# Part 4: Ad-hoc Analysis Report

**Student:** Akshat Dubey  
**Assignment:** Big Data Engineering Assignment 3 — Sydney Airbnb and Census ELT  
**Period covered:** May 2020 – April 2021  
**Warehouse layer:** Postgres `gold.*` (after Bronze load and dbt Silver → Gold)  
**Evidence:** Live DBeaver result grids from the Part 4 SQL in `part_4.sql`

---

## What this analysis is trying to decide

Airbnb activity in Greater Sydney is not evenly spread. Some Local Government Areas (LGAs) and listing neighbourhoods produce far more estimated revenue per active listing than others. The practical question for a platform, council, or host-support programme is not only *where* revenue concentrates, but *what else* sits with that concentration: age structure, household size, listing product mix, how multi-property hosts locate themselves, and whether a single listing’s estimated revenue can cover a local median mortgage.

All figures below come from the Gold fact and Census reference tables after the full pipeline run. Estimated revenue and stays are taken from `gold.fact_listing` for active listings (`has_availability` is true). Census age, household size, median age, and mortgage fields come from the G01/G02 reference tables joined on LGA. Where Gold still stores numeric-looking fields as text, the queries cast to `numeric` before aggregation.

---

## High-revenue LGAs look demographically different from low-revenue ones

**Question 4a.** What are the demographic differences (age mix, household size) between the top 3 and lowest 3 LGAs by average estimated revenue per active listing over the last 12 months?

LGAs were ranked on average estimated revenue per active listing over the last 12 months. The query returns one row per LGA with an explicit `revenue_rank` inside the Top 3 and Bottom 3 groups.

![4a Top and Bottom 3 LGAs](/cursor/stores/self/media/part4/4a_top_bottom_lgas.jpg)

| Group | Rank | LGA | Avg revenue / active listing | % age 0–14 | % age 15–64 | % age 65+ | Avg household size |
|---|---:|---|---:|---:|---:|---:|---:|
| Top 3 | 1 | Mosman | 9,326.52 | 17.61 | 63.26 | 19.15 | 2.4 |
| Top 3 | 2 | Northern Beaches | 7,591.13 | 19.71 | 63.49 | 16.79 | 2.7 |
| Top 3 | 3 | Woollahra | 6,887.47 | 15.62 | 65.75 | 18.64 | 2.3 |
| Bottom 3 | 1 | Fairfield | 1,298.32 | 19.08 | 67.12 | 13.80 | 3.3 |
| Bottom 3 | 2 | Blacktown | 1,310.99 | 22.75 | 66.93 | 10.32 | 3.2 |
| Bottom 3 | 3 | Canterbury-Bankstown | 1,495.72 | 20.51 | 65.56 | 13.93 | 3.0 |

The revenue gap is large. Mosman’s average is roughly seven times Fairfield’s. That scale matters more than small percentage-point moves in any single age band.

Two demographic patterns sit with the revenue split:

1. **Household size.** Every Top 3 LGA has average household size between 2.3 and 2.7. Every Bottom 3 LGA sits at 3.0–3.3. Smaller households line up with higher revenue per active listing in this cut.
2. **Older residents.** The Top 3 show a higher share aged 65+ (about 17–19%) than the Bottom 3 (about 10–14%). The Bottom 3 tend to have a larger working-age and, in Blacktown’s case, child share.

These are associations, not proof that age or household size *cause* higher Airbnb revenue. Coastal and harbour LGAs also differ in tourism appeal, dwelling stock, and nightly rates. The Census fields describe residents of the LGA, while listing revenue describes guests and hosts. Still, for targeting and interpretation, the Top 3 cluster as smaller-household, older-skewed eastern/northern councils, and the Bottom 3 as larger-household western Sydney councils with much lower revenue per active listing.

**Limitation.** The `median_age_persons` column was present in the result grid but not readable in the captured screenshot, so median age is not claimed from 4a itself. Age structure is discussed through the percentage bands that *are* visible.

---

## Older LGAs tend to show higher listing revenue — with caveats

**Question 4b.** Is there a correlation between median age (Census) and revenue per active listing?

The live result used for this report aggregates revenue by **LGA**, joins Census G02 median age, and computes Pearson’s correlation across those LGAs.

![4b Pearson correlation](/cursor/stores/self/media/part4/4b_age_revenue_corr.jpg)

| Metric | Value |
|---|---|
| Number of LGAs (`n_lgas`) | 29 |
| Pearson correlation (median age vs avg revenue per active listing) | **0.6554** |

A coefficient of about 0.66 is a **moderate-to-strong positive** association: LGAs with higher median age tend to show higher average estimated revenue per active listing in this sample of 29 areas.

That fits the 4a picture, where higher-revenue LGAs also had larger 65+ shares. It does **not** mean that raising median age would raise revenue. Amenities, coastal location, dwelling mix, and visitor demand can move both age structure and nightly revenue in the same direction. The correlation is also sensitive to how LGAs are defined and to which LGAs have usable Census median age.

**Method note.** The assignment brief asks for correlation at **neighbourhood** grain (median age of a neighbourhood vs revenue in that neighbourhood). Census median age is published at LGA level, so a neighbourhood analysis would still borrow the mapped LGA’s median age as a proxy. The screenshot evidence provided for this report is the **LGA-level** correlation (`n_lgas = 29`). If a neighbourhood-grain rerun is required for strict brief wording, the SQL in `part_4.sql` (4b-i / 4b-ii) supports that; the numeric claim in this section is limited to what the screenshot shows.

---

## In the top revenue neighbourhoods, entire apartments win on stays

**Question 4c.** For the top 5 `listing_neighbourhood`s by average estimated revenue per active listing, what property type, room type, and capacity maximise total stays?

Neighbourhoods were ranked on average estimated revenue per active listing. Within each of the top five, the listing type with the highest total stays was selected.

![4c Best listing type in top 5 neighbourhoods](/cursor/stores/self/media/part4/4c_top5_listing_types.jpg)

| Rank | Neighbourhood | Best type (by total stays) | Accommodates | Total stays | Avg revenue (that type) |
|---:|---|---|---:|---:|---:|
| 1 | Mosman | Entire apartment / Entire home/apt | 2 | 19,791 | 4,841.60 |
| 2 | Northern Beaches | Entire apartment / Entire home/apt | 4 | 133,597 | 5,204.85 |
| 3 | Woollahra | Entire apartment / Entire home/apt | 2 | 58,328 | 3,597.41 |
| 4 | Hunters Hill | Entire apartment / Entire home/apt | 4 | 1,432 | 4,400.04 |
| 5 | Waverley | Entire apartment / Entire home/apt | 2 | 214,352 | 4,194.42 |

The product recommendation is consistent: **entire apartment (entire home/apt)** dominates stays in every top-five neighbourhood. Capacity splits between two guests (Mosman, Woollahra, Waverley) and four guests (Northern Beaches, Hunters Hill).

Two comparisons matter for interpretation:

- **Rank vs volume.** Waverley has by far the most stays in the winning type (214,352) but ranks fifth on neighbourhood-level average revenue. Mosman ranks first with far fewer stays. High stay volume and high average revenue per listing are not the same thing.
- **Rank vs the revenue column in this grid.** The `neighbourhood_rank` is based on **all** active listings in the neighbourhood. The `avg_revenue_per_active_listing` column in this result is for the **winning listing type only**. That is why Northern Beaches can show a higher type-level average revenue than Mosman while still ranking second overall.

For host guidance in these high-revenue neighbourhoods, the evidence supports promoting entire-home apartments, with capacity tuned locally (2 vs 4), rather than private rooms or unusual property types—at least if the goal is maximising stays among the already high-revenue areas.

**Limitation.** Hunters Hill’s winning type has only 1,432 stays. The type choice there rests on a much thinner base than Waverley or Northern Beaches.

---

## Most multi-listing hosts stay inside one LGA

**Question 4d.** For hosts with multiple listings, are properties concentrated in the same LGA or spread across LGAs?

![4d Multi-listing host concentration](/cursor/stores/self/media/part4/4d_multi_listing_hosts.jpg)

| Measure | Value |
|---|---:|
| Multi-listing hosts | 5,095 |
| Concentrated in one LGA | 3,910 |
| Distributed across LGAs | 1,185 |
| % concentrated | **76.74** |
| % distributed | **23.26** |

Roughly three in four multi-listing hosts keep their portfolio inside a single council area. Only about one in four operate across LGA boundaries.

That pattern suggests local specialisation—shared cleaners, local knowledge, or listings clustered near the host’s own base—more often than geographic diversification. It also concentrates regulatory and demand risk: if one council tightens short-term rental rules, a concentrated host has less natural hedge than a distributed one.

The result does not say *why* hosts concentrate, and it does not separate small “two-listing” hosts from large operators. It does, however, answer the brief clearly: concentration within one LGA is the majority behaviour in this Gold extract.

---

## Northern Beaches leads on single-listing mortgage coverage — with a hard ceiling

**Question 4e.** For single-listing hosts, does estimated revenue over the last 12 months cover the annualised median mortgage in that LGA? Which LGA has the highest percentage of hosts that can cover it?

Annualised median mortgage is `median_mortgage_repay_monthly × 12` from Census G02. Coverage means last-12-month estimated revenue for active listings is at least that amount. LGAs are ranked by the percentage of single-listing hosts who meet the test (minimum 10 hosts in the query filter).

![4e Mortgage coverage by LGA](/cursor/stores/self/media/part4/4e_mortgage_coverage.jpg)

**Highest coverage:** Northern Beaches — **56.41%** of single-listing hosts (2,382 of 4,223) meet the threshold (`coverage_rank` 1). Mosman follows at 52.35%, then Sutherland Shire at 47.15%.

Even in the best-ranked LGA, more than two in five single-listing hosts do **not** cover the annualised median mortgage on this measure. “Best LGA” is therefore relative, not a claim that most hosts are mortgage-safe.

| Coverage rank | LGA | Single-listing hosts | Covering | % covering |
|---:|---|---:|---:|---:|
| 1 | Northern Beaches | 4,223 | 2,382 | 56.41 |
| 2 | Mosman | 340 | 178 | 52.35 |
| 3 | Sutherland Shire | 492 | 232 | 47.15 |
| 4 | Waverley | 4,119 | 1,903 | 46.20 |
| 5 | Hunters Hill | 55 | 25 | 45.45 |
| 6 | Sydney | 5,576 | 2,487 | 44.60 |
| … | … | … | … | … |
| 25 | Georges River | 259 | 51 | 19.69 |

Coastal and eastern LGAs dominate the top of the list; several western and south-western LGAs sit near or below about 22% (for example Canterbury-Bankstown, Penrith, Parramatta, Liverpool, Georges River in this top-25 window).

**Comparisons that matter:**

- **Rate vs volume.** City of Sydney has the most single-listing hosts (5,576) but ranks sixth on coverage (44.60%). Northern Beaches combines high volume with the highest rate.
- **Small samples.** Camden (37 hosts) and Hunters Hill (55) can move sharply with a few hosts; their percentages are less stable than Northern Beaches or Waverley.

**Limitations (these weaken any strong policy claim):**

- Estimated revenue is a pipeline occupancy proxy, not audited net income after fees, cleaning, or vacancies.
- Census mortgage medians are area medians (and from an older Census vintage than the Airbnb months), not the host’s actual loan.
- Matching is by LGA of the listing, not by whether the host owns with a mortgage.

So the honest answer to the brief is: **Northern Beaches has the highest share of single-listing hosts whose estimated 12-month revenue meets the annualised median mortgage test (56.41%),** but coverage is incomplete everywhere, and the measure should be read as a rough affordability screen rather than proof that hosts can service mortgages.

---

## What holds together across the five questions

One geographic story keeps recurring. Eastern and northern harbour/coastal LGAs—Mosman, Northern Beaches, Woollahra, and nearby beach councils—appear at the high end of revenue (4a), in the top neighbourhood set (4c), toward the top of mortgage-coverage ranks (4e), and on the side of the age–revenue association that 4b measures as positive. Western Sydney LGAs such as Fairfield, Blacktown, and Canterbury-Bankstown sit at the low-revenue end in 4a and toward lower mortgage-coverage rates in 4e.

Alongside that geography:

- Smaller Census household size and older age shares sit with higher LGA revenue in the Top 3 vs Bottom 3 cut.
- The listing product that maximises stays in the top revenue neighbourhoods is consistently an entire apartment.
- Multi-listing hosts are mostly local to one LGA, so portfolio risk tracks council-level rules and demand.
- Even the strongest mortgage-coverage LGA clears the bar for only a small majority of single-listing hosts.

---

## Practical implications (proportionate to the evidence)

1. **Host product advice in high-revenue neighbourhoods** should emphasise entire apartments, with capacity 2 or 4 depending on the neighbourhood evidence in 4c—not a generic “any listing type” message.
2. **Geographic targeting** for growth or education campaigns can use the Top 3 revenue LGAs and Northern Beaches / Mosman coverage ranks as priority zones, while treating western low-revenue LGAs as a different market with lower revenue per listing and lower coverage rates.
3. **Multi-listing operations** are mostly single-LGA; compliance and support materials can assume local concentration is the default, with a minority needing multi-council guidance.
4. **Mortgage-related messaging** should stay cautious: even rank 1 reaches only 56%. Pair any “coverage” statistic with the method limits above.

---

## Method summary

| Question | Grain | Main metric | Key live result |
|---|---|---|---|
| 4a | LGA (ranked top/bottom 3) | Avg estimated revenue / active listing (12 months) + Census age % and household size | Mosman / Northern Beaches / Woollahra vs Fairfield / Blacktown / Canterbury-Bankstown |
| 4b | LGA (screenshot evidence) | Pearson corr(median age, avg revenue) | r = 0.6554, n = 29 |
| 4c | Top 5 listing neighbourhoods | Type with max total stays | Entire apartment; capacity 2 or 4 |
| 4d | Multi-listing hosts | Same LGA vs multiple LGAs | 76.74% concentrated |
| 4e | Single-listing hosts by LGA | % covering annualised median mortgage | Northern Beaches 56.41% (rank 1) |

SQL used for these screenshots is maintained in [`part_4.sql`](/cursor/stores/self/docs/part_4.sql).

---

## Evidence gaps (kept separate from the claims above)

- 4a screenshot does not show usable `median_age_persons` values.
- 4b screenshot is LGA-level; neighbourhood-grain correlation was not evidenced in the attached grid.
- Revenue and stays are model estimates from the Gold pipeline, not banked host income.
- No breakdown by month within the 12-month window, so seasonality is not tested here.
- No formal test of whether the age–revenue correlation differs inside vs outside coastal LGAs.

These gaps do not overturn the main patterns above; they bound how strongly the results should be generalised.
