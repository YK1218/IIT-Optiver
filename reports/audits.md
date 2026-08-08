# Step 0 — data audits

Three audits define the themes: **missingness → 1.3**, **imbalance → 1.5**, **sparsity → 1.1**.

|                           |        uci |      paysim |        ieee |
|:--------------------------|-----------:|------------:|------------:|
| rows                      |    3e+04   |   1.049e+06 |   5.905e+05 |
| features                  |   23       |  13         | 431         |
| pos_rate                  |    0.2212  |   0.001089  |   0.03499   |
| imbalance_ratio           |    3.521   | 917.2       |  27.58      |
| cols_any_missing          |    0       |   0         | 414         |
| cols_over_50pct_missing   |    0       |   0         | 214         |
| mean_missing_rate         |    0       |   0         |   0.4539    |
| entities                  |    3e+04   |   1.048e+06 |   1.997e+05 |
| median_events_per_entity  |    1       |   1         |   1         |
| p95_events_per_entity     |    1       |   1         |   9         |
| single_event_entities_pct |    1       |   0.9998    |   0.5809    |
| n_train                   |    2.1e+04 |   7.34e+05  |   4.134e+05 |
| n_val                     | 4500       |   1.573e+05 |   8.858e+04 |
| n_test                    | 4500       |   1.573e+05 |   8.858e+04 |
| median_gap                |  nan       |  14.5       |   2.634e+05 |
| p95_gap                   |  nan       |  34         |   3.974e+06 |
