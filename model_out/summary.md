| # | Модель | Семейство | RMSLE val | RMSLE honest | Δ к наивному |
|---|---|---|---|---|---|
| 1 | LightGBM DART on log1p | gbdt | **1.6764** | 1.7504 | -0.519 |
| 2 | Two-stage LGBM + post calibration | two-stage | **1.6817** | 1.7515 | -0.513 |
| 3 | Two-stage LGBM + BTYD feats | two-stage | **1.6856** | 1.7496 | -0.509 |
| 4 | LightGBM L2 on log1p + BTYD feats | gbdt | **1.6888** | 1.7509 | -0.506 |
| 5 | Two-stage LGBM: P(buy) x E[log1p | buy] | two-stage | **1.6894** | 1.7507 | -0.506 |
| 6 | LightGBM L2 on log1p | gbdt | **1.6910** | 1.7515 | -0.504 |
| 7 | CatBoost RMSE on log1p | gbdt | **1.6926** | 1.7485 | -0.502 |
| 8 | XGBoost L2 on log1p | gbdt | **1.6929** | 1.7526 | -0.502 |
| 9 | XGBoost reg:squaredlogerror (raw) | gbdt | **1.6967** | 1.7599 | -0.498 |
| 10 | LightGBM L2 log1p, deep (255 leaves, 1200 trees) | gbdt | **1.6975** | 1.7578 | -0.498 |
| 11 | EB shrunk mean of log1p 30d blocks (K=10, kappa=0.5, decay=0.7) | btyd | **1.7148** | 1.8049 | -0.480 |
| 12 | Dilated CNN over 364 daily steps | sequence | **1.7265** | 1.7759 | -0.469 |
| 13 | LightGBM Huber on log1p | gbdt | **1.7346** | 1.7992 | -0.461 |
| 14 | GRU over 52 weekly steps | sequence | **1.7384** | 1.7761 | -0.457 |
| 15 | EB log-blocks + log-calibration | btyd | **1.7438** | 1.7908 | -0.451 |
| 16 | EB log-blocks of 14d rescaled to 30d + calibration | btyd | **1.7552** | 1.8182 | -0.440 |
| 17 | MLP wide 1024-512-256 L2 on log1p | neural | **1.7568** | 3.4825 | -0.438 |
| 18 | PatchTST-style transformer (patch 14d) | sequence | **1.7590** | 1.7885 | -0.436 |
| 19 | MLP L2 on log1p (3 seeds averaged) | neural | **1.7723** | 2.1557 | -0.423 |
| 20 | geometric mean of last 6 blocks | baseline | **1.7732** | 1.8532 | -0.422 |
| 21 | MDME-style multi-expert (8 buckets) | neural | **1.7746** | 1.8285 | -0.420 |
| 22 | LightGBM quantile 0.5 on log1p | gbdt | **1.7886** | 1.8665 | -0.406 |
| 23 | LightGBM MAE on log1p | gbdt | **1.7886** | 1.8598 | -0.406 |
| 24 | MLP L2 on log1p | neural | **1.7917** | 2.1035 | -0.403 |
| 25 | EB log-blocks of 60d rescaled to 30d + calibration | btyd | **1.7944** | 1.8442 | -0.401 |
| 26 | CatBoost Quantile 0.5 on log1p | gbdt | **1.7968** | 1.8624 | -0.398 |
| 27 | EB log-blocks of 90d rescaled to 30d + calibration | btyd | **1.8075** | 1.8585 | -0.388 |
| 28 | MMoE (6 experts, 2 tasks) | neural | **1.8183** | 2.3938 | -0.377 |
| 29 | BG/NBD x Gamma-Gamma [fixed] + log-calibration | btyd | **1.8209** | 1.8515 | -0.374 |
| 30 | Two-head MLP: P(buy) x E[log1p | buy] | neural | **1.8225** | 2.1841 | -0.373 |
| 31 | Two-head MLP (3 seeds averaged) | neural | **1.8312** | 2.0586 | -0.364 |
| 32 | TSB + log-calibration | intermittent | **1.8405** | 1.8835 | -0.355 |
| 33 | alpha*prev90/3 (alpha=0.40) | baseline | **1.8517** | 1.9280 | -0.343 |
| 34 | expm1(0.68*log1p(prev90/3)+0.55) | baseline | **1.8523** | 1.9089 | -0.343 |
| 35 | LightGBM Tweedie + log-calibration | gbdt | **1.8693** | 1.9241 | -0.326 |
| 36 | XGBoost Tweedie + log-calibration | gbdt | **1.8702** | 1.9171 | -0.325 |
| 37 | Croston SBA + log-calibration | intermittent | **1.8764** | 1.9256 | -0.319 |
| 38 | LightGBM Poisson + log-calibration | gbdt | **1.8844** | 1.9371 | -0.311 |
| 39 | CatBoost Tweedie + log-calibration | gbdt | **1.8869** | 1.9386 | -0.308 |
| 40 | expm1(0.57*log1p(prev30)+1.10) | baseline | **1.9183** | 1.9631 | -0.277 |
| 41 | median of last 6 blocks | baseline | **1.9388** | 2.0122 | -0.256 |
| 42 | MLP + ZILN loss (Google) | neural | **2.0180** | 2.0039 | -0.177 |
| 43 | alpha*prev30 (alpha=0.43) | baseline | **2.0425** | 2.1380 | -0.153 |
| 44 | ADIDA (alpha=0.1, agg=30) | intermittent | **2.0484** | 2.0408 | -0.147 |
| 45 | Ridge on log-features -> log1p target (alpha=1000) | linear | **2.0519** | 1.7569 | -0.143 |
| 46 | IMAPA (levels [7, 14, 30, 60], alpha=0.1) | intermittent | **2.0590** | 2.0406 | -0.136 |
| 47 | lifetime rate per 30d | baseline | **2.0607** | 2.0486 | -0.134 |
| 48 | prev365 / 12.17 | baseline | **2.0639** | 3.5215 | -0.131 |
| 49 | Ridge on log-features -> log1p target (alpha=100) | linear | **2.0712** | 1.7979 | -0.124 |
| 50 | EWMA of 30d blocks (lambda=0.50) | baseline | **2.0754** | 2.0566 | -0.120 |
| 51 | TSB (alpha=0.3, beta=0.02) | intermittent | **2.0798** | 2.0604 | -0.115 |
| 52 | mean(prev30, prev60/2, prev90/3) | baseline | **2.0823** | 2.0840 | -0.113 |
| 53 | 0.5*prev30 + 0.5*seasonal naive | baseline | **2.0886** | 2.1180 | -0.106 |
| 54 | prev90 / 3 | baseline | **2.0936** | 2.1055 | -0.102 |
| 55 | prev180 / 6 | baseline | **2.1016** | 2.0668 | -0.093 |
| 56 | BG/NBD x empirical mean spend | btyd | **2.1058** | 2.0401 | -0.089 |
| 57 | MBG/NBD x empirical mean spend | btyd | **2.1075** | 2.0400 | -0.088 |
| 58 | prev60 / 2 | baseline | **2.1114** | 2.1299 | -0.084 |
| 59 | Croston SBA (alpha=0.3) | intermittent | **2.1215** | 2.1089 | -0.074 |
| 60 | BG/NBD x Gamma-Gamma [fixed] | btyd | **2.1293** | 2.0534 | -0.066 |
| 61 | Croston classic (alpha=0.15) | intermittent | **2.1496** | 2.1437 | -0.045 |
| 62 | Empirical-Bayes BetaBin x shrunk mean spend | btyd | **2.1661** | 2.1410 | -0.029 |
| 63 | prev30 (naive AR = sample_submit) | baseline | **2.1951** | 2.2362 | +0.000 |
| 64 | const optimal (c=10.25) | baseline | **2.2952** | 2.3496 | +0.100 |
| 65 | LightGBM Tweedie (raw) | gbdt | **2.4401** | 2.3715 | +0.245 |
| 66 | LightGBM Poisson (raw) | gbdt | **2.4786** | 2.4100 | +0.284 |
| 67 | seasonal naive (same window a year ago) | baseline | **2.5130** | 2.3020 | +0.318 |
| 68 | BG/NBD expected transactions (raw count) | btyd | **2.5162** | 2.7829 | +0.321 |
| 69 | const 0 | baseline | **3.2036** | 3.5215 | +1.009 |