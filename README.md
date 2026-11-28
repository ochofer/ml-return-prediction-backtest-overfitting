# Return prediction: machine learning on the S&P 500's members, corrected for overfitting

The study measures two things on the S&P 500's point-in-time members from January 1991 to December 2024, out of sample from 2006. First, how much machine-learning models add to linear models in forecasting each member's return in the next month. Second, how much of the best result is real once corrected for overfitting, that is, for having tried many versions and kept the best one. It follows the method of Gu, Kelly and Xiu (2020), reduced to twelve characteristics and the index's members, with the squared-error loss and a validation sample of five years. The overfitting is measured with the deflated Sharpe ratio of Bailey and López de Prado (2014) and the probability of backtest overfitting of Bailey, Borwein, López de Prado and Zhu (2017).

The study is in build and has reached the first of its five milestones, the panel, i.e., the table with one row per member-month that holds the twelve characteristics and the target. The repository holds the study's settings and their tests, the notebooks that build the panel with the aggregates they write, the code of the two overfitting statistics with its test on made-up returns whose answers are known, and code copied from two other studies, which reads the data and computes the intervals for the difference of two Sharpe ratios.

## Milestone 1, the panel

The panel passes gate G1, three checks fixed in advance that stop the study if any of them is not met:

- (a) The data code copied from Stock-Level Tracking gives that study's 440,724 member-months from January 1926 to December 2024, and files identical, byte for byte, to its yearly comparison of the count of members with CRSP's and to its yearly counts of delistings, at that study's commit 0dabbb0 (`g1a.json`).
- (b) Each of the twelve characteristics is present for at least its level of each month's members in all 408 months from January 1991 to December 2024. The lowest months are book-to-market at 0.924 (May 2024) and operating profitability at 0.930 (July 2023), against a level of 0.80, and the other ten stay at or above 0.982, against 0.90 (`g1b_coverage.json`).
- (c) The 24 months drawn by the study's seed, built a second time from data cut at the last trading day of the month before, equal the full build in all 144,312 cells, with a largest difference of 0 (`g1c_lookahead.csv`).

The panel holds 204,520 member-months, 100 of them without a target, and 114,521 in the test window (`panel_summary.json`). Before the run, ten figures were written into notebooks 03 and 04 as expected values, which are set beside the results and judged by no gate. Seven are forecasts. The other three are read from Stock-Level Tracking's tables on the same extraction: the count of member-months without size, the coverage of the four daily characteristics, and the share with a GVKEY, Compustat's permanent number for a company. The results, from the tables in `results/20260928/`:

- Size is missing in 10 of the 204,520 member-months, where 10 was expected (`market_summary.json`).
- Volatility, the largest daily return, beta and idiosyncratic volatility are present for at least 0.994 of member-months in every year, the lowest in 2006, where above 0.98 was expected (`market_coverage_by_year.csv`).
- Momentum is present for at least 0.987 of member-months in every year, the lowest in 2001, where above 0.97 was expected (`market_coverage_by_year.csv`).
- The mean beta by year runs from 0.733 in 2000 to 1.160 in 2009, where 0.7 to 1.3 was expected (`market_summary.json`).
- The median volatility across members is highest in 2000 (0.0272), then 2009 (0.0267) and 2008 (0.0255), where 2008 or 2009 was expected (`market_medians_by_year.csv`).
- A company in Compustat is found for at least 0.996 of member-months in every year, the lowest in 2023, where at least 0.996 was expected (`accounting_coverage_by_year.csv`).
- An annual record of the right age is found for at least 0.998 of the member-months with a company in every year, where above 0.97 was expected (`accounting_coverage_by_year.csv`).
- A record without book-to-market covers at most 0.069 of member-months in a year, in 2024, where below 0.10 was expected (`accounting_coverage_by_year.csv`).
- Book-to-market and operating profitability are present for at least 0.927 and 0.933 of member-months in every year, and earnings-to-price and asset growth for at least 0.986 and 0.992, where above 0.85 and above 0.95 were expected (`accounting_coverage_by_year.csv`).
- The annual record used is 6 to 11 months old in 0.5000 of member-months and 12 to 17 months old in 0.4995, where about half each was expected (`accounting_summary.json`).

Gate G3 tests the code of the two overfitting statistics on made-up returns whose answers are known, in four clauses. On returns with no true effect, the mean of the best of 32 Sharpe ratios, the median of the best trial's deflated Sharpe ratio and the mean probability of backtest overfitting each lie in the range fixed in advance, and the best trial's deflated Sharpe ratio is above 0.95 in at most 5 per cent of repetitions. With one trial given a true Sharpe ratio of 1.0 a year, that trial has the best Sharpe ratio in at least 90 per cent of repetitions and the mean probability of backtest overfitting is below 0.25. All four clauses are met (`results/simulation/g3_summary.json`).

## Data

CRSP and Compustat, via Wharton Research Data Services, accessed through Queen Mary University of London, in an extraction of 28 September 2026, and Kenneth French's data library. The repository holds no data: code, tests and aggregates only.

The universe, the monthly returns and the reading of the extraction come from the data layer of [Stock-Level Tracking](https://github.com/ochofer/stock-level-tracking), whose seven modules are copied unchanged into `src/mlrp/source/` from its commit 0dabbb0. The interval for the difference of two Sharpe ratios comes from [Volatility-managed portfolios](https://github.com/ochofer/volatility-managed-portfolios), whose `bootstrap.py` is copied unchanged from its commit 74ed6d1. The tests check that every copy is unchanged. `src/mlrp/data.py` calls the copies with this study's settings.

## Notebooks

The notebooks in `notebooks/` are written by `build_notebooks.py` and committed without outputs. Notebooks 01 to 05 build the panel. They read the extraction in place from the folder named by the environment variable `MLRP_DATA_ROOT` and run in order: each writes its aggregates to `results/20260928/`, and writes the tables with one row per member, which a later notebook reads, to `local/`, which git ignores. Notebook 12 reads no data.

## Settings

`src/mlrp/constants.py` holds every gate, expectation, sample boundary and count of the study, each with its reason. A value still to be set is `None` in the file, with the milestone that sets it. The tests run with `python3 -m pytest tests`.

## Terms

- **universe**: the S&P 500's point-in-time members, that is, the stocks in CRSP's list of index members on the last trading day of the month before.
- **member**: a stock in the universe in a given month. A **member-month** is one member in one month.
- **panel**: the table with one row per member-month from January 1991 to December 2024, holding the twelve characteristics and the target.
- **characteristic**: a number measured for each member at the end of the month before, such as its market capitalisation or its return over the past year. The study uses twelve.
- **target**: the member's excess return in the month, its total return less the one-month Treasury bill rate.
- **model**: one of six ways to forecast the target from the characteristics: least squares on three characteristics (OLS-3), least squares on all twelve (OLS-12), the elastic net, a random forest, boosted trees and a network with one hidden layer.
- **configuration**: one choice of a model's settings, such as a penalty or the depth of a tree. The six models have 32 configurations in all.
- **training sample**: for a test year, the months from January 1991 to six years before it, on which a model is fitted.
- **validation sample**: the five years before a test year, on which a model's configuration is chosen.
- **test sample**: the test year itself, which the chosen fit forecasts. Each year from 2006 to 2024 is a test year once.
- **test window**: the 228 months from January 2006 to December 2024, the 19 test samples together.
- **tuned model**: a model with, in each test year, the configuration its validation sample chose.
- **forecast**: a model's number for a member's target in a month.
- **out-of-sample R²**: one minus the sum of squared forecast errors over the sum of squared targets, pooled over the member-months of the test window, in per cent a month. The denominator is not demeaned.
- **pooled-mean forecast**: the mean target over the training sample's member-months, the same number for every member.
- **rank correlation**: the correlation across a month's members between the ranks of their forecasts and the ranks of their targets. Also called the information coefficient.
- **long-short portfolio**: the top tenth of a month's members by forecast held long and the bottom tenth held short.
- **turnover**: the share of a portfolio traded in a month, counted one way.
- **trial**: the long-short portfolio of one configuration held for the whole test window, net of costs. There are 32.
- **deflated Sharpe ratio**: the confidence, from 0 to 1, that the best trial's Sharpe ratio is above the highest one that 32 trials with no true effect would be expected to show.
- **probability of backtest overfitting**: the share of the ways of splitting the test window into two halves in which the trial that is best in one half ranks in the lower half of the other.
- **gate**: a test fixed in advance that the data or the code must pass. A gate that is not met stops the build.
- **expectation**: a test of a result fixed in advance. An expectation that is not met is reported as a finding.

## References

Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance* 20(4), 39 to 69. https://doi.org/10.21314/JCF.2016.322

Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management* 40(5), 94 to 107. https://doi.org/10.3905/jpm.2014.40.5.094

Gu, S., Kelly, B. and Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies* 33(5), 2223 to 2273. https://doi.org/10.1093/rfs/hhaa009

## Other projects

**Optimal versus naive diversification.** Under a long-only index-tracking mandate on French's 49 industries, 1979 to 2026, each of five covariance estimators forecasts less tracking error than its portfolio realises: the realised active return is 22 to 61 per cent more volatile than forecast. The replication of DeMiguel, Garlappi and Uppal (2009) reproduces 29 of the 30 out-of-sample Sharpe ratios it tests within 0.03 of the published values, 26 of 27 with the value-weighted market counted once. [Page](https://www.carlohofer.com/projects/optimal-vs-naive-diversification/), [dashboard](https://www.carlohofer.com/optimal-vs-naive-diversification/dashboard/B-P_dashboard.html), [note](https://www.carlohofer.com/optimal-vs-naive-diversification/note/B-P_note_2026-10.pdf), [code](https://github.com/ochofer/optimal-vs-naive-diversification)

**Climate risk as an equity signal.** An ownership graph linking 328 ownership entities in the US and developed Europe, 302 of them listed companies with a price series, to 5,115 asset holdings, built from Global Energy Monitor's free tracker. Of 19 apparent ownership changes drawn at random and checked by hand, 12 had no corporate event behind them. [Page](https://www.carlohofer.com/projects/climate-risk-equity-signal/), [note](https://github.com/ochofer/paper1-hazard-exposure-data/releases/download/note-v1.4/what-a-vintage-difference-measures.pdf), [code](https://github.com/ochofer/paper1-hazard-exposure-data)

**Volatility-managed portfolios: a replication, the real-time critique, and the months after the sample.** Volatility estimation and position sizing on Ken French's equity factors, replicating Moreira and Muir (2017) and the real-time critique of Cederburg, O'Doherty, Wang and Yan (2020). Scaled by the inverse of last month's realised variance, a factor's Sharpe ratio is higher than the original's by a margin that is statistically significant at the 5 per cent level for momentum alone when each of the six factors is tested on its own, and for no factor after Holm's correction for the six tests. In the 136 months after Moreira and Muir's sample no factor's difference is statistically significant, and for the six volatility-managed factors the intervals are too wide to rule out the differences found on their sample. [Page](https://www.carlohofer.com/projects/volatility-managed-portfolios/), [dashboard](https://www.carlohofer.com/volatility-managed-portfolios/dashboard/B-Q_dashboard.html), [note](https://www.carlohofer.com/volatility-managed-portfolios/note/B-Q_note_2026-11.pdf), [code](https://github.com/ochofer/volatility-managed-portfolios)

**Stock-level index tracking: a point-in-time benchmark, a long-only mandate, and its performance attribution by sector.** Rebuilds the S&P 500 month by month from CRSP's list of members, at a tracking error of 11.5 basis points a year from 1926 to 2024 against CRSP's own return of the same list, and runs a long-only index-tracking mandate on it from 1999 to 2024, with its active return attributed by sector. The portfolios realise 78.9 to 83.6 basis points a year of tracking error, and each of four covariance estimators forecasts less: the realised active return is 48 to 85 per cent more volatile than forecast. Index changes require 11.4 per cent of the turnover of the portfolio built with Ledoit-Wolf shrinkage. [Page](https://www.carlohofer.com/projects/stock-level-tracking/), [dashboard](https://www.carlohofer.com/stock-level-tracking/dashboard/), [note](https://www.carlohofer.com/stock-level-tracking/note/stock-level-tracking_note_2026-11.pdf), [code](https://github.com/ochofer/stock-level-tracking)

**Rules-based portfolio.** Runs a real-money portfolio under rules published before the first order, with every later change dated in the repository: two UCITS ETFs at 70/30, a monthly cycle with pre-trade checks, look-through exposure and ex-ante risk on a public dashboard. [Page](https://www.carlohofer.com/projects/rules-based-portfolio/), [dashboard](https://www.carlohofer.com/rules-based-portfolio/), [guide](https://www.carlohofer.com/rules-based-portfolio/guide/guide.pdf), [code](https://github.com/ochofer/rules-based-portfolio)

## Licence

MIT.
