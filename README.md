# ml-return-prediction-backtest-overfitting

This study is in build. It has reached none of its five milestones: the first, the panel, is in progress, and the repository holds its settings and their tests.

The study measures two things on the S&P 500's point-in-time members from January 1991 to December 2024. First, how much machine-learning models add to linear models in forecasting each member's return in the next month, out of sample. Second, how much of the best backtest's Sharpe ratio is left after a correction for the number of trials. It follows the method of Gu, Kelly and Xiu (2020), reduced to twelve characteristics and the index's members, and measures the overfitting with the deflated Sharpe ratio of Bailey and López de Prado (2014) and the probability of backtest overfitting of Bailey, Borwein, López de Prado and Zhu (2017).

## Data

CRSP and Compustat, via Wharton Research Data Services, accessed through Queen Mary University of London, in an extraction of 28 September 2026, and Kenneth French's data library. The repository holds no data: code, tests and aggregates only.

## Settings

`src/mlrp/constants.py` holds every gate, expectation, sample boundary and count of the study, each with its reason. The levels were set on 18 November 2026. That date rests on the file alone, since the history begins with the commit that holds it. A value still to be set is `None` in the file, with the milestone that sets it. The tests run with `python3 -m pytest tests`.

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

## Licence

MIT.
