"""
The settings of the study, written down before any code that they judge.

Every gate, expectation, sample boundary, count and design parameter is defined
here, one per line, with its reason beside it. A notebook, a script or a module
may read these names. It may not introduce numbers of its own.

The levels of the gates (G1 to G3) and of the expectations (E1 to E10), the
split, the counts of configurations, the cost, the twelve characteristics and
the overfitting statistics were set on 18 November 2026. That date rests on
this file alone: the repository's history begins with the commit that holds
this file, and it cannot show the date. The seed, the names of the random
streams, the data root's variable and the size checks' power were set on
19 November 2026. The commit the copied code is taken from, the copies' SHA-256,
the digest of the manifest's entries for the eight tables, the results folder
and French's folder were set on 20 November 2026, and the folder of the frames
kept between notebooks on 21 November 2026. A value that is still to be set is None here, with the
milestone that sets it, and it is set in a commit of its own before any run
that reads it. A level is never moved after a run that judges it.

Gates and expectations are strict: each is reported as met or not met, in the
same words either way, with its arithmetic. A gate not met stops the build. An
expectation not met is a finding and is reported as one.

Sources cited by short name below:
  CRSP and Compustat: the extraction of 28 September 2026 from
    Wharton Research Data Services, accessed through Queen Mary University of
    London (section 1).
  French: Kenneth French's data library, downloaded at run time and never committed.
  Gu, Kelly and Xiu (2020), "Empirical Asset Pricing via Machine Learning",
    Review of Financial Studies 33(5), 2223-2273.
  Bailey and López de Prado (2014), "The Deflated Sharpe Ratio", Journal of
    Portfolio Management 40(5), 94-107.
  Bailey, Borwein, López de Prado and Zhu (2017), "The probability of backtest
    overfitting", Journal of Computational Finance 20(4).
  Avramov, Cheng and Metzker (2023), "Machine Learning vs. Economic Restrictions:
    Evidence from Stock Return Predictability", Management Science 69(5), 2587-2619.
  Ledoit and Wolf (2008), "Robust performance hypothesis testing with the Sharpe
    ratio", Journal of Empirical Finance 15(5), 850-859.
  The sources the notebooks name, as Crossref gives them, added on 24 November 2026:
  Shumway (1997), "The Delisting Bias in CRSP Data", The Journal of Finance
    52(1), 327-340.
  Fama and French (1992), "The Cross-Section of Expected Stock Returns",
    The Journal of Finance 47(2), 427-465.
  Fama and French (2015), "A five-factor asset pricing model", Journal of
    Financial Economics 116(1), 1-22.
  Basu (1983), "The relationship between earnings' yield, market value and
    return for NYSE common stocks", Journal of Financial Economics 12(1), 129-156.
  Cooper, Gulen and Schill (2008), "Asset Growth and the Cross-Section of Stock
    Returns", The Journal of Finance 63(4), 1609-1651.
  Ang, Hodrick, Xing and Zhang (2006), "The Cross-Section of Volatility and
    Expected Returns", The Journal of Finance 61(1), 259-299.
  Anderson and Dyl (2007), "Trading Volume: NASDAQ and the NYSE", Financial
    Analysts Journal 63(3), 79-86.
  Stock-Level Tracking: github.com/ochofer/stock-level-tracking, whose data layer
    this study copies (section 2).
  Signal Evaluation: github.com/ochofer/volatility-managed-portfolios, whose
    block bootstrap this study copies (section 9).
"""

# ---------------------------------------------------------------------------
# 1. The data: the extraction, read in place and never committed
# ---------------------------------------------------------------------------

# The folder that holds the extraction's manifest.json and raw/ is named by this
# environment variable, set outside the repository on the machine that holds the
# data. Every notebook stops with one sentence when it is missing. Set on
# 19 November 2026.
DATA_ROOT_ENV = "MLRP_DATA_ROOT"

EXTRACTION_DATE = "2026-09-28"   # fetch date of every table this study reads, and the vintage of every result
EXTRACTION_SOURCE = ("CRSP and Compustat, via Wharton Research Data Services, "
                     "accessed through Queen Mary University of London")   # named with the extraction date on every exhibit drawn from them
MANIFEST_FILE = "manifest.json"  # the extraction's record: the SQL, rows, columns, SHA-256 and fetch time of every file
MANIFEST_SHA256 = "23e96d189f4b095936f47a1ebacb4e987fb260726afab4b6e6400c25ebce8693"   # read on 18 and 19 November 2026, equal to manifest.json.sha256
MANIFEST_FILES = 310             # files the manifest lists
MANIFEST_ROWS = 112_161_796      # rows the manifest lists

# The eight tables' entries in the manifest, as one digest: the SHA-256 of the
# compact JSON list of [file, rows, SHA-256] of every file of the keys in TABLES
# below, sorted by key and file. Every run checks it beside the manifest's own
# hash. A later pull that adds tables to the extraction changes the manifest's
# hash and leaves this digest as it is, as long as the eight tables are the same.
# Set on 20 November 2026.
MANIFEST_ENTRIES_SHA256 = "9e0202735911afd462e0ef6f7a9bf5f3937dc3be94faa863d457abae76b09237"
MANIFEST_ENTRIES = 95            # files of the eight tables: 37 yearly files of the daily file, 52 of the fundamentals, one each for the other six
MANIFEST_ENTRIES_ROWS = 73_595_093   # rows of those 95 files, the sum of the manifest's counts for them, read on 21 November 2026

RESULTS_DIR = "results/20260928"   # the folder of the panel's committed aggregates, named by the extraction date. Set on 20 November 2026
FRENCH_CACHE_DIR = "data/french"   # French's files as downloaded, never committed: data/ is ignored by git. Set on 20 November 2026
LOCAL_DIR = "local"                # frames with one row per security kept between notebooks on the machine that holds the extraction, never committed: local/ is ignored by git. Set on 21 November 2026
CRSP_END = "2024-12-31"          # the last date of the CRSP library on this subscription, where the sample ends
DAILY_FILE_START = "1990-01-02"  # the first date of the daily stock file in the extraction

# The extraction's keys for the tables this study reads (its folders raw/<key>/),
# with the WRDS table each holds. No query is made to WRDS.
TABLES = {
    "crsp_dsp500list": "crsp.dsp500list",   # S&P 500 membership by PERMNO with start and end dates: the universe
    "crsp_msp500": "crsp.msp500",           # CRSP's monthly file of the same universe
    "crsp_msf": "crsp.msf",                 # monthly prices, returns, volumes and shares outstanding
    "crsp_dsf": "crsp.dsf",                 # daily returns, from 2 January 1990
    "crsp_msedelist": "crsp.msedelist",     # delisting dates, codes and returns
    "crsp_msenames": "crsp.msenames",       # names, for the company of each security
    "crsp_ccm_lnkhist": "crsp.ccmxpf_lnkhist",   # the links from CRSP's securities to Compustat's companies
    "comp_funda": "comp.funda",             # Compustat's annual fundamentals, the standard filters, all columns
}
TABLE_COUNT = 8

# French's library supplies the market's daily excess return (for beta and
# idiosyncratic volatility) and the one-month bill rate (for the target). Its
# files are downloaded at run time, never committed, and their vintage is
# recorded in the provenance record.
FRENCH_SERIES = ("daily market excess return", "one-month bill rate")

# ---------------------------------------------------------------------------
# 2. The copied data layer: the universe and the returns of Stock-Level Tracking
# ---------------------------------------------------------------------------

# Seven modules of Stock-Level Tracking's package are copied verbatim into
# src/mlrp/source/. The names of the source's constants that they read are in a
# file of their own there, src/mlrp/source/constants.py, each at the source's
# value at SOURCE_COMMIT. Its constants file is not copied. The study's own
# modules call the copies through src/mlrp/data.py with this file's settings.
# The commit and the SHA-256 below were set on 20 November 2026, in a commit
# before any code reads the extraction.
COPIED_INTO = "src/mlrp/source"
SOURCE_REPOSITORY = "github.com/ochofer/stock-level-tracking"
SOURCE_COMMIT = "0dabbb016492707b2f2e14884bceae89f088e1d6"
COPIED_MODULES = ("extraction.py", "universe.py", "returns.py", "sectors.py", "sorts.py", "french.py", "layer.py")
COPIED_FILES = {                 # file: (its path in the source repository, its SHA-256 at SOURCE_COMMIT)
    "extraction.py": ("src/slt/extraction.py", "72f61ed8e8460684b869534c377dbd026fac98779a121e199fac078e1db5c0a3"),
    "universe.py": ("src/slt/universe.py", "c9c6bc68abba106abd3d05632d6ace1a5f751119f17779da99e9fb6db14e0f75"),
    "returns.py": ("src/slt/returns.py", "02f1cc8034a24b6f3677cfa1eb5f312f558e7876251c26f5dc516ec0222aeb45"),
    "sectors.py": ("src/slt/sectors.py", "030fbe948075bbcbd781cfa2e6e7a9fcbcf9bfc38eb29fba8731afab7731c21c"),
    "sorts.py": ("src/slt/sorts.py", "50954e2de053b600f85c118b1c7d605bf713e24b06983fbc6a137d690a79489e"),
    "french.py": ("src/slt/french.py", "1cac5eff1cf3d5cffe2c09d993d5ad35cf5f1400110bcf4e434c20babad2b0d2"),
    "layer.py": ("src/slt/layer.py", "8c8b919f1f26215d933bf57b338e56afd64356ac2e02cc19f6e569eede1964fc"),
}
SOURCE_CONSTANTS_FILE = "src/slt/constants.py"
SOURCE_CONSTANTS_SHA256 = "5a3c817e834a95b95dcd639895196004e717ffce4116974620f167a5bdacee4a"   # at SOURCE_COMMIT

# Gate G1 (a) compares the copied layer's counts with three tables the source
# repository committed at the commit the copies are taken from, SOURCE_COMMIT.
# Read on 19 and 20 November 2026: the three are byte for byte the same at
# 75ce238, at 1d57cbf and at 0dabbb0 of the source repository.
G1A_REFERENCE_TABLES = (
    "results/20260928/universe_summary.json",
    "results/20260928/universe_count_check_by_year.csv",
    "results/20260928/delistings_by_year.csv",
)
G1A_MEMBER_MONTHS = 440_724      # member-months, January 1926 to December 2024, in the first of the three
G1A_FIRST_MONTH = "1926-01"
G1A_LAST_MONTH = "2024-12"

# ---------------------------------------------------------------------------
# 3. The panel
# ---------------------------------------------------------------------------

# One row per member of the universe and month t, for t from January 1991 to
# December 2024. The counts are the data layer's committed counts, read on
# 18 November 2026, and the panel's own counts are checked against them.
PANEL_FIRST_MONTH = "1991-01"
PANEL_LAST_MONTH = "2024-12"
PANEL_MONTHS = 408
PANEL_MEMBER_MONTHS = 204_520            # member-months in the universe over the panel's months
PANEL_MEMBER_MONTHS_NO_RETURN = 100      # with no return and no delisting return: kept in the panel without a target, and left out of every fit
TEST_FIRST_MONTH = "2006-01"
TEST_LAST_MONTH = "2024-12"
TEST_MONTHS = 228
TEST_MEMBER_MONTHS = 114_521             # member-months in the universe over the test window

# The target is the excess return of month t: the total return with the
# delisting rule of the copied returns module applied, minus French's one-month
# bill rate. It is not transformed. Everything on the right-hand side is known on
# the last trading day of month t-1.
TARGET = "excess return of month t"
TARGET_IS_TRANSFORMED = False

# ---------------------------------------------------------------------------
# 4. The twelve characteristics, each measured at the end of month t-1
# ---------------------------------------------------------------------------

CHARACTERISTICS = (
    "size",                       # log of the security's market capitalisation, the data layer's `cap`
    "book_to_market",             # book equity over the company's market value at the fiscal year end of the record used
    "earnings_to_price",          # income before extraordinary items over the same market value, negative values kept
    "operating_profitability",    # revenue less cost of goods sold, selling, general and administrative expenses and interest expense, over book equity
    "asset_growth",               # total assets over the previous fiscal year's total assets of the same company, less one
    "momentum",                   # the return from month t-12 to month t-2, by the copied momentum function
    "reversal",                   # the return of month t-1
    "volatility",                 # the standard deviation of the daily returns in month t-1
    "max_daily_return",           # the largest daily return in month t-1
    "beta",                       # the slope of the daily excess return on the market's daily excess return over the last 252 trading days
    "idiosyncratic_volatility",   # the standard deviation of the residuals of the same regression
    "dollar_volume",              # log of price times share volume in month t-1, from the monthly file
)
CHARACTERISTIC_COUNT = 12

# The groups of expectation E5, after the groups Gu, Kelly and Xiu (2020) find to
# matter most: price trend, risk and liquidity. Eight of the twelve, so their
# share of the importance by count is two thirds.
E5_GROUPS = {
    "price trend": ("momentum", "reversal"),
    "risk": ("volatility", "max_daily_return", "beta", "idiosyncratic_volatility"),
    "liquidity": ("size", "dollar_volume"),
}
E5_SHARE_BY_COUNT = 8 / 12

# Windows of the market characteristics.
MOMENTUM_FIRST_LAG = 12          # momentum starts at month t-12
MOMENTUM_LAST_LAG = 2            # and ends at month t-2, skipping month t-1
DAILY_MIN_DAYS = 15              # volatility and the largest daily return need at least 15 daily returns in month t-1
BETA_WINDOW_DAYS = 252           # beta and idiosyncratic volatility use the 252 trading days to the end of month t-1
BETA_MIN_DAYS = 120              # with at least 120 daily returns among them

# Timing of the accounting data. At the end of month t-1 the study uses the
# annual record of the member's company with the latest fiscal year end that is
# at least six months old and at most 24 months old. The company is the GVKEY of
# the link valid on the last trading day of month t-1.
ACCOUNTING_MIN_AGE_MONTHS = 6
ACCOUNTING_MAX_AGE_MONTHS = 24
# The link types through which a member reaches its company: those of the copied
# sector join, which the tests check against the copied constants. Notebook 04
# counts the member-months a link of any other type would reach. Set on
# 24 November 2026.
ACCOUNTING_LINK_TYPES = ("LC", "LU")

# The Compustat items each accounting characteristic reads, in the order of use.
# Book equity: stockholders' equity (`seq`, else `ceq` plus `pstk`, else `at`
# minus `lt`), plus deferred taxes and investment tax credit (`txditc`, zero when
# missing), minus preferred stock (`pstkrv`, else `pstkl`, else `pstk`, zero when
# missing). A book equity that is not positive is missing, and counted.
BOOK_EQUITY_STOCKHOLDERS = (("seq",), ("ceq", "pstk"), ("at", "lt"))
BOOK_EQUITY_DEFERRED_TAXES = "txditc"
BOOK_EQUITY_PREFERRED = ("pstkrv", "pstkl", "pstk")
MARKET_VALUE_AT_FISCAL_YEAR_END = ("prcc_f", "csho")   # price at the fiscal year end times shares outstanding
EARNINGS = "ib"
REVENUE = "revt"
OPERATING_COSTS = ("cogs", "xsga", "xint")   # revenue and at least one of the three must be present, and a missing cost among the others counts as zero
TOTAL_ASSETS = "at"

# ---------------------------------------------------------------------------
# 5. The transform and the look-ahead test
# ---------------------------------------------------------------------------

# Within each month, across the members for which a characteristic is present,
# the characteristic is ranked and mapped to the interval from -1 to 1, ties at
# their average rank. A missing value is set to zero after the mapping, which is
# the month's median, and the share missing is counted by characteristic and
# year before the fill. This is the treatment of Gu, Kelly and Xiu (2020).
RANK_LOW = -1.0
RANK_HIGH = 1.0
RANK_TIES = "average"
FILL_VALUE = 0.0

# The look-ahead test: for 24 months drawn by the seed from the panel's 408, the
# characteristics of month t are built again from inputs cut at the last trading
# day of month t-1. Every row dated later is removed from the monthly file, the
# daily file, the list and the links, and every annual record less than six
# months old on that day is removed. The rows must equal the full build's rows
# to the tolerance.
LOOKAHEAD_MONTHS = 24
LOOKAHEAD_TOLERANCE = 1e-12

# ---------------------------------------------------------------------------
# 6. Gate G1, the panel
# ---------------------------------------------------------------------------

# (a) The copied data layer reproduces the three reference tables of section 2
#     to the last digit: the member-months, the count check by year and the
#     delisting counts by year.
# (b) In every month from January 1991 to December 2024, each characteristic is
#     present, before the fill, for at least 90 per cent of the month's members,
#     and book-to-market and operating profitability for at least 80 per cent.
#     Those two are missing by definition when book equity is not positive.
# (c) The look-ahead test of section 5 passes on its 24 months.
# On a failure nothing else runs.
G1B_COVERAGE_MIN = 0.90
G1B_COVERAGE_MIN_BOOK_EQUITY = 0.80
G1B_BOOK_EQUITY_CHARACTERISTICS = ("book_to_market", "operating_profitability")
G1B_FIRST_MONTH = "1991-01"
G1B_LAST_MONTH = "2024-12"

# ---------------------------------------------------------------------------
# 7. The split
# ---------------------------------------------------------------------------

# For each test year y from 2006 to 2024: the training sample is January 1991 to
# December of year y-6, the validation sample is the five years y-5 to y-1, and
# the test sample is year y. A model is fitted on the training sample only, its
# configuration is chosen on the validation sample by the mean squared error, and
# the chosen fit forecasts the test year. There is no refit on the training and
# validation samples together. After Gu, Kelly and Xiu (2020), reduced to the
# data held.
TRAIN_FIRST_YEAR = 1991
VALIDATION_YEARS = 5
TEST_YEARS = tuple(range(2006, 2025))
REFITS = 19
FIRST_TRAINING_MONTHS = 120      # 1991 to 2000, for the test year 2006
LAST_TRAINING_MONTHS = 336       # 1991 to 2018, for the test year 2024
REFIT_ON_TRAINING_AND_VALIDATION = False
TUNING_CRITERION = "mean squared error on the validation sample"

# ---------------------------------------------------------------------------
# 8. The six models and the 32 configurations
# ---------------------------------------------------------------------------

# All six in scikit-learn, all with the squared-error loss, all seeded. The
# counts are fixed here. The values of the configurations are proposed in
# milestone 2 on made-up data of the panel's shape, with Table A.5 of Gu, Kelly
# and Xiu (2020) as the model, and set in a commit of their own before any model
# is fitted on the panel.
MODELS = ("ols3", "ols12", "elastic_net", "random_forest", "boosted_trees", "network")
CONFIGURATION_COUNTS = {
    "ols3": 1,               # least squares on size, book-to-market and momentum: the benchmark of Gu, Kelly and Xiu (2020)
    "ols12": 1,              # least squares on the twelve characteristics
    "elastic_net": 12,       # three lasso shares by four penalties
    "random_forest": 6,      # three depths by two counts of characteristics tried at each split
    "boosted_trees": 8,      # two depths by two learning rates by two counts of trees
    "network": 4,            # two weight penalties by two counts of passes over the data
}
CONFIGURATIONS = 32
LOSS = "squared error"

OLS3_CHARACTERISTICS = ("size", "book_to_market", "momentum")

ENET_LASSO_SHARES = (0.0, 0.5, 1.0)    # 0 is ridge, 1 is the lasso
ENET_PENALTY_COUNT = 4
ENET_PENALTY_VALUES = None             # four values, set in milestone 2

RF_TREES = 300                         # regression trees on bootstrap samples, averaged
RF_DEPTH_COUNT = 3
RF_DEPTH_VALUES = None                 # three values, set in milestone 2
RF_CHARACTERISTICS_PER_SPLIT_COUNT = 2
RF_CHARACTERISTICS_PER_SPLIT_VALUES = None   # two values, set in milestone 2

GBT_DEPTHS = (1, 2)                    # shallow trees fitted one after another to the errors left so far, the histogram version
GBT_LEARNING_RATE_COUNT = 2
GBT_LEARNING_RATE_VALUES = None        # two values, set in milestone 2
GBT_TREE_COUNT_COUNT = 2
GBT_TREE_COUNT_VALUES = None           # two values, set in milestone 2

NN_HIDDEN_LAYERS = 1
NN_HIDDEN_UNITS = 32                   # the smallest network of Gu, Kelly and Xiu (2020)
NN_SEEDS_PER_FORECAST = 5              # each forecast is the mean over five seeds
NN_WEIGHT_PENALTY_COUNT = 2
NN_WEIGHT_PENALTY_VALUES = None        # two values, set in milestone 2
NN_PASSES_COUNT = 2
NN_PASSES_VALUES = None                # two values, set in milestone 2

# A tuned model is a family with the configuration its validation sample chose
# that year: six series. A fixed configuration is one configuration held for all
# 19 years: 32 series, the trials of section 11. Each configuration is fitted at
# every refit and forecasts every test year, whether or not validation chose it.
TUNED_MODELS = 6
FIXED_CONFIGURATIONS = 32

# ---------------------------------------------------------------------------
# 9. The measures of the forecasts and the portfolios
# ---------------------------------------------------------------------------

# The out-of-sample R² of Gu, Kelly and Xiu (2020): one minus the sum of
# squared forecast errors over the sum of squared excess returns, pooled over the
# member-months of the test window, in per cent a month. The denominator is not
# demeaned, so the benchmark is a forecast of zero. Beside every R² stands
# that of the pooled-mean forecast: one number per refit, the mean excess return
# of the training sample's member-months.
R2_IN_PER_CENT = True
R2_DENOMINATOR_DEMEANED = False

NEWEY_WEST_LAGS = 12             # for the mean monthly rank correlation and the Diebold-Mariano statistic
DM_BENCHMARK = "ols3"            # each tuned model is compared with OLS-3: five statistics with a verdict (E3)
DM_CRITICAL_VALUE = 1.96         # the absolute value E3 compares each statistic with

# Families of tests read together. Holm's adjusted value stands beside the plain
# one in each. Set on 19 November 2026.
HOLM_FAMILIES = {
    "diebold_mariano": 5,        # the five tuned models against OLS-3
    "rank_correlation": 6,       # the six tuned models' mean rank correlations
    "sharpe_difference": 5,      # the five intervals of E6
}

# Importance, the first measure of Gu, Kelly and Xiu (2020): the fall in the
# R² on the training sample when one characteristic is set to zero and
# the fit is left as it is, at each refit, averaged over the 19 refits, negative
# values set to zero, scaled to sum to one within a model.
IMPORTANCE_NEGATIVE_TO_ZERO = True

# The portfolios. Each month the members with a forecast are sorted into deciles
# on it, ties broken by PERMNO. The long-short portfolio holds the top decile and
# is short the bottom decile, each side weighted by start-of-month market
# capitalisation, with the equal-weighted version beside it. Turnover is one-way,
# each side against its drifted weights, summed over the two sides.
DECILES = 10
COST_BP = 10                     # basis points per unit of one-way turnover
COST_BP_SENSITIVITY = 20         # the one sensitivity
MONTHS_PER_YEAR = 12             # a Sharpe ratio is the monthly mean over the monthly standard deviation, times the square root of 12

# The interval for the difference of two Sharpe ratios after Ledoit and Wolf
# (2008), by the block bootstrap copied from Signal Evaluation. A bootstrap
# sample is 19 blocks of 12 months, 228 months long.
BOOTSTRAP_BLOCK_MONTHS = 12
BOOTSTRAP_DRAWS = 2000
INTERVAL_LEVEL = 0.95
BOOTSTRAP_SOURCE_REPOSITORY = "github.com/ochofer/volatility-managed-portfolios"
BOOTSTRAP_SOURCE_COMMIT = "74ed6d1"
BOOTSTRAP_FILE = "src/sq/bootstrap.py"
BOOTSTRAP_SHA256 = "6b713dce9b1f969e3862acf5fedf2b4f55f88251c963af613a09fdbfdc9ef89e"   # at BOOTSTRAP_SOURCE_COMMIT, copied into src/mlrp/source/bootstrap.py. Set on 20 November 2026

# The size checks of the three tests that expectations E3, E4 and E6 read: the
# rejection rate of each at 228 months on made-up data, and the difference each
# detects at this power. The bootstrap's are run at blocks of 12, 6 and 3 months.
# The other settings of the checks are proposed in milestone 2 and set in a
# commit of their own before they run. Set on 19 November 2026.
SIZE_CHECK_POWER = 0.80
SIZE_CHECK_BOOTSTRAP_BLOCKS = (12, 6, 3)
SIZE_CHECK_SETTINGS = None       # set in milestone 2

# ---------------------------------------------------------------------------
# 10. Gate G2, the benchmark's order of magnitude
# ---------------------------------------------------------------------------

# Over the test window, OLS-3's out-of-sample R² lies between -0.5 and
# +1.5 per cent a month, and it differs from the pooled-mean forecast's by at
# most 0.5 percentage points. Gu, Kelly and Xiu (2020, Table 1) report 0.31 per
# cent a month for OLS-3 with Huber's loss, fitted on all their stocks and scored
# on the 1,000 largest. Three slopes on ranks cannot move the R² far from the
# constant's. A return leaked into a characteristic gives an R² above that
# range. On a failure the panel is counted again against gate G1's tables.
G2_OLS3_R2_RANGE = (-0.5, 1.5)               # per cent a month
G2_OLS3_MINUS_POOLED_MEAN_MAX = 0.5          # percentage points, in absolute value

# ---------------------------------------------------------------------------
# 11. The overfitting statistics
# ---------------------------------------------------------------------------

# A trial is the value-weighted long-short return, net of cost, of one fixed
# configuration over the 228 test months. The best trial is the one with the
# highest Sharpe ratio over the whole test window.
TRIALS = 32

# The deflated Sharpe ratio of Bailey and López de Prado (2014) of the best
# trial, with 32 as the number of trials, the variance of the 32 Sharpe ratios as
# the variance across trials, and the best trial's own skewness and kurtosis.
# Counting all 32 as independent is the conservative choice, since the
# configurations of one family move together.
DSR_TRIAL_COUNT = 32
DSR_LEVEL = 0.95                 # the level gate G3 (b) and expectation E8 read

# The probability of backtest overfitting of Bailey, Borwein, López de Prado and
# Zhu (2017), by combinatorially symmetric cross-validation on the matrix of 228
# months by 32 trials: 12 blocks of 19 months, every choice of six blocks as the
# in-sample half (924 splits), the Sharpe ratio as the measure. The statistic is
# the share of splits in which the in-sample best trial ranks in the lower half
# out of sample.
PBO_BLOCKS = 12
PBO_BLOCK_MONTHS = 19
PBO_SPLITS = 924

# Purged and embargoed cross-validation, a second way to choose a configuration,
# for the elastic net and the boosted trees: five folds contiguous in time over
# the months of the training and validation samples, with the 12 months after
# each test fold kept out of the fit, the longest window inside a characteristic.
CV_FOLDS = 5
CV_EMBARGO_MONTHS = 12
CV_MODELS = ("elastic_net", "boosted_trees")

# ---------------------------------------------------------------------------
# 12. Gate G3, the known-answer simulation
# ---------------------------------------------------------------------------

# On independent normal returns, at the study's own 228 months, 32 trials and 12
# blocks. Sharpe ratios are monthly. No statistic of section 11 is computed on
# the study's returns before all four clauses pass.
# (a) With zero means, the mean over 4,000 repetitions of the highest Sharpe
#     ratio is within 5 per cent of the expected maximum of Bailey and Lopez de
#     Prado (2014) at a variance of 1/(T-1).
# (b) The best trial's deflated Sharpe ratio is above 0.95 in at most 5 per cent
#     of those repetitions, and its median lies between 0.40 and 0.60.
# (c) The mean probability of backtest overfitting over 1,000 repetitions lies
#     between 0.45 and 0.55.
# (d) With one trial given a true Sharpe ratio of 1.0 a year, that trial has the
#     highest Sharpe ratio over the 228 months, as section 11 defines the best
#     trial, in at least 90 per cent of 1,000 repetitions, and the mean
#     probability of backtest overfitting is below 0.25.
#     The words of the first part of (d) were written on 20 November 2026, after
#     the run of 19 November. Its level stands as committed before the run.
# On a failure the statistic's code is corrected, each change recorded, the same
# seed run again, and no level changes.
G3_MONTHS = 228
G3_TRIALS = 32
G3_BLOCKS = 12
G3_REPETITIONS_A_B = 4000
G3_REPETITIONS_C_D = 1000
G3_A_TOLERANCE = 0.05                # relative
G3_B_SHARE_ABOVE_LEVEL_MAX = 0.05    # share of repetitions with the best trial's deflated Sharpe ratio above DSR_LEVEL
G3_B_MEDIAN_RANGE = (0.40, 0.60)
G3_C_PBO_RANGE = (0.45, 0.55)
G3_D_ANNUAL_SHARPE = 1.0             # the planted trial's true Sharpe ratio a year
G3_D_BEST_SHARE_MIN = 0.90
G3_D_PBO_MAX = 0.25

# Beside the four clauses, and read by no level: three sets of figures added on
# 24 November 2026, after the simulation's results were known.
# - The deflated Sharpe ratio of the best trial on clause (d)'s own repetitions.
# - The detectable effects: one trial with each of these true Sharpe ratios a
#   year and 31 trials without one, at the study's months, trials and blocks,
#   with G3_DETECTABLE_REPETITIONS repetitions for each.
# - The bound beside clause (d): its share over G3_BOUND_REPETITIONS further
#   repetitions of clause (d)'s setting.
G3_DETECTABLE_ANNUAL_SHARPES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
G3_DETECTABLE_REPETITIONS = 1000
G3_BOUND_REPETITIONS = 20_000

# ---------------------------------------------------------------------------
# 13. The expectations, over the test window
# ---------------------------------------------------------------------------

# E1: the tuned elastic net's R² is within 0.10 percentage points of
#     OLS-12's. Twelve characteristics on at least 60,000 training rows leave
#     little to shrink.
E1_MAX_DIFFERENCE = 0.10                 # percentage points, in absolute value
# E2: at least two of the three non-linear models (random forest, boosted trees,
#     network) have a higher R² than the tuned elastic net. Table 1 of
#     Gu, Kelly and Xiu (2020) on the 1,000 largest stocks gives the direction.
E2_NONLINEAR_MODELS = ("random_forest", "boosted_trees", "network")
E2_MIN_ABOVE = 2
# E3: none of the five Diebold-Mariano statistics against OLS-3 exceeds 1.96 in
#     absolute value (DM_CRITICAL_VALUE).
E3_MAX_BEYOND = 0
# E4: every tuned model's mean monthly rank correlation is below 0.05, and for at
#     least four of the six it is within two Newey-West standard errors of zero.
#     Two standard errors are a distance and not a test level.
E4_MEAN_RANK_CORRELATION_MAX = 0.05
E4_STANDARD_ERRORS = 2.0
E4_MIN_WITHIN = 4
# E5: in each of the three non-linear models, the eight characteristics of
#     E5_GROUPS carry more than two thirds of the importance, their share by
#     count.
E5_SHARE_MIN = E5_SHARE_BY_COUNT
# E6: for each of the five tuned models other than OLS-3, the 95 per cent
#     interval for the difference between its net long-short Sharpe ratio and
#     OLS-3's includes zero. Avramov, Cheng and Metzker (2023) find the profits of
#     machine-learning forecasts mainly in microcaps, distressed and
#     hard-to-arbitrage stocks, falling under trading costs. The S&P 500 holds no
#     microcap and no distressed stock.
E6_MODELS = ("ols12", "elastic_net", "random_forest", "boosted_trees", "network")
# E7: each non-linear model's long-short turnover is above OLS-3's.
E7_MODELS = E2_NONLINEAR_MODELS
# E8: the best trial's deflated Sharpe ratio is below DSR_LEVEL.
#     How often E8 is met with one trial of a given true Sharpe ratio a year, on
#     made-up returns: results/simulation/g3_detectable_effects.csv (notebook 12,
#     step 6). Its verdict carries that table's figures at 0.5 a year. The
#     pointer was added on 24 November 2026.
# E9: the probability of backtest overfitting is at least 0.25.
#     How often E9 is met with one trial of a given true Sharpe ratio a year, on
#     made-up returns: the same table. Its verdict carries that table's figures
#     at 0.5 a year. The pointer was added on 24 November 2026.
E9_PBO_MIN = 0.25
# E10: for the elastic net and the boosted trees, choosing the configuration by
#      purged and embargoed cross-validation changes the R² by less than
#      0.10 percentage points against the validation sample's choice.
E10_MAX_DIFFERENCE = 0.10                # percentage points, in absolute value

# ---------------------------------------------------------------------------
# 14. The seed
# ---------------------------------------------------------------------------

# One seed for the study. Each use draws from its own stream, so that a change in
# one use leaves the draws of the others as they were:
# numpy.random.default_rng([SEED, SEED_STREAMS[name]]). Set on 19 November 2026.
SEED = 20261119
SEED_STREAMS = {
    "lookahead_months": 1,       # the 24 months of the look-ahead test
    "known_answer": 2,           # gate G3's simulation
    "models": 3,                 # the forests, the boosted trees and the network
    "bootstrap": 4,              # the block bootstrap of E6
    "made_up_data": 5,           # the tests' and the proposals' made-up data
    "size_checks": 6,            # the size checks of E3, E4 and E6
}

# Gate G3's simulation draws each clause from a sub-stream of its stream:
# numpy.random.default_rng([SEED, SEED_STREAMS["known_answer"],
#                           KNOWN_ANSWER_SUB_STREAMS[clause]]),
# so that each clause's draws are the same whatever the number of repetitions of
# the others. Set with the simulation's code on 19 November 2026.
KNOWN_ANSWER_SUB_STREAMS = {"a_b": 0, "c": 1, "d": 2}

# The figures beside gate G3 that need draws of their own take them from
# sub-streams of the same stream, numbered after the clauses':
# numpy.random.default_rng([SEED, SEED_STREAMS["known_answer"],
#                           BESIDE_G3_SUB_STREAMS[name], *position]),
# where the detectable effects add the position of the row's true Sharpe ratio
# in G3_DETECTABLE_ANNUAL_SHARPES, so that each row's draws are the same whatever
# the other rows. The deflated Sharpe ratio beside clause (d) is computed on
# clause (d)'s own draws, sub-stream 2, and takes none of its own. Set on
# 24 November 2026.
BESIDE_G3_SUB_STREAMS = {"detectable_effects": 3, "bound": 4}
