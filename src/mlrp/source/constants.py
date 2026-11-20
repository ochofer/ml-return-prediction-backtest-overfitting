"""
The source repository's values at commit 0dabbb0, kept so that the copies import, and not this study's
parameters. The seven modules copied from Stock-Level Tracking into this folder read their constants by the
source's names; this file holds every name they reference and no other, each at its value in src/slt/constants.py
at that commit (SHA-256 constants.SOURCE_CONSTANTS_SHA256 in this package's parent), with the source line beside
it. bootstrap.py reads no constant. No module outside this folder reads this file: the study's own modules pass
every parameter from src/mlrp/constants.py, through src/mlrp/data.py where they call a copy. Where a name here
names something this study also sets (DATA_ROOT_ENV, MANIFEST_SHA256, RESULTS_DIR, TABLES), the study's value is
the one in src/mlrp/constants.py. tests/test_source_copies.py checks the names against the copies' text and, where
a clone of the source repository sits beside this one, the values against the source's file at the commit.
"""

# Read by extraction.py
DATA_ROOT_ENV = "SLT_DATA_ROOT"                            # line 42
EXTRACTION_DATE = "2026-09-28"                             # line 44
EXTRACTION_SOURCE = "CRSP and Compustat, via Wharton Research Data Services, accessed through Queen Mary University of London" # line 45
GICS_FALLBACK = "comp_company"                             # line 216
GICS_SOURCE = "comp_co_hgic"                               # line 215
MANIFEST_FILE = "manifest.json"                            # line 47
MANIFEST_SHA256 = "23e96d189f4b095936f47a1ebacb4e987fb260726afab4b6e6400c25ebce8693" # line 51
MANIFEST_SHA256_2026_09_28 = "8b2e6b6bf762204bb2c4f70b1a53c66d527d02d2905a540caa53275a54de2b22" # line 51
RESULTS_DIR = "results/20260928"                           # line 978
TABLES = {"crsp_dsp500list": "crsp.dsp500list", "crsp_msp500": "crsp.msp500", "crsp_msf": "crsp.msf", "crsp_msedelist": "crsp.msedelist", "crsp_msenames": "crsp.msenames", "crsp_dsf": "crsp.dsf", "crsp_ccm_lnkhist": "crsp.ccmxpf_lnkhist", "comp_idxcst_his": "comp.idxcst_his", "comp_co_hgic": "comp.co_hgic", "comp_company": "comp.company"} # line 68
TIMEZONE = "Europe/Amsterdam"                              # line 1057

# Read by universe.py
CROSS_CHECK_JOIN_WITHIN_MONTHS = 1                         # line 130
CRSP_END = "2024-12-31"                                    # line 52
G2_USABLE_YEARS_MIN = 15                                   # line 238
SHARE_CODES_COMMON = (10, 11)                              # line 131
SP500_GVKEYX = "000003"                                    # line 113
USABLE_COUNT_WITHIN = 10                                   # line 112

# Read by returns.py; CRSP_END above is read by it too
DELIST_OTHER_MISSING_RETURN = 0.0                          # line 146
DELIST_PERFORMANCE_CODES = (400, 599)                      # line 140
DELIST_PERFORMANCE_MISSING_RETURN = -0.3                   # line 145
G3_BEFORE_WINDOW = "share with a price at the end of the removal month or the month before, monthly file, no verdict" # line 247
G3_SHARE_MIN = 0.9                                         # line 240
G3_TRADING_DAYS = 10                                       # line 239
G3_WINDOW = ("1990-01-02", "2024-12-31")                   # line 246
G4_MONTH_SHARE_MIN = 0.9                                   # line 248
G4_NAME_SHARE_MIN = 0.9                                    # line 249

# Read by sectors.py
EVAL_END = "2024-12"                                       # line 497
EVAL_GICS_CAP_SHARE_MIN = 0.95                             # line 506
EVAL_GICS_MEMBER_SHARE_MIN = 0.95                          # line 505
EVAL_START = "1999-07"                                     # line 496
GICS_SECTORS = {"10": "Energy", "15": "Materials", "20": "Industrials", "25": "Consumer Discretionary", "30": "Consumer Staples", "35": "Health Care", "40": "Financials", "45": "Information Technology", "50": "Communication Services", "55": "Utilities", "60": "Real Estate"} # line 225
LINK_PRIMARY_FLAGS = ("P", "C")                            # line 196
LINK_TYPES = ("LC", "LU")                                  # line 195
SECTOR_LINK_FLAGS = ("P", "C", "J", "N")                   # line 207
SMALL_CELL_MAX_MEMBER_SHARE = 0.8                          # line 1003
SMALL_CELL_MIN_MEMBERS = 3                                 # line 1002
UNCLASSIFIED = "Unclassified"                              # line 230

# Read by sorts.py
E3_CORRELATION_MIN = 0.8                                   # line 346
E3_MEAN_DIFF_SE = 2.0                                      # line 347
E3_NEWEY_WEST_LAGS_REPORTED = 12                           # line 400
MOMENTUM_FIRST_MONTH = "1927-01"                           # line 367
MOMENTUM_MONTHS = (12, 2)                                  # line 355
VARIANCE_DAYS = 60                                         # line 356
VARIANCE_MIN_DAYS = 20                                     # line 371
VARIANCE_SORT_START = "1990-04"                            # line 357

# Read by french.py; CRSP_END, RESULTS_DIR, TIMEZONE above are read by it too
E3_FRENCH_LARGEST_QUINTILE = {"momentum": ("BIG LoPRIOR", "ME5 PRIOR2", "ME5 PRIOR3", "ME5 PRIOR4", "BIG HiPRIOR"), "variance": ("BIG LoVAR", "ME5 VAR2", "ME5 VAR3", "ME5 VAR4", "BIG HiVAR")} # line 388
E3_FRENCH_SPREADS = {"momentum": ("me_prior", "BIG HiPRIOR", "BIG LoPRIOR"), "variance": ("me_var", "BIG HiVAR", "BIG LoVAR")} # line 384
FACTOR_MODEL_FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom") # line 516
FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" # line 88
FRENCH_BLOCK_FIRMS = "Number of Firms in Portfolios"       # line 101
FRENCH_BLOCK_RETURNS = "Value Weighted Returns -- Monthly" # line 100
FRENCH_CACHE_DIR = "data/french"                           # line 99
FRENCH_FILES = {"me_prior": "25_Portfolios_ME_Prior_12_2_CSV.zip", "me_var": "25_Portfolios_ME_VAR_5x5_CSV.zip", "factors5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "momentum_daily": "F-F_Momentum_Factor_daily_CSV.zip", "factors5_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip", "momentum_monthly": "F-F_Momentum_Factor_CSV.zip"} # line 89
FRENCH_MISSING_CODES = (-99.99, -999.0)                    # line 97
FRENCH_RETURNS_ARE_PERCENT = True                          # line 98

# Read by layer.py; CRSP_END above is read by it too
DAILY_FILE_START = "1990-01-02"                            # line 65
