"""
Code copied verbatim from two other repositories, so that this study runs on the same data layer and the same
bootstrap as the studies that built and tested them.

- extraction.py, universe.py, returns.py, sectors.py, sorts.py, french.py and layer.py: Stock-Level Tracking,
  github.com/ochofer/stock-level-tracking, src/slt/ at commit 0dabbb0. They read the extraction in place, build
  the S&P 500's point-in-time members and their monthly returns with the delisting rule, link each member to its
  Compustat company, compute momentum and the Newey-West standard error, and load French's files.
- bootstrap.py: Signal Evaluation, github.com/ochofer/volatility-managed-portfolios, src/sq/bootstrap.py at commit
  74ed6d1. It gives the interval for the difference of two Sharpe ratios after Ledoit and Wolf (2008).

Each file is the source file unchanged, byte for byte, and tests/test_source_copies.py hashes it against
constants.COPIED_FILES and constants.BOOTSTRAP_SHA256. The seven copies from Stock-Level Tracking import their
constants as `from . import constants as C`, by the source's names, so constants.py in this folder holds those
names and nothing else, each at the source's value. The study's own modules call the copies through
src/mlrp/data.py, with the study's own settings, and never read the names in this folder's constants.py.
"""
