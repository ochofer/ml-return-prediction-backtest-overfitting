"""
The study's access to the extraction and to French's library, through the copied data layer.

The modules in src/mlrp/source/ are Stock-Level Tracking's, copied unchanged. Three of their assumptions do not
hold in this repository, and this module wraps the copies so that none of them is edited:

- Where files go. The copied extraction.repo_root() is the folder two levels above its own file, which for a copy
  in src/mlrp/source/ is src/. repo_root() here is the repository's root, and the results and French's files are
  written under it.
- The data root. The copies read the extraction's folder from the source study's environment variable.
  data_root() here reads this study's own, constants.DATA_ROOT_ENV.
- The tables. The copies' lists of columns cover the source study's tables, and Compustat's annual fundamentals
  are not among them. COLUMNS here lists the columns this study reads from each of its eight tables, and
  read_table() reads them.

Every run opens the extraction with open_extraction(). It checks the manifest's hash against
constants.MANIFEST_SHA256 with the copied check, then the eight tables' entries in the manifest against
constants.MANIFEST_ENTRIES_SHA256, and reads nothing when either differs. The functions that write to the
repository are written to receive metadata and aggregates, as the data rules require. The frames with one row
per security that one notebook builds and a later one reads are kept in constants.LOCAL_DIR, which git ignores.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess

from . import constants as C
from .source import extraction as X
from .source import french as F

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The Compustat items the four accounting characteristics read (constants, section 4), each once, in the order
# the constants name them.
FUNDAMENTALS_ITEMS = tuple(dict.fromkeys(
    [item for group in C.BOOK_EQUITY_STOCKHOLDERS for item in group]
    + [C.BOOK_EQUITY_DEFERRED_TAXES, *C.BOOK_EQUITY_PREFERRED, *C.MARKET_VALUE_AT_FISCAL_YEAR_END,
       C.EARNINGS, C.REVENUE, *C.OPERATING_COSTS, C.TOTAL_ASSETS]))

# The columns this study reads, by the extraction's key. For the seven CRSP tables, the copied module's lists,
# with the monthly share volume added for dollar volume. For Compustat's annual fundamentals, the company, the
# fiscal year end and the fiscal year, then the items above.
COLUMNS = {
    "crsp_dsp500list": X.COLUMNS["crsp_dsp500list"],
    "crsp_msp500": X.COLUMNS["crsp_msp500"],
    "crsp_msf": X.COLUMNS["crsp_msf"] + ("vol",),
    "crsp_dsf": X.COLUMNS["crsp_dsf"],
    "crsp_msedelist": X.COLUMNS["crsp_msedelist"],
    "crsp_msenames": X.COLUMNS["crsp_msenames"],
    "crsp_ccm_lnkhist": X.COLUMNS["crsp_ccm_lnkhist"],
    "comp_funda": ("gvkey", "datadate", "fyear") + FUNDAMENTALS_ITEMS,
}

# Date columns this study reads that the copied module does not parse: the fiscal year end of the fundamentals.
DATE_COLUMNS = ("datadate",)

NOT_A_TABLE = "{key} is not one of the study's tables: {keys}."
WRONG_ENTRIES = ("The manifest's entries for the study's {m} tables have SHA-256 {disk} over {n} files, and the "
                 "constants fix {fixed} over {fixed_n}: a table the study reads is not the one the constants "
                 "record, so nothing is read.")


def repo_root() -> pathlib.Path:
    """The repository's root folder, wherever the notebook is opened from."""
    return ROOT


def data_root(environ=None) -> pathlib.Path:
    """The extraction's folder, from the study's environment variable. Stops with one sentence when the variable
    is not set or the folder holds no manifest."""
    env = os.environ if environ is None else environ
    value = env.get(C.DATA_ROOT_ENV, "").strip()
    if not value:
        raise SystemExit(X.MISSING_ROOT.format(env=C.DATA_ROOT_ENV))
    root = pathlib.Path(value).expanduser()
    if not (root / C.MANIFEST_FILE).is_file():
        raise SystemExit(X.MISSING_MANIFEST.format(env=C.DATA_ROOT_ENV, root=root))
    return root


def manifest_entries(manifest: dict, keys=None) -> list:
    """[file, rows, SHA-256] of every file of the study's tables in the manifest, sorted by key and file."""
    keys = tuple(C.TABLES) if keys is None else tuple(keys)
    files = sorted((f for f in manifest["files"] if f["key"] in keys), key=lambda f: (f["key"], f["file"]))
    return [[f["file"], int(f["rows"]), f["sha256"]] for f in files]


def entries_digest(entries: list) -> str:
    """The SHA-256 of the entries written as compact JSON, with no space after a comma or a colon."""
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def check_entries(manifest: dict) -> list:
    """The study's entries in the manifest, after checking their digest and their count against the constants.
    Stops with one sentence when either differs."""
    entries = manifest_entries(manifest)
    digest = entries_digest(entries)
    if digest != C.MANIFEST_ENTRIES_SHA256 or len(entries) != C.MANIFEST_ENTRIES:
        raise SystemExit(WRONG_ENTRIES.format(m=len(C.TABLES), disk=digest[:12], n=len(entries),
                                              fixed=C.MANIFEST_ENTRIES_SHA256[:12], fixed_n=C.MANIFEST_ENTRIES))
    return entries


def open_extraction(environ=None) -> tuple:
    """The extraction's folder and its manifest, after two checks. The copied check compares the manifest's hash
    with the study's constant, MANIFEST_SHA256, and with the manifest's own .sha256 file. The second compares the
    digest of the study's eight tables' entries with MANIFEST_ENTRIES_SHA256."""
    root = data_root(environ)
    manifest = X.check_manifest(root, expected=C.MANIFEST_SHA256)
    check_entries(manifest)
    return root, manifest


def files_used(manifest: dict) -> list:
    """One record per file of the study's eight tables, sorted by path: the manifest's metadata and nothing else."""
    return X.files_used(manifest, keys=tuple(C.TABLES))


def read_table(root: pathlib.Path, key: str, columns=None, years=None) -> tuple:
    """One of the study's tables read from the extraction: only the columns asked for (COLUMNS[key] when none
    are), with its dates parsed. For the daily file, years limits the yearly files read.

    Returns the frame and a dict of counts, by column, of date values that were present and did not parse. A
    date that does not parse becomes missing and is counted, as in the copied module.
    """
    import pandas as pd
    if key not in C.TABLES:
        raise SystemExit(NOT_A_TABLE.format(key=key, keys=", ".join(C.TABLES)))
    columns = list(COLUMNS[key] if columns is None else columns)
    frame, unparsed = X.read_table(root, key, columns=columns, years=years)
    for c in DATE_COLUMNS:
        if c in frame.columns and not pd.api.types.is_datetime64_any_dtype(frame[c]):
            raw = frame[c]
            parsed = pd.to_datetime(raw, format="ISO8601", errors="coerce")
            bad = int((raw.notna() & parsed.isna()).sum())
            if bad:
                unparsed[c] = bad
            frame[c] = parsed
    return frame, unparsed


def missing_columns(root: pathlib.Path, records: list) -> dict:
    """Columns the study reads that a file's schema lacks, by file. Empty when every one is present."""
    import pyarrow.parquet as pq
    out = {}
    for r in records:
        have = set(pq.read_schema(root / r["file"]).names)
        lacking = [c for c in COLUMNS[r["key"]] if c not in have]
        if lacking:
            out[r["file"]] = lacking
    return out


def results_dir() -> pathlib.Path:
    """The folder of the committed aggregates, under the repository's root."""
    return ROOT / C.RESULTS_DIR


def write_results(name: str, frame=None, record: dict = None, folder=None) -> pathlib.Path:
    """Write one committed result: a table as CSV or a record as JSON, in the results folder unless another
    folder is given."""
    folder = results_dir() if folder is None else pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if frame is not None:
        frame.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
    else:
        path.write_text(json.dumps(record, indent=1) + "\n")
    return path


def read_result(name: str, folder=None) -> dict:
    """A record an earlier notebook committed, read back. Stops with one sentence when it is not there."""
    folder = results_dir() if folder is None else pathlib.Path(folder)
    path = folder / name
    if not path.exists():
        raise SystemExit(f"{name} is not in {C.RESULTS_DIR}: run the notebooks in order, from 01.")
    return json.loads(path.read_text())


def provenance(manifest: dict, records: list, written_by: str) -> dict:
    """The provenance record of the study's results: the extraction they were built from, its two checks, the
    commits the copied code comes from and the versions of the tools. It holds no row of any licensed table."""
    return {
        "source": C.EXTRACTION_SOURCE,
        "extraction_date": C.EXTRACTION_DATE,
        "manifest_sha256": C.MANIFEST_SHA256,
        "manifest_files": int(manifest["counts"]["files"]),
        "manifest_rows": int(manifest["counts"]["rows"]),
        "manifest_entries_sha256": entries_digest(manifest_entries(manifest)),
        "fetch_dates": X.fetch_dates(records),
        "tables_used": sorted({r["key"] for r in records}),
        "files_used": len(records),
        "rows_used": sum(r["rows"] for r in records),
        "copied_from": {C.SOURCE_REPOSITORY: C.SOURCE_COMMIT,
                        C.BOOTSTRAP_SOURCE_REPOSITORY: C.BOOTSTRAP_SOURCE_COMMIT},
        "versions": X.versions(),
        "written_by": written_by,
        "written_at": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
    }


def local_dir() -> pathlib.Path:
    """The folder of the frames with one row per security kept between notebooks, under the repository's root
    and ignored by git. It exists only on the machine that holds the extraction."""
    return ROOT / C.LOCAL_DIR


def write_local(name: str, frame) -> pathlib.Path:
    """Keep one frame with one row per security in local_dir(), as parquet, for a later notebook."""
    folder = local_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    frame.to_parquet(path, index=False)
    return path


def read_local(name: str):
    """A frame an earlier notebook kept in local_dir(). Stops with one sentence when it is not there."""
    import pandas as pd
    path = local_dir() / name
    if not path.exists():
        raise SystemExit(f"{name} is not in {C.LOCAL_DIR}: run the notebooks in order, from 01.")
    return pd.read_parquet(path)


# Clones of the source repository that a machine may hold beside this one, by folder name.
SOURCE_CLONES = ("B-S", "stock-level-tracking")


def reference_table(path: str) -> tuple:
    """One of gate G1 (a)'s three reference tables, the bytes of `path` in the source repository at
    constants.SOURCE_COMMIT. It is fetched from GitHub by the commit's full hash, which names one version of the
    file for good. Where GitHub cannot be reached, it is read with git from a clone of the source repository
    beside this one. Returns the bytes and where they were read."""
    repository = C.SOURCE_REPOSITORY.split("/", 1)[1]
    url = f"https://raw.githubusercontent.com/{repository}/{C.SOURCE_COMMIT}/{path}"
    try:
        import requests
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            return response.content, url
    except Exception:
        pass
    for name in SOURCE_CLONES:
        clone = ROOT.parent / name
        if (clone / ".git").exists():
            r = subprocess.run(["git", "show", f"{C.SOURCE_COMMIT}:{path}"], cwd=clone, capture_output=True)
            if r.returncode == 0:
                return r.stdout, f"{clone.name} at {C.SOURCE_COMMIT[:7]}"
    raise SystemExit(f"{path} at {C.SOURCE_COMMIT[:7]} could not be read from GitHub or from a clone beside this one.")


def french_folder() -> pathlib.Path:
    """The folder that holds French's downloaded files, under the repository's root and ignored by git."""
    return ROOT / C.FRENCH_CACHE_DIR


def french_daily(folder=None) -> tuple:
    """French's daily factors and the daily risk-free rate, in decimals, by the copied loader, with the files kept
    in french_folder(): the factors, the risk-free rate and the provenance records of the two files read."""
    return F.daily_factors(french_folder() if folder is None else folder)


def french_monthly(folder=None) -> tuple:
    """French's monthly factors and the monthly risk-free rate, in decimals, by the copied loader, with the files
    kept in french_folder(): the factors, the risk-free rate and the provenance records of the two files read."""
    return F.monthly_factors(french_folder() if folder is None else folder)


def write_french_provenance(records: list, folder=None) -> pathlib.Path:
    """The provenance record of French's files, written beside the other results. A file's record already there
    is replaced by its latest, and the records of other files are kept."""
    folder = results_dir() if folder is None else pathlib.Path(folder)
    path = folder / F.PROVENANCE_NAME
    by_key = {}
    if path.exists():
        by_key = {r["key"]: r for r in json.loads(path.read_text()).get("files", [])}
    by_key.update({r["key"]: r for r in records})
    return write_results(F.PROVENANCE_NAME, folder=folder,
                         record={"source": "Kenneth French's data library, downloaded at run time and never committed",
                                 "files": [by_key[k] for k in sorted(by_key)]})
