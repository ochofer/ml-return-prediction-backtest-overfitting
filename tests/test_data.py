"""The wrappers of the copied data layer, on a made-up extraction written to a temporary folder: the repository's
root, the study's data root, the manifest's two checks, the columns read, the results folder and French's folder.
No test reads the licensed extraction."""
import hashlib
import json
import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlrp import constants as C  # noqa: E402
from mlrp import data as D  # noqa: E402
from mlrp.source import extraction as X  # noqa: E402


def made_up_record(key, file, rows):
    """A manifest record of the shape the extraction writes, with made-up values."""
    return {"key": key, "library": "lib", "table": key, "file": file, "sql": "select 1", "rows": rows,
            "columns": 3, "bytes": 100, "sha256": hashlib.sha256(file.encode()).hexdigest(),
            "fetched_utc": "2026-09-28T10:00:00+00:00"}


@pytest.fixture
def extraction(tmp_path):
    """A made-up extraction: a manifest with one file for each of the study's tables, listed out of order, one
    file of a table the study does not read, its .sha256 file, and one file of made-up fundamentals."""
    records = [made_up_record(k, f"raw/{k}/{k}_2001.parquet", 10 + i) for i, k in enumerate(reversed(C.TABLES))]
    records.append(made_up_record("comp_co_hgic", "raw/comp_co_hgic/comp_co_hgic.parquet", 5))
    manifest = {"files": records, "counts": {"files": len(records), "rows": sum(r["rows"] for r in records)}}
    text = json.dumps(manifest)
    (tmp_path / C.MANIFEST_FILE).write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (tmp_path / (C.MANIFEST_FILE + ".sha256")).write_text(digest + "  " + C.MANIFEST_FILE + "\n")
    folder = tmp_path / "raw" / "comp_funda"
    folder.mkdir(parents=True)
    funda = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in D.COLUMNS["comp_funda"]})
    funda["gvkey"] = ["001000", "001000", "002000"]
    funda["datadate"] = ["2000-12-31", "2001-12-31", "2001-13-45"]
    funda["fyear"] = [2000, 2001, 2001]
    funda["an_item_not_read"] = 0.0
    funda.to_parquet(folder / "comp_funda_2001.parquet", index=False)
    return tmp_path, manifest, digest


def test_the_repository_root_and_the_folders_under_it():
    assert (D.repo_root() / "src" / "mlrp" / "data.py").is_file()
    assert X.repo_root() == D.repo_root() / "src"   # the copy's own root is src/, which is why the wrapper exists
    assert D.results_dir() == D.repo_root() / C.RESULTS_DIR
    assert D.french_folder() == D.repo_root() / C.FRENCH_CACHE_DIR
    ignored = (D.repo_root() / ".gitignore").read_text().split()
    assert C.FRENCH_CACHE_DIR.split("/")[0] + "/" in ignored


def test_the_data_root_comes_from_the_study_variable(extraction):
    root, _, _ = extraction
    assert D.data_root({C.DATA_ROOT_ENV: str(root)}) == root
    with pytest.raises(SystemExit, match=C.DATA_ROOT_ENV):
        D.data_root({})
    with pytest.raises(SystemExit, match=C.DATA_ROOT_ENV):
        D.data_root({"ANOTHER_DATA_ROOT": str(root)})
    with pytest.raises(SystemExit, match="no manifest"):
        D.data_root({C.DATA_ROOT_ENV: str(root / "raw")})


def test_the_entries_and_their_digest(extraction):
    _, manifest, _ = extraction
    entries = D.manifest_entries(manifest)
    assert len(entries) == len(C.TABLES)
    assert [e[0] for e in entries] == sorted(f"raw/{k}/{k}_2001.parquet" for k in C.TABLES)
    assert all(isinstance(e[1], int) for e in entries)
    by_hand = "[" + ",".join(f'["{f}",{r},"{s}"]' for f, r, s in entries) + "]"
    assert D.entries_digest(entries) == hashlib.sha256(by_hand.encode()).hexdigest()


def test_the_extraction_opens_only_when_both_checks_pass(extraction, monkeypatch):
    root, manifest, digest = extraction
    env = {C.DATA_ROOT_ENV: str(root)}
    monkeypatch.setattr(C, "MANIFEST_SHA256", digest)
    monkeypatch.setattr(C, "MANIFEST_ENTRIES_SHA256", D.entries_digest(D.manifest_entries(manifest)))
    monkeypatch.setattr(C, "MANIFEST_ENTRIES", len(C.TABLES))
    opened_root, opened = D.open_extraction(env)
    assert opened_root == root and opened == manifest
    monkeypatch.setattr(C, "MANIFEST_ENTRIES_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="entries"):
        D.open_extraction(env)
    monkeypatch.setattr(C, "MANIFEST_SHA256", "f" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        D.open_extraction(env)


def test_a_change_to_one_of_the_tables_changes_the_digest(extraction):
    _, manifest, _ = extraction
    before = D.entries_digest(D.manifest_entries(manifest))
    other = json.loads(json.dumps(manifest))
    other["files"].append(made_up_record("comp_new_table", "raw/comp_new_table/a.parquet", 7))
    assert D.entries_digest(D.manifest_entries(other)) == before
    other["files"][0]["rows"] += 1
    assert D.entries_digest(D.manifest_entries(other)) != before


def test_the_columns_read():
    assert set(D.COLUMNS) == set(C.TABLES)
    assert "vol" in D.COLUMNS["crsp_msf"] and set(X.COLUMNS["crsp_msf"]) <= set(D.COLUMNS["crsp_msf"])
    items = {C.BOOK_EQUITY_DEFERRED_TAXES, C.EARNINGS, C.REVENUE, C.TOTAL_ASSETS, *C.BOOK_EQUITY_PREFERRED,
             *C.MARKET_VALUE_AT_FISCAL_YEAR_END, *C.OPERATING_COSTS, *(i for g in C.BOOK_EQUITY_STOCKHOLDERS for i in g)}
    assert set(D.FUNDAMENTALS_ITEMS) == items and len(D.FUNDAMENTALS_ITEMS) == len(items)
    assert D.COLUMNS["comp_funda"][:3] == ("gvkey", "datadate", "fyear")


def test_the_fundamentals_are_read_with_their_dates(extraction):
    root, _, _ = extraction
    frame, unparsed = D.read_table(root, "comp_funda")
    assert list(frame.columns) == list(D.COLUMNS["comp_funda"])
    assert pd.api.types.is_datetime64_any_dtype(frame["datadate"])
    assert unparsed == {"datadate": 1} and frame["datadate"].isna().sum() == 1
    with pytest.raises(SystemExit, match="not one of the study's tables"):
        D.read_table(root, "comp_co_hgic")


def test_a_column_missing_from_a_file_is_listed(extraction):
    root, manifest, _ = extraction
    record = next(r for r in D.files_used(manifest) if r["key"] == "comp_funda")
    assert D.missing_columns(root, [record]) == {}
    folder = root / "raw" / "crsp_msf"
    folder.mkdir()
    pd.DataFrame({c: [1.0] for c in X.COLUMNS["crsp_msf"]}).to_parquet(folder / "crsp_msf_2001.parquet", index=False)
    record = next(r for r in D.files_used(manifest) if r["key"] == "crsp_msf")
    assert D.missing_columns(root, [record]) == {record["file"]: ["vol"]}


def test_the_provenance_record(extraction):
    _, manifest, _ = extraction
    records = D.files_used(manifest)
    record = D.provenance(manifest, records, written_by="a test")
    assert record["tables_used"] == sorted(C.TABLES) and record["files_used"] == len(C.TABLES)
    assert record["manifest_entries_sha256"] == D.entries_digest(D.manifest_entries(manifest))
    assert record["copied_from"] == {C.SOURCE_REPOSITORY: C.SOURCE_COMMIT,
                                     C.BOOTSTRAP_SOURCE_REPOSITORY: C.BOOTSTRAP_SOURCE_COMMIT}
    json.dumps(record)


def test_results_are_written_and_read_back_in_the_folder_given(tmp_path):
    frame = pd.DataFrame({"year": [2001, 2002], "share": [0.5, 0.25]})
    assert D.write_results("a.csv", frame=frame, folder=tmp_path).read_text().splitlines()[0] == '"year","share"'
    D.write_results("a.json", record={"n": 3}, folder=tmp_path)
    assert D.read_result("a.json", folder=tmp_path) == {"n": 3}
    with pytest.raises(SystemExit, match="run the notebooks in order"):
        D.read_result("absent.json", folder=tmp_path)
    D.write_french_provenance([{"key": "b", "v": 1}, {"key": "a", "v": 1}], folder=tmp_path)
    D.write_french_provenance([{"key": "a", "v": 2}], folder=tmp_path)
    files = D.read_result("provenance_french.json", folder=tmp_path)["files"]
    assert files == [{"key": "a", "v": 2}, {"key": "b", "v": 1}]
