"""The code copied from Stock-Level Tracking and from Signal Evaluation: each copy is the source file unchanged, and
the constants the copies read are the source's.

Each copy in src/mlrp/source/ is hashed against constants.COPIED_FILES or constants.BOOTSTRAP_SHA256, and fails
here if one byte differs. src/mlrp/source/constants.py must hold exactly the names the copies reference, and no
module outside src/mlrp/source/ may read it. Its values and their source lines are compared with the source's
constants file at constants.SOURCE_COMMIT, and each copy with its source file at its commit, read with git, when a
clone of the source repository sits beside this one (as ../B-S or ../stock-level-tracking, and ../B-Q or
../volatility-managed-portfolios). Those comparisons are skipped otherwise.
"""
import ast
import hashlib
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlrp import constants as C  # noqa: E402
from mlrp.source import constants as SC  # noqa: E402

SOURCE = ROOT / C.COPIED_INTO
BOOTSTRAP = pathlib.PurePosixPath(C.BOOTSTRAP_FILE).name
CLONES = {
    C.SOURCE_REPOSITORY: ("B-S", "stock-level-tracking"),
    C.BOOTSTRAP_SOURCE_REPOSITORY: ("B-Q", "volatility-managed-portfolios"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def at_commit(repository: str, commit: str, path: str):
    """A file of a source repository at a commit, read with git from a clone beside this one; None without one."""
    for name in CLONES[repository]:
        clone = ROOT.parent / name
        if (clone / ".git").exists():
            r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=clone, capture_output=True)
            if r.returncode == 0:
                return r.stdout
    return None


def test_the_copies_are_the_ones_named():
    present = {p.name for p in SOURCE.glob("*.py")} - {"__init__.py", "constants.py"}
    assert present == set(C.COPIED_MODULES) | {BOOTSTRAP}
    assert set(C.COPIED_FILES) == set(C.COPIED_MODULES)


@pytest.mark.parametrize("name", C.COPIED_MODULES)
def test_each_copy_is_the_source_file(name):
    assert sha256((SOURCE / name).read_bytes()) == C.COPIED_FILES[name][1], f"{name} differs from the source file"


def test_the_bootstrap_is_the_source_file():
    assert sha256((SOURCE / BOOTSTRAP).read_bytes()) == C.BOOTSTRAP_SHA256


def test_the_source_constants_hold_exactly_the_names_the_copies_reference():
    referenced = set()
    for name in C.COPIED_MODULES + (BOOTSTRAP,):
        referenced |= set(re.findall(r"\bC\.([A-Z_][A-Z0-9_]*)", (SOURCE / name).read_text()))
    held = {n for n in dir(SC) if n.isupper()}
    assert held == referenced, (f"held and not referenced: {sorted(held - referenced)}; "
                                f"referenced and not held: {sorted(referenced - held)}")


def test_no_module_outside_the_copies_reads_the_source_constants():
    pattern = re.compile(r"source\s+import\s+[^\n]*\bconstants\b|source\.constants|from\s+\.source\s+import\s+constants")
    readers = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "notebooks").glob("*.ipynb")) + [ROOT / "build_notebooks.py"]:
        if path.is_file() and SOURCE not in path.parents and pattern.search(path.read_text()):
            readers.append(str(path.relative_to(ROOT)))
    assert not readers, f"modules outside {C.COPIED_INTO} that read its constants: {readers}"


@pytest.fixture(scope="module")
def source_constants():
    data = at_commit(C.SOURCE_REPOSITORY, C.SOURCE_COMMIT, C.SOURCE_CONSTANTS_FILE)
    if data is None:
        pytest.skip("no clone of the source repository beside this one")
    assert sha256(data) == C.SOURCE_CONSTANTS_SHA256
    values = {}
    exec(compile(data.decode(), "constants.py", "exec"), values)
    return data.decode(), values


def test_the_source_constants_equal_the_source_file_at_the_commit(source_constants):
    text, source = source_constants
    ours = {n: getattr(SC, n) for n in dir(SC) if n.isupper()}
    differ = {n: (v, source.get(n)) for n, v in ours.items() if source.get(n) != v}
    assert not differ, f"values that differ from the source: {differ}"
    lines = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                names = [t] if isinstance(t, ast.Name) else list(getattr(t, "elts", []))
                for e in names:
                    if isinstance(e, ast.Name):
                        lines.setdefault(e.id, node.lineno)
    held = (SOURCE / "constants.py").read_text()
    for n in ours:
        m = re.search(rf"^{n} = .*# line (\d+)$", held, flags=re.M)
        assert m and int(m.group(1)) == lines[n], n


def test_each_copy_equals_its_source_file_at_its_commit():
    pairs = [(C.SOURCE_REPOSITORY, C.SOURCE_COMMIT, path, name) for name, (path, _) in C.COPIED_FILES.items()]
    pairs.append((C.BOOTSTRAP_SOURCE_REPOSITORY, C.BOOTSTRAP_SOURCE_COMMIT, C.BOOTSTRAP_FILE, BOOTSTRAP))
    compared = 0
    for repository, commit, path, name in pairs:
        data = at_commit(repository, commit, path)
        if data is not None:
            assert data == (SOURCE / name).read_bytes(), name
            compared += 1
    if compared == 0:
        pytest.skip("no clone of either source repository beside this one")
