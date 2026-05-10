from __future__ import annotations

import ast
import pkgutil
from pathlib import Path

import rl_recsys.agents

FORBIDDEN_PREFIXES = (
    "rl_recsys.data.loaders.",
    "rl_recsys.environments.rl4rs",
    "rl_recsys.environments.kuairec",
    "rl_recsys.environments.finn_no_slate",
    "rl_recsys.environments.movielens",
    "rl_recsys.environments.open_bandit",
)


def _imported_modules(py_path: Path) -> set[str]:
    tree = ast.parse(py_path.read_text())
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                seen.add(node.module)
    return seen


def test_no_agent_module_imports_dataset_specific_code():
    pkg_path = Path(rl_recsys.agents.__file__).parent
    offenders: list[tuple[str, str]] = []
    for entry in pkgutil.iter_modules([str(pkg_path)]):
        py = pkg_path / f"{entry.name}.py"
        if not py.exists():
            continue
        for mod in _imported_modules(py):
            if mod.startswith(FORBIDDEN_PREFIXES):
                offenders.append((entry.name, mod))
    assert not offenders, (
        f"Agent modules must stay dataset-agnostic. Offenders: {offenders}"
    )
