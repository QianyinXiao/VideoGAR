import ast
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = {"evaluation", "method_core", "utils"}
SOURCE_ROOTS = tuple(
    REPOSITORY_ROOT / package for package in (*sorted(LOCAL_PACKAGES), "tests")
)


def _local_module_exists(module_name):
    module_path = REPOSITORY_ROOT.joinpath(*module_name.split("."))
    return (
        module_path.with_suffix(".py").is_file()
        or (module_path / "__init__.py").is_file()
    )


class RepositoryConsistencyTest(unittest.TestCase):
    def test_absolute_local_imports_resolve(self):
        for source_root in SOURCE_ROOTS:
            for source_path in source_root.rglob("*.py"):
                tree = ast.parse(
                    source_path.read_text(encoding="utf-8"), filename=str(source_path)
                )
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.ImportFrom)
                        and node.level == 0
                        and node.module
                    ):
                        if node.module.split(".", 1)[0] in LOCAL_PACKAGES:
                            self.assertTrue(
                                _local_module_exists(node.module),
                                f"{source_path.relative_to(REPOSITORY_ROOT)}:{node.lineno} imports missing local module {node.module!r}",
                            )
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.split(".", 1)[0] in LOCAL_PACKAGES:
                                self.assertTrue(
                                    _local_module_exists(alias.name),
                                    f"{source_path.relative_to(REPOSITORY_ROOT)}:{node.lineno} imports missing local module {alias.name!r}",
                                )

    def test_removed_experimental_scorer_has_no_code_references(self):
        removed_identifier = "tf" + "vtg"
        for source_path in (REPOSITORY_ROOT / "method_core").rglob("*.py"):
            self.assertNotIn(
                removed_identifier,
                source_path.read_text(encoding="utf-8").lower(),
                f"stale experimental scorer reference in {source_path.relative_to(REPOSITORY_ROOT)}",
            )


if __name__ == "__main__":
    unittest.main()
