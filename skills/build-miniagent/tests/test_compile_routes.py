import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compile_routes.py"
SPEC = importlib.util.spec_from_file_location("miniagent_compile_routes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RouteCompilerTest(unittest.TestCase):
    def _document(self, text: str) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", encoding="utf-8", delete=False
        )
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            temporary.write(text)
        return Path(temporary.name)

    def test_selects_a_later_mermaid_workflow(self) -> None:
        document = self._document(
            """
```mermaid
flowchart TD
  S([Start]) -->|"e1"| E([End])
```
```mermaid
flowchart TD
  S([Start]) -->|"e1"| T[tool_call: lookup]
  T -->|"e2"| E([End])
```
"""
        )
        table = MODULE.compile_routes(document, "W2", block_number=2)
        self.assertIn("| W2-A | e1→e2 | 1 | lookup |", table)

    def test_rejects_edges_outside_terminal_routes(self) -> None:
        document = self._document(
            """
```mermaid
flowchart TD
  S([Start]) -->|"e1"| E([End])
  X{Cycle} -->|"e2"| Y[tool_call: hidden]
  Y -->|"e3"| X
```
"""
        )
        with self.assertRaisesRegex(ValueError, "outside every terminal route"):
            MODULE.compile_routes(document, "W1")

    def test_rejects_standalone_nodes_outside_routes(self) -> None:
        document = self._document(
            """
```mermaid
flowchart TD
  S([Start]) -->|"e1"| E([End])
  X[tool_call: unused]
```
"""
        )
        with self.assertRaisesRegex(ValueError, "nodes outside every terminal route"):
            MODULE.compile_routes(document, "W1")


if __name__ == "__main__":
    unittest.main()
