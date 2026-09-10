#!/usr/bin/env python3
"""Render and verify the MiniAgent application template in a temporary directory."""

from pathlib import Path
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent


def run(command: list[str], cwd: Path = REPOSITORY_ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def expect_gate(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), "[expect exit 2]", flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 2:
        raise RuntimeError(
            f"Expected exit 2 from {' '.join(command)}, observed {result.returncode}."
        )


def expect_failure(command: list[str], cwd: Path = REPOSITORY_ROOT) -> None:
    print("+", " ".join(command), "[expect failure]", flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode == 0:
        raise RuntimeError(f"Expected failure from {' '.join(command)}.")


def main() -> int:
    run([sys.executable, str(SCRIPTS / "check_contract_sync.py")])
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(REPOSITORY_ROOT / "skills" / "build-miniagent" / "tests"),
        ]
    )
    with tempfile.TemporaryDirectory(prefix="miniagent-recipe-") as temporary:
        expect_failure(
            [
                "uvx",
                "--from",
                "copier",
                "copier",
                "copy",
                "--defaults",
                "--data",
                "package_name=class",
                str(REPOSITORY_ROOT / "template"),
                str(Path(temporary) / "invalid_keyword"),
            ]
        )
        application = Path(temporary) / "recipe_smoke"
        run(
            [
                "uvx",
                "--from",
                "copier",
                "copier",
                "copy",
                "--defaults",
                "--data",
                "app_name=Recipe Smoke",
                "--data",
                "package_name=recipe_smoke",
                "--data",
                "description=Verify the MiniAgent recipe.",
                str(REPOSITORY_ROOT / "template"),
                str(application),
            ]
        )
        validator = SCRIPTS / "validate_scaffold.py"
        run([sys.executable, str(validator), str(application)])
        run(["make", "install"], cwd=application)
        run([sys.executable, str(validator), str(application)])
        run(["make", "test"], cwd=application)
        run(["make", "test-standalone"], cwd=application)
        for target in ("validate", "data-build", "train", "eval", "promote-runtime"):
            expect_gate(["make", target], cwd=application)
        fixture = SCRIPTS / "prepare_smoke_fixture.py"
        run([sys.executable, str(fixture), "configure", str(application)])
        for target in ("validate", "data-build"):
            run(["make", target], cwd=application)
        run(["make", "train", "TRAIN_ARGS=--fixture-run"], cwd=application)
        run(["make", "eval", "EVAL_ARGS=--fixture-evaluation"], cwd=application)
        run([sys.executable, str(fixture), "assert-evaluation", str(application)])
        run([sys.executable, str(fixture), "prepare-release", str(application)])
        run(["make", "promote-runtime"], cwd=application)
        run([sys.executable, str(fixture), "assert-release", str(application)])
        run(
            [
                "uv",
                "run",
                "--project",
                "src",
                "python",
                "-m",
                "recipe_smoke.cli",
                "validate",
            ],
            cwd=application,
        )
        run(
            [
                "uv",
                "run",
                "--project",
                "src",
                "python",
                "-c",
                (
                    "from recipe_smoke.agent import Agent; "
                    "result = Agent().respond('alpha'); "
                    "assert result.status == 'completed'; "
                    "assert result.message == 'Catalog result for alpha.'"
                ),
            ],
            cwd=application,
        )
        run(["make", "test"], cwd=application)
        run(["make", "test-standalone"], cwd=application)
        for target in ("data-build", "train", "eval"):
            expect_gate(["make", target], cwd=application)
        run([sys.executable, str(fixture), "tamper-runtime", str(application)])
        expect_gate(
            [
                "uv",
                "run",
                "--project",
                "src",
                "python",
                "-m",
                "recipe_smoke.cli",
                "validate",
            ],
            cwd=application,
        )
    print("MiniAgent template smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
