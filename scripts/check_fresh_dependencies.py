"""Run the full suite against fresh runtime dependencies before a release.

Run from a clean committed checkout:
    python scripts/check_fresh_dependencies.py --output-dir <new evidence directory>

Uses public PyPI and temporary environments. It never updates the project lock or
repo environment. A failed result blocks release; there is no waiver flag.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SNAPSHOT = r'''
import hashlib, importlib.metadata as metadata, json, platform, re, sys, sysconfig, time
from pathlib import Path
import stock_analysis
root = Path(stock_analysis.__file__).resolve().parent
print(json.dumps({
    "python": sys.version, "executable": sys.executable,
    "machine": platform.machine(), "tzset": hasattr(time, "tzset"),
    "source_root": str(root), "purelib": sysconfig.get_paths()["purelib"],
    "source_hashes": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    },
    "packages": {
        re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
        for d in metadata.distributions()
    },
}, sort_keys=True))
'''

_TOOL_REQUIREMENTS = r'''
import importlib.metadata as metadata, json, sys
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
roots, runtime = json.loads(sys.argv[1]), json.loads(sys.argv[2])
def closure(requirements):
    queue = [(Requirement(r).name, frozenset(Requirement(r).extras)) for r in requirements]
    seen, names = set(), set()
    environment = default_environment()
    while queue:
        name, extras = queue.pop()
        name = canonicalize_name(name)
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        names.add(name)
        for raw in metadata.requires(name) or []:
            requirement = Requirement(raw)
            if requirement.marker is not None and not any(
                requirement.marker.evaluate({**environment, "extra": extra})
                for extra in extras | {""}
            ):
                continue
            queue.append((requirement.name, frozenset(requirement.extras)))
    return {name: metadata.version(name) for name in sorted(names)}
print(json.dumps({
    "test_requirements": {k: v for k, v in closure(roots).items() if k not in runtime},
    "production_packages": closure(["stock-analysis"]),
}, sort_keys=True))
'''


class CheckError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _protected_state(repo: Path) -> dict[str, Any]:
    files = {}
    for name in ("uv.lock", "pyproject.toml"):
        path = repo / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    venv = repo / ".venv"
    entries = []
    if venv.exists():
        for path in sorted(venv.rglob("*")):
            stat = path.lstat()
            entries.append((str(path.relative_to(venv)), stat.st_size, stat.st_mtime_ns))
    files["venv_state"] = hashlib.sha256(repr(entries).encode()).hexdigest()
    return files


class Commands:
    def __init__(self, output: Path, env: dict[str, str], executor: Any):
        self.output = output
        self.env = env
        self.executor = executor
        self.records: list[dict[str, Any]] = []
        self.stage = "preflight"

    def run(
        self,
        stage: str,
        command: list[str],
        cwd: Path,
        *,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> tuple[int, str]:
        self.stage = stage
        started = time.monotonic()
        log = self.output / f"{stage}.log"
        with log.open("w") as stream:
            result = self.executor(
                command, cwd=cwd, env=self.env | (env or {}),
                stdout=stream, stderr=subprocess.STDOUT, text=True,
            )
        self.records.append({
            "stage": stage, "command": command, "cwd": str(cwd),
            "exit_code": result.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        _write_json(self.output / "commands.json", self.records)
        if check and result.returncode:
            raise CheckError(stage, f"Command exited {result.returncode}; see {log.name}")
        return result.returncode, log.read_text()


def _snapshot(commands: Commands, stage: str, python: Path, source: Path) -> dict[str, Any]:
    result = json.loads(commands.run(stage, [str(python), "-c", _SNAPSHOT], source)[1])
    _write_json(commands.output / f"{stage}.json", result)
    root = Path(result["source_root"]).resolve()
    if not root.is_relative_to(python.parent.parent.resolve()):
        raise CheckError(stage, "Application imported outside the test environment")
    expected = {
        str(p.relative_to(source / "src/stock_analysis")):
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((source / "src/stock_analysis").rglob("*")) if p.is_file()
    }
    if result["source_hashes"] != expected:
        raise CheckError(stage, "Installed application does not match the committed source")
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    if result["packages"].get("stock-analysis") != project["version"]:
        raise CheckError(stage, "Installed application metadata does not match the release version")
    return result


def _summary(path: Path) -> dict[str, Any]:
    cases = ET.parse(path).getroot().findall(".//testcase")
    skipped = [{"test": f"{case.get('classname')}::{case.get('name')}",
                "reason": case.find("skipped").get("message", "")}
               for case in cases if case.find("skipped") is not None]
    failures = [f"{case.get('classname')}::{case.get('name')}"
                for case in cases
                if case.find("failure") is not None or case.find("error") is not None]
    return {"total": len(cases), "passed": len(cases) - len(skipped) - len(failures),
            "failed": failures, "skipped": skipped}


def check_dependencies(repo: Path, output: Path, *, executor: Any = subprocess.run) -> dict[str, Any]:
    repo, output = repo.resolve(), output.resolve()
    if output.is_relative_to(repo / ".venv") or output.is_relative_to(repo / ".git"):
        raise ValueError("Evidence must not be written into the repo environment or Git directory")
    protected = _protected_state(repo)
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("UV_", "PIP_", "PYTEST_"))
           and k not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    commands = Commands(output, env, executor)
    report: dict[str, Any] = {"status": "FAIL", "started_at": datetime.now(UTC).isoformat()}
    head = None
    try:
        status = commands.run("git-status", ["git", "status", "--porcelain=v1"], repo)[1]
        if status.strip():
            raise CheckError("preflight", "The checkout must be clean and committed")
        head = commands.run("git-head", ["git", "rev-parse", "HEAD"], repo)[1].strip()
        report["commit"] = head
        work = Path(tempfile.mkdtemp(prefix="stock-analysis-fresh-"))
        report["work_directory"] = str(work)
        env["STOCK_ANALYSIS_DATA_DIR"] = str(work / "data")
        source = work / "source"
        commands.run("archive", ["git", "archive", "--format=zip", f"--output={work / 'source.zip'}", head], repo)
        with zipfile.ZipFile(work / "source.zip") as archive:
            archive.extractall(source)
        fresh = work / "fresh/bin/python"
        cache_options = ["--no-config", "--no-python-downloads", "--cache-dir", str(work / "fresh-cache")]
        commands.run("fresh-environment", ["uv", "venv", str(work / "fresh"), *cache_options], work)
        index = ["--default-index", "https://pypi.org/simple", "--keyring-provider", "disabled"]
        commands.run("fresh-install", ["uv", "pip", "install", "--python", str(fresh),
                                      str(source), "--strict", *index, *cache_options], work)
        before = _snapshot(commands, "production", fresh, source)
        report["interpreter"] = {k: before[k] for k in ("python", "machine", "tzset")}
        locked = work / "locked/bin/python"
        locked_options = ["--no-config", "--no-python-downloads", "--cache-dir", str(work / "locked-cache")]
        commands.run("locked-install", ["uv", "sync", "--frozen", "--extra", "dev", "--no-editable",
                                        "--python", str(fresh), *index, *locked_options], source,
                     env={"UV_PROJECT_ENVIRONMENT": str(work / "locked")})
        locked_snapshot = _snapshot(commands, "locked", locked, source)
        if before["python"] != locked_snapshot["python"] or before["machine"] != locked_snapshot["machine"]:
            raise CheckError("locked", "The locked comparison uses a different interpreter")
        roots = tomllib.loads((source / "pyproject.toml").read_text())["project"]["optional-dependencies"]["dev"]
        tooling = json.loads(commands.run("test-tool-requirements", [str(locked), "-c", _TOOL_REQUIREMENTS,
                                                                   json.dumps(roots), json.dumps(before["packages"])], source)[1])
        _write_json(output / "test-tool-requirements.json", tooling)
        requirements = output / "test-tools.txt"
        requirements.write_text("".join(f"{k}=={v}\n" for k, v in tooling["test_requirements"].items()))
        constraints = output / "production-constraints.txt"
        constraints.write_text("".join(f"{k}=={v}\n" for k, v in sorted(before["packages"].items())))
        commands.run("test-tools-install", ["uv", "pip", "install", "--python", str(fresh),
                                            "--requirements", str(requirements), "--constraints", str(constraints),
                                            "--strict", *index, *cache_options], work)
        after = _snapshot(commands, "test-environment", fresh, source)
        if any(after["packages"].get(k) != v for k, v in before["packages"].items()):
            raise CheckError("runtime-preservation", "Test tooling changed production dependency versions")
        if any(after["packages"].get(k) != v for k, v in tooling["test_requirements"].items()):
            raise CheckError("test-tool-versions", "Installed test tools do not match their locked versions")
        locked_packages, fresh_packages = tooling["production_packages"], before["packages"]
        comparison = {k: {"locked": locked_packages.get(k), "fresh": fresh_packages.get(k)}
                      for k in sorted(locked_packages.keys() | fresh_packages.keys())}
        _write_json(output / "version-comparison.json", comparison)
        (output / "version-comparison.md").write_text(
            "| Package | Locked | Fresh |\n|---|---|---|\n" +
            "".join(f"| {k} | {v['locked'] or '-'} | {v['fresh'] or '-'} |\n" for k, v in comparison.items()))
        commands.run("dependency-check", ["uv", "pip", "check", "--python", str(fresh), *cache_options], work)
        code, _ = commands.run("pytest", [str(fresh), "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider",
                                          f"--junitxml={output / 'pytest.xml'}"], source, check=False)
        report["tests"] = _summary(output / "pytest.xml")
        if code or report["tests"]["failed"] or not report["tests"]["passed"]:
            raise CheckError("pytest", f"The fresh-resolution suite failed (exit {code})")
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", failed_stage=getattr(exc, "stage", commands.stage), reason=str(exc))
    finally:
        report["repo_unchanged"] = _protected_state(repo) == protected
        if not report["repo_unchanged"]:
            report.update(status="FAIL", failed_stage="repo-preservation", reason="Repo lock, project or environment changed")
        if head is not None:
            try:
                final_status = commands.run("final-git-status", ["git", "status", "--porcelain=v1"], repo)[1]
                final_head = commands.run("final-git-head", ["git", "rev-parse", "HEAD"], repo)[1].strip()
                if final_status.strip() or final_head != head:
                    report.update(status="FAIL", failed_stage="repo-preservation", reason="Checkout changed during the check")
            except Exception as exc:
                report.update(status="FAIL", failed_stage="repo-preservation", reason=str(exc))
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(output / "result.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = check_dependencies(Path(__file__).resolve().parents[1], args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Fresh dependency check refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
