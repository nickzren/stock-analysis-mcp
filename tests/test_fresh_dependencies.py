"""Offline acceptance tests for the fresh-resolution release gate."""

import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_fresh_dependencies.py"
spec = importlib.util.spec_from_file_location("fresh_dependency_check", SCRIPT)
assert spec is not None and spec.loader is not None
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class FakeSubprocess:
    """Model installs and pytest without starting Python, uv or the network."""

    def __init__(self, repo: Path, *, failure: str = ""):
        self.repo = repo
        self.failure = failure
        self.calls: list[list[str]] = []
        self.environments: dict[str, dict[str, str]] = {}
        self.directories: dict[str, Path] = {}
        self.tools_installed = False
        self.source = b'"""Synthetic installed release."""\n'

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        env = kwargs["env"]
        stage = Path(kwargs["stdout"].name).stem
        self.environments[stage] = env.copy()
        self.directories[stage] = Path(kwargs["cwd"])
        code, text = 0, ""
        if command[:3] == ["git", "status", "--porcelain=v1"]:
            text = " M source.py\n" if self.failure == "dirty" else ""
        elif command[:3] == ["git", "rev-parse", "HEAD"]:
            text = "a" * 40 + "\n"
        elif command[:2] == ["git", "archive"]:
            archive_path = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--output=")))
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("src/stock_analysis/__init__.py", self.source)
                archive.writestr("pyproject.toml", '[project]\nversion="0.5.1"\n[project.optional-dependencies]\ndev=["pytest>=8"]\n')
                archive.writestr("uv.lock", "locked")
        elif command[:2] == ["uv", "venv"]:
            python = Path(command[2]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        elif stage == "locked-install":
            python = Path(env["UV_PROJECT_ENVIRONMENT"]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        elif stage == "test-tools-install":
            self.tools_installed = True
        elif len(command) > 2 and command[1] == "-c":
            if stage == "test-tool-requirements":
                text = json.dumps({"test_requirements": {"pytest": "9.0.3"},
                                   "production_packages": {"stock-analysis": "0.5.1", "numpy": "2.4.6"}})
            else:
                root = Path(command[0]).parent.parent / "lib/stock_analysis"
                packages = {"stock-analysis": "0.5.1", "numpy": "2.4.6" if stage == "locked" else "2.5.4"}
                if stage == "locked" or self.tools_installed:
                    packages["pytest"] = "9.0.3"
                if self.failure == "wrong-tool" and stage == "test-environment":
                    packages["pytest"] = "9.1.0"
                if self.failure == "missing-tool" and stage == "test-environment":
                    packages.pop("pytest")
                if self.failure == "runtime-drift" and stage == "test-environment":
                    packages["numpy"] = "2.5.5"
                if self.failure == "wrong-source" and stage == "production":
                    root = self.repo / "src/stock_analysis"
                source_hash = hashlib.sha256(self.source).hexdigest()
                if self.failure == "wrong-bytes" and stage == "production":
                    source_hash = "wrong"
                if self.failure == "wrong-version" and stage == "production":
                    packages["stock-analysis"] = "0.5.0"
                text = json.dumps({"python": "3.13.16", "machine": "arm64", "tzset": True,
                                   "executable": command[0], "source_root": str(root),
                                   "source_hashes": {"__init__.py": source_hash},
                                   "packages": packages})
        elif stage == "pytest":
            junit = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--junitxml=")))
            content = '<testcase classname="suite" name="test_ok"/>'
            if self.failure == "pytest":
                content = '<testcase classname="suite" name="test_bad"><failure message="bad"/></testcase>'
            elif self.failure == "all-skipped":
                content = '<testcase classname="suite" name="test_skip"><skipped message="no data"/></testcase>'
            junit.write_text("<testsuites><testsuite>" + content + "</testsuite></testsuites>")
        if self.failure == "lock-change" and stage == "fresh-install":
            (self.repo / "uv.lock").write_text("unexpected update")
        if self.failure == "venv-change" and stage == "fresh-install":
            (self.repo / ".venv/sentinel").write_text("unexpected update")
        if stage == self.failure:
            code = 1
        kwargs["stdout"].write(text)
        return subprocess.CompletedProcess(command, code)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(check.tempfile, "tempdir", str(tmp_path))
    path = tmp_path / "repo"
    path.mkdir()
    (path / "uv.lock").write_text("original lock")
    (path / "pyproject.toml").write_text("original project")
    (path / ".venv").mkdir()
    (path / ".venv/sentinel").write_text("original environment")
    return path


def run_check(repo: Path, failure: str = "") -> tuple[dict[str, Any], FakeSubprocess, Path]:
    output = repo.parent / "evidence"
    runner = FakeSubprocess(repo, failure=failure)
    result = check.check_dependencies(repo, output, executor=runner)
    assert json.loads((output / "result.json").read_text()) == result
    return result, runner, output


def test_success_preserves_repo_and_records_full_evidence(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(repo / ".venv"))
    monkeypatch.setenv("PIP_INDEX_URL", "https://unused.invalid")
    result, runner, output = run_check(repo)
    assert result["status"] == "PASS"
    assert result["repo_unchanged"] is True
    assert "UV_PROJECT_ENVIRONMENT" not in runner.environments["fresh-install"]
    assert "PIP_INDEX_URL" not in runner.environments["fresh-install"]
    assert runner.environments["locked-install"]["UV_PROJECT_ENVIRONMENT"] != str(repo / ".venv")
    work = Path(result["work_directory"])
    assert work.is_relative_to(tmp_path)
    assert runner.directories["pytest"] == work / "source"
    assert "PYTHONPATH" not in runner.environments["pytest"]
    pytest_command = next(cmd for cmd in runner.calls if "-m" in cmd and "pytest" in cmd)
    assert pytest_command[:3] == [str(work / "fresh/bin/python"), "-m", "pytest"]
    provenance = json.loads((output / "test-environment.json").read_text())
    assert Path(provenance["source_root"]).is_relative_to(work / "fresh")
    assert provenance["source_hashes"] == {"__init__.py": hashlib.sha256(runner.source).hexdigest()}
    assert result["tests"] == {"total": 1, "passed": 1, "failed": [], "skipped": []}
    assert (repo / "uv.lock").read_text() == "original lock"
    assert (repo / ".venv/sentinel").read_text() == "original environment"
    assert json.loads((output / "version-comparison.json").read_text())["numpy"] == {"locked": "2.4.6", "fresh": "2.5.4"}
    assert (output / "production-constraints.txt").read_text() == "numpy==2.5.4\nstock-analysis==0.5.1\n"
    assert (output / "test-tools.txt").read_text() == "pytest==9.0.3\n"
    installs = [cmd for cmd in runner.calls if cmd[:3] == ["uv", "pip", "install"]]
    assert all(cmd[cmd.index("--default-index") + 1] == "https://pypi.org/simple" for cmd in installs)
    assert all("--no-config" in cmd and "--no-python-downloads" in cmd for cmd in installs)


@pytest.mark.parametrize("failure", ["fresh-install", "locked-install", "test-tools-install", "pytest"])
def test_install_tooling_and_suite_failures_block_release(repo: Path, failure: str) -> None:
    result, _, _ = run_check(repo, failure)
    assert result["status"] == "FAIL"
    assert result["failed_stage"] == failure
    assert result["repo_unchanged"] is True


def test_tooling_cannot_change_runtime_versions(repo: Path) -> None:
    result, runner, _ = run_check(repo, "runtime-drift")
    assert result["status"] == "FAIL"
    assert result["failed_stage"] == "runtime-preservation"
    assert not any("pytest" in cmd and "-m" in cmd for cmd in runner.calls)


@pytest.mark.parametrize("failure", ["wrong-source", "wrong-bytes", "wrong-version"])
def test_wrong_installed_application_is_rejected(repo: Path, failure: str) -> None:
    result, _, _ = run_check(repo, failure)
    assert result["status"] == "FAIL"
    assert result["failed_stage"] == "production"


@pytest.mark.parametrize("failure", ["lock-change", "venv-change"])
def test_unexpected_repo_mutation_is_a_failure(repo: Path, failure: str) -> None:
    result, _, _ = run_check(repo, failure)
    assert result["status"] == "FAIL"
    assert result["failed_stage"] == "repo-preservation"
    assert result["repo_unchanged"] is False


def test_dirty_checkout_is_rejected_before_any_install(repo: Path) -> None:
    result, runner, _ = run_check(repo, "dirty")
    assert result["status"] == "FAIL"
    assert result["failed_stage"] == "preflight"
    assert not any(cmd[0] == "uv" for cmd in runner.calls)


def test_no_executed_tests_is_not_a_passing_release_gate(repo: Path) -> None:
    result, _, _ = run_check(repo, "all-skipped")
    assert result["status"] == "FAIL"
    assert result["tests"]["skipped"] == [{"test": "suite::test_skip", "reason": "no data"}]


def test_existing_output_is_never_overwritten(repo: Path) -> None:
    output = repo.parent / "evidence"
    output.mkdir()
    (output / "result.json").write_text("retained")
    with pytest.raises(FileExistsError):
        check.check_dependencies(repo, output, executor=FakeSubprocess(repo))
    assert (output / "result.json").read_text() == "retained"


@pytest.mark.parametrize("protected", [".venv", ".git"])
def test_output_cannot_be_inside_the_repo_environment_or_git(repo: Path, protected: str) -> None:
    output = repo / protected / "evidence"
    with pytest.raises(ValueError, match="Evidence must not"):
        check.check_dependencies(repo, output, executor=FakeSubprocess(repo))
    assert not output.exists()


@pytest.mark.parametrize("failure", ["wrong-tool", "missing-tool"])
def test_installed_test_tools_must_match_the_lock(repo: Path, failure: str) -> None:
    result, runner, _ = run_check(repo, failure)
    assert result["status"] == "FAIL"
    assert result["failed_stage"] == "test-tool-versions"
    assert not any("-m" in cmd and "pytest" in cmd for cmd in runner.calls)
