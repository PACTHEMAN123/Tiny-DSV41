"""Classify unchanged original verifier evidence; never supply model inputs."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import re

from .evidence_io import HASH, read_json

NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")


def ctrf_counts(report):
    results = report.get("results", {})
    if (not isinstance(results, dict) or not isinstance(results.get("tool"), dict)
            or results["tool"].get("name") != "pytest"):
        raise ValueError("No genuine pytest CTRF tool report")
    summary, tests = results.get("summary", {}), results.get("tests")
    keys = ("passed", "failed", "skipped", "pending", "other")
    if (not isinstance(summary, dict) or not isinstance(tests, list) or not 0 < len(tests) <= 100000 or
            type(summary.get("tests")) is not int or summary["tests"] != len(tests) or
            any(type(summary.get(key)) is not int or summary[key] < 0 for key in keys)):
        raise ValueError("Missing or inconsistent test records")
    counts = Counter(test.get("status") for test in tests if isinstance(test, dict))
    if set(counts) - set(keys) or any(counts[key] != summary[key] for key in keys):
        raise ValueError("CTRF status totals differ")
    if summary["pending"] or summary["other"] or summary["passed"] + summary["failed"] == 0:
        raise ValueError("No completed assertion evaluation")
    return {key: summary[key] for key in ("tests", *keys)}


def verifier_ambiguities(report, terminal):
    results = report["results"]
    issues = []
    summary = results["summary"]
    start, stop = summary.get("start"), summary.get("stop")
    begin, end = terminal.get("started_unix"), terminal.get("finished_unix")
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, stop, begin, end))
            or not begin - 1 <= start <= stop <= end + 1):
        issues.append("CTRF timing is not bound to this original verifier execution")
    for test in results["tests"]:
        if test["status"] == "failed" and (test.get("raw_status") != "call_failed" or
                "assertion error" not in str(test.get("message", ""))):
            issues.append("Failure is not proven to be a test-call assertion; review original verifier logs")
            break
    return issues


def original_verifier_sources(environment_dir):
    """Observe original host test bytes; never read them into the model payload."""
    from .bootstrap_payload import read
    task = Path(environment_dir).parent
    if not task.is_absolute():
        raise ValueError("Original verifier source must be absolute")
    paths = {name: task / name for name in ("tests/test.sh", "tests/test_outputs.py")}
    paths["provenance.json"] = task.parent.parent / "provenance.json"
    if any(part.is_symlink() for path in paths.values() for part in (path, *path.parents)):
        raise ValueError("Linked original verifier source")
    files = {}
    for name, path in paths.items():
        raw = read(path, 16 * 1024**2)
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return {"schema_version": 1, "task_id": task.name, "files": files}


def reviewed_sparql_original_context(path, report, terminal, bootstrap):
    """Bind narrow output-failure rules to unchanged original task evidence."""
    expected = {"tests/test.sh": {"bytes": 629, "sha256":
        "45902b57bd0f1dd6959216bc9943ab27bb349edc7ff557035a1ecc8732e4cc6b"},
        "tests/test_outputs.py": {"bytes": 2399, "sha256":
        "ecf15840fc14f1eeff2f3b6317d7f0febc8fa81f688d24e3054205ba13207b39"},
        "provenance.json": {"bytes": 7637, "sha256":
        "184faa0638cfd08499f51d53a7bf64a492daf1c14ec4f6aa9d889483678e228e"}}
    before, after = (terminal.get(key) for key in
        ("original_verifier_sources_before", "original_verifier_sources_after"))
    if (not isinstance(before, dict) or before != after or type(before.get("schema_version")) is not int
            or before["schema_version"] != 1
            or before.get("task_id") != "sparql-university" or before.get("files") != expected
            or type(terminal.get("return_code")) is not int or terminal["return_code"] != 0
            or not HASH.fullmatch(str(terminal.get("command_sha256")))
            or bootstrap.get("task_id") != "sparql-university"
            or bootstrap.get("manifest_sha256") !=
                "6abd7da26fc66958cd36274edebe7720da9eb24a62ceeb0e6c2ffaed8c9521f5"):
        return None
    if report["results"]["tool"].get("version") != "8.4.1":
        return None
    prerequisites = read_json(path / "agent-prerequisites.json")
    checks = prerequisites.get("checks")
    if (prerequisites.get("task_id") != "sparql-university" or prerequisites.get("verified") is not True
            or prerequisites.get("failure_type") is not None
            or not isinstance(checks, list) or len(checks) != 2
            or any(not isinstance(check, dict) or not isinstance(check.get("diagnostics"), dict) for check in checks)
            or {check.get("name") for check in checks} != {"rdflib_runtime", "sparql_input"}
            or any(type(check.get("return_code")) is not int or check["return_code"] != 0
                or check.get("expected_observation_matches") is not True
                or check.get("diagnostics", {}).get("stderr_python_error_types") != [] for check in checks)):
        return None
    return before


def reviewed_sparql_missing_output(path, report, terminal, bootstrap):
    """Recognize only the reviewed original task's absent required output failure.

    Missing input, fixture and runtime files remain ambiguous. This rule neither
    changes Harbor's reward nor authorizes reusing a previously excluded group.
    """
    from .bootstrap_payload import read
    before = reviewed_sparql_original_context(path, report, terminal, bootstrap)
    if before is None:
        return None
    names = {"test_sparql_file_exists": ("AssertionError", 26),
        "test_sparql_runs_without_error": ("FileNotFoundError", 35),
             "test_sparql_query_results": ("FileNotFoundError", 55)}
    tests = report["results"]["tests"]
    if len(tests) != 3 or {test.get("name") for test in tests} != {
            "test_outputs.py::" + name for name in names}:
        return None
    assertion = r"(?m)^E\s+AssertionError: SPARQL query file /app/solution\.sparql does not exist\s*$"
    # Match the full quoted absolute path, never another absent file or errno.
    missing = r"(?m)^E\s+FileNotFoundError: \[Errno 2\] No such file or directory: ['\"]/app/solution\.sparql['\"]\s*$"
    errors = r"(?m)^E\s+([A-Za-z_]\w*(?:Error|Exception))(?::|\s*$)"
    for test in tests:
        kind, line = names[test["name"].split("::", 1)[1]]
        trace = test.get("trace")
        pattern = assertion if kind == "AssertionError" else missing
        if (test.get("status") != "failed" or test.get("raw_status") != "call_failed"
                or type(test.get("retries")) is not int or test["retries"] != 0
                or test.get("file_path") != "test_outputs.py" or not isinstance(trace, str)
                or re.findall(errors, trace) != [kind] or not re.search(pattern, trace)
                or f"/tests/test_outputs.py:{line}:" not in trace
                or (kind == "FileNotFoundError" and "query_text = SPARQL_PATH.read_text()" not in trace)):
            return None
    raw = read(path / "verifier/test-stdout.txt", 16 * 1024**2)
    stdout = raw.decode("utf-8")
    if (Counter(re.findall(errors, stdout)) != Counter({"AssertionError": 1, "FileNotFoundError": 2})
            or not re.search(assertion, stdout) or len(re.findall(missing, stdout)) != 2
            or any(name not in stdout for name in names)):
        return None
    return {"kind": "reviewed-original-sparql-missing-output-v1", "missing_output": "/app/solution.sparql",
            "original_verifier_sources": before, "stdout_sha256": hashlib.sha256(raw).hexdigest(),
            "ctrf_sha256": hashlib.sha256(read(path / "verifier/ctrf.json", 16 * 1024**2)).hexdigest(),
            "original_reward_preserved": True, "failed_call_cases": sorted(names)}


def reviewed_sparql_query_syntax_failure(path, report, terminal, bootstrap):
    """Recognize the reviewed parser failure of the model's existing query.

    Graph/input parsing, fixture/setup failures and arbitrary runtime errors
    remain unscored. Existing excluded receipts are never rewritten or reused.
    """
    from .bootstrap_payload import read
    before = reviewed_sparql_original_context(path, report, terminal, bootstrap)
    if before is None:
        return None
    names = ("test_sparql_file_exists", "test_sparql_runs_without_error", "test_sparql_query_results")
    tests = report["results"]["tests"]
    if (len(tests) != 3 or {test.get("name") for test in tests} != {
            "test_outputs.py::" + name for name in names}):
        return None
    by_name = {test["name"].split("::", 1)[1]: test for test in tests}
    for name, test in by_name.items():
        if (type(test.get("retries")) is not int or test["retries"] != 0
                or test.get("file_path") != "test_outputs.py"
                or test.get("status") != ("passed" if name == names[0] else "failed")
                or test.get("raw_status") != (None if name == names[0] else "call_failed")):
            return None
    if by_name[names[0]].get("trace") not in (None, ""):
        return None
    # Inspect all pytest E-lines, including qualified exception names and
    # exceptions such as BadSyntax that lack an Error/Exception suffix.
    e_lines = lambda text: [line.strip() for line in re.findall(r"(?m)^E\s+(.+)$", text)]
    parse = re.compile(r"pyparsing\.exceptions\.ParseException: (.+  \(at char \d+\), \(line:[1-9]\d*, col:[1-9]\d*\))\Z")
    run_trace, result_trace = (by_name[name].get("trace") for name in names[1:])
    if not isinstance(run_trace, str) or not isinstance(result_trace, str):
        return None
    run_errors, result_errors = e_lines(run_trace), e_lines(result_trace)
    if len(result_errors) != 1 or not parse.fullmatch(result_errors[0]):
        return None
    reason = parse.fullmatch(result_errors[0]).group(1)
    expected_run = [result_errors[0], "AssertionError: SPARQL query failed to run: " + reason, "assert False"]
    if (run_errors != expected_run
            or "/tests/test_outputs.py:43:" not in run_trace
            or "/tests/test_outputs.py:45: AssertionError" not in run_trace
            or "/tests/test_outputs.py:62:" not in result_trace
            or "g.query(query_text)" not in run_trace or "results = g.query(query_text)" not in result_trace
            or "query_text = SPARQL_PATH.read_text()" not in run_trace
            or "query_text = SPARQL_PATH.read_text()" not in result_trace):
        return None
    raw = read(path / "verifier/test-stdout.txt", 16 * 1024**2)
    stdout = raw.decode("utf-8")
    if (Counter(e_lines(stdout)) != Counter(expected_run + result_errors)
            or any(name not in stdout for name in names)
            or any(frame not in stdout for frame in ("/tests/test_outputs.py:43:",
                "/tests/test_outputs.py:45: AssertionError", "/tests/test_outputs.py:62:"))):
        return None
    return {"kind": "reviewed-original-sparql-query-syntax-failure-v1", "output_path": "/app/solution.sparql",
            "error_type": "pyparsing.exceptions.ParseException", "parse_message": reason,
            "original_verifier_sources": before, "stdout_sha256": hashlib.sha256(raw).hexdigest(),
            "ctrf_sha256": hashlib.sha256(read(path / "verifier/ctrf.json", 16 * 1024**2)).hexdigest(),
            "original_reward_preserved": True, "passed_call_cases": [names[0]],
            "failed_call_cases": list(names[1:]), "original_query_lines": [43, 62], "caught_assertion_line": 45}


def classify_trial(path):
    """Reward zero is valid only with completed original-verifier evidence."""
    result = read_json(path / "result.json")
    if not isinstance(result.get("task_name"), str) or not NAME.fullmatch(result["task_name"]):
        raise ValueError("Malformed trial task identity")
    row = {"task_id": result["task_name"], "trial": path.name, "status": "infrastructure_error",
           "reward": None, "official_reward": None, "cleanup_verified": False, "request_ids": [], "issues": []}
    # Preserve stock Harbor's separately parsed artifact value even when later
    # infrastructure or ambiguity checks make it ineligible as a model score.
    original = result.get("verifier_result")
    rewards = original.get("rewards") if isinstance(original, dict) else None
    if isinstance(rewards, dict):
        official = rewards.get("reward")
        if type(official) in (int, float) and math.isfinite(official):
            row["official_reward"] = official
    try:
        receipt = read_json(path / "pouch-environment.json")
        row["cleanup_verified"] = receipt.get("cleanup_verified") is True
        row["container_id"] = receipt.get("container_id")
        bootstrap = read_json(path / "offline-bootstrap.json")
        terminal = read_json(path / "original-verifier-terminal.json")
        agent = read_json(path / "agent/file-relay/agent-result.json")
        ids = agent.get("request_ids")
        if not isinstance(ids, list) or len(ids) > 1024 or any(
                not isinstance(value, str) or not re.fullmatch("[a-f0-9]{32}", value) for value in ids):
            raise ValueError("Invalid request ID evidence")
        row["request_ids"], row["agent_stop_reason"] = ids, agent.get("stop_reason")
        if bootstrap.get("preprovisioned") is not True or terminal.get("terminal_proven_by_provider") is not True:
            raise ValueError("Missing bootstrap or original-verifier terminal proof")
        report = read_json(path / "verifier/ctrf.json", 16 * 1024**2)
        counts = ctrf_counts(report)
        verifier = result.get("verifier_result")
        if not isinstance(verifier, dict) or not isinstance(verifier.get("rewards"), dict):
            raise ValueError("Missing original verifier reward")
        reward = verifier["rewards"].get("reward")
        if type(reward) not in (int, float) or reward not in (0, 1) or reward != (0 if counts["failed"] else 1):
            raise ValueError("Reward and original assertion outcomes disagree")
        exception_info = result.get("exception_info")
        if exception_info is not None and not isinstance(exception_info, dict):
            raise ValueError("Malformed Harbor exception information")
        exception = (exception_info or {}).get("exception_type")
        if exception not in (None, "AgentTimeoutError"):
            raise ValueError("Harbor reported an infrastructure exception")
        if agent.get("error_type") is not None:
            raise ValueError("Model/tool transport failure is not a benchmark score")
        ambiguous = verifier_ambiguities(report, terminal)
        if (row["task_id"] == "sparql-university" and reward == 0 and ambiguous == [
                "Failure is not proven to be a test-call assertion; review original verifier logs"]):
            reviewed = reviewed_sparql_missing_output(path, report, terminal, bootstrap)
            if reviewed is None:
                reviewed = reviewed_sparql_query_syntax_failure(path, report, terminal, bootstrap)
            if reviewed is not None:
                row["verifier_failure_classification"] = reviewed
                ambiguous = []
        row["issues"].extend(ambiguous)
        row.update(status="review_required" if ambiguous else "evaluated",
                   reward=None if ambiguous else reward, official_reward=reward,
                   ctrf_records=counts, ctrf_records_are_exact_pytest_case_count=False,
                   source_budget_exhausted=exception == "AgentTimeoutError")
    except (ValueError, OSError, KeyError, TypeError) as error:
        row["issues"].append(type(error).__name__ + ": " + str(error))
    if not row["cleanup_verified"]:
        row["issues"].append("Owned container cleanup is unverified; stop further admission")
    return row
