"""Release result classification, using synthetic original-verifier evidence."""
import json
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from rollout_manager.terminal_bench import controller as runner

def trial(path, name="fix-git", failed=False, clean=True, exception=None):
    (path / "agent/file-relay").mkdir(parents=True)
    (path / "verifier").mkdir()
    values = {
        "result.json": {"task_name": name, "verifier_result": {"rewards": {"reward": int(not failed)}},
                        "exception_info": {"exception_type": exception} if exception else None},
        "pouch-environment.json": {"cleanup_verified": clean, "container_id": "c" * 64},
        "offline-bootstrap.json": {"preprovisioned": True},
        "original-verifier-terminal.json": {"terminal_proven_by_provider": True,
                                             "started_unix": 100, "finished_unix": 110},
        "agent/file-relay/agent-result.json": {"request_ids": ["d" * 32], "error_type": None,
                                               "stop_reason": "model_final"},
        "verifier/ctrf.json": {"results": {"tool": {"name": "pytest"},
            "summary": {"tests": 2, "passed": int(not failed) + 1, "failed": int(failed),
                        "skipped": 0, "pending": 0, "other": 0, "start": 101, "stop": 109},
            "tests": [{"status": "passed"}, {"status": "failed" if failed else "passed",
                "raw_status": "call_failed" if failed else None,
                "message": "The test failed in the call phase due to an assertion error" if failed else None}]}}
    }
    for name, data in values.items():
        runner.write_new(path / name, data)

class Classification(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trial"

    def test_pass_and_valid_assertion_zero(self):
        trial(self.path, failed=True)
        result = runner.classify_trial(self.path)
        self.assertEqual((result["status"], result["reward"]), ("evaluated", 0))
        self.assertEqual(result["ctrf_records"]["tests"], 2)
        self.assertFalse(result["ctrf_records_are_exact_pytest_case_count"])

    def test_agent_timeout_with_complete_verifier_is_budget_outcome(self):
        trial(self.path, exception="AgentTimeoutError")
        self.assertTrue(runner.classify_trial(self.path)["source_budget_exhausted"])

    def test_missing_report_not_fake_zero(self):
        trial(self.path, failed=True)
        (self.path / "verifier/ctrf.json").unlink()
        result = runner.classify_trial(self.path)
        self.assertEqual(result["status"], "infrastructure_error")
        self.assertIsNone(result["reward"])
        self.assertTrue(result["cleanup_verified"])

    def test_transport_error_with_reward_stays_infrastructure(self):
        trial(self.path, exception="RelayError")
        result = runner.classify_trial(self.path)
        self.assertEqual(result["status"], "infrastructure_error")
        self.assertIsNone(result["reward"])
        self.assertEqual(result["official_reward"], 1)

    def test_cleanup_separate_from_score(self):
        trial(self.path, clean=False)
        result = runner.classify_trial(self.path)
        self.assertEqual(result["status"], "evaluated")
        self.assertFalse(result["cleanup_verified"])

    def test_fake_qualified_flag_does_not_replace_evidence(self):
        trial(self.path)
        (self.path / "offline-bootstrap.json").write_text('{"qualified": true}')
        self.assertEqual(runner.classify_trial(self.path)["status"], "infrastructure_error")

    def test_counts_and_reward_consistency(self):
        trial(self.path)
        report = runner.read_json(self.path / "verifier/ctrf.json")
        report["results"]["summary"]["passed"] = True
        with self.assertRaises(ValueError):
            runner.ctrf_counts(report)
        report["results"]["summary"]["passed"] = 1
        with self.assertRaises(ValueError):
            runner.ctrf_counts(report)

    def test_fixture_failure_preserves_reward_but_requires_review(self):
        trial(self.path, failed=True)
        report = runner.read_json(self.path / "verifier/ctrf.json")
        report["results"]["tests"][1]["raw_status"] = "setup_failed"
        (self.path / "verifier/ctrf.json").write_text(json.dumps(report))
        result = runner.classify_trial(self.path)
        self.assertEqual((result["status"], result["official_reward"]), ("review_required", 0))
        self.assertIsNone(result["reward"])

    def test_missing_failed_phase_and_exception_need_review(self):
        trial(self.path, failed=True)
        report = runner.read_json(self.path / "verifier/ctrf.json")
        terminal = runner.read_json(self.path / "original-verifier-terminal.json")
        report["results"]["tests"][1].pop("raw_status")
        self.assertTrue(runner.verifier_ambiguities(report, terminal))
        report["results"]["tests"][1]["raw_status"] = "call_failed"
        report["results"]["tests"][1]["message"] = "The test failed in the call phase due to an exception"
        self.assertTrue(runner.verifier_ambiguities(report, terminal))

    def test_stale_report_requires_review(self):
        trial(self.path)
        report = runner.read_json(self.path / "verifier/ctrf.json")
        report["results"]["summary"]["start"] = 80
        (self.path / "verifier/ctrf.json").write_text(json.dumps(report))
        self.assertEqual(runner.classify_trial(self.path)["status"], "review_required")


class SparqlMissingOutput(unittest.TestCase):
    """Original reward0 is usable only for the exact reviewed output failure."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trial"
        trial(self.path, name="sparql-university", failed=True)
        sources = {"schema_version": 1, "task_id": "sparql-university", "files": {
            "tests/test.sh": {"bytes": 629, "sha256":
                "45902b57bd0f1dd6959216bc9943ab27bb349edc7ff557035a1ecc8732e4cc6b"},
            "tests/test_outputs.py": {"bytes": 2399, "sha256":
                "ecf15840fc14f1eeff2f3b6317d7f0febc8fa81f688d24e3054205ba13207b39"},
            "provenance.json": {"bytes": 7637, "sha256":
                "184faa0638cfd08499f51d53a7bf64a492daf1c14ec4f6aa9d889483678e228e"}}}
        self.terminal = {"terminal_proven_by_provider": True, "return_code": 0,
            "command_sha256": "c" * 64, "started_unix": 100, "finished_unix": 110,
            "original_verifier_sources_before": sources, "original_verifier_sources_after": copy.deepcopy(sources)}
        self.bootstrap = {"preprovisioned": True, "task_id": "sparql-university", "manifest_sha256":
            "6abd7da26fc66958cd36274edebe7720da9eb24a62ceeb0e6c2ffaed8c9521f5"}
        self.prerequisites = {"task_id": "sparql-university", "verified": True, "checks": [
            {"name": name, "return_code": 0, "expected_observation_matches": True,
             "diagnostics": {"stderr_python_error_types": []}} for name in ("rdflib_runtime", "sparql_input")]}
        names = ("test_sparql_file_exists", "test_sparql_runs_without_error", "test_sparql_query_results")
        traces = (
            '> assert SPARQL_PATH.exists(), f"SPARQL query file {SPARQL_PATH} does not exist"\n'
            'E AssertionError: SPARQL query file /app/solution.sparql does not exist\nE assert False\n'
            "/tests/test_outputs.py:26: AssertionError",
            "> query_text = SPARQL_PATH.read_text()\nself = PosixPath('/app/solution.sparql')\n"
            "E FileNotFoundError: [Errno 2] No such file or directory: '/app/solution.sparql'\n"
            "/tests/test_outputs.py:35:",
            "> query_text = SPARQL_PATH.read_text()\nself = PosixPath('/app/solution.sparql')\n"
            "E FileNotFoundError: [Errno 2] No such file or directory: '/app/solution.sparql'\n"
            "/tests/test_outputs.py:55:")
        self.report = {"results": {"tool": {"name": "pytest", "version": "8.4.1"},
            "summary": {"tests": 3, "passed": 0, "failed": 3, "skipped": 0, "pending": 0,
                        "other": 0, "start": 101, "stop": 109},
            "tests": [{"name": "test_outputs.py::" + name, "status": "failed", "raw_status": "call_failed",
                "retries": 0, "file_path": "test_outputs.py", "trace": trace,
                "message": "The test failed in the call phase" + (" due to an assertion error" if i == 0 else "")}
                for i, (name, trace) in enumerate(zip(names, traces))]}}
        self.stdout = "\n".join(name + "\n" + trace for name, trace in zip(names, traces))
        self.save()

    def save(self):
        for name, value in (("original-verifier-terminal.json", self.terminal),
                            ("offline-bootstrap.json", self.bootstrap),
                            ("agent-prerequisites.json", self.prerequisites),
                            ("verifier/ctrf.json", self.report)):
            (self.path / name).write_text(json.dumps(value))
        (self.path / "verifier/test-stdout.txt").write_text(self.stdout)

    def test_exact_missing_output_preserves_original_zero_and_source_evidence(self):
        row = runner.classify_trial(self.path)
        self.assertEqual((row["status"], row["reward"], row["official_reward"]), ("evaluated", 0, 0))
        proof = row["verifier_failure_classification"]
        self.assertEqual(proof["missing_output"], "/app/solution.sparql")
        self.assertTrue(proof["original_reward_preserved"])
        self.assertEqual(proof["stdout_sha256"], hashlib.sha256(self.stdout.encode()).hexdigest())
        self.assertEqual(proof["ctrf_sha256"], hashlib.sha256((self.path / "verifier/ctrf.json").read_bytes()).hexdigest())
        # The A seal importer calls this same classifier on unchanged raw files.
        self.assertEqual(row, runner.classify_trial(self.path))

    def test_missing_input_permission_import_and_fixture_remain_unscored(self):
        pristine = copy.deepcopy(self.report)
        for failure in ("E FileNotFoundError: [Errno 2] No such file or directory: '/tests/university_graph_test.ttl'",
                        "E FileNotFoundError: [Errno 2] No such file or directory: '/app/university_graph.ttl'",
                        "E PermissionError: [Errno 13] Permission denied: '/app/solution.sparql'",
                        "E ModuleNotFoundError: No module named 'rdflib'"):
            with self.subTest(failure=failure):
                self.report = copy.deepcopy(pristine)
                self.report["results"]["tests"][1]["trace"] = failure
                self.save()
                row = runner.classify_trial(self.path)
                self.assertEqual((row["status"], row["reward"], row["official_reward"]), ("review_required", None, 0))
        self.report = copy.deepcopy(pristine)
        self.report["results"]["tests"][1]["raw_status"] = "setup_failed"
        self.save()
        self.assertEqual(runner.classify_trial(self.path)["status"], "review_required")

    def test_wrong_source_changed_during_execution_and_old_evidence_fail_closed(self):
        pristine = copy.deepcopy(self.terminal)
        for mutation in ("wrong_source", "changed_after", "no_source", "bad_command", "bad_return"):
            with self.subTest(mutation=mutation):
                self.terminal = copy.deepcopy(pristine)
                if mutation == "wrong_source":
                    for key in ("original_verifier_sources_before", "original_verifier_sources_after"):
                        self.terminal[key]["files"]["tests/test.sh"]["sha256"] = "f" * 64
                elif mutation == "changed_after":
                    self.terminal["original_verifier_sources_after"]["files"]["tests/test_outputs.py"]["sha256"] = "f" * 64
                elif mutation == "no_source":
                    self.terminal.pop("original_verifier_sources_before")
                    self.terminal.pop("original_verifier_sources_after")
                elif mutation == "bad_command":
                    self.terminal["command_sha256"] = "not-a-hash"
                else:
                    self.terminal["return_code"] = 1
                self.save()
                self.assertEqual(runner.classify_trial(self.path)["status"], "review_required")

    def test_stale_time_task_identity_case_mix_and_stdout_disagreement_rejected(self):
        pristine = copy.deepcopy(self.report)
        original_stdout = self.stdout
        for mutation in ("stale", "other_task", "case_name", "duplicate_case", "retry", "mixed", "plugin", "stdout"):
            with self.subTest(mutation=mutation):
                self.report, self.stdout = copy.deepcopy(pristine), original_stdout
                result = runner.read_json(self.path / "result.json")
                result["task_name"] = "sparql-university"
                if mutation == "stale":
                    self.report["results"]["summary"]["start"] = 80
                elif mutation == "other_task":
                    result["task_name"] = "merge-diff-arc-agi-task"
                elif mutation == "case_name":
                    self.report["results"]["tests"][1]["name"] = "test_outputs.py::test_unrelated"
                elif mutation == "duplicate_case":
                    self.report["results"]["tests"][2] = copy.deepcopy(self.report["results"]["tests"][1])
                elif mutation == "retry":
                    self.report["results"]["tests"][1]["retries"] = 1
                elif mutation == "mixed":
                    self.report["results"]["tests"][0]["status"] = "passed"
                    self.report["results"]["summary"].update(passed=1, failed=2)
                elif mutation == "plugin":
                    self.report["results"]["tool"]["version"] = "different-runtime"
                else:
                    self.stdout = self.stdout.replace("/app/solution.sparql", "/tests/university_graph_test.ttl")
                (self.path / "result.json").write_text(json.dumps(result))
                self.save()
                self.assertEqual(runner.classify_trial(self.path)["status"], "review_required")

    def test_prerequisite_and_transport_failure_do_not_become_task_zero(self):
        self.prerequisites["verified"] = False
        self.save()
        self.assertEqual(runner.classify_trial(self.path)["status"], "review_required")
        self.prerequisites["verified"] = True
        self.save()
        agent_path = self.path / "agent/file-relay/agent-result.json"
        agent = runner.read_json(agent_path)
        agent["error_type"] = "RelayError"
        agent_path.write_text(json.dumps(agent))
        row = runner.classify_trial(self.path)
        self.assertEqual((row["status"], row["reward"], row["official_reward"]), ("infrastructure_error", None, 0))

    def test_source_observer_hashes_actual_bytes_and_rejects_links(self):
        task = Path(self.temp.name).resolve() / "copy/tasks/sparql-university"
        (task / "tests").mkdir(parents=True)
        (task / "environment").mkdir()
        (task.parent.parent / "provenance.json").write_bytes(b"original provenance")
        (task / "tests/test.sh").write_bytes(b"original script")
        (task / "tests/test_outputs.py").write_bytes(b"original tests")
        before = runner.original_verifier_sources(task / "environment")
        self.assertEqual(before["files"]["tests/test.sh"], {"bytes": 15,
            "sha256": hashlib.sha256(b"original script").hexdigest()})
        (task / "tests/test.sh").write_bytes(b"changed script")
        self.assertNotEqual(before, runner.original_verifier_sources(task / "environment"))
        (task / "tests/test.sh").unlink()
        (task / "tests/test.sh").symlink_to(task / "tests/test_outputs.py")
        with self.assertRaisesRegex(ValueError, "Linked"):
            runner.original_verifier_sources(task / "environment")


class SparqlQuerySyntax(unittest.TestCase):
    """Synthetic fixture derived from the actual sample4 compact trace review."""
    save = SparqlMissingOutput.save

    def setUp(self):
        SparqlMissingOutput.setUp(self)
        reason = "Expected {SelectQuery | ConstructQuery | DescribeQuery | AskQuery}, found '\\'  (at char 43), (line:1, col:44)"
        parse = "E           pyparsing.exceptions.ParseException: " + reason
        run_trace = "\n".join(("query_text = SPARQL_PATH.read_text()", ">           g.query(query_text)",
            "/tests/test_outputs.py:43:", parse, "g.query(query_text)",
            "E           AssertionError: SPARQL query failed to run: " + reason,
            "E           assert False", "/tests/test_outputs.py:45: AssertionError"))
        result_trace = "\n".join(("query_text = SPARQL_PATH.read_text()", ">       results = g.query(query_text)",
            "/tests/test_outputs.py:62:", parse))
        self.report["results"]["summary"].update(passed=1, failed=2)
        tests = self.report["results"]["tests"]
        tests[0].update(status="passed", raw_status=None, trace="", message="None")
        tests[1].update(trace=run_trace, message="The test failed in the call phase due to an assertion error")
        tests[2].update(trace=result_trace, message="The test failed in the call phase due to an exception")
        self.stdout = "\n".join(test["name"] + "\n" + test["trace"] for test in tests)
        self.save()

    def test_existing_output_query_parse_failure_preserves_original_zero(self):
        before = {path: path.read_bytes() for path in self.path.rglob("*") if path.is_file()}
        result = runner.classify_trial(self.path)
        self.assertEqual((result["status"], result["reward"], result["official_reward"]), ("evaluated", 0, 0))
        proof = result["verifier_failure_classification"]
        self.assertEqual(proof["kind"], "reviewed-original-sparql-query-syntax-failure-v1")
        self.assertEqual(proof["original_query_lines"], [43, 62])
        self.assertEqual(proof["caught_assertion_line"], 45)
        self.assertEqual(proof["stdout_sha256"], hashlib.sha256(self.stdout.encode()).hexdigest())
        self.assertTrue(proof["original_reward_preserved"])
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_graph_input_permission_and_runtime_failures_remain_review_required(self):
        pristine = copy.deepcopy(self.report)
        for replacement in ("RuntimeError", "rdflib.plugins.parsers.notation3.BadSyntax",
                            "FileNotFoundError", "PermissionError", "ModuleNotFoundError"):
            with self.subTest(exception=replacement):
                self.report = copy.deepcopy(pristine)
                for test in self.report["results"]["tests"]:
                    test["trace"] = test["trace"].replace("pyparsing.exceptions.ParseException", replacement)
                self.stdout = "\n".join(test["name"] + "\n" + test["trace"] for test in self.report["results"]["tests"])
                self.save()
                result = runner.classify_trial(self.path)
                self.assertEqual((result["status"], result["reward"], result["official_reward"]), ("review_required", None, 0))
        self.report = copy.deepcopy(pristine)
        for test in self.report["results"]["tests"]:
            test["trace"] = test["trace"].replace("g.query(query_text)", "g.parse(str(DATA_PATH), format='ttl')").replace(":43:", ":40:").replace(":62:", ":59:")
        self.stdout = "\n".join(test["name"] + "\n" + test["trace"] for test in self.report["results"]["tests"])
        self.save()
        self.assertEqual(runner.classify_trial(self.path)["status"], "review_required")

    def test_other_query_syntax_messages_require_the_same_original_query_frames(self):
        pristine, original_stdout = copy.deepcopy(self.report), self.stdout
        original = runner.classify_trial(self.path)["verifier_failure_classification"]["parse_message"]
        for reason in ("Expected PrefixDecl, found 'SELECT'  (at char 7), (line:1, col:8)",
                       "Expected string_literal, found end of text  (at char 81), (line:2, col:35)"):
            with self.subTest(parse_message=reason):
                self.report = copy.deepcopy(pristine)
                for test in self.report["results"]["tests"]:
                    test["trace"] = test["trace"].replace(original, reason)
                self.stdout = original_stdout.replace(original, reason)
                self.save()
                result = runner.classify_trial(self.path)
                self.assertEqual((result["status"], result["reward"], result["official_reward"]), ("evaluated", 0, 0))
                self.assertEqual(result["verifier_failure_classification"]["parse_message"], reason)

    def test_source_prerequisite_retry_frame_time_and_stdout_mismatches_rejected(self):
        pristine = (copy.deepcopy(self.report), copy.deepcopy(self.terminal), copy.deepcopy(self.prerequisites), self.stdout)
        for mutation in ("source", "prerequisite", "retry", "fixture", "frame", "stale", "stdout", "file_exists", "location", "different_detail", "predicate"):
            with self.subTest(mutation=mutation):
                self.report, self.terminal, self.prerequisites, self.stdout = copy.deepcopy(pristine)
                test = self.report["results"]["tests"][2]
                if mutation == "source":
                    self.terminal["original_verifier_sources_after"]["files"]["tests/test_outputs.py"]["sha256"] = "f" * 64
                elif mutation == "prerequisite":
                    self.prerequisites["checks"][1]["expected_observation_matches"] = False
                elif mutation == "retry":
                    test["retries"] = 1
                elif mutation == "fixture":
                    test["raw_status"] = "setup_failed"
                elif mutation == "frame":
                    test["trace"] = test["trace"].replace(":62:", ":59:")
                elif mutation == "stale":
                    self.report["results"]["summary"]["start"] = 80
                elif mutation == "stdout":
                    self.stdout += "\nE RuntimeError: fixture setup failed"
                elif mutation == "location":
                    for case in self.report["results"]["tests"]:
                        case["trace"] = case["trace"].replace("(line:1, col:44)", "(line:0, col:0)")
                    self.stdout = self.stdout.replace("(line:1, col:44)", "(line:0, col:0)")
                elif mutation == "different_detail":
                    test["trace"] = test["trace"].replace("char 43", "char 44")
                    self.stdout = "\n".join(case["name"] + "\n" + case["trace"] for case in self.report["results"]["tests"])
                elif mutation == "predicate":
                    test["trace"] = test["trace"].replace("results = g.query(query_text)", "normalize_countries(row.countries)").replace(":62:", ":71:")
                    self.stdout = "\n".join(case["name"] + "\n" + case["trace"] for case in self.report["results"]["tests"])
                else:
                    self.report["results"]["tests"][0].update(status="failed", raw_status="call_failed")
                    self.report["results"]["summary"].update(passed=0, failed=3)
                self.save()
                result = runner.classify_trial(self.path)
                self.assertEqual((result["status"], result["reward"], result["official_reward"]), ("review_required", None, 0))

    def test_transport_failure_and_cleanup_still_have_independent_meanings(self):
        agent_path = self.path / "agent/file-relay/agent-result.json"
        agent = runner.read_json(agent_path)
        agent["error_type"] = "RelayError"
        agent_path.write_text(json.dumps(agent))
        result = runner.classify_trial(self.path)
        self.assertEqual((result["status"], result["reward"]), ("infrastructure_error", None))
        agent["error_type"] = None
        agent_path.write_text(json.dumps(agent))
        cleanup = self.path / "pouch-environment.json"
        receipt = runner.read_json(cleanup)
        receipt["cleanup_verified"] = False
        cleanup.write_text(json.dumps(receipt))
        result = runner.classify_trial(self.path)
        self.assertEqual((result["status"], result["reward"], result["cleanup_verified"]), ("evaluated", 0, False))
