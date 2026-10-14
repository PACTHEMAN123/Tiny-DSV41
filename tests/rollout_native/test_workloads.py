import concurrent.futures
import time
import unittest

from rollout_manager.workloads import GroupSchedule, suite_groups


class WorkloadTests(unittest.TestCase):
    def cases(self, count):
        return [{"task": {"task_id": str(i)}, "request": {"seed": 0}} for i in range(count)]

    def test_full_coverage_and_reproducible_distinct_sampling(self):
        cases = self.cases(270)
        groups = suite_groups(cases, 8, 2, 42)
        self.assertEqual(len(groups), 540)
        samples = [s for _, group in groups for s in group]
        identities = {(s["task"]["task_id"], s["rollout"]["sample_index"],
                       s["rollout"]["pass_index"]) for s in samples}
        self.assertEqual(len(identities), 4320)
        self.assertEqual(groups, suite_groups(cases, 8, 2, 42))
        self.assertEqual(len({s["request"]["seed"] for s in samples}), 4320)
        self.assertTrue(all(c["request"]["seed"] == 0 for c in cases))

    def test_slow_group_bounds_admission_and_retains_failed_samples(self):
        schedule = GroupSchedule(suite_groups(self.cases(17), 8, 1, 42), 2)
        results = []
        def worker():
            while (case := schedule.take(lambda: False)) is not None:
                with schedule.condition:
                    self.assertLessEqual(len(schedule.active), 2)
                start = time.time_ns()
                if case["rollout"]["sample_index"] == 0:
                    time.sleep(.003)
                record = {"job": str(case["rollout"]), "started_unix_ns": start,
                          "finished_unix_ns": time.time_ns(), "final_correct": False}
                schedule.complete(case, record)
                results.append(record)
        with concurrent.futures.ThreadPoolExecutor(max_workers=13) as pool:
            list(pool.map(lambda _: worker(), range(13)))
        summary = schedule.summary()
        self.assertEqual(len(results), 136)
        self.assertEqual(summary["completed_groups"], 17)
        self.assertEqual(summary["incomplete_groups"], {})
        self.assertTrue(all(g["episodes"] == 8 and g["verified"] == 0 for g in summary["groups"]))

    def test_cancelled_schedule_reports_incomplete_group(self):
        schedule = GroupSchedule(suite_groups(self.cases(3), 8, 1, 42), 1)
        self.assertIsNotNone(schedule.take(lambda: False))
        self.assertIsNone(schedule.take(lambda: True))
        self.assertEqual(schedule.summary()["pending_groups"], 2)
        self.assertEqual(len(schedule.summary()["incomplete_groups"]), 1)
