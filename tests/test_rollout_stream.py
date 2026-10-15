import asyncio
import threading
import unittest

from rollout_manager.contracts import (EpisodeIdentity, EpisodePlan, EpisodeResult,
                                       ResourceRequirements, TaskSpec)
from rollout_manager.coordinator import Coordinator


def plan(group, sample=0):
    task = TaskSpec("prompt-response", "prompt", "a" * 64, {"prompt_response": {}},
                    ResourceRequirements("cpu", cpu=1))
    return EpisodePlan(EpisodeIdentity("stream", "prompt-response", "prompt", str(group), sample), task)


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_group_spans_cpu_waves_and_pause_drains_registered_siblings(self):
        calls, entered = [], asyncio.Event()
        coordinator = None
        test = self

        class Backend:
            async def admit_group(self, selected):
                calls.append(("admit", [p.identity.sample_index for p in selected]))

            async def execute(self, item, token):
                entered.set()
                await asyncio.sleep(.005)
                return EpisodeResult(item.identity, "completed", "evaluated", True, "verified", {}, {})

            async def finish_group(self, selected, observed):
                test.assertEqual(coordinator.reserved["cpu"]["cpu"], len(selected))
                calls.append(("finish", [p.identity.sample_index for p in selected]))
                return {"spool_terminal": True}

        coordinator = Coordinator({"prompt-response": Backend()}, max_workers=3,
            policy="terminal-waves", capacities={"cpu": ResourceRequirements("cpu", cpu=2)})
        await coordinator.open_stream(max_pending=5)
        task = asyncio.create_task(coordinator.execute_stream_group([plan(0, i) for i in range(5)], sequence=0))
        await entered.wait()
        await coordinator.pause_stream()
        observed = await task
        self.assertTrue(all(item.score_eligible for item in observed))
        self.assertEqual(calls, [(kind, members) for members in ([0, 1], [2, 3], [4]) for kind in ("admit", "finish")])
        self.assertEqual(coordinator.reserved["cpu"]["cpu"], 0)
        await coordinator.close_stream()

    async def test_unproved_wave_closure_invalidates_wave_and_stops_later_siblings(self):
        class Backend:
            async def admit_group(self, selected):
                pass

            async def execute(self, item, token):
                return EpisodeResult(item.identity, "completed", "evaluated", True, "verified", {}, {})

            async def finish_group(self, selected, observed):
                return {"spool_terminal": False}

        coordinator = Coordinator({"prompt-response": Backend()}, max_workers=2, policy="terminal-waves")
        await coordinator.open_stream(max_pending=4)
        observed = await coordinator.execute_stream_group([plan(0, i) for i in range(4)], sequence=0)
        self.assertEqual([r.cleanup_state for r in observed], ["unknown", "unknown", "not_required", "not_required"])
        self.assertTrue(all(not r.score_eligible for r in observed))
        self.assertEqual(coordinator.quarantined["cpu"]["cpu"], 2)
        await coordinator.close_stream()

    async def test_persistent_models_and_complete_group_larger_than_worker_pool(self):
        calls, active, peak = [], 0, 0

        class Backend:
            async def start(self):
                calls.append("start")

            async def execute(self, item, token):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(.002)
                active -= 1
                return EpisodeResult(item.identity, "completed", "native_reward", True, "verified", {}, {})

            async def close(self):
                calls.append("close")

        coordinator = Coordinator({"prompt-response": Backend()}, max_workers=2,
                                  capacities={"cpu": ResourceRequirements("cpu", cpu=2)})
        await coordinator.open_stream(max_pending=5)
        for group in range(2):
            results = await coordinator.execute_stream_group([plan(group, i) for i in range(5)], sequence=group)
            self.assertEqual([r.identity.sample_index for r in results], list(range(5)))
            self.assertTrue(all(r.score_eligible for r in results))
            await coordinator.pause_stream()
            self.assertEqual(calls, ["start"])
            coordinator.resume_stream()
        with self.assertRaisesRegex(ValueError, "monotonic"):
            await coordinator.execute_stream_group([plan(0)], sequence=0)
        await coordinator.close_stream()
        self.assertEqual(calls, ["start", "close"])
        self.assertEqual(peak, 2)
        self.assertEqual(coordinator.reserved["cpu"]["cpu"], 0)

    async def test_pending_credit_and_cancelled_waiter_drain_actual_sampler(self):
        entered, release, drained = threading.Event(), threading.Event(), threading.Event()

        class Backend:
            async def execute(self, item, token):
                def work():
                    entered.set()
                    release.wait(2)
                    drained.set()
                await asyncio.to_thread(work)
                return EpisodeResult(item.identity, "completed", "native_reward", True, "verified", {}, {})

        coordinator = Coordinator({"prompt-response": Backend()})
        await coordinator.open_stream(max_pending=1)
        waiter = asyncio.create_task(coordinator.execute_stream_group([plan(0)], sequence=0))
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            with self.assertRaisesRegex(ValueError, "credit"):
                await coordinator.execute_stream_group([plan(1)], sequence=1)
            waiter.cancel()
            await asyncio.sleep(.01)
            self.assertFalse(waiter.done())
            self.assertEqual(coordinator.reserved["cpu"]["cpu"], 1)
            waiter.cancel()
            release.set()
            observed = await waiter
            self.assertTrue(drained.is_set())
            self.assertEqual(observed[0].cleanup_state, "verified")
            self.assertEqual(coordinator.reserved["cpu"]["cpu"], 0)
        finally:
            release.set()
            await waiter
            await coordinator.close_stream()


if __name__ == "__main__":
    unittest.main()
