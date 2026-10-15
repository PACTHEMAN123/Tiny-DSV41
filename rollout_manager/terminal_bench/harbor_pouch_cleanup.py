"""Additive bounded stop acknowledgement handling for future Pouch trials."""

import asyncio

from .harbor_pouch_environment import PouchError


class CleanupPollingMixin:
    async def _wait_stopped(self):
        # Polls share the time formerly available to one post-stop inspection.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.api_timeout_sec
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise PouchError("Owned container did not stop within inspection budget")
            try:
                info = await asyncio.wait_for(self._owned(missing_ok=True), remaining)
            except asyncio.TimeoutError as error:
                raise PouchError("Owned container stop inspection timed out") from error
            if info is None:
                return None
            running = info.get("State", {}).get("Running")
            if running is False:
                return info
            if running is not True:
                raise PouchError("Owned container has unknown running state")
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise PouchError("Owned container did not stop within inspection budget")
            await asyncio.sleep(min(0.1, remaining))

    async def stop(self, delete=True):
        async with self._cleanup_lock:
            if not self._create_attempted:
                return
            if self._creation_task is not None:
                try:
                    await asyncio.shield(self._creation_task)
                except Exception:
                    self._creation_uncertain = True
            info = await self._owned(missing_ok=True)
            if info is None:
                if self._creation_uncertain:
                    self._record(cleanup_verified=False, create_outcome_unknown=True)
                    raise PouchError("Create outcome unknown; absence now cannot prove no late creation")
                self._record(cleanup_verified=True)
                return
            if info.get("State", {}).get("Running") is True:
                if getattr(self, "_cleanup_stop_sent", False):
                    raise PouchError("Previous stop attempt remains unresolved; refusing retry")
                self._cleanup_stop_sent = True
                await self.api.request("POST", "/containers/" + self.container_id + "/stop?t=1",
                                       timeout=self.api_timeout_sec)
                info = await self._wait_stopped()
                if info is None:
                    self._record(stopped=True, deleted=True, cleanup_verified=True)
                    return
            if info.get("State", {}).get("Running") is not False or info.get("State", {}).get("Paused") is True:
                raise PouchError("Owned container terminal state is unproven; refusing deletion")
            if delete:
                if getattr(self, "_cleanup_delete_sent", False):
                    raise PouchError("Previous deletion remains unresolved; refusing retry")
                self._cleanup_delete_sent = True
                await self.api.request("DELETE", "/containers/" + self.container_id + "?force=false&v=true",
                                       timeout=self.api_timeout_sec)
                if await self._owned(missing_ok=True) is not None:
                    raise PouchError("Owned container is still present after deletion")
            self._record(stopped=True, deleted=delete, cleanup_verified=delete)
