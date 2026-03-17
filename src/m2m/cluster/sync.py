import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List

logger = logging.getLogger("m2m_sync_queue")


class SyncQueue:
    """
    Handles offline queuing for Edge Nodes.
    If the coordinator is unreachable, actions (like registering a new document location)
    are queued and flushed securely when connection restores.
    """

    def __init__(self, flush_interval_seconds: float = 10.0):
        self._queue: List[Dict[str, Any]] = []
        self._flush_interval = flush_interval_seconds
        self._is_running = False
        self._flush_task = None
        self._last_flush = 0.0

    def add_action(self, action: Dict[str, Any]):
        """Queue an action to be dispatched."""
        self._queue.append(action)

    def get_pending(self) -> List[Dict[str, Any]]:
        """Return a copy of pending actions."""
        return list(self._queue)

    def clear(self):
        """Empty the queue."""
        self._queue.clear()

    async def start_background_sync(
        self, dispatch_func: Callable[[List[Dict[str, Any]]], Coroutine[Any, Any, bool]]
    ):
        """
        Start the background sync loop.
        dispatch_func must take the list of actions, attempt to send them,
        and return True if successful, False otherwise.
        """
        if self._is_running:
            return

        self._is_running = True

        async def _loop():
            while self._is_running:
                await asyncio.sleep(self._flush_interval)

                if not self._queue:
                    continue

                # Take a snapshot to flush
                snapshot = list(self._queue)
                try:
                    success = await dispatch_func(snapshot)
                    if success:
                        self._last_flush = time.time()
                        # Remove flushed items from queue
                        self._queue = self._queue[len(snapshot) :]
                    else:
                        logger.warning(
                            f"Failed to flush {len(snapshot)} actions to coordinator. Re-queuing."
                        )
                except Exception as e:
                    logger.error(f"Error during queue flush: {e}")

        self._flush_task = asyncio.create_task(_loop())

    async def stop(self):
        """Stop the background loop."""
        self._is_running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
