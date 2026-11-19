from threading import Lock
from time import monotonic
from typing import Any

from mgds.pipelineModules.DiskCache import DiskCache

import torch


class VramAwareDiskCache(DiskCache):
    """Release unused CUDA allocator memory under pressure during cache writes."""

    _CHECK_INTERVAL_SECONDS = 1.0
    _RELEASE_INTERVAL_SECONDS = 5.0
    _MIN_FREE_FRACTION = 0.10
    _MIN_RELEASABLE_BYTES = 256 * 1024**2

    def init(self, pipeline, base_seed, module_index, state):
        super().init(pipeline, base_seed, module_index, state)
        self._first_cache_input = next(iter(self.split_names or self.aggregate_names), None)
        self._memory_check_lock = Lock()
        self._next_memory_check = 0.0

    def _get_previous_item(self, variation: int, name: str, index: int) -> Any:
        # MGDS starts each cache write by fetching its first split or aggregate field.
        if name == self._first_cache_input and self.pipeline.device.type == "cuda":
            self._release_cached_vram_if_needed()
        return super()._get_previous_item(variation, name, index)

    def _release_cached_vram_if_needed(self):
        # Cache workers share one check and release schedule for this pass.
        with self._memory_check_lock:
            now = monotonic()
            if now < self._next_memory_check:
                return
            self._next_memory_check = now + self._CHECK_INTERVAL_SECONDS

            device = self.pipeline.device
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            if free_bytes >= total_bytes * self._MIN_FREE_FRACTION:
                return

            if torch.cuda.get_allocator_backend() == "native":
                stats = torch.cuda.memory_stats(device)
                # Inactive split blocks still share a segment with active allocations.
                releasable_bytes = (
                    stats["reserved_bytes.all.current"]
                    - stats["active_bytes.all.current"]
                    - stats["inactive_split_bytes.all.current"]
                )
                if releasable_bytes < self._MIN_RELEASABLE_BYTES:
                    return

            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            self._next_memory_check = monotonic() + self._RELEASE_INTERVAL_SECONDS
