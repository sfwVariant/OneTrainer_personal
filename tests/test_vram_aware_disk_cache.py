import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.dataLoader.VramAwareDiskCache import VramAwareDiskCache

from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

MIB = 1024**2


class CacheSource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, before_read):
        super().__init__()
        self.before_read = before_read

    def length(self):
        return 8

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ['sample', 'resolution']

    def get_item(self, variation, index, requested_name=None):
        self.before_read()
        return {'sample': f'image-{index}.png', 'resolution': (512, 512)}


class VramAwareDiskCacheTest(unittest.TestCase):
    def test_checks_before_cache_writes_once_across_workers_and_skips_cache_hits(self):
        device = torch.device('cuda:1')
        empty_cache = Mock()
        mem_get_info = Mock(return_value=(64 * MIB, 8192 * MIB))
        state = PipelineState(2)
        self.addCleanup(state.executor.shutdown)

        with (
            tempfile.TemporaryDirectory() as cache_dir,
            patch.multiple(torch.cuda, is_available=Mock(return_value=False),
                           mem_get_info=mem_get_info, empty_cache=empty_cache, device=Mock(side_effect=lambda _: nullcontext()),
                           get_allocator_backend=Mock(return_value='cudaMallocAsync')),
            patch('modules.dataLoader.VramAwareDiskCache.monotonic', return_value=10.0) as clock,
            patch.object(VramAwareDiskCache, '_torch_gc'),
        ):
            cache = VramAwareDiskCache(cache_dir, split_names=['sample'], aggregate_names=['resolution'])
            pipeline = LoadingPipeline(device, [
                CacheSource(empty_cache.assert_called_once), cache, OutputPipelineModule(['sample', 'resolution']),
            ], 1, 42, state)

            pipeline.start_next_epoch()
            mem_get_info.assert_called_once_with(device)
            self.assertEqual([item['sample'] for item in pipeline], [f'image-{i}.png' for i in range(8)])

            clock.return_value = 20.0
            pipeline.start_next_epoch()
            self.assertEqual([item['resolution'] for item in pipeline], [(512, 512)] * 8)
            mem_get_info.assert_called_once_with(device)
            empty_cache.assert_called_once()

    def test_native_allocator_releases_only_under_pressure_with_reclaimable_memory(self):
        device = torch.device('cuda')
        cache = VramAwareDiskCache('unused')
        cache.init(SimpleNamespace(device=device), 0, 0, None)
        mem_get_info = Mock(return_value=(2048 * MIB, 8192 * MIB))
        stats = {
            'reserved_bytes.all.current': 7168 * MIB,
            'active_bytes.all.current': 6144 * MIB,
            'inactive_split_bytes.all.current': 896 * MIB,
        }
        memory_stats = Mock(return_value=stats)
        empty_cache = Mock()

        with (
            patch.multiple(torch.cuda, mem_get_info=mem_get_info, memory_stats=memory_stats,
                           get_allocator_backend=Mock(return_value='native'),
                           empty_cache=empty_cache, device=Mock(side_effect=lambda _: nullcontext())),
            patch('modules.dataLoader.VramAwareDiskCache.monotonic', return_value=0.0) as clock,
        ):
            cache._release_cached_vram_if_needed()
            memory_stats.assert_not_called()

            clock.return_value = 1.0
            mem_get_info.return_value = (64 * MIB, 8192 * MIB)
            cache._release_cached_vram_if_needed()
            memory_stats.assert_called_once_with(device)
            empty_cache.assert_not_called()

            stats['inactive_split_bytes.all.current'] = 512 * MIB
            clock.return_value = 2.0
            cache._release_cached_vram_if_needed()
            empty_cache.assert_called_once()

            clock.return_value = 3.0
            cache._release_cached_vram_if_needed()
            self.assertEqual(mem_get_info.call_count, 3)
            empty_cache.assert_called_once()

            clock.return_value = 7.0
            cache._release_cached_vram_if_needed()
            self.assertEqual(empty_cache.call_count, 2)
