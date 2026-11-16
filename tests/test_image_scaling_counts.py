import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from modules.dataLoader.ImageScalingCounts import ImageScalingCounts

from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class CountSource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, rows, targets):
        super().__init__()
        self.rows = rows
        self.targets = targets

    def length(self):
        return len(self.rows)

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ['image_path', 'original_resolution', 'settings', 'concept']

    def get_item(self, variation, index, requested_name=None):
        path, resolution, override = self.rows[index]
        return {
            'image_path': path,
            'original_resolution': resolution,
            'settings': {'target_resolution': self.targets},
            'concept': {'image': {
                'enable_resolution_override': override is not None,
                'resolution_override': override,
            }},
        }


class ScalingProbe(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, console):
        super().__init__()
        self.console = console
        self.calls = 0

    def length(self):
        return self._get_previous_length('original_resolution')

    def get_inputs(self):
        return ['original_resolution']

    def get_outputs(self):
        return ['scaled']

    def get_item(self, variation, index, requested_name=None):
        assert 'Image scaling preflight:' in self.console.getvalue(), 'Scaling began before the count was logged'
        self.calls += 1
        return {'scaled': index}


class ImageScalingCountsTest(unittest.TestCase):
    def make_pipeline(self, rows, targets):
        console = io.StringIO()
        counter = ImageScalingCounts()
        probe = ScalingProbe(console)
        state = PipelineState(1)
        self.addCleanup(state.executor.shutdown)
        pipeline = LoadingPipeline(torch.device('cpu'), [
            CountSource(rows, targets), counter, probe, OutputPipelineModule(['scaled']),
        ], 1, 42, state)
        return pipeline, counter, probe, console

    def test_independent_targets_boundaries_and_unique_paths_before_scaling(self):
        pipeline, counter, probe, console = self.make_pipeline([
            ('a.png', (900, 1000), None),
            ('b.png', (1100, 1000), None),
            ('c.png', (899, 1000), None),
            ('./c.png', (899, 1000), None),
            ('d.png', (1101, 1000), None),
            ('e.png', (2000, 2000), None),
        ], '1000,2000,1000')
        with redirect_stdout(console):
            pipeline.start_next_epoch()
            self.assertEqual(probe.calls, 0)
            self.assertEqual(counter.counts, {'1000': 3, '2000': 4})
            first_log = console.getvalue()
            self.assertIn('target 1000: 3 of 5 unique images', first_log)
            self.assertIn('target 2000: 4 of 5 unique images', first_log)
            self.assertEqual(len(first_log.splitlines()), 2)
            list(pipeline)
            pipeline.start_next_epoch()
            list(pipeline)
            self.assertEqual(console.getvalue(), first_log)
            self.assertEqual(counter.counts, {'1000': 3, '2000': 4})
            self.assertEqual(probe.calls, 12)

    def test_concept_overrides_and_fixed_resolution_pixel_count(self):
        pipeline, counter, _, console = self.make_pipeline([
            ('a.png', (512, 512), None),
            ('b.png', (256, 256), '256'),
            ('c.png', (480, 640), '640x480'),
            ('b.png', (256, 256), '256,512'),
        ], '512,1024')
        with redirect_stdout(console):
            pipeline.start_next_epoch()
        self.assertEqual(counter.counts, {'512': 1, '1024': 1, '256': 0, '640x480': 0})
        self.assertIn('target 256: 0 of 1 unique images', console.getvalue())

    def test_other_gpu_ranks_do_not_repeat_the_scan_or_log(self):
        pipeline, counter, _, console = self.make_pipeline([('a.png', (100, 100), None)], '512')
        with patch('modules.dataLoader.ImageScalingCounts.multi.is_master', return_value=False), redirect_stdout(console):
            pipeline.start_next_epoch()
        self.assertEqual(counter.counts, {})
        self.assertEqual(console.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
