import os

import modules.util.multi_gpu_util as multi
from modules.dataLoader.resolution_util import is_within_resolution_tolerance

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ImageScalingCounts(PipelineModule, RandomAccessPipelineModule):
    """Count unique inputs per effective target before any resize or cache operations."""

    def __init__(self):
        super().__init__()
        self.counts = {}

    def length(self) -> int:
        return self._get_previous_length('original_resolution')

    def get_inputs(self) -> list[str]:
        return ['image_path', 'original_resolution', 'settings', 'concept']

    def get_outputs(self) -> list[str]:
        return ['image_scaling_counts']

    def start(self, variation: int):
        if not multi.is_master():
            return
        images_by_target = {}
        scaled_by_target = {}
        for index in range(self.length()):
            targets = self._get_previous_item(variation, 'settings.target_resolution', index)
            if self._get_previous_item(variation, 'concept.image.enable_resolution_override', index):
                targets = self._get_previous_item(variation, 'concept.image.resolution_override', index)
            path = os.path.normcase(os.path.realpath(self._get_previous_item(variation, 'image_path', index)))
            resolution = self._get_previous_item(variation, 'original_resolution', index)
            # Count against every applicable target, independently of random target selection.
            for target in str(targets).split(','):
                target = target.strip()
                if 'x' in target:
                    width, height = (int(value.strip()) for value in target.split('x'))
                    label, pixels = f'{width}x{height}', width * height
                else:
                    side = int(target)
                    label, pixels = str(side), side**2
                images_by_target.setdefault(label, set()).add(path)
                scaled = scaled_by_target.setdefault(label, set())
                if not is_within_resolution_tolerance(resolution, pixels):
                    scaled.add(path)
        self.counts = {target: len(scaled_by_target[target]) for target in images_by_target}
        for target, images in images_by_target.items():
            print(
                f'Image scaling preflight: target {target}: {self.counts[target]} of {len(images)} unique images '
                'outside the 10% pixel-count tolerance (scaling required).',
                flush=True,
            )

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {'image_scaling_counts': self.counts}
