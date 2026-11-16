import math

from modules.dataLoader.resolution_util import apply_resolution_tolerance, get_target_resolution

from mgds.pipelineModules.AspectBucketing import AspectBucketing


class TolerantAspectBucketing(AspectBucketing):
    """Keep near-target images at their native size while fitting a bucket crop."""

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        result = super().get_item(variation, index, requested_name)
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)
        target = get_target_resolution(self, variation, index)
        target_pixels = math.prod(target) if isinstance(target, tuple) else target**2
        return apply_resolution_tolerance(self, resolution, target_pixels, result, preserve_crop_aspect=True)
