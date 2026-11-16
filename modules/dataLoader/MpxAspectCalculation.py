import math

from modules.dataLoader.resolution_util import apply_resolution_tolerance, get_target_resolution

from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation


class MpxAspectCalculation(SingleAspectCalculation):
    """Calculate a separate, model-compatible target resolution for each image."""

    def __init__(self, quantization: int, **kwargs):
        super().__init__(**kwargs)
        self.quantization = quantization

    def start(self, variation: int):
        # Unlike fixed buckets, the possible dimensions depend on the input images.
        pass

    def get_inputs(self) -> list[str]:
        return [name for name in (
            self.resolution_in_name,
            self.target_resolutions_in_name,
            self.enable_target_resolutions_override_in_name,
            self.target_resolutions_override_in_name,
        ) if name is not None]

    def get_meta(self, variation: int, name: str):
        if name == self.possible_resolutions_out_name:
            return (
                self.get_item(variation, index)[self.crop_resolution_out_name]
                for index in range(self.length())
            )
        return None

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)
        target = get_target_resolution(self, variation, index)
        height, width = resolution[-2:]
        if isinstance(target, tuple):
            # An explicit WxH resolution remains an explicit crop, as with bucketing.
            target_height, target_width = target
            target_pixels = math.prod(target)
        else:
            target_pixels = target**2
            scale = target / math.sqrt(height * width)
            target_height, target_width = height * scale, width * scale

        target_height = max(self.quantization, round(target_height / self.quantization) * self.quantization)
        target_width = max(self.quantization, round(target_width / self.quantization) * self.quantization)

        # Resize uniformly to cover the rounded dimensions, then crop the small excess.
        # This avoids stretching the image when dimension rounding changes its aspect ratio.
        if height * target_width > width * target_height:
            scale_height = round(height * target_width / width)
            scale_width = target_width
        else:
            scale_height = target_height
            scale_width = round(width * target_height / height)

        result = {
            self.scale_resolution_out_name: (*resolution[:-2], scale_height, scale_width),
            self.crop_resolution_out_name: (*resolution[:-2], target_height, target_width),
        }
        return apply_resolution_tolerance(self, resolution, target_pixels, result, preserve_crop_aspect=isinstance(target, tuple))
