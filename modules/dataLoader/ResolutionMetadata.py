from collections.abc import Callable

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class ResolutionMetadata(PipelineModule, RandomAccessPipelineModule):
    """Expose actual post-cache dimensions for blank inpainting conditioning images."""

    def __init__(self, resolution_in_name: str, possible_resolutions_out_name: str):
        super().__init__()
        self.resolution_in_name = resolution_in_name
        self.possible_resolutions_out_name = possible_resolutions_out_name

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.possible_resolutions_out_name]

    def get_meta(self, variation: int, name: str):
        if name == self.possible_resolutions_out_name:
            return {
                self._get_previous_item(variation, self.resolution_in_name, index)
                for index in range(self.length())
            }
        return None

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {self.possible_resolutions_out_name: self.get_meta(variation, self.possible_resolutions_out_name)}


class ResolutionAwareMaskRemove(PipelineModule, SingleVariationRandomAccessPipelineModule):
    """Refresh blank conditioning images when per-image dimensions change between epochs."""

    def __init__(self, mask_remove: RandomLatentMaskRemove, after_cache_fun: Callable[[], None] | None = None):
        super().__init__()
        self.mask_remove = mask_remove
        self.after_cache_fun = after_cache_fun

    def init(self, pipeline, base_seed, module_index, state):
        super().init(pipeline, base_seed, module_index, state)
        self.mask_remove.init(pipeline, base_seed, module_index, state)

    def length(self) -> int:
        return self.mask_remove.length()

    def get_inputs(self) -> list[str]:
        return self.mask_remove.get_inputs()

    def get_outputs(self) -> list[str]:
        return self.mask_remove.get_outputs()

    def start(self, variation: int):
        self.mask_remove.clear_item_cache()
        self.mask_remove.full_mask_cache.clear()
        self.mask_remove.blank_conditioning_image_cache.clear()
        self.mask_remove.start(variation)
        if self.after_cache_fun is not None:
            self.after_cache_fun()

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return self.mask_remove.get_item(self.current_variation, index, requested_name)
