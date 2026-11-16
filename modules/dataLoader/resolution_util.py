def get_target_resolution(module, variation: int, index: int) -> int | tuple[int, int]:
    targets = module._get_previous_item(variation, module.target_resolutions_in_name, index)
    if module.enable_target_resolutions_override_in_name is not None and module._get_previous_item(
        variation, module.enable_target_resolutions_override_in_name, index,
    ):
        targets = module._get_previous_item(variation, module.target_resolutions_override_in_name, index)
    if isinstance(targets, str) and 'x' in targets and ',' not in targets:
        width, height = (int(value.strip()) for value in targets.split('x'))
        return height, width
    targets = [int(value.strip()) for value in str(targets).split(',')]
    return module._get_rand(variation, index).choice(targets)


def apply_resolution_tolerance(module, resolution, target_pixels, result, preserve_crop_aspect=False):
    height, width = resolution[-2:]
    if not 9 * target_pixels <= 10 * height * width <= 11 * target_pixels:
        return result
    quantization = module.quantization
    if min(height, width) < quantization:
        # A dimension smaller than the required multiple must still be enlarged.
        return result
    if preserve_crop_aspect:
        crop_height, crop_width = result[module.crop_resolution_out_name][-2:]
        fit = min(1, height / crop_height, width / crop_width)
        crop_height = min(height, round(crop_height * fit))
        crop_width = min(width, round(crop_width * fit))
    else:
        crop_height, crop_width = height, width
    result[module.scale_resolution_out_name] = resolution
    result[module.crop_resolution_out_name] = (
        *resolution[:-2],
        max(quantization, crop_height // quantization * quantization),
        max(quantization, crop_width // quantization * quantization),
    )
    return result
