import math
import unittest
from types import SimpleNamespace

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.MpxAspectCalculation import MpxAspectCalculation
from modules.dataLoader.ResolutionMetadata import ResolutionAwareMaskRemove, ResolutionMetadata
from modules.dataLoader.StableDiffusionFineTuneVaeDataLoader import StableDiffusionFineTuneVaeDataLoader
from modules.ui.BaseTrainUIView import BaseTrainUIView
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

from PySide6.QtWidgets import QApplication, QWidget


class ImageSource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, resolutions, target='512', override=None):
        super().__init__()
        self.resolutions = resolutions
        self.target = target
        self.override = override

    def length(self):
        return len(self.resolutions)

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ['image_path', 'image', 'mask', 'custom_conditioning_image', 'settings', 'concept']

    def get_item(self, variation, index, requested_name=None):
        height, width = self.resolutions[index][-2:]
        image = torch.linspace(0, 1, height * width).reshape(1, height, width).repeat(3, 1, 1)
        if len(self.resolutions[index]) == 3:
            image = image.unsqueeze(1).repeat(1, self.resolutions[index][0], 1, 1)
        return {
            'image_path': f'/images/{index}.png',
            'image': image,
            'mask': image[:1].clone(),
            'custom_conditioning_image': image.clone(),
            'settings': {'target_resolution': self.target, 'target_frames': '1'},
            'concept': {'image': {
                'enable_resolution_override': self.override is not None,
                'resolution_override': self.override or self.target,
                'enable_crop_jitter': True,
            }},
        }


class MpxScalingTest(unittest.TestCase):
    def make_pipeline(self, source, bucketing=False, mpx=True, quantization=8, crop=False):
        config = SimpleNamespace(aspect_ratio_bucketing=bucketing, scale_to_mpx_equivalent=mpx)
        modules = DataLoaderText2ImageMixin._aspect_bucketing_in(None, config, quantization)
        outputs = ['scale_resolution', 'crop_resolution']
        if crop:
            modules.append(ScaleCropImage(
                names=['image', 'mask', 'custom_conditioning_image'],
                scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution',
                enable_crop_jitter_in_name='concept.image.enable_crop_jitter', crop_offset_out_name='crop_offset',
            ))
            outputs.extend(['image', 'mask', 'custom_conditioning_image'])
        state = PipelineState(1)
        self.addCleanup(state.executor.shutdown)
        pipeline = LoadingPipeline(
            torch.device('cpu'), [source, *modules, OutputPipelineModule(outputs)], 1, 42, state,
        )
        pipeline.start_next_epoch()
        return pipeline, modules[1]

    def test_portrait_landscape_and_model_multiples(self):
        resolutions = [(1000, 1379), (1379, 1000), (1024, 1024), (1, 100000)]
        for quantization in (8, 16, 64, 128):
            with self.subTest(quantization=quantization):
                pipeline, _ = self.make_pipeline(ImageSource(resolutions, '1024'), quantization=quantization)
                for original, item in zip(resolutions, pipeline, strict=True):
                    crop = item['crop_resolution']
                    scale = item['scale_resolution']
                    for dimension, source_dimension, scaled_dimension in zip(crop, original, scale, strict=True):
                        ideal = source_dimension * 1024 / math.sqrt(math.prod(original))
                        self.assertGreaterEqual(dimension, quantization)
                        self.assertEqual(dimension % quantization, 0)
                        self.assertLessEqual(abs(dimension - max(quantization, ideal)), quantization / 2)
                        self.assertGreaterEqual(scaled_dimension, dimension)

    def test_arbitrary_aspect_does_not_select_a_predefined_bucket(self):
        source = ImageSource([(1000, 1379)], '1024')
        pipeline, _ = self.make_pipeline(source, quantization=64)
        crop = next(pipeline)['crop_resolution']
        self.assertEqual(crop, (896, 1216))
        bucket_pipeline, bucketing = self.make_pipeline(ImageSource([(1000, 1379)], '1024'), bucketing=True, quantization=64)
        self.assertNotIn(crop, bucketing.bucket_resolutions[1024])
        self.assertNotEqual(crop, next(bucket_pipeline)['crop_resolution'])
        self.assertLess(abs(math.prod(crop) / 1024**2 - 1), 0.05)

    def test_off_keeps_square_crops_and_bucketing_takes_precedence(self):
        pipeline, calculation = self.make_pipeline(ImageSource([(600, 800)]), mpx=False)
        self.assertIs(type(calculation), SingleAspectCalculation)
        self.assertEqual(next(pipeline)['crop_resolution'], (512, 512))
        _, calculation = self.make_pipeline(ImageSource([(600, 800)]), bucketing=True, mpx=False)
        self.assertIsInstance(calculation, AspectBucketing)

    def test_tolerance_boundaries_in_both_modes(self):
        for bucketing in (False, True):
            for height in (899, 900, 1000, 1100, 1101):
                with self.subTest(bucketing=bucketing, height=height):
                    pipeline, calculation = self.make_pipeline(
                        ImageSource([(height, 1000)], target='1000'), bucketing=bucketing, mpx=not bucketing,
                    )
                    item = next(pipeline)
                    if 900 <= height <= 1100:
                        self.assertEqual(item['scale_resolution'], (height, 1000))
                        self.assertLessEqual(item['crop_resolution'][0], height)
                        self.assertLessEqual(item['crop_resolution'][1], 1000)
                    elif not bucketing:
                        self.assertNotEqual(item['scale_resolution'], (height, 1000))
                    else:
                        # The existing bucket resize can itself be a no-op outside the tolerance.
                        self.assertEqual(item, AspectBucketing.get_item(calculation, 0, 0))
                    self.assertTrue(all(dimension % 8 == 0 for dimension in item['crop_resolution']))

    def test_tolerance_uses_override_and_keeps_native_image_pixels(self):
        for bucketing in (False, True):
            pipeline, _ = self.make_pipeline(
                ImageSource([(500, 512)], target='1024', override='512'), bucketing=bucketing, crop=True,
            )
            item = next(pipeline)
            self.assertEqual(item['scale_resolution'], (500, 512))
            original = torch.linspace(0, 1, 500 * 512).reshape(1, 500, 512).repeat(3, 1, 1)
            # Every output pixel is an original pixel; cropping introduces no interpolation.
            crop_height, crop_width = item['crop_resolution']
            matches = [
                torch.equal(item['image'], original[:, y:y + crop_height, x:x + crop_width])
                for y in range(500 - crop_height + 1)
                for x in range(512 - crop_width + 1)
            ]
            self.assertTrue(any(matches))

    def test_tolerance_preserves_frames_and_fixed_crop_rules(self):
        for bucketing in (False, True):
            pipeline, _ = self.make_pipeline(
                ImageSource([(3, 52, 80)], target='64', override='80x48'), bucketing=bucketing, crop=True,
            )
            item = next(pipeline)
            self.assertEqual(item['scale_resolution'], (3, 52, 80))
            self.assertEqual(item['crop_resolution'], (3, 48, 80))
            self.assertEqual(tuple(item['image'].shape), (3, 3, 48, 80))

    def test_resolution_overrides_multiple_targets_and_fixed_dimensions(self):
        for override, expected in [('256', (224, 296)), ('768x512', (512, 768))]:
            with self.subTest(override=override):
                pipeline, _ = self.make_pipeline(ImageSource([(600, 800)], override=override))
                self.assertEqual(next(pipeline)['crop_resolution'], expected)
        pipeline, _ = self.make_pipeline(ImageSource([(600, 800)] * 32, target='256,512'))
        self.assertEqual({item['crop_resolution'] for item in pipeline}, {(224, 296), (440, 592)})

    def test_crop_keeps_image_mask_and_conditioning_image_aligned(self):
        pipeline, _ = self.make_pipeline(ImageSource([(600, 800)]), crop=True)
        item = next(pipeline)
        self.assertEqual(tuple(item['image'].shape[-2:]), (440, 592))
        torch.testing.assert_close(item['mask'], item['image'][:1])
        torch.testing.assert_close(item['custom_conditioning_image'], item['image'])

    def test_frame_dimension_is_preserved(self):
        pipeline, _ = self.make_pipeline(ImageSource([(5, 60, 80)], target='64'), crop=True)
        item = next(pipeline)
        self.assertEqual(item['crop_resolution'], (5, 56, 72))
        self.assertEqual(tuple(item['image'].shape), (3, 5, 56, 72))

    def test_vae_training_uses_the_same_mpx_calculation(self):
        config = SimpleNamespace(aspect_ratio_bucketing=False, scale_to_mpx_equivalent=True)
        modules = StableDiffusionFineTuneVaeDataLoader._StableDiffusionFineTuneVaeDataLoader__aspect_bucketing_in(None, config)
        self.assertIsInstance(modules[1], MpxAspectCalculation)
        self.assertEqual(modules[1].quantization, 8)

    def test_default_and_saved_config(self):
        legacy = TrainConfig.default_values().to_dict()
        legacy.pop('scale_to_mpx_equivalent')
        legacy['aspect_ratio_bucketing'] = False
        config = TrainConfig.default_values().from_dict(legacy)
        self.assertTrue(config.scale_to_mpx_equivalent)
        config.scale_to_mpx_equivalent = False
        restored = TrainConfig.default_values().from_dict(config.to_dict())
        self.assertFalse(restored.scale_to_mpx_equivalent)


class InpaintingResolutionSource(PipelineModule, RandomAccessPipelineModule):
    def length(self):
        return 2

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ['crop_resolution', 'latent_mask', 'latent_conditioning_image']

    def get_item(self, variation, index, requested_name=None):
        height, width = 64 + 8 * variation, 96 + 8 * index
        return {
            'crop_resolution': (height, width),
            'latent_mask': torch.zeros(1, height // 8, width // 8),
            'latent_conditioning_image': torch.ones(4, height // 8, width // 8),
        }


class InpaintingScalingTest(unittest.TestCase):
    def test_blank_conditioning_covers_each_actual_shape_in_each_epoch(self):
        encoded_shapes = []

        def encode(image):
            encoded_shapes.append(tuple(image.shape[-2:]))
            latent = torch.zeros(1, 4, image.shape[-2] // 8, image.shape[-1] // 8)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

        remove = RandomLatentMaskRemove(
            latent_mask_name='latent_mask', latent_conditioning_image_name='latent_conditioning_image',
            possible_resolutions_in_name='possible_resolutions', replace_probability=1,
            vae=SimpleNamespace(encode=encode), dtype=torch.float32,
        )
        state = PipelineState(1)
        self.addCleanup(state.executor.shutdown)
        restored_epochs = []
        pipeline = LoadingPipeline(torch.device('cpu'), [
            InpaintingResolutionSource(), ResolutionMetadata('crop_resolution', 'possible_resolutions'),
            ResolutionAwareMaskRemove(remove, after_cache_fun=lambda: restored_epochs.append(True)),
            OutputPipelineModule(['latent_mask', 'latent_conditioning_image']),
        ], 1, 42, state)
        for epoch in range(2):
            pipeline.start_next_epoch()
            for index, item in enumerate(pipeline):
                self.assertTrue(torch.all(item['latent_mask'] == 1))
                self.assertTrue(torch.all(item['latent_conditioning_image'] == 0))
                self.assertEqual(tuple(item['latent_conditioning_image'].shape), (4, 8 + epoch, 12 + index))
        self.assertEqual(set(encoded_shapes), {(64, 96), (64, 104), (72, 96), (72, 104)})
        self.assertEqual(len(restored_epochs), 2)


class DataTabScalingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_checkbox_position_default_lock_and_preset_changes(self):
        config = SimpleNamespace(
            aspect_ratio_bucketing=True, scale_to_mpx_equivalent=False,
            latent_caching=True, clear_cache_before_training=True,
        )
        state = PySide6UIState(config)
        frame = QWidget()
        self.addCleanup(frame.deleteLater)
        view = SimpleNamespace(components=pyside6_components)
        BaseTrainUIView.build_data_tab_content(view, frame, None, state)
        layout = frame.layout()
        checkbox = layout.itemAtPosition(1, 1).widget()
        self.assertEqual(layout.itemAtPosition(1, 0).widget().text(), 'Scale to MPX equivalent')
        self.assertTrue(checkbox.isChecked())
        self.assertFalse(checkbox.isEnabled())
        self.assertTrue(config.scale_to_mpx_equivalent)
        layout.itemAtPosition(0, 1).widget().click()
        self.assertTrue(checkbox.isEnabled())
        checkbox.click()
        self.assertFalse(config.scale_to_mpx_equivalent)
        layout.itemAtPosition(0, 1).widget().click()
        self.assertTrue(config.scale_to_mpx_equivalent)
        self.assertTrue(checkbox.isChecked())
        self.assertFalse(checkbox.isEnabled())
        config.scale_to_mpx_equivalent = False
        state.update(config)
        self.assertTrue(config.scale_to_mpx_equivalent)
        self.assertTrue(checkbox.isChecked())


if __name__ == '__main__':
    unittest.main()
