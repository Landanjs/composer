# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core CopyPaste classes and functions."""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['CopyPaste', 'copypaste_batch']


def copypaste_batch(images, masks, configs):
    """
    copy-paste is a data augmentation method. Two images are randomly chosen (with
    replacement) from a given batch of images and their corresponding masks, i.e.,
    the source and the target target. A number of instances from the source are
    selected to be copied into the target. The number of copied instanes is always
    less than a number determined by the minimum of ``max_copied_instance`` and
    total number of instances in the source. A set of instances are then randomly
    chosen (without replacement) from the source. Each instance goes through a set
    of transformation and jittering, e.g., horizontal flipping, rescaling, and
    cropping. The resulting "jittered" instance is then checked for its area and
    transformed instances with an area smaller than the configured threshold.
    The resulting jittered instances are then pasted on a random position on
    the target. Same procedure is also applied on the masks of the source and
    the target. If a copied instances mask collides with an exisiting mask on
    the target, the copied instance's mask overlays the original mask on the
    target.

    Args:
        input (torch.Tensor): input tensor of shape ``(N, C, H, W)``.
        masks (torch.Tensor): target tensor of shape ``(N, H, W)``.
        configs (dict): dictionary containing the configurable hyperparameters.

    Returns:
        out_images (torch.Tensor): batch of images after copypaste has been
            applied.
        out_masks (torch.Tensor): batch of corresponding masks after applying
        copypaste to them

    Example:
        .. testcode::

            import torch
            from composer.functional import copypaste_batch

            N, C, H, W = 2, 3, 4, 5
            num_classes = 10
            configs = {
                "p": 1.0,
                "max_copied_instances": None,
                "area_threshold": 100,
                "padding_factor": 0.5,
                "jitter_scale": (0.01, 0.99),
                "jitter_ratio": (1.0, 1.0),
                "p_flip": 1.0,
                "bg_color": 0
            }
            X = torch.randn(N, C, H, W)
            y = torch.randint(num_classes, size=(N, H, W))
            out_images, out_masks = cutmix_batch(X, y, configs)
    """
    batch_idx = 0
    out_images = torch.zeros_like(images)
    out_masks = torch.zeros_like(masks)

    assert images.size(dim=0) == masks.size(dim=0), "Number of images and masks in the batch do not match!"
    batch_size = images.size(dim=0)

    # Only samples with instances can be source images
    src_indices = [i for i in range(batch_size) if (torch.unique(masks[i]) != configs['bg_color']).sum()]
    src_indices = np.array(src_indices)

    # Exit if there are no samples with instances
    if len(src_indices) == 0:
        return images, masks

    # Iterate through all samples and maybe apply copy-paste
    rand_samples = np.random.rand(batch_size)
    for batch_idx, sample in enumerate(rand_samples):
        target_img = images[batch_idx]
        target_mask = masks[batch_idx]

        # Array of available source samples based on the current batch
        current_src_indices = src_indices[src_indices != batch_idx]
        if sample < configs["p"] and len(current_src_indices) > 0:
            # Sample the source image to use, excluding the current batch
            src_idx = np.random.choice(current_src_indices)

            # Count the number of instances in the mask, ignoring the background class
            instance_ids = torch.unique(masks[src_idx])
            instance_ids = instance_ids[instance_ids != configs['bg_color']]  # Remove background id
            num_instances = len(instance_ids)

            max_copied_instances = num_instances
            if configs["max_copied_instances"] is not None:
                max_copied_instances = min(max_copied_instances, configs["max_copied_instances"])

            # Sample how many instances to copy-paste
            num_copied_instances = random.randint(1, max_copied_instances)

            # Sample `num_copied_instances` `instance_ids` to copy-paste (without replacement)
            rand_indices = torch.randperm(num_instances)[:num_copied_instances]
            src_instance_ids = instance_ids[rand_indices]

            # Copy-paste each instance onto the target image and mask
            for src_instance_id in src_instance_ids:
                target_img, target_mask = _copypaste_instance(images[src_idx], masks[src_idx], target_img, target_mask,
                                                              src_instance_id, configs)
        out_images[batch_idx] = target_img
        out_masks[batch_idx] = target_mask

    return out_images, out_masks


class CopyPaste(Algorithm):
    """
    Randomly pastes objects onto an image.

    Args:
        p (float, optional): Probability of applying copy-paste augmentation on a
            pair of randomly chosen source and target samples. Default: ``0.5``
        max_copied_instances (int | None, optional): Maximum number of instances
            to be copied from a randomly chosen source sample into another
            randomly chosen target sample. If this value is greater than the total
            number of instances in the source sample, it is overridden by the
            total number of instances in the source sample. If it is set to
            ``None``, the total number of instances in the source sample is set to
            be the limit. Default: ``None``.
        min_instance_area (int, optional): Minimum area (in pixels) of an augmented
            instance to be considered a valid instance. Augmented instances with
            an area smaller than this threshold are removed from the sample.
            Default:``25``.
        max_instance_area (float, optional): Something Something. Default:``0.5``.
        padding_factor (float, optional): The source sample is padded by this
            ratio before applying large scale jittering to it. Default: ``0.5``.
        jitter_scale_min (float, optional): Determines the scale used
            in the large scale jittering of the source instance. Specifies the
            lower and upper bounds for the random area of the crop, before
            resizing. The scale is defined with respect to the area of the
            original image. Default: ``(0.01, 0.99)``.
        jitter_scale_max (float, optional): Determines the scale used
            in the large scale jittering of the source instance. Specifies the
            lower and upper bounds for the random area of the crop, before
            resizing. The scale is defined with respect to the area of the
            original image. Default: ``(0.01, 0.99)``.
        jitter_ratio (Tuple[float, float], optional): Determines the ratio used in
            the large scale jittering of the source instance. Lower and upper
            bounds for the random aspect ratio of the crop, before resizing.
            Default: ``(1.0, 1.0)``.
        p_flip (float, optional): Probability of applying horizontal flipping
            during large scale jittering of the source instance. Default: ``0.9``.
        bg_color (int, optional): Class label (pixel value) of the background
            class. Default: ``-1``.
        input_key (str | int | Tuple[Callable, Callable] | Any, optional): A key
            that indexes to the input from the batch. Can also be a pair of get
            and set functions, where the getter is assumed to be first in the
            pair.  The default is 0, which corresponds to any sequence, where the
            first element is the input. Default: ``0``.
        target_key (str | int | Tuple[Callable, Callable] | Any, optional): A key
            that indexes to the target from the batch. Can also be a pair of get
            and set functions, where the getter is assumed to be first in the
            pair. The default is 1, which corresponds to any sequence, where the
            second element is the target. Default: ``1``.

    Example:
        .. testcode::

            from composer.algorithms import CopyPaste
            algorithm = CopyPaste()
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )

    """

    def __init__(
        self,
        p=0.5,
        max_copied_instances=None,
        min_instance_area=0.01,
        max_instance_area=0.5,
        padding_factor=0.5,
        jitter_scale_min=0.01,
        jitter_scale_max=0.99,
        jitter_ratio=(1.0, 1.0),
        p_flip=0.9,
        bg_color=-1,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
        self.input_key = input_key
        self.target_key = target_key
        self.configs = {
            "p": p,
            "max_copied_instances": max_copied_instances,
            "min_instance_area": min_instance_area,
            'max_instance_area': max_instance_area,
            "padding_factor": padding_factor,
            "jitter_scale": (jitter_scale_min, jitter_scale_max),
            "jitter_ratio": jitter_ratio,
            "p_flip": p_flip,
            "bg_color": bg_color
        }

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        images = state.batch_get_item(key=self.input_key)
        masks = state.batch_get_item(key=self.target_key)

        out_images, out_masks = copypaste_batch(images, masks, self.configs)

        state.batch_set_item(key=self.input_key, value=out_images)
        state.batch_set_item(key=self.target_key, value=out_masks)


def _copypaste_instance(src_image, src_mask, trg_image, trg_mask, src_instance_id, configs):
    """Applies copy-paste augmentation on a set of source and target samples. The
    instance identified by ``src_instance_id`` is selected from the source sample
    to be copied to the target sample.

    Args:
        src_image (torch.Tensor): Source image of shape ``(C, H, W)``,
            C is the number of channels.
        src_mask (torch.Tensor): Source mask of shape ``(H, W)``,
        trg_image (torch.Tensor): Target image of shape ``(C, H, W)``,
            C is the number of channels.
        trg_mask (torch.Tensor): Target mask of shape ``(H, W)``,
        src_instance_id (int): Class ID of the randmoly chosen instance to be
            copied from the source sample into the target sample.
        configs (dict): Configurable hyperparameters.

    Returns:
        trg_image (torch.Tensor): Augmented target image of shape ``(C, H, W)``,
            C is the number of channels.
        trg_mask (torch.Tensor): Augmented target mask of shape ``(H, W)``,
    """
    zero_tensor = torch.zeros(1, dtype=src_image.dtype, device=src_image.device)
    bg_color = configs["bg_color"]

    # Extract the instance from the mask and the image
    src_instance_mask = torch.where(src_mask == src_instance_id, src_instance_id, bg_color)
    src_instance = torch.where(src_mask == src_instance_id, src_image, zero_tensor)

    [src_instance, src_instance_mask] = _jitter_instance(src_instance, src_instance_mask.unsqueeze(0), configs)
    src_instance_mask = src_instance_mask.squeeze(0)

    # Only paste the instance if it meets the pixel area requirements
    instance_area = (src_instance_mask != configs['bg_color']).sum() / (src_instance_mask.shape[0] *
                                                                        src_instance_mask.shape[1])
    if instance_area > configs['min_instance_area'] and instance_area < configs['max_instance_area']:
        trg_image = torch.where(src_instance_mask == src_instance_id, src_instance, trg_image)
        trg_mask = torch.where(src_instance_mask == src_instance_id, src_instance_mask, trg_mask)

    return trg_image, trg_mask


def _jitter_instance(img, mask, configs, n_retry=10):
    """Applies transformations on a tuple of image and mask.

    Args:
        arrs (sequence): Sequence containing the image and mask tensors. Element 0
            always contains the image and element 1 contains the mask.
        configs (dict): Configurable hyperparameters.


    Returns:
        out (sequence): Sequence containing the jittered (transformed) image and mask
            tensors. Element 0 always contains the image and element 1 contains the
            mask.
    """
    jitter_img, jitter_mask = img, mask
    for _ in range(n_retry):
        angle, translate, scale, shear = T.RandomAffine.get_params(degrees=(0, 0),
                                                                   translate=(configs['padding_factor'],
                                                                              configs['padding_factor']),
                                                                   scale_ranges=configs['jitter_scale'],
                                                                   img_size=img.shape[1:],
                                                                   shears=None)
        # Attempt to apply the augmentation
        jitter_mask = T.functional.affine(mask,
                                          angle=angle,
                                          translate=translate,
                                          scale=scale,
                                          shear=shear,
                                          interpolation=T.functional.InterpolationMode.NEAREST,
                                          fill=configs['bg_color'])

        # Check if the jittered instance meets the instance area restrictions
        instance_area = (jitter_mask != configs['bg_color']).sum() / (jitter_mask.shape[0] * jitter_mask.shape[1])
        if instance_area > configs["min_instance_area"] and instance_area < configs["max_instance_area"]:
            jitter_img = T.functional.affine(img,
                                             angle=angle,
                                             translate=translate,
                                             scale=scale,
                                             shear=shear,
                                             interpolation=T.functional.InterpolationMode.BILINEAR,
                                             fill=0)
            break
        else:
            # Reset jitter_img and jitter_mask if the jittered instance does not meet area restrictions
            jitter_img, jitter_mask = img, mask

    is_flip = np.random.rand(1)
    if is_flip < configs['p_flip']:
        T.functional.hflip(jitter_img)
        T.functional.hflip(jitter_mask)

    return jitter_img, jitter_mask
