import math
from typing import Union, Optional

import numpy as np
import albumentations as A
from scipy.ndimage import label

import torch
import torch.nn.functional as F

from lib.data import load_dicom, LABELS


def upsample_pred_if_needed(y_pred: torch.Tensor,
                            x: Union[torch.Tensor, np.ndarray],
                            mode: str = 'bilinear'):
    """Note this is different from the function used at training time, which uses
    the labels as a basis for upsampling (y_true) rather than image (x)
    Also support numpy array as input for x if needed"""
    ph, pw = y_pred.shape[2:]
    h, w = x.shape[2:]
    if ph != h or pw != w:
        y_pred = F.upsample(input=y_pred, size=(h, w), mode=mode)
    return y_pred


def load_anatomy_stack_from_dicom_paths(list_of_dicom_paths: list,
                                        inference_transform: A.Compose,
                                        rescale_to_pixel_size_mm: float,
                                        channels_first: bool = True):
    stack_of_images = []
    pixel_ratios = []
    slice_thicknesses = []
    for dicom_path in list_of_dicom_paths:
        img, pixel_ratio, slice_thickness = load_dicom(dicom_path, rescale_to_pixel_size_mm=rescale_to_pixel_size_mm)
        img = inference_transform(image=img)['image']
        img = np.expand_dims(img, axis=(0 if channels_first else -1))  # Add colour channel
        stack_of_images.append(img)
        pixel_ratios.append(pixel_ratio)
        slice_thicknesses.append(slice_thickness)
    assert all(x == pixel_ratios[0] for x in pixel_ratios), f"Pixel size seems to vary across slices: {list_of_dicom_paths}"
    assert all(x == slice_thicknesses[0] for x in slice_thicknesses), f"Slice thickness seems to vary across slices: {list_of_dicom_paths}"
    stack_of_images = np.stack(stack_of_images).astype(np.float32)
    return stack_of_images, pixel_ratios[0], slice_thicknesses[0]


def rescale_segmented_anatomy_stack_to_isotropy(segmented_anatomy_stack: torch.Tensor,
                                                slice_thickness_mm: float,
                                                pixel_diameter_mm: float):
    n_slices, n_classes, y_in, x_in = segmented_anatomy_stack.shape

    with torch.no_grad():

        # We need to go from n_slices * C * H * W -> <BS=1> * C * n_slices * H * W
        segmented_stack_scaled = segmented_anatomy_stack.transpose(0, 1).unsqueeze(0)

        #  If we have a slice-thickness of pixel_diameter we are already isotropic
        if slice_thickness_mm != pixel_diameter_mm:
            z_out = round(slice_thickness_mm / pixel_diameter_mm * n_slices)

            segmented_stack_scaled = F.interpolate(segmented_stack_scaled,
                                                   size=(z_out, y_in, x_in),
                                                   mode='trilinear',
                                                   align_corners=False)

        # Remove the dummy batch size dimension
        segmented_stack_scaled = segmented_stack_scaled.squeeze(0)

        return segmented_stack_scaled


def volumes_from_volume_and_slice_width(volume: np.ndarray,
                                        pixel_diameter_mm: float):
    volumes_ml = {}
    volumes_px = {}

    pixel_volume_ml = (pixel_diameter_mm / 10)**3

    for label_name, label_id in LABELS.items():
        px = np.sum(volume == label_id)
        volumes_px[label_name] = px
        volumes_ml[label_name] = px * pixel_volume_ml
    return volumes_ml, volumes_px


def get_aortic_diameter_from_volumes_pytorch(anatomy_3d: np.ndarray,
                                             pixel_diameter_mm: float):
    """Takes an isotropic volume (after argmax) and a pixel_diameter
    The scale factor is a tuple of (slope, intercept)"""

    pa_boolean_stack = anatomy_3d == LABELS['pa']
    pa_z = np.any(pa_boolean_stack, axis=(1, 2))
    try:
        mid_pa_z = int(np.median(np.argwhere(pa_z)))

        aorta_boolean_slice = anatomy_3d[mid_pa_z] == LABELS['aorta']

        # https://stackoverflow.com/questions/9440921/identify-contiguous-regions-in-2d-numpy-array
        labels, num_labels = label(aorta_boolean_slice)
        aorta_indices = [(labels == i).nonzero() for i in range(1, num_labels + 1)]
        if num_labels >= 2:  # Should be 2, unless found non-continuous aorta
            aorta_pred_native = math.sqrt(len(aorta_indices[0][0]))
            aorta_pred_mm = aorta_pred_native * pixel_diameter_mm
            return aorta_pred_mm
        else:  # Didn't find aorta
            return False
    except ValueError:
        return False


def remove_nan(y_true, y_pred):
    """Removed NaNs from pairs of prediction/trues"""
    out_true, out_pred = [], []
    for t, p in zip(y_true, y_pred):
        if np.isnan(t) or np.isnan(p):
            continue
        else:
            out_true.append(t)
            out_pred.append(p)
    return out_true, out_pred
