import PIL.Image
import PIL.ImageDraw
import json
import pydicom
import numpy as np
import skimage.transform


MERGE_AORTA = True

LABELS = {label: i for i, label in enumerate([
    'background',  # Background is LAST channel therefore
    'aorta',
    'la_cav',
    'lv_wall',
    'pa',
    'pleuraleffusion',
    'ra_cav',
    'rv_cav',
    'rv_wall',
    'lv_cav'])}


def load_dicom(dicom_path, rescale_to_pixel_size_mm, use_window_levels=False, normalize_by_sd=0.95):
    assert bool(use_window_levels) != bool(normalize_by_sd), f"Either window level normalisation or sd normalisation"

    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array
    slice_thickness = dcm.SpacingBetweenSlices

    if use_window_levels:
        raise NotImplementedError()

    if normalize_by_sd:
        img_min, img_max = np.percentile(img, [5, 95])
        img = img - img_min
        img = img / (img_max - img_min)
        img = np.clip(img, 0, 1)

    if rescale_to_pixel_size_mm:
        pixel_height, pixel_width = dcm.PixelSpacing
        height_ratio = pixel_height / rescale_to_pixel_size_mm
        width_ratio = pixel_width / rescale_to_pixel_size_mm
        img = skimage.transform.rescale(img, (height_ratio, width_ratio), anti_aliasing=True)
        return img, (height_ratio, width_ratio), slice_thickness
    else:
        return img, None, slice_thickness


def load_mask_from_json(json_path, rescale_factors):
    json_data = json.load(open(json_path))
    height, width, shape_data = json_data['imageHeight'], json_data['imageWidth'], json_data['shapes']

    if MERGE_AORTA:
        shape_data = merge_aorta_segments_in_shape_data(shape_data)

    mask = create_mask_from_shape_data(height, width, shape_data)

    if rescale_factors:
        mask = skimage.transform.rescale(mask, rescale_factors, multichannel=False, order=0, anti_aliasing=False, preserve_range=True)
        mask = mask.astype(np.uint8)

    return mask


def create_mask_from_shape_data(height, width, shape_data):
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in sorted(shape_data, key=lambda x: LABELS.get(x['label'], 0)):
        if shape['label'] == "dummy":
            continue
        assert shape['shape_type'] == "polygon"
        label_name = shape['label']
        if label_name not in LABELS:
            # print(f"Skipping label {label_name} as not in LABELS")
            continue
        i_label = LABELS[label_name]
        mask = np.maximum(mask, shape_to_mask(height, width, shape['points'], i_label))

    return mask


def shape_to_mask(height, width, shape, fill_value):
    """Thanks to https://github.com/wkentaro/labelme/blob/master/labelme/utils/shape.py"""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in shape]
    draw.polygon(xy=xy, outline=fill_value, fill=fill_value)
    mask = np.array(mask)
    return mask


def merge_aorta_segments_in_shape_data(shape_data):
    for shape in shape_data:
        if 'aorta' in shape['label']:
            shape['label'] = 'aorta'
    return shape_data




if __name__ == "__main__":
    dcm_path = "/Users/jameshoward/Data/APAYN/dicoms/SIEMENS/RYJ10650155/img0012--22.4602.dcm"
    json_path = "/Users/jameshoward/Data/APAYN/dicoms/SIEMENS/RYJ10650155/img0012--22.4602.dcm_T2_TRA_anatomy.json"
    i, rescale_ratios, _slice_thickness = load_dicom(dcm_path, rescale_to_pixel_size_mm=0.64)
    m = load_mask_from_json(json_path, rescale_ratios)

    import matplotlib.pyplot as plt
    plt.imshow(i, cmap='gray')
    plt.title('image')
    plt.show()

    plt.imshow(m)
    plt.title('mask')
    plt.show()

