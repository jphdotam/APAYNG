import os
import cv2
import csv
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import onnxruntime as ort
import albumentations as A

import torch

from lib.inference import load_anatomy_stack_from_dicom_paths, upsample_pred_if_needed, \
    rescale_segmented_anatomy_stack_to_isotropy, volumes_from_volume_and_slice_width, \
    get_aortic_diameter_from_volumes_pytorch

DICOM_FOLDER = r"D:\Local\APAYN\dicoms"
ONNX_MODEL_PATH = r"E:\Dropbox\Exchange\80_0.80656.pt.onnx"
GROUND_TRUTH_CSV = r"D:\Local\APAYN\dicoms\sz.csv"
USE_PYTORCH_FOR_INFERENCE = True
PYTORCH_DEVICE = "cuda"
PIXEL_DIAMETER_MM = 0.64
FOV_HEIGHT_PX = 576
FOV_WIDTH_PX = 576

inference_transform = A.Compose([
    A.PadIfNeeded(min_height=FOV_HEIGHT_PX, min_width=FOV_WIDTH_PX, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.CenterCrop(height=FOV_HEIGHT_PX, width=FOV_WIDTH_PX)
])

sess = ort.InferenceSession(ONNX_MODEL_PATH)

df_reported = pd.read_csv(GROUND_TRUTH_CSV).dropna(subset=['LVEDV'])

for manufacturer in [d for d in os.listdir(DICOM_FOLDER) if os.path.isdir(os.path.join(DICOM_FOLDER, d))]:

    volumes_by_study_manu = []

    for i_study, study_id in enumerate(tqdm(os.listdir(os.path.join(DICOM_FOLDER, manufacturer)))):

        if study_id not in list(df_reported['accession_number']):
            continue

        dicom_files = sorted(glob(os.path.join(DICOM_FOLDER, manufacturer, study_id, "*.dcm")))

        anatomy_stack, _, slice_thickness = load_anatomy_stack_from_dicom_paths(dicom_files,
                                                                                inference_transform,
                                                                                rescale_to_pixel_size_mm=PIXEL_DIAMETER_MM,
                                                                                channels_first=True if USE_PYTORCH_FOR_INFERENCE else False)

        anatomy_stack_pred = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: anatomy_stack})[0]

        # Data to isotropic 3D model
        if USE_PYTORCH_FOR_INFERENCE:
            with torch.no_grad():
                anatomy_stack_pred = torch.tensor(anatomy_stack_pred).to(PYTORCH_DEVICE).float()

                # Some networks produce smaller predictions per slice, e.g. HRNet - upsample the slices if needed
                anatomy_stack_pred = upsample_pred_if_needed(anatomy_stack_pred, anatomy_stack)

                # Now make pixels isotropic
                anatomy_stack_pred = rescale_segmented_anatomy_stack_to_isotropy(anatomy_stack_pred, slice_thickness)

                # Move back to numpy for the argmax or we end up using 20 Gb of GPU RAM just for inference
                anatomy_3d = np.argmax(anatomy_stack_pred.cpu().numpy(), axis=0)

        else:
            raise NotImplementedError()

        volumes_ml, volumes_px = volumes_from_volume_and_slice_width(anatomy_3d, PIXEL_DIAMETER_MM)

        volumes_ml['study_id'] = study_id
        volumes_ml['manufacturer'] = manufacturer
        volumes_ml['aorta'] = get_aortic_diameter_from_volumes_pytorch(anatomy_3d)

        study_dict = {'study_id': study_id,
                      'manufacturer': manufacturer}
        for structure, vol in volumes_ml.items():
            study_dict[type] = vol
        volumes_by_study_manu.append(volumes_ml)

    with open(f'predictions_{manufacturer}.csv', 'w', encoding='utf8') as output_file:
        fc = csv.DictWriter(output_file, fieldnames=volumes_by_study_manu[0].keys())
        fc.writeheader()
        fc.writerows(volumes_by_study_manu)
