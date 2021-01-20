import os
import math
import hashlib
import albumentations
from glob import glob

import torch
from torch.utils.data import Dataset

from lib.data import load_dicom, load_mask_from_json


class APAYNSliceDataset(Dataset):
    def __init__(self, cfg:dict, train_or_test:str, transforms: albumentations.Compose, fold=1):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.transforms = transforms
        self.fold = fold

        self.n_folds = cfg['data']['n_folds']
        self.samples = self.load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.get_sample(idx)

    def get_sample(self, idx):
        json_path = self.samples[idx]
        dicom_path = json_path.split('.dcm')[0] + '.dcm'

        img, rescale_ratio, _slice_thickness = load_dicom(dicom_path)
        mask = load_mask_from_json(json_path, rescale_ratio)

        trans = self.transforms(image=img, mask=mask)
        img = trans['image']
        mask = trans['mask']

        x = torch.tensor(img).unsqueeze(0).float()
        y = torch.tensor(mask).long()  # Mask is uint8 so should autocast as long

        return x, y, dicom_path

    def load_samples(self):
        samples = []
        dicoms_path = self.cfg['paths']['dicoms']
        json_filepaths = glob(os.path.join(dicoms_path, "**/*.json"), recursive=True)
        for json_filepath in json_filepaths:
            validation_fold = self.get_validation_fold_for_file(json_filepath)
            if (self.train_or_test == 'test') == (validation_fold == self.fold):
                samples.append(json_filepath)
        print(f"{self.train_or_test.upper():<5}: {len(samples)} samples")
        return samples

    def get_validation_fold_for_file(self, file_path):
        """e.g. RYJ9002753/img0006--14.9978.dcm_T2_TRA_anatomy.json -> hash of RYJ9002753 -> 1 to n_folds"""
        randnum = int(hashlib.md5(str.encode(file_path)).hexdigest(), 16) / 16 ** 32
        validation_fold = math.floor(randnum * self.n_folds) + 1
        return validation_fold

