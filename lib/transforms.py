import cv2
import albumentations as A


def load_transforms(cfg):

    def _get_transforms(trans_cfg):
        transforms = []

        if trans_cfg.get('rotate', False):
            rotation = trans_cfg.get('rotate')
            transforms.append(A.Rotate(limit=rotation))

        if trans_cfg.get('randomresizedcrop', False):
            resize_height, resized_width = trans_cfg.get('randomresizedcrop')
            transforms.append(A.RandomResizedCrop(resize_height, resized_width,
                                                  scale=(0.5, 1.0),
                                                  ratio=(0.8, 1.2),
                                                  interpolation=cv2.INTER_CUBIC,
                                                  p=1))

        if trans_cfg.get("grid_dropout", False):
            chance, apply_to_mask = trans_cfg.get("grid_dropout")
            if not apply_to_mask:
                apply_to_mask = None  # None is correct parameter rather than False
            transforms.append(A.GridDropout(ratio=chance,
                                            unit_size_min=10,
                                            unit_size_max=50,
                                            random_offset=True,
                                            fill_value=0,
                                            mask_fill_value=apply_to_mask))

        if trans_cfg.get('centrecrop', False):
            crop_height, crop_width = trans_cfg['centrecrop']
            transforms.append(A.PadIfNeeded(crop_height, crop_width))
            transforms.append(A.CenterCrop(crop_height, crop_width))

        return A.Compose(transforms)

    train_transforms = _get_transforms(cfg['transforms']['train'])
    test_transforms = _get_transforms(cfg['transforms']['test'])

    return train_transforms, test_transforms
