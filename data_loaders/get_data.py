import os
import torch
from torch.utils.data import DataLoader
from data_loaders.humanml_utils import get_inpainting_mask
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate


def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == 'humanact12':
        # used by baseline codepaths in this repo
        from data_loaders.humanml.scripts.motion_process import Text2MotionDatasetV2
        return Text2MotionDatasetV2
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "100style":
        from data_loaders.style.dataset import StyleMotionDataset
        return StyleMotionDataset
    elif name == "vc":
        # motion-only dataset for your exported npz files
        from data_loaders.vc.dataset_vc import VCXYZDataset
        return VCXYZDataset
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    # vc + others use the generic collate of (motion, cond)
    return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train',
                styles=None, motion_type_to_exclude=(), **kwargs):
    DATA = get_dataset_class(name)

    if name in ["humanml", "kit"]:
        # Text datasets (use their own internal dirs and options)
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, **kwargs)
    elif name == "100style":
        dataset = DATA(styles, split, motion_type_to_exclude=motion_type_to_exclude)
    elif name == "vc":
        # Motion-only VC dataset: defaults work with your repo layout
        data_root = kwargs.get('data_root', './data/humanml3d')
        split_dir = kwargs.get('split_dir', './dataset/vc')
        max_motion_length = kwargs.get('max_motion_length', 200)
        split_file = os.path.join(split_dir, f'{split}.txt')
        mean_std_dir = os.path.join(split_dir, 'meta', split)

        # VCXYZDataset returns (motion [22,3,T], {'y': {'age': scalar, 'mask': [1,1,T]}})
        dataset = DATA(
            root=data_root,
            split_file=split_file,
            njoints=22,
            nfeats=3,
            max_motion_length=max_motion_length,
            mean_std_dir=mean_std_dir,
        )
    else:
        # Generic fallback
        dataset = DATA(data_dir=f"dataset/{name}", split=split, num_frames=num_frames, styles=styles)

    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train',
                       styles=None, debug=False, motion_type_to_exclude=(), **kwargs):
    dataset = get_dataset(name, num_frames, split, hml_mode, styles, motion_type_to_exclude, **kwargs)
    collate = get_collate_fn(name, hml_mode)
    batch_size = 5 if debug else min(len(dataset), batch_size)
    drop_last = (name == "humanml")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=drop_last, collate_fn=collate, pin_memory=True
    )
    return loader


def get_prior_dataset_loader(batch_size, num_frames):
    from data_loaders.humanml.data.dataset import HumanML3D
    data = HumanML3D(split='train', num_frames=num_frames, mode='train')
    collate = get_collate_fn('humanml', 'train')
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True, pin_memory=True, collate_fn=collate
    )
    return loader


class InpaintingDataLoader(object):
    def __init__(self, data, inpainting_mask):
        self.data = data
        self.inpainting_mask = inpainting_mask

    def __iter__(self):
        for motion, cond in super().__getattribute__('data').__iter__():
            cond['y']['inpainting_mask'] = torch.tensor(
                get_inpainting_mask(self.inpainting_mask, motion.shape)
            ).to(motion.device)
            yield motion, cond

    def __getattribute__(self, name):
        return super().__getattribute__('data').__getattribute__(name)

    def __len__(self):
        return len(super().__getattribute__('data'))
