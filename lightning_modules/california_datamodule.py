import os
from glob import glob
from typing import Any, Dict, List, Optional

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import skimage.util as util
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import config_to_object


class CaliforniaDataModule(pl.LightningDataModule):
    def __init__(self, **hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()

        def filter_by_notes(x):
            return (
                len(np.intersect1d(str(x).split("-"), self.hparams["comments_filter"]))
                == 0
            )

        pdf = pd.read_csv(self.hparams["csv"])
        pdf = pdf[pdf["valid"].isin(self.hparams["validity_filter"])]
        pdf = pdf[pdf["comments"].map(filter_by_notes)]
        if self.hparams["mode"] == "prepost" or self.hparams["pre_available"]:
            pdf = pdf[pdf["has_prefire"]]
        # Create folders sets
        self.hparams["key"] = int(self.hparams["key"])
        val_fold = (self.hparams["key"] + 1) % pdf["fold"].max()

        if self.hparams["train_fold"] is None:
            train_fold = ~pdf["fold"].isin((self.hparams["key"], val_fold))
        else:
            train_fold = pdf["fold"] == int(self.hparams["train_fold"])
        self.test_folders = pdf[pdf["fold"] == self.hparams["key"]]["folder"].to_list()
        self.val_folders = pdf[pdf["fold"] == val_fold]["folder"].to_list()
        self.train_folders = pdf[train_fold]["folder"].to_list()

        # Assert everything is right
        assert (
            len(self.train_folders) != 0
            and len(self.val_folders) != 0
            and len(self.test_folders) != 0
        )
        if self.hparams["train_fold"] is None:
            assert (
                len(self.test_folders + self.val_folders + self.train_folders)
                == pdf.shape[0]
            )
        assert len(set(self.train_folders) & set(self.test_folders)) == 0
        assert len(set(self.train_folders) & set(self.val_folders)) == 0
        assert len(set(self.test_folders) & set(self.val_folders)) == 0

        self.train_transforms = [
            config_to_object("torchvision.transforms", k, v)
            for k, v in self.hparams["train_transform"].items()
        ]

        self.test_transforms = [
            config_to_object("torchvision.transforms", k, v)
            for k, v in self.hparams["test_transform"].items()
        ]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.generator = torch.Generator().manual_seed(self.hparams["seed"])

        self.batch_size = self.hparams["batch_size"]

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = CaliforniaDataset(
                "data/california",
                transforms=self.train_transforms,
                patch_size=self.hparams["patch_size"],
                keep_burned_only=self.hparams["keep_burned_only"],
                folder_list=self.train_folders,
                mode=self.hparams["mode"],
            )
        if stage in ("fit", "validate", None):
            self.val_dataset = CaliforniaDataset(
                "data/california",
                transforms=self.test_transforms,
                patch_size=self.hparams["patch_size"],
                keep_burned_only=self.hparams["keep_burned_only"],
                folder_list=self.val_folders,
                mode=self.hparams["mode"],
            )
        if stage in ("test", None):
            self.test_dataset = CaliforniaDataset(
                "data/california",
                transforms=self.test_transforms,
                patch_size=self.hparams["patch_size"],
                keep_burned_only=self.hparams["keep_burned_only"],
                folder_list=self.test_folders,
                mode=self.hparams["mode"],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )


class CaliforniaDataset(Dataset):
    def __init__(
        self,
        root: str,
        patch_size: int = 512,
        folder_list: List[str] = None,
        keep_burned_only: bool = True,
        transforms=None,
        mode: str = "post",
    ):
        # Assert validity
        if mode not in ["post", "prepost"]:
            raise ValueError("mode must be post or prepost")

        self.transforms = transforms
        self.patches = []
        # No folder list provided
        if folder_list is None or len(folder_list) == 0:
            folder_list = os.listdir(root)
        # Load all patches
        for folder in tqdm(folder_list):
            if "gdb" in folder or not os.path.exists(
                f"{root}/{folder}/aggregated_bands.npy"
            ):
                continue
            img = np.load(f"{root}/{folder}/aggregated_bands.npy")
            if mode == "prepost":
                img = np.concatenate(
                    [img, np.load(f"{root}/{folder}/pre_aggregated_bands.npy")], axis=-1
                )
            mask = np.load(f"{root}/{folder}/mask.npy")
            # Init
            img = np.concatenate([img, mask.reshape(*mask.shape, 1)], axis=-1)
            img_size = img.shape[0]
            usable_size = img_size // patch_size * patch_size
            overlapping_start = img_size - patch_size
            # Portioning
            to_cut = img[:usable_size, :usable_size]
            last_row = img[overlapping_start:, :usable_size]
            last_column = img[:usable_size, overlapping_start:]
            last_crop = img[overlapping_start:img_size, overlapping_start:img_size]
            # Crop
            wanted_crop_size = (patch_size, patch_size, img.shape[-1])
            last_row = util.view_as_blocks(last_row, wanted_crop_size)
            last_column = util.view_as_blocks(last_column, wanted_crop_size)
            crops = util.view_as_blocks(to_cut, wanted_crop_size)
            # Reshaping
            crops = crops.reshape(crops.shape[0] * crops.shape[1], *wanted_crop_size)
            last_row = last_row.reshape(last_row.shape[1], *wanted_crop_size)
            last_column = last_column.reshape(last_column.shape[0], *wanted_crop_size)
            last_crop = last_crop.reshape(1, *last_crop.shape)
            # Merge
            merged = np.concatenate([crops, last_column, last_row, last_crop])
            if keep_burned_only:
                merged = merged[merged[:, :, :, -1].sum(axis=(1, 2)) > 0]
            self.patches.append(merged)

        self.patches = np.concatenate(self.patches)
        print(f"Dataset len = {self.patches.shape[0]}")

    def __getitem__(self, item):
        result = {"image": self.patches[item, :, :, :-1]}
        if self.transforms is not None:
            for t in self.transforms:
                result = t(result)
        result["mask"] = torch.from_numpy(self.patches[item, :, :, -1]).unsqueeze(0)
        return result

    def __len__(self):
        return self.patches.shape[0]


"""
SECTION BELOW IS NOT TESTED SUFFICIENTLY BUT IT SHOULD WORK
"""


class HDF5CaliforniaDataset(Dataset):
    def __init__(
        self,
        hdf5_folder: str,
        patch_size: int = 512,
        fold_list: List[int] = None,
        keep_burned_only: bool = True,
        transforms=None,
        mode: str = "post",
        pre_available: bool = False,
    ):
        # Assert validity
        if mode not in ["post", "prepost"]:
            raise ValueError("mode must be post or prepost")

        self.transforms = transforms
        self.patches = []
        # No folder list provided
        if fold_list is None or len(fold_list) == 0:
            fold_list = list(range(5))
        fold_list = [str(x) for x in fold_list]
        # Load all patches
        for dataset_file in glob(f"{hdf5_folder}/*.hdf5"):
            with h5py.File(dataset_file, "r") as dataset:
                for fold in fold_list & set(dataset.keys()):
                    for uid in dataset[fold].keys():
                        matrices = dict(dataset[fold][uid].items())
                        if "pre_fire" not in matrices and pre_available:
                            continue
                        if mode != "prepost":
                            matrices.pop("pre_fire")
                        mask = matrices.pop("mask")[...]
                        # Init
                        img = np.concatenate(
                            list(matrices.values()) + [mask.reshape(*mask.shape, 1)],
                            axis=-1,
                        )
                        img_size = img.shape[0]
                        usable_size = img_size // patch_size * patch_size
                        overlapping_start = img_size - patch_size
                        # Portioning
                        to_cut = img[:usable_size, :usable_size]
                        last_row = img[overlapping_start:, :usable_size]
                        last_column = img[:usable_size, overlapping_start:]
                        last_crop = img[
                            overlapping_start:img_size, overlapping_start:img_size
                        ]
                        # Crop
                        wanted_crop_size = (patch_size, patch_size, img.shape[-1])
                        last_row = util.view_as_blocks(last_row, wanted_crop_size)
                        last_column = util.view_as_blocks(last_column, wanted_crop_size)
                        crops = util.view_as_blocks(to_cut, wanted_crop_size)
                        # Reshaping
                        crops = crops.reshape(
                            crops.shape[0] * crops.shape[1], *wanted_crop_size
                        )
                        last_row = last_row.reshape(
                            last_row.shape[1], *wanted_crop_size
                        )
                        last_column = last_column.reshape(
                            last_column.shape[0], *wanted_crop_size
                        )
                        last_crop = last_crop.reshape(1, *last_crop.shape)
                        # Merge
                        merged = np.concatenate(
                            [crops, last_column, last_row, last_crop]
                        )
                        if keep_burned_only:
                            merged = merged[merged[:, :, :, -1].sum(axis=(1, 2)) > 0]
                        self.patches.append(merged)

        self.patches = np.concatenate(self.patches)
        print(f"Dataset len = {self.patches.shape[0]}")

    def __getitem__(self, item):
        result = {"image": self.patches[item, :, :, :-1]}
        if self.transforms is not None:
            for t in self.transforms:
                result = t(result)
        result["mask"] = torch.from_numpy(self.patches[item, :, :, -1]).unsqueeze(0)
        return result

    def __len__(self):
        return self.patches.shape[0]
