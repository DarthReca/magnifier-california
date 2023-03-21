# Copyright 2022 Daniele Rege Cambrin
from itertools import chain, groupby, product
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms.functional as TF
import utils
from loss import AsymmetricUnifiedFocalLoss
from transformers import SegformerConfig, SegformerDecodeHead, SegformerModel

from .unet import UNetDecoder, UNetEncoder


class MagnifierNet(pl.LightningModule):
    """
    A multiple net for patch of different size.

    Parameters
    ----------
    small_patch_size: int, optional
        The size of the patch to be considered as small
    num_classes: int, optional
        the number of classes
    channels: int, optional
        the number of channels
    big_patch_size: int, optional
        the size of the patch to be considered as big
    model: str, optional
        the model to use, can be "segformer" or "unet"
    """

    def __init__(
        self,
        model: str = "segformer",
        big_patch_size: int = 512,
        small_patch_size: int = 64,
        num_classes: int = 2,
        channels: int = 12,
    ):
        super().__init__()
        if model not in ["segformer", "unet"]:
            raise ValueError("Model not supported")
        # Nets
        self.big_net = None
        self.small_net = None
        self.head = None
        self.postprocess = lambda x: x
        self.backbone_forward = lambda backbone, x: backbone(x)

        if model == "segformer":
            self._segformer_init(num_classes, channels, big_patch_size)
        elif model == "unet":
            self._unet_init(num_classes, channels, big_patch_size)

        # Parameters init
        self.small_patch_size = small_patch_size
        # Loss
        self.loss = AsymmetricUnifiedFocalLoss(0.5, 0.6, 0.1)
        # Metrics
        self.test_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.JaccardIndex(
                    "multiclass", num_classes=2, reduction="none"
                ),
                torchmetrics.F1Score(
                    "multiclass", num_classes=2, average="none", mdmc_average="global"
                ),
                torchmetrics.Precision(
                    "multiclass", num_classes=2, average="none", mdmc_average="global"
                ),
                torchmetrics.Recall(
                    "multiclass", num_classes=2, average="none", mdmc_average="global"
                ),
            ]
        )

    def forward(self, x: torch.Tensor):
        # Forward all images to the net
        hidden_states = list(self.backbone_forward(self.big_net, x))
        # Create crops and their positions
        cropped_batch = []
        positions = []
        for image in x:
            imgs, pos = utils.crop_image(image, self.small_patch_size)
            cropped_batch.append(imgs)
            positions.append(pos)
        positions = list(chain(*positions))
        cropped_batch = torch.concat(cropped_batch)
        num_crops = (x.size()[-1] // self.small_patch_size) ** 2
        # Forward to backbone
        crops_out = self.backbone_forward(self.small_net, cropped_batch)
        # Recompose hidden states
        for i, hidden in enumerate(crops_out):
            hidden_states[i] = torch.concat(
                [
                    hidden_states[i],
                    torch.concat(
                        [
                            utils.recompose_image(
                                hidden[i : i + num_crops],
                                positions[i : i + num_crops],
                            ).unsqueeze(0)
                            for i in range(0, hidden.size()[0], num_crops)
                        ]
                    ),
                ],
                dim=1,
            )
        out = self.head(hidden_states)
        return self.postprocess(out)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch, batch_idx):
        masks, images = (
            batch["mask"].float(),
            batch["image"].float(),
        )
        masks = masks.squeeze(1)
        # Forward
        out = self(images)
        out = torch.sigmoid(out)
        # Loss
        loss = self.loss(out, masks)
        # Log
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        masks, images = (
            batch["mask"].float(),
            batch["image"].float(),
        )
        masks = masks.squeeze(1)
        # Forward
        out = self(images)
        out = torch.sigmoid(out)
        # Loss
        loss = self.loss(out, masks)
        # Log
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        masks, images = (
            batch["mask"].float(),
            batch["image"].float(),
        )
        masks = masks.squeeze(1)
        # Forward
        out = self(images)
        out = torch.sigmoid(out)
        # Loss
        loss = self.loss(out, masks)
        # Metrics
        self.test_metrics(out, masks)
        # Log
        self.log("test_loss", loss)
        self.log_dict(self.test_metrics)
        return loss

    def _segformer_init(
        self, num_classes: int = 2, channels: int = 12, big_patch_size: int = 512
    ):
        config = SegformerConfig(
            num_channels=channels,
            num_labels=num_classes,
            semantic_loss_ignore_index=-100,
        )

        self.big_net = SegformerModel(config)
        self.small_net = SegformerModel(config)

        config.hidden_sizes = [h * 2 for h in config.hidden_sizes]
        self.head = SegformerDecodeHead(config)

        self.postprocess = lambda x: nn.functional.interpolate(
            x, size=big_patch_size, mode="bilinear", align_corners=False
        )

        self.backbone_forward = lambda backbone, x: backbone(
            x, output_hidden_states=True
        )[1]

    def _unet_init(
        self, num_classes: int = 2, channels: int = 12, big_patch_size: int = 512
    ):
        self.big_net = UNetEncoder(channels)
        self.small_net = UNetEncoder(channels)
        self.head = UNetDecoder(
            num_classes=num_classes,
            n_channels=2048,
        )
        self.postprocess = lambda x: x
