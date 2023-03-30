import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics as tm
import torchvision.utils as vutils
import utils
from loss import AsymmetricUnifiedFocalLoss
from torch.nn import functional as F


class DeepLabV3Plus(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str,
        n_channels: int = 12,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=n_channels,
            classes=n_classes,
        )
        self.loss = AsymmetricUnifiedFocalLoss(0.5, 0.6, 0.1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["image"].float()
        masks = masks.squeeze(1)
        out = self(images)
        out = torch.softmax(out, dim=1)
        loss = self.loss(out, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["image"].float()
        masks = masks.squeeze(1)
        out = self(images)
        out = torch.softmax(out, dim=1)
        loss = self.loss(out, masks)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["image"].float()
        masks = masks.squeeze(1)
        out = self(images)
        out = torch.softmax(out, dim=1)
        loss = self.loss(out, masks)

        self.test_iou(out, masks)
        self.test_f1(out, masks)

        self.log("test_loss", loss)
        self.log_dict(self.test_iou)
        self.log_dict(self.test_f1)

        if batch_idx in self.batch_to_log:
            self._log_images(
                images,
                masks,
                out.argmax(dim=1),
                out,
                prefix="test",
                draw_rgb=True,
                draw_heatmap=True,
            )
        return loss

    def _log_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        pred_mask: torch.Tensor,
        logits: torch.Tensor,
        prefix: str = "",
        draw_rgb: bool = False,
        draw_heatmap: bool = False,
    ):
        for i, figure in enumerate(utils.draw_figure(masks, pred_mask)):
            if hasattr(self.logger.experiment, "log_image"):
                self.logger.experiment.log_figure(
                    figure=figure,
                    figure_name=f"{prefix}S{self.global_step}N{i}",
                    step=self.global_step,
                )
