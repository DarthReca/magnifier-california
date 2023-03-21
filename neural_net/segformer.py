import pytorch_lightning as pl
import torch
import torchmetrics
from loss import AsymmetricUnifiedFocalLoss
from torch.nn import functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class Segformer(pl.LightningModule):
    def __init__(
        self,
        config_name: str,
        n_channels: int = 12,
        n_classes: int = 2,
        final_size: int = 512,
    ):
        super().__init__()
        config = SegformerConfig.from_pretrained(
            config_name, num_channels=n_channels, num_labels=n_classes
        )
        self.model = SegformerForSemanticSegmentation(config)
        self.size = final_size
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

    def forward(self, x):
        x = self.model(x).logits
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)

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
        self.test_metrics(out, masks)
        self.log("test_loss", loss)
        self.log_dict(self.test_metrics)
        return loss
