import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torchvision.utils as vutils
import utils
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
        self.test_metrics = tm.MetricCollection(
            {
                "IoU": tm.ClasswiseWrapper(
                    tm.JaccardIndex("multiclass", num_classes=2, average="none")
                ),
                "F1Score": tm.ClasswiseWrapper(
                    tm.F1Score(
                        "multiclass",
                        num_classes=2,
                        average="none",
                        multidim_average="global",
                    )
                ),
            }
        )

        self.batch_to_log = [0, 5]

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

        self.test_metrics.update(out, masks)

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

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

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
        images_count = images.size()[0]
        rgb_indexes = (3, 2, 1)
        for i in range(images_count):
            collection = {}
            gt = (masks[i].squeeze() > 0).byte().cpu()
            pr = (pred_mask[i].squeeze() > 0).byte().cpu()
            if draw_rgb:
                img = utils.extract_rgb(images[i], rgb_indexes).cpu()
                collection["original image"] = (img, None)
                if images[i].size()[0] > 12:
                    pre = utils.extract_rgb(images[i][12:], rgb_indexes).cpu()
                    collection["pre image"] = (pre, None)
                collection["ground truth with image"] = (
                    vutils.draw_segmentation_masks(img, gt.bool(), colors=["red"]),
                    None,
                )
                collection["prediction with image"] = (
                    vutils.draw_segmentation_masks(img, pr.bool(), colors=["red"]),
                    None,
                )
            if draw_heatmap:
                collection["prediction heatmap"] = (
                    logits[i][1].unsqueeze(0).cpu(),
                    "viridis",
                )
            collection["ground truth mask"] = (gt.unsqueeze(0), "gray")
            collection["prediction mask"] = (pr.unsqueeze(0), "gray")
            collection = {
                k: (v.permute(1, 2, 0).numpy(), cmap)
                for k, (v, cmap) in collection.items()
            }
            if hasattr(self.logger.experiment, "log_image"):
                figure, axs = plt.subplots(ncols=len(collection), figsize=(20, 20))
                figure.tight_layout()
                for ax, (k, (v, cmap)) in zip(axs, collection.items()):
                    ax.imshow(v, cmap=cmap)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_title(k, {"fontsize": 15})
                self.logger.experiment.log_figure(
                    figure=figure,
                    figure_name=f"{prefix}S{self.global_step}N{i}",
                    step=self.global_step,
                )
                plt.close()
