import torch
import torchmetrics as tm
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import utils

from torch import nn
from .burntnet import BurntNet
from loss import AsymmetricUnifiedFocalLoss

from typing import Union, Dict, List, Literal, Tuple, Any
from torchmetrics import Metric

LossStr = Literal['bce', 'iou', 'bce_iou', 'auf']

class BCEJaccardLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.CrossEntropyLoss()
        self.jaccard = smp.losses.JaccardLoss(mode='multiclass', log_loss=False, from_logits=True)
        return
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        y_true = y_true.long()
        a = self.bce(y_pred, y_true)
        b = self.jaccard(y_pred, y_true)
        return a + b

class BurntNetPL(pl.LightningModule):
    def __init__(
        self,
        features: List[int],
        in_channels: int,
        kernel_size: int,
        nclasses: int,
        lr: float,
        loss: LossStr,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        self.lr = lr
        self.loss_fn = loss
        assert loss in {'bce', 'iou', 'bce_iou', 'auf'}
        self.save_hyperparameters()
        
        self.model = BurntNet(
            features,
            in_channels,
            kernel_size,
            nclasses,
            engine
        )
        
        self.loss = self.create_loss(loss)
        self.create_metrics()
        self.batch_to_log = [0, 5]
        return
        
    def create_loss(self, loss: LossStr) -> nn.Module:
        if loss == 'bce':
            return nn.CrossEntropyLoss()
        elif loss == 'iou':
            return smp.losses.JaccardLoss(mode='multiclass', log_loss=False, from_logits=True)
        elif loss == 'bce_iou':
            return BCEJaccardLoss()
        elif loss == 'auf':
            return AsymmetricUnifiedFocalLoss(0.5, 0.6, 0.1)
        else:
            raise ValueError(f'Invalid loss specified {loss}')
            
    def create_metrics(self) -> Metric:
        metric = tm.MetricCollection({
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
        })
        self.train_metrics = metric.clone(prefix='train_')
        self.val_metrics = metric.clone(prefix='val_')
        self.test_metrics = metric.clone(prefix='test_')
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=55, power=1
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def training_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        out = self(images)
        if self.loss_fn == 'auf':
            out = torch.softmax(out, dim=1)
        self.train_metrics.update(out, masks)
        loss = self.loss(out, masks)
        self.log('train_loss', loss)
        return loss
        
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()
        return
        
    def validation_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        out = self(images)
        if self.loss_fn == 'auf':
            out = torch.softmax(out, dim=1)
        self.val_metrics.update(out, masks)
        loss = self.loss(out, masks)
        self.log('val_loss', loss)
        return loss
        
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()
        return
        
    def test_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
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
        return
        
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
        return
    
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
        return