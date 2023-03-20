import pytorch_lightning as pl
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
            config_name, num_channels=n_channels, num_classes=n_classes
        )
        self.model = SegformerForSemanticSegmentation(config)
        self.size = final_size

    def forward(self, x):
        x = self.model(x)
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
