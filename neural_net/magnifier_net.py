# Copyright 2022 Daniele Rege Cambrin
from itertools import chain, groupby, product
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import utils

from .hf_segformer import HFSegformer, HFSegformerHead, SegformerEncoder


class MagnifierNet(nn.Module):
    """
    A multiple net for patch of different size.

    Parameters
    ----------
    small_patch_size: int, optional
        The size of the patch to be considered as small
    num_classes: int, optional
        the number of classes
    """

    def __init__(
        self,
        small_patch_size: int = 64,
        num_classes: int = 2,
        channels: int = 12,
        freeze_backbones: bool = True
    ):
        super().__init__()
        # Net common parameters
        head_dict = {
            "num_classes": num_classes,
            "classifier_dropout_prob": 0.1,
            "reshape_last_stage": True,
            "decoder_hidden_size": 256,
            "hidden_sizes": [n*2 for n in [32, 64, 160, 256]],
        }
        backbone_dict = {
            "attention_probs_dropout_prob": 0,
            "drop_path_rate": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0,
            "mlp_ratios": [4, 4, 4, 4],
            "num_attention_heads": [1, 2, 5, 8],
            "num_channels": channels,
            "patch_sizes": [7, 3, 3, 3],
            "reshape_last_stage": True,
            "sr_ratios": [8, 4, 2, 1],
            "strides": [4, 2, 2, 2],
            "depths": [2 ,2 , 2, 2],
            "hidden_sizes": [32, 64, 160, 256],
        }
        # Nets
        self.big_net = SegformerEncoder(**backbone_dict)
        self.small_net = SegformerEncoder(**backbone_dict)
        # Lock pretrained
        if freeze_backbones:
            parameters = list(self.small_net.parameters()) + list(self.big_net.parameters())
            for param in parameters:
                param.requires_grad = False

        self.head = HFSegformerHead(**head_dict)
        # Parameters init
        self.small_patch_size = small_patch_size

    def forward(self, x: torch.Tensor):
        # Forward all images to the net
        hidden_states = list(self.big_net(x, output_hidden_states=True)[1])
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
        crops_out = self.small_net(cropped_batch, output_hidden_states=True)[1]
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
        return self.head(hidden_states)


"""
if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(Image.open("../lion1.jpg")).unsqueeze(0)
    img = torch.concat([img, img, img])
    net = MagnifierNet(small_patch_size=64, channels=3)
    out = net(img, hard=torch.tensor([True, False, True]))
    pass
"""
