import torch

from torch import nn
from typing import Union, Dict, List, Optional, Callable, Tuple, Literal

from .morpho import MPMRM, MRMBlock

class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int, 
        nblocks: int,
        f1: int,
        f2: int,
        f3: int,
        with_maxpool: bool=True,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.nblocks = nblocks
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.out_channels = f1 + f2 + f3
        self.engine = engine
        
        acc = []
        inch = in_channels
        for _ in range(nblocks):
            acc.append(
                MPMRM(
                    inch, 
                    self.out_channels,
                    kernel_size,
                    f1, f2, f3,
                    engine
                )
            )
            inch = self.out_channels
        if with_maxpool:
            acc.append(nn.MaxPool2d(2))
        self.enc = nn.Sequential(*acc)
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)
    
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        f1: int,
        f2: int,
        f3: int,
        with_mrm: bool=False,
        with_tconv: bool=True,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.out_channels = f1 + f2 + f3
        self.engine = engine
        
        acc = []
        inch = in_channels
        if with_tconv:
            # halve the number of channels, double the resolution
            acc.append(
                nn.ConvTranspose2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=2,
                    stride=2
                )
            )
            inch = self.out_channels
        acc.append(
            MPMRM(
                inch,
                self.out_channels,
                kernel_size,
                f1, f2, f3,
                engine
            )
        )
        if with_mrm:
            acc.append(
                MRMBlock(
                    self.out_channels,
                    kernel_size,
                    f1, f2, f3,
                    engine
                )
            )
        
        self.dec = nn.Sequential(*acc)
        return
    
    def forward(
        self,
        x: torch.Tensor,
        skip_x: torch.Tensor
    ) -> torch.Tensor:
        if not skip_x is None:
            x = torch.concat(
                (skip_x, x),
                dim=1
            )
        x = self.dec(x)
        return x

class BurntNetEncoder(nn.Module):
    def __init__(
        self,
        features: List[int],
        in_channels: int, 
        kernel_size: int, 
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        
        enc = []
        inch = in_channels
        for idx, f in enumerate(features):
            if idx == 0 or idx == len(features) - 1:
                nblocks = 1
            else:
                nblocks = 2
            f1, f2, f3 = compute_features(f)
            enc.append(
                EncoderBlock(
                    inch,
                    kernel_size, 
                    nblocks,
                    f1, f2, f3,
                    with_maxpool=idx != len(features) - 1,
                    engine=engine
                )
            )
            inch = f
        self.enc = nn.ModuleList(enc)
        return
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        enc_features = []
        feat_map = x
        for m in self.enc:
            t = m(feat_map)
            enc_features.append(t)
            feat_map = t
        return enc_features

class BurntNetDecoder(nn.Module):
    def __init__(
        self,
        features: List[int],
        kernel_size: int,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        dec = []
        reversed_features = list(reversed(features))
        inch = reversed_features[0]
        for idx, f in enumerate(reversed_features):
            next_f = reversed_features[idx + 1] if idx != len(features) - 1 else f
            f1, f2, f3 = compute_features(next_f)
            dec.append(
                DecoderBlock(
                    inch,
                    kernel_size,
                    f1, f2, f3,
                    with_mrm=idx != 0,
                    with_tconv=idx != 0,
                    engine=engine
                )
            )
            inch = f
        self.dec = nn.ModuleList(dec)
        return
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        assert len(x) == len(self.dec)
        out = x[-1]
        for idx, (f, m) in enumerate(zip(reversed(x), self.dec)):
            if idx == 0:
                out = m(out, None)
            else:
                out = m(out, f)
        return out

class BurntNetHead(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        nclasses: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, nclasses, kernel_size=1)
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class BurntNet(nn.Module):
    def __init__(
        self,
        features: List[int],
        in_channels: int,
        kernel_size: int,
        nclasses: int,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        self.features = features
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        
        self.enc = BurntNetEncoder(
            features,
            in_channels,
            kernel_size,
            engine
        )
        
        self.dec = BurntNetDecoder(
            features,
            kernel_size,
            engine
        )
        self.head = BurntNetHead(
            features[0], nclasses,
        )
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = self.enc(x)
        out = self.dec(enc_features)
        out = self.head(out)
        return out
    
def compute_features(f: int) -> Tuple[int, int, int]:
    f1 = f2 = f3 = f // 3
    if f1 + f2 + f3 != f:
        f1 += f - (f1 + f2 + f3)
    assert f1 + f2 + f3 == f
    return f1, f2, f3