import torch

from einops import rearrange
from torch import nn
from abc import abstractmethod, ABC
from typing import Literal, Optional, List, Tuple

class MorphologicalLayer(nn.Module, ABC):
    def __init__(
        self,
        cin: int,
        kernel_size: int,
        padding: int,
        max_computation: Literal['max', 'softmax']='max'
    ) -> None:
        nn.Module.__init__()
        
        self.cin = cin
        self.cout = cout
        self.kernel_size = kernel_size
        self.padding = padding
        self.max_computation = max_computation
        assert max_computation in {'max', 'softmax'}
        
        self.unfold = nn.Unfold((kernel_size, kernel_size), padding=padding)
        # one weight matrix per channel
        self.weights = nn.Parameter(
            torch.rand(cin, kernel_size, kernel_size)
        )
        
        return
    
    @abstractmethod
    def compute_img(
        self,
        unfolded: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        pass
    
    def compute_max(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # x has shape B, C*kernel_size*kernel_size, P
        # reshape it to separate the channels
        k2 = self.kernel_size * self.kernel_size
        c = x.shape[1] / k2
        x = x.view(x.shape[0], c, k2, -1)
        if self.max_computation == 'max':
            x = x.max(dim=2)
        return
    
    def forward(self, x: torch.Tensor) -> torch.tensor:
        # x has shape B, C, H, W
        unfolded = self.unfold(x)
        # unfolded has shape B, C*kernel_size*kernel_size, P (P = number of patches)
        
        # w has shape Cin*kernel_size*kernel_size
        w = self.weights.view(-1)
        # array broadcasting
        # 1, Cin*kernel_size*kernel_size, 1
        w = w.unsqueeze(0).unsqueeze(-1)

        # B, Cin*kernel_size*kernel_size, P
        res = self.compute_img(unfolded, w)
        res = self.compute_max(res)
        
        # TODO reconstruct the image
        
        return res
        
class ErosionMorphoLayer(MorphologicalLayer):
    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: int,
        padding: int
    ) -> None:
        super(MorphologicalLayer, self).__init__(
            cin,
            cout,
            kernel_size,
            padding
        )
        return
    
    def compute_img(
        self,
        unfolded: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        return