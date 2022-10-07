from .cuda_interface import log_sigmoid, window_sum
import torch
import torch.nn.functional as F
from typing import Optional

def geometric_attention_activation(logits: torch.Tensor, pos_offset: int = 0,
                                   alpha: Optional[torch.Tensor] = None, normalize: bool = True) -> torch.Tensor:
        p, one_minus_p = log_sigmoid(logits)
        not_previos = window_sum(one_minus_p.cumsum(-1), pos_offset)
        
        # not_previos = F.pad(one_minus_p.cumsum(-1)[..., :-1], (1, 0), value=0.0)

        # sorted, order = one_minus_p.sort()
        # sorted = F.pad(sorted.cumsum(-1)[..., :-1], (1, 0), value=0.0)
        # order = order.argsort()
        # not_previos = sorted.gather(-1, order)

        if alpha is None:
            probs = (not_previos + p).exp()
        else:
            g = p.exp() * alpha + (1.0 - alpha)
            probs = not_previos.exp() * g

        # return probs
        return F.normalize(probs, 1, -1) if normalize else probs
