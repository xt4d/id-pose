import torch
import torch.nn as nn
import numpy as np
from functools import partial
import kornia
import clip


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPImageEmbedder(AbstractEncoder):
    """
        Uses the CLIP image encoder.
        Not actually frozen... If you want that set cond_stage_trainable=False in cfg
        """
    def __init__(
            self,
            model='ViT-L/14',
            jit=False,
            device='cpu',
            antialias=False,
            clip_root=None
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit, download_root=clip_root)
        # We don't use the text part so delete it
        del self.model.transformer
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # Expects inputs in the range -1, 1
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        if isinstance(x, list):
            # [""] denotes condition dropout for ucg
            device = self.model.visual.conv1.weight.device
            return torch.zeros(1, 768, device=device)
        return self.model.encode_image(self.preprocess(x)).float()

    def encode(self, im):
        return self(im).unsqueeze(1)
