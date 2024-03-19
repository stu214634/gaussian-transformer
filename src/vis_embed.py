#https://github.com/VAST-AI-Research/TriplaneGaussian/blob/main/tgs/models/renderer.py#L76

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .gaussians import GaussianModel, render
from scene import Scene
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter

emb_dim = 128

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply
inverse_sigmoid = lambda x: np.log(x / (1 - x))

class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int = emb_dim,
        dim_out: int = emb_dim,
        n_neurons: int = 256,
        n_hidden_layers: int = 3,
        activation: str = "silu",
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            self.make_linear(
                dim_in, n_neurons, is_first=True, is_last=False, bias=bias
            ),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons, n_neurons, is_first=False, is_last=False, bias=bias
                ),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(
                n_neurons, dim_out, is_first=False, is_last=True, bias=bias
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = self.make_activation(activation)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError
        
class GSLayerOut(nn.Module):
    in_channels: int = emb_dim
    init_scaling: float = -5.0
    init_density: float = 0.1
    keys = ["shs", "scaling", "rotation", "opacity"]
    out_ch = [12, 3, 4, 1]
    def __init__(self) -> None:
        super().__init__() 
        self.out_layers = nn.ModuleList()
        for key, out_ch in zip(self.keys, self.out_ch):
            layer = nn.Linear(self.in_channels, out_ch)

            # initialize
            if key == "shs":
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            elif key == "scaling":
                nn.init.constant_(layer.bias, self.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                nn.init.constant_(layer.bias, inverse_sigmoid(self.init_density))

            self.out_layers.append(layer)

    def forward(self, x, pts):
        ret = {}
        for k, layer in zip(self.keys, self.out_layers):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)    
                v = torch.clamp(v, min=0, max=0.05)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                v = torch.reshape(v, (v.shape[0], -1, 3))
            ret[k] = v
        ret["xyz"] = pts


        return GaussianModel(**ret)
    
class GSLayerIn(nn.Module):
    out_channels: int = emb_dim
    keys = ["shs", "scaling", "rotation", "opacity"]
    in_ch = [12, 3, 4, 1]
    def __init__(self) -> None:
        super().__init__() 
        self.out_layers = nn.ModuleList()
        for key, in_ch in zip(self.keys, self.in_ch):
            layer = nn.Linear(in_ch, self.out_channels)
            self.out_layers.append(layer)

    def forward(self, x : GaussianModel):
        pts = x.xyz
        out = torch.zeros((pts.shape[0], self.out_channels)).cuda()
        for k, layer in zip(self.keys, self.out_layers):
            if k == "rotation":
                out += layer(x.rotation.flatten(1, -1))
            elif k == "scaling":
                out += layer(x.scaling.flatten(1, -1))
            elif k == "opacity":
                out += layer(x.opacity.flatten(1, -1))
            elif k == "shs":
                out += layer(x.shs.flatten(1, -1))

        return out, pts
    

class VisEmbedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gsIn = GSLayerIn()
        self.gsOut = GSLayerOut()
        self.encodeMLP = MLP()
        self.decodeMLP = MLP()
    
    def encode(self, x : GaussianModel):
        vis_embed, pts = self.gsIn(x)
        return self.encodeMLP(vis_embed), pts

    def decode(self, x, pts):
        return self.gsOut(self.decodeMLP(x), pts)
    
    def forward(self, x):
        vis_embed, pts = self.encode(x)
        return self.decode(vis_embed, pts)


def train(gaussians, cameras, pipe, bg):
    model = VisEmbedNet().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, cooldown=5)
    writer = SummaryWriter(f'VisEmbedLog/')
    step = 0
    spe = len(cameras)
    for epoch in range(1000):
        total_loss = 0
        for cam in cameras:
            target_image = render(cam, gaussians, pipe, bg)["render"]
            pred_gaussians = model(gaussians)
            pred_image = render(cam, pred_gaussians, pipe, bg)["render"]
            loss = torch.nn.functional.mse_loss(pred_image, target_image)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.data.item()
            if step % 50 == 0:
                writer.add_image("target", target_image, step)
                writer.add_image("pred", pred_image, step)
            step += 1
        total_loss /= spe
        lr_scheduler.step(total_loss)
        writer.add_scalar("loss", total_loss, step)
        torch.save(model.state_dict(), "VisEmbed.pt")

def load(path="VisEmbed.pt"):
    model = VisEmbedNet()
    model.load_state_dict(torch.load(path))
    return model.cuda()


        

