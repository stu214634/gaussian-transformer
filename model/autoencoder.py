import torch
import torch.nn as nn
import torch.nn.functional as F

from scene.gaussian_model import GaussianModel

from .shared import LayerNorm


class GAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    



# class GAutoEncoder(nn.Module):
#     def __init__(self) -> None:
#         super(GAutoEncoder, self).__init__()
#         self.encoder = GEncoder()
#         self.decoder = GDecoder(self.encoder.d_model)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, xX, xY, xZ):
#         xX = self.dropout(xX)
#         xZ = self.dropout(xZ)
#         xY = self.dropout(xY)
#         features, residuals = self.encoder(xX, xY, xZ)
#         return self.decoder(features, residuals)


# class FeatureBlock(nn.Module):
#     def __init__(self, d_in=64, N=5) -> None:
#         super(FeatureBlock, self).__init__()
#         self.layers = nn.Sequential()
#         self.d_out = d_in*2**N
#         self.layers.add_module("reduce_in", nn.Conv1d(d_in, d_in, 5, 4, 2))
#         for i in range(N):
#             self.layers.add_module(f"block_{i}", self.makeBlock(d_in*2**i))

#     def makeBlock(self, d_in, n=5):
#         d_out = d_in*2
#         block = nn.Sequential()
#         block.add_module("c_reduce", nn.Conv1d(d_in, d_out, 3, 2, 1))
#         for i in range(n):
#             block.add_module(f"c_{i}", nn.Conv1d(d_out, d_out, 3, 1, "same"))
#             block.add_module(f"a_{i}", nn.SiLU())
#         block.add_module("norm", LayerNorm(d_out, dim=1))
#         return block
        
#     def forward(self, x):
#         x = torch.transpose(x, 1, 2)
#         return self.layers(x)
    

# class UBlock(nn.Sequential):
#     def __init__(self, d_in, N=3) -> None:
#         super(UBlock, self).__init__()
#         for i in range(N):
#             self.add_module(f"block_{i}", self.makeBlock(d_in))

#     def makeBlock(self, d_in, n=5):
#         block = nn.Sequential()
#         block.add_module("c_reduce", nn.Conv2d(d_in, d_in, 3, (1,2), (1,1)))
#         for i in range(n):
#             block.add_module(f"c_{i}", nn.Conv2d(d_in, d_in, 3, 1, "same"))
#             block.add_module(f"a_{i}", nn.SiLU())
#         block.add_module("norm", LayerNorm(d_in, dim=1))
#         return block

         
# class GEncoder(nn.Module):
#     def __init__(self) -> None:
#         super(GEncoder, self).__init__()
#         self.fBlockX = FeatureBlock()
#         self.fBlockY = FeatureBlock()
#         self.fBlockZ = FeatureBlock()
#         self.d_model = self.fBlockX.d_out
#         self.uBlock = UBlock(self.d_model)
#         self.reduce = nn.Conv2d(self.d_model, self.d_model, (3, 1), 1, "valid")

#     def forward(self, xX, xY, xZ):
#         residuals = []
#         featX = torch.unsqueeze(self.fBlockX(xX), 2)
#         featY = torch.unsqueeze(self.fBlockX(xY), 2)
#         featZ = torch.unsqueeze(self.fBlockX(xZ), 2)
#         features = torch.cat([featX, featY, featZ], 2)
#         for _, module in enumerate(self.uBlock):
#             residuals.append(features)
#             features = module(features)
#         residuals.append(nn.functional.softmax(features, 1))
#         return self.reduce(features), residuals[::-1]

# class GaussianUnshuffle1D(nn.Module):
#     def __init__(self) -> None:
#         super(GaussianUnshuffle1D, self).__init__()
    
#     def forward(self, x):
#         return torch.reshape(x, (x.shape[0], x.shape[1]//2, x.shape[2]*2))
    
# class GaussianUnshuffle2D(nn.Module):
#     def __init__(self) -> None:
#         super(GaussianUnshuffle2D, self).__init__()
    
#     def forward(self, x):
#         return torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]*2))

# class ReconBlock(nn.Module):
#     def __init__(self, d_in=2048, N=5) -> None:
#         super(ReconBlock, self).__init__()
#         self.layers = nn.Sequential()
#         self.d_out = d_in//2**N
#         for i in range(N):
#             self.layers.add_module(f"block_{i}", self.makeBlock(d_in//2**i))
#         self.layers.add_module("c_expand", nn.Conv1d(self.d_out, self.d_out*4, 1, 1))
#         self.layers.add_module("unshuffle_1", GaussianUnshuffle1D())
#         self.layers.add_module("unshuffle_2", GaussianUnshuffle1D())

#     def makeBlock(self, d_in, n=5):
#         block = nn.Sequential()
#         block.add_module("u_expand", GaussianUnshuffle1D())
#         for i in range(n):
#             block.add_module(f"c_{i}", nn.Conv1d(d_in//2, d_in//2, 3, 1, "same"))
#             block.add_module(f"a_{i}", nn.SiLU())
#         block.add_module("norm", LayerNorm(d_in//2, dim=1))
#         return block
        
#     def forward(self, x):
#         return self.layers(x)

# class CBlock(nn.Sequential):
#     def __init__(self, d_in, N=3) -> None:
#         super(CBlock, self).__init__()
#         for i in range(N):
#             self.add_module(f"block_{i}", self.makeBlock(d_in))

#     def makeBlock(self, d_in, n=5):
#         block = nn.Sequential()
#         block.add_module("u_expand", GaussianUnshuffle2D())
#         for i in range(n):
#             block.add_module(f"c_{i}", nn.Conv2d(d_in, d_in, 3, 1, "same"))
#             block.add_module(f"a_{i}", nn.SiLU())
#         block.add_module("norm", LayerNorm(d_in, dim=1))
#         return nn.Sequential(block)

# class GDecoder(nn.Module):
#     def __init__(self, d_in) -> None:
#         super(GDecoder, self).__init__()
#         self.expand = nn.ConvTranspose2d(d_in, d_in, (3, 1))
#         self.reduce = nn.Conv2d(d_in, d_in, (3, 1), 1, "valid")
#         self.cBlock = CBlock(d_in)
#         self.reconBlock = ReconBlock(d_in)
    
#     def forward(self, features, residuals):
#         features = self.expand(features)
#         for i, module in enumerate(self.cBlock):
#             features = module(torch.cat([features, residuals[i]], 2))
#         features += residuals[-1]
#         features = self.reduce(features)[:, :, 0, :]
#         features = self.reconBlock(features)
#         return torch.transpose(features, 1, 2)




        




