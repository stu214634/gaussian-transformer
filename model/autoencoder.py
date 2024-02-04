import torch
import torch.nn as nn
import torch.nn.functional as F

class GAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(GAutoEncoder, self).__init__()
        self.encoder = GEncoder()
        self.decoder = GDecoder(self.encoder.d_model)

    def forward(self, x):
        features, residuals = self.encoder(x)
        return self.decoder(features, residuals)


class FeatureBlock(nn.Module):
    def __init__(self, d_in=64, N=5) -> None:
        super(FeatureBlock, self).__init__()
        self.layers = nn.Sequential()
        self.d_out = d_in*2**N
        for i in range(N):
            self.layers.add_module(f"block_{i}", self.makeBlock(d_in*2**i))

    def makeBlock(self, d_in, n=5):
        d_out = d_in*2
        block = nn.Sequential()
        block.add_module("c_reduce", nn.Conv1d(d_in, d_out, 3, 2, 1))
        for i in range(n):
            block.add_module(f"c_{i}", nn.Conv1d(d_out, d_out, 3, 1, "same"))
            block.add_module(f"a_{i}", nn.SiLU())
            block.add_module(f"n_{i}", nn.BatchNorm1d(d_out))
        return block
        
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return self.layers(x)
    

class UBlock(nn.Sequential):
    def __init__(self, d_in, N=3) -> None:
        super(UBlock, self).__init__()
        for i in range(N):
            self.add_module(f"block_{i}", self.makeBlock(d_in))

    def makeBlock(self, d_in, n=5):
        block = nn.Sequential()
        block.add_module("c_reduce", nn.Conv2d(d_in, d_in, 3, (1,2), (1,1)))
        for i in range(n):
            block.add_module(f"c_{i}", nn.Conv2d(d_in, d_in, 3, 1, "same"))
            block.add_module(f"a_{i}", nn.SiLU())
            block.add_module(f"n_{i}", nn.BatchNorm2d(d_in))
        return block

         
class GEncoder(nn.Module):
    def __init__(self) -> None:
        super(GEncoder, self).__init__()
        self.fBlockX = FeatureBlock()
        self.fBlockY = FeatureBlock()
        self.fBlockZ = FeatureBlock()
        self.d_model = self.fBlockX.d_out
        self.uBlock = UBlock(self.d_model)
        self.reduce = nn.Conv2d(self.d_model, self.d_model, (3, 1), 1, "valid")
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residuals = []
        idxsX = torch.squeeze(torch.argsort(x[:, :, 17]))
        idxsY = torch.squeeze(torch.argsort(x[:, :, 18]))
        idxsZ = torch.squeeze(torch.argsort(x[:, :, 19]))
        x = self.dropout(x)
        featX = torch.unsqueeze(self.fBlockX(x[:, idxsX]), 2)
        featY = torch.unsqueeze(self.fBlockX(x[:, idxsY]), 2)
        featZ = torch.unsqueeze(self.fBlockX(x[:, idxsZ]), 2)
        features = torch.cat([featX, featY, featZ], 2)
        for _, module in enumerate(self.uBlock):
            residuals.append(features)
            features = module(features)
        residuals.append(nn.functional.softmax(features, 1))
        return self.reduce(features), residuals[::-1]

class GaussianUnshuffle1D(nn.Module):
    def __init__(self) -> None:
        super(GaussianUnshuffle1D, self).__init__()
    
    def forward(self, x):
        return torch.reshape(x, (x.shape[0], x.shape[1]//2, x.shape[2]*2))
    
class GaussianUnshuffle2D(nn.Module):
    def __init__(self) -> None:
        super(GaussianUnshuffle2D, self).__init__()
    
    def forward(self, x):
        return torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]*2))

class ReconBlock(nn.Module):
    def __init__(self, d_in=2048, N=5) -> None:
        super(ReconBlock, self).__init__()
        self.layers = nn.Sequential()
        self.d_out = d_in//2**N
        for i in range(N):
            self.layers.add_module(f"block_{i}", self.makeBlock(d_in//2**i))
        self.layers = nn.Sequential(self.layers)

    def makeBlock(self, d_in, n=5):
        block = nn.Sequential()
        block.add_module("u_expand", GaussianUnshuffle1D())
        for i in range(n):
            block.add_module(f"c_{i}", nn.Conv1d(d_in//2, d_in//2, 3, 1, "same"))
            block.add_module(f"a_{i}", nn.SiLU())
            block.add_module(f"n_{i}", nn.BatchNorm1d(d_in//2))
        return block
        
    def forward(self, x):
        return self.layers(x)

class CBlock(nn.Sequential):
    def __init__(self, d_in, N=3) -> None:
        super(CBlock, self).__init__()
        for i in range(N):
            self.add_module(f"block_{i}", self.makeBlock(d_in))

    def makeBlock(self, d_in, n=5):
        block = nn.Sequential()
        block.add_module("u_expand", GaussianUnshuffle2D())
        for i in range(n):
            block.add_module(f"c_{i}", nn.Conv2d(d_in, d_in, 3, 1, "same"))
            block.add_module(f"a_{i}", nn.SiLU())
            block.add_module(f"n_{i}", nn.BatchNorm2d(d_in))
        return nn.Sequential(block)

class GDecoder(nn.Module):
    def __init__(self, d_in) -> None:
        super(GDecoder, self).__init__()
        self.expand = nn.ConvTranspose2d(d_in, d_in, (3, 1))
        self.reduce = nn.Conv2d(d_in, d_in, (3, 1), 1, "valid")
        self.cBlock = CBlock(d_in)
        self.reconBlock = ReconBlock(d_in)
    
    def forward(self, features, residuals):
        features = self.expand(features)
        for i, module in enumerate(self.cBlock):
            features = module(torch.cat([features, residuals[i]], 2))
        features += residuals[-1]
        features = self.reduce(features)[:, :, 0, :]
        features = self.reconBlock(features)
        return torch.transpose(features, 1, 2)




        




