import torch
import copy
from scene.gaussian_model import GaussianModel


def flattenGaussians(x : GaussianModel):
    features = x.get_features
    features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))
    rot = x._rotation
    opa = x._opacity
    xyz = x._xyz
    sca = x._scaling
    flags = torch.zeros((sca.shape[0], 3), device="cuda")
    out = torch.concat((features, rot, opa, xyz, sca, flags), axis=1)
    return out
    
def unflattenGaussians(x) -> GaussianModel:
    gaussian_model = GaussianModel(1)
    gaussian_model.active_sh_degree = 1
    features =  x[:, :12].reshape((x.shape[0], 4, 3))
    gaussian_model._features_dc = features[:, 0:1, :]
    gaussian_model._features_rest = features[:, 1:, :]
    gaussian_model._rotation = x[:, 12:16]
    gaussian_model._opacity = x[:, 16:17]
    gaussian_model._xyz = x[:, 17:20]
    gaussian_model._scaling = x[:, 20: 23]
    return gaussian_model


class GaussianHandler():
    def __init__(self, gaussians : GaussianModel, interval_num = 10) -> None:
        super(GaussianHandler, self).__init__()
        self.interval_num = interval_num
        self.box_num = self.interval_num**3
        coords = gaussians.get_xyz
        scalings = gaussians._scaling
        self.worldMin, self.worldMax = coords.min(0)[0], coords.max(0)[0]
        self.scalingMin, self.scalingMax = scalings.min(), scalings.max()

    @property
    def get_gaussians(self):
        return self.gaussians

    def normalize(self, gaussians : GaussianModel) -> GaussianModel:
        gaussians._xyz = (gaussians.get_xyz - self.worldMin) / (self.worldMax - self.worldMin) 
        gaussians._scaling = (gaussians._scaling - self.scalingMin) / (self.scalingMax - self.scalingMin)
        return gaussians

    def box_sort(self, gaussians : GaussianModel):
        with torch.no_grad():
            gaussians = flattenGaussians(self.normalize(gaussians))
            sorted = torch.empty_like(gaussians)
            interval_size = 1.0/self.interval_num
            last = 0
            for i in range(self.box_num):
                x = i % self.interval_num
                y = (i // self.interval_num) % self.interval_num
                z = i // self.interval_num**2
                mask = torch.all(torch.cat([gaussians[:, 17:20] >= torch.FloatTensor([interval_size*x, interval_size*y, interval_size*z]).cuda(),
                                gaussians[:, 17:20] < torch.FloatTensor([interval_size*(x+1), interval_size*(y+1), interval_size*(z+1)]).cuda()], -1), -1)
                box_gaussians = gaussians[mask]
                count = box_gaussians.shape[0]
                if count == 0:
                    continue
                sorted[last:last+count] = box_gaussians
                last += count
            return sorted
        
    def denormalize_copy(self, gaussians : GaussianModel):
        g = GaussianModel(3)
        g._features_dc = gaussians._features_dc
        g._features_rest = gaussians._features_rest
        g._opacity = gaussians._opacity
        g._rotation = gaussians._rotation
        g._xyz = gaussians.get_xyz * (self.worldMax - self.worldMin) + self.worldMin
        g._scaling = gaussians._scaling * (self.scalingMax - self.scalingMin) + self.scalingMin
        return g
    
    def denormalize(self, gaussians : GaussianModel):
        gaussians._xyz = gaussians.get_xyz * (self.worldMax - self.worldMin) + self.worldMin
        gaussians._scaling = gaussians._scaling * (self.scalingMax - self.scalingMin) + self.scalingMin
        return gaussians