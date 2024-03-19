import torch
import copy
from scene.gaussian_model import GaussianModel


class GaussianHandler():
    def __init__(self, gaussians : GaussianModel, fl, ufl, interval_num = 10) -> None:
        super(GaussianHandler, self).__init__()
        self.interval_num = interval_num
        self.box_num = self.interval_num**3
        coords = gaussians.get_xyz
        scalings = gaussians._scaling
        self.worldMin, self.worldMax = coords.min(0)[0], coords.max(0)[0]
        self.scalingMin, self.scalingMax = scalings.min(), scalings.max()
        self.fl = fl
        self.ufl = ufl

    @property
    def get_gaussians(self):
        return self.gaussians

    def normalize(self, gaussians : GaussianModel) -> GaussianModel:
        gaussians._xyz = (gaussians.get_xyz - self.worldMin) / (self.worldMax - self.worldMin) 
        gaussians._scaling = (gaussians._scaling - self.scalingMin) / (self.scalingMax - self.scalingMin)
        return gaussians

    def box_sort(self, gaussians : GaussianModel):
        with torch.no_grad():
            f_gaussians = self.fl(self.normalize(gaussians))
            sorted = torch.empty_like(gaussians)
            interval_size = 1.0/self.interval_num
            last = 0
            for i in range(self.box_num):
                x = i % self.interval_num
                y = (i // self.interval_num) % self.interval_num
                z = i // self.interval_num**2
                mask = torch.all(torch.cat([gaussians._xyz >= torch.FloatTensor([interval_size*x, interval_size*y, interval_size*z]).cuda(),
                                gaussians._xyz < torch.FloatTensor([interval_size*(x+1), interval_size*(y+1), interval_size*(z+1)]).cuda()], -1), -1)
                box_gaussians = f_gaussians[mask]
                count = box_gaussians.shape[0]
                if count == 0:
                    continue
                sorted[last:last+count] = box_gaussians
                last += count
            return self.denormalize(self.ufl(sorted))
        
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