from scene import Scene
from src.gaussians import GaussianModel
from src.transformer import train, test
import torch

class ModelParams(): 
    def __init__(self, source_path, model_path):
        self.sh_degree = 1
        self.source_path = source_path
        self.model_path = model_path
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False

 
class Pipe():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

params = ModelParams("C:/Users/anw/repos/gaussian-transformer/tiramisu_ds", "C:/Users/anw/repos/gaussian-transformer/output/tiramisu_gs")
gaussians = GaussianModel(torch.empty((0)), torch.empty((0)), torch.empty((0)), torch.empty((0)), torch.empty((0)))
scene = Scene(params, gaussians, -1)

#test(gaussians)
print("Training")
train(gaussians, scene.getTrainCameras())