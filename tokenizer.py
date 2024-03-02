from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_activation as activ
import sys
import matplotlib.pyplot as plt
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import GaussianModel, network_gui, render
from scene import Scene
from scene.cameras import MiniCam
from train_stacked_transformer import TrainingScene
from torch.utils.tensorboard import SummaryWriter

    
def flattenGaussians(x : GaussianModel):
    features = x.get_features
    features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))
    rot = x._rotation
    opa = x._opacity
    sca = x._scaling
    out = torch.concat((features, rot, opa, sca), axis=1)
    return out
    
def unflattenGaussian(x) -> GaussianModel:
    gaussian_model = GaussianModel(1)
    gaussian_model.active_sh_degree = 1
    features =  x[:12].reshape((1, 4, 3))
    gaussian_model._features_dc = features[:, 0:1, :]
    gaussian_model._features_rest = features[:, 1:, :]
    gaussian_model._rotation = torch.unsqueeze(x[12:16],0)
    gaussian_model._opacity = torch.unsqueeze(x[16:17],0)
    gaussian_model._scaling = torch.unsqueeze(x[17: 20],0)
    gaussian_model._xyz = torch.zeros((1, 3)).cuda()
    return gaussian_model

class Pipe():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

class GMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encode_stack = nn.Sequential(nn.Linear(20, 128),
                                          nn.SiLU(inplace=True),
                                          nn.BatchNorm1d(128),
                                          nn.Linear(128, 192),
                                          nn.SiLU(inplace=True),
                                          nn.BatchNorm1d(192))
        
        self.decode_stack = nn.Sequential(nn.Linear(192, 128),
                                  nn.SiLU(inplace=True),
                                  nn.BatchNorm1d(128),
                                  nn.Linear(128, 128),
                                  nn.SiLU(inplace=True),
                                  nn.BatchNorm1d(128),
                                  nn.Linear(128, 20))


    def encode(self, x):
        return self.encode_stack(x)

    def decode(self, x):
        return self.decode_stack(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))

#standardCam : MiniCam = MiniCam(256, 256)
bg = torch.Tensor([0,0,0]).cuda()
pipe = Pipe()
world_view_transform = torch.Tensor([[ 0.9275, -0.0593,  0.3691, -0.0000],
                                     [-0.2854, -0.7500,  0.5967,  0.0000],
                                     [ 0.2415, -0.6588, -0.7125,  0.0000],
                                     [ 0.0019,  0.0086, -0.2727,  1.0000]]).cuda()

world_view_transform[:,1] = -world_view_transform[:,1]
world_view_transform[:,2] = -world_view_transform[:,2]
full_proj_transform = torch.tensor([[ 0.9232, -0.1053, -0.3691, -0.3691],
                                    [-0.2840, -1.3333, -0.5967, -0.5967],
                                    [ 0.2403, -1.1712,  0.7125,  0.7125],
                                    [ 0.0019,  0.0154,  0.2727,  0.2727]]).cuda()
full_proj_transform[:,1] = -full_proj_transform[:,1]
standardCam = MiniCam(256, 256, 1.024778962135315, 1.57548189163208, 0, 1100, world_view_transform, full_proj_transform)

def inspect(x : GaussianModel, flatten=True):
    im = render(standardCam, x, pipe, bg, 1)["render"]
    for _ in range(5):
        im = F.max_pool2d(im, 2)
    return torch.flatten(im) if flatten else im


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 5000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 2000, 5000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Start GUI server, configure and run training6
    network_gui.init(args.ip, args.port)
    gaussianModel = GaussianModel(1)
    scene = Scene(lp.extract(args), gaussianModel, -1)
    scene.gaussians._xyz = torch.zeros_like(scene.gaussians._xyz)
    flattened = flattenGaussians(scene.gaussians).cuda()
    model = GMLP().train().cuda()
    optim = torch.optim.Adamax(model.parameters(), 0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, cooldown=20)
    tb_writer = SummaryWriter("tokenizer/3")
    torch.autograd.anomaly_mode.set_detect_anomaly(True)
    step = 0
    batch_size = 32_768

    with torch.no_grad():
        hiddens = torch.empty((flattened.shape[0], 192)).cuda()
        for r in range(flattened.shape[0]):
            r_gaussian = unflattenGaussian(flattened[r])
            hiddens[r] = inspect(r_gaussian)

    for epoch in range(10000):
        print(f"Epoch: {epoch}")
        idxs = torch.randperm(flattened.shape[0])
        total_loss = 0
        for i in range(flattened.shape[0] // batch_size):
            idx = idxs[i*batch_size : (i+1)*batch_size]
            gaussians = flattened[idx]
            hidden_target = hiddens[idx]
            r_gaussians = [unflattenGaussian(gaussian) for gaussian in gaussians]
            hidden_pred = model.encode(gaussians)
            h_loss = F.mse_loss(hidden_pred, hidden_target)
            pred = model.decode(hidden_pred)
            i_loss = F.mse_loss(pred, gaussians)

            loss = h_loss + i_loss

            tb_writer.add_scalars("losses", {"h_loss" : h_loss.data.item(), "i_loss" : i_loss.data.item()}, step)

            loss.backward()
            optim.step()
            optim.zero_grad()
            tb_writer.add_scalar("lr", lr_scheduler.optimizer.param_groups[0]['lr'], step)
            total_loss += loss.data.item()
            step += 1
        total_loss /= (i+1)
        lr_scheduler.step(total_loss)
        target = torch.empty((4, 3, 256, 256), device="cuda")
        pred = torch.empty((4, 3, 256, 256), device="cuda")
        hidden_pred = torch.empty((4, 3, 8, 8), device="cuda")
        hidden_target = torch.empty((4, 3, 8, 8), device="cuda")
        idx = idxs[:4]
        encodings = model.encode(flattened[idx])
        preds = model.decode(encodings)
        for i, index in enumerate(idx):
            _flattened = flattened[index]
            target[i] = render(standardCam, unflattenGaussian(_flattened), pipe, bg, 1)["render"]
            pred[i] = render(standardCam, unflattenGaussian(preds[i]), pipe, bg, 1)["render"]
            hidden_pred[i] = torch.reshape(encodings[i], (3, 8, 8))
            hidden_target[i] = inspect(unflattenGaussian(_flattened), False)
        tb_writer.add_images("target", target, step)
        tb_writer.add_images("pred", pred, step)
        tb_writer.add_images("hidden_pred", hidden_pred, step)
        tb_writer.add_images("hidden_target", hidden_target, step)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), "tokenizer.pt")


