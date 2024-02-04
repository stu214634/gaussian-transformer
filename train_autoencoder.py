from random import randint
import time
import torch
import sys
import numpy as np
from argparse import ArgumentParser
from torch.autograd import Variable
from arguments import ModelParams, OptimizationParams, PipelineParams
from model.autoencoder import GAutoEncoder
from model.shared import subsequent_mask
from model.model import make_model, EncoderDecoder
from scene import GaussianModel, Scene
from gaussian_renderer import render, network_gui
from torch.utils.tensorboard import SummaryWriter
import time
import os
import lpips
import math

START_GAUSSIAN = torch.zeros(64,dtype=torch.float32)
START_GAUSSIAN[59] = 1
PAD_GAUSSIAN = torch.zeros(64,dtype=torch.float32)
PAD_GAUSSIAN[61] = 1
END_GAUSSIAN = torch.zeros(64,dtype=torch.float32)
END_GAUSSIAN[63] = 1

def fuzzy_token_equal(gaussian, token):
    return torch.sum(torch.abs(gaussian - token), -1) <= 0.5

    
def flattenGaussians(x : GaussianModel):
    features = x.get_features
    features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))
    rot = x._rotation
    opa = x._opacity
    xyz = x._xyz
    sca = x._scaling
    flags = torch.zeros((sca.shape[0], 5), device="cuda")
    out = torch.concat((features, rot, opa, xyz, sca, flags), axis=1)
    return out
    
def unflattenGaussians(x) -> GaussianModel:
    gaussian_model = GaussianModel(3)
    gaussian_model.active_sh_degree = 3
    features =  x[:, :48].reshape((x.shape[0], 16, 3))
    gaussian_model._features_dc = features[:, 0:1, :]
    gaussian_model._features_rest = features[:, 1:, :]
    gaussian_model._rotation = x[:, 48:52]
    gaussian_model._opacity = x[:, 52:53]
    gaussian_model._xyz = x[:, 53:56]
    gaussian_model._scaling = x[:, 56: 59]
    return gaussian_model

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

if __name__ == "__main__":

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

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, -1)
    f_gaussians = flattenGaussians(scene.gaussians).half().cuda()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    torch.autograd.set_detect_anomaly(True)
    model = GAutoEncoder()
    model.half()
    model.cuda()
    model.train()
    model_opt = NoamOpt(2048, 2, 2000,
            torch.optim.Adamax(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-4))
    
    loss_fn = torch.nn.MSELoss()

    
    if os.path.exists("best_autoencoder.pt"):
        print("Loading Model")
        model.load_state_dict(torch.load("best_autoencoder.pt"))
    
    writer = SummaryWriter('runs/gaussian_autoencoder')
    global_step = 0
    model_opt._step = global_step
    model.train()
    lowest_loss = 1e4
    for epoch in range(0, 20000, 1):

        viewpoint_stack = scene.getTrainCameras().copy()
        for _ in range(len(viewpoint_stack)):
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            view_gaussians = f_gaussians[visibility_filter]
            mask = torch.rand(view_gaussians.shape[0]) >= 0.8
            data = torch.unsqueeze(view_gaussians[mask], 0)
            padding = torch.zeros((1, (-data.shape[1])%256, data.shape[2])).half().cuda()
            padding[:] = PAD_GAUSSIAN
            data = torch.cat([data, padding], 1)
            prediction = model(data)
            loss = loss_fn(prediction, data)
            loss.backward()
            model_opt.step()
            model_opt.optimizer.zero_grad()
            writer.add_scalar("loss", loss.data.item(), model_opt._step)
            
            if loss < lowest_loss:
                lowest_loss = loss
                torch.save(model.state_dict(), "best_autoencoder.pt")
        
    # All done
    print("\nTraining complete.")
