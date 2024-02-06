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

from scene.cameras import MiniCam

START_GAUSSIAN = torch.zeros(64,dtype=torch.float32)-10
START_GAUSSIAN[59] = 1
PAD_GAUSSIAN = torch.zeros(64,dtype=torch.float32)-10
PAD_GAUSSIAN[61] = 1
END_GAUSSIAN = torch.zeros(64,dtype=torch.float32)-10
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
    x = torch.squeeze(x)
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
    f_gaussians = flattenGaussians(scene.gaussians).cuda()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    torch.autograd.set_detect_anomaly(True)
    model = GAutoEncoder()
    model.cuda()
    model.train()
    model_opt = NoamOpt(2048, 8, 2000,
            torch.optim.Adamax(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-4))
    
    loss_fn = torch.nn.MSELoss()
    loss_perceptive = lpips.LPIPS(net='alex').cuda()
    
    if os.path.exists("best_autoencoder.pt"):
        print("Loading Model")
        model.load_state_dict(torch.load("best_autoencoder.pt"))
    
    writer = SummaryWriter('runs/gaussian_autoencoder')
    global_step = 333
    model_opt._step = global_step
    model.half()
    model.train()
    lowest_loss = 1e4
    for epoch in range(21, 20000, 1):

        viewpoint_stack = scene.getTrainCameras().copy()
        for _ in range(len(viewpoint_stack)):
            viewpoint_cam : MiniCam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            with torch.no_grad():
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                view_gaussians = f_gaussians[visibility_filter]
                data = torch.unsqueeze(view_gaussians, 0)
                padding = torch.zeros((1, (-data.shape[1])%1024, data.shape[2])).cuda()
                padding[:] = PAD_GAUSSIAN
                data = torch.cat([data, padding], 1)
                idxsX = torch.squeeze(torch.argsort(data[:, :, 17]))
                idxsY = torch.squeeze(torch.argsort(data[:, :, 18]))
                idxsZ = torch.squeeze(torch.argsort(data[:, :, 19]))
                dataX = data[:, idxsX].half()
                dataY = data[:, idxsY].half()
                dataZ = data[:, idxsZ].half()
            print("predicting")
            prediction = model(dataX, dataY, dataZ)
            if epoch > 20:
                in_im = render(viewpoint_cam, unflattenGaussians(data.float()), pipe, background)["render"]
                out_im = render(viewpoint_cam, unflattenGaussians(prediction.float()), pipe, background)["render"]
                l1_i = torch.nn.functional.l1_loss(out_im, in_im)
                l_p = loss_perceptive(out_im*2-1, in_im*2-1)
                l2_g = loss_fn(prediction.float(), dataX.float())
                loss = l2_g*0.4+l1_i*0.4+l_p*0.2
            else:
                loss = loss_fn(prediction.float(), dataX.float())
            print("backwarding")
            loss.backward()
            print("done")
            model_opt.step()
            model_opt.optimizer.zero_grad()
            with torch.no_grad():
                writer.add_scalar("loss", loss.data.item(), model_opt._step)
                writer.add_scalar("lr", model_opt._rate, model_opt._step)
                if epoch > 20:
                    writer.add_scalar("l1", l1_i.data.item(), model_opt._step)
                    writer.add_scalar("l2", l2_g.data.item(), model_opt._step)
                    writer.add_scalar("lp", l_p.data.item(), model_opt._step)
                    writer.add_image("in", in_im, model_opt._step)
                    writer.add_image("out", out_im, model_opt._step)
                if loss.data.item() < lowest_loss:
                    lowest_loss = loss.data.item()
                    torch.save(model.state_dict(), "best_autoencoder.pt")

        
    # All done
    print("\nTraining complete.")
