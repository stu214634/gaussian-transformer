from random import randint
import time
import torch
import sys
import numpy as np
from argparse import ArgumentParser
from torch.autograd import Variable
from arguments import ModelParams, OptimizationParams, PipelineParams
from model.autoencoder import GAutoEncoder
from model.box_sort import GaussianHandler
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
from utils.loss_utils import ssim

START_GAUSSIAN = torch.zeros(26,dtype=torch.float32)
START_GAUSSIAN[23] = 1
PAD_GAUSSIAN = torch.zeros(26,dtype=torch.float32)
PAD_GAUSSIAN[24] = 1
END_GAUSSIAN = torch.zeros(26,dtype=torch.float32)
END_GAUSSIAN[25] = 1

def fuzzy_token_equal(gaussian, token):
    return torch.sum(torch.abs(gaussian - token), -1) <= 0.5


# TODO optimmize individually
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
    x = torch.squeeze(x)
    features =  x[:, :12].reshape((x.shape[0], 4, 3))
    gaussian_model._features_dc = features[:, 0:1, :]
    gaussian_model._features_rest = features[:, 1:, :]
    gaussian_model._rotation = x[:, 12:16]
    gaussian_model._opacity = x[:, 16:17]
    gaussian_model._xyz = x[:, 17:20]
    gaussian_model._scaling = x[:, 20: 23]
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

    network_gui.init(args.ip, args.port)
    for lrm in range(20, 100, 1):


        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, -1)
        handler = GaussianHandler(scene.gaussians)
        scene.gaussians = handler.denormalize(unflattenGaussians(handler.box_sort(scene.gaussians)))
        f_gaussians = flattenGaussians(scene.gaussians).cuda()
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        torch.autograd.set_detect_anomaly(True)
        model = GAutoEncoder()
        model.cuda()
        model.train()
        model_opt = torch.optim.Adam(model.parameters(), lr=0.0000001*lrm*100, eps=1e-15)
        
        loss_fn = torch.nn.L1Loss()
        loss_perceptive = lpips.LPIPS(net='alex').cuda()
        
        # if os.path.exists("best_autoencoder.pt"):
        #     print("Loading Model")
        #     model.load_state_dict(torch.load("best_autoencoder.pt"))
        
        writer = SummaryWriter(f'LRruns/gaussian_autoencoder_{lrm}')
        step = 0
        model.train()
        lowest_loss = 1e4
        for epoch in range(0, 505, 1):
            print(epoch)
            viewpoint_stack = scene.getTrainCameras().copy()
            for _ in range(len(viewpoint_stack)):
                viewpoint_cam : MiniCam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                with torch.no_grad():
                    render_pkg = render(viewpoint_cam, scene.gaussians, pipe, background)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    view_gaussians = f_gaussians[visibility_filter]
                    data = torch.unsqueeze(view_gaussians, 0)
                    #padding = torch.zeros((1, (-data.shape[1])%1, data.shape[2])).cuda()
                    #padding[:] = PAD_GAUSSIAN
                    #data = torch.cat([data, padding], 1)
                prediction = torch.transpose(model(torch.transpose(data, 1, 2)), 1, 2)
                if network_gui.conn == None:
                    network_gui.try_connect()
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                        if custom_cam != None:
                            net_image = render(custom_cam, unflattenGaussians(torch.squeeze(prediction)), pipe, background, scaling_modifer)["render"]
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        network_gui.send(net_image_bytes, dataset.source_path)
                        if do_training:
                            break
                    except Exception as e:
                        print(e)
                        network_gui.conn = None
                
                try:
                    if epoch > 500:
                        in_im = render(viewpoint_cam, unflattenGaussians(data), pipe, background)["render"]
                        #prediction = torch.clamp(prediction, data.min(), data.max())
                        out_im = render(viewpoint_cam, unflattenGaussians(prediction), pipe, background)["render"]

                        l1_i = torch.nn.functional.l1_loss(out_im, in_im)
                        l1_g = loss_fn(prediction, data)
                        s_los = 1-ssim(in_im, out_im)
                        l_pips = loss_perceptive(torch.clamp(in_im, 0, 1)*2-1, torch.clamp(out_im, 0, 1)*2-1)
                        #loss = l1_g
                        #loss = l1_i*0.8+s_los*0.2
                        #loss = l1_i
                        loss = l1_i*0.6+s_los*0.2+l_pips*0.2
                    else:
                        loss = loss_fn(prediction, data)
                    loss.backward()
                except RuntimeError as e:
                    print(e)
                model_opt.step()
                model_opt.zero_grad()
                with torch.no_grad():
                    writer.add_scalar("loss", loss.data.item(), step)
                    writer.add_scalar("lr", 0.0000001*lrm*100, step)
                    if epoch > 500:
                        writer.add_scalar("l1", l1_i.data.item(), step)
                        writer.add_scalar("ls", s_los.data.item(), step)
                        writer.add_image("in", in_im, step)
                        writer.add_image("out", out_im, step)
                    step += 1
                    # if loss.data.item() < lowest_loss:
                    #     lowest_loss = loss.data.item()
                    #     torch.save(model.state_dict(), "best_autoencoder.pt")

            
        # All done
        print("\nTraining complete.")
