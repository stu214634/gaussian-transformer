import time
import torch
import sys
import numpy as np
from argparse import ArgumentParser
from torch.autograd import Variable
from arguments import ModelParams, OptimizationParams, PipelineParams
from model.box_sort import GaussianHandler
from model.shared import subsequent_mask
from model.model import make_model, EncoderDecoder
from scene import GaussianModel, Scene
from gaussian_renderer import render, network_gui
from torch.utils.tensorboard import SummaryWriter
from scene.cameras import MiniCam
from utils.system_utils import searchForMaxIteration
import time
import os
import lpips
import math
import datetime
import matplotlib.pyplot as plt
from chamfer_distance import ChamferDistance as chamfer_dist

from utils.loss_utils import ssim

START_GAUSSIAN = torch.zeros(26,dtype=torch.float32)
START_GAUSSIAN[23] = 1
PAD_GAUSSIAN = torch.zeros(26,dtype=torch.float32)
PAD_GAUSSIAN[24] = 1
END_GAUSSIAN = torch.zeros(26,dtype=torch.float32)
END_GAUSSIAN[25] = 1

STACK = 7

def fuzzy_token_equal(gaussian, token):
    return torch.sum(torch.abs(gaussian - token), -1) <= 0.5
    
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

class TrainingScene:
    def __init__(self, args, pipe, batch_size = 4) -> None:
        with torch.no_grad():
            self.source_path = args.source_path
            gaussian_model = GaussianModel(1)
            self.scene = Scene(args, gaussian_model, -1)
            self.handler = GaussianHandler(self.scene.gaussians, 40)
            self.scene.gaussians = self.handler.denormalize(unflattenGaussians(self.handler.box_sort(self.scene.gaussians)))
            self.cameras = self.scene.getTrainCameras()
            pipe.debug = False
            pipe.convert_SHs_python = pipe.compute_cov3D_python = None
            self.pipe = pipe
            self.bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            self.batch_size = batch_size
            self.size = len(self.cameras)
            self.dropout = 0.3

    def train_iter(self):
        #self.dropout = min(1.20-math.exp(-0.0001*epoch), 0.8)
        self.idxs = self.make_indexes()
        for _ in range(len(self.idxs)):
            with torch.no_grad():
                idxs = self.idxs.pop(0)
                tgt_im = torch.zeros((self.batch_size, 3, self.cameras[0].image_height, self.cameras[0].image_width))
                cameras = []
                gaussian_list = flattenGaussians(self.scene.gaussians)
                visibility_filter = torch.zeros((gaussian_list.shape[0]), device="cuda", dtype=torch.bool)
                tokens = 0
                for i in range(self.batch_size):
                    render_pkg = render(self.cameras[idxs[i]], self.scene.gaussians, self.pipe, self.bg)
                    image = render_pkg["render"]
                    visibility_filter |= render_pkg["visibility_filter"]
                    tgt_im[i] = image
                    cameras.append(self.cameras[idxs[i]])
                    tokens += 1
                seen_gaussians = gaussian_list[visibility_filter]
                seen_gaussians = seen_gaussians[:(seen_gaussians.shape[0]//(2**STACK))*(2**STACK)]
                for _ in range(STACK):
                    seen_gaussians = torch.cat([seen_gaussians[0::2], seen_gaussians[1::2]], 1)
                mid = seen_gaussians.shape[0]//2
                low = int(mid - mid * self.dropout)
                high = int(mid + mid * self.dropout)
                tgt_count = high - low
                src_count = seen_gaussians.shape[0] - tgt_count
                src_gaussians = torch.zeros((1, src_count, 26*2**STACK))
                tgt_gaussians = torch.zeros((1, tgt_count, 26*2**STACK))
                src_gaussians[0] = torch.cat([seen_gaussians[:low], seen_gaussians[high:]])
                tgt_gaussians[0] = seen_gaussians[low:high]

            src_mask = (False == fuzzy_token_equal(src_gaussians.unsqueeze(-3), PAD_GAUSSIAN.repeat(2**STACK)))
            tgt_gaussians_y = tgt_gaussians
            tgt_gaussians = tgt_gaussians
            tgt_mask = self.make_std_mask(tgt_gaussians)
            src_mask = src_mask.cuda()
            tgt_mask = tgt_mask.cuda()
            src_gaussians = src_gaussians.cuda()
            tgt_gaussians = tgt_gaussians.cuda()
            tgt_gaussians_y = tgt_gaussians_y.cuda()
            tgt_im = tgt_im.cuda()
            yield {"src" : src_gaussians,
                    "src_mask" : src_mask,
                    "trg" : tgt_gaussians,
                    "trg_y" : tgt_gaussians_y,
                    "trg_mask" : tgt_mask,
                    "trg_im" : tgt_im,
                    "cam" : cameras,
                    "ntokens" : tokens}

    @staticmethod
    def make_std_mask(tgt):
        "Create a mask to hide padding and future words."
        tgt_mask = (False == fuzzy_token_equal(tgt.unsqueeze(-3), PAD_GAUSSIAN.repeat(2**STACK)))
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt_mask.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


    def make_indexes(self) -> list:
        idxs : np.ndarray = np.arange(self.size)
        np.random.shuffle(idxs)
        idxs = idxs[:(self.size // self.batch_size)*self.batch_size]
        return list(idxs.reshape(((self.size // self.batch_size), self.batch_size)))
    

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

def unstack(x : torch.Tensor):
    x = torch.squeeze(x)
    for i in range(STACK):
        x = x.reshape(x.shape[0]*2, x.shape[1]//2)
    return x

class ImageLossCompute:
    def __init__(self, generator, pipe, opt=None):
        self.generator = generator
        self.pipe = pipe
        self.bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.opt = opt
        self.chd = chamfer_dist()
        self.criterion = torch.nn.L1Loss()
        self.loss_perceptive = lpips.LPIPS(net='alex').cuda()
        
    def __call__(self, prompt, x, tgt_im, tgt, cameras):
        global global_step
        x = self.generator(x)
        pred_list = unstack(x)
        tgt_list = unstack(tgt)
        dist1, dist2, idx1, idx2 = self.chd(torch.unsqueeze(pred_list, 0),torch.unsqueeze(tgt_list, 0))

        #Start with Chamfer-Dist
        chamfer = ((torch.mean(dist1)) + (torch.mean(dist2))) / tScene.batch_size

        #Check if Gaussians are reasonably reasonably behaved to protect the renderer from running OOM
        if chamfer < 3:
            prompt_list = unstack(prompt)
            gaussians = unflattenGaussians(torch.cat([prompt_list, pred_list], 0))
            images = torch.empty_like(tgt_im)
            #Render the prediction from all camera angles in batch
            for i, cam in enumerate(cameras):
                render_pkg = render(cam, gaussians, self.pipe, self.bg)
                image = render_pkg["render"]
                images[i] = torch.clamp(image, 0, 1)
            
            #Calcuate Image losses
            gen = self.criterion(images, tgt_im)
            perceptive = self.loss_perceptive(images*2-1, tgt_im*2-1).mean()
            ssim_l = 1 - ssim(images, tgt_im)
            gen *= 5.0/(i+1)
            perceptive *= 1.0/(i+1)
            ssim_l *= 0.2/(i+1)

            loss = gen + perceptive + ssim_l + chamfer
            writer.add_scalar("perceptive_loss", perceptive.data.item(), global_step)
            writer.add_scalar("gen_loss", gen.data.item(), global_step)
            writer.add_scalar("ssim_loss", ssim_l.data.item(), global_step)
            if (global_step % 5 == 0):
                with torch.no_grad():
                    gaussian_model_prompt = unflattenGaussians(prompt_list)
                    prompts = torch.empty_like(tgt_im)
                    for i, cam in enumerate(cameras):
                        render_pkg = render(cam, gaussian_model_prompt, self.pipe, self.bg)
                        image = render_pkg["render"]
                        prompts[i] = torch.clamp(image, 0, 1)
                    writer.add_images("prompt", prompts, global_step)
                    writer.add_images("prompt + x", images, global_step)
                    writer.add_images("base", tgt_im, global_step)
        else:
            #Optimize without Image losses
            loss = chamfer


        writer.add_scalar("loss", loss.data.item(), global_step)
        writer.add_scalar("chamfer", chamfer.data.item(), global_step)

        loss.backward()
        self.opt.step()
        self.opt.optimizer.zero_grad()

        writer.add_scalar("lr", self.opt.rate(), global_step)
        writer.flush()
        global_step += 1
        return loss.data.item()
    
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run_epoch(data_iter, model : EncoderDecoder, loss_compute, iter_size):
    "Standard Training and Logging Function"
    total_tokens = 0
    total_loss = 0
    for i, batch in enumerate(data_iter):
        batch = batch
        out = model.forward(batch["src"], batch["trg"], 
                            batch["src_mask"], batch["trg_mask"])
        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, tScene.pipe.convert_SHs_python, tScene.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    gOut = model.generator(out)
                    net_image = render(custom_cam, unflattenGaussians(unstack(torch.cat([torch.squeeze(batch["src"]), torch.squeeze(gOut)], 0))), tScene.pipe, tScene.bg, scaling_modifer)["render"]
                    #net_image = render(custom_cam, unflattenGaussians(unstack(torch.squeeze(batch["trg_y"]))), tScene.pipe, tScene.bg, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, tScene.source_path)
                if not do_training:
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        image = batch["trg_im"]
        loss = loss_compute(batch["src"], out, image, batch["trg_y"], batch["cam"])
        total_loss += loss
        total_tokens += batch["ntokens"]
    
    writer.add_scalar("dropout", tScene.dropout, epoch)
    loss = total_loss / total_tokens
    return loss

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
    tScene = TrainingScene(lp.extract(args), pp.extract(args))
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

 
    V = 26*2**STACK
    model = make_model(STACK, V, V, N=3, d_model=26*2**STACK,)
    model.cuda()
    model_opt = NoamOpt(V, 0.005, 20,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-15))

    
    
    run_name = "runs/" + datetime.datetime.fromtimestamp(time.time()).strftime('%a_%d_%b_%I_%M%p')
    run_name = "runs/Sun_18_Feb_05_12PM"
    if not os.path.exists(run_name): 
        os.mkdir(run_name)
        max_iter = -1
    else:
        max_iter = searchForMaxIteration(run_name)
        print(f"loading Model iter {max_iter}")
        model.load_state_dict(torch.load(f"{run_name}/checkpoint_{max_iter}/model.pt"))
        model_opt.optimizer.load_state_dict(torch.load(f"{run_name}/checkpoint_{max_iter}/optim.pt"))
    
    writer = SummaryWriter(f'logs/{run_name}/base/')

    global_step = 190
    model_opt._step = global_step
    model.train()
    lowest_loss = 1e9
    loss_func = ImageLossCompute(model.generator, pp.extract(args), model_opt)
    for epoch in range(max_iter + 1, 20000, 1):
        loss = run_epoch(tScene.train_iter(), model, loss_func, tScene.size)
        print(f"Epoch: {epoch} Loss: {loss}")
        model_opt.optimizer.zero_grad()
        if epoch % 20 == 0:
            dir = f"{run_name}/checkpoint_{epoch}"
            os.mkdir(dir)
            torch.save(model.state_dict(), f"{dir}/model.pt")
            torch.save(model_opt.optimizer.state_dict(), f"{dir}/optim.pt")
        
    # All done
    print("\nTraining complete.")
