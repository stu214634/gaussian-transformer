import time
import torch
import sys
import numpy as np
from argparse import ArgumentParser
from torch.autograd import Variable
from arguments import ModelParams, OptimizationParams, PipelineParams
from model.shared import subsequent_mask
from model.model import make_model, EncoderDecoder
from scene import GaussianModel, Scene
from gaussian_renderer import render, network_gui
from torch.utils.tensorboard import SummaryWriter
import time
import os
import lpips
import math

START_GAUSSIAN = torch.zeros(26,dtype=torch.float32)
START_GAUSSIAN[23] = 1
PAD_GAUSSIAN = torch.zeros(26,dtype=torch.float32)
PAD_GAUSSIAN[24] = 1
END_GAUSSIAN = torch.zeros(26,dtype=torch.float32)
END_GAUSSIAN[25] = 1

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
    def __init__(self, args, pipe, batch_size = 1, max_len=15_000) -> None:
        self.max_len = max_len
        self.gaussian_model = GaussianModel(1)
        self.scene = Scene(args, self.gaussian_model, -1)
        self.cameras = self.scene.getTrainCameras()
        pipe.debug = False
        pipe.convert_SHs_python = pipe.compute_cov3D_python = None
        self.pipe = pipe
        self.bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.visible_gaussians = []
        self.prep_cameras()
        self.batch_size = batch_size
        self.size = len(self.cameras) // batch_size
        self.dropout = 0.01

    def train_iter(self):
        self.dropout = 1.05-math.exp(-0.0005*epoch)
        self.idxs = self.make_indexes()
        for _ in range(len(self.idxs)):
            idxs = self.idxs.pop(0)
            tgt_im = torch.zeros((self.batch_size, 3, self.cameras[0].image_height, self.cameras[0].image_width))
            cameras = []

            gaussian_list = flattenGaussians(self.gaussian_model)
            tokens = 0
            for i in range(self.batch_size):
                render_pkg = render(self.cameras[idxs[i]], self.gaussian_model, self.pipe, self.bg)
                image, visibility_filter = render_pkg["render"], render_pkg["visibility_filter"]
                seen_gaussians = gaussian_list[visibility_filter]
                mask = torch.rand(seen_gaussians.shape[0]) >= self.dropout
                src_count = mask[mask].shape[0]
                tgt_count = mask.shape[0] - src_count
                src_gaussians = torch.zeros((self.batch_size, src_count+2, 26))
                tgt_gaussians = torch.zeros((self.batch_size, tgt_count+2, 26))
                tgt_gaussians[:, 0] = src_gaussians[:, 0] = START_GAUSSIAN
                src_gaussians[i, 1:(src_count+1)] = seen_gaussians[mask]
                tgt_gaussians[i, 1:(tgt_count+1)] = seen_gaussians[mask == False]
                tgt_gaussians[i, tgt_count+1] = END_GAUSSIAN
                tgt_im[i] = image
                cameras.append(self.cameras[idxs[i]])
                tokens += 1
            src_mask = (False == fuzzy_token_equal(src_gaussians.unsqueeze(-3), PAD_GAUSSIAN))
            tgt_gaussians_y = tgt_gaussians[:, 1:]
            tgt_gaussians = tgt_gaussians[:, :-1]
            tgt_mask = self.make_std_mask(tgt_gaussians)
            src_mask = src_mask.cuda().half()
            tgt_mask = tgt_mask.cuda().half()
            src_gaussians = src_gaussians.cuda().half()
            tgt_gaussians = tgt_gaussians.cuda().half()
            tgt_gaussians_y = tgt_gaussians_y.cuda().half()
            tgt_im = tgt_im.cuda().half()
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
        tgt_mask = (False == fuzzy_token_equal(tgt.unsqueeze(-3), PAD_GAUSSIAN))
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt_mask.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
    def prep_cameras(self):
        too_large = []
        for i in range(len(self.cameras)):
            render_pkg = render(self.cameras[i], self.gaussian_model, self.pipe, self.bg)
            _, _, visibility_filter, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            count = len(visibility_filter[visibility_filter])
            if count+1 >= self.max_len or count <= 5000:
                too_large.append(i)
                continue
            self.visible_gaussians.append(count)
        for idx in sorted(too_large, reverse=True):
            del self.cameras[idx]
        self.visible_gaussians = torch.as_tensor(self.visible_gaussians)


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
    

class ImageLossCompute:
    def __init__(self, generator, pipe, opt=None):
        self.generator = generator
        self.criterion = torch.nn.L1Loss()
        self.loss_perceptive = lpips.LPIPS(net='alex').cuda()
        self.pipe = pipe
        self.bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.opt = opt
        
    def __call__(self, prompt, x, y, tgt, camera):
        global global_step
        x = torch.squeeze(self.generator(x)).float()
        prompt = prompt.float()
        y = y.float()
        gaussian_model = unflattenGaussians(torch.cat([prompt, x],  0))
        render_pkg = render(camera, gaussian_model, self.pipe, self.bg)
        image = render_pkg["render"]
        render_pkg["visibility_filter"]

        original_image = camera.original_image.cuda()
        base = self.criterion(torch.squeeze(y), original_image)
        gen = self.criterion(image, original_image)

        perceptive = self.loss_perceptive(torch.unsqueeze((image*2)-1,0), torch.unsqueeze((original_image*2)-1,0))
        l2 = torch.nn.functional.mse_loss(x , torch.squeeze(tgt).float())

        loss = ((base - (base - gen)) / base)*0.5
        loss += 0.1*l2
        loss += 0.4*torch.squeeze(perceptive)

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        writer.add_scalars("base vs. prompt vs. gen", {"base" : base.data.item(), "gen" : gen.data.item()}, global_step)
        writer.add_scalar("l2_loss", l2.data.item(), global_step)
        writer.add_scalar("perceptive_loss", perceptive.data.item(), global_step)
        writer.add_scalar("loss", loss.data.item(), global_step)
        writer.add_scalar("lr", self.opt.rate(), global_step)
        global_step += 1
        if (global_step % 5 == 0):    
            gaussian_model_prompt = unflattenGaussians(prompt)
            render_pkg_prompt = render(camera, gaussian_model_prompt, self.pipe, self.bg)
            image_prompt = render_pkg_prompt["render"]
            writer.add_image("prompt", image_prompt, global_step)
            writer.add_image("prompt + x", image, global_step)
            writer.add_image("base", torch.squeeze(y), global_step)
            writer.add_image("target", original_image, global_step)
            writer.add_image("diff", torch.abs(image_prompt - image), global_step)
        writer.flush()
        return loss.data.item()
    
def closestFuture(pred, tgt):
    near = torch.abs(tgt - pred)
    close = torch.sum(near, dim=-1)
    closest, closest_idx = torch.min(close, dim=-1)
    return closest, closest_idx

def closestFuturesLoss(pred, tgt):
    tgtReorder = torch.empty_like(tgt)
    for i in range(pred.shape[0]):
        idxs = torch.empty(pred.shape[1], dtype=torch.long).cuda()
        idxs = torch.fill(idxs, tgt.shape[1] + 2)
        rPred, rTgt = pred[i], tgt[i]
        mask = torch.ones(rPred.shape[:-1], dtype=torch.bool)
        for j in range(pred.shape[1]):
            _, idx = closestFuture(rPred[j:j+1], rTgt[mask].reshape(-1, rPred.shape[1]))
            idx += len(idxs[idxs <= idx])
            while not mask[idx]:
                idx += 1
            idxs[j] = idx
            mask[idx] = False
            
        tgtReorder[i] = tgt[i][idxs]
    return torch.nn.functional.l1_loss(pred, tgtReorder)

        
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
        image = batch["trg_im"]
        loss = loss_compute(torch.squeeze(batch["src"]), out, image, batch["trg_y"], batch["cam"][0])
        total_loss += loss
        total_tokens += batch["ntokens"]
    
    writer.add_scalar("dropout", tScene.dropout, epoch)
    loss = total_loss / total_tokens
    return loss

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = start_symbol.unsqueeze(0).unsqueeze(0)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        next_word = prob.data[0]
        ys = torch.cat([ys, 
                        next_word.unsqueeze(0).unsqueeze(0)], dim=1)
    return ys


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

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    tScene = TrainingScene(lp.extract(args), pp.extract(args))



    V = 26
    model = make_model(V, V, N=6, d_model=1024)
    model.half()
    model.cuda()
    model_opt = NoamOpt(model.src_embed[0].d_model, 0.5, 2000,
            torch.optim.Adamax(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-4))

    
    if os.path.exists("best_model.pt"):
        print("Loading Model")
        model.load_state_dict(torch.load("best_model.pt"))
    
    writer = SummaryWriter('runs/gaussian_trainer_embed')
    global_step = 0
    model_opt._step = global_step
    model.train()
    lowest_loss = 1e9
    loss_func = ImageLossCompute(model.generator, pp.extract(args), model_opt)
    for epoch in range(0, 20000, 1):
        loss = run_epoch(tScene.train_iter(), model, loss_func, tScene.size)
        print(f"Epoch: {epoch} Loss: {loss}")
        if loss < lowest_loss:
            lowest_loss = loss
            torch.save(model.state_dict(), "best_model.pt")
        
    # All done
    print("\nTraining complete.")
