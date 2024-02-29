import math
import torch
from gaussian_renderer import network_gui
from src.gaussians import GaussianModel, render
from src.vis_embed import train, load
from src.dvae import DVAEConfig, DiscreteVAE
from scene import Scene
from chamfer_distance import chamfer_distance as chd
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import CosineLRScheduler
import time
import numpy as np

class AverageMeter(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]
        
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

chamfer_dist = chd.ChamferDistance()
def chamferL1(x, y):
    dist1, dist2, _, _ = chamfer_dist(x, y)
    dist1, dist2 = torch.sqrt(dist1), torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2))/2

def chamferL2(x, y):
    dist1, dist2, _, _ = chamfer_dist(x, y)
    return torch.mean(dist1) + torch.mean(dist2)

def compute_loss(loss_1, loss_2, niter, train_writer):
    '''
    compute the final loss for optimization
    For dVAE: loss_1 : reconstruction loss, loss_2 : kld loss
    '''
    start = 0
    target = 0.1
    ntime = 100000

    _niter = niter - 10000
    if _niter > ntime:
        kld_weight = target
    elif _niter < 0:
        kld_weight = 0.
    else:
        kld_weight = target + (start - target) *  (1. + math.cos(math.pi * float(_niter) / ntime)) / 2.

    if train_writer is not None:
        train_writer.add_scalar('Loss/Batch/KLD_Weight', kld_weight, niter)

    loss = loss_1 + kld_weight * loss_2

    return loss

def get_temp(niter):
    start = 1
    target = 0.0625
    ntime = 100000
    if niter > ntime:
        return target
    else:
        temp = target + (start - target) *  (1. + math.cos(math.pi * float(niter) / ntime)) / 2.
        return temp

params = ModelParams("C:/Users/anw/repos/gaussian-transformer/tiramisu_ds", "C:/Users/anw/repos/gaussian-transformer/output/tiramisu_gs")
gaussians = GaussianModel(torch.empty((0)), torch.empty((0)), torch.empty((0)), torch.empty((0)), torch.empty((0)))
scene = Scene(params, gaussians, -1)
bg = torch.Tensor([0,0,0]).cuda()
pipe = Pipe()

#train(, )
model = load()
v_embed, target_pts = model.encode(gaussians)
target_pts = torch.unsqueeze(target_pts, 0)
config = DVAEConfig(128, 2048)
dvae = DiscreteVAE(config).cuda()
train_writer = SummaryWriter("DVAE/")
# parameter setting
start_epoch = 0
best_metrics = None
metrics = None
network_gui.init("127.0.0.1", 6009)

optimizer = torch.optim.AdamW(dvae.parameters(), lr=0.0005, weight_decay=0.0005)
scheduler = CosineLRScheduler(optimizer,
                t_initial=300,
                t_mul=1,
                lr_min=1e-6,
                decay_rate=0.1,
                warmup_lr_init=1e-6,
                warmup_t=10,
                cycle_limit=1,
                t_in_epochs=True)

# Criterion
ChamferDisL1 = chamferL1
ChamferDisL2 = chamferL2

dvae.zero_grad()
for epoch in range(start_epoch, 300 + 1):
    dvae.train()
    epoch_start_time = time.time()
    batch_start_time = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(['Loss1', 'Loss2'])
    num_iter = 0
    dvae.train()  # set model to training mode
    n_batches = 100
    for idx in range(n_batches):
        num_iter += 1
        n_itr = epoch * n_batches + idx
        
        data_time.update(time.time() - batch_start_time)
        
        temp = get_temp(n_itr)
        ret = dvae(target_pts, v_embed, temperature = temp, hard = False)
        whole_coarse, whole_fine, coarse, fine, neighborhood, logits = ret
        loss_1, loss_2 = dvae.get_loss(ret, target_pts)
        _loss = compute_loss(loss_1, loss_2, n_itr, train_writer)
        _loss.backward()
        # forward
        if num_iter == 1:
            num_iter = 0
            optimizer.step()
            dvae.zero_grad()
        losses.update([loss_1.item() * 1000, loss_2.item() * 1000])

        if train_writer is not None:
            train_writer.add_scalar('Loss/Batch/Loss_1', loss_1.item() * 1000, n_itr)
            train_writer.add_scalar('Loss/Batch/Loss_2', loss_2.item() * 1000, n_itr)
            train_writer.add_scalar('Loss/Batch/Temperature', temp, n_itr)
            train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
        scheduler.step(epoch)
    epoch_end_time = time.time()
    if train_writer is not None:
        train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)
        torch.save(dvae.state_dict(), "dvae.pt")
        with torch.no_grad():
            d1, d2, idx1, idx2 = chamfer_dist(target_pts, whole_fine)
            pred_pts = whole_fine[:, idx1[0]]
            pred_gaussians = model.decode(v_embed, torch.squeeze(pred_pts))
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, pred_gaussians, pipe, bg, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, "C:/Users/anw/repos/gaussian-transformer/tiramisu_ds")
                except Exception as e:
                    print(e)
                    network_gui.conn = None
            i = 0
            for cam in scene.getTrainCameras():
                pred_image = render(cam, pred_gaussians, pipe, bg)["render"]
                train_writer.add_image(f"pred_{i}", pred_image, epoch)
                if i > 5:
                    break
                i += 1



    
