import torch
import torch.nn as nn
import torch.nn.functional as F
from chamfer_distance import chamfer_distance as chd
from timm.models.layers import DropPath, trunc_normal_

from Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from .misc import fps

chamfer_dist = chd.ChamferDistance()

def chamferL1(x, y):
    dist1, dist2, _, _ = chamfer_dist(x, y)
    dist1, dist2 = torch.sqrt(dist1), torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2))/2

def chamferL2(x, y):
    dist1, dist2, _, _ = chamfer_dist(x, y)
    return torch.mean(dist1) + torch.mean(dist2)
    
class DGCNN(nn.Module):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(encoder_channel, 128, 1) 

        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 256),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 1024),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer5 = nn.Sequential(nn.Conv1d(2304, output_channel, kernel_size=1, bias=False),
                                nn.GroupNorm(4, output_channel),
                                nn.LeakyReLU(negative_slope=0.2)
                                )

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            coor_k = coor_k.transpose(1, 2).contiguous()
            coor_q = coor_q.transpose(1, 2).contiguous()
            idx = knn_point(k, coor_k, coor_q).transpose(1, 2).contiguous()  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, f, coor):
        # f: B G C
        # coor: B G 3 

        # bs 3 N   bs C N
        feature_list  = []
        coor = coor.transpose(1, 2).contiguous()         # B 3 N
        f = f.transpose(1, 2).contiguous()               # B C N
        f = self.input_trans(f)             # B 128 N

        f = self.get_graph_feature(coor, f, coor, f) # B 256 N k
        f = self.layer1(f)                           # B 256 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 256 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 512 N k
        f = self.layer2(f)                           # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 1024 N k
        f = self.layer3(f)                           # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 1024 N k
        f = self.layer4(f)                           # B 1024 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 1024 N
        feature_list.append(f)

        f = torch.cat(feature_list, dim = 1)         # B 2304 N

        f = self.layer5(f)                           # B C' N
        
        f = f.transpose(-1, -2)

        return f

### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    # missing = []
    # has = []
    # for i in range(xyz.shape[1]):
    #     if i in group_idx:
    #         has.append(i)
    #     else:
    #         missing.append(i)
    # print(missing)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    

def rps(data, number):
    rps_idx = torch.unsqueeze(torch.randperm(data.shape[1], dtype=torch.int32)[:number],0).cuda()
    rps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), rps_idx).transpose(1,2).contiguous()
    return rps_data

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, emb, normalize = True):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        #center = fps(xyz, self.num_group) # B G 3
        # random the centers out
        center = rps(xyz, self.num_group)
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        emb = emb.view(batch_size * num_points, -1)[idx, :]
        emb = emb.view(batch_size, self.num_group, self.group_size, 128).contiguous()
        # normalize
        if normalize:
            neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, emb

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv_f = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.first_conv_s = nn.Conv1d(128, 256, 1)
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups, emb_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        emb_groups = emb_groups.reshape(bs*g, n, 128).transpose(2, 1)
        # encoder
        feature = self.first_conv_f(point_groups.transpose(2,1))  # BG 256 n
        feature_we = feature + emb_groups
        feature = self.first_conv_s(feature_we)
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Decoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4
        assert num_fine % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv1d(512, 3, 1)
        self.emb_head = nn.Conv1d(512, 128, 1)
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2) # 1 2 S


    def forward(self, feature_global):
        '''
            feature_global : B G C
            -------
            coarse : B G M 3
            fine : B G N 3
        
        '''
        bs, g, c = feature_global.shape
        feature_global = feature_global.reshape(bs * g, c)

        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3) # BG M 3

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1) # BG 2 M (S)
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)  # BG 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine) # BG 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # BG C N
    
        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3
        center = center.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N

        out = self.final_conv(feat)
        fine = self.xyz_head(out) + center   # BG 3 N
        fine = fine.reshape(bs, g, 3, self.num_fine).transpose(-1, -2)
        fine_emb = self.emb_head(out)
        fine_emb = fine_emb.reshape(bs, g, 128, self.num_fine).transpose(-1, -2)
        coarse = coarse.reshape(bs, g, self.num_coarse, 3)
        return coarse, fine, fine_emb

class DVAEConfig():
    def __init__(self, group_size = 512, num_group = 4096, encoder_dims = 256, num_tokens=8192, tokens_dims=256, decoder_dims=256) -> None:
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims
        self.num_tokens = num_tokens
        self.tokens_dims = tokens_dims
        self.decoder_dims = decoder_dims   

class DiscreteVAE(nn.Module):
    def __init__(self, config : DVAEConfig = DVAEConfig(), **kwargs):
        super().__init__()
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.tokens_dims = config.tokens_dims

        self.decoder_dims = config.decoder_dims
        self.num_tokens = config.num_tokens

        
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.dgcnn_1 = DGCNN(encoder_channel = self.encoder_dims, output_channel = self.num_tokens)
        self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims))

        self.dgcnn_2 = DGCNN(encoder_channel = self.tokens_dims, output_channel = self.decoder_dims)
        self.decoder = Decoder(encoder_channel = self.decoder_dims, num_fine = self.group_size)
        self.build_loss_func()

        
        
    def build_loss_func(self):
        self.loss_func_cdl1 = chamferL1
        self.loss_func_cdl2 = chamferL2

    def recon_loss(self, ret):
        _, _, coarse, fine, _, _, group_gt, _ = ret

        bs, g, _, _ = coarse.shape

        coarse = coarse.reshape(bs*g, -1, 3).contiguous()
        fine = fine.reshape(bs*g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs*g, -1, 3).contiguous()

        loss_coarse_block = self.loss_func_cdl1(coarse, group_gt)
        loss_fine_block = self.loss_func_cdl1(fine, group_gt)

        loss_recon = loss_coarse_block + loss_fine_block

        return loss_recon
    
    def emb_loss(self, ret):
        _, _, _, _, fine_emb, n_emb, _, _ = ret
        return torch.nn.functional.mse_loss(fine_emb, n_emb)

    def get_loss(self, ret):

        # reconstruction loss
        loss_recon = self.recon_loss(ret)
        loss_emb = self.emb_loss(ret)
        # kl divergence
        logits = ret[-1] # B G N
        softmax = F.softmax(logits, dim=-1)
        mean_softmax = softmax.mean(dim=1)
        log_qy = torch.log(mean_softmax)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device = "cuda"))
        loss_klv = F.kl_div(log_qy, log_uniform.expand(log_qy.size(0), log_qy.size(1)), None, None, 'batchmean', log_target = True)

        return loss_recon, loss_klv, loss_emb

    def encode(self, neighborhood, center, n_emb):
        
        logits = self.encoder(neighborhood, n_emb)   #  B G C
        logits = self.dgcnn_1(logits, center)
        return logits, center, neighborhood, n_emb

    def decode(self, tokens, center):
        feature = self.dgcnn_2(tokens, center)
        coarse, fine, fine_emb = self.decoder(feature)
        return coarse, fine, fine_emb

    def forward(self, neighborhood, center, emb, temperature = 1., hard = False, **kwargs):
        #  B G N
        logits, center, neighborhood, n_emb = self.encode(neighborhood, center, emb)
        soft_one_hot = F.gumbel_softmax(logits, tau = temperature, dim = 2, hard = hard) # B G N
        sampled = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.codebook) # B G C
        coarse, fine, fine_emb = self.decode(sampled, center)

        with torch.no_grad():
            whole_fine = (fine + center.unsqueeze(2)).reshape(1, -1, 3)
            whole_coarse = (coarse + center.unsqueeze(2)).reshape(1, -1, 3)

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, fine_emb, n_emb, neighborhood, logits)
        return ret