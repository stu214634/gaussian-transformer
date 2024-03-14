import copy
import math
import random
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussians import GaussianModel, render
from gaussian_renderer import network_gui
from scene.cameras import Camera
from .vis_embed import VisEmbedNet
from .dvae import DiscreteVAE, DVAEConfig, Group
from timm.models.layers import DropPath,trunc_normal_
import os
from chamfer_distance import chamfer_distance as chd
from torch.utils.tensorboard import SummaryWriter
N_QUERY = 4096//2

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(dim*2, dim)

    def forward(self, q, v, cross_knn_index=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)
        
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)
        q = q + self.drop_path(q_2)

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
#https://github.com/yuxumin/PoinTr/blob/master/models/Transformer.py
class PCTransformer(nn.Module):
    def __init__(self, embed_dim=256, depth=[6, 6], num_heads=8, mlp_ratio=2.,
                  qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                    num_query = N_QUERY, dvae_path = "dvae.pt", tokenizer_path="dvae_emb.pt"):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        cfg = DVAEConfig(64, N_QUERY, 256, 8192, 256, 256)
        self.dvae = DiscreteVAE(cfg).cuda()
        self.dvae.load_state_dict(torch.load(dvae_path))
        self.vis_embed = VisEmbedNet().cuda()
        self.vis_embed.load_state_dict(torch.load(tokenizer_path))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.vocab = self.dvae.codebook

        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        ) 
        self.num_query = num_query
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query)
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1)
        )

        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])
        
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[1])])
        
        self.dict_increase = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 8192)
        ) 
        
    #def tokenize(self, g : GaussianModel):
    #    vis_embed, pts = self.vis_embed.encode(g)
    #    return self.dvae.encode(torch.unsqueeze(pts, 0), torch.unsqueeze(vis_embed, 0))
    
    def detokenize(self, tokens, center):
        _, fine, fine_emb = self.dvae.decode(tokens, center)
        fine_whole = (fine + center.unsqueeze(2)).reshape(-1, 3)
        return self.vis_embed.decode(fine_emb.reshape(-1, 128), fine_whole)
    
    def forward(self, neighborhood, center, n_emb):
        logits, center, neighborhood, n_emb = self.dvae.encode(neighborhood, center, n_emb)
        pos = self.pos_embed(center.detach())
        one_hot = F.gumbel_softmax(logits, 1, dim=-1)
        tokens = torch.einsum('b g n, n c -> b g c', one_hot, self.vocab).detach()
        mem = tokens
        for module in self.encoder:
            mem = module(mem + pos)

        global_feature = self.increase_dim(mem.transpose(1,2)) # B 1024 N 
        global_feature = torch.max(global_feature, dim=-1)[0] # B 1024

        coarse_point_cloud = self.coarse_pred(global_feature).reshape(1, -1, 3)
        query_feature = torch.cat([
        global_feature.unsqueeze(1).expand(-1, self.num_query, -1), 
        coarse_point_cloud], dim=-1) # B M C+3 
        q = self.mlp_query(query_feature.transpose(1,2)).transpose(1,2) # B M C
        for blk in self.decoder:
            q = blk(q, mem)
        
        pred_logits = F.gumbel_softmax(self.dict_increase(q), tau=1, dim=-1)
        pred_tokens = torch.einsum('b g n, n c -> b g c', pred_logits, self.vocab)

        gaussians = self.detokenize(pred_tokens, coarse_point_cloud)
        return coarse_point_cloud, pred_logits, pred_tokens, gaussians

class Pipe():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

def train(gaussians : GaussianModel, cameras : List[Camera], path="GaussianTr.pt"):
    transformer = PCTransformer().cuda()
    if os.path.exists(path):
        print(f"Loading from: {path}")
        transformer.load_state_dict(torch.load(path))
    else:
        print("Initializing from scratch")
    opt = torch.optim.AdamW(transformer.parameters(), 0.0001)

    network_gui.init("127.0.0.1", 6009)
    pipe = Pipe()
    bg = torch.Tensor([0,0,0]).cuda()
    chamferDist = chd.ChamferDistance()
    train_writer = SummaryWriter("GaussianTr/Full_Fill")
    group_div = Group(N_QUERY//2, 64)
    gaussians.box_sort(40)
    gaussians_count = gaussians.xyz.shape[0]
    to_predict = gaussians_count//2
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, cooldown=10)
    #lr = scheduler.optimizer.param_groups[0]['lr']
    step = 0
    for epoch in range(0, 100_000, 1):


        for cam in cameras:
        #t_pts = pts.squeeze()[mask, :]
        #t_pts = t_pts.view(to_predict, 3).contiguous()
        #t_emb = vis_embed.squeeze()[mask, :]
        #t_emb = t_emb.view(to_predict, 128).contiguous()
            with torch.no_grad():
                mask = torch.ones(gaussians_count, dtype=torch.bool)
                start = torch.randint(gaussians_count - to_predict, (1,))[0]
                mask[start:start+to_predict] = False
                to_predict = mask[mask].shape[0]
                idx = torch.nonzero(mask).squeeze()
                input_gaussians = gaussians.get_idx(idx)
                t_idx = torch.nonzero(~mask).squeeze()
                target_gaussians = gaussians.get_idx(t_idx)
                if random.random() > 0.8:
                    target_gaussians, input_gaussians = input_gaussians, target_gaussians

            i_vis_embed, i_pts = transformer.vis_embed.encode(input_gaussians)

            n_in, c_in, n_emb_in = group_div(i_pts.unsqueeze(0), i_vis_embed.unsqueeze(0))

            #n_true = neighborhood.detach()
            #c_true = center.detach()
            #n_emb_true = emb.detach()

            _, _, _, pred_gaussians = transformer.forward(n_in, c_in, n_emb_in)
            d1, d2, _, idx2 = chamferDist(target_gaussians.xyz.unsqueeze(0), pred_gaussians.xyz.unsqueeze(0))
            xyz_loss = torch.mean(d1)*0.5 + torch.mean(d2)*0.5
            with torch.no_grad():
                tgt_gaussians = target_gaussians.get_idx(idx2.squeeze())
            scaling_loss = F.mse_loss(tgt_gaussians.scaling, pred_gaussians.scaling)
            rotation_loss = F.mse_loss(tgt_gaussians.rotation, pred_gaussians.rotation)
            shs_loss = F.mse_loss(tgt_gaussians.shs, pred_gaussians.shs)
            opacity_loss = F.mse_loss(tgt_gaussians.opacity, pred_gaussians.opacity)
            other_loss = xyz_loss + scaling_loss + rotation_loss + shs_loss + opacity_loss
            im_loss = torch.scalar_tensor(0, dtype=torch.float32).cuda()
            loss = other_loss
            if other_loss <= 1:
                pred_im = render(cam, pred_gaussians, pipe, bg, 1)["render"]
                target_im = render(cam, target_gaussians, pipe, bg, 1)["render"].detach()
                im_loss += F.l1_loss(pred_im, target_im)
                loss += im_loss
            loss.backward()
            opt.step()
            opt.zero_grad()

            with torch.no_grad():
                train_writer.add_scalar("loss", loss.data.item(), step)
                train_writer.add_scalar("xyz_loss", xyz_loss.data.item(), step)
                train_writer.add_scalars("mse_losses", {"scaling_loss": scaling_loss.data.item(),
                                        "rotation_loss": rotation_loss.data.item(),
                                        "shs_loss": shs_loss.data.item(),
                                        "opacity_loss": opacity_loss.data.item()}, step)
                train_writer.add_scalar("im_loss", im_loss.data.item(), step)
                step += 1
                first = True
                if network_gui.conn == None:
                    network_gui.try_connect()
                while network_gui.conn != None:
                    if first:
                        xyz = torch.cat([input_gaussians.xyz, pred_gaussians.xyz], 0)
                        rotation = torch.cat([input_gaussians.rotation, pred_gaussians.rotation], 0)
                        shs = torch.cat([input_gaussians.shs, pred_gaussians.shs], 0)
                        opacity = torch.cat([input_gaussians.opacity, pred_gaussians.opacity], 0)
                        scaling = torch.cat([input_gaussians.scaling, pred_gaussians.scaling], 0)
                        g_render_both = GaussianModel(xyz, opacity, rotation, scaling, shs)
                        first = False
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, show_in, show_true, keep_alive, scaling_modifer = network_gui.receive()
                        if custom_cam != None:
                            if show_in:
                                if show_true:
                                    net_image = render(custom_cam, g_render_both, pipe, bg, scaling_modifer)["render"]
                                else:
                                    net_image = render(custom_cam, input_gaussians, pipe, bg, scaling_modifer)["render"]
                            elif show_true:
                                net_image = render(custom_cam, target_gaussians, pipe, bg, scaling_modifer)["render"]
                            else:
                                net_image = render(custom_cam, pred_gaussians, pipe, bg, scaling_modifer)["render"]   
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        network_gui.send(net_image_bytes, "C:/Users/anw/repos/gaussian-transformer/tiramisu_ds")
                        if not do_training:
                            break
                    except Exception as e:
                        print(e)
                        network_gui.conn = None

        torch.save(transformer.state_dict(), "GaussianTr.pt")

        if epoch % 20 == 0:
            target_im = torch.empty((len(cameras)//3 + 1, *cameras[0].original_image.shape)).cuda()
            pred_im = torch.empty_like(target_im)
            input_im = torch.empty_like(target_im)
            for i, cam in enumerate(cameras[::3]):
                pred_im[i] = render(cam, pred_gaussians, pipe, bg, 1)["render"]
                target_im[i] = render(cam, target_gaussians, pipe, bg, 1)["render"]
                input_im[i] = render(cam, input_gaussians, pipe, bg, 1)["render"]
            train_writer.add_images("in", input_im, epoch)
            train_writer.add_images("pred", pred_im, epoch)
            train_writer.add_images("target", target_im, epoch)
                    



def test(gaussians : GaussianModel):
    transformer = PCTransformer()
    network_gui.init("127.0.0.1", 6009)
    pipe = Pipe()
    bg = torch.Tensor([0,0,0]).cuda()
    chamferDist = chd.ChamferDistance()
    train_writer = SummaryWriter("GaussianTr/Test")
    group_div = Group(num_group = 1028, group_size = 64)
    splitAt = int(N_QUERY)
    gaussians.box_sort(5)
    vis_embed, pts = transformer.vis_embed.encode(gaussians)

    mask = torch.ones(pts.shape[0], dtype=torch.bool)
    mask[75_000:125_000] = False
    to_predict = mask[mask].shape[0]
    t_neighborhood = pts.squeeze()[mask, :]
    t_neighborhood = t_neighborhood.view(to_predict, 3).contiguous()
    t_emb = vis_embed.squeeze()[mask, :]
    t_emb = t_emb.view(to_predict, 128).contiguous()
    nidx = ~mask
    print(nidx[nidx].shape[0])
    i_neighborhood = pts.squeeze()[nidx, :]
    i_neighborhood = i_neighborhood.view(pts.shape[0]-to_predict, 3).contiguous()
    i_emb = vis_embed.squeeze()[nidx, :]
    i_emb = i_emb.view(pts.shape[0]-to_predict, 128).contiguous()
    vis_embed = vis_embed.detach()
    pts = pts.detach()
    neighborhood, center, emb = group_div(i_neighborhood.unsqueeze(0), i_emb.unsqueeze(0), normalize=False)
    with torch.no_grad():
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, show_in, show_both, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    ret = transformer.dvae(neighborhood, center, emb, hard=True)
                    whole_coarse, whole_fine, coarse, fine, fine_emb, n_emb, neighborhood, logits = ret
                    g_in = transformer.vis_embed.decode(fine_emb.reshape(-1, 128), whole_fine.reshape(-1, 3))
                    g_render=g_in
                    net_image = render(custom_cam, g_render, pipe, bg, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, "C:/Users/anw/repos/gaussian-transformer/tiramisu_ds")

            except Exception as e:
                print(e)
                network_gui.conn = None



