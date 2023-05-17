from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 num_levels=4,
                 base_resolution=16,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, num_levels=num_levels, base_resolution=base_resolution, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params


class NeRFMultiRes(nn.Module):
    """
    Multi-resolution Radiance Fields
    """
    def __init__(self, reso_num, model_kwargs) -> None:
        super().__init__()
        self.reso_num = reso_num
        self.bg_radius = model_kwargs['bg_radius']
        self.cuda_ray = model_kwargs['cuda_ray']
        # Initialize NeRF models for each dimension
        models = [NeRFNetwork(num_levels=4, base_resolution=16**(i+1), **model_kwargs) for i in range(reso_num)]
        self.models: list[NeRFNetwork] = nn.ModuleList(models)

    def forward(self, x, d):
        """
        @ param x: [N, 3], in [-bound, bound]
        @ param d: [N, 3], nomalized in [-1, 1]
        @ return sigma: 
        @ return color:
        """
        sigmas = []
        colors = []
        for i in range(self.reso_num):
            sigma, color = self.models[i](x, d)
            sigmas.append(sigma)
            colors.append(color)

        sigmas = torch.sum(torch.stack(sigmas, dim=0), dim=0)
        colors = torch.sum(torch.stack(colors, dim=0), dim=0)

        return sigmas, colors
    
    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        sigmas = []
        geo_feats = []
        for i in range(self.reso_num):
            sigma, geo_feat = self.models[i].density(x)
            sigmas.append(sigma)
            geo_feats.append(geo_feat)

        sigmas = torch.sum(torch.stack(sigmas, dim=0), dim=0)
        geo_feats = torch.sum(torch.stack(geo_feats, dim=0), dim=0)

        return {
            'sigma': sigmas,
            'geo_feat': geo_feats,
        }

    # optimizer utils
    def get_params(self, lr):

        params = []

        for i in range(self.reso_num):
            param = [
                {'params': self.models[i].encoder.parameters(), 'lr': lr},
                {'params': self.models[i].sigma_net.parameters(), 'lr': lr},
                {'params': self.models[i].encoder_dir.parameters(), 'lr': lr},
                {'params': self.models[i].color_net.parameters(), 'lr': lr},
            ]
            params += param
        
            if self.bg_radius > 0:
                params.append({'params': self.models[i].encoder_bg.parameters(), 'lr': lr})
                params.append({'params': self.models[i].bg_net.parameters(), 'lr': lr})
        
        return params
    
    # NeRFRenderer methods
    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, reso='all', **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]
        if reso == 'all':
            # Render at combined resolution
            results = {"depth": [], "image": []}
            for i in range(self.reso_num):
                result = self.models[i].render(rays_o, rays_d, staged=staged, max_ray_batch=max_ray_batch, **kwargs)
                results['depth'].append(result['depth'])
                results['image'].append(result['image'])

            results['depth'] = torch.sum(torch.stack(results['depth'], dim=0), dim=0)
            results['image'] = torch.sum(torch.stack(results['image'], dim=0), dim=0)
        
        elif isinstance(reso, int) and reso >= 0 and reso < self.reso_num:
            # Render at certain resolution
            results = self.models[reso].render(rays_o, rays_d, staged=staged, max_ray_batch=max_ray_batch, **kwargs)

        else:
            raise RuntimeError("Illegal resolution in rendering:", reso)

        return results
    
    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]
        for i in range(self.reso_num):
            self.models[i].mark_untrained_grid(poses=poses, intrinsic=intrinsic, S=S)

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.
        for i in range(self.reso_num):
            self.models[i].update_extra_state(decay=decay, S=S)
    
    def mean_count_get(self) -> int:
        mean_count = 0
        for i in range(self.reso_num):
            mean_count += self.models[i].mean_count
        
        return mean_count // self.reso_num
    
    def mean_count_set(self, mean_count) -> None:
        for i in range(self.reso_num):
            self.models[i].mean_count = mean_count
    
    def mean_density_get(self) -> torch.Tensor:
        mean_densities = []
        for i in range(self.reso_num):
            mean_densities.append(self.models[i].mean_density)
        
        return torch.sum(torch.stack(mean_densities, dim=0), dim=0)
    
    def mean_density_set(self, mean_density) -> None:
        for i in range(self.reso_num):
            self.models[i].mean_count = mean_density