import math
import torch
import torch.nn as nn
import collections.abc
from torch import _assert
from itertools import repeat
from torch.utils.checkpoint import checkpoint


__all__ = ['TNT_Small']  

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=None, bias=True, drop=0., use_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    
def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=(), interpolation='bicubic', antialias=False):
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, dim_out, num_pixel, num_heads_in=4, num_heads_out=12, mlp_ratio=4., qkv_bias=False,
                 proj_drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_in = norm_layer(dim)
        self.attn_in = Attention(dim, dim, num_heads=num_heads_in, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.norm_mlp_in = norm_layer(dim)
        self.mlp_in = Mlp(in_features=dim, hidden_features=int(dim * 4), out_features=dim, act_layer=act_layer, drop=proj_drop,)
        self.norm1_proj = norm_layer(dim)
        self.proj = nn.Linear(dim * num_pixel, dim_out, bias=True)

        self.norm_out = norm_layer(dim_out)
        self.attn_out = Attention(dim_out, dim_out, num_heads=num_heads_out, qkv_bias=qkv_bias, 
                                  attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm_mlp = norm_layer(dim_out)
        self.mlp = Mlp(in_features=dim_out, hidden_features=int(dim_out * mlp_ratio), out_features=dim_out,
                       act_layer=act_layer, drop=proj_drop)

    def forward(self, pixel_embed, patch_embed):
        pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        B, N, C = patch_embed.size()
        patch_embed = torch.cat(
            [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
            dim=1)
        patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        return pixel_embed, patch_embed


class PixelEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, in_dim=48, stride=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = (self.grid_size[0]) * (self.grid_size[1])
        self.img_size = img_size
        self.num_patches = num_patches
        self.in_dim = in_dim
        new_patch_size = [math.ceil(ps / stride) for ps in patch_size]
        self.new_patch_size = new_patch_size

        self.proj = nn.Conv2d(in_chans, self.in_dim, kernel_size=7, padding=3, stride=stride)
        self.unfold = nn.Unfold(kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x, pixel_pos):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0],
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        _assert(W == self.img_size[1],
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x)
        x = self.unfold(x)
        x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size[0], self.new_patch_size[1])
        x = x + pixel_pos
        x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
        return x

    
    
class TNT_Small(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=3, global_pool='token', embed_dim=384,
                 inner_dim=24, depth=12, num_heads_inner=4, num_heads_outer=6, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., pos_drop_rate=0., proj_drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, first_stride=4):
        super().__init__()
        assert global_pool in ('', 'token', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.grad_checkpointing = False

        self.pixel_embed = PixelEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      in_dim=inner_dim, stride=first_stride)
        num_patches = self.pixel_embed.num_patches
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size[0] * new_patch_size[1]
        
        self.norm1_proj = norm_layer(num_pixel * inner_dim)
        self.proj = nn.Linear(num_pixel * inner_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pixel_pos = nn.Parameter(torch.zeros(1, inner_dim, new_patch_size[0], new_patch_size[1]))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        for i in range(depth):
            blocks.append(Block(dim=inner_dim, dim_out=embed_dim, num_pixel=num_pixel, num_heads_in=num_heads_inner,
                                num_heads_out=num_heads_outer, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer))
            
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.patch_pos, std=.02)
        trunc_normal_(self.pixel_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)
        
        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.blocks:
                pixel_embed, patch_embed = checkpoint(blk, pixel_embed, patch_embed)
        else:
            for blk in self.blocks:
                pixel_embed, patch_embed = blk(pixel_embed, patch_embed)

        patch_embed = self.norm(patch_embed)
        return patch_embed

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


if __name__ == "__main__":
    model = TNT_Small()
    input = torch.randn(1,3,224,224)
    output = model(input)
    print("Model done")
    print(input.size())
    print(output.size())
    assert output.size()[-1] == 3
    print("Model done again")  




