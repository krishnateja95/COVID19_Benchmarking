import copy
import math
import torch
import torch.nn as nn
import collections.abc
from itertools import repeat
from functools import partial
import torch.nn.functional as F

__all__ = ['RegionViT_Base']


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





class AttentionWithRelPos(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_map_dim=None, num_cls_tokens=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_cls_tokens = num_cls_tokens
        if attn_map_dim is not None:
            one_dim = attn_map_dim[0]
            rel_pos_dim = (2 * one_dim - 1)
            self.rel_pos = nn.Parameter(torch.zeros(num_heads, rel_pos_dim ** 2))
            tmp = torch.arange(rel_pos_dim ** 2).reshape((rel_pos_dim, rel_pos_dim))
            out = []
            offset_x = offset_y = one_dim // 2
            for y in range(one_dim):
                for x in range(one_dim):
                    for dy in range(one_dim):
                        for dx in range(one_dim):
                            out.append(tmp[dy - y + offset_y, dx - x + offset_x])
            self.rel_pos_index = torch.tensor(out, dtype=torch.long)
            trunc_normal_(self.rel_pos, std=.02)
        else:
            self.rel_pos = None

    def forward(self, x, patch_attn=False, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rel_pos is not None and patch_attn:
            rel_pos = self.rel_pos[:, self.rel_pos_index.to(attn.device)].reshape(self.num_heads, N - self.num_cls_tokens, N - self.num_cls_tokens)
            attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] = attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] + rel_pos

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp1d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp2d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNorm2d, self).__init__()

        self.channels = channels
        self.eps = torch.tensor(eps)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(1, keepdim=True)
        std = torch.sqrt(input.var(1, unbiased=False, keepdim=True) + self.eps)
        out = (input - mean) / std
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self):
        return '{channels}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNorm1d, self).__init__()

        self.channels = channels
        self.eps = torch.tensor(eps)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(1, channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(1, keepdim=True)
        std = torch.sqrt(input.var(1, unbiased=False, keepdim=True) + self.eps)
        out = (input - mean) / std
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self):
        return '{channels}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class Attention2d(nn.Module):
    def __init__(self, dim, out_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        out_dim = dim if out_dim is None else out_dim
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, out_dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_dim = out_dim

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).flatten(-2)
        qkv = qkv.reshape(B, 3, self.num_heads, self.out_dim // self.num_heads, H*W).permute(1, 0, 2, 4, 3)  
        q, k, v = qkv[0], qkv[1], qkv[2] 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-2, -1).reshape(B, self.out_dim, H, W) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, patch_conv_type='linear'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches


        if patch_conv_type == '3conv':
            if patch_size[0] == 4:
                tmp = [
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=3, stride=2, padding=1),
                    LayerNorm2d(embed_dim // 4),
                    nn.GELU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    LayerNorm2d(embed_dim // 2),
                    nn.GELU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                ]
            else:
                raise ValueError(f"Unknown patch size {patch_size[0]}")
            self.proj = nn.Sequential(*tmp)
        else:
            if patch_conv_type == '1conv':
                kernel_size = (2 * patch_size[0], 2 * patch_size[1])
                stride = (patch_size[0], patch_size[1])
                padding = (patch_size[0] - 1, patch_size[1] - 1)
            else:
                kernel_size = patch_size
                stride = patch_size
                padding = 0

            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                                  stride=stride, padding=padding)

    def forward(self, x, extra_padding=False):
        B, C, H, W = x.shape
        if extra_padding and (H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0):
            p_l = (self.patch_size[1] - W % self.patch_size[1]) // 2
            p_r = (self.patch_size[1] - W % self.patch_size[1]) - p_l
            p_t = (self.patch_size[0] - H % self.patch_size[0]) // 2
            p_b = (self.patch_size[0] - H % self.patch_size[0]) - p_t
            x = F.pad(x, (p_l, p_r, p_t, p_b))
        x = self.proj(x)
        return x


class R2LAttentionPlusFFN(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_path=0., attn_drop=0., drop=0.,
                 cls_attn=True):
        super().__init__()

        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = [(kernel_size, kernel_size), (kernel_size, kernel_size), 0]
        self.kernel_size = kernel_size

        if cls_attn:
            self.norm0 = norm_layer(input_channels)
        else:
            self.norm0 = None

        self.norm1 = norm_layer(input_channels)
        self.attn = AttentionWithRelPos(
            input_channels, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            attn_map_dim=(kernel_size[0][0], kernel_size[0][1]), num_cls_tokens=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(input_channels)
        self.mlp = Mlp(in_features=input_channels, hidden_features=int(output_channels * mlp_ratio), out_features=output_channels, act_layer=act_layer, drop=drop)

        self.expand = nn.Sequential(
            norm_layer(input_channels),
            act_layer(),
            nn.Linear(input_channels, output_channels)
        ) if input_channels != output_channels else None
        self.output_channels = output_channels
        self.input_channels = input_channels

    def forward(self, xs):
        out, B, H, W, mask = xs
        cls_tokens = out[:, 0:1, ...]

        C = cls_tokens.shape[-1]
        cls_tokens = cls_tokens.reshape(B, -1, C) 

        if self.norm0 is not None:
            cls_tokens = cls_tokens + self.drop_path(self.attn(self.norm0(cls_tokens)))

        cls_tokens = cls_tokens.reshape(-1, 1, C)

        out = torch.cat((cls_tokens, out[:, 1:, ...]), dim=1)
        tmp = out

        tmp = tmp + self.drop_path(self.attn(self.norm1(tmp), patch_attn=True, mask=mask))
        identity = self.expand(tmp) if self.expand is not None else tmp
        tmp = identity + self.drop_path(self.mlp(self.norm2(tmp)))

        return tmp


class Projection(nn.Module):
    def __init__(self, input_channels, output_channels, act_layer, mode='sc'):
        super().__init__()
        tmp = []
        if 'c' in mode:
            ks = 2 if 's' in mode else 1
            if ks == 2:
                stride = ks
                ks = ks + 1
                padding = ks // 2
            else:
                stride = ks
                padding = 0

            if input_channels == output_channels and ks == 1:
                tmp.append(nn.Identity())
            else:
                tmp.extend([
                    LayerNorm2d(input_channels),
                    act_layer(),
                ])
                tmp.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=ks, stride=stride, padding=padding, groups=input_channels))

        self.proj = nn.Sequential(*tmp)
        self.proj_cls = self.proj

    def forward(self, xs):
        cls_tokens, patch_tokens = xs
        # x: BxCxHxW
        cls_tokens = self.proj_cls(cls_tokens)
        patch_tokens = self.proj(patch_tokens)
        return cls_tokens, patch_tokens


def convert_to_flatten_layout(cls_tokens, patch_tokens, ws):
    B, C, H, W = patch_tokens.shape
    _, _, H_ks, W_ks = cls_tokens.shape
    need_mask = False
    p_l, p_r, p_t, p_b = 0, 0, 0, 0
    if H % (H_ks * ws) != 0 or W % (W_ks * ws) != 0:
        p_l, p_r = 0, W_ks * ws - W
        p_t, p_b = 0, H_ks * ws - H
        patch_tokens = F.pad(patch_tokens, (p_l, p_r, p_t, p_b))
        need_mask = True

    B, C, H, W = patch_tokens.shape
    kernel_size = (H // H_ks, W // W_ks)
    tmp = F.unfold(patch_tokens, kernel_size=kernel_size, stride=kernel_size, padding=(0, 0)) 
    patch_tokens = tmp.transpose(1, 2).reshape(-1, C, kernel_size[0] * kernel_size[1]).transpose(-2, -1)

    if need_mask:
        BH_sK_s, ksks, C = patch_tokens.shape
        H_s, W_s = H // ws, W // ws
        mask = torch.ones(BH_sK_s // B, 1 + ksks, 1 + ksks, device=patch_tokens.device, dtype=torch.float)
        right = torch.zeros(1 + ksks, 1 + ksks, device=patch_tokens.device, dtype=torch.float)
        tmp = torch.zeros(ws, ws, device=patch_tokens.device, dtype=torch.float)
        tmp[0:(ws - p_r), 0:(ws - p_r)] = 1.
        tmp = tmp.repeat(ws, ws)
        right[1:, 1:] = tmp
        right[0, 0] = 1
        right[0, 1:] = torch.tensor([1.] * (ws - p_r) + [0.] * p_r).repeat(ws).to(right.device)
        right[1:, 0] = torch.tensor([1.] * (ws - p_r) + [0.] * p_r).repeat(ws).to(right.device)
        bottom = torch.zeros_like(right)
        bottom[0:ws * (ws - p_b) + 1, 0:ws * (ws - p_b) + 1] = 1.
        bottom_right = copy.deepcopy(right)
        bottom_right[0:ws * (ws - p_b) + 1, 0:ws * (ws - p_b) + 1] = 1.

        mask[W_s - 1:(H_s - 1) * W_s:W_s, ...] = right
        mask[(H_s - 1) * W_s:, ...] = bottom
        mask[-1, ...] = bottom_right
        mask = mask.repeat(B, 1, 1)
    else:
        mask = None

    cls_tokens = cls_tokens.flatten(2).transpose(-2, -1)  
    cls_tokens = cls_tokens.reshape(-1, 1, cls_tokens.size(-1))  

    out = torch.cat((cls_tokens, patch_tokens), dim=1)

    return out, mask, p_l, p_r, p_t, p_b, B, C, H, W


def convert_to_spatial_layout(out, output_channels, B, H, W, kernel_size, mask, p_l, p_r, p_t, p_b):
    cls_tokens = out[:, 0:1, ...]
    patch_tokens = out[:, 1:, ...]
    C = output_channels
    kernel_size = kernel_size[0]
    H_ks = H // kernel_size[0]
    W_ks = W // kernel_size[1]
    cls_tokens = cls_tokens.reshape(B, -1, C).transpose(-2, -1).reshape(B, C, H_ks, W_ks)
    patch_tokens = patch_tokens.transpose(1, 2).reshape((B, -1, kernel_size[0] * kernel_size[1] * C)).transpose(1, 2)
    patch_tokens = F.fold(patch_tokens, (H, W), kernel_size=kernel_size, stride=kernel_size, padding=(0, 0))

    if mask is not None:
        if p_b > 0:
            patch_tokens = patch_tokens[:, :, :-p_b, :]
        if p_r > 0:
            patch_tokens = patch_tokens[:, :, :, :-p_r]

    return cls_tokens, patch_tokens


class ConvAttBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, num_blocks, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, pool='sc',
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_path_rate=(0.,), attn_drop_rate=0., drop_rate=0.,
                 cls_attn=True, peg=False):
        super().__init__()
        tmp = []
        if pool:
            tmp.append(Projection(input_channels, output_channels, act_layer=act_layer, mode=pool))

        for i in range(num_blocks):
            kernel_size_ = kernel_size
            tmp.append(R2LAttentionPlusFFN(output_channels, output_channels, kernel_size_, num_heads, mlp_ratio, qkv_bias, qk_scale,
                                           act_layer=act_layer, norm_layer=norm_layer, drop_path=drop_path_rate[i], attn_drop=attn_drop_rate, drop=drop_rate,
                                           cls_attn=cls_attn))

        self.block = nn.ModuleList(tmp)
        self.output_channels = output_channels
        self.ws = kernel_size
        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = [(kernel_size, kernel_size), (kernel_size, kernel_size), 0]
        self.kernel_size = kernel_size

        self.peg = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, groups=output_channels, bias=False) if peg else None

    def forward(self, xs):
        cls_tokens, patch_tokens = xs
        cls_tokens, patch_tokens = self.block[0]((cls_tokens, patch_tokens))
        out, mask, p_l, p_r, p_t, p_b, B, C, H, W = convert_to_flatten_layout(cls_tokens, patch_tokens, self.ws)
        for i in range(1, len(self.block)):
            blk = self.block[i]

            out = blk((out, B, H, W, mask))
            if self.peg is not None and i == 1:
                cls_tokens, patch_tokens = convert_to_spatial_layout(out, self.output_channels, B, H, W, self.kernel_size, mask, p_l, p_r, p_t, p_b)
                cls_tokens = cls_tokens + self.peg(cls_tokens)
                patch_tokens = patch_tokens + self.peg(patch_tokens)
                out, mask, p_l, p_r, p_t, p_b, B, C, H, W = convert_to_flatten_layout(cls_tokens, patch_tokens, self.ws)

        cls_tokens, patch_tokens = convert_to_spatial_layout(out, self.output_channels, B, H, W, self.kernel_size, mask, p_l, p_r, p_t, p_b)
        return cls_tokens, patch_tokens


class RegionViT_Base(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=3, embed_dim=[128, 128, 256, 512, 1024], depth=[2, 2, 14, 2],
                 num_heads=[4, 8, 16, 32], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 kernel_sizes=[7, 7, 7, 7], downsampling=['c', 'sc', 'sc', 'sc'],
                 patch_conv_type='1conv',
                 computed_cls_token=True, peg=False,
                 det_norm=False):
      
        super().__init__()
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.num_features = embed_dim[-1]  
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0],
            patch_conv_type=patch_conv_type)

        if not isinstance(mlp_ratio, (list, tuple)):
            mlp_ratio = [mlp_ratio] * len(depth)

        self.computed_cls_token = computed_cls_token
        self.cls_token = PatchEmbed(
            img_size=img_size, patch_size=patch_size * kernel_sizes[0], in_chans=in_chans, embed_dim=embed_dim[0],
            patch_conv_type='linear'
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        total_depth = sum(depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)] 
        dpr_ptr = 0
        self.layers = nn.ModuleList()
        for i in range(len(embed_dim) - 1):
            curr_depth = depth[i]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            
            self.layers.append(
                ConvAttBlock(embed_dim[i], embed_dim[i + 1], kernel_size=kernel_sizes[i], num_blocks=depth[i], drop_path_rate=dpr_,
                             num_heads=num_heads[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                             pool=downsampling[i], norm_layer=norm_layer, attn_drop_rate=attn_drop_rate, drop_rate=drop_rate,
                             cls_attn=True, peg=peg)
            )
            dpr_ptr += curr_depth
        self.norm = norm_layer(embed_dim[-1])

        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        if not computed_cls_token:
            trunc_normal_(self.cls_token, std=.02)

        self.det_norm = det_norm
        if self.det_norm:
            for i in range(4):
                layer = LayerNorm2d(embed_dim[1 + i])
                layer_name = f'norm{i}'
                self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if not self.computed_cls_token:
            return {'cls_token'}
        else:
            return {}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, detection=False):
        o_x = x
        x = self.patch_embed(x)
        cls_tokens = self.cls_token(o_x, extra_padding=True)
        x = self.pos_drop(x)  # N C H W
        tmp_out = []

        for idx, layer in enumerate(self.layers):
            cls_tokens, x = layer((cls_tokens, x))
            if self.det_norm:
                norm_layer = getattr(self, f'norm{idx}')
                x = norm_layer(x)
            tmp_out.append(x)

        if detection:
            return tmp_out

        N, C, H, W = cls_tokens.shape
        cls_tokens = cls_tokens.reshape(N, C, -1).transpose(1, 2)
        cls_tokens = self.norm(cls_tokens)
        out = torch.mean(cls_tokens, dim=1)

        return out

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

if __name__ == "__main__":
    model = RegionViT_Base()
    input = torch.randn(1,3,224,224)
    output = model(input)
    print("Model done")
    print(input.size())
    print(output.size())
    assert output.size()[-1] == 3
    print("Model done again")