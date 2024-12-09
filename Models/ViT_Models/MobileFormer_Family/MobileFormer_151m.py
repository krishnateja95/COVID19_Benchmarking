import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import warnings
import math
from itertools import repeat
import collections.abc


__all__ = ['MobileFormer_151M']


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()

        out = x.view(b, -1, h, w)
        return out

class DyReLU(nn.Module):
    def __init__(self, num_func=2, use_bias=False, scale=2., serelu=False):
        super(DyReLU, self).__init__()

        assert(num_func>=-1 and num_func<=2)
        self.num_func = num_func
        self.scale = scale

        serelu = serelu and num_func == 1
        self.act = nn.ReLU6(inplace=True) if num_func == 0 or serelu else nn.Sequential()

    def forward(self, x):
        if isinstance(x, tuple):
            out, a = x
        else:
            out = x

        out = self.act(out)


        if self.num_func == 1:    
            a = a * self.scale
            out = out * a
        elif self.num_func == 2:  
            _, C, _, _ = a.shape
            a1, a2 = torch.split(a, [C//2, C//2], dim=1)
            a1 = (a1 - 0.5) * self.scale + 1.0 
            a2 = (a2 - 0.5) * self.scale      
            out = torch.max(out*a1, out*a2)
            
        return out

class HyperFunc(nn.Module):
    def __init__(self, token_dim, oup, sel_token_id=0, reduction_ratio=4):
        super(HyperFunc, self).__init__()

        self.sel_token_id = sel_token_id
        squeeze_dim = token_dim // reduction_ratio
        self.hyper = nn.Sequential(
            nn.Linear(token_dim, squeeze_dim),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_dim, oup),
            h_sigmoid()
        )


    def forward(self, x):
        if isinstance(x, tuple):
            x, attn = x

        if self.sel_token_id == -1:
            hp = self.hyper(x).permute(1, 2, 0)         

            bs, T, H, W = attn.shape
            attn = attn.view(bs, T, H*W)
            hp = torch.matmul(hp, attn)                 
            h = hp.view(bs, -1, H, W)
        else:
            t = x[self.sel_token_id]
            h = self.hyper(t)
            h = torch.unsqueeze(torch.unsqueeze(h, 2), 3)
        return h

class MaxDepthConv(nn.Module):
    def __init__(self, inp, oup, stride):
        super(MaxDepthConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, oup, (3,1), stride, (1, 0), bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inp, oup, (1,3), stride, (0, 1), bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        out = torch.max(y1, y2)
        return out


class Local2Global(nn.Module):
    def __init__(
            self,
            inp,
            block_type='mlp',
            token_dim=128,
            token_num=6,
            inp_res=0,
            attn_num_heads=2,
            use_dynamic=False,
            norm_pos='post',
            drop_path_rate=0.,
            remove_proj_local=True,
        ):
        super(Local2Global, self).__init__()
        #print(f'L2G: {attn_num_heads} heads, inp: {inp}, token: {token_dim}')

        self.num_heads = attn_num_heads
        self.token_num = token_num 
        self.norm_pos = norm_pos
        self.block = block_type
        self.use_dynamic = use_dynamic

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, inp),
                h_sigmoid(),
            )


        if 'mlp' in block_type:
            self.mlp = nn.Linear(inp_res, token_num)

        if 'attn' in block_type:
            self.scale = (inp // attn_num_heads) ** -0.5
            self.q = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(inp, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)
        
        self.remove_proj_local = remove_proj_local
        if self.remove_proj_local == False:
            self.k = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.v = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            

    def forward(self, x):
        features, tokens = x 
                             

        bs, C, H, W = features.shape
        T, _, _ = tokens.shape
        attn = None

        if 'mlp' in self.block:
            t_sum = self.mlp(features.view(bs, C, -1)).permute(2, 0, 1)         

        if 'attn' in self.block:
            t = self.q(tokens).view(T, bs, self.num_heads, -1).permute(1, 2, 0, 3)  
            if self.remove_proj_local:
                k = features.view(bs, self.num_heads, -1, H*W)                          
                attn = (t @ k) * self.scale                                             
    
                attn_out = attn.softmax(dim=-1)                
                attn_out = (attn_out @ k.transpose(-1, -2))     
                                                              
            else:
                k = self.k(features).view(bs, self.num_heads, -1, H*W)                          
                v = self.v(features).view(bs, self.num_heads, -1, H*W)                          
                attn = (t @ k) * self.scale                                             
    
                attn_out = attn.softmax(dim=-1)                 
                attn_out = (attn_out @ v.transpose(-1, -2))    
                                                               
 
            t_a = attn_out.permute(2, 0, 1, 3)             
            t_a = t_a.reshape(T, bs, -1)

            if 'mlp' in self.block:
                t_sum = t_sum + t_a
            else:
                t_sum = t_a

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            t_sum = t_sum * alp

        t_sum = self.proj(t_sum)
        tokens = tokens + self.drop_path(t_sum)
        tokens = self.layer_norm(tokens)

        if attn is not None:
            bs, Nh, Ca, HW = attn.shape
            attn = attn.view(bs, Nh, Ca, H, W)

        return tokens, attn


class DnaBlock3(nn.Module):
    def __init__(self, inp, oup, stride, exp_ratios, kernel_size=(3,3), dw_conv='dw', group_num=1, se_flag=[2,0,2,0], hyper_token_id=0,
        hyper_reduction_ratio=4, token_dim=128, token_num=6, inp_res=49, gbr_type='mlp', gbr_dynamic=[False, False, False], gbr_ffn=False,
        gbr_before_skip=False, mlp_token_exp=4, norm_pos='post', drop_path_rate=0., cnn_drop_path_rate=0., attn_num_heads=2, remove_proj_local=True,):
        super(DnaBlock3, self).__init__()

        #print(f'block: {inp_res}, cnn-drop {cnn_drop_path_rate:.4f}, mlp-drop {drop_path_rate:.4f}')
        if isinstance(exp_ratios, tuple):
            e1, e2 = exp_ratios
        else:
            e1, e2 = exp_ratios, 4
        k1, k2 = kernel_size

        self.stride = stride
        self.hyper_token_id = hyper_token_id

        self.identity = stride == 1 and inp == oup
        self.use_conv_alone = False
        if e1 == 1 or e2 == 0:
            self.use_conv_alone = True
            if dw_conv == 'dw':
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(inp, inp*e1, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp*e1),
                    nn.ReLU6(inplace=True),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                    # pw-linear
                    nn.Conv2d(inp*e1, oup, 1, 1, 0, groups=group_num, bias=False),
                    nn.BatchNorm2d(oup),
                )
            elif dw_conv == 'sepdw':
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(inp, inp*e1//2, (3,1), (stride,1), (1,0), groups=inp, bias=False),
                    nn.BatchNorm2d(inp*e1//2),
                    nn.Conv2d(inp*e1//2, inp*e1, (1,3), (1, stride), (0,1), groups=inp*e1//2, bias=False),
                    nn.BatchNorm2d(inp*e1),
                    nn.ReLU6(inplace=True),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                    # pw-linear
                    nn.Conv2d(inp*e1, oup, 1, 1, 0, groups=group_num, bias=False),
                    nn.BatchNorm2d(oup),
                )
 
        else:
            self.se_flag = se_flag
            hidden_dim1 = round(inp * e1)
            hidden_dim2 = round(oup * e2)

            if dw_conv == 'dw':
                self.conv1 = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim1, k1, stride, k1//2, groups=inp, bias=False),
                    nn.BatchNorm2d(hidden_dim1),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential()
                )
            elif dw_conv == 'maxdw':
                self.conv1 = nn.Sequential(
                    MaxDepthConv(inp, hidden_dim1, stride),
                    ChannelShuffle(group_num) if group_num > 1 else nn.Sequential()
                )
            elif dw_conv == 'sepdw':
                self.conv1 = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim1//2, (3,1), (stride,1), (1,0), groups=inp, bias=False),
                    nn.BatchNorm2d(hidden_dim1//2),
                    nn.Conv2d(hidden_dim1//2, hidden_dim1, (1,3), (1, stride), (0,1), groups=hidden_dim1//2, bias=False),
                    nn.BatchNorm2d(hidden_dim1),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential()
                )
 
            num_func = se_flag[0] 
            self.act1 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper1 = HyperFunc(
                token_dim, 
                hidden_dim1 * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[0] > 0 else nn.Sequential()
                

            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim1, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
            )
            num_func = -1
            self.act2 = DyReLU(num_func=num_func, scale=2.)


            if dw_conv == 'dw':
                self.conv3 = nn.Sequential(
                    nn.Conv2d(oup, hidden_dim2, k2, 1, k2//2, groups=oup, bias=False),
                    nn.BatchNorm2d(hidden_dim2),
                    ChannelShuffle(oup) if group_num > 1 else nn.Sequential()
                )
            elif dw_conv == 'maxdw':
                self.conv3 = nn.Sequential(
                    MaxDepthConv(oup, hidden_dim2, 1),
                )
            elif dw_conv == 'sepdw':
                self.conv3 = nn.Sequential(
                    nn.Conv2d(oup, hidden_dim2//2, (3,1), (1,1), (1,0), groups=oup, bias=False),
                    nn.BatchNorm2d(hidden_dim2//2),
                    nn.Conv2d(hidden_dim2//2, hidden_dim2, (1,3), (1, 1), (0,1), groups=hidden_dim2//2, bias=False),
                    nn.BatchNorm2d(hidden_dim2),
                    ChannelShuffle(oup) if group_num > 1 else nn.Sequential()
                )
           
            num_func = se_flag[2]
            self.act3 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper3 = HyperFunc(
                token_dim, 
                hidden_dim2 * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[2] > 0 else nn.Sequential()
 

            self.conv4 = nn.Sequential(
                nn.Conv2d(hidden_dim2, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup)
            )
            num_func = 1 if se_flag[3] == 1 else -1 
            self.act4 = DyReLU(num_func=num_func, scale=2.)
            self.hyper4 = HyperFunc(
                token_dim, 
                oup * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[3] > 0 else nn.Sequential()
 

            self.drop_path = DropPath(cnn_drop_path_rate)

            # l2g, gb, g2l
            self.local_global = Local2Global(
                inp,
                block_type = gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                inp_res=inp_res,
                use_dynamic = gbr_dynamic[0],
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

            self.global_block = GlobalBlock(
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                mlp_token_exp=mlp_token_exp,
                use_dynamic = gbr_dynamic[1],
                use_ffn=gbr_ffn,
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate
            )
 
            oup_res = inp_res // (stride * stride)

            self.global_local = Global2Local(
                oup,
                oup_res,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                use_dynamic = gbr_dynamic[2],
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

    def forward(self, x):
        features, tokens = x
        if self.use_conv_alone:
            out = self.conv(features)
        else:
            tokens, attn = self.local_global((features, tokens))
            tokens = self.global_block(tokens)

            out = self.conv1(features)

            if self.hyper_token_id == -1:
                attn = attn.mean(dim=1) # bs x T x H x W
                if self.stride > 1:
                    _, _, H, W = out.shape
                    attn = F.adaptive_avg_pool2d(attn, (H, W))
                attn = torch.softmax(attn, dim=1)

            if self.se_flag[0] > 0:
                hp = self.hyper1((tokens, attn))
                out = self.act1((out, hp))
            else:
                out = self.act1(out)

            out = self.conv2(out)
            out = self.act2(out)

            out_cp = out
            out = self.conv3(out)
            if self.se_flag[2] > 0:
                hp = self.hyper3((tokens, attn))
                out = self.act3((out, hp))
            else:
                out = self.act3(out)

            out = self.conv4(out)
            if self.se_flag[3] > 0:
                hp = self.hyper4((tokens, attn))
                out = self.act4((out, hp))
            else:
                out = self.act4(out)

            out = self.drop_path(out) + out_cp

            out = self.global_local((out, tokens))

        if self.identity:
            out = out + features

        return (out, tokens)


class GlobalBlock(nn.Module):
    def __init__(
        self,
        block_type='mlp',
        token_dim=128,
        token_num=6,
        mlp_token_exp=4,
        attn_num_heads=4,
        use_dynamic=False,
        use_ffn=False,
        norm_pos='post',
        drop_path_rate=0.
    ):
        super(GlobalBlock, self).__init__()

        #print(f'G2G: {attn_num_heads} heads')

        self.block = block_type
        self.num_heads = attn_num_heads
        self.token_num = token_num
        self.norm_pos = norm_pos
        self.use_dynamic = use_dynamic
        self.use_ffn = use_ffn
        self.ffn_exp = 2

        if self.use_ffn:
            #print('use ffn')
            self.ffn = nn.Sequential(
                nn.Linear(token_dim, token_dim * self.ffn_exp),
                nn.GELU(),
                nn.Linear(token_dim * self.ffn_exp, token_dim)
            )
            self.ffn_norm = nn.LayerNorm(token_dim)
            

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, token_dim),
                h_sigmoid(),
            )

        
        if 'mlp' in self.block:
            self.token_mlp = nn.Sequential(
                nn.Linear(token_num, token_num*mlp_token_exp),
                nn.GELU(),
                nn.Linear(token_num*mlp_token_exp, token_num),
            )

        if 'attn' in self.block:
            self.scale = (token_dim // attn_num_heads) ** -0.5
            self.q = nn.Linear(token_dim, token_dim)

        self.channel_mlp = nn.Linear(token_dim, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        tokens = x

        T, bs, C = tokens.shape

        if 'mlp' in self.block:
            t = self.token_mlp(tokens.permute(1, 2, 0)) 
            t_sum = t.permute(2, 0, 1)                 

        if 'attn' in self.block:
            t = self.q(tokens).view(T, bs, self.num_heads, -1).permute(1, 2, 0, 3) 
            k = tokens.permute(1, 2, 0).view(bs, self.num_heads, -1, T)            
            attn = (t @ k) * self.scale                                             

            attn_out = attn.softmax(dim=-1)                 
            attn_out = (attn_out @ k.transpose(-1, -2))     
                                                           
            t_a = attn_out.permute(2, 0, 1, 3)             
            t_a = t_a.reshape(T, bs, -1)

            t_sum = t_sum + t_a if 'mlp' in self.block else t_a

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            t_sum = t_sum * alp

        t_sum = self.channel_mlp(t_sum) 
        tokens = tokens + self.drop_path(t_sum)
        tokens = self.layer_norm(tokens)

        if self.use_ffn:
            t_ffn = self.ffn(tokens)
            tokens = tokens + t_ffn
            tokens = self.ffn_norm(tokens)

 
        return tokens

class Global2Local(nn.Module):
    def __init__(self, inp, inp_res=0, block_type='mlp', token_dim=128, token_num=6, attn_num_heads=2, use_dynamic=False,
        drop_path_rate=0., remove_proj_local=True,):
        super(Global2Local, self).__init__()
        #print(f'G2L: {attn_num_heads} heads, inp: {inp}, token: {token_dim}')

        self.token_num = token_num
        self.num_heads = attn_num_heads
        self.block = block_type
        self.use_dynamic = use_dynamic

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, inp),
                h_sigmoid(),
            )


        if 'mlp' in self.block:
            self.mlp = nn.Linear(token_num, inp_res)

        if 'attn' in self.block:
            self.scale = (inp // attn_num_heads) ** -0.5
            self.k = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(token_dim, inp)
        self.drop_path = DropPath(drop_path_rate)

        self.remove_proj_local = remove_proj_local
        if self.remove_proj_local == False:
            self.q = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.fuse = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
 
    def forward(self, x):
        out, tokens = x

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            v = self.proj(tokens)
            v = (v * alp).permute(1, 2, 0)
        else:
            v = self.proj(tokens).permute(1, 2, 0)  

        bs, C, H, W = out.shape
        if 'mlp' in self.block:
            g_sum = self.mlp(v).view(bs, C, H, W)       

        if 'attn' in self.block:
            if self.remove_proj_local:
                q = out.view(bs, self.num_heads, -1, H*W).transpose(-1, -2)                         
            else:
                q = self.q(out).view(bs, self.num_heads, -1, H*W).transpose(-1, -2)                         

            k = self.k(tokens).permute(1, 2, 0).view(bs, self.num_heads, -1, self.token_num)   
            attn = (q @ k) * self.scale                         

            attn_out = attn.softmax(dim=-1)                     
            
            vh = v.view(bs, self.num_heads, -1, self.token_num) 
            attn_out = (attn_out @ vh.transpose(-1, -2))        
                                                                
            g_a = attn_out.transpose(-1, -2).reshape(bs, C, H, W)  

            if self.remove_proj_local == False:
                g_a = self.fuse(g_a)            

            g_sum = g_sum + g_a if 'mlp' in self.block else g_a

        out = out + self.drop_path(g_sum)

        return out


class DnaBlock(nn.Module):
    def __init__(self, inp, oup, stride, exp_ratios, kernel_size=(3,3), dw_conv='dw', group_num=1, se_flag=[2,0,2,0], hyper_token_id=0,
        hyper_reduction_ratio=4, token_dim=128, token_num=6, inp_res=49, gbr_type='mlp', gbr_dynamic=[False, False, False],
        gbr_ffn=False, gbr_before_skip=False, mlp_token_exp=4, norm_pos='post', drop_path_rate=0., cnn_drop_path_rate=0.,
        attn_num_heads=2, remove_proj_local=True,):
        super(DnaBlock, self).__init__()

        #print(f'block: {inp_res}, cnn-drop {cnn_drop_path_rate:.4f}, mlp-drop {drop_path_rate:.4f}')
        if isinstance(exp_ratios, tuple):
            e1, e2 = exp_ratios
        else:
            e1, e2 = exp_ratios, 4
        k1, k2 = kernel_size

        self.stride = stride
        self.hyper_token_id = hyper_token_id

        self.gbr_before_skip = gbr_before_skip
        self.identity = stride == 1 and inp == oup
        self.use_conv_alone = False
        if e1 == 1 or e2 == 0:
            self.use_conv_alone = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp*e1, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp*e1),
                nn.ReLU6(inplace=True),
                ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                # pw-linear
                nn.Conv2d(inp*e1, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.se_flag = se_flag
            hidden_dim = round(inp * e1)

            self.conv1 = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(hidden_dim),
                ChannelShuffle(group_num) if group_num > 1 else nn.Sequential()
            )

            num_func = se_flag[0] 
            self.act1 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper1 = HyperFunc(
                token_dim, 
                hidden_dim * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[0] > 0 else nn.Sequential()
                

            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, k1, stride, k1//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            num_func = se_flag[2] 
            self.act2 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper2 = HyperFunc(
                token_dim, 
                hidden_dim * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[2] > 0 else nn.Sequential()
 

            self.conv3 = nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
                ChannelShuffle(group_num) if group_num > 1 else nn.Sequential()
            )
            num_func = 1 if se_flag[3] == 1 else -1 
            self.act3 = DyReLU(num_func=num_func, scale=2.)
            self.hyper3 = HyperFunc(
                token_dim, 
                oup * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[3] > 0 else nn.Sequential()
 

            self.drop_path = DropPath(cnn_drop_path_rate)

            # l2g, gb, g2l
            self.local_global = Local2Global(
                inp,
                block_type = gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                inp_res=inp_res,
                use_dynamic = gbr_dynamic[0],
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

            self.global_block = GlobalBlock(
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                mlp_token_exp=mlp_token_exp,
                use_dynamic = gbr_dynamic[1],
                use_ffn=gbr_ffn,
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate
            )
 
            oup_res = inp_res // (stride * stride)

            self.global_local = Global2Local(
                oup,
                oup_res,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                use_dynamic = gbr_dynamic[2],
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

    def forward(self, x):
        features, tokens = x
        if self.use_conv_alone:
            out = self.conv(features)
            if self.identity:
                out = self.drop_path(out) + features

        else:
            tokens, attn = self.local_global((features, tokens))
            tokens = self.global_block(tokens)

            out = self.conv1(features)

            if self.hyper_token_id == -1:
                attn = attn.mean(dim=1)
                if self.stride > 1:
                    _, _, H, W = out.shape
                    attn = F.adaptive_avg_pool2d(attn, (H, W))
                attn = torch.softmax(attn, dim=1)

            if self.se_flag[0] > 0:
                hp = self.hyper1((tokens, attn))
                out = self.act1((out, hp))
            else:
                out = self.act1(out)

            out = self.conv2(out)
            if self.se_flag[2] > 0:
                hp = self.hyper2((tokens, attn))
                out = self.act2((out, hp))
            else:
                out = self.act2(out)

            out = self.conv3(out)
            if self.se_flag[3] > 0:
                hp = self.hyper3((tokens, attn))
                out = self.act3((out, hp))
            else:
                out = self.act3(out)

            if self.gbr_before_skip == True:
                out = self.global_local((out, tokens))
                if self.identity:
                    out = self.drop_path(out) + features
            else:
                if self.identity:
                    out = self.drop_path(out) + features
                out = self.global_local((out, tokens))

        return (out, tokens)


class MergeClassifier(nn.Module):
    def __init__(self, inp, oup=1280, ch_exp=6, num_classes=3, drop_rate=0., drop_branch=[0.0, 0.0], group_num=1, 
        token_dim=128, cls_token_num=1, last_act='relu', hyper_token_id=0, hyper_reduction_ratio=4):
        super(MergeClassifier, self).__init__()

        self.drop_branch=drop_branch
        self.cls_token_num = cls_token_num

        hidden_dim = inp * ch_exp
        self.conv = nn.Sequential(
            ChannelShuffle(group_num) if group_num > 1 else nn.Sequential(),
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=group_num, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        self.last_act = last_act
        num_func = 2 if last_act == 'dyrelu' else 0 
        self.act = DyReLU(num_func=num_func, scale=2.)
 
        self.hyper = HyperFunc(
            token_dim, 
            hidden_dim * num_func, 
            sel_token_id=hyper_token_id, 
            reduction_ratio=hyper_reduction_ratio
        ) if last_act == 'dyrelu' else nn.Sequential()
 
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )

        if cls_token_num > 0:
            cat_token_dim = token_dim * cls_token_num 
        elif cls_token_num == 0:
            cat_token_dim = token_dim
        else:
            cat_token_dim = 0

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + cat_token_dim, oup),
            nn.BatchNorm1d(oup),
            h_swish()
        )

        self.classifier = nn.Sequential(
           nn.Dropout(drop_rate),
           nn.Linear(oup, num_classes)
       )

    def forward(self, x):
        features, tokens = x

        x = self.conv(features)

        if self.last_act == 'dyrelu':
            hp = self.hyper(tokens)
            x = self.act((x, hp))
        else:
            x = self.act(x)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ps = [x]
        
        if self.cls_token_num == 0:
            avg_token = torch.mean(F.relu6(tokens), dim=0)
            ps.append(avg_token)
        elif self.cls_token_num < 0:
            pass
        else:
            for i in range(self.cls_token_num):
                ps.append(tokens[i])

        # drop branch
        if self.training and self.drop_branch[0] + self.drop_branch[1] > 1e-8:
            rd = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device)
            keep_local = 1 - self.drop_branch[0]
            keep_global = 1 - self.drop_branch[1]
            rd_local = (keep_local + rd).floor_()
            rd_global = -((rd - keep_global).floor_())
            ps[0] = ps[0].div(keep_local) * rd_local
            ps[1] = ps[1].div(keep_global) * rd_global

        x = torch.cat(ps, dim=1)
        x = self.fc(x)

        x = self.classifier(x)
        return x

dna_blocks = [ 
        ['DnaBlock3', 2,  12, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  16, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  3,  16, 1, 1, 3], #3
        ['DnaBlock3', 6,  32, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  3,  32, 1, 1, 3], #5
        ['DnaBlock3', 6,  64, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4,  64, 1, 1, 4], #7
        ['DnaBlock',  6,  88, 1, 1, 4], #8
        ['DnaBlock',  6,  88, 1, 1, 4], #9
        ['DnaBlock3', 6, 128, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 128, 1, 1, 4], #11
        ['DnaBlock',  6, 128, 1, 1, 4], #12
    ]

class MobileFormer_151M(nn.Module):
    def __init__(self, block_args= dna_blocks, num_classes=3, img_size=224, width_mult=1.0, in_chans=3, stem_chs=12, num_features=1280,dw_conv='dw',
        kernel_size=(3,3), cnn_exp=(6,4), group_num=1, se_flag=[2,0,2,0], hyper_token_id=0,hyper_reduction_ratio=4, token_dim=192,
        token_num=6, cls_token_num=1, last_act='relu', last_exp=6, gbr_type='attn', gbr_dynamic=[True, False, False], gbr_norm='post',
        gbr_ffn=True, gbr_before_skip=False, gbr_drop=[0., 0.], mlp_token_exp=4, drop_rate=0., drop_path_rate=0.,
        cnn_drop_path_rate=0.1, attn_num_heads = 2, remove_proj_local=True,):

        super(MobileFormer_151M, self).__init__()

        cnn_drop_path_rate = drop_path_rate
        mdiv = 8 if width_mult > 1.01 else 4
        self.num_classes = num_classes

        #global tokens
        self.tokens = nn.Embedding(token_num, token_dim) 

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chs, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_chs),
            nn.ReLU6(inplace=True)
        )
        input_channel = stem_chs

        # blocks
        layer_num = len(block_args)
        inp_res = img_size * img_size // 4
        layers = []
        for idx, val in enumerate(block_args):
            b, t, c, n, s, t2 = val # t2 for block2 the second expand
            block = eval(b)

            t = (t, t2)
            output_channel = _make_divisible(c * width_mult, mdiv) if idx > 0 else _make_divisible(c * width_mult, 4) 

            drop_path_prob = drop_path_rate * (idx+1) / layer_num
            cnn_drop_path_prob = cnn_drop_path_rate * (idx+1) / layer_num

            layers.append(block(input_channel, output_channel, s, t, dw_conv=dw_conv, kernel_size=kernel_size,
                group_num=group_num, se_flag=se_flag, hyper_token_id=hyper_token_id, hyper_reduction_ratio=hyper_reduction_ratio,
                token_dim=token_dim, token_num=token_num, inp_res=inp_res, gbr_type=gbr_type, gbr_dynamic=gbr_dynamic, gbr_ffn=gbr_ffn,
                gbr_before_skip=gbr_before_skip, mlp_token_exp=mlp_token_exp, norm_pos=gbr_norm, drop_path_rate=drop_path_prob,
                cnn_drop_path_rate=cnn_drop_path_prob, attn_num_heads=attn_num_heads, remove_proj_local=remove_proj_local,))
            input_channel = output_channel

            if s == 2:
                inp_res = inp_res // 4

            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t, dw_conv=dw_conv, kernel_size=kernel_size, group_num=group_num,
                    se_flag=se_flag, hyper_token_id=hyper_token_id, hyper_reduction_ratio=hyper_reduction_ratio, token_dim=token_dim, 
                    token_num=token_num, inp_res=inp_res, gbr_type=gbr_type, gbr_dynamic=gbr_dynamic, gbr_ffn=gbr_ffn,
                    gbr_before_skip=gbr_before_skip, mlp_token_exp=mlp_token_exp, norm_pos=gbr_norm, drop_path_rate=drop_path_prob, 
                    cnn_drop_path_rate=cnn_drop_path_prob, attn_num_heads=attn_num_heads, remove_proj_local=remove_proj_local,))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # last layer of local to global
        self.local_global = Local2Global(input_channel, block_type = gbr_type, token_dim=token_dim, token_num=token_num,
            inp_res=inp_res, use_dynamic = gbr_dynamic[0], norm_pos=gbr_norm, drop_path_rate=drop_path_rate, attn_num_heads=attn_num_heads)

        # classifer
        self.classifier = MergeClassifier(input_channel, oup=num_features, ch_exp=last_exp, num_classes=num_classes,
            drop_rate=drop_rate, drop_branch=gbr_drop, group_num=group_num, token_dim=token_dim, cls_token_num=cls_token_num, 
            last_act = last_act, hyper_token_id=hyper_token_id, hyper_reduction_ratio=hyper_reduction_ratio)

        #initialize
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        # setup tokens
        bs, _, _, _ = x.shape
        z = self.tokens.weight
        tokens = z[None].repeat(bs, 1, 1).clone()
        tokens = tokens.permute(1, 0, 2)
 
        # stem -> features -> classifier
        x = self.stem(x)
        x, tokens = self.features((x, tokens))
        tokens, attn = self.local_global((x, tokens))
        y = self.classifier((x, tokens))

        return y


if __name__ == "__main__":
    model = MobileFormer_151M()
    input = torch.randn(1,3,224,224)
    model.eval()
    output = model(input)
    print("Model done")
    print(input.size())
    print(output.size())
    assert output.size()[-1] == 3
    print("Model done again")