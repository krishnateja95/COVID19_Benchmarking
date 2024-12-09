import math
from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.stochastic_depth import StochasticDepth

__all__ = ["MaxVit_tiny"]

def _get_conv_output_shape(input_size, kernel_size, stride, padding):
    return ((input_size[0] - kernel_size + 2 * padding) // stride + 1,
            (input_size[1] - kernel_size + 2 * padding) // stride + 1,)

def _make_block_input_shapes(input_size, n_blocks):
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes


def _get_relative_position_index(height, width):
    coords = torch.stack(torch.meshgrid([torch.arange(height), torch.arange(width)]))
    coords_flat = torch.flatten(coords, 1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)




class ConvNormActivation(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = None,
                 groups = 1, norm_layer = torch.nn.BatchNorm2d, activation_layer = torch.nn.ReLU, dilation = 1,
                 inplace = True, bias = None, conv_layer = torch.nn.Conv2d):

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups,bias=bias)]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels




class SqueezeExcitation(torch.nn.Module):
    def __init__(self, input_channels, squeeze_channels, activation = torch.nn.ReLU, scale_activation = torch.nn.Sigmoid):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input



class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, squeeze_ratio, stride, activation_layer,
                 norm_layer, p_stochastic_dropout = 0.0):
        super().__init__()

        proj: Sequence[nn.Module]
        self.proj: nn.Module

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)] + proj
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)

        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode="row") 
        else:
            self.stochastic_depth = nn.Identity()

        _layers = OrderedDict()
        _layers["pre_norm"] = norm_layer(in_channels)
        
        _layers["conv_a"] = ConvNormActivation(in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
                                                 activation_layer=activation_layer, norm_layer=norm_layer, inplace=None)
        
        _layers["conv_b"] = ConvNormActivation(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, 
                                                 activation_layer=activation_layer, norm_layer=norm_layer, 
                                                 groups=mid_channels, inplace=None)
        
        _layers["squeeze_excitation"] = SqueezeExcitation(mid_channels, sqz_channels, activation=nn.SiLU)
        _layers["conv_c"] = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=True)

        self.layers = nn.Sequential(_layers)

    def forward(self, x):
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))
        return res + x


class RelativePositionalMultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, head_dim, max_seq_len):
        super().__init__()

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len

        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3)
        self.scale_factor = feat_dim**-0.5

        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim)
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.empty(((2 * self.size - 1) * (2 * self.size - 1), self.n_heads), dtype=torch.float32))

        self.register_buffer("relative_position_index", _get_relative_position_index(self.size, self.size))
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self):
        bias_index = self.relative_position_index.view(-1)
        relative_bias = self.relative_position_bias_table[bias_index].view(self.max_seq_len, self.max_seq_len, -1)  # type: ignore
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        return relative_bias.unsqueeze(0)

    def forward(self, x):
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)

        k = k * self.scale_factor
        dot_prod = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        pos_bias = self.get_relative_positional_bias()

        dot_prod = F.softmax(dot_prod + pos_bias, dim=-1)

        out = torch.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, G, P, D)

        out = self.merge(out)
        return out


class SwapAxes(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        res = torch.swapaxes(x, self.a, self.b)
        return res


class WindowPartition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, p):
        B, C, H, W = x.shape
        P = p
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, (H // P) * (W // P), P * P, C)
        return x


class WindowDepartition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, p, h_partitions, w_partitions):
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions
        x = x.reshape(B, HP, WP, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, HP * P, WP * P)
        return x


class PartitionAttentionLayer(nn.Module):
    def __init__(self, in_channels, head_dim, partition_size, partition_type, grid_size, mlp_ratio, activation_layer,
                 norm_layer, attention_dropout, mlp_dropout, p_stochastic_dropout):
        super().__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size[0] // partition_size
        self.partition_type = partition_type
        self.grid_size = grid_size

        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partitions
        else:
            self.p, self.g = self.n_partitions, partition_size

        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        self.departition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()

        self.attn_layer = nn.Sequential(norm_layer(in_channels),
                                        RelativePositionalMultiHeadAttention(in_channels, head_dim, partition_size**2),
                                        nn.Dropout(attention_dropout))

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * mlp_ratio),
            activation_layer(),
            nn.Linear(in_channels * mlp_ratio, in_channels),
            nn.Dropout(mlp_dropout),
        )

        self.stochastic_dropout = StochasticDepth(p_stochastic_dropout, mode="row")

    def forward(self, x):
        gh, gw = self.grid_size[0] // self.p, self.grid_size[1] // self.p
        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = x + self.stochastic_dropout(self.attn_layer(x))
        x = x + self.stochastic_dropout(self.mlp_layer(x))
        x = self.departition_swap(x)
        x = self.departition_op(x, self.p, gh, gw)

        return x


class MaxVitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, squeeze_ratio, expansion_ratio, stride, norm_layer, activation_layer,
                 head_dim, mlp_ratio, mlp_dropout, attention_dropout, p_stochastic_dropout, partition_size, grid_size):
        super().__init__()

        layers: OrderedDict = OrderedDict()

        layers["MBconv"] = MBConv(in_channels=in_channels, out_channels=out_channels, expansion_ratio=expansion_ratio,
                                  squeeze_ratio=squeeze_ratio, stride=stride, activation_layer=activation_layer,
                                  norm_layer=norm_layer, p_stochastic_dropout=p_stochastic_dropout)
        
        layers["window_attention"] = PartitionAttentionLayer(in_channels=out_channels, head_dim=head_dim,
                                                             partition_size=partition_size, partition_type="window",
                                                             grid_size=grid_size, mlp_ratio=mlp_ratio,
                                                             activation_layer=activation_layer,
                                                             norm_layer=nn.LayerNorm, attention_dropout=attention_dropout,
                                                             mlp_dropout=mlp_dropout, p_stochastic_dropout=p_stochastic_dropout)
        
        layers["grid_attention"] = PartitionAttentionLayer(in_channels=out_channels, head_dim=head_dim,
                                                           partition_size=partition_size, partition_type="grid",
                                                           grid_size=grid_size, mlp_ratio=mlp_ratio,
                                                           activation_layer=activation_layer, norm_layer=nn.LayerNorm,
                                                           attention_dropout=attention_dropout, mlp_dropout=mlp_dropout,
                                                           p_stochastic_dropout=p_stochastic_dropout)
        
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class MaxVitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, squeeze_ratio, expansion_ratio, norm_layer, activation_layer,
                 head_dim, mlp_ratio, mlp_dropout, attention_dropout, partition_size, input_grid_size, n_layers, p_stochastic):
        
        super().__init__()
        self.layers = nn.ModuleList()
        self.grid_size = _get_conv_output_shape(input_grid_size, kernel_size=3, stride=2, padding=1)

        for idx, p in enumerate(p_stochastic):
            stride = 2 if idx == 0 else 1
            self.layers += [
                MaxVitLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    grid_size=self.grid_size,
                    p_stochastic_dropout=p,
                ),
            ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MaxVit_tiny(nn.Module):
    def __init__(self,  input_size = (224, 224), stem_channels = 64, partition_size = 7, block_channels = [64, 128, 256, 512], 
                 block_layers = [2, 2, 5, 2], head_dim =32, stochastic_depth_prob = 0.2, norm_layer = None, 
                 activation_layer = nn.GELU, squeeze_ratio = 0.25, expansion_ratio = 4, mlp_ratio = 4, 
                 mlp_dropout = 0.0, attention_dropout = 0.0, num_classes = 3):
        super().__init__()
        
        input_channels = 3
        
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99)

        block_input_sizes = _make_block_input_shapes(input_size, len(block_channels))
        
        self.stem = nn.Sequential(ConvNormActivation(input_channels, stem_channels, 3, stride=2, norm_layer=norm_layer,
                                                       activation_layer=activation_layer, bias=False, inplace=None),
                                  
                                  ConvNormActivation(stem_channels, stem_channels, 3, stride=1, norm_layer=None,
                                                       activation_layer=None, bias=True))

        input_size = _get_conv_output_shape(input_size, kernel_size=3, stride=2, padding=1)
        self.partition_size = partition_size

        self.blocks = nn.ModuleList()
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels

        p_stochastic = np.linspace(0, stochastic_depth_prob, sum(block_layers)).tolist()

        p_idx = 0
        for in_channel, out_channel, num_layers in zip(in_channels, out_channels, block_layers):
            self.blocks.append(
                MaxVitBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    input_grid_size=input_size,
                    n_layers=num_layers,
                    p_stochastic=p_stochastic[p_idx : p_idx + num_layers],
                ),
            )
            input_size = self.blocks[-1].grid_size
            p_idx += num_layers

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], num_classes, bias=False),
        )

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

if __name__ == "__main__":
    model = MaxVit_tiny()
    input = torch.randn(1,3,224,224)
    
    output = model(input)
    assert output.size()[-1] == 3
    print("Model done")
    
            