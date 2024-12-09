import copy
import math
import torch
from torch import nn, Tensor
from functools import partial
from typing import Callable

__all__ = ["EfficientNet_v2_s"]

def stochastic_depth(input, p, mode, training = True):
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


torch.fx.wrap("stochastic_depth")


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is not None and v < min_value:
        v = min_value
    else:
        v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return v


class SqueezeExcitation(torch.nn.Module):
    def __init__(self, input_channels, squeeze_channels, activation = torch.nn.ReLU, scale_activation = torch.nn.Sigmoid):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor):
        scale = self._scale(input)
        return scale * input


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


class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels, width_mult, min_value = None):
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    def __init__(self, expand_ratio, kernel, stride, input_channels, out_channels, num_layers, 
                 width_mult = 1.0, depth_mult = 1.0, block = None):
        
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)
        self.block = MBConv
        super().__init__()

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    def __init__(self, expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block = None):
        
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.block = FusedMBConv
        
        super().__init__()


class MBConv(nn.Module):
    def __init__(self, cnf, stochastic_depth_prob, norm_layer, se_layer = SqueezeExcitation):
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels, expanded_channels, kernel_size=1,
                                               norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        layers.append(
            ConvNormActivation(expanded_channels, expanded_channels, kernel_size=cnf.kernel, stride=cnf.stride,
                                 groups=expanded_channels, norm_layer=norm_layer, activation_layer=activation_layer))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(ConvNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, 
                                           activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(nn.Module):
    def __init__(self, cnf, stochastic_depth_prob, norm_layer):
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(ConvNormActivation(cnf.input_channels, expanded_channels, kernel_size=cnf.kernel, stride=cnf.stride,
                                               norm_layer=norm_layer, activation_layer=activation_layer))

            # project
            layers.append(ConvNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, 
                                               norm_layer=norm_layer, activation_layer=None))
        else:
            layers.append(ConvNormActivation(cnf.input_channels, cnf.out_channels, kernel_size=cnf.kernel,
                                               stride=cnf.stride, norm_layer=norm_layer, activation_layer=activation_layer))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


    
class EfficientNet_v2_s(nn.Module):
    def __init__(self, dropout = 0.2, stochastic_depth_prob = 0.2, num_classes = 3, norm_layer = None):
        super().__init__()
        
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
            ]
        
        last_channel = 1280

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU))

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=nn.SiLU))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(lastconv_output_channels, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

                
    def forward(self, x):
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

    
if __name__ == "__main__":
    model = EfficientNet_v2_s()
    input = torch.randn(1,3,224,224)
    output = model(input)
    print(input.size(), output.size())
    assert output.size()[-1] == 3
    print("Model done")
    
    
