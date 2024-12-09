import math
import torch
from torch import nn
from functools import partial
from collections import OrderedDict

__all__ = ["RegNet_x_8gf"]

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

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
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


class SimpleStemIN(ConvNormActivation):
    def __init__(self, width_in, width_out, norm_layer, activation_layer):
        super().__init__(width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer)


class BottleneckTransform(nn.Sequential):
    def __init__(self, width_in, width_out, stride, norm_layer, activation_layer, group_width, bottleneck_multiplier, se_ratio):
        
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = ConvNormActivation(width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, 
                                         activation_layer=activation_layer)
        
        layers["b"] = ConvNormActivation(w_b, w_b, kernel_size=3, stride=stride, groups=g, 
                                         norm_layer=norm_layer, activation_layer=activation_layer)

        if se_ratio:
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(input_channels=w_b, squeeze_channels=width_se_out, activation=activation_layer)

        layers["c"] = ConvNormActivation(w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None)
        
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    def __init__(self, width_in, width_out, stride, norm_layer, activation_layer, group_width = 1, 
                 bottleneck_multiplier = 1.0, se_ratio = None):
        super().__init__()
        
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None)
            
        self.f = BottleneckTransform(width_in, width_out, stride, norm_layer, activation_layer, group_width,
                                     bottleneck_multiplier, se_ratio)
        self.activation = activation_layer(inplace=True)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    def __init__(self, width_in, width_out, stride, depth, block_constructor, norm_layer, activation_layer, group_width,
                 bottleneck_multiplier, se_ratio = None, stage_index = 0):
        super().__init__()

        for i in range(depth):
            block = block_constructor(width_in if i == 0 else width_out, width_out, 
                                      stride if i == 0 else 1, norm_layer, activation_layer, group_width,
                                      bottleneck_multiplier, se_ratio)

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(self, depths, widths, group_widths, bottleneck_multipliers, strides, se_ratio = None):
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(cls, depth, w_0, w_a, w_m, group_width, bottleneck_multiplier = 1.0, se_ratio = None):
        QUANT = 8
        STRIDE = 2

        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(depths=stage_depths, widths=stage_widths, group_widths=group_widths, bottleneck_multipliers=bottleneck_multipliers,
                   strides=strides, se_ratio=se_ratio)

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(stage_widths, bottleneck_ratios, group_widths):
       
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet_x_8gf(nn.Module):
    def __init__(self, num_classes = 3, stem_width = 32, stem_type = None, block_type = None,
                 norm_layer = partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1), activation = None):
        super().__init__()

        block_params = BlockParams.from_init_params(depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120)

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(3, stem_width, norm_layer, activation)

        current_width = stem_width

        blocks = []
        for i, (width_out, stride, depth, group_width, bottleneck_multiplier) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(current_width, width_out, stride, depth, block_type, norm_layer, activation,
                        group_width, bottleneck_multiplier, block_params.se_ratio, stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = RegNet_x_8gf()
    input = torch.randn(1,3,224,224)
    output = model(input)
    print(input.size(), output.size())
    assert output.size()[-1] == 3
    print("Model done")    
    
        
    