import torch
from torch import nn
from functools import partial

__all__ = ["mobilenet_v3_large"]

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


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is not None and v < min_value:
        v = min_value
    else:
        v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return v


class InvertedResidualConfig:
    def __init__(self, input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation, width_mult):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    def __init__(self, cnf, norm_layer, se_layer = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid)):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1, 
                                   norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                               stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                               norm_layer=norm_layer, activation_layer=activation_layer))
        
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(ConvNormActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, 
                                         norm_layer=norm_layer, activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class mobilenet_v3_large(nn.Module):
    def __init__(self, num_classes = 3, block = InvertedResidual,
                 norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), dropout = 0.2):
        super().__init__()
        
        reduce_divider = 1
        dilation = 1

        bneck_conf = partial(InvertedResidualConfig, width_mult=1.0)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=1.0)
        
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
        
        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, 
                                         norm_layer=norm_layer, activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                         norm_layer=norm_layer, activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    
if __name__ == "__main__":
    model = mobilenet_v3_large()
    input = torch.randn(1,3,224,224)
    output = model(input)
    print(input.size(), output.size())
    assert output.size()[-1] == 3
    print("Model done")




