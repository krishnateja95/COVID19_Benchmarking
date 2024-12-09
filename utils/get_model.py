import sys
import torch
sys.path.append('../')
sys.path.append('../../')
import Models
def get_neural_network(model_name):
	if model_name == 'AlexNet':
		 return Models.AlexNet
	if model_name == 'ConvNext_Base':
		 return Models.ConvNext_Base
	if model_name == 'ConvNext_Large':
		 return Models.ConvNext_Large
	if model_name == 'ConvNext_Small':
		 return Models.ConvNext_Small
	if model_name == 'ConvNext_Tiny':
		 return Models.ConvNext_Tiny
	if model_name == 'MNASnet_05':
		 return Models.MNASnet_05
	if model_name == 'MNASnet_075':
		 return Models.MNASnet_075
	if model_name == 'MNASnet_10':
		 return Models.MNASnet_10
	if model_name == 'MNASnet_13':
		 return Models.MNASnet_13
	if model_name == 'Ghost_ResNet':
		 return Models.Ghost_ResNet
	if model_name == 'GhostNetV2':
		 return Models.GhostNetV2
	if model_name == 'MobileNetV1':
		 return Models.MobileNetV1
	if model_name == 'MobileNetV2':
		 return Models.MobileNetV2
	if model_name == 'mobilenet_v3_small':
		 return Models.mobilenet_v3_small
	if model_name == 'mobilenet_v3_large':
		 return Models.mobilenet_v3_large
	if model_name == 'ResNet18':
		 return Models.ResNet18
	if model_name == 'ResNet34':
		 return Models.ResNet34
	if model_name == 'ResNet50':
		 return Models.ResNet50
	if model_name == 'ResNet101':
		 return Models.ResNet101
	if model_name == 'ResNet152':
		 return Models.ResNet152
	if model_name == 'ResNext50':
		 return Models.ResNext50
	if model_name == 'ResNext101':
		 return Models.ResNext101
	if model_name == 'wide_ResNet50':
		 return Models.wide_ResNet50
	if model_name == 'wide_ResNet101':
		 return Models.wide_ResNet101
	if model_name == 'SqueezeNet_1_0':
		 return Models.SqueezeNet_1_0
	if model_name == 'SqueezeNet_1_1':
		 return Models.SqueezeNet_1_1
	if model_name == 'EfficientNet_b0':
		 return Models.EfficientNet_b0
	if model_name == 'EfficientNet_b1':
		 return Models.EfficientNet_b1
	if model_name == 'EfficientNet_b2':
		 return Models.EfficientNet_b2
	if model_name == 'EfficientNet_b3':
		 return Models.EfficientNet_b3
	if model_name == 'EfficientNet_b4':
		 return Models.EfficientNet_b4
	if model_name == 'EfficientNet_b5':
		 return Models.EfficientNet_b5
	if model_name == 'EfficientNet_b6':
		 return Models.EfficientNet_b6
	if model_name == 'EfficientNet_b7':
		 return Models.EfficientNet_b7
	if model_name == 'EfficientNet_v2_s':
		 return Models.EfficientNet_v2_s
	if model_name == 'EfficientNet_v2_m':
		 return Models.EfficientNet_v2_m
	if model_name == 'EfficientNet_v2_l':
		 return Models.EfficientNet_v2_l
	if model_name == 'ShuffleNetV2_x0_5':
		 return Models.ShuffleNetV2_x0_5
	if model_name == 'ShuffleNetV2_x1_0':
		 return Models.ShuffleNetV2_x1_0
	if model_name == 'ShuffleNetV2_x1_5':
		 return Models.ShuffleNetV2_x1_5
	if model_name == 'ShuffleNetV2_x2_0':
		 return Models.ShuffleNetV2_x2_0
	if model_name == 'GoogLeNet':
		 return Models.GoogLeNet
	if model_name == 'BNInception':
		 return Models.BNInception
	if model_name == 'Inception_Resnet_v2':
		 return Models.Inception_Resnet_v2
	if model_name == 'DenseNet_121':
		 return Models.DenseNet_121
	if model_name == 'DenseNet_161':
		 return Models.DenseNet_161
	if model_name == 'DenseNet_169':
		 return Models.DenseNet_169
	if model_name == 'DenseNet_201':
		 return Models.DenseNet_201
	if model_name == 'NFNet_F0':
		 return Models.NFNet_F0
	if model_name == 'NFNet_F1':
		 return Models.NFNet_F1
	if model_name == 'NFNet_F2':
		 return Models.NFNet_F2
	if model_name == 'NFNet_F3':
		 return Models.NFNet_F3
	if model_name == 'NFNet_F4':
		 return Models.NFNet_F4
	if model_name == 'NFNet_F5':
		 return Models.NFNet_F5
	if model_name == 'NFNet_F6':
		 return Models.NFNet_F6
	if model_name == 'NFNet_F7':
		 return Models.NFNet_F7
	if model_name == 'Vgg_11':
		 return Models.Vgg_11
	if model_name == 'Vgg_11_bn':
		 return Models.Vgg_11_bn
	if model_name == 'Vgg_13':
		 return Models.Vgg_13
	if model_name == 'Vgg_13_bn':
		 return Models.Vgg_13_bn
	if model_name == 'Vgg_16':
		 return Models.Vgg_16
	if model_name == 'Vgg_16_bn':
		 return Models.Vgg_16_bn
	if model_name == 'Vgg_19':
		 return Models.Vgg_19
	if model_name == 'Vgg_19_bn':
		 return Models.Vgg_19_bn
	if model_name == 'RegNet_y_16gf':
		 return Models.RegNet_y_16gf
	if model_name == 'RegNet_y_1_6gf':
		 return Models.RegNet_y_1_6gf
	if model_name == 'RegNet_y_8gf':
		 return Models.RegNet_y_8gf
	if model_name == 'RegNet_y_3_2gf':
		 return Models.RegNet_y_3_2gf
	if model_name == 'RegNet_x_800mf':
		 return Models.RegNet_x_800mf
	if model_name == 'RegNet_x_32gf':
		 return Models.RegNet_x_32gf
	if model_name == 'RegNet_x_1_6gf':
		 return Models.RegNet_x_1_6gf
	if model_name == 'RegNet_y_32gf':
		 return Models.RegNet_y_32gf
	if model_name == 'RegNet_x_16gf':
		 return Models.RegNet_x_16gf
	if model_name == 'RegNet_x_3_2gf':
		 return Models.RegNet_x_3_2gf
	if model_name == 'RegNet_y_800mf':
		 return Models.RegNet_y_800mf
	if model_name == 'RegNet_y_400mf':
		 return Models.RegNet_y_400mf
	if model_name == 'RegNet_y_128gf':
		 return Models.RegNet_y_128gf
	if model_name == 'RegNet_x_8gf':
		 return Models.RegNet_x_8gf
	if model_name == 'RegNet_x_400mf':
		 return Models.RegNet_x_400mf
	if model_name == 'Twins_SVT_Large':
		 return Models.Twins_SVT_Large
	if model_name == 'Twins_SVT_Base':
		 return Models.Twins_SVT_Base
	if model_name == 'Twins_SVT_Small':
		 return Models.Twins_SVT_Small
	if model_name == 'Twins_PCPVT_Small':
		 return Models.Twins_PCPVT_Small
	if model_name == 'Twins_PCPVT_Base':
		 return Models.Twins_PCPVT_Base
	if model_name == 'Twins_PCPVT_Large':
		 return Models.Twins_PCPVT_Large
	if model_name == 'FocalTransformer_Base':
		 return Models.FocalTransformer_Base
	if model_name == 'FocalTransformer_Tiny':
		 return Models.FocalTransformer_Tiny
	if model_name == 'FocalTransformer_Small':
		 return Models.FocalTransformer_Small
	if model_name == 'PoolFormer_M36':
		 return Models.PoolFormer_M36
	if model_name == 'PoolFormer_M48':
		 return Models.PoolFormer_M48
	if model_name == 'PoolFormer_S12':
		 return Models.PoolFormer_S12
	if model_name == 'PoolFormer_S24':
		 return Models.PoolFormer_S24
	if model_name == 'PoolFormer_S36':
		 return Models.PoolFormer_S36
	if model_name == 'Sep_ViT_Tiny':
		 return Models.Sep_ViT_Tiny
	if model_name == 'Sep_ViT_Small':
		 return Models.Sep_ViT_Small
	if model_name == 'Sep_ViT_Lite':
		 return Models.Sep_ViT_Lite
	if model_name == 'Sep_ViT_Base':
		 return Models.Sep_ViT_Base
	if model_name == 'MaxVit_tiny':
		 return Models.MaxVit_tiny
	if model_name == 'VAN_b0':
		 return Models.VAN_b0
	if model_name == 'VAN_b1':
		 return Models.VAN_b1
	if model_name == 'VAN_b2':
		 return Models.VAN_b2
	if model_name == 'VAN_b3':
		 return Models.VAN_b3
	if model_name == 'VAN_b4':
		 return Models.VAN_b4
	if model_name == 'VAN_b5':
		 return Models.VAN_b5
	if model_name == 'VAN_b6':
		 return Models.VAN_b6
	if model_name == 'CaiT_XXS24':
		 return Models.CaiT_XXS24
	if model_name == 'CaiT_XXS36':
		 return Models.CaiT_XXS36
	if model_name == 'CaiT_XS24':
		 return Models.CaiT_XS24
	if model_name == 'CaiT_XS36':
		 return Models.CaiT_XS36
	if model_name == 'CaiT_S24':
		 return Models.CaiT_S24
	if model_name == 'CaiT_S36':
		 return Models.CaiT_S36
	if model_name == 'BoTNet':
		 return Models.BoTNet
	if model_name == 'LVT':
		 return Models.LVT
	if model_name == 'DeepViT_16':
		 return Models.DeepViT_16
	if model_name == 'DeepViT_24':
		 return Models.DeepViT_24
	if model_name == 'DeepViT_32':
		 return Models.DeepViT_32
	if model_name == 'DeepViT_L':
		 return Models.DeepViT_L
	if model_name == 'DeepViT_S':
		 return Models.DeepViT_S
	if model_name == 'GCViT_Small':
		 return Models.GCViT_Small
	if model_name == 'GCViT_Small2':
		 return Models.GCViT_Small2
	if model_name == 'GCViT_Base':
		 return Models.GCViT_Base
	if model_name == 'GCViT_Large':
		 return Models.GCViT_Large
	if model_name == 'GCViT_xxTiny':
		 return Models.GCViT_xxTiny
	if model_name == 'GCViT_xTiny':
		 return Models.GCViT_xTiny
	if model_name == 'GCViT_Tiny':
		 return Models.GCViT_Tiny
	if model_name == 'GCViT_Tiny2':
		 return Models.GCViT_Tiny2
	if model_name == 'Swin_ViT_Tiny':
		 return Models.Swin_ViT_Tiny
	if model_name == 'Swin_ViT_Small':
		 return Models.Swin_ViT_Small
	if model_name == 'Swin_ViT_Base':
		 return Models.Swin_ViT_Base
	if model_name == 'Swin_ViT_Tiny_window7':
		 return Models.Swin_ViT_Tiny_window7
	if model_name == 'Swin_ViT_Small_window7':
		 return Models.Swin_ViT_Small_window7
	if model_name == 'Swin_ViT_Base_window7':
		 return Models.Swin_ViT_Base_window7
	if model_name == 'Swin_ViT_Large_window7':
		 return Models.Swin_ViT_Large_window7
	if model_name == 'EdgeNeXt_BNHS_Xsmall':
		 return Models.EdgeNeXt_BNHS_Xsmall
	if model_name == 'EdgeNeXt_BNHS_Small':
		 return Models.EdgeNeXt_BNHS_Small
	if model_name == 'EdgeNeXt_Xsmall':
		 return Models.EdgeNeXt_Xsmall
	if model_name == 'EdgeNeXt_Small':
		 return Models.EdgeNeXt_Small
	if model_name == 'EdgeNeXt_Base':
		 return Models.EdgeNeXt_Base
	if model_name == 'EdgeNeXt_BNHS_Xxsmall':
		 return Models.EdgeNeXt_BNHS_Xxsmall
	if model_name == 'MobileFormer_26M':
		 return Models.MobileFormer_26M
	if model_name == 'MobileFormer_52M':
		 return Models.MobileFormer_52M
	if model_name == 'MobileFormer_96M':
		 return Models.MobileFormer_96M
	if model_name == 'MobileFormer_151M':
		 return Models.MobileFormer_151M
	if model_name == 'MobileFormer_214M':
		 return Models.MobileFormer_214M
	if model_name == 'MobileFormer_294M':
		 return Models.MobileFormer_294M
	if model_name == 'MobileFormer_508M':
		 return Models.MobileFormer_508M
	if model_name == 'CCT_7':
		 return Models.CCT_7
	if model_name == 'CCT_7_sine':
		 return Models.CCT_7_sine
	if model_name == 'CCT_14':
		 return Models.CCT_14
	if model_name == 'CCT_14_sine':
		 return Models.CCT_14_sine
	if model_name == 'TNT_Base':
		 return Models.TNT_Base
	if model_name == 'TNT_Small':
		 return Models.TNT_Small
	if model_name == 'ViT_Tiny_patch16':
		 return Models.ViT_Tiny_patch16
	if model_name == 'ViT_Small_patch8':
		 return Models.ViT_Small_patch8
	if model_name == 'ViT_Small_patch16':
		 return Models.ViT_Small_patch16
	if model_name == 'ViT_Small_patch32':
		 return Models.ViT_Small_patch32
	if model_name == 'ViT_Base_patch8':
		 return Models.ViT_Base_patch8
	if model_name == 'ViT_Base_patch16':
		 return Models.ViT_Base_patch16
	if model_name == 'ViT_Base_patch32':
		 return Models.ViT_Base_patch32
	if model_name == 'ViT_Large_patch14':
		 return Models.ViT_Large_patch14
	if model_name == 'ViT_Large_patch16':
		 return Models.ViT_Large_patch16
	if model_name == 'ViT_Large_patch32':
		 return Models.ViT_Large_patch32
	if model_name == 'ViT_Huge_patch14':
		 return Models.ViT_Huge_patch14
	if model_name == 'ViT_Giant_patch14':
		 return Models.ViT_Giant_patch14
	if model_name == 'ViT_Gigantic_patch14':
		 return Models.ViT_Gigantic_patch14
	if model_name == 'MLPMixer':
		 return Models.MLPMixer
	if model_name == 'ResMLP':
		 return Models.ResMLP
	if model_name == 'EfficientFormer_L1':
		 return Models.EfficientFormer_L1
	if model_name == 'EfficientFormer_L3':
		 return Models.EfficientFormer_L3
	if model_name == 'EfficientFormer_L7':
		 return Models.EfficientFormer_L7
	if model_name == 'EfficientFormerV2_L':
		 return Models.EfficientFormerV2_L
	if model_name == 'EfficientFormerV2_S1':
		 return Models.EfficientFormerV2_S1
	if model_name == 'EfficientFormerV2_S0':
		 return Models.EfficientFormerV2_S0
	if model_name == 'EfficientFormerV2_S2':
		 return Models.EfficientFormerV2_S2
	if model_name == 'T2T_ViT_T_24':
		 return Models.T2T_ViT_T_24
	if model_name == 'T2T_ViT_T_14':
		 return Models.T2T_ViT_T_14
	if model_name == 'T2T_ViT_12':
		 return Models.T2T_ViT_12
	if model_name == 'T2T_ViT_14':
		 return Models.T2T_ViT_14
	if model_name == 'T2T_ViT_7':
		 return Models.T2T_ViT_7
	if model_name == 'T2T_ViT_24':
		 return Models.T2T_ViT_24
	if model_name == 'T2T_ViT_14_resnext':
		 return Models.T2T_ViT_14_resnext
	if model_name == 'T2T_ViT_10':
		 return Models.T2T_ViT_10
	if model_name == 'T2T_ViT_14_wide':
		 return Models.T2T_ViT_14_wide
	if model_name == 'T2T_ViT_T_19':
		 return Models.T2T_ViT_T_19
	if model_name == 'T2T_ViT_19':
		 return Models.T2T_ViT_19
	if model_name == 'RegionViT_Small_w14_peg':
		 return Models.RegionViT_Small_w14_peg
	if model_name == 'RegionViT_Medium':
		 return Models.RegionViT_Medium
	if model_name == 'RegionViT_Small':
		 return Models.RegionViT_Small
	if model_name == 'RegionViT_Base_w14_peg':
		 return Models.RegionViT_Base_w14_peg
	if model_name == 'RegionViT_Tiny':
		 return Models.RegionViT_Tiny
	if model_name == 'RegionViT_Base':
		 return Models.RegionViT_Base
	if model_name == 'RegionViT_Small_w14':
		 return Models.RegionViT_Small_w14
	if model_name == 'RegionViT_Base_w14':
		 return Models.RegionViT_Base_w14
	if model_name == 'CrossViT_15':
		 return Models.CrossViT_15
	if model_name == 'CrossViT_15_dagger':
		 return Models.CrossViT_15_dagger
	if model_name == 'CrossViT_18':
		 return Models.CrossViT_18
	if model_name == 'CrossViT_9':
		 return Models.CrossViT_9
	if model_name == 'CrossViT_9_dagger':
		 return Models.CrossViT_9_dagger
	if model_name == 'CrossViT_Base':
		 return Models.CrossViT_Base
	if model_name == 'CrossViT_Tiny':
		 return Models.CrossViT_Tiny
	if model_name == 'CrossViT_Small':
		 return Models.CrossViT_Small
	if model_name == 'CrossFormer_tiny':
		 return Models.CrossFormer_tiny
	if model_name == 'CrossFormer_small':
		 return Models.CrossFormer_small
	if model_name == 'CrossFormer_base':
		 return Models.CrossFormer_base
	if model_name == 'CrossFormer_large':
		 return Models.CrossFormer_large
	if model_name == 'PvT_v2_b0':
		 return Models.PvT_v2_b0
	if model_name == 'PvT_v2_b1':
		 return Models.PvT_v2_b1
	if model_name == 'PvT_v2_b2':
		 return Models.PvT_v2_b2
	if model_name == 'PvT_v2_b3':
		 return Models.PvT_v2_b3
	if model_name == 'PvT_v2_b4':
		 return Models.PvT_v2_b4
	if model_name == 'PvT_v2_b5':
		 return Models.PvT_v2_b5
	if model_name == 'PvT_v2_b2_Linear':
		 return Models.PvT_v2_b2_Linear
	if model_name == 'PvT_Small':
		 return Models.PvT_Small
	if model_name == 'PvT_Tiny':
		 return Models.PvT_Tiny
	if model_name == 'PvT_Medium':
		 return Models.PvT_Medium
	if model_name == 'PvT_Large':
		 return Models.PvT_Large
	if model_name == 'PvT_Huge_v2':
		 return Models.PvT_Huge_v2
	if model_name == 'LeViT_128S':
		 return Models.LeViT_128S
	if model_name == 'LeViT_128':
		 return Models.LeViT_128
	if model_name == 'LeViT_192':
		 return Models.LeViT_192
	if model_name == 'LeViT_256':
		 return Models.LeViT_256
	if model_name == 'LeViT_384':
		 return Models.LeViT_384
	if model_name == 'CvT_21':
		 return Models.CvT_21
	if model_name == 'CvT_13':
		 return Models.CvT_13
	if model_name == 'PiT_Small':
		 return Models.PiT_Small
	if model_name == 'PiT_XS':
		 return Models.PiT_XS
	if model_name == 'PiT_Base':
		 return Models.PiT_Base
	if model_name == 'PiT_TI':
		 return Models.PiT_TI

if __name__ == '__main__':
	model = get_neural_network(model_name='AlexNet')
	model = get_neural_network(model_name='ConvNext_Base')
	model = get_neural_network(model_name='ConvNext_Large')
	model = get_neural_network(model_name='ConvNext_Small')
	model = get_neural_network(model_name='ConvNext_Tiny')
	model = get_neural_network(model_name='MNASnet_05')
	model = get_neural_network(model_name='MNASnet_075')
	model = get_neural_network(model_name='MNASnet_10')
	model = get_neural_network(model_name='MNASnet_13')
	model = get_neural_network(model_name='Ghost_ResNet')
	model = get_neural_network(model_name='GhostNetV2')
	model = get_neural_network(model_name='MobileNetV1')
	model = get_neural_network(model_name='MobileNetV2')
	model = get_neural_network(model_name='mobilenet_v3_small')
	model = get_neural_network(model_name='mobilenet_v3_large')
	model = get_neural_network(model_name='ResNet18')
	model = get_neural_network(model_name='ResNet34')
	model = get_neural_network(model_name='ResNet50')
	model = get_neural_network(model_name='ResNet101')
	model = get_neural_network(model_name='ResNet152')
	model = get_neural_network(model_name='ResNext50')
	model = get_neural_network(model_name='ResNext101')
	model = get_neural_network(model_name='wide_ResNet50')
	model = get_neural_network(model_name='wide_ResNet101')
	model = get_neural_network(model_name='SqueezeNet_1_0')
	model = get_neural_network(model_name='SqueezeNet_1_1')
	model = get_neural_network(model_name='EfficientNet_b0')
	model = get_neural_network(model_name='EfficientNet_b1')
	model = get_neural_network(model_name='EfficientNet_b2')
	model = get_neural_network(model_name='EfficientNet_b3')
	model = get_neural_network(model_name='EfficientNet_b4')
	model = get_neural_network(model_name='EfficientNet_b5')
	model = get_neural_network(model_name='EfficientNet_b6')
	model = get_neural_network(model_name='EfficientNet_b7')
	model = get_neural_network(model_name='EfficientNet_v2_s')
	model = get_neural_network(model_name='EfficientNet_v2_m')
	model = get_neural_network(model_name='EfficientNet_v2_l')
	model = get_neural_network(model_name='ShuffleNetV2_x0_5')
	model = get_neural_network(model_name='ShuffleNetV2_x1_0')
	model = get_neural_network(model_name='ShuffleNetV2_x1_5')
	model = get_neural_network(model_name='ShuffleNetV2_x2_0')
	model = get_neural_network(model_name='GoogLeNet')
	model = get_neural_network(model_name='BNInception')
	model = get_neural_network(model_name='Inception_Resnet_v2')
	model = get_neural_network(model_name='DenseNet_121')
	model = get_neural_network(model_name='DenseNet_161')
	model = get_neural_network(model_name='DenseNet_169')
	model = get_neural_network(model_name='DenseNet_201')
	model = get_neural_network(model_name='NFNet_F0')
	model = get_neural_network(model_name='NFNet_F1')
	model = get_neural_network(model_name='NFNet_F2')
	model = get_neural_network(model_name='NFNet_F3')
	model = get_neural_network(model_name='NFNet_F4')
	model = get_neural_network(model_name='NFNet_F5')
	model = get_neural_network(model_name='NFNet_F6')
	model = get_neural_network(model_name='NFNet_F7')
	model = get_neural_network(model_name='Vgg_11')
	model = get_neural_network(model_name='Vgg_11_bn')
	model = get_neural_network(model_name='Vgg_13')
	model = get_neural_network(model_name='Vgg_13_bn')
	model = get_neural_network(model_name='Vgg_16')
	model = get_neural_network(model_name='Vgg_16_bn')
	model = get_neural_network(model_name='Vgg_19')
	model = get_neural_network(model_name='Vgg_19_bn')
	model = get_neural_network(model_name='RegNet_y_16gf')
	model = get_neural_network(model_name='RegNet_y_1_6gf')
	model = get_neural_network(model_name='RegNet_y_8gf')
	model = get_neural_network(model_name='RegNet_y_3_2gf')
	model = get_neural_network(model_name='RegNet_x_800mf')
	model = get_neural_network(model_name='RegNet_x_32gf')
	model = get_neural_network(model_name='RegNet_x_1_6gf')
	model = get_neural_network(model_name='RegNet_y_32gf')
	model = get_neural_network(model_name='RegNet_x_16gf')
	model = get_neural_network(model_name='RegNet_x_3_2gf')
	model = get_neural_network(model_name='RegNet_y_800mf')
	model = get_neural_network(model_name='RegNet_y_400mf')
	model = get_neural_network(model_name='RegNet_y_128gf')
	model = get_neural_network(model_name='RegNet_x_8gf')
	model = get_neural_network(model_name='RegNet_x_400mf')
	model = get_neural_network(model_name='Twins_SVT_Large')
	model = get_neural_network(model_name='Twins_SVT_Base')
	model = get_neural_network(model_name='Twins_SVT_Small')
	model = get_neural_network(model_name='Twins_PCPVT_Small')
	model = get_neural_network(model_name='Twins_PCPVT_Base')
	model = get_neural_network(model_name='Twins_PCPVT_Large')
	model = get_neural_network(model_name='FocalTransformer_Base')
	model = get_neural_network(model_name='FocalTransformer_Tiny')
	model = get_neural_network(model_name='FocalTransformer_Small')
	model = get_neural_network(model_name='PoolFormer_M36')
	model = get_neural_network(model_name='PoolFormer_M48')
	model = get_neural_network(model_name='PoolFormer_S12')
	model = get_neural_network(model_name='PoolFormer_S24')
	model = get_neural_network(model_name='PoolFormer_S36')
	model = get_neural_network(model_name='Sep_ViT_Tiny')
	model = get_neural_network(model_name='Sep_ViT_Small')
	model = get_neural_network(model_name='Sep_ViT_Lite')
	model = get_neural_network(model_name='Sep_ViT_Base')
	model = get_neural_network(model_name='MaxVit_tiny')
	model = get_neural_network(model_name='VAN_b0')
	model = get_neural_network(model_name='VAN_b1')
	model = get_neural_network(model_name='VAN_b2')
	model = get_neural_network(model_name='VAN_b3')
	model = get_neural_network(model_name='VAN_b4')
	model = get_neural_network(model_name='VAN_b5')
	model = get_neural_network(model_name='VAN_b6')
	model = get_neural_network(model_name='CaiT_XXS24')
	model = get_neural_network(model_name='CaiT_XXS36')
	model = get_neural_network(model_name='CaiT_XS24')
	model = get_neural_network(model_name='CaiT_XS36')
	model = get_neural_network(model_name='CaiT_S24')
	model = get_neural_network(model_name='CaiT_S36')
	model = get_neural_network(model_name='BoTNet')
	model = get_neural_network(model_name='LVT')
	model = get_neural_network(model_name='DeepViT_16')
	model = get_neural_network(model_name='DeepViT_24')
	model = get_neural_network(model_name='DeepViT_32')
	model = get_neural_network(model_name='DeepViT_L')
	model = get_neural_network(model_name='DeepViT_S')
	model = get_neural_network(model_name='GCViT_Small')
	model = get_neural_network(model_name='GCViT_Small2')
	model = get_neural_network(model_name='GCViT_Base')
	model = get_neural_network(model_name='GCViT_Large')
	model = get_neural_network(model_name='GCViT_xxTiny')
	model = get_neural_network(model_name='GCViT_xTiny')
	model = get_neural_network(model_name='GCViT_Tiny')
	model = get_neural_network(model_name='GCViT_Tiny2')
	model = get_neural_network(model_name='Swin_ViT_Tiny')
	model = get_neural_network(model_name='Swin_ViT_Small')
	model = get_neural_network(model_name='Swin_ViT_Base')
	model = get_neural_network(model_name='Swin_ViT_Tiny_window7')
	model = get_neural_network(model_name='Swin_ViT_Small_window7')
	model = get_neural_network(model_name='Swin_ViT_Base_window7')
	model = get_neural_network(model_name='Swin_ViT_Large_window7')
	model = get_neural_network(model_name='EdgeNeXt_BNHS_Xsmall')
	model = get_neural_network(model_name='EdgeNeXt_BNHS_Small')
	model = get_neural_network(model_name='EdgeNeXt_Xsmall')
	model = get_neural_network(model_name='EdgeNeXt_Small')
	model = get_neural_network(model_name='EdgeNeXt_Base')
	model = get_neural_network(model_name='EdgeNeXt_BNHS_Xxsmall')
	model = get_neural_network(model_name='MobileFormer_26M')
	model = get_neural_network(model_name='MobileFormer_52M')
	model = get_neural_network(model_name='MobileFormer_96M')
	model = get_neural_network(model_name='MobileFormer_151M')
	model = get_neural_network(model_name='MobileFormer_214M')
	model = get_neural_network(model_name='MobileFormer_294M')
	model = get_neural_network(model_name='MobileFormer_508M')
	model = get_neural_network(model_name='CCT_7')
	model = get_neural_network(model_name='CCT_7_sine')
	model = get_neural_network(model_name='CCT_14')
	model = get_neural_network(model_name='CCT_14_sine')
	model = get_neural_network(model_name='TNT_Base')
	model = get_neural_network(model_name='TNT_Small')
	model = get_neural_network(model_name='ViT_Tiny_patch16')
	model = get_neural_network(model_name='ViT_Small_patch8')
	model = get_neural_network(model_name='ViT_Small_patch16')
	model = get_neural_network(model_name='ViT_Small_patch32')
	model = get_neural_network(model_name='ViT_Base_patch8')
	model = get_neural_network(model_name='ViT_Base_patch16')
	model = get_neural_network(model_name='ViT_Base_patch32')
	model = get_neural_network(model_name='ViT_Large_patch14')
	model = get_neural_network(model_name='ViT_Large_patch16')
	model = get_neural_network(model_name='ViT_Large_patch32')
	model = get_neural_network(model_name='ViT_Huge_patch14')
	model = get_neural_network(model_name='ViT_Giant_patch14')
	model = get_neural_network(model_name='ViT_Gigantic_patch14')
	model = get_neural_network(model_name='MLPMixer')
	model = get_neural_network(model_name='ResMLP')
	model = get_neural_network(model_name='EfficientFormer_L1')
	model = get_neural_network(model_name='EfficientFormer_L3')
	model = get_neural_network(model_name='EfficientFormer_L7')
	model = get_neural_network(model_name='EfficientFormerV2_L')
	model = get_neural_network(model_name='EfficientFormerV2_S1')
	model = get_neural_network(model_name='EfficientFormerV2_S0')
	model = get_neural_network(model_name='EfficientFormerV2_S2')
	model = get_neural_network(model_name='T2T_ViT_T_24')
	model = get_neural_network(model_name='T2T_ViT_T_14')
	model = get_neural_network(model_name='T2T_ViT_12')
	model = get_neural_network(model_name='T2T_ViT_14')
	model = get_neural_network(model_name='T2T_ViT_7')
	model = get_neural_network(model_name='T2T_ViT_24')
	model = get_neural_network(model_name='T2T_ViT_14_resnext')
	model = get_neural_network(model_name='T2T_ViT_10')
	model = get_neural_network(model_name='T2T_ViT_14_wide')
	model = get_neural_network(model_name='T2T_ViT_T_19')
	model = get_neural_network(model_name='T2T_ViT_19')
	model = get_neural_network(model_name='RegionViT_Small_w14_peg')
	model = get_neural_network(model_name='RegionViT_Medium')
	model = get_neural_network(model_name='RegionViT_Small')
	model = get_neural_network(model_name='RegionViT_Base_w14_peg')
	model = get_neural_network(model_name='RegionViT_Tiny')
	model = get_neural_network(model_name='RegionViT_Base')
	model = get_neural_network(model_name='RegionViT_Small_w14')
	model = get_neural_network(model_name='RegionViT_Base_w14')
	model = get_neural_network(model_name='CrossViT_15')
	model = get_neural_network(model_name='CrossViT_15_dagger')
	model = get_neural_network(model_name='CrossViT_18')
	model = get_neural_network(model_name='CrossViT_9')
	model = get_neural_network(model_name='CrossViT_9_dagger')
	model = get_neural_network(model_name='CrossViT_Base')
	model = get_neural_network(model_name='CrossViT_Tiny')
	model = get_neural_network(model_name='CrossViT_Small')
	model = get_neural_network(model_name='CrossFormer_tiny')
	model = get_neural_network(model_name='CrossFormer_small')
	model = get_neural_network(model_name='CrossFormer_base')
	model = get_neural_network(model_name='CrossFormer_large')
	model = get_neural_network(model_name='PvT_v2_b0')
	model = get_neural_network(model_name='PvT_v2_b1')
	model = get_neural_network(model_name='PvT_v2_b2')
	model = get_neural_network(model_name='PvT_v2_b3')
	model = get_neural_network(model_name='PvT_v2_b4')
	model = get_neural_network(model_name='PvT_v2_b5')
	model = get_neural_network(model_name='PvT_v2_b2_Linear')
	model = get_neural_network(model_name='PvT_Small')
	model = get_neural_network(model_name='PvT_Tiny')
	model = get_neural_network(model_name='PvT_Medium')
	model = get_neural_network(model_name='PvT_Large')
	model = get_neural_network(model_name='PvT_Huge_v2')
	model = get_neural_network(model_name='LeViT_128S')
	model = get_neural_network(model_name='LeViT_128')
	model = get_neural_network(model_name='LeViT_192')
	model = get_neural_network(model_name='LeViT_256')
	model = get_neural_network(model_name='LeViT_384')
	model = get_neural_network(model_name='CvT_21')
	model = get_neural_network(model_name='CvT_13')
	model = get_neural_network(model_name='PiT_Small')
	model = get_neural_network(model_name='PiT_XS')
	model = get_neural_network(model_name='PiT_Base')
	model = get_neural_network(model_name='PiT_TI')
	input = torch.randn(1,3,224,224)
	output = model(input)
	print('Running PiT_TI')
	print(input.size())
	print(output.size())
	assert output.size()[-1] == 3
	print('Model done')
	print('Model done again')

