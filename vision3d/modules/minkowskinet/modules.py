import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiFunctional as MEF


class MEGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(MEGroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, input_tensor):
        feats = self.gn(input_tensor.F)
        if isinstance(input_tensor, ME.TensorField):
            return ME.TensorField(
                feats,
                coordinate_field_map_key=input_tensor.coordinate_field_map_key,
                coordinate_manager=input_tensor.coordinate_manager,
                quantization_mode=input_tensor.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                feats,
                coordinate_map_key=input_tensor.coordinate_map_key,
                coordinate_manager=input_tensor.coordinate_manager,
            )


class MEConvBlock(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=None,
                 batch_norm=True,
                 relu=True,
                 dimension=-1):
        super().__init__()
        if bias is None:
            bias = not batch_norm
        layers = []
        layers.append(('conv', ME.MinkowskiConvolution(input_dim,
                                                       output_dim,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       dilation=dilation,
                                                       bias=bias,
                                                       dimension=dimension)))
        if batch_norm:
            layers.append(('bn', ME.MinkowskiBatchNorm(output_dim)))
        if relu:
            layers.append(('relu', ME.MinkowskiReLU()))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super().forward(inputs)


class MEConvTrBlock(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=None,
                 batch_norm=True,
                 relu=True,
                 dimension=-1):
        super().__init__()
        if bias is None:
            bias = not batch_norm
        layers = []
        layers.append(('conv', ME.MinkowskiConvolutionTranspose(input_dim,
                                                                output_dim,
                                                                kernel_size=kernel_size,
                                                                stride=stride,
                                                                dilation=dilation,
                                                                bias=bias,
                                                                dimension=dimension)))
        if batch_norm:
            layers.append(('bn', ME.MinkowskiBatchNorm(output_dim)))
        if relu:
            layers.append(('relu', ME.MinkowskiReLU()))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super().forward(inputs)


class MEResidualBlock(nn.Module):
    def __init__(self, feature_dim, kernel_size, stride=1, dimension=-1):
        super().__init__()
        self.conv1 = MEConvBlock(feature_dim, feature_dim, kernel_size=kernel_size, stride=stride, dimension=dimension)
        self.conv2 = MEConvBlock(feature_dim,
                                 feature_dim,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 relu=False,
                                 dimension=dimension)

    def forward(self, x):
        identity = x

        residual = self.conv1(x)
        residual = self.conv2(residual)

        residual += identity
        residual = MEF.relu(residual)

        return residual


class MEResUNet(nn.Module):
    def __init__(self,
                 input_dim,
                 conv_dims,
                 conv_tr_dims,
                 output_dim,
                 conv1_kernel_size,
                 normalize_features=False,
                 dimension=3):
        super().__init__()
        self.normalize_features = normalize_features
        self.conv1 = MEConvBlock(input_dim,
                                 conv_dims[0],
                                 kernel_size=conv1_kernel_size,
                                 relu=False,
                                 dimension=dimension)
        self.block1 = MEResidualBlock(conv_dims[0], kernel_size=3, dimension=dimension)

        self.conv2 = MEConvBlock(conv_dims[0],
                                 conv_dims[1],
                                 kernel_size=3,
                                 stride=2,
                                 relu=False,
                                 dimension=dimension)
        self.block2 = MEResidualBlock(conv_dims[1], kernel_size=3, dimension=dimension)

        self.conv3 = MEConvBlock(conv_dims[1],
                                 conv_dims[2],
                                 kernel_size=3,
                                 stride=2,
                                 relu=False,
                                 dimension=dimension)
        self.block3 = MEResidualBlock(conv_dims[2], kernel_size=3, dimension=dimension)

        self.conv4 = MEConvBlock(conv_dims[2],
                                 conv_dims[3],
                                 kernel_size=3,
                                 stride=2,
                                 relu=False,
                                 dimension=dimension)
        self.block4 = MEResidualBlock(conv_dims[3], kernel_size=3, dimension=dimension)

        self.conv4_tr = MEConvTrBlock(conv_dims[3],
                                      conv_tr_dims[3],
                                      kernel_size=3,
                                      stride=2,
                                      relu=False,
                                      dimension=dimension)
        self.block4_tr = MEResidualBlock(conv_tr_dims[3], kernel_size=3, dimension=dimension)

        self.conv3_tr = MEConvTrBlock(conv_dims[2] + conv_tr_dims[3],
                                      conv_tr_dims[2],
                                      kernel_size=3,
                                      stride=2,
                                      relu=False,
                                      dimension=dimension)
        self.block3_tr = MEResidualBlock(conv_tr_dims[2], kernel_size=3, dimension=dimension)

        self.conv2_tr = MEConvTrBlock(conv_dims[1] + conv_tr_dims[2],
                                      conv_tr_dims[1],
                                      kernel_size=3,
                                      stride=2,
                                      relu=False,
                                      dimension=dimension)
        self.block2_tr = MEResidualBlock(conv_tr_dims[1], kernel_size=3, dimension=dimension)

        self.conv1_tr = MEConvBlock(conv_dims[0] + conv_tr_dims[1],
                                    conv_tr_dims[0],
                                    kernel_size=1,
                                    bias=False,
                                    batch_norm=False,
                                    dimension=dimension)

        self.final = MEConvBlock(conv_tr_dims[0],
                                 output_dim,
                                 kernel_size=1,
                                 batch_norm=False,
                                 relu=False,
                                 dimension=dimension)

    def forward(self, x):
        out_s1 = self.conv1(x)
        out_s1 = self.block1(out_s1)

        out_s2 = self.conv2(out_s1)
        out_s2 = self.block2(out_s2)

        out_s4 = self.conv3(out_s2)
        out_s4 = self.block3(out_s4)

        out_s8 = self.conv4(out_s4)
        out_s8 = self.block4(out_s8)

        out_s4_tr = self.conv4_tr(out_s8)
        out_s4_tr = self.block4_tr(out_s4_tr)

        out_s4_tr = ME.cat(out_s4_tr, out_s4)

        out_s2_tr = self.conv3_tr(out_s4_tr)
        out_s2_tr = self.block3_tr(out_s2_tr)

        out_s2_tr = ME.cat(out_s2_tr, out_s2)

        out_s1_tr = self.conv2_tr(out_s2_tr)
        out_s1_tr = self.block2_tr(out_s1_tr)

        out_s1_tr = ME.cat(out_s1_tr, out_s1)

        out = self.conv1_tr(out_s1_tr)
        out = self.final(out)

        if self.normalize_features:
            out = ME.SparseTensor(F.normalize(out.F, p=2, dim=1),
                                  coordinate_map_key=out.coordinate_map_key,
                                  coordinate_manager=out.coordinate_manager)

        return out
