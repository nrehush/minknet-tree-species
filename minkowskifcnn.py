# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import torch.nn as nn
import MinkowskiEngine as ME
        
class MinkowskiFCNN(ME.MinkowskiNetwork):
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3,
    ):
        ME.MinkowskiNetwork.__init__(self, D)

        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.mlp1 = self.get_mlp_block(in_channel, channels[0])
        self.conv1 = self.get_conv_block(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = self.get_conv_block(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv3 = self.get_conv_block(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv4 = self.get_conv_block(
            channels[3],
            channels[4],
            kernel_size=kernel_size,
            stride=2,
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(
                channels[1] + channels[2] + channels[3] + channels[4],
                embedding_channel // 4,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 4,
                embedding_channel // 2,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 2,
                embedding_channel,
                kernel_size=3,
                stride=2,
            ),
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            self.get_mlp_block(embedding_channel * 2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

        # No, Dropout, last 256 linear, AVG_POOLING 92%

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.TensorField):
        x = self.mlp1(x)
        y = x.sparse()

        y = self.conv1(y)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        y4 = self.pool(y)

        x1 = y1.slice(x)
        x2 = y2.slice(x)
        x3 = y3.slice(x)
        x4 = y4.slice(x)

        x = ME.cat(x1, x2, x3, x4)

        y = self.conv5(x.sparse())
        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)

        return self.final(ME.cat(x1, x2)).F

   
