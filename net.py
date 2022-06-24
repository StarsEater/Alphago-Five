from numpy import pad
import torch
import torch.nn as nn
import copy
from configs import args
'''
'''

act_maps = {
    'relu': nn.ReLU(inplace=True),
    'selu': nn.SELU(inplace=True),
    'tanh': nn.Tanh(inplace=True),
    'sigmoid': nn.Sigmoid(inplace=True)
}


class ResBlock(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()
        # [b,k,sz,sz]
        # h + 2*padding - dilation*(kernel_size - 1) -1 /stride + 1
        # (h + 2*p - k )/s + 1 = nh
        # s = 1, nh = h + 2*p - k + 1
        # k = 3,p = 1, nh = h,
        conv_layer1 = nn.Conv2d(in_channels=feature_dim,
                                out_channels=args.n_res_filters,
                                stride=args.res_stride,
                                padding=args.res_padding,
                                kernel_size=args.res_kernel_size)
        batchnorm_layer1 = nn.BatchNorm2d(args.n_res_filters)
        active_layer1 = act_maps['relu']
        conv_layer2 = nn.Conv2d(in_channels=args.n_res_filters,
                                out_channels=feature_dim,
                                stride=args.res_stride,
                                padding=args.res_padding,
                                kernel_size=args.res_kernel_size)
        batchnorm_layer2 = nn.BatchNorm2d(feature_dim)
        self.sequence_module = nn.Sequential(conv_layer1, batchnorm_layer1,
                                             active_layer1, conv_layer2,
                                             batchnorm_layer2)
        self.active_layer2 = self.act_maps['relu']

    def forward(self, input_state):
        # input_state [bs, k,sz,sz]
        return self.active_layer2(input_state +
                                  self.sequence_module(input_state))


class PolicyHead(nn.Module):
    # 行为概率预测头
    def __init__(self, feature_dim, in_dense_size):
        super().__init__()
        conv_layer1 = nn.Conv2d(in_channels=feature_dim,
                                out_channels=args.n_policy_filters,
                                stride=args.policy_stride,
                                padding=args.policy_padding,
                                kernel_size=args.policy_kernel_size)
        batchnorm_layer1 = nn.BatchNorm2d(args.n_policy_filters)

        active_layer1 = act_maps[args.policy_act]
        self.post_res_p1 = nn.Sequential(conv_layer1, batchnorm_layer1,
                                         active_layer1)

        self.dense_layer1 = nn.Linear(
            args.n_policy_filters * in_dense_size * in_dense_size, 1)

    def forward(self, post_res_inputs):
        bsz = post_res_inputs.shape[0]
        input_state = self.post_res_p1(post_res_inputs)
        return self.dense_layer1(input_state.reshape(bsz, -1))


class ValueHead(nn.Module):
    # 局面价值评估头
    def __init__(self, feature_dim, in_dense_size):
        super().__init__()
        conv_layer1 = nn.Conv2d(in_channels=feature_dim,
                                out_channels=args.n_value_filters,
                                stride=args.value_stride,
                                padding=args.value_padding,
                                kernel_size=args.value_kernel_size)
        batchnorm_layer1 = nn.BatchNorm2d(args.n_value_filters)
        active_layer1 = act_maps[args.value_act1]
        dense_layer1 = nn.Linear(
            args.n_value_filters * in_dense_size * in_dense_size,
            args.dense_hidden_size)
        active_layer2 = act_maps[args.value_act2]
        dense_layer2 = nn.Linear(args.dense_hidden_size, 1)
        out_active_layer = act_maps[args.value_act_out]
        self.value_layer = nn.Sequential(conv_layer1, batchnorm_layer1,
                                         active_layer1, dense_layer1,
                                         active_layer2, dense_layer2,
                                         out_active_layer)

    def forward(self, post_res_inputs):
        return self.value_layer(post_res_inputs)


class ResNetForState(nn.Module):

    def __init__(self, pre_feature_dim: int, in_dense_size: int):
        # 默认是方阵的情况下，in_dense_size表示res block之后的第三第四维度大小
        super().__init__()
        # block前
        conv_layer1 = nn.Conv2d(in_channels=pre_feature_dim,
                                out_channels=args.n_pre_blk_filters,
                                stride=args.pre_blk_stride,
                                padding=args.pre_blk_padding,
                                kernel_size=args.pre_blk_kernel_size)
        batchnorm_layer1 = nn.BatchNorm2d(args.n_pre_blk_filters)
        active_layer1 = act_maps[args.pre_blk_act]
        self.pre_block = nn.Sequential(conv_layer1, batchnorm_layer1,
                                       active_layer1)
        # block后
        self.res_blocks = nn.ModuleList([
            copy.deepcopy(ResBlock(args.n_pre_blk_filters))
            for _ in range(args.n_res_blocks)
        ])

        # policy head
        self.policy_head = PolicyHead(args.n_res_filters, in_dense_size)
        # value  head
        self.value_head = ValueHead(args.n_res_filters, in_dense_size)

    def forward(self, input_state: torch.Tensor):
        input_state = self.pre_block(input_state)
        for module in self.res_blocks:
            input_state = module(input_state)
        return self.policy_head(input_state), self.value_head(input_state)
