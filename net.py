from dataclasses import dataclass
from numpy import save
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import os
import time
import torch.utils.data as tdata
from datetime import datetime
from configs import args
from tools import get_available_path, pickle_load
'''
'''

act_maps = {
    'relu': nn.ReLU(inplace=True),
    'selu': nn.SELU(inplace=True),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid()
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
        self.active_layer2 = act_maps['relu']

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
            args.n_policy_filters * in_dense_size * in_dense_size,
            args.border_size * args.border_size)

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
        self.value_layer1 = nn.Sequential(conv_layer1, batchnorm_layer1,
                                          active_layer1)
        self.value_layer2 = nn.Sequential(dense_layer1, active_layer2,
                                          dense_layer2, out_active_layer)

    def forward(self, post_res_inputs):
        bsz = post_res_inputs.shape[0]
        return self.value_layer2(
            self.value_layer1(post_res_inputs).reshape(bsz, -1))


class ResNetForState(nn.Module):

    def __init__(self, pre_feature_dim: int, in_dense_size: int):
        # 默认是方阵的情况下
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
        # 模型初始化
        self.init_parameters()

    def init_parameters(self):
        pass

    def forward(self,
                input_state: torch.Tensor,
                trained: bool = True,
                target_policy=None,
                target_value=None):
        assert trained and target_policy is not None and target_value is not None or not trained, print(
            "训练模式但是没有目标数据")
        input_state = self.pre_block(input_state)
        for module in self.res_blocks:
            input_state = module(input_state)
        pred_policy, pred_value = self.policy_head(
            input_state), self.value_head(input_state)
        if trained:
            policy_loss = F.cross_entropy(pred_policy, target_policy)
            value_loss = F.mse_loss(pred_value, target_value)
            return args.c_policy * policy_loss + args.c_value * value_loss, pred_policy, pred_value
        else:
            return pred_policy, pred_value

    def load_model(self, model_path):
        model_path = get_available_path(args.model_save_dir, model_path)
        if os.path.exists(model_path):
            print(f">>>> 载入模型/参数中 {model_path}")
            model_state_dict = torch.load(model_path)
            self.load_state_dict(model_state_dict['model'])
            return True
        else:
            print(f">>>> 从头开始训练")
            return False

    @torch.no_grad()
    def policy_value_fn(self, state_board_np, valid_action_ids=None):
        self.eval()
        if valid_action_ids is None:
            valid_action_ids = list(range(args.border_size * args.border_size))
        state_board_tensor = state_board_np
        if not isinstance(state_board_np, torch.Tensor):
            state_board_tensor = torch.from_numpy(
                state_board_np).float().unsqueeze(0)

        policy_res, value_res = self.forward(state_board_tensor, trained=False)
        valid_policy = policy_res[0][valid_action_ids]
        valid_policy = torch.softmax(valid_policy, dim=-1)
        value_state = value_res[0].item()
        # todo
        return {
            act_id: policy.item()
            for act_id, policy in zip(valid_action_ids, valid_policy)
        }, value_state


class SelfPlayDataset(tdata.Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.train_data = pickle_load(data_path)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        state_np, policy_np, state_value = self.train_data[index]
        state_tensor = torch.from_numpy(state_np).float()
        policy_tensor = torch.from_numpy(policy_np).float()
        value_tensor = torch.FloatTensor([state_value])
        return state_tensor, policy_tensor, value_tensor

    def get_dataloader(self):
        return tdata.DataLoader(self,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=False)


class TrainPipeline:

    def __init__(self):
        pre_feature_dim = args.state_board_deque_maxlen + 2
        in_dense_size = args.border_size
        self.model = ResNetForState(pre_feature_dim, in_dense_size)
        if args.optim_choice == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.init_lr,
                                        weight_decay=args.weight_decay)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save_model(self, epoch, _path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch
            }, _path)

    def train_step(self, train_data):
        self.model.train()
        device = args.device if torch.cuda.is_available() else 'cpu'
        print(f"model train on {device}")
        self.model.to(device)
        # 优化器
        lr_multiplier = args.lr_multiplier
        best_loss = 10**5
        for epoch in range(args.n_epochs):
            cur_lr = self.optimizer.param_groups[0]['lr']
            loss_epoch = 0
            num_in_epoch = 0
            for batched_data in train_data:
                num_in_epoch += 1
                self.optimizer.zero_grad()
                batched_state, batched_policy, batched_value = map(
                    lambda x: x.to(device), batched_data)
                old_prob, _ = self.model.policy_value_fn(batched_state)
                loss, new_prob, _ = self.model.forward(batched_state, True,
                                                       batched_policy,
                                                       batched_value)
                tmp_loss = loss.item()
                loss_epoch += tmp_loss
                loss.backward()
                self.optimizer.step()
                old_prob = torch.FloatTensor(
                    [old_prob[i] for i in range(len(old_prob))])
                kl_loss = F.kl_div(old_prob.log(),
                                   new_prob[0].softmax(dim=-1)).item()
                print(f">>> 当前损失为{tmp_loss} kl_loss 为 {kl_loss}")

            print(
                f"第{epoch}个epoch, lr = {cur_lr/num_in_epoch}, loss = {loss_epoch} ,kl_loss = {kl_loss}"
            )

            if kl_loss > args.kl_div_threshold and cur_lr > args.lr_min_threshold:
                lr_multiplier *= 1.5
            elif kl_loss < args.kl_div_threshold and cur_lr > args.lr_min_threshold:
                lr_multiplier *= 0.5

            if best_loss > loss_epoch:
                best_loss = loss_epoch
                newest_model_path = get_available_path(args.model_save_dir,
                                                       args.newest_model_path)
                # 保存最新的模型
                self.save_model(epoch, newest_model_path)
                # 每隔一定轮次保存模型
                if epoch % args.save_loop == 0:
                    now_time = datetime.now()
                    epoch_model_path = now_time + "-" + f"{epoch}.ckpt"
                    epoch_model_path = get_available_path(
                        args.model_save_dir, epoch_model_path)
                    self.save_model(epoch, epoch_model_path)

            self.update_lr(cur_lr * lr_multiplier)

    def start_train(self):
        self.model.load_model(args.newest_model_path)
        train_data_path = get_available_path(args.train_data_dir,
                                             args.newest_data_path)
        # 构建dataset 和 dataloader

        try:
            while True:
                while (not os.path.exists(train_data_path)):
                    print(">>> 目前不存在训练数据")
                    time.sleep(1000)
                dataloader = SelfPlayDataset(train_data_path).get_dataloader()
                self.train_step(dataloader)

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    # x = torch.ones((10, 5, 8, 8))
    # modelx = ResNetForState(pre_feature_dim=5, in_dense_size=8)
    # print(modelx)
    # w1, w2 = modelx(x)
    # print(w1.shape, w2.shape)
    import numpy as np
    tpl = TrainPipeline()
    tpl.start_train()
    # x = np.ones((args.state_board_deque_maxlen + 2, args.border_size,
    #              args.border_size))
    # print(x)
    # y = [0, 1, 2]
    # w = tpl.model.policy_value_fn(state_board_np=x, valid_action_ids=y)
    # print(w)
    # tpl.start_train()
