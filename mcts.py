import copy
import time
import random
import numpy as np
from torch import rand
from configs import args
from game_control import Game, Board
from net import ResNetForState
from tools import stable_softmax


class TreeNode:
    # 蒙特卡洛树的节点
    def __init__(self, parent, p):
        # 父亲节点
        self._parent = parent
        # 平均动作价值
        # 从父亲节点选择动作之后对于父亲节点的价值
        # 这是由select 阶段的操作决定的
        self._q = 0
        # 本节点的先验概率
        self._p = p
        # 置信上限（不懂，这是个啥）
        self._u = 0
        # 节点被访问的总次数
        self._n_visits = 0
        # 孩子节点，表示为行为-孩子的键值对
        self._children = {}

    def select(self):
        # 选择孩子当中得分最高的孩子返回
        return max(
            self._children.items(),
            key=lambda act: act[1].get_value(args.c_puct),
        )

    def expand(self, act_probs):
        # 根据模型预测得到的行为概率分布扩展新的节点
        for act, prior in act_probs.items():
            if act not in self._children:
                self._children[act] = TreeNode(self, prior)

    def update(self, leaf_value):
        # 将leaf_value的值或者是模型预测得到的分数，或者是在对弈结束产生的最终分数，反传
        # 注意向上传的时候，数值是不断取负的
        if self._parent is not None:
            self._parent.update(-leaf_value)
        self._n_visits += 1
        self._q += (leaf_value - self._q) / self._n_visits

    def get_value(self, c_puct):
        # 获得自身的分数
        self._u = c_puct * self._p * (np.sqrt(self._parent._n_visits) /
                                      (1 + self._n_visits))
        return self._q + self._u

    @property
    def is_leaf(self):
        # 不同于数据结构的叶子节点，叶子节点定义为可以继续扩展的节点
        return len(self._children) == 0

    @property
    def is_root(self):
        # 没有父亲的节点为根节点
        return self._parent is None

    def __str__(self):
        return f"q= {self._q}, p = {self._p} ,u = {self._u}, visits = {self._n_visits}"


class MCTS:
    # 蒙特卡洛树本体
    def __init__(self, policy_value_fn=None):
        self._root = TreeNode(None, 0)
        self._policy_value_fn = policy_value_fn

    def _playout(self, game: Game):
        # 一次内心戏,实际啥也没发生
        # 效果是延展mc树
        # select 直到leaf node
        # expand leaf node
        # backup update
        node = self._root
        action = None
        # 选择
        while True:
            if node.is_leaf:
                break
            action_id, node = node.select()
            action = game.id2action[action_id]
            game.do_move(action)  # 推进局面
            game.state_update(action)

        # 先判断当前是否已经结束，否则才扩展
        if action is None:
            is_end = False
            winner = ""
        else:
            is_end, winner = game.is_end(action)
        valid_action_ids = game.valid_action_ids
        act_probs, leaf_value = self._policy_value_fn(
            game.encode_state2numpy(), valid_action_ids)
        # print("result:", action, "--", is_end, "--", winner)
        if is_end:
            # 平局
            if winner == 'tie':
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == args.first_player_id else -1.0)
        else:
            node.expand(act_probs)
        # node 在选择阶段之后，为叶子节点
        # 用模型预测当前局面的分数/游戏给出的分数
        """
                          黑
                       /      \
                     白1       白2 
                  /   |   \
                黑21 黑22  黑23
        update更新的是节点的q值
        节点的q值在select阶段被使用，是父亲选择动作节点考量之一
        如白1的q值表示黑回合选择白1的分数之一
        
        而模型对白1的预测分数表示这个局面对于白1的意义，如果是-1就是输了，
        取负就是对于黑的分数
        """
        node.update(-leaf_value)

    def get_move_probs(self, game: Game):
        for _ in range(args.n_playout):
            virtual_game = copy.deepcopy(
                game)  # 必须拷贝，因为game的局面在一次游戏后已经固定了，需要从头开始下。
            self._playout(virtual_game)

        act_visits = [(act, child._n_visits)
                      for act, child in self._root._children.items()]
        act, visits = zip(*act_visits)
        # temp todo 也不知道是个啥
        act_probs = stable_softmax(1 / args.temp * np.log(np.array(visits)))
        return act, act_probs

    def update_move_root(self, move_action):
        if move_action in self._root._children:
            self._root = self._root._children[move_action]
        else:
            self.root = TreeNode(None, 1)

    def __str__(self):
        from collections import deque
        q = deque()
        q.append(((-1, -1), self._root))
        while (q):
            h = len(q)
            for i in range(h):
                move, node = q.popleft()
                print(move,
                      "-->",
                      id(node),
                      " ",
                      node,
                      "==>",
                      id(node._parent),
                      end="|")
                for k, v in node._children.items():
                    q.append((k[0], v))
                print()
            print("*" * 6)
        return "above is mcts "


class MCTSPlayer:
    # 蒙特卡洛树玩家
    def __init__(self, policy_value_fn=None, selfplay=False):
        self.policy_value_fn = policy_value_fn
        self.mcts = MCTS(policy_value_fn)
        self._selfplay = selfplay

    def reset_player(self):
        # 重置mcts的root节点
        self.mcts.update_move_root(-1)

    def get_action(self, game: Game, think_time_record=False):
        if think_time_record:
            start = time.time()
        acts, probs = self.mcts.get_move_probs(game)
        acts = list(acts)
        # 不是_selfplay的情况下，不添加噪声
        act = np.random.choice(
            acts,
            p=args.p_d_coff * np.array(probs) +
            (1 - args.p_d_coff) * int(self._selfplay) *
            np.random.dirichlet(args.dirichlet_coff * np.ones(len(acts))))
        self.mcts.update_move_root(act)
        full_probs = np.zeros(len(game.id2action))
        full_probs[acts] = probs
        if think_time_record:
            print(f"think for {time.time()-start}")
        return game.id2action[act], full_probs


def policy_for_test(game_np, valid_ids):
    probs = np.exp(np.ones(len(valid_ids)))
    probs = probs / np.sum(probs)
    return {id: pb for id, pb in zip(valid_ids, probs)}, 0.4


if __name__ == '__main__':
    g = Game(Board())
    m = MCTS(policy_value_fn=ResNetForState())
    # for i in range(3):
    #     print(f"iteration {i}")
    #     m._playout(copy.deepcopy(g))
    #     print(m)
