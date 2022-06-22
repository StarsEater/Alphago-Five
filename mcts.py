import numpy as np 
from configs import args 
class TreeNode:
    # 蒙特卡洛树的节点
    def __init__(self, parent, p):
        # 父亲节点
        self._parent = parent 
        # 平均动作价值
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
        return max(self._children.items(), key = lambda act: act[1].get_value(c_puct),)
    
    def expand(self, action_prior):
        # 根据模型预测得到的行为概率分布扩展新的节点
        for act, prior in action_prior.items():
            if act not in self._children:
                self._children[act] = TreeNode(self, prior)
    
    def update(self, leaf_value):
        # 将leaf_value的值或者是模型预测得到的分数，或者是在对弈结束产生的最终分数，反传
        if self._parent:
            self.update(self._parent, leaf_value)
        self._n_visits +=1
        self._q += self._n_visits*(leaf_value-self._q)/(self._n_visits+1)

    def get_value(self, c_puct):
        # 获得自身的分数
        self._u = c_puct * self._p * (np.sqrt(self._parent._n_visits)/(1+self._n_visits))
        return self._q + self._u 
    
    @property
    def is_leaf(self):
        # 不同于数据结构的叶子节点，叶子节点定义为可以继续扩展的节点
        return len(self._children) == 0
    
    @property
    def is_root(self):
        # 没有父亲的节点为根节点
        return self._parent is None 


class MCTS:
    # 蒙特卡洛树本体
    def __init__(self):
        self._root = TreeNode(None,0)
    
    
    def _playout(self):
        # 一次内心戏，实际啥也没发生
        # select 直到leaf node
        # expand leaf node 
        # backup update 
        




    

class MCTS_player:
    def __init__(self):
        pass 

    def get_action(self):
        actions = []
        actions_probs = []
        return action, action_probs
    
    def 

    