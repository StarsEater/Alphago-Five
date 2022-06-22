class TreeNode:
    # 蒙特卡洛树的节点
    def __init__(self, parent, prior):
        self._parent = parent 
        self._Q = 0
        self._U = 0
        self._N = 0
        self.prior = prior 
        self._children = {}
    
    def get_value()


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
    def __init__(self) :
        pass
    
    def select(self):
        pass 

    def expand(self):
        pass 
    
    def backup(self):
        pass 

    

class MCTS_player:
    def __init__(self):
        pass 

    def get_action(self):
        actions = []
        actions_probs = []
        return action, action_probs
    
    def 

    