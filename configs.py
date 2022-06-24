import argparse

parser = argparse.ArgumentParser()

# 和棋盘强相关
parser.add_argument("--border_size", type=int, default=3, help="表示棋盘的边长,需要为奇数")
parser.add_argument("--placeholder_id",
                    type=int,
                    default=0,
                    help="未落子点的编号,默认为0")
# 和游戏局面强相关
parser.add_argument("--init_move_id",
                    type=int,
                    default=-100,
                    help="初始动作编号,需要设置非法，默认为-100")
parser.add_argument("--init_move_action",
                    type=tuple,
                    default=(-1, -1),
                    help="初始动作落子,需要设置为非法，默认为(-1,-1)")
parser.add_argument("--first_player_id",
                    type=int,
                    default=1,
                    help="先手的编号,默认为1")
parser.add_argument("--first_player_color",
                    type=str,
                    default="黑",
                    help="先手的棋子颜色,默认为黑色（O）")
parser.add_argument("--second_player_id",
                    type=int,
                    default=-1,
                    help="后手的编号,默认为-1")

# 棋盘和游戏连接的部分
parser.add_argument("--state_board_deque_maxlen",
                    type=int,
                    default=2,
                    help="保存棋盘的双端队列的最大容量，默认为2,建议为偶数")

# 蒙特卡洛树搜索部分 # TODO
parser.add_argument("--c_puct",
                    type=float,
                    default=2,
                    help="表示在mcts过程中探索的权重，衡量Q和U之间的权重，默认为5")
parser.add_argument("--n_playout",
                    type=int,
                    default=100,
                    help="在蒙特卡洛树搜索中为了得到当前节点的下一步动作的概率，假想模拟的次数，默认为100")
# todo 查询一下文献
parser.add_argument("--temp",
                    type=float,
                    default=0.15,
                    help="温度系数，在返回mcts的行动概率使用的超参数，默认为0.15")
# todo 查询一下文献
parser.add_argument("--dirichlet_coff",
                    type=float,
                    defult=0.3,
                    help="返回dirichlet系数,在行为概率分布中添加dirichlet噪声")
parser.add_argument("--p_d_coff",
                    type=float,
                    default=0.75,
                    help="平衡行为概率和dirichlet噪声的系数，默认为0.75")
# 模型部分
# resnet
# 输入block之前
parser.add_argument("--pre_blk_kernel_size",
                    type=int,
                    default=3,
                    help="输入block之前卷积核的尺寸")
parser.add_argument("--n_pre_blk_filters",
                    type=int,
                    default=64,
                    help="输入block之前卷积核的个数")
parser.add_argument("--pre_blk_stride", type=int, default=1, help="输入block之前卷积核的步进")
parser.add_argument("--pre_blk_padding", type=int, default=1, help="输入block之前卷积核的填充")
parser.add_argument("--pre_blk_act",
                    type=str,
                    choices=['relu', 'selu', 'gelu'],
                    default='relu',
                    help="输入block之前激活函数的选择")

# block 内部
parser.add_argument("--res_kernel_size",
                    type=int,
                    default=3,
                    help="resnet中卷积核的尺寸")
parser.add_argument("--n_res_filters",
                    type=int,
                    default=64,
                    help="resnet中卷积核的个数")
parser.add_argument("--res_stride", type=int, default=1, help="resnet中卷积核的步进")
parser.add_argument("--res_padding", type=int, default=1, help="resnet中卷积核的填充")
parser.add_argument("--n_res_blocks",
                    type=int,
                    default=19,
                    help="resnet中block的个数")
parser.add_argument("--res_act",
                    type=str,
                    choices=['relu', 'selu', 'gelu'],
                    default='relu',
                    help="resnet中激活函数的选择")

# policy 头
parser.add_argument("--n_policy_filters",
                    type=int,
                    default=2,
                    help="policy头的卷积核个数")
parser.add_argument("--policy_kernel_size",
                    type=int,
                    default=1,
                    help="policy头的卷积和尺寸")
parser.add_argument("--policy_stride",
                    type=int,
                    default=1,
                    help="policy头中卷积核的步进")
parser.add_argument("--policy_padding",
                    type=int,
                    default=0,
                    help="policy头中卷积核的填充")
parser.add_argument("--policy_act",
                    type=str,
                    choices=['relu', 'selu', 'gelu'],
                    default='relu',
                    help="policy头中激活函数的选择")

# value 头
parser.add_argument("--n_value_filters",
                    type=int,
                    default=1,
                    help="value头的卷积核个数")
parser.add_argument("--value_kernel_size",
                    type=int,
                    default=1,
                    help="value头的卷积和尺寸")
parser.add_argument("--value_stride",
                    type=int,
                    default=1,
                    help="value头中卷积核的步进")
parser.add_argument("--value_padding",
                    type=int,
                    default=0,
                    help="value头中卷积核的填充")
parser.add_argument("--value_act1",
                    type=str,
                    choices=['relu', 'selu', 'gelu'],
                    default='relu',
                    help="value头中前面一个激活函数的选择")
parser.add_argument("--dense_hidden_size",
                    type=int,
                    default=128,
                    help="value头中dense层的隐层输出维度")
parser.add_argument("--value_act2",
                    type=str,
                    choices=['relu', 'selu', 'gelu'],
                    default='relu',
                    help="value头中后面一个激活函数的选择")
parser.add_argument("--value_act_out",
                    type=str,
                    choices=['relu', 'selu', 'gelu'],
                    default='tanh',
                    help="value头输出映射激活函数的选择，保证数值在[-1,1]")

# transformer
# todo
args = parser.parse_args()
