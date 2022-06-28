import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--debug", type=bool, default=False, help="是否为debug阶段")
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
                    default=5,
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
                    default=0.3,
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
                    default=2,
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
parser.add_argument("--c_policy",
                    type=float,
                    default=1.0,
                    help="损失函数中policy的系数")
parser.add_argument("--c_value",
                    type=float,
                    default=1.0,
                    help="损失函数中value的系数")

# transformer
# todo
# 模型保存/载入路径
parser.add_argument("--model_save_dir",
                    type=str,
                    default='./model_saves',
                    help="模型保存的目录路径")
# 数据保存/载入路径
parser.add_argument("--train_data_dir",
                    type=str,
                    default='./train_data',
                    help="数据保存的目录路径")
# 最新的模型保存文件名称
parser.add_argument("--newest_model_path",
                    type=str,
                    default='newest_model.ckpt',
                    help="模型保存的目录路径")
# 最新的数据保存文件名称
parser.add_argument("--newest_data_path",
                    type=str,
                    default='newest_data.pkl',
                    help="数据保存的目录路径")


# 训练过程中的超参数
parser.add_argument("--n_epochs",
                    type=int,
                    default=2,
                    help="训练轮次，默认为1000")
parser.add_argument("--batch_size",
                    type=int,
                    default=8,
                    help="批大小，默认为1000")

parser.add_argument("--device",
                    type=int,
                    default=0,
                    help="模型映射的GPU设备号")
parser.add_argument("--optim_choice",
                    type=str,
                    choices=['adam', 'sgd'],
                    default='adam',
                    help="优化器的选择")
parser.add_argument("--init_lr",
                    type=float,
                    default=1e-3,
                    help="初始学习率")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1.0,
                    help="权重衰减L2系数")
# TODO
parser.add_argument("--kl_div_threshold",
                    type=float,
                    default=0.02,
                    help="权重衰减L2系数")
parser.add_argument("--lr_multiplier",
                    type=float,
                    default=1.0,
                    help="学习率乘法器，每次和学习率做乘法")
parser.add_argument("--lr_min_threshold",
                    type=float,
                    default=1e-5,
                    help="学习率下限")
parser.add_argument("--save_loop",
                    type=int,
                    default=10,
                    help="每过多少轮次保存一次")

# 数据收集阶段
parser.add_argument("--buffer_size",
                    type=int,
                    default=100000,
                    help="存储数据的容量")
parser.add_argument("--n_games",
                    type=int,
                    default=100,
                    help="收集的总轮次，每次都会加载最新的模型")
parser.add_argument("--save_data_loop",
                    type=int,
                    default=4,
                    help="保存收集的数据的间隔")


args = parser.parse_args()
