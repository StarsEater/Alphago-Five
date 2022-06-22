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

args = parser.parse_args()
