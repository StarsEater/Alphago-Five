import random
import copy
from tkinter.tix import Tree
import numpy as np
from collections import deque
from configs import args


class board(object):

    def __init__(self):
        assert args.border_size % 2 == 1, print("border_size 需要为奇数")
        self.border_x = args.border_size
        self.border_y = args.border_size
        self.state_board = np.zeros((self.border_x, self.border_y))

    @property
    def move_id2move_action(self):
        return [(x, y) for x in range(self.border_x)
                for y in range(self.border_y)]

    @property
    def move_action2move_id(self):
        return {pos: idx for idx, pos in enumerate(self.move_id2move_action)}

    def _valid_move_action(self, move_action):
        x, y = move_action
        return self.state_board[x][
            y] == args.placeholder_id and 0 <= x < self.border_x and 0 <= y < self.border_y

    def next_valid_move_action(self):
        next_move_actions = list(
            filter(lambda act: self._valid_move_action(act),
                   [(i, j) for i in range(self.border_x)
                    for j in range(self.border_y)]))
        next_move_ids = [
            self.move_action2move_id[act] for act in next_move_actions
        ]
        return list(zip(next_move_actions, next_move_ids))

    def visualize_board_state(self):
        pass


class game(object):

    def __init__(self, board):
        self.board = board

        self.player_id2player_color = {
            args.first_player_id: args.first_player_color,
            args.second_player_id:
            '白' if args.first_player_color == '黑' else '黑',
            args.placeholder_id: '空'
        }
        self.player_color2player_id = {
            v: k
            for k, v in self.player_id2player_color.items()
        }

        self.border_x = self.board.border_x
        self.border_y = self.board.border_y
        self.state_board = self.board.state_board

        self.state_reset()

    def has_win(self, cur_move_action):

        x, y = cur_move_action
        cur_player_id = self.cur_player_id
        # 同一行的棋子
        row_nums = 1
        for j in range(y - 1, max(0, y - 5) - 1, -1):  # 向左
            if self.state_board[x][j] != cur_player_id:
                break
            row_nums += 1
        for j in range(y + 1, min(self.border_y, y + 5)):  # 向右
            if self.state_board[x][j] != cur_player_id:
                break
            row_nums += 1
        if row_nums >= 5:  # TODO ==5 / >=5
            print(f"行胜 {row_nums} ,落子为({x},{y})")
            return True

        # 同一列的棋子
        col_nums = 1
        for i in range(x - 1, max(0, x - 5) - 1, -1):  # 向上
            if self.state_board[i][y] != cur_player_id:
                break
            col_nums += 1
        for i in range(x + 1, min(self.border_x, x + 5)):  # 向下
            if self.state_board[i][y] != cur_player_id:
                break
            col_nums += 1
        if col_nums >= 5:  # TODO ==5 / >=5
            print(f"列胜 {col_nums},落子为({x},{y})")
            return True

        # 对角线的棋子
        diag_nums = 1
        for d in range(1, min(x, y) + 1):  # 左上
            if self.state_board[x - d][y - d] != cur_player_id:
                break
            diag_nums += 1
        for d in range(1, min(self.border_x - x, self.border_y - y, 5)):  # 右下
            if self.state_board[x + d][y + d] != cur_player_id:
                break
            diag_nums += 1
        if diag_nums >= 5:
            print(f"对角线胜 {diag_nums},落子为({x},{y})")
            return True

        # 反对角线
        bdiag_nums = 1
        for d in range(1, min(y + 1, self.border_x - x)):
            if self.state_board[x + d][y - d] != cur_player_id:
                break
            bdiag_nums += 1
        for d in range(1, min(x + 1, self.border_y - y, 5)):
            if self.state_board[x - d][y + d] != cur_player_id:
                break
            bdiag_nums += 1
        if bdiag_nums >= 5:
            print(f"反对角线胜 {bdiag_nums},落子为({x},{y})")
            return True

        return False

    def is_tie(self):
        return self.round == self.border_x * self.border_y - 1

    def is_end(self, cur_move_action):

        # 判断是否可以结束
        if self.has_win(cur_move_action):
            return True, self.cur_player_id
        if self.is_tie():
            return True, 'tie'
        return False, 'continue'

    def print_board(self):

        # 打印棋盘
        head_words = f">>>>>> 第{self.round}轮 棋盘为: <<<<"
        print(head_words)
        print(" " * 4, end=" ")
        for i in range(self.border_y):
            print(i, end=" ")
        print()
        for i in range(self.border_x):
            print(" " * 2, end=" ")
            print(i, end=" ")
            for j in range(self.border_y):
                if self.state_board[i][j] == args.placeholder_id:
                    print("_", end=" ")
                elif self.state_board[i][j] == args.first_player_id:
                    print("O", end=" ")
                else:
                    print("X", end=" ")
            print()
        print("*" * (len(head_words) + 4))

    def take_move_action(self, player, shown=True):

        next_move_action_id = self.board.next_valid_move_action()

        # 玩家决策
        (p_x, p_y), _ = player.get_action(self.state_board,
                                          next_move_action_id)

        # 落子
        self.state_board[p_x][p_y] = self.cur_player_id

        if shown:
            # 根据当前的玩家，选择不同的策略，下一步棋
            now_hand = self.player_id2player_color[self.cur_player_id]
            print(f"第{self.round}轮对局,{now_hand} 正在落子....")
            print(f"落子位置为({p_x},{p_y})")

        return p_x, p_y

    def state_update(self, cur_move_action):

        # 当前手和下一手互换
        self.cur_player_id, self.rival_player_id = self.rival_player_id, self.cur_player_id

        # 当前轮次加1
        self.round += 1

        # 更新上一手落子
        self.last_move_action = cur_move_action
        self.last_move_id = self.board.move_action2move_id[
            self.last_move_action]

        # 棋盘压入当前的队列
        self.board_state_deque.append(copy.deepcopy(self.state_board))

    def state_reset(self):
        # 当前手
        self.cur_player_id = args.first_player_id

        # 下一手
        self.rival_player_id = args.second_player_id

        # 当前轮次
        self.round = 0

        # 上一手的落子设置为无法访问
        self.last_move_action = args.init_move_action
        self.last_move_id = args.init_move_id

        # 初始化棋盘状态队列
        self.board_state_deque = deque(maxlen=args.state_board_deque_maxlen)
        for _ in range(args.state_board_deque_maxlen):
            self.board_state_deque.append(
                np.zeros((self.border_x, self.border_y)))

    def flip_move_action(self, move_action, center_pos):
        # 根据move_action=(x,y),得到对称的操作,从而扩充数据集
        # 对中心进行旋转 (border_x//2,border//2) (0~10,0~10) -> (5,5)
        # [
        #    [cos,-sin]
        #    [sin,+cos]
        # ]
        center_x, center_y = center_pos
        absolute_x, absolute_y = move_action
        relative_x, relative_y = absolute_x - center_x, absolute_y - center_y
        # 1、 旋转90度
        #  [[0,-1],[1,0]]
        # (x,y) - > (-y,x)
        rotate_90 = (-relative_y + center_x, relative_x + center_y)
        # 2、 旋转180度
        # [[-1,0],[0,-1]]
        rotate_180 = (-relative_x + center_x, -relative_y + center_y)
        # 3、 旋转270度
        # [[0,1],[-1,0]]
        rotate_270 = (relative_y + center_x, -relative_x + center_y)

        # 4、水平翻转
        horiz_symm = (relative_x + center_x, -relative_y + center_y)
        # 5、竖直翻转
        vert_symm = (-relative_x + center_x, relative_y + center_y)
        # 6、对角线翻转
        diag_symm = (relative_y + center_x, relative_x + center_y)
        # 7、反对角线翻转
        bdiag_symm = (-relative_y + center_x, -relative_x + center_y)

        pos_symm = [
            rotate_90, rotate_180, rotate_270, horiz_symm, vert_symm,
            diag_symm, bdiag_symm
        ]
        return pos_symm

    def encode_state2numpy(self):
        # 编码当前的棋盘和玩家状态为一个numpy矩阵
        # [border_size,border_size,]
        # [..,k//2]:前k轮当中当前手的棋盘状态
        # [..,k//2]:前k轮当中当前手的棋盘状态
        # [...,1]: 上一手的落子位置，其余位置为0
        # [...,1]: 剩余部分可以下的棋子
        last_k = args.state_board_deque_maxlen
        total_dim = last_k + 2
        state_board_np = np.zeros((total_dim, self.border_x, self.border_y))
        for k in range(0, last_k // 2):
            state_board_np[k][:, :] = (
                self.board_state_deque[k * 2][:, :] == self.cur_player_id)
            state_board_np[k + last_k // 2] = (
                self.board_state_deque[k * 2 +
                                       1][:, :] == self.rival_player_id)

        x, y = self.last_move_action
        state_board_np[last_k][x][y] = 1
        for i in range(self.border_x):
            for j in range(self.border_y):
                state_board_np[last_k + 1][i][j] = int(
                    self.state_board[i][j] == args.placeholder_id)
        return state_board_np

    def play_chess(self, player1, player2):
        # 先手为player1, 后手为player2
        self.state_reset()
        player_id2player_inst = {
            args.first_player_id: player1,
            args.second_player_id: player2
        }
        first_hand = self.player_id2player_color[self.cur_player_id]
        print(f"先手为 {first_hand}")

        while True:
            cur_player = player_id2player_inst[self.cur_player_id]
            cur_move_action = self.take_move_action(cur_player)
            is_end, winner = self.is_end(cur_move_action)
            if is_end:
                self.print_board()
                if winner == 'tie':
                    print('平局')
                else:
                    winner = self.player_id2player_color[winner]
                    print(f"胜者为 {winner}")
                break
            self.state_update(cur_move_action)

    def _self_play(self, mcts_player):
        # TODO 未测试
        # 先后手都是自己
        # (encoded_state, award, winner)
        self.state_reset()
        states = []
        act_probs = []
        players = []
        winner_z = []

        while True:

            states.append(self.encode_state2numpy())
            players.append(self.cur_player_id)

            next_move_action_id = self.board.next_valid_move_action()
            cur_move_action, act_prob = mcts_player.get_action(
                self.state_board, next_move_action_id)
            act_probs.append(act_prob)
            # 落子
            (p_x, p_y) = cur_move_action
            self.state_board[p_x][p_y] = self.cur_player_id

            is_end, winner = self.is_end(cur_move_action)

            if is_end:
                winner_z = [0] * len(states)
                if winner == 'tie':
                    pass
                else:
                    # 落子前的状态为必胜态，对应的，奖励1
                    winner_z[-1::-2] = 1.0
                    # 其他状态
                    winner_z[-2::-2] = -1.0

                mcts_player.reset_player()
                break
            self.state_update(cur_move_action)
        # 对应玩家落子序列，棋盘状态,action概率分布, 局面的value值
        return players, zip(states, act_probs, winner_z)


class human1:

    def get_action(self, state_board, valid_move_ids):
        return random.choice(valid_move_ids)


class human2:

    def get_action(self, state_board, valid_move_ids):
        return random.choice(valid_move_ids)


if __name__ == '__main__':
    b = board()
    g = game(b)
    # 测试对称操作
    # print(g.flip_move_action(move_action=(2, 1), center_pos=(0, 0)))
    # 测试棋盘
    print(b.next_valid_move_action())
    # 测试随机下棋
    g.play_chess(human1(), human2())

    print(g.encode_state2numpy())
'''
     0 1 2 3 4
   0 X X _ _ O
   1 O O _ _ _
   2 O X _ X O
   3 _ X X _ O
   4 O X _ _ _

   X X _ _ O
   O O _ _ _
   O X _ X O
   _ X X _ O
   O X _ _ _
'''
