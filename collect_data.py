import copy
import time
from collections import deque
from game_control import Board, Game, self_play
from configs import args
from mcts import MCTSPlayer
from net import ResNetForState
from tools import get_available_path, pickle_dump, pickle_load


class CollectPipeline:

    def __init__(self):
        self.board = Board()
        self.game = Game(self.board)
        self.data_buffer = deque(maxlen=args.buffer_size)
        self.iters = 0
        pre_feature_dim = args.state_board_deque_maxlen + 2
        in_dense_size = args.border_size
        self.model = ResNetForState(pre_feature_dim, in_dense_size)
        self.mcts_player = MCTSPlayer(
            policy_value_fn=self.model.policy_value_fn, selfplay=True)

    def get_equi_data(self, play_data):
        extend_data = []
        for state_board, mcts_prob, winner in play_data:
            center_pos = (args.border_size // 2, args.border_size // 2)
            state_board = state_board.transpose(1, 2, 0)  # x,y,f
            state_board_copys = [copy.deepcopy(state_board) for _ in range(8)]
            mcts_prob_copys = [copy.deepcopy(mcts_prob) for _ in range(8)]
            winner_copys = [copy.deepcopy(winner) for _ in range(8)]
            for i in range(args.border_size):
                for j in range(args.border_size):
                    equal_moves = self.game.flip_move_action((i, j),
                                                             center_pos)
                    for state_board_cp, (x, y), mcts_prob_cp, winner_cp in zip(
                            state_board_copys, equal_moves, mcts_prob_copys,
                            winner_copys):
                        state_board_cp[i][j] = state_board[x][y]
                        i_j, x_y = self.board.move_action2move_id[(
                            i, j)], self.board.move_action2move_id[(x, y)]
                        mcts_prob_cp[i_j] = mcts_prob[x_y]
            for i, v in enumerate(state_board_copys):
                state_board_copys[i] = v.transpose(2, 0, 1)
            extend_data.extend(
                list(zip(state_board_copys, mcts_prob_copys, winner_copys)))
        return extend_data

    def start_collect(self):

        for i in range(args.n_games):
            start = time.time()
            # 数据收集之前先载入最新的模型
            if not self.model.load_model(args.newest_model_path):
                print("模型不存在")

            # 载入已有的数据
            data_path = get_available_path(args.train_data_dir,
                                           args.newest_data_path)
            old_data = pickle_load(data_path)

            # players, zip(states, act_probs, winner_z)
            winners, play_data = self_play(self.game, self.mcts_player)
            play_data = list(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(old_data)
            self.data_buffer.extend(play_data)
            if i % args.save_data_loop == 0:
                print(
                    f">> game_epoch {i} 总计{len(self.data_buffer)}条数据存入, 耗时 {time.time()-start}"
                )
                pickle_dump(self.data_buffer, data_path)


if __name__ == '__main__':
    cp = CollectPipeline()
    cp.start_collect()
    # data_path = get_available_path(args.train_data_dir, args.newest_data_path)
    # test_data = pickle_load(data_path)
    # print(test_data[0])
