import copy
import random

import numpy as np
import time

from rllib2.game.base.game_interface import GymLikeWrapper
from . import config
from .gridworld_base import GridWorldEnv
from .maze import Maze


def get_manhattan_distance(source, target):
    sx, sy = source
    tx, ty = target
    return abs(tx - sx) + abs(ty - sy)


class MyGridWorldEnv(GridWorldEnv):

    def __init__(self, n_width=10, n_height=10, u_size=40, \
                 default_reward=-1, default_type=0, windy=False):
        super().__init__(n_width, n_height, u_size, default_reward, \
                         default_type, windy)

    def render(self, duration=None):
        if config.render:
            self._render()
            sleep_time = config.render_duration if duration is None else duration
            time.sleep(sleep_time)

    def reset(self):
        # cand_start_pos = [(3,5), (5,3), (7,3), (7,1), (2,1)]
        # cand_start_pos = [(0,0), (0,9), (9,0), (9,9), (3,5)]
        self.start = (3, 5)
        # self.start = random.choice(cand_start_pos)
        self.bad_pos = [(3, 2), (3, 6), (5, 2), (6, 2), (8, 3),
                        (8, 4), (6, 4), (5, 5), (6, 5)]
        self.good_ends = [(5, 4)]
        self.types = [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                      (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                      (8, 7, 1)]  # , (4,1,1), (7,2,1), (8,7,1), (9,7,1), (4,8,1)]
        self.disappeared_types = []
        self.refresh()
        return self._reset()

    def reset_for_eval(self):
        cand_start_pos = [(0, 0), (0, 9), (9, 0), (9, 9), (3, 5)]
        self.start = random.choice(cand_start_pos)
        self.bad_pos = [(3, 2), (3, 6), (5, 2), (6, 2), (8, 3),
                        (8, 4), (6, 4), (5, 5), (6, 5)]
        self.good_ends = [(5, 4)]
        self.types = [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                      (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                      (8, 7, 1)]  # , (4,1,1), (7,2,1), (8,7,1), (9,7,1), (4,8,1)]
        self.disappeared_types = []
        self.refresh()
        return self._reset()

    def step(self, action):
        target = self.good_ends[0]
        old_position = self._state_to_xy(self.state)
        old_dis = get_manhattan_distance(old_position, target)
        ret = self._step(action)
        new_position = self._state_to_xy(self.state)
        new_dis = get_manhattan_distance(new_position, target)
        if new_dis < old_dis:
            ret[1] += 0.01
        self.render(duration=config.step_duration)
        return ret

    def refresh(self):
        self.grids.reset()
        self.ends = self.good_ends
        for x, y in self.disappeared_types:
            old_obstacle = (x, y, 1)
            if old_obstacle in self.types:
                self.types.remove(old_obstacle)
        self.rewards = []
        for x, y in self.bad_pos:
            self.rewards.append((x, y, -20))
        for x, y in self.good_ends:
            self.rewards.append((x, y, 100))
        self.refresh_setting()

    def parse_action(self, pos, next_pos):
        dx = next_pos[0] - pos[0]
        if dx == -1:
            return 0
        elif dx == 1:
            return 1
        dy = next_pos[1] - pos[1]
        if dy == 1:
            return 2
        elif dy == -1:
            return 3
        raise RuntimeError("Not Valid Next Position", pos, next_pos)

class RandomStartEndGridWorld(MyGridWorldEnv):

    def __init__(self, n_width=10, n_height=10, u_size=40, \
                 default_reward=-1, default_type=0, windy=False):
        self.all_grids = set()
        self.easy = False
        for x in range(n_width):
            for y in range(n_height):
                self.all_grids.add((x, y))
        super().__init__(n_width, n_height, u_size, default_reward, \
                         default_type, windy)

    def reset(self, start=None, end=None):
        if config.render:
            self._render(close=True)
        self.bad_pos = []
        if self.easy:
            self.types = [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                          (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                          (8, 7, 1)]  # , (4,1,1), (7,2,1), (8,7,1), (9,7,1), (4,8,1)]
        else:
            self.types = [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                          (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                          (8, 7, 1), (1, 2, 1), (2, 2, 1), (3, 2, 1), (4, 2, 1),
                          (5, 2, 1), (6, 2, 1), (7, 2, 1), (7, 3, 1), (7, 4, 1), (7, 5, 1),
                          (7, 6, 1), (4, 8, 1), (4, 9, 0)]
        # self.disappeared_types = [(4,6),]# (2,7)]
        self.disappeared_types = []
        for x, y, t in self.types:
            if (x, y) in self.all_grids:
                self.all_grids.remove((x, y))
        s, t = random.sample(self.all_grids, 2)
        if start is None:
            self.start = copy.deepcopy(s)
        else:
            self.start = start
        if end is None:
            self.good_ends = [copy.deepcopy(t)]
        else:
            self.good_ends = [end]
        # self.start = (1,3)
        # self.good_ends = [(5,4)]
        self.refresh()
        return self._reset()

    def reset_for_eval(self):
        return self.reset()

    def step(self, action):
        ret = self._step(action)
        self.render(duration=config.step_duration)
        return ret

    def refresh(self):
        self.grids.reset()
        self.ends = self.good_ends
        for x, y in self.disappeared_types:
            old_obstacle = (x, y, 1)
            if old_obstacle in self.types:
                self.types.remove(old_obstacle)
        self.rewards = []
        for x, y in self.bad_pos:
            self.rewards.append((x, y, -20))
        for x, y in self.good_ends:
            self.rewards.append((x, y, 0))
        self.refresh_setting()


class RandomGridWorldEnv(MyGridWorldEnv):

    def __init__(self, n_width=10, n_height=10, u_size=40, \
                 default_reward=0, default_type=0, windy=False):
        super().__init__(n_width, n_height, u_size, default_reward, \
                         default_type, windy)

    def reset(self):
        cand_start_pos = [(random.randint(0, 4), random.randint(0, 4)), \
                          (random.randint(0, 4), random.randint(5, 9)), \
                          (random.randint(5, 9), random.randint(0, 4))]
        sx, sy = random.choice(cand_start_pos)
        tx, ty = random.randint(5, 9), random.randint(5, 9)
        self.start = (sx, sy)
        self.good_ends = [(tx, ty)]
        self.bad_pos = []
        self.disappeared_types = []
        self.refresh()
        return self._reset()

    def reset_for_eval(self):
        sx, sy = random.randint(5, 9), random.randint(5, 9)
        tx, ty = random.randint(0, 4), random.randint(0, 4)
        self.start = (sx, sy)
        self.good_ends = [(tx, ty)]
        self.bad_pos = []
        self.disappeared_types = []
        self.refresh()
        return self._reset()


class BigRandomStartEndGridWorld(RandomStartEndGridWorld):

    def __init__(self, n_width=100, n_height=100, u_size=10, \
                 default_reward=-1, default_type=0, windy=False):
        self.all_grids = set()
        self.easy = False
        for x in range(n_width):
            for y in range(n_height):
                self.all_grids.add((x, y))
        super().__init__(n_width, n_height, u_size, default_reward, \
                         default_type, windy)

    def reset(self, start=None, end=None):
        # start = (1, 3)
        # end = (33,46)
        # start = (31, 10)
        # end = (45, 33)
        # start = (0, 0)
        # end = (self.n_width-1, self.n_height-1)
        # start = (0, self.n_height-1)
        # end = (self.n_width-1, 0)
        if config.render:
            self._render(close=True)
        self.bad_pos = []
        if self.easy:
            self.types = [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                          (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                          (8, 7, 1)]  # , (4,1,1), (7,2,1), (8,7,1), (9,7,1), (4,8,1)]
        else:
            self.types = [(0, 7, 1), (4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                          (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                          (8, 7, 1), (1, 2, 1), (2, 2, 1), (3, 2, 1), (4, 2, 1),
                          (5, 2, 1), (6, 2, 1), (7, 2, 1), (7, 3, 1), (7, 4, 1), (7, 5, 1),
                          (7, 6, 1), (4, 8, 1), (4, 9, 1), (9, 7, 0)]
            copy_types = []
            random.seed(1000)
            for i in range(int(self.n_width / 10)):
                for j in range(int(self.n_height / 10)):
                    if i == 0 and j == 0:
                        continue
                    if (i + j) % 4 == 0:
                        for x, y, t in self.types:
                            if random.random() < 0.3:
                                continue
                            copy_types.append((x + i * 10, y + j * 10, 1))
                    elif (i + j) % 4 == 2:
                        for x, y, t in self.types:
                            if random.random() < 0.2:
                                continue
                            copy_types.append((j * 10 - x + 9, i * 10 - y + 9, 1))
                    elif (i + j) % 4 == 1:
                        for x, y, t in self.types:
                            if random.random() < 0.3:
                                continue
                            copy_types.append((y + i * 10, x + j * 10, 1))
                    else:
                        for x, y, t in self.types:
                            if random.random() < 0.2:
                                continue
                            copy_types.append((j * 10 - y + 9, i * 10 - x + 9, 1))
            # add new obstacle in copy_types
            # copy_types.append((9,13,1))
            self.types.extend(copy_types)
            for x, y in [(4, 49), (44, 49)]:
                if (x, y, 1) in self.types:
                    self.types.remove((x, y, 1))
        # self.disappeared_types = [(34,45)]
        self.disappeared_types = []
        for x, y, t in self.types:
            if (x, y) in self.all_grids:
                self.all_grids.remove((x, y))
        random.seed(int(time.time()))
        s, t = random.sample(self.all_grids, 2)
        if start is None:
            self.start = copy.deepcopy(s)
        else:
            self.start = start
        if end is None:
            self.good_ends = [copy.deepcopy(t)]
        else:
            self.good_ends = [end]
        # self.start = (1,3)
        # self.good_ends = [(5,4)]
        self.refresh()
        return self._reset()


class LargeGridWorldEnv(MyGridWorldEnv):
    def __init__(self, n_width=28, n_height=28, u_size=20,
                 default_reward=-0.1, default_type=0, windy=False):
        self.n_width = n_width
        self.n_height = n_height
        self.m = Maze(n_height, n_width)
        self.m.create_maze(True)
        self.initialtype = self.get_type()
        self.start = [(2, 2)]
        self.good_ends = [(3, 4)]
        super().__init__(n_width, n_height, u_size, default_reward, \
                         default_type, windy)
        self.bad_pos = []
        self.disappeared_types = []
        self.image = self.m.get_image()

    def get_value_map(self):
        value_map = np.zeros((self.n_height, self.n_width), dtype=int)
        for end in self.good_ends:
            value_map[end[0], end[1]] = 10
        return value_map

    def get_image(self):
        return self.m.get_image()

    def create_maze(self, easy=True):
        self.m.create_maze(easy)

    def get_type(self):
        types = self.m.get_type()
        for i in range(self.n_width):  # width 和 height 一样，所以可以这么搞
            types.append((0, i, 1))
            types.append((self.n_height - 1, i, 1))
            types.append((i, 0, 1))
            types.append((i, self.n_width - 1, 1))
        return types

    def reset(self, start=None, end=None):
        """
        default reset
        """
        if config.render:
            self._render(close=True)
        self.bad_pos = []
        self.types = self.get_type()
        self.disappeared_types = []
        if start is not None:
            self.start = start
        if end is not None:
            self.good_ends = [end]
        self.refresh()
        return self._reset()

    def initial_end(self):
        end_x = None
        end_y = None
        obstacle = True
        while obstacle:
            end_x = np.random.randint(1, self.n_width - 2)
            end_y = np.random.randint(1, self.n_height - 2)
            obstacle = (self.grids.get_type(end_x, end_y) == 1)
        self.good_ends = [(end_x, end_y)]
        return self.good_ends

    def initial_start(self, distance=10):
        obstacle = True
        start_x = None
        start_y = None
        end_x = self.good_ends[0][0]
        end_y = self.good_ends[0][1]
        while obstacle:
            start_x = np.random.randint(1, self.n_width - 2)
            start_y = np.random.randint(1, self.n_height - 2)
            obstacle = (self.grids.get_type(start_x, start_y) == 1) or \
                       (end_x == start_x and end_y == start_y) or \
                       abs(start_x - end_x) + abs(start_y - end_y) > distance  # 设置距离长度，用于后期课程学习
        self.start = (start_x, start_y)
        return self.start

    def reset_for_eval(self):
        return self.reset()


def CreateGridWorld(n_width=8, n_height=8):
    # env = MyGridWorldEnv()
    # env = RandomStartEndGridWorld()
    # env = LargeGridWorldEnv(n_width,n_height)
    # env = RandomGridWorldEnv()
    env = BigRandomStartEndGridWorld()
    return env


class GridWorld(GymLikeWrapper):

    def __init__(self, **kwargs):
        super().__init__()
        imsize = 28
        if 'imsize' in kwargs.keys():
            imsize = kwargs['imsize']
        self.env = CreateGridWorld(imsize, imsize)
        self.observation = self.env.state
        self.view_size = 5

    def reset_for_eval(self):
        self.observation = self.env.reset_for_eval()

    def get_game_model_info(self, team_id, model_id):
        if config.state_type == 1:
            return 2, 4
        elif config.state_type == 2:
            return [self.env.n_width, self.env.n_height, 1], 4
        elif config.state_type == 3:
            return [self.env.n_width, self.env.n_height, 4], 4
        elif config.state_type == 4:
            return [2 * self.view_size + 1, 2 * self.view_size + 1, 4], 4
        elif config.state_type == 5:
            return [2 * self.view_size + 1, 2 * self.view_size + 1, 1], 4
        elif config.state_type == 6:
            return 4, 4
        elif config.state_type == 7:
            return self.env.n_width * 2 + self.env.n_height * 2, 4
        elif config.state_type == 8:
            return [[2 * self.view_size + 1, 2 * self.view_size + 1, 1],
                    [self.env.n_width * 2 + self.env.n_height * 2], ], 4
        elif config.state_type == 9:
            return [[2 * self.view_size + 1, 2 * self.view_size + 1, 1],
                    [self.env.n_width * 2 + self.env.n_height * 2], ], 4
        else:
            raise RuntimeError("Not valid state space dimension and action space dimension")

    def get_observation(self):
        self.observation = self.env._state_to_xy(self.env.state)
        return self.observation

    def get_relative_image_state(self):
        state = np.zeros((2 * self.view_size + 1, 2 * self.view_size + 1, 1), dtype=np.float32)
        # agent is always in center
        x, y = self.observation
        ax, ay = int(x), int(y)
        # bad pos
        for x, y in self.env.bad_pos:
            bx, by = x - ax + self.view_size, y - ay + self.view_size
            if self.is_in_view(bx, by):
                state[bx][by][0] = -0.5
        # obstacle
        for x, y, t in self.env.types:
            if t == 1:
                ox, oy = x - ax + self.view_size, y - ay + self.view_size
                if self.is_in_view(ox, oy):
                    state[ox][oy][0] = -1
        # target
        for x, y in self.env.good_ends:
            tx, ty = self.clip_in_view(x - ax, y - ay)
            state[tx][ty][0] = 1
        return state

    def get_raw_pos_state(self):
        state = np.zeros(4, dtype=np.float32)
        x, y = self.observation
        ax, ay = int(x), int(y)
        state[0], state[1] = ax, ay
        for x, y in self.env.good_ends:
            state[2], state[3] = x, y
        return state

    def get_state(self, team_id, member_id, observation=None, poster_process=False):
        if observation is None:
            self.observation = np.array(self.env._state_to_xy(self.env.state), dtype=np.float32)
        else:
            self.observation = observation
        if config.state_type == 1:
            state = self.observation / 10.
        elif config.state_type == 2:
            state = np.zeros((self.env.n_width, self.env.n_height, 1), dtype=np.float32)
            # obstacle
            for x, y, t in self.env.types:
                if t == 1:
                    state[x][y][0] = -1
            # bad_pos
            for x, y in self.env.bad_pos:
                state[x][y][0] = -0.5
            # target position
            for x, y in self.env.good_ends:
                state[x][y][0] = 1
            # now position
            x, y = self.observation
            state[int(x)][int(y)][0] = 0.5
        elif config.state_type == 3:
            state = np.zeros((self.env.n_width, self.env.n_height, 4), dtype=np.float32)
            # agent channel
            x, y = self.observation
            state[int(x)][int(y)][0] = 1
            # target channel
            for x, y in self.env.good_ends:
                state[x][y][1] = 1
            # bad pos channel
            for x, y in self.env.bad_pos:
                state[x][y][2] = 1
            # obstacle channel
            for x, y, t in self.env.types:
                if t == 1:
                    state[x][y][3] = 1
        elif config.state_type == 4:
            state = np.zeros((2 * self.view_size + 1, 2 * self.view_size + 1, 4), dtype=np.float32)
            # agent is always in center
            x, y = self.observation
            ax, ay = int(x), int(y)
            # target channel
            for x, y in self.env.good_ends:
                tx, ty = self.clip_in_view(x - ax, y - ay)
                state[tx][ty][0] = 1
            # bad pos channel
            for x, y in self.env.bad_pos:
                bx, by = x - ax + self.view_size, y - ay + self.view_size
                if self.is_in_view(bx, by):
                    state[bx][by][1] = 1
            # obstacle channel
            for x, y, t in self.env.types:
                if t == 1:
                    ox, oy = x - ax + self.view_size, y - ay + self.view_size
                    if self.is_in_view(ox, oy):
                        state[ox][oy][2] = 1
            # bound channel
            for x in range(2 * self.view_size + 1):
                for y in range(2 * self.view_size + 1):
                    dx, dy = x - self.view_size + ax, y - self.view_size + ay
                    if self.is_outside(dx, dy):
                        state[x][y][3] = 1
        elif config.state_type == 5:
            state = self.get_relative_image_state()
        elif config.state_type == 6:
            state = self.get_raw_pos_state()
        elif config.state_type == 7:
            state = np.zeros(self.env.n_width * 2 + self.env.n_height * 2, dtype=np.float32)
            x, y = self.observation
            ax, ay = int(x), int(y)
            state[ax] = 1.
            state[self.env.n_width + ay] = 1.
            for x, y in self.env.good_ends:
                state[self.env.n_width + self.env.n_height + x] = 1.
                state[self.env.n_width * 2 + self.env.n_height + y] = 1.
        elif config.state_type == 8:
            state = []
            state.append(self.get_relative_image_state())
            raw_pos_state = self.get_raw_pos_state()
            state2 = np.zeros(self.env.n_width * 2 + self.env.n_height * 2, dtype=np.float32)
            state2[int(raw_pos_state[2] - raw_pos_state[0]) + self.env.n_width] = 1.
            state2[self.env.n_width * 2 + int(raw_pos_state[3] - raw_pos_state[1]) + self.env.n_height] = 1.
            state.append(state2)
            state = np.array(state)
        elif config.state_type == 9:
            state = []
            state1 = np.zeros((2 * self.view_size + 1, 2 * self.view_size + 1, 1), dtype=np.float32)
            # agent is always in center
            x, y = self.observation
            ax, ay = int(x), int(y)
            # bad pos
            for x, y in self.env.bad_pos:
                bx, by = x - ax + self.view_size, y - ay + self.view_size
                if self.is_in_view(bx, by):
                    state1[bx][by][0] = -0.5
            # obstacle
            for x, y, t in self.env.types:
                if t == 1:
                    ox, oy = x - ax + self.view_size, y - ay + self.view_size
                    if self.is_in_view(ox, oy):
                        state1[ox][oy][0] = -1
            state.append(state1)
            # state2
            raw_pos_state = self.get_raw_pos_state()
            state2 = np.zeros(self.env.n_width * 2 + self.env.n_height * 2, dtype=np.float32)
            state2[int(raw_pos_state[2] - raw_pos_state[0]) + self.env.n_width] = 1.
            state2[self.env.n_width * 2 + int(raw_pos_state[3] - raw_pos_state[1]) + self.env.n_height] = 1.
            state.append(state2)
            state = np.array(state)
        return state, None

    def handle_states(self, states):
        if config.state_type in [8, 9]:
            x1 = np.array([state[0] for state in states])
            x2 = np.array([state[1] for state in states])
            x = [x1, x2]
        else:
            x = np.array(states)
        return x

    def clip_in_view(self, dx, dy):
        new_x = max(min(dx + self.view_size, 2 * self.view_size), 0)
        new_y = max(min(dy + self.view_size, 2 * self.view_size), 0)
        return new_x, new_y

    def is_in_view(self, dx, dy):
        return dx >= 0 and dx < 2 * self.view_size + 1 and dy >= 0 and dy < 2 * self.view_size + 1

    def is_outside(self, dx, dy):
        return dx < 0 or dx >= self.env.n_width or dy < 0 or dy >= self.env.n_height

    def step(self, action, team_id, member_id):
        self.observation, reward, done, info = self.env.step(action)
        log = self._logpack()
        log.add_scalar("succ", info["succ"])
        return reward, done, done, log
