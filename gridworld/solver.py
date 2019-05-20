from collections import deque

import math
import numpy as np
import time

from rllib2.algos.factory import create_instance
from rllib2.config import config as PkgConfig
from rllib2.framework.utils import get_model_save_path
from rllib2.utils.astar import AStar
from . import config
from .game_interface import GridWorld

DETECTION_THRESHOLD = 10

def is_equal(n1, n2):
    return n1[0] == n2[0] and n1[1] == n2[1]

# astar solver
class AStarSolver(AStar):

    def __init__(self):
        self.gw = GridWorld()
        self.width = self.gw.env.n_width
        self.height = self.gw.env.n_height
        self.init_model()

    def heuristic_cost_estimate(self, n1, n2):
        if config.heuristic_type == 1:
            x1, y1 = n1
            x2, y2 = n2
            return math.hypot(x2 - x1, y2 - y1)
        elif config.heuristic_type == 2:
            x1, y1 = n1
            x2, y2 = n2
            return abs(x2-x1)+abs(y2-y1)
        elif config.heuristic_type == 3:
            if n1[0] == n2[0] and n1[1] == n2[1]:
                return 0
            q_model = self.model.model
            old_state = self.gw.env.state
            old_targets = self.gw.env.good_ends
            self.gw.env.state = self.gw.env._xy_to_state(n1)
            self.gw.env.good_ends = [n2]
            state, _ = self.gw.get_state(0, 0)
            q_values = q_model.calc_q([state,])[0]
            self.gw.env.state = old_state
            self.gw.env.good_ends = old_targets
            ret = max(0, np.min(q_values*(-1.)))*PkgConfig.reward_scale
            # at least add one step cost
            return ret
        elif config.heuristic_type == 4:
            # v-value of policy-net
            if n1[0] == n2[0] and n1[1] == n2[1]:
                return 0
            old_state = self.gw.env.state
            old_targets = self.gw.env.good_ends
            self.gw.env.state = self.gw.env._xy_to_state(n1)
            self.gw.env.good_ends = [n2]
            state, _ = self.gw.get_state(0, 0)
            x = self.gw.handle_states([state,])
            v = self.model.predict_v(x)[0]
            self.gw.env.state = old_state
            self.gw.env.good_ends = old_targets
            ret = max(0, np.min(v*(-1.)))*PkgConfig.reward_scale
            return ret
        else:
            raise RuntimeError("Not valid heuristic type")

    def distance_between(self, n1, n2):
        return 1

    def neighbors(self, node):
        x, y = node
        return [(nx, ny) for nx, ny in [(x, y-1), (x, y+1), (x-1, y), \
                (x+1, y)] if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny, 1) not in self.gw.env.types]

    def solve(self, gw):
        start = gw.env.start
        end = gw.env.good_ends[0]
        path = list(self.astar(start, end, env=gw.env))
        return True, {'path':path}

    def run(self):
        self.gw.reset()
        _, ext = self.solve(self.gw)
        path = ext['path']
        prev_pos = None
        for idx, pos in enumerate(path):
            if idx == 0:
                prev_pos = pos
                continue
            else:
                action = self.gw.env.parse_action(prev_pos, pos)
                self.gw.env.step(action)
                prev_pos = pos
        # close render
        self.gw.env._render(close=True)

    def init_model(self):
        worker_tf_conf = {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1}
        # worker_tf_conf = {}
        save_path = get_model_save_path(group_id=0, team_id=0, model_id=0)
        state_space_size, action_space_size = self.gw.get_game_model_info(0, 0)
        self.model = create_instance('network',
                device=PkgConfig.actor_device,
                model_name=PkgConfig.algorithm,
                state_space_size=state_space_size,
                action_space_size=action_space_size,
                save_path=save_path,
                init_time=time.time(),
                tf_conf=worker_tf_conf)
        self.model.load(train_step=config.train_step, save_path=save_path)

# solver using rl algorithm
class DLSolver(AStarSolver):
    def __init__(self):
        self.gw = GridWorld()
        self.width = self.gw.env.n_width
        self.height = self.gw.env.n_height
        self.init_model()

    def solve(self, gw):
        pos_set = set()
        while True:
            pos_set.add(gw.get_observation())
            state, _ = gw.get_state(0, 0)
            x = gw.handle_states([state,])
            action = self.model.inference(x)[0]
            reward, _, done, _ = gw.step(action, 0, 0)
            if gw.get_observation() in pos_set:
                return False, None
            if done:
                return True, None

    def run(self):
        # reset environment
        self.gw.reset()
        # self.gw.env._render()
        # time.sleep(1000)
        succ, _ = self.solve(self.gw)
        if not succ:
            time.sleep(5)
        self.gw.env._render(close=True)

# hybridsolver
class HybridNavigationSolver(AStarSolver):

    def get_new_position(self, n1, action):
        new_x, new_y = n1
        if action == 0:
            new_x = max(0, new_x-1)
        elif action == 1:
            new_x = min(self.width-1, new_x+1)
        elif action == 2:
            new_y = min(self.height-1, new_y+1)
        elif action == 3:
            new_y = max(0, new_y-1)
        return (new_x, new_y)

    def solve(self, gw):
        # first use dl to find the planning path
        old_state = gw.env.state
        old_targets = gw.env.good_ends
        # init n1, n2
        start, end = tuple(gw.env._state_to_xy(old_state)), tuple(old_targets[0])
        n1, n2 = start, end
        # print('start, end', n1, n2)
        # init memory path
        memory_path_set = set()
        memory_path_deque = deque()
        while True:
            if is_equal(n1, n2):
                break
            gw.env.state = gw.env._xy_to_state(n1)
            gw.env.good_ends = [n2]
            state, _ = gw.get_state(0, 0)
            x = gw.handle_states([state,])
            action = self.model.inference(x)[0]
            if n1 not in memory_path_set:
                memory_path_set.add(n1)
                memory_path_deque.append([n1, action])
            else:
                break
            new_position = self.get_new_position(n1, action)
            n1 = new_position
        # reset old_state, old_targets
        gw.env.state = old_state
        gw.env.good_ends = old_targets
        # fix wrong path
        if not is_equal(n1, n2):
            # do not find the path to the target
            # forget some path, so use astar to find the rest path
            path = list(self.astar(n1, n2, env=gw.env, reversePath=True))
            first_idx = len(path) - 1
            for idx, n in enumerate(path):
                if n in memory_path_set:
                    first_idx = idx
                    break
            while memory_path_deque[-1][0] != path[first_idx]:
                pos, action = memory_path_deque.pop()
                memory_path_set.remove(pos)
            # pop the last path
            if len(memory_path_deque) > 0:
                pos, action = memory_path_deque.pop()
                memory_path_set.remove(pos)
            idx = first_idx
            while idx > 0:
                action = gw.env.parse_action(path[idx], path[idx-1])
                memory_path_deque.append((path[idx], action))
                memory_path_set.add(path[idx])
                idx -= 1
        ext = {
                'memory_path_set':memory_path_set,
                'memory_path_deque':memory_path_deque,
                }
        return True, ext

    def run(self):
        # reset environment
        self.gw.reset()
        succ, ext = self.solve(self.gw)
        memory_path_set = ext['memory_path_set']
        memory_path_deque = ext['memory_path_deque']
        start, end = self.gw.env.start, self.gw.env.good_ends[0]
        checked_obstacles = set()

        def _manhattan_distance(n1, n2):
            return abs(n1[0]-n2[0]) + abs(n1[1]-n2[1])
        #
        def _check_disappeared_obstacle(memory_path_set, memory_path_deque):
            # check memory path if obstacle disappear
            if config.check_disappeared_obstacle:
                for disappeared_obstacle in self.gw.env.disappeared_types:
                    disappeared_obstacle = tuple(disappeared_obstacle)
                    if disappeared_obstacle in checked_obstacles:
                        continue
                    now_pos = self.gw.get_observation()
                    if _manhattan_distance(disappeared_obstacle, now_pos) > DETECTION_THRESHOLD:
                        continue
                    checked_obstacles.add(disappeared_obstacle)
                    # find new path
                    path1 = list(self.astar(disappeared_obstacle, now_pos, env=self.gw.env, local_path=memory_path_set, reversePath=True))
                    path2 = list(self.astar(disappeared_obstacle, end, env=self.gw.env, local_path=memory_path_set))
                    new_memory_path_deque = deque()
                    # create new path
                    old_idx = 0
                    for pos, action in memory_path_deque:
                        if is_equal(pos, path1[0]):
                            break
                        new_memory_path_deque.append([pos, action])
                        old_idx += 1
                    # add path1
                    for idx, pos in enumerate(path1):
                        if idx == len(path1)-1:
                            break
                        action = self.gw.env.parse_action(pos, path1[idx+1])
                        new_memory_path_deque.append([pos, action])
                    # add path2
                    for idx, pos in enumerate(path2):
                        if idx == len(path2)-1:
                            break
                        action = self.gw.env.parse_action(pos, path2[idx+1])
                        new_memory_path_deque.append([pos, action])
                    # add tail path
                    tail_idx = old_idx
                    for tail_idx in range(old_idx, len(memory_path_deque)):
                        if is_equal(memory_path_deque[tail_idx][0], path2[-1]):
                            break
                    for idx in range(tail_idx, len(memory_path_deque)):
                        new_memory_path_deque.append(memory_path_deque[idx])
                    def estimate_cost(_path_deque):
                        if len(_path_deque) <= 0:
                            return 0
                        ret = 0
                        for idx, item in enumerate(_path_deque):
                            if idx == 0:
                                continue
                            ret += self.distance_between(_path_deque[idx-1][0], _path_deque[idx][0])
                        ret += self.distance_between(_path_deque[-1][0], end)
                        return ret
                    if estimate_cost(new_memory_path_deque) < estimate_cost(memory_path_deque):
                        # replace path deque
                        memory_path_deque = new_memory_path_deque
                        memory_path_set = set()
                        for pos, action in new_memory_path_deque:
                                memory_path_set.add(pos)
            return memory_path_set, memory_path_deque
        # smart walk
        new_position = tuple(self.gw.env._state_to_xy(self.gw.env.state))
        target = tuple(self.gw.env.good_ends[0])
        while not is_equal(new_position, target):
            # use memory path to navigation
            while len(memory_path_deque) > 0:
                old_position, action = memory_path_deque.popleft()
                # print('old position is', old_position)
                # step in the environment
                self.gw.env.step(action)
                # check new position is really reached
                new_position = tuple(self.gw.env._state_to_xy(self.gw.env.state))
                memory_path_set.remove(old_position)
                # check disappeared obstacle
                memory_path_set, memory_path_deque = _check_disappeared_obstacle(memory_path_set, memory_path_deque)
                # beyond expectation, don't move
                if is_equal(new_position, old_position):
                    break
            # use a-star to find a local path
            path = list(self.astar(new_position, self.gw.env.good_ends[0], env=self.gw.env, local_path=memory_path_set))
            # pop old path
            while len(memory_path_deque) > 0:
                if is_equal(path[-1], memory_path_deque[0][0]):
                    break
                invalid_position, action = memory_path_deque.popleft()
                memory_path_set.remove(invalid_position)
            # re-calc new path
            poster_pos = None
            for pos in path[::-1]:
                if poster_pos is None:
                    poster_pos = pos
                    continue
                else:
                    action = self.gw.env.parse_action(pos, poster_pos)
                    memory_path_set.add(pos)
                    memory_path_deque.appendleft((pos, action))
                    poster_pos = pos
        # close render
        self.gw.env._render(close=True)

    def check_succ_only_on_dl(self):
        self.try_cnt = 0
        self.succ_cnt = 0
        while True:
            # reset environment
            self.gw.reset()
            # first use dl to find the planning path
            old_state = self.gw.env.state
            old_targets = self.gw.env.good_ends
            # init n1, n2
            start, end = tuple(self.gw.env._state_to_xy(old_state)), tuple(old_targets[0])
            n1, n2 = start, end
            # print('start, end', n1, n2)
            # init memory path
            memory_path_set = set()
            memory_path_deque = deque()
            while True:
                if is_equal(n1, n2):
                    break
                self.gw.env.state = self.gw.env._xy_to_state(n1)
                self.gw.env.good_ends = [n2]
                state, _ = self.gw.get_state(0, 0)
                action = self.model.inference([state, ])[0]
                if n1 not in memory_path_set:
                    memory_path_set.add(n1)
                    memory_path_deque.append([n1, action])
                else:
                    break
                new_position = self.get_new_position(n1, action)
                n1 = new_position
            # reset old_state, old_targets
            self.gw.env.state = old_state
            self.gw.env.good_ends = old_targets

            self.try_cnt += 1
            if is_equal(n1, n2):
                self.succ_cnt += 1
            if self.try_cnt % 100 == 0:
                print('succ rate {:.4f}'.format(float(self.succ_cnt)/self.try_cnt))

Solver = None
if config.solver_type == 1:
    Solver = AStarSolver
elif config.solver_type == 2:
    Solver = DLSolver
elif config.solver_type == 3:
    Solver = HybridNavigationSolver
