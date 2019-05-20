from multiprocessing import Process, Queue

import numpy as np
import time

from rllib2.algos.base.base_network import ValueNetwork
from rllib2.config import config as PkgConfig
from rllib2.framework.utils import get_model_save_path
from rllib2.utils.astar import AStar
from . import config
from .dijkstra.dijkstra import dijkstra
from .game_interface import CreateGridWorld, GridWorld
from .record_and_train import BaseTrainer

SAVE_FREQUENCY = 1000
BATCH_SIZE = 1024

def is_equal(n1, n2):
    return n1[0] == n2[0] and n1[1] == n2[1]

def distance(t1, t2):
    # manhattan distance
    t1_x, t1_y = t1.center
    t2_x, t2_y = t2.center
    return abs(t1_x-t2_x) + abs(t1_y-t2_y)

class Tile(object):

    def __init__(self, gw, sx, sy, ex, ey, level):
        self.gw = gw
        self.sx, self.sy = sx, sy
        self.ex, self.ey = ex, ey
        self.level = level
        self.leaf = False
        self.leaf_type = -1
        self.children = []
        #
        self._check()

    def _has_diff_type(self):
        obstacles = False
        roads = False
        for x in range(self.sx, self.ex+1):
            for y in range(self.sy, self.ey+1):
                if (x,y) not in self.gw.all_grids:
                    obstacles = True
                else:
                    roads = True
        if obstacles == roads:
            return True
        if obstacles:
            self.leaf_type = 1
        elif roads:
            self.leaf_type = 0
        return False

    def _check(self):
        if self._has_diff_type():
            self._split()
        else:
            self.leaf = True

    def _check_child(self, sx, sy, ex, ey):
        if sx > ex or sy > ey:
            return False
        return True

    def _split(self):
        sx, sy = self.sx, self.sy
        ex, ey = self.ex, self.ey
        dx = ex - sx
        dy = ey - sy
        half_dx, half_dy = dx//2, dy//2
        self.children = []
        if self._check_child(sx, sy, sx+half_dx, sy+half_dy):
            ld = Tile(self.gw, sx, sy, sx+half_dx, sy+half_dy, self.level+1)
            if not ld.leaf or ld.leaf_type != 1:
                self.children.append(ld)
        if self._check_child(sx, sy+half_dy+1, sx+half_dx, ey):
            lu = Tile(self.gw, sx, sy+half_dy+1, sx+half_dx, ey, self.level+1)
            if not lu.leaf or lu.leaf_type != 1:
                self.children.append(lu)
        if self._check_child(sx+half_dx+1, sy, ex, sy+half_dy):
            rd = Tile(self.gw, sx+half_dx+1, sy, ex, sy+half_dy, self.level+1)
            if not rd.leaf or rd.leaf_type != 1:
                self.children.append(rd)
        if self._check_child(sx+half_dx+1, sy+half_dy+1, ex, ey):
            ru = Tile(self.gw, sx+half_dx+1, sy+half_dy+1, ex, ey, self.level+1)
            if not ru.leaf or ru.leaf_type != 1:
                self.children.append(ru)

    def count(self):
        if self.leaf:
            return 1
        return sum([child.count() for child in self.children])

    def is_in_tile(self, n1):
        x, y = n1
        return self.sx <= x and x <= self.ex and self.sy <= y and y <= self.ey

    def get_leaf_tile(self, n1):
        if self.leaf:
            if self.is_in_tile(n1):
                return self
            else:
                return None
        for child in self.children:
            if child.is_in_tile(n1):
                return child.get_leaf_tile(n1)
        return None

    def find_all_tiles(self, tiles):
        if self.leaf:
            tiles.add(self)
        else:
            for child in self.children:
                child.find_all_tiles(tiles)

    @property
    def center(self):
        return ((self.sx+self.ex)//2, (self.sy+self.ey)//2)

    def grids(self):
        _grids = []
        for x in range(self.sx, self.ex+1):
            for y in range(self.sy, self.ey+1):
                _grids.append((x,y))
        return _grids

    def __str__(self):
        return str(self.center)

    def __repr__(self):
        return str(self.center)

class QuadTree(object):

    def __init__(self):
        self.gw = CreateGridWorld()
        self.root = Tile(self.gw, 0, 0, self.gw.n_width-1, self.gw.n_height-1, 0)

    def get_all_tiles(self):
        tiles = set()
        self.root.find_all_tiles(tiles)
        return tiles

    def get_leaf_tile(self, n1):
        return self.root.get_leaf_tile(n1)

    def neighbors(self, t1):
        _tiles = set()
        # lself.quadtree.get_leaf_tileeft
        if t1.sx-1 >= 0:
            i = t1.sx-1
            for j in range(t1.sy, t1.ey+1):
                t = self.get_leaf_tile((i, j))
                t and _tiles.add(t)
        # down
        if t1.sy-1 >= 0:
            j = t1.sy-1
            for i in range(t1.sx, t1.ex+1):
                t = self.get_leaf_tile((i, j))
                t and _tiles.add(t)
        # right
        if t1.ex+1 < self.gw.n_width:
            i = t1.ex+1
            for j in range(t1.sy, t1.ey+1):
                t = self.get_leaf_tile((i, j))
                t and _tiles.add(t)
        # up
        if t1.ey+1 < self.gw.n_height:
            j = t1.ey+1
            for i in range(t1.sx, t1.ex+1):
                t = self.get_leaf_tile((i, j))
                t and _tiles.add(t)
        return _tiles

    def run(self):
        print('Tile count', self.root.count())

class Boundary(object):

    def __init__(self, sx, sy, ex, ey):
        self.sx, self.sy = sx, sy
        self.ex, self.ey = ex, ey
        if self.sy == self.ey:
            self.type = 0
        elif self.sx == self.ex:
            self.type = 1

class QuadTreeAStarSolver(AStar):

    def __init__(self):
        self.quadtree = QuadTree()
        if config.heuristic_type == 4:
            self.gw = GridWorld()
            self.init_model()

    def init_model(self):
        state_space_size, action_space_size = self.gw.get_game_model_info(0, 0)
        init_time = time.time()
        save_path = get_model_save_path(group_id=0, team_id=0, model_id=0)
        self.model = ValueNetwork('/cpu:0', 'value-net', state_space_size, action_space_size, init_time, save_path,
                 create_tensorboard=False, model_id=0)
        # load model 
        print('save_path is', save_path, state_space_size, action_space_size)
        self.model.load(train_step=0, save_path=save_path)

    def heuristic_cost_estimate(self, t1, t2):
        if config.heuristic_type == 4:
            n1, n2 = t1.center, t2.center
            old_state = self.gw.env.state
            old_targets = self.gw.env.good_ends
            self.gw.env.state = self.gw.env._xy_to_state(n1)
            self.gw.env.good_ends = [n2]
            state, _ = self.gw.get_state(0, 0)
            x = self.gw.handle_states([state,])
            v = self.model.predict_v(x)[0]
            self.gw.env.state = old_state
            self.gw.env.good_ends = old_targets
            ret = max(0, v)*PkgConfig.reward_scale
            return ret
        else:
            return distance(t1, t2)

    def distance_between(self, t1, t2):
        # manhattan distance
        t1_x, t1_y = t1.center
        t2_x, t2_y = t2.center
        return abs(t1_x-t2_x) + abs(t1_y-t2_y)

    def neighbors(self, t1):
        return list(self.quadtree.neighbors(t1))

    def find_boundary(self, t, next_t):
        # t
        # left
        sy, ey = -1, -1
        for j in range(t.sy, t.ey+1):
            if next_t.is_in_tile((t.sx-1, j)):
                if sy == -1:
                    sy = j
                ey = j
        if sy != -1:
            return Boundary(t.sx, sy, t.sx, ey), Boundary(t.sx-1, sy, t.sx-1, ey)
        # right
        sy, ey = -1, -1
        for j in range(t.sy, t.ey+1):
            if next_t.is_in_tile((t.ex+1, j)):
                if sy == -1:
                    sy = j
                ey = j
        if sy != -1:
            return Boundary(t.ex, sy, t.ex, ey), Boundary(t.ex+1, sy, t.ex+1, ey)
        # down
        sx, ex = -1, -1
        for i in range(t.sx, t.ex+1):
            if next_t.is_in_tile((i, t.sy-1)):
                if sx == -1:
                    sx = i
                ex = i
        if sx != -1:
            return Boundary(sx, t.sy, ex, t.sy), Boundary(sx, t.sy-1, ex, t.sy-1)
        # up
        sx, ex = -1, -1
        for i in range(t.sx, t.ex+1):
            if next_t.is_in_tile((i, t.ey+1)):
                if sx == -1:
                    sx = i
                ex = i
        if sx != -1:
            return Boundary(sx, t.ey, ex, t.ey), Boundary(sx, t.ey+1, ex, t.ey+1)

    @property
    def _gw(self):
        return self.quadtree.gw

    @property
    def now_pos(self):
        return tuple(self._gw._state_to_xy(self._gw.state))

    def move_to_in_tile(self, target):
        while not is_equal(self.now_pos, target):
            dx = target[0] - self.now_pos[0]
            dy = target[1] - self.now_pos[1]
            if dx < 0:
                action = 0
            elif dx > 0:
                action = 1
            elif dy > 0:
                action = 2
            elif dy < 0:
                action = 3
            self._gw.step(action)

    def cross_tile(self, next_t):
        now_x, now_y = self.now_pos
        if next_t.is_in_tile((now_x-1, now_y)):
            action = 0
        elif next_t.is_in_tile((now_x+1, now_y)):
            action = 1
        elif next_t.is_in_tile((now_x, now_y+1)):
            action = 2
        elif next_t.is_in_tile((now_x, now_y-1)):
            action = 3
        self._gw.step(action)

    def find_path(self):
        self._gw.reset()
        start = self._gw.start
        end = self._gw.good_ends[0]
        s_tile = self.quadtree.get_leaf_tile(start)
        e_tile = self.quadtree.get_leaf_tile(end)
        tile_path = list(self.astar(s_tile, e_tile, env=self._gw, is_tile=True))
        # print('start tile is', s_tile)
        # print('end tile is', e_tile)
        # print('path is', len(tile_path))
        # for t in tile_path:
        #     print(t)
        # parse action
        target = end
        idx = 0
        while True:
            t = tile_path[idx]
            if t.is_in_tile(target):
                self.move_to_in_tile(target)
                break
            else:
                next_t = tile_path[idx+1]
                # find boundary
                boundary_a, boundary_b = self.find_boundary(t, next_t)
                now_x, now_y = self.now_pos
                # boundary sy==ey
                if boundary_a.type == 0:
                    new_x = min(max(now_x, boundary_a.sx), boundary_a.ex)
                    self.move_to_in_tile((new_x, boundary_a.sy))
                # boundary sx==ex
                elif boundary_a.type == 1:
                    new_y = min(max(now_y, boundary_a.sy), boundary_a.ey)
                    self.move_to_in_tile((boundary_a.sx, new_y))
                else:
                    raise RuntimeError
                # cross tile
                self.cross_tile(next_t)
                idx += 1
        self._gw._render(close=True)

    def run(self):
        while True:
            self.find_path()

class DijkstraRecord(object):

    def __init__(self):
        self.quadtree = QuadTree()
        # get all tiles
        all_tiles = self.quadtree.get_all_tiles()
        # node dict for dijkstra
        self.t_dict = dict()
        for t in all_tiles:
            self.t_dict[t] = dict()
            for n_t in self.quadtree.neighbors(t):
                self.t_dict[t][n_t] = distance(t, n_t)

    def run(self):
        for t in self.t_dict.keys():
            dist, pred = dijkstra(self.t_dict, start=t)

class QuadTreeBaseTrainer(BaseTrainer):

    def __init__(self):
        self.gw = GridWorld()

    def init_model(self):
        state_space_size, action_space_size = self.gw.get_game_model_info(0, 0)
        init_time = time.time()
        save_path = get_model_save_path(group_id=0, team_id=0, model_id=0)
        self.model = ValueNetwork('/cpu:0', 'value-net', state_space_size, action_space_size, init_time, save_path,
                 create_tensorboard=False, model_id=0)

    def get_x(self, t, data):
        rets = []
        for d in data:
            rets.append((t.center[0], t.center[1], d[0].center[0], d[0].center[1]))
        states = [self.get_state(r) for r in rets]
        x = self.gw.handle_states(states)
        return x

    def train(self, t, train_data):
        x = self.get_x(t, train_data)
        v = np.array([d[1] for d in train_data]).astype(np.float32)/PkgConfig.reward_scale

        feed_dict = self.model._base_feed_dict()
        feed_dict = self.model._update_feed_dict(x, feed_dict)
        feed_dict.update({
            self.model.v:v
        })
        if PkgConfig.use_multi_learner:
            return self.model.sess.run(self.model.opt_grad, feed_dict=feed_dict)
        else:
            return self.model.sess.run(self.model.train_op, feed_dict=feed_dict)

    def apply_gradients(self, gradients):
        if not gradients:
            return
        feed_dict = self.model._base_feed_dict()
        for i in range(len(gradients)):
            feed_dict[self.model.grad_holder[i][0]] = gradients[i][0]
        self.model.sess.run(self.model.train_op, feed_dict=feed_dict)

    def evaluate(self, idx, t, train_data):
        x = self.get_x(t, train_data)
        v = np.array([d[1] for d in train_data]).astype(np.float32)/PkgConfig.reward_scale
        values = self.model.predict_v(x)
        ground_truth_v = v
        loss_v = np.sqrt(np.mean(np.square(values - ground_truth_v)))
        _output = 'train_step(1000)={}, loss_v={:.8f}'.format(int(idx/SAVE_FREQUENCY), loss_v)
        print(_output)

class QuadTreeTrainer(Process, QuadTreeBaseTrainer):

    def __init__(self):
        super().__init__()
        self.gw = GridWorld()
        self.gradients_queue = Queue()
        # init quadtree
        self.quadtree = QuadTree()
        # get all tiles
        all_tiles = self.quadtree.get_all_tiles()
        # node dict for dijkstra
        self.t_dict = dict()
        for t in all_tiles:
            self.t_dict[t] = dict()
            for n_t in self.quadtree.neighbors(t):
                self.t_dict[t][n_t] = distance(t, n_t)
        self.quadtree_value_dict = dict()
        for t in self.t_dict.keys():
            dist, pred = dijkstra(self.t_dict, start=t)
            self.quadtree_value_dict[t] = list(dist.items())
        self.tile_keys = list(self.t_dict.keys())
        self.last_t = 0
        self.last_v = 0
        # print('fff', self.quadtree_value_dict)

    def get_training_data(self):
        _start = self.tile_keys[self.last_t]
        total_values = self.quadtree_value_dict[self.tile_keys[self.last_t]]
        _values = total_values[self.last_v:self.last_v+BATCH_SIZE]
        self.last_v += BATCH_SIZE
        if self.last_v >= len(total_values):
            self.last_v = 0
            self.last_t += 1
            if self.last_t >= len(self.tile_keys):
                self.last_t = 0
        # print('training data is', _start, _values)
        return _start, _values

    def broadcast_model(self, idx):
        new_m = self.model.dumps()
        for sub_trainer in self.sub_trainers:
            sub_trainer.model_queue.put((idx, new_m))

    def run(self):
        if PkgConfig.use_multi_learner:
            self.sub_trainers = []
            for i in range(config.sub_trainers):
                self.sub_trainers.append(QuadTreeSubTrainer(i, self))
                self.sub_trainers[-1].start()
            self.init_model()
            idx = 0
            while True:
                if idx % SAVE_FREQUENCY == 0:
                    self.model.save(0)
                self.broadcast_model(idx)
                for sub_trainer in self.sub_trainers:
                    t, train_data = self.get_training_data()
                    sub_trainer.training_queue.put((t, train_data))
                for i in range(config.sub_trainers):
                    id, gradients = self.gradients_queue.get()
                    self.apply_gradients(gradients)
                idx += 1
        else:
            self.init_model()
            idx = 0
            self.last_epoch = 0
            while True:
                t, train_data = self.get_training_data()
                if idx % SAVE_FREQUENCY == 0:
                    self.model.save(0)
                    self.evaluate(idx, t, train_data)
                self.train(t, train_data)
                idx += 1

class QuadTreeSubTrainer(Process, QuadTreeBaseTrainer):

    def __init__(self, id, trainer):
        super().__init__()
        self.id = id
        self.trainer = trainer
        self.gw = self.trainer.gw
        self.model_queue = Queue(maxsize=1)
        self.training_queue = Queue(maxsize=1)

    def run(self):
        self.init_model()
        idx, new_m = self.model_queue.get()
        self.model.update_params(new_m)
        self.train_data = []
        while True:
            t, train_data = self.training_queue.get()
            if self.id == 0 and idx % SAVE_FREQUENCY == 0:
                self.evaluate(idx, t, train_data)
            gradients = self.train(t, train_data)
            self.trainer.gradients_queue.put((self.id, gradients))
            idx, new_m = self.model_queue.get()
            self.model.update_params(new_m)
