import json
import os
from multiprocessing import Process, Queue

import time

from rllib2.utils.astar import AStar
from . import config
from .game_interface import CreateGridWorld

N_PROCESSOR = config.n_processors

TASK_DONE = 1

SEARCH_TYPE = 2
# 1=astar, 2=breadth-first, 3=dijkstra

def get_output_filename(id):
    _dir = 'data/'
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return _dir+'big_gridworld_with_v_'+str(id)+'.dat'

class Record(object):

    def __init__(self):
        self.gw = CreateGridWorld()
        # reset environment
        self.gw.reset()
        print('all_grids is', len(self.gw.all_grids))
        if SEARCH_TYPE == 1:
            self.path_pairs = dict()
            for s in self.gw.all_grids:
                for t in self.gw.all_grids:
                    if s == t:
                        continue
                    self.path_pairs.setdefault(t, set()).add(s)
            self.length = len(self.path_pairs)
        elif SEARCH_TYPE == 2:
            self.t_idx = 0
            self.targets = list(self.gw.all_grids)
            self.length = len(self.targets)
        self.response_queue = Queue()

    def pop_next_pair(self):
        if SEARCH_TYPE == 1:
            if len(self.path_pairs) <= 0:
                return None, None
            t = list(self.path_pairs.keys())[0]
            starts = self.path_pairs.pop(t)
            return t, starts
        elif SEARCH_TYPE == 2:
            if self.t_idx >= len(self.targets):
                return None, None
            t = self.targets[self.t_idx]
            self.t_idx += 1
            return t, self.targets

    def complete_rate(self):
        if SEARCH_TYPE == 1:
            return 1-len(self.path_pairs)/float(self.length)
        elif SEARCH_TYPE == 2:
            return self.t_idx/float(self.length)

    def astar(self):
        self.total_targets = len(self.path_pairs)
        self.last_log_time = time.time()
        while True:
            # dispatch task
            for r in self.runners:
                if r.request_queue.full():
                    continue
                if len(self.path_pairs) > 0:
                    t = list(self.path_pairs.keys())[0]
                    starts = self.path_pairs.pop(t)
                    r.request_queue.put((t, starts))
                else:
                    r.request_queue.put(TASK_DONE)
            time.sleep(0.1)
            if time.time() - self.last_log_time >= 10:
                print('~~~', len(self.path_pairs) / self.total_targets)
                self.last_log_time = time.time()
            # response
            if not self.response_queue.empty():
                done_id = self.response_queue.get()
                self.done_set.add(done_id)
                if len(self.done_set) >= N_PROCESSOR:
                    break

    def b_search(self):
        self.total_targets = len(self.targets)
        self.last_log_time = time.time()
        idx = 0
        while True:
            # dispatch task
            for r in self.runners:
                if r.request_queue.full():
                    continue
                if idx < len(self.targets):
                    t = self.targets[idx]
                    r.request_queue.put((t, None))
                    idx += 1
                else:
                    r.request_queue.put(TASK_DONE)
            time.sleep(0.01)
            if time.time() - self.last_log_time >= 10:
                print('calc rate {:.8f}'.format(float(idx) / self.total_targets))
                self.last_log_time = time.time()
            # response
            if not self.response_queue.empty():
                done_id = self.response_queue.get()
                self.done_set.add(done_id)
                if len(self.done_set) >= N_PROCESSOR:
                    break

    def run(self):
        self.runners = []
        self.done_set = set()
        for id in range(N_PROCESSOR):
            self.runners.append(Runner(id, self.response_queue, self))
            self.runners[-1].start()
        # do
        if SEARCH_TYPE == 1:
            self.astar()
        elif SEARCH_TYPE == 2:
            self.b_search()
        # end 
        for runner in self.runners:
            runner.join()

class Runner(Process, AStar):

    def __init__(self, id, response_queue, master):
        super().__init__()
        self.id = id
        self.records = dict()
        self.request_queue = Queue(maxsize=1)
        self.response_queue = response_queue
        self.master = master
        self.gw = self.master.gw
        self.width = self.gw.n_width
        self.height = self.gw.n_height

    def heuristic_cost_estimate(self, n1, n2):
        x1, y1 = n1
        x2, y2 = n2
        return abs(x2-x1)+abs(y2-y1)

    def neighbors(self, node):
        x, y = node
        return [(nx, ny) for nx, ny in [(x, y-1), (x, y+1), (x-1, y), \
                (x+1, y)] if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny, 1) not in self.gw.types]

    def distance_between(self, n1, n2):
        return 1

    def do_astar(self, task):
        t, starts = task
        for s in starts:
            if s in self.records:
                continue
            self.gw.reset(start=s, end=t)
            path = list(self.astar(s, t, env=self.gw))
            prev_pos = None
            for idx, pos in enumerate(path):
                if idx == 0:
                    prev_pos = pos
                    continue
                else:
                    action = self.gw.parse_action(prev_pos, pos)
                    # record
                    key = (tuple(prev_pos), t)
                    self.records[tuple(prev_pos)] = [action, -(len(path)-idx)]
                    self.gw.step(action)
                    prev_pos = pos

    def is_valid(self, n):
        if n in self.records:
            return False
        if n not in self.gw.all_grids:
            return False
        return True

    def search_next_nodes(self, node, first=False):
        x, y = node
        if first:
            a, value = 0, 0
        else:
            a, value = self.records[(x,y)]
        neighbors = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
        actions = [1, 0, 2, 3]
        valid_neighbors = []
        for n, a in zip(neighbors, actions):
            if not self.is_valid(n):
                continue
            self.records[n] = [a, value-1]
            self.all_nodes.append(n)

    def do_b_search(self, task):
        t, _ = task
        tx, ty = t
        self.all_nodes = []
        # first search
        self.search_next_nodes(t, first=True)
        idx = 0
        while True:
            if idx >= len(self.all_nodes):
                break
            self.search_next_nodes(self.all_nodes[idx])
            idx += 1

    def run(self):
        print('child process start running', self.id)
        # clear old data
        with open(get_output_filename(self.id), 'w') as f:
            pass
        while True:
            task = self.request_queue.get()
            if task == TASK_DONE:
                break
            self.records = dict()
            self.t = task[0]
            if SEARCH_TYPE == 1:
                self.do_astar(task)
            elif SEARCH_TYPE == 2:
                self.do_b_search(task)
            rets = []
            for key, value in self.records.items():
                a, v = value
                # rets.append(json.dumps((key, self.t, a, v)))
                rets.append((key, a, v))
            dct = {str(self.t):rets}
            # _str = '\n'.join(rets) + '\n'
            # print('self.records', self.records)
            _str = json.dumps(dct) + '\n'
            with open(get_output_filename(self.id), 'a+') as f:
                f.write(_str)
        self.response_queue.put(self.id)
