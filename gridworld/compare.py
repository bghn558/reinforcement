import os
import random
from multiprocessing import Process, Queue

import time

from . import config
from .game_interface import GridWorld
from .record import Record
from .solver import Solver

TASK_DONE = 1

def get_output_filename(id):
    _dir = 'data/'
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return _dir+'method_cost_'+str(id)+'.dat'


class Comparison(Record):

    def __init__(self):
        super().__init__()

    def run(self):
        self.max_cost = 0
        self.min_cost = 9999999
        self.total_cost = 0
        self.n = 0
        self.succ_n = 0
        # 
        self.runners = []
        self.done_set = set()
        for id in range(config.n_processors):
            self.runners.append(Runner(id, self.response_queue))
            self.runners[-1].start()
        total_targets = self.length
        while True:
            for r in self.runners:
                if r.request_queue.full():
                    continue
                t, starts = self.pop_next_pair()
                if t is None:
                    r.request_queue.put(TASK_DONE)
                else:
                    r.request_queue.put((t, starts))
            if not self.response_queue.empty():
                id, cost, succ, done = self.response_queue.get()
                if done:
                    self.done_set.add(id)
                    if len(self.done_set) >= config.n_processors:
                        break
                else:
                    self.n += 1
                    if succ:
                        self.succ_n += 1
                        self.total_cost += cost
                        self.max_cost = max(self.max_cost, cost)
                        self.min_cost = min(self.min_cost, cost)
                        print('complete {:.6f}, max cost {:.6f}, min cost {:.6f}, avg cost {:.6f}, \
                                succ_rate {:.4f}'.format(self.complete_rate(),\
                                self.max_cost, self.min_cost, float(self.total_cost)/self.succ_n, float(self.succ_n)/self.n))
        # task done
        for runner in self.runners:
            runner.join()

class Runner(Process):

    def __init__(self, id, response_queue):
        super().__init__()
        self.id = id
        self.records = dict()
        self.request_queue = Queue(maxsize=1)
        self.response_queue = response_queue
        # init environment
        self.gw = GridWorld()
        # reset environment
        self.gw.reset()

    def run(self):
        print('{} start running'.format(self.id))
        self.solver = Solver()
        with open(get_output_filename(self.id), 'w') as f:
            pass
        while True:
            task = self.request_queue.get()
            if task == TASK_DONE:
                break
            t, starts = task
            costs = []
            tests = random.sample(starts, 100)
            for idx, s in enumerate(tests):
                if s == t:
                    continue
                self.gw.env.reset(start=s, end=t)
                _ts = time.time()
                succ, _ = self.solver.solve(self.gw)
                cost = time.time() - _ts
                self.response_queue.put((self.id, cost, succ, False))
                if succ:
                    costs.append('{:.8f}'.format(cost))
            _str = '\n'.join(costs) + '\n'
            with open(get_output_filename(self.id), 'a+') as f:
                f.write(_str)
        self.response_queue.put((self.id, 0, False, True))
