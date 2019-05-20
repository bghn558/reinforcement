from .compare import get_output_filename, Comparison
from . import config
from multiprocessing import Process, Queue
import os

TASK_DONE = 1

class Statistics(Comparison):

    def run(self):
        self.max_cost = 0
        self.min_cost = 9999999
        self.total_cost = 0
        self.n = 0
        self.succ_n = 0

        self.runners = []
        self.done_set = set()
        for id in range(config.n_processors):
            self.runners.append(Runner(id, self.response_queue))
            self.runners[-1].start()
        now_idx = 0
        while True:
            filename =  get_output_filename(now_idx)
            if not os.path.exists(filename):
                for r in self.runners:
                    if r.request_queue.full():
                        continue
                    r.request_queue.put(TASK_DONE)
            else:
                for r in self.runners:
                    if r.request_queue.full():
                        continue
                    r.request_queue.put(filename)
                    now_idx += 1
                    break
            if not self.response_queue.empty():
                id, max_cost, min_cost, total_cost, n, succ_n, done = self.response_queue.get()
                if done:
                    self.done_set.add(id)
                    if len(self.done_set) >= config.n_processors:
                        break
                else:
                    self.n += n
                    self.succ_n += succ_n
                    self.total_cost += total_cost
                    self.max_cost = max(self.max_cost, max_cost)
                    self.min_cost = min(self.min_cost, min_cost)
                    print('max cost {:.6f}, min cost {:.6f}, avg cost {:.6f}, \
                            succ_rate {:.4f}'.format(self.max_cost, self.min_cost, \
                            float(self.total_cost)/self.succ_n, float(self.succ_n)/self.n))

        for runner in self.runners:
            runner.join()


class Runner(Process):

    def __init__(self, id, response_queue):
        super().__init__()
        self.id = id
        self.request_queue = Queue(maxsize=1)
        self.response_queue = response_queue

    def run(self):
        print('{} start running'.format(self.id))
        while True:
            task = self.request_queue.get()
            if task == TASK_DONE:
                break
            filename = task
            max_cost = 0
            min_cost = 9999999
            total_cost = 0
            n = 0
            with open(filename, 'r') as f:
                for line in f.readlines():
                    cost = float(line)
                    n += 1
                    total_cost += cost
                    max_cost = max(max_cost, cost)
                    min_cost = min(min_cost, cost)
            self.response_queue.put((self.id, max_cost, min_cost, total_cost, n, n, False))
        self.response_queue.put((self.id, 0, 0, 0, 0, 0, True))
