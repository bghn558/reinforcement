#-*-coding:utf-8-*-

import json
import os
import random
from queue import Queue
from threading import Thread

import numpy as np
import time

from rllib2.algos.factory import create_instance
from rllib2.config import config as PkgConfig
from rllib2.framework.utils import get_model_save_path
from . import config
from .game_interface import GridWorld
from .record import get_output_filename


def get_log_file():
    _dir = 'data/'
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return _dir+'pre_train.log'

class TrainCore(object):

    def init(self):
        self.gw = GridWorld()
        self.train_data = list()
        self.init_model()

    def init_model(self, group_id=0, team_id=0, model_id=0):
        # worker_tf_conf = {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1}
        save_path = get_model_save_path(group_id=group_id, team_id=team_id, model_id=model_id)
        state_space_size, action_space_size = self.gw.get_game_model_info(0, 0)
        init_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.model = create_instance('network',
                device=PkgConfig.actor_device,
                model_name=PkgConfig.algorithm,
                save_path=save_path,
                state_space_size=state_space_size,
                action_space_size=action_space_size,
                init_time=init_time
                )

    def load_data(self, id=0, interval=1):
        id = id
        train_data = list()
        while True:
            filename = get_output_filename(id)
            if not os.path.exists(filename):
                break
            with open(filename, 'r') as f:
                for line in f.readlines():
                    if not line:
                        continue
                    dct = json.loads(line)
                    t = list(dct.keys())[0]
                    split_t = t.split(', ')
                    t = [int(split_t[0][1:]), int(split_t[1][:-1])]
                    starts = list(dct.values())[0]
                    for s, a, v in starts:
                        state = []
                        state.extend(s)
                        state.extend(t)
                        train_data.append([state, a, v])
            id += interval
        random.shuffle(train_data)
        return train_data

    def get_state(self, raw_state):
        s = (raw_state[0], raw_state[1])
        t = (raw_state[2], raw_state[3])
        observation = np.array(s, dtype=np.float32)
        self.gw.env.good_ends = [t]
        state, _ = self.gw.get_state(0, 0, observation=observation)
        return state

    def get_x(self, data):
        states = [self.get_state(d[0]) for d in data]
        x = self.gw.handle_states(states)
        return x

    def train(self, data):
        x = self.get_x(data)
        a = np.eye(self.model.num_actions)[np.array([d[1] for d in data])].astype(np.float32)
        v = np.array([d[2] for d in data]).astype(np.float32)/PkgConfig.reward_scale
        feed_dict = self.model._base_feed_dict()
        feed_dict = self.model._update_feed_dict(x, feed_dict)
        feed_dict.update({
            self.model.action_index:a,
            self.model.v: v,
            })
        self.model.sess.run(self.model.pre_train_op1, feed_dict=feed_dict)

    def evaluate(self):
        self.eval_cnt += 1
        self.train_length = 0
        data = self.train_data[:10000]
        x = self.get_x(data)
        predict_actions = self.model.inference(x)
        ground_truth = np.array([d[1] for d in data])
        values = self.model.predict_v(x)
        ground_truth_v = np.array([d[2] for d in data]).astype(np.float32)/PkgConfig.reward_scale
        loss_v = np.sqrt(np.mean(np.square(values - ground_truth_v)))
        if self.eval_cnt == 1:
            mode = 'w'
        else:
            mode = 'a+'
        with open(get_log_file(), mode) as f:
            _output = '{}, {:.6f}, {:.4f}'.format(self.eval_cnt, sum(np.equal(predict_actions, ground_truth))/float(len(ground_truth)), loss_v)
            f.write(_output+'\n')
            print(_output)
        self.model.save(0)

class PreTrain(TrainCore):

    def __init__(self):
        super().__init__()
        self.response_queue = Queue()

    def run(self):
        self.init()
        print('start load data...')
        self.train_data_list = []
        self.train_data = []
        total_runners = 0
        for id in range(config.n_processors):
            _data = self.load_data(id, config.n_processors)
            if len(_data) <= 0:
                break
            self.train_data_list.append(_data)
            self.train_data.extend(self.train_data_list[-1])
            total_runners += 1
        random.shuffle(self.train_data)
        print('load succ!')
        # create runners
        self.runners = []
        for id in range(total_runners):
            print('id , runner', id)
            self.runners.append(Runner(id, self))
            self.runners[-1].start()
        # first evaluate
        self.eval_cnt = 0
        self.evaluate()
        last_print_time = time.time()
        # update model
        all_reduce_runners = set([runner.id for runner in self.runners])
        for runner in self.runners:
            if runner.id in all_reduce_runners:
                runner.request_queue.put(True)
        # start loop
        while True:
            tmp_reduce_runners = set()
            for _ in range(len(all_reduce_runners)):
                id, epoch_done, train_length = self.response_queue.get()
                self.train_length += train_length
                if time.time() - last_print_time >= 5:
                    print('epoch={}, train rate={:.8f}, timecost={:.8f}'.format(self.eval_cnt, float(self.train_length)/len(self.train_data), time.time()-last_print_time))
                    last_print_time = time.time()
                if not epoch_done:
                    tmp_reduce_runners.add(id)
            # broadcast
            if len(tmp_reduce_runners) <= 0:
                tmp_reduce_runners = set([runner.id for runner in self.runners])
                # evalutate
                self.evaluate()
            all_reduce_runners = tmp_reduce_runners
            for runner in self.runners:
                if runner.id in all_reduce_runners:
                    runner.request_queue.put(True)
        for runner in self.runners:
            runner.join()

class Runner(TrainCore, Thread):

    def __init__(self, id, master):
        super().__init__()
        self.id = id
        # request queue
        self.master = master
        self.model = self.master.model
        self.request_queue = Queue(maxsize=1)
        self.gw = GridWorld()

    def run(self):
        self.train_data = self.master.train_data_list[self.id]
        batch_size = 256
        s_idx = 0
        while True:
            self.request_queue.get()
            self.train(self.train_data[s_idx:s_idx+batch_size])
            train_length = len(self.train_data[s_idx:s_idx+batch_size])
            epoch_done = False
            s_idx += batch_size
            if s_idx >= len(self.train_data):
                epoch_done = True
                s_idx = 0
            self.master.response_queue.put((self.id, epoch_done, train_length))
