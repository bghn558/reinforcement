import os
from multiprocessing import Process, Queue

import numpy as np
import time

from rllib2.algos.factory import create_instance
from rllib2.config import config as PkgConfig
from rllib2.framework.utils import get_model_save_path
from . import config
from .game_interface import GridWorld

BATCH_SIZE = 1024

USE_SUB_TRAINER = True

EVALUATION_SIZE = 10000

SEND_RAW_DATA = False

SAVE_FREQUENCY = 1000

def get_log_file():
    _dir = 'data/'
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return _dir+'record_and_train.log'

class Manager(object):

    def __init__(self):
        self.gw = GridWorld()
        self.gw.reset()
        self.targets = list(self.gw.env.all_grids)
        _, self.action_nums = self.gw.get_game_model_info(0, 0)
        print('all grids is', len(self.targets))
        self.training_queue = Queue(maxsize=config.n_processors*16)

    def run(self):
        self.generators = []
        for i in range(config.n_processors):
            self.generators.append(DataGenerator(i, self))
            self.generators[-1].start()
        trainer = Trainer(self)
        trainer.start()
        while True:
            time.sleep(10)
        for g in self.generators:
            g.join()

class DataGenerator(Process):
    def __init__(self, id, manager):
        super().__init__()
        self.id = id
        self.manager = manager
        self.gw = self.manager.gw
        self.targets = self.manager.targets

    def is_valid(self, n):
        if n in self.records:
            return False
        if n not in self.gw.env.all_grids:
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

    def do_b_search(self, t):
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

    def run(self):
        epoch = 1
        idx = self.id
        last_print_time = time.time()
        while True:
            self.records = dict()
            t = self.targets[idx]
            # generate new data
            self.do_b_search(t)
            # send new data
            train_data = []
            for n, value in self.records.items():
                train_data.append(((n[0], n[1], t[0], t[1]), value[0], value[1]))
                if len(train_data) >= BATCH_SIZE:
                    if SEND_RAW_DATA:
                        self.manager.training_queue.put((epoch, train_data))
                    else:
                        x = self.get_x(train_data)
                        a = np.eye(self.manager.action_nums)[np.array([d[1] for d in train_data])].astype(np.float32)
                        v = np.array([d[2] for d in train_data]).astype(np.float32)/PkgConfig.reward_scale
                        self.manager.training_queue.put((epoch, (x, a, v)))
                    train_data = []
            idx += config.n_processors
            if self.id == 0 and time.time() - last_print_time >= 5:
                print('epoch {:d}, process {:.8f}, time cost {:.8f}'.format(epoch, float(idx)/len(self.targets), time.time()-last_print_time))
                last_print_time = time.time()
            if idx >= len(self.targets):
                idx = self.id
                epoch += 1

class BaseTrainer(object):

    def init_model(self):
        group_id, team_id, model_id = 0, 0, 0
        # worker_tf_conf = {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1}
        worker_tf_conf = {}
        save_path = get_model_save_path(group_id=group_id, team_id=team_id, model_id=model_id)
        state_space_size, action_space_size = self.gw.get_game_model_info(0, 0)
        init_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.model = create_instance('network',
                device=PkgConfig.actor_device,
                model_name=PkgConfig.algorithm,
                save_path=save_path,
                state_space_size=state_space_size,
                action_space_size=action_space_size,
                init_time=init_time,
                tf_conf=worker_tf_conf
                )

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
        if SEND_RAW_DATA:
            train_data = data
            x = self.get_x(train_data)
            a = np.eye(self.manager.action_nums)[np.array([d[1] for d in train_data])].astype(np.float32)
            v = np.array([d[2] for d in train_data]).astype(np.float32)/PkgConfig.reward_scale
        else:
            x, a, v = data
        feed_dict = self.model._base_feed_dict()
        feed_dict = self.model._update_feed_dict(x, feed_dict)
        feed_dict.update({
            self.model.action_index:a,
            self.model.v: v,
            })
        if USE_SUB_TRAINER:
            return self.model.sess.run(self.model.pre_train_grad, feed_dict=feed_dict)
        else:
            return self.model.sess.run(self.model.pre_train_op1, feed_dict=feed_dict)

    def apply_gradients(self, gradients):
        if not gradients:
            return
        feed_dict = self.model._base_feed_dict()
        for i in range(len(gradients)):
            feed_dict[self.model.pre_grad_holder[i][0]] = gradients[i][0]
        self.model.sess.run(self.model.pre_train_op2, feed_dict=feed_dict)

    def evaluate(self, idx, train_data):
        if SEND_RAW_DATA:
            x = self.get_x(train_data)
            a = np.eye(self.manager.action_nums)[np.array([d[1] for d in train_data])].astype(np.float32)
            v = np.array([d[2] for d in train_data]).astype(np.float32)/PkgConfig.reward_scale
        else:
            x, a, v = train_data
        predict_actions = self.model.inference(x)
        ground_truth = np.argmax(a, axis=1)
        values = self.model.predict_v(x)
        ground_truth_v = v
        loss_v = np.sqrt(np.mean(np.square(values - ground_truth_v)))
        _output = 'train_step(1000)={}, acc={:.8f}, loss_v={:.8f}'.format(int(idx/SAVE_FREQUENCY), sum(np.equal(predict_actions, ground_truth))/float(len(ground_truth)), loss_v)
        print(_output)


class Trainer(Process, BaseTrainer):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.gw = self.manager.gw
        self.gradients_queue = Queue()

    def broadcast_model(self, idx):
        new_m = self.model.dumps()
        for sub_trainer in self.sub_trainers:
            sub_trainer.model_queue.put((idx, new_m))

    def run(self):
        if USE_SUB_TRAINER:
            self.sub_trainers = []
            for i in range(config.sub_trainers):
                self.sub_trainers.append(SubTrainer(i, self))
                self.sub_trainers[-1].start()
            self.init_model()
            idx = 0
            while True:
                if idx % SAVE_FREQUENCY == 0:
                    self.model.save(0)
                self.broadcast_model(idx)
                for i in range(config.sub_trainers):
                    id, gradients = self.gradients_queue.get()
                    self.apply_gradients(gradients)
                idx += 1
        else:
            self.init_model()
            idx = 0
            self.last_epoch = 0
            while True:
                epoch, train_data = self.manager.training_queue.get()
                if idx % SAVE_FREQUENCY == 0:
                    self.model.save(0)
                    self.evaluate(idx, train_data)
                self.train(train_data)
                idx += 1


class SubTrainer(Process, BaseTrainer):
    def __init__(self, id, trainer):
        super().__init__()
        self.id = id
        self.trainer = trainer
        self.manager = trainer.manager
        self.gw = trainer.manager.gw
        self.model_queue = Queue(maxsize=1)

    def run(self):
        self.init_model()
        idx, new_m = self.model_queue.get()
        self.model.update_params(new_m)
        self.train_data = []
        while True:
            epoch, train_data = self.manager.training_queue.get()
            if self.id == 0 and idx % SAVE_FREQUENCY == 0:
                self.evaluate(idx, train_data)
            gradients = self.train(train_data)
            self.trainer.gradients_queue.put((self.id, gradients))
            idx, new_m = self.model_queue.get()
            self.model.update_params(new_m)
