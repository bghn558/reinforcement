import numpy as np
import tensorflow as tf
import os
from rllib2.game.gridworld.vin_utils import *
import time
import sys


def VI_Block(X, S1, S2, config):
    k = config.k  # Number of value iterations performed
    ch_i = config.ch_i  # Channels in input layer
    ch_h = config.ch_h  # Channels in initial hidden layer
    ch_q = config.ch_q  # Channels in q layer (~actions)
    state_batch_size = config.statebatchsize  # k+1 state inputs for each channel

    bias = tf.Variable(np.random.randn(1, 1, 1, ch_h) * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w0 = tf.Variable(np.random.randn(3, 3, ch_i, ch_h) * 0.01, dtype=tf.float32)
    w1 = tf.Variable(np.random.randn(1, 1, ch_h, 1) * 0.01, dtype=tf.float32)
    w = tf.Variable(np.random.randn(3, 3, 1, ch_q) * 0.01, dtype=tf.float32)
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb = tf.Variable(np.random.randn(3, 3, 1, ch_q) * 0.01, dtype=tf.float32)
    w_o = tf.Variable(np.random.randn(ch_q, config.action) * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h = conv2d_flipkernel(X, w0, name="h0") + bias

    r = conv2d_flipkernel(h, w1, name="r")
    q = conv2d_flipkernel(r, w, name="q")
    v = tf.reduce_max(q, axis=3, keepdims=True, name="v")

    for i in range(0, k - 1):
        rv = tf.concat([r, v], 3)
        wwfb = tf.concat([w, w_fb], 2)
        q = conv2d_flipkernel(rv, wwfb, name="q")
        v = tf.reduce_max(q, axis=3, keepdims=True, name="v")

    # do one last convolution
    q = conv2d_flipkernel(tf.concat([r, v], 3),
                          tf.concat([w, w_fb], 2), name="q")

    # CHANGE TO THEANO ORDERING
    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

    # add logits
    logits = tf.matmul(q_out, w_o)
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return logits, output


# similar to the normal VI_Block except there are separate weights for each q layer
def VI_Untied_Block(X, S1, S2, config):
    k = config.k  # Number of value iterations performed
    ch_i = config.ch_i  # Channels in input layer
    ch_h = config.ch_h  # Channels in initial hidden layer
    ch_q = config.ch_q  # Channels in q layer (~actions)
    state_batch_size = config.statebatchsize  # k+1 state inputs for each channel

    bias = tf.Variable(np.random.randn(1, 1, 1, ch_h) * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w0 = tf.Variable(np.random.randn(3, 3, ch_i, ch_h) * 0.01, dtype=tf.float32)
    w1 = tf.Variable(np.random.randn(1, 1, ch_h, 1) * 0.01, dtype=tf.float32)
    w_l = [tf.Variable(np.random.randn(3, 3, 1, ch_q) * 0.01, dtype=tf.float32) for i in range(0, k + 1)]
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb_l = [tf.Variable(np.random.randn(3, 3, 1, ch_q) * 0.01, dtype=tf.float32) for i in range(0, k)]
    w_o = tf.Variable(np.random.randn(ch_q, config.action) * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h = conv2d_flipkernel(X, w0, name="h0") + bias

    r = conv2d_flipkernel(h, w1, name="r")
    q = conv2d_flipkernel(r, w_l[0], name="q")
    v = tf.reduce_max(q, axis=3, keepdims=True, name="v")

    for i in range(0, k - 1):
        rv = tf.concat([r, v], 3)
        wwfb = tf.concat([w_l[i + 1], w_fb_l[i]], 2)
        q = conv2d_flipkernel(rv, wwfb, name="q")
        v = tf.reduce_max(q, axis=3, keepdims=True, name="v")

    # do one last convolution
    q = conv2d_flipkernel(tf.concat([r, v], 3),
                          tf.concat([w_l[k], w_fb_l[k - 1]], 2), name="q")

    # CHANGE TO THEANO ORDERING
    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

    # add logits
    logits = tf.matmul(q_out, w_o)
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return logits, output


class ILModel:
    def __init__(self, config):
        self.config = config
        self.build_net()
        self.saver = tf.train.Saver(max_to_keep=10)
        self.sess = tf.Session()
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(config.logdir, self.sess.graph)
        self.load_model()

    def build_net(self):
        self.X = tf.placeholder(dtype=tf.float32, name="X",
                                shape=[None, self.config.imsize, self.config.imsize, self.config.ch_i])
        # symbolic input batches of vertical positions
        self.S1 = tf.placeholder(tf.int32, name="S1", shape=[None, ])
        # symbolic input batches of horizontal positions
        self.S2 = tf.placeholder(tf.int32, name="S2", shape=[None, ])
        self.y = tf.placeholder(tf.int32, name="y", shape=[None])
        self.global_step = tf.train.get_or_create_global_step()

        self.logits, self.nn = VI_Block(self.X, self.S1, self.S2, self.config)

        self.cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=tf.cast(self.y, tf.int64),
            name='cross_entropy'
        ),
            name="cross_entropy_mean"
        )

        tf.add_to_collection('losses', self.cross_entropy_mean)
        self.cost = tf.add_n(tf.get_collection("losses"), name="total_loss")

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.lr, epsilon=1e-6, centered=True). \
            minimize(self.cost, self.global_step)

        # Test model & calculate accuracy
        self.cp = tf.cast(tf.argmax(self.nn, 1), tf.int32)
        self.err = tf.reduce_mean(tf.cast(tf.not_equal(self.cp, self.y), dtype=tf.float32))

    def load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.modeldir)
        if latest_checkpoint is not None:
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            # Initializing the variables
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def save_model(self):
        if not os.path.exists(os.path.join(self.config.modeldir)):
            os.makedirs(self.config.modeldir)
        self.saver.save(self.sess, os.path.join(self.config.modeldir, "%dgrid" % self.config.imsize),
                        global_step=tf.train.global_step(self.sess, global_step_tensor=self.global_step))

    def save_log(self, avg_err, avg_cost, num_batches, acc, epoch):
        summary = tf.Summary()
        summary.ParseFromString(self.sess.run(self.summary_op))
        summary.value.add(tag='Average error', simple_value=float(avg_err / num_batches))
        summary.value.add(tag='Average cost', simple_value=float(avg_cost / num_batches))
        summary.value.add(tag="Average val acc", simple_value=float(1 - np.mean(acc)))
        self.summary_writer.add_summary(summary, epoch)

    def predict(self, X, S1, S2):
        action = self.sess.run(self.cp, feed_dict={
            self.X: X,
            self.S1: S1,
            self.S2: S2,
        })
        return action

    def train(self, Xtrain, S1train, S2train, ytrain, Xval, S1val, S2val, yval, Xtest, S1test, S2test, ytest):
        batch_size = self.config.batchsize
        print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
        for epoch in range(int(self.config.epochs)):
            tstart = time.time()
            avg_err, avg_cost = 0.0, 0.0
            num_batches = int(Xtrain.shape[0] / batch_size)
            # Loop over all batches

            for i in range(0, Xtrain.shape[0], batch_size):
                j = i + batch_size
                if j <= Xtrain.shape[0]:
                    # Run optimization op (backprop) and cost op (to get loss value)
                    fd = {self.X: Xtrain[i:j], self.S1: S1train[i:j], self.S2: S2train[i:j],
                          self.y: ytrain[i * self.config.statebatchsize:j * self.config.statebatchsize]}
                    _, e_, c_ = self.sess.run([self.optimizer, self.err, self.cost], feed_dict=fd)
                    avg_err += e_
                    avg_cost += c_
            # Display logs per epoch step
            if epoch % self.config.display_step == 0:
                elapsed = time.time() - tstart
                print(fmt_row(10, [epoch, avg_cost / num_batches, avg_err / num_batches, elapsed]))

            # summary_writer.add_summary(summary, epoch)
            self.save_model()
            acc = []
            for i in range(0, Xval.shape[0], self.config.batchsize):
                j = i + self.config.batchsize
                if j <= Xval.shape[0]:
                    acc.append(self.sess.run(self.err, {self.X: Xval[i:j],
                                                        self.S1: S1val[i:j],
                                                        self.S2: S2val[i:j],
                                                        self.y: yval[
                                                                i * self.config.statebatchsize:j * self.config.statebatchsize]}))
            print("Accuracy: {0}".format(100 * (1 - np.mean(acc))))
            self.save_log(avg_err, avg_cost, num_batches, acc, epoch)

        print("Finished training!")

        acc = []
        for i in range(0, Xtest.shape[0], self.config.batchsize):
            j = i + self.config.batchsize
            if j <= Xtest.shape[0]:
                acc.append(self.sess.run(self.err, {self.X: Xtest[i:j],
                                                    self.S1: S1test[i:j],
                                                    self.S2: S2test[i:j],
                                                    self.y: ytest[
                                                            i * self.config.statebatchsize:j * self.config.statebatchsize]}))
        print("Accuracy: {0}".format(100 * (1 - np.mean(acc))))

    def test(self, n_domains, n_traj, gen):
        correct, total = 0.0, 0.0
        reward_total = []
        for i in range(n_domains):
            gen.gw.m.create_maze(True)
            gen.gw.initial_end()  # 为了简单，先将goal 和 image 都固定，和作者代码一致
            end = gen.gw.good_ends[0]
            s1, s2, label = gen.get_optimal_path(n_traj)
            s1 = s1.reshape([n_traj, -1])
            s2 = s2.reshape([n_traj, -1])
            s1.dtype = int
            s2.dtype = int

            image_map = (gen.gw.m.get_image()).reshape([1, self.config.imsize, self.config.imsize, 1])
            value_map = (gen.gw.get_value_map()).reshape([1, self.config.imsize, self.config.imsize, 1])
            X = np.concatenate((image_map, value_map), axis=-1)

            for j in range(n_traj):
                L = self.config.max_distance * 4
                pred_traj = [[s1[j][0], s2[j][0]]]
                reward_temp = 0
                gen.gw.reset((int(pred_traj[0][0]), int(pred_traj[0][1])), end)
                gen.gw.render()
                for k in range(1, L):
                    S1 = [pred_traj[k - 1][0]]
                    S2 = [pred_traj[k - 1][1]]
                    a = self.predict(X, S1, S2)[0]
                    state, reward, isdone, info = gen.gw.step(a)
                    pred_traj.append([info['x'], info['y']])
                    reward_temp += reward
                    if pred_traj[j][0] == end[0] and pred_traj[j][0] == end[1]:
                        correct += 1
                        break
                    gen.gw.render()
                reward_total.append(reward_temp)
                total += 1
                if self.config.plot:
                    s1 = np.append(s1[j], end[0])
                    s2 = np.append(s2[j], end[1])
                    gt_traj = np.stack((s1, s2), -1)
                    visualize(gen.gw.m.get_image().T, gt_traj, np.array(pred_traj))

            sys.stdout.write("\r" + str(int(
                (float(i) / n_domains) * 100.0)) + "%")
            sys.stdout.flush()
            sys.stdout.write("\n")
        print('Rollout Success rate: {:.2f}%'.format(100 * (correct / total)))
        print("Acc reward:{:.2f}%".format(np.mean(reward_total)))
