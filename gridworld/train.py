import time
import numpy as np
import tensorflow as tf
from rllib2.game.gridworld.data import *
from rllib2.game.gridworld.model import VI_Block, VI_Untied_Block
from rllib2.game.gridworld.vin_utils import fmt_row
import os
import argparse

parser = argparse.ArgumentParser()
# Data
parser.add_argument('-i', '--input', default='/data/gridworld_28_12_train.npz', type=str,
                    help='Path to data')
parser.add_argument('-t', '--test', default='/data/gridworld_28_12_test.npz',
                           help='Path to data')
parser.add_argument('--imsize', type= int,default=28, help='Size of input image')
parser.add_argument('-a','--action',type=int,default=4,help='action space')
# Parameters
parser.add_argument('-l','--lr', type=int, default=0.001, help='Learning rate for RMSProp')
parser.add_argument('-e','--epochs', type=int, default=10, help='Maximum epochs to train for')
parser.add_argument('-k','--k', type=int, default=36, help='Number of value iterations')
parser.add_argument('--ch_i', type=int, default=2, help='Channels in input layer')
parser.add_argument('--ch_h', type=int, default=150, help='Channels in initial hidden layer')
parser.add_argument('--ch_q', type=int, default=10, help='Channels in q layer (~actions)')
parser.add_argument('-bs','--batchsize', type=int, default=20, help='Batch size')
parser.add_argument('-ss','--statebatchsize', type=int, default=1,
                    help='Number of state inputs for each sample (real number, technically is k+1)')
parser.add_argument('--untied_weights', type=bool, default=False, help='Untie weights of VI network')
# Misc.
parser.add_argument('--seed', type=int  ,default=0, help='Random seed for numpy')
parser.add_argument('--display_step', type=int ,default=1, help='Print summary output every n epochs')
parser.add_argument('--log', type=bool  ,default=True, help='Enable for tensorboard summary')
parser.add_argument('--logdir', type=str  ,default='/data/outputs/vintf/28grid_12/',
                    help='Directory to store tensorboard summary')
parser.add_argument('--modeldir', type=str, default='/data/outputs/model/grid_28_12/',
                    help='Directory to store checkpoints')
config = parser.parse_args()

np.random.seed(config.seed)

# symbolic input image tensor where typically first channel is image, second is the reward prior
X = tf.placeholder(tf.float32, name="X", shape=[None, config.imsize, config.imsize, config.ch_i])
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32, name="S1", shape=[None, ])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32, name="S2", shape=[None, ])
y = tf.placeholder(tf.int32, name="y", shape=[None])
global_step = tf.train.get_or_create_global_step()
# Construct model (Value Iteration Network)
if (config.untied_weights):
    logits, nn = VI_Untied_Block(X, S1, S2, config)
else:
    logits, nn = VI_Block(X, S1, S2, config)

# Define loss and optimizer
y_ = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y_, name='cross_entropy')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
tf.add_to_collection('losses', cross_entropy_mean)

cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-6, centered=True).minimize(cost, global_step)

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))

# correct_prediction = tf.cast(tf.argmax(nn, 1), tf.int32)
# # Calculate accuracy
# accuracy = tf.reduce_mean(tf.cast(tf.not_equal(correct_prediction, y), dtype=tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=config.input, imsize=config.imsize)
Xtrain, S1train, S2train, ytrain, = process_gridworld_np_data(input=config.input, imsize=config.imsize)
Xtest, S1test, S2test, ytest = process_gridworld_np_data(input=config.input, imsize=config.imsize)
Xval, S1val, S2val, yval, Xtest, S1test, S2test, ytest = split(Xtest, S1test, S2test, ytest, frac=2 / 3)

# Launch the graph
with tf.Session() as sess:
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)

    latest_checkpoint = tf.train.latest_checkpoint(config.modeldir)
    if latest_checkpoint is not None:
        saver.restore(sess, latest_checkpoint)
    else:
        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)
    sess.run(init)

    batch_size = config.batchsize
    print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
    for epoch in range(int(config.epochs)):
        tstart = time.time()
        avg_err, avg_cost = 0.0, 0.0
        num_batches = int(Xtrain.shape[0] / batch_size)
        # Loop over all batches

        for i in range(0, Xtrain.shape[0], batch_size):
            j = i + batch_size
            if j <= Xtrain.shape[0]:
                # Run optimization op (backprop) and cost op (to get loss value)
                fd = {X: Xtrain[i:j], S1: S1train[i:j], S2: S2train[i:j],
                      y: ytrain[i * config.statebatchsize:j * config.statebatchsize]}
                _, e_, c_ = sess.run([optimizer, err, cost], feed_dict=fd)
                avg_err += e_
                avg_cost += c_
        # Display logs per epoch step
        if epoch % config.display_step == 0:
            elapsed = time.time() - tstart
            print(fmt_row(10, [epoch, avg_cost / num_batches, avg_err / num_batches, elapsed]))

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Average error', simple_value=float(avg_err / num_batches))
        summary.value.add(tag='Average cost', simple_value=float(avg_cost / num_batches))
        # summary_writer.add_summary(summary, epoch)
        if not os.path.exists(os.path.join(config.modeldir)):
            os.makedirs(config.modeldir)
        saver.save(sess, os.path.join(config.modeldir, "%dgrid" % config.imsize),
                   global_step=tf.train.global_step(sess, global_step_tensor=global_step))
        acc = []
        for i in range(0, Xval.shape[0], config.batchsize):
            j = i + config.batchsize
            if j <= Xval.shape[0]:
                acc.append(sess.run(err, {X: Xval[i:j],
                                          S1: S1val[i:j],
                                          S2: S2val[i:j],
                                          y: yval[
                                             i * config.statebatchsize:j * config.statebatchsize]}))
        print("Accuracy: {0}".format(100 * (1 - np.mean(acc))))
        summary.value.add(tag="Average val acc", simple_value=float(1 - np.mean(acc)))
        summary_writer.add_summary(summary, epoch)

    print("Finished training!")

    acc = []
    for i in range(0, Xtest.shape[0], config.batchsize):
        j = i + config.batchsize
        if j <= Xtest.shape[0]:
            acc.append(sess.run(err, {X: Xtest[i:j],
                                      S1: S1test[i:j],
                                      S2: S2test[i:j],
                                      y: ytest[
                                         i * config.statebatchsize:j * config.statebatchsize]}))
    print("Accuracy: {0}".format(100 * (1 - np.mean(acc))))
