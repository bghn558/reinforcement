import os

import matplotlib.pyplot as plt
import tensorflow as tf

from data_gen import *
from rllib2.game.gridworld.model import *

parser = argparse.ArgumentParser()
# Data
parser.add_argument('-i', '--input', default='F:\project\RLlib2 for test\data\gridworld_28_6_train.npz', type=str,
                    help='Path to data')
parser.add_argument('-t', '--test', default='F:\project\RLlib2 for test\data\gridworld_28_6_test.npz',
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
parser.add_argument('-bs','--batchsize', type=int, default=64, help='Batch size')
parser.add_argument('-ss','--statebatchsize', type=int, default=1,
                    help='Number of state inputs for each sample (real number, technically is k+1)')
parser.add_argument('--untied_weights', type=bool, default=False, help='Untie weights of VI network')
# Misc.
parser.add_argument('--seed', type=int  ,default=0, help='Random seed for numpy')
parser.add_argument('--display_step', type=int ,default=1, help='Print summary output every n epochs')
parser.add_argument('--log', type=bool  ,default=True, help='Enable for tensorboard summary')
parser.add_argument('--logdir', type=str  ,default='./outputs/vintf/28_6/',
                    help='Directory to store tensorboard summary')
parser.add_argument('--modeldir', type=str, default='./grid_28_6/',
                    help='Directory to store checkpoints')
# for test
parser.add_argument('-md','--max_distance',type=int,default=10,help="max distance of road")
parser.add_argument("-p","--plot",type=bool,default=False,help="show difference between ground truth path & pred path")
config = parser.parse_args()


class ILModel:
    def __init__(self,config):
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
        self.X = tf.placeholder(dtype=tf.float32, name="X",shape=[None,self.config.imsize,self.config.imsize,self.config.ch_i])
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

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.lr,epsilon=1e-6,centered=True).\
            minimize(self.cost,self.global_step)

        # Test model & calculate accuracy
        self.cp = tf.cast(tf.argmax(self.nn, 1), tf.int32)
        self.err = tf.reduce_mean(tf.cast(tf.not_equal(self.cp, self.y), dtype=tf.float32))

    def load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(config.modeldir)
        if latest_checkpoint is not None:
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            # Initializing the variables
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def save_model(self):
        if not os.path.exists(os.path.join(config.modeldir)):
            os.makedirs(config.modeldir)
        self.saver.save(self.sess, os.path.join(config.modeldir, "%dgrid" % config.imsize),
                   global_step=tf.train.global_step(self.sess, global_step_tensor=self.global_step))

    def predict(self,X,S1,S2):
        action = self.sess.run(self.cp, feed_dict={
            self.X:X,
            self.S1:S1,
            self.S2:S2,
        })
        return action


def visualize(dom, gt_traj, pred_traj):
    fig, ax = plt.subplots()
    implot = plt.imshow(dom, cmap="Greys_r")
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], c='b', label='Optimal Path')
    ax.plot(
        pred_traj[:, 0], pred_traj[:, 1], '-X', c='r', label='Predicted Path')
    ax.plot(gt_traj[0, 0], gt_traj[0, 1], '-o', label='Start')
    ax.plot(gt_traj[-1, 0], gt_traj[-1, 1], '-s', label='Goal')
    legend = ax.legend(loc='upper right', shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-small')  # the legend text size
    for label in legend.get_lines():
        label.set_linewidth(0.5)  # the legend line width
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

def main(config):
    model = ILModel(config)
    n_domains = 1000
    gen = DataGenerator(config.imsize, 10, config.max_distance)
    n_traj = 1
    correct, total = 0.0, 0.0
    reward_total = []
    for i in range(n_domains):
        gen.gw.m.create_maze(True)
        gen.gw.reset()
        end = gen.gw.initial_end()[0]  # 为了简单，先将goal 和 image 都固定，和作者代码一致
        state_x, steat_y, label = gen.get_optimal_path(n_traj)
        s1test = state_x.reshape([n_traj,-1])
        s2test = steat_y.reshape([n_traj,-1])
        s1test.dtype=int
        s2test.dtype=int

        image_map = (gen.gw.get_image()).reshape([1, config.imsize, config.imsize, 1])
        value_map = (gen.gw.get_value_map()).reshape([1, config.imsize, config.imsize, 1])
        X = np.concatenate((image_map, value_map), axis=-1)

        for j in range(n_traj):
            L = len(s1test[j]) * 2
            pred_traj = [[s1test[j][0], s2test[j][0]]]
            reward_temp = 0
            gen.gw.reset((int(pred_traj[0][0]),int(pred_traj[0][1])), end)
            gen.gw.render()
            for k in range(1,L):
                S1 = [pred_traj[k-1][0]]
                S2 = [pred_traj[k-1][1]]
                a = model.predict(X,S1,S2)[0]
                state, reward, isdone, info = gen.gw.step(a)
                pred_traj.append([info['x'], info['y']])
                reward_temp += reward
                if pred_traj[-1][0] == end[0] and pred_traj[-1][1] == end[1]:
                    correct += 1
                    break
                if pred_traj[-1][0] == pred_traj[-2][0] and pred_traj[-1][0] == pred_traj[-2][1]:
                    break
                gen.gw.render()
            reward_total.append(reward_temp)
            total += 1
            if config.plot == True:
                s1 = np.append(s1test[j],end[0])
                s2 = np.append(s2test[j],end[1])
                gt_traj = np.stack((s1,s2),-1)
               # print("ground_truth path:",gt_traj)
               # print("pred_traj",np.array(pred_traj))
                visualize(gen.gw.m.get_image().T, gt_traj, np.array(pred_traj))

        sys.stdout.write("\r" + str(int(
                (float(i) / n_domains) * 100.0)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    print('Rollout Success rate: {:.2f}%'.format(100 * (correct / total)))
    print("Acc reward:{:.2f}".format(np.mean(reward_total)))


if __name__ == "__main__":
    main(config)













