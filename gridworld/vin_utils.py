import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# helper methods to print nice table (taken from CGT code)
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep


def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out


def flipkernel(kern):
    return kern[(slice(None, None, -1),) * 2 + (slice(None), slice(None))]


def conv2d_flipkernel(x, k, name=None):
    return tf.nn.conv2d(x, flipkernel(k), name=name,
                        strides=(1, 1, 1, 1), padding='SAME')


def in_bound(x, y, width, height):
    if 0 <= x < width and 0 <= y < height:
        return True
    else:
        return False


def getPos(width, height):
    w = np.zeros([width * height, 2])
    for i in range(width * height):
        w[i][0] = i%width
        w[i][1] = i/width
    return w


def extract_label(theta_matrix, start_pos, label_pos, discrete=True):
    if discrete :
        labels = []
        for i in range(len(start_pos)):
            label = []
            tmp = theta_matrix[i].toarray()[start_pos[i], label_pos[i]]
            for j in range(len(tmp)):
                if tmp[j] <= np.pi/8 or tmp[j] > 15*np.pi/8:
                    label.append(0)
                elif tmp[j] <= 3*np.pi/8 and tmp[j] > np.pi/8:
                    label.append(1)
                elif tmp[j] <= 5*np.pi/8 and tmp[j] > 3*np.pi/8:
                    label.append(2)
                elif tmp[j] <= 7*np.pi/8 and tmp[j] > 5*np.pi/8:
                    label.append(3)
                elif tmp[j] <= 9*np.pi/8 and tmp[j] > 7*np.pi/8:
                    label.append(4)
                elif tmp[j] <= 11*np.pi/8 and tmp[j] > 9*np.pi/8:
                    label.append(5)
                elif tmp[j] <= 13*np.pi/8 and tmp[j] > 11*np.pi/8:
                    label.append(6)
                elif tmp[j] <= 15*np.pi/8 and tmp[j] > 13*np.pi/8:
                    label.append(7)
            labels.append(label)
        labels = np.array(labels)
    else :
        labels = []
        for i in range(len(start_pos)):
            labels.append(theta_matrix[i].toarray()[start_pos[i], label_pos[i]])
        labels = np.array(labels)
    return labels

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
