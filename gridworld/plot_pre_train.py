import matplotlib.pyplot as plt
import glob
import sys
import re
import numpy as np

colors = ['blue', 'green', 'red', 'magenta', 'cyan', \
            'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkcyan']

def plot_pre_train_log():
    dir = 'results/'
    plt.figure()
    for idx, filename in enumerate(glob.glob(dir+'*.log')):
        with open(filename, 'r') as f:
            label = re.split('[\\\\/]', filename)[-1].replace('.log', '')
            color = colors[idx]
            x_axis, y_axis = [], []
            for idx, line in enumerate(f.readlines()):
                epoch, predict_acc, loss_v = line.split(', ')
                epoch, predict_acc, loss_v = int(epoch), float(predict_acc), float(loss_v)
                if epoch > 10:
                    break
                x_axis.append(epoch)
                y_axis.append(predict_acc)
            linestyle = '--' if 'dark' in color else '-'
            plt.plot(x_axis, y_axis, c=color, label=label, linestyle=linestyle)

    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(dir+'pre_train.png')
    plt.show()

if __name__ == '__main__':
    plot_pre_train_log()
