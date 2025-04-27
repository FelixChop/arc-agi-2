import matplotlib.pyplot as plt
from matplotlib import colors
import random

# Colors: 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_task(task):
    cols = len(task['train']) + len(task['test']) # number of columns
    fig, axs  = plt.subplots(2, cols, figsize=(2*cols,2*2)) # number of rows = 2: input and output

    for train_or_test in ['train', 'test']:
        for i in range(len(task[train_or_test])):
            offset = 0 if train_or_test == 'train' else len(task['train'])
            l = ['input', 'output'] if 'output' in task[train_or_test][i] else ['input'] # to make it work with the test_challenges dataset
            for k,input_or_output in enumerate(l):
                axs[k, offset+i].imshow(task[train_or_test][i][input_or_output], cmap=cmap, norm=norm) # plot colors
                axs[k, offset+i].grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5) # adds a grid
                plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[]) # removes axes' ticks
                axs[k, offset+i].set_xticks([x-0.5 for x in range(1 + len(task[train_or_test][i][input_or_output][0]))]) # adds x-ticks
                axs[k, offset+i].set_yticks([x-0.5 for x in range(1 + len(task[train_or_test][i][input_or_output]))]) # adds y-ticks
                axs[k, offset+i].set_title(train_or_test + ' ' + input_or_output, fontsize=10)

    plt.show()

def plot_grid(grid):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5) # adds a grid
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[]) # removes axes' ticks
    ax.set_xticks([x-0.5 for x in range(1 + len(grid[0]))]) # adds x-ticks
    ax.set_yticks([x-0.5 for x in range(1 + len(grid))]) # adds y-ticks
    plt.show()

if __name__=='__main__':
    from arcagi.data import Arcagi2
    data = Arcagi2().get_data()
    task = data['training_challenges'][list(data['training_challenges'])[random.randint(1, 1000)]]
    plot_task(task)
