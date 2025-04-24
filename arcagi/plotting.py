import matplotlib.pyplot as plt
from matplotlib import colors
import random

# Colors:
# 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_task(task, task_solutions, i, t):
    """    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app    """
    fs = 12 # font size
    num_train = len(task['train'])
    num_test  = len(task['test']) # =1 for training
    columns = num_train + num_test # number of columns

    fig, axs  = plt.subplots(2, columns, figsize=(2*columns,2*2)) # number of rows = 2: input and output
    plt.suptitle(f'Set #{i}, {t}:', fontsize=fs, fontweight='bold', y=1)

    for j in range(num_train):
        plot_one(axs[0, j], j, task, 'train', 'input')
        plot_one(axs[1, j], j, task, 'train', 'output')

    for j in range(num_test):
        plot_one(axs[0, num_train+j], 0, task, 'test', 'input')

    axs[1, num_train+j].imshow(task_solutions, cmap=cmap, norm=norm)
    axs[1, num_train+j].grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    axs[1, num_train+j].set_yticks([x-0.5 for x in range(1 + len(task_solutions))])
    axs[1, num_train+j].set_xticks([x-0.5 for x in range(1 + len(task_solutions[0]))])
    axs[1, num_train+j].set_xticklabels([])
    axs[1, num_train+j].set_yticklabels([])
    axs[1, num_train+j].set_title('Test output')

    axs[1, j+1] = plt.figure(1).add_subplot(111)
    axs[1, j+1].set_xlim([0, num_train+1])

    for m in range(1, num_train):
        axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color = 'black')

    axs[1, j+1].plot([num_train,num_train],[0,1],'-', linewidth=3, color = 'black')
    axs[1, j+1].axis("off")

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')

    plt.tight_layout()

    print(f'#{i}, {t}') # for fast and convenience search
    plt.show()

    print()


def plot_one(ax, i, task, train_or_test, input_or_output):
    fs = 12
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)

    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])

    ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs-2)

if __name__=='__main__':
    from arcagi.data import Arcagi2
    data = Arcagi2().get_data()
    i = random.randint(1, 1000)
    key = list(data['training_challenges'])[i]
    task = data['training_challenges'][key]
    task_solution = data['training_solutions'][key][0]
    plot_task(task,  task_solution, i, key)
