import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from arcagi.plotting import cmap, norm

def first_not_zero(ar):
    for i in range(0,len(ar),1):
        if ar[i] != 0:
            return i

def last_not_zero(ar):
    for i in range(len(ar)-1,-1,-1):
        if ar[i] != 0:
            return i + 1

class Shapes:

    def __init__(self, grid, shapes):
        self.grid = grid # to get the original colors for each shape
        self.shapes = shapes
        self.color_to_shapes, self.shape_to_color = self.associate_color_shape()
        self.shapes_id, self.color_shapes_id = self.group_shapes()

    def associate_color_shape(self):
        color_to_shapes = {} # color ID: shape IDs
        shape_to_color = {} # shape ID: color ID
        for shape in range(1, self.shapes.max()+1):
            color_shape = int(self.grid[tuple(np.argwhere(self.shapes == shape)[0])])
            shape_to_color[shape] = color_shape
            if color_shape in color_to_shapes.keys():
                color_to_shapes[color_shape] += [shape]
            else:
                color_to_shapes[color_shape] = [shape]
        return color_to_shapes, shape_to_color

    def group_shapes(self):
        '''Returns a table: rows: colors, columns: shapes, values: count / coordinates x_0,y_0'''
        shapes_id = {}
        color_shapes_id = {}
        for color in self.color_to_shapes.keys():
            color_shapes_id[color] = {}
            for shape in self.color_to_shapes[color]:
                shape_0_1 = self.isolate_shape(shape, self.shapes)
                id_shape = self.shape_to_id(shape_0_1)
                if id_shape in shapes_id.keys():
                    shapes_id[id_shape] += 1
                else:
                    shapes_id[id_shape] = 1
                if id_shape in color_shapes_id[color].keys():
                    color_shapes_id[color][id_shape] += 1
                else:
                    color_shapes_id[color][id_shape] = 1
        return shapes_id, color_shapes_id

    def isolate_shape(self, shape, shapes):
        sh = np.where(shapes==shape, np.ones(shapes.shape, dtype=int), np.zeros(shapes.shape, dtype=int))
        sum_rows = sh.sum(axis=0)
        sum_cols = sh.sum(axis=1)
        x_0, y_0, x_1, y_1 = first_not_zero(sum_rows), first_not_zero(sum_cols), last_not_zero(sum_rows), last_not_zero(sum_cols)
        return sh[y_0:y_1, x_0:x_1]

    def shape_to_id(self, shape_0_1):
        shape_0_1 = shape_0_1.astype(str)
        for x in range(1,shape_0_1.shape[0]):
            shape_0_1[x,0] = '_'+str(shape_0_1[x,0])
        return ''.join(map(str, (shape_0_1.ravel().tolist())))

    def id_to_shape(self, shape_id):
        return np.array([list(s) for s in shape_id.split('_')]).astype(int)

    def table_colors_shapes(self):
        res = pd.DataFrame(self.color_shapes_id).fillna(0).astype(int)
        return res[sorted(res.columns)] # réordonne par couleurs croissantes

    def plot_shapes(self):
        '''
        Plots a table with index row: color ID, index column: shape ID, values: count
        '''
        table_colors_shapes = self.table_colors_shapes()
        fig, axs = plt.subplots(nrows=table_colors_shapes.shape[0]+1,
                                ncols=table_colors_shapes.shape[1]+1)

        for row in range(table_colors_shapes.shape[0]+1):
            for col in range(table_colors_shapes.shape[1]+1):
                axis = axs[row, col]
                if col == 0 and row == 0: # plot nothing
                    axis.axis('off')
                elif row == 0: # plot color
                    square = patches.Rectangle((0, 0), 1, 1, linewidth=0, edgecolor='none',
                                               facecolor=cmap.colors[table_colors_shapes.columns[col-1]])
                    axis.add_patch(square)
                    axis.set_ylim(0, 0.5)
                    axis.axis('off')
                elif col == 0: # plot shape in black and white
                    shape = self.id_to_shape(table_colors_shapes.index[row-1])
                    axis.imshow(shape, cmap='gray_r', vmin=0, vmax=1)
                    axis.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5) # adds a grid
                    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[]) # removes axes' ticks
                    axis.set_xticks([x-0.5 for x in range(1 + len(shape[0]))]) # adds x-ticks
                    axis.set_yticks([x-0.5 for x in range(1 + len(shape))]) # adds y-ticks
                elif col>0 and row>0:
                    axis.text(0.5, 0.5, str(table_colors_shapes.iloc[row-1, col-1]), fontsize=24, ha='center', va='center')
                    axis.axis('off')

        plt.show()

    def __repr__(self):
        self.plot_shapes()
        return ''

if __name__=='__main__':
    from arcagi.data import Arcagi2
    data = Arcagi2().get_data()
    import random
    # i = random.randint(1, 1000)
    i = 1
    key = list(data['training_challenges'])[i]
    task = data['training_challenges'][key]
    i = random.randint(1, len(task['train']))
    from arcagi.grid import Grid
    g = Grid(task['train'][i-1]['input'])
    s = Shapes(g.grid, g.shapes[2])
    print(g.grid)
    print(s.shapes_id)
