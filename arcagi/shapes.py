import numpy as np
import matplotlib.pyplot as plt

def first_not_zero(ar):
    for i in range(0,len(ar),1):
        if ar[i] != 0:
            return i

def last_not_zero(ar):
    for i in range(len(ar)-1,-1,-1):
        if ar[i] != 0:
            return i + 1

class Shapes():

    def __init__(self, shapes, grid):
        self.shapes = shapes
        self.grid = grid
        self.associate_color_shape()
        self.group_shapes()

    def associate_color_shape(self):
        color_to_shapes = {} # color ID: shape IDs
        shape_to_color = {} # shape ID: color ID
        for shape in range(1,self.shapes.max()+1):
            color_shape = self.grid[tuple(np.argwhere(self.shapes == shape)[0])]
            shape_to_color[shape] = color_shape
            if color_shape in color_to_shapes.keys():
                color_to_shapes[color_shape] += [shape]
            else:
                color_to_shapes[color_shape] = [shape]
        self.color_to_shapes, self.shape_to_color = color_to_shapes, shape_to_color

    def group_shapes(self):
        '''Returns a table: rows: colors, columns: shapes, values: count / coordinates x_0,y_0'''
        shapes_id = {}
        color_shapes_id = {}
        for color in self.color_to_shapes.keys():
            color_shapes_id[color] = {}
            for shape in self.color_to_shapes[color]:
                shape_0_1 = self.isolate_shape(shape)
                id_shape = self.shape_to_id(shape_0_1)
                if id_shape in shapes_id.keys():
                    shapes_id[id_shape] += 1
                else:
                    shapes_id[id_shape] = 1
                if id_shape in color_shapes_id[color].keys():
                    color_shapes_id[color][id_shape] += 1
                else:
                    color_shapes_id[color][id_shape] = 1
        self.shapes_id, self.color_shapes_id = shapes_id, color_shapes_id

    def isolate_shape(self, shape):
        sh = np.where(self.shapes==shape, np.ones(self.shapes.shape, dtype=int), np.zeros(self.shapes.shape, dtype=int))
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
        # shape_id = str(shape_id) if type(shape_id)==float else shape_id
        return np.array([list(s) for s in shape_id.split('_')]).astype(int)

if __name__=='__main__':
    from arcagi.data import Arcagi2
    data = Arcagi2().get_data()
    import random
    i = random.randint(1, 1000)
    key = list(data['training_challenges'])[i]
    task = data['training_challenges'][key]
    from arcagi.task import Task
    t = Task(task)
    i = random.randint(1, len(t.train))
    from arcagi.grid import Grid
    g = Grid(t.train[i-1]['input'])
    s = Shapes(g.shapes2, g.grid)
    fig, axes = plt.subplots(figsize=(10, 5), ncols=len(list(s.shapes_id.keys())))
    for i in range(len(list(s.shapes_id.keys()))):
        g._plot(axes[i], '', s.id_to_shape(list(s.shapes_id.keys())[i]), 'gray')
    plt.show()
