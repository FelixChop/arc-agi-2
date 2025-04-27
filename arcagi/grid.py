import matplotlib.pyplot as plt
from arcagi.plotting import cmap, plot_grid
import numpy as np
import pandas as pd
import seaborn as sns

class Grid():

    def __init__(self, grid):
        self.grid = np.array(grid)
        self.create_background()
        self.shapes = {i:np.zeros(self.grid.shape, dtype=int) for i in [1,2]}
        self.create_shapes(1)
        self.create_shapes(2)
        self.colors = self.count_colors(as_dict=False)
        self.shape_splits = {i:self.create_image_splits(shape_type=i, width=1) for i in [1,2]}

    def count_colors(self, normalize:bool = False, as_dict:bool = True):
        res = pd.Series(self.grid.reshape(self.grid.size)).value_counts(normalize=normalize).sort_index()
        return res.to_dict() if as_dict else res

    def neighbor_w(self, cursor:tuple): # west
        return (cursor[0]-1,cursor[1]) if cursor[0]-1>=0 else None

    def neighbor_s(self, cursor:tuple): # south
        return (cursor[0],cursor[1]+1) if cursor[1]+1<self.grid.shape[1] else None

    def neighbor_e(self, cursor:tuple): # east
        return (cursor[0]+1,cursor[1]) if cursor[0]+1<self.grid.shape[0] else None

    def neighbor_n(self, cursor:tuple): # north
        return (cursor[0],cursor[1]-1) if cursor[1]-1>=0 else None

    def neighbor_sw(self, cursor:tuple): # south west
        return (cursor[0]-1,cursor[1]+1) if cursor[0]-1>=0 and cursor[1]+1<self.grid.shape[1] else None

    def neighbor_se(self, cursor:tuple): # south east
        return (cursor[0]+1,cursor[1]+1) if cursor[0]+1<self.grid.shape[0] and cursor[1]+1<self.grid.shape[1] else None

    def neighbor_ne(self, cursor:tuple): # north east
        return (cursor[0]+1,cursor[1]-1) if cursor[0]+1<self.grid.shape[0] and cursor[1]-1>=0 else None

    def neighbor_nw(self, cursor:tuple): # north west
        return (cursor[0]-1,cursor[1]-1) if cursor[0]-1>=0 and cursor[1]-1>=0 else None

    def neighbors1(self, cursor:tuple):
        return self._c([self.neighbor_w(cursor),
                        self.neighbor_s(cursor),
                        self.neighbor_e(cursor),
                        self.neighbor_n(cursor)])

    def neighbors2(self, cursor:tuple):
        return self._c([self.neighbor_w(cursor),
                        self.neighbor_s(cursor),
                        self.neighbor_e(cursor),
                        self.neighbor_n(cursor),
                        self.neighbor_sw(cursor),
                        self.neighbor_se(cursor),
                        self.neighbor_nw(cursor),
                        self.neighbor_ne(cursor)])

    def _c(self, l):
        return [el for el in l if el is not None]

    def get_all_neighbors_same_color(self, cursor:tuple, known_neighbors:list = [], is_neighbor2:bool = False) -> list:
        if cursor in known_neighbors:
            return []
        known_neighbors.append(cursor)
        neighbors = self.neighbors2(cursor) if is_neighbor2 else self.neighbors1(cursor)
        new_neighbors_same_color = [n for n in neighbors if self.grid[cursor] == self.grid[n] and n not in known_neighbors]
        for neighbor in new_neighbors_same_color:
            neighbors = self.get_all_neighbors_same_color(cursor=neighbor, known_neighbors=known_neighbors, is_neighbor2=is_neighbor2)
            for n in neighbors:
                if n not in known_neighbors:
                    known_neighbors.append(n)
        return known_neighbors

    def create_shapes(self, shape_type:int):
        while (self.shapes[shape_type] + self.background).min() == 0:
            point_without_shape = np.unravel_index((self.shapes[shape_type] + self.background).argmin(),
                                                   self.shapes[shape_type].shape)
            point_color = self.grid[point_without_shape]
            # Create new shape ID for the point with value 0
            shape_number = self.shapes[shape_type].max() + 1
            self.shapes[shape_type][point_without_shape] = shape_number
            # Get all neighbors of the same color with the same shape
            for close_neighbor in self.get_all_neighbors_same_color(cursor=point_without_shape,
                                                                    known_neighbors=[],
                                                                    is_neighbor2=shape_type==2):
                self.shapes[shape_type][close_neighbor] = shape_number
            self.create_shapes(shape_type=shape_type)

    def get_border(self, grid, width:int = 1, increment=0):
        if grid.shape[0] > 2 and grid.shape[1] > 2:
            rows, cols = grid.shape
            border = []
            for col in range(cols):
                border.append((0+increment, col+increment))
            for row in range(rows):
                border.append((row+increment, cols-1+increment))
            for col in range(cols - 1, -1, -1):
                border.append((rows - 1+increment, col+increment))
            for row in range(rows - 1, 0, -1):
                border.append((row+increment, 0+increment))
            if width == 1:
                return border
            else:
                return border + self.get_border(grid[1:(rows-1), 1:(cols-1)], width-1, increment+1)

    def get_opposite_borders_left_right(self, grid, width:int = 1):
        return [(i,j) for i in range(grid.shape[0]) for j in range(width)], \
               [(grid.shape[0]-i-1,grid.shape[1]-j-1) for i in range(grid.shape[0]) for j in range(width)]

    def get_opposite_borders_top_bottom(self, grid, width:int = 1):
        return [(i,j) for j in range(grid.shape[1]) for i in range(width)], \
               [(grid.shape[0]-i-1,grid.shape[1]-j-1) for j in range(grid.shape[1]) for i in range(width)]

    def create_image_splits(self, shape_type:int, width:int):
        # An image split splits the image into sub images
        # Basically, there exists a path between 2 same-color points from opposite sides
        result = []
        border_top, border_bottom = self.get_opposite_borders_top_bottom(self.grid, width=width)
        border_left, border_right = self.get_opposite_borders_left_right(self.grid, width=width)
        shapes_top = set([self.shapes[shape_type][point] for point in border_top])
        shapes_bottom = set([self.shapes[shape_type][point] for point in border_bottom])
        shapes_left = set([self.shapes[shape_type][point] for point in border_left])
        shapes_right = set([self.shapes[shape_type][point] for point in border_right])
        for shape in shapes_top:
            if shape in shapes_bottom and shape not in result:
                result.append(int(shape))
        for shape in shapes_left:
            if shape in shapes_right and shape not in result:
                result.append(int(shape))
        return result

    def get_background(self):
        dict_most_used = self.count_colors(normalize=True, as_dict=False)
        color = list(dict_most_used.sort_values(ascending=False).to_dict().keys())[0]
        if dict_most_used[color] > .5:
            return color
        elif dict_most_used[color] > .4 and color in self.shape_splits[2]:
            return color
        else:
            return

    def has_background(self):
        return type(self._background) == int

    def create_background(self):
        self._background = self.get_background()
        if self.has_background():
            self.background = np.where(self.grid == self._background,
                                       np.ones(self.grid.shape, dtype=int),
                                       np.zeros(self.grid.shape, dtype=int))
        else:
            self.background = np.zeros(self.grid.shape, dtype=int)

    def points_inside_shape(shape):
        # Identify points that are contoured by/within/inside the shape
        pass

    def __repr__(self):
        plot_grid(self.grid)
        return ''

if __name__=='__main__':
    from arcagi.data import Arcagi2
    data = Arcagi2().get_data()
    import random
    i = random.randint(1, 1000)
    key = list(data['training_challenges'])[i]
    task = data['training_challenges'][key]
    from arcagi.task import Task
    t = Task(task)
    i = random.randint(0, len(t.train)-1)
    for io in ['input', 'output']:
        print(io.capitalize())
        g = Grid(t.train[i][io])
        print('grid')
        print(g.grid)
        if g.background.sum()==0:
            print('no background found')
        else:
            print('background:')
            print(g.background)
        print('shapes 1:')
        print(g.shapes[1])
        print('shapes 2:')
        print(g.shapes[2])
        print('grid split:')
        print(g.shape_splits[2])
        print()
