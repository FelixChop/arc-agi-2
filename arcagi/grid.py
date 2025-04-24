import matplotlib.pyplot as plt
from arcagi.plotting import cmap
import numpy as np
import pandas as pd
import seaborn as sns

class Grid():

    def __init__(self, grid):
        self.grid = np.array(grid)
        self.shapes1 = np.zeros(self.grid.shape, dtype=int)
        self.shapes2 = np.zeros(self.grid.shape, dtype=int)
        self.create_shapes1()
        self.create_shapes2()
        self.colors = self.count_colors(as_dict=False)
        self.shape_splits1 = self.create_image_splits(width=1,shape=1)
        self.shape_splits2 = self.create_image_splits(width=1,shape=2)
        self._background = self.get_background()
        if type(self._background) == int:
            self.background = np.where(self.grid == self._background, np.ones(self.grid.shape, dtype=int), np.zeros(self.grid.shape, dtype=int))

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

    def create_shapes1(self):
        while self.shapes1.min()==0:
            point_without_shape = np.unravel_index(self.shapes1.argmin(), self.shapes1.shape)
            point_color = self.grid[point_without_shape]
            # Create new shape ID for the point with value 0
            shape_number = self.shapes1.max() + 1
            self.shapes1[point_without_shape] = shape_number
            # Get all neighbors of the same color with the same shape
            for close_neighbor in self.get_all_neighbors_same_color(cursor=point_without_shape, known_neighbors=[], is_neighbor2=False):
                self.shapes1[close_neighbor] = shape_number
            self.create_shapes1()

    def create_shapes2(self):
        while self.shapes2.min()==0:
            point_without_shape = np.unravel_index(self.shapes2.argmin(), self.shapes2.shape)
            point_color = self.grid[point_without_shape]
            shape_number = self.shapes2.max() + 1
            self.shapes2[point_without_shape] = shape_number
            for close_neighbor in self.get_all_neighbors_same_color(cursor=point_without_shape, known_neighbors=[], is_neighbor2=True):
                self.shapes2[close_neighbor] = shape_number
            self.create_shapes2()

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

    def create_image_splits(self, width:int = 1, shape:int = 1):
        # An image split splits the image into sub images
        # Basically, there exists a path between 2 same-color points from opposite sides
        result = []
        shape_grid = self.shapes1 if shape==1 else self.shapes2
        border_top, border_bottom = self.get_opposite_borders_top_bottom(self.grid, width=width)
        border_left, border_right = self.get_opposite_borders_left_right(self.grid, width=width)
        shapes_top = set([shape_grid[point] for point in border_top])
        shapes_bottom = set([shape_grid[point] for point in border_bottom])
        shapes_left = set([shape_grid[point] for point in border_left])
        shapes_right = set([shape_grid[point] for point in border_right])
        for shape in shapes_top:
            if shape in shapes_bottom and shape not in result:
                result.append(shape)
        for shape in shapes_left:
            if shape in shapes_right and shape not in result:
                result.append(shape)
        return result

    def get_background(self):
        dict_most_used = self.count_colors(normalize=True, as_dict=False)
        color = list(dict_most_used.sort_values(ascending=False).to_dict().keys())[0]
        if dict_most_used[color] > .5:
            return color
        elif dict_most_used[color] > .3 and color in self.shape_splits2:
            return color
        else:
            return

    def points_inside_shape(shape):
        # Identify points that are contoured by/within/inside the shape
        pass

    def describe_plot(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(13, 3), ncols=4)
        fig.tight_layout()
        fig.suptitle('Raw grid and corresponding shapes found')

        self._plot(ax=ax1, title="Raw grid", grid=self.grid, cmap=cmap)
        self._plot(ax=ax2, title='#{} shapes-1 found'.format(self.shapes1.max()), grid=self.shapes1, cmap=sns.color_palette(palette='rocket', as_cmap=True))
        self._plot(ax=ax3, title='#{} shapes-2 found'.format(self.shapes2.max()), grid=self.shapes2, cmap=sns.color_palette(palette='rocket', as_cmap=True))
        if self._background:
            self._plot(ax=ax4, title='Backgroud', grid=self.background, cmap='gray')

        plt.show()

    def _plot(self, ax, title, grid, cmap):
        ax.imshow(grid, cmap=cmap)
        ax.set_title(title)
        ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + grid.shape[1])])
        ax.set_yticks([x-0.5 for x in range(1 + grid.shape[0])])

    def plot(self, indices_highlight):
        array = np.zeros(self.grid.shape, dtype=int)
        for idx in indices_highlight:
            array[idx] = 1
        fig, ax = plt.subplots(figsize=(2,2))
        ax.imshow(array, cmap='gray', vmin=0, vmax=1)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + array.shape[1])])
        ax.set_yticks([x-0.5 for x in range(1 + array.shape[0])])
        fig

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
    g = Grid(t.train[i-1]['input'])
    g.describe_plot()
