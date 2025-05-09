import numpy as np
from arcagi.plotting import plot_task
from arcagi.grid import Grid
from arcagi.shapes import Shapes
from utils import io, tt
from arcagi.llm import llm, prompt_input_output

class Task():

    def __init__(self, task):
        # transforms lists within each task into np.array
        self.task = task
        self.train = task['train']
        self.train  = [{k:np.array(i[k]) for k in i.keys()} for i in task['train']]
        self.test = task['test']

        # saves interesting info about the task
        self.training_examples_number = len(self.train) # number of training examples
        self.training_examples_sizes = [(self.train[i]['input'].shape,
                                         self.train[i]['output'].shape)
                                        for i in range(self.training_examples_number)] # shapes of grids

        # creates Grids and Shapes

    def is_same_size_each_input_output(self):
        # checks if each pair of input/output is stable by size
        l = [self.training_examples_sizes[i][0] == self.training_examples_sizes[i][1]
             for i in range(len(self.training_examples_sizes))]
        return len(l)==sum(l)

    def is_same_size_all_input_output(self):
        # checks if all pais of input/output have exactly the same size
        shape = self.training_examples_sizes[0][0]
        l = [self.training_examples_sizes[i][j] == shape for j in (0,1)
             for i in range(len(self.training_examples_sizes))]
        return len(l)==sum(l)

    def __repr__(self):
        plot_task(self.task)
        return ''

    def create_grid_and_shapes_to_text(self):
        res = {}
        for a in tt():
            res[a] = {}
            for b in range(len(self.task[a])):
                res[a][b] = {}
                for c in io():
                    if a == 'test' and c == 'output':
                        continue # do not handle test data
                    g = Grid(self.task[a][b][c])
                    s = Shapes(g.grid, g.shapes[2])
                    res[a][b][c] = self.grid_and_shapes_to_text(g,s)
        return res

    def grid_and_shapes_to_text(self, grid, shapes):
        return f'''
        background color ID: {grid._background},
        task delimiter color ID: {grid.shape_splits[2][0]},
        for each color ID, it tells how many shape ID there are: {shapes.color_shapes_id}
        '''

    def tasks_to_text(self):
        grid_and_shapes_to_text = self.create_grid_and_shapes_to_text()
        res = {}
        for i in range(len(self.train)):
            prompt = prompt_input_output(grid_and_shapes_to_text['train'][i]['input'],
                                         grid_and_shapes_to_text['train'][i]['output'])
            print(prompt)
            res[i] = llm(prompt)
        self.analysis = res

if __name__=='__main__':
    from arcagi.data import Arcagi2
    data = Arcagi2().get_data()
    task = Task(data['training_challenges']['009d5c81'])
    task.analyze()
