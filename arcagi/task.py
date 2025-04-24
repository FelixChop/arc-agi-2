import numpy as np

class Task():

    def __init__(self, task):
        # transforms lists within each task into np.array
        self.train = task['train']
        self.train  = [{k:np.array(i[k]) for k in i.keys()} for i in task['train']]
        self.test = task['test']
        self.test[0]['input'] = np.array(self.test[0]['input'])
        if self.test[0].get('output'):
            self.test[0]['output'] = np.array(self.test[0]['output'])

        # saves interesting info about the task
        self.training_examples_number = len(self.train) # number of training examples
        self.training_examples_sizes = [(self.train[i]['input'].shape,
                                         self.train[i]['output'].shape)
                                        for i in range(self.training_examples_number)] # shapes of grids
        self.test_size = self.test[0]['input'].shape

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

if __name__=='__main__':
    from arcagi.data import Arcagi2
    data = Arcagi2().get_data()
    import random
    i = random.randint(1, 1000)
    key = list(data['training_challenges'])[i]
    task = data['training_challenges'][key]
    t = Task(task)
    print({'key':key,
          'num_training_examples':t.training_examples_sizes,
          'each_input_output_have_same_size':t.is_same_size_each_input_output(),
          'all_input_output_have_same_size':t.is_same_size_all_input_output()})
