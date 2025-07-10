from ollama import chat
model = 'gemma'

def llm(prompt):
    messages = [{'role': 'system', 'content': prompt_system()}, {'role': 'user','content': prompt}]
    response = chat(model=model, messages=messages)
    return response.message.content

def prompt_system():
    return '''
    You are an expert at logic. Your goal is to understand the logic that leads
    from input grids to output grids. There is a unique logic for all input and
    output grids.

    Grids are composed with squares. A grid size varies from 3 times 3 to 20
    times 20 for instance. Grids contain colored squares. Each square has a
    color ID ranging from 0 to 9.

    To help you with identifying the logic, we provide for each grid all the
    shapes that exist in it: their color, the count, etc. This will help you to
    check how shapes change from input to output grids, and ultimately surmise
    the logic that transforms shapes.

    You will be given input grids and output grids and the goal is to define
    what transformation leads from one to another.

    Once you think you understand the logic, be sure to test it against all
    input grids to make sure you retrieve the output grids with the logic. It
    is extremely important that your logic is correct. An error in the logic is
    fatal.
    '''

def prompt_grid_shapes(grid, shapes):
    prompt_color_shapes = ''
    for color in shapes.color_shapes_id.keys():
        for shape in shapes.color_shapes_id[color]:
            prompt_color_shapes += '- color ID '+str(color)+': '+str(shapes.color_shapes_id[color])+'shapes that have the following ID '+shape+'\n'
    return f'''
    The grid's attributes are:
    - {grid.grid.shape[0]} squares on the x-axis
    - {grid.grid.shape[1]} squares on the y-axis
    - background color ID is {grid._background}
    - colors ID proportions are in python dictionary type: {grid.colors}

    Following is the list of all its shapes with their color and their count and their ID:
    {prompt_color_shapes}
    '''

if __name__=='__main__':
    prompt = '''
    Here is the input grid: [[0,0], [0,1]]. Here is the output grid [[1,1], [1,0]].
    '''
    print(llm(prompt))
