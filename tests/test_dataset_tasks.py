import sys
import types
import os

# Ensure the arcagi package is importable when running the tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Stub out arcagi.llm before importing Task
sys.modules['arcagi.llm'] = types.SimpleNamespace(
    llm=lambda *a, **k: None,
    prompt_input_output=lambda *a, **k: ''
)

from arcagi.data import Arcagi2
from arcagi.task import Task


def load_task(key: str) -> Task:
    data = Arcagi2().get_data(add_solutions=False)
    return Task(data['training_challenges'][key])


def test_dataset_equal_shapes():
    task = load_task('009d5c81')
    assert task.is_same_size_each_input_output()
    assert task.is_same_size_all_input_output()


def test_dataset_varying_shapes():
    task = load_task('00576224')
    assert not task.is_same_size_each_input_output()
    assert not task.is_same_size_all_input_output()
