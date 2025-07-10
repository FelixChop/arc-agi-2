import sys
import types
import numpy as np

# Stub out arcagi.llm before importing Task to avoid optional dependencies
sys.modules['arcagi.llm'] = types.SimpleNamespace(
    llm=lambda *a, **k: None,
    prompt_input_output=lambda *a, **k: ''
)

from arcagi.task import Task


def make_task(train_pairs):
    return {
        'train': [{'input': inp, 'output': out} for inp, out in train_pairs],
        'test': []
    }


def test_same_shapes_true():
    train_pairs = [
        (np.zeros((2, 2), dtype=int).tolist(), np.ones((2, 2), dtype=int).tolist()),
        (np.zeros((2, 2), dtype=int).tolist(), np.ones((2, 2), dtype=int).tolist()),
    ]
    task = Task(make_task(train_pairs))
    assert task.is_same_size_each_input_output()
    assert task.is_same_size_all_input_output()


def test_varying_shapes_false():
    train_pairs = [
        (np.zeros((2, 2), dtype=int).tolist(), np.ones((2, 2), dtype=int).tolist()),
        (np.zeros((1, 1), dtype=int).tolist(), np.ones((1, 2), dtype=int).tolist()),
    ]
    task = Task(make_task(train_pairs))
    assert not task.is_same_size_each_input_output()
    assert not task.is_same_size_all_input_output()
