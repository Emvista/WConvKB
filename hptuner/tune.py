from itertools import product
from typing import Tuple, Iterable, Mapping, Any


def generate_grid(hyperparameter: Mapping[str, Iterable[Any]]) -> Iterable[Mapping[str, Iterable[Any]]]:
    """Generates an iterator over the cartesian product of hyperparameters values.

    :param hyperparameter: a mapping from hyperparameter name to possible values
    :return: an iterator over the cartesian product of hyperparameters values
    """
    # sort is used to make sure order is always the same
    sorted_keys = sorted(hyperparameter.keys())
    hpp = product(*[hyperparameter[k] for k in sorted_keys])
    for hp in hpp:
        grid_item = dict()
        for idx, key in enumerate(sorted_keys):
            grid_item[key] = hp[idx]
        yield grid_item
