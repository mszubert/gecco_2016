import copy
from functools import wraps
import random


def static_limit(key, max_value):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if key(ind) > max_value:
                    new_inds[i] = copy.deepcopy(random.choice(keep_inds))
            return new_inds
        return wrapper
    return decorator


def internally_biased_node_selector(individual, bias):
    internal_nodes = []
    leaves = []

    for index, node in enumerate(individual):
        if node.arity == 0:
            leaves.append(index)
        else:
            internal_nodes.append(index)

    if internal_nodes and random.random() < bias:
        return random.choice(internal_nodes)
    else:
        return random.choice(leaves)


def one_point_xover_biased(ind1, ind2, node_selector):
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    index1 = node_selector(ind1)
    index2 = node_selector(ind2)
    slice1 = ind1.searchSubtree(index1)
    slice2 = ind2.searchSubtree(index2)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2
