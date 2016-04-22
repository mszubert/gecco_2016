import random

import numpy


def get_homologous_nodes_and_leaves(ind1, ind2):
    homologous_internal_nodes = []
    homologous_leaves = []

    index_1 = 0
    index_2 = 0
    depth_stack = [0]
    while index_1 < len(ind1) and index_2 < len(ind2):
        depth = depth_stack.pop()
        if ind1[index_1].arity == 0 and ind2[index_2].arity == 0:
            homologous_leaves.append((index_1, index_2, depth))
        else:
            homologous_internal_nodes.append((index_1, index_2, depth))

        if ind1[index_1].arity == ind2[index_2].arity:
            depth_stack.extend([depth + 1] * ind1[index_1].arity)
            index_1 += 1
            index_2 += 1
        else:
            subtree_slice_1 = ind1.searchSubtree(index_1)
            index_1 = subtree_slice_1.stop
            subtree_slice_2 = ind2.searchSubtree(index_2)
            index_2 = subtree_slice_2.stop

    return homologous_internal_nodes, homologous_leaves


def get_node_depth(ind, node_index):
    stack = [0]
    for node in ind[:node_index]:
        depth = stack.pop()
        stack.extend([depth + 1] * node.arity)
    return stack.pop()


def homologous_lgx(ind1, ind2, library_selector, internal_bias, distance_measure, distance_threshold, max_height=None):
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    internal_nodes, leaves = get_homologous_nodes_and_leaves(ind1, ind2)
    if not leaves or random.random() < internal_bias:
        if len(internal_nodes) > 1:
            indices = random.choice(internal_nodes[1:])
        else:
            indices = internal_nodes[0]
    else:
        indices = random.choice(leaves)

    slice1 = ind1.searchSubtree(indices[0])
    slice2 = ind2.searchSubtree(indices[1])
    semantics1 = ind1.semantics[indices[0]]
    semantics2 = ind2.semantics[indices[1]]

    max_replacement_height = max_height - indices[2] if max_height is not None else numpy.inf
    if distance_measure(semantics1, semantics2) > distance_threshold:
        mid_point = (semantics1 + semantics2) / 2.0
        replacement, _ = library_selector(mid_point, max_height=max_replacement_height)
    else:
        replacement, _ = library_selector(numpy.nan, max_height=max_replacement_height)

    ind1[slice1] = replacement[:]
    ind2[slice2] = replacement[:]

    return ind1, ind2
