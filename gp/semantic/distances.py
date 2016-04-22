import numpy
from gp.experiments import symbreg


def euclidean_distance(a, b, axis=0):
    diff = a - b
    squares = numpy.multiply(diff, diff)
    return numpy.sqrt(numpy.sum(squares, axis=axis))


def cumulative_absolute_difference(a, b, axis=0):
    return numpy.sum(numpy.abs(a - b), axis=axis)


def mean_canberra(a, b, axis=0, eps=10e-6):
    abs_difference = numpy.abs(a - b)
    protected_canberra = symbreg.numpy_protected_div_dividend(abs_difference, numpy.abs(a) + numpy.abs(b))
    protected_canberra[abs_difference < eps] = 0
    return numpy.mean(protected_canberra, axis=axis)
