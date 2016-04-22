import numpy
from deap import gp


def numpy_protected_div_dividend(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = left[numpy.isinf(x)]
            x[numpy.isnan(x)] = left[numpy.isnan(x)]
        elif numpy.isinf(x) or numpy.isnan(x):
            x = left
    return x


def numpy_protected_log_abs(x):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.log(numpy.abs(x))
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = -1e300
            x[numpy.isnan(x)] = 0
        elif numpy.isinf(x):
            x = -1e300
        elif numpy.isnan(x):
            x = 0
    return x


def get_numpy_pset(arity, div_function=None, log_function=None, prefix="ARG"):
    if div_function is None:
        div_function = numpy_protected_div_dividend

    if log_function is None:
        log_function = numpy_protected_log_abs

    pset = gp.PrimitiveSet("MAIN", arity, prefix=prefix)
    pset.addPrimitive(numpy.add, 2)
    pset.addPrimitive(numpy.subtract, 2)
    pset.addPrimitive(numpy.multiply, 2)
    pset.addPrimitive(div_function, 2)
    pset.addPrimitive(log_function, 1)
    pset.addPrimitive(numpy.cos, 1)
    pset.addPrimitive(numpy.sin, 1)
    pset.addPrimitive(numpy.exp, 1)
    return pset
