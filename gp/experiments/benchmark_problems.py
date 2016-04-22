from functools import partial
import math

import numpy
import inspect


# 4 * x^4 + 3 * x^3 + 2 * x^2 + x
def mod_quartic(x):
    return x * (1 + x * (2 + x * (3 + x * 4)))


# Koza-1: x^4 + x^3 + x^2 + x
def quartic(x):
    return x * (1 + x * (1 + x * (1 + x)))


# Koza-2: x^5 - 2x^3 + x
def quintic(x):
    return x * (1 - x * x * (2 - x * x))


# Koza-3: x^6 - 2x^4 + x^2
def sextic(x):
    return x * x * (1 - x * x * (2 - x * x))


# x^7 - 2x^6 + x^5 - x^4 + x^3 - 2x^2 + x
def septic(x):
    return x * (1 - x * (2 - x * (1 - x * (1 - x * (1 - x * (2 - x))))))


# sum_{1}^9{x^i}
def nonic(x):
    return x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x))))))))


# x^3 + x^2 + x
def nguyen1(x):
    return x * (1 + x * (1 + x))


# x^5 + x^4 + x^3 + x^2 + x
def nguyen3(x):
    return x * (1 + x * (1 + x * (1 + x * (1 + x))))


# x^6 + x^5 + x^4 + x^3 + x^2 + x
def nguyen4(x):
    return x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x)))))


def nguyen5(x):
    return math.sin(x * x) * math.cos(x) - 1


def nguyen6(x):
    return math.sin(x) + math.sin(x * (1 + x))


def nguyen7(x):
    return math.log(x + 1) + math.log(x * x + 1)


def nguyen9(x, y):
    return math.sin(x) + math.sin(y * y)


def nguyen10(x, y):
    return 2 * math.sin(x) * math.cos(y)


def keijzer1(x):
    return 0.3 * x * math.sin(2 * math.pi * x)


def keijzer4(x):
    return x ** 3 * math.exp(-x) * math.cos(x) * math.sin(x) * (math.sin(x) ** 2 * math.cos(x) - 1)


def keijzer11(x, y):
    return (x * y) + math.sin((x - 1) * (y - 1))


def keijzer12(x, y):
    return x ** 4 - x ** 3 + (y ** 2 / 2.0) - y


def keijzer13(x, y):
    return 6 * math.sin(x) * math.cos(y)


def keijzer14(x, y):
    return 8.0 / (2 + x ** 2 + y ** 2)


def r1(x):
    return ((x + 1) ** 3) / (x ** 2 - x + 1)


def r2(x):
    return (x ** 5 - (3 * (x ** 3)) + 1) / (x ** 2 + 1)


def pagie1(x, y):
    return (1 / (1 + x ** -4)) + (1 / (1 + y ** -4))


univariate_problems = [mod_quartic, quartic, quintic, sextic, septic, nonic, nguyen5, nguyen6, keijzer1, r1, r2]
bivariate_problems = [nguyen9, nguyen10, keijzer11, keijzer12, keijzer13, keijzer14]

for problem in univariate_problems + [nguyen4]:
    problem.sample_features = partial(numpy.random.uniform, low=-1, high=1)
    problem.sample_features_evenly = partial(numpy.linspace, start=-1, stop=1)

for problem in bivariate_problems:
    problem.sample_features = partial(numpy.random.uniform, low=-1, high=1)
    problem.sample_features_evenly = lambda num: numpy.dstack(
        numpy.meshgrid(numpy.linspace(start=-1, stop=1, num=int(numpy.sqrt(num))),
                       numpy.linspace(start=-1, stop=1, num=int(numpy.sqrt(num)))))

keijzer4.sample_features = partial(numpy.random.uniform, low=0, high=10)
keijzer4.sample_features_evenly = partial(numpy.linspace, start=0, stop=10)

pagie1.sample_features_evenly = lambda num: numpy.dstack(
    numpy.meshgrid(numpy.linspace(start=-5, stop=5, num=int(numpy.sqrt(num))),
                   numpy.linspace(start=-5, stop=5, num=int(numpy.sqrt(num)))))


def get_training_set(target_func, num_points=20, sample_evenly=False):
    arity = len(inspect.getargspec(target_func)[0])
    if sample_evenly:
        predictors = target_func.sample_features_evenly(num=num_points).reshape((num_points, arity))
    else:
        predictors = target_func.sample_features(size=(num_points, arity))
    response = numpy.array([target_func(*x) for x in predictors])
    return predictors, response
