from collections import defaultdict
import logging
import random

from deap import tools
import numpy
import time

from gp.semantic import semantics


def breed(parents, toolbox, xover_prob, mut_prob):
    offspring = [toolbox.clone(ind) for ind in parents]

    for i in range(1, len(offspring), 2):
        if random.random() < xover_prob:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            max_age = max(offspring[i - 1].age, offspring[i].age)
            offspring[i].age = offspring[i - 1].age = max_age
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mut_prob:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def find_pareto_front(population):
    """Finds a subset of nondominated indivuals in a given list

    :param population: a list of individuals
    :return: a set of indices corresponding to nondominated individuals
    """

    pareto_front = set(range(len(population)))

    for i in range(len(population)):
        if i not in pareto_front:
            continue

        ind1 = population[i]
        for j in range(i + 1, len(population)):
            ind2 = population[j]

            # if individuals are equal on all objectives, mark one of them (the first encountered one) as dominated
            # to prevent excessive growth of the Pareto front
            if ind2.fitness.dominates(ind1.fitness) or ind1.fitness == ind2.fitness:
                pareto_front.discard(i)

            if ind1.fitness.dominates(ind2.fitness):
                pareto_front.discard(j)

    return pareto_front


def reduce_population(population, tournament_size, target_popsize, nondominated_size):
    num_iterations = 0
    new_population_indices = list(range(len(population)))
    while len(new_population_indices) > target_popsize and len(new_population_indices) > nondominated_size:
        if num_iterations > 10e6:
            logging.info("Pareto front size may be exceeding the size of population! Stopping the execution!")
            # random.sample(new_population_indices, len(new_population_indices) - target_popsize)
            exit()
        num_iterations += 1
        tournament_indices = random.sample(new_population_indices, tournament_size)
        tournament = [population[index] for index in tournament_indices]
        nondominated_tournament = find_pareto_front(tournament)
        for i in range(len(tournament)):
            if i not in nondominated_tournament:
                new_population_indices.remove(tournament_indices[i])
    population[:] = [population[i] for i in new_population_indices]


def assign_semantic_novelty_fitness(population, k, distance_measure, novelty_scale_function=None):
    finite_inds = []
    finite_semantics = []
    for ind in population:
        if numpy.isfinite(numpy.sum(ind.semantics[0])):
            finite_semantics.append(ind.semantics[0])
            finite_inds.append(ind)
        else:
            ind.fitness.values = (ind.error, 0)

    semantic_array = numpy.array(finite_semantics)
    for index, ind in enumerate(finite_inds):
        distances = distance_measure(semantic_array, semantic_array[index], axis=1)
        if k + 1 < len(distances):
            smallest_k_distances = numpy.partition(distances, k + 1)
            novelty = numpy.mean(smallest_k_distances[:k + 1])
        else:
            novelty = numpy.mean(distances)

        if novelty_scale_function is not None:
            novelty = novelty_scale_function(ind, novelty)
        ind.fitness.values = (ind.error, novelty)


def assign_semantic_novelty_angle_fitness(population, k, target):
    finite_inds = []
    finite_semantics = []
    for ind in population:
        if numpy.isfinite(numpy.sum(ind.semantics[0])):
            finite_semantics.append(ind.semantics[0])
            finite_inds.append(ind)
        else:
            ind.fitness.values = (ind.error, 0)

    unit_residuals = [semantics.calc_unit_vector(target - sem) for sem in finite_semantics]
    unit_residual_array = numpy.array(unit_residuals)

    for index, ind in enumerate(finite_inds):
        dot_products = numpy.dot(unit_residual_array, unit_residual_array[index])
        angles = numpy.arccos(numpy.clip(dot_products, -1.0, 1.0))
        if k + 1 < len(angles):
            smallest_k_angles = numpy.partition(angles, k + 1)
            novelty = numpy.mean(smallest_k_angles[:k + 1])
        else:
            novelty = numpy.mean(angles)
        ind.fitness.values = (ind.error, novelty)


def assign_density_fitness(population, tag_depth):
    tag_density = defaultdict(int)
    for ind in population:
        ind.tag = build_tag(ind, tag_depth)
        tag_density[ind.tag] += 1

    for ind in population:
        ind.fitness.values = (ind.error, tag_density[ind.tag])


def build_tag(ind, tag_depth):
    tag = []
    depth_stack = [0]
    for node in ind:
        depth = depth_stack.pop()
        if depth <= tag_depth:
            tag.append(node.name)
        depth_stack.extend([depth + 1] * node.arity)
    return ":".join(tag)


def assign_pure_fitness(population):
    for ind in population:
        ind.fitness.values = (ind.error,)


def assign_age_fitness(population):
    for ind in population:
        ind.fitness.values = (ind.error, ind.age)


def pareto_optimization(population, toolbox, xover_prob, mut_prob, ngen, tournament_size, num_randoms=1, archive=None,
                        stats=None, calc_pareto_front=True, verbose=False):
    start = time.time()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'cpu_time'] + (stats.fields if stats else [])

    target_popsize = len(population)
    for ind in population:
        ind.error = toolbox.evaluate_error(ind)[0]
    toolbox.assign_fitness(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), cpu_time=time.time() - start, **record)
    if archive is not None:
        archive.update(population)
    if verbose:
        print logbook.stream

    for gen in range(1, ngen + 1):
        parents = toolbox.select(population, len(population) - num_randoms)
        offspring = breed(parents, toolbox, xover_prob, mut_prob)
        offspring += [toolbox.individual() for _ in range(num_randoms)]
        for ind in offspring:
            ind.error = toolbox.evaluate_error(ind)[0]

        population.extend(offspring)
        toolbox.assign_fitness(population)

        if calc_pareto_front:
            pareto_front_size = len(find_pareto_front(population))
            logging.debug("Generation: %5d - Pareto Front Size: %5d", gen, pareto_front_size)
            if pareto_front_size > target_popsize:
                logging.info("Pareto front size exceeds the size of population")
                break
        else:
            pareto_front_size = 0

        reduce_population(population, tournament_size, target_popsize, pareto_front_size)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), cpu_time=time.time() - start, **record)
        if archive is not None:
            archive.update(population)
        if verbose:
            print logbook.stream

        for ind in population:
            ind.age += 1

    return population, logbook
