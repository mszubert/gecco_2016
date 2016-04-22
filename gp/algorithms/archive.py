import csv
from collections import defaultdict
from copy import deepcopy

import numpy

from gp.semantic import semantics, distances


class FitnessDistributionArchive(object):
    def __init__(self, frequency):
        self.fitness = []
        self.generations = []
        self.frequency = frequency
        self.generation_counter = 0

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            fitnesses = [ind.fitness.values for ind in population]
            self.fitness.append(fitnesses)
            self.generations.append(self.generation_counter)
        self.generation_counter += 1

    def save(self, log_file):
        fitness_distribution_file = "fitness_" + log_file
        with open(fitness_distribution_file, 'wb') as f:
            writer = csv.writer(f)
            for gen, ages in zip(self.generations, self.fitness):
                writer.writerow([gen, ages])


class MultiArchive(object):
    def __init__(self, archives):
        self.archives = archives

    def update(self, population):
        for archive in self.archives:
            archive.update(population)

    def save(self, log_file):
        for archive in self.archives:
            archive.save(log_file)


class PopulationSavingArchive(object):
    def __init__(self, frequency, simplifier=None):
        self.inds = []
        self.generations = []
        self.frequency = frequency
        self.generation_counter = 0
        self.simplifier = simplifier

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            pop_copy = [deepcopy(ind) for ind in population]
            if self.simplifier is not None:
                self.simplifier(pop_copy)
            self.inds.append(pop_copy)
            self.generations.append(self.generation_counter)
        self.generation_counter += 1

    def save(self, log_file):
        inds_file = "inds_" + log_file
        with open(inds_file, 'wb') as f:
            writer = csv.writer(f)
            for gen, inds in zip(self.generations, self.inds):
                tuples = [(ind.fitness.values, str(ind)) for ind in inds]
                writer.writerow([gen, len(inds)] + tuples)


class TreeCountingArchive(object):
    def __init__(self, file_prefix="tree_num_", ind_selector=None):
        self.ind_selector = ind_selector
        self.num_distinct_trees = []
        self.num_sem_distinct_trees = []
        self.num_sem_distinct_trees_threshold = []
        self.num_infinite_trees = []
        self.num_constant_trees = []
        self.num_novel_trees = []
        self.syntactic_entropy = []
        self.semantic_entropy = []
        self.fitness_entropy = []
        self.gini_index = []
        self.file_prefix = file_prefix

    def update(self, population):
        num_infinite_trees = 0
        num_constant_trees = 0

        fitness_count_dict = defaultdict(int)
        tree_expression_dict = defaultdict(int)
        tree_semantics_set = set()
        tree_semantics_list = list()

        inds = population if self.ind_selector is None else self.ind_selector(population)
        for ind in inds:
            tree_semantics = ind.semantics[0]
            tree_expression_dict[tree_semantics.expr] += 1
            tree_semantics_set.add(tree_semantics.tostring())
            tree_semantics_list.append(tree_semantics)
            fitness_count_dict[ind.error] += 1

            if not numpy.isfinite(numpy.sum(tree_semantics)):
                num_infinite_trees += 1
            elif numpy.ptp(tree_semantics) <= 10e-8:
                num_constant_trees += 1

        unique_semantics, semantic_groups = semantics.get_unique_semantics(tree_semantics_list, 10e-6,
                                                                           distances.cumulative_absolute_difference)
        self.semantic_entropy.append(calculate_entropy(semantic_groups))
        self.syntactic_entropy.append(calculate_entropy(tree_expression_dict.values()))
        self.fitness_entropy.append(calculate_entropy(fitness_count_dict.values()))
        self.gini_index.append(calculate_gini_index([ind.error for ind in population]))

        self.num_infinite_trees.append(num_infinite_trees)
        self.num_constant_trees.append(num_constant_trees)
        self.num_distinct_trees.append(len(tree_expression_dict))
        self.num_sem_distinct_trees.append(len(tree_semantics_set))
        self.num_sem_distinct_trees_threshold.append(len(unique_semantics))

    def save(self, log_file):
        semantic_statistics_file = self.file_prefix + log_file
        with open(semantic_statistics_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(
                ["gen",
                 "num_infinite_trees",
                 "num_constant_trees",
                 "num_distinct_trees",
                 "num_sem_distinct_trees",
                 "num_sem_distinct_trees_threshold",
                 "syntactic_entropy",
                 "semantic_entropy",
                 "fitness_entropy",
                 "gini_index"])
            for gen in range(len(self.num_infinite_trees)):
                writer.writerow(
                    [gen,
                     self.num_infinite_trees[gen],
                     self.num_constant_trees[gen],
                     self.num_distinct_trees[gen],
                     self.num_sem_distinct_trees[gen],
                     self.num_sem_distinct_trees_threshold[gen],
                     self.syntactic_entropy[gen],
                     self.semantic_entropy[gen],
                     self.fitness_entropy[gen],
                     self.gini_index[gen]])


def calculate_entropy(values):
    n = float(sum(values))
    if n == 0:
        return 0

    entropy = 0
    for v in values:
        entropy -= v * numpy.log2(v)
    return entropy / n + numpy.log2(n)


def calculate_gini_index(values):
    md = 0
    n = len(values)
    for i in range(n):
        for j in range(n):
            md += abs(values[i] - values[j])
    md /= n * n
    rmd = md / numpy.mean(values)
    gini = rmd / 2.0
    return gini


def mean_absolute_error_dist(vector1, vector2):
    return numpy.sum(numpy.abs(vector1 - vector2)) / len(vector1)


def get_percentiles(dists):
    sorted_distances = numpy.sort(dists)
    percentile_index = len(dists) / 100.0
    return [sorted_distances[int(percentile_index * q)] for q in range(0, 100)]


class SemanticMedianDistanceArchive(object):
    def __init__(self, distance_measures, distance_names, file_prefix="distance_medians_", ind_selector=None):
        self.ind_selector = ind_selector
        self.medians = defaultdict(list)
        self.distance_names = distance_names
        self.distance_measures = distance_measures
        self.file_prefix = file_prefix

    def update(self, population):
        inds = population if self.ind_selector is None else self.ind_selector(population)
        root_semantics = [ind.semantics[0] for ind in inds if numpy.isfinite(numpy.sum(ind.semantics[0]))]
        for distance_measure, distance_name in zip(self.distance_measures, self.distance_names):
            dists = semantics.calculate_pairwise_distances(root_semantics, distance_measure)
            self.medians[distance_name].append(numpy.nanmedian(dists))

    def save(self, log_file):
        semantic_statistics_file = self.file_prefix + log_file
        with open(semantic_statistics_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["gen"] + self.distance_names)
            values = [self.medians[name] for name in self.distance_names]
            for gen, medians in enumerate(zip(*values)):
                writer.writerow([gen] + list(medians))


class SemanticStatisticsArchive(object):
    def __init__(self, file_prefix):
        self.means = []
        self.medians = []
        self.first_quartiles = []
        self.third_quartiles = []
        self.file_prefix = file_prefix

    def update(self, population):
        root_semantics = [ind.semantics[0] for ind in population if numpy.isfinite(numpy.sum(ind.semantics[0]))]
        values = self.get_values(root_semantics)
        self.means.append(numpy.nanmean(values))
        self.medians.append(numpy.nanmedian(values))
        self.first_quartiles.append(numpy.nanpercentile(values, 25))
        self.third_quartiles.append(numpy.nanpercentile(values, 75))

    def get_values(self, root_semantics):
        raise NotImplementedError

    def save(self, log_file):
        single_statistics_file = self.file_prefix + log_file
        with open(single_statistics_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["gen", "mean", "median", "first_quartile", "third_quartile"])
            for gen in range(len(self.means)):
                writer.writerow([gen, self.means[gen], self.medians[gen],
                                 self.first_quartiles[gen], self.third_quartiles[gen]])


class AngularStatisticsArchive(SemanticStatisticsArchive):
    def __init__(self, target, file_prefix="angle_"):
        SemanticStatisticsArchive.__init__(self, file_prefix)
        self.target = target

    def get_values(self, root_semantics):
        return semantics.calculate_pairwise_angles(root_semantics, self.target)