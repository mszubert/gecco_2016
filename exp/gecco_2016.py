import logging
import operator
import sys

import cachetools
from deap import creator, gp, base, tools

from gp.algorithms import afpo, ea_simple_semantics, operators, initialization, archive, fast_evaluate
from gp.experiments import runner, benchmark_problems
from gp.experiments import symbreg, reports
from gp.semantic import distances, semantics, library, locally_geometric
from gp.semantic.semantics import SemanticPrimitiveTree

NGEN = 10
POP_SIZE = 256
TOURN_SIZE = 7
MIN_DEPTH_INIT = 2
MAX_DEPTH_INIT = 6
MAX_HEIGHT = 17
MAX_SIZE = 300
XOVER_PROB = 0.9
MUT_PROB = 0.0
INTERNAL_NODE_SELECTION_BIAS = 0.9
NOVELTY_NEIGHBORS = 15
LIBRARY_SEARCH_NEIGHBORS = 8
LIBRARY_DEPTH = 3
TAG_DEPTH = 2

WEIGHT_FITNESS = -1.0
WEIGHT_AGE_DENSITY = -1.0
WEIGHT_NOVELTY = 1.0
ERROR_FUNCTION = fast_evaluate.euclidean_error


def get_archive(response):
    fitness_archive = archive.FitnessDistributionArchive(100)
    population_archive = archive.PopulationSavingArchive(100)

    angular_stats_archive = archive.AngularStatisticsArchive(response)
    tree_counting_archive = archive.TreeCountingArchive()
    median_distance_archive = archive.SemanticMedianDistanceArchive(
        [distances.mean_canberra, distances.euclidean_distance, distances.cumulative_absolute_difference],
        ["canberra", "euclidean", "manhattan"])

    multi_archive = archive.MultiArchive(
        [tree_counting_archive, median_distance_archive, angular_stats_archive, population_archive, fitness_archive])
    return multi_archive


def get_gp_toolbox(predictors, response):
    creator.create("FitnessMax", base.Fitness, weights=(WEIGHT_FITNESS,))
    creator.create("Individual", SemanticPrimitiveTree, fitness=creator.FitnessMax, age=int)

    toolbox = base.Toolbox()
    pset = symbreg.get_numpy_pset(len(predictors[0]))
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH_INIT, max_=MAX_DEPTH_INIT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", initialization.syntactically_distinct, individual=toolbox.individual, retries=100)
    toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

    expression_dict = cachetools.LRUCache(maxsize=10000)
    toolbox.register("error_func", ERROR_FUNCTION, response=response)
    toolbox.register("evaluate_error", semantics.calc_eval_semantics, context=pset.context, predictors=predictors,
                     eval_semantics=toolbox.error_func, expression_dict=expression_dict)
    toolbox.register("assign_fitness", afpo.assign_pure_fitness)

    toolbox.register("koza_node_selector", operators.internally_biased_node_selector, bias=INTERNAL_NODE_SELECTION_BIAS)
    toolbox.register("mate", operators.one_point_xover_biased, node_selector=toolbox.koza_node_selector)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_SIZE))

    mstats = reports.configure_inf_protected_stats()
    multi_archive = get_archive(response)

    pop = toolbox.population(n=POP_SIZE)
    toolbox.register("run", ea_simple_semantics.ea_simple, population=pop, toolbox=toolbox, cxpb=XOVER_PROB,
                     mutpb=MUT_PROB, ngen=NGEN, elite_size=0, stats=mstats, verbose=False, archive=multi_archive)

    toolbox.register("save", reports.save_log_to_csv)
    toolbox.decorate("save", reports.save_archive(multi_archive))
    return toolbox


def get_po_toolbox(predictors, response):
    creator.create("FitnessAge", base.Fitness, weights=(WEIGHT_FITNESS, WEIGHT_AGE_DENSITY))
    creator.create("Individual", SemanticPrimitiveTree, fitness=creator.FitnessAge, age=int)

    toolbox = base.Toolbox()
    pset = symbreg.get_numpy_pset(len(predictors[0]))
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH_INIT, max_=MAX_DEPTH_INIT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", initialization.syntactically_distinct, individual=toolbox.individual, retries=100)
    toolbox.register("select", tools.selRandom)

    expression_dict = cachetools.LRUCache(maxsize=10000)
    toolbox.register("error_func", ERROR_FUNCTION, response=response)
    toolbox.register("evaluate_error", semantics.calc_eval_semantics, context=pset.context, predictors=predictors,
                     eval_semantics=toolbox.error_func, expression_dict=expression_dict)

    toolbox.register("koza_node_selector", operators.internally_biased_node_selector, bias=INTERNAL_NODE_SELECTION_BIAS)
    toolbox.register("mate", operators.one_point_xover_biased, node_selector=toolbox.koza_node_selector)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_SIZE))

    mstats = reports.configure_inf_protected_stats()
    multi_archive = get_archive(response)

    pop = toolbox.population(n=POP_SIZE)
    toolbox.register("run", afpo.pareto_optimization, population=pop, toolbox=toolbox, xover_prob=XOVER_PROB,
                     mut_prob=MUT_PROB, ngen=NGEN, tournament_size=TOURN_SIZE, num_randoms=1, stats=mstats,
                     archive=multi_archive, calc_pareto_front=False)

    toolbox.register("save", reports.save_log_to_csv)
    toolbox.decorate("save", reports.save_archive(multi_archive))
    return toolbox


def get_afpo_toolbox(predictors, response):
    toolbox = get_po_toolbox(predictors, response)
    toolbox.register("assign_fitness", afpo.assign_age_fitness)
    return toolbox


def get_dfpo_toolbox(predictors, response):
    toolbox = get_po_toolbox(predictors, response)
    toolbox.register("assign_fitness", afpo.assign_density_fitness, tag_depth=TAG_DEPTH)
    return toolbox


def get_novelty_po_toolbox(predictors, response):
    creator.create("FitnessNovelty", base.Fitness, weights=(WEIGHT_FITNESS, WEIGHT_NOVELTY))
    creator.create("Individual", SemanticPrimitiveTree, fitness=creator.FitnessNovelty, age=int)

    toolbox = base.Toolbox()
    pset = symbreg.get_numpy_pset(len(predictors[0]))
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH_INIT, max_=MAX_DEPTH_INIT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", initialization.syntactically_distinct, individual=toolbox.individual, retries=100)
    toolbox.register("select", tools.selRandom)

    expression_dict = cachetools.LRUCache(maxsize=10000)
    toolbox.register("error_func", ERROR_FUNCTION, response=response)
    toolbox.register("evaluate_error", semantics.calc_eval_semantics, context=pset.context, predictors=predictors,
                     eval_semantics=toolbox.error_func, expression_dict=expression_dict)

    toolbox.register("koza_node_selector", operators.internally_biased_node_selector, bias=INTERNAL_NODE_SELECTION_BIAS)
    toolbox.register("mate", operators.one_point_xover_biased, node_selector=toolbox.koza_node_selector)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_SIZE))

    mstats = reports.configure_inf_protected_stats()
    multi_archive = get_archive(response)

    pop = toolbox.population(n=POP_SIZE)
    toolbox.register("run", afpo.pareto_optimization, population=pop, toolbox=toolbox, xover_prob=XOVER_PROB,
                     mut_prob=MUT_PROB, ngen=NGEN, tournament_size=TOURN_SIZE, num_randoms=1, stats=mstats,
                     archive=multi_archive, calc_pareto_front=False)

    toolbox.register("save", reports.save_log_to_csv)
    toolbox.decorate("save", reports.save_archive(multi_archive))
    return toolbox


def get_snfpo_toolbox(predictors, response):
    toolbox = get_novelty_po_toolbox(predictors, response)
    toolbox.register("assign_fitness", afpo.assign_semantic_novelty_fitness, k=NOVELTY_NEIGHBORS,
                     distance_measure=distances.mean_canberra)
    return toolbox


def get_angle_snfpo_toolbox(predictors, response):
    toolbox = get_novelty_po_toolbox(predictors, response)
    toolbox.register("assign_fitness", afpo.assign_semantic_novelty_angle_fitness, k=NOVELTY_NEIGHBORS, target=response)
    return toolbox


def get_gp_lgx_toolbox(predictors, response):
    creator.create("FitnessMax", base.Fitness, weights=(WEIGHT_FITNESS,))
    creator.create("Individual", SemanticPrimitiveTree, fitness=creator.FitnessMax, age=int)

    toolbox = base.Toolbox()
    pset = symbreg.get_numpy_pset(len(predictors[0]))
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH_INIT, max_=MAX_DEPTH_INIT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", initialization.syntactically_distinct, individual=toolbox.individual, retries=100)
    toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

    lib = library.SemanticLibrary()
    lib.generate_trees(pset, LIBRARY_DEPTH, predictors)
    toolbox.register("lib_selector", lib.get_closest, distance_measure=distances.cumulative_absolute_difference,
                     k=LIBRARY_SEARCH_NEIGHBORS, check_constant=True)
    toolbox.register("mate", locally_geometric.homologous_lgx, library_selector=toolbox.lib_selector,
                     internal_bias=INTERNAL_NODE_SELECTION_BIAS, max_height=MAX_HEIGHT,
                     distance_measure=distances.cumulative_absolute_difference, distance_threshold=0.0)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_SIZE))

    expression_dict = cachetools.LRUCache(maxsize=10000)

    toolbox.register("error_func", ERROR_FUNCTION, response=response)
    toolbox.register("evaluate_error", semantics.calc_eval_semantics, context=pset.context, predictors=predictors,
                     eval_semantics=toolbox.error_func, expression_dict=expression_dict)
    toolbox.register("assign_fitness", afpo.assign_pure_fitness)

    mstats = reports.configure_inf_protected_stats()
    multi_archive = get_archive(response)

    pop = toolbox.population(n=POP_SIZE)
    toolbox.register("run", ea_simple_semantics.ea_simple, population=pop, toolbox=toolbox, cxpb=XOVER_PROB,
                     mutpb=MUT_PROB, ngen=NGEN, elite_size=0, stats=mstats, verbose=False, archive=multi_archive)

    toolbox.register("save", reports.save_log_to_csv)
    toolbox.decorate("save", reports.save_archive(multi_archive))
    return toolbox


def get_po_lgx_toolbox(predictors, response):
    creator.create("FitnessAge", base.Fitness, weights=(WEIGHT_FITNESS, WEIGHT_AGE_DENSITY))
    creator.create("Individual", SemanticPrimitiveTree, fitness=creator.FitnessAge, age=int)

    toolbox = base.Toolbox()
    pset = symbreg.get_numpy_pset(len(predictors[0]))
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH_INIT, max_=MAX_DEPTH_INIT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", initialization.syntactically_distinct, individual=toolbox.individual, retries=100)
    toolbox.register("select", tools.selRandom)

    lib = library.SemanticLibrary()
    lib.generate_trees(pset, LIBRARY_DEPTH, predictors)
    toolbox.register("lib_selector", lib.get_closest, distance_measure=distances.cumulative_absolute_difference,
                     k=LIBRARY_SEARCH_NEIGHBORS, check_constant=True)
    toolbox.register("mate", locally_geometric.homologous_lgx, library_selector=toolbox.lib_selector,
                     internal_bias=INTERNAL_NODE_SELECTION_BIAS, max_height=MAX_HEIGHT,
                     distance_measure=distances.cumulative_absolute_difference, distance_threshold=0.0)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_SIZE))

    expression_dict = cachetools.LRUCache(maxsize=10000)
    toolbox.register("error_func", ERROR_FUNCTION, response=response)
    toolbox.register("evaluate_error", semantics.calc_eval_semantics, context=pset.context, predictors=predictors,
                     eval_semantics=toolbox.error_func, expression_dict=expression_dict)

    mstats = reports.configure_inf_protected_stats()
    multi_archive = get_archive(response)

    pop = toolbox.population(n=POP_SIZE)
    toolbox.register("run", afpo.pareto_optimization, population=pop, toolbox=toolbox, xover_prob=XOVER_PROB,
                     mut_prob=MUT_PROB, ngen=NGEN, tournament_size=TOURN_SIZE, num_randoms=1, stats=mstats,
                     archive=multi_archive, calc_pareto_front=False)

    toolbox.register("save", reports.save_log_to_csv)
    toolbox.decorate("save", reports.save_archive(multi_archive))
    return toolbox


def get_afpo_lgx_toolbox(predictors, response):
    toolbox = get_po_lgx_toolbox(predictors, response)
    toolbox.register("assign_fitness", afpo.assign_age_fitness)
    return toolbox


def get_dfpo_lgx_toolbox(predictors, response):
    toolbox = get_po_lgx_toolbox(predictors, response)
    toolbox.register("assign_fitness", afpo.assign_density_fitness, tag_depth=TAG_DEPTH)
    return toolbox


def get_novelty_po_lgx_toolbox(predictors, response):
    creator.create("FitnessNovelty", base.Fitness, weights=(WEIGHT_FITNESS, WEIGHT_NOVELTY))
    creator.create("Individual", SemanticPrimitiveTree, fitness=creator.FitnessNovelty, age=int)

    toolbox = base.Toolbox()
    pset = symbreg.get_numpy_pset(len(predictors[0]))
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH_INIT, max_=MAX_DEPTH_INIT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", initialization.syntactically_distinct, individual=toolbox.individual, retries=100)
    toolbox.register("select", tools.selRandom)

    lib = library.SemanticLibrary()
    lib.generate_trees(pset, LIBRARY_DEPTH, predictors)
    toolbox.register("lib_selector", lib.get_closest, distance_measure=distances.cumulative_absolute_difference,
                     k=LIBRARY_SEARCH_NEIGHBORS, check_constant=True)
    toolbox.register("mate", locally_geometric.homologous_lgx, library_selector=toolbox.lib_selector,
                     internal_bias=INTERNAL_NODE_SELECTION_BIAS, max_height=MAX_HEIGHT,
                     distance_measure=distances.cumulative_absolute_difference, distance_threshold=0.0)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_SIZE))

    expression_dict = cachetools.LRUCache(maxsize=10000)
    toolbox.register("error_func", ERROR_FUNCTION, response=response)
    toolbox.register("evaluate_error", semantics.calc_eval_semantics, context=pset.context, predictors=predictors,
                     eval_semantics=toolbox.error_func, expression_dict=expression_dict)

    mstats = reports.configure_inf_protected_stats()
    multi_archive = get_archive(response)

    pop = toolbox.population(n=POP_SIZE)
    toolbox.register("run", afpo.pareto_optimization, population=pop, toolbox=toolbox, xover_prob=XOVER_PROB,
                     mut_prob=MUT_PROB, ngen=NGEN, tournament_size=TOURN_SIZE, num_randoms=1, stats=mstats,
                     archive=multi_archive, calc_pareto_front=False)

    toolbox.register("save", reports.save_log_to_csv)
    toolbox.decorate("save", reports.save_archive(multi_archive))
    return toolbox


def get_snfpo_lgx_toolbox(predictors, response):
    toolbox = get_novelty_po_lgx_toolbox(predictors, response)
    toolbox.register("assign_fitness", afpo.assign_semantic_novelty_fitness, k=NOVELTY_NEIGHBORS,
                     distance_measure=distances.mean_canberra)
    return toolbox


def get_angle_snfpo_lgx_toolbox(predictors, response):
    toolbox = get_novelty_po_lgx_toolbox(predictors, response)
    toolbox.register("assign_fitness", afpo.assign_semantic_novelty_angle_fitness, k=NOVELTY_NEIGHBORS, target=response)
    return toolbox


random_seed = int(sys.argv[1])
runner.run_benchmarks(random_seed,
                      [benchmark_problems.mod_quartic, benchmark_problems.nonic, benchmark_problems.keijzer4,
                       benchmark_problems.r1, benchmark_problems.r2],
                      [get_gp_toolbox, get_gp_lgx_toolbox,
                       get_afpo_toolbox, get_afpo_lgx_toolbox,
                       get_dfpo_toolbox, get_dfpo_lgx_toolbox,
                       get_snfpo_toolbox, get_snfpo_lgx_toolbox,
                       get_angle_snfpo_toolbox, get_angle_snfpo_lgx_toolbox],
                      ["GP", "GP_LGX",
                       "AFPO", "AFPO_LGX",
                       "DFPO", "DFPO_LGX",
                       "SNFPO", "SNFPO_LGX",
                       "ASNFPO", "ASNFPO_LGX"],
                      logging_level=logging.INFO, sample_evenly=True)
