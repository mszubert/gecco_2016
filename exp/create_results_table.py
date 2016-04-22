import csv
import glob
import random
from collections import defaultdict
from functools import partial

import numpy
from deap import gp

from gp.algorithms import fast_evaluate
from gp.experiments import symbreg, benchmark_problems

numpy.random.seed(123)
random.seed(123)

algorithms = ["GP", "AFPO", "DFPO", "SNFPO", "ASNFPO", "GP_LGX", "AFPO_LGX", "DFPO_LGX", "SNFPO_LGX", "ASNFPO_LGX"]
problems = ["mod_quartic", "nonic", "keijzer4", "r1", "r2"]

pset = symbreg.get_numpy_pset(1)

AGGREGATE_FUNCTION = numpy.median
ERROR_MEASURE = fast_evaluate.root_mean_square_error
NUM_TEST_POINTS = 1000
GENERATION_LINE = 10

problem_training_errors = defaultdict(list)
problem_test_errors = defaultdict(list)
problem_model_sizes = defaultdict(list)

for problem in problems:
    print "Processing problem {}".format(problem)
    function = getattr(benchmark_problems, problem)
    p, r = benchmark_problems.get_training_set(function, sample_evenly=True)
    training_error_function = partial(ERROR_MEASURE, response=r)
    training_error = partial(fast_evaluate.fast_numpy_evaluate, context=pset.context, predictors=p,
                             error_function=training_error_function)

    test_p, test_r = benchmark_problems.get_training_set(function, num_points=NUM_TEST_POINTS, sample_evenly=False)
    test_error_function = partial(ERROR_MEASURE, response=test_r)
    test_error = partial(fast_evaluate.fast_numpy_evaluate, context=pset.context, predictors=test_p,
                         error_function=test_error_function)

    for algorithm in algorithms:
        print "Processing algorithm {}".format(algorithm)
        algorithm_training_errors = []
        algorithm_test_errors = []
        algorithm_model_sizes = []

        pattern = "inds_{}_{}_*.log".format(algorithm, problem)
        log_files = glob.glob(pattern)
        for log_file in log_files:
            with open(log_file) as f:
                lines = f.readlines()
                eval_line = eval(lines[GENERATION_LINE].replace("inf", "10e7"))
                best_error = numpy.inf
                best_ind_string = None
                for i in range(2, len(eval_line)):
                    ind_tuple = eval(eval_line[i])
                    error = float(ind_tuple[0][0])
                    if error < best_error:
                        best_error = error
                        best_ind_string = ind_tuple[1]

                best_ind = gp.PrimitiveTree.from_string(best_ind_string, pset)
                algorithm_training_errors.append(training_error(best_ind))
                algorithm_test_errors.append(test_error(best_ind))
                algorithm_model_sizes.append(len(best_ind))

        problem_training_errors[problem].append(AGGREGATE_FUNCTION(algorithm_training_errors))
        problem_test_errors[problem].append(AGGREGATE_FUNCTION(algorithm_test_errors))
        problem_model_sizes[problem].append(AGGREGATE_FUNCTION(algorithm_model_sizes))

values = []
for problem in problems:
    values.append(problem_training_errors[problem])
    values.append(problem_test_errors[problem])
    values.append(problem_model_sizes[problem])

rows = zip(*values)
with open("test_errors.csv", 'wb') as f:
    writer = csv.writer(f)
    header = [p + "_" + measure for p in problems for measure in ["train", "test", "size"]]
    writer.writerow(["algorithm"] + header)
    for algorithm, row in zip(algorithms, rows):
        rounded_row = ["{0:.3f}".format(a) if a < 10 else int(a) for a in row]
        writer.writerow([algorithm] + rounded_row)
