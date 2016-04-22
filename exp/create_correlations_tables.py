import csv
import glob
from collections import defaultdict

import pandas as pd
from scipy import stats


def get_data_frame(alg_log_files, algorithm_names, xover, problem, colnames=None, add_generation=True):
    exp_dfs = []
    for files, alg_name in zip(alg_log_files, algorithm_names):
        print alg_name, len(files)
        csvs = []
        for i, log_file in enumerate(files):
            seed = int(log_file.split("_")[-1][:-4])
            c = pd.read_csv(log_file, names=colnames)
            if add_generation:
                c["gen"] = c.index
            c["run"] = i
            c["crossover"] = xover
            c["problem"] = problem
            c["seed"] = seed
            csvs.append(c)
        csvs_df = pd.concat(csvs)
        csvs_df["algorithm"] = alg_name
        exp_dfs.append(csvs_df)
    df = pd.concat(exp_dfs)
    return df


measure = "euclidean"
prefix = "distance_medians"
# measure = "mean"
# prefix = "angle"


problems = ["mod_quartic", "nonic", "keijzer4", "r1", "r2"]
generations = [0, 10]#, 25, 50, 100, 250, 500, 1000]
final_generation = 10

algs_standard = ["GP", "AFPO", "DFPO", "SNFPO", "ASNFPO"]
algs_lgx = [alg + "_LGX" for alg in algs_standard]
algorithms_all = [algs_standard, algs_lgx]
xovers = ["standard", "geometric"]

results = defaultdict(list)
for problem in problems:
    print "Processing problem {}".format(problem)

    for i, xover in enumerate(xovers):
        algorithms = algorithms_all[i]
        log_files = [glob.glob("{}_{}_*.log".format(algorithm, problem)) for algorithm in algorithms]
        sd_files = [glob.glob("{}_{}_{}_*.log".format(prefix, algorithm, problem)) for algorithm in algorithms]

        data_frame = get_data_frame(log_files, algorithms, xover, problem)
        sd_data_frame = get_data_frame(sd_files, algorithms, xover, problem, add_generation=False)
        final_fitnesses = data_frame.loc[data_frame["gen"] == final_generation, "min_fitness"]

        correlations = []
        for gen in generations:
            diversity = sd_data_frame.loc[sd_data_frame["gen"] == gen, measure]
            spearman_results = stats.spearmanr(diversity, final_fitnesses)
            correlations.append(spearman_results.correlation)
        results[xover].append(correlations)

rows_std = zip(*results["standard"])
rows_geometric = zip(*results["geometric"])

with open("correlations_{}_{}.csv".format(prefix, measure), 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(["generation"] + problems * 2)
    for gen, row_std, row_geometric in zip(generations, rows_std, rows_geometric):
        rounded_row_std = ["{0:+.3f}".format(a) for a in row_std]
        rounded_row_geometric = ["{0:+.3f}".format(a) for a in row_geometric]
        writer.writerow(["{:5d}".format(gen)] + rounded_row_std + rounded_row_geometric)
