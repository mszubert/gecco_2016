import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(color_codes=True, context="poster")
sns.set_style("whitegrid")


def create_plot(df, column, ax, title, ylim=None, ylabel="Error", ylog=True, xlog=True, palette=None, legend=True):
    sns.despine(left=True)
    if ylog:
        ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log")
    ax.tick_params(which="both", width=1, length=10)
    sns.tsplot(data=df[df["gen"] % 10 == 0], time="gen", value=column, unit="run", condition="algorithm", ax=ax,
               color=palette, legend=legend)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Generations")
    ax.set_title(title)


def get_data_frame(alg_log_files, algorithm_names, xover, problem, colnames=None):
    exp_dfs = []
    for files, alg_name in zip(alg_log_files, algorithm_names):
        print alg_name, len(files)
        csvs = []
        for i, log_file in enumerate(files):
            c = pd.read_csv(log_file, names=colnames)
            c["gen"] = c.index
            c["run"] = i
            c["crossover"] = xover
            c["problem"] = problem
            csvs.append(c)
        csvs_df = pd.concat(csvs)
        csvs_df["algorithm"] = alg_name
        exp_dfs.append(csvs_df)
    df = pd.concat(exp_dfs)
    return df

problems = ["mod_quartic", "nonic", "keijzer4", "r1", "r2"]
problem_patterns = ["{}_mod_quartic_*.log",
                    "{}_nonic_*.log",
                    "{}_keijzer4_*.log",
                    "{}_r1_*.log",
                    "{}_r2_*.log"]

ylims = [1.0, 1.0, 0.6, 0.8, 0.6]
algs_standard = ["GP", "AFPO", "DFPO", "SNFPO", "ASNFPO"]
algs_lgx = [alg + "_LGX" for alg in algs_standard]
algs = [algs_standard, algs_lgx]

alg_names_standard = ["GP", "AFPO", "DFPO", "ESNFPO", "ASNFPO"]
alg_names_lgx = [alg_name + "_LGX" for alg_name in alg_names_standard]
alg_names = [alg_names_standard, alg_names_lgx]

num_columns = 2
columns = ["min_fitness", "min_fitness"]
xovers = ["standard", "geometric"]

standard_palette = sns.color_palette("deep", 5)
lgx_palette = sns.color_palette("bright", 5)
palettes = [standard_palette, lgx_palette]

f, axes = plt.subplots(nrows=len(problems), ncols=num_columns, figsize=(num_columns * 5, len(problems) * 5),
                       sharex=True, sharey=False)

for i, ax in enumerate(axes.flat):
    problem = problems[i / num_columns]
    algorithms = algs[i % num_columns]
    algorithm_names = alg_names[i % num_columns]
    column = columns[i % num_columns]

    print "Plotting problem {}".format(problem)
    problem_pattern = problem_patterns[i / num_columns]
    log_files = [glob.glob(problem_pattern.format(algorithm)) for algorithm in algorithms]
    data_frame = get_data_frame(log_files, algorithm_names, xovers[i % num_columns], problem)
    create_plot(data_frame, column, ax, problem + ", " + xovers[i % num_columns], ylim=ylims[i / num_columns],
                ylog=False, xlog=False, palette=palettes[i % num_columns])

plt.tight_layout()
plt.savefig("final_results.pdf")
