import sys
from typing import List

import pandas as pd
import os
import json
from autorank import autorank, latex_table, create_report
from scipy.stats import wilcoxon, ranksums, mannwhitneyu
import math
import numpy as np
from evaluate import get_types, get_comments
from cliffs_delta import cliffs_delta


model = sys.argv[1]  # codegpt, incoder, unixcoder
single_token = len(sys.argv) > 2 and sys.argv[2] == "single"

model_predictions_folder = os.path.join("./data/predictions", model)
metrics_file_name = "metrics.txt"

print("Analyzing", model_predictions_folder)
print("Single token:", single_token)

metrics = [
    'exact_match',
    'levenshtein',
    'bleu',
    'rouge',
    'meteor'
]

# 5 * 12 = 60

if single_token:
    metrics = ['exact_match', 'levenshtein']
    metrics_file_name = "metrics_single_token.txt"

comparisons = [
    # no types vs normal types (no comments)
    ['untyped-none', 'normal-none'],
    # no types vs normal types (all comments)
    ['untyped-all', 'normal-all'],

    # normal types vs explicit types (no comments)
    ['normal-none', 'explicit-none'],
    # normal types vs explicit types (all comments)
    ['normal-all', 'explicit-all'],

    # no comments vs all comments (no types)
    ['untyped-none', 'untyped-all'],
    # no comments vs single line comments (no types)
    ['untyped-none', 'untyped-single_line'],
    # no comments vs multi line comments (no types)
    ['untyped-none', 'untyped-multi_line'],
    # no comments vs docblock comments (no types)
    ['untyped-none', 'untyped-docblock'],

    # no comments vs all comments (normal types)
    ['normal-none', 'normal-all'],
    # no comments vs single line comments (normal types)
    ['normal-none', 'normal-single_line'],
    # no comments vs multi line comments (normal types)
    ['normal-none', 'normal-multi_line'],
    # no comments vs docblock comments (normal types)
    ['normal-none', 'normal-docblock'],
]

def get_metric_abbrev(metric: str):
    if metric == 'exact_match':
        return 'EM'
    if metric == 'levenshtein':
        return 'ES'
    if metric == 'bleu':
        return 'B4'
    if metric == 'rouge':
        return 'RL'
    if metric == 'meteor':
        return 'MT'
    raise Exception("Unknown Metric")

# TODO: Comments as well
if __name__ == '__main__':
    table_start = """
\\begin{table}[tb]
\\caption{Significant $p$-values (Test)}
\\centering
\\begin{tabular}{lllllllrr}
\\toprule
Model & Types 1 & CMT 1 & Types 2 & CMT 2 & Metric & $p$ & $\\delta$ \\\\
\\midrule
""".lstrip()

    table_end = """
\\bottomrule
\\end{tabular}
\\end{table}
""".lstrip()

    rows = []

    for populations in comparisons:
        row = ""
        data = []
        all_exist = True
        for population in populations:
            if not os.path.exists(os.path.join(model_predictions_folder, population)):
                all_exist = False
                break
        if not all_exist:
            print(f"Bad comparison: {populations}")
            continue
        for population in populations:
            cur_data = []
            data.append(cur_data)
            # one sample on each line of metrics.txt. line = { "empty": boolean, bleu: x, exact_match, y } etc
            with open(os.path.join(model_predictions_folder, population, metrics_file_name)) as f:
                for line in f:
                    obj = json.loads(line)
                    for key, value in obj.items():
                        # multiply by 100 to make it [0, 100] scale if numeric
                        if type(obj[key]) == float:
                            obj[key] = value * 100
                    cur_data.append(obj)

        # only take the data that every population is non empty for
        data_zipped = [d for d in zip(*data) if all(map(lambda s: not s["empty"], d))]

        pop_names = list(map(lambda pop: pop, populations))
        name_pop1 = populations[0]
        name_pop2 = populations[1]
        # Model & Types 1 & CMT 1 & Types 2 & CMT 2 & Metric & $p$ & $\\delta$ \\\\
        row_start = f"{model} & {get_types(name_pop1)} & {get_comments(name_pop1)} & {get_types(name_pop2)} & {get_comments(name_pop2)}"

        local_rows = ""
        significant = False
        for metric in metrics:
            data_pop1 = list(map(lambda d: d[0][metric], data_zipped))
            data_pop2 = list(map(lambda d: d[1][metric], data_zipped))

            result = wilcoxon(data_pop1, data_pop2)
            cd = cliffs_delta(data_pop1, data_pop2)[0]
            if result.pvalue <= 0.05:
                significant = True
                print(f"{metric}: {' vs '.join(pop_names)}: "
                      f"p = {round(result.pvalue, 3):.3f} {'!' if result.pvalue <= 0.05 else ''} ")
                local_rows += f"{row_start} & {get_metric_abbrev(metric)} & \\numprint{{{round(result.pvalue, 3):.3f}}} & \\numprint{{{round(cd, 3):.3f}}} \\\\\n"

        if significant:
            rows.append(local_rows)

    rows = "\\midrule\n".join(rows)

    print(table_start + rows + table_end)

