import json
import os
import pprint
import re
import sys

import numpy as np

from src.evaluation.nbrhood_stats_aggregator import aggregate_stats


def aggregate_ensemble(results_dir):
    all_percentages = {}
    for fname in os.listdir(results_dir):
        if not re.search(r'results_\d{1,2}\.json', fname): continue
        with open(os.path.join(results_dir, fname)) as f:
            results = json.load(f)
        stats, percentages = aggregate_stats(results)
        for d, d_percentage in percentages.items():
            if d not in all_percentages:
                all_percentages[d] = [d_percentage]
            else:
                all_percentages[d].append(d_percentage)

    ensemble_aggregate = {d:[np.mean(d_percentages), np.std(d_percentages)]
                          for d, d_percentages in all_percentages.items()}
    return ensemble_aggregate

def main():
    results_dir = sys.argv[1]
    ensemble_aggregate = aggregate_ensemble(results_dir)
    to_print = {d: '{} \u00B1 {}'.format(int(np.round(ensemble_aggregate[d][0])),
                                         int(np.round(ensemble_aggregate[d][1])))
                for d in sorted(ensemble_aggregate.keys())}
    pp = pprint.PrettyPrinter()
    pp.pprint(to_print)

if __name__ == '__main__':
    main()