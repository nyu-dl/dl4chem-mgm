import json
import pprint
import sys

import numpy as np


def main():
    json_path = sys.argv[1]
    results = json.load(open(json_path))
    stats, percentages = aggregate_stats(results)
    pp = pprint.PrettyPrinter()
    pp.pprint(stats)
    pp.pprint(percentages)

def dct_string_keys_to_int(dct):
    new_dct = {}
    for old_key in dct.keys():
        new_key = int(old_key)
        new_dct[new_key] = dct[old_key]
    return new_dct

def aggregate_stats(results):
    stats = {}
    results = [dct_string_keys_to_int(dct) for dct in results]
    for dct in results:
        for d, preds in dct.items():
            if d == 0: continue
            if d not in stats:
                stats[d] = {'equal': 0, 'total': 0}
            stats[d]['equal'] += (np.array(preds) == dct[0][0]).sum()
            stats[d]['total'] += len(preds)
    percentages = {d: int(np.round(aggregation['equal'] / aggregation['total'] * 100))
                   for d, aggregation in stats.items()}
    return stats, percentages

if __name__ == '__main__':
    main()