import json
import os
import sys

import pandas as pd

results_dir = sys.argv[1]

results_list, hps = [], []
max_num_iters = 10
incomplete_path = os.path.join(results_dir, 'distribution_results_{}_{}.json') 
for n in range(1, max_num_iters+1):
    for s in range(n+1):
        a = n - s 
        complete_path = incomplete_path.format(s, a)
        if not os.path.exists(complete_path): continue
        with open(complete_path) as f:
            benchmark_output = json.load(f)
        hps.append('{}_{}'.format(s, a))
        results_dict = {}
        for result in benchmark_output['results']:
            results_dict[result['benchmark_name']] = result['score']
        results_list.append(results_dict)

df = pd.DataFrame(results_list, index=hps)
print(df)
best_results = pd.concat([pd.DataFrame(df.max()), pd.DataFrame(df.idxmax())], axis=1)
best_results.columns = ['Max Value', 'Configuration']
print(best_results)
