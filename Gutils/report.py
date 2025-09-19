from sys import stdout

import numpy as np


def pretty_print_with_baselines(metrics, verbose=True, file=stdout):
    metric_col = list(list(metrics.values())[0].keys())
    baseline_keys = list(metrics.keys())
    line_width = (35 + len(metric_col) * 17)
    print('=' * line_width, file=file)

    print((' ' * 35) + (str('{:15s}' * len(metric_col))).format(*metric_col), file=file)
    print('_' * line_width, file=file)
    for j, ba in enumerate(baseline_keys):
        tmp = '{:35s}' + ('{:15s}' * len(metric_col))
        msg = ['.'.join(ba.split('.')[-2:])]
        for col in metric_col:
            val = f'{np.mean(metrics[ba][col]):.5f}'

            msg.append(val)
        print(tmp.format(*msg), file=file)
        if j != len(baseline_keys) - 1:
            print('_' * line_width, file=file)
    print('=' * line_width, file=file)
    if verbose:
        print(metrics)
