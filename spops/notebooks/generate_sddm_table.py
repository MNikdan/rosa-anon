import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sys

raw_data_path = sys.argv[1]
layer_meta_path = sys.argv[2]


def sddm_bench():
    title = 'SDDMM Benchmark'
    duration_name = 'Duration (ms)'
    method_name = 'Method'
    layer_name = 'Layer Name'

    # Assuming you have already loaded your data
    perf_table = pd.read_csv(raw_data_path, sep=';')
    layer_table = pd.read_csv(layer_meta_path, sep=';')
    perf_table = perf_table.round(2)
    Ms = layer_table['rows'].to_list()
    Ns = layer_table['cols'].to_list()
    perf_table['layer_name'] = perf_table['layer_name'].str.replace('model.model.', 'model.')
    perf_table['layer_name'] = perf_table['layer_name'].str.replace('_proj', '')
    perf_table = perf_table.rename(columns={'median': duration_name, 'method': method_name, 'layer_name': layer_name})

    sns.set_style('darkgrid', {'legend.frameon': True})

    num_axis = 5

    layer_names = perf_table[layer_name][perf_table[method_name] == 'Ours'].to_list()

    for i in range(len(layer_names)):
        layer_names[i] = f'M = {Ms[i]} N = {Ns[i]} K = 512 {layer_names[i]}'

    items_per_axis = 450 // num_axis
    labels_per_axis = items_per_axis // 2
    fig, axs = plt.subplots(nrows=num_axis, ncols=1, tight_layout=True, figsize=(60, 50))
    fig.suptitle(title, y=1.0001, fontsize=50)
    plt.autoscale(enable=True, axis='y', tight=True)  # tight layout

    handles = []  # Collect legend handles
    x = np.arange(labels_per_axis)

    width = 0.3
    for i in range(num_axis):
        subplot = perf_table[(i * items_per_axis):((i + 1) * items_per_axis)]

        axis_names = layer_names[(i * labels_per_axis):((i + 1) * labels_per_axis)]

        for j, method in zip([0, 1], ['Sputnik', 'Ours']):
            bars = axs[i].bar(x + j * width, subplot[subplot[method_name] == method][duration_name], width, label=method)
            handles.extend(bars)
            axs[i].set_ylabel(duration_name, fontdict={'fontsize': 20})
            # Add labels to the bars
            axs[i].bar_label(bars, labels=subplot[subplot[method_name] == method][duration_name], padding=3)

        axs[i].set_xticks(x + width / 2)
        axs[i].set_xticklabels(axis_names, rotation=30, ha='right', fontsize=20)

    axs[0].legend(title='', fontsize=30, loc='upper right')

    plt.subplots_adjust(top=0.85)  # Add space at the top
    plt.xlabel('')  # Removes X-label
    plt.savefig('sddmm.pdf')
    plt.show()

def mask_meta():
    title = 'SDDMM Benchmark'
    duration_name = 'Duration (ms)'
    method_name = 'Method'
    layer_name = 'Layer Name'

    # Assuming you have already loaded your data
    layer_table = pd.read_csv(layer_meta_path, sep=';')
    Ms = layer_table['rows'].to_list()
    Ns = layer_table['cols'].to_list()

    layer_table['layer_name'] = layer_table['layer_name'].str.replace('model.model.', 'model.').str.replace('_proj', '')
    layer_table = layer_table.rename(columns={'layer_name': 'Layer Name', 'rows': 'Rows', 'cols': 'Cols', 'empty_rows_perc': 'Empty Rows (%)', 'empty_cols_perc': 'Empty Columns (%)'})
    for i in range(len(Ms)):
        layer_names[i] = f'M = {Ms[i]} N = {Ns[i]} K = 512 {layer_names[i]}'


if __name__ == '__main__':
    mask_meta()
