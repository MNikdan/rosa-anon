import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import sys
import threading


def count_empty_rows(tensor):
    empty_rows_mask = torch.all(tensor == 0, dim=1)
    num_empty_rows = torch.sum(empty_rows_mask).item()
    return num_empty_rows

def count_empty_columns(tensor):
    empty_cols_mask = torch.all(tensor == 0, dim=0)
    num_empty_cols = torch.sum(empty_cols_mask).item()
    return num_empty_cols

kernel_size = 4

path = sys.argv[1]
result = sys.argv[2]

image_id = 0
masks_to_plot = []
Round = lambda x, n: eval('"%.'+str(int(n))+'f" % '+repr(int(x)+round(float('.'+str(float(x)).split('.')[1]),n)))

raw_data = [['Model Name', 'Maximum Empty Rows (%)', 'Maximum Empty Column (%)', 'Mean Maximum Empty Row or Column']]
model_names = []


for model_id, model_name in enumerate(os.listdir(path)):
    print(f'Processing {model_id}')
    model = os.path.join(path, model_name)
    t = torch.load(model)

    max_empty_pct = 0
    max_empty_rows = 0
    max_empty_cols = 0
    num_masks = 0
    # Skip the embeddings
    for i, mask_name in enumerate(list(t.keys())[1:]):
        mask = t[mask_name].int()

        mask_name = mask_name.replace('model.model.layers.', 'Layer:')
        mask_name = mask_name.replace('. attn', ' attn')
        mask_name = mask_name.replace('.self', ', self')

        rows, cols = mask.shape

        num_empty_rows = count_empty_rows(mask)
        num_empty_cols = count_empty_columns(mask)

        empty_rows_pct = (num_empty_rows / rows) * 100
        empty_cols_pct = (num_empty_cols / cols) * 100

        max_empty_rows = max(max_empty_rows, empty_rows_pct)
        max_empty_cols = max(max_empty_cols, empty_cols_pct)

        max_empty_pct += max(empty_rows_pct, empty_cols_pct)
        num_masks += 1

    max_empty_pct /= num_masks
    raw_data.append([model_name, max_empty_rows, max_empty_cols, max_empty_pct])

pd.DataFrame(raw_data).to_csv(os.path.join(result, 'masks.csv'), sep=';', index=False, header=False)

