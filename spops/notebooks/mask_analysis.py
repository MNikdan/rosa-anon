import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import torch.nn as nn

def pool2d(img, kernel_size: int):
    pool = nn.MaxPool2d(kernel_size, stride=kernel_size)
    img4d = img[None, None, :, :]
    return pool(img4d).squeeze()

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
# result = sys.argv[2]

image_id = 0
masks_to_plot = []
Round = lambda x, n: eval('"%.'+str(int(n))+'f" % '+repr(int(x)+round(float('.'+str(float(x)).split('.')[1]),n)))

for model in os.listdir(path):
    if 'd0.006' not in model:
        continue
    model = os.path.join(path, model)
    t = torch.load(model)
    for i, mask_name in enumerate(t.keys()):
        mask = t[mask_name].int()

        mask_name = mask_name.replace('model.model.layers.', 'Layer:')
        mask_name = mask_name.replace('. attn', ' attn')
        mask_name = mask_name.replace('.self', ', self')
        rows, cols = mask.shape

        if rows != cols:
            continue

        num_empty_rows = count_empty_rows(mask)
        num_empty_cols = count_empty_columns(mask)

        empty_rows_pct = round((num_empty_rows / rows) * 100)
        empty_cols_pct = round((num_empty_cols / cols) * 100)

        mask_pooled = pool2d(mask, kernel_size).int() != 0

        mask_name = f'{mask_name}\nEmpty Rows: {empty_rows_pct}%\nEmpty Columns: {empty_cols_pct}%'
        # Add mask and its name to the list
        masks_to_plot.append((mask_name, mask_pooled.numpy()))

        image_id += 1
        if image_id == 24:
            break

# Plot masks in a grid
num_rows = 4
num_cols = 6

# Increase the figsize parameter to make the images larger
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))

for i in range(num_rows):
    for j in range(num_cols):
        index = i * num_cols + j
        mask_name, mask = masks_to_plot[index]
        axes[i, j].imshow(mask, cmap='binary')
        axes[i, j].set_title(mask_name)

# Increase the fontsize parameter to make the suptitle larger
# Use the y parameter to adjust the vertical position of the suptitle
plt.suptitle('Mask Visualization', fontsize=20, y=1.0005)

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(result, 'masks.pdf'))
