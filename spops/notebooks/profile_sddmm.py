#!/usr/bin/env python3

import os
import sys
import spops
import utils
import torch
from torch import nn
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
from utils import DEFAULT_CUDA_DEVICE

DEFAULT_REPS = 5


def pool2d(img, kernel_size: int):
    pool = nn.MaxPool2d(kernel_size, stride=kernel_size)
    img4d = img[None, None, :, :]
    return pool(img4d).squeeze()


def profile_mask(mask, data_prep, algorithm, num_warmups=3,
                 num_reps=DEFAULT_REPS):
    M, N = mask.shape
    K = 512
    A = torch.ones(M, K, dtype=torch.float16).to(DEFAULT_CUDA_DEVICE)
    BT = torch.ones(N, K, dtype=torch.float16).to(DEFAULT_CUDA_DEVICE)
    prepped_data = data_prep(mask)

    for _ in range(num_warmups):
        duration = torch.tensor(0, dtype=torch.float32)
        algorithm(*prepped_data, A, BT, duration)

    durations = torch.zeros(num_reps, dtype=torch.float32)
    for i in range(num_reps):
        duration = torch.tensor(0, dtype=torch.float32)
        C = algorithm(*prepped_data, A, BT, duration)
        durations[i] = duration

    return C, durations


class RoSAProfile:
    def __init__(self, base_path, output_path):

        paths = os.listdir(base_path)
        for path in paths:
            print(f'Running {path}')
            t = torch.load(os.path.join(base_path, path))

            # model_id/
            model_report_path = os.path.join(output_path, path)

            if not os.path.isdir(model_report_path):
                os.makedirs(model_report_path)

            test_names = []
            sputnik = []
            sputnik_v2 = []

            num_masks = 20  # len(t.keys())
            sample_avg = 0
            sample_avg_v2 = 0
            for _, name in zip(range(num_masks), t.keys()):
                print(f'Processing {name}')
                torch.cuda.empty_cache()
                mask = t[name].int()
                C, durations = profile_mask(mask, utils.prepare_mask_csr, sputnik_algorithm)
                C_v2, durations_v2 = profile_mask(mask, utils.prepare_mask_v2, sputnik_v2_algorithm)

                assert torch.allclose(C, C_v2)

                sample_avg += durations.mean()
                sample_avg_v2 += durations_v2.mean()
                assert (torch.allclose(C, C_v2))

                test_names.append(name)
                sputnik.append(durations.median())
                sputnik_v2.append(durations_v2.median())

            sample_avg /= num_masks
            sample_avg_v2 /= num_masks
            bar_width = 0.25
            plt.bar(np.arange(num_masks) - bar_width / 2, sputnik, width=bar_width, label='Sputnik', edgecolor='black')
            plt.bar(np.arange(num_masks) + bar_width / 2, sputnik_v2, width=bar_width, label='Sputnik V2',
                    edgecolor='black')
            plt.xticks(np.arange(num_masks), test_names, rotation=45)
            plt.title(path + f'\nAverage Run: {sample_avg}\nAverage Run {sample_avg_v2} (V2)')
            plt.tight_layout()
            plt.legend()

            plt.savefig(os.path.join(output_path, 'sddmm_benchmark', path + '_benchmark.png'))
            plt.close()
            del t
            torch.cuda.empty_cache()


class SparsityProfile:
    def __init__(self, base_path, output_path):
        paths = os.listdir(base_path)
        BINS = 101
        ROW_AXIS = 1
        COL_AXIS = 0

        for M, N, K in [
            # (64, 64, 64),
            # (128, 128, 128),
            (512, 512, 512),
            (1024, 1024, 1024),
            (4096, 4096, 4096),
        ]:
            test_names = []
            sputnik = []
            sputnik_v2 = []

            num_tests = 0  # len(t.keys())
            first = True
            for i in range(1, 20, 1):
                print(f'Running {i}')
                torch.cuda.empty_cache()
                num_tests += 1
                density = 0.001 * i
                # name = str(int(density * 100)) + '%'
                name = f'{(density * 100):.1f}%'

                mask = sparse.random(M, N, density=0.01 * i, format='dense', dtype='f')
                mask = torch.Tensor(np.array(mask.data))

                A = torch.rand(M, K, dtype=torch.float32).cuda()
                BT = torch.rand(N, K, dtype=torch.float32).cuda()

                duration = torch.tensor(0, dtype=torch.float32)
                duration_v2 = torch.tensor(0, dtype=torch.float32)

                csr_mask = utils.prepare_mask_csr(mask)
                csr_mask_v2 = utils.prepare_mask_v2(mask)

                if first:
                    # first = False
                    num_warmups = 3
                    for _ in range(num_warmups):
                        C_val = spops.sddmm_benchmark(*csr_mask, A, BT, duration)
                        C_val_v2 = spops.sddmm_benchmark(*csr_mask_v2, A, BT, duration_v2, backend='sputnik_v2')

                num_reps = 20
                durations = torch.zeros(num_reps, dtype=torch.float32)
                durations_v2 = torch.zeros(num_reps, dtype=torch.float32)

                for i in range(num_reps):
                    C_val = spops.sddmm_benchmark(*csr_mask, A, BT, duration)
                    durations[i] = duration

                for i in range(num_reps):
                    C_val_v2 = spops.sddmm_benchmark(*csr_mask_v2, A, BT, duration_v2, backend='sputnik_v2')
                    durations_v2[i] = duration_v2

                assert (torch.allclose(C_val, C_val_v2))

                test_names.append(name)
                sputnik.append(torch.median(durations))
                sputnik_v2.append(torch.median(durations_v2))

            bar_width = 0.25
            plt.bar(np.arange(num_tests) - bar_width / 2, sputnik, width=bar_width, label='Sputnik', edgecolor='black')
            plt.bar(np.arange(num_tests) + bar_width / 2, sputnik_v2, width=bar_width, label='Sputnik V2',
                    edgecolor='black')
            plt.xticks(np.arange(num_tests), test_names, rotation=45)
            plt.title(f'Benchmark M={M} N={N} K={K}')
            plt.legend()
            plt.xlabel('Density (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'sddmm_benchmark', f'M={M}_N={N}_K={K}_density_benchmark.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    if len(sys.argv) == 3:
        base_path = sys.argv[1]
        output_path = sys.argv[2]
        # stats = RoSAProfile(base_path, output_path)
        stats = SparsityProfile(base_path, output_path)
    else:
        print('Error - run the script like this:\npython3 kernel_analysis <data_path> <output_path>')
