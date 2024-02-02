import torch
import spops_backend
from torch import int32, float16, float32, bfloat16

from notebooks.utils import Storage


def sddmm_benchmark(mask_row_offsets, mask_row_indices, mask_col_indices, last0: int, last1: int,
        storage: Storage, backend: str, A, BT, t, return_type=None):
    assert backend in ['sputnik', 'sputnik_v2']
    assert all([t.dtype in [float16, float32, bfloat16] for t in [A, BT]]), 'Only fp32, bf16 and fp16 are supported for sddmm.'


    if backend == 'sputnik_v2':
        out = spops_backend.sputnik_v2_sddmm_fp32_benchmark(mask_row_offsets.to(int32), mask_row_indices, mask_col_indices, last0, last1, A.to(float32), BT.to(float32), storage == Storage.CSR, t)[0]
    else:
        out = spops_backend.sputnik_sddmm_fp32(mask_row_offsets.to(int32), mask_row_indices.to(int32), mask_col_indices.to(int32), A.to(float32), BT.to(float32), t)[0]


    if return_type is None:
        return out
    else:
        return out.to(return_type)


def sddmm(mask_row_offsets, mask_row_indices, mask_col_indices, A, BT, backend='sputnik', return_type=None):
    assert backend in ['sputnik', 'sputnik_v2']
    assert all([t.dtype in [float16, float32, bfloat16] for t in [A, BT]]), 'Only fp32, bf16 and fp16 are supported for sddmm.'

    if backend == 'sputnik_v2':
        sorted_row_counts = torch.diff(mask_row_offsets)[mask_row_indices.int()]
        last0 = (sorted_row_counts != 0).int().sum().cpu().item()
        last1 = sorted_row_counts[0].int()
        out = spops_backend.sputnik_v2_sddmm_fp32(mask_row_offsets.to(int32), mask_row_indices, mask_col_indices, last0, last1, A.to(float32), BT.to(float32))[0]
    else:
        out = spops_backend.sputnik_sddmm_fp32(mask_row_offsets.to(int32), mask_row_indices.to(int32), mask_col_indices.to(int32), A.to(float32), BT.to(float32))[0]

    if return_type is None:
        return out
    else:
        return out.to(return_type)
