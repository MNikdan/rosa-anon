import torch
from enum import Enum
from scipy.sparse import csr_matrix, csc_matrix

DEFAULT_CUDA_DEVICE = 'cuda:0'

class Storage(Enum):
    CSR = 1
    CSC = 2

    # TODO: Why do I need this?
    def __eq__(self, other):
        return other.value == self.value


class IndexType(Enum):
    INT_16 = 1
    INT_32 = 2


def prepare_mask_csr(mask):
    device = DEFAULT_CUDA_DEVICE
    mask_sp = csr_matrix(mask)
    mask_row_offsets = torch.tensor(mask_sp.indptr).int().to(device)
    mask_col_indices = torch.tensor(mask_sp.indices).int().to(device)
    mask_row_indices = torch.argsort(-1 * torch.diff(mask_row_offsets)).int().to(device)
    return mask_row_offsets, mask_row_indices, mask_col_indices, 0


def prepare_mask_csc(mask):
    device = DEFAULT_CUDA_DEVICE
    mask_sp = csc_matrix(mask)
    mask_row_offsets = torch.tensor(mask_sp.indptr).int().to(device)
    mask_col_indices = torch.tensor(mask_sp.indices).int().to(device)
    mask_row_indices = torch.argsort(-1 * torch.diff(mask_row_offsets)).int().to(device)
    return mask_row_offsets, mask_row_indices, mask_col_indices, 0


def prepare_mask(mask, storage: Storage, index_type: IndexType, device):
    if storage == Storage.CSR:
        mask_sp = csr_matrix(mask)
    else:
        mask_sp = csc_matrix(mask)

    if index_type == IndexType.INT_16:
        indptr = torch.Tensor(mask_sp.indptr).int().to(device)
        indices = torch.Tensor(mask_sp.indices).short().to(device)
        row_counts = torch.diff(indptr)
        ordered_indices = torch.argsort(-1 * row_counts).short().to(device)
        sorted_row_counts = row_counts[ordered_indices.long()]
        last0 = (sorted_row_counts != 0).int().sum().cpu().item()
        last1 = sorted_row_counts[0].int()
        return indptr, ordered_indices, indices, last0, last1, storage
    else:
        indptr = torch.Tensor(mask_sp.indptr).int().to(device)
        indices = torch.Tensor(mask_sp.indices).int().to(device)
        ordered_indices = torch.argsort(-1 * torch.diff(indptr)).int().to(device)
        row_counts = torch.diff(indptr)
        sorted_row_counts = row_counts[ordered_indices.long()]
        last0 = (ordered_indices.squeeze() != 0).int().sum()
        last1 = sorted_row_counts[0].int()
        return indptr, ordered_indices, indices, last0, last1, storage


def prepare_mask_v1(mask):
    device = DEFAULT_CUDA_DEVICE
    backend = 'sputnik'
    return *prepare_mask(mask, Storage.CSR, IndexType.INT_32, device), backend


def prepare_mask_v2_csr_32(mask):
    device = DEFAULT_CUDA_DEVICE
    backend = 'sputnik_v2'
    return *prepare_mask(mask, Storage.CSR, IndexType.INT_32, device), backend


def prepare_mask_v2_csc_32(mask):
    device = DEFAULT_CUDA_DEVICE
    backend = 'sputnik_v2'
    return *prepare_mask(mask, Storage.CSC, IndexType.INT_32, device), backend


def prepare_mask_v2_csr_16(mask):
    device = DEFAULT_CUDA_DEVICE
    backend = 'sputnik_v2'
    return *prepare_mask(mask, Storage.CSR, IndexType.INT_16, device), backend


def prepare_mask_v2_csc_16(mask):
    device = DEFAULT_CUDA_DEVICE
    backend = 'sputnik_v2'
    return *prepare_mask(mask, Storage.CSC, IndexType.INT_16, device), backend
