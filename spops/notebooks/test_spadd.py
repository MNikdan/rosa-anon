import torch
import spops
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import unittest

device = 'cuda:2'

def csr_to_torch_fp32(A_sp):
    A_val = torch.tensor(A_sp.data).float().to(device)
    A_row_offsets = torch.tensor(A_sp.indptr).int().to(device)
    A_col_indices = torch.tensor(A_sp.indices).short().to(device)
    A_row_indices = torch.argsort(-1 * torch.diff(A_row_offsets)).short().to(device)
    return A_val, A_row_offsets, A_row_indices, A_col_indices


def csr_to_torch_fp16(A_sp):
    A_val = torch.tensor(A_sp.data).half().to(device)
    A_row_offsets = torch.tensor(A_sp.indptr).int().to(device)
    A_col_indices = torch.tensor(A_sp.indices).short().to(device)
    A_row_indices = torch.argsort(-1 * torch.diff(A_row_offsets)).short().to(device)
    return A_val, A_row_offsets, A_row_indices, A_col_indices


def csr_to_torch_bf16(A_sp):
    A_val = torch.tensor(A_sp.data).bfloat16().to(device)
    A_row_offsets = torch.tensor(A_sp.indptr).int().to(device)
    A_col_indices = torch.tensor(A_sp.indices).short().to(device)
    A_row_indices = torch.argsort(-1 * torch.diff(A_row_offsets)).short().to(device)
    return A_val, A_row_offsets, A_row_indices, A_col_indices


class TestCSRAdd(unittest.TestCase):
    def test_fp32(self):
        for M in [32, 33, 65, 65]:
            for N in [32, 33, 65, 65]:
                for density in [0.01, 0.1, 0.2, 0.3, 0.99]:
                    A_sp = sp.rand(M, N, density=density, format='csr')
                    A_sp_torch = csr_to_torch_fp32(A_sp)
                    A_dense = torch.tensor(csr_matrix.todense(A_sp)).float().to(device)
                    B = torch.zeros(M, N).cuda()
                    spops.csr_add(*A_sp_torch, B)
                    torch.cuda.synchronize()
                    B_gt = torch.zeros(M, N).cuda()
                    B_gt += A_dense
                    torch.cuda.synchronize()
                    self.assertTrue(torch.allclose(B, B_gt))

    def test_fp16(self):
        for M in [32, 33, 65, 65]:
            for N in [32, 33, 65, 65]:
                for density in [0.01, 0.1, 0.2, 0.3, 0.99]:
                    A_sp = sp.rand(M, N, density=density, format='csr')
                    A_sp_torch = csr_to_torch_fp16(A_sp)
                    A_dense = torch.tensor(csr_matrix.todense(A_sp)).half().to(device)
                    B = torch.zeros(M, N).half().cuda()
                    spops.csr_add(*A_sp_torch, B)
                    torch.cuda.synchronize()
                    B_gt = torch.zeros(M, N).half().cuda()
                    B_gt += A_dense
                    torch.cuda.synchronize()
                    self.assertTrue(torch.allclose(B, B_gt))

    def test_bf16(self):
        for M in [32, 33, 65, 65]:
            for N in [32, 33, 65, 65]:
                for density in [0.01, 0.1, 0.2, 0.3, 0.99]:
                    A_sp = sp.rand(M, N, density=density, format='csr')
                    A_sp_torch = csr_to_torch_bf16(A_sp)
                    A_dense = torch.tensor(csr_matrix.todense(A_sp)).bfloat16().to(device)
                    B = torch.zeros(M, N).bfloat16().cuda()
                    spops.csr_add(*A_sp_torch, B)
                    torch.cuda.synchronize()
                    B_gt = torch.zeros(M, N).bfloat16().cuda()
                    B_gt += A_dense
                    torch.cuda.synchronize()
                    self.assertTrue(torch.allclose(B, B_gt))


if __name__ == '__main__':
    unittest.main()
