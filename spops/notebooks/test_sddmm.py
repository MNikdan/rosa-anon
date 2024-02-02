import torch
import spops
from scipy.sparse import csr_matrix
import unittest

SPUTNIK_V2 = 'sputnik_v2'

class TestSDMM(unittest.TestCase):
    def test_sddmm(self):
        M, N, K = 3072, 768, 2048
        device = 'cuda:0'

        # (M, K) * (N, K) -> (M, N)
        # (N, B) * (M, B) -> (N, M)

        A = torch.randn(M, K).to(device)
        BT = torch.randn(N, K).to(device)

        mask = (torch.rand(M, N) > 0.5).int()
        mask_sp = csr_matrix(mask)
        mask = mask.to(device).bool()
        mask_val = torch.tensor(mask_sp.data).float().to(device)
        mask_row_offsets = torch.tensor(mask_sp.indptr).int().to(device)
        mask_col_indices = torch.tensor(mask_sp.indices).int().to(device)
        mask_row_indices = torch.argsort(-1 * torch.diff(mask_row_offsets)).int()

        C_val = spops.sddmm(mask_row_offsets, mask_row_indices.short(), mask_col_indices.short(), A, BT, backend=SPUTNIK_V2)
        real_C_val = torch.mm(A, BT.T.contiguous())[mask]
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(C_val, real_C_val, 1e-4, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
