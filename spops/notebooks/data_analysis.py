#!/usr/bin/env python3

from profile_sddmm import *
import seaborn as sns
from utils import DEFAULT_CUDA_DEVICE

configurations = {
    'Sputnik V1 CSR INT 32': utils.prepare_mask_v1,
    'Sputnik V2 CSR INT 16': utils.prepare_mask_v2_csr_16,
    # 'Sputnik V2 CSC INT 16': utils.prepare_mask_v2_csc_16,
    # 'Sputnik V2 CSC INT 32': utils.prepare_mask_v2_csc_32,
}



class RoSAProfile:
    def __init__(self, base_path, output_path):
        paths = os.listdir(base_path)
        for path in paths[:1]:
            print(f'Running {path}')
            t = torch.load(os.path.join(base_path, path))

            # model_id/
            model_report_path = os.path.join(output_path, path)

            if not os.path.isdir(model_report_path):
                os.makedirs(model_report_path)

            test_names = []
            sputnik = []
            sputnik_v2 = []
            num_algorithms = len(configurations.keys())
            algorithm_names = list(configurations.keys())

            num_masks = len(t.keys())
            sample_avg = 0
            sample_avg_v2 = 0
            results = []
            for _, name in zip(range(num_masks), list(t.keys())[1:]):
                torch.cuda.empty_cache()
                mask = t[name].int()

                row_offsets, _, col_indices, last = utils.prepare_mask_csr(mask)
                row_counts = torch.sort(torch.diff(row_offsets)).values

                col_offsets, _, _, _ = utils.prepare_mask_csc(mask)
                col_counts = torch.sort(torch.diff(col_offsets)).values

                empty_rows_perc = ((row_counts == 0).int().sum() / row_counts.shape[0]) * 100
                empty_cols_perc = ((col_counts == 0).int().sum() / col_counts.shape[0]) * 100
                empty_max_perc = max(empty_rows_perc, empty_cols_perc)
                print(
                    f'name: {name}\n'
                    f'dims: {mask.shape}\n'
                    f'sparsity: {(col_indices.shape[0] / (mask.shape[0] * mask.shape[1])) * 100}%\n'
                    f'empty rows: {empty_rows_perc}%\nempty cols: {empty_cols_perc}%\nmax_empty: {empty_max_perc}%\n'
                )

                test_names.append(name)

                mask = t[name].int()
                C_true, durations = profile_mask(mask, utils.prepare_mask_v1, spops.sddmm_benchmark)

                sample_results = []
                for name, f in zip(configurations.keys(), configurations.values()):
                    C, durations = profile_mask(mask, f, spops.sddmm_benchmark)
                    sample_results.append(torch.median(durations).cpu().item())
                    assert torch.allclose(C_true, C)
                results.append(sample_results)

                """
                fig, axs = plt.subplots(2, 2, layout='constrained')
                plt.title(name)
                axs[0][0].set_title('Row Counts')
                axs[0][1].bar(np.arange(num_algorithms), results, edgecolor='black', align='center')
                axs[0][1].set_xticks(np.arange(num_algorithms), algorithm_names)
                axs[0][0].plot(range(row_counts.shape[0]), row_counts.cpu())
                axs[1][0].set_title('Col Counts')
                axs[1][0].plot(range(col_counts.shape[0]), col_counts.cpu())
                axs[1][1].imshow(pool2d(mask, 8).cpu(), cmap='binary')
                plt.show()
                """

            results = np.array(results)
            title = f"{path}\n"
            for i in range(num_algorithms):
                mean_run = np.mean(np.squeeze(results[:, i]))
                sns.barplot(results[:, i], label=algorithm_names[i], edgecolor='black')
                title += f'Average Run of {algorithm_names[i]}: {mean_run}\n'

            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'{path}.png'), dpi=4000, bbox_inches='tight')
            plt.close()

class SparsityProfile:
    def __init__(self, base_path, output_path):
        device = 'cuda:0'
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

                results = []

                C_true, durations = profile_mask(mask, utils.prepare_mask_v1, spops.sddmm)

                for name, f in configurations:
                    params = f(mask)
                    if first:
                        num_warmups = 3
                        for _ in range(num_warmups):
                            __ = spops.sddmm(*params, A, BT, duration)

                    num_reps = 5
                    durations = torch.zeros(num_reps, dtype=torch.float32)

                    for i in range(num_reps):
                        C = spops.sddmm(*params, A, BT, duration)
                        durations[i] = duration

                    assert (torch.allclose(C_true, C))

                    results.append(torch.median(durations))

                bar_width = 0.1
                plt.bar(np.arange(num_tests) - bar_width / 2, sputnik, width=bar_width, label='Sputnik', edgecolor='black')
                plt.bar(np.arange(num_tests) + bar_width / 2, sputnik_v2, width=bar_width, label='Sputnik V2', edgecolor='black')
                plt.xticks(np.arange(num_tests), configurations.keys(), rotation=45)
                plt.title(f'Benchmark M={M} N={N} K={K}')
                plt.legend()
                plt.xlabel('Density (%)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, 'sddmm_benchmark', f'M={M}_N={N}_K={K}_density_benchmark.png'), dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == '__main__':
    if len(sys.argv) == 3:
        base_path = sys.argv[1]
        output_path = sys.argv[2]
        stats = RoSAProfile(base_path, output_path)
        # stats = SparsityProfile(base_path, output_path)
    else:
        print('Error - run the script like this:\npython3 kernel_analysis <data_path> <output_path>')
