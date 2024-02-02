#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
from profile_sddmm import *
from utils import DEFAULT_CUDA_DEVICE

configurations = {
    'Sputnik': utils.prepare_mask_v1,
    'Ours': utils.prepare_mask_v2_csr_16,
    # 'Sputnik V2 CSC INT 16': utils.prepare_mask_v2_csc_16,
    # 'Sputnik V2 CSC INT 32': utils.prepare_mask_v2_csc_32,
}



class RoSAProfile:
    def __init__(self, path, output_path):
        perf_table = os.path.join(output_path, 'raw_data.csv')
        layer_table = os.path.join(output_path, 'layer_meta.csv')

        perf_file = open(perf_table, 'w')
        layer_file = open(layer_table, 'w')

        num_runs = DEFAULT_REPS

        perf_file.write(f'layer_name;method;{";".join(["run" + str(i) for i in range(num_runs)])};mean;median\n')
        layer_file.write('layer_name;empty_rows_perc;empty_cols_perc;max_empty_perc;sparsity_perc\n')

        print(f'Running {path}')
        t = torch.load(os.path.join(path))

        # model_id/
        model_report_path = os.path.join(output_path, path)

        # if not os.path.isdir(model_report_path):
            # os.makedirs(model_report_path)

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
            sparsity_perc = (col_indices.shape[0] / (mask.shape[0] * mask.shape[1])) * 100
            print(
                f'name: {name}\n'
                f'dims: {mask.shape}\n'
                f'sparsity: {(col_indices.shape[0] / (mask.shape[0] * mask.shape[1])) * 100}%\n'
                f'empty rows: {empty_rows_perc}%\nempty cols: {empty_cols_perc}%\nmax_empty: {empty_max_perc}%\n'
            )

            test_names.append(name)

            mask = t[name].int()
            C_true, durations = profile_mask(mask, utils.prepare_mask_v1, spops.sddmm)

            sample_results = []
            row_col_info = f'{name};{empty_rows_perc};{empty_cols_perc};{empty_max_perc};{sparsity_perc}\n'
            layer_file.write(row_col_info)
            layer_name = name
            for name, f in zip(configurations.keys(), configurations.values()):
                C, durations = profile_mask(mask, f, spops.sddmm)
                sample_results.append(torch.median(durations).cpu().item())

                durations_str = [str(i) for i in durations.tolist()]

                perf_file.write(f'{layer_name};{name};{";".join(durations_str)};{torch.mean(durations)};{torch.median(durations)}\n')


                assert torch.allclose(C_true, C)

            results.append(sample_results)
            perf_file.flush()
            layer_file.flush()


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

        perf_file.close()
        layer_file.close()

if __name__ == '__main__':
    if len(sys.argv) == 3:
        base_path = sys.argv[1]
        output_path = sys.argv[2]
        stats = RoSAProfile(base_path, output_path)
        # stats = SparsityProfile(base_path, output_path)
    else:
        print('Error - run the script like this:\npython3 kernel_analysis <data_path> <output_path>')
