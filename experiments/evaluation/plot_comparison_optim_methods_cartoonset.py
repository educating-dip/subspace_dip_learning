from typing import List, Dict
import hydra
import argparse
import os
import pickle
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from subspace_dip.utils import find_log_files, extract_tensorboard_scalars, print_dct, sorted_dict

METHODS = ['dip', 'edip', 'sub[adam]', 'sub[lbfgs]', 'sub[ngd]']
METHODS_NAMES = {
        'dip': 'DIP',
        'edip': 'EDIP',
        'sub[adam]': 'Sub-DIP (adam)', 
        'sub[lbfgs]': 'Sub-DIP (lbfgs)', 
        'sub[ngd]': 'Sub-DIP (ngd)'
    } 

DEFAULT_COLORS = {
    'dip': '#e63946',
    'edip': '#5555ff',
    'sub[adam]': '#55c3ff',
    'sub[lbfgs]': '#5a6c17',
    'sub[ngd]': '#54db39',
}

DEFAULT_SYM_HATCH = {
    45: {'marker': 'o', 'hatch': 3*'-'},  
    95: {'marker': 's', 'hatch': 4*'+'},
    285: {'marker': '*', 'hatch': 3*'//'}
}

DEFAULT_SPAN = {}
cnt = 0
for method in METHODS:
    DEFAULT_SPAN[method] = [cnt, cnt+2]
    cnt += 4

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='cartoonset', help='name of the dataset used')
parser.add_argument('--runs_file', type=str, default='../cartoonset.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experimental_outputs_path', type=str, default='validation', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--load_data_from_path', type=str, default=None, help='load data cached from a previous run with')
args = parser.parse_args()

def extract_data_from_paths_all_method(
    runs_file_dict: Dict,
    methods: List = METHODS
    ):
    dt = {}
    for key in runs_file_dict.keys():
        dt.setdefault(key, {})
        for method in methods:
            print(f'\nloading data reconstructed with: {method}\n')
            path_method_runs = runs_file_dict[key][method]
            print(f'loading data from: {path_method_runs}\n')
            dt[key].setdefault(method, {})
            dt[key][method].setdefault('psnr_histories', [])
            dt[key][method].setdefault('loss_histories', [])
            dt[key][method].setdefault('time_histories', [])
            for path_method_run in path_method_runs:
                for dirpath, dirnames, _ in os.walk(path_method_run):
                    if '.hydra' in dirnames:
                        samples_log_files = find_log_files(dirpath)
                        with tqdm(range(len(samples_log_files)), desc='loading') as pbar:
                            for i in pbar:
                                tb = extract_tensorboard_scalars(samples_log_files[i])
                                dt[key][method]['psnr_histories'].append(tb['min_loss_output_psnr_scalars'])
                                dt[key][method]['loss_histories'].append(tb['loss_scalars'])
                                dt[key][method]['time_histories'].append( tb['loss_times'])
    return dt

def plot_methods_comparison() -> None:
    
    os.makedirs(args.experimental_outputs_path, exist_ok=True)
    if args.load_data_from_path is None:
        runs_file_dict = OmegaConf.load(args.runs_file)
        dt = extract_data_from_paths_all_method(runs_file_dict, METHODS) 
        with open(
                os.path.join(
                    args.experimental_outputs_path, f'extracted_{args.dataset_name}_data.pkl'), 'wb') as f:
            pickle.dump(dt, f)
    else:
        with open(
                os.path.join(
                    args.load_data_from_path, f'extracted_{args.dataset_name}_data.pkl'), 'rb') as f:
            dt = pickle.load(f)
    
    info_plot_dict = {}
    for key in dt.keys():
        info_plot_dict.setdefault(key, {})
        for method in METHODS:
            psnr_max_samples, psnr_steady_samples, time_conv_per_sample, psnr_conv_per_sample = [], [], [], []
            info_plot_dict[key].setdefault(method, {})
            for psnr, loss, time in zip(
                    dt[key][method]['psnr_histories'], 
                        dt[key][method]['loss_histories'], 
                            dt[key][method]['time_histories']):
        
                psnr_max_samples.append(np.max(psnr))
                steady_start = len(psnr) - int(0.1*len(psnr))
                psnr_steady_samples.append(np.mean(psnr[steady_start:]))
                loss_at_conv = np.mean(loss[steady_start:])
                loss_diff_one_perc = loss_at_conv + 0.01*loss_at_conv
                idx = np.where(loss[:steady_start] < loss_diff_one_perc)[0][0]
                # idx = (np.abs(loss[:steady_start] - loss_diff_one_perc)).argmin()
                psnr_conv_per_sample.append(psnr[idx])
                time_conv_per_sample.append(time[idx] - time[0])

            info_plot_dict[key][method]['max_mean'] = np.mean(psnr_max_samples)
            info_plot_dict[key][method]['steady_mean'] = np.mean(psnr_steady_samples)
            info_plot_dict[key][method]['max_std'] = np.std(psnr_max_samples)  / np.sqrt(len(psnr_max_samples))
            info_plot_dict[key][method]['steady_std'] = np.std(psnr_steady_samples) / np.sqrt(len(psnr_max_samples))
            info_plot_dict[key][method]['psnr_loss_conv'] = np.mean(psnr_conv_per_sample)
            info_plot_dict[key][method]['time_loss_conv'] = np.mean(time_conv_per_sample)

    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{dashbox}",              
        r"\setmainfont{xcolor}",
        ])
    fig, _ = plt.subplots(1, len(dt)+1 , figsize=(15, 3), gridspec_kw={
        'width_ratios': [1.]*len(dt) + [1.],  # includes spacer columns
        'hspace': 0.2})
    legend_kwargs = {'loc': 'lower right', 'ncol':2, 'borderpad': .85, 'prop': {'size': 'small'}}
    
    for k, (key, ax) in enumerate(zip(dt.keys(), fig.axes[:len(dt)])):
        for i, method in enumerate(METHODS):
            ax.plot(
                DEFAULT_SPAN[method],
                [info_plot_dict[key][method]['max_mean'], info_plot_dict[key][method]['max_mean']], 
                color=DEFAULT_COLORS[method],
                linewidth=1.5,
                linestyle='solid',
                alpha=1, 
                label='MAX' if i == len(METHODS)-1 else None
                )
            ax.fill_between(
                DEFAULT_SPAN[method],
                [info_plot_dict[key][method]['max_mean'] + info_plot_dict[key][method]['max_std'], info_plot_dict[key][method]['max_mean'] + info_plot_dict[key][method]['max_std']], 
                [info_plot_dict[key][method]['max_mean'] - info_plot_dict[key][method]['max_std'], info_plot_dict[key][method]['max_mean'] - info_plot_dict[key][method]['max_std']],
                alpha=.2,
                edgecolor=DEFAULT_COLORS[method], facecolor=DEFAULT_COLORS[method],
                linewidth=0
                )
            ax.plot(
                DEFAULT_SPAN[method], 
                [info_plot_dict[key][method]['steady_mean'],info_plot_dict[key][method]['steady_mean']],
                color=DEFAULT_COLORS[method],
                linewidth=1.75,
                linestyle=':',
                alpha=1, 
                label='STEADY' if i == len(METHODS)-1 else None
                )
            ax.fill_between(
                DEFAULT_SPAN[method],
                [info_plot_dict[key][method]['steady_mean'] + info_plot_dict[key][method]['steady_std'], info_plot_dict[key][method]['steady_mean'] + info_plot_dict[key][method]['steady_std']], 
                [info_plot_dict[key][method]['steady_mean'] - info_plot_dict[key][method]['steady_std'], info_plot_dict[key][method]['steady_mean'] - info_plot_dict[key][method]['steady_std']],
                alpha=.2,
                hatch='xxx',
                edgecolor=DEFAULT_COLORS[method], facecolor=DEFAULT_COLORS[method],
                linewidth=0
                )
        ax.set_title(f'#angles: {key}', fontsize=plt.rcParams['axes.titlesize'], loc='left') 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([], minor=True)
        ax.set_xticks([])
        yticks = matplotlib.ticker.MaxNLocator(4)
        ax.yaxis.set_major_locator(yticks)
        if k in [1, 2]: 
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_ticklabels([])
            ax.yaxis.set_ticks_position('none')

        ax.grid(alpha=0.3)
        ax.set_ylim([26, 37])
        if k == 0:
            ax.set_ylabel('PSNR [dB]', fontsize=plt.rcParams['axes.titlesize'], )
            leg = ax.legend(**(legend_kwargs or {}))
            for handle in leg.legendHandles:
                handle.set_color('k')
    
    for key in dt.keys():
        for i, method in enumerate(METHODS):
            fig.axes[-1].scatter(
                info_plot_dict[key][method]['psnr_loss_conv'], 
                info_plot_dict[key][method]['time_loss_conv'], 
                color=DEFAULT_COLORS[method],
                facecolor='white',
                s=150,
                **DEFAULT_SYM_HATCH[key],
                label = key if (i == len(METHODS)-1) else None)

        fig.axes[-1].grid(alpha=0.3)        
        fig.axes[-1].spines['top'].set_visible(False)
        fig.axes[-1].spines['right'].set_visible(False)
        yticks = matplotlib.ticker.MaxNLocator(3)
        fig.axes[-1].yaxis.set_major_locator(yticks)
    leg = fig.axes[-1].legend(**({'bbox_to_anchor': (0.41, 0., 0.5, 1.17), 'ncol':3, 'borderpad': .85, 'prop': {'size': 'small'}} or {}))
    for handle in leg.legendHandles:
        handle._original_edgecolor = '#000000'
    fig.axes[-1].set_ylabel('time (1%-loss) [s]')
    fig.axes[-1].set_xlabel('PSNR (1%-loss) [dB]')

    handles = []
    for name in METHODS:
        handles.append(
            matplotlib.lines.Line2D([], [], 
                color=DEFAULT_COLORS[name], 
                marker='s', 
                ls='', 
                label=METHODS_NAMES[name])
            )
    fig.legend(
        handles=handles,
        **{'bbox_to_anchor': (0.48, -0.15, 0.1, 0.25), 'ncol':5, 'borderpad': .85, 'prop': {'size': 'small'}}
        )
    fig.savefig(os.path.join(args.experimental_outputs_path, f'psnr_comparison_{args.dataset_name}.png'), bbox_inches='tight', pad_inches=0., dpi=600)
plot_methods_comparison()
