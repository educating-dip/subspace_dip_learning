from typing import Optional, List
import hydra
import os
import os.path
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
import copy
import difflib

def find_log_files(log_dir: str) -> str:
    log_files = []
    for path, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                log_files.append(os.path.join(path, file))
    if not log_files:
        raise RuntimeError(f'did not find log file in {log_dir}')
    return log_files

def extract_tensorboard_scalars(
        log_file: str, save_as_npz: str = '', tags: Optional[List[str]] = None) -> dict:
    """
    From https://github.com/educating-dip/bayes_dip/blob/5ae7946756d938a7cd00ad56307a934b8dd3685e/bayes_dip/utils/evaluation_utils.py#L693
    Extract scalars from a tensorboard log file.
    Parameters
    ----------
    log_file : str
        Tensorboard log filepath.
    save_as_npz : str, optional
        File path to save the extracted scalars as a npz file.
    tags : list of str, optional
        If specified, only extract these tags.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ModuleNotFoundError:
        raise RuntimeError('Tensorboard\'s event_accumulator could not be imported, which is '
                           'required by `extract_tensorboard_scalars`')

    ea = event_accumulator.EventAccumulator(
            log_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    tags = tags or ea.Tags()['scalars']

    scalars = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        scalars[tag + '_steps'] = np.asarray(steps)
        scalars[tag + '_scalars'] = np.asarray(values)

    if save_as_npz:
        np.savez(save_as_npz, **scalars)

    return scalars

def print_dct(dct):
    for (item, values) in dct.items():
        print(item)
        for value in values:
            print(value)

# from https://stackoverflow.com/a/47882384
def sorted_dict(d):
    return {k: sorted_dict(v) if isinstance(v, dict) else v
            for k, v in sorted(d.items())}

def collect_runs_paths_per_gamma(base_paths, raise_on_cfg_diff=False):
    paths = {}
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    ref_cfg = None
    ignore_keys_in_cfg_diff = [
            'dip.optim.gamma', 'dip.torch_manual_seed']
    for base_path in base_paths:
        path = os.path.join(os.getcwd().partition('src')[0], base_path)
        for dirpath, dirnames, filenames in os.walk(path):
            if '.hydra' in dirnames:
                cfg = OmegaConf.load(
                        os.path.join(dirpath, '.hydra', 'config.yaml'))
                paths.setdefault(cfg.dip.optim.gamma, []).append(dirpath)

                if ref_cfg is None:
                    ref_cfg = copy.deepcopy(cfg)
                    for k in ignore_keys_in_cfg_diff:
                        OmegaConf.update(ref_cfg, k, None)
                    ref_cfg_yaml = OmegaConf.to_yaml(sorted_dict(OmegaConf.to_object(ref_cfg)))
                    ref_dirpath = dirpath
                else:
                    cur_cfg = copy.deepcopy(cfg)
                    for k in ignore_keys_in_cfg_diff:
                        OmegaConf.update(cur_cfg, k, None)
                    cur_cfg_yaml = OmegaConf.to_yaml(sorted_dict(OmegaConf.to_object(cur_cfg)))
                    try:
                        assert cur_cfg_yaml == ref_cfg_yaml
                    except AssertionError:
                        print('Diff between config at path {} and config at path {}'.format(ref_dirpath, dirpath))
                        differ = difflib.Differ()
                        diff = differ.compare(ref_cfg_yaml.splitlines(),
                                              cur_cfg_yaml.splitlines())
                        print('\n'.join(diff))
                        # print('\n'.join([d for d in diff if d.startswith('-') or d.startswith('+')]))
                        if raise_on_cfg_diff:
                            raise

    paths = {k:sorted(v) for k, v in sorted(paths.items()) if v}
    return paths

@hydra.main(config_path='hydra_cfg', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if not cfg.val.select_gamma_multirun_base_paths:
        raise ValueError

    runs = collect_runs_paths_per_gamma(
            cfg.val.select_gamma_multirun_base_paths)  # , raise_on_cfg_diff=False)  # -> check diff output manually
    print_dct(runs) # visualise runs and models checkpoints

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.select_gamma_run_paths_filename)), exist_ok=True)
    with open(cfg.val.run_paths_filename, 'w') as f:
        json.dump(runs, f, indent=1)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.select_gamma_results_filename)), exist_ok=True)

    infos = {}
    for i_run, (gamma, histories_path) in enumerate(runs.items()):
        psnr_histories = []
        samples_log_files = find_log_files(histories_path[0])
        for tb_path in samples_log_files:
            extracted_min_loss_psnr = extract_tensorboard_scalars(tb_path)['min_loss_output_psnr_scalars']
            psnr_histories.append(extracted_min_loss_psnr)
        
        median_psnr_output = np.median(psnr_histories, axis=0)
        psnr_steady = np.max(median_psnr_output[
                cfg.val.psnr_steady_start:cfg.val.psnr_steady_stop])
        infos[gamma] = {
                'PSNR_steady': psnr_steady, 'PSNR_0': median_psnr_output[0]}

        with open(cfg.val.select_gamma_results_filename, 'w') as f:
            json.dump(infos, f, indent=1)

    def key(info):
        return -info['PSNR_steady']

    infos = {k: v for k, v in sorted(infos.items(), key=lambda item: key(item[1]))}
    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.select_gamma_results_sorted_filename)), exist_ok=True)
    with open(cfg.val.select_gamma_results_sorted_filename, 'w') as f:
        json.dump(infos, f, indent=1)

if __name__ == '__main__':
    coordinator()