import os
import urllib.request

import gymnasium
import numpy as np
from tqdm import tqdm

from ogbench.relabel_utils import add_oracle_reps, relabel_dataset

DEFAULT_DATASET_DIR = '~/.ogbench/data'
DATASET_URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench'


def load_dataset(dataset_path, ob_dtype=np.float32, action_dtype=np.float32, compact_dataset=False, add_info=False):
    """Load OGBench dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        add_info: Whether to add observation information ('qpos', 'qvel', and 'button_states') to the dataset.

    Returns:
        Dictionary containing the dataset. The dictionary contains the following keys: 'observations', 'actions',
        'terminals', and 'next_observations' (if `compact_dataset` is False) or 'valids' (if `compact_dataset` is True).
        If `add_info` is True, the dictionary may also contain additional keys for observation information.
    """
    file = np.load(dataset_path)

    dataset = dict()
    for k in ['observations', 'actions', 'terminals']:
        if k == 'observations':
            dtype = ob_dtype
        elif k == 'actions':
            dtype = action_dtype
        else:
            dtype = np.float32
        dataset[k] = file[k][...].astype(dtype, copy=False)

    if add_info:
        # Read observation information.
        info_keys = []
        for k in ['qpos', 'qvel', 'button_states']:
            if k in file:
                dataset[k] = file[k][...]
                info_keys.append(k)

    # Example:
    # Assume each trajectory has length 4, and (s0, a0, s1), (s1, a1, s2), (s2, a2, s3), (s3, a3, s4) are transition
    # tuples. Note that (s4, a4, s0) is *not* a valid transition tuple, and a4 does not have a corresponding next state.
    # At this point, `dataset` loaded from the file has the following structure:
    #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
    # -------------------------------------------------------------
    # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
    # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
    # 'terminals'   : [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  1, ...]

    if compact_dataset:
        # Compact dataset: We need to invalidate the last state of each trajectory so that we can safely get
        # `next_observations[t]` by using `observations[t + 1]`.
        # Our goal is to have the following structure:
        #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
        # -------------------------------------------------------------
        # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
        # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
        # 'terminals'   : [ 0,  0,  0,  1,  1,  0,  0,  0,  1,  1, ...]
        # 'valids'      : [ 1,  1,  1,  1,  0,  1,  1,  1,  1,  0, ...]

        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)
    else:
        # Regular dataset: Generate `next_observations` by shifting `observations`.
        # Our goal is to have the following structure:
        #                       |<- traj 1 ->|  |<- traj 2 ->|  ...
        # ----------------------------------------------------------
        # 'observations'     : [s0, s1, s2, s3, s0, s1, s2, s3, ...]
        # 'actions'          : [a0, a1, a2, a3, a0, a1, a2, a3, ...]
        # 'next_observations': [s1, s2, s3, s4, s1, s2, s3, s4, ...]
        # 'terminals'        : [ 0,  0,  0,  1,  0,  0,  0,  1, ...]

        ob_mask = (1.0 - dataset['terminals']).astype(bool)
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
        dataset['next_observations'] = dataset['observations'][next_ob_mask]
        dataset['observations'] = dataset['observations'][ob_mask]
        dataset['actions'] = dataset['actions'][ob_mask]
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

        if add_info:
            for k in info_keys:
                dataset[k] = dataset[k][ob_mask]

    return dataset


def download_datasets(dataset_names, dataset_dir=DEFAULT_DATASET_DIR):
    """Download OGBench datasets.

    Args:
        dataset_names: List of dataset names to download.
        dataset_dir: Directory to save the datasets.
    """
    # Make dataset directory.
    dataset_dir = os.path.expanduser(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download datasets.
    dataset_file_names = []
    for dataset_name in dataset_names:
        dataset_file_names.append(f'{dataset_name}.npz')
        dataset_file_names.append(f'{dataset_name}-val.npz')
    for dataset_file_name in dataset_file_names:
        dataset_file_path = os.path.join(dataset_dir, dataset_file_name)
        if not os.path.exists(dataset_file_path):
            dataset_url = f'{DATASET_URL}/{dataset_file_name}'
            print('Downloading dataset from:', dataset_url)
            response = urllib.request.urlopen(dataset_url)
            tmp_dataset_file_path = f'{dataset_file_path}.tmp'
            with open(tmp_dataset_file_path, 'wb') as tmp_file:
                with tqdm.wrapattr(
                    tmp_file,
                    'write',
                    miniters=1,
                    desc=dataset_url.split('/')[-1],
                    total=getattr(response, 'length', None),
                ) as file:
                    for chunk in response:
                        file.write(chunk)
            os.rename(tmp_dataset_file_path, dataset_file_path)


def make_env_and_datasets(
    dataset_name,
    dataset_dir=DEFAULT_DATASET_DIR,
    dataset_path=None,
    compact_dataset=False,
    env_only=False,
    dataset_only=False,
    cur_env=None,
    add_info=False,
    **env_kwargs,
):
    """Make OGBench environment and load datasets.

    Args:
        dataset_name: Dataset name.
        dataset_dir: Directory to save/load the datasets.
        dataset_path: (Optional) Path to the dataset file.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        env_only: Whether to return only the environment.
        dataset_only: Whether to return only the datasets.
        cur_env: Current environment (only used when `dataset_only` is True).
        add_info: Whether to add observation information ('qpos', 'qvel', and 'button_states') to the datasets.
        **env_kwargs: Keyword arguments to pass to the environment.
    """
    # Make environment.
    splits = dataset_name.split('-')
    dataset_add_info = add_info
    env = cur_env
    if 'singletask' in splits:
        # Single-task environment.
        pos = splits.index('singletask')
        env_name = '-'.join(splits[: pos - 1] + splits[pos:])  # Remove the dataset type.
        if not dataset_only:
            env = gymnasium.make(env_name, **env_kwargs)
        dataset_name = '-'.join(splits[:pos] + splits[-1:])  # Remove the words 'singletask' and 'task\d' (if exists).
        dataset_add_info = True
    elif 'oraclerep' in splits:
        # Environment with oracle goal representations.
        env_name = '-'.join(splits[:-3] + splits[-1:])  # Remove the dataset type and the word 'oraclerep'.
        if not dataset_only:
            env = gymnasium.make(env_name, use_oracle_rep=True, **env_kwargs)
        dataset_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the word 'oraclerep'.
        dataset_add_info = True
    else:
        # Original, goal-conditioned environment.
        env_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the dataset type.
        if not dataset_only:
            env = gymnasium.make(env_name, **env_kwargs)

    if env_only:
        return env

    # Load datasets.
    if dataset_path is None:
        dataset_dir = os.path.expanduser(dataset_dir)
        download_datasets([dataset_name], dataset_dir)
        train_dataset_path = os.path.join(dataset_dir, f'{dataset_name}.npz')
        val_dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val.npz')
    else:
        train_dataset_path = dataset_path
        val_dataset_path = dataset_path.replace('.npz', '-val.npz')

    ob_dtype = np.uint8 if ('visual' in env_name or 'powderworld' in env_name) else np.float32
    action_dtype = np.int32 if 'powderworld' in env_name else np.float32
    train_dataset = load_dataset(
        train_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
    )
    val_dataset = load_dataset(
        val_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
    )

    if 'singletask' in splits:
        # Add reward information to the datasets.
        relabel_dataset(env_name, env, train_dataset)
        relabel_dataset(env_name, env, val_dataset)

    if 'oraclerep' in splits:
        # Add oracle goal representations to the datasets.
        add_oracle_reps(env_name, env, train_dataset)
        add_oracle_reps(env_name, env, val_dataset)

    if not add_info:
        # Remove information keys.
        for k in ['qpos', 'qvel', 'button_states']:
            if k in train_dataset:
                del train_dataset[k]
            if k in val_dataset:
                del val_dataset[k]

    if dataset_only:
        return train_dataset, val_dataset
    else:
        return env, train_dataset, val_dataset
