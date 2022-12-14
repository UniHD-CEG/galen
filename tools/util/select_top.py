import pickle
import shutil
from pathlib import Path

import numpy as np


def _read_in_run(run_identifier, directory="./logs"):
    dir_path = Path(directory)
    episode_data_paths = dir_path.glob(f"{run_identifier}-episode-*.pickle")
    episode_dict = dict()
    for p in episode_data_paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
            episode_number = data["_episode_number"]
            episode_dict[episode_number] = data

    return episode_dict


def select_top_k(episode_dict, target_c, k=10, var=0.05, last_n_episodes=100):
    pairs = np.zeros((last_n_episodes, 3))
    idx = 0
    for episode, data in episode_dict.items():
        if episode < (len(episode_dict) - last_n_episodes):
            continue
        pairs[idx, :] = np.array(
            [episode, data["_episode_metrics"]["episode-lat-ratio"], data["_episode_metrics"]["episode-acc"]])
        idx += 1

    meeting_c_ids = np.where(np.logical_and(pairs[:, 1] >= (target_c - var), pairs[:, 1] <= (target_c + var)))

    matching_pairs = pairs[meeting_c_ids]
    sorted_ids = np.argsort(matching_pairs[:, 2])[::-1]

    top_sorted_ids = sorted_ids[:k]
    top_sorted = matching_pairs[top_sorted_ids]

    return [episode_dict[int(key)] for key in top_sorted[:, 0]], top_sorted


def select_top_k_reward(episode_dict, k=10, skip_n_episodes=100):
    if len(episode_dict) > skip_n_episodes:
        pairs = np.zeros((len(episode_dict) - skip_n_episodes, 3))
    else:
        return [], None
    idx = 0
    for episode, data in episode_dict.items():
        if episode < skip_n_episodes:
            continue
        pairs[idx, :] = np.array(
            [episode, data["_episode_metrics"]["episode-lat-ratio"], data["_episode_metrics"]["episode-mean-reward"]])
        idx += 1

    matching_pairs = pairs
    sorted_ids = np.argsort(matching_pairs[:, 2])[::-1]

    top_sorted_ids = sorted_ids[:k]
    top_sorted = matching_pairs[top_sorted_ids]

    return [episode_dict[int(key)] for key in top_sorted[:, 0]], top_sorted


def copy_to(run_identifier, top_array, src="./logs", dst="./best"):
    for ep in top_array[:, 0]:
        name = f"{run_identifier}-episode-{int(ep):03d}.pickle"
        src_path = Path(src) / name
        dst_path = Path(dst)
        dst_path.mkdir(parents=True, exist_ok=True)
        dst_path = dst_path / name

        shutil.copyfile(src_path, dst_path)


def copy_top_k_matching(run_identifier, target_c, k=10, var=0.05, last_n_episodes=100, src="./logs", dst="./best"):
    episode_dict = _read_in_run(run_identifier, src)
    _, top_array = select_top_k(episode_dict, target_c, k=k, var=var, last_n_episodes=last_n_episodes)
    if top_array is not None:
        copy_to(run_identifier, top_array, src, dst)


def copy_top_k_matching_reward(run_identifier, k=10, skip_n_episodes=100, src="./logs", dst="./best"):
    episode_dict = _read_in_run(run_identifier, src)
    _, top_array = select_top_k_reward(episode_dict, k=k, skip_n_episodes=skip_n_episodes)
    if top_array is not None:
        copy_to(run_identifier, top_array, src, dst)


def read_as_list(run_identifier, directory):
    runs_dict = _read_in_run(run_identifier, directory)
    return list(runs_dict.values())
