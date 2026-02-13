# diffusion_policy/common/sampler.py

from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer

#返回所有的可处理9帧索引，返回嵌套链表训练集13193个
@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
) -> (np.ndarray, np.ndarray):  # <--- 修改返回类型
    episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    episode_ids = list() # <--- 新增：用于存储每个序列对应的episode ID
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
            episode_ids.append(i) # <--- 新增：记录下当前序列属于第 i 个 episode

    indices_arr = np.array(indices)
    episode_ids_arr = np.array(episode_ids, dtype=np.int64) # <--- 新增
    return indices_arr, episode_ids_arr # <--- 修改返回


def get_val_mask(n_episodes, val_ratio, seed=0):
    # ... (此函数无需修改)
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = -1
    val_mask[val_idxs] = True
    return val_mask

def downsample_mask(mask, max_n, seed=0):
    # ... (此函数无需修改)
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class SequenceSampler:
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            sequence_length: int,
            pad_before: int = 0,
            pad_after: int = 0,
            keys=None,
            key_first_k=dict(),
            episode_mask: Optional[np.ndarray] = None,
    ):
        super().__init__()
        assert sequence_length >= 1
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        indices, episode_ids = np.zeros((0, 4), dtype=np.int64), np.zeros(0, dtype=np.int64) # <--- 修改
        if np.any(episode_mask):
            # create_indices 现在返回两个值
            indices, episode_ids = create_indices( # <--- 修改
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )

        self.indices = indices
        self.episode_ids = episode_ids # <--- 新增：存储 episode ID
        self.keys = list(keys)
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        # ... (此方法无需修改)，给一个idx，然后从所有的13507里面去取三个键的数据
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (self.indices[idx])
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                sample = np.full(
                    (n_data, ) + input_arr.shape[1:],
                    fill_value=np.nan,
                    dtype=input_arr.dtype,
                )
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]
                except Exception as e:
                    import pdb
                    pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length, ) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result