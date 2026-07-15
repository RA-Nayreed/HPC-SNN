import numpy as np
import torch

def bin_events(times: np.ndarray, channels: np.ndarray, dt_ms: float=10.0, features: int=140) -> torch.Tensor:
    if len(times) != len(channels): raise ValueError("times/channels length mismatch")
    if np.any(channels < 0) or np.any(channels >= 700): raise ValueError("channel outside [0, 699]")
    steps=max(1, int(np.floor(float(times.max())*1000/dt_ms))+1) if len(times) else 1
    output=torch.zeros((steps, features), dtype=torch.float32)
    if len(times): output.index_put_((torch.from_numpy(np.floor(times*1000/dt_ms).astype(np.int64)), torch.from_numpy((channels//5).astype(np.int64))), torch.ones(len(times)), accumulate=True)
    return output

def pad_batch(samples):
    sequences, labels=zip(*samples); lengths=torch.tensor([x.shape[0] for x in sequences], dtype=torch.long)
    output=torch.zeros((len(samples), int(lengths.max()), sequences[0].shape[1]), dtype=torch.float32)
    for i, sequence in enumerate(sequences): output[i,:sequence.shape[0]]=sequence
    return output, torch.tensor(labels, dtype=torch.long), lengths
