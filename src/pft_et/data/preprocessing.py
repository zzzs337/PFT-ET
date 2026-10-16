from pathlib import Path
import numpy as np
from PIL import Image

def uniform_indices(length, count):
    if length < 1: raise ValueError('Empty temporal window')
    return np.rint(np.linspace(0, length-1, count)).astype(int)

def resampling_indices(timestamps, fps):
    timestamps = np.asarray(timestamps, dtype=float)
    if len(timestamps) == 0 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Source timestamps must be finite and strictly increasing')
    timestamps = timestamps - timestamps[0]
    grid = np.arange(int(np.floor(timestamps[-1]*fps+1e-8))+1) / fps
    # Causal resampling: no source frame timestamp may lie after the target time.
    indices = (np.searchsorted(timestamps, grid + 1e-9, side='right') - 1).clip(0,len(timestamps)-1)
    return grid, indices

def normalize_skeleton(pose, width, height):
    pose = np.asarray(pose,dtype=np.float32).copy()
    if pose.shape[-2:] != (17,3): raise ValueError('Expected COCO pose [...,17,3]')
    pose[...,0] /= width; pose[...,1] /= height
    if not np.isfinite(pose).all(): raise ValueError('Non-finite keypoints')
    return pose

def resize_rgb(frame, size):
    return np.asarray(Image.fromarray(frame).resize((size,size),Image.Resampling.BILINEAR))
