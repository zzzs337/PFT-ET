"""Checkpoint input/model identity; paths may relocate if content fingerprints match."""


def check_checkpoint_config(cfg,saved):
    for key in ('fps','observation_frames','future_seconds','sampled_frames','image_size','rgb_mean','rgb_std','bin_edges'):
        if cfg['data'][key]!=saved['data'][key]:raise ValueError('Checkpoint input/preprocessing mismatch: '+key)
    for key in ('rgb_backend','dim','skeleton_layers','skeleton_heads','skeleton_ffn','head_hidden','mixtures','strategy','use_ema','use_auxiliary','sigma_min','history_windows','history_stride_frames'):
        if cfg['model'][key]!=saved['model'][key]:raise ValueError('Checkpoint model mismatch: '+key)
    # N_MC and marginalization are explicit inference settings and may vary.
