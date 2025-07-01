import torch
import numpy as np

def compute_random_expl_chunk_basis(action_chunk_size: int, n_components=10, delta_chunk_normalization = 'max', distr_type='uniform0_1'):

    if distr_type == 'uniform0_1':
        expl_comps = np.random.rand(n_components, 7*action_chunk_size).astype(np.float32)
    elif distr_type == 'gaussian':
        expl_comps = np.random.normal(size=(n_components, 7*action_chunk_size)).astype(np.float32)

    if delta_chunk_normalization == 'max':
        expl_comps = expl_comps / np.abs(expl_comps).max(axis=1, keepdims=True)
    elif delta_chunk_normalization == 'mean_l1':
        expl_comps = expl_comps / np.abs(expl_comps).mean(axis=1, keepdims=True)
    else:
        raise NotImplementedError("only max, mean_l1 normalization implemented right now")

    print("Random Explore Chunk Basis:", expl_comps)

    return expl_comps

def compute_random_expl_chunk_basis_QR_decomp(action_chunk_size: int, n_components=10, delta_chunk_normalization = 'max'):

    #TODO: implement QR decomp to get orthogonal basis
    pass


def get_repeat_chunk_basis(action_chunk_size: int):

    expl_comps = np.zeros((7, action_chunk_size, 7), dtype=np.float32)

    for i in range(7):
        expl_comps[i, :, i] = 1.

    expl_comps = expl_comps.reshape(7, action_chunk_size*7)

    print("Repeat Chunk Basis", expl_comps)

    return expl_comps
