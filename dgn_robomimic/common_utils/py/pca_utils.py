from bc.dataset import DatasetConfig, RobomimicDataset
import torch
import numpy as np
from sklearn.decomposition import PCA
from common_utils import ibrl_utils as utils

def compute_delta_action_pca(agent, dataset: RobomimicDataset, action_chunk_size: int, n_components=10, scale='none'):
    """
    Compute PCA on delta action chunks based on the RobomimicDataset.
    Args:
        dataset (RobomimicDataset): The dataset containing demonstrations.
        action_chunk_size (int): The size of action chunks for PCA.
    Returns:
        tuple: (pca_transformed_chunks, pca_model, weighted_scaled_components)
            - pca_transformed_chunks: Numpy array of shape (num_chunks, num_pca_components)
            - pca_model: The fitted PCA model.
            - weighted_scaled_components: PCA components weighted by the explained variance ratio and scaled.
    """
    all_delta_chunks = []
    # Process each demonstration
    for demo_idx in range(len(dataset)):
        demo = dataset[demo_idx]  # Get the demo as a list of timestep entries
        print(f"Processing demo {demo_idx + 1}/{len(dataset)}...")
        # Extract states and actions for the demonstration
        #states = [entry["state"] for entry in demo]  # Shape: (T, state_stack * state_dim)
        actions = [entry["action"] for entry in demo]  # Shape: (T, action_dim)
        #states_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to("cuda")
        actions_tensor = torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions]).to("cuda")
        # Use the current policy to predict actions for each state
        predicted_actions = []
        with torch.no_grad() and utils.eval_mode(agent):
            for entry in demo:
                entry = entry.copy()
                del entry['action']
                for k, v in entry.items():
                    entry[k] = v.cuda()
                predicted_action = agent.act(entry, eval_mode=True)
                predicted_actions.append(predicted_action.cpu().numpy())
        predicted_actions = np.vstack(predicted_actions)  # Shape: (T, action_dim)
        # Compute delta actions
        deltas = actions_tensor.cpu().numpy() - predicted_actions  # Shape: (T, action_dim)
        # Extract delta action chunks
        for i in range(0, len(deltas), action_chunk_size):
            chunk = deltas[i:i + action_chunk_size]
            if chunk.shape[0] == action_chunk_size:  # Include only full chunks
                all_delta_chunks.append(chunk.flatten())
    # Perform PCA on all collected delta chunks
    all_delta_chunks = np.array(all_delta_chunks)
    if all_delta_chunks.size > 0:
        if n_components > 0:
            pca = PCA(n_components=n_components)
        else:
            pca = PCA()
        pca_transformed_chunks = pca.fit_transform(all_delta_chunks)
        # Scale PCA components so that the largest magnitude is 1
        scaled_components = pca.components_ / np.abs(pca.components_).max(axis=1, keepdims=True)
        # Weight each component by the explained variance ratio
        if scale == 'expl_variance':
            weighted_scaled_components = scaled_components * pca.explained_variance_ratio_[:, np.newaxis]
        else:
            weighted_scaled_components = scaled_components
        print("Delta action PCA successfully computed.")
        return pca_transformed_chunks, pca, weighted_scaled_components
    else:
        print("No valid delta action chunks found for PCA.")
        return np.array([]), None, None