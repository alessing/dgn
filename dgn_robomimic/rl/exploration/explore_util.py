from bc.dataset import DatasetConfig, RobomimicDataset
import torch
import torch.nn as nn
import numpy as np
from common_utils import ibrl_utils as utils


def compute_deltas(agent, dataset: RobomimicDataset):
    """
    Compute delta action chunks based on the RobomimicDataset.
    Args:
        dataset (RobomimicDataset): The dataset containing demonstrations.
        action_chunk_size (int): The size of action chunks for PCA.
    Returns:
        numpy array of delta chunks
    """
    
    all_deltas = []
    all_states = []
    # Process each demonstration
    for demo_idx in range(len(dataset)):
        demo = dataset[demo_idx]  # Get the demo as a list of timestep entries
        print(f"Processing demo {demo_idx + 1}/{len(dataset)}...")
        # Extract states and actions for the demonstration
        states = [entry["state"] for entry in demo]  # Shape: (T, state_stack * state_dim)
        actions = [entry["action"] for entry in demo]  # Shape: (T, action_dim)
        states_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to("cuda")
        actions_tensor = torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions]).to("cuda")
        # Use the current policy to predict actions for each state
        predicted_actions = []
        with torch.no_grad() and utils.eval_mode(agent):
            for state in states_tensor:
                predicted_action = agent.act({"state": state}, eval_mode=True)
                predicted_actions.append(predicted_action.cpu().numpy())
        predicted_actions = np.vstack(predicted_actions)  # Shape: (T, action_dim)
        # Compute delta actions
        deltas = actions_tensor.cpu().numpy() - predicted_actions  # Shape: (T, action_dim)

        assert len(deltas) == len(states)
        
        all_deltas.extend(deltas)
        all_states.extend(states)

    all_deltas = np.array(all_deltas)
    all_states = torch.stack(all_states, dim=0).detach().cpu().numpy()

    assert all_deltas.shape[0] == all_states.shape[0]

    return all_states, all_deltas


def make_mlp(output_shape, hidden_size, n_layers, dropout, include_output_linear=True):

    layers = []

    for _ in range(n_layers):
        layers.extend([
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
        ])
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_size = hidden_size

    if include_output_linear:
        layers.append(nn.LazyLinear(output_shape))

    return nn.Sequential(*layers)