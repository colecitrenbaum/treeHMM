
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def generate_tree_hmm_data(
    key,
    num_timesteps=60,
    max_cells=50,
    num_states=3,
    emission_dim=2,
    division_prob=0.05,
    death_prob=0.01,
    new_root_prob=0.4
):
    """
    Generates dummy data for a Tree HMM with AR Gaussian emissions.
    Uses STRICT UNIQUE IDS: Every biological cell gets a unique column index.
    - If a cell dies, its column is never reused.
    - If a cell divides, the parent column ends, and 2 NEW columns are assigned to children.
    - This makes the arrays sparse but guarantees Column K = Cell ID K.
    """
    k1, k2, k3, k4 = jr.split(key, 4)

    # --- 1. Define Model Parameters ---
    P_std = jnp.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    P_div = jnp.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])

    AR_weights = jr.normal(k1, (num_states, emission_dim, emission_dim)) * 0.5
    AR_biases = jr.normal(k2, (num_states, emission_dim))
    Emission_cov = jnp.eye(emission_dim) * 0.1

    # Initial distribution (used for roots, including spontaneous new roots)
    initial_distribution = jnp.array([1.0, 0.0, 0.0])

    # --- 2. Initialize Arrays ---
    parent_indices = np.zeros((num_timesteps, max_cells), dtype=np.int32)
    is_division_mask = np.zeros((num_timesteps, max_cells), dtype=bool)
    is_new_root_mask = np.zeros((num_timesteps, max_cells), dtype=bool)
    active_mask = np.zeros((num_timesteps, max_cells), dtype=bool)
    true_states = np.zeros((num_timesteps, max_cells), dtype=np.int32)
    observations = np.zeros((num_timesteps, max_cells, emission_dim))
    parent_observations = np.zeros((num_timesteps, max_cells, emission_dim))

    # --- 3. Simulation Loop ---
    
    # Global counter for Unique Cell IDs (mapped to column indices)
    # Start with 1 because 0 is taken by the initial root
    next_unique_id = 1 
    
    # State tracking: mapping column_idx -> {'state': int, 'obs': array}
    # Only currently active cells are in this map
    current_active_cells = {}

    # Initialize Root at t=0 in Column 0
    current_active_cells[0] = {
        'state': 0,
        'obs': np.zeros(emission_dim)
    }
    
    # Set t=0 arrays for Root
    active_mask[0, 0] = True
    is_new_root_mask[0, 0] = True
    true_states[0, 0] = 0
    observations[0, 0] = np.zeros(emission_dim)
    # parent_indices[0, 0] is 0
    # parent_observations[0, 0] is 0

    sim_keys = jr.split(k3, num_timesteps)
    new_root_keys = jr.split(k4, num_timesteps)

    for t in range(num_timesteps - 1):
        key_t = sim_keys[t]
        key_root_t = new_root_keys[t]
        
        next_active_cells = {}
        
        # A. Process Currently Active Cells
        sorted_ids = sorted(current_active_cells.keys())
        
        for pid in sorted_ids: # pid = Parent Column ID
            cell = current_active_cells[pid]
            
            fate_key, trans_key, obs_key = jr.split(jr.fold_in(key_t, pid), 3)
            fate_roll = float(jr.uniform(fate_key))
            
            fate = "continue"
            if fate_roll < death_prob:
                fate = "die"
            elif fate_roll < death_prob + division_prob:
                fate = "divide"
            
            # Check capacity before branching
            # If we need 2 new IDs but only have 1 slot left, force death or continue
            if fate == "divide" and next_unique_id + 2 > max_cells:
                fate = "die" # Out of unique IDs
            
            # --- Execute Fate ---
            
            if fate == "die":
                pass # Parent pid is not added to next_active_cells. It ends here.
                
            elif fate == "continue":
                # Cell persists in the SAME column (pid)
                target_id = pid
                
                # Transition
                state_probs = P_std[cell['state']]
                next_state = int(jr.choice(trans_key, np.arange(num_states), p=state_probs))
                
                # AR Obs (Input is Self)
                ar_input = cell['obs']
                mean = AR_weights[next_state] @ ar_input + AR_biases[next_state]
                next_obs = mean + np.array(jr.multivariate_normal(obs_key, np.zeros(emission_dim), Emission_cov))
                
                next_active_cells[target_id] = {'state': next_state, 'obs': next_obs}
                
                # Record in Arrays (t+1)
                active_mask[t+1, target_id] = True
                parent_indices[t+1, target_id] = pid # Parent is Self
                true_states[t+1, target_id] = next_state
                observations[t+1, target_id] = next_obs
                parent_observations[t+1, target_id] = ar_input
            
            elif fate == "divide":
                # Parent pid ends. Two NEW unique IDs are born.
                child_id_1 = next_unique_id
                child_id_2 = next_unique_id + 1
                next_unique_id += 2
                
                # Transition (Both children use Division matrix)
                state_probs = P_div[cell['state']]
                
                # Create Children
                # Note: We share the division transition logic for both but sample independently
                sub_keys = jr.split(trans_key, 2)
                obs_keys = jr.split(obs_key, 2)
                
                for i, cid in enumerate([child_id_1, child_id_2]):
                    # Sample State
                    next_state = int(jr.choice(sub_keys[i], np.arange(num_states), p=state_probs))
                    
                    # AR Obs (Input is 0.0 for division)
                    ar_input = np.zeros(emission_dim)
                    mean = AR_weights[next_state] @ ar_input + AR_biases[next_state]
                    next_obs = mean + np.array(jr.multivariate_normal(obs_keys[i], np.zeros(emission_dim), Emission_cov))
                    
                    next_active_cells[cid] = {'state': next_state, 'obs': next_obs}
                    
                    # Record
                    active_mask[t+1, cid] = True
                    parent_indices[t+1, cid] = pid # Parent is 'pid'
                    is_division_mask[t+1, cid] = True
                    true_states[t+1, cid] = next_state
                    observations[t+1, cid] = next_obs
                    parent_observations[t+1, cid] = ar_input # 0.0

        # B. Spontaneous New Roots
        if next_unique_id < max_cells:
            if float(jr.uniform(key_root_t)) < new_root_prob:
                new_root_id = next_unique_id
                next_unique_id += 1
                
                k_s, k_o = jr.split(key_root_t)
                next_state = int(jr.choice(k_s, jnp.arange(num_states), p=initial_distribution))
                
                # Input 0.0
                ar_input = np.zeros(emission_dim)
                mean = AR_biases[next_state]
                next_obs = mean + np.array(jr.multivariate_normal(k_o, np.zeros(emission_dim), Emission_cov))
                
                next_active_cells[new_root_id] = {'state': next_state, 'obs': next_obs}
                
                active_mask[t+1, new_root_id] = True
                is_new_root_mask[t+1, new_root_id] = True
                true_states[t+1, new_root_id] = next_state
                observations[t+1, new_root_id] = next_obs
                parent_observations[t+1, new_root_id] = ar_input
                parent_indices[t+1, new_root_id] = 0 # Dummy

        # Advance
        current_active_cells = next_active_cells

    # --- 4. Finalize ---
    observations_jax = jnp.array(observations)
    parent_obs_jax = jnp.array(parent_observations)
    active_mask_jax = jnp.array(active_mask)

    def compute_ar_log_prob(obs_t, obs_tm1, weights, biases, cov):
        means = jnp.einsum('kij,j->ki', weights, obs_tm1) + biases
        log_probs = jax.vmap(
            lambda m: multivariate_normal.logpdf(obs_t, m, cov)
        )(means)
        return log_probs

    log_likelihoods = jax.vmap(jax.vmap(
        lambda y, y_prev: compute_ar_log_prob(y, y_prev, AR_weights, AR_biases, Emission_cov)
    ))(observations_jax, parent_obs_jax)

    log_likelihoods = log_likelihoods * active_mask_jax[:, :, None]

    print(f"Generated sparse data for {num_timesteps} steps.")
    print(f"Total Unique Cells Created: {next_unique_id}")
    
    return {
        "initial_distribution": initial_distribution,
        "transition_matrices": (P_std, P_div),
        "log_likelihoods": log_likelihoods,
        "parent_indices": jnp.array(parent_indices),
        "is_division_mask": jnp.array(is_division_mask),
        "is_new_root_mask": jnp.array(is_new_root_mask),
        "active_mask": active_mask_jax,
        "observations": observations_jax,
        "parent_observations": parent_obs_jax,
        "params": {"AR_weights": AR_weights, "AR_biases": AR_biases, "cov": Emission_cov}
    }


def visualize_lineage(data):
    """
    Visualizes the tree lineage vertically.
    Y-axis: Time (starts at 0 at the top, goes down).
    X-axis: Unique Cell ID.
    """
    parent_indices = np.array(data['parent_indices'])
    active_mask = np.array(data['active_mask'])
    is_new_root = np.array(data['is_new_root_mask'])
    is_division = np.array(data['is_division_mask'])
    
    T, MaxCells = active_mask.shape
    
    plt.figure(figsize=(12, 10))
    
    # 1. Draw Cell Tracks (Vertical Lines)
    # We iterate by ID to draw contiguous vertical lines for each unique cell
    # Limit visualization to first 64 cells
    max_cells_to_show = 64
    
    for cell_id in range(min(MaxCells, max_cells_to_show)):
        # Find time points where this cell is active
        active_times = np.where(active_mask[:, cell_id])[0]
        
        if len(active_times) > 0:
            # Draw the life of the cell vertically
            # X = Cell ID (Constant), Y = Time (Variable)
            plt.plot([cell_id] * len(active_times), active_times, 
                     marker='o', markersize=4, linewidth=3, alpha=0.7, 
                     color='skyblue', label='Cell Life' if cell_id == 0 else "")
            
            # Label the start of the track
            t_start = active_times[0]
            
            # 2. Draw Connections to Parents (Upwards)
            if t_start > 0:
                parent_id = parent_indices[t_start, cell_id]
                
                # Only draw parent connections if parent is also in the visible range
                if parent_id < max_cells_to_show:
                    # Check if this is a "New Root" (Spontaneous) or a Child
                    if is_new_root[t_start, cell_id]:
                        # Star marker for spontaneous appearance
                        plt.plot(cell_id, t_start, marker='*', color='green', markersize=12, label='New Root' if cell_id==0 else "")
                    else:
                        # Draw connection line from Parent(t-1) to Child(t)
                        # Parent is at (parent_id, t_start - 1)
                        # Child is at (cell_id, t_start)
                        plt.plot([parent_id, cell_id], [t_start - 1, t_start], 
                                 color='gray', linestyle='--', linewidth=1, alpha=0.5)
                        
                        # If this was a division, mark the birth point
                        if is_division[t_start, cell_id]:
                             plt.plot(cell_id, t_start, marker='^', color='red', markersize=8, label='Division Child' if cell_id==0 else "")

    plt.ylabel("Time Step (t=0 at Top)")
    plt.xlabel("Unique Cell ID (Column Index)")
    plt.title("Lineage Tree Visualization (Vertical Flow)")
    
    # Invert Y axis so time goes down
    plt.gca().invert_yaxis()
    
    # Grid and Ticks
    plt.grid(True, alpha=0.3)
    # Crop x-axis to show only first 64 cells
    plt.xlim(-0.5, max_cells_to_show - 0.5)
    plt.xticks(np.arange(0, max_cells_to_show, 1))
    plt.yticks(np.arange(0, T, 1))
    
    # Create a custom legend manually to avoid duplicate labels
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='skyblue', lw=3, marker='o', label='Active Cell (Vertical)'),
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Parent Link'),
        Line2D([0], [0], color='red', lw=0, marker='^', label='Born from Division'),
        Line2D([0], [0], color='green', lw=0, marker='*', label='Spontaneous Root')
    ]
    plt.legend(handles=custom_lines, loc='upper right')
    
    plt.tight_layout()
    plt.show()