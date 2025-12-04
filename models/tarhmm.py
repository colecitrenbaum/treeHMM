"""
Tree Autoregressive Hidden Markov Model (tARHMM). 
We have data (T, C, D) where T is the number of timesteps, C is the number of cells, and D is the dimension of the data.
At each time point, an active cell can either divide into two daughter cells, or it can stay. 
It's a typical Gaussian AR HMM, except we have a new transition matrix for division events. 
"""
from typing import NamedTuple, Optional, Tuple, Union, Callable
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jit, vmap
from jax.tree_util import tree_map
from jaxtyping import Int, Float, Array, PyTree
from functools import partial
from dynamax.utils.utils import ensure_array_has_batch_dim
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.hidden_markov_model.models.linreg_hmm import LinearRegressionHMMEmissions, ParamsLinearRegressionHMMEmissions
from dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMMEmissions
from dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMM
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar, IntScalar
from typing import Any, Optional, Tuple, runtime_checkable, Union, Dict
from jaxtyping import Real
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.hidden_markov_model.inference import *
from dynamax.hidden_markov_model.inference import _condition_on
from dynamax.utils.utils import pytree_slice
from fastprogress.fastprogress import progress_bar

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


## NOTE turning off jit for debugging only
import jax.debug as dbg
from jax import config
config.update("jax_disable_jit", True)
class TreeHMMPosterior(NamedTuple):
    r"""Simple wrapper for properties of an HMM posterior distribution.

    Transition probabilities may be either 2D or 3D depending on whether the
    transition matrix is fixed or time-varying.

    :param marginal_loglik: $p(y_{1:T} \mid \theta) = \log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.
    :param filtered_probs: $p(z_t \mid y_{1:t}, \theta)$ for $t=1,\ldots,T$
    :param predicted_probs: $p(z_t \mid y_{1:t-1}, \theta)$ for $t=1,\ldots,T$
    :param smoothed_probs: $p(z_t \mid y_{1:T}, \theta)$ for $t=1,\ldots,T$
    :param initial_probs: $p(z_1 \mid y_{1:T}, \theta)$ (also present in `smoothed_probs` but here for convenience)
    :param trans_probs: $p(z_t, z_{t+1} \mid y_{1:T}, \theta)$ for $t=1,\ldots,T-1$. (If the transition matrix is fixed, these probabilities may be summed over $t$. See note above.)
    """
    marginal_loglik: Scalar
    filtered_probs: Float[Array, "num_timesteps num_states"]
    predicted_probs: Float[Array, "num_timesteps num_states"]
    smoothed_probs: Float[Array, "num_timesteps num_states"]
    initial_probs: Float[Array, " num_states"]
    trans_probs: Optional[Union[Float[Array, "num_states num_states"],
                                Float[Array, "num_timesteps_minus_1 num_states num_states"]]] = None
    division_trans_probs: Optional[Union[Float[Array, "num_states num_states"],
                                Float[Array, "num_timesteps_minus_1 num_states num_states"]]] = None


class HMMDivisionTransitions(StandardHMMTransitions):
    def collect_suff_stats(
            self,
            params,
            posterior: HMMPosterior,
            inputs=None
    ) -> Union[Float[Array, "num_states num_states"],
               Float[Array, "num_timesteps_minus_1 num_states num_states"]]:
        """Collect the sufficient statistics for the model."""
        return posterior.division_trans_probs
    def m_step(
            self,
            params: ParamsStandardHMMTransitions,
            props: ParamsStandardHMMTransitions,
            batch_stats: Float[Array, "batch num_states num_states"],
            m_step_state: Any
        ) -> Tuple[ParamsStandardHMMTransitions, Any]:
        """Perform the M-step of the EM algorithm."""
        if props.transition_matrix.trainable:
            if self.num_states == 1:
                transition_matrix = jnp.array([[1.0]])
            else:
                expected_trans_counts = batch_stats
                transition_matrix = tfd.Dirichlet(self.concentration + expected_trans_counts).mode()
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state
class TreeTransitions(StandardHMMTransitions):
    def m_step(
            self,
            params: ParamsStandardHMMTransitions,
            props: ParamsStandardHMMTransitions,
            batch_stats: Float[Array, "batch num_states num_states"],
            m_step_state: Any
        ) -> Tuple[ParamsStandardHMMTransitions, Any]:
        """Perform the M-step of the EM algorithm."""
        if props.transition_matrix.trainable:
            if self.num_states == 1:
                transition_matrix = jnp.array([[1.0]])
            else:
                expected_trans_counts = batch_stats
                transition_matrix = tfd.Dirichlet(self.concentration + expected_trans_counts).mode()
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state
class TreeInitialState(StandardHMMInitialState):
    def m_step(
            self,
            params: ParamsStandardHMMInitialState,
            props: ParamsStandardHMMInitialState,
            batch_stats: Float[Array, "batch num_states"],
            m_step_state: Any
    ) -> Tuple[ParamsStandardHMMInitialState, Any]:
        """Perform the M-step of the EM algorithm."""
        if props.probs.trainable:
            if self.num_states == 1:
                probs = jnp.array([1.0])
            else:
                expected_initial_counts = batch_stats.sum(axis=0)
                probs = tfd.Dirichlet(self.initial_probs_concentration + expected_initial_counts).mode()
            params = params._replace(probs=probs)
        return params, m_step_state


@partial(jit, static_argnames=["transition_fn", "division_transition_fn"])
def tree_hmm_filter(
    initial_distribution,   # (num_states,)
    transition_matrices,    # tuple of (P_std, P_div) each (num_states, num_states)
    log_likelihoods,        # (T, MAX_CELLS, K)
    parent_indices,         # (T, MAX_CELLS)
    is_division_mask,       # (T, MAX_CELLS)
    active_mask,            # (T, MAX_CELLS)
    is_new_root_mask,       # (T, MAX_CELLS)
    transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None,
    division_transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None
):
    '''
    Compute filtering and predictive distributions, ie p(z_t | x_1:t) and p(z_t+1 | x_1:t)
    '''

    num_timesteps, max_cells, num_states = log_likelihoods.shape
    P_std, P_div = transition_matrices

    # Initial state 
    init_probs_template = jnp.tile(initial_distribution, (max_cells, 1)) 

    # --- Init Carry ---
    # make this all zeros - we will swap in the init_probs_template anywhere there's a new root, including t=0

    init_carry = (0.0, jnp.zeros((max_cells, num_states)))

    def _step(carry, t):
        '''
        Note here that I'm using globals (is_division_mask, etc) instead of passing as inputs to scan. I think this is easier since we need both present and future values. 
        '''
        log_normalizer, predicted_probs = carry
        ## predicted_probs[new root] = init_probs 
        is_root = is_new_root_mask[t] # (MAX_CELLS,)
        predicted_probs = jnp.where(
            is_root[:, None], 
            init_probs_template, 
            predicted_probs 
        ) #(num_cells, num_states)

        # Inputs for time t
        ll = log_likelihoods[t]
        is_active = active_mask[t]
        
        # vmap the _condition_on to apply it per-cell
        filtered_probs, log_norm_t = vmap(_condition_on)(predicted_probs, ll)
        # mask only the active ones 
        filtered_probs = filtered_probs * is_active[:, None]
        
        log_normalizer += jnp.sum(log_norm_t * is_active)

        # alpha_{t+1} predictive distribution computation 
        def perform_transition():
            parents_map = parent_indices[t+1]      
            div_mask = is_division_mask[t+1]       
            next_active = active_mask[t+1]         
            
            parents_filtered = filtered_probs[parents_map]
            
            def apply_trans(parent_vec, is_d):
                '''
                apply the transition matrix depending on if it is a division or not
                '''
                A_T = lax.cond(is_d, lambda _: P_div.T, lambda _: P_std.T, None)
                return A_T @ parent_vec

            preds_next = vmap(apply_trans)(parents_filtered, div_mask)
            preds_next = preds_next * next_active[:, None]
            return preds_next


        ##NOTE : for new roots, parent_vec will be zeros, so the below gives a zero. We handle new roots at the top of the function.
        predicted_probs_next = lax.cond(
            t == num_timesteps - 1,
            lambda: jnp.zeros_like(predicted_probs),
            perform_transition
        )

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    (log_normalizer, _), (filtered_probs, predicted_probs) = lax.scan(
        _step, init_carry, jnp.arange(num_timesteps)
    )

    post = HMMPosteriorFiltered(marginal_loglik=log_normalizer,
                                filtered_probs=filtered_probs,
                                predicted_probs=predicted_probs)
    return post

@partial(jit, static_argnames=["transition_fn"])
def tree_hmm_backward_filter(
    transition_matrices,    # tuple of (P_std, P_div) each (num_states, num_states)
    log_likelihoods,        # (T, MAX_CELLS, K)
    parent_indices,         # (T, MAX_CELLS)
    is_division_mask,       # (T, MAX_CELLS)
    active_mask,            # (T, MAX_CELLS)
    is_new_root_mask,       # (T, MAX_CELLS)
    transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None,
    division_transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None
) -> Tuple[Scalar, Float[Array, "num_timesteps num_states"]]:
    r"""Run the filter backwards in time. This is the second step of the forward-backward algorithm.

Note that is_death is 1 for  the LAST timestep that the cell is alive-- ie the backwards analogue of is_new_root_mask.

#NOTE only implement at first for stationary transition matrices

    """
    num_timesteps, max_cells, num_states = log_likelihoods.shape
    P_std, P_div = transition_matrices
    init_betas = jnp.ones((max_cells, num_states))
    init_carry = (0.0, init_betas) # log_normalizer, betas_future
    def _step(carry, t):
        """Backward filtering step.
        Note here that we're going backwards in time. We start at time T-1. 
        Note that we implement with diff logic than for backward filter in regular HMM since it's easier to aggregate children messages like this. 
        At each "t" in the step, we collect information from the future (t+1). 
        """
        log_normalizer, betas_future = carry
        ll_future = log_likelihoods[t+1]
        parents_map = parent_indices[t+1] # what are the parents of cells at t+1 ? 
        div_mask = is_division_mask[t+1]      # is the cell at t+1 the child of a division event 
        active_future = active_mask[t+1] 
        is_new_root_future = is_new_root_mask[t+1]

        backward_filtered_probs, log_norm_t = vmap(_condition_on)(betas_future, ll_future)
        backward_filtered_probs = backward_filtered_probs * active_future[:, None]
        log_normalizer += jnp.sum(log_norm_t * active_future)

        def backwards_transition(child_vec, is_d):
            A = lax.cond(is_d, lambda _: P_div, lambda _: P_std, None)
            return A @ child_vec
        msgs_to_parent = vmap(backwards_transition)(backward_filtered_probs, div_mask) # these are all the messages from t+1 children backwards to t
        # below is for safety-- if we have parent = 0 for inactive cells by default, then we don't multiply by 0...
        # also if there's a new root, and the parent_id is 0 by default, we don't want to propogate the message to cell 0, since it's not a division. 
        msgs_to_parent = jnp.where(active_future[:, None] & ~is_new_root_future[:, None], msgs_to_parent, 1.0) 
        betas_current = jnp.ones((max_cells, num_states))
        betas_current = betas_current.at[parents_map].multiply(msgs_to_parent)

        return (log_normalizer, betas_current), betas_current
    
    scan_range = jnp.arange(num_timesteps - 2, -1, -1)
    (log_normalizer, _), backward_history = lax.scan(_step, init_carry, scan_range)
    ## reverse to get 0 -> T-2 , append initial betas for T-1 state. 
    backward_history = backward_history[::-1]
    backward_history = jnp.concatenate([backward_history,init_betas[None,...]],axis=0)
        
    return log_normalizer, backward_history

@partial(jit, static_argnames=["transition_fn"])
def tree_hmm_two_filter_smoother(
    initial_distribution,   # (num_states,)
    transition_matrices,    # tuple of (P_std, P_div) each (num_states, num_states)
    log_likelihoods,        # (T, MAX_CELLS, K)
    parent_indices,         # (T, MAX_CELLS)
    is_division_mask,       # (T, MAX_CELLS)
    active_mask,            # (T, MAX_CELLS)
    is_new_root_mask,       # (T, MAX_CELLS)
    transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None,
    division_transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None,
    compute_trans_probs: bool = True
) -> TreeHMMPosterior:
    r"""
    """
    post = tree_hmm_filter(initial_distribution, transition_matrices, log_likelihoods, parent_indices, is_division_mask, active_mask, is_new_root_mask, transition_fn, division_transition_fn)
    ll = post.marginal_loglik   
    filtered_probs, predicted_probs = post.filtered_probs, post.predicted_probs

    _, backward_pred_probs = tree_hmm_backward_filter(transition_matrices, log_likelihoods, parent_indices, is_division_mask, active_mask, is_new_root_mask, transition_fn, division_transition_fn)

    # Compute smoothed probabilities
    smoothed_probs = filtered_probs * backward_pred_probs
    norm = smoothed_probs.sum(axis=2, keepdims=True)
    smoothed_probs /= norm




    # compute initial probs
    first_root_times = jnp.argmax(is_new_root_mask,axis=0)
    # find the first time each root appears , then get smoothed_probs for that cell at that time.
    # note that this will give a NaN for cells that are not roots , so you'll nansum in the m step 
    initial_probs = jnp.take_along_axis(smoothed_probs,first_root_times[None,:,None],axis=0)[0]
    posterior = TreeHMMPosterior(
        marginal_loglik=ll,
        filtered_probs=filtered_probs,
        predicted_probs=predicted_probs,
        smoothed_probs=smoothed_probs,
        initial_probs=initial_probs # ie p(z_1 | x_1:T) in the case that all roots start at 0. 
    )

    # Compute the transition probabilities if specified
    if compute_trans_probs:
        trans_probs = compute_transition_probs(transition_matrices, posterior, parent_indices, is_division_mask, active_mask, is_new_root_mask)
        posterior = posterior._replace(trans_probs=trans_probs[0], division_trans_probs=trans_probs[1])

    return posterior

def _compute_sum_transition_probs(
    transition_matrices, # (P_std, P_div)
    hmm_posterior,
    parent_indices,
    is_division_mask,
    active_mask,
    is_new_root_mask,
) -> Tuple[Float[Array, "num_states num_states"], Float[Array, "num_states num_states"]]:
    """Compute the transition probabilities from the HMM posterior messages.

    Args:
        transition_matrix (_type_): _description_
        hmm_posterior (_type_): _description_
    """
    #TODO 
    def _step(carry, args: Tuple[Array, Array, Array, Int[Array, ""]]):
        """Compute the sum of transition probabilities."""
        sum_transition_probs, sum_division_transition_probs = carry
        filtered_probs, smoothed_probs_next, predicted_probs_next, parents_map, div_mask, next_active_mask, next_is_new_root_mask, t = args
        #TODO am i handling the active mask correctly? i think it's getting zerod out bc the posterior is getting 0'd correctly
        # Get parameters for time t
        # A = _get_params(transition_matrix, 2, t)

        A_std = transition_matrices[0]
        A_div = transition_matrices[1]
        # Compute smoothed transition probabilities (Eq. 8.4 of Saarka, 2013)
        # If hard 0. in predicted_probs_next, set relative_probs_next as 0. to avoid NaN values
        valid_transition_mask = next_active_mask & (~next_is_new_root_mask)
        relative_probs_next = jnp.where(jnp.isclose(predicted_probs_next, 0.0), 0.0,
                                        smoothed_probs_next / predicted_probs_next)
        parent_filtered_probs = filtered_probs[parents_map]
        smoothed_trans_probs = parent_filtered_probs[:, :,None] * A_std * relative_probs_next[:,None, :]
        smoothed_trans_probs *= (~div_mask[:,None,None]) & valid_transition_mask[:,None,None]
        smoothed_trans_probs /= smoothed_trans_probs.sum(axis=[1,2],keepdims=True)

        smoothed_division_trans_probs = parent_filtered_probs[:, :,None] * A_div * relative_probs_next[:,None ,:]
        smoothed_division_trans_probs *= div_mask[:,None,None] & valid_transition_mask[:,None,None]
        smoothed_division_trans_probs /= smoothed_division_trans_probs.sum(axis=[1,2],keepdims=True)

        ## 
        sum_transition_probs += jnp.nansum(smoothed_trans_probs,axis=0)
        sum_division_transition_probs += jnp.nansum(smoothed_division_trans_probs,axis=0)
        return (sum_transition_probs, sum_division_transition_probs), None

    # Initialize the recursion
    num_states = transition_matrices[0].shape[-1]
    num_timesteps = len(hmm_posterior.filtered_probs)
    sum_transition_probs, sum_division_transition_probs = lax.scan(
        _step,
        (jnp.zeros((num_states, num_states)), jnp.zeros((num_states, num_states))),
        (
            hmm_posterior.filtered_probs[:-1],
            hmm_posterior.smoothed_probs[1:],
            hmm_posterior.predicted_probs[1:],
            parent_indices[1:],
            is_division_mask[1:],
            active_mask[1:],
            is_new_root_mask[1:],
            jnp.arange(num_timesteps - 1),
        ),
    )
    return sum_transition_probs, sum_division_transition_probs

def compute_transition_probs(
    transition_matrices,
    hmm_posterior: TreeHMMPosterior,
    parent_indices,
    is_division_mask,
    active_mask,
    is_new_root_mask,
    transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None,
    division_transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None
) -> Union[Float[Array, "num_states num_states"],
           Float[Array, "num_timesteps_minus_1 num_states num_states"]]:
    r"""Compute the posterior marginal distributions $p(z_{t+1}, z_t \mid y_{1:T}, u_{1:T}, \theta)$.

    Args:
        transition_matrix: the (possibly time-varying) transition matrix
        hmm_posterior: Output of `hmm_smoother` or `hmm_two_filter_smoother`
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        array of smoothed transition probabilities.
    """
    if transition_matrices is None:
        raise ValueError("`transition_matrices` must be specified.")

    return _compute_sum_transition_probs(transition_matrices, hmm_posterior,parent_indices,is_division_mask,active_mask,is_new_root_mask)

class ParamsLinearTreeARHMM(NamedTuple):
    """Model parameters for a Tree Autoregressive HMM."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    division_transitions: ParamsStandardHMMTransitions
    emissions: ParamsLinearRegressionHMMEmissions




class TreeARHMMEmissions(LinearAutoregressiveHMMEmissions):
    r"""Emissions for a Tree Autoregressive HMM.
    
    Inherits everything from LinearAutoregressiveHMMEmissions, but just requires some flattening. 
    """

    def _compute_conditional_logliks(self, params, emissions, inputs):
        """
        Compute log likelihoods, handling potentially 3D inputs (Time, Cells, Dim).
        """
        # Case 1: If just one cell.. 
        if emissions.ndim == 2:
            return super()._compute_conditional_logliks(params, emissions, inputs)
            
        # Case 2: Tree 3D Input (Time, Cells, Dim)
        # we flatten T and Cells into a single 'Batch' dimension
        T, C, D = emissions.shape
        flat_emissions = emissions.reshape(-1, D)
        
        flat_inputs = None
        if inputs is not None:
            # inputs shape is (T, C, Input_Dim)
            flat_inputs = inputs.reshape(-1, inputs.shape[-1])
            
        # Compute using parent's vectorized logic on the flattened batch
        flat_log_probs = super()._compute_conditional_logliks(params, flat_emissions, flat_inputs)
        
        # Reshape back to (T, C, K)
        return flat_log_probs.reshape(T, C, -1)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances: Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions: Optional[Union[Float[Array, "num_timesteps emission_dim"], 
                                             Float[Array, "num_timesteps max_cells emission_dim"]]]=None
                   ) -> Tuple[ParamsLinearRegressionHMMEmissions, ParamsLinearRegressionHMMEmissions]:
        r"""Initialize parameters, handling 3D tree data."""

        # Handle K-Means Pre-processing (Flatten & Filter)
        # If method is "prior", we pass emissions=None or ignored, so this logic is skipped safely.
        if method.lower() == "kmeans" and emissions is not None and emissions.ndim == 3:
            # Flatten: (T, Cells, D) -> (N, D)
            flat_emissions = emissions.reshape(-1, self.emission_dim)
            
            # Filter: Remove zero-padded (unborn) cells to avoid clustering them
            # Heuristic: If norm is effectively 0, it's padding.

            ##NOTE this is only used for filtering out zero padded cells, probably not the best way to do this.
            norms = jnp.linalg.norm(flat_emissions, axis=1)
            valid_mask = norms > 1e-6
            
            if jnp.sum(valid_mask) > self.num_states:
                emissions = flat_emissions[valid_mask]
            else:
                emissions = flat_emissions

        # Initialize all the AR weights from the super. 
        return super().initialize(
            key=key, 
            method=method, 
            emission_weights=emission_weights,
            emission_biases=emission_biases,
            emission_covariances=emission_covariances,
            emissions=emissions
        )
    def collect_suff_stats(
            self,
            params: ParamsLinearTreeARHMM,
            posterior: HMMPosterior,
            emissions: Float[Array, "num_timesteps emission_dim"],
            inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None
    ) -> Dict[str, Float[Array, "..."]]:
        """Collect sufficient statistics for the emission parameters."""
        expected_states = jnp.nan_to_num(posterior.smoothed_probs) # zeroes out nan values which is safe for summing here
        sum_w = jnp.einsum("tnk->k", expected_states)
        sum_x = jnp.einsum("tnk,tni->ki", expected_states, inputs)
        sum_y = jnp.einsum("tnk,tni->ki", expected_states, emissions)
        sum_xxT = jnp.einsum("tnk,tni,tnj->kij", expected_states, inputs, inputs)
        sum_xyT = jnp.einsum("tnk,tni,tnj->kij", expected_states, inputs, emissions)
        sum_yyT = jnp.einsum("tnk,tni,tnj->kij", expected_states, emissions, emissions)
        return dict(sum_w=sum_w, sum_x=sum_x, sum_y=sum_y, sum_xxT=sum_xxT, sum_xyT=sum_xyT, sum_yyT=sum_yyT)
    # def m_step(
    #         self,
    #         params: ParamsLinearTreeARHMM,
    #         props: ParamsLinearRegressionHMMEmissions,
    #         batch_stats: Dict[str, Float[Array, "..."]],
    #         m_step_state: Any
    # ) -> Tuple[ParamsLinearTreeARHMM, Any]:
    #     """Perform the M-step of the EM algorithm."""
        
    #     def _single_m_step(stats):
    #         """Perform the M-step for a single state."""
    #         sum_w = stats['sum_w']
    #         sum_x = stats['sum_x']
    #         sum_y = stats['sum_y']
    #         sum_xxT = stats['sum_xxT']
    #         sum_xyT = stats['sum_xyT']
    #         sum_yyT = stats['sum_yyT']

    #         # Make block matrices for stacking features (x) and bias (1)
    #         sum_x1x1T = jnp.block(
    #             [[sum_xxT,                   jnp.expand_dims(sum_x, 1)],
    #              [jnp.expand_dims(sum_x, 0), jnp.expand_dims(sum_w, (0, 1))]]
    #         )
    #         sum_x1yT = jnp.vstack([sum_xyT, sum_y])

    #         # Solve for the optimal A, b, and Sigma
    #         Ab = jnp.linalg.solve(sum_x1x1T, sum_x1yT).T
    #         Sigma = 1 / sum_w * (sum_yyT - Ab @ sum_x1yT)
    #         Sigma = 0.5 * (Sigma + Sigma.T)                 # for numerical stability
    #         return Ab[:, :-1], Ab[:, -1], Sigma

    #     # emission_stats = pytree_sum(batch_stats, axis=0)
    #     As, bs, Sigmas = vmap(_single_m_step)(batch_stats)
    #     params = params._replace(weights=As, biases=bs, covs=Sigmas)
    #     return params, m_step_state
    def m_step(
                self,
                params: ParamsLinearTreeARHMM,
                props: ParamsLinearRegressionHMMEmissions,
                batch_stats: Dict[str, Float[Array, "..."]],
                m_step_state: Any
        ) -> Tuple[ParamsLinearTreeARHMM, Any]:
            """Perform the M-step of the EM algorithm."""
            
            def _single_m_step(stats):
                """Perform the M-step for a single state."""
                sum_w = stats['sum_w']
                sum_x = stats['sum_x']
                sum_y = stats['sum_y']
                sum_xxT = stats['sum_xxT']
                sum_xyT = stats['sum_xyT']
                sum_yyT = stats['sum_yyT']

                # --- STABILITY FIX 1: Check for dead states ---
                # If a state has effectively 0 observations, don't update (return identity/zeros)
                # or return the prior (if you had one). Here we return zeros/identity to prevent NaNs.
                # In a real implementation, you might want to return the *old* params.
                is_valid_state = sum_w > 1e-4

                # Make block matrices for stacking features (x) and bias (1)
                sum_x1x1T = jnp.block(
                    [[sum_xxT,                   jnp.expand_dims(sum_x, 1)],
                    [jnp.expand_dims(sum_x, 0), jnp.expand_dims(sum_w, (0, 1))]]
                )
                
                # --- STABILITY FIX 2: Ridge Regularization ---
                # Add a small constant to the diagonal to ensure invertibility
                ridge = 1e-4
                diag_indices = jnp.arange(sum_x1x1T.shape[0])
                sum_x1x1T = sum_x1x1T.at[diag_indices, diag_indices].add(ridge)

                sum_x1yT = jnp.vstack([sum_xyT, sum_y])

                # Solve for the optimal A, b
                Ab = jnp.linalg.solve(sum_x1x1T, sum_x1yT).T
                
                # --- STABILITY FIX 3: Safe Sigma calculation ---
                # Use sum_w + epsilon to avoid div by zero
                Sigma = (sum_yyT - Ab @ sum_x1yT) / (sum_w + 1e-10)
                
                # Enforce symmetry and PSD
                Sigma = 0.5 * (Sigma + Sigma.T) 
                # Optional: Add small jitter to diagonal of Sigma too
                Sigma = Sigma + 1e-6 * jnp.eye(Sigma.shape[0])

                return Ab[:, :-1], Ab[:, -1], Sigma

            # NOTE: Using the fix from our previous conversation (no pytree_sum if already summed)
            # batch_stats should already be aggregated over the batch
            As, bs, Sigmas = vmap(_single_m_step)(batch_stats)
            
            # NOTE: You might want to carry over old parameters if the state died (sum_w approx 0)
            # This requires passing 'params' into _single_m_step to select between new and old.
            
            params = params._replace(weights=As, biases=bs, covs=Sigmas)
            return params, m_step_state

class tARHMM(LinearAutoregressiveHMM):
    def __init__(self,
                num_states: int,
                emission_dim: int,
                num_lags: int=1,
                initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                transition_matrix_stickiness: Scalar=0.0):
        super().__init__(num_states, emission_dim, num_lags, initial_probs_concentration, transition_matrix_concentration, transition_matrix_stickiness)
        self.initial_component = TreeInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        self.emission_component = TreeARHMMEmissions(num_states, emission_dim, num_lags=num_lags)
        self.transition_component = TreeTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        self.division_transition_component = HMMDivisionTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        # The inference functions all need the same arguments
    def _inference_args(self, params: HMMParameterSet, 
                        emissions: Array, 
                        inputs: Optional[Array],
                        parent_indices: Array,
                        is_division_mask: Array,
                        active_mask: Array,
                        is_new_root_mask: Array) -> Tuple:
        """Return the arguments needed for inference."""
        return (self.initial_component._compute_initial_probs(params.initial, inputs), # initial distribution
                (self.transition_component._compute_transition_matrices(params.transitions, inputs), self.division_transition_component._compute_transition_matrices(params.division_transitions, inputs)), # transition matrix
                self.emission_component._compute_conditional_logliks(params.emissions, emissions, inputs),# emission log likelihoods
                parent_indices, # parent indices
                is_division_mask, # is division mask
                active_mask, # active mask
                is_new_root_mask # is new root mask
                
                ) 
    def initialize(self,
                key: Array=jr.PRNGKey(0),
                method: str="prior",
                initial_probs: Optional[Float[Array, " num_states"]]=None,
                transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                division_transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                emission_weights: Optional[Float[Array, "num_states emission_dim emission_dim_times_num_lags"]]=None,
                emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                emissions:  Optional[Float[Array, "num_timesteps num_cells emission_dim"]]=None
    ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        key1, key2, key3, key4 = jr.split(key , 4)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_covariances=emission_covariances, emissions=emissions)
        params["division_transitions"], props["division_transitions"] = self.division_transition_component.initialize(key4, method=method, transition_matrix=division_transition_matrix)
        return ParamsLinearTreeARHMM(**params), ParamsLinearTreeARHMM(**props)

    def compute_inputs(self,
                       emissions: Float[Array, "num_timesteps max_cells emission_dim"],
                       parent_indices: Int[Array, "num_timesteps max_cells"],
                       is_division_mask: Array,
                       is_new_root_mask: Array,
                       active_mask: Array,
    ) -> Float[Array, "num_timesteps max_cells {num_lags}*{emission_dim}"]:
        r"""Compute the matrix of lagged emissions for a Tree HMM.
        
        Logic:
        1. Gather parent observation from t-1.
        2. For active cells, set input to 0 if:
           - No parent (eg t=0, new root)
           - Cell is a Division -- no autoregression from parents -> daughter
        """
        num_timesteps, max_cells, _ = emissions.shape
        
        if self.num_lags > 1:
            raise NotImplementedError("Tree ARHMM currently supports num_lags=1 only.")

        def get_inputs_for_t(t, p_indices, is_div, is_root, is_active):
            # 1. Gather Parent Obs from t-1
            def gather_prev():
                prev_obs = emissions[t-1] 
                return prev_obs[p_indices]

            # At t=0, there is no t-1. Return zeros.
            gathered_obs = lax.cond(
                t == 0,
                lambda: jnp.zeros((max_cells, self.emission_dim)),
                gather_prev
            )
            
            # 2. Apply Masks
            # Input is 0 if: Division OR New Root OR Inactive
            should_zero_out = is_div | is_root | (~is_active)
            final_input = jnp.where(should_zero_out[:, None], 0.0, gathered_obs)
            return final_input

        inputs = vmap(get_inputs_for_t)(
            jnp.arange(num_timesteps), 
            parent_indices, 
            is_division_mask, 
            is_new_root_mask, 
            active_mask
        )
        return inputs
    #TODO some day 
    def sample(self,
               params: HMMParameterSet,
               key: Array,
               num_timesteps: int,
               prev_emissions: Optional[Float[Array, "num_lags emission_dim"]]=None
    ) -> Tuple[Int[Array, " num_timesteps"], Float[Array, "num_timesteps emission_dim"]]:
        raise NotImplementedError("Sampling from the tARHMM is not implemented yet.")



    # Expectation-maximization (EM) code
    def e_step(
            self,
            params: HMMParameterSet,
            emissions: Array,
            inputs: Array,
            parent_indices: Array,
            is_division_mask: Array,
            active_mask: Array,
            is_new_root_mask: Array
            ) -> Tuple[PyTree, Scalar]:
        """
        """
        args = self._inference_args(params, emissions, inputs, parent_indices, is_division_mask, active_mask, is_new_root_mask)
        posterior = tree_hmm_two_filter_smoother(*args)

        # initial_stats = self.initial_component.collect_suff_stats(params.initial, posterior, inputs)
        #TODO ?? check the collect_suff_stats stuff 
        # initial_stats = self.initial_component.collect_suff_stats(params.initial, posterior, inputs)
        initial_stats = posterior.initial_probs
        transition_stats = self.transition_component.collect_suff_stats(params.transitions, posterior, inputs)
        division_transition_stats = self.division_transition_component.collect_suff_stats(params.division_transitions, posterior, inputs)
        emission_stats = self.emission_component.collect_suff_stats(params.emissions, posterior, emissions, inputs)
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik
    def m_step(
            self,
            params: HMMParameterSet,
            props: HMMPropertySet,
            batch_stats: PyTree,
            m_step_state: Any
            ) -> Tuple[HMMParameterSet, Any]:
        """
        """
        batch_initial_stats, batch_transitions_stats, batch_emission_stats = batch_stats
        batch_transition_stats, batch_division_transition_stats = batch_transitions_stats
        initial_m_step_state, transitions_m_step_state, division_transitions_m_step_state, emissions_m_step_state = m_step_state

        initial_params, initial_m_step_state = self.initial_component.m_step(params.initial, props.initial, jnp.nan_to_num(batch_initial_stats), initial_m_step_state)
        transition_params, transitions_m_step_state = self.transition_component.m_step(params.transitions, props.transitions, batch_transition_stats, transitions_m_step_state)
        division_transition_params, division_transitions_m_step_state = self.division_transition_component.m_step(params.division_transitions, props.division_transitions, batch_division_transition_stats, division_transitions_m_step_state)
        emission_params, emissions_m_step_state = self.emission_component.m_step(params.emissions, props.emissions, batch_emission_stats, emissions_m_step_state)
        params = params._replace(initial=initial_params, transitions=transition_params, division_transitions=division_transition_params, emissions=emission_params)
        m_step_state = initial_m_step_state, transitions_m_step_state, division_transitions_m_step_state, emissions_m_step_state
        return params, m_step_state

    def fit_em(
        self,
        params,
        props,
        emissions,
        inputs,
        parent_indices,
        is_division_mask,
        active_mask,
        is_new_root_mask,
        num_iters: int=50,
        verbose: bool=True
    ) -> Tuple[HMMParameterSet, Float[Array, " num_iters"]]:
        r"""Compute parameter MLE/ MAP estimate using Expectation-Maximization (EM).

        EM aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        It does so by iteratively forming a lower bound (the "E-step") and then maximizing it (the "M-step").

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of emissions
            inputs: one or more sequences of corresponding inputs
            num_iters: number of iterations of EM to run
            verbose: whether or not to show a progress bar

        Returns:
            tuple of new parameters and log likelihoods over the course of EM iterations.

        """

        # Make sure the emissions and inputs have batch dimensions
        #NOTE remove this ? 
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)
        # batch_parent_indices = ensure_array_has_batch_dim(parent_indices, parent_indices.shape)
        # batch_is_division_mask = ensure_array_has_batch_dim(is_division_mask, is_division_mask.shape)
        # batch_active_mask = ensure_array_has_batch_dim(active_mask, active_mask.shape)
        # batch_is_new_root_mask = ensure_array_has_batch_dim(is_new_root_mask, is_new_root_mask.shape)
        @jit
        def em_step(params, m_step_state):
            """Perform one EM step."""
            # batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs, parent_indices, is_division_mask, active_mask, is_new_root_mask)
            batch_stats, lls = self.e_step(params, batch_emissions, batch_inputs, parent_indices, is_division_mask, active_mask, is_new_root_mask)
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
            if jnp.isnan(params.emissions.weights).any():
                print("nan in emissions weights")

            # debug.print('e_step: {x}', x=(batch_stats, lls))
            # debug.print('m_step{y}', y=params)
            return params, m_step_state, lp

        log_probs = []
        # m_step_state = self.initialize_m_step_state(params, props) # none 
        m_step_state = (None,None,None,None)
        pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
        for _ in pbar:
            params, m_step_state, marginal_logprob = em_step(params, m_step_state)
            log_probs.append(marginal_logprob)
        return params, jnp.array(log_probs)