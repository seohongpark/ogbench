from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCIQEValue, GCMRNValue, LogParam


class QRLAgent(flax.struct.PyTreeNode):
    """Quasimetric RL (QRL) agent.

    This implementation supports the following variants:
    (1) Value parameterizations: IQE (quasimetric_type='iqe') and MRN (quasimetric_type='mrn').
    (2) Actor losses: AWR (actor_loss='awr') and latent dynamics-based DDPG+BC (actor_loss='ddpgbc').

    QRL with AWR only fits a quasimetric value function and an actor network. QRL with DDPG+BC fits a quasimetric value
    function, an actor network, and a latent dynamics model. The latent dynamics model is used to compute
    reparameterized gradients for the actor loss. The original implementation of QRL uses IQE and DDPG+BC.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def value_loss(self, batch, grad_params):
        """Compute the QRL value loss."""
        d_neg = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        d_pos = self.network.select('value')(batch['observations'], batch['next_observations'], params=grad_params)
        lam = self.network.select('lam')(params=grad_params)

        # Apply loss shaping following the original implementation.
        d_neg_loss = (100 * jax.nn.softplus(5 - d_neg / 100)).mean()
        d_pos_loss = (jax.nn.relu(d_pos + batch['rewards']) ** 2).mean() # the reward of the goal is not -1
        
        value_loss = d_neg_loss + d_pos_loss * jax.lax.stop_gradient(lam)
        lam_loss = lam * (self.config['eps'] - jax.lax.stop_gradient(d_pos_loss))

        total_loss = value_loss + lam_loss

        return total_loss, {
            'total_loss': total_loss,
            'value_loss': value_loss,
            'lam_loss': lam_loss,
            'd_neg_loss': d_neg_loss,
            'd_neg_mean': d_neg.mean(),
            'd_neg_max': d_neg.max(),
            'd_neg_min': d_neg.min(),
            'd_pos_loss': d_pos_loss,
            'd_pos_mean': d_pos.mean(),
            'd_pos_max': d_pos.max(),
            'd_pos_min': d_pos.min(),
            'lam': lam,
        }

    def dynamics_loss(self, batch, grad_params):
        """Compute the dynamics loss."""
        _, ob_reps, next_ob_reps = self.network.select('value')(
            batch['observations'], batch['next_observations'], info=True, params=grad_params
        )
        # Dynamics model predicts the delta of the next observation.
        pred_next_ob_reps = ob_reps + self.network.select('dynamics')(
            jnp.concatenate([ob_reps, batch['actions']], axis=-1), params=grad_params
        )

        dist1 = self.network.select('value')(next_ob_reps, pred_next_ob_reps, is_phi=True, params=grad_params)
        dist2 = self.network.select('value')(pred_next_ob_reps, next_ob_reps, is_phi=True, params=grad_params)
        dynamics_loss = (dist1 + dist2).mean() / 2

        return dynamics_loss, {
            'dynamics_loss': dynamics_loss,
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        if self.config['actor_loss'] == 'awr':
            # Compute AWR loss based on V(s', g) - V(s, g).
            v = -self.network.select('value')(batch['observations'], batch['actor_goals'])
            nv = -self.network.select('value')(batch['next_observations'], batch['actor_goals'])
            adv = nv - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # Compute DDPG+BC loss based on latent dynamics model.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

            _, ob_reps, goal_reps = self.network.select('value')(batch['observations'], batch['actor_goals'], info=True)
            pred_next_ob_reps = ob_reps + self.network.select('dynamics')(
                jnp.concatenate([ob_reps, q_actions], axis=-1)
            )
            q = -self.network.select('value')(pred_next_ob_reps, goal_reps, is_phi=True)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        if self.config['actor_loss'] == 'ddpgbc':
            dynamics_loss, dynamics_info = self.dynamics_loss(batch, grad_params)
            for k, v in dynamics_info.items():
                info[f'dynamics/{k}'] = v
        else:
            dynamics_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + dynamics_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        ex_latents = np.zeros((ex_observations.shape[0], config['latent_dim']), dtype=np.float32)
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        if config['quasimetric_type'] == 'mrn':
            value_def = GCMRNValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                encoder=encoders.get('value'),
            )
        elif config['quasimetric_type'] == 'iqe':
            value_def = GCIQEValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                dim_per_component=8,
                layer_norm=config['layer_norm'],
                encoder=encoders.get('value'),
            )
        else:
            raise ValueError(f'Unsupported quasimetric type: {config["quasimetric_type"]}')

        if config['actor_loss'] == 'ddpgbc':
            # DDPG+BC requires a latent dynamics model.
            dynamics_def = MLP(
                hidden_dims=(*config['value_hidden_dims'], config['latent_dim']),
                layer_norm=config['layer_norm'],
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        # Define the dual lambda variable.
        lam_def = LogParam()

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
            lam=(lam_def, ()),
        )
        if config['actor_loss'] == 'ddpgbc':
            network_info.update(
                dynamics=(dynamics_def, np.concatenate([ex_latents, ex_actions], axis=-1)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='qrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            quasimetric_type='iqe',  # Quasimetric parameterization type ('iqe' or 'mrn').
            latent_dim=512,  # Latent dimension for the quasimetric value function.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            eps=0.05,  # Margin for the dual lambda loss.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.003,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=1.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # negative rewards
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
