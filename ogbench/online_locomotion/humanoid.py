import contextlib
import os
import warnings

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

_DEFAULT_VALUE_AT_MARGIN = 0.1
_STAND_HEIGHT = 1.4
_WALK_SPEED = 1
_RUN_SPEED = 10


def _sigmoids(x, value_at_1, sigmoid):
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError('`value_at_1` must be nonnegative and smaller than 1, ' 'got {}.'.format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, ' 'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == 'hyperbolic':
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='invalid value encountered in cos')
            cos_pi_scaled_x = np.cos(np.pi * scaled_x)
        return np.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == 'quadratic':
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == 'tanh_squared':
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return float(value) if np.isscalar(x) else value


class HumanoidEnv(MujocoEnv, utils.EzPickle):
    """DMC Humanoid environment.

    Several methods are reimplemented to remove the dependency on the `dm_control` package. It is supposed to work
    identically to the original Humanoid environment.
    """

    xml_file = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid.xml')
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array', 'rgbd_tuple'],
        'render_fps': 40,
    }

    def __init__(
        self,
        xml_file=None,
        task='run',
        **kwargs,
    ):
        if xml_file is None:
            xml_file = self.xml_file
        utils.EzPickle.__init__(
            self,
            xml_file,
            task,
            **kwargs,
        )

        self._move_speed = {
            'walk': _WALK_SPEED,
            'run': _RUN_SPEED,
        }[task]

        observation_space = Box(low=-np.inf, high=np.inf, shape=(67,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs,
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward = self._get_reward()

        if self.render_mode == 'human':
            self.render()

        return (
            observation,
            reward,
            False,
            False,
            {
                'xy': self.get_xy(),
            },
        )

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        # DMC-style stepping (see Page 6 of https://arxiv.org/abs/2006.12983).
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4.value:
            mujoco.mj_step2(self.model, self.data)
            if n_frames > 1:
                mujoco.mj_step(self.model, self.data, n_frames - 1)
        else:
            mujoco.mj_step(self.model, self.data, n_frames)

        mujoco.mj_step1(self.model, self.data)

    def _get_obs(self):
        joint_angles = self.data.qpos[7:]  # Skip the 7 DoFs of the free root joint.
        head_height = self.data.xpos[2, 2]  # ['head', 'z']
        torso_frame = self.data.xmat[1].reshape(3, 3)  # ['torso']
        torso_pos = self.data.xpos[1]  # ['torso']
        positions = []
        for idx in [16, 10, 13, 7]:  # ['left_hand', 'left_foot', 'right_hand', 'right_foot']
            torso_to_limb = self.data.xpos[idx] - torso_pos
            positions.append(torso_to_limb.dot(torso_frame))
        extremities = np.hstack(positions)
        torso_vertical_orientation = self.data.xmat[1, [6, 7, 8]]  # ['torso', ['zx', 'zy', 'zz']]
        center_of_mass_velocity = self.data.sensordata[0:3]  # ['torso_subtreelinvel']
        velocity = self.data.qvel

        return np.concatenate(
            [joint_angles, [head_height], extremities, torso_vertical_orientation, center_of_mass_velocity, velocity]
        )

    def _get_reward(self):
        head_height = self.data.xpos[2, 2]  # ['head', 'z']
        torso_upright = self.data.xmat[1, 8]  # ['torso', 'zz']
        center_of_mass_velocity = self.data.sensordata[0:3]  # ['torso_subtreelinvel']
        control = self.data.ctrl.copy()

        standing = tolerance(head_height, bounds=(_STAND_HEIGHT, float('inf')), margin=_STAND_HEIGHT / 4)
        upright = tolerance(torso_upright, bounds=(0.9, float('inf')), margin=1.9, sigmoid='linear', value_at_margin=0)
        stand_reward = standing * upright
        small_control = tolerance(control, margin=1, value_at_margin=0, sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5
        if self._move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move
        else:
            com_velocity = np.linalg.norm(center_of_mass_velocity[[0, 1]])
            move = tolerance(
                com_velocity,
                bounds=(self._move_speed, float('inf')),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid='linear',
            )
            move = (5 * move + 1) / 6
            return small_control * stand_reward * move

    @contextlib.contextmanager
    def disable(self, *flags):
        old_bitmask = self.model.opt.disableflags
        new_bitmask = old_bitmask
        for flag in flags:
            if isinstance(flag, str):
                field_name = 'mjDSBL_' + flag.upper()
                flag = getattr(mujoco.mjtDisableBit, field_name)
            elif isinstance(flag, int):
                flag = mujoco.mjtDisableBit(flag)
            new_bitmask |= flag.value
        self.model.opt.disableflags = new_bitmask
        try:
            yield
        finally:
            self.model.opt.disableflags = old_bitmask

    def reset_model(self):
        penetrating = True
        while penetrating:
            quat = self.np_random.uniform(size=4)
            quat /= np.linalg.norm(quat)
            self.data.qpos[3:7] = quat

            for joint_id in range(1, self.model.njnt):
                range_min, range_max = self.model.jnt_range[joint_id]
                self.data.qpos[6 + joint_id] = self.np_random.uniform(range_min, range_max)

            with self.disable('actuation'):
                mujoco.mj_forward(self.model, self.data)
            penetrating = self.data.ncon > 0

        observation = self._get_obs()

        return observation

    def get_xy(self):
        return self.data.qpos[:2].copy()

    def set_xy(self, xy):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[:2] = xy
        self.set_state(qpos, qvel)
