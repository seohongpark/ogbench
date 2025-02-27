import contextlib
import os

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


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
        render_mode='rgb_array',
        width=200,
        height=200,
        **kwargs,
    ):
        """Initialize the Humanoid environment.

        Args:
            xml_file: Path to the XML description (optional).
            render_mode: Rendering mode.
            width: Width of the rendered image.
            height: Height of the rendered image.
            **kwargs: Additional keyword arguments.
        """
        if xml_file is None:
            xml_file = self.xml_file
        utils.EzPickle.__init__(
            self,
            xml_file,
            **kwargs,
        )

        observation_space = Box(low=-np.inf, high=np.inf, shape=(69,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            **kwargs,
        )

    def step(self, action):
        prev_qpos = self.data.qpos.copy()
        prev_qvel = self.data.qvel.copy()

        self.do_simulation(action, self.frame_skip)

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        observation = self.get_ob()

        if self.render_mode == 'human':
            self.render()

        return (
            observation,
            0.0,
            False,
            False,
            {
                'xy': self.get_xy(),
                'prev_qpos': prev_qpos,
                'prev_qvel': prev_qvel,
                'qpos': qpos,
                'qvel': qvel,
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

    def get_ob(self):
        xy = self.data.qpos[:2]
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
            [
                xy,
                joint_angles,
                [head_height],
                extremities,
                torso_vertical_orientation,
                center_of_mass_velocity,
                velocity,
            ]
        )

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
            self.data.qvel = 0.1 * self.np_random.standard_normal(self.model.nv)

            for joint_id in range(1, self.model.njnt):
                range_min, range_max = self.model.jnt_range[joint_id]
                self.data.qpos[6 + joint_id] = self.np_random.uniform(range_min, range_max)

            with self.disable('actuation'):
                mujoco.mj_forward(self.model, self.data)
            penetrating = self.data.ncon > 0

        observation = self.get_ob()

        return observation

    def get_xy(self):
        return self.data.qpos[:2].copy()

    def set_xy(self, xy):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[:2] = xy
        self.set_state(qpos, qvel)
