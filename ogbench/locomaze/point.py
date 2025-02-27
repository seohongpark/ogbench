import os

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class PointEnv(MujocoEnv, utils.EzPickle):
    """PointMass environment.

    This is a simple 2-D point mass environment, where the agent is controlled by an x-y action vector that is added to
    the current position of the point mass.
    """

    xml_file = os.path.join(os.path.dirname(__file__), 'assets', 'point.xml')
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array', 'rgbd_tuple'],
        'render_fps': 10,
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

        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

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

        action = 0.2 * action

        self.data.qpos[:] = self.data.qpos + action
        self.data.qvel[:] = np.array([0.0, 0.0])

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

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

    def get_ob(self):
        return self.data.qpos.flat.copy()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1

        self.set_state(qpos, qvel)

        return self.get_ob()

    def get_xy(self):
        return self.data.qpos.copy()

    def set_xy(self, xy):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[:] = xy
        self.set_state(qpos, qvel)
