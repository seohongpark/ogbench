import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class AntEnv(MujocoEnv, utils.EzPickle):
    """Gymnasium Ant environment.

    Unlike the original Ant environment, this environment uses a restricted joint range for the actuators, as typically
    done in previous works in hierarchical reinforcement learning. It also uses a control frequency of 10Hz instead of
    20Hz, which is the default in the original environment.
    """

    xml_file = os.path.join(os.path.dirname(__file__), 'assets', 'ant.xml')
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array', 'rgbd_tuple'],
        'render_fps': 10,
    }

    def __init__(
        self,
        xml_file=None,
        reset_noise_scale=0.1,
        render_mode='rgb_array',
        width=200,
        height=200,
        **kwargs,
    ):
        """Initialize the Ant environment.

        Args:
            xml_file: Path to the XML description (optional).
            reset_noise_scale: Scale of the noise added to the initial state during reset.
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
            reset_noise_scale,
            **kwargs,
        )

        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64)

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

    def get_ob(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        return np.concatenate([position, velocity])

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)

        observation = self.get_ob()

        return observation

    def get_xy(self):
        return self.data.qpos[:2].copy()

    def set_xy(self, xy):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[:2] = xy
        self.set_state(qpos, qvel)
