import mujoco
import numpy as np
from dm_control import mjcf

from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv


class PuzzleEnv(ManipSpaceEnv):
    """Puzzle environment.

    This environment implements the "Lights Out" puzzle game. The goal is to set all buttons to a specific state. It
    supports the following variants:
    - `env_type`: '3x3', '4x4', '4x5', '4x6'.

    In addition to `qpos` and `qvel`, it maintains the following state variables.
    - `button_states`: A binary array of size `num_buttons` representing the state of each button. Stored in
        `_cur_button_states`.
    """

    def __init__(self, env_type, *args, **kwargs):
        """Initialize the Puzzle environment.

        Args:
            env_type: Environment type. One of '3x3', '4x4', '4x5', or '4x6'.
            *args: Additional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self._env_type = env_type

        # Set the puzzle size.
        self._num_button_states = 2

        if env_type == '3x3':
            self._num_rows = 3
            self._num_cols = 3
        elif env_type == '4x4':
            self._num_rows = 4
            self._num_cols = 4
        elif env_type == '4x5':
            self._num_rows = 4
            self._num_cols = 5
        elif env_type == '4x6':
            self._num_rows = 4
            self._num_cols = 6
        else:
            raise ValueError(f'Unknown env_type: {env_type}')

        self._num_buttons = self._num_rows * self._num_cols
        self._cur_button_states = np.array([0] * self._num_buttons)

        super().__init__(*args, **kwargs)

        # Adjust arm sampling bounds to a smaller region.
        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.25]])

        # Target info.
        self._target_task = 'button'
        self._target_button = 0
        self._target_button_states = np.array([0] * self._num_buttons)

    def set_state(self, qpos, qvel, button_states):
        self._cur_button_states = button_states.copy()
        self._apply_button_states()
        super().set_state(qpos, qvel)

    def set_tasks(self):
        if self._num_rows == 3 and self._num_cols == 3:
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_button_states=np.array(
                        [
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task2',
                    init_button_states=np.array(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task3',
                    init_button_states=np.array(
                        [
                            [0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task4',
                    init_button_states=np.array(
                        [
                            [0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task5',
                    init_button_states=np.array(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 0, 1],
                            [1, 0, 1],
                            [1, 0, 1],
                        ]
                    ).flatten(),
                ),
            ]
        elif self._num_rows == 4 and self._num_cols == 4:
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task2',
                    init_button_states=np.array(
                        [
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task3',
                    init_button_states=np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [1, 0, 1, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task4',
                    init_button_states=np.array(
                        [
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task5',
                    init_button_states=np.array(
                        [
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
            ]
        elif self._num_rows == 4 and self._num_cols == 5:
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_button_states=np.array(
                        [
                            [1, 1, 0, 1, 1],
                            [0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0],
                            [1, 1, 0, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task2',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task3',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task4',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task5',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 0, 0, 0, 1],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [1, 0, 0, 0, 1],
                        ]
                    ).flatten(),
                ),
            ]
        elif self._num_rows == 4 and self._num_cols == 6:
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_button_states=np.array(
                        [
                            [1, 1, 0, 1, 1, 1],
                            [0, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 0],
                            [1, 1, 1, 0, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task2',
                    init_button_states=np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task3',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1, 1, 1, 0],
                            [1, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 1],
                            [0, 1, 1, 1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task4',
                    init_button_states=np.array(
                        [
                            [0, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task5',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 0, 0, 0, 0, 1],
                            [0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0],
                            [1, 0, 0, 0, 0, 1],
                        ]
                    ).flatten(),
                ),
            ]

        if self._reward_task_id == 0:
            # Set default task.
            if self._num_rows == 3 and self._num_cols == 3:
                self._reward_task_id = 4
            elif self._num_rows == 4 and self._num_cols == 4:
                self._reward_task_id = 4
            elif self._num_rows == 4 and self._num_cols == 5:
                self._reward_task_id = 2
            elif self._num_rows == 4 and self._num_cols == 6:
                self._reward_task_id = 2

    def add_objects(self, arena_mjcf):
        # Add button scene.
        button_outer_mjcf = mjcf.from_path((self._desc_dir / 'button_outer.xml').as_posix())
        arena_mjcf.include_copy(button_outer_mjcf)

        # Add buttons to the scene.
        distance = 0.05
        for i in range(self._num_rows):
            for j in range(self._num_cols):
                button_mjcf = mjcf.from_path((self._desc_dir / 'button_inner.xml').as_posix())
                pos_x = 0.425 - distance * (self._num_rows - 1) + 2 * distance * i
                pos_y = 0.0 - distance * (self._num_cols - 1) + 2 * distance * j
                button_mjcf.find('body', 'buttonbox_0').pos[:2] = np.array([pos_x, pos_y])
                for tag in ['body', 'joint', 'geom', 'site']:
                    for item in button_mjcf.find_all(tag):
                        if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                            item.name = item.name[:-2] + f'_{i * self._num_cols + j}'
                arena_mjcf.include_copy(button_mjcf)

        # Save button geoms.
        self._button_geoms_list = []
        for i in range(self._num_buttons):
            self._button_geoms_list.append([arena_mjcf.find('geom', f'btngeom_{i}')])

        # Add cameras.
        cameras = {
            'front': {
                'pos': (1.139, 0.000, 0.821),
                'xyaxes': (0.000, 1.000, 0.000, -0.627, 0.000, 0.779),
            },
            'front_pixels': {
                'pos': (0.905, 0.000, 0.762),
                'xyaxes': (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        # Button geom IDs.
        self._button_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in button_geoms]
            for button_geoms in self._button_geoms_list
        ]
        self._button_site_ids = [self._model.site(f'btntop_{i}').id for i in range(self._num_buttons)]

    def _apply_button_states(self):
        # Adjust button colors based on the current state.
        for i in range(self._num_buttons):
            for gid in self._button_geom_ids_list[i]:
                color_zero = self._colors['red']
                color_one = self._colors['blue']
                self._model.geom(gid).rgba = color_zero if self._cur_button_states[i] == 0 else color_one

        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == 'data_collection':
            # Randomize the scene.

            self.initialize_arm()

            # Randomize button states.
            for i in range(self._num_buttons):
                self._cur_button_states[i] = self.np_random.choice(self._num_button_states)
            self._apply_button_states()

            # Set a new target.
            self.set_new_target(return_info=False)
        else:
            # Set button states based on the current task.

            # Get the current task info.
            init_button_states = self.cur_task_info['init_button_states'].copy()
            goal_button_states = self.cur_task_info['goal_button_states'].copy()

            # First, force set the current scene to the goal state to obtain the goal observation.
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            self._cur_button_states = goal_button_states.copy()
            self._apply_button_states()
            mujoco.mj_forward(self._model, self._data)

            # Do a few random steps to make the scene stable.
            for _ in range(5):
                action = self.action_space.sample()
                action[-1] = 1  # Close gripper.
                self.step(action)

            # Save the goal observation.
            self._cur_goal_ob = (
                self.compute_oracle_observation() if self._use_oracle_rep else self.compute_observation()
            )
            if self._render_goal:
                self._cur_goal_rendered = self.render()
            else:
                self._cur_goal_rendered = None

            # Now, do the actual reset.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            self._cur_button_states = init_button_states.copy()
            self._target_button_states = goal_button_states.copy()
            self._apply_button_states()

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Set a new random target for data collection.

        Args:
            return_info: Whether to return the observation and reset info.
            p_stack: Unused; defined for compatibility with the other environments.
        """
        assert self._mode == 'data_collection'

        # Set target button.
        self._target_button = self.np_random.choice(self._num_buttons)
        self._target_button_states[self._target_button] = (
            self._cur_button_states[self._target_button] + 1
        ) % self._num_button_states

        mujoco.mj_kinematics(self._model, self._data)

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def pre_step(self):
        self._prev_button_states = self._cur_button_states.copy()
        super().pre_step()

    def _compute_successes(self):
        """Compute object successes."""
        button_successes = [
            (self._cur_button_states[i] == self._target_button_states[i]) for i in range(self._num_buttons)
        ]

        return button_successes

    def post_step(self):
        # Update button states.
        for i in range(self._num_buttons):
            prev_joint_pos = self._prev_ob_info[f'privileged/button_{i}_pos'][0]
            cur_joint_pos = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()[0]
            if prev_joint_pos > -0.02 and cur_joint_pos <= -0.02:
                # Button pressed: change the state of the button and its neighbors.
                x, y = i // self._num_cols, i % self._num_cols
                for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self._num_rows and 0 <= ny < self._num_cols:
                        self._cur_button_states[nx * self._num_cols + ny] = (
                            self._cur_button_states[nx * self._num_cols + ny] + 1
                        ) % self._num_button_states
        self._apply_button_states()

        # Evaluate successes.
        button_successes = self._compute_successes()
        if self._mode == 'data_collection':
            self._success = button_successes[self._target_button]
        else:
            self._success = all(button_successes)

    def add_object_info(self, ob_info):
        # Button states.
        for i in range(self._num_buttons):
            ob_info[f'privileged/button_{i}_state'] = self._cur_button_states[i]
            ob_info[f'privileged/button_{i}_pos'] = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()
            ob_info[f'privileged/button_{i}_vel'] = self._data.joint(f'buttonbox_joint_{i}').qvel.copy()

        if self._mode == 'data_collection':
            # Target button info.
            ob_info['privileged/target_task'] = self._target_task

            ob_info['privileged/target_button'] = self._target_button
            ob_info['privileged/target_button_state'] = self._target_button_states[self._target_button]
            ob_info['privileged/target_button_top_pos'] = self._data.site_xpos[
                self._button_site_ids[self._target_button]
            ].copy()

        ob_info['prev_button_states'] = self._prev_button_states.copy()
        ob_info['button_states'] = self._cur_button_states.copy()

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0
            button_scaler = 120.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
            for i in range(self._num_buttons):
                button_state = np.eye(self._num_button_states)[self._cur_button_states[i]]
                ob.extend(
                    [
                        button_state,
                        ob_info[f'privileged/button_{i}_pos'] * button_scaler,
                        ob_info[f'privileged/button_{i}_vel'],
                    ]
                )

            return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        return self._cur_button_states.astype(np.float64)

    def compute_reward(self):
        if self._reward_task_id is None:
            return super().compute_reward()

        # Compute the reward based on the task.
        successes = self._compute_successes()
        reward = float(sum(successes) - len(successes))
        return reward
