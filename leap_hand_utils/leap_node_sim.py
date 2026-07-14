import atexit
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

import leap_hand_utils.leap_hand_utils as lhu
from ui import console, viewer_key_callback

MOTOR_OF_ACTUATOR = np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15])


class LeapNodeSim:
    def __init__(self, cfg):
        self.cfg = cfg
        self.init_pos = cfg.get("init_pos", None)

        if self.init_pos is not None:
            self.prev_pos = self.curr_pos = np.array(self.init_pos, dtype=float)
        else:
            self.prev_pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))

        self.motors = list(range(16))

        from robot_descriptions import leap_hand_mj_description

        scene_path = os.path.join(
            os.path.dirname(leap_hand_mj_description.MJCF_PATH), "scene_right.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        if self.model.nu != 16:
            raise RuntimeError(f"LEAP Hand 模型应有 16 个执行器，实际 {self.model.nu}")

        joint_ids = self.model.actuator_trnid[:16, 0]
        self._qpos_adr = self.model.jnt_qposadr[joint_ids]
        self._dof_adr = self.model.jnt_dofadr[joint_ids]

        qpos_target = self._motor_to_qpos(self.curr_pos)
        self.data.qpos[self._qpos_adr] = qpos_target
        self.data.ctrl[:16] = qpos_target
        mujoco.mj_forward(self.model, self.data)

        self.free_drag_active = False

        threads_before = set(threading.enumerate())
        self._viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=viewer_key_callback
        )
        self._viewer_threads = [
            t for t in threading.enumerate() if t not in threads_before
        ]
        self._running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        atexit.register(self.close)
        console.print(
            "[green]MuJoCo 仿真已启动[/] (LEAP Hand, mujoco_menagerie scene_right.xml)"
        )

    def _motor_to_qpos(self, motor_pos):
        return np.asarray(motor_pos, dtype=float)[MOTOR_OF_ACTUATOR] - 3.14159

    def _qpos_to_motor(self, qpos_vals):
        motor = np.empty(16)
        motor[MOTOR_OF_ACTUATOR] = np.asarray(qpos_vals, dtype=float) + 3.14159
        return motor

    def _sim_loop(self):
        timestep = self.model.opt.timestep
        while self._running:
            step_start = time.perf_counter()

            if self.free_drag_active:
                self.data.ctrl[:16] = self.data.qpos[self._qpos_adr]
                self.curr_pos = self._qpos_to_motor(self.data.qpos[self._qpos_adr])
            else:
                self.data.ctrl[:16] = self._motor_to_qpos(self.curr_pos)

            mujoco.mj_step(self.model, self.data)

            if self._viewer.is_running():
                self._viewer.sync()

            elapsed = time.perf_counter() - step_start
            if elapsed < timestep:
                time.sleep(timestep - elapsed)

    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose, dtype=float)

    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose, dtype=float)

    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose, dtype=float)

    def read_pos(self):
        return self._qpos_to_motor(self.data.qpos[self._qpos_adr])

    def read_vel(self):
        vel = np.empty(16)
        vel[MOTOR_OF_ACTUATOR] = self.data.qvel[self._dof_adr]
        return vel

    def read_cur(self):
        cur = np.empty(16)
        cur[MOTOR_OF_ACTUATOR] = self.data.actuator_force[:16]
        return cur

    def enable_free_drag_mode(self):
        if self.free_drag_active:
            return
        self.free_drag_active = True
        console.print(
            "[yellow]仿真自由拖拽模式:[/] 在 MuJoCo 窗口中双击选中指节，Ctrl+右键拖拽即可摆姿势"
        )

    def disable_free_drag_mode(self):
        if not self.free_drag_active:
            return
        self.free_drag_active = False
        self.curr_pos = self.read_pos()

    def close(self):
        if not self._running:
            return
        self._running = False
        if self._sim_thread.is_alive():
            self._sim_thread.join(timeout=2.0)
        self._viewer.close()
        for t in self._viewer_threads:
            t.join(timeout=5.0)
