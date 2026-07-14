import atexit
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from ui import console, viewer_key_callback

XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "hand2",
    "right.xml",
)
NUM_JOINTS = 20


def create_hand_node(cfg):
    if not cfg.get("sim", False):
        raise RuntimeError("hand2 只支持 MuJoCo 仿真后端，请设置 cfg['sim']=True")
    return Hand2NodeSim(cfg)


class Hand2NodeSim:
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        if self.model.nu != NUM_JOINTS:
            raise RuntimeError(
                f"hand2 模型应有 {NUM_JOINTS} 个执行器，实际 {self.model.nu}"
            )

        joint_ids = self.model.actuator_trnid[:NUM_JOINTS, 0]
        self._qpos_adr = self.model.jnt_qposadr[joint_ids]
        self._dof_adr = self.model.jnt_dofadr[joint_ids]

        base_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free"
        )
        if base_jid < 0:
            raise RuntimeError("hand2 模型缺少浮动基座关节 base_free")
        self._base_qadr = self.model.jnt_qposadr[base_jid]
        self._base_dadr = self.model.jnt_dofadr[base_jid]
        self._eef_pose = self.model.qpos0[
            self._base_qadr : self._base_qadr + 7
        ].copy()

        init_pos = cfg.get("init_pos", None)
        if init_pos is not None:
            self.curr_pos = np.array(init_pos, dtype=float)
        else:
            self.curr_pos = np.zeros(NUM_JOINTS)
        self.prev_pos = self.curr_pos

        self.data.qpos[self._qpos_adr] = self.curr_pos
        self.data.ctrl[:NUM_JOINTS] = self.curr_pos
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
        console.print("[green]MuJoCo 仿真已启动[/] (hand2, assets/hand2/right.xml)")

    def _sim_loop(self):
        timestep = self.model.opt.timestep
        while self._running:
            step_start = time.perf_counter()

            if self.free_drag_active:
                self.data.ctrl[:NUM_JOINTS] = self.data.qpos[self._qpos_adr]
                self.curr_pos = self.data.qpos[self._qpos_adr].copy()
                self._eef_pose = self.data.qpos[
                    self._base_qadr : self._base_qadr + 7
                ].copy()
            else:
                self.data.ctrl[:NUM_JOINTS] = self.curr_pos
                self.data.qpos[self._base_qadr : self._base_qadr + 7] = self._eef_pose
                self.data.qvel[self._base_dadr : self._base_dadr + 6] = 0.0

            mujoco.mj_step(self.model, self.data)

            if self._viewer.is_running():
                self._viewer.sync()

            elapsed = time.perf_counter() - step_start
            if elapsed < timestep:
                time.sleep(timestep - elapsed)

    def set_pos(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose, dtype=float)

    def read_eef(self):
        return self.data.qpos[self._base_qadr : self._base_qadr + 7].copy()

    def set_eef(self, pose):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (7,):
            raise ValueError(f"EEF 位姿应为 7 维 [x y z qw qx qy qz]，实际 {pose.shape}")
        self._eef_pose = pose

    def read_pos(self):
        return self.data.qpos[self._qpos_adr].copy()

    def read_vel(self):
        return self.data.qvel[self._dof_adr].copy()

    def read_cur(self):
        return self.data.actuator_force[:NUM_JOINTS].copy()

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
