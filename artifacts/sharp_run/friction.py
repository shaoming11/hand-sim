"""Does the M1 scripted gait still turn the cap as thread friction rises?"""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/shaoming/Documents/GitHub/hand-sim")
from capturn import scene

z = np.load("artifacts/squeeze_twist.npz")
ctrl = z["ctrl"]; ctrl_dt = float(z["ctrl_dt"]) if "ctrl_dt" in z else 0.05
print(f"replaying the M1 gait: {len(ctrl)} control steps, {len(ctrl)*ctrl_dt:.1f}s\n")
print(f"{'frictionloss':>13} {'needs Fn':>9} {'final cap angle':>16} {'vs 0.05':>9}")
base = None
for fl in (0.05, 0.08, 0.10, 0.125, 0.15, 0.20):
    m = mujoco.MjModel.from_xml_path(scene.SCENE.as_posix())
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, scene.CAP_JOINT)
    m.dof_frictionloss[m.jnt_dofadr[jid]] = fl
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("pregrasp").id)
    n_sub = max(1, int(round(ctrl_dt / m.opt.timestep)))
    for c in ctrl:
        d.ctrl[:] = c
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
    ang = np.degrees(d.qpos[m.jnt_qposadr[jid]])
    if base is None: base = ang
    print(f"{fl:13.3f} {fl/0.016:8.2f}N {ang:+15.2f} deg {100*ang/base:8.0f}%")
