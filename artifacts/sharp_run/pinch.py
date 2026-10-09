"""Cap torque ceiling for a thumb+index pinch alone, vs the whole hand.

Repeats NOTES.md's M1 flexion-ramp measurement but restricted to the two pinch fingers,
which is what decides whether a two-finger grasp can beat a given thread friction.
"""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/shaoming/Documents/GitHub/hand-sim")
from capturn import scene

MU, R = 1.0, 0.016
PINCH = ["rh_A_THJ2", "rh_A_THJ1", "rh_A_FFJ3", "rh_A_FFJ0"]          # thumb + index flexion
ALL = PINCH + ["rh_A_MFJ3", "rh_A_MFJ0", "rh_A_RFJ3", "rh_A_RFJ0", "rh_A_LFJ3", "rh_A_LFJ0"]

m = mujoco.MjModel.from_xml_path(scene.SCENE.as_posix())
cap_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cap")

def normal_force_on_cap(groups, frac):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("pregrasp").id)
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in groups]
    target = d.ctrl.copy()
    for i in ids:
        lo, hi = m.actuator_ctrlrange[i]
        target[i] = d.ctrl[i] + frac * (hi - d.ctrl[i])
    for _ in range(600):                      # settle
        d.ctrl[:] = target
        mujoco.mj_step(m, d)
    total = 0.0
    f6 = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        if cap_gid in (con.geom1, con.geom2):
            mujoco.mj_contactForce(m, d, c, f6)
            total += abs(f6[0])
    return total

print(f"{'extra flexion':>14} {'thumb+index':>22} {'whole hand':>22}")
print(f"{'':>14} {'sum(Fn)':>10}{'torque':>12} {'sum(Fn)':>10}{'torque':>12}")
for frac in (0.10, 0.20, 0.35, 0.50):
    fp = normal_force_on_cap(PINCH, frac)
    fa = normal_force_on_cap(ALL, frac)
    print(f"{frac*100:13.0f}% {fp:9.2f}N {MU*R*fp:10.3f}Nm {fa:9.2f}N {MU*R*fa:10.3f}Nm")
