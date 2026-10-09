import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/shaoming/Documents/GitHub/hand-sim")
from capturn import scene
m = mujoco.MjModel.from_xml_path(scene.SCENE.as_posix()); d = mujoco.MjData(m)
cap_r, cap_hh = scene.cap_dimensions(m)
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cap_site")
TIPS = scene.ALL_TIPS  # thumb, index(ff), middle(mf), ring(rf), little(lf)
ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIPS]
for path in sys.argv[1:]:
    q = np.load(path)["qpos"]
    gaps = []
    for qi in q:
        d.qpos[:] = qi; d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        o = d.site_xpos[site]; R = d.site_xmat[site].reshape(3, 3)
        local = (d.site_xpos[ids] - o) @ R
        gaps.append(scene.surface_distance(local, cap_r, cap_hh))
    g = np.array(gaps)
    print(f"\n== {path.split('/')[-1]}  ({len(q)} frames)")
    for k, name in enumerate(TIPS):
        touch = (g[:, k] < 0.010).mean() * 100
        print(f"   {name:10} median gap {np.median(g[:,k])*1e3:7.2f} mm   within 10mm on {touch:5.1f}% of frames")
