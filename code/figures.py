"""Figures for 'Long-Distance Universality for the Berger Sphere and the Unicycle'.
Usage: python3 figures.py OUTDIR
Draws (1) the pencil of circular arcs joining two points of the sphere of radius 1/2,
(2) the parking manoeuvre built in Section 5 for I = 4 and I = infinity."""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "."

def rot(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K

# ---------- Figure 1: pencil of arcs ----------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
R = 0.5
u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
ax.plot_surface(R*np.cos(u)*np.sin(v), R*np.sin(u)*np.sin(v), R*np.cos(v),
                color="#dfe7f2", alpha=0.35, linewidth=0, shade=True)
a = 0.85
p = np.array([-np.sin(a), 0, np.cos(a)]); q = np.array([np.sin(a), 0, np.cos(a)])
cmap = plt.get_cmap("viridis")
ts = np.linspace(-np.pi + 0.35, np.pi - 0.35, 9)
for i, t in enumerate(ts):
    c = np.array([0, np.sin(t), np.cos(t)])
    w1 = p - (c @ p) * c; w2 = q - (c @ q) * c
    ang = np.arctan2(c @ np.cross(w1, w2), w1 @ w2)
    s = np.linspace(0, 1, 200)
    pts = np.array([rot(c, ang * x) @ p for x in s]) * R
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=cmap(i / len(ts)), lw=2)
for P, lab in ((p, "$p$"), (q, "$q$")):
    ax.scatter(*(R*P), color="k", s=30)
    ax.text(*(R*P*1.12), lab, fontsize=14)
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
ax.view_init(elev=55, azim=-90); ax.set_xlim(-0.45,0.45); ax.set_ylim(-0.45,0.45); ax.set_zlim(-0.42,0.5)
plt.tight_layout()
plt.savefig(f"{out}/fig_pencil.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close()

# ---------- Figure 2: parking manoeuvre ----------
def manoeuvre(I, z, eps, n=4000):
    k = 1.0 if np.isinf(I) else np.sqrt(1 - 1/I)
    iI = 0.0 if np.isinf(I) else 1/I
    th1 = np.pi/2 - eps
    th = np.linspace(0, th1, n)
    dzdth = (np.sin(th)**2 + iI*np.cos(th)**2) / (k*np.cos(th))
    Z1 = np.concatenate([[0], np.cumsum(0.5*(dzdth[1:]+dzdth[:-1])*np.diff(th))])
    Y1 = k*(1 - np.cos(th))
    Y = 2*Y1[-1]; m = z - 2*Z1[-1]
    delta = np.arctan(Y/m)
    ys = list(Y1); zs = list(Z1); hs = list(th)
    # straight tilted segment
    ell = np.hypot(m, Y)
    for x in np.linspace(0, 1, 200):
        ys.append(Y1[-1] - x*ell*np.sin(delta)); zs.append(Z1[-1] + x*ell*np.cos(delta)); hs.append(np.pi/2 + delta)
    y0, z0 = ys[-1], zs[-1]
    for t, yy, zz in zip(th[::-1], Y1[::-1], Z1[::-1]):
        ys.append(y0 + (Y1[-1] - yy)); zs.append(z0 + (Z1[-1] - zz)); hs.append(t)
    return np.array(ys), np.array(zs), np.array(hs)

fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), sharey=False)
for axx, I, ttl in ((axs[0], 4.0, r"$\mathcal{I}=4$"), (axs[1], np.inf, r"$\mathcal{I}=\infty$")):
    ys, zs, hs = manoeuvre(I, 24.0, 0.02)
    axx.plot(zs, ys, color="#1f5aa6", lw=2)
    for j in np.linspace(0, len(zs)-1, 22).astype(int):
        axx.arrow(zs[j], ys[j], 0.9*np.sin(hs[j]), 0.45*np.cos(hs[j]), head_width=0.12, head_length=0.3,
                  color="#c0392b", length_includes_head=True)
    axx.set_title(ttl); axx.set_xlabel("$z$ (towards the curb)")
    axx.grid(alpha=0.3)
axs[1].set_ylabel("$y$ (along the curb)"); axs[0].set_ylabel("$y$ (along the curb)")
plt.tight_layout()
plt.savefig(f"{out}/fig_parking.png", dpi=170, bbox_inches="tight", facecolor="white")
