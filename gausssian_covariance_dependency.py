import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  
from scipy.stats import multivariate_normal as mvn

plt.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":   ["Computer Modern"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.linewidth": 0.7,
})

covs   = [np.eye(2), # Isotropic
          np.diag([2, .5]), # Anisotropic
          np.array([[1, .8], # Correlated
                    [.8,  1]])]

titles = ["(a) Isotropic", "(b) Anisotropic", "(c) Correlated"]
mu     = np.zeros(2) # Origin mean

fig, axes = plt.subplots(
    1, 3,
    figsize=(24/2.54, 8/2.54),
    subplot_kw={"projection": "3d"},
    constrained_layout=True
)

x = y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)
pos  = np.dstack((X, Y))

for ax, Sigma, title in zip(axes, covs, titles):
    Z = mvn(mu, Sigma).pdf(pos)

    ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                    cmap=cm.viridis, linewidth=0, antialiased=True)


    ax.set_title(title, pad=2)
    ax.set_box_aspect((1, 1, 0.35))
    ax.view_init(elev=35, azim=-45)

    ax.set_xlabel(r"$x_{1}$", labelpad=4)
    ax.set_ylabel(r"$x_{2}$", labelpad=4)
    ax.set_zlabel("")                 

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    ax.text2D(0.5, -0.16,
              (r"$\displaystyle \Sigma ="
               r"\begin{bmatrix}"
               f"{Sigma[0,0]:.1f} & {Sigma[0,1]:.1f} \\\\ "
               f"{Sigma[1,0]:.1f} & {Sigma[1,1]:.1f}"
               r"\end{bmatrix}$"),
              transform=ax.transAxes,
              ha="center", va="center", fontsize=12)

fig.savefig("gaussian_surfaces_final.pdf")
fig.savefig("gaussian_surfaces_final.png", dpi=600, transparent=True)
plt.show()
