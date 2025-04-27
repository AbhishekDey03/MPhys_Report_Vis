from CLEAN_Algorithm_Simulation import RadioImagerCLEAN
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":   ["Computer Modern"],
    "axes.linewidth": 0.7,
})


# Simulate a radio imager with CLEAN algorithm
ri = RadioImagerCLEAN(uv_taper_fwhm_frac=0.1)
sky, dirty, restored, psf,mask = ri.simulate()

# Plot the resultsss
fig, axs = plt.subplots(2,2, figsize=(8,8))
axs = axs.flatten()
for ax, img, title in zip(axs, [sky, restored, psf, dirty],
                          ["True sky","CLEANed","PSF","Dirty"]):
    ax.imshow(img, origin="lower", cmap="gist_gray")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]) 
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.imshow(mask, origin="lower", cmap="gist_gray")
ax.set_title(r"$(u,v)$ sampling mask"); ax.set_xticks([]); ax.set_yticks([])
plt.show()