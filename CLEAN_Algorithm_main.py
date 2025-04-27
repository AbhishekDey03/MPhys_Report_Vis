from CLEAN_Algorithm_Simulation import RadioImagerCLEAN
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":   ["Gulliver"],
    "axes.linewidth": 0.7,
})

fwhm = 0.3

# Simulate a radio imager with CLEAN algorithm
ri = RadioImagerCLEAN(uv_taper_fwhm_frac=fwhm)
sky, dirty, restored, psf,mask = ri.simulate()

# Plot the resultsss
fig, axs = plt.subplots(1,4, figsize=(9,3))
axs = axs.flatten()
for ax, img, title in zip(axs, [sky, restored, dirty,psf ],
                          ["(a)True sky","(b)CLEANed image","(c)Dirty image","(d)Point-spread"]):
    ax.imshow(img, origin="lower", cmap="gist_gray")
    ax.set_xlabel(title); ax.set_xticks([]); ax.set_yticks([]) 
plt.suptitle(f"CLEAN Algorithm Deconvolution of Simulated Radio Sky", fontsize=14)
plt.savefig("CLEAN_Algorithm_Simulation.pdf", dpi=300)
plt.show()

"""fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.imshow(mask, origin="lower", cmap="gist_gray")
ax.set_title(r"$(u,v)$ sampling mask"); ax.set_xticks([]); ax.set_yticks([])
plt.show()"""