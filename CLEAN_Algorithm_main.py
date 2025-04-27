from CLEAN_Algorithm_Simulation import RadioImagerCLEAN
import matplotlib.pyplot as plt

ri = RadioImagerCLEAN(uv_taper_fwhm_frac=0.5)
sky, dirty, restored, psf = ri.simulate()

fig, axs = plt.subplots(1,4, figsize=(16,4))
for ax, img, title in zip(axs, [sky, dirty, psf, restored],
                          ["True sky","Dirty","PSF","CLEANed"]):
    ax.imshow(img, origin="lower", cmap="inferno")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
