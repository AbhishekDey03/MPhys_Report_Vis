import numpy as np
from scipy.signal import fftconvolve

class RadioImagerCLEAN:
    
    def __init__(self, image_size = 256, fov = 1.0, fill_frac=0.5, uv_taper_fwhm_frac:float|None=None):
        """
        Initialize the RadioImagerCLEAN class with image size, field of view, fill fraction, and optional taper.
        
        Parameters:
        - image_size (int): Size of the image (image_size x image_size).
        - fov (float): Field of view in radians.
        - fill_frac (float): Fill fraction of the u-v plane.
        - u_v_taper_fwhm_frac (float or None): Optional tapering function FWHM in units of the image size.
        """
        self.image_size = image_size
        self.fov = fov
        self.fill_frac = fill_frac
        self.uv_taper_fwhm_frac = uv_taper_fwhm_frac
    
    def generate_sky(self,seed: int=8032003, n_sources: int = 5) -> np.ndarray:
        """
        Creates a sky populated with toy delta function sources at random positions.
        """
        rng = np.random.default_rng(seed)
        sky = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for _ in range(n_sources):
            l,m = rng.uniform(0.2,0.8,size=2)
            amp = rng.uniform(0.5,1.0)
            i,j = int(l * self.image_size), int(m * self.image_size)
            sky[i, j] = amp
        return sky
    
    def compute_visibilities(self,sky:np.ndarray)->np.ndarray:
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky)))

    def sample_uv(self, seed: int =8032003) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mask = np.zeros((self.image_size, self.image_size), dtype=bool)
        n = int(self.fill_frac * self.image_size**2)
        idx = rng.choice(self.image_size**2, size=n, replace=False)
        mask.flat[idx] = True
        mask[self.image_size//2, self.image_size//2] = True
        return mask
    
    def gaussian_weights(self) -> np.ndarray:
        if self.uv_taper_fwhm_frac is None:
            return np.ones((self.image_size, self.image_size), dtype=float)
        y, x = np.indices((self.image_size, self.image_size))
        c = (self.image_size - 1) / 2
        r2 = (x - c)**2 + (y - c)**2
        fwhm_pix = self.uv_taper_fwhm_frac * self.image_size
        sigma = fwhm_pix / 2.355
        w = np.exp(-0.5 * r2 / sigma**2)
        return w / w.max()

    def dirty_image(self, vis: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
        meas = vis * mask * weights
        return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(meas))).real

    def dirty_beam(self, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
        beam_uv = mask * weights
        b = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(beam_uv))).real
        return b / b.max()

    def clean(self, dirty: np.ndarray, psf: np.ndarray,
              gain: float = 0.3, thresh_frac: float = 1e-5,
              max_iters: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
        residual = dirty.copy()
        model = np.zeros_like(dirty)
        centre = np.array(psf.shape) // 2

        peak = residual.max()
        threshold = thresh_frac * peak
        it = 0

        while peak > threshold and it < max_iters:
            idx = np.unravel_index(np.argmax(residual), residual.shape)
            amp = gain * residual[idx]
            model[idx] += amp

            shift = np.array(idx) - centre
            psf_sh = np.roll(np.roll(psf, shift[0], axis=0), shift[1], axis=1)
            residual -= amp * psf_sh

            peak = residual.max()
            it += 1

        print(f"CLEAN completed in {it} iterations; final peak={peak:.3e}")
        return model, residual

    def restore(self, model: np.ndarray, residual: np.ndarray, fwhm_pix: float) -> np.ndarray:
        # build a clean beam
        y, x = np.indices((self.image_size, self.image_size))
        c = (self.image_size - 1) / 2
        r2 = (x - c)**2 + (y - c)**2
        sigma = fwhm_pix / 2.355
        clean_beam = np.exp(-0.5 * r2 / sigma**2)
        clean_beam /= clean_beam.sum()

        return fftconvolve(model, clean_beam, mode="same") + residual

    def simulate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run sky → uv sampling → dirty image & beam → CLEAN → restore.

        Returns
        -------
        sky, dirty, restored, psf
        """
        sky = self.generate_sky()
        vis = self.compute_visibilities(sky)
        mask = self.sample_uv()
        weights = self.gaussian_weights()

        dirty = self.dirty_image(vis, mask, weights)
        psf = self.dirty_beam(mask, weights)
        model, residual = self.clean(dirty, psf)
        restored = self.restore(model, residual, fwhm_pix=self.image_size/200)

        return sky, dirty, restored, psf

