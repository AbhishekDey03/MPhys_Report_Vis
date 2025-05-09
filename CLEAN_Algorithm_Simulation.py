import numpy as np
from scipy.signal import fftconvolve

class RadioImagerCLEAN:
    
    def __init__(self, image_size = 256, fov = 1.0, fill_frac=0.25, uv_taper_fwhm_frac:float|None=None):
        """
        Initializes the RadioImagerCLEAN class.

        Parameters:
        - image_size (int): Number of pixels per side of the image.
        - fov (float): Field of view in radians.
        - fill_frac (float): Fraction of the u-v plane sampled.
        - uv_taper_fwhm_frac (float or None): Optional taper FWHM as a fraction of the image size.
        """

        self.image_size = image_size
        self.fov = fov
        self.fill_frac = fill_frac
        self.uv_taper_fwhm_frac = uv_taper_fwhm_frac
    
    def generate_sky(self, seed: int = 8032003, n_sources: int = 5) -> np.ndarray:
        """
        Generates a sky image with small Gaussian sources instead of pure delta functions.
        
        Parameters:
        - seed (int): Random seed for reproducibility.
        - n_sources (int): Number of sources to generate.

        Returns:
        - np.ndarray: 2D array representing the sky brightness distribution.
        """
        rng = np.random.default_rng(seed)
        sky = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        sigma_pix = self.image_size / 200  # width of Gaussian sources (~few pixels)

        y, x = np.indices((self.image_size, self.image_size))

        for _ in range(n_sources):
            l, m = rng.uniform(0.2, 0.8, size=2)
            amp = rng.uniform(0.5, 1.0)
            i, j = int(l * self.image_size), int(m * self.image_size)
            
            # Create a small 2D Gaussian centered at (i,j)
            r2 = (x - i)**2 + (y - j)**2
            source = amp * np.exp(-0.5 * r2 / sigma_pix**2)

            # Add the Gaussian to the sky
            sky += source

        return sky

    
    def compute_visibilities(self,sky:np.ndarray)->np.ndarray:
        """
        Computes the visibilities by Fourier transforming the sky image.

        Applies an ifftshift before the transform and fftshift afterward to center the u-v plane.

        Parameters:
        - sky (np.ndarray): Input sky image.

        Returns:
        - np.ndarray: Simulated complex visibility data.
        """

        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky)))

    def sample_uv(self,
                n_antennas: int = 10,
                n_times:    int = 240,
                seed:       int =8032003
                ) -> np.ndarray:
        """
        Simulate Earth-rotation aperture synthesis uv-coverage.

        For a given array of n_antennas (randomly laid out in a disk)
        and n_times samples over a full 2pi Earth rotation, this builds
        all baselines and rotates them through hour angle to trace uv-tracks.

        Parameters
        ----------
        n_antennas : int
            Number of antennas in the array.
        n_times : int
            Number of time samples (i.e. rotation steps over 0→2π).
        seed : int
            RNG seed for reproducible antenna layouts.

        Returns
        -------
        mask : np.ndarray (bool)
            Boolean sampling mask of shape (image_size, image_size),
            with True wherever a u-v point is “measured.”
        """
        N      = self.image_size
        centre = N // 2
        rng    = np.random.default_rng(seed)

        # Random antenna positions in a disk of radius N/4 pixels
        phi = rng.uniform(0, 2*np.pi, size=n_antennas)
        r = rng.uniform(0, N/4,      size=n_antennas)
        ant_x = r * np.cos(phi)
        ant_y = r * np.sin(phi)

        # All unique baselines (i < j)
        baselines = []
        for i in range(n_antennas):
            for j in range(i+1, n_antennas):
                # Compute the baseline vector (bx,by)
                baselines.append((ant_x[j] - ant_x[i],
                                ant_y[j] - ant_y[i]))
        baselines = np.array(baselines)

        # Time sampling: We measure each baseline at n_times
        # evenly spaced angles around the circle
        thetas = np.linspace(0, 2*np.pi, n_times, endpoint=False)

        mask = np.zeros((N, N), dtype=bool)
        for (bx, by) in baselines:
            for theta in thetas:
                # rotate baseline by theta
                u =  bx * np.cos(theta) - by * np.sin(theta)
                v =  bx * np.sin(theta) + by * np.cos(theta)

                # map to pixel indices
                ui = int(round(centre + u))
                vi = int(round(centre + v))
                if 0 <= ui < N and 0 <= vi < N:
                    mask[ui, vi] = True

                # also sample the conjugate point (−u, −v)
                ui_c = int(round(centre - u))
                vi_c = int(round(centre - v))
                # Ennsure conjugate point is within bounds
                if 0 <= ui_c < N and 0 <= vi_c < N:
                    # Store conjugate point
                    mask[ui_c, vi_c] = True

        # always include zero-spacing
        mask[centre, centre] = True
        return mask


    
    def gaussian_weights(self) -> np.ndarray:
        """
        Generates a Gaussian taper function over the u-v plane.

        If no taper is specified, returns uniform weights.

        Returns:
        - np.ndarray: 2D array of weights normalized to a maximum of 1.
        """


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
        """
        Constructs the dirty image from masked and weighted visibilities.

        Applies the inverse Fourier transform and returns the real part.

        Parameters:
        - vis (np.ndarray): Full visibility data.
        - mask (np.ndarray): Sampling mask.
        - weights (np.ndarray): Weighting function.

        Returns:
        - np.ndarray: The dirty image.
        """

        meas = vis * mask * weights
        return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(meas))).real

    def dirty_beam(self, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Computes the dirty beam (point spread function) from the sampling mask and weights.

        Applies an inverse Fourier transform and normalizes the peak to 1.

        Parameters:
        - mask (np.ndarray): Sampling mask.
        - weights (np.ndarray): Weighting function.

        Returns:
        - np.ndarray: The dirty beam.
        """

        beam_uv = mask * weights
        b = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(beam_uv))).real
        return b / b.max()

    def CLEAN(self, dirty: np.ndarray, psf: np.ndarray,
              gain: float = 0.1, thresh_frac: float = 1e-10,
              max_iters: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs CLEAN deconvolution on the dirty image.

        Iteratively subtracts scaled and shifted copies of the PSF at peak residual locations.

        Parameters:
        - dirty (np.ndarray): Input dirty image.
        - psf (np.ndarray): Dirty beam (PSF).
        - gain (float): Fraction of residual to subtract each iteration.
        - thresh_frac (float): Stopping threshold as a fraction of initial peak.
        - max_iters (int): Maximum number of iterations.

        Returns:
        - tuple[np.ndarray, np.ndarray]: The CLEAN model and residual images.
        """


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
        """
        Restores the final CLEAN image by convolving the model with a clean beam and adding residuals.

        The clean beam is a normalized 2D Gaussian with specified FWHM.

        Parameters:
        - model (np.ndarray): CLEAN model image.
        - residual (np.ndarray): Residual image.
        - fwhm_pix (float): Clean beam full width at half maximum in pixels.

        Returns:
        - np.ndarray: The restored image.
        """

        y, x = np.indices((self.image_size, self.image_size))
        c = (self.image_size - 1) / 2
        r2 = (x - c)**2 + (y - c)**2
        sigma = fwhm_pix / 2.355
        clean_beam = np.exp(-0.5 * r2 / sigma**2)
        clean_beam /= clean_beam.sum()

        return fftconvolve(model, clean_beam, mode="same") + residual

    def simulate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the full sky simulation, u-v sampling, imaging, deconvolution, and restoration.

        Returns:
        - tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        Sky image, dirty image, restored image, and dirty beam (PSF).
        """

        sky = self.generate_sky()
        vis = self.compute_visibilities(sky)
        mask = self.sample_uv()
        weights = self.gaussian_weights()

        dirty = self.dirty_image(vis, mask, weights)
        psf = self.dirty_beam(mask, weights)
        model, residual = self.CLEAN(dirty, psf)
        restored = self.restore(model, residual, fwhm_pix=self.image_size/200)

        return sky, dirty, restored, psf,mask

