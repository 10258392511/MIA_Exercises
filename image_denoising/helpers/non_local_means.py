import numpy as np
import config


class NonLocalMeansDenoiser(object):
    def __init__(self, imgs, ref_img=None):
        """
        Parameters
        ----------
        imgs: list
            list of ndarrays
        ref_img: None or np.ndarray
        """
        assert len(imgs) > 0
        self.imgs = imgs
        self.denoised_img = np.zeros_like(imgs[0])
        self.ref_img = ref_img

    def compute_weights(self, patch_vec, patches_vec, tau) -> np.ndarray:
        """
        Implement different distance metric here.

        Parameters
        ----------
        patch_vec: np.ndarray
            Vectorized patch, of shape (num_vox_in_window,) (i.e (N,))
        patches_vec: np.ndarray
            Vectorized patches to compare, of shape (num_patches, num_vox_in_window) (i.e (B, N))
        tau: float
            Temperature parameter

        Returns
        -------
        coeffs: np.ndarray
            Normalized weights for each patch to compare
        """
        # cos similarity
        dot_product = (patch_vec * patches_vec).sum(axis=-1)  # (B, N) * (N,) -> (B, N) -> ((B,))
        den = np.linalg.norm(patch_vec) * np.linalg.norm(patches_vec, axis=-1)  # (1,) * (B,) -> (B,)
        coeffs = dot_product / den
        coeffs = np.exp(coeffs / tau)  # (B,)
        coeffs /= coeffs.sum()  # (B,)

        return coeffs

    def compute_one_voxel_patch(self, patch, patches, tau) -> float:
        """
        Parameters
        ----------
        patch: np.ndarray
            Current patch
        patches: list
            list of patches to compare
        tau: float
            Temperature parameter
        """
        patch_vec = patch.flatten()
        patches_vec = np.stack(patches, axis=0).reshape(len(patches), -1)  # [(S, S, S)...] -> (B, S, S, S) -> (B, S^3)
        coeffs = self.compute_weights(patch_vec, patches_vec, tau)  # (B,)
        out_patch_vec = patch_vec.T @ coeffs  # (S^3, B) @ (B,) -> (S^3,)

        return out_patch_vec.reshape(patch.shape)

    def denoise_one_itr(self, win_size, tau):
        pass

    def denoise(self):
        pass
