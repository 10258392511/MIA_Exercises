import numpy as np

from skimage.restoration import denoise_nl_means, estimate_sigma
from .config import *
from .utils import visualize_nii_slice


class NonLocalMeansDenoiser(object):
    def __init__(self, imgs, ref_img=None, notebook=True):
        """
        Parameters
        ----------
        imgs: list
            list of ndarrays
        ref_img: None or np.ndarray
        """
        assert len(imgs) > 0
        self.imgs = imgs
        self.denoised_img = np.mean(np.stack(self.imgs, axis=0), axis=0)
        self.ref_img = ref_img
        self.notebook = notebook

    def compute_weights(self, patch_vec, patches_vec, tau, mode="l2") -> np.ndarray:
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
        assert mode in ["l2", "cos_sim"]
        # cos similarity
        if mode == "cos_sim":
            dot_product = (patch_vec * patches_vec).sum(axis=-1)  # (B, N) * (N,) -> (B, N) -> (B,)
            den = np.linalg.norm(patch_vec) * np.linalg.norm(patches_vec, axis=-1)  # (1,) * (B,) -> (B,)
            coeffs = dot_product / den
            coeffs = np.exp(coeffs / tau)  # (B,)
            coeffs /= coeffs.sum()  # (B,)

        elif mode == "l2":
            diff = patches_vec - patch_vec  # (B, N) - (N,) -> (B, N)
            coeffs = np.exp(-(diff ** 2).mean(axis=-1) / tau)  # (B,)
            coeffs /= coeffs.sum()  # (B,)
            # print(coeffs)

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
        out_patch_vec = patches_vec.T @ coeffs  # (S^3, B) @ (B,) -> (S^3,)

        return out_patch_vec.reshape(patch.shape)

    def _extract_patches(self, range_dict, win_size):
        # top-left corner
        H, W, D = self.imgs[0].shape
        patches = []
        # # all patches: too slow
        # for i in range(max(range_dict["i_min"], 0), min(range_dict["i_max"], H - win_size)):
        #     for j in range(max(range_dict["j_min"], 0), min(range_dict["j_max"], W - win_size)):
        #         for k in range(max(range_dict["k_min"], 0), min(range_dict["k_max"], D - win_size)):
        #             # for img in self.imgs:
        #             #     patches.append(img[i:i+win_size, j:j+win_size, k:k+win_size])
        #             patches.append(self.denoised_img[i:i+win_size, j:j+win_size, k:k+win_size])

        # sample
        for _ in range(config_num_patches):
            i = np.random.randint(max(range_dict["i_min"], 0), min(range_dict["i_max"], H - win_size))
            j = np.random.randint(max(range_dict["j_min"], 0), min(range_dict["j_max"], W - win_size))
            k = np.random.randint(max(range_dict["k_min"], 0), min(range_dict["k_max"], D - win_size))
            patches.append(self.denoised_img[i:i + win_size, j:j + win_size, k:k + win_size])

        return patches

    def denoise_one_itr(self, win_size, tau):
        H, W, D = self.imgs[0].shape
        counter = 0

        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange

        # # full iteration: too slow
        # pbar_i = trange(H - win_size)
        # for i in pbar_i:
        #     for j in trange(W - win_size, leave=False):
        #         for k in range(D - win_size):
        #             # counter += 1
        #             # if counter % config_print_interval == 0:
        #             #     print(f"current: ({i}, {j}, {k}), rmse: {self.rmse(self.ref_img, self.denoised_img)}")
        #             range_dict = dict(i_min = i - config_half_search_scale * win_size,
        #                               i_max = i + (config_half_search_scale + 1) * win_size,
        #                               j_min = j - config_half_search_scale * win_size,
        #                               j_max = j + (config_half_search_scale + 1) * win_size,
        #                               k_min = k - config_half_search_scale * win_size,
        #                               k_max = k + (config_half_search_scale + 1) * win_size)
        #
        #             patches = self._extract_patches(range_dict, win_size)
        #             patch = self.denoised_img[i:i+win_size, j:j+win_size, k:k+win_size]
        #             self.denoised_img[i:i+win_size, j:j+win_size, k:k+win_size] = \
        #                 self.compute_one_voxel_patch(patch, patches, tau)
        #             del patches
        #
        #     pbar_i.set_description(f"current: {i + 1}/{H}, rmse: {self.rmse(self.ref_img, self.denoised_img)}")


        pbar = trange(config_num_locations)
        for itr in pbar:
            i = np.random.randint(H - win_size)
            j = np.random.randint(W - win_size)
            k = np.random.randint(D - win_size)
            range_dict = dict(i_min=i - config_half_search_scale * win_size,
                              i_max=i + (config_half_search_scale + 1) * win_size,
                              j_min=j - config_half_search_scale * win_size,
                              j_max=j + (config_half_search_scale + 1) * win_size,
                              k_min=k - config_half_search_scale * win_size,
                              k_max=k + (config_half_search_scale + 1) * win_size)

            patches = self._extract_patches(range_dict, win_size)
            patch = self.denoised_img[i:i + win_size, j:j + win_size, k:k + win_size]
            self.denoised_img[i:i + win_size, j:j + win_size, k:k + win_size] = \
                self.compute_one_voxel_patch(patch, patches, tau)
            del patches

            pbar.set_description(f"current: {itr}/{config_num_locations}, "
                                 f"rmse: {self.rmse(self.ref_img, self.denoised_img)}")

    def rmse(self, X, X_hat):
        return np.sqrt(1 / X.size * ((X - X_hat) ** 2).sum())

    def denoise(self, plot=True):
        slice_num = np.random.randint(self.denoised_img.shape[-1])

        doc = {"imgs": [self.denoised_img.copy()]}
        if self.ref_img is not None:
            doc["rmse"] = [self.rmse(self.ref_img, self.denoised_img)]

        for win_size, tau in zip(config_win_size_grid, config_tau_grid):
            self.denoise_one_itr(win_size, tau)
            doc["imgs"].append(self.denoised_img.copy())

            if self.ref_img is not None:
                doc["rmse"].append(self.rmse(self.ref_img, self.denoised_img))

            if self.ref_img is not None:
                print(f"current win_size: {win_size}, tau: {tau:.3f}, rmse: {doc['rmse'][-1]:.3f}")
            else:
                print(f"current win_size: {win_size}, tau: {tau:.3f}")

            if plot:
                self.plot_img(self.denoised_img, slice_num)

        return doc

    @staticmethod
    def plot_img(img, slice_num):
        visualize_nii_slice(img, slice_num)


class NonLocalMeansDenoiserSkimage(object):
    def __init__(self, imgs, ref_img=None, notebook=True):
        """
        Parameters
        ----------
        imgs: list
            list of ndarrays
        ref_img: None or np.ndarray
        """
        assert len(imgs) > 0
        self.imgs = imgs
        self.denoised_img = np.mean(np.stack(self.imgs, axis=0), axis=0)
        self.ref_img = ref_img
        self.notebook = notebook

    def _denoise_one_image(self, img, **kwargs):
        assert kwargs["dim"] in [2, 3]
        if kwargs["dim"] == 3:
            sigma_est = estimate_sigma(img)
            img_denoised = denoise_nl_means(img, patch_size=kwargs["patch_size"],
                                            patch_distance=config_search_dist, h=kwargs["h"] * sigma_est, sigma=sigma_est)
        else:
            img_denoised = img.copy()
            if self.notebook:
                from tqdm.notebook import trange
            else:
                from tqdm import trange
            pbar = trange(img.shape[-1], leave=False)

            for k in pbar:
                sigma_est = estimate_sigma(img[..., k])
                img_denoised[..., k] = denoise_nl_means(img[..., k], patch_size=kwargs["patch_size"],
                                                patch_distance=config_search_dist, h=kwargs["h"] * sigma_est,
                                                sigma=sigma_est)

                if self.ref_img is not None:
                    pbar.set_description(f"rmse: {self.rmse(self.ref_img, img_denoised)}")

        return img_denoised

    def rmse(self, X, X_hat):
        return np.sqrt(1 / X.size * ((X - X_hat) ** 2).sum())

    def denoise(self, plot=True):
        slice_num = np.random.randint(self.denoised_img.shape[-1])

        # imgs = self.imgs + [self.denoised_img]
        imgs = [self.denoised_img]
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        pbar = tqdm(zip(config_win_size_grid, config_h_grid), total=len(config_win_size_grid))
        doc = {"imgs": []}
        if self.ref_img is not None:
            doc["rmse"] = []

        for win_size, h in pbar:
            params = dict(patch_size=win_size, h=h, dim=config_denoise_dim)
            for i in range(len(imgs)):
                # print(f"img {i + 1}/{len(imgs)}")
                imgs[i] = self._denoise_one_image(imgs[i], **params)

            imgs[-1] = np.stack(imgs, axis=0).mean(axis=0)
            doc["imgs"].append(imgs[-1].copy())
            self.denoised_img = imgs[-1].copy()
            if self.ref_img is not None:
                doc["rmse"].append(self.rmse(self.ref_img, self.denoised_img))
            pbar.set_description(f"win_size: {win_size}, h: {h}, "
                                 f"rmse: {doc['rmse'][-1]}")

        if plot:
            self.plot_img(self.denoised_img, slice_num)

        return doc

    @staticmethod
    def plot_img(img, slice_num):
        visualize_nii_slice(img, slice_num)
