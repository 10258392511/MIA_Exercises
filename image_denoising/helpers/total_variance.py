import numpy as np
import matplotlib.pyplot as plt

from skimage.restoration import denoise_tv_chambolle
from .config import *
from .utils import rmse


class TVDenoiser(object):
    def __init__(self, imgs, ref_img=None, notebook=True):
        self.imgs = imgs
        self.denoised_img = np.mean(np.stack(self.imgs, axis=0), axis=0)
        self.ref_img = ref_img
        self.notebook = notebook

    def compute_diff_weights(self, img=None, plot=True):
        """
        Only for evaluation
        """
        assert self.ref_img is not None
        if img is None:
            img = self.denoised_img
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(config_weights_grid)

        doc = []
        for weight in pbar:
            # print(weight)
            img_denoised = denoise_tv_chambolle(img, weight)
            doc.append(rmse(self.ref_img, img_denoised))
            pbar.set_description(f"weight: {weight}, rmse: {doc[-1]}")

        if plot:
            plt.plot(config_weights_grid, doc)
            plt.show()

        return doc

