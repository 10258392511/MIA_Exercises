import numpy as np
import matplotlib.pyplot as plt
import pickle
import helpers.config as config

from helpers.non_local_means import NonLocalMeansDenoiserSkimage

from helpers.utils import read_nii_image


if __name__ == '__main__':
    filename1 = "data/MPR_500um_32_run01_nuCorrect_aligned.nii"
    img1 = read_nii_image(filename1)
    filename2 = "data/MPR_500um_32_run02_nuCorrect_aligned.nii"
    img2 = read_nii_image(filename2)
    filename_ref = "data/MPR_500um_32_avg.norm.norm.nii"
    img_ref = read_nii_image(filename_ref)

    args = dict(imgs=[img1, img2], ref_img=img_ref, notebook=False)
    denoiser = NonLocalMeansDenoiserSkimage(**args)
    doc = denoiser.denoise(False)

    with open("doc_3d.pkl", "wb") as wf:
        pickle.dump(doc, wf)
