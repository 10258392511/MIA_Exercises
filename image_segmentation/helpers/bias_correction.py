import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def bias_field_correction(img_array: np.ndarray, num_levels=5, num_iters=50, if_plot=False):
    """
    img_array: [0, 1]
    """
    img = sitk.GetImageFromArray(img_array)
    img = sitk.Cast(img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iters] * num_levels)
    img_corrected = corrector.Execute(img)
    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    img_corrected_array = sitk.GetArrayFromImage(img_corrected)
    log_bias_field_array = sitk.GetArrayFromImage(log_bias_field)

    if if_plot:
        bias_field_correction_plot_(img_array, img_corrected_array, log_bias_field_array)

    return img_corrected_array, log_bias_field_array


def bias_field_correction_plot_(img_array, img_corrected_array, log_bias_field_array, **kwargs):
    fig_size = kwargs.get("figsize", (7.2, 7.2))
    bias_field_array = np.exp(log_bias_field_array)
    img_biased_back_array = img_corrected_array * bias_field_array
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    imgs = [[img_array, img_corrected_array],
            [bias_field_array, img_biased_back_array]]
    titles = [["original", "corrected"],
              ["bias field", f"biased back, diff: {np.linalg.norm(img_biased_back_array - img_array):.5f}"]]

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axis = axes[i, j]
            img_iter = imgs[i][j]
            title_iter = titles[i][j]
            if not (i == 1 and j == 1):
                img_iter = img_iter
            handle = axis.imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axis)
            axis.set_title(title_iter)

    fig.tight_layout()
    plt.show()
    plt.close()
