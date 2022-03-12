import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def read_nii_image(filename: str, normalize=True) -> np.ndarray:
    img_nii = nib.load(filename)
    img_array = img_nii.get_fdata()
    if normalize:
        img_array = normalize_img(img_array)

    return img_array


def visualize_nii_slice(img: np.ndarray, slice_num: int, transpose=True, **kwargs) -> None:
    figsize = kwargs.get("figsize", (4.8, 4.8))
    fraction = kwargs.get("fraction", 0.3)
    fig, axis = plt.subplots(figsize=figsize)
    img_slice = img[..., slice_num] if not transpose else img[..., slice_num].T
    handle = axis.imshow(img_slice, cmap="gray")
    plt.colorbar(handle, ax=axis, fraction=fraction)
    axis.set_title(f"slice {slice_num}/{img.shape[-1]}")
    plt.show()


def normalize_img(img: np.ndarray, eps=2) -> np.ndarray:
    """
    Min-max normalization with quantile eps% and (1 - eps)%.
    """
    eps /= 100
    img_out = np.zeros_like(img)
    for slice_num in range(img.shape[-1]):
        img_slice = img[..., slice_num]
        min_val = np.quantile(img_slice, eps)
        max_val = np.quantile(img_slice, 1 - eps)
        img_out[..., slice_num] = np.clip((img_slice - min_val) / (max_val - min_val), 0, 1)

    return img_out
