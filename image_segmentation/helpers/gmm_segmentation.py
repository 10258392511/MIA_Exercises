import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture


COMPONENTS = ("BG", "CSF", "GM", "WM")


def segmentation(img_array, num_components=4, **kwargs):
    gmm = GaussianMixture(n_components=num_components, covariance_type=kwargs.get("covariance_type", "full"))
    img_array_data = img_array.reshape(-1, 1)
    gmm.fit(img_array_data)
    img_seg_data = gmm.predict(img_array_data)
    img_seg = img_seg_data.reshape(img_array.shape)

    if_plot = kwargs.get("if_plot", False)
    if if_plot:
        label_indices = map_labels_(gmm)
        # print(label_indices)
        segmentation_plot_(img_seg, label_indices)

    return img_seg, gmm


def map_labels_(gmm):
    indices = np.argsort(gmm.means_.ravel())

    return indices


def segmentation_plot_(img_seg: np.ndarray, label_indices, **kwargs):
    fig_size = kwargs.get("figsize", (7.2, 7.2))
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    labels = label_indices.reshape(2, 2)

    counter = 0
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axis = axes[i, j]
            label = labels[i, j]
            mask = (img_seg == label).astype(np.float32)
            axis.imshow(mask, cmap="gray")
            axis.set_title(COMPONENTS[counter])
            counter += 1

    fig.tight_layout()
    plt.show()
    plt.close()
