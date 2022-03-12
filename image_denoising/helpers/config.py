import numpy as np

# self defined
config_start_tau = 0.01
config_end_tau = 0.001
config_start_win_size = 11
config_end_win_size = 3
config_search_scale = 3  # e.g. Current location: 5: 2, 3, 4 | 5, 6, 7 | 8, 9, 10
config_print_interval = 100
config_num_patches = 5
config_num_locations = int(1e3)

# skimage
config_search_dist = 4
config_start_h = 0.3
config_end_h = 0.05
config_denoise_dim = 3

# TV
config_min_weight = 0.038
config_max_weight = 0.042
config_weights_pts = 5

# config_win_size_grid = np.arange(config_end_win_size, config_start_win_size + 1, 2)[::-1]
config_win_size_grid = np.array([5, 5, 3, 3])
config_tau_grid = np.linspace(config_end_tau, config_start_tau, config_win_size_grid.shape[0])[::-1]
config_half_search_scale = (config_search_scale - 1) // 2
config_h_grid = np.linspace(config_end_h, config_start_h, config_win_size_grid.shape[0])[::-1]
config_weights_grid = np.linspace(config_min_weight, config_max_weight, config_weights_pts)
