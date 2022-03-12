import numpy as np


config_start_tau = 5
config_end_tau = 0.5
config_start_win_size = 21
config_end_win_size = 3
search_scale = 3  # e.g. Current location: 5: 2, 3, 4 | 5, 6, 7 | 8, 9, 10

config_win_size_grid = np.arange(config_start_win_size, config_end_win_size + 1, 2)
config_tau_grid = np.linspace(config_start_tau, config_end_tau, config_win_size_grid.shape[0])
