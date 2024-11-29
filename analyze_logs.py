import pathlib as pl
import numpy as np

logs = ["short_plane_log_interactive_50.npy",
        "short_plane_log_sqpnp_30.npy",
        "short_plane_log_sqpnp_50.npy",
        "short_sphere_log_sqpnp_50.npy",
        "short_sphere_log_sqpnp_30.npy",
        "long_plane_log_sqpnp_50.npy",
        "long_plane_log_sqpnp_50_p_light.npy",
        "long_plane_log_sqpnp_50_p_light_7.npy",
        "short_led_log_sqpnp_50_10.npy",
        "short_led_log_sqpnp_50_10_pad_2.npy",
        "short_plane_log_sqpnp_50_8_pad_2.npy",
        "short_plane_log_sqpnp_50_8_pad_2_50_gain.npy",
        "short_plane_log_sqpnp_50_8_pad_4_0_gain.npy",
        "short_plane_log_sqpnp_50_8_pad_2_0_gain.npy",
        "short_plane_log_sqpnp_50_8_pad_2_50_gain.npy",
        "short_plane_log_sqpnp_50_8_pad_2_25_gain.npy",
        "long_plane_log_sqpnp_50_8_pad_2_0_gain.npy",
        "long_plane_log_sqpnp_50_9_pad_2_50_gain.npy",
        "long_plane_log_sqpnp_50_9_pad_2_0_gain.npy",
        "short_3dplane_log_sqpnp_50_9_pad_2_0_gain.npy",
        "short_plane_log_sqpnp_50_9_pad_2_0_gain.npy",
        "short_7p_log_sqpnp_50_9_pad_2_0_gain.npy",
        "short_p4_log_sqpnp_50_9_pad_2_0_gain.npy",
        "short_p5_log_sqpnp_50_9_pad_2_0_gain.npy",
        "short_p5_large_10_exp_pad_2_0_gain.npy",
        "short_p5_large_9_exp_pad_2_0_gain.npy",
        "short_p5_small_9_exp_pad_2_0_gain.npy"]

base_path = pl.Path("C:\\Users\\v3n0w\\Downloads\\Camera")
log_paths = [base_path / log_name for log_name in logs]

last_n = 30
for log_path in log_paths:
    log_data = np.load(str(log_path))
    rot_x_std, rot_y_std, rot_z_std = np.std(log_data[-last_n:, 2, 0]), np.std(log_data[-last_n:, 2, 1]), np.std(log_data[-last_n:, 2, 2])
    x_std, y_std, z_std = np.std(log_data[-last_n:, 0, 3]), np.std(log_data[-last_n:, 1, 3]), np.std(log_data[-last_n:, 2, 3])

    rot_x_mean, rot_y_mean, rot_z_mean = np.mean(log_data[-last_n:, 2, 0]), np.mean(log_data[-last_n:, 2, 1]), np.mean(log_data[-last_n:, 2, 2])
    x_mean, y_mean, z_mean = np.mean(log_data[-last_n:, 0, 3]), np.mean(log_data[-last_n:, 1, 3]), np.mean(log_data[-last_n:, 2, 3])

    print(log_path.name)
    print(rot_x_std, rot_y_std, rot_z_std)
    print(rot_x_mean, rot_y_mean, rot_z_mean)
    print(x_std, y_std, z_std)
    print(x_mean, y_mean, z_mean)