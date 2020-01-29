import numpy as np
from data_engineering.dask_image_stats_collector import stats_for_numpy_images

# Test slicing
arr = np.stack([np.ones((120, 120, 3)) for x in range(100)])
print(arr.shape)
assert arr[:, :, :, 0].sum() == 120 * 120 * 100

# Test method output
files = [str(x) for x in range(100)]
stats = stats_for_numpy_images(files, use_test_data=True)

for stat in ["mean", "min", "max"]:
    for band_name, expected_value in [("red", 1), ("blue", 2), ("green", 3)]:
        assert stats[stat][band_name] == expected_value

assert sum(stats['std'].values()) == 0
