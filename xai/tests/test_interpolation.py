import unittest
import xarray as xr
import numpy as np
import sys
sys.path.append('.')
from libs.preprocessing_functions import interpolate_features

class Testinterpolation(unittest.TestCase):
    def test_interpolation(self):
        # load_example_data
        ds = xr.open_dataset('example_data/cml_raw_example.nc').load()
        ds = interpolate_features(
            ds,
            features=['TL_1', 'TL_2'],
            interpolation_max='5min',
            method="linear",
        )
        interpolated = ds.TL_1.sel(sensor_id='BY0452_2_BY4006_3').sel(time=slice('2019-07-03T19:15:00.000000000','2019-07-03T19:35:00.000000000')).values

        truth = np.array(
            [65.2, 64.9, 65.2, 64.9, 64.9, 65.2, 65.2, 64.9, 64.9, 64.9, 64.9,
       64.9, 64.9, 64.9, 64.9, 64.9, 64.9, 64.9, 64.9, 64.9, 64.9]
        )
        np.testing.assert_almost_equal(interpolated, truth)
