import numpy as np
import pandas as pd

from lightcurver.utilities.lightcurves_postprocessing import group_observations


def test_grouping_multiple_observations():
    # two groups: first two observations close in time,
    # next two observations form the second group.
    df = pd.DataFrame({
        'mjd': [1.0, 1.2, 2.5, 2.6],
        'A_flux': [10.0, 12.0, 20.0, 22.0],
        'A_d_flux': [1.0, 1.0, 2.0, 2.0],
        'other': [100, 200, 300, 400]
    })
    result = group_observations(df, threshold=0.8)
    # expect 2 groups
    assert len(result) == 2, "expected 2 groups based on time differences."

    # for group 1: indices 0 and 1, weights are 1/1^2 = 1 so weighted average = (10+12)/2 = 11.
    np.testing.assert_almost_equal(result.loc[0, 'A_flux'], 11.0, decimal=3)
    # for group 2: indices 2 and 3, weighted average = (20+22)/2 = 21.
    np.testing.assert_almost_equal(result.loc[1, 'A_flux'], 21.0, decimal=3)

    # optional column "other" is averaged as well.
    np.testing.assert_almost_equal(result.loc[0, 'other'], 150.0, decimal=3)
    np.testing.assert_almost_equal(result.loc[1, 'other'], 350.0, decimal=3)


def test_single_observation_group():
    # a single observation should yield a group that, due to sigma clipping, produces nan flux, inf uncertainty.
    df = pd.DataFrame({
        'mjd': [1.0],
        'A_flux': [10.0],
        'A_d_flux': [1.0]
    })
    result = group_observations(df, threshold=0.8)
    assert len(result) == 1, "expected a single group."
    # with one observation, sigma clipping returns an empty set; hence, the else branch applies.
    np.testing.assert_almost_equal(result.loc[0, 'A_flux'], 10.0, decimal=3)
    np.testing.assert_almost_equal(result.loc[0, 'A_d_flux'], 1.0, decimal=3)
    assert result.loc[0, 'A_count_flux'] == 1, "expected count of 1 for a single observation group."


def test_last_group_inclusion():
    # test that the last observation is correctly grouped even if it stands alone.
    df = pd.DataFrame({
        'mjd': [1.0, 1.2, 3.0],
        'A_flux': [10.0, 12.0, 20.0],
        'A_d_flux': [1.0, 1.0, 2.0]
    })
    result = group_observations(df, threshold=0.8)
    # expected groups: group 1 from indices 0,1 and group 2 from index 2.
    assert len(result) == 2, "expected 2 groups when the last observation is isolated."
    # group 1 should aggregate correctly.
    np.testing.assert_almost_equal(result.loc[0, 'A_flux'], 11.0, decimal=5)
    # group 2 is a single observation; expect sigma clipping to remove it.
    np.testing.assert_almost_equal(result.loc[1, 'A_flux'], 20.0, decimal=3)
    np.testing.assert_almost_equal(result.loc[1, 'mjd'], 3.0, decimal=5)

