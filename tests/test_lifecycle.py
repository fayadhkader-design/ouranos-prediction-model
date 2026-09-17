import numpy as np
import pytest
pytest.importorskip('torch')
from src.neural.lifecycle import lifecycle_alarm


def test_recovers_only_after_quiet_and_stops_at_missingness():
    score=np.array([0,9,4,3,0,0,9,4,np.nan,4,0],float)
    alarm=lifecycle_alarm(score,8,1,2,15)
    np.testing.assert_array_equal(alarm,[False,True,True,True,True,False,True,True,False,False,False])


def test_timeout_and_new_trigger_refresh():
    score=np.array([9,4,4,4,4,4,4],float)
    np.testing.assert_array_equal(lifecycle_alarm(score,8,1,1,3),[True,True,True,False,False,False,False])
    score[2]=9
    np.testing.assert_array_equal(lifecycle_alarm(score,8,1,1,3),[True,True,True,True,True,False,False])


def test_lifecycle_is_prefix_causal_and_never_invents_a_trigger():
    rng=np.random.default_rng(1);score=rng.normal(0,4,500);score[99]=np.nan
    full=lifecycle_alarm(score,8,1,5,15)
    for stop in [1,50,99,100,101,300]:
        np.testing.assert_array_equal(full[:stop],lifecycle_alarm(score[:stop],8,1,5,15))
    assert not lifecycle_alarm(np.ones(100),8,1,5,15).any()
