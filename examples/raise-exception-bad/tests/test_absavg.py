#
# Unit tests for absavg.py
#

import pytest
from sample.avg import Avg

def test_allpositive():
    val = Avg.absAvg([1, 2, 3, 4, 5])
    assert val == 3


def test_allnegative():
    val = Avg.absAvg([-1, -2, -3, -4, -5])
    assert val == 3

# These are weak tests because they assert a general Exception gets caught, which
# could come from several kinds of problem.

def test_input_none():
    with pytest.raises(Exception):
        val = Avg.absAvg(None)
        assert False


def test_input_empty():
    with pytest.raises(Exception):
        val = Avg.absAvg([])
        assert False
	
