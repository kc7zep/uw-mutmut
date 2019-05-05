#
# Unit tests for absavg.py
#

import pytest
from sample.avg import Avg
from sample.avg import EmptyInputException


def test_allpositive():
    val = Avg.absAvg([1, 2, 3, 4, 5])
    assert val == 3


def test_allnegative():
    val = Avg.absAvg([-1, -2, -3, -4, -5])
    assert val == 3

# Changed exception throw test to check a more specific exception type than Exception.
# This change was added after discovering a weakness in original design which was highlighted by adding the
# raise suppression mutation.


def test_input_none():
    with pytest.raises(EmptyInputException):
        val = Avg.absAvg(None)
        assert False


def test_input_empty():
    with pytest.raises(EmptyInputException):
        val = Avg.absAvg([])
        assert False
	
