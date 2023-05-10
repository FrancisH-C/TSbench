import pytest

from TSbench import TSmodels
from numpy.random import Generator
from randomgen import Xoshiro256


def test_simple():
    cnst_model = TSmodels.Constant(1)
    cnst_model.generate(200)["returns"]


test_simple()


@pytest.mark.R
def test_R():
    """Using R packages"""
    TSmodels.rGARCH()


def test_arma():
    """Comparisson with statmodels."""
    ar = [0.5, 0.33]
    ma = [0.5, 0.6]
    rg = Generator(Xoshiro256(2134))
    arma_model = TSmodels.ARMA(ar=ar, ma=ma, rg=rg)
    arma_model.generate(200)["returns"]
