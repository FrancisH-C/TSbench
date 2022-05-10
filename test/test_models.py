from TSbench import models
from numpy.random import Generator
from randomgen import Xoshiro256

def test_arma():
    """Comparisson with statmodels."""
    ar = [0.5, 0.33]
    ma = [0.5, 0.6]
    rg = Generator(Xoshiro256(2134))
    arma_model = models.ARMA(ar=ar, ma=ma, rg=rg)
    arma_model.generate(200)["returns"]

def test_R():
    """Using R packages"""
    models.rGARCH()
