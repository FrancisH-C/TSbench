# from TSbench import models
#
# # from statsmodels.tsa.arima_process import arma_generate_sample
# #
# # import numpy as np
#
# from TSbench import models
#
# ar = np.array([1.0, 1.0])
# ma = np.array([1.0, 1.0])
#
# arma = models.ARMA(ar=ar, d=0, ma=ma)
#
# arma.rg
#
# from randomgen import Generator, Xoshiro256
#
# rg = [Generator(Xoshiro256(1234)) for _ in range(10)]
# # Advance each Xoshiro256 instance by i jumps
# for i in range(10):
#     rg[i].bit_generator.jump(i)
#
# from TSbench.utils.corr_mat import Corr_mat
# from numpy.random import Generator
# from randomgen import Xoshiro256
#
# rg = Generator(Xoshiro256(1234))
# rg.exponential(scale=0.1, size=1)
#
# x = Corr_mat(dim=2)
# x.mat
#
# from statsmodels.tsa.arima_process import arma_generate_sample
#
# import numpy as np
#
# from numpy.random import Generator, PCG64, MT19937, default_rng
# from randomgen import Xoshiro256
#
# from TSbench import models
#
#
# n = 10
# seed = 2134
# ar = np.array([1.0])
# ma = np.array([1.0])
# arma_model = models.ARMA(ar=ar, d=0, ma=ma, rg=Generator(MT19937(seed)))
# # arma_model = models.ARMA(ar=ar, d=0, ma=ma, rg=default_rng(seed))
# generated = arma_model.generate(n)["returns"]
#
# np.random.seed(seed)
# sm_generated = np.transpose(np.array(arma_generate_sample(ar, ma, n), ndmin=2))
#
# # Are all points equals for the two models?
# eq = np.linalg.norm(generated - sm_generated) <= (10 ** (-1))
# print("test")
# print(generated)
# print("test")
# print(sm_generated)
# # assert eq
#
# np.random.default_rng(seed)
