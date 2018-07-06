import hlm
import numpy as np


nobs = 50
ntar = 100
nfac = 10

Z = np.random.standard_normal((ntar, nfac)) + 1
X = np.random.standard_normal((nfac, nobs))
Y = Z @ X + np.random.standard_normal((ntar, nobs))

s0 = (.01, .01)
S0 = (nfac, nfac * np.identity(nfac))

map = hlm.estimate(1000, Y, X, S0, s0)
samples = hlm.sample(1000, Y, X, S0, s0)