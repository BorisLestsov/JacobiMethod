import numpy as np
from numpy import matrix
from numpy import linalg as lg
import types

data = np.loadtxt("matrix.txt")
m = matrix(data, dtype=np.float)

def i_max(self):
    idx = np.nanargmax(self, axis=None)
    return np.unravel_index(idx, self.shape)
m.i_max = types.MethodType(i_max, m)

