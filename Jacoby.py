import numpy as np
import math
import types


precision = 10**-8


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def check_symmetric(m):
    for i in np.arange(m.shape[0]):
        for j in np.arange(i + 1, m.shape[1]):
            if m[i, j] != m[j, i]:
                raise Exception("Matrix is not symmetric")


class JacobiSolver:
    """
    Finds eigenvalues and eigenvectors using Jacobi method.
    Constructor takes a symmetric matrix
    "solve()" method returns vector of eigenvalues and vector of eigenvectors
    """

    def __init__(self, matrix_p):
        check_symmetric(matrix_p)
        self.m = np.matrix(matrix_p)

        # Create matrix with eigenvectors and make it identity matrix
        self.eigenvectors = np.zeros(shape=self.m.shape)
        for i in xrange(self.eigenvectors.shape[0]):
            self.eigenvectors[i, i] = 1

        self.eigenvalues = np.zeros(self.m.shape[0])

    # Public methods:
    def solve(self):
        return self.eigenvalues, self.eigenvectors

    # Private methods:
    def _rotate(self, alpha, i, j):
        cosa = math.cos(alpha)
        sina = math.sin(alpha)

        # for row in np.arange(self.shape[0]):

        for col in np.arange(self.m.shape[1]):
            self.m[i, col] = self.m[i, col] * cosa + self.m[j][col] * sina
            self.m[j, col] = self.m[j, col] * cosa + self.m[i][col] * sina

    def _ind_max(self):
        i_max = 0
        j_max = 0
        _max = self.m[0, 0]

        for i in np.arange(self.m.shape[0]):
            for j in np.arange(i + 1, self.m.shape[1]):
                if self.m[i, j] > _max:
                    _max = self.m[i, j]
                    i_max = i
                    j_max = j
        return i_max, j_max

    def _compute_alpha(self, i, j):
        if (self.m[j, j] - self.m[i, i]) <= precision:
            return sign(self.m[i, j]) * math.pi/2.0
        else:
            return 0.5 * math.atan(2.0 * self.m[i, j] / (self.m[j, j] - self.m[i, i]))


def main():
    data = np.loadtxt("matrix.txt")
    inp_m = np.matrix(data, dtype=np.float)

    J = JacobiSolver(inp_m)
    values, vectors = J.solve()

    print "Eigenvalues:\n", values
    print "Eigenvectors:\n", vectors


if __name__ == "__main__":
    main()

