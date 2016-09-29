import numpy as np
import math

precision = 10**-8

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def check_symmetric(m):
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise Exception("Matrix is not symmetric")
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
        self.eigenvectors = np.zeros_like(self.m)
        for i in np.arange(self.eigenvectors.shape[0]):
            self.eigenvectors[i, i] = 1

        self.eigenvalues = np.zeros(self.m.shape[0])

    # Public methods:
    def solve(self):
        if self.m.shape[0] == 1:
            return [self.m[0, 0]], np.array([1])

        i_max, j_max = self._find_ind_max()
        while abs(self.m[i_max, j_max]) >= precision:
            a = self._compute_alpha(i_max, j_max)
            self._rotate(alpha=a, i=i_max, j=j_max)
            i_max, j_max = self._find_ind_max()

        for i in np.arange(self.m.shape[0]):
            self.eigenvalues[i] = self.m[i, i]
        return self.eigenvalues, self.eigenvectors

    # Private methods:
    def _rotate(self, alpha, i, j):
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)

        self._tmp = np.copy(self.m)
        for row in np.arange(self.m.shape[1]):
            self._tmp[row, i] = self.m[row, i] * cos_a + self.m[row, j] * sin_a
            self._tmp[row, j] = self.m[row, j] * cos_a - self.m[row, i] * sin_a

        self.m = np.copy(self._tmp)
        for col in np.arange(self.m.shape[0]):
            self.m[i, col] = self._tmp[i, col] * cos_a + self._tmp[j, col] * sin_a
            self.m[j, col] = self._tmp[j, col] * cos_a - self._tmp[i, col] * sin_a

        self._tmp = np.copy(self.eigenvectors)
        for col in np.arange(self.m.shape[0]):
            self.eigenvectors[i, col] = self._tmp[i, col] * cos_a + self._tmp[j, col] * sin_a
            self.eigenvectors[j, col] = self._tmp[j, col] * cos_a - self._tmp[i, col] * sin_a

    def _find_ind_max(self):
        i_max = 0
        j_max = 1
        _max = abs(self.m[0, 1])

        for i in np.arange(self.m.shape[0]):
            for j in np.arange(i + 1, self.m.shape[1]):
                if abs(self.m[i, j]) > _max:
                    _max = abs(self.m[i, j])
                    i_max = i
                    j_max = j
        return i_max, j_max

    def _compute_alpha(self, i, j):
        if (self.m[i, i] - self.m[j, j]) <= precision:
            return sign(self.m[i, j]) * math.pi/4.0
        else:
            return 0.5 * math.atan(2.0 * self.m[i, j] / (self.m[i, i] - self.m[j, j]))


def main():
    data = np.loadtxt("matrix.txt")
    inp_m = np.matrix(data, dtype=np.float)

    J = JacobiSolver(inp_m)
    values, vectors = J.solve()

    print "Eigenvalues:"
    for i, lbd in enumerate(values):
        print "{}{} = {}".format("lambda", i + 1, lbd)
    print "Eigenvectors:"
    for i, vec in enumerate(vectors):
        print "{}{}: {}".format("eigenvector for lambda", i + 1, vec)


if __name__ == "__main__":
    main()

