import cvxpy
import matplotlib.pyplot as plt

from data_util import DataLoader, add_noise

class TVDenoise():
	def __init__(self, lamb):
		self.lamb = lamb

	def fit(self, img):
		m,n = img.shape
		X = cvxpy.Variable([m, n])
		E = cvxpy.Variable(shape=[m, n])
		p = cvxpy.Problem(
			cvxpy.Minimize(
				cvxpy.sum(cvxpy.abs(X[1:, :] - X[:-1, :])) + cvxpy.sum(
					cvxpy.abs(X[:, 1:] - X[:, :-1])) + 2. * cvxpy.sum(cvxpy.abs(E))
			),
			[img == X + E]
		)
		p.solve()
		return X.value


