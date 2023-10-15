import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def hyperbola(x):
    return 1 / x

def polynomial_3(x):
    return 3 * x**3 - 2 * x**2 + 5 * x + 1

#generate random points
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y_hyperbola = hyperbola(X).ravel()
y_polinomial_3 = polynomial_3(X).ravel()

regr_hyperbola = RandomForestRegressor(n_estimators=100, random_state=0)
regr_polynomial_3 = RandomForestRegressor(n_estimators=100, random_state=0)

regr_hyperbola.fit(X, y_hyperbola)
regr_polynomial_3.fit(X, y_polinomial_3)

y_pred_hyperbola = regr_hyperbola.predict(X)
y_pred_polynomial_3 = regr_polynomial_3.predict(X)

mse_hyperbola = mean_squared_error(y_hyperbola, y_pred_hyperbola)
mse_polynomial_3 = mean_squared_error(y_polinomial_3, y_pred_polynomial_3)

print(f"MSE for hyperbola: {mse_hyperbola:.4f}")
print(f"MSE for Polynomial: {mse_polynomial_3:.4f}")

plt.figure()
plt.scatter(X, y_hyperbola, c="k", label="Data")
plt.plot(X, y_pred_hyperbola, c="g", label="Prediction", linewidth=2)
plt.title("RandomForestRegressor for Hyperbola")
plt.legend()
plt.show()

plt.figure()
plt.scatter(X, y_polinomial_3, c="k", label="Data")
plt.plot(X, y_pred_polynomial_3, c="g", label="Prediction", linewidth=2)
plt.title("RandomForestRegressor for Polynomial 3")
plt.legend()
plt.show()
