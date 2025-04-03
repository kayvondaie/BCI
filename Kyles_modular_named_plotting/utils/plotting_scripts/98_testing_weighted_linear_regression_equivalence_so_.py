import matplotlib.pyplot as plt
import numpy as np

def plot_cell_98():
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LinearRegression
    from scipy.stats import linregress

    np.random.seed(111111)

    x = np.arange(5, 30, dtype=np.float32)
    y = 0.3 * x + np.random.normal(size=x.shape)

    # weights = np.abs(np.random.normal(size=x.shape))
    weights = None

    reg = LinearRegression()
    reg.fit(x[:, np.newaxis], y[:, np.newaxis], sample_weight=weights)

    print('LinearRegression slope:', reg.coef_[0, 0])

    if weights is not None:
    #     x_copy = np.sqrt(weights) * x
    #     y_copy = np.sqrt(weights) * y
        x_copy = np.matmul(np.diag(np.sqrt(weights)), x)
        y_copy = np.matmul(np.diag(np.sqrt(weights)), y)    
    else:
        x_copy = x.copy()
        y_copy = y.copy()

    X = x_copy[:, np.newaxis]
    Y = y_copy[:, np.newaxis]
    print('Inverse slope:', np.linalg.inv(X.T @ X) @ X.T @ Y)

    slope, intercept, rvalue, pvalue, se = linregress(x_copy, y_copy)
    print('linregress slope:', slope)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.scatter(x, y, color='k', marker='.')

    fig.show()

if __name__ == '__main__':
    plot_cell_98()
