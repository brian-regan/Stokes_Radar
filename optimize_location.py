from bayes_opt import BayesianOptimization
import numpy as np
from plotutil import Figure
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn

MAX_N_RADARS = 3
N_CURRENT_RADARS = 0
# init_points and n_iter for each n_radars
INIT_POINTS = [10, 10, 10]
N_ITER = [5, 5, 5]

xy_range = ((0, 3), (0, 3))

grid_for_optimization = (np.linspace(*xy_range[0], 11),
                         np.linspace(*xy_range[1], 11))

grid_for_QR = np.meshgrid(np.linspace(*xy_range[0], 21),
                          np.linspace(*xy_range[1], 21))

plt.rcParams['image.cmap'] = 'jet'

# optimization

def posterior(bo, x):
    bo.gp.fit(bo.X, bo.Y)
    mu, sigma = bo.gp.predict(x, return_std=True)
    return mu, sigma


def get_optimal_location(bo, n_radars, grid=grid_for_optimization):
    grid_multi = []
    for _ in range(n_radars):
        grid_multi.extend(grid)
    points = np.meshgrid(*grid_multi)
    points_flat = np.array([x.reshape(-1) for x in points]).T
    mu, sigma = posterior(bo, points_flat)
    return points_flat[np.argmax(mu)]


def optimize_for_n(Q, R, n_radars, init_points, n_iter):
    def S(*args, **kwargs):
        return np.sum((Q * R(*args, **kwargs)).reshape(-1, 1))

    ranges = {}
    for i in range(n_radars):
        ranges.update({'x' + str(i): xy_range[0], 'y' + str(i): xy_range[1]})
    bo = BayesianOptimization(S, ranges)
    bo.maximize(init_points=init_points, n_iter=n_iter)
    x_optimum = get_optimal_location(bo, n_radars)
    plot_result(Q, R(*x_optimum), x_optimum, n_radars)
    return S(*x_optimum)


def optimize(Q, R, max_n=MAX_N_RADARS):
    def S(*args, **kwargs):
        return np.sum((Q * R(*args, **kwargs)).reshape(-1, 1))

    score = [S()]
    for n in range(1, max_n + 1):
        score.append(optimize_for_n(Q, R, n, init_points=INIT_POINTS[n+1],
                                    n_iter=N_ITER[n+1]))

    fig = Figure(1)
    fig[0].plot(list(range(N_CURRENT_RADARS, N_CURRENT_RADARS + max_n + 1)), score)
    fig[0].set_ylim([0, np.max(np.array(score))])
    fig[0].set_xlabel('#radars')
    fig[0].set_ylabel('score')
    fig.close('figs', 'score')

# figure

def set_lim(axis, grid, p=0.1):
    x_lim = (np.min(grid[0].reshape(-1)), np.max(grid[0].reshape(-1)))
    y_lim = (np.min(grid[1].reshape(-1)), np.max(grid[1].reshape(-1)))

    def extend(x_a, x_b, p=p):
        return x_a - (x_b - x_a) * p, x_b + (x_b - x_a) * p

    axis.set_xlim(extend(*x_lim))
    axis.set_ylim(extend(*y_lim))


def plot_result(Q, R, x_optimum, n_radars, grid=grid_for_QR):
    levels = 20
    fig = Figure(1, 2, figsize=(8, 6))
    fig[0].contourf(*grid, Q, levels)
    fig[1].contourf(*grid, R, levels)
    fig[0].set_ylabel('Q(x)', fontdict={'size': 20})
    # fig[1].set_ylabel('R(x)', fontdict={'size':20})
    fig[1].set_ylabel('R(x)', fontdict={'size': 20})
    for x, y in zip(*[iter(x_optimum)] * 2):
        fig[1].plot(x, y, 'D', markersize=8, label=u'Last observation', color='gold')

    for axis in fig:
        set_lim(axis, grid)
    fig.close('figs', f'optimal_location_for_{n_radars}')


def main():
    Q = np.random.random(grid_for_QR[0].shape)

    def R(x_c, y_c, grid=grid_for_QR):
        a = 3
        r = a ** 2 - ((grid[0] - x_c) ** 2 + (grid[1] - y_c) ** 2)
        r[r < 0] = 0
        return r / 20

    def R_multi(*args, **kwargs):
        R_each = []
        for i in range(int(len(args)/2)):
            x = args[i*2]
            y = args[i*2+1]
            R_each.append(R(x, y))

        for i in range(int(len(kwargs) / 2)):
            x = kwargs['x' + str(i)]
            y = kwargs['y' + str(i)]
            R_each.append(R(x, y))

        if len(R_each) == 0:
            return 0

        R_total = R_each[0]
        for R_temp in R_each[1:]:
            R_total = np.maximum(R_total, R_temp)
        return R_total

    optimize(Q, R_multi)


if __name__ == '__main__':
    main()