from bayes_opt import BayesianOptimization
import numpy as np
from plotutil import Figure
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn

MAX_N_RADARS = 3
N_CURRENT_RADARS = 0
# init_points and n_iter for each n_radars
INIT_POINTS = [10, 20, 40]
N_ITER = [10, 10, 10]

xy_range = ((0, 1), (0, 1))
POINTS = (334, 418)

grid_for_optimization = (np.linspace(*xy_range[0], 11),
                         np.linspace(*xy_range[1], 11))

grid_for_QR = np.meshgrid(np.linspace(*xy_range[0], POINTS[0]),
                          np.linspace(*xy_range[1], POINTS[1]))

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


def optimize_for_n(Q, R, top, n_radars, j, init_points, n_iter):

    # score function
    def S(*args, **kwargs):
        return np.sum((Q * R(*args, **kwargs)).reshape(-1, 1))

    # ranges should be {'x0': (0, 1), 'y0': (0, 1), 'x1': (0, 1), 'y1': (0, 1), ... }
    ranges = {}
    for i in range(n_radars):
        ranges.update({'x' + str(i): xy_range[0], 'y' + str(i): xy_range[1]})

    # bayesian optimization
    bo = BayesianOptimization(S, ranges)
    bo.maximize(init_points=init_points, n_iter=n_iter)

    # get optimal location
    x_optimum = get_optimal_location(bo, n_radars)

    # plot result
    plot_result(Q, R(*x_optimum), x_optimum, n_radars, top, j)

    # save data
    np.savez(f'dat/optimal_location_for_{n_radars}_{i}.npz',
             Q=Q, R=R(*x_optimum), x_optimum=x_optimum, n_radars=n_radars, top=top)
    return S(*x_optimum)


def optimize(Q, R, top, i, max_n=MAX_N_RADARS):
    # iterate over different numbers of radars (including n=0)

    def S(*args, **kwargs):
        return np.sum((Q * R(*args, **kwargs)).reshape(-1, 1))

    score = [S()]
    for n in range(1, max_n + 1):
        score.append(optimize_for_n(Q, R, top, n, i, init_points=INIT_POINTS[n-1],
                                    n_iter=N_ITER[n-1]))

    fig = Figure(1)
    fig[0].plot(list(range(N_CURRENT_RADARS, N_CURRENT_RADARS + max_n + 1)), score)
    fig[0].set_ylim([0, np.max(np.array(score))])
    fig[0].set_xlabel('#radars')
    fig[0].set_ylabel('score')
    fig.close('figs', f'score_{i}')
    return score

# figure

def set_lim(axis, grid, p=0.1):
    x_lim = (np.min(grid[0].reshape(-1)), np.max(grid[0].reshape(-1)))
    y_lim = (np.min(grid[1].reshape(-1)), np.max(grid[1].reshape(-1)))

    def extend(x_a, x_b, p=p):
        return x_a - (x_b - x_a) * p, x_b + (x_b - x_a) * p

    axis.set_xlim(extend(*x_lim))
    axis.set_ylim(extend(*y_lim))


def draw_image(matrix, axis):
    max = matrix.reshape(-1).max()
    min = matrix.reshape(-1).min()
    if max == min:
        matrix_normalized = np.ones_like(matrix)
    else:
        matrix_normalized = (matrix - min) / (max - min)
    axis.imshow(matrix_normalized, interpolation=None)


def plot_result(Q, R, x_optimum, n_radars, top, i):
    fig = Figure(1, 3, figsize=(12, 4))
    draw_image(Q, fig[0])
    draw_image(top, fig[1])
    draw_image(R, fig[2])
    fig[0].set_title('Q(x)', fontdict={'size': 20})
    fig[1].set_title('topography', fontdict={'size': 20})
    fig[2].set_title('R(x)', fontdict={'size': 20})
    max_x, max_y = R.shape
    for x, y in zip(*[iter(x_optimum)] * 2):
        fig[1].plot(max_y*x, max_x*y, 'D', markersize=8, label=u'Last observation', color='gold')
        fig[2].plot(max_y*x, max_x*y, 'D', markersize=8, label=u'Last observation', color='gold')
    fig.close('figs', f'optimal_location_for_{n_radars}_{i}')


def plot_comparison_score(score, max_n=MAX_N_RADARS):
    fig = Figure(1)
    fig[0].plot(list(range(N_CURRENT_RADARS, N_CURRENT_RADARS + max_n + 1)), score)
    fig[0].set_ylim([0, np.max(np.array(score))])
    fig[0].set_xlabel('#radars')
    fig[0].set_ylabel('score')
    fig.close('figs', f'comparison_of_score')


# toy problem

def toy_problem():
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
    toy_problem()