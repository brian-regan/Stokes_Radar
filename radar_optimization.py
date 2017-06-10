import radar_map
from optimize_location import optimize, plot_comparison_score
import numpy as np
from PIL import Image


xy_range = ((0, 1), (0, 1))
POINTS = (334, 418)

grid_for_QR = np.meshgrid(np.linspace(*xy_range[0], POINTS[0]),
                          np.linspace(*xy_range[1], POINTS[1]))


def main():
    city_centers = [(0.3, 0.3), (0.7, 0.4), (0.2, 0.7)]

    def Q_each(x_c, y_c, grid=grid_for_QR):
        a = 0.3
        r = a ** 2 - (grid[0] - x_c) ** 2 - (grid[1] - y_c) ** 2
        r[r < 0] = 0
        return r

    Q = np.zeros_like(grid_for_QR[0])
    for city_center in city_centers:
        Q = np.maximum(Q, Q_each(*city_center))

    Q = np.ones_like(Q)

    #image = Image.open("topography_1.png").convert("L")
    #arr = np.asarray(image)
    #arr = (255 - arr) * (10 / 255)

    topo = np.loadtxt('topo.mat')
    importance = np.loadtxt('importance.mat')


    def R(*args, **kwargs):
        centers = []
        for i in range(int(len(args) / 2)):
            x = int(args[i * 2] * POINTS[0])
            y = int(args[i * 2 + 1] * POINTS[1])
            centers.append((x, y))

        for i in range(int(len(kwargs) / 2)):
            x = int(kwargs['x' + str(i)] * POINTS[0])
            y = int(kwargs['y' + str(i)] * POINTS[1])
            centers.append((x, y))

        r_maxs = [250 for _ in centers]

        return radar_map.field_of_vision(centers, r_maxs, topo, 30, 'linear', resolution=1000, max_height=5)

    n_trials = 3
    score =np.zeros((n_trials, 4))
    for i in range(n_trials):
        score[i, :] = optimize(importance, R, topo, i)
    plot_comparison_score(score.T)

if __name__ == '__main__':
    main()