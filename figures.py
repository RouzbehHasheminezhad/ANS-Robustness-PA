from engine.utils.io import *
from engine.utils.visualization import experiment, consistency, near_optimal_consistency, adaptive_sensetivity


def generate_figures():
    mkdir_(os.getcwd() + "/results/figs/")
    experiment(10000, 3, [0.05, 0.1, 0.2])
    experiment(10000, 3, [0.25, 0.3, 0.4])
    consistency(is_centered=True, transparency=0.8, alphas=[0.05, 0.10, 0.20, 0.25, 0.3, 0.4], ref="pa")
    near_optimal_consistency([1000, 10000, 20000], [3, 4, 5])
    adaptive_sensetivity([1000, 10000, 20000], [3, 4, 5])


if __name__ == '__main__':
    generate_figures()
