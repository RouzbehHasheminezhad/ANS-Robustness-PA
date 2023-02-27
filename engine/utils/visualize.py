import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def z_score(a, b):
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=0)
    var_b = np.var(b, ddof=0)
    assert len(a) == len(b)
    n = len(a)
    return (np.sqrt(n) * (mean_a - mean_b)) / (np.sqrt((var_a + var_b)))


def compute_z_score(beta, compared_group, reference_group):
    index = int(beta * 100) - 1
    points = {}
    with open(os.getcwd() + '/results/scores.pkl', 'rb') as f:
        scores = pickle.load(f)
        for removal_strategy in ["static-targeted-attack", "adaptive-targeted-attack", "random-failure"]:
            points[removal_strategy] = {}
            for (n, k) in scores:
                a = [x[index] for x in scores[(n, k)][compared_group][removal_strategy]]
                b = [x[index] for x in scores[(n, k)][reference_group][removal_strategy]]
                points[removal_strategy][(n, k)] = z_score(a, b)

    return points


def get_robustness_score(n, k, removal_strategy, beta):
    index = int(beta * 100) - 1
    res = {}
    with open(os.getcwd() + '/results/scores.pkl', 'rb') as f:
        scores = pickle.load(f)
        for graph_type in ["pa", "random", "random_min_deg"]:
            res[graph_type] = [x[index] for x in scores[(n, k)][graph_type][removal_strategy]]
    return res


def get_ratios_score(n, k, alphas):
    data = {}
    for removal_strategy in ["static-targeted-attack", "adaptive-targeted-attack", "random-failure"]:
        data[removal_strategy] = []
        for alpha in alphas:
            data[removal_strategy].append(
                np.mean(get_robustness_score(n, k, removal_strategy, alpha)["random_min_deg"]) / (
                        1 - (np.ceil(n * alpha) / (2 * n))))
    return data


def get_res_map():
    res = {}
    for (n, k) in list(itertools.product([1000, 10000, 20000], [3, 4, 5])):
        res[(n, k)] = {}
        for graph_type in ["pa", "random", "random_min_deg"]:
            res[(n, k)][graph_type] = {}
            for removal_strategy in ["static-targeted-attack", "adaptive-targeted-attack", "random-failure"]:
                res[(n, k)][graph_type][removal_strategy] = []
                for beta in [i / 100 for i in range(1, 100)]:
                    res[(n, k)][graph_type][removal_strategy].append(
                        np.mean(get_robustness_score(n, k, removal_strategy, beta)[graph_type]))
    return res


def experiment(n, k, alpha_list):
    import matplotlib as mpl
    mpl.use('pdf')
    plt.rcParams['hatch.linewidth'] = 2
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    width = 3.487 * 3
    height = width / 1.27

    fig, ax = plt.subplots(3, 1, figsize=(width, height))
    markers = {"random": "^",
               "random_min_deg": "s",
               "pa": "o"}
    labels = {"random_min_deg": r"$G(n,m,k)$",
              "random": r"$G(n,m)$",
              "pa": r"$\textsf{PA}(n,k)$"}

    offset_x = {"random": {0.05: -0.00, 0.1: -0.0, 0.2: -0.0, 0.25: -0.013, 0.3: -0.02, 0.4: -0.029},
                "random_min_deg": {0.05: -0.0008, 0.1: -0.0022, 0.2: -0.0075, 0.25: -0.016, 0.3: -0.025,
                                   0.4: -0.035},
                "pa": {0.05: 0., 0.1: 0.0, 0.2: 0.0, 0.25: 0.0, 0.3: 0.0, 0.4: 0.0}}

    offset_y = {"random": {0.05: 0.0002, 0.1: 0.0003, 0.2: 0.0005, 0.25: 0.000, 0.3: 0.000, 0.4: 0.000},
                "random_min_deg": {0.05: -0.00008, 0.1: -0.00016, 0.2: -0.00032, 0.25: -0.0, 0.3: -0.00,
                                   0.4: -0.00},
                "pa": {0.05: -0.00014, 0.1: -0.0003, 0.2: -0.0006, 0.25: 0.0005, 0.3: 0.0006, 0.4: 0.001}}
    for i in range(3):
        alpha = alpha_list[i]
        ax_ = ax[i]
        scores_x = get_robustness_score(n, k, "static-targeted-attack", alpha)
        scores_y = get_robustness_score(n, k, "random-failure", alpha)
        y_min, x_min, y_max, x_max = 1, 1, 0, 0
        for net_type in ['pa', 'random', 'random_min_deg']:
            ax_.scatter(scores_x[net_type], scores_y[net_type], label=net_type, marker=markers[net_type],
                        edgecolors='black', facecolors='none')
            ymin, ymax = ax_.get_ylim()
            xmin, xmax = ax_.get_xlim()
            x_min, y_min = min(xmin, x_min), min(ymin, y_min)
            x_max, y_max = max(xmax, x_max), max(ymax, y_max)
            if net_type == "random":
                ax_.annotate(labels[net_type],
                             (np.mean(scores_x[net_type]) + offset_x[net_type][alpha],
                              np.mean(scores_y[net_type]) + offset_y[net_type][alpha]),
                             textcoords="offset points",
                             xytext=(0., 0),
                             ha='center', size=20, color='black')
            else:
                ax_.annotate(labels[net_type],
                             (np.mean(scores_x[net_type]) + offset_x[net_type][alpha],
                              np.mean(scores_y[net_type]) + offset_y[net_type][alpha]),
                             textcoords="offset points",
                             xytext=(0., 0),
                             ha='center', size=20, color='black')

        ax_.tick_params(axis='x', labelsize=16)
        ax_.tick_params(axis='y', labelsize=16)
        if alpha_list == [0.05, 0.1, 0.2]:
            if i == 0:
                ax_.set_xlim(xmax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 3 / n)
                ax_.set_ylim(ymax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 2 / n)
            elif i == 1:
                ax_.set_xlim(xmax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 9 / n)
                ax_.set_ylim(ymax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 3.5 / n)
            else:
                ax_.set_xlim(xmax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 34 / n)
                ax_.set_ylim(ymax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 7 / n)
        else:
            if i == 0:
                ax_.set_xlim(xmax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 36 / n)
                ax_.set_ylim(ymax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 6 / n)
            elif i == 1:
                ax_.set_xlim(xmax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 60 / n)
                ax_.set_ylim(ymax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 7.5 / n)
            else:
                ax_.set_xlim(xmax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 84 / n)
                ax_.set_ylim(ymax=1 - np.ceil(alpha_list[i] * n) / (2 * n) + 12 / n)
        ax_.add_patch(Rectangle((0, 0), 1 - np.ceil(alpha * n) / (2 * n),
                                1 - np.ceil(alpha_list[i] * n) / (2 * n), fill=None, alpha=1, linestyle="--",
                                color='black'))
        ax_.text((x_max + x_min) / 2, 1 - (np.ceil(alpha * n) / (2 * n)) + 0.04 * (y_max - y_min),
                 r'$ \beta=$' + str(alpha),
                 c='black', fontsize=21)

    ax[1].set_ylabel('robustness score for random failure', rotation=90, size=24)
    ax[2].set_xlabel('robustness score for targeted attack', rotation=0, size=24)
    ax[2].xaxis.set_label_coords(0.55, -0.15)
    ax[1].yaxis.set_label_coords(-0.1, 0.5)

    plt.subplots_adjust(left=0.12, right=0.98, top=0.99, bottom=0.08)
    if alpha_list == [0.05, 0.1, 0.2]:
        fig.savefig("results/figs/scatterplot.pdf", format="pdf")
    else:
        fig.savefig("results/figs/scatterplot_consistency.pdf", format="pdf")

    plt.close()


def consistency(is_centered, alphas, transparency, ref):
    import matplotlib as mpl
    mpl.use('pdf')
    width = 3.487 * 6
    height = width * 1.15 * 1.5

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('axes', labelsize=60)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=100)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=100)  # fontsize of the tick labels
    plt.rc('legend', fontsize=60)  # legend fontsize
    plt.rc('legend', fontsize=60)

    markers = {"random_min_deg": "s",
               "random": "^",
               "pa": "o"
               }
    labels = {"random_min_deg": r"$G(n,m,k)$",
              "random": r"$G(n,m)$",
              "pa": r"$\textsf{PA}(n,k)$"
              }

    net_list = list(markers.keys())
    removal_types = ["static-targeted-attack", "random-failure"]

    f, ax = plt.subplots(6, 2, figsize=(width, height))
    f.subplots_adjust(left=.05, bottom=.12, right=.95, top=.95)

    for is_effect_n in [True, False]:
        if is_effect_n:
            cases = {(20000, 3): {}, (10000, 3): {}, (1000, 3): {}}
        else:
            cases = {(10000, 5): {}, (10000, 4): {}, (10000, 3): {}}
        for k in [0, 1, 2, 3, 4, 5]:
            alpha = alphas[k]

            if is_effect_n:
                ax_ = ax[k, 0]
            else:
                ax_ = ax[k, 1]

            ax_.axvline(x=0, ls='--', lw=1.5, c='k')
            ax_.axhline(y=0, ls='--', lw=1.5, c='k')
            ax_.tick_params(axis='x', labelsize=50)
            ax_.tick_params(axis='y', labelsize=50)

            if k == 5:
                ax_.set_xlabel("z-score targeted attack", fontsize=70)
            if is_effect_n and k == 2:
                ax_.set_ylabel("z-score random failure", fontsize=70)

            ax_.text(.17, 0.8, r'$\beta=$' + "%.2f" % alpha,
                     horizontalalignment='center',
                     transform=ax_.transAxes, fontsize=55)

            sizes = [72, 144, 288]
            sizes = [x * 5 for x in sizes][::-1]
            i = 0
            for (n, m) in cases:
                zscores_attack = {}
                zscores_random = {}
                for key in net_list:
                    if key != ref:
                        zscores_attack[key] = compute_z_score(alpha, key, ref)[removal_types[0]][(n, m)]
                        zscores_random[key] = compute_z_score(alpha, key, ref)[removal_types[1]][(n, m)]

                for net_type in net_list:
                    if net_type != ref:
                        c = "black"
                        x = zscores_attack[net_type]
                        y = zscores_random[net_type]
                        ax_.scatter(x, y, c=c, marker=markers[net_type], label=labels[net_type], s=sizes[i],
                                    alpha=transparency)

                i += 1
            if is_centered:
                yabs_max = abs(max(ax_.get_ylim(), key=abs))
                ax_.set_ylim(ymin=-1.1 * yabs_max, ymax=1.1 * yabs_max)
                xabs_max = abs(max(ax_.get_xlim(), key=abs))
                ax_.set_xlim(xmin=-1.1 * xabs_max, xmax=1.1 * xabs_max)
    lgn = plt.legend(numpoints=1, markerscale=2., loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True,
                     shadow=True, ncol=4)

    handles = lgn.legendHandles[4:8]
    legend_labels = [labels[x] for x in net_list if x != ref]
    plt.legend(handles=handles, labels=legend_labels, fancybox=True, shadow=True, loc="upper left",
               bbox_to_anchor=(-1., -0.42),
               ncol=4)
    pad = 16

    for a, col in zip(ax[0], [r"Effect of $n$", r"Effect of $k$", ]):
        a.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                   xycoords='axes fraction', textcoords='offset points',
                   size=60, ha='center', va='baseline')

    f.suptitle(r"Reference is $\textsf{PA}(n,k)$", fontsize=60, y=0.995)
    plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12)
    f.savefig("results/figs/zscores.pdf", format="pdf")
    plt.close()


def near_optimal_consistency(n_list, min_deg_list):
    import matplotlib as mpl
    mpl.use('pdf')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize=38)
    plt.rc('legend', fontsize=38)
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    width = 3.487 * 6
    height = width / 1.618
    lw = 5

    f, ax = plt.subplots(3, 3, figsize=(width, height))
    f.subplots_adjust(left=.1, bottom=.2, right=.98, top=.98)
    styles = {"static-targeted-attack": "solid",
              "random-failure": "solid",
              "adaptive-targeted-attack": "solid"
              }
    colors = {
        "random-failure": "darkgrey",
        "static-targeted-attack": "dimgrey",
        "adaptive-targeted-attack": "black"
    }
    legends = {
        "random-failure": "random failure",
        "static-targeted-attack": "non-adaptive attack",
        "adaptive-targeted-attack": "adaptive attack"
    }

    for i in range(3):
        min_deg = min_deg_list[i]
        for j in [0, 1, 2]:
            n = n_list[j]
            alphas = np.linspace(0.01, 1, 100)
            results = get_ratios_score(n, min_deg, alphas)

            ax_ = ax[i, j]
            if i == 0:
                ax_.text(.55, 1.05,
                         r'\begin{align*}' + r'n=' + str(n) + r'\end{align*}',
                         horizontalalignment='center',
                         transform=ax_.transAxes, fontsize=38)
            if j == 2:
                ax_.text(1.15, .55,
                         r'\begin{align*}' + r'k=' + str(min_deg) + r'\end{align*}',
                         horizontalalignment='center',
                         transform=ax_.transAxes, fontsize=38)
            for removal_type in ["random-failure", "static-targeted-attack", "adaptive-targeted-attack"]:
                ax_.plot(alphas, results[removal_type], label=legends[removal_type],
                         linestyle=styles[removal_type], c=colors[removal_type], lw=lw)
            ax_.tick_params(axis='x', labelsize=32)
            ax_.tick_params(axis='y', labelsize=32)
            ax_.set_xlim(0, 1)
            ax_.set_ylim(0, 1.1)
            ax_.tick_params(width=3, length=4.854)
            ax_.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
            ax_.set_yticks([0.2, 0.4, 0.6, 0.8, 1])

            if i in [0, 1]:
                ax_.xaxis.set_ticklabels([])
            if j in [1, 2]:
                ax_.yaxis.set_ticklabels([])
            ax_.spines['top'].set_visible(False)
            ax_.spines['right'].set_visible(False)

    f.text(0.5, 0.115, 'proportion of vertices removed', ha='center', fontsize=42)
    f.text(0.03, 0.6, 'normalized robustness score', va='center', rotation='vertical', fontsize=42)
    f.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.2)
    leg = plt.legend(loc='upper center', bbox_to_anchor=(-0.65, -0.45), fancybox=True, shadow=False, ncol=3,
                     frameon=True)
    leg.get_frame().set_linewidth(3)
    f.savefig("results/figs/near_optimal_consistency" + ".pdf", format="pdf")
    plt.close()


def adaptive_sensetivity(n_list, min_deg_list):
    import matplotlib as mpl
    mpl.use('pdf')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize=38)
    plt.rc('legend', fontsize=38)
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    width = 3.487 * 6
    height = width / 1.618
    lw = 5

    f, ax = plt.subplots(3, 3, figsize=(width, height))
    f.subplots_adjust(left=.1, bottom=.2, right=.98, top=.98)

    colors = {
        "pa": "dimgrey",
        "random": "darkgrey",
        "random_min_deg": "black"
    }
    line_styels = {
        "pa": "solid",
        "random": "solid",
        "random_min_deg": ':'
    }
    legends = {"random_min_deg": r"$\textsf{G}(n,m,k)$",
               "random": r"$\textsf{G}(n,m)$",
               "pa": r"$\textsf{PA}(n,k)$"}

    results = get_res_map()
    alphas = [i / 100 for i in range(100)]
    for i in range(3):
        min_deg = min_deg_list[i]
        for j in [0, 1, 2]:
            n = n_list[j]
            ax_ = ax[i, j]
            if i == 0:
                ax_.text(.55, 1.05,
                         r'\begin{align*}' + r'n=' + str(n) + r'\end{align*}',
                         horizontalalignment='center',
                         transform=ax_.transAxes, fontsize=38)
            if j == 2:
                ax_.text(1.15, .55,
                         r'\begin{align*}' + r'k=' + str(min_deg) + r'\end{align*}',
                         horizontalalignment='center',
                         transform=ax_.transAxes, fontsize=38)
            for graph_type in ["pa", "random", "random_min_deg"]:
                ax_.plot(alphas, [1.] + results[(n, min_deg)][graph_type]["adaptive-targeted-attack"],
                         label=legends[graph_type], c=colors[graph_type], lw=lw, linestyle=line_styels[graph_type])

            ax_.tick_params(axis='x', labelsize=32)
            ax_.tick_params(axis='y', labelsize=32)
            ax_.set_xlim(0, 1)
            ax_.set_ylim(0, 1.1)
            ax_.tick_params(width=3, length=4.854)
            ax_.set_xticks([0.2, 0.4, 0.6, 0.8, 1])
            ax_.set_yticks([0.2, 0.4, 0.6, 0.8, 1])

            if i in [0, 1]:
                ax_.xaxis.set_ticklabels([])
            if j in [1, 2]:
                ax_.yaxis.set_ticklabels([])
            ax_.spines['top'].set_visible(False)
            ax_.spines['right'].set_visible(False)

    f.text(0.5, 0.115, 'proportion of vertices removed', ha='center', fontsize=42)
    f.text(0.03, 0.6, 'robustness score', va='center', rotation='vertical', fontsize=42)
    f.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.2)
    leg = plt.legend(loc='upper center', bbox_to_anchor=(-0.65, -0.45), fancybox=True, shadow=False, ncol=3,
                     frameon=True)
    leg.get_frame().set_linewidth(3)
    f.savefig("results/figs/adaptive_sensitivity.pdf", format="pdf")
    plt.close()
