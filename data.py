import itertools

import graph_tool.all as gt
import numpy as np
from tqdm.contrib.concurrent import process_map

from engine.utils.io import *
from engine.utils.network_generator import *
from engine.utils.robustness_score import *


def gen_dir():
    base_dir = get_working_dir() + "data/"
    mkdir(base_dir)
    for arg in get_param_list():
        n, k = arg[0], arg[1]
        mkdir(base_dir + "/" + str(n))
        mkdir(base_dir + str(n) + "/" + str(k))
        mkdir(base_dir + str(n) + "/" + str(k) + "/" + "pa")
        mkdir(base_dir + str(n) + "/" + str(k) + "/" + "random")
        mkdir(base_dir + str(n) + "/" + str(k) + "/" + "random_min_deg")


def generate_data():
    # --------------------------------------------------------------------------------
    # setup
    # --------------------------------------------------------------------------------
    set_num_engines()
    set_num_sampled_graphs(100)
    set_num_reps(10)
    set_seed(6814)
    set_max_tries(10000000)
    set_param_list(list(itertools.product([1000, 10000, 20000], [3, 4, 5])))
    set_working_dir(os.getcwd() + "/results/")
    log_initial_parameters()
    gen_dir()
    n_engines = get_num_engines()
    # --------------------------------------------------------------------------------
    # graph generation
    # --------------------------------------------------------------------------------
    args_pa = []
    args_random = []
    args_random_min_deg = []
    seeds = np.random.RandomState(get_seed()).choice(10000000, get_num_sampled_graphs())
    for (n, k) in get_param_list():
        base_dir = get_working_dir() + "data/" + str(n) + "/" + str(k) + "/"
        args_pa.extend(
            [(base_dir + "pa" + "/" + str(i), n, k, seeds[i]) for i in
             range(get_num_sampled_graphs())])
        args_random.extend(
            [(base_dir + "random" + "/" + str(i), n, n * k, seeds[i],
              get_max_tries()) for i in
             range(get_num_sampled_graphs())])
        args_random_min_deg.extend(
            [(base_dir + "random_min_deg" + "/" + str(i), n, n * k, k, seeds[i],
              get_max_tries())
             for
             i in
             range(get_num_sampled_graphs())])

    set_logger("pa.log")
    logging.info("The format is: n (number of vertices), k (minimum degree), seed (random seed)")
    results = process_map(generate_pa, args_pa, max_workers=n_engines, desc="pa", chunksize=1)
    for (exit_code, n, k, seed) in results:
        if exit_code == 0:
            logging.info("Finished the generation of: %s", (n, k, seed))
        else:
            logging.error("Failed in the generation of: %s", (n, k, seed))
    reset_logger()

    set_logger("random.log")
    logging.info("The format is: n (number of vertices), m (number of edges), seed (random seed)")
    results = process_map(generate_random, args_random, max_workers=n_engines, desc="random", chunksize=1)
    for (exit_code, n, m, seed) in results:
        if exit_code == 0:
            logging.info("Finished the generation of: %s", (n, m, seed))
        elif exit_code == 1:
            logging.error("Timeout in the generation of: %s", (n, m, seed))
        else:
            logging.error("Failed in the generation of: %s", (n, m, seed))
    reset_logger()

    set_logger("random_min_deg.log")
    logging.info("The format is: n (number of vertices), m (number of edges), k (minimum degree), seed (random seed)")
    results = process_map(generate_random_min_deg, args_random_min_deg, max_workers=n_engines, desc="random_min_deg",
                          chunksize=1)
    for (exit_code, n, m, k, seed) in results:
        if exit_code == 0:
            logging.info("Finished the generation of: %s", (n, m, k, seed))
        elif exit_code == 1:
            logging.error("Timeout in the generation of: %s", (n, m, k, seed))
        else:
            logging.error("Failed in the generation of: %s", (n, m, k, seed))
    reset_logger()
    # --------------------------------------------------------------------------------
    # score computation
    # --------------------------------------------------------------------------------
    for (n, k) in get_param_list():
        for graph_type in ["pa", "random", "random_min_deg"]:
            assert len(os.listdir(
                get_working_dir() + "data/" + str(n) + "/" + str(k) + "/" + graph_type)) == get_num_sampled_graphs()

    score_base_dir = get_working_dir() + "scores/"
    mkdir(score_base_dir)
    robustness_score_dirs = [score_base_dir + "static-targeted-attack/",
                             score_base_dir + "adaptive-targeted-attack/",
                             score_base_dir + "random-failure/"
                             ]
    mkdir(robustness_score_dirs[0])
    mkdir(robustness_score_dirs[1])
    mkdir(robustness_score_dirs[2])

    for path_prefix in robustness_score_dirs:
        for (n, k) in get_param_list():
            mkdir(path_prefix + str(n))
            mkdir(path_prefix + str(n) + "/" + str(k))
            mkdir(path_prefix + str(n) + "/" + str(k) + "/" + "pa")
            mkdir(path_prefix + str(n) + "/" + str(k) + "/" + "random")
            mkdir(path_prefix + str(n) + "/" + str(k) + "/" + "random_min_deg")
    args = []
    for (n, k) in get_param_list():
        for i in range(get_num_sampled_graphs()):
            for graph_type in ["pa", "random", "random_min_deg"]:
                read_path = get_working_dir() + "data/" + str(n) + "/" + str(k) + "/" + graph_type + "/" + str(
                    i) + ".gt"
                write_path_static_attack = score_base_dir + "static-targeted-attack/" + str(n) + "/" + str(
                    k) + "/" + graph_type + "/" + str(i) + ".npy"
                write_path_adaptive_attack = score_base_dir + "adaptive-targeted-attack/" + str(n) + "/" + str(
                    k) + "/" + graph_type + "/" + str(i) + ".npy"
                write_path_random = score_base_dir + "random-failure/" + str(n) + "/" + str(
                    k) + "/" + graph_type + "/" + str(i) + ".npy"
                args.append((read_path, write_path_static_attack, write_path_adaptive_attack, write_path_random,
                             score_base_dir, get_num_reps()))

    set_logger("score_generation.log")
    logging.info(
        "The format is: n (number of vertices), k (minimum degree), graph_type, index (among all generated graphs of same kind), seed (random seed)")
    results = process_map(compute_robustness_score, args, max_workers=n_engines, desc="compute_scores", chunksize=1)
    for (exit_code, n, k, graph_type, index, seed) in results:
        if exit_code == 0:
            logging.info("Computed the score with the following parameters: %s", (n, k, graph_type, index, seed))
        else:
            logging.error("Failed to compute the score with the following parameters: %s",
                          (n, k, graph_type, index, seed))
    reset_logger()

    res_scores = {}
    res_graphs = {}
    for (n, k) in get_param_list():
        res_scores[(n, k)] = {}
        res_graphs[(n, k)] = {}
        for graph_type in ["pa", "random", "random_min_deg"]:
            res_scores[(n, k)][graph_type] = {}
            res_graphs[(n, k)][graph_type] = {}
            for i in range(get_num_sampled_graphs()):
                res_graphs[(n, k)][graph_type][i] = gt.load_graph(
                    get_working_dir() + "data/" + str(n) + "/" + str(k) + "/" + graph_type + "/" + str(i) + ".gt")
            for removal_type in ["random-failure", "static-targeted-attack", "adaptive-targeted-attack"]:
                res_scores[(n, k)][graph_type][removal_type] = np.empty(shape=(get_num_sampled_graphs(), 100),
                                                                        dtype=float)
                for i in range(get_num_sampled_graphs()):
                    res_scores[(n, k)][graph_type][removal_type][i] = np.load(
                        score_base_dir + removal_type + "/" + str(n) + "/" + str(k) + "/" + graph_type + "/" + str(
                            i) + ".npy")
    # --------------------------------------------------------------------------------
    # clean up
    # --------------------------------------------------------------------------------
    logging.shutdown()
    remove_dir(get_working_dir() + "scores")
    remove_dir(get_working_dir() + "data")
    with open(get_working_dir() + "scores.pkl", "wb") as scores_file:
        pickle.dump(res_scores, scores_file)
    with open(get_working_dir() + "graphs.pkl", "wb") as graphs_file:
        pickle.dump(res_graphs, graphs_file)


if __name__ == '__main__':
    generate_data()
