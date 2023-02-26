def get_reverse_removal_orders(g, removal_type, number_removal_orders):
    import numpy as np
    import random
    seed = hash(tuple(sorted(g.get_out_degrees(list(range(g.num_vertices()))), reverse=True)))
    seed = seed % (2 ^ 32 - 1)
    new_seeds = np.random.RandomState(seed).choice(10000000, number_removal_orders, replace=False)
    removal_orders = []
    if removal_type == "static-targeted-attack":
        deg_dict = {}
        for v in g.vertices():
            if v.out_degree() not in deg_dict:
                deg_dict[v.out_degree()] = [g.vertex_index[v]]
            else:
                deg_dict[v.out_degree()].append(g.vertex_index[v])
        deg_dict = dict(sorted(deg_dict.items(), reverse=True))
        for seed_ in new_seeds:
            attack_order = []
            for key in sorted(deg_dict, reverse=True):
                random.Random(seed_).shuffle(deg_dict[key])
                attack_order += deg_dict[key]
            removal_orders.append(attack_order[::-1])
    elif removal_type == "random-failure":
        for seed_ in new_seeds:
            attack_order = [i for i in range(g.num_vertices())]
            random.Random(seed_).shuffle(attack_order)
            removal_orders.append(attack_order[::-1])
    elif removal_type == "adaptive-targeted-attack":
        for seed_ in new_seeds:
            deg_dict = {i: set() for i in range(g.num_vertices())}
            vertices = [v for v in g.vertices()]
            random.Random(seed_).shuffle(vertices)
            for v in vertices:
                deg_dict[v.out_degree()].add(g.vertex_index[v])
            neighbors = {i: list(g.get_out_neighbors(i)) for i in range(g.num_vertices())}
            for i in range(g.num_vertices()):
                neighbors[i] = sorted(neighbors[i])
                random.Random(seed_).shuffle(neighbors[i])
                neighbors[i] = set(neighbors[i])
            attack_order = []
            for k in range(g.num_vertices() - 1, -1, -1):
                while len(deg_dict[k]) != 0:
                    u = deg_dict[k].pop()
                    for neighbor in neighbors[u]:
                        d = len(neighbors[neighbor])
                        deg_dict[d].remove(neighbor)
                        deg_dict[d - 1].add(neighbor)
                        neighbors[neighbor].remove(u)
                    del neighbors[u]
                    attack_order.append(u)
            removal_orders.append(attack_order[::-1])
    return removal_orders


# noinspection PyArgumentList
def compute_robustness_score(args):
    import graph_tool.all as gt
    import numpy as np
    read_path, write_path_static_attack, write_path_adaptive_attack, write_path_random, score_base_dir, reps = args[0], args[1], args[2], args[3], args[4], args[5]

    def get_scores(graph, reverse_removal_orders):
        score = np.zeros(100)
        for reverse_removal_order in reverse_removal_orders:
            n_ = len(reverse_removal_order)
            res = np.concatenate((gt.vertex_percolation(graph, reverse_removal_order)[0][::-1][1:], [0])) / n_
            endpoints = [int(np.ceil(alpha * n_)) for alpha in np.linspace(0.01, 1, 100)]
            score += [np.mean(res[:end]) for end in endpoints]
        return score / len(reverse_removal_orders)
    g = gt.load_graph(read_path)
    deg_seq = sorted(list([v.out_degree() for v in g.vertices()]), reverse=True)
    seed = hash(tuple(deg_seq))
    try:
        reverse_static_attack_orders = get_reverse_removal_orders(g, "static-targeted-attack", reps)
        reverse_adaptive_attack_orders = get_reverse_removal_orders(g, "adaptive-targeted-attack", reps)
        reverse_random_orders = get_reverse_removal_orders(g, "random-failure", reps)

        np.save(write_path_static_attack, get_scores(g, reverse_static_attack_orders))
        np.save(write_path_adaptive_attack, get_scores(g, reverse_adaptive_attack_orders))
        np.save(write_path_random, get_scores(g, reverse_random_orders))

        return (0,) + tuple(
            write_path_static_attack[len(score_base_dir) + len("static-targeted-attack/"):][:-len('.npy')].split(
                "/")) + (seed,)
    except (Exception,):
        return (1,) + tuple(
            write_path_static_attack[len(score_base_dir) + len("static-targeted-attack/"):][:-len('.npy')].split(
                "/")) + (seed,)
