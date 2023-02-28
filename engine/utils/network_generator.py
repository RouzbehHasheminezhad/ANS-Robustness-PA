def generate_random(args):
    import random
    import igraph as ig
    import graph_tool.all as gt
    net_dir, n, m, seed, max_tries = args[0], args[1], args[2], args[3], args[4]
    num_tries = 0
    try:
        while num_tries < max_tries:
            random.seed(seed)
            # noinspection PyArgumentList
            g = ig.Graph.Erdos_Renyi(n=n, m=m, directed=False, loops=False).to_graph_tool()
            lcc_of_g = gt.extract_largest_component(g, directed=False, prune=True)
            if (lcc_of_g.num_vertices() / g.num_vertices()) > 0.96:
                lcc_of_g.save(net_dir + ".gt", fmt="gt")
                return 0, n, m, seed
            else:
                num_tries += 1
                seed += 1
        return 1, n, m, seed
    except (Exception,):
        return 2, n, m, seed


def generate_random_min_deg(args):
    import random
    import igraph as ig
    from scipy.optimize import bisect
    from scipy.stats import poisson

    net_dir, n, m, k, seed, max_tries = args[0], args[1], args[2], args[3], args[4], args[5]
    c = (2 * m) / n

    def f(x):
        return (x * (poisson(x).sf(k - 2) / poisson(x).sf(k - 1))) - c

    random.seed(seed)
    lamda = bisect(f, a=c - k, b=c, maxiter=max_tries)
    n_ = round(n / poisson(lamda).sf(k - 1))
    m_ = round(0.5 * (lamda / poisson(lamda).sf(k - 2)) * n_)
    num_tries = 0
    try:
        while num_tries < max_tries:
            # noinspection PyArgumentList
            g = ig.Graph.Erdos_Renyi(n=n_, m=m_)
            g_ = g.k_core(k)
            if g_.vcount() == n and g_.ecount() == m:
                g_.to_graph_tool().save(net_dir + ".gt", fmt="gt")
                return 0, n, m, k, seed
            else:
                num_tries += 1
        return 1, n, m, k, seed
    except (Exception,):
        return 2, n, m, k, seed


def generate_pa(args):
    import random
    import igraph as ig
    net_dir, n, k, seed = args[0], args[1], args[2], args[3]
    try:
        random.seed(seed)
        g = ig.Graph().Barabasi(n=n, m=k, directed=False, zero_appeal=0,
                                start_from=ig.Graph.Full(n=2 * k + 1, directed=False, loops=False)).to_graph_tool()
        g.save(net_dir + ".gt", fmt="gt")
        return 0, n, k, seed
    except (Exception,):
        return 2, n, k, seed


def generate_ua(args):
    import numpy as np
    import graph_tool.all as gt
    net_dir, n, k, seed = args[0], args[1], args[2], args[3]
    try:
        gt.seed_rng(seed)
        np.random.seed(seed)
        g = gt.price_network(N=n - 2 * k - 1, m=k, c=0, gamma=0, directed=False,
                             seed_graph=gt.complete_graph(directed=False, self_loops=False, N=2 * k + 1))
        g.save(net_dir + ".gt", fmt="gt")
        return 0, n, k, seed
    except (Exception,):
        return 2, n, k, seed
