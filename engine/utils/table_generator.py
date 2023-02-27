def get_cluster_score(n, k, beta):
    from engine.utils.visualization import get_robustness_score
    from sklearn import metrics
    import numpy as np
    types = ["pa", "random", "random_min_deg"]
    scores_1 = get_robustness_score(n, k, "static-targeted-attack", beta)
    scores_2 = get_robustness_score(n, k, "random-failure", beta)
    features = []
    labels = []
    for x in types:
        assert len(scores_1[x]) == len(scores_2[x])
        for i in range(len(scores_1[x])):
            features.append([scores_1[x][i], scores_2[x][i]])
            labels.append(types.index(x))
    return metrics.silhouette_score(np.array(features), np.array(labels))


def get_effect_of_n():
    import pandas as pd
    effect_of_n = {"": ["$\\beta=" + "{:.2f}".format(x) + "$" for x in [0.05, 0.1, 0.2, 0.25, 0.3, 0.4]], 1000: [], 10000: [],
                   20000: []}

    for alpha in [0.05, 0.1, 0.2, 0.25, 0.3, 0.4]:
        for n in [1000, 10000, 20000]:
            if n == 20000:
                effect_of_n[n].append("\\textbf{" + "{:.3f}".format(get_cluster_score(n, 3, alpha)) + "}")
            else:
                effect_of_n[n].append("{:.3f}".format(get_cluster_score(n, 3, alpha)))

    effect_of_n = pd.DataFrame(effect_of_n)
    effect_of_n.rename(
        columns={x: ("$n=" if len(str(x)) > 0 else "") + str(x) + ("$" if len(str(x)) > 0 else "") for x in
                 effect_of_n.columns},
        inplace=True,
    )
    effect_of_n = effect_of_n.to_latex(index=False, escape=False, column_format="|c|c|c|c|").replace("\$", "$").replace(
        "\\toprule",
        "\cline{1-4} \n  \multirow{2}{*}{Effect of $n, \\beta$} & \multicolumn{3}{|c|}{$k=3$} \\\ \n \cline{2-4}").replace(
        "midrule",
        "hline").replace(
        "bottomrule", "hline")
    effect_of_n = "\subfloat[Fixed $k$ and varying $n, \\beta$]{\n" + effect_of_n + "}"
    return effect_of_n


def get_effect_of_k():
    import pandas as pd
    effect_of_k = {"": ["$\\beta=" + "{:.2f}".format(x) + "$" for x in [0.05, 0.1, 0.2, 0.25, 0.3, 0.4]], 3: [], 4: [],
                   5: []}

    for alpha in [0.05, 0.1, 0.2, 0.25, 0.3, 0.4]:
        for k in [3, 4, 5]:
            if k == 3:
                effect_of_k[k].append("\\textbf{" + "{:.3f}".format(get_cluster_score(10000, k, alpha)) + "}")
            else:
                effect_of_k[k].append("{:.3f}".format(get_cluster_score(10000, k, alpha)))
    effect_of_k = pd.DataFrame(effect_of_k)
    effect_of_k.rename(
        columns={x: ("$k=" if len(str(x)) > 0 else "") + str(x) + ("$" if len(str(x)) > 0 else "") for x in
                 effect_of_k.columns},
        inplace=True,
    )
    effect_of_k = effect_of_k.to_latex(index=False, escape=False, column_format="|c|c|c|c|").replace("\$", "$").replace(
        "\\toprule",
        "\cline{1-4} \n  \multirow{2}{*}{Effect of $k, \\beta$} & \multicolumn{3}{|c|}{$n=10000$} \\\ \n \cline{2-4}").replace(
        "midrule",
        "hline").replace(
        "bottomrule", "hline")
    effect_of_k = "\subfloat[Fixed $n$ and varying $k,\\beta$]{\n" + effect_of_k + "}"
    return effect_of_k


def generate_table_2():
    header = "\\begin{table}[!htb]\n\centering\n"
    footer = "\n\end{table}"
    caption = "For each fixed combination of $n$ and $k$, we generate three clusters of networks, as described at the beginning of Section~\\ref{sec:Experiments}. Then, for each fixed $\\beta\in\{0.05, 0.1, 0.2, 0.25, 0.3, 0.4\}$, we compute a silhouette score for the three corresponding groups, using the same procedure that we used to obtain the values in Table~$1$."
    label = "\n\label{tab:silhouette_consistency}"
    res = header + "\caption{" + caption + "}\n" + get_effect_of_n() + "\\\\" + "\n" + get_effect_of_k() + label + footer
    with open("results/tables/table_2.tex", "w") as f:
        f.write(res)


def generate_table_1():
    import pandas as pd
    header = "\\begin{table}[!htb]\n\centering\n"
    footer = "\n\end{table}"
    caption = "Silhouette scores corresponding to network clusters in Figures~\\ref{fig:scatterplot} and~\\ref{fig:scatterplot_consistency}."
    label = "\label{tab:silhouette}"
    df = pd.DataFrame({"$\\beta=" + "{:.2f}".format(x) + "$": ["{:.3f}".format(get_cluster_score(10000, 3, x))] for x in
                       [0.05, 0.1, 0.2, 0.25, 0.3, 0.4]})
    body = df.to_latex(index=False, escape=False, column_format="|c|c|c|c|c|c|c|").replace("\\toprule\n",
                                                                                           "\cline{1-7}\n\multirow{2}{*}{$n=10000, k=3$} &").replace(
        "\midrule\n", "\cline{2-7}\n& ").replace("\\bottomrule", "\hline")
    res = header + "\caption{" + caption + "}" + body + label + footer
    with open("results/tables/table_1.tex", "w") as f:
        f.write(res)
