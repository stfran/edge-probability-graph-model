# Analyzing the random graphs across different metrics

This directory contains the **Python-side evaluation code** for extending the original EPGM project with a small set of additional realism metrics. The goal is not to make new computational-complexity claims; it is to provide an easy-to-read, easy-to-modify evaluation pipeline that fits naturally into the existing repository structure.

## Metrics in scope

- GCC - reported in the original paper, reproduced here for convenience
- ALCC - reported in the original paper, reproduced here for convenience
- 4-clique density - the original repo has 4-clique counting code; here we expose the normalized density form as a metric
- C_3 - 3rd-order higher-order clustering coefficient from Yin et al. (2018)
- GCD-11 - graphlet correlation distance using the 11 non-redundant orbits for 2- to 4-node graphlets


## Files in this directory

- `metrics.py` - metric implementations and graph-loading helpers
- `*.ipynb` - metric exploration notebooks using the Facebook data as an example
- `README.md` - project notes, formulas, and citation record


## Basic facts of the project

This extension is designed around three ideas:

1. **Preserve continuity with the original EPGM paper.** The original paper evaluates clustering primarily with triangles, GCC, and ALCC.
2. **Add a higher-order closure metric.** `C_3` is a natural extension of triangle clustering because it measures whether triangles tend to expand into 4-cliques.
3. **Add a broader local-topology realism metric.** `GCD-11` compares two graphs using graphlet-orbit correlation structure rather than a single scalar summary.

## References
### Primary project paper

Bu, F., Yang, R., Bogdan, P., & Shin, K. (2025). *Edge probability graph models beyond edge independency: Concepts, analyses, and algorithms*. Proceedings of the IEEE International Conference on Data Mining.

### Higher-order clustering

Yin, H., Benson, A. R., & Leskovec, J. (2018). Higher-order clustering in networks. *Physical Review E, 97*(5), 052306. https://doi.org/10.1103/PhysRevE.97.052306

### Graphlet correlation distance: original method paper

Yaveroglu, O. N., Malod-Dognin, N., Davis, D., Levnajic, Z., Janjic, V., Karapandza, R., Stojmirovic, A., & Przulj, N. (2014). Revealing the hidden language of complex networks. *Scientific Reports, 4*, 4547. https://doi.org/10.1038/srep04547

### Graphlet correlation distance: evaluation paper supporting the GCD-11 choice

Yaveroglu, O. N., Milenkovic, T., & Przulj, N. (2015). Proper evaluation of alignment-free network comparison methods. *Bioinformatics, 31*(16), 2697-2704. https://doi.org/10.1093/bioinformatics/btv170

## Formulas 

### 1. Global clustering coefficient (GCC)

For an undirected graph $G=(V,E)$, let $d(v)$ denote the degree of node $v\in V$, i.e.,

$$d(v)=|\{u\in V : \{u,v\}\in E\}|.$$

Define

$$n_w(G)=
\sum_{v\in V} \binom{d(v)}{2}$$

to be the number of wedges in $G$, and let $\Delta(G)$ be the number of triangles in $G$, i.e., the number of distinct 3-node subsets $\{u,v,w\}\subseteq V$ such that all three edges $\{u,v\}, \{u,w\}, \{v,w\}$ belong to $E$. Then

$$\mathrm{GCC}(G)=\frac{\Delta(G)}{n_w(G)}.$$

Interpretation: among all places where a triangle could happen, how often does it happen?

Note, the operating GCC in Bu et al and its associated repository counts all trianlges as opposed to all *unique* triangles

---

### 2. Average local clustering coefficient (ALCC)

We follow the operational convention from the original repository code. For each node $v\in V$, let $\Delta(v)$ denote the number of **unique** triangles containing $v$, i.e., the number of distinct unordered pairs $\{u,w\}\subseteq V\setminus\{v\}$ such that $\{u,v\}, \{w,v\}, \{u,w\}\in E$.

Then define the local clustering coefficient of $v$ by

$$C(v)=\begin{cases}
\frac{\Delta(v)}{\binom{d(v)}{2}}, & d(v)\ge 2 \\
0, & d(v)<2
\end{cases}$$

and then

$$\mathrm{ALCC}(G)=\frac{1}{|V|}\sum_{v\in V} C(v).$$

Interpretation: for the average node, how interconnected are its neighbors?

---

### 3. 4-clique density

For a simple undirected graph $G=(V,E)$, let $K_k(G)$ denote the set of all 4-cliques in $G$, i.e.,

$$K_4(G)=\left\{S\subseteq V : |S|=4 \text{ and } \{u,v\}\in E \text{ for all distinct } u,v\in S\right\}.$$

Thus, a 4-clique is a 4-node subset of $V$ whose induced subgraph is complete.

The 4-clique density of $G$ is then defined by

$$\delta_4(G)=\frac{|K_4(G)|}{\binom{|V|}{4}}.$$

Interpretation: how often do fully connected groups of four nodes occur, relative to all possible 4-node sets?

---

### 4. 3rd-order higher-order clustering coefficient $C_3$

Following Yin et al. (2018), for an integer $\ell\ge 2$, let $K_{\ell+1}(G)$ denote the set of all $(\ell+1)$-cliques in $G$.

An $\ell$-wedge is formed by taking an $\ell$-clique together with an adjacent edge that shares exactly one node with that clique. Let $W_\ell(G)$ denote the set of all such $\ell$-wedges in $G$.

The global $\ell$-th order clustering coefficient is

$$C_\ell(G)=\frac{(\ell^2+\ell)\,|K_{\ell+1}(G)|}{|W_\ell(G)|}.$$

For $\ell=3$, this specializes to the 3rd-order higher-order clustering coefficient

$$C_3(G)=\frac{12\,|K_4(G)|}{|W_3(G)|}.$$

Equivalently, $C_3(G)$ measures the fraction of triangle-based 3-wedges that close into 4-cliques.

For a node $u\in V$, let $K_\ell(u)$ denote the set of $\ell$-cliques containing $u$, and let $K_{\ell+1}(u)$ denote the set of $(\ell+1)$-cliques containing $u$. Let $d(u)$ denote the degree of node $u$.

A useful identity is

$$|W_\ell(u)| = |K_\ell(u)|\,(d(u)-\ell+1),$$

which yields the local $\ell$-th order clustering coefficient

$$C_\ell(u)=\frac{|K_{\ell+1}(u)|}{(d(u)-\ell+1)\,|K_\ell(u)|},$$

whenever the denominator is nonzero.

Interpretation: if we start from a triangle and try to close one more level of dense structure, how often do we get a 4-clique?

---

### 5. Graphlet Correlation Distance (GCD-11)

Let $G=(V,E)$ be a simple undirected graph. Consider the 11 non-redundant graphlet orbits used in GCD-11. For each node $v\in V$, define its graphlet degree vector to be the 11-dimensional vector whose $a$-th entry counts how many times node $v$ participates in orbit $a$.

Stacking these vectors over all nodes gives the node-by-orbit matrix

$$X_G\in\mathbb{R}^{|V|\times 11},$$

where row $v$ corresponds to node $v$ and column $a$ corresponds to orbit $a$.

The graphlet correlation matrix of $G$ is the $11\times 11$ matrix whose $(a,b)$ entry is the Spearman rank correlation between orbit-columns $a$ and $b$ of $X_G$:

$$\mathrm{GCM}_{ab}(G)=\rho_s\big(X_G[:,a], X_G[:,b]\big).$$

Now let $H$ be another graph. Let $\operatorname{vec}_{\triangle}(\cdot)$ denote the operation that extracts the entries strictly above the diagonal of a symmetric matrix and stacks them into a vector. Then the Graphlet Correlation Distance between $G$ and $H$ is

$$\mathrm{GCD}_{11}(G,H)=
\left\|
\operatorname{vec}_{\triangle}(\mathrm{GCM}(G))-
\operatorname{vec}_{\triangle}(\mathrm{GCM}(H))
\right\|_2.$$

Thus, GCD-11 compares two graphs by measuring the Euclidean distance between their graphlet-correlation summaries.

Interpretation: summarize how graphlet roles co-vary across nodes in each graph, then measure how different those summaries are.

Lower values mean the two graphs are more similar in local topology.

## Caveats

- 5-clique counting and `C_4` may get expensive on larger graphs.
- This directory for **evaluation and analysis** of the graphs and we're not concerned with the computational intensity of computing the metrics