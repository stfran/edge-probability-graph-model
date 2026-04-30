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


## Running the project
Running all the project code involved some manual efforts and some error corrections that we document below. Alternatively, unzip the results file as we describe in the next section.

Setup a virtual environment or conda environment and install requirements
```
pip isntall cluster_analysis/requirements.txt
```

Generate the graphs as described in main README (restating here for completeness)
```
cd data
python kron_seeds.py 
python er_cl_gen.py # updated line 10 of this file after observing an error
python sbm_gen.py # updated line 51 after observing an error, by sampling the unordered edge list instead of the dense matrix
```

fit the models to the ground truth data
```
cd fitting
bash fitting.sh all # split up the elif's into independent if's so the "all" option works
```

Note that this will not generate the ER models. To generate those, we placed the ER_iid.wls and ER_iter.wls files into wolfram cloud notebooks, manually iterated through the datasets, and then copied the results into the folder structure expected by the downstream


Generate the random graphs
```
cd generation
bash compile.sh # updated 
bash gen.sh
```

## Consolidating results
experiment_pipeline.py assumes all the edge-independent, local-binding, and parallel binding test results have been consolidated into the following directories. Copy all the following directories to ./cluster_analysis/results
```
mkdir -p ./cluster_analysis/results/data
cp -r ./data/orig_cl/ ./cluster_analysis/results/data/orig_cl/
cp -r ./data/orig_er/ ./cluster_analysis/results/data/orig_er/
cp -r ./data/orig_kr/ ./cluster_analysis/results/data/orig_kr/
cp -r ./data/orig_sbm/ ./cluster_analysis/results/data/orig_sbm/
cp -r ./gen_res/ ./cluster_analysis/results/gen_res/
```

Or if you have results.tar.gz, just unpack with
```
cd cluster_analysis
tar -xzvf results.tar.gz results
```

Edge-independent (baseline results)
- ./cluster_analysis/results/data/orig_cl (move from ./data/orgi_cl, etc.)
- ./cluster_analysis/results/data/orig_er
- ./cluster_analysis/results/data/orig_kr
- ./cluster_analysis/results/data/orig_sbm

Local Binding 
- ./cluster_analysis/results/gen_res/CL_iter/t1 
- ./cluster_analysis/results/gen_res/ER_iter/traingle
- ./cluster_analysis/results/gen_res/KR_iter/t1
- ./cluster_analysis/results/gen_res/SBM_iter/t1

Parallel Binding
- ./cluster_analysis/results/gen_res/CL_iid/t1
- ./cluster_analysis/results/gen_res/ER_iid/triangle
- ./cluster_analysis/results/gen_res/KR_iid/t1
- ./cluster_analysis/results/gen_res/SBM_iid/t1

Ground Truth (not part of results)
- ./data/gt_txt

Note that for ER, the original code outputs a directory named 'relax' that we do not include because it does not appear to be what is reported in the paper. Indeed, the other methods have commented out hyperparamenter choices, but we likewise only keep those reported in the paper.

## What are we comparing
Running all the project code (or unzipping the results) gives us all the data to reproduce the results reported in Table II of Bu et al. Beyond reproducing the table, we extend the original measurements in two ways. First, we add two metrics and a graph topology distance measurement. Second, we evaluate the metrics with a hypothesis test. 

For each metric, dataset, probability model, and generation method, we ask if the ground-truth graph is plausible under the distribution of the randomly generated graphs. 


## Metric Formulas 

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

For an undirected graph $G=(V,E)$, let $K_k(G)$ denote the set of all 4-cliques in $G$, i.e.,

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

Let $G=(V,E)$ be an undirected graph. Consider the 11 non-redundant graphlet orbits used in GCD-11. For each node $v\in V$, define its graphlet degree vector to be the 11-dimensional vector whose $a$-th entry counts how many times node $v$ participates in orbit $a$.

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

### Caveats

- 5-clique counting and `C_4` may get expensive on larger graphs.
- We're not concerned with the computational intensity of computing the metrics

## Hypothesis Tests

For each dataset, probability model, and generation method, we have one ground-truth graph and $B=100$ randomly generated graphs. Let

$$G_0$$

denote the ground-truth graph, and let

$$G_1,G_2,\ldots,G_B$$

denote the generated graphs for the same dataset/model/method combination.

The hypothesis tests below ask whether the ground-truth graph is plausible under the fitted random-graph generator. These are simulation-based goodness-of-fit tests, conditional on the fitted model parameters used to generate the random graphs.

We use

$$B=100$$

generated graphs and a per-test significance level of

$$\alpha=0.05.$$

Because we have a finite Monte Carlo sample of B=100 generated graphs, we use empirical p-values with a plus-one correction (Phipson and Smith).

---

### 1. Scalar-metric tests: GCC, ALCC, 4-clique density, and $C_3$

This test applies to scalar graph metrics:

$$
m \in \{\mathrm{GCC}, \mathrm{ALCC}, \delta_4, C_3\}.
$$

For a fixed metric $m$, compute the observed ground-truth value

$$
x_0 = m(G_0)
$$

and the generated values

$$
x_i = m(G_i), \qquad i=1,\ldots,B.
$$

The null and alternative hypotheses are

$$
H_0: x_0 \text{ is consistent with the distribution of } m(G_i)
$$

and

$$
H_A: x_0 \text{ is unusually small or unusually large under the distribution of } m(G_i).
$$

This is a two-sided test because a model can fail by either under-producing or over-producing the metric. Put another way, rejecting the null hypothesis would be evidence that the model and generation combo is not realistic with respect to the metric, at least not in the real structure of the ground truth data. 

Define the lower-tail and upper-tail counts as

$$
n_{\leq} = \sum_{i=1}^{B} \mathbf{1}(x_i \leq x_0)
$$

and

$$
n_{\geq} = \sum_{i=1}^{B} \mathbf{1}(x_i \geq x_0).
$$

The two-sided empirical Monte Carlo p-value is

$$
p_m =
\min\left(
1,\,
2\min\left\{
\frac{n_{\leq}+1}{B+1},
\frac{n_{\geq}+1}{B+1}
\right\}
\right).
$$

With $B=100$, this becomes

$$
p_m =
\min\left(
1,\,
2\min\left\{
\frac{n_{\leq}+1}{101},
\frac{n_{\geq}+1}{101}
\right\}
\right).
$$

We reject $H_0$ when

$$
p_m \leq 0.05.
$$

For the scalar metrics, the p-value answers whether the ground-truth metric value is unusally extreme relative to the 100 generated graphs.


For reporting, we also compute the generated mean, generated standard deviation, and simulation z-score:

$$
\bar{x} = \frac{1}{B}\sum_{i=1}^{B}x_i,
$$

$$
s_x =
\sqrt{
\frac{1}{B-1}
\sum_{i=1}^{B}
(x_i-\bar{x})^2
},
$$

and

$$
z_m = \frac{x_0-\bar{x}}{s_x}.
$$

The z-score measures how many standard deviations (in the generated distribution) away is the ground-truth value from the generated mean. We can use the z-score to rank model and method combinations under the same dataset and metric. 

For each scalar metric, report:

- ground-truth value $x_0$
- generated mean $\bar{x}$
- generated standard deviation $s_x$
- simulation z-score $z_m$
- empirical p-value $p_m$
- reject / fail-to-reject decision at $\alpha=0.05$

---

### 2. GCD-11 distance test

GCD-11 is a distance measurement between two graphs. Therefore, we test whether the ground-truth graph is unusually far from the generated graphs compared with how far the generated graphs are from each other.

Let

$$
d(G,H)=\mathrm{GCD}_{11}(G,H).
$$

First compute the average GCD-11 distance from the ground-truth graph to the generated graphs:

$$
T_0 =
\frac{1}{B}
\sum_{i=1}^{B}
d(G_0,G_i).
$$

Then compute an analogous leave-one-out distance for each generated graph. For each $i=1,\ldots,B$, define

$$
T_i =
\frac{1}{B-1}
\sum_{\substack{j=1 \\ j\neq i}}^{B}
d(G_i,G_j).
$$

Here, $T_i$ measures how far generated graph $G_i$ is from the other generated graphs. The collection

$$
T_1,T_2,\ldots,T_B
$$

forms the empirical reference distribution for typical generated-graph variability.

The null and alternative hypotheses are

$$
H_0: G_0 \text{ is exchangeable with the generated graphs under GCD-11 distance}
$$

and

$$
H_A: G_0 \text{ is farther from the generated graphs than generated graphs are from each other}.
$$

This is a one-sided test because smaller GCD-11 values indicate greater local-topology similarity. The model fails this test only when the ground-truth graph is unusually far from the generated graphs.

The empirical Monte Carlo p-value is

$$
p_{\mathrm{GCD}} =
\frac{
1+\sum_{i=1}^{B}\mathbf{1}(T_i \geq T_0)
}{
B+1
}.
$$

With $B=100$, this becomes

$$
p_{\mathrm{GCD}} =
\frac{
1+\sum_{i=1}^{100}\mathbf{1}(T_i \geq T_0)
}{
101
}.
$$

We reject $H_0$ when

$$
p_{\mathrm{GCD}} \leq 0.05.
$$

With 100 generated graphs, this means the ground-truth graph must be farther from the generated graphs than almost all generated graphs are from each other. Specifically, rejection at $\alpha=0.05$ requires

$$
\sum_{i=1}^{100}\mathbf{1}(T_i \geq T_0) \leq 4.
$$

For reporting, we also compute

$$
\bar{T} =
\frac{1}{B}
\sum_{i=1}^{B}
T_i
$$

and

$$
s_T =
\sqrt{
\frac{1}{B-1}
\sum_{i=1}^{B}
(T_i-\bar{T})^2
},
$$

along with the GCD-11 distance z-score

$$
z_{\mathrm{GCD}} =
\frac{T_0-\bar{T}}{s_T}.
$$

As with the scalar metrics, the z-score is only an effect-size summary. The empirical p-value is the hypothesis-test decision rule.

For GCD-11, report:

- ground-truth-to-generated mean distance $T_0$
- generated-to-generated mean distance $\bar{T}$
- generated-to-generated standard deviation $s_T$
- simulation z-score $z_{\mathrm{GCD}}$
- empirical p-value $p_{\mathrm{GCD}}$
- reject / fail-to-reject decision at $\alpha=0.05$

---

### Interpretation of test outcomes

Failing to reject $H_0$ does not prove that the fitted model is correct. It means that, for the selected metric and the available 100 generated graphs, the ground-truth graph is not statistically unusual relative to the generated graphs.

Rejecting $H_0$ means that the fitted generator does not reproduce the ground-truth graph with respect to the tested metric. For scalar metrics, this means the ground-truth value is unusually low or unusually high. For GCD-11, this means the ground-truth graph is unusually far from the generated graphs in local graphlet-correlation structure.

Because these tests are run across multiple datasets, models, generation methods, and metrics, p-values should be interpreted as metric-specific evidence rather than as a single global claim about model realism.


## References
### Primary project paper

Bu, F., Yang, R., Bogdan, P., & Shin, K. (2025). *Edge probability graph models beyond edge independency: Concepts, analyses, and algorithms*. Proceedings of the IEEE International Conference on Data Mining.

### Higher-order clustering

Yin, H., Benson, A. R., & Leskovec, J. (2018). Higher-order clustering in networks. *Physical Review E, 97*(5), 052306. https://doi.org/10.1103/PhysRevE.97.052306

### Graphlet correlation distance: original method paper

Yaveroglu, O. N., Malod-Dognin, N., Davis, D., Levnajic, Z., Janjic, V., Karapandza, R., Stojmirovic, A., & Przulj, N. (2014). Revealing the hidden language of complex networks. *Scientific Reports, 4*, 4547. https://doi.org/10.1038/srep04547

### Graphlet correlation distance: evaluation paper supporting the GCD-11 choice

Yaveroglu, O. N., Milenkovic, T., & Przulj, N. (2015). Proper evaluation of alignment-free network comparison methods. *Bioinformatics, 31*(16), 2697-2704. https://doi.org/10.1093/bioinformatics/btv170

### Hypothesis-test references

Hope, A. C. A. (1968). A simplified Monte Carlo significance test procedure. *Journal of the Royal Statistical Society: Series B (Methodological), 30*(3), 582-598. https://doi.org/10.1111/j.2517-6161.1968.tb00759.x

Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero: Calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology, 9*(1). https://doi.org/10.2202/1544-6115.1585