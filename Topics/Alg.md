






*2024-02-01*

#### [Keyword Search over Knowledge Graphs via Static and Dynamic Hub Labelings](https://dl.acm.org/doi/10.1145/3366423.3380110)

*Yuxuan Shi, Gong Cheng, Evgeny Kharlamov*

*WWW 2020*

This paper proposes practical approximation algorithms with a guaranteed quality to solve the GST problem. It relies on Hub Labeling (HL), which is a structure that labels each vertex in a graph with a list of vertices reachable from it for computing distances and shortest paths. It proposes two HLs: one is a conventional static HL that uses a new heuristic to improve pruned landmark labeling, and the other is a dynamic HL that inverts and aggregates query-relevant static labels to more efficiently process vertex sets.


*2023-10-26*

#### [Fast Online Node Labeling for Very Large Graphs](https://proceedings.mlr.press/v202/zhou23k.html)

*Baojian Zhou, Yifan Sun, Reza Babanezhad Harikandeh*

*ICML 2023*

This paper proposes an online node labeling framework under the transductive learning setting. Based on an online relaxation technique it provides an approximate algorithm with an effective regret.


*2023-07-27*

#### [Graph Summarization via Node Grouping: A Spectral Algorithm](https://dl.acm.org/doi/10.1145/3539597.3570441)

*Arpit Merchant, Michael Mathioudakis, Yanhao Wang*

*WSDM 2023*

Graph summarization via node grouping builds a concise graph by grouping neighboring nodes into supernodes and encoding edges as superedges to minimize loss of adjacency information. This paper reformulates the loss minimization as an integer maximization problem, and proposes a two-stage algorithm to solve that. It first assigns nodes to supernodes with k-means clustering on the k largest (in magnitude) eigenvectors of the adjacency matrix, and secondly uses a greedy heuristic to further improve the summary.


*2023-07-21*

#### [Everything Evolves in Personalized PageRank](https://dl.acm.org/doi/10.1145/3543507.3583474)

*Zihao Li, Dongqi Fu, Jingrui He*

*WWW 2023*

This paper aims to solve the Personalized PageRank effectively and efficiently in a fully dynamic setting, by tracking the exact solution at each time stamp. It proposes an offset score propagation method and shows its correctness. It can be applied for dynamic knowledge graph alignment.


*2023-05-13*

#### [Efficient Approximation Algorithms for the Diameter-Bounded Max-Coverage Group Steiner Tree Problem](https://dl.acm.org/doi/10.1145/3543507.3583257)

*Ke Zhang, Xiaoqing Wang, Gong Cheng*

*TheWebConf 2023*

This paper formalizes the problem of Diameter-bounded max-Coverage GST (DCGST) as a generalized version of typical GST problem. It proposes a first approximation algorithm, analyzes its approximation ratio, and demonstrates its effectiveness over real-world datasets.


*2022-10-12*

#### [Efficient and Effective Similarity Search over Bipartite Graphs](https://dl.acm.org/doi/10.1145/3485447.3511959)

*Renchi Yang*

*TheWebConf 2022*

This paper investigates the problem of similarity search over bipartite graphs. Based on the recently proposed hidden personalized PageRank (HPP) algorithm, it firstly identifies a drawback of HPP, and then proposes bidirectional HPP (BHPP) to overcome it. It formulates the similarity search over bipartite graphs as a BHPP problem, and also proposes an approximate algorithm to apply it to real-world graphs.


*2022-08-18*

#### [A Unifying Framework for Mining Approximate Top-k Binary Patterns](https://doi.org/10.1109/TKDE.2013.181)

*Claudio Lucchese, Salvatore Orlando, Raffaele Perego*

*IEEE TKDE 2014*

This paper focuses on approximate pattern mining for binary matrices. Firstly, it reviews several greedy pattern mining algorithms, i.e., Asso, Hyper+, and PaNDa, and introduces their cost functions. Then it proposes PaNDa+ as a framework able to optimize different cost functions within a unifying formulation. The algorithms is evaluated on synthetic and real-world datasets, respectively. The source codes of PaNDa+ are provided on [the author's website](https://claudio-lucchese.github.io/archives/20131113/index.html). 


*2022-08-15*

#### [Scaling up top-K cosine similarity search](https://doi.org/10.1016/j.datak.2010.08.004)

*Shiwei Zhu, Junjie Wu, Hui Xiong, Guoping Xia*

*Data & Knowledge Engineering 2011*

Instead of setting a similarity threshold, this paper proposes a top-K similarity search method with cosine similarity. Based on the monotone of upper bound of the cosine measure, it adopts a diagonal traversal strategy for pruning the search space. Besides, it also proposes a max-first strategy to optimize the I/O cost. 


*2022-08-14*

#### [Scaling Up All Pairs Similarity Search](https://doi.org/10.1145/1242572.1242591)

*Roberto J. Bayardo, Yiming Ma, Ramakrishnan Srikant*

*WWW 2007*

This paper investigates the problem of all pairs similarity search, i.e., finding all pairs of vectors whose similarity score (e.g., cosine distance) is above a given threshold. Starting with a typical inverted index-based approach, it optimizes the threshold during the indexing process and refines a sorted order for the record list. It also optimizes the matching process by refining the threshold and customizes for binary vector data. In the experiment, the proposed method is compared with existing inverted index-based method and signature-based methods. 


*2022-08-13*

#### [Fast Matching for All Pairs Similarity Search](https://doi.org/10.1109/WI-IAT.2009.52)

*Amit C. Awekar, Nagiza F. Samatova*

*Web Intelligence 2009*

This paper proposes a general framework for all-pairs similarity search which contains 3 main steps, i.e., data preprocessing, pairs matching, and indexing. It optimizes the matching step by incorporating (1) a lower bound on the number of non-zero components in any record, (2) a upper bound on the similarity score for any record pair. The detailed algorithm and the upper/lower bounds are also presented in the paper. 


*2022-08-12*

#### [Efficient Similarity Joins for Near Duplicate Detection](https://doi.org/10.1145/1367497.1367516)

*Chuan Xiao, Wei Wang, Xuemin Lin, Jeffrey Xu Yu*

*WWW 2008*

This paper proposes PPJOIN, an efficient set similarity join approach. For a given similarity threshold and a list of records ranked by length and dictionary (for Jaccard similarity), it iterates over the records, filters them by prefixes, and verifies the exact similarity of a few candidates. It also incorporates a map-based optimization for the verification process. 


*2022-08-11*

#### [Efficient Set Similarity Joins Using Min-prefixes](https://doi.org/10.1007/978-3-642-03973-7_8)

*Leonardo Ribeiro, Theo Härder*

*ADBIS 2009*

This paper proposes a new approach for set similarity join. Unlike existing methods focusing on efficiently pruning the number of candidate sets, it achieves a trade-off between the cost of candidate generation and the verification phase. The workload for verification is increased while the computational cost for candidate generation is decreased, i.e., the length of prefixes is decreased. It modifies the SOTA similarity join method PPJOIN with minimal length of the prefixes to construct the inverted index. The proposed method is thus named MPJOIN. 


*2022-08-10*

#### [Efficient set joins on similarity predicates](https://doi.org/10.1145/1007568.1007652)

*Sunita Sarawagi, Alok Kirpal*

*SIGMOD 2004*

This paper proposes an optimized similarity join method based on existing algorithms and inverted indexes. Here, the predicates mean similarity measurements, such as set intersect, Jaccard coefficient or cosine similarity. The task is formulated as finding all pairs of word sets for which the total weight of common words exceeds the threshold. The approach begins by building an inverted index, then ranking the retrieved set list using a heap. Besides, it also adds a clustering step for related sets to further optimize the search space and memory cost.


*2022-08-08*

#### [Efficient Exact Set-Similarity Joins](https://dl.acm.org/doi/10.5555/1182635.1164206)

*Arvind Arasu, Venkatesh Ganti, Raghav Kaushik*

*VLDB 2006*

This paper proposes a precise algorithm named PartEnum for set similarity joins. It firstly generates signatures for the input sets, and finds all the pairs of sets with signatures overlap. Based on the overlap of signatures, it outputs all the final similar pairs as the result. The signature is like a compression of the original vector to speed up the comparison. This paper also proposes an evaluation framework for the performance of set similarity join methods. 


*2022-08-07*

#### [A Primitive Operator for Similarity Joins in Data Cleaning](https://doi.org/10.1109/ICDE.2006.9)

*Surajit Chaudhuri, Venkatesh Ganti, Raghav Kaushik*

*ICDE 2006*

This paper proposes SSJOIN, a primitive operator for set similarity join. SSJOIN can support different kinds of similarity measures such as edit distance, Jaccard similarity, co-occurrence. Apart from the examples given in the paper, it proposes implementing details and the comparison with existing similarity join methods. 


*2022-05-13*

#### [Top-k Set Similarity Joins](https://doi.org/10.1109/ICDE.2009.111)

*Chuan Xiao, Wei Wang, Xuemin Lin, Haichuan Shang*

*ICDE 2009*

This paper introduces an algorithm to identify top-k similar record pairs from a large given corpus, which doesn't require a pre-defined similarity threshold. It is adapted from an existing similarity join algorithm named All-Pairs, by changing the current threshold while keeping top-k similar pairs, and devising a new stopping condition. In the experiment, this top-k join algorithm was compared with ppjoin-topk on several large datasets. 

