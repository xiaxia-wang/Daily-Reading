


*2023-11-08*

#### [Combining Inductive and Deductive Reasoning for Query Answering over Incomplete Knowledge Graphs](https://dl.acm.org/doi/10.1145/3583780.3614816)

*Medina Andresel, Trung-Kien Tran, Csaba Domokos, Pasquale Minervini, Daria Stepanova*

*CIKM 2023*

This paper proposes an embedding-based method by taking both inductive and deductive reasoning into consideration for question answering. Specifically, it applies ontology-aware data sampling to obtain the training data for the embedding model (e.g., Query2Box, etc.). It considers the DL-Lite_R as the description logic fragment, and uses a set of query shapes (DAGs) in the process of ontology-based data sampling.


*2023-03-20*

#### [SMORE: Knowledge Graph Completion and Multi-hop Reasoning in Massive Knowledge Graphs](https://dl.acm.org/doi/10.1145/3534678.3539405)

*Hongyu Ren, Hanjun Dai, Bo Dai, Xinyun Chen, Denny Zhou, Jure Leskovec, Dale Schuurmans*

*KDD 2022*

This paper proposes a unified method for single-hop (i.e., link prediction) and multi-hop reasoning over knowledge graphs. It implements neural transformations to simulate logical relations. To improve the computation efficiency, it proposes a bidirectional negative sampling method to obtain better training data, and operates on the full KG directly in a shared memory environment with multiple GPUs.


*2023-03-19*

#### [Compute Like Humans: Interpretable Step-by-step Symbolic Computation with Deep Neural Network](https://dl.acm.org/doi/10.1145/3534678.3539276)

*Shuai Peng, Di Fu, Yong Cao, Yijun Liang, Gu Xu, Liangcai Gao, Zhi Tang*

*KDD 2022*

This paper proposes a deep neural network model for simulating math expressions' transformation in a step-by-step manner. The task is formulated as to find an acyclic path of transformation steps based on given start expression and target expression. It is divided into 2 subtasks, (1) valid transformation prediction, and (2) next expression generation. It does not consider the optimality of different paths. The model is built upon the encoder-decoder architecture, with the positional encoding based on the tree structure of the math expression.


*2023-03-13*

#### [Mask and Reason: Pre-Training Knowledge Graph Transformers for Complex Logical Queries](https://dl.acm.org/doi/10.1145/3534678.3539472)

*Xiao Liu, Shiyu Zhao, Kai Su, Yukuo Cen, Jiezhong Qiu, Mengdi Zhang, Wei Wu, Yuxiao Dong, Jie Tang*

*KDD 2022*

This paper proposes a pre-training framework for logical query answering over knowledge graphs. It formulates a KG transformation strategy to turn relations into nodes thus eliminating the edge labels, and proposes a Mixture-of-Experts strategy to enhance the activation component of the transformers' feed-forward layers.


*2023-03-09*

#### [GammaE: Gamma Embeddings for Logical Queries on Knowledge Graphs](https://aclanthology.org/2022.emnlp-main.47/)

*Dong Yang, Peijun Qing, Yang Li, Haonan Lu, Xiaodong Lin*

*EMNLP 2022*

This paper proposes a probabilistic embedding model for FOL queries over knowledge graphs. It implements a Gamma mixture method to alleviate the non-closure problem on union operators, and enables query embeddings to have strict boundaries on the negation operator.


*2023-03-08*

#### [Embedding Logical Queries on Knowledge Graphs](https://proceedings.neurips.cc/paper/2018/hash/ef50c335cca9f340bde656363ebd02fd-Abstract.html)

*William L. Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, Jure Leskovec*

*NeurIPS 2018*

This paper introduces a method for conjunctive queries over knowledge graphs. The input contains a set of anchor nodes, and the target is to get the embedding of the query variable. The conjunctive query is formulated as a DAG, and two transformation operators are learned for projection and intersection in the embedding space, respectively.
