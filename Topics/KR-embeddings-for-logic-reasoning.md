








*2024-09-30*

#### [unKR: A Python Library for Uncertain Knowledge Graph Reasoning by Representation Learning](https://dl.acm.org/doi/abs/10.1145/3626772.3657661 )

*王靖婷、吴天星、陈仕林、刘云畅、朱曙曈、李伟、许婧怡、漆桂林*

*SIGIR 2024 Demonstration*

An uncertain knowledge graph (UKG) is typically a set of triples where each triple is associated with a (non-negative) confidence score. This paper proposes an embedding-based library for UKG with representation learning models under normal and few-shot settings, as well as evaluation on down-streaming tasks such as confidence prediction and link prediction.


*2024-05-14*

#### [Learning Rule-Induced Subgraph Representations for Inductive Relation Prediction](https://papers.nips.cc/paper_files/paper/2023/hash/0b06c8673ebb453e5e468f7743d8f54e-Abstract-Conference.html)

*Tianyu Liu, Qitan Lv, Jie Wang, Shuling Yang, Hanzhu Chen*

*NeurIPS 2023*

This paper proposes an inductive link prediction model, with a single-source initialization approach to assign a nonzero initial embedding for the target link according to its relation and zero embeddings for other links. It also proposes several RNN-based functions for edge-wise message passing to model the sequential property of mined rules, and uses the representation of the target link as the final subgraph representation.


*2024-05-11*

#### [Differentiable Neuro-Symbolic Reasoning on Large-Scale Knowledge Graphs](https://papers.nips.cc/paper_files/paper/2023/hash/5965f3a748a8d41415db2bfa44635cc3-Abstract-Conference.html)

*Shengyuan Chen, Yunfeng Cai, Huang Fang, Xiao Huang, Mingming Sun*

*NeurIPS 2023*

This paper proposes a link prediction model based on continuous Markov Logic Network. The model has three components: (1) a grounding module as a filter to identify crucial ground formulas and extract triples connected to them; (2) a KG-embedding model to compute truth scores for the extracted triples; and (3) a tailored continuous MLN framework that takes the truth scores as input and assesses the overall probability. It is optimized using an EM algorithm, alternating between embedding optimization and weight updating. During the E-step, the rule weights are fixed and embeddings are optimized in an end-to-end fashion by maximizing the overall probability. In the M-step, the rule weights are updated by leveraging the sparsity of violated rules. Note that the model requires a set of rules from certain rule-mining process or domain experts.


*2024-05-07*

#### [Prompt-fused framework for Inductive Logical Query Answering](https://arxiv.org/abs/2403.12646)

*Zezhong Xu, Peng Ye, Lei Liang, Huajun Chen, Wen Zhang*

*COLING 2024*

This paper proposes an embedding-based model for inductive QA task that involves emergence of new entities. It considers the query embeddings and addresses the embedding of emerging entities through contextual information aggregation.


*2024-03-05*

#### [On the Markov Property of Neural Algorithmic Reasoning: Analyses and Methods](https://openreview.net/forum?id=Kn7tWhuetn)

*Montgomery Bohde, Meng Liu, Alexandra Saxton, Shuiwang Ji*

*ICLR 2024 Spotlight*

A common paradigm of neural algorithmic reasoning uses historical embeddings in predicting the results of future execution steps. This paper argues that such historical dependence intrinsically contradicts the Markov nature of algorithmic reasoning tasks. Motivated by this, it presents ForgetNet, which does not use historical embeddings and thus is consistent with the Markov nature of the tasks. To address challenges in training ForgetNet at early stages, it further introduces G-ForgetNet, which uses a gating mechanism to allow for the selective integration of historical embeddings.


*2024-02-17*

#### [Knowledge Graph Reasoning over Entities and Numerical Values](https://dl.acm.org/doi/abs/10.1145/3580305.3599399)

*Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, Yangqiu Song*

*KDD 2023*

This paper proposes an approach for query answering over knowledge graphs that involves numerical reasoning. Specifically, the input query is processed sequentially by relational projection, attribute projection, numerical projection and reverse attribute projection. Besides, this paper provides a test set for query answering with numerical values based on FB15k, DB15k, and YAGO15k.


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
