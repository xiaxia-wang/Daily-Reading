






*2022-11-01*

#### [Exploring Edge Disentanglement for Node Classification](https://dl.acm.org/doi/10.1145/3485447.3511929)

*Tianxiang Zhao, Xiang Zhang, Suhang Wang*

*TheWebConf 2022*

This paper proposes a GNN model to identify different edge attributives (edge disentanglement) for node classification task. It implements a set of disentangled channels to capture different edge attributes, and provides three self-supervision signals to learn edge disentanglement. 


*2022-10-31*

#### [Learning and Evaluating Graph Neural Network Explanations based on Counterfactual and Factual Reasoning](https://doi.org/10.1145/3485447.3511948)

*Juntao Tan, Shijie Geng, Zuohui Fu, Yingqiang Ge, Shuyuan Xu, Yunqi Li, Yongfeng Zhang*

*TheWebConf 2022*

This paper proposes a GNN model to generate subgraphs as explanation for the graph classification task based on factual and counterfactual reasoning. It proposes two objectives that a good explanation should be (1) sufficient and necessary (related to factual and counterfactual reasoning), and (2) simple (driven by the Occam’s Razor Principle). 


*2022-10-24*

#### [Rethinking Graph Convolutional Networks in Knowledge Graph Completion](https://doi.org/10.1145/3485447.3511923)

*Zhanqiu Zhang, Jie Wang, Jieping Ye, Feng Wu*

*TheWebConf 2022*

This paper proposes an idea that the graph structure modeling is unimportant for GCN-based KGC models. Instead, the ability to distinguish different entities and the transformations for entity embeddings account for the performance improvements. To prove this, firstly, it randomly changes the adjacency tensors in message passing and surprisingly gets similar results. Besides, removing the self-loop information also results in similar performance. Based on that, this paper also proposes a KG embedding model which applies linear transformation to entity representations. 


*2022-10-22*

#### [Trustworthy Knowledge Graph Completion Based on Multi-sourced Noisy Data](https://doi.org/10.1145/3485447.3511938)

*Jiacheng Huang, Yao Zhao, Wei Hu, Zhen Ning, Qijin Chen, Xiaoxia Qiu, Chengfu Huo, Weijun Ren*

*TheWebConf 2022*

This paper works on open knowledge graph completion based on noisy data. It firstly proposes a holistic fact scoring function for both relational facts and literal facts (triples). Then it proposes a neural network model to align the heterogeneous values from different facts. It also implements a semi-supervised inference model to predict the trustworthiness of the claims. 


*2022-10-18*

#### [Knowledge Graph Reasoning with Relational Digraph](https://doi.org/10.1145/3485447.3512008)

*Yongqi Zhang, Quanming Yao*

*TheWebConf 2022*

This paper proposes a knowledge graph reasoning network named RED-GNN to answer the queries in the form of $\langle$ subject entity, relation, ? $\rangle$. It introduces a relational directed graph (r-digraph) to capture the entity relation connections in the KG, and uses dynamic programming to recursively encode multiple r-digraphs with shared edges. It achieves relatively good performance among existing GNN models, and provides their codes. 


*2022-10-17*

#### [KoMen: Domain Knowledge Guided Interaction Recommendation for Emerging Scenarios](https://dl.acm.org/doi/10.1145/3485447.3512177)

*Yiqing Xie, Zhen Wang, Carl Yang, Yaliang Li, Bolin Ding, Hongbo Deng, Jiawei Han*

*TheWebConf 2022*

This paper studies a problem of user-user interaction recommendation in emerging scenarios, which is formulated as a few-shot link prediction task over a multiplex graph. To solve the problem, it proposes a model containing two levels of attention mechanism. Each of the several experts is trained on one type of edges (i.e., learns the attention over each type of edges), while different scenarios choose different combination of experts (i.e., learn the attention over experts). 


*2022-09-28*

#### [INDIGO: GNN-Based Inductive Knowledge Graph Completion Using Pair-Wise Encoding](https://proceedings.neurips.cc/paper/2021/hash/0fd600c953cde8121262e322ef09f70e-Abstract.html)

*Shuwen Liu, Bernardo Cuenca Grau, Ian Horrocks, Egor V. Kostylev*

*NeurIPS 2021*

This paper focuses on the task of knowledge graph completion. While existing GNN models for knowledge graph completion generally use transductive features of the KG and cannot handle unseen entities during the test phase, this paper improves the GCN model to address the problem by changing the encoder and decoder components. Each node in this GNN model is designed to represent a pair of entities in the original KG. Besides, its transparent nature allows the predicted triples to be read out directly without the need of an additional predicting layer. 


*2022-04-25*

#### [Graph Neural Networks: Taxonomy, Advances, and Trends](https://dl.acm.org/doi/10.1145/3495161)

*Yu Zhou, Haixia Zheng, Xin Huang, Shufeng Hao, Dengao Li, Jumin Zhao*

*ACM TIST 2022*

It's a comprehensive survey paper that introduces the GNN roadmap on four levels, and worth reading for several times. (1) The fundamental architectures present the basic GNN modules and operations such as graph attention and pooling. (2) The extended architectures and applications discuss various tasks such as pre-training framework, reinforcement learning and application scenarios of GNN modules such as NLP, recommendation systems. (3) The implementations and evaluations introduce commonly used tools and benchmark datasets. (4) The future research directions provide potential directions includes highly scalable/robust/interpretable GNNs, and GNNs going beyond WL test. 


*2022-04-22*

#### [Graph Neural Networks for Graphs with Heterophily: A Survey](https://arxiv.org/abs/2202.07082)

*Authors: Xin Zheng, Yixin Liu, Shirui Pan, Miao Zhang, Di Jin, Philip S. Yu*

*Arxiv 2022*

A survey about GNNs for heterophilic graphs. Generally, most existing GNN models depends on the homophily assumption, i.e., the nodes with same or similar labels are more likely to be linked than those with different labels, such as the citation network. But there are also many real-world graphs do not obey this rule, e.g., online transaction networks, dating networks. This paper surveys existing research efforts for GNNs over heterophilic graphs in two folds: (1) non-local neighbor extension, which try to obtain information from similar but distant nodes (2) GNN architecture refinement, which try to modify the information aggregation methods on the model side. This paper has also suggested 4 future work directions: (1) interpretability and robustness, (2) scalable heterophilic GNNs (how to implement large models, how to train, how to sample batches), (3) heterophily and over-smoothing, (4) comprehensive benchmark and metrics (real-world, larger graphs). 

