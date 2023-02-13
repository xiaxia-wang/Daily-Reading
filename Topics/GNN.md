

*2023-02-13*

#### [Relational Graph Attention Network for Aspect-based Sentiment Analysis](https://doi.org/10.18653/v1/2020.acl-main.295)

*Kai Wang, Weizhou Shen, Yunyi Yang, Xiaojun Quan, Rui Wang*

*ACL 2020*

This paper proposes a GAT-based model for fine-grained (aspect-based) text sentiment analysis. It firstly constructs a dependency tree based on direct and indirect dependencies between tokens for each aspect. Then it applies the GAT mechanism with respect to each relation, and aggregates them as overall message-passing weights. The model contains a set of $K$ attention heads and $M$ relational heads.


*2023-02-12*

#### [Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ)

*Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio*

*ICLR 2018*

This is a milestone paper that firstly introduce the attention mechanism to graph neural network. It implements the graph attention mechanism as a feed-forward network on each layer to learn the weights of message passing between neighboring nodes. It also incorporates the multi-head mechanism to capture different features over the same edge.

*2023-02-10*

#### [Modeling Relational Data with Graph Convolutional Networks](https://link.springer.com/chapter/10.1007/978-3-319-93417-4_38)

*Michael Sejr Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling*

*ESWC 2018*

This paper proposes a graph convolutional network model for relational data named R-GCN. It optimizes the structure of basic GCN model with a normalized aggregation over each relation for massage passing. The evaluation over the link prediction and entity classificaation tasks demonstrates the effectiveness of R-GCN, especially its relation-normalized encoder.

*2023-02-08*

#### [Inductive Relation Prediction by Subgraph Reasoning](http://proceedings.mlr.press/v119/teru20a.html)

*Komal K. Teru, Etienne G. Denis, William L. Hamilton*

*ICML 2020*

This paper proposes a GNN-based inductive relation prediction model named Grail. It assumes the underlying relation between two nodes (entities) can be represented by the local subgraph structure (paths between the nodes), thus can be applied under the inductive setting. It adopts one-hot encoding vectors as the node feature, and implements message passing over the radius-bounded subgraphs. The experimental result shows it achieves SOTA performance among existing link prediction methods.

*2022-11-16*

#### [Contrastive Knowledge Graph Error Detection](https://doi.org/10.1145/3511808.3557264)

*Qinggang Zhang, Junnan Dong, Keyu Duan, Xiao Huang, Yezi Liu, Linchuan Xu*

*CIKM 2022*

Unlike existing knowledge graph error detection methods which generally rely on negative sampling, this paper introduces a contrastive learning model by creating different hyper-views of the KG, and regards each relational triple as a node. The optimize target includes the consistency of triple representations among the multi-views and the self-consistency within each triple. In this paper, the two views of the KG are defined by two link patterns, i.e., two triples sharing head entity, or sharing tail entity.

*2022-11-15*

#### [Taxonomy-Enhanced Graph Neural Networks](https://doi.org/10.1145/3511808.3557467)

*Lingjun Xu, Shiyin Zhang, Guojie Song, Junshan Wang, Tianshu Wu, Guojun Liu*

*CIKM 2022*

This paper proposes to incorporate the external taxonomy knowledge into the GNN learning process of nodes embeddings. For the taxonomy, instead of using a vector, it firstly maps each category to a Gaussian distribution, and calculates the mutual information between them. In the downstream GNN model, these categories are used to characterize the similarity of node pairs. The context of each node is represented by the mean vector of categories of neighboring nodes.

*2022-11-14*

#### [Large-scale Entity Alignment via Knowledge Graph Merging, Partitioning and Embedding](https://doi.org/10.1145/3511808.3557374)

*Kexuan Xin, Zequn Sun, Wen Hua, Wei Hu, Jianfeng Qu, Xiaofang Zhou*

*CIKM 2022*

This paper proposes three strategies for scalable GNN-based entity alignment without losing too much structural information. It follows the pipeline of partitioning, merging the knowledge graph and generating the alignment. In the partitioning process, it identifies a set of landmark entities to connect different subgraphs. To reduce the structure loss, it also applies an entity reconstruction mechanism to incorporate information from its neighborhood. Besides, it implements entity search in a fused unified space of multiple subgraphs.

*2022-11-13*

#### [Incorporating Peer Reviews and Rebuttal Counter-Arguments for Meta-Review Generation](https://doi.org/10.1145/3511808.3557360)

*Po-Cheng Wu, An-Zi Yen, Hen-Hsen Huang, Hsin-Hsi Chen*

*CIKM 2022*

This paper investigates the problem of meta-review generation based on peer reviews and the authors' rebuttal. The authors collect a dataset of submissions, reviews and rebuttal responses from ICLR 2017--2021. To solve the problem, they firstly extract all the argumentative discourse units (ADUs) and three level of relations (i.e., intra-document, intra-discussion, and inter-discussion relations) between the ADUs. Then they construct a content (text) encoder model with a graph attention network, and aggregate them to generate the meta-review. The overall model is trained in a seq2seq manner.

*2022-11-11*

#### [Reinforced Continual Learning for Graphs](https://doi.org/10.1145/3511808.3557427)

*Appan Rakaraddi, Siew-Kei Lam, Mahardhika Pratama, Marcus de Carvalho*

*CIKM 2022*

This paper proposes a graph continual learning model for the task of node classification. It consists of a reinforcement learning based controller to manage adding and deleting node features, and a GNN as child network to deal with the tasks. It supports both task-incremental and class-incremental settings for node classification.

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
