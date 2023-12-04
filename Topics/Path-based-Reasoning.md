








*2023-12-04*

#### [On the Correspondence Between Monotonic Max-Sum GNNs and Datalog](https://proceedings.kr.org/2023/64/)

*David Tena Cucala, Bernardo Cuenca Grau, Boris Motik, Egor V. Kostylev*

*KR 2023*

This paper investigates the max-sum GNN model which is based on a canonical encoding scheme for a graph and extra restrictions (e.g., non-negative weight matrices and activation functions) to the model.


*2023-07-23*

#### [PaGE-Link: Path-based Graph Neural Network Explanation for Heterogeneous Link Prediction](https://dl.acm.org/doi/10.1145/3543507.3583511)

*Shichang Zhang, Jiani Zhang, Xiang Song, Soji Adeshina, Da Zheng, Christos Faloutsos, Yizhou Sun*

*WWW 2023*

This paper proposes a GNN-based model for explaining link prediction results produced by the model. Given a GNN model and a predicted link on a heterogeneous graph (e.g., user-item-attribute in e-commerce), the model generates paths as explanation for the predicted link.


*2023-07-04*

#### [Topic-enhanced Graph Neural Networks for Extraction-based Explainable Recommendation](https://le-wu.com/files/Publications/CONFERENCES/SIGIR-23-shuai.pdf)

*Jie Shuai, Le Wu, Kun Zhang, Peijie Sun, Richang Hong, Meng Wang*

*SIGIR 2023*

This paper proposes a topic-based GNN model based on user reviews to achieve explainable recommendation. It first uses a pre-trained topic model to analyze reviews at the topic level, and designs a sentence-enhanced topic graph to explicitly model user preference, where topics are intermediate nodes between users and items. Corresponding sentences serve as edge features. Secondly, a review-enhanced rating graph is built to implicitly model user preference. The user and item representations from two graphs are then used for final rating prediction and explanation extraction.


*2023-06-13*

#### [M-Walk: Learning to Walk over Graphs using Monte Carlo Tree Search](https://proceedings.neurips.cc/paper/2018/hash/c6f798b844366ccd65d99bc7f31e0e02-Abstract.html)

*Yelong Shen, Jianshu Chen, Po-Sen Huang, Yuqing Guo, Jianfeng Gao*

*NeurIPS 2018*

This is another work using reinforcement learning to handle path-based knowledge graph reasoning. To overcome the challenge of sparse rewards, it develops a graph-walking agent called M-Walk, which consists of a deep recurrent neural network and Monte Carlo Tree Search (MCTS). The RNN encodes the state (i.e., history of the walked path) and maps it separately to a policy and Q-values.


*2023-06-12*

#### [Multi-Hop Knowledge Graph Reasoning with Reward Shaping](https://doi.org/10.18653/v1/d18-1362)

*Xi Victoria Lin, Richard Socher, Caiming Xiong*

*EMNLP 2018*

Similar to some other papers (e.g., MINERVA), this paper also formulates knowledge graph reasoning as a path-based reinforcement learning problem, and uses a MDP model to solve it. Specifically, it improves the model by (1) adopting pre-trained embeddings to estimate the model rewards for reducing the impact of false negative samples, and (2) forcing the agent to explore diverse paths with randomly generated edge masks.


*2023-06-11*

#### [Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning](https://openreview.net/forum?id=Syg-YfWCW)

*Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishnamurthy, Alex Smola, Andrew McCallum*

*ICLR 2018*

This paper proposes a reinforcement learning method for knowledge graph completion. Specifically, it formulates the problem as given a query entity and a relation, to predict the other entity connected to the given one via the relation. It uses a MDP architecture to search for the reasoning path by representing each state with a 4-tuple consisting of the given entity, given relation, current entity in the path, and the answer entity. It applies a LSTM network to encode the historical states.


*2023-06-10*

#### [Scalable Rule Learning via Learning Representation](https://doi.org/10.24963/ijcai.2018/297)

*Pouya Ghiasnezhad Omran, Kewen Wang, Zhe Wang*

*IJCAI 2018*

This paper also investigates the problem of path-based rule learning. Instead of using a refinement operator to search the rule space, it applies embedding models to effectively prune the search. Specifically, for a given rule head, it firstly samples a small part from the original KG containing only entities and relations related to the rule head. Then it performs rule search over the embedded subgraph with rule evaluation for instant pruning the search space.


*2023-06-09*

#### [Variational Knowledge Graph Reasoning](https://doi.org/10.18653/v1/n18-1165)

*Wenhu Chen, Wenhan Xiong, Xifeng Yan, William Yang Wang*

*NAACL 2018*

This paper proposes a path-based reasoning model for knowledge graph completion. It assumes that each relation between a pair of query entities can be represented by a "latent" path in the graph. Therefore, it combines a CNN-based path reasoner with an LSTM-based path finder in the model. It also takes negative samples into the training process.


*2023-06-08*

#### [DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning](https://doi.org/10.18653/v1/d17-1060)

*Wenhan Xiong, Thien Hoang, William Yang Wang*

*EMNLP 2017*

This paper applies a reinforcement learning method to select the most promising relation over the knowledge graph for extending the reasoning path. The selection is embedding-based, which takes accuracy, diversity, and efficiency into consideration.
