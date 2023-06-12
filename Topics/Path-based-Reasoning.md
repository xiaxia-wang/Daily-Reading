




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
