









*2025-04-22*

#### [Advancing Abductive Reasoning in Knowledge Graphs through Complex Logical Hypothesis Generation](https://arxiv.org/abs/2312.15643)

*Jiaxin Bai, Yicheng Wang, Tianshi Zheng, Yue Guo, Xin Liu, Yangqiu Song*

*AAAI 2025*

This paper introduces the task of complex logical hypothesis generation, which aims to generate complex logical hypotheses that explain a set of observations. It trains a hypothesis generation model by (1) randomly sampling logical hypotheses with diverse patterns and graph searches on the training graphs to obtain observations which are then tokenized, (2) training a conditional generation model to generate hypotheses based on tokenized observations.


*2025-03-20*

#### [SymAgent: A Neural-Symbolic Self-Learning Agent Framework for Complex Reasoning over Knowledge Graphs](https://arxiv.org/abs/2502.03283)

*Ben Liu, Jihai Zhang, Fangquan Lin, Cheng Yang, Min Peng, Wotao Yin*

*Arxiv 2025*

This paper proposes a KG-LLM collaboation framework that uses KG as dynamic environment and transforms complex questions into a multi-step interactive process. It mainly consists of 2 modules: an agent planner that leverages LLMs to guide question decomposition, and an agent executor that invokes action tools. Besides, it designs a self-learning framework to automatically synthesize reasoning trajectories and improve the agent performance.


*2024-10-03*

#### [KnowFormer: Revisiting Transformers for Knowledge Graph Reasoning](https://arxiv.org/abs/2409.12865)

*Junnan Liu, Qianren Mao, Weifeng Jiang, Jianxin Li*

*ICML 2024*

This paper revisits the application of transformers for knowledge graph reasoning to address the constraints faced by path-based methods and proposes a new model. Specifically, it utilizes a transformer architecture to perform reasoning on knowledge graphs from the message-passing perspective, rather than reasoning by textual information as PLM-based methods.


*2024-07-05*

#### [Better Together: Enhancing Generative Knowledge Graph Completion with Language Models and Neighborhood Information](https://arxiv.org/abs/2311.01326)

*Alla Chepurova, Aydar Bulatov, Yuri Kuratov, Mikhail Burtsev*

*EMNLP 2023 Findings*

This paper applies a LLM for knowledge graph completion/QA tasks by utilizing the neighboring information of each entity. Specifically, it first translate the triple-form query as (s, r, ?) into a natural language sentence like what holds the relation r with the subject s. Then by iterating over all possible entities as s, it collects the one-hop neighborhood for each entity s, organizes them as a sequence of entities and relations, and then feeds it to a LLM for predicting the result entity.


*2024-06-21*

#### [Temporal Inductive Logic Reasoning over Hypergraphs](https://arxiv.org/abs/2206.05051)

*Yuan Yang, Siheng Xiong, Ali Payani, James C Kerce, Faramarz Fekri*

*IJCAI 2024*

This paper proposes temporal inductive logic reasoning (TILR), an ILP method that reasons on temporal hypergraphs. To enable hypergraph reasoning, it introduces the multi-start random B-walk, a graph traversal method for hypergraphs. By combining it with a path-consistency algorithm, TILR learns logic rules by generalizing from both temporal and relational data.


*2024-06-20*

#### [A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs](https://papers.nips.cc/paper_files/paper/2023/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html)

*Zhaocheng Zhu, Xinyu Yuan, Michael Galkin, Louis-Pascal A. C. Xhonneux, Ming Zhang, Maxime Gazeau, Jian Tang*

*NeurIPS 2023*

For path-based reasoning, exhaustive search algorithms (e.g., Path-RNN, PathCon) enumerate all paths in exponential time, and Bellman-Ford algorithms (e.g., NeuralLP, DRUM, NBFNet, RED-GNN) compute all paths in polynomial time, but need to propagate through all nodes and edges. To mitigate the problem, this paper proposes A*Net that learns a priority function to select a subset of nodes and edges at each iteration, thus avoiding exploring all nodes and edges.


*2024-06-15*

#### [Less is More: One-shot Subgraph Reasoning on Large-scale Knowledge Graphs](https://openreview.net/forum?id=QHROe7Mfcb)

*Zhanke Zhou, Yongqi Zhang, Jiangchao Yao, quanming yao, Bo Han*

*ICLR 2024*

This paper proposes a two-stage KG link prediction approach, by firstly extracting a query-dependent subgraph from the large-scale KG with PPR as heuristics, and then conducting link prediction (with existing approaches such as DRUM) on the sampled subgraph to improve the computational efficiency.


*2024-05-30*

#### [Towards Foundation Models for Knowledge Graph Reasoning](https://openreview.net/forum?id=jVEoydFOl9)

*Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, Zhaocheng Zhu*

*ICLR 2024*

This paper proposes an approach for learning universal and transferable (knowledge) graph representations. Specifically, it builds **relational representations** as a function conditioned on their interactions. Such a conditioning strategy allows a pre-trained model to inductively generalize to any unseen KG with any relation vocabulary and to be fine-tuned on any graph.


*2024-02-13*

#### [Understanding the Reasoning Ability of Language Models From the Perspective of Reasoning Paths Aggregation](https://arxiv.org/abs/2402.03268)

*Xinyi Wang, Alfonso Amayuelas, Kexun Zhang, Liangming Pan, Wenhu Chen, William Yang Wang*

*Arxiv 2024*

To understand how pre-training with a next-token prediction objective contributes to the emergence of such reasoning capability, this paper proposes that an LM can be viewed as deriving new conclusions by aggregating indirect reasoning paths seen at pre-training time. Specifically, it formalizes the reasoning paths as random walk paths on the knowledge/reasoning graphs. Analyses of learned LM distributions suggest that a weighted sum of relevant random walk path probabilities is a reasonable way to explain how LMs reason.


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
