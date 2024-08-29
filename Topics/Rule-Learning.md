
















*2024-08-29*

#### [Relational Graph Convolutional Networks Do Not Learn Sound Rules](https://arxiv.org/abs/2408.10261)

*Matthew Morris, David J. Tena Cucala, Bernardo Cuenca Grau, Ian Horrocks*

*KR 2024*

This paper focuses on R-GCN and proposes sound rule extraction approaches for this model. Besides, it discusses the classes of R-GCN models that no Datalog rule is sound for these models.


*2024-06-23*

#### [Learning logic programs by discovering higher-order abstractions](https://arxiv.org/abs/2308.08334)

*Céline Hocquette, Sebastijan Dumančić, Andrew Cropper*

*IJCAI 2024*

This paper studies the higher-order refactoring problem, where the goal is to refactor a logic program by discovering higher-order abstractions such as map, filter, and fold. To achieve this, it encodes the refactoring problem as a constraint optimization problem, and demonstrates the cross-domain transferability of discovered abstractions.


*2024-06-22*

#### [Learning big logical rules by joining small rules](https://arxiv.org/abs/2401.16215)

*Céline Hocquette, Andreas Niskanen, Rolf Morel, Matti Järvisalo, Andrew Cropper*

*IJCAI 2024*

This paper focuses on the inductive logic programming task that aims at learning complex rules. Specifically, it introduces an approach that joins smaller rules to make up bigger rules. The approach is implemented in a constraint-driven system with constraint solvers to efficiently join rules.


*2024-05-27*

#### [Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers](https://openreview.net/forum?id=XNa6r6ZjoB)

*Awni Altabaa, Taylor Whittington Webb, Jonathan D. Cohen, John Lafferty*

*ICLR 2024*

This paper proposes a Transformer-based approach to simulate binary relations between objects. It applies a variant of attention named relational cross-attention, which is motivated by an architectural inductive bias for relational learning that disentangles relational information from object-level features.


*2024-05-20*

#### [Learning Reliable Logical Rules with SATNet](https://papers.nips.cc/paper_files/paper/2023/hash/2ff46d83d1dcc063e075058b29d55efe-Abstract-Conference.html)

*Zhaoyu Li, Jinpei Guo, Yuhe Jiang, Xujie Si*

*NeurIPS 2023*

SATNet is a differentiable MaxSAT solver based on a low-rank semidefinite programming (SDP) approach that utilizes a learnable matrix $S$ to capture the underlying structure of logical rules and dynamically adjusts its weight from input-output examples. To interpret the weights learned by the SATNet model, this paper proposes a maximum equality approach to extract a set of propositional logical rules in weighted MaxSAT form from the learned weights of SATNet.


*2024-04-22*

#### [Fire: An Optimization Approach for Fast Interpretable Rule Extraction](https://dl.acm.org/doi/10.1145/3580305.3599353)

*Brian Liu, Rahul Mazumder*

*KDD 2023*

This paper proposes a rule extraction approach to obtain decision rules based on tree ensembles. It selects sparse representative subsets of rules from tree ensembles that are easy for a practitioner to examine. To enhance the interpretability of the extracted model, it encourages fusing rules during selection, so that many selected decision rules share common antecedents. Besides, to solve this non-convex optimization problem, it proposes a specialized solver based on block coordinate descent principles, which shows relatively high efficiency in the experiments.


*2024-04-07*

#### [Differentiable Learning of Rules with Constants in Knowledge Graphs](https://www.sciencedirect.com/science/article/pii/S0950705123004367)

*Zezhong Xu, Peng Ye, Juan Li, Huajun Chen, Wen Zhang*

*Knowledge-Based Systems 2023*

This paper extends the idea of Neural-LP framework for rule learning, and includes constants connected to each variable in the rule body. It provides rule extraction algorithm, but no faithfulness guarantee, and the performance is just close / slightly better than DRUM.


*2024-04-06*

#### [Learning Interpretable Rules for Scalable Data Representation and Classification](https://ieeexplore.ieee.org/document/10302393)

*Zhuo Wang, Wei Zhang, Ning Liu, Jianyong Wang*

*TAPMI 2023*

This paper proposes a rule learning model for classification task, including "logical layers" to simulate the conjunction and disjunction operators in the rule body. However, it provides neither the rule extraction algorithms, nor formal guarantee for the faithfulness of extracted rules.


*2024-03-29*

#### [Document-level Relationship Extraction by Bidirectional Constraints of Beta Rules](https://aclanthology.org/2023.emnlp-main.138.pdf)

*Yichun Liu, Zizhong Zhu, Xiaowang Zhang, Zhiyong Feng, Daoqi Chen, Yaxin Li*

*EMNLP 2023*

This paper proposes bidirectional constraints of beta rules (BCBR) as a logic constraint framework for rule learning. It first introduces a new rule miner which simulates rules by beta distribution, and proposes forward and reverse logic constraints based on beta rules. Then it formulates rule consistency loss with bidirectional constraints to regulate the output of document-level relation extraction models.


*2023-11-12*

#### [AMIE: association rule mining under incomplete evidence in ontological knowledge bases](https://dl.acm.org/doi/10.1145/2488388.2488425)

*Luis Antonio Galárraga, Christina Teflioudi, Katja Hose, Fabian M. Suchanek*

*WWW 2013*

This paper proposes a rule mining model named AMIE from RDF-style KBs under the OWA setting. It focuses on mining chain-like closed Horn rules, i.e., every variable in the rule appears at least twice. AMIE relies on a "language bias" to restrict the search space, i.e., every atom in the rule is transitively connected via a sharing variable to every other atom.


*2023-10-14*

#### [Do Machine Learning Models Learn Statistical Rules Inferred from Data?](https://proceedings.mlr.press/v202/naik23a.html)

*Aaditya Naik, Yinjun Wu, Mayur Naik, Eric Wong*

*ICML 2023*

This paper proposes an approach to extract statistical rules from the training data, and incorporates them in the test time to model adaptation thus improving performance. Specifically, the statistical rules are usually characterized as an interval for a random variable with lower and upper thresholds. In the test time, it first finds the violated rules, then updates the model to have fewer violations.


*2023-08-30*

#### [Answering Complex Logical Queries on Knowledge Graphs via Query Computation Tree Optimization](https://proceedings.mlr.press/v202/bai23b.html)

*Yushi Bai, Xin Lv, Juanzi Li, Lei Hou*

*ICML 2023*

This paper also handles the problem of answering logical queries with neural adjacency matrices representing different operators. For a given FOL query, it performs forward/backward propagation on the tree-like computation graph with matrices for transformation.


*2023-06-07*

#### [Logical Entity Representation in Knowledge-Graphs for Differentiable Rule Learning](https://arxiv.org/pdf/2305.12738.pdf)

*Chi Han, Qizheng He, Charles Yu, Xinya Du, Hanghang Tong, Heng Ji*

*ICLR 2023*

This paper follows the existing differentiable rule learning methods by extending neural representations (functions) for logical connectives (e.g., negation).


*2023-04-29*

#### [Anytime Bottom-Up Rule Learning for Knowledge Graph Completion](https://doi.org/10.24963/ijcai.2019/435)

*Christian Meilicke, Melisachew Wudage Chekol, Daniel Ruffinelli, Heiner Stuckenschmidt:*

*IJCAI 2019*

This paper proposes an anytime bottom-up rule learning algorithm, whose basic idea is to sample paths from a given knowledge graph. A sampling strategy is used to efficiently compute the confidences of a rule using scoring function.


*2023-04-28*

#### [Ruleformer: Context-aware Differentiable Rule Mining over Knowledge Graph](https://arxiv.org/pdf/2209.05815.pdf)

*Zezhong Xu, Peng Ye, Hui Chen, Meng Zhao, Huajun Chen, Wen Zhang*

*COLING 2022*

This paper formulates rule mining as a sequence generation problem, and proposes an encoder-decoder model for the link prediction task. Specifically, to mitigate the heavy tailed problem of rarely-seen entities in the KG, it introduces domain and range embeddings for each relation to provide the entity with some background information. It starts with extracting a subgraph of the given source entity, and applies the transformer-based encoder-decoder model to predict the next relation.


*2023-04-27*

#### [Learn to Explain Efficiently via Neural Logic Inductive Learning](https://openreview.net/pdf?id=SJlh8CEYDB)

*Yuan Yang, Le Song*

*ICLR 2020*

This paper proposes an optimized model for neural logic ILP. Notice that previous work of ILP has several limitations, including (1) the NP-hardness of ILP that involves exponentially growing number of parameters, (2) can only handle chain-like rules, and (3) the relation paths are binded to specific query. To address these problems, this paper proposes a transformer-based model that divides the search space into 3 hierarchical subspaces, and introduces functions to represent unary and binary predicates thus transforming the original queries into skolem forms. It applies multi-head attention to capture the relations in the rule body between the source and target entities.


*2023-04-01*

#### [A Minimal Deductive System for RDFS with Negative Statements](https://proceedings.kr.org/2022/35/)

*Umberto Straccia, Giovanni Casini*

*KR 2022*

This paper extends RDFS to deal with negative statements under the Open World Assumption (OWA), by extending $\rho df$, a minimal RDFS fragment, to $\rho df_\bot^\neg$. The extended logic remains syntactically a triple language, and can still be handled by general RDFS reasoners (by simply ignoring the extra parts).


*2023-03-18*

#### [The Logical Expressiveness of Graph Neural Networks](https://openreview.net/forum?id=r1lZ7AEKvB)

*Pablo Barceló, Egor V. Kostylev, Mikaël Monet, Jorge Pérez, Juan L. Reutter, Juan Pablo Silva*

*ICLR 2020*

It is a paper that studies the intersection of logical classifiers with (general) GNNs. Clearly GNNs can capture functions that are not logical classifiers, but one of the ideas of the paper is that if a a GNN captures a logical classifier, then this classifier can be expressed in the Description Logic ALCQ. Also, the converse holds: every ALCQ logical classifier can be captured by a GNN. (they also have some other results about adding readout functions to GNNs, but those are less relevant to us)


*2023-03-17*

#### [Fuzzy Logic Based Logical Query Answering on Knowledge Graphs](https://doi.org/10.1609/aaai.v36i4.20310)

*Xuelu Chen, Ziniu Hu, Yizhou Sun*

*AAAI 2022*

This paper proposes an embedding method for answering FOL queries over knowledge graphs based on fuzzy logic. It designs the embedding vector of entities from a fuzzy space, and requires the fuzzy logical operators to satisfy some rules as model properties.


*2023-03-07*

#### [LinE: Logical Query Reasoning over Hierarchical Knowledge Graphs](https://dl.acm.org/doi/10.1145/3534678.3539338)

*Zijian Huang, Meng-Fen Chiang, Wang-Chien Lee*

*KDD 2022*

This paper proposes a neural reasoner for question answering over knowledge graphs. It is motivated by the limitation of modeling entity embeddings under specific distributions (e.g., Beta distribution). Instead, it modifies the embeddings of both queries and entities in a line embedding (LinE) space, in which each dimension is expressed as a sequence of $k$ values. The embeddings are firstly initialized using Beta distribution. Then a MLP-based transformation is applied to generate the LinE embeddings. It also specifically considers the reasoning over hierarchical structures possessed by the KG.


*2023-03-07*

#### [LinE: Logical Query Reasoning over Hierarchical Knowledge Graphs](https://dl.acm.org/doi/10.1145/3534678.3539338)

*Zijian Huang, Meng-Fen Chiang, Wang-Chien Lee*

*KDD 2022*

This paper proposes a neural reasoner for question answering over knowledge graphs. It is motivated by the limitation of modeling entity embeddings under specific distributions (e.g., Beta distribution). Instead, it modifies the embeddings of both queries and entities in a line embedding (LinE) space, in which each dimension is expressed as a sequence of $k$ values. The embeddings are firstly initialized using Beta distribution. Then a MLP-based transformation is applied to generate the LinE embeddings. It also specifically considers the reasoning over hierarchical structures possessed by the KG.


*2023-03-03*

#### [Automatic Rule Generation for Time Expression Normalization](https://aclanthology.org/2021.findings-emnlp.269/)

*Wentao Ding, Jianhao Chen, Jinmao Li, Yuzhong Qu*

*EMNLP Findings 2022*

This paper proposes a method for learning rule sequences applied for normalizing time expressions. It firstly designs 10 atomic rules to transform the time expression from the source format to the target format. Its goal is to obtain the chain of rules for transforming any input time expression to the standard format. It applies DFS over the graph of available rules, under the assumption that low-redundancy chain of rules (shorter paths) is more ideal.


*2023-03-02*

#### Materialisation-Based Reasoning in DatalogMTL with Bounded Intervals

*Przemysław A. Wał˛ega, Michał Zawidzki, Dingmin Wang, Bernardo Cuenca Grau*

*AAAI 2023*

This paper proposes a fact entailment checking method for an extension of Datalog of predicates/concepts satisfiability with time intervals. The reasoning process is based on forward-chaining and captures repeating patterns between continuous time intervals to block infinite expansion. After obtaining all the available patterns the algorithm is ensured to terminate and is able to return all possible entailment results.


*2023-02-24*

#### [RLogic: Recursive Logical Rule Learning from Knowledge Graphs](https://dl.acm.org/doi/10.1145/3534678.3539421)

*Kewei Cheng, Jiahao Liu, Wei Wang, Yizhou Sun*

*KDD 2022*

This paper proposes a rule learning method over knowledge graphs for link prediction. Its motivation includes that, firstly, most of existing rule learning methods rely on the rule instances appear in the datasets, and secondly, existing methods typically regard the learned rules as independent with each other, instead of in a deductive manner. To address the first limitation, this paper proposes a probabilistic measurement based on the conditional probability to evaluate each rule. For the second limitation, it also learns the probabilistic to replace a continuous pair of relations with a single relation for recursively reducing the relation path. The rules learned in this paper are represented as Horn rules.


*2023-01-28*

#### [Inductive Logic Programming](https://link.springer.com/article/10.1007/BF03037089)

*Stephen H. Muggleton*

*New Generation Computing 1991*

This paper introduces the foundation of inductive logic programming. It begins with introducing induction. Then it reviews existing ILP systems with theoretical foundations, such as the relations to PAC-learning. It further analyzes the relationship between inverse resolution (IR) and relative least general generalization (RLCG), as well as the feasibility of extending the RLGG framework to allow for the invention of new predicates.

