

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

