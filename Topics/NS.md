









*2022-11-25*

#### [GNNQ: A Neuro-Symbolic Approach to Query Answering over Incomplete Knowledge Graphs](https://doi.org/10.1007/978-3-031-19433-7_28)

*Maximilian Pflueger, David J. Tena Cucala, Egor V. Kostylev*

*ISWC 2022*

To improve the performance of question answering over incomplete knowledge graphs, this paper proposes a method based on symbolic rules and relational graph convolutional network. For the input incomplete KG, it firstly augments the part matching the query fragments based on symbolic rules (i.e., Datalog rules as reported in the experiment). Then it applies an RGCN model to predict the answers. Furthermore, it proves that the proposed model is able to use fewer layers to work with the help of KG augmentation.


*2022-11-24*

#### [Faithful Approaches to Rule Learning](https://proceedings.kr.org/2022/50/)

*David J. Tena Cucala, Bernardo Cuenca Grau, Boris Motik*

*KR 2022*

This paper analyzes the rule learning approach Neural-LP and proposes to improve its soundness and completeness. It firstly introduces the concept of rule learning, and describes Neural-LP with its rule extraction process. It proves that existing Neural-LP can be unsound and incomplete under some settings. To address that, it provides a new kind of max-Neural-LP models which can ensure the faithfulness of prediction by replacing a sum calculation with max product. 


*2022-11-23*

#### [Explainable GNN-Based Models over Knowledge Graphs](https://openreview.net/forum?id=CrCvGNHAIrz)

*David Jaime Tena Cucala, Bernardo Cuenca Grau, Egor V. Kostylev, Boris Motik*

*ICLR 2022*

This paper proposes a transformation of knowledge graphs to GNN, in which the predictions can be interpreted symbolically as Datalog rules. It firstly introduces the transformation of a KG to a GNN, in which each entity and relation is represented by a vertex, and coloured edges represent different roles. Then it proves that, such a monotonic graph neural network (MGNN) can equally derive a set of facts as a set of Datalog rules, and an operator on the MGNN can also be mapped to a symbolic rule or program. 


*2022-11-05*

#### [Neuro-Symbolic Interpretable Collaborative Filtering for Attribute-based Recommendation](https://doi.org/10.1145/3485447.3512042)

*Wei Zhang, Junbing Yan, Zhuo Wang, Jianyong Wang*

*TheWebConf 2022*

This paper proposes a neural-symbolic approach for attribute-based recommendation. The goal of the task is to predict the user-item interaction based on their attribute-value pairs. It proposes a three-tower shaped model, in which the three towers represent the user, item, and the concatenation of user and item. It incorporates logical layers in each tower with conjunction and disjunction nodes. 


*2022-11-04*

#### [Explainable Neural Rule Learning](https://dl.acm.org/doi/10.1145/3485447.3512023)

*Shaoyun Shi, Yuexiang Xie, Zhen Wang, Bolin Ding, Yaliang Li, Min Zhang*

*TheWebConf 2022*

This paper proposes an explainable neural rule learning method for binary predictions. It constructs a set of explainable condition modules (ECMs) as units of the neural network, and organizes them into a forest (multiple trees). From each root to the leaves, a path is greedily identified. Then the final prediction is given by a voting layer with different weights of the trees. 


*2022-09-29*

#### [AdaLoGN: Adaptive Logic Graph Network for Reasoning-Based Machine Reading Comprehension](https://doi.org/10.18653/v1/2022.acl-long.494)

*Xiao Li, Gong Cheng, Ziheng Chen, Yawei Sun, Yuzhong Qu*

*ACL 2022*

This paper proposes a neural-symbolic approach for the task of machine reading comprehension. For a given document, it firstly applies symbolic reasoning to extend the existing text logic graph. Then it adopts neural reasoning with a subgraph-to-node message passing mechanism to predict the answer for multiple-choice questions.

