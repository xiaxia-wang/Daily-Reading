






*2023-01-24*

#### [Synthesizing Datalog Programs using Numerical Relaxation](https://doi.org/10.24963/ijcai.2019/847)

*Xujie Si, Mukund Raghothaman, Kihong Heo, Mayur Naik*

*IJCAI 2019*

This paper presents a technique to synthesize Datalog programs using numerical optimization. The central idea is to formulate the problem as an instance of rule selection, and then relax classical Datalog to a refinement named DIFFLOG.


*2023-01-23*

#### [Learning Explanatory Rules from Noisy Data](https://doi.org/10.1613/jair.5714)

*Richard Evans, Edward Grefenstette*

*Journal of Artificial Intelligence Research 2018*

This paper proposes Differentiable Inductive Logic Programming ($\partial$ILP), an end-to-end ILP system that can address noisy and error data in the training process. This paper firstly introduces the basic concepts of logic programming and ILP. Then it formulates learning under ILP as a satisfiability problem, and learns continuous representations of rules through back-propagation against a likelyhood objective. In the experiments, it evaluates the proposed system over 20 standard ILP tasks from four domains: arithmetic, lists, group-theory, and family tree relations. Note that it is limited to at most binary predicates.  


*2023-01-22*

#### [Efficient Embeddings of Logical Variables for Query Answering over Incomplete Knowledge Graphs](https://ora.ox.ac.uk/objects/uuid:27c9b239-20d0-449f-b425-b5065eb128fe)

*DingminWang, Yeyuan Chen, Bernardo Cuenca Grau*

*AAAI 2023*

This paper proposes a neural model for query answering (not KBQA actually, but only a name) over KGs. Each query is formulated as a FOL expression (it covers only a part of FOL but not all) which has exactly one free variable. The model is based on a general entity/relation embedding model, and further trained to generate embeddings for intermediate variables appear in the FOL expression. It is able to handle negation symbols and is evaluated on the datasets with given query patterns.


*2023-01-21*

#### [DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs](https://proceedings.neurips.cc/paper/2019/hash/0c72cb7ee1512f800abe27823a792d03-Abstract.html)

*Ali Sadeghian, Mohammadreza Armandpour, Patrick Ding, Daisy Zhe Wang*

*NeurIPS 2019*

This paper proposes DRUM as a differentiable rule mining framework which can address inductive link prediction. It is based on bidirectional RNN (LSTM), which formulates the problem as learning first-order logical Horn clauses from a KB. Similar to Neural LP, it generates each rule with a confidence value $\alpha$, and is further optimized to limit generating incorrect rules. 


*2023-01-20*

#### [Differentiable Learning of Logical Rules for Knowledge Base Reasoning](https://proceedings.neurips.cc/paper/2017/hash/0e55666a4ad822e0e34299df3591d979-Abstract.html)

*Fan Yang, Zhilin Yang, William W. Cohen*

*NIPS 2017*

This paper proposes NeuralLP, a differentiable framework for learning inductive logical rules over knowledge bases. It represents each learnable first-order logical rule as a pair $\langle \alpha, \beta \rangle$ where $\alpha$ is a confidence score and $\beta$ is an ordered list of relations in the rule. Furthermore, the $\alpha$ performs as an attention score over different rules. In the experiments, NeuralLP is evaluated over several tasks, including statistical relation learning, grid path finding, knowledge base completion, and KBQA.


*2023-01-13*

#### [Dynamic Neuro-Symbolic Knowledge Graph Construction for Zero-shot Commonsense Question Answering](https://ojs.aaai.org/index.php/AAAI/article/view/16625)

*Antoine Bosselut, Ronan Le Bras, Yejin Choi*

*AAAI 2021*

This paper proposes a dynamic knowledge graph construction method for commonsense question answering with given contexts. Based on a pre-trained commonsense knowledge graph, it firstly generates dynamic nodes based on the given context. Then it performs inference over the generated part of the graph, to achieve zero-shot question answering (instead of retrieving over existing knowledge graph parts).


*2023-01-11*

#### [Differentiable learning of numerical rules in knowledge graphs](https://openreview.net/forum?id=rJleKgrKwS)

*Po-Wei Wang, Daria Stepanova, Csaba Domokos, J. Zico Kolter*

*ICLR 2020*

This paper proposes an extension of Neural LP, which can better capture the numerical features and rules from the knowledge graph. It efficiently expresses the comparison and classification operators, negation as
well as multi-atom symbol matching. 


*2023-01-01*

#### [From Statistical Relational to Neuro-Symbolic Artificial Intelligence](https://www.ijcai.org/proceedings/2020/688)

*Luc De Raedt, Sebastijan Dumancic, Robin Manhaeve, Giuseppe Marra*

*IJCAI 2020*

This survey paper discusses the differences between statistical relation learning and neuro-symbolic learning methods according to seven dimensions, such as directed/undirected graphical model, semantics, etc. It also identifies some open challenges for neuro-symbolic researches. 


*2022-12-30*

#### [LNN-EL: A Neuro-Symbolic Approach to Short-text Entity Linking](https://aclanthology.org/2021.acl-long.64/)

*Hang Jiang, Sairam Gurajada, Qiuhao Lu, Sumit Neelam, Lucian Popa, Prithviraj Sen, Yunyao Li, Alexander G. Gray*

*ACL 2021*

This paper introduces to use logical neural network (LNN) with rules to enhance the performance of entity linking for short texts. It implements conjunction and disjunction using LNNs to express the rules, to compute the overall similarity score between the mention and the entity in the KG. It also uses box embedding for entities to transform logical operations to geometric computations (e.g., intersection). 


*2022-12-28*

#### [Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval](https://proceedings.mlr.press/v162/alon22a.html)

*Uri Alon, Frank F. Xu, Junxian He, Sudipta Sengupta, Dan Roth, Graham Neubig*

*ICML 2022*

This paper proposes a retrieval-based language modeling method with an unsupervised automaton to enhance the efficiency. Retrieval-based model generally searches for the nearest neighbor examples in an external datastore for reference. The main idea is the retrieved neighbors at the current time step also hint at the neighbors that will be retrieved at future time steps, and can thus save repetitive searches in the prediction period. 


*2022-12-27*

#### [Neuro-Symbolic XAI: Application to Drug Repurposing for Rare Diseases](https://link.springer.com/chapter/10.1007/978-3-031-00129-1_51)

*Martin Drancé*

*DASFAA 2022*

This is a research proposal of a PhD project, discussing the possibility of using link prediction methods for drug repurposing with explanability. 


*2022-12-26*

#### [Neuro-Symbolic Visual Dialog](https://aclanthology.org/2022.coling-1.17/)

*Adnen Abdessaied, Mihai Bâce, Andreas Bulling*

*COLING 2022*

This paper proposes a visual dialog model using neuro-symbolic methods. It mainly consists of four parts. Scene understanding processes the input figure. Program generator generates programs based on the captions and encoded question (two parts, one for captions and one for the question). Program executor mainly conducts symbolic reasoning based on a dynamic knowledge base.


*2022-12-25*

#### [An Interpretable Neuro-Symbolic Reasoning Framework for Task-Oriented Dialogue Generation](https://aclanthology.org/2022.acl-long.338/)

*Shiquan Yang, Rui Zhang, Sarah M. Erfani, Jey Han Lau*

*ACL 2022*

This paper introduces an interpretable reasoning method based on a general generation model for tast-oriented dialogue generation. It implements a hypothesis generator to create candidate triples based on an external knowledge base. Then each hypothesis is executed on a hierarchical reasoner with a belief score. The final answer is generated based on BERT model with attention over entities in the KB.


*2022-12-24*

#### [Neuro-Symbolic Inductive Logic Programming with Logical Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/20795)

*Prithviraj Sen, Breno W. S. R. de Carvalho, Ryan Riegel, Alexander G. Gray*

*AAAI 2022*

This paper works on the task of inductive logic programming. It uses logical neural network to learn rules based on a tree-structured program template, and combines it with base facts.


*2022-12-23*

#### [Weakly Supervised Neuro-Symbolic Module Networks for Numerical Reasoning over Text](https://ojs.aaai.org/index.php/AAAI/article/view/21374)

*Amrita Saha, Shafiq R. Joty, Steven C. H. Hoi*

*AAAI 2022*

This paper proposes a weakly supervised model for numerical reasoning over text. It trains both neural and discrete reasoning modules end-to-end in a Deep RL framework with only discrete reward based on exact answer match. 


*2022-12-05*

#### [Modular materialisation of Datalog programs](https://doi.org/10.1016/j.artint.2022.103726)

*Pan Hu, Boris Motik, Ian Horrocks*

*Artificial Intelligence 2022*

This paper proposes an optimized framework for the computation and maintenance of Datalog materialisations. By pruning duplicate computation process it achieves more efficient performance over Datalog reasoning systems without loss of completeness. 


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

