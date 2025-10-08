








*2025-10-08*

#### [CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering](https://arxiv.org/abs/2509.21035)

*Yang Zhao, Chengxiao Dai, Wei Zhuo, Yue Xiu, Dusit Niyato*

*Arxiv 2025*

This paper introduces an agentic framework that treats context construction as a sequential decision process over knowledge graphs, deciding what to expand, which paths to follow or backtrack, what evidence to keep, and when to stop. Latency (interaction steps) and prompt cost (selected tokens) are exposed as user-specified budgets or prices, allowing per-query adaptation. CLAUSE employs a Lagrangian-Constrained Multi-Agent Proximal Policy Optimization algorithm to coordinate three agents: Subgraph Architect, Path Navigator, and Context Curator, so that subgraph construction, reasoning-path discovery, and evidence selection are jointly optimized under per-query resource budgets on edge edits, interaction steps, and selected tokens.


*2025-04-13*

#### [Learning Horn envelopes via queries from language models](https://www.sciencedirect.com/science/article/pii/S0888613X23001573)

*Sophie Blum, Raoul Koudijs, Ana Ozaki, Samia Touileb*

*International Journal of Approximate Reasoning*

This paper explores an approach for probing a trained neural network to extract a symbolic abstraction as a Boolean formula, where a learner attempts to extract information from an oracle by posing membership and equivalence queries. The goal is to learn the smallest representation of all Horn clauses implied by a Boolean formula.


*2025-04-02*

#### [Neuro-Symbolic AI and the Semantic Web](https://www.semantic-web-journal.net/system/files/swj3711.pdf)

*Pascal Hitzler, Monireh Ebrahimi, Md. Kamruzzaman Sarker, Daria Stepanova*

*SWJ 2024*

A special issue of Semantic Web Journal with 6 papers focusing on neuro-symbolic AI in the context of Semantic Web.


*2025-03-14*

#### [Efficient Rectification of Neuro-Symbolic Reasoning Inconsistencies by Abductive Reflection](https://arxiv.org/abs/2412.08457)

*Wen-Chao Hu, Wang-Zhou Dai, Yuan Jiang, Zhi-Hua Zhou*

*AAAI 2025 Best Paper*

This paper proposes an extension to the existing abductive learning framework, which replaces the cumbersome consistency optimization module with a more efficient reflection approach. In particular, the reflection mechnism does not rely on external KB, while being part of the learning framework, thereby easing the complex computation and scaling issue to larger datasets.


*2025-01-30*

#### [神经符号系统：非确定性管理的视角](https://doi.org/10.1360/SSI-2024-0163)

*李泽南, 姚远, 马晓星, 吕建*

*中国科学：信息科学*

非确定性 (Uncertainty) 是理解高效融合神经符号之内在技术挑战的一个关键。神经网络和符号规则两类软件实体对非确定性的处理方式截然不同。前者基于学习泛化理论，把非确定性隐式地包含于期望损失最小化的学习目标之中，而且主流的神经网络学习技术常常牺牲对于非确定性的精确管理，并以此来换取更好的预测准确性。后者的构建过程中，人类开发者负责抽象掉现实世界问题的非确定性，以换取符号规则推理的可计算性、可解释性和可组合性。本文以非确定性管理的视角，来分析“端到端”神经符号学习的关键步骤和技术挑战。首先将神经符号学习建模成一个双层优化问题，用以讨论“端到端”学习的优越性和必要性，以及当前实现“端到端”学习的两个主要技术瓶颈：“神经符号接地”以及“可微符号合成”。


*2024-07-09*

#### [Are Logistic Models Really Interpretable?](https://arxiv.org/abs/2406.13427)

*Danial Dervovic, Freddy Lécué, Nicolás Marchesotti, Daniele Magazzeni*

*IJCAI 2024*

This paper first shows via a user study that skilled participants are unable to reliably reproduce the action of small LR models given the trained parameters. Then it proposes a new Linearised Additive model, and argues that LAMs are more interpretable than logistic models for users.


*2023-12-24*

#### [LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers](https://arxiv.org/pdf/2310.15164.pdf)

*Theo X. Olausson, Alex Gu, Ben Lipkin, Cedegao E. Zhang, Armando Solar-Lezama, Joshua B. Tenenbaum, Roger P. Levy*

*EMNLP 2023 Outstanding*

This paper handles the logical reasoning task by using LLMs as semantic parser to translate premises and conclusions from natural language to expressions in first-order logic, and uses an external symbolic solver to perform deductive inference.


*2023-10-11*

#### [Neuro-Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal](https://proceedings.mlr.press/v202/marconato23a.html)

*Emanuele Marconato, Gianpaolo Bontempo, Elisa Ficarra, Simone Calderara, Andrea Passerini, Stefano Teso*

*ICML 2023*

This paper investigates the task of neuro-symbolic continual learning, which handles the issue of catastrophic forgetting by leveraging prior knowledge, and overcomes the problem of reasoning shortcuts. Reasoning shortcuts appears when the model predicts correct labels while knowledge is not enough to support the prediction, usually related to how the knowledge and training data are structured. This paper also proposes a model to effectively learn high-quality knowledge based on a small number of densely annotated examples and a concept rehearsal strategy.


*2023-08-15*

#### [Binding Language Models in Symbolic Languages](https://openreview.net/pdf?id=lH1PV42cbF)

*Zhoujun Cheng, Tianbao Xie, Peng Shi, Chengzu Li, Rahul Nadkarni, Yushi Hu, Caiming Xiong, Dragomir Radev, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, Tao Yu*

*ICLR 2023*

This paper builds a pipeline for solving tasks described in natural language by mapping it to an executable program and obtaining the answer by executing it (as in a fixed, interpretable way, instead of directly getting the answer with the LM as a black box). It involves two stages, parsing and execution. In the parsing stage, a LM maps the input to a customized program. In the execution stage, another LM serves to realize the API call (e.g., just like pretending the SQL engine) and return values. A deterministic program interpreter is used to generate the final answer.


*2023-08-08*

#### [Learning where and when to reason in neuro-symbolic inference](https://openreview.net/pdf?id=en9V5F8PR-)

*Cristina Cornelio, Jan Stuehmer, Shell Xu Hu, Timothy M. Hospedales*

*ICLR 2023*

This paper proposes a pipeline for solving visual-sudoku problems by combining both neural and symbolic reasoners. Specifically, an input instance is firstly processed by a neuro-solver that outputs an approximate solution. The solution is then analyzed by a mask-predictor to identify the components that do not satisfy domain-knowledge constraints. The masking output is combined with the solution predicted by the neuro-solver, and fed to a symbolic solver to correct the errors identified by the mask-predictor.


*2023-04-30*

#### [NeuPSL: Neural Probabilistic Soft Logic](https://arxiv.org/abs/2205.14268)

*Connor Pryor, Charles Dickens, Eriq Augustine, Alon Albalak, William Wang, Lise Getoor*

*IJCAI 2023*

This paper proposes an architecture combining both neural and symbolic inference. It applies an energy function to capture the symbolic potentials where dependencies between relations and attributes of atoms are encoded with weighted first-order logical clauses and linear arithmetic inequalities as rules. (To be honest I do not understand this part very well...but it seems indeed broad and effective.) The experiments were conducted over MNIST-Add1 and MNIST-Add2.


*2023-04-08*

#### [Learning Typed Rules over Knowledge Graphs](https://proceedings.kr.org/2022/51/)

*Hong Wu, Zhe Wang, Kewen Wang, Yi-Dong Shen*

*KR 2022*

This paper proposes a rule learning method for Horn clauses with class (type) information. It firstly samples related paths starting from some head predicates $p$ in a BFS manner. Then it applies path-based and embedding-based methods to score the paths, supported by the type information and sub/super-class relations. Finally, the generated rules are evaluated using statistical metrics (standard confidence and head coverage).


*2023-04-07*

#### [Learning Generalized Policies without Supervision Using GNNs](https://proceedings.kr.org/2022/49/)

*Simon Ståhlberg, Blai Bonet, Hector Geffner*

*KR 2022*

This paper proposes a GNN model for the classical planning problem, which is related to logic programs and finite-state controllers. It represents the features of states in the planning by node embeddings, and modifies the way of message passing accordingly.


*2023-04-05*

#### [Embed2Sym - Scalable Neuro-Symbolic Reasoning via Clustered Embeddings](https://proceedings.kr.org/2022/44/)

*Yaniv Aspis, Krysia Broda, Jorge Lobo, Alessandra Russo*

*KR 2022*

This paper proposes a two-stage model as perception and reasoning for symbolic reasoning tasks. As an example, it deals with the problem of identifying numbers from figures and computing their additions. It firstly gets the embeddings of each figure using a neural network, and conducts k-NN based clustering of embedding vectors. Then, each cluster is labeled with a distinct concept (i.e., the number). Finally, the symbolic reasoning part makes predictions based on the given rules.


*2023-04-03*

#### [Looking Inside the Black-Box: Logic-based Explanations for Neural Networks](https://proceedings.kr.org/2022/45/)

*João Ferreira, Manuel de Sousa Ribeiro, Ricardo Gonçalves, João Leite*

*KR 2022*

This paper proposes to extract logical concepts from a trained (and fixed) neural network model. Specifically, based on the fixed neural network model, it selects different tensors from each level, and trains a mapping network to produces connections between neuron activations and human-defined concepts.


*2023-04-02*

#### [Open Relation Extraction with Non-existent and Multi-span Relationships](https://proceedings.kr.org/2022/37/)

*Huifan Yang, Da-Wei Li, Zekun Li, Donglin Yang, Bin Wu*

*KR 2022*

Open relation extraction (ORE) aims to assign semantic relationships among arguments. This paper proposes a query-based open relation extractor to extract single/multi-span relations, and detect non-existent relationships. The model is based on BERT, and implements single-span extraction head for single-span relations, as well as a query-based sequence labeling head for multi-span and non existent relations.


*2023-03-14*

#### [Discovering Invariant and Changing Mechanisms from Data](https://dl.acm.org/doi/10.1145/3534678.3539479)

*Sarah Mameche, David Kaltenpoth, Jilles Vreeken*

*KDD 2022*

This paper investigates the algorithmic model for causation to mechanism changes and instantiating it using Minimum Description Length (MDL). For a continuous variable 𝑌 in multiple contexts C, it identifies variables 𝑋 as causal if the regression functions 𝑔 : 𝑋 →𝑌 have succinct descriptions in all contexts.


*2023-01-27*

#### [Techniques for Symbol Grounding with SATNet](https://proceedings.neurips.cc/paper/2021/hash/ad7ed5d47b9baceb12045a929e7e2f66-Abstract.html)

*Sever Topan, David Rolnick, Xujie Si*

*NeurlPS 2021*

SATNet is a differentiable MAXSAT solver based on a low-rank semidefinite relaxation approach. This paper further introduces a self-supervised pre-training method based on SATNet for the symbol grounding problem. It trains a visual classifier without direct supervision data (i.e., "ungrounded") with a symbol grounding loss. It also introduces a proofreader which can be applied to improve the performance of any SATNet system. 


*2023-01-26*

#### [End-to-end Differentiable Proving](https://proceedings.neurips.cc/paper/2017/hash/b2ab001909a8a6f04b51920306046ce5-Abstract.html)

*Tim Rocktäschel, Sebastian Riedel*

*NIPS 2017*

This paper proposes a neural network model for differentiable proving of queries over knowledge bases. It is motivated by the backward chaining algorithm used in Prolog. It replaces symbolic unification with a differentiable computation on vector representations of symbols using a radial basis function kernel, thus learning the proximity of symbols. It is presented to have better performance than ComplEX. 


*2023-01-25*

#### [Syntax guided synthesis of Datalog programs](https://dl.acm.org/doi/10.1145/3236024.3236034)

*Xujie Si, Woosuk Lee, Richard Zhang, Aws Albarghouthi, Paraschos Koutris, Mayur Naik*

*ESEC/SIGSOFT FSE 2018*

This paper proposes a system named ALPS for ILP problem. It is based on Datalog rules and formulates the problem as synthesizing Datalog rules from the set of facts where all positive/negative atoms are satisfied. It incorporates three main techniques, including (1) meta-rule-guided synthesis which extracts similar meta-rules with different predicates' names from the hypothesis space, (2) query-by-committee which is applied to pick up the query that can prune the most space to converge to the desired program, and (3) bidirectional search aiming to maintain the most concise programs with available examples. 


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

