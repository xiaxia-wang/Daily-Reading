



*2023-06-18*

#### [What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings](https://arxiv.org/pdf/2306.02622.pdf)

*Zequn Sun, Jiacheng Huang, Xiaozhou Xu, Qijin Chen, Weijun Ren, Wei Hu*

*ICML 2023*

This paper analyzes the translation-based (e.g., TransE) and aggregation-based (e.g., GCN) entity alignment models via a unified similarity flooding perspective. It regards the learning process of both kinds of models as to find a fix-point of pairwise similarities between two KGs in the embedding space.


*2023-06-16*

#### [Shrinking Embeddings for Hyper-Relational Knowledge Graphs](https://arxiv.org/pdf/2306.02199.pdf)

*Bo Xiong, Mojtaba Nayyer, Shirui Pan, Steffen Staab*

*ACL 2023*

This paper proposes an embedding-based method for link prediction over hyper-relational knowledge graphs. A hyper-relational fact is represented by a primal triple and a set of qualifiers comprising a key-value pair that allows for expressing complicated semantics. The proposed method ShrinkE models the primal triple as a spatial-functional transformation from the head into a relation-specific box. Each qualifier “shrinks” the box to narrow down the possible answer set and, thus, realizing qualifier monotonicity.


*2023-06-02*

#### [Graph Propagation Transformer for Graph Representation Learning](https://arxiv.org/pdf/2305.11424.pdf)

*Zhe Chen, Hao Tan, Tao Wang, Tianrun Shen, Tong Lu, Qiuying Peng, Cheng Cheng, Yue Qi*

*IJCAI 2023*

This paper proposes a transformer-based architecture for graph representation learning. Specifically, it considers three types of message passing as node-to-node, node-to-edge and edge-to-node, and proposes a graph propagation attention mechanism.


*2023-05-10*

#### [Improving Knowledge Graph Entity Alignment with Graph Augmentation](https://arxiv.org/pdf/2304.14585.pdf)

*Feng Xie, Xiang Zeng, Bin Zhou, Yusong Tan*

*PAKDD 2023*

This paper proposes a model for entity alignment with a simple Entity-Relation (ER) encoder to generate latent representations for entities via jointly modeling structural information and relation semantics. Besides, it uses graph augmentation to create two graph views for margin-based alignment learning and contrastive entity representation learning, thus mitigating structural heterogeneity and improving the alignment performance.


*2023-05-02*

#### [Semantic Specialization for Knowledge-based Word Sense Disambiguation](https://arxiv.org/pdf/2304.11340.pdf)

*Sakae Mizuki, Naoaki Okazaki*

*EACL 2023*

This paper proposes a knowledge-based method for word sense disambiguation (WSD) by learning two transformations bringing the sense embedding and the context embedding close to each other. The initial sense and context embeddings are generated using a frozen BERT. For each sense, 3 sets of senses, i.e., related senses, different senses, and unrelated senses, are collected from WordNet. The 2 transformations are implemented as 2-layer feed-forward networks. The loss function to train the transformations incorporates a contrastive loss: bring related senses closer while unrelated senses further away.


*2023-04-24*

#### [Learning to Defer with Limited Expert Predictions](https://arxiv.org/pdf/2304.07306.pdf)

*Patrick Hemmer, Lukas Thede, Michael Vössing, Johannes Jakubik, Niklas Kühl*

*AAAI 2023*

Learning to defer algorithms are used by models to decide whether to generate predictions by itself or to pass them to human experts (comment: feels like a combination of human-in-the-loop and crowdsourcing). This paper proposes a new approach with 3 steps: (1) Training an embedding model with ground truth labels, which is used to extract feature representations. (2) They serve as input for the training of an expertise predictor model to approximate the human expert’s capabilities. (3) The expertise predictor model generates artificial expert predictions for the instances not labeled by the human expert.


*2023-04-19*

#### [Hyperbolic Embedding Inference for Structured Multi-Label Prediction](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d51ab0fc62fe2d777c7569952f518f56-Abstract-Conference.html)

*Bo Xiong, Michael Cochez, Mojtaba Nayyeri, Steffen Staab*

*NeurIPS 2022*

This paper investigates the problem of multi-label prediction under implication and mutual exclusion constraints. It considers the problem as an embedding inference task, and formulates a Poincaré ball model to encode different labels. The logical relationships (implication and exclusion) are geometrically encoded using insideness and disjointedness of these convex regions, where labels are separated by linear decision boundaries.


*2023-04-17*

#### [A Survey on Knowledge Graphs: Representation, Acquisition, and Applications](https://doi.org/10.1109/TNNLS.2021.3070843)

*Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, Philip S. Yu*

*TNNLS 2022*

This survey paper generally summarizes the research fields related to knowledge graphs, including knowledge representation learning, knowledge acquisition, temporal knowledge graphs and knowledge-aware applications. Each of them are further divided into 3-4 specific research directions. Their basic ideas and representative work are also introduced.


*2023-04-15*

#### [Rethinking GNN-based Entity Alignment on Heterogeneous Knowledge Graphs: New Datasets and A New Method](https://arxiv.org/pdf/2304.03468.pdf)

*Xuhui Jiang, Chengjin Xu, Yinghan Shen, Fenglong Su, Yuanzhuo Wang, Fei Sun, Zixuan Li, Huawei Shen*

*Arxiv 2023*

This paper discusses existing entity alignment methods from the perspectives of datasets and focuses of models. Firstly, it presents that existing benchmarking datasets for the EA task are not well enough (unrealistic overlapping rates, similar structure and scale, etc.) Then it compares and analyzes the important features (entity name, structure, temporal information) for EA based on experiments. It also proposes highly heterogeneous (i.e., more realistic) datasets and a preliminary simple EA model based on the important features.


*2023-03-12*

#### [Joint Knowledge Graph Completion and Question Answering](https://dl.acm.org/doi/10.1145/3534678.3539289)

*Lihui Liu, Boxin Du, Jiejun Xu, Yinglong Xia, Hanghang Tong*

*KDD 2022*

This paper proposes a model which jointly handles KGC and multi-hop KGQA by formulating them as a multi-task learning problem, using a shared embedding space and an answer scoring module. It allows the two tasks to automatically share latent features and learn the interactions between natural language question decoder and answer scoring module.


*2023-03-11*

#### [Dual-Geometric Space Embedding Model for Two-View Knowledge Graphs](https://dl.acm.org/doi/10.1145/3534678.3539350)

*Roshni G. Iyer, Yunsheng Bai, Wei Wang, Yizhou Sun*

*KDD 2022*

This paper introduces a two-view model for knowledge graphs, which divides a KG into an ontology view and an instance view. Based on that, it further proposes an embedding model by mapping the ontology part and instance part of a KG into different non-Euclidean spaces, respectively.


*2023-03-05*

#### [Subset Node Anomaly Tracking over Large Dynamic Graphs](https://dl.acm.org/doi/10.1145/3534678.3539389)

*Xingzhi Guo, Baojian Zhou, Steven Skiena*

*KDD 2022*

This paper investigates the problem of node anomaly detection using generalized personalized PageRank vectors (PPVs). It formulates the problem of node anomaly tracking as to quantify the status of each node between two snapshots of a dynamic weighted graph with an anomaly measure function. This paper also proposes a framework for node anomaly tracking, in which the PPVs are used as node representations.


*2023-02-28*

#### [Disentangled Ontology Embedding for Zero-shot Learning](https://dl.acm.org/doi/10.1145/3534678.3539453)

*Yuxia Geng, Jiaoyan Chen, Wen Zhang, Yajing Xu, Zhuo Chen, Jeff Z. Pan, Yufeng Huang, Feiyu Xiong, Huajun Chen*

*KDD 2022*

This paper proposes an ontology embedding method for unseen concepts (i.e., zero-shot) by learning the representations of different semantic aspects from the original graph. It firstly learns multiple disentangled vector representations (embeddings) for each class according to its semantics of different aspects defined in an ontology, and then incorporate these disentangled
class representations using a graph adversarial network based generative model.


*2023-02-25*

#### [FreeKD: Free-direction Knowledge Distillation for Graph Neural Networks](https://dl.acm.org/doi/10.1145/3534678.3539320)

*Kaituo Feng, Changsheng Li, Ye Yuan, Guoren Wang*

*KDD 2022*

This paper proposes a knowledge graph distillation method based on reinforcement learning. It collaboratively build two shallower GNNs to exchange knowledge between them via reinforcement learning in a hierarchical way. It incorporates node-level and structure-level actions to determine the propagation of knowledge to be conducted.


*2023-02-17*

#### [Knowledge Graph Embedding by Adaptive Limit Scoring Loss Using Dynamic Weighting Strategy](https://aclanthology.org/2022.findings-acl.91/)

*Jinfa Yang, Xianghua Ying, Yongjie Shi, Xin Tong, Ruibin Wang, Taiyan Chen, Bowei Xing*

*ACL Findings 2022*

This paper designs an adaptive loss function to emphasize the deviated triples (negative samples) by re-weighting strategy. It can be incorporated in existing KG embedding models to optimize their performance.


*2023-02-11*

#### [A Comprehensive Survey of Graph Embedding: Problems, Techniques, and Applications](https://ieeexplore.ieee.org/document/8294302/)

*Hongyun Cai, Vincent W. Zheng, Kevin Chen-Chuan Chang*

*IEEE TKDE 2018*

This survey paper introduces existing graph embedding methods. It firstly categorizes the methods by different types of inputs and outputs. For each group this paper identifies its unique challenge. Then it also proposes another taxonomy by dividing the graph embedding techniques into 5 categories, including matrix factorization, deep learning, etc. Finally it also introduces the applications of different graph embedding methods.

*2023-02-09*

#### [Knowledge Graph Embedding for Link Prediction: A Comparative Analysis](https://dl.acm.org/doi/10.1145/3424672)

*Andrea Rossi, Denilson Barbosa, Donatella Firmani, Antonio Matinata, Paolo Merialdo*

*TKDD 2021*

This survey paper analyzes 18 SOTA graph embedding methods for the link prediction task, which also includes a rule-based baseline. These methods are evaluated over several popular benchmarks.

*2023-01-12*

#### [Towards the Web of Embeddings: Integrating multiple knowledge graph embedding spaces with FedCoder](https://www.sciencedirect.com/science/article/pii/S1570826822000270?via%3Dihub)

*Matthias Baumgartner, Daniele Dell'Aglio, Heiko Paulheim, Abraham Bernstein*

*Journal of Web Semantics*

This paper proposes to integrate multiple knowledge graph embedding spaces into a universal latent space, to achieve better interopreability and comparability. It answers two main questions: (1) how do different embedding space integration models compare in the face of heterogeneous embeddings, and (2) how do different embedding space integration models perform in the presence of multiple KGs.

*2023-01-10*

#### [TransEdge: Translating Relation-Contextualized Embeddings for Knowledge Graphs](https://link.springer.com/chapter/10.1007/978-3-030-30793-6_35)

*Zequn Sun, Jiacheng Huang, Wei Hu, Muhao Chen, Lingbing Guo, Yuzhong Qu*

*ISWC 2019*

This paper proposes a knowledge graph embedding method based on the translating model inherited from TransE. It is motivated by the idea that different relations between the same pair of head-tail entities should not necessarily possess similar embedding vectors. Therefore, it implements a contextualization operator which aggregates the relation embedding and the head/tail entity embeddings to formulate the overall edge embedding.

*2023-01-09*

#### [Knowledge Association with Hyperbolic Knowledge Graph Embeddings](https://doi.org/10.18653/v1/2020.emnlp-main.460)

*Zequn Sun, Muhao Chen, Wei Hu, Chengming Wang, Jian Dai, Wei Zhang*

*EMNLP 2020*

This paper proposes a hyperbolic relational graph neural network for KG embedding and entity alignment. It is motivated by the fact that hyperbolic space grows exponentially with the radius. Unlike the Euclidean space which grows linearly, it can better fit the needs of embedding hierarchical structures in a KG.

*2023-01-08*

#### [Knowing the No-match: Entity Alignment with Dangling Cases](https://aclanthology.org/2021.acl-long.278/)

*Zequn Sun, Muhao Chen, Wei Hu*

*ACL 2021*

This paper proposes a framework to identify dangling entities from the source and target knowledge graphs to improve entity alignment. The dangling entity detection is based on nearest neighbor classification, marginal ranking and background ranking. It also proposes a multi-lingual dataset DBP2.0 for evaluation.

*2023-01-06*

#### [Informed Multi-context Entity Alignment](https://dl.acm.org/doi/10.1145/3488560.3498523)

*Kexuan Xin, Zequn Sun, Wen Hua, Wei Hu, Xiaofang Zhou*

*WSDM 2022*

This paper introduces transformer model for entity alignment, and incorporates holistic entity and relation inferences to measure the similarity.

*2023-01-05*

#### [Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs](http://proceedings.mlr.press/v97/guo19c.html)

*Lingbing Guo, Zequn Sun, Wei Hu*

*ICML 2019*

This paper proposes a new framework for training entity embeddings. It utilizes long-term entity relation chains and implements a recurrent skipping network (RSN), to better capture long-term relational dependencies between the entities.

*2023-01-04*

#### [Multi-view Knowledge Graph Embedding for Entity Alignment](https://www.ijcai.org/proceedings/2019/754)

*Qingheng Zhang, Zequn Sun, Wei Hu, Muhao Chen, Lingbing Guo, Yuzhong Qu*

*IJCAI 2019*

This paper proposes an entity alignment method which utilizes different views of the entities within the source KG. It generates embeddings for an entity based on its name, relations and attributes. Then it combines these different "views" into an overall embedding. Besides, it also adopts cross-KG relation and attribute identity inference to enhance the performance.

*2023-01-03*

#### [A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf)

*Zequn Sun, Qingheng Zhang, Wei Hu, Chengming Wang, Muhao Chen, Farahnaz Akrami, Chengkai Li*

*VLDB 2020*

This paper investigates the embedding-based methods for entity alignment over knowledge graphs. It proposes a set of benchmarking datasets which fit the entity distribution of real-world KGs, and develops OpenEA as an open-sourced library with 12 KG embedding models. It also compares the performances between these models and with traditional methods for entity alignment.

*2022-12-21*

#### [CRNet: Modeling Concurrent Events over Temporal Knowledge Graph](https://doi.org/10.1007/978-3-031-19433-7_30)

*Shichao Wang, Xiangrui Cai, Ying Zhang, Xiaojie Yuan*

*ISWC 2022*

This paper proposes a model to capture the concurrence of events in temporal knowledge graphs. It uses a concurrent evolution module with graph attention network to capture the historical event relations. Then it builds a candidate graph and predicts the missing events (i.e., entities) based on it.

*2022-12-20*

#### [HybridFC: A Hybrid Fact-Checking Approach for Knowledge Graphs](https://doi.org/10.1007/978-3-031-19433-7_27)

*Umair Qudus, Michael Röder, Muhammad Saleem, Axel-Cyrille Ngonga Ngomo*

*ISWC 2022*

This paper proposes an ensemble method which combines text-based, KG-based and path-based features for fact checking over knowledge graphs. (submitted to JoWS, reviewed by me)

*2022-12-18*

#### [Each Snapshot to Each Space: Space Adaptation for Temporal Knowledge Graph Completion](https://doi.org/10.1007/978-3-031-19433-7_15)

*Yancong Li, Xiaoming Zhang, Bo Zhang, Haiying Ren*

*ISWC 2022*

This paper proposes a space adaptation network for temporal knowledge graph completion (TKGC). It extends a convolutional neural network to map the facts with different timestamps into different latent spaces. Besides, it adapts the overlap between different spaces to control the balance between time-variability and time-stability.

*2022-12-16*

#### [Radar Station: Using KG Embeddings for Semantic Table Interpretation and Entity Disambiguation](https://doi.org/10.1007/978-3-031-19433-7_29)

*Jixiong Liu, Viet-Phi Huynh, Yoan Chabot, Raphaël Troncy*

*ISWC 2022*

This paper proposes a system named Radar Station, for entity disambiguation of relational table cells. It links the cells (i.e., entities) in the table to a target KG (e.g., Wikidata in this paper). Based on the rows and columns context, it can identify different entities with the same label using semantic similarities between them. It implements a graph embedding for the entities to represent the latent relationships between them.

*2022-12-08*

#### [Enhancing Document-Level Relation Extraction by Entity Knowledge Injection](https://doi.org/10.1007/978-3-031-19433-7_3)

*Xinyi Wang, Zitao Wang, Weijian Sun, Wei Hu*

*ISWC 2022*

This paper proposes to improve general relation extraction (RE) models by adding a knowledge injection layer between the encoding and the prediction layers. It considers coreference distillation and context exchanging in knowledge injection, and reconciles KG encoding to the representations.

*2022-11-22*

#### [OWL2Vec*: embedding of OWL ontologies](https://doi.org/10.1007/s10994-021-05997-6)

*Jiaoyan Chen, Pan Hu, Ernesto Jiménez-Ruiz, Ole Magnus Holter, Denvar Antonyrajah, Ian Horrocks*

*Machine Learning 2021*

This paper proposes an embedding method named OWL2Vec* for OWL ontologies. The overall framework is divided into two parts. The random walks over the input ontology generate a set of sentences which capture the graph structure, while the lexical information provided by e.g., *rdfs:label* and *rdfs:comment* retains the semantics of the ontology. These two documents, together with a combination of them as a third document, are fed into a pretrained language model to generate embeddings for IRIs and words.

*2022-11-18*

#### [Numerical Feature Representation with Hybrid N-ary Encoding](https://doi.org/10.1145/3511808.3557090)

*Bo Chen, Huifeng Guo, Weiwen Liu, Yue Ding, Yunzhe Li, Wei Guo, Yichao Wang, Zhicheng He, Ruiming Tang, Rui Zhang*

*CIKM 2022*

This paper proposes an encoding framework for numerical features in CTR models. It relies on the natural binary representations of numerical systems to characterize the discretization of elements, and incorporates intra-ary and inter-ary attention to avoid collision risk among neighboring representations.

*2022-11-10*

#### [Inductive Knowledge Graph Reasoning for Multi-batch Emerging Entities](https://doi.org/10.1145/3511808.3557361)

*Yuanning Cui, Yuxin Wang, Zequn Sun, Wenqiang Liu, Yiqiao Jiang, Kexin Han, Wei Hu*

*CIKM 2022*

This paper proposes a reinforcement learning model with graph convolutional network to handle knowledge graph reasoning with chronologically updated entity batches. In this model, the GCN is used to encode and update entity embeddings based on walk-based path samples. It also incorporates a link augmentation strategy to add more facts for new entities with few links.

*2022-11-08*

#### [Explainable Link Prediction in Knowledge Hypergraphs](https://doi.org/10.1145/3511808.3557316)

*Zirui Chen, Xin Wang, Chenxu Wang, Jianxin Li*

*CIKM 2022*

This paper introduces a link prediction method under the knowledge hypergraph model. It applies an EM algorithm with Markov Logic Network model to enrich and update the hypergraph embeddings.

*2022-11-03*

#### [Entity Type Prediction Leveraging Graph Walks and Entity Descriptions](https://doi.org/10.1007/978-3-031-19433-7_23)

*Russa Biswas, Jan Portisch, Heiko Paulheim, Harald Sack, Mehwish Alam*

*ISWC 2022*

This paper focuses on the problem of predicting classes for each entity in a knowledge graph. It uses three graph walk strategies, namely, node walks, edge walks, and classic node-edge walks. It feeds these different walks into a word2vec model to generate entity embeddings, which is used in downstream class prediction. The model allows to output multiple classes for an entity.

*2022-11-02*

#### [Context-Enriched Learning Models for Aligning Biomedical Vocabularies at Scale in the UMLS Metathesaurus](https://dl.acm.org/doi/10.1145/3485447.3511946)

*Vinh Nguyen, Hong Yung Yip, Goonmeet Bajaj, Thilini Wijesiriwardene, Vishesh Javangula, Srinivasan Parthasarathy, Amit P. Sheth, Olivier Bodenreider*

*TheWebConf 2022*

This paper incorporates the knowledge graph embeddings to conduct entity alignment. It concatenates the embedding vector of each UMLS term with lexical-based (LSTM) embedding vector to compute the similarity score. It also compares several kinds of knowledge graph embeddings.

*2022-10-16*

#### [TaxoEnrich: Self-Supervised Taxonomy Completion via Structure-Semantic Representations](https://dl.acm.org/doi/10.1145/3485447.3511935)

*Minhao Jiang, Xiangchen Song, Jieyu Zhang, Jiawei Han*

*TheWebConf 2022*

This paper investigates the problem of taxonomy completion, which means adding new concepts to existing hierarchical taxonomy. It aims to identify a position $\langle n_p, n_c \rangle$ where the new concept should be added. To achieve this, it proposes a contextualized embedding for each node (i.e., existing concept) in the tree based on pseudo sentences and PLM. Then it uses a LSTM-based encoder to extract the features of the paths from root to $n_p$ and $n_c$, and also extracts some relevant siblings to provide more information. It aggregates the above parent/child/siblings to perform query matching. The overall framework is evaluated on Microsoft Academic Graph (MAG) and WordNet datasets.
