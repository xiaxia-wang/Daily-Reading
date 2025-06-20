



*2025-06-16*

#### [Prompting Disentangled Embeddings for Knowledge Graph Completion with Pre-trained Language Model](https://arxiv.org/abs/2312.01837)

*Yuxia Geng, Jiaoyan Chen, Yuhang Zeng, Zhuo Chen, Wen Zhang, Jeff Z. Pan, Yuxiang Wang, Xiaoliang Xu*

*Expert Systems With Applications 268 (2025) 126175*

This paper proposes a PLM-based KGC method which includes a hard task prompt (i.e., a pre-defined template containing the [MASK] token) and a disentangled structure prompt (i.e., a series of trainable vectors generated from entity embeddings learned by a graph learner with selective aggregations over different neighboring entities).


*2025-06-06*

#### [Croppable Knowledge Graph Embedding](https://arxiv.org/abs/2407.02779)

*Yushan Zhu, Wen Zhang, Zhiqiang Liu, Mingyang Chen, Lei Liang, Huajun Chen*

*ACL 2025*

This paper proposes a training framework to get a croppable KGE model. It utilizes a mutual learning mechanism to improve low-dimensional sub-models performance and make the high-dimensional sub-models retain the capacity that low-dimensional sub-models have.


*2025-05-05*

#### [Embedding Ontologies via Incorporating Extensional and Intensional Knowledge](https://www.sciengine.com/DI/doi/10.3724/2096-7004.di.2024.0088)

*Keyu Wang, Guilin Qi, Jiaoyan Chen, Yi Huang, Tianxing Wu*

*Data Intelligence 2025*

Ontologies can be divided into extensional knowledge that describes instances by concepts, and intensional knowledge that illustrates relations between concepts. This paper proposes an ontology embedding model that distinguishes the two types of knowledge with distinct spaces and loss design.


*2025-04-29*

#### [KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge](https://arxiv.org/abs/2405.16412)

*Pengcheng Jiang, Lang Cao, Cao Xiao, Parminder Bhatia, Jimeng Sun, Jiawei Han*

*NeurIPS 2024*

This paper proposes an approach to fine-tune knowledge graph embeddings by utilizing semantic knowledge from LLM, implemented by LLM-guided hierarchical refinement and local KG fine-tuning.


*2025-04-27*

#### [Large Language Model Enhanced Knowledge Representation Learning: A Survey](https://link.springer.com/article/10.1007/s41019-025-00285-y)

*Xin Wang, Zirui Chen, Haofen Wang, Leong Hou U, Zhao Li, Wenbin Guo*

*Data Science and Engineering*

This paper summarizes existing works on LLM-enhanced knowledge graph embeddings with their downstreaming tasks such as relation classification and question answering.


*2025-03-02*

#### [Knowledge Graph Pooling and Unpooling for Concept Abstraction](https://aclanthology.org/2025.coling-main.359)

*Juan Li, Wen Zhang, Zhiqiang Liu, Mingchen Tu, Mingyang Chen, Ningyu Zhang, Shijian Li*

*COLING 2025*

This paper proposes a KGE method by KG pooling and unpooling with contrastive learning to encode latent concepts for better embedding performance. Specifically, given an input KG, it first constructs a new graph by KG pooling and unpooling. Then the new graph is processed by relational KGE to get the representations of entities and relations, with local and global contrastive losses being applied for jointly training.


*2025-02-26*

#### [InBox: Recommendation with Knowledge Graph using Interest Box Embedding](https://arxiv.org/abs/2403.12649)

*Zezhong Xu, Yincen Qu, Wen Zhang, Lei Liang, Huajun Chen*

*VLDB 2025*

Given an interaction graph G𝑢 that captures the historical interactions between users and items, as well as an external knowledge graph G𝑘 that depicts relationships between various entities, the recommendation task is to train a model to predict the probability for a user 𝑢 will positively interact with a candidate item 𝑖, given the user’s interaction history in G𝑢 and the relational information in G𝑘. To avchieve this, it formulates items as points within the boxes representing tags. By viewing tags combination and user interest as boxes, it conducts three training steps to obtain representations for items, tags, relations, and leverage the user’s interest box to compute the matching score, which serves as the recommendation result.


*2025-01-06*

#### [Position-Aware Relational Transformer for Knowledge Graph Embedding](https://ieeexplore.ieee.org/document/10092525)

*Guangyao Li, Zequn Sun, Wei Hu, Gong Cheng, Yuzhong Qu*

*TNNLS 35(8)*

Due to the self-attention mechanism in Transformer being invariant to the order of input tokens, it is unable to distinguish a (true) relation triple from its shuffled (false) variants. To address the issue, this paper incorporates relational compositions for entity representations into the Transformer architecture, which captures the role of entities based on their position (subject or object) in the relation triple.


*2025-01-01*

#### [Tokenization, Fusion, and Augmentation: Towards Fine-grained Multi-modal Entity Representation](https://arxiv.org/abs/2404.09468)

*Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Binbin Hu, Ziqi Liu, Wen Zhang, Huajun Chen*

*AAAI 2025*

This paper proposes a framework for generating multi-modal representation for entities. Specifically, it first tokenizes raw modality data into fine-grained tokens and then fuzes them to obtain fine-grained entity representations.


*2024-12-26*

#### [Triple confidence measurement in knowledge graph with multiple heterogeneous evidences](https://link.springer.com/article/10.1007/s11280-024-01307-x)

*吴天星, 姚凯, 李伟, 漆桂林, 于益俊, 赵能文, 张人友, 段沛博*

*World Wide Web Journal 2024*

To measure the confidence (i.e., truthfulness) of each triple in the KG, this paper proposes an approach that integrates multiple features, including neighborhood node embedding, GCN-based subgraph embedding, LLM-based embedding, and path-based contrastive learning, and aggreagtes them with a confidence estimator for triples.


*2024-10-04*

#### [Centrality-Based Node Feature Augmentation for Robust Network Alignment](https://arxiv.org/abs/2304.12751)

*Jin-Duk Park, Cong Tran, Won-Yong Shin, Xin Cao*

*(Extended version of) CIKM 2022*

Network alignment (NA) is the task of discovering node correspondences across multiple networks. This paper presents an NA approach that gradually discovers a part of node pairs until all node pairs are found. Specifically, it contains 3 components as: (1) centrality-based node feature augmentation, (2) GNN-aided embedding similarity calculation alongside the augmented node features, and (3) gradual NA with similarity calculation using aligned cross-network neighbor-pairs.


*2024-09-20*

#### [BioBridge: Bridging Biomedical Foundation Models via Knowledge Graphs](https://arxiv.org/abs/2310.03320)

*Zifeng Wang, Zichen Wang, Balasubramaniam Srinivasan, Vassilis N. Ioannidis, Huzefa Rangwala, Rishita Anubhai*

*ICLR 2024*

Foundation models (FMs) for biomedical domains have largely remained unimodel. To overcome the limitation, this paper presents a parameter-efficient learning framework to bridge independently trained unimodal FMs to establish multimodal behavior. BioBridge achieves it by utilizing Knowledge Graphs (KG) to learn transformations between one unimodal FM and another without fine-tuning any underlying unimodal FMs.


*2024-09-01*

#### [Beyond Entity Alignment: Towards Complete Knowledge Graph Alignment via Entity-Relation Synergy](https://arxiv.org/abs/2407.17745)

*Xiaohan Fang, Chaozhuo Li, Yi Zhao, Qian Zang, Litian Zhang, Jiquan Peng, Xi Zhang, Jibing Gong*

*Arxiv 2024*

Existing works on KG alignment primarily emphasize the linkage of cross-graph entities but overlook aligning relations across KGs. Motivated by that, this paper decomposes KG alignment into two subtasks, entity alignment and relation alignment. Then it proposes an EM-based model to iteratively optimize both subtasks.


*2024-05-04*

#### [Editing Language Model-based Knowledge Graph Embeddings](https://arxiv.org/abs/2301.10405)

*Siyuan Cheng, Ningyu Zhang, Bozhong Tian, Xi Chen, Qingbing Liu, Huajun Chen*

*AAAI 2024*

This paper investigates the performance of LLM-based KGE model in editing tasks, i.e., to (1) identify the false entity in an existing triple then replace it with the correct one, and (2) add some new fact as a whole. It compares two paradigms of KG editing models. Finetuning KGE models are trained with the description of triples that represent relations and entities connected by the [SEP] and [CLS] tokens and then take the description sequences as the input for finetuning. Prompt-tuning models consider entities as special tokens in the language model, turning link prediction into a masked entity prediction. It also proposes datasets for evaluating the KG editing performances.


*2024-05-02*

#### [Distributed representations of entities in open-world knowledge graphs](https://www.sciencedirect.com/science/article/pii/S095070512400217X)

*Lingbing Guo, Zhuo Chen, Jiaoyan Chen, Yichi Zhang, Zequn Sun, Zhongpu Bo, Yin Fang, Xiaoze Liu, Huajun Chen, Wen Zhang*

*Knowledge-Based Systems 2023*

This paper introduces Decentralized Attention Network (DAN) as a GNN-based entity embedding model. It leverages neighboring context as the query vector to score the neighbors for each entity, thereby distributing the entity semantics only among its neighbors' embeddings. To effectively train the model, self-distillation is proposed to guide the network in generating ideal representations.


*2024-04-19*

#### [Representation Learning on Hyper-Relational and Numeric Knowledge Graphs with Transformers](https://dl.acm.org/doi/10.1145/3580305.3599490)

*Chanyoung Chung, Jaejun Lee, Joyce Jiyoung Whang*

*KDD 2023*

This paper proposes an encoder-decoder model for the link prediction task over hyper-relation knowledge graphs that also involve numeric values. Specifically, it designs context transformer layer (i.e., the encoder) to aggregate information related to each triplet, as well as a prediction transformer layer (i.e., the decoder) for making predictions from the representation. It also proposes different types of masks for different learning tasks, including link prediction of discrete entities/numeric values, and relation prediction.


*2024-04-18*

#### [Classification of Edge-dependent Labels of Nodes in Hypergraphs](https://doi.org/10.1145/3580305.3599274)

*Minyoung Choe, Sunwoo Kim, Jaemin Yoo, Kijung Shin*

*KDD 2023*

This paper studies the classification task of edge-dependent node labels in hypergraphs, e.g., the orders of authors (first, last, or others) in each publication hyperedge. To solve the problem, it proposes a transformer-based network model that incorporates edge-dependent node embeddings and positional encoding for the classification task.


*2024-04-08*

#### [A Survey on Hypergraph Representation Learning](https://dl.acm.org/doi/10.1145/3605776)

*Alessia Antelmi, Gennaro Cordasco, Mirko Polato, Vittorio Scarano, Carmine Spagnuolo, Dingqi Yang*

*ACM Computing Surveys 2023*

This paper reviews recent works on hypergraph representation learning by offering a new taxonomy of hypergraph embedding methods, including three main families of techniques, i.e., spectral, proximity-preserving, and deep neural networks.


*2024-04-05*

#### [Toward Practical Entity Alignment Method Design: Insights from New Highly Heterogeneous Knowledge Graph Datasets](https://arxiv.org/abs/2304.03468)

*Xuhui Jiang, Chengjin Xu, Yinghan Shen, Yuanzhuo Wang, Fenglong Su, Fei Sun, Zixuan Li, Zhichao Shi, Jian Guo, Huawei Shen*

*WWW 2024*

This paper focuses on the entity alignment (EA) task over heterogeneous knowledge graphs. First, it argues that existing EA benchmarks usually hold too strong assumptions that are not close to practical application, such as the two KGs having similar density, structure, and the entities are almost 1-to-1 mapped to each other KG. By proposing new EA datasets with relatively different scale and density, it shows the structural information are not well utilized by most existing GNN-based EA models. Motivated by this, it further proposes a new EA model based on entity name, local structure (by random walk) and temporal information.


*2024-01-22*

#### [LHP: Logical hypergraph link prediction](https://www.sciencedirect.com/science/article/pii/S0957417423003433?via%3Dihub)

*Yang Yang, Xue Li, Yi Guan, Haotian Wang, Chaoran Kong, Jingchi Jiang*

*Expert Systems with Applications 2023*

This paper proposes a hyperlink prediction approach for knowledge hypergraph completion, where each directed hyperlink $(h, r, t)$ consists a head $h$, and a tail $t$ as conjunctions of entities, respectively. It proposes a prediction model with an information aggregating layer and a constrained scoring layer, which also holds permutation invariance.


*2024-01-21*

#### [Knowledge Hypergraphs: Prediction Beyond Binary Relations](https://www.ijcai.org/proceedings/2020/303)

*Bahare Fatemi, Perouz Taslakian, David Vázquez, David Poole*

*IJCAI 2020*

This paper works with knowledge hypergraph where relations are defined over any number of entities. It proposes HSimplE and HypE as two embedding-based methods that work directly with knowledge hypergraphs. For both models, the prediction is a function of the relation embedding, the entity embeddings and their corresponding positions in the relation. It also introduces benchmark datasets for knowledge hyperlink prediction.


*2024-01-15*

#### [Learning Latent Relations for Temporal Knowledge Graph Reasoning](https://aclanthology.org/2023.acl-long.705/)

*Mengqi Zhang, Yuwei Xia, Qiang Liu, Shu Wu, Liang Wang*

*ACL 2023*

This paper focuses on the link prediction task over temporal knowledge graphs. In particular, it first uses a structural encoder (SE) to get representations of entities at each timestamp, then applies a relation prediction module (LRL) to mine the intra- and inter-time relations, and finally extracts the temporal representations from the output of SE and LRL for entity prediction.


*2024-01-08*

#### [APAN: Asynchronous Propagation Attention Network for Real-time Temporal Graph Embedding](https://dl.acm.org/doi/10.1145/3448016.3457564)

*Xuhong Wang, Ding Lyu, Mengjian Li, Yang Xia, Qi Yang, Xinwen Wang, Xinguang Wang, Ping Cui, Yupu Yang, Bowen Sun, Zhenyu Guo*

*SIGMOD 2021*

Due to the high time complexity of querying k-hop neighbors, most graph algorithms cannot be deployed in a giant dense temporal network to execute millisecond-level inference. Different from previous graph algorithms, this paper decouples model inference and graph computation to alleviate the damage of the heavy graph query operation to the speed of model inference.


*2023-12-30*

#### [Relation-aware Ensemble Learning for Knowledge Graph Embedding](https://aclanthology.org/2023.emnlp-main.1034/)

*Ling Yue, Yongqi Zhang, Quanming Yao, Yong Li, Xian Wu, Ziheng Zhang, Zhenxi Lin, Yefeng Zheng*

*EMNLP 2023*

This paper proposes an ensemble approach to learn the entity embeddings in a relation-aware manner. To handle the large search space of aggregating weights for each relation of different methods, it uses a divide-search-combine algorithm that searches the relation-wise ensemble weights independently.


*2023-12-20*

#### [Meta-Knowledge Transfer for Inductive Knowledge Graph Embedding](https://dl.acm.org/doi/10.1145/3477495.3531757)

*Mingyang Chen, Wen Zhang, Yushan Zhu, Hongting Zhou, Zonggang Yuan, Changliang Xu, Huajun Chen*

*SIGIR 2022*

To achieve inductive knowledge graph embedding, this paper proposes a model MorsE that learns not entity embeddings but transferable meta-knowledge to be used for producing entity embeddings. The meta-knowledge is modeled by entity-independent modules and learned by meta learning. Specifically, it designs modules that can produce entity embeddings based on the neighbor structural information. To achieve this, it proposes an entity initializer to initialize the embedding of each entity using the information of relations connected to it, and a GNN Modulator to modulate the initialized embedding for each entity based on its multi-hop neighborhood structure.


*2023-12-03*

#### [Explainable Representations for Relation Prediction in Knowledge Graphs](https://proceedings.kr.org/2023/62/)

*Rita Torres Sousa, Sara Silva, Catia Pesquita*

*KR 2023*

This paper proposes explainable representations for link prediction over knowledge graphs, which is based on identifying similar subgraphs between entities and learning representations for each subgraph. Then the embeddings of entity pairs are generated based on their mutual classes. These embbedings are then fed into a supervised ML model with a perturbation-style learning process to generate the explanations.


*2023-10-29*

#### [Deep Active Alignment of Knowledge Graph Entities and Schemata](https://dl.acm.org/doi/10.1145/3589304)

*Jiacheng Huang, Zequn Sun, Qijin Chen, Xiaozhou Xu, Weijun Ren, Wei Hu*

*SIGMOD 2023*

This paper proposes a framework for knowledge graph alignment, aiming to not only align the entities from different KGs but also align the classes and relations in the KG. It proposes a deep active framework for achieving this, which consists of 3 main modules, i.e., an embedding-based joint alignment module producing the alignment between the KGs, an inference power measurement module that estimates the similarity of entity, class and relation pairs, and a batch active learning module to interactively enhance the alignment by getting truth labels from the oracle.


*2023-07-17*

#### [Link Prediction with Attention Applied on Multiple Knowledge Graph Embedding Models](https://dl.acm.org/doi/10.1145/3543507.3583358)

*Cosimo Gregucci, Mojtaba Nayyeri, Daniel Hernández, Steffen Staab*

*WWW 2023*

This paper proposes an integrated KGE model by combining several existing models, in which attention is used to find the most suitable model for an input query. Meanwhile, the representation of queries and candidate answers are mapped onto a Poincare ball, which is regarded better for retaining the hierarchical graph structure.


*2023-07-16*

#### [Structure Pretraining and Prompt Tuning for Knowledge Graph Transfer](https://dl.acm.org/doi/10.1145/3543507.3583301)

*Wen Zhang, Yushan Zhu, Mingyang Chen, Yuxia Geng, Yufeng Huang, Yajing Xu, Wenting Song, Huajun Chen*

*WWW 2023*

This paper introduces a pretraining-finetuning model for learning knowledge graph representation. Based on traditional Transformer layers, it reforms the input triples as sequences of tokens, and extracts subgraphs to constrain the interactions between entities. The pretraining tasks include masked entity modeling, masked relation modeling, and entity pair modeling.


*2023-07-12*

#### [Heterogeneous Federated Knowledge Graph Embedding Learning and Unlearning](https://dl.acm.org/doi/10.1145/3543507.3583305)

*Xiangrong Zhu, Guangyao Li, Wei Hu*

*WWW 2023*

This paper proposes a framework for federated learning and unlearning of KG embedding. It is based on a client-server architecture, where each client contains a learning module and an unlearning module. In the learning process, it maintains local and global embeddings in parallel that mutually reinforce each other but are not identical. For the unlearning module, it conducts a retroactive interference step with hard and soft confusions followed by a passive decay step.


*2023-06-19*

#### [Joint Pre-training and Local Re-training: Transferable Representation Learning on Multi-source Knowledge Graphs](https://arxiv.org/pdf/2306.02679.pdf)

*Zequn Sun, Jiacheng Huang, Jinghao Lin, Xiaozhou Xu, Qijin Chen, Wei Hu*

*KDD 2023*

This paper proposes a joint pre-training and re-training framework for learning KG embeddings. It explores three path encoders based on RNNs, recurrent skipping networks and Transformers. A re-training process is applied to the linked subgraph with multi-level knowledge distillation.


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
