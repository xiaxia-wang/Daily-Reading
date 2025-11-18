







*2025-11-17*

#### [Scalable Feature Learning on Huge Knowledge Graphs for Downstream Machine Learning](https://openreview.net/pdf/f22b26fa13d0b0488d2674d5991216d381b43261.pdf)

*Félix Lefebvre, Gaël Varoquaux*

*NeurIPS 2025*

This paper introduces a KG embedding approach, by optimizing embeddings only on a small core of entities, selected based on node degrees, and then propagating them to the rest of the graph with message passing.


*2025-08-26*

#### [SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation](https://arxiv.org/abs/2502.05424)

*Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, Hui Zhang*

*WWW 2025*

This paper introduces a graph foundation model for text-free graphs, that uses a set of structure tokens to harmonize structure-based aggregation across source domains during the pre-training phase.


*2025-08-25*

#### [All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining](https://arxiv.org/abs/2402.09834)

*Haihong Zhao, Aochuan Chen, Xiangguo Sun, Hong Cheng, Jia Li*

*KDD 2024*

This paper introduces a pretrain-finetune pipeline of graph foundational model. It combines multiple graphs using 'coordinator nodes' with learnable parameters in the pretraining phase, and samples subgraphs by random walk for training a 'foundational' GNN with contrastive loss.


*2025-07-30*

#### [Fine-tuning Graph Neural Networks by Preserving Graph Generative Patterns](https://arxiv.org/abs/2312.13583)

*Yifei Sun, Qi Zhu, Yang Yang, Chunping Wang, Tianyu Fan, Jiajun Zhu, Lei Chen*

*AAAI 2024*

Given a downstream graph G, the core idea is to tune the pre-trained GNN so that it can reconstruct the generative patterns of G, i.e., the graphon W.


*2025-07-27*

#### [Benchmarking Graph Foundation Models](http://shichuan.org/doc/199.pdf)

*Jinyu Yang et al.*

*KDD 2025*

This work introduces an open-source pipeline that standardizes the training, evaluation, and deployment of graph foundation models across diverse real-world graph applications.


*2025-06-20*

#### [Can Classic GNNs Be Strong Baselines for Graph-level Tasks? Simple Architectures Meet Excellence](https://arxiv.org/abs/2502.09263)

*罗元凯，时磊，吴晓明*

*ICML 2025*

This work explores the potential of classical GNNs for handling graph-level task with the help of commonly-used techniques, including edge feature integration, normalization, dropout, residual connections, feed-forward networks, and positional encoding.


*2025-05-23*

#### [Harnessing Language Model for Cross-Heterogeneity Graph Knowledge Transfer](https://ojs.aaai.org/index.php/AAAI/article/view/33421)

*Jinyu Yang, Ruijia Wang, Cheng Yang, Bo Yan, Qimin Zhou, Yang Juan, Chuan Shi*

*AAAI 2025*

This paper proposes a Language Model-enhanced Cross-Heterogeneity learning model for the node classification task. Specifically, it first uses a metapath-based corpus construction method to unify HG representations as languages. The corpus of source HGs is used to fine-tune a pretrained LM, enabling the LM to autonomously extract general knowledge across different HGs. Then it applies an iterative training pipeline with a GNN predictor, enhanced by LM-GNN contrastive alignment at each iteration.


*2025-03-21*

#### [Towards Neural Scaling Laws on Graphs](https://arxiv.org/abs/2402.02054)

*Jingzhe Liu, Haitao Mao, Zhikai Chen, Tong Zhao, Neil Shah, Jiliang Tang*

*Arxiv 2024*

This paper investigates the scaling law for neural network models over graphs. For model scaling, it identifies that despite the number of parameters, the model depth also plays an important role in the model scaling behavior, which differs from observations in other domains. For data scaling, it describes the data scaling law with number of nodes or edges as metrics.


*2025-03-13*

#### [SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation](https://arxiv.org/abs/2502.05424)

*Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, Hui Zhang*

*WWW 2025*

This paper proposes a structure alignment framework for text-free multi-domain graph pre-training and cross-domain adaptation. It is designed to learn multi-domain knowledge from graphs originating in multiple source domains, which is then adapted to unseen target domain. Specifically, it introduces a set of structure tokens to harmonize structure-based aggregation across source domains during the pre-training phase. Next, for cross-domain adaptation, holistic prompts and specific prompts are used to adapt unified multi-domain structural knowledge and fine-grained, domain-specific information, respectively.


*2025-01-16*

#### [HOGDA: Boosting Semi-supervised Graph Domain Adaptation via High-Order Structure-Guided Adaptive Feature Alignment](https://dl.acm.org/doi/10.1145/3664647.3680765)

*Jun Dan, Weiming Liu, Mushui Liu, Chunfeng Xie, Shunjie Dong, Guofang Ma, Yanchao Tan, Jiazheng Xing*

*MM 2024*

Semi-supervised graph domain adaptation, as a subfield of graph transfer learning, seeks to precisely annotate unlabeled target graph nodes by leveraging transferable features acquired from the limited labeled source nodes. This paper introduces a high-order structure information mixing module to capture graph structure information, and applies adaptive weighted domain alignment to dynamically adjust the node weight during adversarial domain adaptation. Besides, to mitigate overfitting caused by limited labeled nodes, it designs a trust-aware node clustering strategy to guide the unlabeled nodes for discriminative clustering.


*2024-12-23*

#### [Bi-Level Attention Graph Neural Networks](https://arxiv.org/abs/2304.11533)

*Roshni G. Iyer, Wei Wang, Yizhou Sun*

*Arxiv 2023*

To effectively model both multi-relational and multi-entity large-scale heterogeneous graphs, this paper presents a bi-level graph neural network model, that includes both node-node and relation-relation attention mechanisms by hierarchically attending to both types of information from local neighborhood contexts instead of the global graph context.


*2024-12-20*

#### [Heterogeneous Information Networks: the Past, the Present, and the Future](https://www.vldb.org/pvldb/vol15/p3807-sun.pdf)

*Yizhou Sun, Jiawei Han, Xifeng Yan, Philips S. Yu, Tianyi Wu*

*VLDB Vol 15*

This perspective paper introduces the heterogeneous information networks, where both nodes and links have different types. Based on the notion of meta-paths, it introduces the enbeddings and representation learning for heterogeneous networks, and mentions several connections with KGs and GNNs.


*2024-12-16*

#### [Data-centric Graph Learning: A Survey](https://arxiv.org/abs/2310.04987)

*Yuxin Guo, Deyu Bo, Cheng Yang, Zhiyuan Lu, Zhongjian Zhang, Jixi Liu, Yufei Peng, Chuan Shi*

*IEEE Transactions on Big Data*

This paper summarizes existing works on graph learning, categorizing them by different stages in the learning pipeline, including data preparation, pre-processing, training and inference. It also highlights several potential future directions such as standardized graph data processing, continuous learning, and graph-model co-development.


*2024-11-20*

#### [Understanding over-squashing and bottlenecks on graphs via curvature](https://arxiv.org/abs/2111.14522)

*Jake Topping, Francesco Di Giovanni, Benjamin Paul Chamberlain, Xiaowen Dong, Michael M. Bronstein*

*ICLR 2022*

This paper introduces and analyzes the issue of over-squashing in message-passing neural networks, consisting in the distortion of messages being propagated from distant nodes.


*2024-11-17*

#### [SE-GSL: A General and Effective Graph Structure Learning Framework through Structural Entropy Optimization](https://arxiv.org/abs/2303.09778)

*Dongcheng Zou, Hao Peng, Xiang Huang, Renyu Yang, Jianxin Li, Jia Wu, Chunyang Liu, Philip S. Yu*

*WWW 2023*

This paper proposes a general graph strcture learning framework, through structural entropy and the graph hierarchy abstracted in the encoding tree. Particularly, it uses the one-dimensional structural entropy to maximize embedded information content while auxiliary neighbourhood attributes are fused to enhance the original graph. It constructs optimal encoding trees to minimize the uncertainty and noises in the graph whilst assuring proper community partition in hierarchical abstraction. A sample-based mechanism was proposed ssssfor restoring the graph structure via node structural entropy distribution.


*2024-11-17*

#### [SE-GSL: A General and Effective Graph Structure Learning Framework through Structural Entropy Optimization](https://arxiv.org/abs/2303.09778)

*Dongcheng Zou, Hao Peng, Xiang Huang, Renyu Yang, Jianxin Li, Jia Wu, Chunyang Liu, Philip S. Yu*

*WWW 2023*

This paper proposes a general graph strcture learning framework, through structural entropy and the graph hierarchy abstracted in the encoding tree. Particularly, it uses the one-dimensional structural entropy to maximize embedded information content while auxiliary neighbourhood attributes are fused to enhance the original graph. It constructs optimal encoding trees to minimize the uncertainty and noises in the graph whilst assuring proper community partition in hierarchical abstraction. A sample-based mechanism was proposed ssssfor restoring the graph structure via node structural entropy distribution.


*2024-11-08*

#### [Redundancy-Free Computation for Graph Neural Networks](https://dl.acm.org/doi/10.1145/3394486.3403142)

*Zhihao Jia, Sina Lin, Rex Ying, Jiaxuan You, Jure Leskovec, Alex Aiken*

*KDD 2020*

This paper proposes an adjusted GNN framework that explicitly avoids redundancy by managing intermediate aggregation results hierarchically to eliminate repeated computations and unnecessary data transfers in GNN training and inference. HAGs perform the same computations and give the same models/accuracy as traditional GNNs, but in a much shorter time due to optimized computations. To identify redundant computations, an accurate cost function is introduced with a search algorithm to find optimized HAGs.


*2024-11-07*

#### [Minimal Variance Sampling with Provable Guarantees for Fast Training of Graph Neural Networks](https://dl.acm.org/doi/10.1145/3394486.3403192)

*Weilin Cong, Rana Forsati, Mahmut Kandemir, Mehrdad Mahdavi*

*KDD 2020*

Existing sampling methods are mostly based on the graph structural information and ignore the dynamicity of optimization, which leads to the issue of high variance in estimating the stochastic gradients. To address that, this paper shows that the variance of any sampling method can be decomposed into embedding approximation variance in the forward stage and stochastic gradient variance in the backward stage. Then it proposes a decoupled variance reduction strategy that employs (approximate) gradient information to adaptively sample nodes with minimal variance, and explicitly reduces the variance introduced by embedding approximation.


*2024-11-06*

#### [Subgraph neural networks](https://dl.acm.org/doi/10.5555/3495724.3496396)

*Emily Alsentzer, Samuel G. Finlayson, Michelle M. Li, Marinka Zitnik*

*NeurIPS 2020*

This paper proposes a subgraph neural network to learn disentangled subgraph representations. Specifically, it introduces a subgraph routing mechanism that propagates neural messages between the subgraph’s connected components and (randomly sampled) anchor patches from the underlying graph. It specifies three channels, namely, position, neighborhood, and structure, each designed to capture a distinct aspect of subgraph topology.


*2024-11-05*

#### [Open graph benchmark: datasets for machine learning on graphs](https://dl.acm.org/doi/10.5555/3495724.3497579)

*Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, Jure Leskovec*

*NeurIPS 2020*

This paper introduces the OPEN GRAPH BENCHMARK, a diverse set of challenging and realistic benchmark datasets to facilitate scalable, robust, and reproducible graph machine learning (ML) research.


*2024-11-04*

#### [Mixup for Node and Graph Classification](https://dl.acm.org/doi/10.1145/3442381.3449796)

*Yiwei Wang, Wei Wang, Yuxuan Liang, Yujun Cai, Bryan Hooi*

*WWW 2021*

This paper proposes mixup approaches for augmenting graph data for node and graph classification. To interpolate the irregular graph topology, it proposes a two-branch graph convolution to mix the receptive field subgraphs for the paired nodes. mixup on different node pairs can interfere with the mixed features for each other due to the connectivity between nodes. To block this interference, a two-stage mixup framework was proposed by using each node’s neighbors’ representations before mixup for graph convolutions. For graph classification, it interpolates complex and diverse graphs in the semantic space.


*2024-11-03*

#### [Pathfinder Discovery Networks for Neural Message Passing](https://dl.acm.org/doi/10.1145/3442381.3449882)

*Benedek Rozemberczki, Peter Englert, Amol Kapoor, Martin Blais, Bryan Perozzi*

*WWW 2021*

This paper proposes the pathfinder layer, a differentiable neural network layer which is able to combine multiple sources of proximity information defined over a set of nodes to form a single weighted graph. The pathfinder layer uses a feed forward neural network to learn the edge weights while the sparsity of the underlying weighted multiplex graph is unchanged. This layer feeds directly into a downstream GNN model that is set up to learn arbitrary tasks – in this paper, semi-supervised node classification.


*2024-11-02*

#### [Towards Self-Explainable Graph Neural Network](https://dl.acm.org/doi/10.1145/3459637.3482306)

*Enyan Dai, Suhang Wang*

*CIKM 2021*

This paper proposes a GNN framework to find K-nearest labeled nodes for any unlabeled node, as an explanation for node classification, where nearest labeled nodes are found by ``interpretable'' similarity module in terms of both node similarity and local structure similarity.


*2024-11-01*

#### [Node2Grids: A Cost-Efficient Uncoupled Training Framework for Large-Scale Graph Learning](https://dl.acm.org/doi/10.1145/3459637.3482456)

*Dalong Yang, Chuan Chen, Youhao Zheng, Zibin Zheng, Shih-wei Liao*

*CIKM 2021*

This paper proposes a training framework that leverages the independent mapped data for obtaining the embedding. Instead of directly processing the coupled nodes, it maps the coupled graph data into independent grid-like data which can be fed into uncoupled models as CNN. To begin with, the central node is mapped to the Euclidean structured grid with the size of $k$ × 1 × 3. Then a convolutional layer is used to extract information from the grid-like data. Next, the proposed model employs attention filters to learn the weight of each grid, and fully-connected layers are finally applied to obtain the output.


*2024-10-31*

#### [VQ-GNN: a universal framework to scale-up graph neural networks using vector quantization](https://dl.acm.org/doi/10.5555/3540261.3540777)

*Mucong Ding, Kezhi Kong, Jingling Li, Chen Zhu, John Dickerson, Furong Huang, Tom Goldstein*

*NeurIPS 2021*

This paper proposes a framework to scale up any convolution-based GNNs using Vector Quantization (VQ) without compromising the performance. In contrast to sampling-based techniques, it is able to preserve all the messages passed in a mini-batch of nodes by learning and updating a small number of quantized reference vectors of global node representations, using VQ within each GNN layer.


*2024-10-28*

#### [Decoupling the depth and scope of graph neural networks](https://dl.acm.org/doi/10.5555/3540261.3541765)

*Hanqing Zeng, Muhan Zhang, Yinglong Xia, Ajitesh Srivastava, Andrey Malevich, Rajgopal Kannan, Viktor Prasanna, Long Jin, Ren Chen*

*NeurIPS 2021*

To handle the issue of oversmoothing of GNN models, this paper decouples the depth and scope of GNNs – to generate representation of a target entity by first extracting a localized subgraph as the bounded-size scope, and then applying a GNN of arbitrary depth on top of the subgraph. A properly extracted subgraph consists of a small number of critical neighbors, while excluding irrelevant ones.


*2024-10-27*

#### [Large scale learning on non-homophilous graphs: new benchmarks and strong simple methods](https://dl.acm.org/doi/10.5555/3540261.3541859)

*Derek Lim, Felix Hohne, Xiuyu Li, Sijia Linda Huang, Vaishnavi Gupta, Omkar Bhalerao, Ser-Nam Lim*

*NeurIPS 2021*

Graph machine learning on homophilous graphs usually assumes that nodes with similar labels are likely to connect with each other However, previous works demonstrated performance degradation on non-homophilous graphs. To address the issue, this paper first collects diverse nonhomophilous datasets from a variety of application areas to form a large non-homophilous graph. Then it proposes a simple method that admits straightforward minibatch training and inference, with superior performance.


*2024-10-26*

#### [SAGES: Scalable Attributed Graph Embedding With Sampling for Unsupervised Learning](https://ieeexplore.ieee.org/document/9705119)

*Jialin Wang, Xiaoru Qu, Jinze Bai, Zhao Li, Ji Zhang, Jun Gao*

*IEEE TKDE 2023*

This paper proposes a graph sampler that considers both the node connections and node attributes, thus nodes having a high influence on each other will be sampled in the same subgraph. After that, an unbiased Graph Autoencoder (GAE) with structure-level, content-level, and community-level reconstruction loss is built on the properly-sampled subgraphs in each epoch.


*2024-10-25*

#### [Sampling Methods for Efficient Training of Graph Convolutional Networks: A Survey](https://ieeexplore.ieee.org/document/9601152/)

*Xin Liu, Mingyu Yan, Lei Deng, Guoqi Li, Xiaochun Ye, Dongrui Fan*

*IEEE/CAA Journal of Automatica Sinica 2022*

This paper categorizes sampling methods for efficient training of GCN, including node-wise, layer-wise, subgraph-based, and heterogeneous sampling. To highlight the characteristics and differences of sampling methods, it also presents a detailed comparison within each category and gives an overall comparative analysis for the sampling methods in all categories.


*2024-10-24*

#### [ByteGNN: efficient graph neural network training at large scale](https://dl.acm.org/doi/10.14778/3514061.3514069)

*Chenguang Zheng, Hongzhi Chen, Yuxuan Cheng, Zhezheng Song, Yifan Wu, Changji Li, James Cheng, Hao Yang, Shuai Zhang*

*Proceedings of the VLDB Endowment, Volume 15, Issue 6, 2022*

To address the issues of existing distributed GNN training systems including high network communication cost, low CPU utilization, and poor end-to-end performance, this paper proposes ByteGNN with three key designs: (1) an abstraction of mini-batch graph sampling to support high parallelism, (2) a two-level scheduling strategy to improve resource utilization and to reduce the end-to-end GNN training time, and (3) a graph partitioning algorithm tailored for GNN workloads.


*2024-10-23*

#### [Ada-GNN: Adapting to Local Patterns for Improving Graph Neural Networks](https://dl.acm.org/doi/10.1145/3488560.3498460)

*Zihan Luo, Jianxun Lian, Hong Huang, Hai Jin, Xing Xie*

*WSDM 2022*

Instead of using a unified model to learn representations for all nodes on a large graph, this paper proposes a model-agnostic framework for scalable GNNs to improve their performance by generating personalized models at the subgroup-level. Specifically, it first applies a graph partition algorithm like METIS to divide the whole graph into multiple non-overlapped subgraphs, and tags each node with its corresponding subgraph ID as a group-wise label. Then a meta adapter is designed to learn a good global model from all subgroups and adapts to local models with a few instances in a subgraph, which helps preserve global coherence and learn local distinction.


*2024-10-22*

#### [Graph Embedding with Hierarchical Attentive Membership](https://dl.acm.org/doi/10.1145/3488560.3498499)

*Lu Lin, Ethan Blaser, Hongning Wang*

*WSDM 2022*

This paper studies the property of graphs as latent hierarchical grouping of nodes, where each node manifests its membership to a specific group based on the context composed by its neighboring nodes. In each layer of the aggregation operation, a group membership is firstly sampled for each node. Then the information from neighbors is attended by the inferred membership to generate node states for the next layer. The node states for each layer are learned by recovering the context within a certain neighborhood scope. Meanwhile, inter-layer constraints are introduced to inject the inclusive relation between membership assignments across layers.


*2024-10-21*

#### [PaSca: A Graph Neural Architecture Search System under the Scalable Paradigm](https://dl.acm.org/doi/10.1145/3485447.3511986)

*Wentao Zhang, Yu Shen, Zheyu Lin, Yang Li, Xiaosen Li, Wen Ouyang, Yangyu Tao, Zhi Yang, Bin Cui*

*WWW 2022*

This paper summarizes a new paradigm that offers a principled approach to systemically construct the design space for scalable GNNs, rather than studying individual designs. Through deconstructing the message passing mechanism, it presents a novel Scalable Graph Neural Architecture Paradigm, together with a general architecture design space consisting of 150k different designs. Following this paradigm, it implements an auto-search engine that automatically searches well-performing and scalable GNN architectures to balance the trade-off between multiple criteria (e.g., accuracy and efficiency) via multi-objective optimization.


*2024-10-19*

#### [Rethinking Node-wise Propagation for Large-scale Graph Learning](https://dl.acm.org/doi/10.1145/3589334.3645450)

*Xunkai Li, Jingyuan Ma, Zhengyu Wu, Daohan Su, Wentao Zhang, Rong-Hua Li, Guoren Wang*

*WWW 2024*

To address the issue that existing node-wise propagation optimization strategies are insufficient on web-scale graphs with intricate topology, where a full portrayal of nodes' local properties is required, this paper proposes Adaptive Topology-aware Propagation (ATP), which reduces potential high-bias propagation and extracts structural patterns of each node in a scalable manner to improve running efficiency and predictive performance. Specifically, ATP is crafted to be a plug-and-play node-wise propagation optimization strategy, allowing for offline execution independent of the graph learning process.


*2024-10-18*

#### [NPA: Improving Large-scale Graph Neural Networks with Non-parametric Attention](https://dl.acm.org/doi/10.1145/3626246.3653399)

*Wentao Zhang, Guochen Yan, Yu Shen, Yang Ling, Yangyu Tao, Bin Cui, Jian Tang*

*SIGMOD 2024*

This paper proposes non-parametric attention (NPA), a plug-and-play module for non-parametric GNNs. To address the issue of over-smoothing, NPA introduces global attention by assigning node-adaptive attention weights to newly propagated features to measure the newly generated global information compared with the previous features. In this way, it can adaptively remove the over-smoothed and redundant feature information in each propagation step. To tackle the fixed propagation weights issue in local feature propagation, NPA further proposes local attention to consider the similarity of propagated features and assigns more attention (i.e., larger propagation weight) to similar nodes in each propagation step.


*2024-10-17*

#### [Hierarchical Dynamic Graph Clustering Network](https://ieeexplore.ieee.org/document/10320217)

*Jie Chen, Licheng Jiao, Xu Liu, Lingling Li, Fang Liu, Puhua Chen, Shuyuan Yang, Biao Hou*

*IEEE TKDE 2023*

This paper proposes a hierarchical dynamic graph clustering network for visual feature learning. The initial graph is constructed in high-dimensional feature domain of images. To mine the hierarchical geometric features in latent graph space, adaptive clustering network (ClusterNet) is performed to learn discriminative clusters and generates cluster-based coarse graph. Then, graph convolutional networks are used to diffuse, transform and aggregate information among clusters.


*2024-10-16*

#### [SHINE: A Scalable Heterogeneous Inductive Graph Neural Network for Large Imbalanced Datasets](https://ieeexplore.ieee.org/document/10478631)

*Rafaël Van Belle, Jochen De Weerdt*

*IEEE TKDE 2024*

This paper proposes a scalable heterogeneous inductive GNN for node classification over large imbalanced datasets, which comprises three core components: (1) a sampler based on nearest-neighbor (NN) search, (2) a heterogeneous GNN (HGNN) layer with a novel relationship aggregator, and (3) aggregator functions tailored to skewed class distributions.


*2024-10-10*

#### [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://dl.acm.org/doi/10.1145/3292500.3330925)

*Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh*

*KDD 2019*

This paper proposes Cluster-GCN as an algorithm that is suitable for SGD-based training by exploiting the graph clustering structure. It works as the following: at each step, it samples a block of nodes that associate with a dense subgraph identified by a graph clustering algorithm, and restricts the neighborhood search within this subgraph. This simple but effective strategy leads to significantly improved memory and computational efficiency while being able to achieve comparable test accuracy with previous algorithms.


*2024-10-09*

#### [Hierarchical Graph Representation Learning with Differentiable Pooling](https://dl.acm.org/doi/10.5555/3327345.3327389)

*Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, Jure Leskovec*

*NeurIPS 2018*

DIFFPOOL learns a differentiable soft cluster assignment (implemented as softmax attention among nodes) for nodes at each layer of a deep GNN, mapping nodes to a set of clusters, which then form the coarsened input for the next GNN layer.


*2024-09-24*

#### [Calibrating Graph Neural Networks from a Data-centric Perspective](https://dl.acm.org/doi/10.1145/3589334.3645562)

*Cheng Yang, Chengdong Yang, Chuan Shi, Yawen Li, Zhiqiang Zhang, Jun Zhou*

*WWW 2024*

Existing calibration methods primarily focus on improving GNN models, e.g., adding regularization during training or introducing temperature scaling after training. However, this paper argues that the miscalibration of GNNs may stem from the graph data and can be alleviated through topology modification. Data observations were conducted by examining the impacts of decisive and homophilic edges on calibration performance, where decisive edges play a critical role in GNN predictions and homophilic edges connect nodes of the same class. By assigning larger weights to these edges in the adjacency matrix, an improvement in calibration performance was observed without sacrificing classification accuracy. Motivated by that, this paper proposes Data-centric Graph Calibration (DCGC), which uses two edge weighting modules to adjust the input graph for GNN calibration.


*2024-09-10*

#### [Expanding the Scope: Inductive Knowledge Graph Reasoning with Multi-Starting Progressive Propagation](https://arxiv.org/abs/2407.10430)

*Zhoutian Shao, Yuanning Cui, Wei Hu*

*ISWC 2024*

This paper proposes a GNN-based model for inductive knowledge graph link prediction. Specifically, it uses a pre-embedded GNN and selects a set of starting entities for the given query relation. The selected entities are categorized and added with additional links to the head entity given in the query. Then the model conducts message passing with attention mechanism, and finally decodes the target entity with an MLP over node (entity) embeddings.


*2024-09-08*

#### [Efficient Topology-aware Data Augmentation for High-Degree Graph Neural Networks](https://arxiv.org/abs/2406.05482)

*Yurui Lai, Xiaoyang Lin, Renchi Yang, Hongtao Wang*

*Technical report for the paper accepted to KDD 2024*

The message-passing paradigm of GNN brings over-smoothing and efficiency issues over high-degree graphs, where most nodes have dozens of neighbors, such as social networks, transaction graphs, power grids, etc. Additionally, such graphs usually encompass rich and complex structure semantics, which are hard to capture merely by feature aggregations in GNNs. To address these limitations, it proposes a front-mounted data augmentation framework for GNNs on HDGs, which includes two key modules: (i) feature expansion with structure embeddings, and (ii) topology- and attribute-aware graph sparsification.


*2024-09-07*

#### [Mitigating Label Noise on Graph via Topological Sample Selection](https://arxiv.org/abs/2403.01942)

*Yuhao Wu, Jiangchao Yao, Xiaobo Xia, Jun Yu, Ruxin Wang, Bo Han, Tongliang Liu*

*ICML 2024*

Existing approaches for sample selection over noisily labelled graph data have 2 limitations, (1) nodes located near topological class boundaries are very informative for classification but cannot be successfully distinguished by the heuristic sample selection, (2) there is no available measure that considers the graph topological information to promote sample selection in a graph. To address them, this paper proposes a Topological Sample Selection method that boosts the informative sample selection process in a graph by utilizing topological information.


*2024-09-04*

#### [Generalized Graph Prompt: Toward a Unification of Pre-Training and Downstream Tasks on Graphs](https://arxiv.org/abs/2311.15317)

*Xingtong Yu, Zhenghao Liu, Yuan Fang, Zemin Liu, Sihong Chen, Xinming Zhang*

*IEEE TKDE (Extension of WWW'23 paper [GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3543507.3583386))*

This paper extends the graph pre-training framework GraphPrompt to GraphPrompt+ with two major enhancements. First, it generalizes a few graph pre-training tasks beyond simple link prediction to broaden the compatibility with the task template. Second, it designs a more generalized prompt that incorporates a series of prompt vectors within every layer of the pre-trained graph encoder.


*2024-09-03*

#### [NeurCAM: Interpretable Neural Clustering via Additive Models](https://arxiv.org/abs/2408.13361)

*Nakul Upadhya, Eldan Cohen*

*ECAI 2024*

This paper introduces the Neural Clustering Additive Model for the interpretable clustering problem, that leverages neural generalized additive models to provide fuzzy cluster membership with additive explanations of the obtained clusters. To promote the sparsity in model’s explanations, it uses selection gates that explicitly limit the number of features and pairwise interactions.


*2024-08-31*

#### [Graph Reinforcement Learning for Combinatorial Optimization: A Survey and Unifying Perspective](https://arxiv.org/abs/2404.06492)

*Victor-Alexandru Darvariu, Stephen Hailes, Mirco Musolesi*

*TMLR*

This paper summarizes graph reinforcement learning as a decision-making method for graph combinatorial problems. The works are reviewed along the dividing line of whether the goal is to optimize graph structure given a process of interest, or to optimize the outcome of the process itself under fixed graph structure. It also discusses the common challenges facing the field and open research questions.


*2024-08-19*

#### [Graph Neural Networks with Learnable Structural and Positional Representations](https://arxiv.org/abs/2110.07875)

*Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, Xavier Bresson*

*ICLR 2022*

This paper investigates the usage of positional encoding (PE) of the nodes as GNN input. Possible graph PE are Laplacian eigenvectors. Specifically, it proposes a framework that decouples structural and positional representations to make easy for the network to learn these two essential properties.


*2024-07-20*

#### [A Survey of Data-Efficient Graph Learning](https://arxiv.org/abs/2402.00447)

*Wei Ju, Siyu Yi, Yifan Wang, Qingqing Long, Junyu Luo, Zhiping Xiao, Ming Zhang*

*IJCAI 2024 Survey*

This paper summarizes existing research efforts on Data-Efficient Graph Learning with a taxonomy of three top categories: self-supervised graph learning, semi-supervised graph learning, and few-shot graph learning. Specifically, self-supervised graph learning designs tasks that encourage the model to learn meaningful representations from the graph data itself; semi-supervised graph learning leverages the relationships within the labeled and unlabeled instances to guide the model in learning representations for unlabeled nodes or graphs; few-shot graph learning equips the model with the ability to learn from a few annotated instances and then apply this acquired knowledge to make predictions on new, unseen data.


*2024-07-16*

#### [MuGSI: Distilling GNNs with Multi-Granularity Structural Information for Graph Classification](https://arxiv.org/abs/2406.19832)

*Tianjun Yao, Jiaqi Sun, Defu Cao, Kun Zhang, Guangyi Chen*

*TheWebConf 2024*

This paper explores the GNN-to-MLP knowledge distillation (KD) approach for the graph classification task. Compared with typical node-level classification using KD, graph classification suffers from (1) the inherent sparsity of learning signals due to soft labels being generated at the graph level, and (2) the limited expressiveness of student MLPs. To alleviate the problem, it employs a loss function composed of three distinct targets: graph-level distillation, subgraph-level distillation, and node-level distillation.


*2024-06-14*

#### [Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks](https://openreview.net/forum?id=up6hr4hIQH)

*Xu Zheng, Farhad Shirani, Tianchun Wang, Wei Cheng, Zhuomin Chen, Haifeng Chen, Hua Wei, Dongsheng Luo*

*ICLR 2024*

An explanation function for GNNs takes a pre-trained GNN along with a graph as input, to produce a "sufficient statistic" subgraph with respect to the graph label in a classification task. A main challenge in studying GNN explainability is to provide fidelity measures that evaluate the performance of these explanation functions. This paper identifies shortcomings of current metrics, attributing them to the out-of-distribution nature of explanation subgraphs. To address the problem, modified fidelity metrics are proposed to transform explanation subgraphs to approximate the underlying data distribution.


*2024-06-13*

#### [Graph Metanetworks for Processing Diverse Neural Architectures](https://openreview.net/forum?id=ijK5hyxs0n)

*Derek Lim, Haggai Maron, Marc T. Law, Jonathan Lorraine, James Lucas*

*ICLR 2024*

This is an interesting work that tries to convert typical neural networks into a meta-structure, by taking their weights as inputs, representing them as graphs and using graph neural networks to process them. The main contribution is that of handing a much more diverse set of neural networks, and it shows good flexibility while being able to control the original network outputs by adjusting the corresponding graph meta-network.


*2024-06-12*

#### [Learning Large DAGs is Harder than you Think: Many Losses are Minimal for the Wrong DAG](https://openreview.net/forum?id=gwbQ2YwLhD)

*Jonas Seng, Matej Zečević, Devendra Singh Dhami, Kristian Kersting*

*ICLR 2024*

In DAGs, a node corresponds to a random variable and each edge marks a direct statistical dependence between two random variables. The absence of an edge encodes (in)direct independencies between random variables. The paper discusses that the optimization of a data-fitting loss can be misleading to find the DAG structure underlying the data. Basically, a root cause intervenes in the loss through its variance; and the optimization objective can thus prefer considering a root cause with high variance as an effect, than as a root cause.


*2024-06-09*

#### [Quantifying Network Similarity using Graph Cumulants](https://arxiv.org/abs/2107.11403)

*Gecia Bravo-Hermsdorff, Lee M. Gunderson, Pierre-André Maugis, Carey E. Priebe*

*JMLR 2023*

To test whether networks are sampled from the same distribution, this paper evaluates two statistical metrics based on subgraph counts. The first uses empirical subgraph densities to estimate the underlying distribution, and the second converts these subgraph densities into estimates of the **graph cumulants** of the distribution. It demonstrates (theoretically and empirically) the superior performance of the graph cumulants.


*2024-06-08*

#### [Lie Group Decompositions for Equivariant Neural Networks](https://openreview.net/forum?id=p34fRKp8qA)

*Mircea Mironenco, Patrick Forré*

*ICLR 2024*

Existing works incorporating invariance and equivariance to geometrical transformations usually focus on the case where the symmetry group employed is compact or abelian or both. This paper explores enlarged transformation groups principally through the use of Lie algebra. Specifically, it presents a framework by which it is possible to work with Lie groups 𝐺=GL+(𝑛,𝑅) and 𝐺=SL(𝑛,𝑅), as well as their representation as affine transformations 𝑅𝑛⋊𝐺. Invariant integration as well as a global parametrization is realized by decomposing the "larger" groups into subgroups and submanifolds which can be handled individually.


*2024-06-07*

#### [GNNBoundary: Towards Explaining Graph Neural Networks through the Lens of Decision Boundaries](https://openreview.net/forum?id=WIzzXCVYiH)

*Xiaoqi Wang, Han Wei Shen*

*ICLR 2024*

Focusing on the graph classification task, this paper aims to identify decision boundaries between a pair of classes, which is defined as a set of graphs having the same highest classification score for two neighboring classes. To achieve this, it first proposes an algorithm to identify pairs of classes whose decision regions are adjacent. Then for pairs of adjacent classes, the near-boundary graphs between them are effectively generated by optimizing an objective function specifically designed for boundary graph generation.


*2024-06-05*

#### [Efficient Subgraph GNNs by Learning Effective Selection Policies](https://openreview.net/forum?id=gppLqZLQeY)

*Beatrice Bevilacqua, Moshe Eliasof, Eli Meirom, Bruno Ribeiro, Haggai Maron*

*ICLR 2024*

Subgraph GNNs are provably expressive neural architectures that learn graph representations from sets of subgraphs, despite the high computational complexity associated with message passing on many subgraphs. This paper studies the problem of selecting a small subset of subgraphs in a data-driven fashion. It first proves that there are families of WL-indistinguishable graphs for which there exist efficient subgraph selection policies: small subsets of subgraphs can already identify all the graphs within the family. Then it proposes an approach for learning how to select subgraphs in an iterative manner.


*2024-06-04*

#### [Counting Graph Substructures with Graph Neural Networks](https://openreview.net/forum?id=qaJxPhkYtD)

*Charilaos Kanatsoulis, Alejandro Ribeiro*

*ICLR 2024*

This paper studies the substructure counting ability of a type of message-passing networks, where node features are randomly initialized and the output representation is obtained by taking expectation over the randomness. Based on these representations, it proves that GNNs can learn how to count cycles, cliques, quasi-cliques, and the number of connected components in a graph.


*2024-05-26*

#### [Polynormer: Polynomial-Expressive Graph Transformer in Linear Time](https://openreview.net/forum?id=hmv1LpNfXa)

*Chenhui Deng, Zichao Yue, Zhiru Zhang*

*ICLR 2024*

This paper proposes a polynomial-expressive graph transformer model with linear complexity, which is built upon a base model that learns a high-degree polynomial on input features. To enable the base model permutation equivariant, the graph topology and node features are separately integerated, resulting in local and global equivariant attention models. Consequently, it adopts a linear local-to-global attention scheme to learn high-degree equivariant polynomials whose coefficients are controlled by attention scores.


*2024-05-25*

#### [From Graphs to Hypergraphs: Hypergraph Projection and its Reconstruction](https://openreview.net/forum?id=qwYKE3VB2h)

*Yanbang Wang, Jon Kleinberg*

*ICLR 2024*

This work proposes a ML-based hypergraph reconstruction approach that recovers the original hypergraph from its projected simple graph (via clique-expansion). The reason to use learning-based approach is that accurate reconstruction is beyond reach with structural information loss in the projection process.


*2024-05-21*

#### [Rethinking and Extending the Probabilistic Inference Capacity of GNNs](https://openreview.net/forum?id=7vVWiCrFnd)

*Tuo Xu, Lei Zou*

*ICLR 2024*

Different from the WL-test, this paper adopts an approach to examine the expressive power of GNNs from a probabilistic perspective. Specifically, it formulates GNN models as a family of Markov random fields that are applicable to different graph structures and invariant to permutations. Then it investigates the problem of to what extent can GNNs approximate the inference of graphical models.


*2024-05-15*

#### [SAME: Uncovering GNN Black Box with Structure-aware Shapley-based Multipiece Explanations](https://papers.nips.cc/paper_files/paper/2023/hash/14cdc9013d80338bf81483a7736ea05c-Abstract-Conference.html)

*Ziyuan Ye, Rihan Huang, Qilin Wu, Quanying Liu*

*NeurIPS 2023*

This paper proposes a post-hoc explanation framework for GNN models. Specifically, it first leverages an expansion-based Monte Carlo tree search to explore the multi-grained structure-aware connected substructure. Then the explanation results are encouraged to be informative of the graph properties by optimizing the combination of distinct single substructures. The final explanation is possible to be as explainable as the theoretically optimal explanation obtained by the Shapley value within polynomial time. *Note: Shapley value, originating from cooperative game theory, is the unique credit allocation scheme that satisfies the fairness axioms. This concept is similar to the importance scoring function for explanation with the consideration of feature interactions.*


*2024-04-27*

#### [On Structural Expressive Power of Graph Transformers](https://dl.acm.org/doi/10.1145/3580305.3599451)

*Wenhao Zhu, Tianyu Wen, Guojie Song, Liang Wang, Bo Zheng*

*KDD 2023*

Motivated by the well-known WL-test for measuring the expressive power of GNN models, this paper proposes Structural Encoding enhanced Global Weisfeiler-Lehman (SEG-WL) test as a generalized graph isomorphism test for the structural discriminative power of graph Transformers. With the SEG-WL test, it shows how graph Transformers’ expressive power is determined by the design of structural encodings, and presents conditions that make the expressivity of graph Transformers beyond WL-test and GNNs.


*2024-04-26*

#### [MixupExplainer: Generalizing Explanations for Graph Neural Networks with Data Augmentation](https://dl.acm.org/doi/10.1145/3580305.3599435)

*Jiaxing Zhang, Dongsheng Luo, Hua Wei*

*KDD 2023*

Graph Information Bottleneck (GIB) is a post-hoc explanation approach that maximizes the mutual information between the target label 𝑌 and the explanation 𝐺∗ while constraining the size of the explanation as the mutual information between the original graph 𝐺 and the explanation 𝐺∗. This paper studies the distribution shifting issue in existing GIB framework that affects explanation quality. To address the issue, it introduces a generalized GIB form that includes a label-independent graph variable, which is equivalent to the vanilla GIB. With the generalized GIB, it proposes a graph mixup method with theoretical guarantee to resolve the distribution shifting issue.


*2024-04-25*

#### [Improving the Expressiveness of K-hop Message-Passing GNNs by Injecting Contextualized Substructure Information](https://dl.acm.org/doi/10.1145/3580305.3599390)

*Tianjun Yao, Yingxu Wang, Kun Zhang, Shangsong Liang*

*KDD 2023*

This paper first analyzes the (limited) expressiveness of K-hop MPNNs, where the node representation is updated by iteratively aggregating information from neighbors within K-hop of the node. Then it proposes an substructure encoding function that improves the expressive power of any K-hop MPNN.


*2024-04-21*

#### [Are Message Passing Neural Networks Really Helpful for Knowledge Graph Completion?](https://aclanthology.org/2023.acl-long.597.pdf)

*Juanhui Li, Harry Shomer, Jiayuan Ding, Yiqi Wang, Yao Ma, Neil Shah, Jiliang Tang, Dawei Yin*

*ACL 2023*

By comparing the performance of KGC models with Message-Passing (MP) or simple MLP layers, i.e., replacing the adjacency matrix of the graph with identity matrix, this paper surprisingly finds that simple MLP models are able to achieve comparable performance to MPNNs, suggesting that MP may not be as crucial as previously believed. With further exploration, it shows careful scoring function and loss function design has a much stronger influence on KGC model performance.


*2024-04-14*

#### [Improving Expressivity of GNNs with Subgraph-specific Factor Embedded Normalization](https://dl.acm.org/doi/10.1145/3580305.3599388)

*Kaixuan Chen, Shunyu Liu, Tongtian Zhu, Ji Qiao, Yun Su, Yingjie Tian, Tongya Zheng, Haofei Zhang, Zunlei Feng, Jingwen Ye, Mingli Song*

*KDD 2023*

This paper proposes a subgraph-specific factor computed from the number of nodes, edges, and the eigenvalues of the adjacency matrix for each graph, which is exclusive for each non-isomorphic graph. Then it shows that by using this factor in a plug-and-play manner, the GNN models can be alleviated from the over-smoothing issue, and is at least as powerful as the 1-WL test in distinguishing non-isomorphism graphs.


*2024-04-13*

#### [When to Pre-Train Graph Neural Networks? From Data Generation Perspective!](https://dl.acm.org/doi/10.1145/3580305.3599548)

*Yuxuan Cao, Jiarong Xu, Carl Yang, Jiaan Wang, Yunchao Zhang, Chunping Wang, Lei CHEN, Yang Yang*

*KDD 2023*

This paper investigates the problem "when to pretrain models over graph data" without any usual "pretrain-finetune" attempts. Specifically, it first fits the pretraining data into graphon bases, where each element of graphon bases (i.e., a graphon) identifies a fundamental transferable pattern shared by a collection of pretraining graphs. Then the feasibility of pretraining is quantified as the generation probability of the downstream data from any generator in the generator space.


*2024-04-12*

#### [On the Theoretical Expressive Power and the Design Space of Higher-Order Graph Transformers](https://arxiv.org/abs/2404.03380)

*Cai Zhou, Rose Yu, Yusu Wang*

*AISTATS 2024*

This paper studies the theoretical expressive power of order-$k$ graph transformers and sparse variants. It first shows that, an order-$k$ graph transformer without additional structural information is less expressive than the $k$-Weisfeiler Lehman ($k$-WL) test despite its high computational cost. Then it explores sparsification strategies, and shows that a natural neighborhood-based sparse order-$k$ transformer model is not only computationally efficient, but also expressive – as expressive as $k$-WL test.


*2024-04-04*

#### [Globally Interpretable Graph Learning via Distribution Matching](https://arxiv.org/abs/2306.10447)

*Yi Nian, Yurui Chang, Wei Jin, Lu Lin*

*WWW 2024*

This paper investigates the global interpretation for GNN models, whose goal is to generate a few compact interpretive graphs to summarize class discriminative patterns the GNN model learns for decision making. It assumes that if the interpretation indeed contains essential patterns the model captures during training, then if use these interpretive graphs to train a model from scratch, this surrogate model should present similar behavior as the original model. Based on this, it proposes a new metric as model fidelity, which evaluates the predictive similarity between the surrogate model (trained via interpretative graphs) and the original model (normally trained via the training set). Then it also proposes a global interpretable graph learning model that can provide graph patterns as global interpretation during training.


*2024-03-27*

#### [Data-centric Graph Learning: A Survey](https://arxiv.org/abs/2310.04987)

*Yuxin Guo, Deyu Bo, Cheng Yang, Zhiyuan Lu, Zhongjian Zhang, Jixi Liu, Yufei Peng, Chuan Shi*

*Arxiv 2024*

This survey reviews graph learning approaches from the data-centric perspective. It categorizes existing data-centric graph learning methods by the stages in the graph learning pipeline, including pre-processing, training, and inference. Then it highlights the processing methods for different data structures in the graph data, i.e., topology, feature and label. Besides, it analyzes the potential influence of problematic graph data on the graph models and discusses how to alleviate these problems in a data-centric manner.


*2024-03-26*

#### [Graph Fairness Learning under Distribution Shifts](https://arxiv.org/abs/2401.16784)

*Yibo Li, Xiao Wang, Yujie Xing, Shaohua Fan, Ruijia Wang, Yaoqi Liu, Chuan Shi*

*WWW 2024*

This paper investigates the problem of fair prediction under distribution shifts, which relies on the assumption that the test data distribution is not the same as training data distribution. It proposes a model for fair prediction under distribution shifts with three modules, including a generative adversarial debiasing module, a graph generation module, and a group alignment module.


*2024-03-21*

#### [Link Prediction with Relational Hypergraphs](https://arxiv.org/abs/2402.04062)

*Xingyue Huang, Miguel Romero Orth, Pablo Barceló, Michael M. Bronstein, İsmail İlkan Ceylan*

*Arxiv 2024*

The presence of relational hyperedges makes link prediction a task between $k$ nodes for varying choices of $k$, which is substantially harder than link prediction with knowledge graphs, where every relation is binary ($k = 2$). This paper proposes two frameworks for link prediction with relational hypergraphs and conduct analysis of the expressive power of the resulting model architectures via corresponding relational Weisfeiler-Lehman algorithms.


*2024-03-16*

#### [One For All: Towards Training One Graph Model For All Classification Tasks](https://openreview.net/forum?id=4IT2pgc9v6)

*Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, Muhan Zhang*

*ICLR 2024 Spotlight*

This paper proposes a general framework for node classification over graphs based on LMs. Specifically, it proposes text-attributed graphs to unify different graph data by describing nodes and edges with natural language and uses LMs to encode the diverse and possibly cross-domain text attributes to feature vectors in the same embedding space. Further, it introduces the concept of nodes-of-interest to standardize different tasks with a single task representation. For in-context learning on graphs, it introduces a novel graph prompting paradigm that appends prompting substructures to the input graph, which enables it to address varied tasks without fine-tuning.


*2024-03-06*

#### [Hybrid Directional Graph Neural Network for Molecules](https://openreview.net/forum?id=BBD6KXIGJL)

*Junyi An, Chao Qu, Zhipeng Zhou, Fenglei Cao, Xu Yinghui, Yuan Qi, Furao Shen*

*ICLR 2024 Spotlight*

The equivariant operations in each layer of a GNN model may impose excessive constraints on the function form and network flexibility. This paper proposes a new model called the Hybrid Directional Graph Neural Network (HDGNN), which combines strictly equivariant operations with learnable modules, and is applied in chemical molecule properties prediction.


*2024-03-02*

#### [Improving Generalization in Equivariant Graph Neural Networks with Physical Inductive Biases](https://openreview.net/forum?id=3oTPsORaDH)

*Yang Liu, Jiashun Cheng, Haihong Zhao, Tingyang Xu, Peilin Zhao, Fugee Tsung, Jia Li, Yu Rong*

*ICLR 2024 Spotlight*

This paper proposes second-order equivariant graph neural ordinary differential equation as a method for modeling dynamics by computing second derivatives with equivariant GNNs. It theoretically analyzes the method's expressivity and empirically demonstrates the method on n-body problems, molecular dynamics, and motion capture.


*2024-02-23*

#### [SaNN: Simple Yet Powerful Simplicial-aware Neural Networks](https://openreview.net/forum?id=eUgS9Ig8JG)

*Sravanthi Gurugubelli, Sundeep Prabhakar Chepuri*

*ICLR 2024 Spotlight*

Simplicial neural networks (SNNs) are deep models for higher-order graph representation learning that learn low-dimensional embeddings of simplices in a simplicial complex by aggregating features of their respective upper, lower, boundary, and coboundary adjacent simplices. This paper proposes a scalable simplicial-aware neural network (SaNN) model with a constant run-time and memory requirements independent of the size of the simplicial complex and the density of interactions in it. SaNN is based on pre-aggregated simplicial-aware features as inputs to a neural network, thus having a strong simplicial-structural inductive bias.


*2024-02-20*

#### [Beyond Weisfeiler-Lehman: A Quantitative Framework for GNN Expressiveness](https://openreview.net/forum?id=HSKaGOi7Ar)

*Anonymous so far*

*ICLR 2024 Oral*

This paper proposes a framework for quantitatively studying the expressiveness of GNN architectures by  identifying a fundamental expressivity measure termed homomorphism expressivity, which quantifies the ability of GNN models to count graphs under homomorphism.


*2024-02-16*

#### [Graph Neural Networks for Learning Equivariant Representations of Neural Networks](https://openreview.net/forum?id=oO6FsMyDBt)

*Miltiadis Kofinas, Boris Knyazev, Yan Zhang, Yunlu Chen, Gertjan J. Burghouts, Efstratios Gavves, Cees G. M. Snoek, David W. Zhang*

*ICLR 2024 Oral*

This paper applies a graph neural network model to capture the permutation invariance in typical neural networks such as CNN. Specifically, it firstly represents the neural network as a layered computation graph, where $W_{i, j}$ is viewed as edge feature, and $b_i$ is viewed as node feacture. Then it adopts a GNN model that includes edge features to process the computation graph, which enables a single model to learn from neural graphs with diverse architectures.


*2024-01-23*

#### [A Generalized Neural Diffusion Framework on Graphs](https://arxiv.org/abs/2312.08616)

*Yibo Li, Xiao Wang, Hongrui Liu, Chuan Shi*

*AAAI 2024*

This paper proposes a general diffusion equation framework with a fidelity term for graph diffusion networks. It also proposes a high-order neighbor-aware diffusion equation, and derives a new type of graph diffusion network based on the framework.


*2024-01-17*

#### [Pitfalls in Link Prediction with Graph Neural Networks: Understanding the Impact of Target-link Inclusion & Better Practices](https://arxiv.org/abs/2306.00899)

*Jing Zhu, Yuhang Zhou, Vassilis N. Ioannidis, Shengyi Qian, Wei Ai, Xiang Song, Danai Koutra*

*WSDM 2024*

This paper first identifies the problem of including the edges being predicted at training/test time, which is a common practice in link prediction tasks, of low-degree nodes. In particular, it demonstrates 3 issues related to this practice, i.e., overfitting, distribution shift, and implicit test leakage. To address these issues, it proposes an effective GNN training framework by excluding edges which are incident to low-degree nodes in the training/test time, thus improving the accuracy especially for sparse graphs.


*2024-01-09*

#### [FreeDyG: Frequency Enhanced Continuous-Time Dynamic Graph Model for Link Prediction](https://openreview.net/forum?id=82Mc5ilInM)

*Anonymous*

*ICLR 2024 in Submission*

This paper proposes a link prediction model for continuous-time dynamic graphs. It extracts node representations based on their historical first-hop neighbors thus transforming the dynamic graph learning problem into time series analysis where node interactions are observed over sequential time points. Unlike previous works that primarily focus on the time domain, it focuses on the frequency domain to capture the periodic and "shift" behaviors of interaction patterns.


*2024-01-07*

#### [All in One: Multi-task Prompting for Graph Neural Networks](https://dl.acm.org/doi/10.1145/3580305.3599256)

*Xiangguo Sun, Hong Cheng, Jia Li, Bo Liu, Jihong Guan*

*KDD 2023 Best Research Paper*

Given the situation that graph tasks with node level, edge level, and graph level are far diversified, making the pre-training pretext often incompatible with these multiple tasks, this paper proposes a multi-task prompting method for graph models. Specifically, it first unifies the format of graph prompts and language prompts with the prompt token, token structure, and inserting pattern.  To narrow the gap between various graph tasks and pre-training strategies, it reformulates downstream problems to the graph-level task, and introduces meta-learning to efficiently learn a better initialization for the multi-task prompt of graphs.


*2024-01-06*

#### [One For All: Towards Training One Graph Model For All Classification Tasks](https://openreview.net/forum?id=4IT2pgc9v6)

*Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, Muhan Zhang*

This paper proposes a general framework that uses a single graph model for multiple classification tasks over graphs, including node classification, link prediction, and graph classification. Specifically, it proposes text-attributed graphs to unify different graph data by describing nodes and edges with natural language and uses language models to encode the diverse and possibly cross-domain text attributes to feature vectors in the same embedding space. Besides, it introduces the concept of nodes-of-interest to standardize different tasks with a single task representation. For in-context learning on graphs, it introduces a novel graph prompting paradigm that appends prompting substructures to the input graph, which enables it to address varied tasks without fine-tuning.


*2024-01-04*

#### [Graph-based Knowledge Distillation: A survey and experimental evaluation](https://arxiv.org/abs/2302.14643)

*Jing Liu, Tongya Zheng, Guanzheng Zhang, Qinfen Hao*

*Arxiv 2023*

This paper first introduces the background of graph and KD. It then provides a comprehensive summary of three types of Graph-based Knowledge Distillation methods, namely, Graph-based Knowledge Distillation for deep neural networks (DKD), Graph-based Knowledge Distillation for GNNs (GKD), and Self-Knowledge Distillation based Graph-based Knowledge Distillation (SKD). Each type is further divided into knowledge distillation methods based on the output layer, middle layer, and constructed graph.


*2024-01-03*

#### [Knowledge Distillation on Graphs: A Survey](https://arxiv.org/abs/2302.00219)

*Yijun Tian, Shichao Pei, Xiangliang Zhang, Chuxu Zhang, Nitesh V. Chawla*

*Arxiv 2023*

This paper summarizes the recent approaches on knowledge distillation on graphs (KDG). Specifically, it first introduces KDG challenges and bases, then categorizes and summarizes existing works of KDG by answering the following three questions: 1) what to distillate, 2) who to whom, and 3) how to distillate, and finally, shares some thoughts on future research directions.


*2024-01-02*

#### [Spectral Augmentation for Self-Supervised Learning on Graphs](https://openreview.net/forum?id=DjzBCrMBJ_p)

*Lu Lin, Jinghui Chen, Hongning Wang*

*ICLR  2023*

This paper proposes a spectral augmentation method which uses graph spectrum to capture structural properties and guide topology augmentations for graph self-supervised learning. The proposed method also brings promising generalization capability in transfer learning.


*2024-01-01*

#### [Revisiting Graph Contrastive Learning from the Perspective of Graph Spectrum](https://openreview.net/forum?id=L0U7TUWRt_X)

*Nian Liu, Xiao Wang, Deyu Bo, Chuan Shi, Jian Pei*

*NeurIPS 2022*

This paper investigates the problem of what information is essentially encoded into the learned representations by graph contrastive learning (GCL). It answers the question by establishing connections between GCL and graph spectrum. It finds that the difference of the high-frequency parts between two augmented graphs should be larger than that of low-frequency parts, and the learned representations by GCL essentially encode the low-frequency information. Based on these, it proposes a general GCL plug-in module which further improves the existing GCL performance.


*2023-12-29*

#### [Efficient and degree-guided graph generation via discrete diffusion modeling](https://dl.acm.org/doi/10.5555/3618408.3618589)

*Xiaohui Chen, Jiaxing He, Xu Han, Liping Liu*

*ICML 2023*

This paper proposes EDGE as an efficient discrete graph generative model, which is developed by reversing a discrete diffusion process that randomly removes edges until obtaining an empty graph. Specifically, it only focuses on a small portion of graph nodes and only adds edges between these nodes to improve the computational efficiency. Besides, it can explicitly model the node degrees of training graphs and then gain performance improvement in capturing graph statistics.


*2023-12-28*

#### [DiGress: Discrete Denoising diffusion for graph generation](https://openreview.net/forum?id=UaAD-Nu86WX)

*Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, Pascal Frossard*

*ICLR 2023*

Unlike typical graph generative models which represent the graph into continuous space, this paper introduces DiGress as a discrete denoising diffusion model for generating graphs with categorical node and edge attributes. It utilizes a discrete diffusion process that progressively edits graphs with noise, through the process of adding or removing edges and changing the categories. A graph transformer network is trained to revert this process, simplifying the problem of distribution learning over graphs into a sequence of node and edge classification tasks.


*2023-12-27*

#### [NVDiff: Graph Generation through the Diffusion of Node Vectors](https://arxiv.org/abs/2211.10794)

*Xiaohui Chen, Yukun Li, Aonan Zhang, Li-Ping Liu*

*Arxiv 2023*

To combine the generation process of nodes and edges in graph generation instead of conducting them in parallel, this paper takes the VGAE (variational graph auto-encoders) structure and uses a score-based generative model (SGM) as a flexible prior to sample node vectors. By modeling only node vectors in the latent space, NVDiff significantly reduces the dimension of the diffusion process and thus improves sampling speed.


*2023-12-26*

#### [Permutation Invariant Graph Generation via Score-Based Generative Modeling](https://proceedings.mlr.press/v108/niu20a.html)

*Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, Stefano Ermon*

*AISTATS 2020*

This paper proposes a permutation invariant approach to modeling graphs using a score-based generative modeling. In particular, it designs a permutation equivariant, multi-channel graph neural network to model the gradient of the data distribution at the input graph (a.k.a., the score function). This permutation equivariant model of gradients implicitly defines a permutation invariant distribution for graphs. Then the GNN model is trained with score matching and sampled with annealed Langevin dynamics.


*2023-12-18*

#### [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km)

*Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka*

*ICLR 2019*

This paper firstly analyzes the theoretical expressive power of graph neural networks by connecting the MPNN framework with 1-WL test. It proves that the 1-WL test is both the upper bound and the lower bound for the expressive power of MPNN. Further, it introduces GIN as a GNN architecture that reaches the maximum expressive power within this theoretical framework.


*2023-12-06*

#### [Newton-Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems](https://arxiv.org/abs/2305.14642)

*Lingbing Guo, Weiqing Wang, Zhuo Chen, Ningyu Zhang, Zequn Sun, Yixuan Lai, Qiang Zhang, Huajun Chen*

*NeurIPS 2023*

This paper investigates the GNN-based system dynamics following the observation that they actually share a common paradigm that learns the integration of the velocity over the interval between the initial and terminal coordinates. Then it proposes a new approach to predict the integration based on several velocity estimations with Newton–Cotes formulas and theoretically prove its effectiveness.


*2023-11-16*

#### [Bridged-GNN: Knowledge Bridge Learning for Effective Knowledge Transfer](https://dl.acm.org/doi/10.1145/3583780.3614796)

*Wendong Bi, Xueqi Cheng, Bingbing Xu, Xiaoqian Sun, Easton Li Xu, Huawei Shen*

*CIKM 2023*

This paper proposes an architecture for knowledge transfer based on GNN models, which includes an adaptive knowledge retrieval module for building a bridge-graph and a graph knowledge transfer module. It defines the task of knowledge transfer as knowledge bridge learning that conducts sample-wise knowledge transfers within the learned scope.


*2023-11-14*

#### [Wasserstein Barycenter Matching for Graph Size Generalization of Message Passing Neural Networks](https://proceedings.mlr.press/v202/chu23a.html)

*Xu Chu, Yujie Jin, Xin Wang, Shanghang Zhang, Yasha Wang, Wenwu Zhu, Hong Mei*

*ICML 2023*

To address the uncontrollable convergence rate caused by correlations across nodes in the underlying dimensional signal-generating space, this paper proposes to use Wasserstein bary centers as graph-level consensus to combat node-level correlations. Specifically, it proposes a Wasserstein bary center matching (WBM) layer that represents an input graph by Wasserstein distances between its MPNN-filtered node embeddings versus some learned class-wise bary centers. Theoretically, it shows that the convergence rate of an MPNN with a WBM layer is controllable and independent to the dimensionality of the signal-generating space.


*2023-11-13*

#### [Generalization Analysis of Message Passing Neural Networks on Large Random Graphs](https://papers.nips.cc/paper_files/paper/2022/hash/1eeaae7c89d9484926db6974b6ece564-Abstract-Conference.html)

*Sohir Maskey, Ron Levie, Yunseok Lee, Gitta Kutyniok*

*NeurIPS 2022*

This paper studies the generalization error of MPNNs in graph classification and regression. By assuming that graphs of different classes are sampled from different random graph models, it shows that, when training a MPNN on a dataset sampled from such a distribution, the generalization gap increases in the complexity of the MPNN, and decreases, not only with respect to the number of training samples, but also with the average number of nodes in the graphs. This shows how a MPNN with high complexity can generalize from a small dataset of graphs, as long as the graphs are large.


*2023-11-07*

#### [Heterogeneous Temporal Graph Neural Network Explainer](https://dl.acm.org/doi/10.1145/3583780.3614909)

*Jiazheng Li, Chunhui Zhang, Chuxu Zhang*

*CIKM 2023*

This paper proposes a link-prediction model for temporal knowledge graphs with an explanation generation module. Specifically, it has a typical encoder-decoder pipeline for the link prediction task. After getting the node embeddings from the encoder, it also feeds them into an explainer network which samples subgraphs and applies a MLP network to compute the importance score for the subgraphs and identifies the one that contributes most to the prediction.


*2023-11-04*

#### [Recipe for a General, Powerful, Scalable Graph Transformer](https://papers.nips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html)

*Ladislav Rampásek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini*

*NeurIPS 2022*

This paper proposes an architecture of a general graph transformer. It decouples the positional and structural encodings as local positional encodings and local structural encodings. Then it incorporates graph features including node features, global features and edge features into a general network layer which consists of MPNN layers and global attention layers to be a plug-and-play module.


*2023-11-03*

#### [Structure-Aware Transformer for Graph Representation Learning](https://proceedings.mlr.press/v162/chen22r.html)

*Dexiong Chen, Leslie O'Bray, Karsten M. Borgwardt*

*ICML 2022*

This paper proposes a model to encode the graph structural information into the attention mechanism. Given an input graph, it first extracts k-hop subgraphs using a structure extractor and updates the node representation using a GNN model with attention over subgraphs. Then it applies a usual transformer layer on top of that. To achieve a trade-off between computational efficiency and expressiveness, it implements k-subtree and k-subgraph extractors to compute local structures around each node.


*2023-11-02*

#### [Do Transformers Really Perform Badly for Graph Representation?](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)

*Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu*

*NeurIPS 2021*

This is an initial work to apply the idea of transformer to graphs, especially for graph representation learning. On the one hand, it represents the node features as Q, K, V, like a typical transformer, and meanwhile it also utilizes structural encodings to capture the graph structural information, including spatial encoding for connections between nodes, edge encoding, and centrality encoding to capture the node importance.


*2023-10-27*

#### [From Relational Pooling to Subgraph GNNs: A Universal Framework for More Expressive Graph Neural Networks](https://proceedings.mlr.press/v202/zhou23n.html)

*Cai Zhou, Xiyuan Wang, Muhan Zhang*

*ICML 2023*

Weisfeiler-Lehman test (1-WL) is a common graph isomorphism test, which also bounds the expressivity of message passing neural networks (MPNNs). k-dimensional Weisfeiler-Lehman test has stronger expressivity. It assigns colors to all k-tuples and iteratively updates them. This paper proposes k-WL with l labels as k,l-WL being an enhanced test framework, and demonstrates its expressivity hierarchy.


*2023-10-21*

#### [Graph Generative Model for Benchmarking Graph Neural Networks](https://proceedings.mlr.press/v202/yoon23d.html)

*Minji Yoon, Yue Wu, John Palowitch, Bryan Perozzi, Russ Salakhutdinov*

*ICML 2023*

This paper proposes a graph generative model to produce benchmarking graph datasets for evaluating GNN models while maintaining the data privacy. Specifically, given a source graph represented by the adjacency matrix, node features and node label matrices, the target is to produce a new graph which can reflect similar performance measurements w.r.t. the source graph for GNN models, being scalable and can guarantee data privacy. To achieve this, the paper proposes to learn the distribution of the computation graph by reforming the task into a discrete-value sequence generation problem.


*2023-10-20*

#### [LazyGNN: Large-Scale Graph Neural Networks via Lazy Propagation](https://proceedings.mlr.press/v202/xue23c.html)

*Rui Xue, Haoyu Han, MohamadAli Torkamani, Jian Pei, Xiaorui Liu*

*ICML 2023*

Typical deep GNN models suffer from scalability issues due to the exponential neighborhood search requirement. To address the problem, this paper proposes a shallower GNN model LazyGNN, which introduces a mixing of current and history features using a hyperparameter in each layer. It demonstrates compatible performance with deep models while being much more efficient compared with existing GNN models.


*2023-10-18*

#### [Relevant Walk Search for Explaining Graph Neural Networks](https://proceedings.mlr.press/v202/xiong23b.html)

*Ping Xiong, Thomas Schnake, Michael Gastegger, Grégoire Montavon, Klaus-Robert Müller, Shinichi Nakajima*

*ICML 2023*

This paper proposes a max-product based approach of relevant walk search over GNN, which provides instance-level explanations for the model by searching for the most prominent feature interaction path. It also proposes an efficient polynomial-time algorithm for fining the top-K relevant walks.


*2023-10-06*

#### [Dink-Net: Neural Clustering on Large Graphs](https://proceedings.mlr.press/v202/liu23v.html)

*Yue Liu, Ke Liang, Jun Xia, Sihang Zhou, Xihong Yang, Xinwang Liu, Stan Z. Li*

*ICML 2023*

This paper proposes a method for node clustering on large graphs. It aggregates the two-step framework, i.e., representation learning and clustering optimization, into an end-to-end pipeline, where the clustering centers are initialized as learnable neural parameters.


*2023-10-05*

#### [Structural Re-weighting Improves Graph Domain Adaptation](https://proceedings.mlr.press/v202/liu23u.html)

*Shikun Liu, Tianchun Li, Yongbin Feng, Nhan Tran, Han Zhao, Qiang Qiu, Pan Li*

*ICML 2023*

To minimize the distributional gap between training data and real-world data in applications, typical graph domain adaptation (gda) methods aligning the distributions of node representations output by a single GNN encoder shared across the training and testing domains, which yields sub-optimal performance. This paper first identifies a type of distributional shift of node attributes and demonstrates that existing GDA methods are sub-optimal for this shift, and then proposes a structural re-weighting scheme to address this issue.


*2023-10-01*

#### [Alternately Optimized Graph Neural Networks](https://proceedings.mlr.press/v202/han23c.html)

*Haoyu Han, Xiaorui Liu, Haitao Mao, MohamadAli Torkamani, Feng Shi, Victor Lee, Jiliang Tang*

*ICML 2023*

This paper proposes a new optimization framework for GNN, by considering the typical end-to-end node classification problem as a bi-level optimization process. It uses a single-level optimization problem to couple the node features and graph structure information through a multi-view semi-supervised learning framework.


*2023-09-20*

#### [E(n) Equivariant Message Passing Simplicial Networks](https://proceedings.mlr.press/v202/eijkelboom23a.html)

*Floor Eijkelboom, Rob Hesselink, Erik J. Bekkers*

*ICML 2023*

A common model used on geometric graphs is the E(n) Equivariant Graph Neural Network (EGNN), which augments the message passing formulation to use the positional information while being equivariant to the Euclidean group E(n). This paper proposes a new model E(n) Equivariant Message Passing Simplicial Networks, to learn on geometric graphs and point clouds that is equivariant to rotations, translations and reflections.


*2023-09-19*

#### [Graph Neural Tangent Kernel: Convergence on Large Graphs](https://proceedings.mlr.press/v202/krishnagopal23a.html)

*Sanjukta Krishnagopal, Luana Ruiz*

*ICML 2023*

This paper investigates the training dynamics of GNNs with graph neural tangent kernals (GNTKs) and graphons. It proves that, on a sequence of graphs, the GNTKs converge to the graphon neural tangent kernal (NTK).


*2023-09-15*

#### [GOAT: A Global Transformer on Large-scale Graphs](https://proceedings.mlr.press/v202/kong23a.html)

*Kezhi Kong, Jiuhai Chen, John Kirchenbauer, Renkun Ni, C. Bayan Bruss, Tom Goldstein*

*ICML 2023*

This paper proposes a scalable global graph transformer model where each node theoretically attends to all nodes in the graph. For efficient implementation, it proposes a dimensionality reduction algorithm based on k-means thus reducing the memory complexity from quadratic to linear.


*2023-09-06*

#### [On the Expressive Power of Geometric Graph Neural Networks](https://proceedings.mlr.press/v202/joshi23a.html)

*Chaitanya K. Joshi, Cristian Bodnar, Simon V. Mathis, Taco Cohen, Pietro Lio*

*ICML 2023*

This paper proposes a geometric version of the WL test for discriminating geometric graphs embedded in Euclidean space. It concludes that (1) Invariant layers have limited expressivity as they cannot distinguish one-hop identical geometric graphs; (2) Equivariant layers distinguish a larger class of graphs by propagating geometric information beyond local neighbourhoods; (3) Higher order tensors and scalarisation enable maximally powerful geometric GNNs; and (4) GWL’s discrimination-based perspective is equivalent to universal approximation.


*2023-08-04*

#### [Logical Message Passing Networks with One-hop Inference on Atomic Formulas](https://openreview.net/forum?id=SoyOsp7i_l)

*Zihao Wang, Yangqiu Song, Ginny Wong, Simon See*

*ICLR 2023*

This paper proposes a Logical Message Passing Neural Network (LMPNN) that connects local one-hop inferences on atomic formulas to the global logical reasoning, based on the underlying entity embed


*2023-07-19*

#### [Message Function Search for Knowledge Graph Embedding](https://dl.acm.org/doi/10.1145/3543507.3583546)

*Shimin Di, Lei Chen*

*WWW 2023*

This paper proposes an integrated message passing module to be applied in GNN models. Motivated by existing GNNs usually have a single message passing function and can only handle some kind of KGs, this paper firstly proposes a message passing module as a MLP, in which the operators (i.e., intermediate message passing functions) can be customized (i.e., optimized) for the input KG. An algorithm is also proposed for searching the best combination of functions in a data-aware manner.


*2023-07-09*

#### [Graph Neural Networks without Propagation](https://dl.acm.org/doi/10.1145/3543507.3583419)

*Liang Yang, Qiuliang Zhang, Runjie Shi, Wenmiao Zhou, Bingxin Niu, Chuan Wang, Xiaochun Cao, Dongxiao He, Zhen Wang, Yuanfang Guo*

*WWW 2023*

Unlike typical GNNs using propagation-based message passing, this paper employs low-rank matrix decomposition as a local operator to update node features based on the ego-network of each node.


*2023-07-08*

#### [GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks](https://dl.acm.org/doi/10.1145/3543507.3583386)

*Zemin Liu, Xingtong Yu, Yuan Fang, Xinming Zhang*

*WWW 2023*

This paper proposes a pretraining-prompting framework for GNN, by applying link prediction as pre-training because it is self-supervised without requiring extra annotation. Afterwards, it applies prompts to guide node classification and graph classification as downstream tasks.


*2023-07-07*

#### [Global Counterfactual Explainer for Graph Neural Networks](https://dl.acm.org/doi/10.1145/3539597.3570376)

*Zexi Huang, Mert Kosan, Sourav Medya, Sayan Ranu, Ambuj K. Singh*

*WSDM 2023*

Counterfactual reasoning aims to find a way as explanation, by minimal changes in the input graph for changing the GNN prediction. Existing methods for counterfactual explanation of GNNs are limited to instance-specific local reasoning, while this paper aims to find a small set of representative counterfactual graphs that explains all input graphs. To achieve this, it proposes GCFExplainer that powered by vertex-reinforced random walks on an edit map of graphs with a greedy summary.


*2023-07-06*

#### [BLADE: Biased Neighborhood Sampling based Graph Neural Network for Directed Graphs](https://dl.acm.org/doi/10.1145/3539597.3570430)

*Srinivas Virinchi, Anoop Saladi*

*WSDM 2023*

This paper studies the problem of node recommendation in non-attributed directed graphs, which is to recommend top-k nodes with highest likelihood of a link with the query node. To learn the edge direction, it applies an asymmetric loss function and dual embeddings for each node. It also uses a biased sampling scheme to generate locally varying neighborhoods based on the graph structure.


*2023-07-05*

#### [Self-Supervised Graph Structure Refinement for Graph Neural Networks](https://dl.acm.org/doi/10.1145/3539597.3570455)

*Jianan Zhao, Qianlong Wen, Mingxuan Ju, Chuxu Zhang, Yanfang Ye*

*WSDM 2023*

Graph structure learning (GSL) aims to learn the adjacency matrix for graph neural networks. While existing methods usually learn the adjacency matrix and optimize for the downstream task in a joint manner, this paper proposes a pretrain-finetune pipeline for GSL. By regarding GSL as a link prediction task, it firstly uses a multi-view contrastive learning framework with both intra- and inter-view for pretraining. Then the graph structure is refined by modifying edges based on the pretrained model, and finetuned on the downstream task.


*2023-06-26*

#### [Evolving Computation Graphs](https://arxiv.org/pdf/2306.12943.pdf)

*Andreea Deac, Jian Tang*

*ICML 2023*

This paper proposes a model called ECG, short for Evolving Computation Graphs, for enhancing GNNs on heterophilic datasets. Specifically, it designs a module called weak classifier, and applies BGRL (i.e., bootstrapped graph latents, a graph representation learning method) in the model. The general architecture consists of two main steps, (1) embedding extraction, and (2) parallel message passing.


*2023-06-23*

#### [Finding the Missing-half: Graph Complementary Learning for Homophily-prone and Heterophily-prone Graphs](https://arxiv.org/pdf/2306.07608.pdf)

*Yizhen Zheng, He Zhang, Vincent CS Lee, Yu Zheng, Xiao Wang, Shirui Pan*

*ICML 2023*

Motivated by the existing GNN models usually focus only on either homophilic or heterophilic structures of the input graph and tend to ignore the other part, this paper proposes a new architecture to mitigate the problem. It consists of two modules. The first part, graph complementation, is designed to balance and utilize both kinds of structures in the graph, while the second is a modified GCN module.


*2023-06-22*

#### [Path Neural Networks: Expressive and Accurate Graph Neural Networks](https://arxiv.org/pdf/2306.05955.pdf)

*Gaspard Michel, Giannis Nikolentzos, Johannes Lutzeyer, Michalis Vazirgiannis*

*ICML 2023*

This paper proposes Path Neural Networks (PathNNs), that is a model that updates node representations by aggregating paths emanating from nodes. It presents three different variants of the PathNN model that aggregate single shortest paths, all shortest paths and all simple paths of length up to K. It also proves that two of these variants are strictly more powerful than the 1-WL algorithm.


*2023-06-17*

#### [Graph-Aware Language Model Pre-Training on a Large Graph Corpus Can Help Multiple Graph Applications](https://arxiv.org/pdf/2306.02592.pdf)

*Han Xie, Da Zheng, Jun Ma, Houyu Zhang, Vassilis N. Ioannidis, Xiang Song, Qing Ping, Sheng Wang, Carl Yang, Yi Xu, Belinda Zeng, Trishul Chilimbi*

*KDD 2023*

This paper introduces an architecture for graph-aware LM pre-training on a large graph corpus. The graph corpus generally refers to a heterogeneous graph in which nodes preserve text information and different types of edges can connect the same pair of nodes. It proposes two ways to pre-train GaLM, i.e., The LMs are pre-trained on a given large graph corpus either with or without the incorporation of GNN aggregators.


*2023-06-04*

#### [Stability and Generalization of ℓp-Regularized Stochastic Learning for GCN](https://arxiv.org/pdf/2305.12085.pdf)

*Shiyu Liu, Linsen Wei, Shaogao Lv, Ming Li*

*IJCAI 2023*

For GCN models, l-1 norm generally promotes sparsity while l-2 norm optimizes the smoothness. This paper further studies the tradeoff between sparsity and smoothness using an l-p norm (1 < p <= 2) and stochastic learning.


*2023-05-28*

#### [FedHGN: A Federated Framework for Heterogeneous Graph Neural Networks](https://arxiv.org/pdf/2305.09729.pdf)

*Xinyu Fu, Irwin King*

*IJCAI 2023*

This paper proposes a federated graph learning method by collaborative HGNN training among clients with private heterogeneous graphs, which also includes a schema-weight decoupling strategy and a schema coefficients alignment component.


*2023-05-03*

#### [Rethinking the Expressive Power of GNNs via Graph Biconnectivity](https://openreview.net/forum?id=r9hNv76KoT3)

*Bohang Zhang, Shengjie Luo, Shengjie Luo, Liwei Wang, Di He*

*ICLR 2023 Best Paper*

This paper investigates the ability of GNNs to identify the biconnectivity of graphs as a measure of their expressive power. Unlike the well-known WL test being an upper-bound of GNNs' expressive power, the problem of graph biconnectivity is easy to solve in practice. However, most of existing GNN models are unable to even identify whether or not a given graph contains a cut vertex or cut edge. Motivated by this, this paper analyzes a special kind of GNN, named Equivariant Subgraph Aggregation Network (ESAN), which is being able to identify the graph biconnectivity. It further proposes a framework called Generalized Distance Weisfeiler-Lehman (GD-WL) test, as an advanced method to test the GNN's expressive power.


*2023-04-14*

#### [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://doi.org/10.1145/3534678.3539321)

*Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Hongxia Yang, Chunjie Wang, Jie Tang*

*KDD 2022*

Generally, self-supervised methods can be divided into contrastive learning and generative learning. This paper proposes a generative learning method based on graph autoencoder model to re-construct the graph structure. It improves existing GAE models in 3 ways: (1) incorporating node features as target of reconstruction, (2) using a re-mask strategy with a 1-layer GNN as decoder, and (3) applying scaled cosine error instead of MSE as the loss function.


*2023-04-12*

#### [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](http://proceedings.mlr.press/v97/abu-el-haija19a.html)

*Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina Lerman, Hrayr Harutyunyan, Greg Ver Steeg, Aram Galstyan*

*ICML 2019*

This paper proposes to extend basic GCN models with message aggregation from higher-order neighbors. It proves that the model with higher-order graph convolution layers has different expressiveness power compared with basic GCNs.


*2023-04-10*

#### [Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling](https://arxiv.org/pdf/2304.02806.pdf)

*Haotao Wang, Ziyu Jiang, Yan Han, Zhangyang Wang*

*Arxiv 2023*

This paper proposes a Mixture-of-Experts strategy for the message aggregation of graph neural network. Indeed, the different "experts" are 1-hop and 2-hop GCN-like message passing functions with different weights. Besides, it also introduces the loss function to balance the workload between different experts.


*2023-03-30*

#### [Compressing Deep Graph Neural Networks via Adversarial Knowledge Distillation](https://dl.acm.org/doi/10.1145/3534678.3539315)

*Huarui He, Jie Wang, Zhanqiu Zhang, Feng Wu*

*KDD 2022*

To improve the discrepancy evaluation of the teacher-student model for knowledge distillation, this paper proposes an automatic discriminator and a generator based on adversarial training, which adaptively detects and decreases the discrepancy between the teacher and student model.


*2023-03-28*

#### [Graph Attention Multi-Layer Perceptron](https://doi.org/10.1145/3534678.3539121)

*Wentao Zhang, Ziqi Yin, Zeang Sheng, Yang Li, Wen Ouyang, Xiaosen Li, Yangyu Tao, Zhi Yang, Bin Cui*

*KDD 2022*

This paper proposes a new GNN architecture named Graph Attention MLP. It consists of node-level feature and label propagation with recursive attention. Then they are feeded into a downstream MLP. In the experiments it achieves slightly better performance than GAT but with much shorter training time.


*2023-03-21*

#### [Learning to Distill Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3539597.3570480)

*Cheng Yang, Yuxin Guo, Yao Xu, Chuan Shi, Jiawei Liu, Chunchen Wang, Xin Li, Ning Guo, Hongzhi Yin*

*WSDM 2023*

Knowledge distillation is to train a shallow student model w.r.t. a deep pretrained teacher model with soft feedbacks from the teacher model and the gold labels. Previous work generally implements the "temperature" to be a global hyper-parameter to adjust the distillation process. In contrast, this paper proposes the temperature to be learnable parameters w.r.t. the prediction loss of the student model.


*2023-03-15*

#### [Specformer: Spectral Graph Neural Networks Meet Transformers](https://openreview.net/pdf?id=0pdSt3oyJa1)

*Deyu Bo, Chuan Shi, Lele Wang, Renjie Liao*

*ICLR 2023*

This paper is motivated by the transformer module and tries to apply it to spectral graph neural network models. It uses the transformer module to process the eigenvalues of the spectrum. The self-attention mechanism is expected to capture the influences between the eigenvalues and produce new eigenvalues by decoding for computing graph convolution.


*2023-03-10*

#### [Global Self-Attention as a Replacement for Graph Convolution](https://dl.acm.org/doi/10.1145/3534678.3539296)

*Md. Shamim Hussain, Mohammed J. Zaki, Dharmashankar Subramanian*

*KDD 2022*

This paper proposes a general graph learning model (called Edge-augmented Graph Transformer, EGT) based on global graph attention. It uses global self-attention as an aggregation mechanism rather than static localized convolutional aggregation. It can also take structural information of arbitrary form by adding a dedicated pathway for pairwise structural information.


*2023-03-01*

#### [Adversarial Attack and Defense on Graph Data: A Survey](https://ieeexplore.ieee.org/document/9878092)

*Lichao Sun, Yingtong Dou, Carl Yang, Kai Zhang, Ji Wang, Yixin Liu, Philip S. Yu, Lifang He, Bo Li,*

*IEEE TKDE 2022*

This survey paper summarizes existing research about graph adversarial attack and defense methods. It firstly introduces the fundamental knowledge and the problem of GAT. Then it describes the process of GAT step-by-step, as well as adversarial defense methods. It also summarizes the metrics used in both kinds of works, and proposes an online under-maintain version of collected work in this direction.


*2023-02-26*

#### [On Structural Explanation of Bias in Graph Neural Networks](https://dl.acm.org/doi/10.1145/3534678.3539319)

*Yushun Dong, Song Wang, Yu Wang, Tyler Derr, Jundong Li*

*KDD 2022*

This paper studies the structural explanation of bias in node prediction for GNN. It formulates the problem as to identify a set of edges in the computation graph which mainly accounts for the bias of a given node (binary) prediction, as well as a set of edges which accounts for the fairness. It measures the extent of bias/fairness given by the set of edges as distances (specifically, Wasserstein distance) between the probabilistic distributions. The core idea is the edges in the bias edge set maximally account for the node-level bias, while those in the fairness edge set maximally alleviates the node-level bias.


*2023-02-23*

#### [MA-GCL: Model Augmentation Tricks for Graph Contrastive Learning](https://arxiv.org/pdf/2212.07035.pdf)

*Xumeng Gong, Cheng Yang, Chuan Shi*

*AAAI 2023*

This paper proposes data augmentation methods for graph contrastive learning. The motivation is, unlike the assumption in e.g., vision area that typical data augmentation methods cannot change the label of training data, graph data need more precise methods for augmentation. Therefore, this paper proposes three tricks for doing that, including asymmetrical, random, and shuffling tricks. It directly learns the transformation units in the GNN model for each layer, by designing a proper loss function implementing each of these tricks.


*2023-02-22*

#### [Interpretable Graph Convolutional Neural Networks for Inference on Noisy Knowledge Graphs](https://arxiv.org/abs/1812.00279)

*Daniel Neil, Joss Briody, Alix Lacoste, Aaron Sim, Páidí Creed, Amir Saffari*

*Arxiv 2018*

This paper introduces a graph convolutional neural network (GCNN) model for link prediction over noisy knowledge graphs. The model is based on a regularized attention mechanism. A visualization method is also proposed for interpreting how the learned representation can help dataset denoising.


*2023-02-21*

#### [GNNExplainer: Generating Explanations for Graph Neural Networks](https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html)

*Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, Jure Leskovec*

*NeurIPs 2019*

This paper proposes GNNExplainer, a model-agnostic approach to generate explanations for any GNN model. Given an input instance, it returns a small subgraph of the input graph together with a subset of node features that are most influential for the prediction.


*2023-02-20*

#### [Explainability Methods for Graph Convolutional Neural Networks](https://ieeexplore.ieee.org/document/8954227/)

*Phillip E. Pope, Soheil Kolouri, Mohammad Rostami, Charles E. Martin, Heiko Hoffmann*

*CVPR 2019*

This paper investigates three methods of explainability of GNN, namely, contrastive gradient-based (CG) salience maps, class activation mapping (CAM), and excitation backpropagation (EB). Basically, the CG method proposes a heat-map based on the importance of model output w.r.t. the input. The CAM method identifies the class-specific features at the last convolutional layer. The EB method generates heap-maps based on the back propagation while ignoring the nonlinearity. It also investigates several variants based on the three main methods.


*2023-02-19*

#### [Explainability in Graph Neural Networks: A Taxonomic Survey](https://arxiv.org/abs/2012.15445)

*Hao Yuan, Haiyang Yu, Shurui Gui, Shuiwang Ji*

*Arxiv 2020*

This paper investigates existing research efforts of the explainability of GNN. It proposes a taxonomy which categorizes explanations of GNN into instance-level and model-level, and further divides the instance-level explanations into 4 subsets. It analyzes each category with an overview of methods and examples.


*2023-02-18*

#### [Explainability techniques for graph convolutional networks](https://arxiv.org/pdf/1905.13686.pdf)

*Federico Baldassarre, Hossein Azizpour*

*Arxiv 2019*

This short paper proposes a very initial study of explainable graph neural network. It explores three back-propagation-based analyzing methods to explain the output of a GNN. Experiments are conducted over a toy dataset simulating disease infection chains, and another chemical solubility graph prediction task.


*2023-02-16*

#### [TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation](https://dl.acm.org/doi/10.1145/3397271.3401319)

*Feng Yu, Yanqiao Zhu, Qiang Liu, Shu Wu, Liang Wang, Tieniu Tan*

*SIGIR 2020*

This paper proposes a gated graph neural network model for session-based recommendation. It computes an embedding for each item over the session graph as well as an embedding for the session. Beyond that, it also implements an attention mechanism between the items and the sessions, to measure the importance of different items for the session.


*2023-02-15*

#### [Heterogeneous Graph Transformer for Graph-to-Sequence Learning](https://doi.org/10.18653/v1/2020.acl-main.640)

*Shaowei Yao, Tianming Wang, Xiaojun Wan*

*ACL 2020*

This paper proposes a graph transformer architecture to encode the graph structure and applies it to downstream sequence generation tasks (i.e., graph to sequence). Given an input graph, it firstly splits and aggregates it to be four different subgraphs with splitting the original nodes to finer-grained subnodes. Then it uses stacked heterogeneous graph transformer to encode the subgraphs with layer aggregation. The model is evaluated over the AMT-to-text and NMT tasks, and demonstrates SOTA performance on them.


*2023-02-14*

#### [Graph Structure Learning for Robust Graph Neural Networks](https://dl.acm.org/doi/10.1145/3394486.3403049)

*Wei Jin, Yao Ma, Xiaorui Liu, Xianfeng Tang, Suhang Wang, Jiliang Tang*

*KDD 2020*

This paper proposes a GNN framework to "clean" a perturbed graph to achieve better robustness when faced with adversarial attack. The motivation is that real-world graphs are usually low-rank and sparse. Besides, the features of two adjacent nodes tend to be similar. In the proposed model Pro-GNN, it incorporates these targets into the loss function, and uses a forward-backward method to iteratively optimize the model.


*2023-02-13*

#### [Relational Graph Attention Network for Aspect-based Sentiment Analysis](https://doi.org/10.18653/v1/2020.acl-main.295)

*Kai Wang, Weizhou Shen, Yunyi Yang, Xiaojun Quan, Rui Wang*

*ACL 2020*

This paper proposes a GAT-based model for fine-grained (aspect-based) text sentiment analysis. It firstly constructs a dependency tree based on direct and indirect dependencies between tokens for each aspect. Then it applies the GAT mechanism with respect to each relation, and aggregates them as overall message-passing weights. The model contains a set of $K$ attention heads and $M$ relational heads.


*2023-02-12*

#### [Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ)

*Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio*

*ICLR 2018*

This is a milestone paper that firstly introduce the attention mechanism to graph neural network. It implements the graph attention mechanism as a feed-forward network on each layer to learn the weights of message passing between neighboring nodes. It also incorporates the multi-head mechanism to capture different features over the same edge.

*2023-02-10*

#### [Modeling Relational Data with Graph Convolutional Networks](https://link.springer.com/chapter/10.1007/978-3-319-93417-4_38)

*Michael Sejr Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling*

*ESWC 2018*

This paper proposes a graph convolutional network model for relational data named R-GCN. It optimizes the structure of basic GCN model with a normalized aggregation over each relation for massage passing. The evaluation over the link prediction and entity classificaation tasks demonstrates the effectiveness of R-GCN, especially its relation-normalized encoder.

*2023-02-08*

#### [Inductive Relation Prediction by Subgraph Reasoning](http://proceedings.mlr.press/v119/teru20a.html)

*Komal K. Teru, Etienne G. Denis, William L. Hamilton*

*ICML 2020*

This paper proposes a GNN-based inductive relation prediction model named Grail. It assumes the underlying relation between two nodes (entities) can be represented by the local subgraph structure (paths between the nodes), thus can be applied under the inductive setting. It adopts one-hot encoding vectors as the node feature, and implements message passing over the radius-bounded subgraphs. The experimental result shows it achieves SOTA performance among existing link prediction methods.

*2022-11-16*

#### [Contrastive Knowledge Graph Error Detection](https://doi.org/10.1145/3511808.3557264)

*Qinggang Zhang, Junnan Dong, Keyu Duan, Xiao Huang, Yezi Liu, Linchuan Xu*

*CIKM 2022*

Unlike existing knowledge graph error detection methods which generally rely on negative sampling, this paper introduces a contrastive learning model by creating different hyper-views of the KG, and regards each relational triple as a node. The optimize target includes the consistency of triple representations among the multi-views and the self-consistency within each triple. In this paper, the two views of the KG are defined by two link patterns, i.e., two triples sharing head entity, or sharing tail entity.

*2022-11-15*

#### [Taxonomy-Enhanced Graph Neural Networks](https://doi.org/10.1145/3511808.3557467)

*Lingjun Xu, Shiyin Zhang, Guojie Song, Junshan Wang, Tianshu Wu, Guojun Liu*

*CIKM 2022*

This paper proposes to incorporate the external taxonomy knowledge into the GNN learning process of nodes embeddings. For the taxonomy, instead of using a vector, it firstly maps each category to a Gaussian distribution, and calculates the mutual information between them. In the downstream GNN model, these categories are used to characterize the similarity of node pairs. The context of each node is represented by the mean vector of categories of neighboring nodes.

*2022-11-14*

#### [Large-scale Entity Alignment via Knowledge Graph Merging, Partitioning and Embedding](https://doi.org/10.1145/3511808.3557374)

*Kexuan Xin, Zequn Sun, Wen Hua, Wei Hu, Jianfeng Qu, Xiaofang Zhou*

*CIKM 2022*

This paper proposes three strategies for scalable GNN-based entity alignment without losing too much structural information. It follows the pipeline of partitioning, merging the knowledge graph and generating the alignment. In the partitioning process, it identifies a set of landmark entities to connect different subgraphs. To reduce the structure loss, it also applies an entity reconstruction mechanism to incorporate information from its neighborhood. Besides, it implements entity search in a fused unified space of multiple subgraphs.

*2022-11-13*

#### [Incorporating Peer Reviews and Rebuttal Counter-Arguments for Meta-Review Generation](https://doi.org/10.1145/3511808.3557360)

*Po-Cheng Wu, An-Zi Yen, Hen-Hsen Huang, Hsin-Hsi Chen*

*CIKM 2022*

This paper investigates the problem of meta-review generation based on peer reviews and the authors' rebuttal. The authors collect a dataset of submissions, reviews and rebuttal responses from ICLR 2017--2021. To solve the problem, they firstly extract all the argumentative discourse units (ADUs) and three level of relations (i.e., intra-document, intra-discussion, and inter-discussion relations) between the ADUs. Then they construct a content (text) encoder model with a graph attention network, and aggregate them to generate the meta-review. The overall model is trained in a seq2seq manner.

*2022-11-11*

#### [Reinforced Continual Learning for Graphs](https://doi.org/10.1145/3511808.3557427)

*Appan Rakaraddi, Siew-Kei Lam, Mahardhika Pratama, Marcus de Carvalho*

*CIKM 2022*

This paper proposes a graph continual learning model for the task of node classification. It consists of a reinforcement learning based controller to manage adding and deleting node features, and a GNN as child network to deal with the tasks. It supports both task-incremental and class-incremental settings for node classification.

*2022-11-01*

#### [Exploring Edge Disentanglement for Node Classification](https://dl.acm.org/doi/10.1145/3485447.3511929)

*Tianxiang Zhao, Xiang Zhang, Suhang Wang*

*TheWebConf 2022*

This paper proposes a GNN model to identify different edge attributives (edge disentanglement) for node classification task. It implements a set of disentangled channels to capture different edge attributes, and provides three self-supervision signals to learn edge disentanglement.

*2022-10-31*

#### [Learning and Evaluating Graph Neural Network Explanations based on Counterfactual and Factual Reasoning](https://doi.org/10.1145/3485447.3511948)

*Juntao Tan, Shijie Geng, Zuohui Fu, Yingqiang Ge, Shuyuan Xu, Yunqi Li, Yongfeng Zhang*

*TheWebConf 2022*

This paper proposes a GNN model to generate subgraphs as explanation for the graph classification task based on factual and counterfactual reasoning. It proposes two objectives that a good explanation should be (1) sufficient and necessary (related to factual and counterfactual reasoning), and (2) simple (driven by the Occam’s Razor Principle).

*2022-10-24*

#### [Rethinking Graph Convolutional Networks in Knowledge Graph Completion](https://doi.org/10.1145/3485447.3511923)

*Zhanqiu Zhang, Jie Wang, Jieping Ye, Feng Wu*

*TheWebConf 2022*

This paper proposes an idea that the graph structure modeling is unimportant for GCN-based KGC models. Instead, the ability to distinguish different entities and the transformations for entity embeddings account for the performance improvements. To prove this, firstly, it randomly changes the adjacency tensors in message passing and surprisingly gets similar results. Besides, removing the self-loop information also results in similar performance. Based on that, this paper also proposes a KG embedding model which applies linear transformation to entity representations.

*2022-10-22*

#### [Trustworthy Knowledge Graph Completion Based on Multi-sourced Noisy Data](https://doi.org/10.1145/3485447.3511938)

*Jiacheng Huang, Yao Zhao, Wei Hu, Zhen Ning, Qijin Chen, Xiaoxia Qiu, Chengfu Huo, Weijun Ren*

*TheWebConf 2022*

This paper works on open knowledge graph completion based on noisy data. It firstly proposes a holistic fact scoring function for both relational facts and literal facts (triples). Then it proposes a neural network model to align the heterogeneous values from different facts. It also implements a semi-supervised inference model to predict the trustworthiness of the claims.

*2022-10-18*

#### [Knowledge Graph Reasoning with Relational Digraph](https://doi.org/10.1145/3485447.3512008)

*Yongqi Zhang, Quanming Yao*

*TheWebConf 2022*

This paper proposes a knowledge graph reasoning network named RED-GNN to answer the queries in the form of $\langle$ subject entity, relation, ? $\rangle$. It introduces a relational directed graph (r-digraph) to capture the entity relation connections in the KG, and uses dynamic programming to recursively encode multiple r-digraphs with shared edges. It achieves relatively good performance among existing GNN models, and provides their codes.

*2022-10-17*

#### [KoMen: Domain Knowledge Guided Interaction Recommendation for Emerging Scenarios](https://dl.acm.org/doi/10.1145/3485447.3512177)

*Yiqing Xie, Zhen Wang, Carl Yang, Yaliang Li, Bolin Ding, Hongbo Deng, Jiawei Han*

*TheWebConf 2022*

This paper studies a problem of user-user interaction recommendation in emerging scenarios, which is formulated as a few-shot link prediction task over a multiplex graph. To solve the problem, it proposes a model containing two levels of attention mechanism. Each of the several experts is trained on one type of edges (i.e., learns the attention over each type of edges), while different scenarios choose different combination of experts (i.e., learn the attention over experts).

*2022-09-28*

#### [INDIGO: GNN-Based Inductive Knowledge Graph Completion Using Pair-Wise Encoding](https://proceedings.neurips.cc/paper/2021/hash/0fd600c953cde8121262e322ef09f70e-Abstract.html)

*Shuwen Liu, Bernardo Cuenca Grau, Ian Horrocks, Egor V. Kostylev*

*NeurIPS 2021*

This paper focuses on the task of knowledge graph completion. While existing GNN models for knowledge graph completion generally use transductive features of the KG and cannot handle unseen entities during the test phase, this paper improves the GCN model to address the problem by changing the encoder and decoder components. Each node in this GNN model is designed to represent a pair of entities in the original KG. Besides, its transparent nature allows the predicted triples to be read out directly without the need of an additional predicting layer.

*2022-04-25*

#### [Graph Neural Networks: Taxonomy, Advances, and Trends](https://dl.acm.org/doi/10.1145/3495161)

*Yu Zhou, Haixia Zheng, Xin Huang, Shufeng Hao, Dengao Li, Jumin Zhao*

*ACM TIST 2022*

It's a comprehensive survey paper that introduces the GNN roadmap on four levels, and worth reading for several times. (1) The fundamental architectures present the basic GNN modules and operations such as graph attention and pooling. (2) The extended architectures and applications discuss various tasks such as pre-training framework, reinforcement learning and application scenarios of GNN modules such as NLP, recommendation systems. (3) The implementations and evaluations introduce commonly used tools and benchmark datasets. (4) The future research directions provide potential directions includes highly scalable/robust/interpretable GNNs, and GNNs going beyond WL test.

*2022-04-22*

#### [Graph Neural Networks for Graphs with Heterophily: A Survey](https://arxiv.org/abs/2202.07082)

*Authors: Xin Zheng, Yixin Liu, Shirui Pan, Miao Zhang, Di Jin, Philip S. Yu*

*Arxiv 2022*

A survey about GNNs for heterophilic graphs. Generally, most existing GNN models depends on the homophily assumption, i.e., the nodes with same or similar labels are more likely to be linked than those with different labels, such as the citation network. But there are also many real-world graphs do not obey this rule, e.g., online transaction networks, dating networks. This paper surveys existing research efforts for GNNs over heterophilic graphs in two folds: (1) non-local neighbor extension, which try to obtain information from similar but distant nodes (2) GNN architecture refinement, which try to modify the information aggregation methods on the model side. This paper has also suggested 4 future work directions: (1) interpretability and robustness, (2) scalable heterophilic GNNs (how to implement large models, how to train, how to sample batches), (3) heterophily and over-smoothing, (4) comprehensive benchmark and metrics (real-world, larger graphs).
