







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
