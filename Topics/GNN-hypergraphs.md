









*2025-08-06*

#### [Structure Is All You Need: Structural Representation Learning on Hyper-Relational Knowledge Graphs](https://openreview.net/forum?id=2tH2vexW1Z)

*Jaejun Lee, Joyce Jiyoung Whang*

*ICML 2025*

The paper proposes a structure-driven representation learning method for hyper-relational knowledge graphs. It leverages the HKG’s structure through a structure-driven initializer and attentive neural message passing.


*2025-08-02*

#### [HyCubE: Efficient Knowledge Hypergraph 3D Circular Convolutional Embedding](https://ieeexplore.ieee.org/document/10845179)

*李钊、王鑫、赵军、郭文斌、李建新*

*TKDE 2025*

This paper proposes an end-to-end knowledge hypergraph embedding model, which uses a *3D circular convolutional neural network* and the *alternate mask stack* strategy to enhance the interaction and extraction of feature information.


*2025-02-20*

#### [HySAE: An Efficient Semantic-Enhanced Representation Learning Model for Knowledge Hypergraph Link Prediction](https://openreview.net/forum?id=OLLYLTb8FC#discussion)

*Zhao Li, Xin Wang, Zhao Jun, Feng Feng, Zirui Chen, Jianxin Li*

*WWW 2025*

This paper proposes a knowledge hypergraph representation learning model, aiming at achieving a trade-off between effectiveness and efficiency. In particular, it builds a semantic-enhanced 3D scalable end-to-end embedding architecture to capture n-ary relations with fewer parameters. Also, it applies a position-aware entity role semantic embedding and enhanced semantic learning strategies to further improve the effectiveness and scalability of the proposed approach.


*2024-10-15*

#### [FastHGNN: A New Sampling Technique for Learning with Hypergraph Neural Networks](https://dl.acm.org/doi/full/10.1145/3663670)

*Fengcheng Lu, Michael Kwok-Po Ng*

*ACM Transactions on Knowledge Discovery from Data 2024*

This paper proposes a sampling technique for learning with hypergraph neural networks. The core idea is to design a layer-wise sampling scheme for nodes and hyperedges to approximate the original hypergraph convolution. Specifically, it rewrites hypergraph convolution in the form of double integral and leverages Monte Carlo to achieve a discrete and consistent estimator. In addition, it applies importance sampling and derives feasible probability mass functions for both nodes and hyperedges in consideration of variance reduction.


*2024-10-13*

#### [HJE: Joint Convolutional Representation Learning for Knowledge Hypergraph Completion](https://ieeexplore.ieee.org/document/10436025)

*Zhao Li, Chenxu Wang, Xin Wang, Zirui Chen, Jianxin Li*

*TKDE 2024*

This paper proposes an embedding-based hypergraph link prediction approach, which achieves embedding of position information into all convolutional paths by constructing a unified learnable position embedding matrix for each entity position in the knowledge tuple. It is commendable that it not only realizes the complete embedding of entity position information but also avoids constructing redundant convolutional kernels to reduce the complexity of the model.


*2024-10-12*

#### [Knowledge Hypergraph Embedding Meets Relational Algebra](https://www.jmlr.org/papers/v24/22-063.html)

*Bahare Fatemi, Perouz Taslakian, David Vazquez, David Poole*

*JMLR 2023*

This paper proposes a simple embedding-based model called Relational Algebra Embedding (ReAlE) that performs link prediction in knowledge hypergraphs. By exploring the space between relational algebra foundations and machine learning techniques for knowledge completion, it shows that ReAlE is fully expressive and can represent the relational algebra operations of renaming, projection, set union, selection, and set difference.


*2024-10-11*

#### [Generalizing Tensor Decomposition for N-ary Relational Knowledge Bases](https://dl.acm.org/doi/10.1145/3366423.3380188)

*Yu Liu, Quanming Yao, Yong Li*

*WWW 2020*

This paper focuses on link prediction over n-ary relation knowledge graphs for completion. To generalize tensor decomposition for n-ary relational KBs, it proposes GETD as a generalized model based on Tucker decomposition and Tensor Ring decomposition. The existing negative sampling technique is also generalized to the n-ary case for GETD.


*2024-09-29*

#### [Beyond Link Prediction: Predicting Hyperlinks in Adjacency Space](https://aaai.org/papers/11780-beyond-link-prediction-predicting-hyperlinks-in-adjacency-space/)

*Muhan Zhang, Zhicheng Cui, Shali Jiang, Yixin Chen*

*AAAI 2018*

This paper works on the task of hyperlink prediction over hypernetworks. It proposes an algorithm Coordinated Matrix Minimization (CMM), which alternately performs non-negative matrix factorization and least square matching in the vertex adjacency space of the hypernetwork. Then it evaluates the model over two tasks: predicting Chinese food recipes, and finding missing reactions of metabolic networks.


*2024-09-28*

#### [Message Passing for Hyper-Relational Knowledge Graphs](https://aclanthology.org/2020.emnlp-main.596.pdf)

*Mikhail Galkin, Priyansh Trivedi, Gaurav Maheshwari, Ricardo Usbeck, Jens Lehmann*

*EMNLP 2020*

This paper proposes a GNN-based model for link prediction over hyper-relational KGs. Specifically, each fact in a hyper-relational KG is formed by a relation, a main triple as $\langle s, p, o \rangle$, plus unlimited number of qualifiers as key-value pairs to describe the fact, or to help distinguish facts with the same main triple. For example, a person may get different degrees from distinct universities, which are supposed to be identified by the attached key-value pairs, so link prediction in this case should be able to return different $o$ (e.g., representing the universities) in the main triple with the same $s$ and $p$ (e.g., representing the person and "studies at" relation) with different sets of attached key-value pairs.


*2024-09-09*

#### [A Survey on Hyperlink Prediction](https://ieeexplore.ieee.org/document/10163497)

*Can Chen, Yang-Yu Liu*

*TNNLS 2023*

This paper reviews existing works on hypergraph link prediction/hyperlink prediction by adopting a classical taxonomy from link prediction that classifies existing methods into four categories: similarity-based, probability-based, matrix optimization-based, and deep learning-based methods. To compare the performance of methods from different categories, it performs a benchmark study on various hypergraph applications using representative methods from each category.


*2024-08-20*

#### [HypeBoy: Generative Self-Supervised Representation Learning on Hypergraphs](https://arxiv.org/abs/2404.00638)

*Sunwoo Kim, Shinhwan Kang, Fanchen Bu, Soo Yong Lee, Jaemin Yoo, Kijung Shin*

*ICLR 2024*

This paper introduces a generative self-supervised learning task on hypergraphs named hyperedge filling, and its connection to node classification. The task of hyperedge filling is sometimes also called hypergraph link prediction in other works. Then it proposes an embedding-based model for this task and applies it on node classification.


*2024-08-18*

#### [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://arxiv.org/abs/2106.13264)

*Eli Chien, Chao Pan, Jianhao Peng, Olgica Milenkovic*

*ICLR 2022*

Unlike existing approaches that generally use clique expansion to decompose hyperedges in the hypergraph, this paper proposes a new framework that implements hypergraph neural network layers as compositions of two multiset functions that can be efficiently learned for each task and each dataset.


*2024-08-17*

#### [Hypergraph modeling and hypergraph multi-view attention neural network for link prediction](https://dl.acm.org/doi/10.1016/j.patcog.2024.110292)

*Lang Chai, Lilan Tu, Xianjia Wang, Qingqing Su*

*Pattern Recognition 2024*

This paper proposes an approach for inductive link prediction without requiring node attribute information. It uses a network structure linear representation to model hypergraph for general networks without node attributes.


*2024-08-16*

#### [Hypergraph Structure Learning for Hypergraph Neural Networks](https://www.ijcai.org/proceedings/2022/267)

*Derun Cai, Moxian Song, Chenxi Sun, Baofeng Zhang, Shenda Hong, Hongyan Li*

*IJCAI 2022*

This paper proposes an end-to-end approach to optimize the hypergraph structure and the node/edge representations simultaneously with a joint loss. Specifically, it adopts a two-stage sampling process: (1) hyperedge sampling for pruning redundant hyperedges, and (2) incident node sampling for pruning irrelevant incident nodes and discovering potential implicit connections. The consistency between the optimized structure and the original structure is maintained by an intra-hyperedge contrastive learning module. The proposed model is evaluated via transductive node classification.


*2024-08-15*

#### [Hypergraph Convolution and Hypergraph Attention](https://arxiv.org/abs/1901.08150)

*Song Bai, Feihu Zhang, Philip H.S. Torr*

*Pattern Recognition 2020*

This paper proposes the convolution and attention operators over hypergraphs, and establishes the similar relationship between the two operators as on binary graphs. The operators are built based on the incidence matrix, weight matrices, and vector representations of vertexes and hyperedges.


*2024-08-14*

#### [Totally Dynamic Hypergraph Neural Network](https://www.ijcai.org/proceedings/2023/275)

*Peng Zhou, Zongqian Wu, Xiangxiang Zeng, Guoqiu Wen, Junbo Ma, Xiaofeng Zhu*

*IJCAI 2023*

Dynamic hypergraph neural networks aim to update the hypergraph structure in the learning process. While existing works typically cannot adjust the hyperedge number, this paper proposes a model that is able to adjust the hyperedge number for optimizing the hypergraph structure.


*2024-08-11*

#### [A Survey on Hypergraph Neural Networks: An In-Depth and Step-By-Step Guide](https://arxiv.org/abs/2404.01039)

*Sunwoo Kim, Soo Yong Lee, Yue Gao, Alessia Antelmi, Mirko Polato, Kijung Shin*

*KDD 2024*

This paper reviews existing works on hypergraph neural network (HNN) models by their 4 design components: (1) input features, (2) input structures, (3) message-passing schemes, and (4) training strategies. It examines how HNNs address and learn higher-order interactions with each of their components, and then introduces recent applications of HNNs in recommendation, bioinformatics and medical science, time series analysis, and computer vision.


*2024-08-08*

#### [HyConvE: A Novel Embedding Model for Knowledge Hypergraph Link Prediction with Convolutional Neural Networks](https://dl.acm.org/doi/10.1145/3543507.3583256)

*Chenxu Wang, Xin Wang, Zhao Li, Zirui Chen, Jianxin Li*

*WWW 2023*

This paper proposes an embedding-based neural network model for link prediction over hypergraphs. Specifically, it extends the idea of convolutional neural network, using a 3D convolution to capture the interactions of entities and relations to extract explicit and implicit knowledge in each n-ary fact without compromising the translation property. Additionally, appropriate relation and position-aware filters are utilized sequentially to perform two-dimensional convolution operations to capture the intrinsic patterns and position information in each n-ary fact, respectively.


*2024-08-07*

#### [NHP: Neural Hypergraph Link Prediction](https://dl.acm.org/doi/10.1145/3340531.3411870)

*Naganand Yadati, Vikram Nitin, Madhav Nimishakavi, Prateek Yadav, Anand Louis, Partha Talukdar*

*CIKM 2020*

This paper proposes a hypergraph neural network model by extending the GCN model on binary graphs to hypergraphs. It includes two variants for undirected and directed hypergraph link prediction, respectively.


*2024-07-01*

#### [Hypergraph Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/4235)

*Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao*

*AAAI 2019*

This paper proposes HGNN as a hypergraph neural network for node classification task, and evaluates it on citation network classification and visual object recognition tasks.


*2024-06-29*

#### [HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs](https://papers.nips.cc/paper_files/paper/2019/hash/1efa39bcaec6f3900149160693694536-Abstract.html)

*Naganand Yadati, Madhav Nimishakavi, Prateek Yadav, Vikram Nitin, Anand Louis, Partha Talukdar*

*NeurIPS 2019*

This paper extends the GCN framework to hypergraphs as HyperGCN. It is evaluated over citation/coauthoring hypergraph cora/citeceer, though the model ignores the hyperedge labels.


*2024-06-28*

#### [Neural Message Passing for Multi-Relational Ordered and Recursive Hypergraphs](https://dl.acm.org/doi/pdf/10.5555/3495724.3496000)

*Naganand Yadati*

*NeurIPS 2020*

This paper proposes a message passing neural network framework named Generalized-MPNN for multi-relational hypergraphs, and an extension MPNN-R (MPNN-Recursive) to handle recursively-structured data. Besides, it proposes three datasets for the inductive hyper-link prediction task.

