





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

