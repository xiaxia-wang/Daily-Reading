















*2024-09-23*

#### [Advancing Molecule Invariant Representation via Privileged Substructure Identification](https://dl.acm.org/doi/abs/10.1145/3637528.3671886)

*Ruijia Wang, Haoran Dai, Cheng Yang, Le Song, Chuan Shi*

*KDD 2024*

This paper proposes a specific framework for molecule invariant representation. It first formalizes molecule invariant learning based on privileged substructure identification and introduces substructure invariance constraint. Building on that, it theoretically establishes two criteria for environment splits conducive to molecule invariant learning. Inspired by the criteria, a dual-head graph neural network is developed with a shared identifier that identifies privileged substructures, while environment and task heads generates predictions based on variant and privileged substructures.


*2024-09-13*

#### [Do We Really Need Graph Convolution During Training? Light Post-Training Graph-ODE for Efficient Recommendation](https://arxiv.org/abs/2407.18910)

*Weizhi Zhang, Liangwei Yang, Zihe Song, Henry Peng Zou, Ke Xu, Liancheng Fang, Philip S. Yu*

*CIKM 2024*

This paper claims a new idea that the graph convolution layers are less effective in the training stage than the testing stage for recommendation task, and then proposes a framework that directly optimizes the embedding alignment (for the paired nodes) in the training process, while only applying the graph convolution layers in the testing process. The results indicate that by doing so, the training time could be largely reduced without sacrificing the overall performance.


*2024-08-21*

#### [PolygonGNN: Representation Learning for Polygonal Geometries with Heterogeneous Visibility Graph](https://arxiv.org/abs/2407.00742)

*Dazhou Yu, Yuntong Hu, Yun Li, Liang Zhao*

*KDD 2024*

To effectively capture and utilize the representation of multipolygons, this paper first proposes a unified representation framework as heterogeneous visibility graph for single and multipolygons. Then, to enhance computational efficiency and minimize graph redundancy, it implements a heterogeneous spanning tree sampling method. Additionally, it devises a rotation-translation invariant geometric representation to ensure broader applicability across scenarios. Finally, it proposes a model Multipolygon-GNN to leverage the spatial and semantic heterogeneity inherent in the visibility graph.


*2024-08-04*

#### [Unifying Graph Retrieval and Prompt Tuning for Graph-Grounded Text Classification](https://dl.acm.org/doi/10.1145/3626772.3657934)

*Le Dai, Yu Yin, Enhong Chen, Hui Xiong*

*SIGIR 2024 Short*

For the document classification task where documents are represented as nodes in a graph, unlike existing approaches typically applying GNN for classification by using the document as context, this paper proposes a prompt tuning based approach that searches on the graph and generates node-based and path-based contexts for document classification.


*2024-08-03*

#### [Graph Reasoning Enhanced Language Models for Text-to-SQL](https://dl.acm.org/doi/10.1145/3626772.3657961)

*Zheng Gong, Ying Sun*

*SIGIR 2024 Short*

Existing Text-to-SQL approaches typically introduce some useful multi-hop structures manually and then incorporate them into graph neural networks by stacking multiple layers, which (1) ignore the difficult-to-identify but meaningful semantics embedded in the multi-hop reasoning path, and (2) are limited by the expressive capability of GNN to capture long-range dependencies among the heterogeneous graph. To address these shortcomings, this paper proposes a graph reasoning enhanced language model, which applies structure encoding to capture the dependencies between node pairs, encompassing one-hop, multi-hop and distance information, subsequently enriched through self-attention for enhanced representational power over GNNs.


*2024-07-23*

#### [Graph Neural Networks for Brain Graph Learning: A Survey](https://arxiv.org/abs/2406.02594)

*Xuexiong Luo, Jia Wu, Jian Yang, Shan Xue, Amin Beheshti, Quan Z. Sheng, David McAlpine, Paul Sowman, Alexis Giral, Philip S. Yu*

*IJCAI 2024 Survey*

A novel approach in neuroimaging has emerged that models the human brain as a graph-structured pattern, with different brain regions represented as nodes and functional relationships among these regions as edges. This paper reviews GNN-based approaches for learning brain graph representations. It first introduces the process of brain graph modeling based on common neuroimaging data, and then systematically categorizes current works into static-, dynamic-, and multi-modal brain graphs with their targeted research problems. Also, it summarizes existing available datasets and resources.


*2024-07-07*

#### [Leveraging Pedagogical Theories to Understand Student Learning Process with Graph-based Reasonable Knowledge Tracing](https://arxiv.org/abs/2406.12896)

*Jiajun Cui, Hong Qian, Bo Jiang, Wei Zhang*

*KDD 2024*

Knowledge tracing (KT) focuses on predicting students' performance on given questions to trace their evolving knowledge. This paper utilizes the educational theory and divides the KT process into 3 stages, (1) knowledge retrieval, (2) memory strengthening, and (3) knowledge learning/forgetting. By applying a GNN model over a knowledge concept graph for predicting (binary) answers for anchor questions, the three-stage modeling is recurrent along the student response sequence. After learning/forgetting knowledge in the third stage, the updated knowledge memory is prepared for the first stage to answer the next question.


*2024-01-19*

#### [Unified Pretraining for Recommendation via Task Hypergraphs](https://arxiv.org/abs/2310.13286)

*Mingdai Yang, Zhiwei Liu, Liangwei Yang, Xiaolong Liu, Chen Wang, Hao Peng, Philip S. Yu*

*WSDM 2024*

This paper proposes a multitask pretraining model for recommendation. It introduces task hypergraphs for different pretext recommendation tasks, and applies a transitional attention layer to discriminatively learn from each task. The trainable parameters are only the initial user and item embeddings, while the hypergraph embeddings are conducted in a standard way of matrix factorization.


*2023-12-16*

#### [KGTrust: Evaluating Trustworthiness of SIoT via Knowledge Enhanced Graph Neural Networks](https://dl.acm.org/doi/10.1145/3543507.3583549)

*Zhizhi Yu, Di Jin, Cuiying Huo, Zhiqiang Wang, Xiulong Liu, Heng Qi, Jia Wu, Lingfei Wu*

*WWW 2023*

Social Internet of Things (SIoT) is an emerging paradigm that injects the notion of social networking into smart objects, i.e., things. To better utilize the various information provided in SIoT, this paper proposes a knowledge-enhanced GNN model for trust evaluation. It first extracts useful knowledge from users’ comment behaviors and external structured triples related to object descriptions, and further introduces a discriminative convolutional layer that utilizes heterogeneous graph structure, node semantics, and augmented trust relationships to learn node embeddings from the perspective of a user as a trustor or a trustee.


*2023-12-15*

#### [GATrust: A Multi-Aspect Graph Attention Network Model for Trust Assessment in OSNs](https://doi.org/10.1109/TKDE.2022.3174044)

*Nan Jiang, Jie Wen, Jin Li, Ximeng Liu, Di Jin*

*IEEE TKDE 2023*

To capture the pairwise trustworthiness in online social networks, existing approaches typically use GCNs that overlooked the various user features. This paper proposes a model in which each layer contains a GAT module and two GCN module for aggregating the users' context-specific information, network topological structure information, and locally-generated social trust relationships.


*2023-09-29*

#### [Linkless Link Prediction via Relational Distillation](https://proceedings.mlr.press/v202/guo23f.html)

*Zhichun Guo, William Shiao, Shichang Zhang, Yozen Liu, Nitesh V. Chawla, Neil Shah, Tong Zhao*

*ICML 2023*

This paper explores to distill link prediction-relevant knowledge from GNN models to MLPs. Specifically, it applies teacher-student models to (1) logit-based matching of predicted link existence probabilities, and (2) representation-based matching of the generated latent node representations.


*2023-07-29*

#### [Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs](https://dl.acm.org/doi/10.1145/3539597.3570465)

*Linhao Luo, Gholamreza Haffari, Shirui Pan*

*WSDM 2023*

This paper proposes a Graph Sequential Neural ODE Process model for link prediction over dynamic graphs. Specifically, it uses a dynamic GNN encoder and a sequential ODE aggregator to model the dynamic-changing stochastic process. A neural ordinary differential equation (ODE) is a continuous-time model that defines the derivative of the hidden state with a neural network.


*2023-07-26*

#### [Search Behavior Prediction: A Hypergraph Perspective](https://dl.acm.org/doi/10.1145/3539597.3570403)

*Yan Han, Edward W. Huang, Wenqing Zheng, Nikhil Rao, Zhangyang Wang, Karthik Subbian*

*WSDM 2023*

Typical user-item correlations in e-commerce are modeled as bipartite graph for downstreaming tasks. To address the challenges of (1) long-tail distribution, and (2) disassortative mixing (i.e., infrequently queries are more likely to link to popular items), this paper models the user-item correlations as a hypergraph, where all items appear in the same customer session are treated as a single hypernode. An attention-based model is proposed to use information from both the original query-item edges and item-item hyperedges.


*2023-07-25*

#### [Effective Seed-Guided Topic Discovery by Integrating Multiple Types of Contexts](https://dl.acm.org/doi/10.1145/3539597.3570475)

*Yu Zhang, Yunyi Zhang, Martin Michalski, Yucheng Jiang, Yu Meng, Jiawei Han*

*WSDM 2023*

Seed-guided topic discovery aims to find coherent topics based on the user-provided seed words. To achieve this, this paper integrates three types of context signals to model the correlation between seed words and topic-indicative terms, i.e., word embeddings learned from local contexts, pre-trained language model representations obtained from general-domain training, and topic-indicative sentences retrieved based on seed information.


*2023-07-18*

#### [Knowledge Graph Completion with Counterfactual Augmentation](https://dl.acm.org/doi/10.1145/3543507.3583401)

*Heng Chang, Jie Cai, Jia Li*

*WWW 2023*

This paper proposes an instantiation of a casual model for KG based on counterfactual facts estimation. It further incorporates it into existing KGC models and achieves performance improvement.


*2023-06-28*

#### [Explainable Conversational Question Answering over Heterogeneous Sources via Iterative Graph Neural Networks](https://arxiv.org/abs/2305.01548)

*Philipp Christmann, Rishiraj Saha Roy, Gerhard Weikum*

*SIGIR 2023*

This paper proposes a pipeline for conversational question answering based on multiple information sources, and is able to provide explanatory evidences for the user. It contains 3 major stages, i.e., question understanding stage receives the conversational history as input and generates structural representation (SR) of the information need (using BART), evidence retrieval stage retrieves evidence form heterogeneous information sources, and heterogeneous answering stage constructs a graph, iteratively applies GNN to perform multitask learning to predict the output answer with explanations.


*2023-06-20*

#### [Enabling tabular deep learning when d≫n with an auxiliary knowledge graph](https://arxiv.org/pdf/2306.04766.pdf)

*Camilo Ruiz, Hongyu Ren, Kexin Huang, Jure Leskovec*

*Arxiv 2023*

To overcome the problem of overfitting in tabular data with extremely high d-dimensional features but limited n samples (i.e. d ≫ n), this paper uses an auxiliary KG describing input features to regularize a MLP. Each input feature corresponds to a node in the auxiliary KG. It is based on the inductive bias that two input features corresponding to similar nodes in the auxiliary KG should have similar weight vectors in the MLP’s first layer.


*2023-05-26*

#### [Faithful Knowledge Graph Explanations in Commonsense Question Answering](https://aclanthology.org/2022.emnlp-main.743/)

*Guy Aglionby, Simone Teufel*

*EMNLP 2022*

For commonsense QA, this paper argues that faithful graph-based explanations cannot be extracted from a kind of typical models that combine a text encoder with a graph encoder, mainly due to the text encoder separately conducts reasoning. Motivated by this, it presents two empirical changes to the model, which results in the increase of reasoning proportion done by the graph encoder, thus "increasing the explanation faithfulness".


*2023-04-26*

#### [Addressing Variable Dependency in GNN-based SAT Solving](https://arxiv.org/pdf/2304.08738.pdf)

*Zhiyuan Yan, Min Li, Zhengyuan Shi, Wenjie Zhang, Yingcong Chen, Hongce Zhang*

*Arxiv 2023*

This paper proposes a new GNN-based architecture for predicting a satisfiable boolean assignment for a given CNF formula. Notice that some previous work cannot handle a kind of CNFs which have symmetric variables and asymmetric values (e.g., a XOR b), this paper incorporates recurrent neural networks, specifically, GRU units, into the model. The bi-directional massage passing on the computation graph follows the topological order of the circuit graph.


*2023-04-25*

#### [Temporal Aggregation and Propagation Graph Neural Networks for Dynamic Representation](https://arxiv.org/pdf/2304.07503)

*Tongya Zheng, Xinchao Wang, Zunlei Feng, Jie Song, Yunzhi Hao, Mingli Song, Xingen Wang, Xinyu Wang, Chun Chen*

*IEEE TKDE 2023*

Compared with previous works which usually generate dynamic representation of the graph with limited neighbors, this paper proposes TAP-GNN (short for Temporal Aggregation and Propagation Graph Neural Networks), which applies temporal graph convolution over the whole neighborhood, supported by AP (aggregation and propagation) blocks.


*2023-04-22*

#### [GNNUERS: Fairness Explanation in GNNs for Recommendation via Counterfactual Reasoning](https://arxiv.org/pdf/2304.06182.pdf)

*Giacomo Medda, Francesco Fabbri, Mirko Marras, Ludovico Boratto, Mihnea Tufis, Gianni Fenu*

*Arxiv 2023*

This paper introduces a GNN-based method for identifying and modifying the unfairness in recommendation, where the user-item interactions are represented as a bipartite graph. It proposes a graph perturbation mechanism to alter the user-item interactions. The edges to be perturbed are selected with a loss function that combines: (1) minimizing the absolute pair-wise difference across demographic groups, and (2) minimizing the distance between the original adjacency matrix and the perturbed one.


*2023-04-16*

#### [Continual Graph Convolutional Network for Text Classification](https://arxiv.org/pdf/2304.04152.pdf)

*Tiandeng Wu, Qijiong Liu, Yi Cao, Yao Huang, Xiao-Ming Wu, Jiandong Ding*

*AAAI 2023*

This paper proposes a GCN-based text classification method that can be applied for inductive inference on streaming data. It firstly fixes a universal token set (e.g., taken from PLM such as BERT). Then each document can be tokenized into a set of seen tokens. In this way, a token-document graph can be built, where edge weights are dynamically computed based on tf-idf. 

