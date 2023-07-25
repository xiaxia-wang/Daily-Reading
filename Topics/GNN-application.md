




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

