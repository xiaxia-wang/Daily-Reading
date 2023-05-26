

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

