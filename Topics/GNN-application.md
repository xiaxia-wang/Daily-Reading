

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

