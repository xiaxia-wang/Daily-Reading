







*2024-05-22*

#### [REFACTOR: Learning to Extract Theorems from Proofs](https://openreview.net/forum?id=fgKjiVrm6u)

*Jin Peng Zhou, Yuhuai Wu, Qiyang Li, Roger Baker Grosse*

*ICLR 2024*

This paper aims to train a GNN for extracting a sub-component as the ``theorem'' from a proof tree. Specifically, given a proof tree with a set of nodes $V$, edges $E$, and node features $x_v$ which correspond to the name N and the proposition PROP associated with each node. The task is to output a subset of nodes $V_{target} ⊂ V$ that correspond to an embedded proof of a useful theorem. The problem is formulated as a node-level binary classification that predicts whether each node belongs to $V_{target}$. To solve the problem, it uses a GNN parametrized by $θ$ to take a graph with its node features as input, and outputs a scalar $\tilde{P}_v$ between 0 and 1 for each node $v \in V$, representing the probability belonging to $V_{target}$. The objective is a binary cross entropy loss between the node level probabilities and ground truth targets of a graph.


*2024-05-03*

#### [PEACH: Pretrained-embedding Explanation Across Contextual and Hierarchical Structure](https://arxiv.org/abs/2404.13645)

*Feiqi Cao, Caren Han, Hyunsuk Chung*

*IJCAI 2024*

This paper introduces PEACH, a tree-based explanation framework for text-based classification using pretrained contextual embeddings. Specifically, it obtains a pretrained embedding vector for each text document in the given corpus, then applies a feature extraction model (statistical or CNN-based) with decision tree algorithms to generate the tree-structured explanation.

