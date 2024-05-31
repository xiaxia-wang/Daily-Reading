











*2024-05-31*

#### [Faithful Explanations of Black-box NLP Models Using LLM-generated Counterfactuals](https://openreview.net/forum?id=UMfcdRIotC)

*Yair Ori Gat, Nitay Calderon, Amir Feder, Alexander Chapanin, Amit Sharma, Roi Reichart*

*ICLR 2024*

To obtain explanation for predictions over causal graphs with order-faithfulness, this paper first proposes counterfactual generation as an approach, by prompting an LLM to change a specific text concept while keeping confounding concepts unchanged. As this approach is too costly for inference-time, it also presents a second matching-based method guided by an LLM at training-time and learns a dedicated embedding space.


*2024-05-29*

#### [A Multimodal Automated Interpretability Agent](https://arxiv.org/abs/2404.14394)

*Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, Antonio Torralba*

*Arxiv 2024*

This paper describes a multimodal automated interpretability agent that is equipped with a pre-trained vision-language model and a set of tools that support iterative experimentation on subcomponents of other models to explain their behavior. Not truly "interpretable" though.


*2024-05-28*

#### [Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning](https://openreview.net/forum?id=ZGNWW7xZ6Q)

*LINHAO LUO, Yuan-Fang Li, Reza Haf, Shirui Pan*

*ICLR 2024*

To enhance LLMs' reasoning trustfulness, this paper proposes a collaboration approach between an LLM and a KG. Specifically, it presents a planning-retrieval-reasoning framework, where relation paths grounded by the KG are firstly generated as 'plans' by optimizing a maximal probability target. Then the plans are used to retrieve reasoning paths from the KG, which are afterwards used by the LLM to conduct reasoning.


*2024-05-24*

#### [Expressivity of ReLU-Networks under Convex Relaxations](https://openreview.net/forum?id=awHTL3Hpto)

*Maximilian Baader, Mark Niklas Mueller, Yuhao Mao, Martin Vechev*

*ICLR 2024*

This paper studies the expressive power of finite ReLU neural networks by considering their various convex relaxations. It measures the networks ability to represent continuous piecewise linear (CPWL) functions.


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

