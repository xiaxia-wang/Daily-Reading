




*2025-03-17*

#### [SpArX: Sparse Argumentative Explanations for Neural Networks [Technical Report]](https://arxiv.org/abs/2301.09559)

*Hamed Ayoobi, Nico Potyka, Francesca Toni*

*ECAI 2023*

This paper investigates the explainability of MLPs through the lens of quantitative argumentative framework. In particular, it formulates a general MLP with a QAF instance, and identifies the attack/support relationships between neurons to give explanations for one or all the MLP outputs. It does not guarantee fully faithfulness of the explaination to the output, instead, it uses a quantitative value to indicate the faithfulness/unfaithfulness.


*2024-08-13*

#### [SES: Bridging the Gap Between Explainability and Prediction of Graph Neural Networks](https://arxiv.org/abs/2407.11358)

*Zhenhua Huang, Kunhao Li, Shaojie Wang, Zhaohong Jia, Wentao Zhu, Sharad Mehrotra*

*ICDE 2024*

This paper proposes an "explainable" GNN model that comprises two processes: explainable training and enhanced predictive learning. During explainable training, it employs a global mask generator co-trained with a graph encoder and directly produces crucial structure and feature masks, reducing time consumption and providing node feature and subgraph explanations. During enhanced predictive learning, mask-based positive-negative pairs are constructed utilizing the explanations to compute a triplet loss and enhance the node representations by contrastive learning.


*2024-08-12*

#### [Expressivity and Generalization: Fragment-Biases for Molecular GNNs](https://arxiv.org/abs/2406.08210)

*Tom Wollschläger, Niklas Kemper, Leon Hetzel, Johanna Sommer, Stephan Günnemann*

*ICML 2024*

This paper conducts theoretical analyses of GNN models that explicitly use fragment information as inductive bias for molecular property prediction. Specifically, it proposes Fragment-WL test as an extension to the standard WL-test. Based on that, it proposes a new GNN architecture and a fragmentation with infinite vocabulary that improves the model expressiveness.


*2024-08-05*

#### [RAG-Ex: A Generic Framework for Explaining Retrieval Augmented Generation](https://dl.acm.org/doi/10.1145/3626772.3657660)

*Viju Sudhi, Sinchana Ramakanth Bhat, Max Rudat, Roman Teucher*

*SIGIR 2024 Short*

This paper proposes a framework to identify the key information (which could be viewed as `approximate' explanation) for LLMs' generation w.r.t. the input for tasks such as RAG or QA. Specifically, it introduces several perturbation approaches to distort the input, and identify the common part in the input of the cases when the LLM produces the correct answer.


*2024-07-22*

#### [Building Expressive and Tractable Probabilistic Generative Models: A Review](https://arxiv.org/abs/2402.00759)

*Sahil Sidheekh, Sriraam Natarajan*

*IJCAI 2024 Survey*

A Probabilistic Circuit is a computational graph that compactly encodes a probability distribution via factorizations and mixtures, which consists of three types of nodes - Sums, Products and Leaf Distributions. Each node in the graph computes a non-negative function, which can be interpreted as an unnormalized probability measure over a subset of random variables as the scope of the node. The computational graph is evaluated bottom-up recursively. This paper first introduces the building blocks, properties, learning methodologies and challenges for tractable probabilistic circuits, and then discusses hybrid techniques that merge tractable PCs with deep generative models to achieve the best of both worlds.


*2024-07-04*

#### [A Logic for Reasoning About Aggregate-Combine Graph Neural Networks](https://arxiv.org/abs/2405.00205)

*Pierre Nunn, Marco Sälzer, François Schwarzentruber, Nicolas Troquard*

*IJCAI 2024*

This paper proposes a modal logic in which counting modalities appear in linear inequalities, and each formula can be transformed into an equivalent graph neural network. In contrast, a class of GNNs can be transformed into a formula, thus making it more clear about the logical expressiveness of GNNs. It also shows that the satisfiability problem is PSPACE-complete.


*2024-06-11*

#### [Understanding Expressivity of GNN in Rule Learning](https://openreview.net/forum?id=43cYe4oogi)

*Haiquan Qiu, Yongqi Zhang, Yong Li, quanming yao*

*ICLR 2024*

This paper investigates the form of FOL rule structures that can be captured by a GNN model (but not in the reversed order, i.e., all possible rules that can be captured), which can be viewed as a lower bound of expressivity of certain types of GNNs. Besides, it proposes another GNN model that is proved to be able to capture more rule structures.


*2024-06-03*

#### [Logical Languages Accepted by Transformer Encoders with Hard Attention](https://openreview.net/forum?id=gbrHZq07mq)

*Pablo Barcelo, Alexander Kozachinskiy, Anthony Widjaja Lin, Vladimir Podolskii*

*ICLR 2024*

This paper investigates the formal languages that can be recognized by (1) Unique Hard Attention Transformers (UHAT) and (2) Average Hard Attention Transformers (AHAT). It obtains a characterization of which counting properties are expressible by UHAT and AHAT, in relation to regular languages.


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

