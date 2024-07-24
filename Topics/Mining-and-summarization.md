













*2024-07-24*

#### [A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation](https://arxiv.org/abs/2402.03358)

*Mohammad Hashemi, Shengbo Gong, Juntong Ni, Wenqi Fan, B. Aditya Prakash, Wei Jin*

*IJCAI 2024 Survey*

This paper reviews existing works for graph reduction (summarization), the process of finding a graph dataset of smaller size while preserving its key information. It categorizes existing works into 3 types. Graph  sparsification selects a subset of existing nodes or edges from the original graph. Graph coarsening produces a graph consisting of supernodes and superedges, with a surjective mapping from the original graph to a coarse graph. Graph condensation produces a synthetic graph that contains learnable lebels, such that a GNN trained on the condensed graph has comparable performance as the original graph.


*2024-06-27*

#### [To prompt or not to prompt: Navigating the use of Large Language Models for integrating and modeling heterogeneous data](https://doi.org/10.1016/j.datak.2024.102313)

*Adel Remadi, Karim El Hage, Yasmina Hobeika, Francesca Bugiotti*

*Data & Knowledge Engineering 2024*

This paper investigates the problem of integrating heterogeneous data sources by applying LLMs. It demonstrates LLMs’ capability to effectively extract data from unstructured sources, and further highlights that LLMs can enhance data integration by providing the ability to resolve entities originating from multiple data sources.


*2024-05-19*

#### [Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data](https://papers.nips.cc/paper_files/paper/2023/hash/13183a224208671a6fc33ba1aa661ec4-Abstract-Conference.html)

*Xin Zheng, Miao Zhang, Chunyang Chen, Quoc Viet Hung Nguyen, Xingquan Zhu, Shirui Pan*

*NeurIPS 2023*

Unlike typical graph condensation approaches jointly optimize node representations, topological structure, and GNN parameters, this paper proposes a condensation framework that distills the original large-scale graph into a graph-free data form. The model contains two parts, (1) a **training trajectory meta-matching** scheme for effectively synthesizing small-scale graph-free data; (2) a graph neural feature score metric for dynamically evaluating the quality of the condensed data. Through training trajectory meta matching, it aligns the long-term GNN learning behaviors between the large-scale graph and the condensed small-scale graph-free data. Then the underlying condensed graph-free data would be dynamically evaluated with the graph neural feature score.


*2024-03-23*

#### [A Survey on Neural Data-to-Text Generation](https://ieeexplore.ieee.org/document/10215344)

*Yupian Lin, Tong Ruan, Jingping Liu, Haofen Wang*

*IEEE TKDE 2023*

Data-to-text Generation (D2T) aims to generate textual natural language statements that can fluently and precisely describe the structured data such as graphs, tables, and meaning representations (MRs) in the form of key-value pairs. This paper provides a comprehensive review on existing neural data-to-text generation approaches by firstly introducing available D2T resources, and surveying existing works based on the taxonomy along two axes: neural end-to-end D2T and neural modular D2T. Then it discusses the potential applications and adverse impacts.


*2024-03-22*

#### [Domain Adaptation and Summary Distillation for Unsupervised Query Focused Summarization](https://ieeexplore.ieee.org/document/10185622)

*Jiancheng Du, Yang Gao*

*IEEE TKDE 2023*

This paper proposes an unsupervised domain adaptation and summary distillation method (DASD). It firstly transforms a large-scale query-free benchmark into a query-focused dataset (Query-CNNDM) while preserving its informative summaries. To achieve the domain adaptation for unsupervised query-focused abstractive summarization (QFS), it designs a query-aware gap sentence generation (q-GSG) strategy to equip the model with the capability of learning target textual knowledge and obtaining a good initialization at the target domain. For instance-specific regularization, it trains a teacher model with the Query-CNNDM to generate pseudo-labels for summary distillation.


*2023-11-17*

#### [Inducing Causal Structure for Abstractive Text Summarization](https://dl.acm.org/doi/10.1145/3583780.3614934)

*Lu Chen, Ruqing Zhang, Wei Huang, Wei Chen, Jiafeng Guo, Xueqi Cheng*

*CIKM 2023*

This paper proposes a model for causal structure-guided generative text summarization. Specifically, it separately learns the hidden factors for the core-content, side-content, document style and summary style, and combines them into the generative model of the summary. The decoder consists a reconstruction module to simulate the input text, and a prediction module to provide the generated summary. The loss function includes the reconstruction loss, prediction loss, KL loss and a content-guidance loss.


*2023-11-01*

#### [CompressGraph: Efficient Parallel Graph Analytics with Rule-Based Compression](https://dl.acm.org/doi/abs/10.1145/3588684)

*Zheng Chen, Feng Zhang, JiaWei Guan, Jidong Zhai, Xipeng Shen, Huanchen Zhang, Wentong Shu, Xiaoyong Du*

*SIGMOD 2023*

This paper proposes CompressGraph as an efficient rule-based graph analytics engine that leverages data redundancy in graphs to achieve both performance boost and space reduction for common graph applications. Its main advantages include (1) the rule-based abstraction supports the reuse of intermediate results during graph traversal, (2) it has intense expressiveness for a wide range of graph applications, and (3) it scales well under high parallelism because the context-free rules have few dependencies.


*2023-10-31*

#### [Near-Duplicate Sequence Search at Scale for Neural Language Model Memorization Evaluation](https://dl.acm.org/doi/abs/10.1145/3589324)

*Zhencan Peng, Zhizhi Wang, Dong Deng*

*SIGMOD 2023*

This paper proposes an efficient and scalable near-duplicate sequence search algorithm at scale for LM training corpus, which can be applied to evaluate the training data memorization by the model. The algorithm generates and groups the min-hash values for all sequences with at least t tokens in the corpus within reasonable time, then it searches for all sequences sharing enough min-hash values with the query using inverted indexes and prefix filtering.


*2023-07-30*

#### [FineSum: Target-Oriented, Fine-Grained Opinion Summarization](https://dl.acm.org/doi/10.1145/3539597.3570397)

*Suyu Ge, Jiaxin Huang, Yu Meng, Jiawei Han*

*WSDM 2023*

This paper proposes a weakly-supervised model for target-oriented, fine-grained opinion summarization, which includes (1) extracting candidate opinion phrases by identifying objects and associated description from the raw corpus, (2) identifying possible aspect and sentiment in each candidate phrase, and (3) aggregating phrases within each aspect and sentiment to obtain fine-grained opinion clusters.
