







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
