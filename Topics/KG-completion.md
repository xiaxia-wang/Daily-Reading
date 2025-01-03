








*2024-12-29*

#### [Uncertain Knowledge Graph Completion with Rule Mining](https://link.springer.com/chapter/10.1007/978-981-97-7707-5_9)

*Yilin Chen, Tianxing Wu, Yunchang Liu, Yuxiang Wang, Guilin Qi*

*WISA 2024*

This paper proposes a framework that utilize an LLM module to measure the confidence of each triple in a KG.


*2024-12-21*

#### [DynaSemble: Dynamic Ensembling of Textual and Structure-Based Models for Knowledge Graph Completion](https://arxiv.org/abs/2311.03780)

*Ananjan Nandi, Navdeep Kaur, Parag Singla, Mausam*

*ACL 2024*

This paper proposes an ensemble method for knowledge graph completion, which learns query-dependent ensemble weights to combine several models, and uses the distributions of scores assigned by the models in the ensemble to all candidate entities.


*2024-11-09*

#### [A Prompt-Based Knowledge Graph Foundation Model for Universal In-Context Reasoning](https://arxiv.org/abs/2410.12288)

*Yuanning Cui, Zequn Sun, Wei Hu*

*NeurIPS 2024*

This paper proposes a prompt-based KG pretrained model with ICL for reasoning. Specifically, it introduces a prompt graph with a query-related example fact as context to understand the query relation. To encode prompt graphs with the generalization ability to unseen entities and relations in queries, it uses a unified tokenizer that maps entities and relations in prompt graphs to predefined tokens. Then, two message passing neural networks was applied to perform prompt encoding and KG reasoning, respectively.


*2024-09-12*

#### [Finetuning Generative Large Language Models with Discrimination Instructions for Knowledge Graph Completion](https://www.arxiv.org/abs/2407.16127)

*Yang Liu, Xiaobin Tian, Zequn Sun, Wei Hu*

*ISWC 2024*

This paper proposes a KG completion model that based on a fine-tuned LLM and a simple embedding-based model for KG link prediction. Specifically, the query for link prediction is firstly fed into the embedding-based model to get top-m entities as answer candidates. Then the LLM is used with a discrimination instruction to select the most plausible entity. Besides, to enhance the graph reasoning ability of the LLM, a knowledge adaption module is applied to inject the embeddings of query and candidate entities to the LLM.


*2024-07-06*

#### [Start from Zero: Triple Set Prediction for Automatic Knowledge Graph Completion](https://ieeexplore.ieee.org/document/10529617)

*Wen Zhang, Yajing Xu, Peng Ye, Zhiwei Huang, Zezhong Xu, Jiaoyan Chen, Jeff Z. Pan, Huajun Chen*

*TKDE 2024*

This paper proposes a knowledge graph completion approach which is formulated as predicting a set of missing facts based on a set of given positive facts, without specifying any query such as (s, r, ?). It first proposes 3 classification metrics and 1 ranking metric, considering both the partial-open world and the closed-world assumptions. Then it applies an efficient subgraph-based method to predict the missing triple set.


*2024-06-18*

#### [Learning from Both Structural and Textual Knowledge for Inductive Knowledge Graph Completion](https://papers.nips.cc/paper_files/paper/2023/hash/544242770e8333875325d013328b2079-Abstract-Conference.html)

*Kunxun Qi, Jianfeng Du, Hai Wan*

*NeurIPS 2023*

This paper proposes a two-stage framework by using both structural and textual knowledge to learn rule-based KGC model. In the first stage, it computes a set of triples with confidence scores (called soft triples) from a text corpus by distant supervision. In the second stage, these soft triples are added into the given incomplete graph to jointly train a rule-based model for KGC.

