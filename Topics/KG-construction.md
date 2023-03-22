







*2023-03-22*

#### [Multi-View Clustering for Open Knowledge Base Canonicalization](https://dl.acm.org/doi/10.1145/3534678.3539449)

*Wei Shen, Yang Yang, Yinan Liu*

*KDD 2022*

This paper proposes a model for clustering open information extraction results based on two views of actual data, namely, a fact view and a context view. For the fact view, this paper utilizes the KG embeddings to represent the actual data. Meanwhile, for the context view, it uses PLM to compute k-NNs and pseudo labels. These two views are combined iteratively in an EM process, and without further need of supervised training data.


*2023-02-04*

#### [Improving Mental Health Support Response Generation with Eventuality-based Knowledge Graph](https://knowledge-nlp.github.io/aaai2023/papers/006-MHKG-oral.pdf)

*Lingbo Tong, Qi Liu, Wenhao Yu, Mengxia Yu, Zhihan Zhang, Meng Jiang*

*KnowledgeNLP-AAAI 2023*

To help generate better mental health support response for online forums, this paper constructs a knowledge graph MHKG consisting of eventualities related to mental health support. This KG is evaluated with the text generation task. Result shows that enriching the input sequence with the ground-truth neighbors in MHKG is able to significantly improve model performance of response generation.


*2023-01-29*

#### [Extracting Cultural Commonsense Knowledge at Scale](https://arxiv.org/abs/2210.07763)

*Tuan-Phong Nguyen, Simon Razniewski, Aparna Varde, Gerhard Weikum*

*WWW 2023*

This paper works on extracting cultural commonsense knowledge from large corpus of Web pages. It introduces an overall pipeline for doing this. First and second, it uses general NER (spaCy’s NER) and string matching (hand-crafted lexico-syntactic rules) to identify potential assertions. Then it uses PLM to perform zero-shot classification to divide the assertions into specific cultural facets. After that, the assertions are clustered (sentence embeddings + Hierarchical Agglomerative Clustering) and concepts are extracted (as frequent n-grams) from the clusters. Finally, some scores (e.g., frequency, distinctiveness) are computed for these assertions. 


*2022-11-07*

#### [UnCommonSense: Informative Negative Knowledge about Everyday Concepts](https://dl.acm.org/doi/10.1145/3511808.3557484)

*Hiba Arnaout, Simon Razniewski, Gerhard Weikum, Jeff Z. Pan*

*CIKM 2022*

This paper focuses on capturing informative negations about concepts in commonsense knowledge bases. Since general commonsense knowledge bases hold an open-world assumption, they cannot answer the questions (i.e., unknown) related to absent triples in the KB. In this paper, for a given target concept, a set of comparable (similar) concepts are firstly extracted, then a local closed-world assumption is applied to compute the negative relations. Then these negative candidates are scrutinized over the input KB using sentence embeddings, and also evaluated over a external PLM. Finally, the most informative negations are selected as the result.

