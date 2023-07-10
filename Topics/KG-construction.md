





*2023-07-10*

#### [Wikidata as a seed for Web Extraction](https://dl.acm.org/doi/10.1145/3543507.3583236)

*Kunpeng Guo, Dennis Diefenbach, Antoine Gourru, Christophe Gravier*

*WWW 2023*

This paper proposes a pipeline for extracting facts from Web pages to be added in Wikidata, which is motivated by QA-based relation extraction from textual data. The framework consists of different modules: knowledge selection (which identifies facts to be completed), data cleaning (which fetches websites that contain the underlying fact and perform general cleaning), relation extraction (which extracts the actual fact from a website), object-linking (which links the identifies object to a Wikidata item), WikidataComplete integration (which proposes extracted facts to users for fact verification).


*2023-06-30*

#### [Schema-aware Reference as Prompt Improves Data-Efficient Knowledge Graph Construction](https://arxiv.org/abs/2210.10709)

*Yunzhi Yao, Shengyu Mao, Ningyu Zhang, Xiang Chen, Shumin Deng, Xi Chen, Huajun Chen*

*SIGIR 2023*

This paper proposes a plug-in approach of schema-augmented prompting methods for KG construction. It is applied for triple-form event extraction. In a (offline) reference store construction process, it builds a map between example texts and a schema graph. In the real extraction phase, given the input text, it firstly retrieves similar text contexts in the reference store, and uses the mapped schema graph information to enhance the prompts thus benefiting KG construction.


*2023-05-27*

#### [Structured prompt interrogation and recursive extraction of semantics (SPIRES): A method for populating knowledge bases using zero-shot learning](https://arxiv.org/abs/2304.02711)

*J. Harry Caufield, Harshad Hegde, Vincent Emonet, Nomi L. Harris, Marcin P. Joachimiak, Nicolas Matentzoglu, HyeongSik Kim, Sierra A.T. Moxon, Justin T. Reese, Melissa A. Haendel, Peter N. Robinson, Christopher J. Mungall*

*Arxiv 2023*

This paper presents a process to use prompt interrogation for constructing knowledge bases. The result shows its accuracy is comparable to existing relation extraction methods, while being easier, more customizable and more flexible.


*2023-05-24*

#### [Improving Continual Relation Extraction by Distinguishing Analogous Semantics](https://arxiv.org/pdf/2305.06620.pdf)

*Wenzheng Zhao, Yuanning Cui, Wei Hu*

*ACL 2023*

Existing methods for continual relation extraction usually retain a small set of typical samples to re-train the model, which will potentially results in model overfitting. To address this problem, this paper proposes memory-insensitive relation prototypes and memory augmentation methods, as well as a framework  especially for addressing analogous relations that are typically difficult to process.


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

