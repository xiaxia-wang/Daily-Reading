







*2025-05-28*

#### [Graphusion: A RAG Framework for Knowledge Graph Construction with a Global Perspective](https://arxiv.org/abs/2410.17600)

*Rui Yang, Boming Yang, Aosong Feng, Sixun Ouyang, Moritz Blum, Tianwei She, Yuang Jiang, Freddy Lecue, Jinghui Lu, Irene Li*

*WWW 2025 NLP4KGC*

This work presents a zero-shot KG construction framework from free text. It contains three steps: (1) extract a list of seed entities by topic modeling; (2) conduct candidate triplet extraction by LLMs; (3) use a fusion module that provides a global view of the extracted knowledge, incorporating entity merging, conflict resolution, and triplet discovery.


*2025-05-08*

#### [RAKG:Document-level Retrieval Augmented Knowledge Graph Construction](https://arxiv.org/abs/2504.09823)

*Hairong Zhang, Jiaheng Si, Guohang Yan, Boyuan Qi, Pinlong Cai, Song Mao, Ding Wang, Botian Shi*

*Arxiv 2025*

This paper proposes a LLM-based approach for knowledge graph construction from text corpus. In particular, it retrieves relevant nodes and text-chunks to identify relations between entities and extends the graph.


*2025-04-30*

#### [Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction](https://arxiv.org/abs/2404.03868)

*Bowen Zhang, Harold Soh*

*EMNLP 2024*

This paper introduces a pipeline for LLM-based KG construction, which consists of open information extraction followed by schema definition and post-hoc canonicalization.


*2025-02-27*

#### [Continual Multimodal Knowledge Graph Construction](https://www.ijcai.org/proceedings/2024/0688.pdf)

*Xiang Chen, Jintian Zhang, Xiaohan Wang, Ningyu Zhang, Tongtong Wu, Yuxiang Wang, Yongheng Wang, Huajun Chen*

*IJCAI 2024*

This paper introduces a framework for continual multimodal KGC, which consists of two critical modules: (1) a gradient modulation technique to handle the imbalanced learning dynamics across modalities, and (2) instead of typical cross-attention multimodal interaction, it computes an inter-modal self query affinity for decoupling the parameters with attention distillation.


*2025-02-12*

#### [OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System](https://arxiv.org/abs/2412.20005)

*Yujie Luo, Xiangyuan Ru, Kangwei Liu, Lin Yuan, Mengshu Sun, Ningyu Zhang, Lei Liang, Zhiqiang Zhang, Jun Zhou, Lanning Wei, Da Zheng, Haofen Wang, Huajun Chen*

*TheWebConf 2025 Demo Track*

This pape introduces a dockerized schema-guided system that extracts knowledge from the Web and raw PDF books, which supports various domains (science, news, etc.). In particular, it employs three agents, (1) Schema Agent for schema analysis with various data types, (2) Extraction Agent for extracting knowledge with various LLMs, and (3) Reflection Agent to debug and handle erroneous cases.


*2025-01-26*

#### [Can LLMs be Good Graph Judger for Knowledge Graph Construction?](https://arxiv.org/abs/2411.17388)

*Haoyu Huang, Chong Chen, Conghui He, Yang Li, Jiawei Jiang, Wentao Zhang*

*Arxiv 2024*

This paper investigates the ability of LLMs for extracting schema-free triples from plain texts to conduct knowledge graph construction. In particular, it identifies 3 limitations from existing approaches: (1) a large amount of excessive noise exists in real-world documents, resulting in extracting messy information, (2) native LLMs struggle to effectively extract accurate knowledge from domain-specific documents, and (3) hallucinations. To overcome these limitations, this paper introduces three innovative modules, i.e., entity-centric iterative text denoising, knowledge aware instruction tuning, and graph judgement.


*2024-12-27*

#### [UrbanKGent: A Unified Large Language Model Agent Framework for Urban Knowledge Graph Construction](https://arxiv.org/abs/2402.06861)

*Yansong Ning, Hao Liu*

*NeurIPS 2024*

This paper proposes a LLM-based framework for urban knowledge graph construction, which consists of two subtasks, i.e., relational triple extraction and knowledge graph completion. It implements a knowledgeable instruction generation module and a tool-augmented iterative trajectory refinement method, which align LLMs to UrbanKGC tasks and compensate for their geospatial computing and reasoning inability.


*2024-12-24*

#### [SAC-KG: Exploiting Large Language Models as Skilled Automatic Constructors for Domain Knowledge Graph](https://aclanthology.org/2024.acl-long.238/)

*Hanzhu Chen, Xu Shen, Qitan Lv, Jie Wang, Xiaoqi Ni, Jieping Ye*

*ACL 2024*

This paper proposes a framework that utilizes LLMs to construct domain KGs. Specifically, it consists of three components: Generator, Verifier, and Pruner. For a given entity, Generator produces its relations and tails from raw domain corpora, to construct a specialized single-level KG. Verifier and Pruner then work together to ensure precision by correcting generation errors and determining whether newly produced tails require further iteration for the next-level KG.


*2024-11-21*

#### [Knowledge Graph Error Detection with Contrastive Confidence Adaption](https://arxiv.org/abs/2312.12108)

*Xiangyu Liu, Yang Liu, Wei Hu*

*AAAI 2024*

This paper proposes a KG error detection model that integrates both textual and graph structural information from triplet reconstruction for better distinguishing semantics. It employs interactive contrastive learning to measure the agreement between textual and structural embeddings for each candidate entity, and evaluates on datasets with semantically-similar noise and adversarial noise.


*2024-04-02*

#### [Knowledge graph extension with a pre-trained language model via unified learning method](https://linkinghub.elsevier.com/retrieve/pii/S0950705122013417)

*Bonggeun Choi, Youngjoong Ko*

*Knowledge-Based Systems 2023*

This paper proposes a knowledge graph extension approach, i.e., to link new entity with some existing entity in the graph, by formulating it as a classification task. Specifically, given a pair as (head-entity, relation), it aims to find the target entity to form a triple (head-entity, relation, target-entity) by classifying all existing entities, which is hard to scale to large graphs.


*2023-12-13*

#### [Schema-adaptable Knowledge Graph Construction](https://arxiv.org/abs/2305.08703)

*Hongbin Ye, Honghao Gui, Xin Xu, Xi Chen, Huajun Chen, Ningyu Zhang*

*EMNLP 2023 Findings*

This paper proposes a task called schema-adaptable KGC, which aims to continually extract entity, relation, and event based on a dynamically changing schema graph without re-training. It first converts existing datasets based on horizontal schema expansion, vertical schema expansion, and hybrid schema expansion, and then evaluates several baseline encoder-decoder models for this task.


*2023-08-16*

#### [NLIRE: A Natural Language Inference method for Relation Extraction](https://www.jws-volumes.com/_files/ugd/c6c160_e11821035f99466591d8fa2ca0f71aec.pdf)

*Wenfei Hu, Lu Liu, Yupeng Sun, Yu Wu, Zhicheng Liu, Ruixin Zhang, Tao Peng*

*JoWS System paper*

This is the only "system" paper among all issues of JoWS in the past 3 years. However, although it is categorized as system paper, it is more like a "weak" version of research paper, with all the sections, contributions, or writing style matching the structure of research paper - which is quite different from what I assume a system paper should be like.


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

