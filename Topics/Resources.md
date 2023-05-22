




*2023-05-22*

#### [FactKG: Fact Verification via Reasoning on Knowledge Graphs](https://arxiv.org/pdf/2305.06590.pdf)

*Jiho Kim, Sungjin Park, Yeonsu Kwon, Yohan Jo, James Thorne, Edward Choi*

*ACL 2023*

This paper proposes a dataset named FactKG for fact verification by reasoning over knowledge graphs. It consists of 5 kinds of claims/reasoning, namely, one-hop, conjunction, existence, multi-hop and negation. The facts are extracted from DBpedia, and the claims are generated based on WebNLG.


*2023-03-29*

#### [Interpretability, Then What? Editing Machine Learning Models to Reflect Human Knowledge and Values](https://dl.acm.org/doi/10.1145/3534678.3539074)

*Zijie J. Wang, Alex Kale, Harsha Nori, Peter Stella, Mark E. Nunnally, Duen Horng Chau, Mihaela Vorvoreanu, Jennifer Wortman Vaughan, Rich Caruana*

*KDD 2022*

This paper introduces an interactive system which allows domain experts to manually edit each prediction function within a Generalized Additive Model (GAM), in order to modify the prediction results ("to fix the problematic parts").


*2023-03-26*

#### [GraphWorld: Fake Graphs Bring Real Insights for GNNs](https://doi.org/10.1145/3534678.3539203)

*John Palowitch, Anton Tsitsulin, Brandon Mayer, Bryan Perozzi*

*KDD 2022*

This paper introduces GraphWorld, a system for benchmarking GNN models on an arbitrarily-large population of synthetic graphs for any conceivable GNN task. It allows a user to efficiently generate a world with millions of statistically diverse datasets. The user has fine-grained control over graph generator parameters, and can benchmark arbitrary GNN models with built-in hyperparameter tuning.


*2023-02-02*

#### [Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering](https://knowledge-nlp.github.io/aaai2023/papers/004-ScienceQA-oral.pdf)

*Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, Ashwin Kalyan*

*KnowledgeNLP-AAAI 2023*

This paper proposes a benchmarking dataset named ScienceQA for multimodal question answering. It consists about 21k multimodal multiple choice questions of scientific topics, as well as annotations of their answers with corresponding lectures and explanations (which simulate a chain of thoughts, CoT). This paper also tries to design models for generating lectures and explanations. The results demonstrate the utility of CoT in language models as it improves the question answering performance of SOTA models.


*2023-02-01*

#### [Knowledge Retrieval Over Public and Private Data](https://knowledge-nlp.github.io/aaai2023/papers/003-PQA-oral.pdf)

*Simran Arora, Patrick Lewis, Angela Fan, Jacob D Kahn, Christopher Re*

*KnowledgeNLP-AAAI 2023*

This paper studies the task of multi-hop retrieval for question answering over public and private corpora. It firstly formulates the task as a multi-hop (ordered) sequence for retrieval over several public and private corpora, and requires the visiting order to strictly include public corpora before private ones. In this way, the leakage of private data can be avoided. Then it also proposes a public-private benchmarking dataset for evaluating multi-hop retrieval-based QA methods.


*2023-01-31*

#### [ComFact: A Benchmark for Linking Contextual Commonsense Knowledge](https://knowledge-nlp.github.io/aaai2023/papers/002-ComFact-oral.pdf)

*Silin Gao, Jena D Hwang, Saya Kanno, Hiromi Wakaki, Yuki Mitsufuji, Antoine Bosselut*

*KnowledgeNLP-AAAI 2023*

This paper proposes a benchmark dataset for contextual commonsense knowledge linking, an example application of which is to find relevant pieces of fact in the KG for a dialog or story (contexts). To build this benchmark dataset, the authors firstly apply string matching and SentenceBERT to generate and filter candidate facts. Then the relevance of these facts are judged by crowdsourcing workers. In the experiments, several popular fact linking methods are evaluated over the dataset, showing that their performances are still far behind human beings. 


*2023-01-19*

#### [Bidimensional Leaderboards: Generate and Evaluate Language Hand in Hand](https://aclanthology.org/2022.naacl-main.259/)

*Jungo Kasai, Keisuke Sakaguchi, Ronan Le Bras, Lavinia Dunagan, Jacob Morrison, Alexander R. Fabbri, Yejin Choi, Noah A. Smith*

*NAACL 2022*

Unlike typical leaderboards which generally compare language models based on BLEU/ROUGE, this paper proposes a bidimensional leaderboard accepting both generation models and evaluation metrics. The leaderboard automatically creates an ensemble metric based on global analysis of the models. It will also rank and compare the metrics based on the correlation with human judgements. 


*2023-01-07*

#### [μ KG: A Library for Multi-source Knowledge Graph Embeddings and Applications](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_35)

*Xindi Luo, Zequn Sun, Wei Hu*

*ISWC 2022*

This resource paper provides an open-source Python library consisting of 26 popular knowledge graph embedding models and 16 benchmark datasets. 


*2022-12-22*

#### [WDV: A Broad Data Verbalisation Dataset Built from Wikidata](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_32)

*Gabriel Amaral, Odinaldo Rodrigues, Elena Simperl*

*ISWC 2023*

This resource paper proposes a dataset consists of over 7.6k Wikidata entries and their verbalized textual version. The dataset is made up from crowdsourcing annotations. Each entry contains textual descriptions of the subject, predicate and object in Wikidata, respectively.  


*2022-11-28*

#### [HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data](https://doi.org/10.18653/v1/2020.findings-emnlp.91)

*Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, William Yang Wang*

*EMNLP Findings 2020*

This paper proposes a question answering dataset containing both a Wikipedia table and a set of passages as resources. It begins with selecting Wikipedia tables which have hyperlinked cells. Then it retrieves the Wikipedia pages of each cell and collects the first sentences as supporting textual data. The questions are collected using AMT and post-processed to avoid bias. 


*2022-11-27*

#### [FinQA: A Dataset of Numerical Reasoning over Financial Data](https://doi.org/10.18653/v1/2021.emnlp-main.300)

*Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R. Routledge, William Yang Wang*

*EMNLP 2021*

This paper proposes a dataset named FinQA containing question-answer pairs with annotated reasoning explanations based on financial reports. By implementing existing QA methods as baseline, the experimental results suggest existing pre-trained language models still fall far short than humans in acquiring finance knowledge and conducting multi-step numerical reasoning. 
