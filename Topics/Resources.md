





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
