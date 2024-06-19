








*2024-06-19*

#### [Complex Query Answering on Eventuality Knowledge Graph with Implicit Logical Constraints](https://papers.nips.cc/paper_files/paper/2023/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract-Conference.html)

*Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, Yangqiu Song*

*NeurIPS 2023*

This paper introduces Complex Eventuality Query Answering (CEQA) that considers the implicit logical constraints over temporal order and occurrence of eventualities. It proposes to use theorem provers for constructing benchmark datasets to ensure the answers satisfy implicit logical constraints. Besides, it also proposes a memory-enhanced query encoding approach to improve the performance of state-of-the-art neural query encoders on the CEQA task.


*2024-04-03*

#### [Enhancing Complex Question Answering over Knowledge Graphs through Evidence Pattern Retrieval](https://arxiv.org/abs/2402.02175)

*Wentao Ding, Jinmao Li, Liangchuan Luo, Yuzhong Qu*

*WWW 2024*

This paper proposes an IR-based KGQA approach by improving the subgraph extraction module. Specifically, it firstly uses dense retrieval to obtain atomic patterns formed by resource pairs, and then enumerates their combinations to construct candidate evidence patterns. These evidence patterns are scored using a neural model, and the best one is selected to extract a subgraph for downstream answer reasoning.


*2023-10-03*

#### [Variational Open-Domain Question Answering](https://proceedings.mlr.press/v202/lievin23a.html)

*Valentin Liévin, Andreas Geert Motzfeldt, Ida Riis Jensen, Ole Winther*

*ICML 2023*

Open-domain question answering (ODQA) consists of augmenting LMs with external knowledge bases indexed with a retrieval mechanism. This paper proposes a variational inference framework for end-to-end training and evaluation of retrieval-augmented models by extending a previous Renyi divergence metric.


*2023-08-03*

#### [DecAF: Joint Decoding of Answers and Logical Forms for Question Answering over Knowledge Bases](https://openreview.net/forum?id=XHc5zRPxqV9)

*Donghan Yu, Sheng Zhang, Patrick Ng, Henghui Zhu, Alexander Hanbo Li, Jun Wang, Yiqun Hu, William Wang, Zhiguo Wang, Bing Xiang*

*ICLR 2023*

Compared with directly obtaining answers from the texts, generating logical forms for KBQA shows higher accuracy but suffers from non-execution issue. To address this problem, this paper proposes to decode the  answer together with its logical form. It firstly applies free-text retrieval to get relevant texts from the knowledge base. Then it separately uses two encoder-decoder pipeline to get direct answers and the logical form, and combines them as the output.


*2023-07-14*

#### [Hierarchy-Aware Multi-Hop Question Answering over Knowledge Graphs](https://dl.acm.org/doi/10.1145/3543507.3583376)

*Junnan Dong, Qinggang Zhang, Xiao Huang, Keyu Duan, Qiaoyu Tan, Zhimeng Jiang*

*WWW 2023*

Given a complex question containing question entities and a domain specific KG, the answer entities (i.e., candidates) are firstly extracted, as well as a subgraph containing question entities, answer entities and their k-hop neighbors. With these being the input, the problem of hierarchy-aware multi-hop QA is formulated as finding the target answer entities among the given ones. The performance is evaluated using the prediction accuracy.


*2023-07-13*

#### [Knowledge Graph Question Answering with Ambiguous Query](https://dl.acm.org/doi/10.1145/3543507.3583316)

*Lihui Liu, Yuzhong Chen, Mahashweta Das, Hao Yang, Hanghang Tong*

*WWW 2023*

This paper proposes a model for ambiguous question answering over knowledge graphs. It firstly applies an embedding-based model to find top-k candidate answers for the input query. Then a query inference part is used to infer the true relation between the anchor entity and each of the candidates. After that, a neighborhood embedding based VGAE model is used to prune low quality relations, and the rest of relations are used for query ranking and answer re-ranking.


*2023-05-23*

#### [Long-Tailed Question Answering in an Open World](https://arxiv.org/pdf/2305.06557.pdf)

*Yi Dai, Hao Lang, Yinhe Zheng, Fei Huang, Yongbin Li*

*ACL 2023*

This paper proposes a prompt-enhanced encoder-decoder model for open world long-tailed question answering. The training data is formulated as a tuple consisting of a context $c$, a question $q$ and the answer $a$. Two crucial components of this model are (1) instance-level knowledge sharing, and (2) knowledge mining from a PLM.


*2023-05-12*

#### [Few-shot In-context Learning for Knowledge Base Question Answering](https://arxiv.org/pdf/2305.01750.pdf)

*Tianle LI, Xueguang Ma, Alex Zhuang, Yu Gu, Yu Su, Wenhu Chen*

*ACL 2023*

This paper proposes a KBQA method based on in-context learning. Given an input query, the LLM model firstly generates corresponding logical forms for the query. Then it applies entity and relation binders to ground the variables to the actual dataset, and executes the query under relation- and class- constraints.


*2023-03-04*

#### [Semantic Framework based Query Generation for Temporal Question Answering over Knowledge Graphs](https://aclanthology.org/2022.emnlp-main.122/)

*Wentao Ding, Hao Chen, Huayu Li, Yuzhong Qu*

*EMNLP 2022*

This paper proposes a model for temporal question answering over knowledge graphs by utilizing intrinsic connections between events to improve performance. Based on several temporal constraints over questions, it proposes interpretation structures for the query. Then it also designs a pipeline for temporal question answering based on the interpretations.


*2023-02-07*

#### [Can Open-Domain QA Reader Utilize External Knowledge Efficiently like Humans?](https://knowledge-nlp.github.io/aaai2023/papers/009-QAHuman-poster.pdf)

*Neeraj Varshney, Man Luo, Chitta Baral*

*KnowledgeNLP-AAAI 2023*

Unlike typical open-domain QA methods using a retriever-reader approach, this paper proposes a more efficient model with a "closed-book" inference module and an iterative "open-book" prediction part. The "closed-book" inference is achieved by pre-trained models, which already shows a non-trivial performance. Meanwhile, a confidence score is predicted for the current answer at this stage. If the confidence score is not sufficiently high, then the external knowledge (by the "retriever") is iteratively provided to the predictor (the "reader"). Multiple ways for computing the confidence score are evaluated in the paper. 


*2022-12-29*

#### [Logical Form Generation via Multi-task Learning for Complex Question Answering over Knowledge Bases](https://aclanthology.org/2022.coling-1.145/)

*Xixin Hu, Xuan Wu, Yiheng Shu, Yuzhong Qu*

*COLING 2022*

This paper proposes a KBQA method that generates logical forms (s-expressions in the paper, equivalent to SPARQL queries) for given natural language questions. It implements entity retrieval, relation retrieval and multitask classification based on a shared T5 encoder, to improve both the precision and recall of the results. 


*2022-12-03*

#### [Open Domain Question Answering over Tables via Dense Retrieval](https://doi.org/10.18653/v1/2021.naacl-main.43)

*Jonathan Herzig, Thomas Müller, Syrine Krichene, Julian Eisenschlos*

*NAACL 2021*

This paper proposes an open-domain question answering model over tabular data. It contains two parts. The first is a dense retrieval model based on TAPAS (a BERT-based encoder) over all the tables. After retrieving the top-k tables, the second part is a question answering model which takes all the top-k tables as input, and returns a span over the table as the final answer.


*2022-07-23*

#### [Evaluating question answering over linked data](https://doi.org/10.1016/j.websem.2013.05.006)

*Vanessa López, Christina Unger, Philipp Cimiano, Enrico Motta*

*Journal of Web Semantics 2013*

This paper investigates the evaluation of question answering systems over linked data. It proposes a set of challenges and discusses the performances of existing systems. It reviews existing research efforts for evaluating question answering systems, and especially QALD, including its datasets, gold standards, metrics, etc. Then it further analyzes the results of the challenges QALD-1 and 2. It introduces each of the participating systems, and summarizes some difficulties and problems in the QA process.


*2022-07-22*

#### [9th Challenge on Question Answering over Linked Data (QALD-9)](http://ceur-ws.org/Vol-2241/paper-06.pdf)

*Ricardo Usbeck, Ria Hari Gusmita, Axel-Cyrille Ngonga Ngomo, Muhammad Saleem*

*Semdeep/NLIWoD@ISWC 2018*

The Question Answering over Linked Data (QALD) challenges aim to provide up-to-date benchmarks for assessing and comparing SOTA systems that mediate between the user and RDF data. The challenges are formulated as given one or several RDF dataset(s) and natural language questions or keywords, the system should return the correct answers or a SPARQL query that retrieves these answers. All the datasets of QALD-1~9 are available on the Web.


*2022-07-21*

#### [LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and DBpedia](https://doi.org/10.1007/978-3-030-30796-7_5)

*Mohnish Dubey, Debayan Banerjee, Abdelrahman Abdelkawi, Jens Lehmann*

*ISWC 2019, Part 2*

This paper proposes LC-QuAD 2.0, a subsequent version of LC-QuAD. Compared with the previous dataset, it has larger size with 30,000 questions, their paraphrases and corresponding SPARQL queries. Besides, it is compatible with both Wikidata and DBpedia 2018 knowledge graphs.


*2022-07-20*

#### [LC-QuAD: A Corpus for Complex Question Answering over Knowledge Graphs](https://doi.org/10.1007/978-3-319-68204-4_22)

*Priyansh Trivedi, Gaurav Maheshwari, Mohnish Dubey, Jens Lehmann*

*ISWC 2017*

This resource paper proposes a dataset for evaluating question answering methods based on DBpedia. It provides 5,000 questions and their corresponding SPARQL queries over the DBpedia dataset. Compared with existing datasets, this dataset is larger in size, and has better variety and complexity. The generation process begins with creating SPARQL patterns. Then seed entities are selected, and 2-hop subgraphs are extracted. The SPARQL queries are then interpreted from these subgraphs. 


*2022-07-19*

#### [On Generating Characteristic-rich Question Sets for QA Evaluation](https://doi.org/10.18653/v1/d16-1054)

*Yu Su, Huan Sun, Brian M. Sadler, Mudhakar Srivatsa, Izzeddin Gur, Zenghui Yan, Xifeng Yan*

*EMNLP 2016*

This paper focuses on constructing a dataset for evaluating factoid knowledge base question answering systems. Different from previous work which follows a pipeline of collecting questions then manually characterizing them, this paper proposes a reverse framework by firstly generating graph-structured logical forms and then converting them into questions. In this way, the characteristics of the generated dataset are more controllable. A set of question characteristics (i.e., features) are formalized in this paper, such as structure complexity, function, answer cardinality. It also proposes a new dataset named *GraphQuestions* with over 5,000 question-answer pairs for evaluation.


*2022-07-18*

#### [Constraint-Based Question Answering with Knowledge Graph](https://aclanthology.org/C16-1236/)

*Junwei Bao, Nan Duan, Zhao Yan, Ming Zhou, Tiejun Zhao*

*COLING 2016*

This paper proposes a novel benchmark dataset named *ComplexQuestions* for knowledge base question answering. Compared with previous datasets such as *WebQuestions* with most of the questions can be answered by a single relation in the knowledge base, this new dataset contains more "complex" questions requiring multiple relations to obtain the answer. Besides, this paper also proposes a new question answering method which outperforms the existing methods especially on *ComplexQuestions*. 


*2022-07-17*

#### [The Value of Semantic Parse Labeling for Knowledge Base Question Answering](https://doi.org/10.18653/v1/p16-2033)

*Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, Jina Suh*

*ACL 2016*

This paper also focuses on knowledge base question answering over FreeBase. Compared with existing work such as *Free917* and *WebQuestions* learning from question-answer pairs, this paper uses question-parse (i.e., label) pairs to train the semantic parser. Based on the *WebQuestions* dataset, it constructs a new dataset named *WebQuestionsSP*, which contains semantic parses for the answerable questions in *WebQuestions* over Freebase. Besides, it also demonstrates that with appropriate user interfaces, collecting semantic parse labels can be achieved with relatively low cost. 


*2022-07-16*

#### [Semantic Parsing on Freebase from Question-Answer Pairs](https://aclanthology.org/D13-1160/)

*Jonathan Berant, Andrew Chou, Roy Frostig, Percy Liang*

*EMNLP 2013*

This paper also discusses the task of semantic parsing by learning from question-answer pairs to build a semantic parser over FreeBase. It firstly builds a coarse mapping from phrases to the predicates of FreeBase and large text corpus. Then it generates additional predicates based on neighboring predicates. The proposed semantic parser is evaluated over *Free917* and outperforms the SOTA method. This paper also proposes a new benchmarking dataset *WebQuestions* with almost 6,000 records. 


*2022-07-15*

#### [Large-scale Semantic Parsing via Schema Matching and Lexicon Extension](https://aclanthology.org/P13-1042/)

*Qingqing Cai, Alexander Yates*

*ACL 2013*

Semantic parsing is the task of translating natural language utterances to a formal meaning representation language. This paper applies schema matching to find correspondences between natural language words and ontological symbols, and uses pattern-based regression model to incorporate such pairs into the lexicon of the trained semantic parser. Here, a schema $S = (E, C, R, I)$ consists of an entity set $E$, a category set $C$, a relation set $R$, and an instance set $I$. Besides, such a standard semantic parsing model proposed in this paper can also make use of relevant information such as schema alignments. This paper also releases a dataset *Free917* with 917 question-representation pairs for evaluation.
