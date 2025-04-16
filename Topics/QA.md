












*2025-04-16*

#### [Rule-KBQA: Rule-Guided Reasoning for Complex Knowledge Base Question Answering with Large Language Models](https://aclanthology.org/2025.coling-main.562/)

*Zhiqiang Zhang, Liqiang Wen, Wen Zhao*

*COLING 2025*

This paper proposes a LLM-based KBQA framework with the idea of rule-based reasoning. The induction phase extracts rules from existing data (SPARQL queries) and employs Rule-Following Fine-Tuned (RFFT) LLM to generate additional rules. The deduction phase uses a symbolic agent guided by learned rules to incrementally construct executable logical forms.


*2025-04-12*

#### [Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models](https://arxiv.org/abs/2402.15131)

*Guanming Xiong, Junwei Bao, Wen Zhao*

*ACL 2024*

This paper introduces a KBQA framework that takes natural language questions as input, and produces SPARQL queries using a Thought-Action-Observation iterative pipeline.


*2025-04-06*

#### [Final: Combining First-order Logic with Natural Logic for Question Answering](https://www.computer.org/csdl/journal/tk/5555/01/10926899/25588G8f8ti)

*石继豪，丁效，Siu Cheung HUI，鄢宇雄，赵恒维，刘挺，秦兵*

*IEEE TKDE*

This paper proposes an approach for question answering by incorporating deductive reasoning. It starts from the hypothesis transformed from the question, decomposes the hypothesis into sub-hypotheses, and retrieves facts from the KB to verify them.


*2025-04-04*

#### [Augmented Knowledge Graph Querying leveraging LLMs](https://arxiv.org/abs/2502.01298v1)

*Marco Arazzi, Davide Ligari, Serena Nicolazzo, Antonino Nocera*

*Arxiv 2025*

This paper proposes an LLM-based pipeline for general users to query KG, including KG construction (without LLM), searching for the most appropriate SPARQL template for the query, and execution with result visualization.


*2025-03-04*

#### [Retrieval and Reasoning on KGs: Integrate Knowledge Graphs into Large Language Models for Complex Question Answering](https://aclanthology.org/2024.findings-emnlp.446/)

*Yixin Ji, Kaixin Wu, Juntao Li, Wei Chen, Mingjie Zhong, Xu Jia, Min Zhang*

*EMNLP 2024 Findings*

This paper proposes an approach for enhancing QA performance of LLMs based on knowledge retrieved from KGs. In particular, it uses the triples forming the path from query entity to answer entity for each multi-hop question as a positive example, and ramdomly samples the rest irrelevant triples as negative examples. Besides, to ease the token redundancy of triple-form representation of retrieved subgraphs, it uses YAML-style representation as a more compact form, which is easier for LLMs to process.


*2025-02-19*

#### [Goal-Driven Query Answering for Existential Rules With Equality](https://ojs.aaai.org/index.php/AAAI/article/view/11563)

*Michael Benedikt, Boris Motik, Efthymia Tsamoura*

*AAAI 2018*

The paper presents an approach for goal-driven query answering for existential rules with equality, which gives the benefits of top-down approaches, while radically pruning the set of considered proofs. Additionally, it does not require UNA, but certain steps can be optimized if UNA is satisfied. It also preserves chase termination, and includes an optimized magic set transformation using the symmetry of equality to greatly reduce the number of output rules.


*2025-02-15*

#### [CogMG: Collaborative Augmentation Between Large Language Model and Knowledge Graph](https://aclanthology.org/2024.acl-demos.35)

*Tong Zhou, Yubo Chen, Kang Liu, Jun Zhao*

*ACL 2024 Demo*

This paper proposes a framework for leveraging knowledge graphs to address the limitations of LLMs in QA scenarios, explicitly targeting at problems of incomplete knowledge coverage and knowledge update misalignment. The LLMs identify and decompose required knowledge triples that are not present in the KG, enriching them and aligning updates with real-world demands. For implementation, it fine-tunes an open-sourced LLM and develops an online system.


*2025-01-27*

#### [DAGE: DAG Query Answering via Relational Combinator with Logical Constraints](https://arxiv.org/abs/2410.22105)

*Yunjie He, Bo Xiong, Daniel Hernández, Yuqicheng Zhu, Evgeny Kharlamov, Steffen Staab*

*TheWebConf 2025*

This paper proposes an embedding-based approach for multi-hop QA over KGs, where the query graphs are not limited to tree-forms but can also be DAGs. It presents a plug-and-play module that extends tree-form query embedding approaches with a relational combinator, which is able to merge the potential multiple paths between nodes to obtain the answer embedding.


*2025-01-24*

#### [Harnessing Large Language Models for Knowledge Graph Question Answering via Adaptive Multi-Aspect Retrieval-Augmentation](https://arxiv.org/abs/2412.18537)

*Derong Xu, Xinhang Li, Ziheng Zhang, Zhenxi Lin, Zhihong Zhu, Zhi Zheng, Xian Wu, Xiangyu Zhao, Tong Xu, Enhong Chen*

*AAAI 2025*

This paper introduces an Adaptive Multi-Aspect Retrieval-augmented over KGs framework, which comprises two sub-components: (1) a self-alignment module that aligns commonalities among entities, relations, and subgraphs to enhance retrieved text, thereby reducing noise interference; (2) a relevance gating module that employs a soft gate to learn the relevance score between question and multi-aspect retrieved data, to determine which information be used to enhance LLMs' output.


*2025-01-19*

#### [Knowledge Base Question Answering: A Semantic Parsing Perspective](https://arxiv.org/abs/2209.04994)

*Yu Gu, Vardaan Pahuja, Gong Cheng, Yu Su*

*AKBC 2022*

This paper first introcudes knowledge base question answering, and attributes two unique challenges of KBQA as schema-level complexity and fact-level complexity. Then it situates KBQA in the literature of semantic parsing and analyzes how existing KBQA approaches attempt to address the unique challenges, and how to take further inspiration from semantic parsing.


*2025-01-17*

#### [Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering](https://aclanthology.org/2024.acl-long.397)

*Zhengliang Shi, Shuo Zhang, Weiwei Sun, Shen Gao, Pengjie Ren, Zhumin Chen, Zhaochun Ren*

*ACL 2024*

To handle the task of multi-hop question answering, existing RAG framework is limited by the retriever, and is inevitably affected by the involved noise. In contrast, this paper introduces a generate-then-ground framework, synergizing the parametric knowledge of LLMs and external documents to solve a multi-hop question. It empowers LLMs to alternate two phases until the final answer is derived: (1) formulate a simpler, single-hop question and directly generate the answer; (2) ground the question-answer pair in retrieved documents, amending any wrong predictions in the answer.


*2024-12-30*

#### [TrustUQA: A Trustful Framework for Unified Structured Data Question Answering](https://arxiv.org/abs/2406.18916)

*Wen Zhang, Long Jin, Yushan Zhu, Jiaoyan Chen, Zhiwei Huang, Junjie Wang, Yin Hua, Lei Liang, Huajun Chen*

*AAAI 2024*

This paper proposes a unified NL2Query framework for multiple types of structured data, including tabular data, knowledge graphs, etc. It consists of 3 modules: (1) a dynamic demonstration retriever, which receives the natural language question and generates relavant prompts to get a LLM-generated query, (2) a functional translation process that receives the LLM-generated query and outputs a fixed execution to get the answers, and (3) a conditional graph translator that reforms the original structured data into a uniform graph for execution.


*2024-10-08*

#### CoTKR: Chain-of-Thought Enhanced Knowledge Rewriting for Complex Knowledge Graph Question Answering

*吴亦珂, 黄毅, 胡楠, 花云程, 漆桂林, 陈矫彦, Jeff Z. Pan*

*EMNLP 2024*

To incorporate RAG-LLM into KGQA, existing works need to transform the retrieved subgraph into natural language, which may involve losing or adding irrelevant information. To improve this process, this paper proposes a chain-of-thought enhanced knowledge rewriting approach, which iteratively generates the reasoning paths and corresponding (natural language) results to ensure the accuracy. Besides, it also presents a preference alignment strategy to further optimize the rewriter.


*2024-08-28*

#### [CABINET: Content Relevance based Noise Reduction for Table Question Answering](https://arxiv.org/abs/2402.01155)

*Sohan Patnaik, Heril Changwal, Milan Aggarwal, Sumit Bhatia, Yaman Kumar, Balaji Krishnamurthy*

*ICLR 2024 Spotlight*

For question answering over tables, typically, only a small part of the whole table is relevant to derive the answer, while the irrelevant parts act as distracting noise, resulting in sub-optimal performance due to the vulnerability of LLMs to noise. To improve the performance of LLM on table QA, this paper proposes an unsupervised relevance scorer, trained differentially with the QA LLM, that weighs the table content based on its relevance to the input question before feeding it to the QA LLM. Besides, it employs a weakly supervised module that generates a parsing statement describing the criteria of rows and columns relevant to the question and highlights the content of corresponding table cells.


*2024-08-27*

#### [Temporal Knowledge Question Answering via Abstract Reasoning Induction](https://arxiv.org/abs/2311.09149)

*Ziyang Chen, Dongfang Li, Xiang Zhao, Baotian Hu, Min Zhang*

*ACL 2024*

To improve the LLM performance on temporal knowledge graph reasoning, this paper proposes Abstract Reasoning Induction (ARI) framework, which divides temporal reasoning into two distinct phases: Knowledge-based and Knowledge-agnostic. The knowledge-based phase extracts a temporal knowledge subgraph (one-hop subgraph of a target entity) and generates all possible next-step actions, while the knowledge-agnostic phase learns from historic examples to choose the exact next action to go.


*2024-08-26*

#### [Knowledge Graph Prompting for Multi-Document Question Answering](https://ojs.aaai.org/index.php/AAAI/article/view/29889)

*Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa Siu, Ruiyi Zhang, Tyler Derr*

*AAAI 2024*

This paper proposes an approach for multi-document QA by constructing a graph based on the documents, and applying the LLM to traverse over the graph for answering the question. Specifically, the graph contains three types of nodes, i.e., passage/table/page nodes, and two types of edges, i.e., structural relation, and common keywords/passage similarity relation. Each document is split into passages and encoded for computing similarity to get similarity relations. The table/page nodes are extracted by external APIs and connected by structural relations.


*2024-08-09*

#### [AutoAct: Automatic Agent Learning from Scratch for QA via Self-Planning](https://arxiv.org/abs/2401.05268)

*Shuofei Qiao, Ningyu Zhang, Runnan Fang, Yujie Luo, Wangchunshu Zhou, Yuchen Eleanor Jiang, Chengfei Lv, Huajun Chen*

*ACL 2024*

This paper introduces AutoAct, an automatic agent learning framework for natural language QA with a tool library. It first automatically synthesizes planning trajectories without human assistance or strong closed-source models. Then, it leverages a division-of-labor strategy to automatically differentiate based on the target task information and synthesized trajectories, producing a sub-agent group to complete the task.


*2024-07-15*

#### [BeamAggR: Beam Aggregation Reasoning over Multi-source Knowledge for Multi-hop Question Answering](https://arxiv.org/abs/2406.19820)

*Zheng Chu, Jingchang Chen, Qianglong Chen, Haotian Wang, Kun Zhu, Xiyuan Du, Weijiang Yu, Ming Liu, Bing Qin*

*ACL 2024*

This paper proposes a beam-search like framework for multi-hop question answering. Specifically, it first decomposes the natural language questions into tree-shape, where each edge indicates a simple question. Then for each simple question, it aggregates the answers from multiple sources, including close-book generation, web search, wikipedia retrieval, etc., and retains several candidates of answers for the next simple question. In the end, it outputs the candidate with top scores retained in the beam search process.


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
