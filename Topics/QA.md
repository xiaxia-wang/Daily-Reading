






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
