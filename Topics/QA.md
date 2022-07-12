




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
