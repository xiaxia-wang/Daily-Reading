




*2022-05-30*

#### [Tensors: An abstraction for general data processing](http://www.vldb.org/pvldb/vol14/p1797-koutsoukos.pdf)

*Dimitrios Koutsoukos, Supun Nakandala, Konstantinos Karanasos, Karla Saur, Gustavo Alonso, Matteo Interlandi*

*VLDB 2021 Vision*

This vision paper presents a new idea of using tensors, which are basic components of deep leaning models, to achieve non-ML data processing tasks, such as graph processing and relational operations. As a preliminary attempt, in this paper it firstly presents using tensors to compute PageRank on the graph, and compares the performance with typical implementation. Besides, it also implements relational operators using tensors, and evaluates them with cardinality calculation. The result shows the tensor implementation performs quite well in these tasks thus being promising for future work. 


*2022-05-18*

#### [NOAH: Interactive Spreadsheet Exploration with Dynamic Hierarchical Overviews](http://www.vldb.org/pvldb/vol14/p970-rahman.pdf)

*Sajjadur Rahman, Mangesh Bendre, Yuyang Liu, Shichu Zhu, Zhaoyuan Su, Karrie Karahalios, Aditya G. Parameswaran*

*VLDB 2021*

"This paper can be used as a model for similar papers to profile a system or investigate user interactions." - by GC

This paper introduces a novel system for the user to interactively explore large spreadsheets with hierarchical overviews. The introduction mainly presents the motivation of this work, including limitations of existing systems, supporting researches and the contributions of this paper. Then it describes the design consideration and use cases of the system with examples. The details and methods for each component are introduced in the next section, followed by the system architecture. Then two user studies for verifying system effectiveness are presented with discussion of results. Finally it ends up with related work and conclusion. 


*2022-05-13*

#### [Top-k Set Similarity Joins](https://doi.org/10.1109/ICDE.2009.111)

*Chuan Xiao, Wei Wang, Xuemin Lin, Haichuan Shang*

*ICDE 2009*

This paper introduces an algorithm to identify top-k similar record pairs from a large given corpus, which doesn't require a pre-defined similarity threshold. It is adapted from an existing similarity join algorithm named All-Pairs, by changing the current threshold while keeping top-k similar pairs, and devising a new stopping condition. In the experiment, this top-k join algorithm was compared with ppjoin-topk on several large datasets. 


*2022-04-29*

#### [A Framework to Conduct and Report on Empirical User Studies in Semantic Web Contexts](https://doi.org/10.1007/978-3-030-03667-6_36)

*Catia Pesquita, Valentina Ivanova, Steffen Lohmann, Patrick Lambrix*

*EKAW 2018*

This paper analyses and discusses the general framework of Semantic Web user studies by reviewing 87 conference papers. The authors read and categorize the papers from 6 aspects, namely, (1) purpose, (2) users, (3) tasks, (4) setup, (5) procedure and (6) analysis and presentation of data. For each aspect, this paper presents the distributions of the reviewed paper with brief analysis. According to the analysis results, it also talks about an appropriate user study design for Semantic Web related systems, also following the six aspects/steps. Finally, it gives some suggestions and indicates challenges for conducting user studies in practice.


*2022-04-26*

#### [From Natural Language Processing to Neural Databases](https://dl.acm.org/doi/10.14778/3447689.3447706)

*James Thorne, Majid Yazdani, Marzieh Saeidi, Fabrizio Silvestri, Sebastian Riedel, Alon Y. Levy*

*VLDB Vision 2021*

This is a vision paper by Facebook AI which proposes an idea of neural database. The motivation is that existing NLP techniques, especially transformers, can handle select-project-join queries which are expressed as short natural language sentences. Neural databases can base on that as a starting point to handle the data with pre-defined schema, though currently they cannot deal with set-based or non-trivial queries. It provides a set of examples to illustrate how the neural database works, strengthen the idea to "ground the vision" and reports some preliminary evaluation results. It also poses a research agenda with potential challenges in the future fulfillment of the comprehensive neural database system.
