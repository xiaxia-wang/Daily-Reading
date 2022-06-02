




*2022-06-06*

#### [SaS: SSD as SQL Database System](http://www.vldb.org/pvldb/vol14/p1481-lee.pdf)

*Jong-Hyeok Park, Soyee Choi, Gihwan Oh, Sang Won Lee*

*VLDB 2021 Vision*

This vision paper introduces a new database architecture of making a full SQL database engine run inside SSD rather than on the operating system. Following the introduction, the advantages and design goals are firstly introduced. Then a prototype system is presented with introduction of each component. The prototype is also preliminarily evaluated and analyzed. Finally it ends up with discussions about future research directions. 


*2022-06-05*

#### [Software-Defined Data Protection: Low Overhead Policy Compliance at the Storage Layer is Within Reach!](http://www.vldb.org/pvldb/vol14/p1167-istvan.pdf)

*Zsolt István, Soujanya Ponnapalli, Vijay Chidambaram*

*VLDB 2021 Vision*

This vision paper mainly introduces an idea to decouple the data protection policies from request-level enforcement, and allow the distributed storage nodes to implement the latter. In other words, it uses in-storage computation together with a control-plane/data-plane separation to allow enforcing data protection rules at line-rate. In this paper, following the introduction, the background and related work is firstly presented. Then the software-defined model is introduced,  including the function of each component. After that, a preliminary evaluation is presented to show its effectiveness. Finally, the paper is ended up with discussion about future challenges. 


*2022-06-04*

#### [RPT: Relational Pre-trained Transformer Is Almost All You Need towards Democratizing Data Preparation](http://www.vldb.org/pvldb/vol14/p1254-tang.pdf)

*Nan Tang, Ju Fan, Fangyi Li, Jianhong Tu, Xiaoyong Du, Guoliang Li, Samuel Madden, Mourad Ouzzani*

*VLDB 2021 Vision*

This vision paper introduces a novel idea of relational pre-trained transformer model for data preparation tasks. The data preparation tasks refer to a set of human-easy but computer-hard tasks such as data cleaning, attribute filling, schema matching, entity resolution, etc. The proposed model RPT takes tuples as input. It uses a BERT-like, bidirectional encoder and a GPT-like, left-to-right decoder with tuple-aware masking mechanisms. It is also preliminarily evaluated with table completion task, and compared with BART. In the experiment, since RPT has the same architecture with BART, it directly uses the parameters pre-trained on BART rather than random initializing.


*2022-06-03*

#### [Towards Cost-Optimal Query Processing in the Cloud](http://www.vldb.org/pvldb/vol14/p1606-leis.pdf)

*Viktor Leis, Maximilian Kuschewski*

*VLDB 2021 Vision*

This vision paper introduces a model for determining the optimal configuration of cloud-based database systems. It aims to reduce the costs of query processing, which relates to scanning data,  caching, materializing operators, transferring data between nodes, etc. It begins with a basic model with only 2 variables, i.e., `CPU hours` and `scanned data`, and increasingly adds more variables (until six) to the model. The model is constructed based on intuitive rules (rather than learning). 


*2022-06-02*

#### [The Case for NLP-Enhanced Database Tuning: Towards Tuning Tools that “Read the Manual”](http://www.vldb.org/pvldb/vol14/p1159-trummer.pdf)

*Immanuel Trummer*

*VLDB 2021 Vision*

This vision paper introduces a novel idea of tuning database (parameters or settings, for example) automatically using hints mined from texts. The texts are collected from the Web such as community forums, and used as pre-trained language model. Database tuning aims at optimizing the computation process and improving the efficiency. In this paper, it describes two use cases of NLP-enhanced database tuning, i.e., system configuration and cardinality estimation. It also conducts a simple evaluation of the system configuration task based on hints from Web documents. 


*2022-06-01*

#### [Declarative Data Serving: The Future of Machine Learning Inference on the Edge](http://www.vldb.org/pvldb/vol14/p2555-shaowang.pdf)

*Ted Shaowang, Nilesh Jain, Dennis Matthews, Sanjay Krishnan*

*VLDB 2021 Vision*

This vision paper describes an edge-based computational system architecture. It takes machine learning model serving task as an example. Firstly, it reviews existing supporting technologies and describes an industrial example. Then it discusses the optimization objectives for such an edge-based system, and provides an overview architecture of the system. Besides, it also indicates the research challenges with a research agenda about future work directions. No experiments are conducted. 


*2022-05-31*

#### [Database Technology for the Masses: Sub-Operators as First-Class Entities](http://www.vldb.org/pvldb/vol14/p2483-bandle.pdf)

*Maximilian Bandle, Jana Giceva*

*VLDB 2021 Vision*

This vision paper introduces a new idea of building relational databases and computation processes over sub-operators, which can be regarded as components of existing SQL operators. The authors argue that using sub-operators can make databases more flexible, extend their functionality to more data types, and achieve lower level of abstraction. In this paper, some initial sub-operators are listed, such as scan, map, and loop. Advantages and challenges for related research fields about implementing such sub-operators are also discussed in the paper. 


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

*VLDB 2021 Vision*

This is a vision paper by Facebook AI which proposes an idea of neural database. The motivation is that existing NLP techniques, especially transformers, can handle select-project-join queries which are expressed as short natural language sentences. Neural databases can base on that as a starting point to handle the data with pre-defined schema, though currently they cannot deal with set-based or non-trivial queries. It provides a set of examples to illustrate how the neural database works, strengthen the idea to "ground the vision" and reports some preliminary evaluation results. It also poses a research agenda with potential challenges in the future fulfillment of the comprehensive neural database system.
