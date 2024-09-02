




*2024-09-02*

#### [nsDB: Architecting the Next Generation Database by Integrating Neural and Symbolic Systems](https://www.vldb.org/pvldb/vol17/p3283-tang.pdf)

*Ye Yuan, Bo Tang, Tianfei Zhou, Zhiwei Zhang, Jianbin Qin*

*VLDB 2024*

This paper designs a neural-symbolic database system that aims to integrate neural and symbolic architectures natively to address the weaknesses of both. It consists of 5 major components: (i) user interface, (ii) model manager, (iii) query optimizer, (iv) execution engine, and (v) storage engine, and these components are loosely decoupled to support diverse analytical queries efficiently.


*2024-06-17*

#### [CHORUS: Foundation Models for Unified Data Discovery and Exploration](https://arxiv.org/abs/2306.09610)

*Moe Kayali, Anton Lykov, Ilias Fountalis, Nikolaos Vasiloglou, Dan Olteanu, Dan Suciu*

*VLDB 2024*

This paper applies foundation models (LLMs) to data discovery and exploration tasks, including table-class
detection, column-type annotation and join-column prediction. Specifically, it utilizes 6 types of model inputs in the experiments, including instructions, demonstrations, data samples, metadata, task-specific knowledge, and prefixes.


*2023-04-11*

#### [Querying Large Language Models with SQL[Vision]](https://arxiv.org/pdf/2304.00472.pdf)

*Mohammed Saeed, Nicola De Cao, Paolo Papotti*

*Arxiv 2023*

This vision paper introduces a new possibility of implementing SQL with LLMs. To leverage the LLMs for SQL query, it firstly decomposes the original query into components. For each components, it uses LLMs in an in-context learning manner to get the query results. The LLMs generally perform as an underlying source of tuple sets, providing relevant answers/intermediate information for the query.


*2022-12-12*

#### [Mapping Relational Database Constraints to SHACL](https://doi.org/10.1007/978-3-031-19433-7_13)

*Ratan Bahadur Thapa, Martin Giese*

*ISWC 2022*

This paper discusses the relational to RDF (R2R) transformation and its constraint rewriting. It firstly introduces two fundamental properties of constraint rewriting, i.e., maximal semantics preservation and monotonicity. Then it proposes a constraint rewriting method which satisfies both properties. 


*2022-10-08*

#### [BABOONS: Black-Box Optimization of Data Summaries in Natural Language](https://www.vldb.org/pvldb/vol15/p2980-trummer.pdf)

*Immanuel Trummer*

*VLDB 2022*

This paper proposes BABOONS, a model for generating comparative data summaries based on a user-given utility target. Basically, the data source is characterized by a relational table *D*, a set of dimension columns and a set of aggregation columns. A summary template *T* specifies how abstract facts are translated into natural language texts by providing text snippets with placeholder for values. The problem is to obtain summaries for a set of data items *I* based on *D* and *T* that maximize mean utility over *I*. To achieve this, BABOONS applies deep reinforcement learning for optimization and uses proactive caching to expedite the computation. 


*2022-09-22*

#### [Generating Titles for Web Tables](https://doi.org/10.1145/3308558.3313399)

*Braden Hancock, Hongrae Lee, Cong Yu*

*WWW 2019*

This paper provides a method to generate titles to describe table contents. It is based on a sequence-to-sequence model, using text snippets extracted from the table as the input, and applies a copy mechanism in the decoder to improve the readability. The extracted text snippets include metadata fields such as page title, section headings, prefixes, and the table rows. Then it concatenates these fields into a sequence and feeds it to the model. Besides, the copy mechanism in the decoder helps retain some rare words such as human names related to the table content. 


*2022-09-20*

#### [Automatically Generating Interesting Facts from Wikipedia Tables](https://dl.acm.org/doi/10.1145/3299869.3314043)

*Flip Korn, Xuezhi Wang, You Wu, Cong Yu*

*SIGMOD 2019*

This paper proposes a method to extract "fun facts" from relational tables in Wikipedia. A fun fact is a fact-based sentence to be presented in an entity card. To extract fun facts from the tables, this paper proposes a template-based method with two kinds of templates, namely, *rank-ordered* and *distributional*. It begins with the tables in Wikipedia’s “Lists of Superlatives” table category, by firstly collecting curated templates created by human. Based on that, it then derives templates for new tables using a language model.


*2022-09-19*

#### [Interactive Summarization and Exploration of Top Aggregate Query Answers](https://doi.org/10.14778/3275366.3275369)

*Yuhao Wen, Xiaodan Zhu, Sudeepa Roy, Jun Yang*

*VLDB 2018*

This paper proposes a summarization method for database query results, to ease the user with comprehension and exploration. In this paper, a query result summary is considered to be simple, diverse and discriminative. Based on that, the summary is designed to consists k (a given number) clusters and cover top-l records. The distance between each pair of clusters is no less than a given threshold to ensure the diversity. It applies greedy algorithms to solve the problem.  Besides, the interactivity of the method means it allows the user to customize the parameters in the problem. 


*2022-09-18*

#### [Probabilistic Database Summarization for Interactive Data Exploration](https://doi.org/10.14778/3115404.3115419)

*Laurel J. Orr, Dan Suciu, Magdalena Balazinska*

*VLDB 2017*

This paper proposes a probabilistic approach to generate database summaries based on maximum entropy. Such a summary is able to approximately answer linear database queries such as COUNT, which can be represented as a dot product. It also proposes three optimizations for summary compression, query processing and statistics selection, respectively.  


*2022-09-17*

#### [Comprehension-Based Result Snippets](https://doi.org/10.1145/2396761.2398405)

*Abhijith Kashyap, Vagelis Hristidis*

*CIKM 2012*

This paper investigates the problem of generating snippets for structured data query results. Unlike the previous work which typically uses the textual length of snippets to measure the difficulty for the user's comprehension, this paper considers the comprehension effort as the cost of locating specific attributes based on the user's interest. Therefore, the number and location of the attributes (in the snippet) are optimized in the proposed snippet generation method. It designs an attribute level informativeness for the snippet, formulates the generation as an optimization problem, and applies heuristic-based algorithm to solve the problem.


*2022-09-16*

#### [Summary Graphs for Relational Database Schemas](http://www.vldb.org/pvldb/vol4/p899-yang.pdf)

*Xiaoyan Yang, Cecilia M. Procopiuc, Divesh Srivastava*

*VLDB 2011*

This paper proposes to generate a query-relevant summary graph for a relational database. Given a set of query-related tables hit by the user's query, the generation algorithm automatically selects some relevant (not hit) tables and joins to be incorporated in the result summary. The number of relevant tables is limited by the size constraint, and the informativeness of join edges is maximized. The problem is formulated and solved as an integer program. 


*2022-09-15*

#### [Summarizing Relational Databases](https://doi.org/10.14778/1687627.1687699)

*Xiaoyan Yang, Cecilia M. Procopiuc, Divesh Srivastava*

*VLDB 2009*

Motivated by the difficulty of understanding the unfamiliar schema of database tables before querying, this paper proposes an approach to summarize the information contained in a relational database. The approach has three main components: (1) defining the importance of each table based on random walk, (2) proposing a metric space to compute the distances between the tables, (3) using a k-center algorithm to cluster the tables, and returns the centers of the clusters as the summarization result. 


*2022-09-14*

#### [MDL Summarization with Holes](http://www.vldb.org/archives/website/2005/program/paper/wed/p433-bu.pdf)

*Shaofeng Bu, Laks V. S. Lakshmanan, Raymond T. Ng*

*VLDB 2005*

This paper proposes a (database) query result summarization method based on Minimum Description Length (MDL). It mainly considers the trade-off between generalizing the query results and the number of exceptions. For cases with high dimension ($\geq 2$), the problem is NP-hard. To solve this, it also proposes greedy and dynamic programming algorithms to improve the time efficiency. 



*2022-09-13*

#### [Schema Summarization](http://dl.acm.org/citation.cfm?id=1164156)

*Cong Yu, H. V. Jagadish*

*VLDB 2006*

This paper proposes to generate a schema summary for a given database. It mainly focuses on three main features of an ideal summary: (1) a relatively small size, (2) containing important elements, (3) achieving a broad coverage. To achieve this, it designs the local importance of each schema element (similar to PageRank), and aggregates the importance of the elements in the summary as the overall importance score. The covered elements are counted by instantiation. The affinity of the schema elements is measured by the distance on the graph and the number of common instances. 
