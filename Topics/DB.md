








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
