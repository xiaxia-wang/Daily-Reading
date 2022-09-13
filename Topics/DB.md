










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
