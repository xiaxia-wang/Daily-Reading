





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
