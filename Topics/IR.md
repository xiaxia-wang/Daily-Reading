






*2022-07-09*

#### [Finding Top-k Answers in Keyword Search over Relational Databases Using Tuple Units](https://doi.org/10.1109/TKDE.2011.61)

*Jianhua Feng, Guoliang Li, Jianyong Wang*

*IEEE TKDE 2011*

This paper investigates the problem of keyword search over relational database tables. It focuses on the computation and indexing of tuple units. Each tuple unit is a set of related tuples containing query keywords and connected by primary/foreign keys. In this paper, the database is modeled as a graph, where nodes represents tuple units, and edges are keys connecting these units. By splitting the original database into tuple units and indexing them, it retrieves and ranks these units to form the query answer. In this paper, 2 kinds of indexes are introduces. One is single-keyword-based tuple index, the other is keyword-pair-based index. 


*2022-07-08*

#### [EASE: An Effective 3-in-1 Keyword Search Method for Unstructured, Semi-structured and Structured Data](https://doi.org/10.1145/1376616.1376706)

*Guoliang Li, Beng Chin Ooi, Jianhua Feng, Jianyong Wang, Lizhu Zhou*

*SIGMOD 2008*

This paper proposes an inverted index method for keyword search over graphs, and presents a ranking method for improving search effectiveness. It regards a r-radius Steiner graph containing all or some of the query keywords as an answer to the input query. A given dataset, as a graph, will firstly be partitioned into a set of r-radius subgraphs. These subgraphs will then be clustered based on similarity, and indexed. Given a keyword query, the indexed subgraphs will be retrieved and ranked using TF-IDF to form an answer list. 


*2022-07-01*

#### [Are There Any Differences in Data Set Retrieval Compared to Well-Known Literature Retrieval?](https://doi.org/10.1007/978-3-319-24592-8_15)

*Dagmar Kern, Brigitte Mathiak*

*TPDL 2015*

This paper investigates the similarities and differences between dataset retrieval and typical literature retrieval. In this paper two user studies are conducted with social sciences dataset and literatures. In the user studies, the participants are interviewed about their retrieval experiences in using a system DBK for retrieving datasets. By analyzing the results, some open problems are discussed, such as search within metadata, grouping of the datasets. Besides, it is found that for social sciences researchers, choosing datasets is a much more important decision than choosing literature. 


*2022-06-30*

#### [A Test Collection for Ad-hoc Dataset Retrieval](https://doi.org/10.1145/3404835.3463261)

*Makoto P. Kato, Hiroaki Ohshima, Ying-Hsang Liu, Hsin-Liang Chen*

*SIGIR 2021*

This resource paper mainly proposes a test collection for ad-hoc dataset retrieval. The datasets are collected from the US and Japanese government portals (data.gov and e-Stat). As a resource, it constructs a test collection containing 192 training topics (i.e., queries), 192 test topics, and over 2,000 qrels. It clearly introduces the construction process of the test collection. The relevance labels are rated by crowd-sourcing workers. This resource is build upon a shared task for retrieval systems. By analyzing the results, the system effectiveness, topic difficulty and variability are discussed. 


*2022-05-20*

#### [Characteristics of Dataset Retrieval Sessions: Experiences from a Real-Life Digital Library](https://doi.org/10.1007/978-3-030-54956-5_14)

*Zeljko Carevic, Dwaipayan Roy, Philipp Mayr*

*TPDL 2020*

This paper discusses the different features when people search for documents and datasets by analyzing search logs of a digital library. It also compares the results with the dataset search query analysis by [Koesten et al.]. In this paper, by comparing the query lengths and other statistics, the main findings include dataset queries are shorter than document queries, which is on the contrary with [Koesten et al.]. Besides, it also counts the number of views and presents interaction sequences for documents and datasets,  respectively. The use of a Sankey diagram to present interaction sequences is interesting. 


*2022-05-16*

#### [A Scalable Virtual Document-Based Keyword Search System for RDF Datasets](https://dl.acm.org/doi/10.1145/3331184.3331284)

*Dennis Dosso, Gianmaria Silvello*

*SIGIR 2019 short*

This paper proposes a "TSA+VDP" pipeline to perform keyword search on a graph, whose output is a (not necessarily connected) subgraph containing all the keywords. It first splits the original graph into a set of subgraphs with limited radius by performing a BFS-like exploration (TSA step). Then for each subgraph, a virtual document is built for retrieval. Given a keyword query, a list of subgraphs are retrieved, merged by overlapping triples, and pruned to be minimal (VDP step). Finally the subgraphs are ranked and output.


*2022-05-10*

#### [Analysis of Long Queries in a Large Scale Search Log](https://dl.acm.org/doi/10.1145/1507509.1507511)

*Michael Bendersky, W. Bruce Croft*

*WSCD@WSDM 2009*

This paper analyzes real-world search log to characterize the queries, search behaviors, and information needs behind them. The search log is MSN search query log excerpt containing about 15M queries and associate clicks. This paper firstly presents statistical distributions of the query length, query types and click data. It divides queries into short queries whose length <= 4 and long queries, and categorizes long queries into 4 types, i.e., questions, operators, composites and non-composites. Then, it analyzes the relationships between pairs of query lengths, types and clicks. In the evaluation, this paper proposes to use click data as relevance labels to evaluate the performances of different retrieval models. Experimental results show that the measurements by click data are consistent with explicit manual relevance judgments.


*2022-05-08*

#### [Context-Aware Document Term Weighting for Ad-Hoc Search](https://dl.acm.org/doi/10.1145/3366423.3380258)

*Zhuyun Dai, Jamie Callan*

*WWW 2020*

Ad-hoc search is to retrieve a list of relevant documents for a given keyword query. Since existing dense embedding models such as BERT cannot handle relatively long texts, bag-of-words (sparse) retrieval is more easy and applicable. This paper proposes a context-aware hierarchical document term weighting framework HDCT, using BERT to generate weights for the terms in each documents, and the weights are used in sparse retrieval. It firstly splits each document into passages, feeds the passages into BERT to generate passage-level term weights, then aggregates the passage-level bag-of-words into document-level weighted term representations to build the inverted index. The experimental results demonstrate HDCT outperforms existing sparse and SOTA dense models.


*2022-05-07*

#### [Binary and graded relevance in IR evaluations--Comparison of the effects on ranking of IR systems](https://doi.org/10.1016/j.ipm.2005.01.004)

*Jaana Kekäläinen*

*IPM 2005*

This paper investigates the correlation between binary relevance and graded relevance in IR evaluation based on TREC test collections. It firstly recruits some students to reassess the relevances of TREC documents on an 0-3 scale, then conducts evaluation based on both rankings. It computes CG, DCG, nDCG, etc., over the rankings, and calculates Kendall correlation value between them. The result shows when fairly and highly relevant documents are given more weight, the correlation diminishes. 


*2022-05-06*

#### [Some Common Mistakes In IR Evaluation, And How They Can Be Avoided](https://dl.acm.org/doi/10.1145/3190580.3190586)

*Norbert Fuhr*

*SIGIR Forum 2017*

This paper points out some mistakes that can be frequently found in IR publications: MRR and ERR violate basic requirements for a metric, MAP is based on unrealistic assumptions, the numbers shown overstate the precision of the result, relative improvements of arithmetic means are inappropriate, the simple holdout method yields unreliable results, hypotheses are often formulated after the experiment, significance tests frequently ignore the multiple comparisons problem, effect sizes are ignored, reproducibility of the experiments might be nearly impossible, and sometimes authors claim proof by experimentation. 
