









*2022-10-06*

#### [Entity Summarization with User Feedback](https://link.springer.com/chapter/10.1007/978-3-030-49461-2_22)

*Qingxia Liu, Yue Chen, Gong Cheng, Evgeny Kharlamov, Junyou Li, Yuzhong Qu*

*ESWC 2020*

This paper investigates the problem of using the user's feedback to improve the entity summarization result. It formulates this problem as a Markov decision process and the entity summarizer as a reinforcement learning agent. The interdependence of triples in the current summary and the user feedback is represented by a deep neural network. In the experiment, the authors conduct both online user study and offline evaluation based on ESBM which is an entity summarization benchmark. 


*2022-10-05*

#### [An Empirical Evaluation of Techniques for Ranking Semantic Associations](https://doi.org/10.1109/TKDE.2017.2735970)

*Gong Cheng, Fei Shao, Yuzhong Qu*

*TKDE 2017*

This paper evaluates 8 techniques for ranking semantic associations based on 1,200 ground-truth rankings created by 30 human experts. To evaluate the semantic associations, this paper divides the techniques into data-centric and user-centric ones. For data-centric techniques, it implements 8 metrics such as size and frequency. In the human evaluation, the participants are required to conduct pair-wise comparison to the importance of the semantic associations, and give brief explanation to their judgments. 


*2022-10-04*

#### [ESBM: An Entity Summarization BenchMark](https://doi.org/10.1007/978-3-030-49461-2_32)

*Qingxia Liu, Gong Cheng, Kalpa Gunaratna, Yuzhong Qu*

*ESWC 2020*

This paper proposes a benchmark for evaluating general-purpose entity summarization. ESBM contains 175 heterogeneous entities sampled from DBpedia and LinkedMDB, with 30 human experts creating 2,100 ground-truth summaries under 2 size constraints. Besides, it also compares the performance of 9 existing entity summarization methods, and implements a supervised learning-based method in the experiments. 


*2022-10-03*

#### [Generating Compact and Relaxable Answers to Keyword Queries over Knowledge Graphs](https://doi.org/10.1007/978-3-030-62419-4_7)

*Gong Cheng, Shuxin Li, Ke Zhang, Chengkai Li*

*ISWC 2020*

The problem of keyword search on the graph is usually formulated as a GST problem. However, a GST that connects all the query keywords may not exist in practice. To address the problem, this paper proposes to generate compact but relaxable (i.e., containing only a subset of keywords) subgraphs as answers. It is formulated as an optimization problem of computing a minimally relaxed answer with a compactness guarantee, and is solved with a best-first search algorithm. 


*2022-10-02*

#### [Fast Algorithms for Semantic Association Search and Pattern Mining](https://doi.org/10.1109/TKDE.2019.2942031)

*Gong Cheng, Daxin Liu, Yuzhong Qu*

*IEEE TKDE 2021*

This paper investigates the problems of semantic association (SA) search, and semantic association pattern (SAP) mining. Given a set of related entities, a SA is a compact subgraph connecting all the entities, while a SAP is the conceptual graph pattern corresponding to the SA. This paper proposes fast algorithms for searching SAs and mining frequent SAPs. It designs a canonical code for each SA to help sort them in search, along with a distance oracle for pruning the search space. For frequent SAP mining, it applies a partition-based SA filtering method to optimize the SAP enumeration process. 


*2022-10-01*

#### [Fast and Practical Snippet Generation for RDF Datasets](https://doi.org/10.1145/3365575)

*Daxin Liu, Gong Cheng, Qingxia Liu, Yuzhong Qu*

*TWeb 2019*

This is an extension of the WSDM 2017 paper. This paper discusses the snippet generation method IlluSnip and its two optimized versions called OPT-MAX and OPT-RANK by changing the ways of iteration and the starting triple. In the second part of the paper, it also proposes an adaptation to Web-based query service, i.e., for SPARQL endpoint. Instead of directly manipulating the dataset, it submits queries to the SPARQL endpoint and uses the answers to construct the dataset snippet. 


*2022-09-30*

#### [Searching Linked Objects with Falcons: Approach, Implementation and Evaluation](https://doi.org/10.4018/jswis.2009081903)

*Gong Cheng, Yuzhong Qu*

*International Journal on Semantic Web and Information Systems 2009*

This paper introduces the keyword search strategy applied in Falcons and its implementation. It firstly illustrates the search functions within several scenarios. Then it presents the system architecture, which involves metadata, inverted indexes with the retrieval component, query parsing and snippet generation. In its implementation, it incorporates type information discovery using class-inclusive reasoning. 


*2022-09-27*

#### [Diversified and Verbalized Result Summarization for Semantic Association Search](https://doi.org/10.1007/978-3-030-02922-7_26)

*Yu Gu, Yue Liang, Gong Cheng, Daxin Liu, Ruidi Wei, Yuzhong Qu*

*WISE 2018*

This paper investigates the problem called semantic association search, which extracts a subgraph containing a set of connected entities in the given query. This paper focuses on reducing the redundancy of top-ranked graph patterns by formulating a combinatorial optimization model to select top-*k* diversified patterns. Besides, to ease user's comprehension, it also proposes to translate graph patterns to textual sentences based on a set of rules. 


*2022-09-24*

#### [VISION-KG: Topic-centric Visualization System for Summarizing Knowledge Graph](https://doi.org/10.1145/3336191.3371863)

*Jiaqi Wei, Shuo Han, Lei Zou*

*WSDM 2020 demo*

This demo paper introduces a visualization method to profile entities in a knowledge graph. For each entity given as a query, the system extracts a summary graph by ranking the related facts with importance, relatedness and diversity measures. Further, the system will also split the summary graph with respect to different topics, where each subgraph corresponds to a specific topic. Each subgraph is a cluster of nodes gathered using an approximate k-NN algorithm. 


*2022-09-23*

#### [Query-Based Entity Comparison in Knowledge Graphs Revisited](https://link.springer.com/chapter/10.1007/978-3-030-30793-6_32)

*Alina Petrova, Egor V. Kostylev, Bernardo Cuenca Grau, Ian Horrocks*

*ISWC 2019*

Entity comparison is to describe the similarities between two given entities in a knowledge graph. This paper improves a previous method which formulates the problem as to find a SPARQL query having both entities in its answer. In this paper, it also considers the similarity queries with numeric filter expressions. The optimize target is to compute the most specific and exact similarity queries for a pair of given entities. To achieve this, it also proposes an approximate algorithm for the problem.


*2022-09-21*

#### [Data summarization: a survey](https://doi.org/10.1007/s10115-018-1183-0)

*Mohiuddin Ahmed*

*Knowledge and Information Systems 2019*

This survey paper introduces the summarization categories and methods for structured and unstructured data. It firstly proposes a taxonomy of data summarization. Structured data summarization methods are categorized into machine learning, statistical and semantics ones, while unstructured data summarization is related to several research areas including NLP, graph spreading, etc. It also introduces some representative work in each kind, and discusses common evaluation metrics. 


*2022-09-02*

#### [RDF Data Storage and Query Processing Schemes: A Survey](https://doi.org/10.1145/3177850)

*Marcin Wylot, Manfred Hauswirth, Philippe Cudré-Mauroux, Sherif Sakr*

*ACM Computing Surveys 2018*

This paper reviews the RDF data storage, indexing and query processing techniques. It summarizes existing RDF data management systems, and categorizes them into centralized and distributed ones. For each kind of system, it also introduces some examples, along with their architecture and specific techniques. In the end, it also introduces some benchmarking RDF systems. 


*2022-08-26*

#### [Summarizing Ontology-based Schemas in PDMS](https://doi.org/10.1109/ICDEW.2010.5452706)

*Carlos Eduardo S. Pires, Paulo Orlando Queiroz-Sousa, Zoubida Kedad, Ana Carolina Salgado*

*ICDE Workshops 2010*

This paper introduces a summarization approach for peer data management system (PDMS). In a PDMS system, peers can be semantically clustered, and each cluster can be represented by its schema. Firstly, to cluster the peers based on the ontology, it considers the centrality and frequency of each concept in the local schema. Then the paths between different clusters of concepts are identified. The clusters and relations jointly form the output summary. In the experiment, the proposed summaries are compared with gold standards generated by human experts.


*2022-08-25*

#### [Summarizing Linked Data RDF Graphs Using Approximate Graph Pattern Mining](https://doi.org/10.5441/002/edbt.2016.86)

*Mussab Zneika, Claudio Lucchese, Dan Vodislav, Dimitris Kotzinos*

*EDBT 2016 Poster*

This is a poster paper reporting the [RDF summarization method based on approximate binary patterns](https://link.springer.com/chapter/10.1007/978-3-319-43862-7_4). 


*2022-08-24*

#### [Relatedness-based Multi-Entity Summarization](https://doi.org/10.24963/ijcai.2017/147)

*Kalpa Gunaratna, Amir Hossein Yazdavar, Krishnaprasad Thirunarayan, Amit P. Sheth, Gong Cheng*

*IJCAI 2017*

This paper proposes an idea of generating a summary for a set of related entities in a knowledge graph, instead of generating isolated summaries for each of them. For the set of entities, the approach focuses on (1) inter-entity facts that are similar, and (2) intra-entity facts that are important and diverse. It is formulated as a constrained knapsack problem and solved with an efficient approximate algorithm. 


*2022-08-23*

#### [RDF Graph Summarization Based on Approximate Patterns](https://link.springer.com/chapter/10.1007/978-3-319-43862-7_4)

*Mussab Zneika, Claudio Lucchese, Dan Vodislav, Dimitris Kotzinos*

*ISIP 2015*

This paper proposes a RDF summarization method based on top-k approximate patterns. In this paper, the original RDF graph is firstly transformed into a binary matrix. Then based on the top-k binary pattern mining method PaNDa+, a set of binary patterns are extracted. They are further reconstructed to be RDF graph patterns as the result summary. 


*2022-08-22*

#### [Quality metrics for RDF graph summarization](https://doi.org/10.3233/SW-190346)

*Mussab Zneika, Dan Vodislav, Dimitris Kotzinos*

*Semantic Web Journal 2019*

This paper introduces a set of metrics to evaluate the quality of RDF graph summaries. These metrics mainly focus on the precision and recall (i.e., in IR style) of the classes/properties/instances in the summary based on gold summary or some given schema. It also discusses some existing RDF summarization algorithms such as ExpLOD, and evaluates the quality of the generated summaries. 


*2022-08-21*

#### [Generating Preview Tables for Entity Graphs](https://doi.org/10.1145/2882903.2915221)

*Ning Yan, Sona Hasani, Abolfazl Asudeh, Chengkai Li*

*SIGMOD 2016*

This paper discusses the task of generating preview tables for entity graphs. Given an entity-relation graph and its schema graph which contains the corresponding properties and classes, preview table generation aims to extract a few key and non-key attributes, and construct a set of tables as a graph preview. This paper also proposes scoring methods for preview tables and algorithms for generating them. 


*2022-08-20*

#### [Evaluation Metrics](https://hal.inria.fr/inria-00174152)

*Jovan Pehcevski, Benjamin Piwowarski*

*Encyclopedia of Database Systems 2007*

This report reviews a set of metrics for evaluating XML retrieval systems, including INEX, XCG, etc. It also analyzes and categories the metrics based on their properties such as recall, precision, overlap and ideality. 


*2022-08-19*

#### [Efficiency and Precision Trade-Offs in Graph Summary Algorithms](https://doi.org/10.1145/2513591.2513654)

*Stéphane Campinas, Renaud Delbru, Giovanni Tummarello*

*IDEAS 2013*

This paper discusses the trade-off between the size of graph summary (also, the efficiency to generate graph summary) and the accuracy of keeping the original graph information. In this paper, the summary of graph data is formulated as a quotient graph. Several rules with different level of equivalence are compared. The edge precision and errors are evaluated and discussed.


*2022-08-17*

#### [Understanding the Structure of Knowledge Graphs with ABSTAT Profiles](https://www.semantic-web-journal.net/content/understanding-structure-knowledge-graphs-abstat-profiles-1)

*Blerina Spahiu, Matteo Palmonari, Renzo Arturo Alva Principe, Anisa Rula*

*Semantic Web Jorunal (under review)*

This journal paper is an extended version of the previous conference paper that proposes ABSTAT. It also introduces a user study to evaluate the effectiveness of ABSTAT. In the user study, the participants are required to complete SPARQL query patterns for given datasets w/ or w/o the help of ABSTAT profiles.


*2022-08-16*

#### [ABSTAT: Ontology-Driven Linked Data Summaries with Pattern Minimalization](https://doi.org/10.1007/978-3-319-47602-5_51)

*Blerina Spahiu, Riccardo Porrini, Matteo Palmonari, Anisa Rula, Andrea Maurino*

*ESWC 2016*

This paper proposes ABSTAT, an ontology-based summarization framework for linked datasets. Based on the ontology especially the subClass information, it extracts a set of triple patterns as the summary for each dataset. The property and the minimal type of the subject and object are retained in each triple pattern.


*2022-08-09*

#### [Efficient Semantic-Aware Detection of Near Duplicate Resources](https://doi.org/10.1007/978-3-642-13489-0_10)

*Ekaterini Ioannou, Odysseas Papapetrou, Dimitrios Skoutas, Wolfgang Nejdl*

*ESWC 2010*

This paper proposes a detection method to identify near duplicate RDF resources based on their RDF representations and the literal and structural information. Detecting duplicate RDF resources, in this paper, is formulated as finding the resources whose similarity is higher than a given threshold. Each RDF resource is represented as a graph containing all the triples describing the resource. The graph is transformed into a list-shape representation, and indexed using Locality-Sensitive Hashing (LSH). The indexes are organized as a B-tree to accelerate the search process. This paper also analyzes the trade-off between the prefix length and the false positive rates. 


*2022-08-06*

#### [Utilizing Resource Importance for Ranking Semantic Web Query Results](https://link.springer.com/chapter/10.1007/978-3-540-31839-2_14)

*Bhuvan Bamba, Sougata Mukherjea*

*SWDB 2004*

This paper proposes a ranking technique for semantic Web resources using link analyses. It firstly builds a PropertyGraph and an IsaGraph to represent the relations between the resources, properties and classes. Then it modifies HITS to compute the subjectivity and objectivity of the resources. The overall importance of each resource is a linear combination of these factors. For ranking the semantic Web query results, it computes a score for each result graph, which is a combination of scores of resources and properties (using Inverse Property Frequency, IPF) contained in the graph.


*2022-08-05*

#### [Using Naming Authority to Rank Data and Ontologies for Web Search](https://doi.org/10.1007/978-3-642-04930-9_18)

*Andreas Harth, Sheila Kinsella, Stefan Decker*

*ISWC 2009*

Each semantic Web resource, represented as a URI, has a naming authority. The naming authority is usually the name of the dereferenced document, or it can also be the name of pay-level domain. Naming authorities can reference each other to create links by importing documents. In this way, this paper creates a naming authority graph and computes PageRank over the graph to rank the naming authorities. Using their scores, the ranking score of each resource (URI) is the total score of all naming authorities which have referenced it. 


*2022-08-04*

#### [ReConRank: A Scalable Ranking Method for Semantic Web Data with Context](https://web.archive.org/web/20170808045350/https://aran.library.nuigalway.ie/bitstream/handle/10379/492/paper.pdf?sequence=1)

*Aidan Hogan, Andreas Harth, Stefan Decker*

*2nd Workshop on Scalable Semantic Web Knowledge Base Systems (2006)*

This paper proposes a ranking method for semantic data resources. It creates a heterogeneous graph where nodes can be either a resource or a RDF graph. With the bi-directional links between the resources and RDF graphs, it utilizes the contextual information of the resources. Then it proposes a modified PageRank method to rank the resources on the graph. 


*2022-08-03*

#### [Ranking Ontologies with AKTiveRank](https://doi.org/10.1007/11926078_1)

*Harith Alani, Christopher Brewster, Nigel Shadbolt*

*ISWC 2006*

This paper presents AKTiveRank, a prototype system for ranking ontologies based on structural metrics. It is designed to be used at the back end of ontology search engines (e.g. Swoogle). The query submitted to the search engine is used to identify the concepts that match the user’s request. The ranking measures are based on the representation of those concepts and their neighborhoods.


*2022-08-02*

#### [Query-Independent Learning to Rank for RDF Entity Search](https://doi.org/10.1007/978-3-642-30284-8_39)

*Lorand Dali, Blaz Fortuna, Duc Thanh Tran, Dunja Mladenic*

*ESWC 2012*

This paper proposes to rank entities using learning to rank. Given a structured query, lots of entities with equal substructures are returned as potential answers. Therefore, it is useful to incorporate query-independent features to enhance learning to rank. This paper adopts the frequency of human accessing the entities from the search logs as a feature to train L2R model, and achieves better performance compared with baselines such as PageRank. 


*2022-08-01*

#### [NAGA: Searching and Ranking Knowledge](https://doi.org/10.1109/ICDE.2008.4497504)

*Gjergji Kasneci, Fabian M. Suchanek, Georgiana Ifrim, Maya Ramanath, Gerhard Weikum*

*ICDE 2008*

This paper proposes a semantic search engine built over a Web-based knowledge graph. By organizing the entities as nodes and relations as edges, it searches proper answer subgraphs using graph queries. Then it proposes a ranking model for the answer graphs with respect to confidence, informativeness and compactness. It evaluates the search system with user studies and TREC datasets. 


*2022-07-31*

#### [Learning to Rank for Semantic Search](https://link.springer.com/chapter/10.1007/978-3-319-56608-5_60)

*Luca Soldaini, Nazli Goharian*

*ECIR 2017*

This paper considers the ranking features for RDF resources, including the widely-used PageRank, and other domain and query-independent features. It combines these features based on learning-to-rank, and compares their ranking performances over DBpedia and YAGO.


*2022-07-30*

#### [Hierarchical Link Analysis for Ranking Web Data](https://doi.org/10.1007/978-3-642-13489-0_16)

*Renaud Delbru, Nickolai Toupikov, Michele Catasta, Giovanni Tummarello, Stefan Decker*

*ESWC 2010*

This paper introduces a ranking approach for entities. The computation method is divided into two layers. Firstly, a global graph is introduced on the dataset layer, where datasets are the nodes linked by properties. Secondly, each dataset is considered as a individual graph where the nodes represent entities. Using the "global" rank computed on the dataset graph and the "local" rank computed over each dataset graph, the two parts are combined to rank the entities. 


*2022-07-29*

#### [A survey of approaches for ranking on the web of data](https://doi.org/10.1007/s10791-014-9240-0)

*Antonio J. Roa-Valverde, Miguel-Ángel Sicilia*

*Informational Retrieval 2014*

This survey paper investigates the problem of ranking linked data. It firstly identifies some open challenges related to this problem, e.g., how to deal with the large size and the heterogeneous nature of linked data, the query execution, and result consolidation. Then it classifies existing ranking approaches based on four aspects, namely, query dependency, granularity, features and heuristics. It also analyzes some SOTA approaches and semantic search engines, and reviews evaluation approaches for ranking.


*2022-07-28*

#### [A review of ranking approaches for semantic search on Web](https://doi.org/10.1016/j.ipm.2013.10.004)

*Vikas Jindal, Seema Bawa, Shalini Batra*

*IP&M 2014*

This paper reviews the semantic ranking approaches and compares their features with traditional keyword search methods. It firstly analyzes the differences between semantic-based search and keyword-based search, and introduces the motivation to review ranking approaches. It classifies existing ranking methods into 3 categories, (1) entity ranking, (2) relationship ranking, and (3) semantic document ranking. It also discusses 6 important features for ranking semantic documents such as heterogeneity, portability, and evaluation benchmarks. 


*2022-07-27*

#### [A Hybrid Approach for Searching in the Semantic Web](https://doi.org/10.1145/988672.988723)

*Cristiano Rocha, Daniel Schwabe, Marcus Poggi de Aragão*

*WWW 2004*

This paper proposes a search method over ontology based on typical search and involving spread activation techniques. It firstly assigns initial weights to each concept, i.e., nodes on the graph, based on typical search relevance. Then it assigns a weight to each edge to characterize the strength of relations. Using the edge weights it spreads the concepts' weight to neighboring related concepts, and retains ones with the highest relevances as the search result. 


*2022-07-26*

#### [A survey on semantic schema discovery](https://doi.org/10.1007/s00778-021-00717-x)

*Kenza Kellou-Menouer, Nikolaos Kardoulakis, Georgia Troullinou, Zoubida Kedad, Dimitris Plexousakis, Haridimos Kondylakis*

*VLDB J 2022*

This survey paper studies the problem of semantic schema discovery. The motivation is that given RDF datasets' irregularity such as incomplete schema definition, schema discovery aims to enrich the data and facilitate the user to understand and make use of them. It categorizes existing methods into 3 kinds, (1) ones that exploit implicit data structures, (2) ones that use explicit schema and enrich them, (3) ones that discover structural patterns. By comparing these approaches, it summarizes the advantages, limitations, and remaining open problems. 


*2022-07-25*

#### [Improving Curated Web-Data Quality with Structured Harvesting and Assessment](https://doi.org/10.4018/ijswis.2014040103)

*Kevin Chekov Feeney, Declan O'Sullivan, Wei Tai, Rob Brennan*

*International Journal on Semantic Web and Information Systems 2014*

This paper proposes a framework for maintaining and curating linked datasets with quality control. It divides the data process workflow into several layers with different purposes and managers, such as the data architects manage the schema and update them, the domain experts act mainly as consultants and provide facts. Each role is incorporated and closely related to the layer and proper tasks. This paper also evaluates the quality of linked datasets under the curating framework. 


*2022-07-24*

#### [Charaterizing RDF graphs through graph-based measures - framework and assessment](https://doi.org/10.3233/SW-200409)

*Matthäus Zloch, Maribel Acosta, Daniel Hienert, Stefan Conrad, Stefan Dietze*

*Semantic Web 2021*

This paper is an extended version of a previous paper which implements a set of statistical graph metrics. In this paper, quantitative graph metrics for linked datasets are proposed and categorized into groups such as degree-based, edge-based, centrality, etc. The corresponding software implementations are also proposed. These metrics are then applied to the LOD cloud datasets to analyze their features.


*2022-07-12*

#### [Entity summarization: State of the art and future challenges](https://www.sciencedirect.com/science/article/pii/S1570826821000226)

*Qingxia Liu, Gong Cheng, Kalpa Gunaratna, Yuzhong Qu*

*Journal of Web Semantics*

This survey paper introduces existing research and future directions of entity summarization. With literature review, this paper firstly summarizes technical features of entity summarization, such as frequency and diversity. Secondly, it classifies existing methods combining these technical features. Then it introduces existing evaluation benchmarks for entity summarization, and discusses potential directions for future work.


*2022-07-11*

#### [Querying Linked Ontological Data Through Distributed Summarization](http://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5110)

*Achille Fokoue, Felipe Meneguzzi, Murat Sensoy, Jeff Z. Pan*

*AAAI 2012*

This paper introduces a summarization method for distributed ontologies. For each distributed resource, a summary as a RDF graph, is generated to efficiently handle queries. Given a query, the individuals (instances) are firstly mapped to some summaries using same concept sets and global hash values. Then the local summaries are filtered to select relevant small subsets, and merged to perform queries. These selected summaries are generally helpful for query pruning, source selection and transformation of distributed joins into local joins. 


*2022-07-07*

#### [ODArchive – Creating an Archive for Structured Data from Open Data Portals](https://doi.org/10.1007/978-3-030-62466-8_20)

*Thomas Weber, Johann Mitlöhner, Sebastian Neumaier, Axel Polleres*

*ISWC 2020*

This paper introduces a collection of over 260 open data portals, and analyzes the datasets provided in these portals. By analyzing the metadata of all the datasets, it presents statistical distributions of the datasets, such as popular domains and portals. Besides, it also analyzes the tabular data based on collected CSV files, including the number of rows, column labeling, references, etc. The collected portal list is open sourced with available APIs and some basic visual summaries. 


*2022-07-06*

#### [Leveraging Schema Labels to Enhance Dataset Search](https://doi.org/10.1007/978-3-030-45439-5_18)

*Zhiyu Chen, Haiyan Jia, Jeff Heflin, Brian D. Davison*

*ECIR 2020*

Existing dataset retrieval methods mainly match queries to the dataset description. To improve the performance of (tabular) dataset retrieval, this paper proposes a method to generate schema labels for the table content, and incorporates these labels in the ranking model. Schema labels refer to the header row of each data table. In this paper, the schema label generation is treated as a multi-label classification problem, and all the candidates are existing labels collected from the training set. This paper also proposes a test collection for the task of table retrieval.


*2022-07-05*

#### [Google Dataset Search: Building a search engine for datasets in an open Web ecosystem](https://doi.org/10.1145/3308558.3313685)

*Dan Brickley, Matthew Burgess, Natasha F. Noy*

*WWW 2019*

This is the first research paper that I read at the very beginning of my work towards dataset search. This paper introduces Google Dataset Search, which is based on Web-crawled metadata and provides search service for the user. It firstly introduces the motivations and challenges to build such a system. The datasets discussed in the paper refer to anything that a data provider considers to be datasetS (i.e., a natural but non-specific definition). Then it presents the technical challenges and implementation process of the system, and reports some statistical results of the current system. Lastly, it mentions some suggestions and feedbacks from the user, such as to index the datasets themselves apart from metadata.


*2022-07-04*

#### [Goods: Organizing Google’s Datasets](https://doi.org/10.1145/2882903.2903730)

*Alon Y. Halevy, Flip Korn, Natalya Fridman Noy, Christopher Olston, Neoklis Polyzotis, Sudip Roy, Steven Euijong Whang*

*SIGMOD 2016*

This paper is prior to the release of Google Dataset Search. It discusses Goods as a project for collecting, organizing, and analyzing distributed datasets from the Web. Specifically, Goods is a post-hoc system used in Google to organize datasets. It automatically collects and generates metadata, and enables the user to find datasets more easily. This paper discusses some challenges in designing the system, such as the large scale of datasets, the uncertainty of metadata discovery. According to the implementation, it relies on the Google Web crawler to collect metadata, and organizes the datasets into clusters. It customizes the catalog storage for the large scale of data, and implements a front end including a profile page for each dataset. In the end, it presents some insights learned from building the system.


*2022-07-03*

#### [Characterising dataset search—An analysis of search logs and data requests](https://doi.org/10.1016/j.websem.2018.11.003)

*Emilia Kacprzak, Laura Koesten, Luis-Daniel Ibáñez, Tom Blount, Jeni Tennison, Elena Simperl*

*Journal of Web Semantics 2019*

This journal paper has similar topic with the analytical report in 2018 [Characterising Dataset Search on the European Data Portal: An Analysis of Search Logs](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf). By analyzing the search logs of 4 open government data portals, in this paper the authors use a set of statistical metrics to characterize the activity of dataset retrieval. It analyzes the search logs from different perspectives including the users, the queries, and the data requests. Some findings are summarized, such as the geospatial and temporal features are the most common features for dataset retrieval, and current search functionalities are used in an exploratory manner rather than to retrieval a specific resource. 


*2022-07-02*

#### [Browsing Linked Data Catalogs with LODAtlas](https://doi.org/10.1007/978-3-030-00668-6_9)

*Emmanuel Pietriga, Hande Gözükan, Caroline Appert, Marie Destandau, Sejla Cebiric, François Goasdoué, Ioana Manolescu*

*ISWC 2018*

To help application developers and end-users easily find datasets of their interest, this resource paper proposes [LODAtlas](http://purl.org/lodatlas) as a Web-based search service. The datasets provided in the system are collected from open data portals (e.g., datahub.io, europeandataportal.eu). To enable users to submit different types of queries, view and compare the content of datasets, it indexes the metadata, incorporates several visual summarization methods, and provides statistical charts for comparison between datasets. Besides, in this paper, the overview of system structure and the [source codes](https://gitlab.inria.fr/epietrig/LODAtlas) are also provided. 


*2022-06-29*

#### [Towards An Objective Assessment Framework for Linked Data Quality: Enriching Dataset Profiles with Quality Indicators](https://doi.org/10.4018/IJSWIS.2016070104)

*Ahmad Assaf, Aline Senart, Raphaël Troncy*

*International Journal on Semantic Web and Information Systems 2016*

This paper mainly introduces a set of objective quality indicators for linked open datasets, and how to choose the quality metrics as a data provider or data consumer. In Section 3, it firstly discusses the criteria for data quality measurements, and groups them into different categories. In Section 4, it presents a survey of tools to measure linked open data quality. In Section 5, it proposes a framework for quality assessment, and evaluates it over the LOD datasets. 


*2022-06-28*

#### [Empirical Analysis of Ranking Models for an Adaptable Dataset Search](https://doi.org/10.1007/978-3-319-93417-4_4)

*Angelo Batista Neves, Rodrigo G. G. de Oliveira, Luiz André P. Paes Leme, Giseli Rabello Lopes, Bernardo Pereira Nunes, Marco A. Casanova*

*ESWC 2018*

This paper introduces the task of similarity search for datasets. Given a dataset $d_t$, it aims at ranking existing datasets $d_1, d_2, ...d_n$ according to the probability of their entities could be linked to $d_t$. There are three kinds of methods for this task, (1) similarity ranking, (2) using known links and metadata to learn linking rules, (3) identifying relevant hubs. In this paper, 5 ranking models are evaluated and compared with 2 use cases. The experiment is conducted over LOD Cloud datasets. 


*2022-06-27*

#### [Discovering and Maintaining Links on the Web of Data](https://doi.org/10.1007/978-3-642-04930-9_41)

*Julius Volz, Christian Bizer, Martin Gaedke, Georgi Kobilarov*

*ISWC 2009*

This paper proposes a linking framework named Silk, to maintain RDF links between linked open datasets. The framework mainly consists of three main parts, (1) a link discovery engine, (2) a tool for evaluating and modifying RDF links (e.g., generating `owl:sameAs` between entities), and (3) a protocol for maintaining RDF links. To discover interlinks, the Silk - Link Specification Language is proposed to express heuristics rules. The datasets are generally accessed through SPARQL endpoints, and the similarity between entities is measured by a set of statistical-based metrics. Then a Web-based interface allows user to inspect the links by comparing pairs of entities and verify the correctness. Thirdly, a link maintenance protocol is proposed to support updates of datasets and links. 


*2022-06-26*

#### [DING! Dataset Ranking using Formal Descriptions](http://ceur-ws.org/Vol-538/ldow2009_paper21.pdf)

*Nickolai Toupikov, Jürgen Umbrich, Renaud Delbru, Michael Hausenblas, Giovanni Tummarello*

*LDOW 2009*

DING, short for **D**ataset Rank**ING**, is proposed in this paper as a link analysis method for ranking semantic Web datasets based on interlinks identified by VoID vocabulary. VoID (Vocabulary of Interlinked Datasets) is a RDFS vocabulary for describing linked datasets, which is generally used as metadata and published alongside with the datasets. Following the PageRank model, in this paper the VoID interlinks between datasets are treated like hyperlinks between Webpages. Moreover, the weights of the links are designed in a TF-IDF manner. For each dataset, both its popularity (number of links) and rarity (the IDF part, uniqueness of the links) are considered. The datasets are ranked according to the computed PageRank-style scores. In the experiment, the proposed ranking method is compared with some other popular methods such as Google's PageRank, HITS, etc. 


*2022-06-25*

#### [Assessing Trust with PageRank in the Web of Data](http://ceur-ws.org/Vol-1597/PROFILES2016_paper5.pdf)

*José M. Giménez-García, Harsh Thakkar, Antoine Zimmermann*

*PROFILES@ESWC 2016*

This paper is motivated by the idea of assessing trust of datasets. The first research question is to decide what a dataset is. In this paper it uses PLD to identify each dataset. Based on that it identifies links between different datasets. Using these links, PageRank values are computed for each dataset, and the influence of ignoring the predicates is also discussed. The experiments are conducted on 300+ LOD datasets. The results show that reuse of a dataset is not strictly correlated with its size. 


*2022-06-24*

#### [RDFSync: Efficient Remote Synchronization of RDF Models](https://doi.org/10.1007/978-3-540-76298-0_39)

*Giovanni Tummarello, Christian Morbidoni, Reto Bachmann-Gmür, Orri Erling*

*ISWC 2007*

This paper introduces a graph decomposition method for efficiently synchronizing and merging RDF graphs. It decomposes the original RDF graph into a set of Minimum Self-Contained Graphs (MSGs). Each MSG is a minimum set of RDF statements containing a closure of IRIs and blank nodes. MSG decomposition could possibly bring redundant MSGs such as the same structure with replaceable blank nodes, which could be safely removed. For the usage, MSG decomposition could assist remote data synchronization (e.g., over HTTP) for it can be stream-like transported and does not require huge bandwidth. 


*2022-06-23*

#### [Canonical Forms for Isomorphic and Equivalent RDF Graphs: Algorithms for Leaning and Labelling Blank Nodes](https://doi.org/10.1145/3068333)

*Aidan Hogan*

*ACM Transactions on the Web 2017*

The existence of blank nodes in RDF graphs brings huge difficulty in deciding whether two RDF graphs have equivalent semantics. Motivated by this, the paper proposes two effective leaning-and-labeling methods for RDF graphs with blank nodes. It firstly proposes a canonical form for RDF graphs which preserves isomorphism. Then the second canonical form is under simple interpretation of semantics, which means graphs with same semantics but are not isomorphic could be equivalent. By leaning the graphs then canonically labeling the blank nodes, the complete graph can be transformed to a standard hash value. (The methods for hashing can be customized such as MD5, etc.) The worst case of the algorithm involves exponential steps, but for most of ordinary cases the computation is relatively fast. 


*2022-06-22*

#### [On the Graph Structure of the Web of Data](https://doi.org/10.4018/IJSWIS.2018040104)

*Alberto Nogales, Miguel-Ángel Sicilia, Elena García Barriocanal*

*International Journal on Semantic Web and Information Systems 2018*

This paper analyzes the Web of Data based on its graph structure and reports empirical findings, including the Web of Data also complying the bow-tie theory, and statistical features like node distances, degree centralities, etc. In Section 2, it firstly introduces the background of Web analysis, such as the bow-tie theory for the Web, the structure levels (i.e., page, host, and pay-level domain), and social network analysis (SNA) metrics. Then, in Section 3, by formulating the Web of Data as a directed graph where nodes represent datasets and edges represent RDF links, approaches used in analyzing the Web are applied. Section 4 introduces the process of gathering and cleaning the LOD Cloud datasets which are used in the following analyses. Section 5 reports the analysis results, including the overall structure, degree distribution, and connectivity. Finally, to check if the graph structure complies the bow-tie theory, it counts the specific structures in the graph and visualizes them into different groups, which demonstrates the compliance. 


*2022-06-21*

#### [Content-based Union and Complement Metrics for Dataset Search over RDF Knowledge Graphs](https://dl.acm.org/doi/10.1145/3372750)

*Michalis Mountantonakis, Yannis Tzitzikas*

*Journal of Data and Information Quality 2020*

In practice, dataset search users often propose needs for dataset unions and complements, such as finding top-k datasets that maximize the information (coverage), contain most complementary information (enrichment), or contribute the most unique content (uniqueness). However, existing dataset search methods including interlinking, cannot satisfy these needs. Therefore, in this paper a semantics-aware index is proposed to support efficient union and complement computation. Basically, for a set of given datasets, it stores each of the entities, properties, classes and literals as an inverted index. Then each of the computation tasks is presented as steps and metrics based on the index. The efficiency of the proposed methods are evaluated on a set of 400 RDF datasets, which is open-sourced.


*2022-06-20*

#### [Avoiding Chinese Whispers: Controlling End-to-End Join Quality in Linked Open Data Stores](https://doi.org/10.1145/2786451.2786466)

*Jan-Christoph Kalo, Silviu Homoceanu, Jewgeni Rose, Wolf-Tilo Balke*

*WebSci 2015*

This paper discusses the problem of incorrectness propagation for end-to-end large-scale entity linking, especially when several data sources are involved. In this paper, a benchmark for multi-source entity linking  is firstly presented, involving seven popular linked open datasets such as DBpedia, Freebase. Based on the proposed benchmark, existing instance matching systems are evaluated. The results shows a decrease of end-to-end performance especially when the chain of transitive links is relatively long. This also verifies the proposed problem of "Chinese Whispers" for instance matching. To handle this problem, this paper also proposes four approaches, including using similarity values, and relying on equivalent groups, cliques, or Markov clustering to identify correct links. The experimental results show that, existing systems achieve better performances with these approaches. 


*2022-06-18*

#### [Everything you always wanted to know about a dataset: Studies in data summarisation](https://doi.org/10.1016/j.ijhcs.2019.10.004)

*Laura Koesten, Elena Simperl, Tom Blount, Emilia Kacprzak, Jeni Tennison*

*International Journal of Human-Computer Studies 2019*

Data summarization not only helps end user quickly make sense of datasets, but also helps search systems, e.g., with keyword queries. To investigate data summarization, including important features of summaries and what people do care about, in this paper, the authors conduct two user studies. In the first user study, 69 students are recruited to record their data search diaries. In the second user study, crowdsourcing workers are recruited to produce summaries for given datasets. Then the data search diaries and crowdsourcing summaries are analyzed by components, topics, etc. Some findings and ideas about how to improve data summaries and related information such as metadata are discussed in the end.


*2022-06-17*

#### [Dataset search: a survey](https://doi.org/10.1007/s00778-019-00564-x)

*Adriane Chapman, Elena Simperl, Laura Koesten, George Konstantinidis, Luis-Daniel Ibáñez, Emilia Kacprzak, Paul Groth*

*VLDB J 2020*

This survey paper investigates existing systems and researches of dataset search, and discusses the problems and challenges that dataset search faces. It firstly introduces what dataset search is with definitions and example usages. Then in Section 2, it presents a general overview of dataset search process and system architecture. A general search process, as it is presented, contains four main parts, namely, query language, query handling, data handling and results presentation.  It also briefly reviews the other search communities, such as tabular search, information retrieval, etc. In Section 3, it reviews current systems related to dataset search, such as data portals, Google Dataset Search, etc. In Section 4, it reviews existing research efforts according to the four steps of dataset search. In Section 5 it talks about the open problems and challenges for dataset search researches and systems, mainly related to the nature of datasets which is different for existing techniques and systems. In Section 6, it highlights the importance of building benchmarks for dataset search. 


*2022-06-16*

#### [Characterising Dataset Search on the European Data Portal: An Analysis of Search Logs](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf)

*Luis-Daniel Ibáñez, Laura Koesten, Emilia Kacprzak, Elena Simperl*

*Analytical Report 18*

By analyzing over two years of European Data Portal (EDP) search and interaction logs, this paper mainly studies the following questions: (1) dataset search in the context of the EDP, i.e., how is it applied in EDP, (2) dataset search strategies and search query characteristics, (3) EDP versus Web search engines in dataset search, (4) success criteria in dataset search. Following the four stages in a dataset search process identified in the [survey](https://doi.org/10.1007/s00778-019-00564-x), this paper firstly introduces the general process of how EDP works. Then it introduces some key terms and interaction features recorded in the search logs, and presents statistical analyses for them, e.g., number of sessions, distribution of the facets. It also summarizes the statistical results into a set of findings and answers the previous questions. 


*2022-06-15*

#### [TripleRank: Ranking Semantic Web Data by Tensor Decomposition](https://doi.org/10.1007/978-3-642-04930-9_14)

*Thomas Franz, Antje Schultz, Sergej Sizov, Steffen Staab*

*ISWC 2009*

This paper introduces a method for faceted authority ranking of RDF data elements, i.e., properties and entities, based on tensor decomposition. It is motivated that existing graph-based ranking methods lack support for fine-grained latent coherence between resources and predicates, which can be complemented by tensor decomposition. Given the entities and properties connecting them in the RDF dataset as an input matrix, TripleRank computes the most significant groups of resources and predicates, and gives each of them an importance score. Such results can be applied in a faceted browsing scenario. 


*2022-06-14*

#### [SchemEX — Efficient construction of a data catalogue by stream-based indexing of linked data](https://doi.org/10.1016/j.websem.2012.06.002)

*Mathias Konrath, Thomas Gottron, Steffen Staab, Ansgar Scherp*

*Journal of Web Semantics 2012*

This paper introduces SchemEX, an indexing method for RDF datasets. Its structure contains three main layers, (1) RDF class layer, which stores the classes for RDF instances, (2) RDF type cluster layer, which combines the co-occurred classes of a single instance as a cluster, (3) equivalence class layer, based on a equivalence relation between instances, uses bisimulation method to characterize the relations of classes and properties in the dataset. This paper also proposes a SchemEX vocabulary based on VoID, which supports SPARQL queries over the indexes. The indexes are evaluated over the BTC dataset. 


*2022-06-13*

#### [SAKey: Scalable Almost Key Discovery in RDF Data](https://doi.org/10.1007/978-3-319-11964-9_3)

*Danai Symeonidou, Vincent Armant, Nathalie Pernelle, Fatiha Saïs*

*ISWC 2014*

This paper introduces SAKey, an approach to efficiently compute almost keys in RDF datasets. In practice, erroneous or duplicate data often exists thus causing trouble for precise key identification. An $n$-almost key is a set of properties with exception sets whose size is no more than $n$. SAKey approach derives such almost keys based on $n$-non keys, and incorporates filtering and pruning methods. An $n$-non key means for the set of properties, there is an exception set with size $>n$. The experiments are conducted over three datasets, i.e., DBpedia, YAGO and OAEI, where SAKey is also compared with other key identification methods. 


*2022-06-12*

#### [ROCKER – A Refinement Operator for Key Discovery](https://doi.org/10.1145/2736277.2741642)

*Tommaso Soru, Edgard Marx, Axel-Cyrille Ngonga Ngomo*

*WWW 2015*

This paper proposes an algorithm named ROCKER for finding keys, which is sets of properties for uniquely describing resources in linked datasets. Given a set of properties, if all the resources in the dataset are distinguishable, such a set can be called a key. For all properties in a given dataset, it firstly presents a quasi-ordering relation over the power set of the properties, with a score for each set of properties representing the fraction of subject resources that are distinguishable. Then it introduces a refinement operator for gradually adding and refining the keys. The algorithm for identifying keys is presented based on a priority queue, and is evaluated over twelve datasets. 


*2022-06-11*

#### [Profiling and Mining RDF Data with ProLOD++](https://doi.org/10.1109/ICDE.2014.6816740)

*Ziawasch Abedjan, Toni Grütze, Anja Jentzsch, Felix Naumann*

*ICDE 2014*

This demo paper introduces ProLOD++, the extension of a previous prototype ProLOD (at ICDE Workshop 2010) for profiling RDF datasets. It presents three kinds of tasks for ProLOD++, i.e., profiling, mining, and cleansing. Compared with the previous version, it adds uniqueness analysis to the prototype for analyzing the predicate combinations and usages to uniquely represent an entity. It also implements rule-based analysis methods for fact auto-completion and prediction. 


*2022-06-10*

#### [LODStats – An Extensible Framework for High-Performance Dataset Analytics](https://doi.org/10.1007/978-3-642-33876-2_31)

*Sören Auer, Jan Demter, Michael Martin, Jens Lehmann*

*EKAW 2012*

This paper introduces a set of evaluation metrics of RDF datasets analytics. The metrics are statement-stream-based and on triple level. In this paper, it firstly introduces the use cases of dataset analytics. Then it proposes the definition and criteria for statistical metrics, and presents a total of 32 schema level statistical metrics. Finally it talks about the implementation as python package and metadata representations. 


*2022-06-09*

#### [ExpLOD: Summary-Based Exploration of Interlinking and RDF Usage in the Linked Open Data Cloud](https://doi.org/10.1007/978-3-642-13489-0_19)

*Shahan Khatchadourian, Mariano P. Consens*

*ESWC 2010*

This paper introduces a RDF summarization method named ExpLOD, based on RDF graph bisimulation. Given a RDF graph, it firstly applies (literal) bisimulation labels to each of the instances, predicates and classes, including the prefix, types of the node, etc. Then it hierarchically contracts the labeled graph to generate the summary. It also provides several ways to implement the summarization, such as PRAIG which is a automatic construction method, SPARQL query-based subgraph extraction, etc. In the experiment, ExpLOD is evaluated on several popular RDF datasets. Its different implementations are also compared. 


*2022-06-08*

#### [Profiling Linked Open Data with ProLOD](https://doi.org/10.1109/ICDEW.2010.5452762)

*Christoph Böhm, Felix Naumann, Ziawasch Abedjan, Dandy Fenz, Toni Grütze, Daniel Hefenbrock, Matthias Pohl, David Sonnabend*

*ICDE Workshops 2010*

This paper introduces a Web-based prototype system named ProLOD for profiling RDF datasets. The back-end involves three main processes. For each dataset, (1) Clustering and Labeling mainly pre-computes the clusters of the dataset based on predicate similarity, and assigns a human-readable label for each element. (2) Schema Discovery uses statistical-based rules and algorithms to identify relations between predicates and attributes, such as co-occurrence, exclusive, etc., and uses these identified relation to refine the clusters. (3) Data Types and Pattern Statistics mainly counts and presents the numbers of the elements and patterns. 


*2022-06-07*

#### [RDF graph summarization for first-sight structure discovery](https://doi.org/10.1007/s00778-020-00611-y)

*François Goasdoué, Pawel Guzewicz, Ioana Manolescu*

*VLDB J 2020*

This paper discusses about quotient-based RDF graph summarization. It firstly introduces the equality conditions between nodes, strong/weak relations, etc. Then it presents the methods and conditions to generate summaries for RDF graphs, and also discusses the relationship between RDF graph saturation and summarization. It presents some sufficient conditions for saturated graph summary to be isomorphic to the summary of saturated graph (although seems strict and naive). It also talks about the visualization of RDF graph summaries which is in E-R style, with numbers of instances, properties, and inline types of entities. 


*2022-05-29*

#### [Characterizing the Semantic Web on the Web](https://link.springer.com/chapter/10.1007/11926078_18)

*Li Ding, Tim Finin*

*ISWC 2006*

This paper analyzes the semantic Web environment based on the data and resources collected by Swoogle. It firstly presents a model for semantic Web, including concepts about RDF graphs and their provenances (i.e., Web documents and agents). Then it proposes an effective way for harvesting documents and applies them in Swoogle to collect and analyze SWDs. The collected data is then analyzed to show the increasing trend of SWDs, distribution of domains and sizes, etc., and summarized into global statistics and implications. 


*2022-05-28*

#### [Swoogle: Searching for Knowledge on the Semantic Web](https://www.aaai.org/Papers/AAAI/2005/ISD05-007.pdf)

*Timothy W. Finin, Li Ding, Rong Pan, Anupam Joshi, Pranam Kolari, Akshay Java, Yun Peng*

*AAAI 2005 Demo*

This paper mainly presents the architecture of Swoogle and its interaction flow between RDF graphs and the semantic Web Documents. The architecture of Swoogle consists four bottom-up layers, namely, discovery, digest, analysis and service. For the service part, Swoogle provides both search service for human users and an ontology dictionary for software agents. This paper also provides an agent access model through navigation and links between semantic Web terms, documents, and ontologies. 


*2022-05-27*

#### [Swoogle: A Search and Metadata Engine for the Semantic Web](https://doi.org/10.1145/1031171.1031289)

*Li Ding, Timothy W. Finin, Anupam Joshi, Rong Pan, R. Scott Cost, Yun Peng, Pavan Reddivari, Vishal Doshi, Joel Sachs*

*CIKM 2004*

This paper introduces an influential search engine prototype "Swoogle" for semantic Web documents (SWDs). It firstly introduces the system architecture, including four parts: SWD discovery, metadata creation, data analysis and interface. Then it presents an analysis about the retrieved documents and their metadata. By identifying six relations such as imports, previous version and assigning their weights, Swoogle computes PageRank scores for document ranking. It also applies some IR-based methods such as n-grams for document retrieving. Overall, Swoogle acts as a famous search engine for SWDs at that time. 


*2022-05-26*

#### [An Intelligent Linked Data Quality Dashboard](http://ceur-ws.org/Vol-2563/aics_32.pdf)

*Ramneesh Vaidyambath, Jeremy Debattista, Neha Srivatsa, Rob Brennan*

*AICS 2019*

This paper introduces a tool for helping the user understand the linked datasets and make quality assessments. The system architecture consists of a Web-based service wrapper, an analytical dashboard, a triplestore, and an existing quality assessment tool. It supports detection of a set of quality problems, mainly related to undefined or incorrect classes and properties.  The dashboard provides an overview of the quality metric scores with further details. Finally, the usability of dashboard is evaluated by real human users, including "experts" and "general users".


*2022-05-25*

#### [How Matchable Are Four Thousand Ontologies on the Semantic Web](https://doi.org/10.1007/978-3-642-21034-1_20)

*Wei Hu, Jianfeng Chen, Hang Zhang, Yuzhong Qu*

*ESWC 2011*

This paper presents an overview about how and to what extent do ontologies match with each other. The analysis starts from identifying term-level mappings. Based on Falcon-AO, term mappings are computed. An undirected term mapping graph is built, and statistical results are reported, such as the number and distribution of connected components. Then, based on the term mapping, a directed ontology mapping graph is similarly built and analyzed. Besides, also based on the terms, an undirected pay-level-domain mapping graph is built. The metrics used in this paper are mainly from the network analysis fields.


*2022-05-24*

#### [A Snapshot of the OWL Web](https://doi.org/10.1007/978-3-642-41335-3_21)

*Nicolas Matentzoglu, Samantha Bail, Bijan Parsia*

*ISWC 2013*

This paper builds a corpus of OWL ontologies and compares it with other existing ontology collections, and reports the statistical results with analysis. It firstly introduces basic concepts about OWL and its profiles. Then the pipeline of obtaining and curating ontologies is introduced. Ontologies are crawled from the Web, filtered to reduce errors or duplicates, and grouped into clusters. Each remaining OWL DL document is parsed, and analyzed by domain sources, syntax distributions, etc. This paper also compares the entity usage of the proposed corpora with existing ontology collections. 


*2022-05-23*

#### [An Empirical Study of Vocabulary Relatedness and Its Application to Recommender Systems](https://doi.org/10.1007/978-3-642-25073-6_7)

*Gong Cheng, Saisai Gong, Yuzhong Qu*

*ISWC 2011*

This paper discusses relatedness measurements for Semantic Web vocabularies. It firstly presents six metrics from four aspects, and proposes relatedness-based recommendation strategy using linear combination of the proposed metrics. Four main aspects are considered to reflect relatedness in this paper. (1) Explicit relations between vocabularies such as `owl:imports`. An edge-weighted graph is built to capture the number of relations between vocabularies. Relatedness is then computed from the length of shortest path between vocabularies. Similarly, another graph indicating implicit references is also built and used as the second metric. (2) Content similarity is mainly based on string overlaps of class and property labels between vocabularies. (3) Expressivity closeness is measured by the extent of meta-terms (e.g., super classes) overlaps. (4) Distributional relatedness uses cosine similarity between term co-occurrence vectors to measure the closeness between vocabularies. 


*2022-05-22*

#### [Summarizing semantic graphs: a survey](https://doi.org/10.1007/s00778-018-0528-3)

*Sejla Cebiric, François Goasdoué, Haridimos Kondylakis, Dimitris Kotzinos, Ioana Manolescu, Georgia Troullinou, Mussab Zneika*

*VLDB J 2019*

This survey paper presents a taxonomy for semantic graph summarization, with the approaches and applications. It firstly introduces basic concepts about semantic graphs, such as RDF, RDFS, OWL, and BGP query. Then it describes the scope of this paper, the usage of different summarization methods, and the classification for the methods. In this paper, summarization methods are categorized into 4 kinds, namely, structural, statistical, pattern-mining and hybrid. In the following sections, it discusses each kind of methods respectively, including generic graph (non-RDF) summarization, structural RDF summarization, pattern-based RDF summarization, statistical summarization, and others. 


*2022-05-21*

#### [Profiling relational data: a survey](https://link.springer.com/article/10.1007/s00778-015-0389-y)

*Ziawasch Abedjan, Lukasz Golab, Felix Naumann*

*VLDB J 2015*

This survey paper introduces tasks and approaches for profiling relational tables. It firstly identifies the usages for data profiling and illustrates its importance. Then it introduces profiling tasks including single-column profiling, multi-column profiling, dependency detection, storage, etc. In the following sections, it comprehensively reviews existing efforts for each profiling task along with the primary usages. After that, this paper summarizes existing profiling tools, and presents a discussion about next generation of profiling integration and non-relational data (e.g., RDF) profiling. Some of the approaches for relational data might also be applicable or extendable for non-relational data. 


*2022-05-19*

#### [RDF dataset profiling – a survey of features, methods, vocabularies and applications](https://content.iospress.com/articles/semantic-web/sw294)

*Mohamed Ben Ellefi, Zohra Bellahsene, John G. Breslin, Elena Demidova, Stefan Dietze, Julian Szymanski, Konstantin Todorov*

*Semantic Web 2018*

This is a comprehensive and important survey paper for RDF dataset profiling that worth reading for several times. It reviews 85 publications over the past 2 decades about RDF datasets. Following the introduction and description of the survey process, it firstly presents a taxonomy for dataset profile features, including 7 main categories. Then it provides a systematic overview of dataset profile extraction tools and approaches based on the taxonomy. The third part is an overview and a classification of vocabularies used to characterize dataset profiles. The fourth part proposes an illustration for the use of dataset profiles within the application contexts. 


*2022-05-17*

#### [Visual Querying LOD sources with LODeX](https://dl.acm.org/doi/10.1145/2815833.2815849)

*Fabio Benedetti, Sonia Bergamaschi, Laura Po*

*K-CAP 2015*

This paper introduces a visual query system for linked open datasets named LODeX. Based on the datasets available on Data Hub, LODeX identifies four main steps to process the datasets and incorporate them into the system. (1) Indexes extraction builds statistical indexes including the number of usages for each class, property, and the domain-range relations between them. Based on the indexes, (2) Schema summary generation aims at producing statistical summaries for schema elements, including classes, properties, attributes, their labels, and the mapping function between them. (3)  Schema summary visualization provides the user with node-link style visualized summaries. (4) Query orchestration supports the user to create query by selecting nodes, links and classes. The query will be automatically transformed into SPARQL query and perform. 


*2022-05-15*

#### [OptiqueVQS: a Visual Query System over Ontologies for Industry](https://content.iospress.com/articles/semantic-web/sw293)

*Ahmet Soylu, Evgeny Kharlamov, Dmitriy Zheleznyakov, Ernesto Jiménez-Ruiz, Martin Giese, Martin G. Skjæveland, Dag Hovland, Rudolf Schlatte, Sebastian Brandt, Hallstein Lie, Ian Horrocks*

*Semantic Web 2018*

This paper introduces a visual query system named OptiqueVQS, aiming at assisting the user to formulate structured queries over industrial ontologies. It firstly proposes three main challenges for such a system, including identifying common query types, identifying query, task, user types, and identifying quality attributes. Then it accordingly lists a set of requirements for expressivity (supporting which type of queries) and quality attributes (how to measure the system). It demonstrates the system by examples and illustrates the function of each component, and conducts user studies with human experts and ordinary users.


*2022-05-14*

#### [RDF Explorer: A Visual Query Builder for Semantic Web Knowledge Graphs](http://ceur-ws.org/Vol-2456/paper60.pdf)

*Hernán Vargas, Carlos Buil Aranda, Aidan Hogan*

*ISWC Satellites 2019*

This demo paper introduces a visual query builder that allows non-expert user to formulate SPARQL queries by adding nodes and links rather than directly editing in the text box. In this paper, the query creation process is explained with an example based on Wikidata. 


*2022-05-12*

#### [HIEDS: A Generic and Efficient Approach to Hierarchical Dataset Summarization](https://www.ijcai.org/Abstract/16/521)

*Gong Cheng, Cheng Jin, Yuzhong Qu*

*IJCAI 2016*

This paper introduces HIEDS, a hierarchical summarization method for RDF datasets. It applies a hierarchical grouping method for entities by property-value pairs in their descriptions. To form each hierarchical level, a property is selected, and entities are grouped by the values of the property. There are also links and overlapping entities between sibling subgroups. The problem is formulated as a multidimensional knapsack problem to achieve a balanced division with higher cohesion in each group. Then this NP-hard problem is solved by a greedy strategy.


*2022-05-11*

#### [Knowledge graph exploration: where are we and where are we going?](https://dl.acm.org/doi/10.1145/3409481.3409485)

*Matteo Lissandrini, Torben Bach Pedersen, Katja Hose, Davide Mottin*

*SIGWEB Newsletter 2020*

This paper surveys and summarizes the knowledge graph exploration techniques in existing work, and proposes a taxonomy of them over a spectrum. It categories the exploration techniques into three kinds, (1) Summarization and profiling, which requires no user interaction, personalization and no domain knowledge, and whose output is high level overview. (2) Exploratory analytics, which requires median-level interactivity and high-level information need, whose output is overview of specific aspects. (3) Exploratory search, which requires high interactivity, personalization and detailed sample or query intent, whose output is detailed answers. This taxonomy also categories more specific exploration tasks, and discusses potential future work directions.


*2022-05-09*

#### [RDF Data Storage and Query Processing Schemes: A Survey](https://dl.acm.org/doi/10.1145/3177850)

*Marcin Wylot, Manfred Hauswirth, Philippe Cudré-Mauroux, Sherif Sakr*

*ACM Computing Surveys 2018*

This survey paper introduces a taxonomy of linked data/RDF data management systems. Here, management means storage and querying. In this taxonomy, systems are firstly categorized into centralized and distributed ones, then divided into more specific kinds such as statement table, graph-based, etc. Each kind of systems are introduced with typical example systems. 


*2022-05-05*

#### [Estimating Characteristic Sets for RDF Dataset Profiles Based on Sampling](https://link.springer.com/chapter/10.1007/978-3-030-49461-2_10)

*Lars Heling, Maribel Acosta*

*ESWC 2020*

This paper discusses statistical profile features of RDF datasets. It proposes the concept of "characteristic set" which in fact is the set of properties used to describe each entity in the dataset. This paper mainly focuses on the count of such characteristic sets, and the multiplicity of some properties to profile the RDF dataset. To address the problem of large RDF datasets are not easy to straight-forwardly count the characteristic sets for all entities,  this paper proposes a solution by firstly sampling a small subset of triples, then estimating the overall counts using designed projection function. It tries several sample methods (by entities, by weighted triples, etc.) and statistics-enhanced projection methods. Finally, the paper is concluded by answering 4 research questions. 


*2022-05-04*

#### [Personalized Knowledge Graph Summarization: From the Cloud to Your Pocket](https://ieeexplore.ieee.org/document/8970788)

*Tara Safavi, Caleb Belth, Lukas Faber, Davide Mottin, Emmanuel Müller, Danai Koutra*

*ICDM 2019*

This paper proposes GLIMPSE, a summarization method to extract triples that best meet the user's potential interest from the original large knowledge graph. It relies on the user's history list of queries to implement a probabilistic-based sampling. It measures the importance of each entity from its query history and all its neighbors, and measures the importance of each triple both by the entities it contains and whether the triple itself has appeared in the query history. The paper also proves that the probabilistic-based sample framework is sub-modular. Therefore, a greedy algorithm is able to achieve a constant approximation ratio of (1 - 1/e). It further improves the time complexity by incorporating pruning process. Finally, it conducts experiments over large real-world knowledge graphs and achieves good performance. 


*2022-05-03*

#### [Structural Properties as Proxy for Semantic Relevance in RDF Graph Sampling](https://doi.org/10.1007/978-3-319-11915-1_6)

*Laurens Rietveld, Rinke Hoekstra, Stefan Schlobach, Christophe Guéret*

*ISWC 2014*

This paper introduces SampLD, an RDF graph sampling method aiming to select a subset of triples from the original large RDF dataset, while trying to retain the same ability to answer SPARQL queries as the original one. It relies on network analysis methods and statistical-based features, such as PageRank, in-degree and out-degree. SampLD firstly rewrites the graph, then analyzes it with network features, assigns node weights, and finally performs sampling from the ranked list of triples. This paper also conducts experiments over several large (in 2014) RDF datasets. 


*2022-05-02*

#### [Dataset Discovery in Data Lakes](https://ieeexplore.ieee.org/document/9101607)

*Alex Bogatu, Alvaro A. A. Fernandes, Norman W. Paton, Nikolaos Konstantinou*

*ICDE 2020*

It is a database style paper. This paper mainly studies the data discovery problem in a data lake, i.e., how to identify relevant datasets (for a given target dataset, if provided) from a bunch of large, mixed, complex datasets. It introduces a data lake as "a repository whose items are datasets about which, we assume, we have no more metadata than, when in tabular form, their attribute names, and possible domain-independent types." This paper proposes "D3L" for data discovery. The datasets discussed in this paper are tabular data. It proposes a locality-sensitive hashing-based method to index each table, including hashing attribute names, data values, attribute formats, etc. The hash value itself for each dataset can be regarded as an embedding, which can be used for computing distances between different datasets. It conducts experiments over synthetic table sets and real-world tables. At least two test collections are open-sourced.


*2022-05-01*

#### [Google Dataset Search by the Numbers](https://link.springer.com/chapter/10.1007/978-3-030-62466-8_41)

*Omar Benjelloun, Shiyu Chen, Natasha F. Noy*

*ISWC 2020*

Google Dataset Search (or Google DS in short) has developed for about 4 years since launched in 2018. This paper was published in 2020, which introduces an overview about the snapshot of Google Dataset Search at that time with a series of statistical distributions and numbers. Generally, Google DS relies on schema.org, DCAT and other similar vocabularies to identify datasets from Web pages. By analyzing the Web pages crawled by Google's general Web crawler, the pages with specific Semantic Web annotations and especially the information about datasets are extracted, downloaded and indexed. This paper analyzes these records, provides the distributions of datasets by domains, languages, topics, formats, vocabularies within the metadata, etc. It also proposes some suggestions for Semantic Web researchers about how to further improve the data reusability and interchangeability. 


*2022-04-30*

#### [The Trials and Tribulations of Working with Structured Data - a Study on Information Seeking Behaviour](https://dl.acm.org/doi/10.1145/3025453.3025838)

*Laura M. Koesten, Emilia Kacprzak, Jenifer Fay Alys Tennison, Elena Simperl*

*CHI 2017*

(Probably the first paper that let me know about Dr. Koesten.)

This paper studies information seeking behaviors of people, i.e., how people find data, make sense of them, obtain information from the data, and use them. The authors conducted two experiments for this purpose. One is a user study by interview, the other is a search log analysis of data.gov.uk. For the user study, the recruited participants were required to have data-centric tasks in their daily work, the interview was about their activities such as data collections, data analysis, etc. The search log was mainly statistically analyzed. The lengths and distributions of queries and sessions were reported. By analyzing the user study and statistical results, this paper concludes a framework with 4 steps in a data interaction process, and major tasks within each step.


*2022-04-28*

####  [Talking datasets - Understanding data sensemaking behaviours](https://www.sciencedirect.com/science/article/pii/S1071581920301646?via%3Dihub)

*Laura Koesten, Kathleen Gregory, Paul Groth, Elena Simperl*

*Journal of Human-Computer Studies 2021*

This is an interesting article discussing how people make sense of data and how to reuse them. Specifically, it mainly focuses on two research questions: (1) what are the common patterns of sense-making activities for people to know about data, (2) how do these sense-making patterns enable potential data reuse, like what will be needed for reusing the data. This paper is mainly based on a large, comprehensive user interview with 31 independent participants. The interview process was recorded and analyzed. Each participant was asked to provide a familiar dataset, and introduced it in the interview, then the interviewer would also provide a unfamiliar dataset to the participant and let the participant try to make sense of it and summarize its content, followed by some other questions about their opinion of data reuse, data sharing, etc. After analyzing the interview results, this paper also provides some suggestions about how to better understand, analyze, and reuse the data. 


*2022-04-27*

#### [Open data User Needs: A Preliminary Synthesis](https://dl.acm.org/doi/10.1145/3366424.3386586)

*Auriol Degbelo*

*WWW 2020*

This is a brief "survey" paper which aims to act as a starting point to analyze the open government data (OGD) usages and user needs. Mainly by reviewing existing papers about users' data needs, search behaviors, etc., it selects 7 representative papers which focus on user behaviors with data. It aggregates all discussed user need statements and re-categorizes them from different aspects. It follows the pattern: ‘[A user] needs [need] in order to accomplish [goal]’, to identify the needs and goals, divides them in different types (informational / transactional / navigational) of information needs, and summarizes them into 10 kinds of data needs as an open data user needs taxonomy. The table-1 is quite clear and informational. The paper also explains the usefulness of this work, e.g., for Web agents, Web mining, etc.


*2022-04-24*

#### [IDOL: Comprehensive & Complete LOD Insights](https://doi.org/10.1145/3132218.3132238)

*Ciro Baron Neto, Dimitris Kontokostas, Amit Kirschenbaum, Gustavo Correa Publio, Diego Esteves, Sebastian Hellmann*

*Semantics 2017*

This paper provides an analysis about some existing RDF dataset repositories, the datasets they contain, and their subsets and distributions. Some ideas are very close to ours. Firstly, they obtain real-world RDF datasets and repositories from several well-known websites or data sources, including DBpedia, LOD laundromat, CKAN.org, etc., and they explain the reasons and steps to process the data sources. Secondly, according to the pipeline in Section 4, they follow the way to (1) get and store the metadata, (2) get, store and process the datasets by triples, (3) modify and republish the datasets' metadata. They have also conducted the deduplication process among triples, which seems as to determine subset relations. They have implemented a "Bloom Filter" to check triple equality, which is an approximate hash function with guaranteed accuracy. The time and space costs seem to be relatively large, they also incorporate some optimizing methods. Finally they report the analysis results of the overall repositories and datasets (a total of 174 repositories and 13,237 datasets), provide counts and distributions about them. 


*2022-04-23*

#### [Quality assessment for Linked Data: A Survey](https://content.iospress.com/articles/semantic-web/sw175)

*Amrapali Zaveri, Anisa Rula, Andrea Maurino, Ricardo Pietrobon, Jens Lehmann, and Sören Auer*

*Semantic Web Journal, 2016*

I've read this paper when preparing the submission to theWebConf'21, at that time we wanted to implement some quality measurements for each dataset, to improve the re-ranking performance for the retrieved result datasets. Intuitively it was because "Relying only on links mined from metadata is far from enough, incorporating the content can add some links, but still not enough. So we needed some quality measurements to help improve ranking as a complement." For this paper, it indeed provides various metrics from 4 aspects, include: (1) accessibility, e.g., availability, licensing, (2) intrinsic dimensions, which mainly related to semantic accuracy and schema consistency, (3) contextual dimensions, which mainly related to the contextual task or user satisfactory, e.g., trustworthiness, understandability, timeliness, (4) representational dimensions, measuring if the dataset is typical and usable, e.g., w/ or w/o multi-language version. Besides, this paper also illustrates the relations between the 4 kinds of dimensions, and provides a comparison among all surveyed papers and tools of all the summarized quality dimensions. 
