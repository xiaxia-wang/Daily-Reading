





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
