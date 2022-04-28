

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
