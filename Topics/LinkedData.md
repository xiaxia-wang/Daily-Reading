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
