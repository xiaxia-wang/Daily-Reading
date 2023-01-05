









*2023-01-05*

#### [Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs](http://proceedings.mlr.press/v97/guo19c.html)

*Lingbing Guo, Zequn Sun, Wei Hu*

*ICML 2019*

This paper proposes a new framework for training entity embeddings. It utilizes long-term entity relation chains and implements a recurrent skipping network (RSN), to better capture long-term relational dependencies between the entities. 


*2023-01-04*

#### [Multi-view Knowledge Graph Embedding for Entity Alignment](https://www.ijcai.org/proceedings/2019/754)

*Qingheng Zhang, Zequn Sun, Wei Hu, Muhao Chen, Lingbing Guo, Yuzhong Qu*

*IJCAI 2019*

This paper proposes an entity alignment method which utilizes different views of the entities within the source KG. It generates embeddings for an entity based on its name, relations and attributes. Then it combines these different "views" into an overall embedding. Besides, it also adopts cross-KG relation and attribute identity inference to enhance the performance. 


*2023-01-03*

#### [A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf)

*Zequn Sun, Qingheng Zhang, Wei Hu, Chengming Wang, Muhao Chen, Farahnaz Akrami, Chengkai Li*

*VLDB 2020*

This paper investigates the embedding-based methods for entity alignment over knowledge graphs. It proposes a set of benchmarking datasets which fit the entity distribution of real-world KGs, and develops OpenEA as an open-sourced library with 12 KG embedding models. It also compares the performances between these models and with traditional methods for entity alignment. 


*2022-12-21*

#### [CRNet: Modeling Concurrent Events over Temporal Knowledge Graph](https://doi.org/10.1007/978-3-031-19433-7_30)

*Shichao Wang, Xiangrui Cai, Ying Zhang, Xiaojie Yuan*

*ISWC 2022*

This paper proposes a model to capture the concurrence of events in temporal knowledge graphs. It uses a concurrent evolution module with graph attention network to capture the historical event relations. Then it builds a candidate graph and predicts the missing events (i.e., entities) based on it. 


*2022-12-20*

#### [HybridFC: A Hybrid Fact-Checking Approach for Knowledge Graphs](https://doi.org/10.1007/978-3-031-19433-7_27)

*Umair Qudus, Michael Röder, Muhammad Saleem, Axel-Cyrille Ngonga Ngomo*

*ISWC 2022*

This paper proposes an ensemble method which combines text-based, KG-based and path-based features for fact checking over knowledge graphs. (submitted to JoWS, reviewed by me)


*2022-12-18*

#### [Each Snapshot to Each Space: Space Adaptation for Temporal Knowledge Graph Completion](https://doi.org/10.1007/978-3-031-19433-7_15)

*Yancong Li, Xiaoming Zhang, Bo Zhang, Haiying Ren*

*ISWC 2022*

This paper proposes a space adaptation network for temporal knowledge graph completion (TKGC). It extends a convolutional neural network to map the facts with different timestamps into different latent spaces. Besides, it adapts the overlap between different spaces to control the balance between time-variability and time-stability. 


*2022-12-16*

#### [Radar Station: Using KG Embeddings for Semantic Table Interpretation and Entity Disambiguation](https://doi.org/10.1007/978-3-031-19433-7_29)

*Jixiong Liu, Viet-Phi Huynh, Yoan Chabot, Raphaël Troncy*

*ISWC 2022*

This paper proposes a system named Radar Station, for entity disambiguation of relational table cells. It links the cells (i.e., entities) in the table to a target KG (e.g., Wikidata in this paper). Based on the rows and columns context, it can identify different entities with the same label using semantic similarities between them. It implements a graph embedding for the entities to represent the latent relationships between them. 


*2022-12-08*

#### [Enhancing Document-Level Relation Extraction by Entity Knowledge Injection](https://doi.org/10.1007/978-3-031-19433-7_3)

*Xinyi Wang, Zitao Wang, Weijian Sun, Wei Hu*

*ISWC 2022*

This paper proposes to improve general relation extraction (RE) models by adding a knowledge injection layer between the encoding and the prediction layers. It considers coreference distillation and context exchanging in knowledge injection, and reconciles KG encoding to the representations. 


*2022-11-22*

#### [OWL2Vec*: embedding of OWL ontologies](https://doi.org/10.1007/s10994-021-05997-6)

*Jiaoyan Chen, Pan Hu, Ernesto Jiménez-Ruiz, Ole Magnus Holter, Denvar Antonyrajah, Ian Horrocks*

*Machine Learning 2021*

This paper proposes an embedding method named OWL2Vec* for OWL ontologies. The overall framework is divided into two parts. The random walks over the input ontology generate a set of sentences which capture the graph structure, while the lexical information provided by e.g., *rdfs:label* and *rdfs:comment* retains the semantics of the ontology. These two documents, together with a combination of them as a third document, are fed into a pretrained language model to generate embeddings for IRIs and words. 


*2022-11-18*

#### [Numerical Feature Representation with Hybrid N-ary Encoding](https://doi.org/10.1145/3511808.3557090)

*Bo Chen, Huifeng Guo, Weiwen Liu, Yue Ding, Yunzhe Li, Wei Guo, Yichao Wang, Zhicheng He, Ruiming Tang, Rui Zhang*

*CIKM 2022*

This paper proposes an encoding framework for numerical features in CTR models. It relies on the natural binary representations of numerical systems to characterize the discretization of elements, and incorporates intra-ary and inter-ary attention to avoid collision risk among neighboring representations.


*2022-11-10*

#### [Inductive Knowledge Graph Reasoning for Multi-batch Emerging Entities](https://doi.org/10.1145/3511808.3557361)

*Yuanning Cui, Yuxin Wang, Zequn Sun, Wenqiang Liu, Yiqiao Jiang, Kexin Han, Wei Hu*

*CIKM 2022*

This paper proposes a reinforcement learning model with graph convolutional network to handle knowledge graph reasoning with chronologically updated entity batches. In this model, the GCN is used to encode and update entity embeddings based on walk-based path samples. It also incorporates a link augmentation strategy to add more facts for new entities with few links. 


*2022-11-08*

#### [Explainable Link Prediction in Knowledge Hypergraphs](https://doi.org/10.1145/3511808.3557316)

*Zirui Chen, Xin Wang, Chenxu Wang, Jianxin Li*

*CIKM 2022*

This paper introduces a link prediction method under the knowledge hypergraph model. It applies an EM algorithm with Markov Logic Network model to enrich and update the hypergraph embeddings. 


*2022-11-03*

#### [Entity Type Prediction Leveraging Graph Walks and Entity Descriptions](https://doi.org/10.1007/978-3-031-19433-7_23)

*Russa Biswas, Jan Portisch, Heiko Paulheim, Harald Sack, Mehwish Alam*

*ISWC 2022*

This paper focuses on the problem of predicting classes for each entity in a knowledge graph. It uses three graph walk strategies, namely, node walks, edge walks, and classic node-edge walks. It feeds these different walks into a word2vec model to generate entity embeddings, which is used in downstream class prediction. The model allows to output multiple classes for an entity. 


*2022-11-02*

#### [Context-Enriched Learning Models for Aligning Biomedical Vocabularies at Scale in the UMLS Metathesaurus](https://dl.acm.org/doi/10.1145/3485447.3511946)

*Vinh Nguyen, Hong Yung Yip, Goonmeet Bajaj, Thilini Wijesiriwardene, Vishesh Javangula, Srinivasan Parthasarathy, Amit P. Sheth, Olivier Bodenreider*

*TheWebConf 2022*

This paper incorporates the knowledge graph embeddings to conduct entity alignment. It concatenates the embedding vector of each UMLS term with lexical-based (LSTM) embedding vector to compute the similarity score. It also compares several kinds of knowledge graph embeddings.  


*2022-10-16*

#### [TaxoEnrich: Self-Supervised Taxonomy Completion via Structure-Semantic Representations](https://dl.acm.org/doi/10.1145/3485447.3511935)

*Minhao Jiang, Xiangchen Song, Jieyu Zhang, Jiawei Han*

*TheWebConf 2022*

This paper investigates the problem of taxonomy completion, which means adding new concepts to existing hierarchical taxonomy. It aims to identify a position $\langle n_p, n_c \rangle$ where the new concept should be added. To achieve this, it proposes a contextualized embedding for each node (i.e., existing concept) in the tree based on pseudo sentences and PLM. Then it uses a LSTM-based encoder to extract the features of the paths from root to $n_p$ and $n_c$, and also extracts some relevant siblings to provide more information. It aggregates the above parent/child/siblings to perform query matching. The overall framework is evaluated on Microsoft Academic Graph (MAG) and WordNet datasets. 
