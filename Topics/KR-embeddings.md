








*2022-11-10*

#### [Inductive Knowledge Graph Reasoning for Multi-batch Emerging Entities](https://doi.org/10.1145/3511808.3557361)

*Yuanning Cui, Yuxin Wang, Zequn Sun, Wenqiang Liu, Yiqiao Jiang, Kexin Han, Wei Hu*

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
