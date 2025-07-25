






*2025-07-24*

#### [Shifting from Ranking to Set Selection for Retrieval Augmented Generation](https://arxiv.org/abs/2507.06838)

*Dahyun Lee, Yongrae Jo, Haeju Park, Moontae Lee*

*ACL 2025*

While traditional search engines rank individual results by relevance, RAG systems need a curated set of passages to generate accurate answers, requiring not only relevance but also diversity, completeness, and the comprehensiveness of retrieved passages. To address these challenges, this paper proposes a setwise passage selection approach that optimizes the passage set as a whole, rather than treating retrieval as an independent ranking task.


*2025-06-18*

#### [SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis](https://arxiv.org/abs/2505.16834)

*Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, Lei Fang, Zhongyuan Wang, Ji-Rong Wen*

*Arxiv 2025*

This work introduces a lightweight yet effective framework that bridges this gap through strategic data engineering rather than complex training paradigms, by simulating realistic user interactions in live web search environments, coupled with a multi-criteria curation strategy that optimizes the diversity and quality of input and output side.


*2025-03-09*

#### [KNOWNET: Guided Health Information Seeking from LLMs via Knowledge Graph Integration](https://arxiv.org/abs/2407.13598)

*Youfu Yan, Yu Hou, Yongkang Xiao, Rui Zhang, Qianwen Wang*

*IEEE VIS 2024*

This paper proposes a visualization system for health information retrieval by improving LLMs via integrating external KGs. In particular, to enhance accuracy, it extracts triples from LLM outputs and maps them to supporting evidence in external KGs. For structured exploration, it provides recommendations for further inquiry. Considering that multi-step exploration might introduce information overwhelming, it follows the focus+context design and proposes a progressive graph visualization to track inquiries, connecting the history with current queries and next-step recommendations.


*2025-02-21*

#### [ShapeShifter: Workload-Aware Adaptive Evolving Index Structures Based on Learned Models](https://openreview.net/forum?id=diaXdjqLGW#discussion)

*Hui Wang, Xin Wang, Jiake Ge, Lei Liang, Peng Yi*

*WWW 2025*

This paper proposes a dynamic approach to update point search index structure according to evolving topics, which has two main modules: (1) the classifier module which, in the first phase, determines the "temporature state" as either hot or code of the target node in wach look up or insert operation; and (2) the adaptation module which, in the second phase, triggers an evolving operation if a node is classified as either hot or cold; otherwise, the index structurere mains unchanged.


*2025-02-04*

#### [Behavior Modeling Space Reconstruction for E-Commerce Search](https://arxiv.org/abs/2501.18216)

*王叶晶，张持，赵翔宇，刘启东，王茂林，危学涛，刘子韬，施兴，杨旭东，钟灵，林伟*

*TheWebConf 2025*

This paper proposes an e-commerce search framework that enhances accuracy via two components to reconstruct the behavior modeling space: (1) preference editing to proactively remove the relevance effect from preference predictions, yielding untainted user preferences; (2) adaptive fusion to dynamically adjust fusion criteria to align with the varying relevance and preference within the reconstructed modeling space.


*2025-02-03*

#### [LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations](https://arxiv.org/abs/2406.12433)

*高璟桐，陈渤，赵翔宇，刘卫文，李向阳，王奕超，王婉玉，郭慧丰，唐睿明*

*TheWebConf 2025*

Existing reranking approaches often fail to harmonize diverse criteria effectively at the model level. In response, this paper introduces a LLM-based reranking framework to seamlessly integrate various reranking criteria while maintaining scalability and facilitating personalized recommendations. The framework has a fully connected graph structure, allowing the LLM to simultaneously consider multiple aspects such as accuracy, diversity, and fairness through a coherent Chain-of-Thought process. A customizable input mechanism is integrated to enable the tuning of the language model's focus to meet specific reranking needs.


*2024-09-19*

#### [Perspectives on Large Language Models for Relevance Judgment](https://arxiv.org/abs/2304.09161)

*Guglielmo Faggioli, Laura Dietz, Charles Clarke, Gianluca Demartini, Matthias Hagen, Claudia Hauff, Noriko Kando, Evangelos Kanoulas, Martin Potthast, Benno Stein, Henning Wachsmuth*

*SIGIR 2023*

This perspectives paper discusses possible ways for LLMs to support relevance judgments along with concerns and issues that arise. It devises a human-machine collaboration spectrum that allows to categorize different relevance judgment strategies, based on how much humans rely on machines. For the extreme point of "fully automated judgments", it further includes a pilot experiment on whether LLM-based relevance judgments correlate with judgments from trained human assessors.


*2024-09-18*

#### [Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community](https://arxiv.org/abs/2307.09751)

*Qingyao Ai, Ting Bai, Zhao Cao, Yi Chang, Jiawei Chen, Zhumin Chen, Zhiyong Cheng, Shoubin Dong, Zhicheng Dou, Fuli Feng, Shen Gao, Jiafeng Guo, Xiangnan He, Yanyan Lan, Chenliang Li, Yiqun Liu, Ziyu Lyu, Weizhi Ma, Jun Ma, Zhaochun Ren, Pengjie Ren, Zhiqiang Wang, Mingwen Wang, Ji-Rong Wen, Le Wu, Xin Xin, Jun Xu, Dawei Yin, Peng Zhang, Fan Zhang, Weinan Zhang, Min Zhang, Xiaofei Zhu*

*Arxiv 2023 (worth reading a few more times)*

This report is a record of a strategic workshop conducted by the Chinese IR community, which discusses the transformative impact of LLMs on IR research. It first discusses the fundamental value of IR with its boundaries and extensions. Then it analyzes the opportunities and potential enhancements that LLM/IR could bring to each other, and the new paradigms in the new era for LLM+IR.


*2024-08-02*

#### [Old IR Methods Meet RAG](https://dl.acm.org/doi/abs/10.1145/3626772.3657935)

*Oz Huly, Idan Pogrebinsky, David Carmel, Oren Kurland, Yoelle Maarek*

*SIGIR 2024 Short*

This paper shows that, some traditional sparse retrieval approach, such as BM25, may perform better than recent LLM-based dense retrieval methods. By comparing two LLMs with sparse models, it reveals that a broad set of sparse retrieval methods achieve better results than dense retrieval methods for varying lengths of queries induced from the prompt. This finding calls for further study of classical retrieval methods for RAG.


*2024-08-01*

#### [Evaluating Retrieval Quality in Retrieval-Augmented Generation](https://dl.acm.org/doi/10.1145/3626772.3657957)

*Alireza Salemi, Hamed Zamani*

*SIGIR 2024 Short*

This paper proposes an approach for evaluating the retrieval quality in RAG systems, where each document in the retrieval list is individually utilized by the large language model within the RAG system. The output generated for each document is then evaluated based on the downstream task ground truth labels. In this manner, the downstream performance for each document serves as its relevance label.


*2024-07-30*

#### [Large Language Models and Future of Information Retrieval: Opportunities and Challenges](https://dl.acm.org/doi/10.1145/3626772.3657848)

*ChengXiang Zhai*

*SIGIR 2024*

In the era of LLMs, this perspective paper investigates the following questions related to future IR: (1) How can we both exploit the strengths of LLMs and mitigate any risk caused by their weaknesses when applying LLMs to IR? (2) What are the best opportunities for us to apply LLMs to IR, both for improving the current generation search engines and for developing the next-generation search engines? (3) What are the major challenges that we will need to address in the future to fully exploit such opportunities? (4) Given the anticipated growth of LLMs, what will future information retrieval systems look like? Will LLMs eventually replace an IR system?


*2024-07-13*

#### [Retrieval Augmented Zero-Shot Text Classification](https://arxiv.org/abs/2406.15241)

*Tassallah Abdullahi, Ritambhara Singh, Carsten Eickhoff*

*SIGIR 2024*

Zero-shot text learning enables text classifiers to handle unseen classes efficiently, alleviating the need for task-specific training data. A simple approach often relies on comparing embeddings of query text to those of potential classes. To improve the performance without doing extensive training, this paper proposes a training-free knowledge augmentation approach that reformulates queries by retrieving supporting categories from Wikipedia for zero-shot text classification.


*2024-05-08*

#### [From Matching to Generation: A Survey on Generative Information Retrieval](https://arxiv.org/abs/2404.14851)

*Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, Zhicheng Dou*

*Arxiv 2024*

Currently, research in generative information retrieval (GenIR) can be categorized into two directions: generative document retrieval (GR) and reliable response generation. This survey paper summarizes the advancements in GR regarding model training, document identifier, incremental learning, downstream tasks adaptation, multi-modal GR and generative recommendation, as well as progress in reliable response generation in aspects of internal knowledge memorization, external knowledge augmentation, generating response with citations and personal information assistant. It also discusses the evaluation, challenges and future prospects in GenIR systems.


*2024-04-20*

#### [Stream of Search (SoS): Learning to Search in Language](https://arxiv.org/abs/2404.03683)

*Kanishk Gandhi, Denise Lee, Gabriel Grand, Muxin Liu, Winson Cheng, Archit Sharma, Noah D. Goodman*

*Arxiv 2024*

This paper shows how language models can be taught to search by representing the process of search in language, as a flattened string — a stream of search (SoS). Specifically, by pretraining a transformer-based language model from scratch on a dataset of search streams generated by heuristic solvers, SoS pretraining increases search accuracy by 25% over models trained to predict only the optimal search trajectory, and even more after fine-tuning. The results indicate that language models can learn to solve problems via search, self-improve to flexibly use different search strategies, and potentially discover new ones.


*2024-03-31*

#### [ACORDAR 2.0 A Test Collection for Ad Hoc Dataset Retrieval with Densely Pooled Datasets and Question-Style Queries]()

*Qiaosheng Chen, Weiqing Luo, Zixian Huang, Tengteng Lin, Xiaxia Wang, Ahmet Soylu, Basil Ell, Baifan Zhou, Evgeny Kharlamov and Gong Cheng*

*SIGIR 2024*

This paper proposes an improved version of a test collection for ad hoc dataset search, by incorporating dense retrieval models in the pooling process to avoid lexical bias, and using LLMs to rewrite keyword queries into question-style queries to diversify available query forms.


*2023-12-31*

#### [CTRL: Connect Collaborative and Language Model for CTR Prediction](https://arxiv.org/abs/2306.02841)

*Xiangyang Li, Bo Chen, Lu Hou, Ruiming Tang*

*Arxiv 2023*

Traditional click-through rate (CTR) prediction models convert the tabular data into one-hot vectors while ignoring the essential semantic information. To address this problem, this paper first converts the original tabular data into textual data, regards them as two different modalities and separately feeds them into the collaborative CTR model and pre-trained language model. Then it preforms a cross-modal knowledge alignment procedure to align the collaborative and semantic signals.


*2023-12-22*

#### [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents](https://arxiv.org/abs/2304.09542)

*Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren*

*EMNLP 2023 Outstanding*

This paper investigates the passage reranking ability of LLMs with prompts. Specifically, it reveals with experiments that properly instructed LLMs can deliver even superior results to state-of-the-art supervised methods on popular IR benchmarks. Further, it proposes a new test set aiming to verify the model’s ability to rank unknown knowledge. Besides, it tries to distill the ranking capabilities of ChatGPT into small specialized models with a permutation distillation scheme.


*2023-10-08*

#### [Which Tricks are Important for Learning to Rank?](https://proceedings.mlr.press/v202/lyzhin23a.html)

*Ivan Lyzhin, Aleksei Ustimenko, Andrey Gulin, Liudmila Prokhorenkova*

*ICML 2023*

This paper empirically compares several learning-to-rank models and investigates two questions: (1) Is direct optimization of a smoothed ranking loss preferable over optimizing a convex surrogate? (2) How to properly construct and smooth surrogate ranking losses?


*2023-08-29*

#### [Large Language Models for Information Retrieval: A Survey](https://arxiv.org/abs/2308.07107)

*Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Zhicheng Dou, Ji-Rong Wen*

*Arxiv 2023*

This paper reviews existing research efforts of LLMs in the field of information retrieval, and organizes them according to different modules of IR, including rewriter, retriever, reranker and reader.


*2023-08-01*

#### [Modeling Fine-grained Information via Knowledge-aware Hierarchical Graph for Zero-shot Entity Retrieval](https://dl.acm.org/doi/10.1145/3539597.3570415)

*Taiqiang Wu, Xingyu Bai, Weigang Guo, Weijie Liu, Siheng Li, Yujiu Yang*

*WSDM 2023*

Notice that existing sentence-embedding based models cannot fully capture the mentions for entities, this paper proposes a model to learn extra fine-grained information based on an entity graph with hierarchical graph attention to obtain entity embeddings.


*2023-07-31*

#### [A Bird's-eye View of Reranking: From List Level to Page Level](https://dl.acm.org/doi/10.1145/3539597.3570399)

*Yunjia Xi, Jianghao Lin, Weiwen Liu, Xinyi Dai, Weinan Zhang, Rui Zhang, Ruiming Tang, Yong Yu*

*WSDM 2023*

This paper proposes a dual-attention based model for page-level reranking, which requires a to rerank multiple lists of products simultaneously with a unified model. To achieve this, it applies hierarchical dual-side attention and a spatial-scaled attention network to learn the fine-grained spatial-aware item interactions across lists.


*2023-07-28*

#### [Heterogeneous Graph-based Context-aware Document Ranking](https://dl.acm.org/doi/10.1145/3539597.3570390)

*Shuting Wang, Zhicheng Dou, Yutao Zhu*

*WSDM 2023*

Context-aware document ranking aims to rerank candidate documents of query $𝑞_𝑖$ based on its search context (i.e., queries $q_1$ to $q_{i-1}$ and corresponding clicked documents) so as to rank the clicked document as high as possible, and each candidate document is scored by its relevance with respect to the current query and its context. This paper proposes a GNN-based method consisting of a session graph and a query graph with shared parameters.


*2023-07-22*

#### [A Reference-Dependent Model for Web Search Evaluation: Understanding and Measuring the Experience of Boundedly Rational Users](https://dl.acm.org/doi/pdf/10.1145/3543507.3583551)

*Nuo Chen, Jiqun Liu, Tetsuya Sakai*

*WWW 2023*

Generally, the user's actions in Web search interaction are associated with relative gains and losses to reference points, known as the reference dependence effect. This paper proposes an evaluation metric, namely Reference Dependent Metric (ReDeM), for assessing query-level search by incorporating the effect of reference dependence into the modelling of user search behavior. The experimental results show that, integrating ReDeMs with a proper reference point achieves better correlations with user satisfaction than existing metrics, such as Discounted Cumulative Gain (DCG) and Rank-Biased Precision (RBP).


*2023-07-20*

#### [Improving Content Retrievability in Search with Controllable Query Generation](https://dl.acm.org/doi/10.1145/3543507.3583261)

*Gustavo Penha, Enrico Palumbo, Maryam Aziz, Alice Wang, Hugues Bouchard*

*WWW 2023*

This paper proposes a query generation method being able to improve: (1) the training data used for dense retrieval models, and (2) the distribution of narrow and broad intent queries issued in the system. It is proposed as a generative model based on the Transformer architecture, whose input is the entity with all its serialized metadata, and an optional input query is used as weakly supervised label.


*2023-07-15*

#### [TRAVERS: A Diversity-Based Dynamic Approach to Iterative Relevance Search over Knowledge Graphs](https://dl.acm.org/doi/10.1145/3543507.3583429)

*Ziyang Li, Yu Gu, Yulin Shen, Wei Hu, Gong Cheng*

*WWW 2023*

Traditional relevance search task (over KG) requires the user to provide example question-answer pairs, which are sometimes not so useful and cannot handle cases with code start. Motivated by this, this paper proposes a labeling-based iterative relevance search method that asks the user to label current answer entities in an iterative manner. The labels will be used as reward to optimize answer entities in the next iteration. It performs a learning-to-rank process involving two rankers, i.e., a diversity-oriented ranker for supporting cold start and avoiding converging to sub-optimum caused by noisy labels, and a relevance-oriented ranker capable of handling unbalanced labels.


*2023-05-09*

#### [Multivariate Representation Learning for Information Retrieval](https://arxiv.org/pdf/2304.14522.pdf)

*Hamed Zamani, Michael Bendersky*

*SIGIR 2023*

Instead of using k-dim vectors to represent the queries and documents in dense retrieval, this paper proposes another framework by representing both of them using k-variate normals. The query-document similarity is then computed by KL-divergence between distributions.


*2023-05-08*

#### [Understand the Dynamic World: An End-to-End Knowledge Informed Framework for Open Domain Entity State Tracking](https://arxiv.org/pdf/2304.13854.pdf)

*Mingchen Li, Lifu Huang*

*SIGIR 2023*

Open domain entity state tracking aims to predict reasonable state changes of entities. This paper proposes a model for this purpose by firstly retrieving related entities and attributes from an external knowledge graph, and designing a dynamic encoder-decoder process to generate state changes.


*2023-03-24*

#### [Unsupervised Key Event Detection from Massive Text Corpora](https://dl.acm.org/doi/10.1145/3534678.3539395)

*Yunyi Zhang, Fang Guo, Jiaming Shen, Jiawei Han*

*KDD 2022*

This paper proposes the task of key event detection, aiming to detect from a news corpus key events that happen at a particular time/location and focus on the same topic. It also proposes an unsupervised extraction method, consisting of (1) key phrase mining, (2) phrase clustering, and (3) iterative document selection based on pseudo labels generated by LLMs w.r.t. the key phrases.


*2023-03-23*

#### [Knowledge Enhanced Search Result Diversification](https://dl.acm.org/doi/10.1145/3534678.3539459)

*Zhan Su, Zhicheng Dou, Yutao Zhu, Ji-Rong Wen*

*KDD 2022*

This paper proposes a GNN-based model for improving the diversity of document retrieval results. It firstly builds an entity-relation graph based on the document corpus and query entities. Then it applies a GCN to adjust the node weights w.r.t. the entity set covered by the selected document sequence, and aggregates the final document score with the relevance feature and diversity feature.


*2023-02-27*

#### [Noisy Interactive Graph Search](https://dl.acm.org/doi/10.1145/3534678.3539267)

*Qianhao Cong, Jing Tang, Kai Han, Yuming Huang, Lei Chen, Yeow Meng Chee*

*KDD 2022*

This paper introduces the problem of noisy interactive graph search, which is formulated as to find a target node in a given hierarchy based on several rounds of noisy reachability queries. The noise is adopted as an one-coin model with a fixed probability to give wrong answer. It firstly uses the Bayesian model to compute the posterior probability of the prediction. Then it also shows the posterior probability is monotonic w.r.t. the number of queries performed in the interaction.


*2022-11-30*

#### [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://doi.org/10.1145/3397271.3401075)

*Omar Khattab, Matei Zaharia*

*SIGIR 2020*

This paper proposes ColBERT, which is a passage retrieval model based on BERT. It independently encodes the documents and the query, then uses a "late interaction" layer to capture the similarity between the query and documents.


*2022-11-17*

#### [Efficient and Effective SPARQL Autocompletion on Very Large Knowledge Graphs](https://doi.org/10.1145/3511808.3557093)

*Hannah Bast, Johannes Kalmbach, Theresa Klumpp, Florian Kramer, Niklas Schnelle*

*CIKM 2022*

This paper investigates the problem of autocompletion of SPARQL queries. It applies prefix matching, filters reachable predicates, and computes the relevance of IRIs to provide suggestions to the user. It also optimizes the implementation of SPARQL query execution. 


*2022-11-12*

#### [Dense Retrieval with Entity Views](https://doi.org/10.1145/3511808.3557285)

*Hai Dang Tran, Andrew Yates*

*CIKM 2022*

Existing dense retrieval methods typically use the average embedding of all the tokens in a passage as its representation. Considering one passage could have multiple views in which entities are loosely related, this paper proposes to generate multiple representations of a passage based on clusters of entities. For a given query, it firstly selects a cluster of most relevant entities (from the passage) to the query, and adopts the average embedding of these entities as a query-aware representation of the passage to perform dense retrieval. 


*2022-11-09*

#### [SpaDE: Improving Sparse Representations using a Dual Document Encoder for First-stage Retrieval](https://doi.org/10.1145/3511808.3557456)

*Eunseong Choi, Sunkyung Lee, Minjin Choi, Hyeseon Ko, Young-In Song, Jongwuk Lee*

*CIKM 2022*

This paper proposes an improved sparse representation for document retrieval. It proposes a uni-encoder approach with term weighting encoder for adjusting the weights of existing terms in the document, and term expansion strategy for adding top-k predicted (masked) terms into the document representation. The length of documents in the evaluation dataset is relatively small (56 words on average) so that they can be directly fed into the BERT encoder. 


*2022-11-06*

#### [ExpScore: Learning Metrics for Recommendation Explanation](https://doi.org/10.1145/3485447.3512269)

*Bingbing Wen, Yunhe Feng, Yongfeng Zhang, Chirag Shah*

*TheWebConf 2022*

This paper investigates the problem of explainable item recommendation. It firstly collects a dataset containing machine-generated explanations for item recommendation and crowd-sourcing ratings. Then it proposes a neural-network based metric to measure the explainability of the sentences. It extracts several numeric factors such as relevance, length, readability of the explanations and feeds them into a network, and adopts the prediction score as the metric. 


*2022-10-15*

#### [Learning Probabilistic Box Embeddings for Effective and Efficient Ranking](https://dl.acm.org/doi/10.1145/3485447.3512073)

*Lang Mei, Jiaxin Mao, Gang Guo, Ji-Rong Wen*

*TheWebConf 2022*

This paper proposes a probabilistic box embedding for queries and items in information retrieval. Each query or item is represented by a high-dimension axis-aligned hyper rectangle to characterize its diversity and uncertainty. To improve the efficiency, it also proposes a box embedding-based indexing method to filter irrelevant items and reduce the retrieval latency. A box embedding for each object is a *d*-dimension axis-aligned hyper-rectangle. On each of the dimension there is a lower bound and an upper bound (as a "box"). The overlapping volume of two embeddings are used to model the relationship of the objects, which is proved as a kernel function. 


*2022-10-14*

#### [Global or Local: Constructing Personalized Click Models for Web Search](https://dl.acm.org/doi/10.1145/3485447.3511950)

*Junqi Zhang, Yiqun Liu, Jiaxin Mao, Xiaohui Xie, Min Zhang, Shaoping Ma, Qi Tian*

*TheWebConf 2022*

This paper investigates personalized click models in Web search. Based on a 6-month search log of a commercial search engine, in this paper the authors firstly construct a dataset with search records of 1,249 users to train and evaluate the models. Then it combines different personalized (local) search strategies with global model by keeping the personalized relevance or behavior model parameters. 


*2022-10-13*

#### [Towards a Better Understanding of Human Reading Comprehension with Brain Signals](https://dl.acm.org/doi/10.1145/3485447.3511966)

*Ziyi Ye, Xiaohui Xie, Yiqun Liu, Zhihong Wang, Xuesong Chen, Min Zhang, Shaoping Ma*

*TheWebConf 2022*

This paper investigates the brain activities and neural signals when handling reading comprehension tasks. It firstly conducts a user study and collects neural signals when people conducting reading comprehension tasks. Then it uses the collected data to train a sentence classification model and evaluates its effectiveness. 


*2022-10-11*

#### [Exploring Heterogeneous Data Lake based on Unified Canonical Graphs](https://doi.org/10.1145/3477495.3531759)

*Qin Yuan, Ye Yuan, Zhenyu Wen, He Wang, Chen Chen, Guoren Wang*

*SIGIR 2022*

This paper focuses on the problem of keyword search over heterogeneous datasets in a data lake. It constructs a canonical graph where nodes represent entities, and edges represent either relations between entities or linkages of matching nodes across data sources. The answer tree is formulated as the minimum Steiner tree with respect to the keyword query. 


*2022-10-10*

#### [A Sketch-based Index for Correlated Dataset Search](https://doi.org/10.1109/ICDE53745.2022.00264)

*Aécio S. R. Santos, Aline Bessa, Christopher Musco, Juliana Freire*

*ICDE 2022*

This paper investigates the problem of, for a given dataset, searching for its related (i.e., query-biased and joinable) datasets from a large corpus. To address this problem, this paper proposes a hashing scheme and constructs a sketch-based index to support efficient search for correlated tables. 


*2022-09-26*

#### [Identifying Facet Mismatches In Search Via Micrographs](https://doi.org/10.1145/3357384.3357911)

*Sriram Srinivasan, Nikhil S. Rao, Karthik Subbian, Lise Getoor*

*CIKM 2019*

In e-commerce, each query from a customer contains several facets such as product type, color and brand. This paper aims to identify the mismatched facets between the products returned by the search engine and the user's query intent. It firstly constructs a micrograph based on the connections between the products and the query. Then it applies statistical relational learning to predict the existence of certain structures. It also improves the model structure with a confidence passing technique to achieve a better performance. 


*2022-09-12*

#### [Thematic ranking of object summaries for keyword search](https://doi.org/10.1016/j.datak.2017.08.002)

*Georgios John Fakas, Yilun Cai, Zhi Cai, Nikos Mamoulis*

*Data & Knowledge Engineering 2018*

This paper proposes a model to rank OSs according to their relevance to a set of thematic keywords. The authors argue that the effective thematic ranking should incorporate IR-style properties, authoritative ranking and affinity. In the paper, each of these three features is formulated as an importance score of an OS. Then these scores are combined as a product to measure the overall importance of the OS. It also proposes several optimized algorithms to solve the problem. 


*2022-09-11*

#### [Diverse and proportional size-*l* object summaries using pairwise relevance](https://doi.org/10.1007/s00778-016-0433-6)

*Georgios John Fakas, Zhi Cai, Nikos Mamoulis*

*VLDBJ 2016*

This paper is an extension of [Diverse and Proportional Size-*l* Object Summaries for Keyword Search](https://doi.org/10.1145/2723372.2737783). In this paper, beyond the importance of each node in the OS, it also consider the pairwise relevance between the nodes. 


*2022-09-10*

#### [Diverse and Proportional Size-*l* Object Summaries for Keyword Search](https://doi.org/10.1145/2723372.2737783)

*Georgios John Fakas, Zhi Cai, Nikos Mamoulis*

*SIGMOD 2015*

This paper improves the previous method to generate Size-*l* OSs in [Versatile Size-*l* Object Summaries for Relational Keyword Search](https://doi.org/10.1109/TKDE.2013.110). It adds two more features to be considered in the local importance of each attribute-value pair. Firstly, diversity means that the same entity (e.g., the name matched to the keywords) should not be repeated for too many times. Secondly, proportionality means the frequency of each entity in the OS should be close to its original frequency in the dataset. Based on these two features, this paper proposes two kinds of Size-*l* OSs, named DSize-*l* and PSize-*l* OSs, with efficient generation algorithms. 


*2022-09-09*

#### [Versatile Size-*l* Object Summaries for Relational Keyword Search](https://doi.org/10.1109/TKDE.2013.110)

*Georgios John Fakas, Zhi Cai, Nikos Mamoulis*

*IEEE TKDE 2014*

This paper is an extension of [Size-*l* Object Summaries for Relational Keyword Search](https://doi.org/10.14778/2078331.2078338). It proposes two types of size-*l* OS snippets, namely, size-*l* OS(t) limits the number of tuples and size-*l* OS(a) consists of no more than *l* attributes. Compared with the previous paper, it (1) improves the DP algorithms with lower computational complexity, (2) changes the problem formulation to let *l* represent the number of attribute-value pairs instead of the tuples, and (3) introduces a method to automatically select *l* based on a pre-defined importance function. 


*2022-09-08*

#### [Size-*l* Object Summaries for Relational Keyword Search](https://doi.org/10.14778/2078331.2078338)

*Georgios John Fakas, Zhi Cai, Nikos Mamoulis*

*VLDB 2011*

Compared with [A novel keyword search paradigm in relational databases: Object summaries](https://doi.org/10.1016/j.datak.2010.11.003), this paper proposes a new problem of selecting exact *I* (i.e., a given number of size) tuples instead of using thresholds to limit the number of results. Besides, this paper mainly focuses on effective algorithms of finding the optimal tuples. It proposes a dynamic programming algorithm with heuristics to reduce the search space. 


*2022-09-07*

#### [Ranking of Object Summaries](https://doi.org/10.1109/ICDE.2009.171)

*Georgios John Fakas, Zhi Cai*

*ICDE 2009*

This paper is an extension of [Automated generation of object summaries from relational databases: A novel keyword searching paradigm](https://doi.org/10.1109/ICDEW.2008.4498381). Compared with the previous work which considers the search result as a list of OSs, this paper further investigates the ranking of OSs and their tuples. This is to facilitate the generation of size-constrained OSs and identify the most important tuples. Based on a proposed global importance score for each tuple, it is able to generate a top-k (i.e., at most k OSs in the list), size-*l* (i.e., each OS in the list must has at most *l* tuples) OS list as the query result. 


*2022-09-06*

#### [Automated generation of object summaries from relational databases: A novel keyword searching paradigm](https://doi.org/10.1109/ICDEW.2008.4498381)

*Georgios John Fakas*

*ICDE Workshops 2008*

This paper is a previous work of [A novel keyword search paradigm in relational databases: Object summaries](https://doi.org/10.1016/j.datak.2010.11.003). In this paper the concept of keyword search in relational databases is firstly introduced. The result of such a keyword search is a ranked set of object summaries. It also preliminarily introduce the summarization method based on table and attribute affinity. 


*2022-09-05*

#### [A novel keyword search paradigm in relational databases: Object summaries](https://doi.org/10.1016/j.datak.2010.11.003)

*Georgios John Fakas*

*Data & Knowledge Engineering 2011*

This paper proposes a method to generate object summaries (OS) for keyword search in relational databases. The input is a keyword query which can hit data subjects (DS) in the database. An object summary is a tree with the data subject as the root, and other neighboring tuples as children. Beginning with this data subject, it firstly selects close relational tables, and computes the affinity of the attributes in these tables. Controlled by given thresholds, it filters some tuples and attributes to appear in the final OS. Besides, when the keyword query hits multiple data subjects, the final OSs are further ranked based on local importance and their size. 


*2022-09-04*

#### [Unsupervised Extraction of Template Structure in Web Search Queries](https://doi.org/10.1145/2187836.2187892)

*Sandeep Pandey, Kunal Punera*

*WWW 2012*

This paper investigates the template extraction of the Web search queries. It firstly proposes three properties (i.e., assumptions) of the query templates. Then it introduces a generative model based on the probabilistic distribution of the words to the templates. It also describes some existing methods such as LDA and k-means. In the experiment, it evaluates the performance of the model based on manually labeled ground truth. 


*2022-09-03*

#### [Understanding User Goals in Web Search](https://doi.org/10.1145/988672.988675)

*Daniel E. Rose, Danny Levinson*

*WWW 2004*

This paper investigates the underlying goals behind the user's search behavior, i.e., why do people search. By analyzing a set of queries from the AltaVista search engine, it proposes a framework to categorize the search queries. There are three kinds of goals, namely, navigational, informational, and resource. Then all the queries are manually categorized into the three categorizes. The results indicate that the navigational queries are less prevalent than generally believed, while the resource seeking goal may account for a large part of Web search.


*2022-09-01*

#### [Inside the Search Process: Information Seeking from the User’s Perspective](https://doi.org/10.1002/(SICI)1097-4571(199106)42:5%3C361::AID-ASI6%3E3.0.CO;2-%23)

*Carol Collier Kuhlthau*

*Journal of the American Society for Information Science 1991*

This paper investigates the information seeking activities from the user's perspective. It firstly proposes a information search process (ISP) model with six stages. In each stage it also gives a set of feelings, thoughts and tasks of the user. Then it analyzes five user studies to verify the model, and indicates the gap between the user's experience in information seeking and existing researches.


*2022-08-31*

#### [Exploratory search: from finding to understanding](https://doi.org/10.1145/1121949.1121979)

*Gary Marchionini*

*Communications of the ACM 2006*

This paper firstly proposes three kinds of search activities when people use Web search service, including lookup, learn and investigate. Specifically, exploratory search mainly includes the learn and investigate process. In the rest of the paper, each of these activities is analyzed in detail with example systems and techniques. 


*2022-08-30*

#### [Efficient and Progressive Group Steiner Tree Search](https://doi.org/10.1145/2882903.2915217)

*Rong-Hua Li, Lu Qin, Jeffrey Xu Yu, Rui Mao*

*SIGMOD 2016*

This paper proposes the keyword search algorithms PrunedDP and PrunedDP++ to solve the GST problem. They are mainly based on optimal-tree decomposition and conditional tree merging techniques, and incorporate several lower-bounding techniques to further optimize the computation. 


*2022-08-29*

#### [Characterising Dataset Search Queries](https://doi.org/10.1145/3184558.3191597)

*Emilia Kacprzak, Laura Koesten, Jeni Tennison, Elena Simperl*

*PROFILES@WWW 2018*

This workshop paper analyzes queries for dataset search. The queries are generated by a crowd-sourcing experiment based on data requests to the UK Government Open Data portal. Then these 449 queries are compared with the search log analysis of four open data portals. The comparison results indicate some differences between them. For example, the generated queries are much longer than the portal queries, and they contain more geospatial and temporal information. 


*2022-08-28*

#### [Automatic Identification of User Goals in Web Search](https://doi.org/10.1145/1060745.1060804)

*Uichin Lee, Zhenyu Liu, Junghoo Cho*

*WWW 2005*

This paper proposes an automatic way to identify the user's goal in Web search. In this paper, the goals are categorized into two kinds, navigational and informational. Firstly, it verifies with a user study that the goals in Web search can indeed be identified. Then it adopts two features of (1) user-click behavior and (2) anchor-link distribution, to automatically predict the goal of the user. 


*2022-08-27*

#### [A taxonomy of web search](https://doi.org/10.1145/792550.792552)

*Andrei Z. Broder*

*SIGIR 2002*

This paper discusses the purpose of classic information retrieval. It proposes that IR mainly focuses on predicting the user's information need, which can be categorized as informational, navigational and transactional. With this taxonomy of Web search, it also discusses how existing search engines evolve to deal with the information needs. 


*2022-07-14*

#### [Performance Measures for Multi-Graded Relevance](http://ceur-ws.org/Vol-781/paper7.pdf)

*Christian Scheel, Andreas Lommatzsch, Sahin Albayrak*

*SPIM@ISWC 2011*

This paper proposes an extension of the relevance metric Mean Average Precision (MAP), by introducing graded relevances to the computation. Graded relevances mean that each retrieved object could be rated on a graded scale (e.g., 1 -- 5), rather than binary ratings (i.e., relevant or not). In this paper, 3 relevance levels are used in the evaluation. It firstly introduces the dataset used in this paper and the computation of average precision. Then it extends the MAP by using different relevance thresholds and computing the overall average score to handle the graded relevances. 


*2022-07-13*

#### [Extending Average Precision to Graded Relevance Judgments](https://doi.org/10.1145/1835449.1835550)

*Stephen E. Robertson, Evangelos Kanoulas, Emine Yilmaz*

*SIGIR 2010*

This paper discusses evaluation metrics for information retrieval. It proposes the graded average precision (GAP) by incorporating graded relevance to the traditional average precision (AP) metric. It introduces the user model, definition and properties of GAP, and evaluates it in learning-to-rank tasks by using it as an optimize objective.


*2022-07-10*

#### [Retune: Retrieving and Materializing Tuple Units for Effective Keyword Search over Relational Databases](https://doi.org/10.1007/978-3-540-87877-3_34)

*Guoliang Li, Jianhua Feng, Lizhu Zhou*

*ER 2008*

This paper is the previous conference version of the [TKDE paper](https://doi.org/10.1109/TKDE.2011.61). This paper mainly proposes the concept of tuple units, and the way to generate and materialize the tuple units over databases using SQL statements. For retrieving and ranking the tuple units, it incorporates both TF-IDF-based relevance measure and graph-based distance measure to compute the relevance score of the tuple units for the given query.


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
