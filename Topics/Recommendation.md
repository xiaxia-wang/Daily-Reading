








*2025-09-29*

#### [SPARK: Adaptive Low-Rank Knowledge Graph Modeling in Hybrid Geometric Spaces for Recommendation](https://arxiv.org/abs/2509.11094)

*Binhao Wang, Yutian Xiao, Maolin Wang, Zhiqi Li, Tianshuo Wei, Ruocheng Guo, Xiangyu Zhao*

*CIKM 2025*

This paper proposes an embedding-based KG recommendation approach, which uses low-rank decomposition to generate entity representations. Then an SVD-initialized hybrid geometric GNN learns representations in Euclidean and Hyperbolic spaces to capture semantic features of sparse, long-tail items. A core contribution is an item popularity-aware adaptive fusion strategy that dynamically weights signals from collaborative filtering, refined KG embeddings, and geometric spaces for modeling both mainstream and long-tail items. Finally, contrastive learning aligns these multi-source representations.


*2025-08-11*

#### [Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation](https://aclanthology.org/2025.acl-long.1317/)

*Shijie Wang, Wenqi Fan, Yue Feng, Lin Shanru, Xinyu Ma, Shuaiqiang Wang, Dawei Yin*

*ACL 2025*

The model first indexes hop-field knowledge sub-graphs within the KG, and uses a popularity selective retrieval policy to determine which items should be retrieved or augmented. Then it retrieves specific subgraphs from the knowledge vector database. Subsequently, the retrieved knowledge sub-graphs are re-ranked to refine the retrieval quality. Finally, the retrieved knowledge sub-graphs are utilized with the original prompt to generate recommendations.


*2025-07-18*

#### [CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models](http://arxiv.org/abs/2506.17281)

*Junze Chen, Xinjie Yang, Cheng Yang, Junfei Bao, Zeyuan Guo, Yawen Li, Chuan Shi*

*SIGIR 2025*

This work proposes an approach for graph-based recommendation that includes: (1) an LLM for preference reasoning based on user profiles, with the response serving as a query to extract relevant users and items from the interaction graph as preference-assisted retrieval; (2) the information retrieved in the previous step along with the purchase history of target user is used by LLM to refine a smaller interaction subgraph as intent-assisted retrieval; (3) a GNN to capture high-order collaborative filtering information from the extracted subgraph and generate the final recommendation results.


*2025-05-10*

#### [Efficient Multi-task Prompt Tuning for Recommendation](https://doi.org/10.48550/arXiv.2408.17214)

*Ting Bai, Le Huang, Yue Yu, Cheng Yang, Cheng Hou, Zhe Zhao, Chuan Shi*

*ACM Transactions on Information Systems*

This paper proposes a model for multi-task recommendation that includes two modules, namely, multi-task pre-training and multi-task fine-tuning. In the pre-training phase it uses a task-specific expert for each task to decouple the task features, and a fused task representation for the final prediction.


*2025-02-24*

#### [Graph Foundation Models for Recommendation: A Comprehensive Survey](https://arxiv.org/abs/2502.08346)

*Bin Wu, Yihang Wang, Yuanhao Zeng, Jiawei Liu, Jiashu Zhao, Cheng Yang, Yawen Li, Long Xia, Dawei Yin, Chuan Shi*

*Arxiv 2024*

This paper reviews exisitng works of utilizing graph foundation models to enhance recommender systems. It categorizes existing works in the view of collaboration between GNNs and LLMs, including (1) graph-augmented LLMs, which use graph structural information to assist LLMs' decision process, (2) LLM-augmented graphs, which use LLMs' world knowledge to enhance GNNs' performance, and (3) LLM-graph harmonization, which aligns both embedding spaces of GNN and LLM via transformation.


*2025-01-18*

#### [Rethinking Byzantine Robustness in Federated Recommendation from Sparse Aggregation Perspective](https://arxiv.org/abs/2501.03301)

*张中健，张梦玫，王啸，吕灵娟，闫博，杜军平，石川*

*AAAI 2025*

This paper explores the Byzantine robustness in federated recommendation from the sparse aggregation perspective. It first reformulates the Byzantine robustness under sparse aggregation by defining the aggregation for a single item as the smallest execution unit. Then it proposes a family of effective attack strategies which exploit the vulnerability in sparse aggregation, categorized along the adversary's knowledge and capability.


*2025-01-04*

#### [Instructing and Prompting Large Language Models for Explainable Cross-domain Recommendations](https://dl.acm.org/doi/10.1145/3640457.3688137)

*Alessandro Petruzzelli, Cataldo Musto, Lucrezia Laraspata, Ivan Rinaldi, Marco de Gemmis, Pasquale Lops, Giovanni Semeraro*

*RecSys 2024*

This paper proposes a pipeline for cross-domain recommendation, which utilizes a personalized prompt, based on the preferences of the user in a source domain, and a list of items to be ranked in target domain. It prompts the LLM in both zero-shot and one-shot settings, and processes the answer to extract the recommendations with a natural language explanation.


*2024-12-03*

#### [Deep Ensemble Shape Calibration: Multi-Field Post-hoc Calibration in Online Advertising](https://dl.acm.org/doi/10.1145/3637528.3671529)

*Shuai Yang, Hao Yang, Zhuang Zou, Linhe Xu, Shuo Yuan, Yifan Zeng*

*KDD 2024*

In the e-commerce advertising scenario, estimating the true probabilities (known as a calibrated estimate) on Click-Through Rate and Conversion Rate is critical. This paper proposes an ensemble approach for multi-field calibration, by composing it into value calibration and shape calibration, and introducing innovative basis calibration functions, which enhance both function expression capabilities and data utilization by combining these basis calibration functions.


*2024-12-02*

#### [Pay Attention to Attention for Sequential Recommendation](https://arxiv.org/abs/2410.21048)

*Yuli Liu, Min Liu, Xiaojing Liu*

*RecSys 2024*

This paper proposes a new approach for sequential recommendation called attention weight refinement. Specifically, it implements an "attention over attention scores" mechanism, aiming at allowing for more refined attention distributions of correlations among items.


*2024-12-01*

#### [DLCRec: A Novel Approach for Managing Diversity in LLM-Based Recommender Systems](https://arxiv.org/abs/2408.12470)

*Jiaju Chen, Chongming Gao, Shuai Yuan, Shuchang Liu, Qingpeng Cai, Peng Jiang*

*KDD 2024*

To improve the ability of diversified recommendation, this paper proposes a framework with fine-grained control over diversity of user preferences. Specifically, the framework consists of three sub-tasks, namely, genre prediction, genre filling, and item prediction. These sub-tasks are trained independently and inferred sequentially according to user-defined control numbers, ensuring more precise control over diversity. Besides, to overcome the challenge of data scarcity and uneven distribution of diversity-related user behavior for fine-tuning, two data augmentation techniques are introduced to enhance the model's robustness to noisy and out-of-distribution data.


*2024-11-23*

#### [Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation](https://arxiv.org/abs/2307.16089)

*Andreea Iana, Goran Glavaš, Heiko Paulheim*

*EMNLP 2024*

This paper proposes a news recommendation framework that facilitates model reusements. Specifically, the framework consists of not only a content-based recommender (by fine-tuning a PLM), but also individual recommenders for multiple aspects. Each aspect contains multiple classes, and each downstream recommendation task has a personalized preference over the aspects, thus achieving personalized recommendation without changing each individual aspect recommender.


*2024-11-13*

#### [Meta Graph Learning for Long-tail Recommendation](https://dl.acm.org/doi/10.1145/3580305.3599428)

*Chunyu Wei, Jian Liang, Di Liu, Zehui Dai, Mang Li*

*KDD 2023*

To handle the task of recommendation for long-tail entities, this paper proposes a meta-learning framework that avoids the issues of skewed downstream information and negative transfer. Specifically, it first trains a meta edge generator to reconstruct a debiased item co-occurrence matrix from historical interactions, from which it is fine-tuned to extract the collaborative relations among items from their attributes. In the evaluation phase, the unbiased meta edge generator produces item relations as an auxiliary graph, which will participate in the representation learning of items and adapt to the downstream recommendation tasks.


*2024-11-12*

#### [Improving Long-Tail Item Recommendation with Graph Augmentation](https://dl.acm.org/doi/10.1145/3583780.3614929)

*Sichun Luo, Chen Ma, Yuanzhang Xiao, Linqi Song*

*CIKM 2023*

This paper proposes a graph augmentation approach for long-tail recommendation, which can be plugged into any graph-based recommendation models to improve the performance for tail items. It incorporates an edge addition module that enriches the graph's connectivity for tail items by injecting item-to-item edges between similar items based on user-item interaction records. Besides, to balance the graph structure, it utilizes a degree-aware edge dropping strategy, preserving the more valuable edges from the tail items while selectively discarding less informative edges from the head items.


*2024-07-29*

#### [A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models](https://dl.acm.org/doi/10.1145/3626772.3657813)

*Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, Guido Zuccon*

*SIGIR 2024*

This paper proposes a LLM-based setwise document ranking approach, which improves the existing listwise ranking approach by applying batch inference of the LLMs to generate ranking scores. Compared with existing pointwise, pairwise and listwise approaches, it improves the efficiency with fewer LLM calls, while also retaining high zero-shot ranking effectiveness.


*2024-05-12*

#### [Wyze Rule: Federated Rule Dataset for Rule Recommendation Benchmarking](https://papers.nips.cc/paper_files/paper/2023/hash/02b9d1e6d1b5295a6f883969ddc1bbbd-Abstract-Datasets_and_Benchmarks.html)

*Mohammad Mahdi Kamani, Yuhang Yao, Hanjia Lyu, Zhongwei Cheng, Lin Chen, Liangju Li, Carlee Joe-Wong, Jiebo Luo*

*NeurIPS 2023*

Rules are an essential component for smart home automation and IoT devices. To provide a resource for developing and evaluating intelligent smart home rule recommendations, this paper proposes a large-scale dataset designed specifically for smart home rule recommendation research. To establish a usable benchmark for comparison and evaluation, several baselines are also implemented in both centralized and federated settings.


*2024-05-10*

#### [REASONER: An Explainable Recommendation Dataset with Comprehensive Labeling Ground Truths](https://papers.nips.cc/paper_files/paper/2023/hash/2ebf43d20e5933ab6d98225bbb908ade-Abstract-Datasets_and_Benchmarks.html)

*Xu Chen, Jingsen Zhang, Lei Wang, Quanyu Dai, Zhenhua Dong, Ruiming Tang, Rui Zhang, Li Chen, Xin Zhao, Ji-Rong Wen*

*NeurIPS 2023*

This paper proposes a dataset for explainable recommendation that includes a large amount of real user labeled multi-modal and multi-aspect explanation ground truths. Specifically, the authors firstly develop a video recommendation platform, where a series of questions around the recommendation explainability are designed. Then they recruit about 3,000 human labelers with different backgrounds to use the system, and collect their behaviors and feedback to the questions.


*2024-04-30*

#### [SIGformer: Sign-aware Graph Transformer for Recommendation](https://arxiv.org/abs/2404.11982)

*Sirui Chen, Jiawei Chen, Sheng Zhou, Bohao Wang, Shen Han, Chanfei Su, Yuqing Yuan, Can Wang*

*SIGIR 2024*

To collaboratively utilize the positive and negative feedback in the recommendation system, this paper proposes a model that employs the transformer architecture to sign-aware graph-based recommendation. It incorporates two positional encodings that capture the spectral properties and path patterns of the signed graph, to improve exploitation of the entire graph.


*2024-04-01*

#### [Can Small Language Models be Good Reasoners for Sequential Recommendation?](https://arxiv.org/abs/2403.04260)

*Yuling Wang, Changxin Tian, Binbin Hu, Yanhua Yu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Liang Pang, Xiao Wang*

*WWW 2024*

This paper proposes a distillation-based model for sequential recommendation. Specifically, it utilizes a frozen large LM to produce CoT prompts and rationales. Then feeds both to the small student model as prompts and labels to let the student model acquire step-by-step recommendation capability.


*2024-03-30*

#### [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545)

*Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Daifeng Guo, Yanli Zhao, Shen Li, Yuchen Hao, Yantao Yao, Guna Lakshminarayanan, Ellie Dingqiao Wen, Jongsoo Park, Maxim Naumov, Wenlin Chen*

*Arxiv 2024*

To capture the scaling law in the recommendation field, this paper proposes an effective network architecture based purely on stacked factorization machines, and a synergistic upscaling strategy, to establish a scaling law in the domain of recommendation. This unique design makes it possible to capture diverse, any-order of interactions simply through taller and wider layers.


*2024-03-28*

#### [PPM : A Pre-trained Plug-in Model for Click-through Rate Prediction](https://arxiv.org/abs/2403.10049)

*Yuanbo Gao, Peng Lin, Dongyue Wang, Feng Mei, Xiwei Zhao, Sulong Xu, Jinghe Hu*

*WWW 2024*

This paper proposes a pre-trained plug-in CTR model, namely PPM. PPM employs multi-modal features as input and utilizes large scale data for pre-training. Then, PPM is plugged in IDRec model to enhance unified model’s performance and iteration efficiency. Upon incorporating IDRec model, certain intermediate results within the network are cached, with only a subset of the parameters participating in training and serving. Hence, our approach can successfully deploy an end-to-end model without causing huge latency increases.


*2024-01-18*

#### [Knowledge Graph Context-Enhanced Diversified Recommendation](https://arxiv.org/abs/2310.13253)

*Xiaolong Liu, Liangwei Yang, Zhiwei Liu, Mingdai Yang, Chen Wang, Hao Peng, Philip S. Yu*

*WSDM 2024*

This paper investigates diversified recommendation with knowledge graphs as context, where the KGs act as repositories of interconnected information concerning entities and items. It first proposes metrics of entity and relation coverage to quantify the diversity within the KG, and introduces the diversified embedding learning approach to characterize diversity-aware user representations. Besides, it encodes KG items while preserving contextual integrity using conditional alignment and uniformity techniques.


*2024-01-12*

#### [AT4CTR: Auxiliary Match Tasks for Enhancing Click-Through Rate Prediction](https://arxiv.org/abs/2312.06683)

*Qi Liu, Xuyang Hou, Defu Lian, Zhe Wang, Haoran Jin, Jia Cheng, Jun Lei*

*AAAI 2024*

This paper proposes auxiliary match tasks to enhance Click-through rate (CTR) prediction by alleviating the data sparsity problem. Specifically, it designs two match tasks inspired by collaborative filtering to enhance the relevance modeling between user and item. The first match task aim at pulling closer the representation between the user and the item regarding the positive samples, and the second match task is next item prediction.


*2024-01-11*

#### [STEM: Unleashing the Power of Embeddings for Multi-task Recommendation](https://arxiv.org/abs/2308.13537)

*Liangcai Su, Junwei Pan, Ximei Wang, Xi Xiao, Shijie Quan, Xihua Chen, Jie Jiang*

*AAAi 2024*

A key challenge for multitask learning (MTL) where all tasks employ a shared-embedding paradigm is negative transfer. Existing studies explored negative transfer on all samples, but surprisingly, an evaluation shows that negative transfer still occurs in existing MTL methods on samples that receive comparable feedback across tasks. Based on this, the paper assumes it is limited by the shared embeddings, and proposes a Shared and Task-specific EMbeddings (STEM) scheme that aims to incorporate both shared and task-specific embeddings to effectively capture task-specific user preferences.


*2023-12-19*

#### [RecExplainer: Aligning Large Language Models for Recommendation Model Interpretability](https://arxiv.org/abs/2311.10947)

*Yuxuan Lei, Jianxun Lian, Jing Yao, Xu Huang, Defu Lian, Xing Xie*

*Arxiv 2023*

This paper proposes to use LLMs for explaining recommendation models. Specifically, it considers two ways for aligning the LLM with the recommendation model, including (1) behavior alignment for which the LLM is trained to emulate the recommendation model’s predictive patterns—given a user’s profile as input, the LLM is fine-tuned to predict the items that the recommendation model would suggest to the user, and (2) intention alignment aims to incorporate the embeddings from the recommendation model to the LLM's prompts in a cross-modal manner.


*2023-12-14*

#### [ControlRec: Bridging the Semantic Gap between Language Model and Personalized Recommendation](https://arxiv.org/abs/2311.16441)

*Junyan Qiu, Haitao Wang, Zhaolin Hong, Yiping Yang, Qiang Liu, Xingxing Wang*

*Arxiv 2023*

An existing obstacle for using LLMs in the recommendation field is the representations in the semantic space used in recommendation such as user and item IDs, is distinct from natural language. To address this issue, this paper proposes two approaches to improve the alignment between user IDs and NL: (1) Heterogeneous Feature Matching (HFM) aligning item description (NL) with the corresponding ID or user’s next preferred ID based on their interaction sequence, and (2) Instruction Contrastive Learning (ICL) effectively merging these two crucial data sources by contrasting probability distributions of output sequences generated by diverse tasks.


*2023-12-10*

#### [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)

*Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy*

*NeurIPS 2023*

Modern recommender systems perform large-scale retrieval by embedding queries and item candidates in the same unified space, followed by approximate nearest neighbor search to select top candidates given a query embedding. In contrast, this paper enables the retrieval model to autoregressively decode the identifiers of the target candidates by creating semantically meaningful tuple of codewords to serve as a Semantic ID for each item. Given Semantic IDs for items in a user session, a Transformer-based seq2seq model is trained to predict the Semantic ID of the next item that the user will interact with.


*2023-12-09*

#### [Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction](https://dl.acm.org/doi/10.1145/3583780.3615089)

*Fangye Wang, Hansu Gu, Dongsheng Li, Tun Lu, Peng Zhang, Ning Gu*

*CIKM 2023*

CTR prediction aims to estimate the probability of a user clicking on a recommended item or an advertisement on a web page. This paper proposes a Gated Cross Network (GCN) that captures explicit high-order feature interactions and dynamically filters important interactions with an information gate in each order. Then it also uses a Field-level Dimension Optimization (FDO) approach to learn condensed dimensions for each field based on their importance.


*2023-11-30*

#### [LLM4Vis: Explainable Visualization Recommendation using ChatGPT](https://arxiv.org/abs/2310.07652)

*Lei Wang, Songheng Zhang, Yun Wang, Ee-Peng Lim, Yong Wang*

*EMNLP 2023*

Visualization recommendation aims to find the most appropriate type of visualization approach for a given dataset. To achieve this, this paper proposes a prompt-based pipeline, which involves feature description, example selection, and explanation generation (by LLM). Then in the test phase, it searches for the top-k nearest examples from the example set, and combines them with the input test example to generate the visualization recommendation.


*2023-11-23*

#### [Session-based Recommendations with Recurrent Neural Networks](https://github.com/hidasib/GRU4Rec)

*Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk*

*ICLR 2016*

This is an early work that uses RNN, specifically, GRU-based network models, to simulate the long history of user-item interactions for recommendation.


*2023-11-19*

#### [LLMRec: Large Language Models with Graph Augmentation for Recommendation](https://arxiv.org/abs/2311.00423)

*Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang*

*WSDM 2024*

This paper proposes to use LLM to generate extra information for augmenting the user-item interation graph thus improving the recommendation performance. Specifically, it applies LLM to (1) generate pseudo user-item edges as extra positive/negative examples, (2) generate user profiles such as name, gender from the recommendation history, and (3) encode the generated information as auxiliary features used as input for the recommender.


*2023-11-11*

#### [How Expressive are Graph Neural Networks in Recommendation?](https://dl.acm.org/doi/10.1145/3583780.3614917)

*Xuheng Cai, Lianghao Xia, Xubin Ren, Chao Huang*

*CIKM 2023*

This paper studies the problem of measuring the expressiveness of GNN models in the recommendation context. Specifically, it first introduces expressiveness metrics on three levels, i.e., graph level, node level, and link level. In particular, it proposes a topological closeness metric to evaluate GNNs’ ability to capture the structural distance between nodes, which closely aligns with the recommendation objective.


*2023-11-06*

#### [Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach](https://arxiv.org/abs/2305.07001)

*Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, Ji-Rong Wen*

*Arxiv 2023*

This paper proposes to apply LLMs for the sequence recommendation task. It first proposes a general instruction format to describe the preference, intention, task form and the context for recommendation. Then it manually designs 39 instruction templates and uses them to generate a set of user-personalized instructions data.


*2023-11-05*

#### [Prompt Distillation for Efficient LLM-based Recommendation](https://dl.acm.org/doi/10.1145/3583780.3615017)

*Lei Li, Yongfeng Zhang, Li Chen*

*CIKM 2023*

This work explores to use LLM for recommendation by prompt distillation. Generally, recommendation requires user and item IDs as the input for the LLM, usually used as prompts. This paper proposes to learn a shorter dense vector as prompt to replace the original longer sparse prompt. Concretely, it appends a set of vectors at the beginning of an input sample that already filled in a discrete prompt template, and allow the vectors to be shared by the samples of the same recommendation task. It evaluates the proposed method for sequence recommendation, top-N recommendation, and generation of explanations.

