






*2025-05-21*

#### [LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!](https://arxiv.org/abs/2502.07374)

*Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde, Kourosh Hakhamaneshi, Shishir G. Patil, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica*

*ICLR 2025*

This work shows that a LLM can effectively learn Long CoT reasoning through data-efficient supervised fine-tuning (SFT) and LoRA. Importantly, it uncovers that the structure of Long CoT is critical to the learning process, whereas the content of individual reasoning steps has minimal impact. Perturbations affecting content, such as training on incorrect samples or removing reasoning keywords, have little impact on performance. In contrast, structural modifications that disrupt logical consistency in the Long CoT, such as shuffling or deleting reasoning steps, significantly degrade accuracy.


*2025-05-20*

#### [CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction](https://arxiv.org/abs/2502.07316)

*Junlong Li, Daya Guo, Dejian Yang, Runxin Xu, Yu Wu, Junxian He*

*ICLR 2025*

This paper proposes CodeI/O, an approach that condenses diverse reasoning patterns inherently embedded in contextually-grounded codes, through transforming the original code into a code input-output prediction format, and training models to predict inputs/outputs given code and test cases entirely in natural language as Chain-of-Thought (CoT) rationales.


*2025-05-19*

#### [Even Small Reasoners Should Quote Their Sources: Introducing the Pleias-RAG Model Family](https://arxiv.org/abs/2504.18225)

*Pierre-Carl Langlais, Pavel Chizhov, Mattia Nee, Carlos Rosas Hinostroza, Matthieu Delsart, Irène Girard, Othman Hicheur, Anastasia Stasenko, Ivan P. Yamshchikov*

*Arxiv 2025*

This paper proposes an efficient RAG model with ~1B parameters. Its workflow involves: query analysis to draw assumptions on the intent of the query and clarify the kind and format of information the user wants to retrieve; a query report with standardized output; a source analysis identifying the sources most likely to contain answering elements to query and hierarchize this information; a source report: either extensive (sources provide all the required material to write an answer in detail), basic (enough to write an answer), incomplete (potentially enough to answer partially), or infeasible (will result in a refusal).


*2025-05-17*

#### [From Superficial to Deep: Integrating External Knowledge for Follow-up Question Generation Using Knowledge Graph and LLM](https://aclanthology.org/2025.coling-main.55/)

*Jianyu Liu, Yi Huang, Sheng Bi, Junlan Feng, Guilin Qi*

*COLING 2025*

This paper proposes an external knowledge-enhanced follow-up question generation method, which identifies contextual topics, constructs a knowledge graph online, and finally combines them with a LLM to generate the final question.


*2025-05-11*

#### [Synergizing RAG and Reasoning: A Systematic Review](https://arxiv.org/abs/2504.15909)

*Yunfan Gao, Yun Xiong, Yijie Zhong, Yuxi Bi, Ming Xue, Haofen Wang*

*Arxiv 2025*

This survey paper introduces the techniques, pipeline, and key aspects for the combination of RAG and reasoning. The collaboration targets include reasoning-augmented retrieval (the reasoning result guides next-step retrieval), and retrieval-augmented reasoning (to provide more valid facts for reasoning).


*2025-05-06*

#### [LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph](https://arxiv.org/abs/2504.03137)

*Tu Ao, Yanhua Yu, Yuling Wang, Yang Deng, Zirui Guo, Liang Pang, Pinghui Wang, Tat-Seng Chua, Xiao Zhang, Zhen Cai*

*AAAI 2025*

This paper proposes a retrieval-embed-generate pipeline for multi-hop knowledge graph question answering. Given a question, it identifies the number of hops, gets all relational paths by BFS, ranks the paths by LLMs, embeds the knowledge paths into soft prompts, and finally feds them with hard prompts to generate the answer.


*2025-05-04*

#### [OSCAR: Online Soft Compression And Reranking](https://arxiv.org/abs/2504.07109)

*Maxime Louis, Thibault Formal, Hervé Dejean, Stéphane Clinchant*

*Arxiv 2025*

The query $q$, the $i$-th retrieved document $d_i$, and a set of memory tokens $\text{MEM}_i$ are fed forward to a compressor LLM $\mathcal{C}$. It uses the hidden states corresponding to each memory token as query-dependent embedding representations.


*2025-05-03*

#### [HyperGraphRAG: Retrieval-Augmented Generation with Hypergraph-Structured Knowledge Representation](https://arxiv.org/abs/2503.21322)

*Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin Kuang, Meina Song, Yifan Zhu, Luu Anh Tuan*

*Arxiv 2025*

This paper proposes a graph-RAG for hypergraphs, including hypergraph construction, retrieval strategy implementation, and combination with typical RAG approach.


*2025-05-02*

#### [Retrieval-Augmented Generation with Hierarchical Knowledge](https://arxiv.org/abs/2503.10150v1)

*Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, James Cheng*

*Arxiv 2025*

This paper proposes a hierarchical approach based on graph structure to index the underlying documents. By extracting entities and relations from the texts, it first constructs a fundamental KG at the ground level. Then it performs community detection and generating summaries for each cluster, to gradually build the upper level of indexes.


*2025-05-01*

#### [GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning](https://arxiv.org/abs/2405.20139)

*Costas Mavromatis, George Karypis*

*Arxiv 2024*

This paper utilizes a GNN model to retrieve answer candidates for a given question, then the shortest paths that connect question entities and answer candidates are extracted to represent KG reasoning paths for LLM’s final output.


*2025-04-28*

#### [SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](https://arxiv.org/abs/2504.08600)

*Peixian Ma, Xialie Zhuang, Chengjin Xu, Xuhui Jiang, Ran Chen, Jian Guo*

*Arxiv 2025*

This paper proposes a reinforcement learning based approach for supervised fine-tuning of NL2SQL LMs. The RL reward function includes: Format Reward, Execution Reward, Result Reward, and Length Reward.


*2025-04-26*

#### [OneEval]([http://oneeval.openkg.cn/](https://mp.weixin.qq.com/s/BeKah91_texXN3s1WAOcKg))

A systematic evaluation framework for LLM+KG reasoning and understanding abilities.


*2025-04-25*

#### [DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2503.00223)

*Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng Wang, Jimeng Sun, Jiawei Han*

*Arxiv 2025*

This paper proposes a reinforcement learning approach that trains LLMs for query generation through trial and error without supervised data (reference query). Using retrieval metrics as rewards, the system generates queries that maximize retrieval performance.


*2025-04-24*

#### [Pre-train, Align, and Disentangle: Empowering Sequential Recommendation with Large Language Models](https://arxiv.org/abs/2412.04107)

*汪宇豪，潘军伟，贾鹏越，王婉玉，王茂林，冯志祥，李笑天，蒋杰，赵翔宇*

*SIGIR 2025*

This paper proposes a LLM-based sequential recommendation approach. It first pre-trains both the SR and LLM models to get collaborative and textual embeddings. Next, it proposes a recommendation-anchored alignment loss using multi-kernel maximum mean discrepancy with Gaussian kernels. Finally, it fine-tunes a triple-experts architecture.


*2025-04-23*

#### [ZeroED: Hybrid Zero-shot Error Detection through Large Language Model Reasoning](https://arxiv.org/abs/2504.05345)

*Wei Ni, Kaihang Zhang, Xiaoye Miao, Xiangyu Zhao, Yangyang Wu, Yaoshu Wang, Jianwei Yin*

*ICDE 2025*

This paper proposes a tabular data error detection framework that works in 4 steps: (1) generate data representations using error reason-aware binary features, pre-trained embeddings, and statistical features, (2) use LLMs to label errors holistically via in-context learning, guided by a two-step reasoning process for detailed error detection guidelines, (3) to reduce token costs, LLMs are applied only to representative data selected via clustering-based sampling. High-quality training data is constructed through in-cluster label propagation and LLM augmentation with verification, and (4) train the classifier.


*2025-04-21*

#### [Knowledge Graph-Driven Retrieval-Augmented Generation: Integrating Deepseek-R1 with Weaviate for Advanced Chatbot Applications](https://arxiv.org/abs/2502.11108)

*Alexandru Lecu, Adrian Groza, Lezan Hawizy*

*Arxiv 2025*

This paper introduces a medical chatbot system based on KG-supported RAG.


*2025-04-19*

#### [GOFA: A Generative One-For-All Model for Joint Graph Language Modeling](https://arxiv.org/abs/2407.09709)

*Lecheng Kong, Jiarui Feng, Hao Liu, Chengsong Huang, Jiaxin Huang, Yixin Chen, Muhan Zhang*

*ICLR 2025*

The model interleaves randomly initialized GNN layers into a frozen pre-trained LLM so that the semantic and structural modeling abilities are combined. It is pre-trained on graph-level next-word prediction, question-answering, and structural tasks.


*2025-04-18*

#### [A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond](https://arxiv.org/abs/2503.21614)

*Xiaoye Qu, Yafu Li, Zhaochen Su, Weigao Sun, Jianhao Yan, Dongrui Liu, Ganqu Cui, Daizong Liu, Shuxian Liang, Junxian He, Peng Li, Wei Wei, Jing Shao, Chaochao Lu, Yue Zhang, Xian-Sheng Hua, Bowen Zhou, Yu Cheng*

*Arxiv 2025*

A growing concern lies in existing LLMs to produce excessively long reasoning traces, which are often filled with redundant content (e.g., repeated definitions), over-analysis of simple problems, and superficial exploration of multiple reasoning paths for harder tasks.


*2025-04-17*

#### [Why do LLMs attend to the first token?](https://arxiv.org/abs/2504.02732)

*Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu*

*Arxiv 2025*

This paper explores the attention sink phenomenon in LLMs, i.e., LLMs tend to attend heavily to the first token (e.g., <bos>) in the sequence. It argues that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers.


*2025-04-15*

#### [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100)

*Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, Tianqi Chen*

*Arxiv 2024*

This paper proposes an efficient structure generation engine for large language models. Combined with an LLM inference engine, it can achieve near-zero overhead structure generation in end-to-end low-LLM serving.


*2025-04-09*

#### [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)

*Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, Jiawei Han*

*Arxiv 2025*

This paper introduces Search-R1, an extension of the DeepSeek-R1 model where the LLM learns solely through reinforcement learning to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM rollouts with multi-turn search interactions, leveraging retrieved token masking for stable RL training and an outcome-based reward function.


*2025-04-08*

#### [Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs](https://openreview.net/forum?id=CwCUEr6wO5)

*Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun, Jieping Ye, Hui Xiong*

*NeurIPS 2024*

This paper proposes a KG-augmented LLM named Plan-on-Graph (PoG), which first decomposes the question into several sub-objectives and then repeats the process of adaptively exploring reasoning paths, updating memory, and reflecting on the need to self-correct erroneous reasoning paths until arriving at the answer.


*2025-04-07*

#### [StructGPT: A General Framework for Large Language Model to Reason over Structured Data](https://aclanthology.org/2023.emnlp-main.574/)

*Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Xin Zhao, Ji-Rong Wen*

*EMNLP 2023*

This paper proposes a unified approach to retrieve relevant facts from structured data to support LLM-based question answering. The structured data considered in this paper include KGs, single tables and databases (i.e., multiple tables).


*2025-04-05*

#### [Code Search is All You Need? Improving Code Suggestions with Code Search](https://dl.acm.org/doi/10.1145/3597503.3639085)

*Junkai Chen, Xing Hu, Zhenhao Li, Cuiyun Gao, Xin Xia, David Lo*

*ICSE 2024*

This paper introduces a retrieval-augmented code generation approach, by firstly searching for similar code pieces, combining them with context prompts, and generating token-level/line-level/complete codes for the context.


*2025-04-03*

#### [KnowGPT: Knowledge Graph based Prompting for Large Language Models](https://openreview.net/forum?id=PacBluO5m7)

*Qinggang Zhang, Junnan Dong, Hao Chen, Daochen Zha, Zailiang Yu, Xiao Huang*

*NeurIPS 2024*

This paper proposes a knowledge graph prompting framework. Given the question context with multiple choices, it first retrieves a question-specific subgraph from the KG. Then the Prompt Construction module combines the paths retrieved from the subgraph into the prompts, and feeds the priorotized prompt into LLM to get the answer.


*2025-04-01*

#### [Estimation of single-cell and tissue perturbation effect in spatial transcriptomics via Spatial Causal Disentanglement](https://openreview.net/forum?id=Tqdsruwyac)

*Stathis Megas, Daniel G. Chen, Krzysztof Polanski, Moshe Eliasof, Carola-Bibiane Schönlieb, Sarah A Teichmann*

*ICLR 2025*

This paper proposes a generative graph neural network to infer and disentangle the causal structure diagram of feature interactions from spatial samples such as spatial transcriptomics data. The method can also be used to predict spatial perturbation effect in silico.


*2025-03-31*

#### [From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions](https://arxiv.org/abs/2410.08197)

*Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, Ji-Rong Wen*

*ICLR 2025*

This paper proposes a framework to promote the ability of LLMs to utilize external tools, by prompting the LLM to (1) explore tool documentations, (2) learning from them, and (3) rewriting tool documentation to better fit the utilization by LLMs.


*2025-03-29*

#### [OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning](https://arxiv.org/abs/2503.08398)

*Jiawei Zhou, Lei Chen*

*Arxiv 2025*

This paper proposes an end-to-end RAG framework, which aims to retrieve documents on-the-fly and identify them as positive or negative for contrastive learning (of the retriever).


*2025-03-28*

#### [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779)

*Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang*

*Arxiv 2024*

This paper proposes a RAG framework that incorporates low-level and high-level knowledge retrieval, with graph structure to organize related entities. The evaluation also includes generating abstract questions, which is mainly by prompting LLMs to simulate RAG users with given knowledge acquisition tasks.


*2025-03-27*

#### [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)

*Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson*

*Arxiv 2024*

The technical report of Microsoft GraphRAG framework.


*2025-03-26*

#### [In-depth Analysis of Graph-based RAG in a Unified Framework](https://arxiv.org/abs/2503.04338)

*Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He, Yongwei Zhang, Sicong Liang, Xilin Liu, Yuchi Ma, Yixiang Fang*

*Arxiv 2025*

This paper compares existing graph-based RAG approaches under a unified framework and analyzes their results. The main process of the framework includes (1) graph building, (2) index construction, (3) operator configuration, and (4) retrieve and response. Note that this paper also talks about abstract questions, i.e., questions that are not specific to some fact, but at a higher level and more general, example datasets such as Mix [65], MultihopSum [74], Agriculture [65], CS [65], and Legal [65].


*2025-03-25*

#### [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)

*Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef*

*Arxiv 2025*

While existing RAG frameworks for LLM-based agents are limited to static document retrieval, enterprise applications demand dynamic knowledge integration from diverse sources including ongoing conversations and business data. This paper introduces a memory layer service for AI agents with a temporally-aware knowledge graph engine to dynamically synthesize both unstructured conversational data and structured business data while maintaining historical relationships.


*2025-03-23*

#### [SampleLLM: Optimizing Tabular Data Synthesis in Recommendations](https://arxiv.org/abs/2501.16125)

*Jingtong Gao, Zhaocheng Du, Xiaopeng Li, Yichao Wang, Xiangyang Li, Huifeng Guo, Ruiming Tang, Xiangyu Zhao*

*WWW 2025*

This paper proposes an LLM-based approach to generate synthetic tabular data in the recommendation domain. Specifically, it uses a two-stage process, by first employing LLMs with CoT prompts and diverse exemplars to generate data that closely aligns with the target dataset distribution, and the second stage uses a feature attribution-based importance sampling method to refine feature relationships within the synthesized data, reducing any distribution biases introduced by the LLM.


*2025-03-19*

#### [Graph Foundation Models: Concepts, Opportunities and Challenges](https://ieeexplore.ieee.org/document/10915556)

*Jiawei Liu*, Cheng Yang*, Zhiyuan Lu, Junze Chen, Yibo Li, Mengmei Zhang, Ting Bai, Yuan Fang, Lichao Sun, Philip S. Yu, Chuan Shi*

*TPAMI 2025*

This paper first presents the graph foundation models, and summarizes their similarities and differences to LLMs. It analyzes existing graph foundational models by 3 categories, namely, GNN-based models, LLM-based models, and GNN+LLM-based models, respectively.


*2025-03-18*

#### [Graph RAG-Tool Fusion](https://arxiv.org/abs/2502.07223)

*Elias Lumer, Pradeep Honaganahalli Basavaraju, Myles Mason, James A. Burke, Vamse Kumar Subbiah*

*Arxiv 2025*

Existing RAG-based frameworks for selecting agents or tools to complete complex tasks uaually fail to capture the structural dependencies between tools. To mitigate the issue, this paper introduces a plug-and-play approach by vector-based retrieval with graph traversal to capture all relevant tools (nodes), along with any nested dependencies (edges) within a predefined tool knowledge graph.


*2025-03-16*

#### [A Pilot Empirical Study on When and How to Use Knowledge Graphs as Retrieval Augmented Generation](https://arxiv.org/abs/2502.20854)

*Xujie Yuan, Yongxu Liu, Shimin Di, Shiwen Wu, Libin Zheng, Rui Meng, Lei Chen, Xiaofang Zhou, Jian Yin*

*Arxiv 2025*

To evaluate existing KG-RAG frameworks, this paper analyzes their performance in various application scenarios associated with different technical configurations. It conducts an empirical study of KG-RAG works to reimplement and evaluate 6 KG-RAG methods across 7 datasets in multiple scenarios, analyzing the impact of 9 KG-RAG configurations in combination with 17 LLMs.


*2025-03-15*

#### [A Survey on Complex Reasoning of Large Language Models through the Lens of Self-Evolution](https://www.researchgate.net/publication/389209259_A_Survey_on_Complex_Reasoning_of_Large_Language_Models_through_the_Lens_of_Self-Evolution)

*Tao He, Hao Li, Jingchang Chen, Runxuan Liu, Yixin Cao, Lizi Liao, Zihao Zheng, Zheng Chu, Jiafeng Liang, Ming Liu, Bing Qin*

*Preprint*

Self-evolution cycles self-play and iterative learning to enable autonomous evolution of reasoning capabilities. This paper reviews existing works related to self-evolution under a category whth 3 groups: data evolution, model evolution, and combined data-and-model evolution.


*2025-03-12*

#### [From Specific-MLLMs to Omni-MLLMs: A Survey on MLLMs Aligned with Multi-modalities](https://arxiv.org/abs/2412.11694)

*Shixin Jiang, Jiafeng Liang, Jiyuan Wang, Xuan Dong, Heng Chang, Weijiang Yu, Jinhua Du, Ming Liu, Bing Qin*

*Arxiv 2025*

This paper reviews the potential roadmap for realizing omni-multimodal LLM, by (1) investigating relevant research and describing core components with a meticulous taxonomy, (2) introducing the effective integration of two-stage training with corresponding datasets as well as evaluation, (3) summarizing challenges and future directions.


*2025-03-10*

#### [Do LLMs Really Adapt to Domains? An Ontology Learning Perspective](https://arxiv.org/abs/2407.19998)

*Huu Tan Mai, Cuong Xuan Chu, Heiko Paulheim*

*ISWC 2024*

This paper investigates the question: Do LLMs really adapt to domains and remain consistent in structured knowledge extraction, or do they only learn lexical senses instead of reasoning? By devising a controlled experiment that uses WordNet to synthesize parallel corpora, with English and gibberish terms, empirical results show that while adapting to the gibberish corpora, off-the-shelf LLMs do not consistently reason over semantic relationships between concepts, instead, they rely on the lexical senses. Fine-tuning improves the performance on lexical semantic tasks even when domain-specific terms are unseen during pre-training, hinting at the applicability of pre-trained LLMs for ontology learning.


*2025-03-07*

#### [Knowledge Graph-Guided Retrieval Augmented Generation](https://arxiv.org/abs/2502.06864)

*Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, Wei Hu*

*NAACL 2025*

To address the hallucination issue of LLMs, existing RAG approaches usually retrieve relevant but isolated passage chunks, while ignoring the semantic relations between these retrieved results. This paper proposes a solution by introducing a KG-guided RAG framework. In particular, it first pre-processes documents by dividing them into chunks and linked to a KG. Then the retrieval process is conducted with two stages, including semantic-based retrieval and graph-guided expansion.


*2025-03-06*

#### [Controllable Protein Sequence Generation with LLM Preference Optimization](https://arxiv.org/abs/2501.15007)

*Xiangyu Liu, Yi Liu, Silei Chen, Wei Hu*

*AAAI 2025*

Although pre-trained protein large language models (LLMs) have shown promising results on protein sequence generation, existing works still struggle on controlling sequence generation for specific attributes, with poor functionality and structural stability. To solve this problem, this paper proposes a controllable protein design method by finetuning a protein LLM with a new multi-listwise preference optimization strategy to improve generation quality and support multi-attribute controllable generation.


*2025-03-05*

#### [Targeted training for numerical reasoning with large language models](https://link.springer.com/article/10.1007/s10115-024-02216-1)

*Xiao Li, Sichen Liu, Yin Zhu, Gong Cheng*

*Knowledge and Information Systems*

This paper explores the task of instructing LLMs to generate CoT examples for fine-tuning smaller models for numerical reasoning. As smaller models are usually passive in this line of work and may not be able to exploit the provided training data, this paper proposes a targeted training strategy to match LLM’s assistance with small models’ capacities. The small model proactively requestd LLM’s assistance when it sifts out confusing training data. Then, LLM refines such data by successively revising reasoning steps and reducing question complexity before feeding the small model.


*2025-03-03*

#### [PIKE-RAG: sPecIalized KnowledgE and Rationale Augmented Generation](https://arxiv.org/abs/2501.11551)

*Jinyu Wang, Jingjing Fu, Rui Wang, Lei Song, Jiang Bian*

*Arxiv 2025*

Despite notable advancements in RAG systems that expand LLM capabilities through external retrieval, these systems often struggle to meet the complex and diverse needs of real-world industrial applications. To address that, this technical report introduces PIKE-RAG, focusing on extracting, understanding, and applying specialized knowledge, while constructing coherent rationale to incrementally steer LLMs toward accurate responses. Recognizing the diverse challenges of industrial tasks, it introduces a new paradigm to classify tasks based on their complexity in knowledge extraction and application, allowing for a systematic evaluation of RAG systems' problem-solving capabilities.


*2025-02-28*

#### [LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities](https://arxiv.org/abs/2305.13168)

*Yuqi Zhu, Xiaohan Wang, Jing Chen, Shuofei Qiao, Yixin Ou, Yunzhi Yao, Shumin Deng, Huajun Chen, Ningyu Zhang*

*WWWJ*

This paper presents an evaluation of LLMs for KG construction and reasoning, focusing on representative tasks including entity and relation extraction, event extraction, link prediction, and question-answering. Besides, it also proposes a multi-agent-based approach employing LLMs and external sources for KG construction and reasoning.


*2025-02-25*

#### [Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research](https://arxiv.org/abs/2502.04644)

*Junde Wu, Jiayuan Zhu, Yuyuan Liu*

*Arxiv 2025*

This paper proposes a framework to enhance LLM reasoning by integrating external tool-using agents. Unlike conventional LLM-based reasoning which relies solely on internal inference, it dynamically engages web search, code execution, and structured reasoning-context memory to solve complex problems requiring deep research and multi-step logical deduction. The proposed framework introduces an agent to construct a knowledge graph for tracking logical relationships, and the integration of web-search and coding agents enables real-time retrieval and computational analysis.


*2025-02-22*

#### [Goedel-Prover: A Frontier Model for Open-Source Automated Theorem Proving](https://arxiv.org/abs/2502.07640v1)

*Yong Lin, Shange Tang, Bohan Lyu, Jiayun Wu, Hongzhou Lin, Kaiyu Yang, Jia Li, Mengzhou Xia, Danqi Chen, Sanjeev Arora, Chi Jin*

*Arxiv 2025*

This paper introduces Goedel-Prover, an open-source LLM for automated formal proof generation for mathematical problems. The key challenge in this field is the scarcity of formalized math statements and proofs. To tackle the challenge, it trains statement formalizers to translate the natural language math problems from Numina into formal language (Lean 4), creating a dataset of 1.64 million formal statements. LLMs are used to check that the formal statements accurately preserve the content of the original natural language problems. Then the authors iteratively build a large dataset of formal proofs by training a series of provers. Each prover succeeds in proving many statements that the previous ones could not, and these new proofs are added to the training set for the next prover. The final prover outperforms all existing open-source models in whole-proof generation.


*2025-02-13*

#### [KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents](https://arxiv.org/abs/2403.03101)

*Yuqi Zhu, Shuofei Qiao, Yixin Ou, Shumin Deng, Shiwei Lyu, YUE SHEN, Lei Liang, Jinjie GU, Huajun Chen, Ningyu Zhang*

*NAACL 2025*

This paper proposes an approach for enhancing the planning capability of LLMs by incorporating explicit action knowledge. Specifically, it employs an action knowledge base and a knowledgeable self-learning strategy to constrain the action path during planning, enabling more reasonable trajectory synthesis, and thereby enhancing the planning performance of language agents. Initially, *Action Knowledge to Text* converts task-specific action knowledge into textual descriptions. Next, *Planning Path Generation* uses prompts and this knowledge to lead LLMs in planning path creation. Lastly, *Knowledgeable Self-Learning* enables the model iteratively optimize using generated planning trajectories to improve performance.


*2025-02-11*

#### [OntoTune: Ontology-Driven Self-training for Aligning Large Language Models](https://openreview.net/forum?id=d7spHwemKX#discussion)

*Zhiqiang Liu, Chengtao Gan, Junjie Wang, Yichi Zhang, Zhongpu Bo, Mengshu Sun, Wen Zhang, Huajun Chen*

*TheWebConf 2025*

To effectively utilize domain knowledge to improve domain-specific LLMs, this paper proposes an ontology-driven self-training framework that aims to align LLMs with ontology through in-context learning, enabling the generation of responses guided by the ontology. It uses in-context learning to identify whether the LLM has acquired the specific concept's ontology knowledge, and selects the entries not yet mastered by LLM as training set to further align the LLM with ontology.


*2025-02-10*

#### [MLLM can see? Dynamic Correction Decoding for Hallucination Mitigation](http://arxiv.org/abs/2410.11779)

*Chenxi Wang, Xiang Chen, Ningyu Zhang, Bozhong Tian, Haoming Xu, Shumin Deng, Huajun Chen*

*ICLR 2025*

This paper explores the hallucination issue of MLLMs, and finds that the confidence of generated tokens is influenced by the knowledge priors of MLLMs, leading to a reduction in the probability of ground truth tokens in the deeper layers. Further, it proposes a dynamic correction decoding method for MLLMs that adaptively selects appropriate preceding layers and proportionally integrates knowledge into the final layer to adjust the output logits.


*2025-02-09*

#### [Benchmarking Agentic Workflow Generation](https://arxiv.org/abs/2410.07869)

*Shuofei Qiao, Runnan Fang, Zhisong Qiu, Xiaobin Wang, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen*

*ICLR 2025*

This paper introduces a unified workflow generation benchmark with multi-faceted scenarios and graph workflow structures. Also, it provides a systemic evaluation protocol utilizing subsequence and subgraph matching algorithms to quantify the LLM agent's workflow generation capabilities.


*2025-02-07*

#### [SaMer: A Scenario-aware Multi-dimensional Evaluator for Large Language Models](https://openreview.net/forum?id=aBnVU5DL3I)

*Kehua Feng, Keyan Ding, Jing Yu, Yiwen Qu, Zhiwen Chen, chengfei lv, Gang Yu, Qiang Zhang, Huajun Chen*

*ICLR 2025*

This paper proposes a multi-dimensional evaluator to evaluate the response quality of LLMs for open-ended questions, which mainly involves four steps: (1) identifying the appropriate evaluation dimensions, (2) scoring the response quality in those dimensions, (3) weighting the contribution of those dimensions, and (4) calculating an overall score through weighted summation. It consists of a text embedding model and three MLP-based prediction heads (i.e., dimension prediction, scoring, and weighting layers).


*2025-02-02*

#### [SampleLLM: Optimizing Tabular Data Synthesis in Recommendations](https://arxiv.org/abs/2501.16125)

*高璟桐，杜昭呈，李晓鹏，赵翔宇，王奕超，李向阳，郭慧丰，唐睿明*

*TheWebConf 2025 Industry Track*

Tabular data synthesis is to generate synthetic tabular data that closely resemble those in a given dataset. This paper proposes a two-stage framework to improve the quality of LLM-based tabular data synthesis for recommendations by ensuring better distribution alignment. The first stage employs Chain-of-Thought prompts with examples to generate data that closely aligns with the target dataset distribution, while the second stage uses a feature attribution-based importance sampling method to refine feature relationships within the synthetic data, reducing any distribution biases introduced by the LLM.


*2025-01-29*

#### [KaLM-Embedding: Superior Training Data Brings A Stronger Embedding Model](https://arxiv.org/abs/2501.01028)

*Xinshuo Hu, Zifei Shan, Xinping Zhao, Zetian Sun, Zhenyu Liu, Dongfang Li, Shaolin Ye, Xinyuan Wei, Qian Chen, Baotian Hu, Haofen Wang, Jun Yu, Min Zhang*

*Arxiv 2025*

This paper proposes a model trained with key techniques proven to enhance performance: (1) persona-based synthetic data to create diversified examples distilled from LLMs, (2) ranking consistency filtering to remove less informative samples, and (3) semi-homogeneous task batch sampling to improve training efficacy.


*2025-01-28*

#### [DOGE: Towards Versatile Visual Document Grounding and Referring](https://arxiv.org/abs/2411.17125)

*Yinan Zhou, Yuxin Chen, Haokun Lin, Shuyu Yang, Li Zhu, Zhongang Qi, Chen Ma, Ying Shan*

*Arxiv 2024*

This paper proposes an approach for producing high-quality fine-grained document data for visual document understanding, including multi-granular parsing data for enhancing fundamental text localization and recognition capabilities; and instruction-tuning data to activate MLLM's grounding and referring capabilities during dialogue and reasoning.


*2025-01-23*

#### [OneGen: Efficient One-Pass Unified Generation and Retrieval for LLMs](https://aclanthology.org/2024.findings-emnlp.237)

*张锦添, 彭成, 孙梦殊, 陈想, 梁磊, 张志强, 周俊, 陈华钧, 张宁豫*

*EMNLP 2024 Findings*

To better integrate retrieval and generation, this paper introduces an one-pass framework by incorporating a special token representing the retrieval request in the generation process, which enables a single LLM to handle both tasks simultaneously in a unified forward pass.


*2025-01-15*

#### [Chatlaw: A Multi-Agent Collaborative Legal Assistant with Knowledge Graph Enhanced Mixture-of-Experts Large Language Model](https://arxiv.org/abs/2306.16092)

*Jiaxi Cui, Munan Ning, Zongjian Li, Bohua Chen, Yang Yan, Hao Li, Bin Ling, Yonghong Tian, Li Yuan*

*Arxiv 2023*

This paper first collects a comprehensive and diverse legal dataset including multi-sourced data, by deduplication, denoising, and human finetuning. Then it proposes a multi-agent collaborative framework, which involves several roles, and each agent follows a ‘sense-think-action’ process: The Legal Assistant interacts with users to gather information and fill in knowledge graph nodes. The Legal Researcher analyzes and extracts legal entities, relationships, and substantial cases from the legal datasets. The Legal Editor assists users in consulting documents, selecting templates, and filling documents, while also ensuring a firewall strategy for data security. The Senior Lawyer conducts case studies, evaluates the relevance of items, and provides comprehensive results.


*2025-01-14*

#### [KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation](https://arxiv.org/abs/2409.13731)

*Lei Liang, Mengshu Sun, Zhengke Gui, Zhongshu Zhu, Zhouyu Jiang, Ling Zhong, Yuan Qu, Peilong Zhao, Zhongpu Bo, Jin Yang, Huaidong Xiong, Lin Yuan, Jun Xu, Zaoyang Wang, Zhiqiang Zhang, Wen Zhang, Huajun Chen, Wenguang Chen, Jun Zhou*

*Arxiv 2024*

This paper proposes a domain knowledge service framework that aims to utilize KGs and improve reasoning and generation performances of LLMs. Specifically, it focuses on 5 aspects, including (1) LLM-friendly knowledge representation, (2) mutual-indexing between knowledge graphs and original chunks, (3) logical-form-guided hybrid reasoning engine, (4) knowledge alignment with semantic reasoning, and (5) model capability enhancement for KAG.


*2025-01-13*

#### [GraphInsight: Unlocking Insights in Large Language Models for Graph Structure Understanding](https://arxiv.org/abs/2409.03258)

*Yukun Cao, Shuo Han, Zengyi Gao, Zezhong Ding, Xike Xie, S. Kevin Zhou*

*Arxiv 2024*

To address the positional bias of LLMs in handling long sequences of graph description, this paper proposes a framework that utilizes two key strategies: (1) placing critical graphical information in positions where LLMs exhibit stronger memory performance, and (2) investigating a lightweight external knowledge base for regions with weaker memory performance, inspired by retrieval-augmented generation.


*2025-01-12*

#### [Make Your LLM Fully Utilize the Context](https://openreview.net/forum?id=YGTVEmBXtV)

*Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, Weizhu Chen*

*NeurIPS 2024*

To overcome the lost-in-the-middle problem, this paper hypothesizes that it stems from insufficient explicit supervision during the long-context training, which fails to emphasize that any position in a long context can hold crucial information. Based on this intuition, it presents information-intensive training as a purely data-driven solution. Specifically, it leverages a synthesized long-context question-answer dataset, where the answer requires (1) fine-grained information awareness on a short segment (~128 tokens) within a synthesized long context (4K-32K tokens), and (2) the integration and reasoning of information from two or more short segments.


*2025-01-11*

#### [Lost in the Middle: How Language Models Use Long Contexts](https://aclanthology.org/2024.tacl-1.9/)

*Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang*

*Transactions of the Association for Computational Linguistics, Volume 12*

This paper analyzes the performance of LMs on multi-document question answering and key-value retrieval, and finds that performance can degrade significantly when changing the position of relevant information, indicating that current LMs do not robustly make use of information in long input contexts. Specifically, the performance is often highest when relevant information occurs at the beginning or end of the input context, while significantly lower when the relevant information in the middle of long contexts.


*2025-01-05*

#### [Thinking with Knowledge Graphs: Enhancing LLM Reasoning Through Structured Data](https://arxiv.org/abs/2412.10654)

*Xue Wu, Kostas Tsioutsiouliklis*

*Arxiv 2024*

To improve the reasoning ability of LLMs with the help of structured data, this paper introduces a programming language style representation of knowledge graphs, which aims to facilitate the seamless integration of structured knowledge into the language modeling process.


*2025-01-03*

#### [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200)

*Xuezhi Wang, Denny Zhou*

*NeurIPS 2024*

This paper investigates whether LLMs can reason effectively without prompting. Their findings reveal that, intriguingly, CoT reasoning paths can be elicited from pre-trained LLMs by altering the decoding process. Rather than conventional greedy decoding, it takes top-k alternative tokens, uncovering that CoT paths are frequently inherent in these sequences. This approach not only bypasses the confounders of prompting but also allows to assess the LLMs' intrinsic reasoning abilities. Besides, the presence of a CoT in the decoding path correlates with a higher confidence in the model's decoded answer. This confidence metric effectively differentiates between CoT and non-CoT paths.


*2025-01-02*

#### [Making LLaMA SEE and Draw with SEED Tokenizer](https://arxiv.org/abs/2310.01218)

*Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, Ying Shan*

*ICLR 2024*

This paper introduces an image tokenizer that empowers LLMs with the ability to see and draw at the same time. Two crucial design principles: (1) Image tokens should be independent of 2D physical patch positions and instead be produced with a 1D causal dependency, exhibiting intrinsic interdependence that aligns with the left-to-right autoregressive prediction mechanism in LLMs. (2) Image tokens should capture high-level semantics consistent with the degree of semantic abstraction in words, and be optimized for both discriminativeness and reconstruction during the tokenizer training phase.


*2024-12-31*

#### [Advancing Tool-Augmented Large Language Models: Integrating Insights from Errors in Inference Trees](https://arxiv.org/abs/2406.07115)

*Sijia Chen, Yibo Wang, Yi-Feng Wu, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, Lijun Zhang*

*NeurIPS 2024*

To improve tool-augmented LLMs, this paper utilizes the tree-of-thought for supervised fine-tuning. Instead of having all planned trajectories at once, it adopts an iterative reasoning approach and generates a step-wise preference dataset for tool use. It utilizes tool-usage expert trajectories as step-wise preference pairs for direct preference optimization to update the policy of the LLM.


*2024-12-28*

#### [Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?](https://arxiv.org/abs/2408.08685)

*Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi*

*KDD 2025*

This paper proposes an LLM-based robust graph structure inference framework, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges, and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure.


*2024-12-22*

#### [Human Bias in the Face of AI: The Role of Human Judgement in AI Generated Text Evaluation](https://arxiv.org/abs/2410.03723)

*Tiffany Zhu, Iain Weissburg, Kexun Zhang, William Yang Wang*

*Arxiv 2024*

This paper investigates the overwhelming issue when human readers treating AI-generated texts. Specifically, it evaluates how human raters respond to labeled and unlabeled content. While the raters could not differentiate the two types of texts in the blind test, they overwhelmingly favored content labeled as "Human Generated," over those labeled "AI Generated," by a preference score of over 30%, and the same pattern was observed even when the labels were deliberately swapped.


*2024-12-19*

#### [Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs](https://arxiv.org/abs/2406.14282)

*Junjie Wang, Mingyang Chen, Binbin Hu, Dan Yang, Ziqi Liu, Yue Shen, Peng Wei, Zhiqiang Zhang, Jinjie Gu, Jun Zhou, Jeff Z. Pan, Wen Zhang, Huajun Chen*

*EMNLP 2024*

To improve the performance of LLMs in complex question-answering (QA) scenarios, recent studies attempt to enhance LLMs' performance by combining step-wise planning with external retrieval. In contrast to previous works that rely on manual annotation and knowledge distillation from teacher LLMs, this paper introduces a framework to enhance LLMs' planning capability using planning data derived from knowledge graphs.


*2024-12-17*

#### [Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents](https://arxiv.org/abs/2411.06559)

*Yu Gu, Boyuan Zheng, Boyu Gou, Kai Zhang, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, Huan Sun, Yu Su*

*Arxiv 2024*

This paper proposes a LLM-based approach for agent planning over webpages, which uses examples to guide the model, i.e., planning through simulaiton.


*2024-12-15*

#### [Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning](https://aclanthology.org/2024.findings-emnlp.882.pdf)

*Debjit Paul, Robert West, Antoine Bosselut, Boi Faltings*

*EMNLP 2024 Findings*

This paper evaluates to what degree the LLM's final answer is faithful to the stated reasoning steps. Analyses on 12 LLMs show that they do not reliably use their intermediate reasoning steps when generating an answer. To address the issue, it proposes a framework to tailor small-sized LMs (<10B parameters) for generating correct reasoning steps and robustly reasoning over these steps. It consists of an inference module that learns to generate correct reasoning steps using an implicit causal reward function, and a reasoning module that learns to faithfully reason over these intermediate inferences using a counterfactual and causal preference objective.


*2024-12-14*

#### [Ada-Instruct: Adapting Instruction Generators for Complex Reasoning](https://aclanthology.org/2024.findings-emnlp.409.pdf)

*Wanyun Cui, Qianle Wang*

*EMNLP 2024 Findings*

This paper proposes a framework that first uses a few examples for fine-tuning an LM to specifically generate prompts, which help improve the following task performance.


*2024-12-13*

#### [Deciphering the Factors Influencing the Efficacy of Chain-of-Thought: Probability, Memorization, and Noisy Reasoning](https://aclanthology.org/2024.findings-emnlp.212.pdf)

*Akshara Prabhakar, Thomas L. Griffiths, R. Thomas McCoy*

*EMNLP 2024 Findings*

To evaluate whether LLMs exhibit abstract generalization or rely on shallow heuristics when given CoT prompts, this paper provides a detailed case study of the symbolic reasoning task of decoding shift ciphers. The results indicate 3 factors that affect CoT performance: the probability of the task’s expected output (probability), what the model has implicitly learned during pre-training (memorization), and the number of intermediate operations involved in reasoning (noisy reasoning).


*2024-12-12*

#### [Language Models as Compilers: Simulating Pseudocode Execution Improves Algorithmic Reasoning in Language Models](https://aclanthology.org/2024.emnlp-main.1253.pdf)

*Hyungjoo Chae, Yeonghyeon Kim, Seungone Kim, Kai Tzu-iunn Ong, Beong-woo Kwak, Moohyeon Kim, Sunghwan Kim, Taeyoon Kwon, Jiwan Chung, Youngjae Yu, Jinyoung Yeo*

*EMNLP 2024*

This paper proposes a think-and-execute framework, which consists of two LMs as an instructor and a reasoner, respectively. The instructor LM takes the input question and is prompted to generate a pseudo code-style description for the task. Then the reasoner LM follows the pseudo code with the input to simulate the execution process and reaches the final result.


*2024-12-10*

#### [Divide-or-Conquer? Which Part Should You Distill Your LLM?](https://aclanthology.org/2024.findings-emnlp.145.pdf)

*Zhuofeng Wu, Richard He Bai, Aonan Zhang, Jiatao Gu, V.G.Vinod Vydiswaran, Navdeep Jaitly, Yizhe Zhang*

*EMNLP 2024 Findings*

To improve the LLMs' capability for reasoning tasks, this work divides the process into a problem decomposition phase and a problem solving phase, rather than in a single stage. Besides, empirical results show that the decomposition ability is easier to be distilled into a smaller model, while the problem solving ability is more difficult to distill without performance degradation.


*2024-12-10*

#### [Divide-or-Conquer? Which Part Should You Distill Your LLM?](https://aclanthology.org/2024.findings-emnlp.145.pdf)

*Zhuofeng Wu, Richard He Bai, Aonan Zhang, Jiatao Gu, V.G.Vinod Vydiswaran, Navdeep Jaitly, Yizhe Zhang*

*EMNLP 2024 Findings*

To improve the LLMs' capability for reasoning tasks, this work divides the process into a problem decomposition phase and a problem solving phase, rather than in a single stage. Besides, empirical results show that the decomposition ability is easier to be distilled into a smaller model, while the problem solving ability is more difficult to distill without performance degradation.


*2024-12-09*

#### [Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs](https://aclanthology.org/2024.emnlp-main.629/)

*Haritz Puerto, Martin Tutek, Somak Aditya, Xiaodan Zhu, Iryna Gurevych*

*EMNLP 2024*

This paper investigates the hypothesis that representing natural language tasks as code can enhance specific reasoning abilities such as entity tracking or logical reasoning. To study this, it proposes code prompting, a chain of prompts that transforms a natural language problem into code and directly prompts the LLM using the generated code without resorting to external code execution. The result shows that code prompting exhibits a high-performance boost for multiple LLMs across multiple conditional reasoning datasets. Subsequent analysis on GPT 3.5 reveals that the code formatting of the input problem is essential for boosting the performance.


*2024-12-08*

#### [Few shot chain-of-thought driven reasoning to prompt LLMs for open-ended medical question answering](https://aclanthology.org/2024.findings-emnlp.31/)

*Saeel Sandeep Nachane, Ojas Gramopadhye, Prateek Chanda, Ganesh Ramakrishnan, Kshitij Sharad Jadhav, Yatin Nandwani, Dinesh Raghu, Sachindra Joshi*

*EMNLP 2024 Findings*

This paper proposes a modified version of the MedQA-USMLE dataset that changes the type of multi-choice questions into open-ended natural language questions. Besides, it constructs a prompt using chain-of-thought reasoning, which empirically improves the performance compared with existing 5-shot CoT-based prompts. Further, it presents a pipeline by first exploring the multiple possibility and subsequently narrowing down to the final diagnosis.


*2024-12-07*

#### [Puzzle Solving using Reasoning of Large Language Models: A Survey](https://aclanthology.org/2024.emnlp-main.646/)

*Panagiotis Giadikiaroglou, Maria Lymperaiou, Giorgos Filandrianos, Giorgos Stamou*

*EMNLP 2024*

This paper investigates the capabilities of LLMs in puzzle solving, by firstly categorizing the puzzles into rule-based and rule-less puzzles, highlighting the distinct knowledge demands to tackle each of them. Then it reviews existing approaches that LLMs use to solve puzzles, assesses their impacts on both categories of puzzles, and compares them with conventional problem-solving techniques. It also reviews existing benchmarks for evaluating LLMs' reasoning abilities.


*2024-12-06*

#### [Zero-Resource Hallucination Prevention for Large Language Models](https://aclanthology.org/2024.findings-emnlp.204/)

*Junyu Luo, Cao Xiao, Fenglong Ma*

*EMNLP 2024 Findings*

To prevent the hallucination of LLMs, existing approaches usually identify hallucinations post-generation that cannot prevent their occurrence, and suffer from inconsistent performance due to the instruction format and model style. In contrast, this paper introduces a pre-detection self-evaluation technique, which focuses on evaluating the model’s familiarity with the concepts present in the input instruction and withholding the generation of response in case of unfamiliar concepts under the zero-resource setting, where external ground-truth information is unavailable.


*2024-12-05*

#### [Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate](https://aclanthology.org/2024.emnlp-main.992/)

*Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, Zhaopeng Tu*

*EMNLP 2024*

To enhance the capability of LLM for complex reasoning tasks, one representative strategy is self-reflection, which asks an LLM to refine the solution with feedbacks generated by itself iteratively. However, such reflection-style methods suffer from the Degeneration-of-Thought (DoT) problem: once the LLM has established confidence in its solutions, it is unable to generate novel thoughts later through reflection even if its initial stance is incorrect. To address the DoT problem, this paper proposes a Multi-Agent Debate framework, in which multiple agents express their arguments in the state of “tit for tat” and a judge manages the debate process to obtain a final solution.


*2024-12-04*

#### [MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models](https://aclanthology.org/2024.acl-long.558/)

*Yilin Wen, Zifeng Wang, Jimeng Sun*

*ACL 2024*

This paper aims to build a plug-and-play prompting approach to elicit the graph-of-thoughts reasoning capability in LLMs. Specifically, it first uses an LLM to identify key entities from the question. Then it builds an evidence sub-graph by path-based and neighbor-based exploration from the keyword-related entities in a source KG, and let LLM conduct reasoning over the evidence graph.


*2024-11-29*

#### [Can Large Language Models Understand DL-Lite Ontologies? An Empirical Study](https://arxiv.org/abs/2406.17532)

*Keyu Wang, Guilin Qi, Jiaqi Li, Songlin Zhai*

*EMNLP 2024 Findings*

This paper investigates the ability of LLMs to understand DL-Lite ontologies. Specifically, for the syntactic aspect, it evaluates whether LLMs can comprehend structural rules, valid statements, and expressions of DL-Lite through syntax checking. For the semantic aspect, it tests whether LLMs can understand the semantics of concepts and roles from intension and extension, by subsumption of concepts or roles, and instance checking, respectively. Additionally, it probes property characteristics in DL-Lite ontologies, such as inverse roles and functional roles. Query answering and ontology satisfiability checking were conducted to evaluate whether LLMs can understand the semantics of the whole ontologies.


*2024-11-28*

#### [To Forget or Not? Towards Practical Knowledge Unlearning for Large Language Models](https://arxiv.org/abs/2407.01920)

*Bozhong Tian, Xiaozhuan Liang, Siyuan Cheng, Qingbin Liu, Mengru Wang, Dianbo Sui, Xi Chen, Huajun Chen, Ningyu Zhang*

*EMNLP 2024 Findings*

This paper introduces a benchmark containing copyrighted content and user privacy domains to evaluate if the unlearning process for LLMs inadvertently erases essential knowledge.The findings suggest that existing unlearning methods often suffer from excessive unlearning.


*2024-11-24*

#### [GPTKB: Building Very Large Knowledge Bases from Language Models](https://arxiv.org/abs/2411.04920)

*Yujia Hu, Shrestha Ghosh, Tuan-Phong Nugyen, Simon Razniewski*

*Arxiv 2024*

This paper proposes to build a large general-domain KB entirely from large language models, which demonstrates the feasibility of large-scale KB construction from LLMs, while highlighting specific challenges arising around entity recognition, entity and property canonicalization, and taxonomy construction.


*2024-11-22*

#### [MKGL: Mastery of a Three-Word Language](https://arxiv.org/abs/2410.07526)

*Lingbing Guo, Zhongpu Bo, Zhuo Chen, Yichi Zhang, Jiaoyan Chen, Yarong Lan, Mengshu Sun, Zhiqiang Zhang, Yangyifei Luo, Qian Li, Qiang Zhang, Wen Zhang, Huajun Chen*

*NeurIPS 2024*

This paper proposes a specialized KG language model. The instruction to the LLM includes a dictionary exemplifying the entity $e_i$ and relation $r_k$, and the task is to construct new KG sentences initialized with $e_ir_k$. It first tokenizes the input text, where the entities and relations are represented as special tokens out of the original vocabulary. To process the special tokens, it first collects embeddings of their constituting text tokens. Then a retriever performs a 4-step process to aggregate textual and relational information into KGL token embeddings, where the first and the last steps are LoRA-like down-scaling and up-scaling operations. The output is embeddings of the special KGL tokens. Similar to the context retriever, a score retriever obtains the score information and outputs a probability distribution among candidate entities.


*2024-11-19*

#### [From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning](https://arxiv.org/abs/2308.12032)

*Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, Jing Xiao*

*NAACL 2024*

This paper proposes a 3-stage process for fine-tuning LLMs, aiming to pick up and utilize the most difficult examples to make the model training more effective. Specifically, it begins by familiarizing the model with a small subset of the dataset, i.e., “Learning from Brief Experience”. Then it computes the Instruction-Following Difficulty (IFD) score for each training example, to evaluate how much help the instruction provides to the generation of the corresponding response. The final step is to re-train the model using examples with relatively large IFD scores as the *cherry data*.


*2024-11-15*

#### [Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334)

*Stephanie Lin, Jacob Hilton, Owain Evans*

*TMLR 2022*

This paper shows that a GPT-3 model can learn to express uncertainty about its own answers in natural language – without use of model logits. When given a question, the model generates both an answer and a level of confidence (e.g. “90% confidence” or “high confidence”), and these levels map to well calibrated probabilities.


*2024-11-14*

#### [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://arxiv.org/abs/2306.13063)

*Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, Bryan Hooi*

*ICLR 2024*

To better investigate the ability of LLM to estimate the uncertainty of generated answers, this paper proposes a framework for evaluating LLMs, which contains 3 components: *prompting strategies* for eliciting verbalized confidence, *sampling methods* for generating multiple responses, and *aggregation techniques* for computing consistency. The results suggest that (1) LLMs tend to be overconfident, (2) both calibration and failure prediction performance improve with the model capability, (3) some proposed strategies such as prompts are helpful for mitigating the overconfident issue, (4) while white-box methods perform better, the gap is narrow.


*2024-11-11*

#### [GAugLLM: Improving Graph Contrastive Learning for Text-Attributed Graphs with Large Language Models](https://dl.acm.org/doi/10.1145/3637528.3672035)

*Yi Fang, Dongzhe Fan, Daochen Zha, Qiaoyu Tan*

*KDD 2024*

To utilize text attributes for node representations, this paper proposes GAugLLM as a framework for augmenting text-attributed graphs. It applies LLMs like Mistral to enhance self-supervised graph learning. Specifically, it introduces a mixture-of-prompt-expert technique to generate augmented node features, It adaptively maps multiple prompt experts, each of which modifies raw text attributes using prompt engineering, into numerical feature space. Besides, based on both structural and textual commonalities, it effectively perturbs edges deemed most spurious or likely to be connected, thereby achieving structural (edge) augmentation.


*2024-11-10*

#### [Agent Planning with World Knowledge Model](https://arxiv.org/abs/2405.14205)

*Shuofei Qiao, Runnan Fang, Ningyu Zhang, Yuqi Zhu, Xiang Chen, Shumin Deng, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen*

*NeurIPS 2024*

This paper proposes a LLM-agent model that synthesizes knowledge from both expert and sampled trajectories, by providing prior task knowledge to guide the global planning and dynamic state knowledge to assist the local planning.


*2024-10-29*

#### [Tree-of-Traversals: A Zero-Shot Reasoning Algorithm for Augmenting Black-box Language Models with Knowledge Graphs](https://arxiv.org/abs/2407.21358)

*Elan Markowitz, Anil Ramakrishna, Jwala Dhamala, Ninareh Mehrabi, Charith Peris, Rahul Gupta, Kai-Wei Chang, Aram Galstyan*

*ACL 2024*

This paper applies the tree-of-thought approach for KBQA with the help of LLMs to select the appropriate entity/relaiton for expanding the subgraph. Specifically, the initial graph contains seed entities from the query, and each subsequent step consists of selecting a tree node, sampling thoughts or actions by the LLM, performing actions, and evaluating outputs with the LLM.


*2024-10-14*

#### [Linguini: A benchmark for language-agnostic linguistic reasoning](https://arxiv.org/abs/2409.12126)

*Eduardo Sánchez, Belen Alastruey, Christophe Ropers, Pontus Stenetorp, Mikel Artetxe, Marta R. Costa-jussà*

*Arxiv 2024*

This paper proposes a benchmark to measure a language model's linguistic reasoning skills without relying on pre-existing language-specific knowledge. The test covers 894 questions grouped in 160 problems across 75 (mostly) extremely low-resource languages, extracted from the International Linguistic Olympiad corpus. To attain high accuracy on this benchmark, models don't need previous knowledge of the tested language, as all the information needed to solve the linguistic puzzle is presented in the context. The results show that, while all analyzed models rank below 25% accuracy, there is a significant gap between open and closed models.


*2024-10-06*

#### [Application of Generative AI as an Enterprise Wikibase Knowledge Graph Q&A System](https://aclanthology.org/2024.kallm-1.4/)

*Renê Mendes, Dimas Oliveira, Victor Garcia*

*KaLLM@ACL 2024*

This paper proposes a RAG Q&A system that accesses data from an Enterprise Knowledge Graph using open-source Wikibase software, created to integrate company data into natural language conversations.


*2024-10-02*

#### [LLMR: Knowledge Distillation with a Large Language Model-Induced Reward](https://arxiv.org/abs/2409.12500)

*Dongheng Li, Yongchang Hao, Lili Mou*

*COLING 2024*

This paper proposes a knowledge distillation method based on a reward function induced from large language models. Specifically, it prompts an LLM and treats it as the teacher. It does not follow the common KD that minimizes the divergence between LLM’s probability $p_{\text{LLM}}$ and the student $q_\theta$. Instead, it induces a reward function from $p_{\text{LLM}}$ that evaluates the appropriateness of a word at every step given its previous context.


*2024-10-01*

#### [Evaluating the Factuality of Large Language Models using Large-Scale Knowledge Graphs](https://arxiv.org/pdf/2404.00942)

*Xiaoze Liu, Feijie Wu, Tianyang Xu, Zhuo Chen, Yichi Zhang, Xiaoqian Wang, Jing Gao*

*ACL 2024*

To evaluate the factuality of LLM-generated responses, this paper proposes GraphEval that contains a large test dataset and a light-weighted judge model. Specifically, the test dataset is retrieved from a knowledge graph with more than 10 million facts without human efforts. Unlike conventional methods that evaluate LLMs based on generated responses, GraphEval streamlines the evaluation process by creating a judge model to estimate the correctness of the answers given by the LLM.


*2024-09-27*

#### [KAM-CoT: Knowledge Augmented Multimodal Chain-of-Thoughts Reasoning](https://arxiv.org/abs/2401.12863)

*Debjyoti Mondal, Suraj Modi, Subhadarshi Panda, Rituraj Singh, Godawari Sudhakar Rao*

*AAAI 2024*

Given a (text) question $q$ along with k answer choices {a1, a2, . . . , ak}, the task is to pick the correct choice. The question $q$ is optionally accompanied by an image $X_{img}$ and a text $c$ that adds context to it. This paper proposes an approach named KAM-CoT, which consists of an LM that takes language context, a vision encoder to encode visual features and a graph neural network (GNN) that reasons over the KGs. Then the extracted information (as embeddings of text, image, kg, respectively) are fused using cross attention and then fed into the LM for generating the final result.


*2024-09-26*

#### [How Easily do Irrelevant Inputs Skew the Responses of Large Language Models?](https://arxiv.org/abs/2404.03302)

*Siye Wu, Jian Xie, Jiangjie Chen, Tinghui Zhu, Kai Zhang, Yanghua Xiao*

*COLM 2024*

With retrieval-augmented generation, Large Language Models exhibit enhanced capabilities for many knowledge-intensive tasks. However, due to the inherent flaws of current retrieval systems, there might exist irrelevant information within the retrieving top-ranked passages. This work presents an investigation into the robustness of LLMs to different types of irrelevant information under various conditions. It first introduces a framework to construct high-quality irrelevant information that ranges from semantically unrelated, partially related, and related to questions. Furthermore, it demonstrates that the constructed irrelevant information not only scores highly on similarity metrics, being highly retrieved by existing systems, but also bears semantic connections to the context.


*2024-09-25*

#### [OneEdit: A Neural-Symbolic Collaboratively Knowledge Editing System](https://arxiv.org/abs/2409.07497)

*Ningyu Zhang, Zekun Xi, Yujie Luo, Peng Wang, Bozhong Tian, Yunzhi Yao, Jintian Zhang, Shumin Deng, Mengshu Sun, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, Huajun Chen*

*LLM+KG@VLDB2024*

This paper proposes a neural-symbolic prototype system for collaborative knowledge editing using natural language, which consists of three modules: 1) The Interpreter serves for user interaction with natural language; 2) The Controller manages editing requests from various users, leveraging the KG with rollbacks to handle knowledge conflicts and prevent toxic knowledge attacks; 3) The Editor utilizes the knowledge from the Controller to edit KG and LLM.


*2024-09-15*

#### [Resolving Knowledge Conflicts in Large Language Models](https://arxiv.org/abs/2310.00935)

*Yike Wang, Shangbin Feng, Heng Wang, Weijia Shi, Vidhisha Balachandran, Tianxing He, Yulia Tsvetkov*

*COLM 2024*

For LLMs to solve knowledge conflicts, this paper claims that the LLM should (1) identify knowledge conflicts, (2) pinpoint conflicting information segments, and (3) provide distinct answers or viewpoints in conflicting scenarios. Then it introduces an evaluation framework for simulating contextual knowledge conflicts and quantitatively evaluating to what extent LLMs achieve these goals.


*2024-09-14*

#### [TaxoLLaMA: WordNet-based Model for Solving Multiple Lexical Semantic Tasks](https://arxiv.org/abs/2403.09207)

*Viktor Moskvoretskii, Ekaterina Neminova, Alina Lobanova, Alexander Panchenko, Irina Nikishina*

*ACL 2024*

This paper explores the capabilities of LLMs in capturing lexical-semantic knowledge from WordNet on the example of the LLaMA-2-7b model and tests it on multiple lexical semantic tasks including hypernym discovery, taxonomy enrichment, lexical entailment and taxonomy construction.  As the outcome, it presents TaxoLLaMA, an "all-in-one" model for taxonomy-related tasks, lightweight due to 4-bit quantization and LoRA.


*2024-09-11*

#### [Multi-aspect controllable text generation with disentangled counterfactual augmentation](https://arxiv.org/abs/2405.19958)

*Yi Liu, Xiangyu Liu, Xiangrong Zhu, Wei Hu*

*ACL 2024*

Multi-aspect controllable text generation aims to control the generated texts in attributes from multiple aspects. However, existing works usually neglect attribute correlations formed by the intertwining of different attributes, thus affecting multi-aspect control. To address the problem, this paper proposes a multi-aspect controllable text generation method with disentangled counterfactual augmentation, which alleviates the issue of imbalanced attribute correlations during training using counterfactual feature vectors in the attribute latent space by disentanglement. During inference, it also enhances attribute correlations by target-guided counterfactual augmentation to further improve multi-aspect control.


*2024-09-06*

#### [Can LLMs perform structured graph reasoning?](https://arxiv.org/abs/2402.01805)

*Palaash Agrawal, Shavak Vasania, Cheston Tan*

*ICPR 2024*

To test the ability of navigating through representations beyond plain text in various LLMs, it designs 10 distinct problems of graph traversal with increasing levels of complexity, and benchmarks 5 different instruct-finetuned LLMs. The result highlights various limitations, biases and properties of LLMs, such as an inverse relation to the average degrees of freedom of traversal per node in graphs, the overall negative impact of k-shot prompting on graph reasoning tasks, and a positive response bias which prevents LLMs from identifying the absence of a valid solution. Finally, it introduces a new prompting technique specially designed for graph traversal tasks.


*2024-09-05*

#### [AlignBench: Benchmarking Chinese Alignment of Large Language Models](https://arxiv.org/abs/2311.18743)

*Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue Huang, Zhuoer Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Lam Tam, Xiaohan Zhang, Lichao Sun, Xiaotao Gu, Hongning Wang, Jing Zhang, Minlie Huang, Yuxiao Dong, Jie Tang*

*ACL 2024*

This paper proposes a benchmark for Chinese LLM alignment. It designs a human-in-the-loop data curation pipeline, containing 8 main categories, 683 real-scenario rooted queries and corresponding human verified references. To ensure the correctness of references, each knowledge-intensive query is accompanied with evidences collected from reliable web sources (including URLs and quotations) by annotators. For automatic evaluation, the benchmark employs a rule-calibrated multi-dimensional LLM-as-Judge approach with Chain-of-Thought to generate explanations and final ratings.


*2024-08-30*

#### [Elephants Never Forget: Memorization and Learning of Tabular Data in Large Language Models](https://arxiv.org/abs/2404.06209)

*Sebastian Bordt, Harsha Nori, Vanessa Rodrigues, Besmira Nushi, Rich Caruana*

*COLM 2024*

This paper investigates the issue of data contamination for tabular data by introducing a variety of techniques to assess whether a language model has seen a tabular dataset during training. Then it compares few-shot learning performance of LLMs on datasets that were seen during training to the performance on datasets released after training. The result shows that LLMs perform better on datasets seen during training, indicating that memorization leads to overfitting. Meanwhile, LLMs show non-trivial performance on novel datasets and are surprisingly robust to data transformations. Overall, the results highlight the importance of testing whether an LLM has seen an evaluation dataset during pre-training.


*2024-08-25*

#### [Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation](https://arxiv.org/abs/2408.04187)

*Junde Wu, Jiayuan Zhu, Yunli Qi*

*Arxiv 2024*

This paper proposes a graph-based RAG framework in medical domain. It begins with a hybrid static-semantic document chunking approach. Then the extracted entities form a three-tier hierarchical graph structure, linking entities to foundational medical knowledge sourced from medical papers and dictionaries. These entities are connected to form meta-graphs, which are merged based on semantic similarities to build a comprehensive global graph. This structure supports information retrieval and response generation. Further, a U-retrieve method is employed to balance global awareness and indexing efficiency of the LLM.


*2024-08-24*

#### [GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework](https://arxiv.org/abs/2407.10793)

*Hannah Sansford, Nicholas Richardson, Hermina Petric Maretic, Juba Nait Saada*

*KDD 2024 Workshop*

This paper proposes a KG-based framework to detect the LLM hallucination in the output texts. First, the LLM output is fed into the KG construction prompt to produce the KG depicted on the right. Next, each individual triple in the KG is fed into an out-of-the-box hallucination detection method such as an NLI model, and compared to the provided context for inconsistency. Finally, any triples that are flagged as inconsistent are returned to the user, along with the overall hallucination decision.


*2024-08-23*

#### [Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph-based Retrofitting](https://arxiv.org/abs/2311.13314)

*Xinyan Guan, Yanjiang Liu, Hongyu Lin, Yaojie Lu, Ben He, Xianpei Han, Le Sun*

*AAAI 2024*

This paper proposes a KG-based retrofitting framework that automatically detects and updates the LLM responses to mitigate factual errors. Specifically, the framework consists of 5 steps, namely, claim extraction, entity detection and KG retrieval, fact selection, claim verification, response retrofitting. Each step involves usage of LLMs and prompt-tuning.


*2024-08-10*

#### [Knowledge Graph Tuning: Real-time Large Language Model Personalization based on Human Feedback](https://arxiv.org/abs/2405.19686)

*Jingwei Sun, Zhixu Du, Yiran Chen*

*Arxiv 2024*

This paper proposes a LLM-based model for the natural language QA task by editing a personalized (small) KG during human-LLM interactions, and generating answers based on the personalized KG. In each round of conversation, the LLM first extracts the posterior distribution of the personalized knowledge triples Q(z|q, a) from the human-LLM interaction. Then the personalized triples are used to optimize the KG to achieve two goals: The model can (1) retrieve the personalized triples with high probability and (2) generate the user’s feedback with the retrieved triples in high confidence.


*2024-08-06*

#### [Knowledgeable Preference Alignment for LLMs in Domain-specific Question Answering](https://arxiv.org/abs/2311.06503)

*Yichi Zhang, Zhuo Chen, Yin Fang, Yanxi Lu, Fangming Li, Wen Zhang, Huajun Chen*

*ACL 2024*

This paper introduces preference alignment for domain-specific QA with LLMs and domain KBs, which aims to ensure that (1) the response from LLM accommodates the user's question, and (2) it utilizes proper external KB as knowledge resource.


*2024-07-28*

#### [Masked Thought: Simply Masking Partial Reasoning Steps Can Improve Mathematical Reasoning Learning of Language Models](https://arxiv.org/abs/2403.02178)

*Changyu Chen, Xiting Wang, Ting-En Lin, Ang Lv, Yuchuan Wu, Xin Gao, Ji-Rong Wen, Rui Yan, Yongbin Li*

*ACL 2024*

This paper proposes a training approach to improve the LLMs reasoning ability, by randomly masking certain tokens within the chain of thought. When applied to fine-tuning with GSM8K on Llama-2-7B, it achieved a 5% improvement in GSM8K accuracy and a 10% improvement in GSM-IC accuracy over standard supervised fine-tuning with a few codes modified.


*2024-07-27*

#### [From Supervised to Generative: A Novel Paradigm for Tabular Deep Learning with Large Language Models](https://arxiv.org/abs/2310.07338)

*Xumeng Wen, Han Zhang, Shun Zheng, Wei Xu, Jiang Bian*

*KDD 2024*

This paper introduces Generative Tabular Learning (GTL), a framework for continued pretraining of an LLM on extensive tabular data, transcribed in an instruction-oriented language format and spanning multiple domains. It introduces the pipeline for constructing tabular data in an instruction-oriented language format across various domains, and the detailed optimization process in GTL.


*2024-07-26*

#### [SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks](https://arxiv.org/abs/2302.13939)

*Rui-Jie Zhu, Qihang Zhao, Guoqi Li, Jason K. Eshraghian*

*Transactions on Machine Learning Research*

Spiking Neural Networks (SNNs) emerge as an energy-efficient approach to deep learning that leverage sparse and event-driven activations to reduce the computational overhead associated with model inference. While they have become competitive with non-spiking models on many computer vision tasks, SNNs have proven to be more challenging to train. As a result, their performance lags behind modern deep learning, and until now, SNNs have yet to succeed at language generation on large-scale datasets. This paper, inspired by the Receptance Weighted Key Value (RWKV) language model, successfully implements ‘SpikeGPT’, a generative language model with binary, event-driven spiking activation units.


*2024-07-21*

#### [A Survey of Graph Meets Large Language Model: Progress and Future Directions](https://arxiv.org/abs/2311.12399)

*Yuhan Li, Zhixun Li, Peisong Wang, Jia Li, Xiangguo Sun, Hong Cheng, Jeffrey Xu Yu*

*IJCAI 2024 Survey*

This paper summarizes existing efforts on LLMs for graph-related tasks. It categorizes existing approaches into 3 kinds, namely, LLM-as-Enhancer generates extra text attributes or features to improve the graph embedding; LLM-as-Predictor applies LLMs more directly for the tasks such as classification, with flatten-based or GNN-based modules to extract structural features; GNN-LLM-Alighment focuses on the embedding spaces of GNNs and LLMs to integrate the graph modality with the text modality.


*2024-07-18*

#### [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://openreview.net/forum?id=WZH7099tgfM)

*Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, Ed H. Chi*

*ICLR 2023*

To improve the performance of chain-of-thought prompting on complex tasks, this paper proposes a framework that first decomposes the complex task into a series of simpler subproblems using a fixed prompting scheme. Then it sequentially solve each subproblem using minimal prompts (e.g., a single example) with answers from previous subproblems until the final answer.


*2024-07-17*

#### [XL3M: A Training-free Framework for LLM Length Extension Based on Segment-wise Inference](https://arxiv.org/abs/2405.17755)

*Shengnan Wang, Youhui Bai, Lin Zhang, Pingyi Zhou, Shixiong Zhao, Gong Zhang, Sen Wang, Renhai Chen, Hua Xu, Hongwei Sun*

*Arxiv 2024*

This paper proposes XL3M, which means extra-long large language model, to enable the LLMs trained on short sequences to reason extremely long sequence without any further training or fine-tuning. The input context is firstly decomposed into multiple short sub-contexts, where each sub-context contains an independent segment and a common “question” which is a few tokens from the end of the original context. Then it measures the relevance between each segment and the “question”, and constructs a concise key context by splicing all relevant segments in chronological order. The key context is further used instead of the original context to complete the inference task.


*2024-07-14*

#### [Enhanced Story Comprehension for Large Language Models through Dynamic Document-Based Knowledge Graphs](https://ojs.aaai.org/index.php/AAAI/article/view/21286)

*Berkeley R Andrus, Yeganeh Nasiri, Shilong Cui, Benjamin Cullen, Nancy Fulda*

*AAAI 2022*

This paper proposes a framework for long story comprehension and story-based question answering. Specifically, to mitigate the document length limitation that comes with finite context windows, it first constructs a KG using knowledge extraction tools from the source text, and then searches for the top ranked triples from the KG for augmenting the story comprehension task. Next, the extracted triples are transformed into prompts and fed to the LLM for generating the answers.


*2024-07-12*

#### [Detoxifying Large Language Models via Knowledge Editing](https://arxiv.org/abs/2403.14472)

*Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen*

*ACL 2024*

This paper proposes a benchmark of toxic question-answers including 9 categories, in which each category consists of 60 malicious questions, and each question is associated with safe and unsafe answers. Next, it proposes a toxic editing approach by tracing the difference of the model's hidden states with safe and unsafe answers as input. It identifies the next MLP layer of the maximum difference between hidden states as the "toxic region". Then it performs back-propagation using a single training example by freezing the rest of parameters other than the ones in the identified toxic region.


*2024-07-02*

#### [Efficient Tuning and Inference for Large Language Models on Textual Graphs](https://arxiv.org/abs/2401.15569)

*Yun Zhu, Yaoke Wang, Haizhou Shi, Siliang Tang*

*IJCAI 2024*

This paper proposes ENGINE, a parameter- and memory-efficient fine-tuning method for textual graphs with an LLM encoder. The key insight is to combine the LLMs and GNNs through a tunable side structure, which significantly reduces the training complexity without impairing the joint model’s capacity.


*2024-06-30*

#### [KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM Does and Doesn't Know](https://arxiv.org/abs/2312.11539)

*Shangshang Zheng, He Bai, Yizhe Zhang, Yi Su, Xiaochuan Niu, Navdeep Jaitly*

*Arxiv 2023*

This paper proposes a framework to evaluate the alignment between KGs and LLMs. Specifically, it includes a graph-guided question generator for converting KGs into natural language, along with a carefully designed sampling strategy based on parameterized KG structure to expedite KG traversal.


*2024-06-26*

#### [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)

*Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, Jeffrey Wu*

*Arxiv 2024*

This paper develops a state-of-the-art methodology to reliably train extremely wide and sparse autoencoders with very few dead latents on the activations of any language model. It studies the scaling laws with respect to sparsity, autoencoder size, and language model size by training a 16 million latent autoencoder on GPT-4 residual stream activations.


*2024-06-25*

#### [Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization](https://arxiv.org/abs/2405.15071)

*Boshi Wang, Xiang Yue, Yu Su, Huan Sun*

*Arxiv 2024*

This paper investigates the finding that transformers can learn to reason implicitly, but this skill is only robustly acquired through grokking, i.e., an extended period of training far beyond overfitting. Besides, the transformer fails to systematically generalize for composition, yet succeeds for comparison. This paper conducts a mechanistic study into the model internals throughout grokking, which reveals distinct generalizing circuits across the two tasks that explains the variation.


*2024-06-24*

#### [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)

*Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su*

*Arxiv 2024*

This paper presents a RAG framework, which first processes perceptual input with an LLM to transform a corpus into a schemaless knowledge graph as index. Given a new query, it identifies the key concepts in the query and runs the Personalized PageRank algorithm on the KG. Then it uses the query concepts as seeds to integrate information across passages for retrieval.


*2024-06-16*

#### [Dense Text Retrieval Based on Pretrained Language Models: A Survey](https://dl.acm.org/doi/10.1145/3637870)

*Wayne Xin Zhao, Jing Liu, Ruiyang Ren, Ji-Rong Wen*

*ACM Transactions on Information Systems*

Dense retrieval effectively learns the semantic representations of queries and texts in the latent representation space, and further constructs the semantic matching function between the dense vectors for relevance modeling. This survey paper organizes related works into 4 aspects, namely, architecture, training, indexing and integration. It also maintains a website for updating available papers and resources.


*2024-06-02*

#### [What does the Knowledge Neuron Thesis Have to do with Knowledge?](https://openreview.net/forum?id=2HJRwwbV3G)

*Jingcheng Niu, Andrew Liu, Zining Zhu, Gerald Penn*

*ICLR 2024*

This paper examines the “knowledge neuron” hypothesis of LLMs that factual knowledge can be localized to a small number of neurons, and that ablation of those neurons alters the probability of, and/or the final chosen output token. It further extends knowledge to include formal knowledge, and similarly finds small number of neurons that can be ablated to suppress their respective represented knowledge, particularly distributed throughout the later layers. However, through previous and additionally proposed metrics, in particular emphasizing bi-directionality and synonym-agnosticism, the authors argue that the discovered knowledge neurons cannot be considered to contain anything like “knowledge”, but simply conserve token correlations found in the training text.


*2024-05-09*

#### [Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples](https://papers.nips.cc/paper_files/paper/2023/hash/09425891e393e64b0535194a81ba15b7-Abstract-Conference.html)

*Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Mehran Kazemi, Najoung Kim, He He*

*NeurIPS 2023*

This paper investigates the deductive reasoning ability of LLMs over OOD CoT prompts, in terms of depth-, width-, and compositional generalization. The findings suggest that in-context learning is best applied to reasoning tasks by including examples that cover a diverse set of deduction rules, and keeping the examples simple. The in-context examples should especially contain examples of deduction rules that are less familiar to the model (i.e. proof by cases and proof by contradiction), and distractors should be provided for such examples as the model is more prone to overfitting.


*2024-05-05*

#### [Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering](https://arxiv.org/abs/2404.14741)

*Yao Xu, Shizhu He, Jiabei Chen, Zihao Wang, Yangqiu Song, Hanghang Tong, Kang Liu, Jun Zhao*

*Arxiv 2024*

This paper proposes to leverage LLMs for QA under Incomplete Knowledge Graph (IKGQA), where the given KG doesn't include all the factual triples involved in each question. To solve the task, it proposes a training-free method called Generate-on-Graph (GoG) that produces new triples while exploring on KGs. Specifically, a select-generate-answer framework works by not only using the LLM as an agent to explore the KG, but also as a KG to generate new facts based on the explored subgraph and its inherent knowledge.


*2024-05-01*

#### [QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering](https://arxiv.org/abs/2104.06378)

*Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, Jure Leskovec*

*NAACL 2021*

To leverage LM and KG for QA task with given question and selections, this paper proposes QA-GNN as an end-to-end framework with two insights: (i) Relevance scoring: Since the KG subgraph consists of all few-hop neighbors of the topic entities, some entity nodes are more relevant than others with respect to the given QA context. By proposing KG node relevance scoring, it scores each entity in the KG subgraph by concatenating the entity with the QA context and calculating the likelihood using a pretrained LM. This presents a general framework to weight information on the KG. (ii) Joint reasoning: It proposes a joint graph representation of the QA context and KG, where the QA context is viewed as an additional node (QA context node) and connected to the topic entities in the KG subgraph.


*2024-04-28*

#### [Augmenting Knowledge Graph Hierarchies Using Neural Transformers](https://arxiv.org/abs/2404.08020)

*Sanat Sharma, Mayank Poddar, Jayant Kumar, Kosta Blank, Tracy King*

*ECIR 2024*

This paper reports a LLM-based approach for generating KG hierarchies from a set of initial broad categories. It applies cyclical generation to obtain each level of categories in a loop, and uses one-shot generation to add each candidate into the hierarchy. The quality of generated hierarchy is manually evaluated by human experts.


*2024-04-15*

#### [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)

*Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang*

*Arxiv 2023*

This paper reviews existing works on Retrieval-Augmented Generation (RAG), which aims to retrieving pre-indexed information (e.g., text corpus, KG, etc.) in helping generation tasks such as QA. LLMs are used in both the indexing and the generation processes. This paper summarizes several RAG paradigms, as well as existing approaches used in each module of the RAG framework, such as retrieval sources, indexing optimization, query optimization, etc.


*2024-04-10*

#### [cTBLS: Augmenting Large Language Models with Conversational Tables](https://arxiv.org/abs/2303.12024)

*Anirudh S Sundar, Larry Heck*

*Arxiv 2023*

This paper proposes Conversational Tables (cTBLS), a three-step encoder-decoder architecture designed to augment LLMs with tabular data in conversational settings. Dense Table Retrieval identifies the table most relevant to the initial query. The retrieved table is provided to the state tracker for follow-up queries. State Tracking ranks cells in the table based on their ability to answer a follow-up query. Response Generation utilizes a LLM Decoder provided with the ranked cell information and the follow-up query to convert tabular data into a natural language response and continue the conversation.


*2024-04-09*

#### [Large Language Models are few(1)-shot Table Reasoners](https://arxiv.org/abs/2210.06710)

*Wenhu Chen*

*EACL 2023*

This work preliminarily evaluates the performance of LLMs on reasoning tasks over tabular data. Specifically, it tests several LLMs over tabular data benchmarks for QA that involves reasoning. The results show that, when combined with chain-of-thoughts prompting, LLMs can achieve very strong performance with only 1-shot demonstration. However, it also shows that the performance degrades vastly with the increase of table size, and smaller models do not perform well enough.


*2024-03-18*

#### [KnowLA: Enhancing Parameter-efficient Finetuning with Knowledgeable Adaptation](https://openreview.net/pdf?id=dmTNWtJ58V)

*Xindi Luo, Zequn Sun, Jing Zhao, Zhe Zhao, Wei Hu*

*NAACL 2024*

This paper proposes a knowledgeable adaptation method by inserting an adaptation layer into a LLM to integrate the embeddings of entities that appear in the input text. The adaptation layer is trained in combination with LoRA on instruction data.


*2024-03-15*

#### [At Which Training Stage Does Code Data Help LLMs Reasoning?](https://openreview.net/forum?id=KIPJKST4gw)

*YINGWEI MA, Yue Liu, Yue Yu, Yuanliang Zhang, Yu Jiang, Changjian Wang, Shanshan Li*

*ICLR 2024 Spotlight*

This paper investigates the impact of code data on LLMs at the pre-training stage and the fine-tuning stage. First, pre-training LLMs with the mixture of code and text can significantly enhance LLMs’ general reasoning capability almost without negative transfer on other tasks. Second, at the instruction-tuning stage, code data endows LLMs the task-specific reasoning capability. Moreover, the dynamic mixing strategy of code and text data assists LLMs to learn reasoning capability step-by-step during training.


*2024-03-09*

#### [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://openreview.net/forum?id=N8N0hgNDRt)

*Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, Weiyang Liu*

*ICLR 2024 Spotlight*

To improve the LLMs's ability on solving mathematical problems, this paper proposes a finetuned language model that specializes in mathematical reasoning. It firstly rewrites the original question from multiple perspectives, which results in a new dataset called MetaMathQA, and then finetunes the LLaMA-2 models on MetaMathQA.


*2024-03-07*

#### [Unbiased Watermark for Large Language Models](https://openreview.net/forum?id=uWVC5FVidc)

*Zhengmian Hu, Lichang Chen, Xidong Wu, Yihan Wu, Hongyang Zhang, Heng Huang*

*ICLR 2024 Spotlight*

As existing studies suggested a trade-off between watermark strength and output quality, this study examines how significantly watermarks impact the quality of model-generated outputs. Surprisingly, this work demonstrates that it is possible to integrate watermarks without affecting the output probability distribution with appropriate implementation, which is referred to in this paper as an unbiased watermark.


*2024-03-03*

#### [Linearity of Relation Decoding in Transformer Language Models](https://openreview.net/forum?id=w7LU2s14kE)

*Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, Yonatan Belinkov, David Bau*

*ICLR 2024 Spotlight*

Much of the knowledge encoded in transformer language models (LMs) may be expressed in terms of relations: relations between words and their synonyms, entities and their attributes, etc. This paper shows that, for a subset of relations, this computation is well-approximated by a single linear transformation on the subject representation. Linear relation representations can be obtained by constructing a first-order approximation to the LM from a single prompt, and they exist for a variety of factual, commonsense, and linguistic relations. Meanwhile, there are also many cases in which LM predictions capture relational knowledge accurately, but this knowledge is not linearly encoded in their representations.


*2024-02-29*

#### [In-Context Pretraining: Language Modeling Beyond Document Boundaries](https://openreview.net/forum?id=LXVswInHOo)

*Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Xi Victoria Lin, Noah A. Smith, Luke Zettlemoyer, Wen-tau Yih, Mike Lewis*

*ICLR 2024 Spotlight*

This paper proposes in-context pretraining, an approach where language models are trained on a sequence of related documents, thereby encouraging them to read and reason across document boundaries. Different from standard pretraining strategy that places randomly shuffled documents in the input context, in-context pretraining places related documents in the same context.


*2024-02-26*

#### [MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning](https://openreview.net/forum?id=jenyYQzue1)

*Zayne Rea Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, Greg Durrett*

*ICLR 2024 Spotlight*

This paper introduces a reasoning benchmark, MuSR, consisting of 756 examples across three domains that challenge state-of-the-art LLMs such as GPT-4, Llama 2, and Vicuna. The benchmark is built with an algorithm for generating natural language narratives grounded in reasoning trees, by firstly generating gold facts that are used to deduce the correct answer, and then using an LLM to create a reasoning tree leading to those deductions from facts in a story combined with commonsense. After that, a narrative one chunk is iteratively generated at a time using the facts.


*2024-02-24*

#### [Overthinking the Truth: Understanding how Language Models Process False Demonstrations](https://openreview.net/forum?id=Tigr1kMDZy)

*Danny Halawi, Jean-Stanislas Denain, Jacob Steinhardt*

*ICLR 2024 Spotlight*

This paper studies harmful imitation through the lens of a LM’s internal representations, and identifies two related phenomena: overthinking and false induction heads. The overthinking phenomenon appears when decoding predictions from intermediate layers, given correct vs. incorrect few-shot demonstrations. At early layers, both demonstrations induce similar model behavior, but the behavior diverges sharply at some “critical layer”, after which the accuracy given incorrect demonstrations progressively decreases. The false induction heads are a possible mechanistic cause of overthinking: these are heads in late layers that attend to and copy false information from previous demonstrations, whose ablation reduces overthinking.


*2024-02-22*

#### [LEGO-Prover: Neural Theorem Proving with Growing Libraries](https://openreview.net/forum?id=3f5PALef5B)

*Anonymous so far*

*ICLR 2024 Oral*

To improve the performance of LLM-based theorem prover, this paper presents LEGO-Prover, which employs a growing library containing verified lemmas as skills to augment the capability of LLMs used in theorem proving. By constructing the proof modularly, LEGO-Prover enables LLMs to utilize existing skills retrieved from the library and to create new skills during the proving process.


*2024-02-19*

#### [Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with Hypothesis Refinement](https://openreview.net/forum?id=bNt7oajl2a)

*Linlu Qiu, Liwei Jiang, Ximing Lu, Melanie Sclar, Valentina Pyatkin, Chandra Bhagavatula, Bailin Wang, Yoon Kim, Yejin Choi, Nouha Dziri, Xiang Ren*

*ICLR 2024 Oral*

This work investigates LMs’ inductive reasoning capabilities through the lens of iterative hypothesis refinement: hypotheses generation, selection, and refinement. The results show that LMs are particularly good at generating candidate rules, and when coupled with a symbolic interpreter that can provide accurate feedback with which to refine hypotheses, this hybrid induction approach is effective. However, a closer inspection shows that LMs are often unable to correctly applying their own proposed rules.


*2024-02-15*

#### [The mechanistic basis of data dependence and abrupt learning in an in-context classification task](https://openreview.net/forum?id=aN4Jf6Cx69)

*Gautam Reddy*

*ICLR 2024 Oral*

This paper characterizes the loss landscape of an in-context classification task and identify the factors that lead to abrupt transitions during learning. By identifying progress measures and designing experiments, it shows that ICL is driven by the abrupt formation of an induction head. It constructs a minimal two-parameter model of an induction head stacked with a deep classifier, which reproduces all data distributional dependencies and captures the dynamics of learning. Then it develops a phenomenological model of an induction head’s loss landscape. The analysis enables to trace the abrupt learning phenomenon to cliffs in the landscape created by nested nonlinearities in a multilayer attention-based network.


*2024-02-14*

#### [Controlled Text Generation via Language Model Arithmetic](https://openreview.net/forum?id=SLw9fp4yI6)

*Jasper Dekoninck, Marc Fischer, Luca Beurer-Kellner, Martin Vechev*

*ICLR 2024 Spotlight*

This paper proposes an inference framework named model arithmetic, for composing and biasing LLMs without the need for model (re)training or highly specific datasets. Specifically, it relies on the KL-divergence to combine multiple models with biased topics, which is applied for controlled text generation tasks.


*2024-02-10*

#### [Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions](https://openreview.net/forum?id=ekeyCgeRfC)

*Anonymous Authors*

*ICLR 2024 Oral*

To investigate the ability of Transformers to implement learning algorithms and learn other forms (apart from gradient-based ones) of algorithms, this paper firstly finds that Transformers can nearly match the optimal learning algorithm for 'simpler' tasks, while their performance deteriorates on more 'complex' tasks. Besides, when provided a *teaching sequence*, i.e. a set of examples that uniquely identifies a function in a class, the Transformers learn more sample-efficiently.


*2024-02-09*

#### [BooookScore: A systematic exploration of book-length summarization in the era of LLMs](https://openreview.net/forum?id=7Ttk3RzDeu)

*Anonymous Authors*

*ICLR 2024 Oral*

This paper evaluates the ability of LLMs to generate summaries for book-length texts via hierarchical and incremental workflows. With the help of human annotation, it first proposes a protocol for evaluating the coherence of summaries. Then it implements an automatic LLM-based metric to assess the summary coherence, and presents a systematic evaluation of existing SOTA LLMs on generating summaries from book-length texts -- which suggests potential improvements.


*2024-02-08*

#### [Proving Test Set Contamination for Black-Box Language Models](https://openreview.net/attachment?id=KS8mIvetg2&name=pdf)

*Anonymous Authors*

*ICLR 2024 Oral*

This paper proposes an approach for speculating potential test set contamination of LLMs without having access to the training corpus or the model weights. Specifically, it relies on a key insight that if a language model shows a preference for any particular ordering of the dataset – a canonical ordering that appears in publicly available repositories – this violates exchangeability and can only occur by observing the dataset during training.


*2024-01-20*

#### [Let the LLMs Talk: Simulating Human-to-Human Conversational QA via Zero-Shot LLM-to-LLM Interactions](https://arxiv.org/abs/2312.02913)

*Zahra Abbasiantaeb, Yifei Yuan, Evangelos Kanoulas, Mohammad Aliannejadi*

*WSDM 2024*

To simulate the human-to-human conversational question-answering process, this paper proposes to use two LLMs (specifically, two GPT-4 models) for the role of 'teacher' and 'student' interacting on a specific topic, where the 'student' LLM generates questions to explore the topic, and the 'teacher' LLM answers the questions. Both LLMs are used in a zero-shot manner. The evaluation over both models shows that the 'teacher' LLM generates lengthier answers that tend to be more accurate and complete, while the 'student' LLM generates more diverse questions, covering more aspects of a given topic.


*2024-01-16*

#### [GPT4Table: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study](https://arxiv.org/abs/2305.13062)

*Yuan Sui, Mengyu Zhou, Mingjie Zhou, Shi Han, Dongmei Zhang*

*WSDM 2024*

This paper proposes a benchmark for evaluating the structural understanding capabilities (SUC) of LLMs, which includes 7 tasks such as table partition, cell lookup and column retrieval. Evaluation on GPT-3.5 and GPT-4 demonstrate that the performance varied depending on several input choices, including table input format, content order, role prompting, and partition marks. It further proposes a self-augmentation structural prompting method based on the findings.


*2024-01-13*

#### [When Do Program-of-Thoughts Work for Reasoning?](https://arxiv.org/abs/2308.15452)

*Zhen Bi, Ningyu Zhang, Yinuo Jiang, Shumin Deng, Guozhou Zheng, Huajun Chen*

*AAAI 2024*

This paper explores the correlation between program-of-thoughts and LLM's ability for reasoning, by proposing a measurement of program-of-thoughts' complexity. Specifically, it considers the abstract syntax tree to encode the structural information and calculates logical complexity by considering the difficulty and the cyclomatic complexity. Experiments demonstrate the positive correlation between the proposed complexity measure and LLM's reasoning ability.


*2024-01-05*

#### [Thought Propagation: An Analogical Approach to Complex Reasoning with Large Language Models](https://arxiv.org/abs/2310.03965)

*Junchi Yu, Ran He, Rex Ying*

*Arxiv 2023*

To overcome the problem of existing LLM reasoning that prompting approaches cannot reuse insights of solving similar problems and suffer from accumulated errors in multi-step reasoning from scratch, this paper proposes Thought Propagation (TP), which explores the analogous problems and leverages their solutions to enhance the complex reasoning ability of LLMs. Specifically, it first prompts LLMs to propose and solve a set of analogous problems that are related to the input one. Then, it reuses the results of analogous problems to directly yield a new solution or derive a knowledge-intensive plan for execution to amend the initial solution obtained from scratch.


*2023-12-25*

#### [Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning](https://arxiv.org/abs/2305.12295)

*Liangming Pan, Alon Albalak, Xinyi Wang, William Yang Wang*

*EMNLP 2023 Findings*

To handle the logical reasoning task, this paper utilizes LLMs to translate the natural language problem into a symbolic formulation. Afterward, a deterministic symbolic solver performs inference on the formulated problem. Besides, it introduces a self-refinement module which utilizes the symbolic solver’s error messages to revise symbolic formalizations. The system's effectiveness is demonstrated on five logical reasoning datasets: ProofWriter, PrOntoQA, FOLIO, LogicalDeduction, and AR-LSAT, which inclues deductive reasoning problems, FOL reasoning problems, constraint satisfaction problems (CSP), and analytical reasoning (AR) problems.


*2023-12-21*

#### [Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning](https://aclanthology.org/2023.emnlp-main.609/)

*Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, Xu Sun*

*EMNLP 2023 Best Paper*

This paper investigates the working mechanism of ICL through an information flow lens. It shows that label words in the demonstration examples function as anchors: (1) semantic information aggregates into label word representations during the shallow computation layers’ processing; (2) the consolidated information in label words serves as a reference for LLMs’ final predictions. Based on the findings, it also proposes an anchor re-weighting method to improve ICL performance, a demonstration compression technique to expedite inference, and an analysis framework for diagnosing ICL errors in GPT2-XL.


*2023-12-12*

#### [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)

*Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen*

*Arxiv 2023*

This is so far the most comprehensive survey of LLMs including their theoretical foundation and available tools. (Still in updating. Chinese version and github resources link are available.)


*2023-12-11*

#### [StructGPT: A General Framework for Large Language Model to Reason over Structured Data](https://arxiv.org/abs/2305.09645)

*Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, Ji-Rong Wen*

*EMNLP 2023*

This paper proposes an Iterative Reading-then-Reasoning (IRR) framework to solve question answering tasks based on structured data, called StructGPT. The framework constructs the specialized interfaces to collect relevant evidence from structured data (i.e., reading), and let LLMs concentrate on the reasoning task based on the collected information (i.e., reasoning). Specially, it proposes an invoking-linearization-generation procedure to support LLMs in reasoning on the structured data with the help of the interfaces.


*2023-12-08*

#### [UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation](https://arxiv.org/abs/2311.15296)

*Xun Liang, Shichao Song, Simin Niu, Zhiyu Li, Feiyu Xiong, Bo Tang, Zhaohui Wy, Dawei He, Peng Cheng, Zhonghao Wang, Haiying Deng*

*Arxiv 2023*

This paper introduces a benchmark dataset using unconstrained hallucination generation, comprising a dataset curated for hallucinated news continuation, which encompasses over 5,000 instances annotated at the keyword level. Besides, it also proposes an evaluation framework to facilitate comprehensive assessments.


*2023-12-02*

#### [Leveraging Large Language Models to Generate Answer Set Programs](https://proceedings.kr.org/2023/37/)

*Adam Ishay, Zhun Yang, Joohyung Lee*

*KR 2023*

This paper evaluates the ability of LLMs to transform natural language descriptions of logic puzzles into answer set programs by designing prompts to convert descriptions into answer set programs in a step by step manner and in-context learning.


*2023-11-28*

#### [Making Large Language Models Perform Better in Knowledge Graph Completion](https://arxiv.org/abs/2310.06671)

*Yichi Zhang, Zhuo Chen, Wen Zhang, Huajun Chen*

*Arxiv 2023*

This paper proposes a new LLM-based method for knowledge graph completion. To feed the structural information of the KG into the LLM, it propose a knowledge prefix adapter (KoPA), which employs structural embedding pre-training to capture the structural information, and project it into the textual token space as input prefix for the LLM.


*2023-11-26*

#### [Don't Generate, Discriminate: A Proposal for Grounding Language Models to Real-World Environments](https://aclanthology.org/2023.acl-long.270/)

*Yu Gu, Xiang Deng, Yu Su*

*ACL 2023*

This paper proposes a generic framework for grounded language understanding that capitalizes on the discriminative ability of LMs instead of their generative ability. It consists of a symbolic agent and a neural LM where the agent explores the environment to incrementally construct valid plans, and the LM evaluates the plausibility of the candidate plans to guide the search process. It is evaluated on the KBQA task and achieved relatively good results.


*2023-11-25*

#### [Evaluating the Logical Reasoning Ability of ChatGPT and GPT-4](https://arxiv.org/abs/2304.03439)

*Hanmeng Liu, Ruoxi Ning, Zhiyang Teng, Jian Liu, Qiji Zhou, Yue Zhang*

*Arxiv 2023*

This report evaluates the performance of ChatGPT and GPT-4 on multi-choice reading comprehension and natural language inference tasks (in prompt-style) with benchmarks requiring logical reasoning. The results show that both ChatGPT and GPT-4 perform quite well on most existing benchmark datasets, and GPT-4 is even better than ChatGPT. However, their performance drops significantly when handling newly released and out-of-distribution datasets, which suggests the remaining challenge for LLMs on these datasets.


*2023-11-24*

#### [A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity](https://arxiv.org/abs/2302.04023)

*Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, Pascale Fung*

*Arxiv 2023*

As title, an extensive technical evaluation of ChatGPT using 23 datasets covering 8 different common NLP application tasks.


*2023-11-20*

#### [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)

*Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu*

*Arxiv 2023*

To automatically evaluate the text generation performance of LLMs, this paper present G-EVAL, a framework of using LLMs with chain-of-thoughts (CoT) and a form-filling paradigm, to assess the quality of NLG outputs. It evaluates with two generation tasks, text summarization and dialogue generation.


*2023-11-18*

#### [Hallucination Detection: Robustly Discerning Reliable Answers in Large Language Models](https://dl.acm.org/doi/10.1145/3583780.3614905)

*Yuyan Chen, Qiang Fu, Yichen Yuan, Zhihao Wen, Ge Fan, Dayiheng Liu, Dongmei Zhang, Zhixu Li, Yanghua Xiao*

*CIKM 2023*

This paper proposes a PLM-based model for hallucination detection for LLMs. Specifically, it first constructs a dataset including QA pairs, and evaluates the generated answers of LLMs as positive/negative hallucination examples. Then it formulates the discriminator as a classification model based on multiple metrics such as LLM self-evaluation metrics, human metrics, machine metrics and composite metrics. It conducts evaluation on several PLMs as backbone.


*2023-10-23*

#### [Prompting Large Language Model for Machine Translation: A Case Study](https://proceedings.mlr.press/v202/zhang23m.html)

*Biao Zhang, Barry Haddow, Alexandra Birch*

*ICML 2023*

This paper investigates the prompting techniques for machine translation by LLM. Specifically, it evaluates different kinds of prompts including monolingual data, cross-lingual, cross-domain and sentence-to-document transfer learning for prompting. The results shows, (1) the number and the quality of prompt examples matter, where using suboptimal examples degenerates translation; (2) several features of prompt examples, such as semantic similarity, show significant Spearman correlation with their prompting performance; yet, none of the correlations are strong enough; (3) using pseudo parallel prompt examples constructed from monolingual data via zero-shot prompting could improve translation; and (4) improved performance is achievable by transferring knowledge from prompt examples selected in other settings.


*2023-10-22*

#### [Bag of Tricks for Training Data Extraction from Language Models](https://proceedings.mlr.press/v202/yu23c.html)

*Weichen Yu, Tianyu Pang, Qian Liu, Chao Du, Bingyi Kang, Yan Huang, Min Lin, Shuicheng Yan*

*ICML 2023*

This paper investigates the effectiveness of tricks of training data extraction for language models. Specifically, it measures the precision, recall and Hamming distance increment brought by different tricks including sampling, probability distribution adjustment, exposure bias reduction, look ahead, sentence-level criteria and token-level criteria. The empirical results show that several of these tricks improve the results significantly, while interactions between different tricks are more subtle than expected. Besides, the commonly used versatile methods for general text generation are not always effective for extraction tasks.


*2023-10-17*

#### [Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models](https://proceedings.mlr.press/v202/shao23a.html)

*Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, Weizhu Chen*

*ICML 2023*

This paper proposes a prompt generation method, which is based on a few handcrafted examples. It generates more examples and selects effective demonstrations to improve reasoning. Specifically, it applies an alternating forward-backward process, in which the backward process involves generating a question for the given reasoning chain, and the forward process produces a better reasoning chain for the generated question.


*2023-10-15*

#### [Can Large Language Models Reason about Program Invariants?](https://proceedings.mlr.press/v202/pei23a.html)

*Kexin Pei, David Bieber, Kensen Shi, Charles Sutton, Pengcheng Yin*

*ICML 2023*

This paper investigates the application of using LLMs to identify program invariants. Typical invariant prediction approaches usually rely on dynamic analysis which requires multiple traces collected from executing the codes. However, this paper shows that LMs trained on source codes and fine-tuned for invariant generation are able to perform static invariant prediction with a scratch-pad approach.


*2023-09-25*

#### [The Unreasonable Effectiveness of Few-shot Learning for Machine Translation](https://proceedings.mlr.press/v202/garcia23a.html)

*Xavier Garcia, Yamini Bansal, Colin Cherry, George F. Foster, Maxim Krikun, Melvin Johnson, Orhan Firat*

*ICML 2023*

This paper demonstrates the few-shot training with high-quality data can produce translation models with relatively high performance. It highlights the quality of few-shot demonstrations can heavily determine the quality of translations generated by the model. Besides, the few-shot paradigm also provides a way to control certain variabilities of machine translation such as regional varieties.


*2023-09-23*

#### [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://proceedings.mlr.press/v202/frantar23a.html)

*Elias Frantar, Dan Alistarh*

*ICML 2023*

This paper proposes to prune the GPT model family to at least 50% sparsity of parameters in one-shot without retraining. It is based on a set of techniques including post-training pruning, layer-wise pruning, mask selection and weight reconstruction. It further proposes an algorithm for the pruning process.


*2023-09-14*

#### [A Watermark for Large Language Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html)

*John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein*

*ICML 2023*

This paper proposes a watermarking method for LLMs based on the selection of synonyms when there are multiple choices for generating texts. A watermark is a hidden pattern in text that is imperceptible to humans, while making the text algorithmically identifiable as synthetic. The proposed watermarking method uses a list of "favored" words in text generation, which is invisible to humans while can be captured by computing entropy and probability distribution among words.


*2023-09-08*

#### [Large Language Models Struggle to Learn Long-Tail Knowledge](https://proceedings.mlr.press/v202/kandpal23a.html)

*Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, Colin Raffel*

*ICML 2023*

This paper explores the relationship between the knowledge learned by an LLM and the information in its pretraining data. Specifically, it studies how an LLM’s ability to answer a question relates to how many documents associated with that question were seen during pretraining. The results indicate a strong correlational and causal relationships between accuracy and relevant document count for numerous question answering datasets (e.g., TriviaQA), pretraining corpora (e.g., ROOTS), and model sizes (e.g., 176B parameters). It also concludes that model scaling and retrieval-augmentation help better capture the knowledge that rarely appears in the pretraining data.


*2023-09-05*

#### [Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172)

*Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang*

*Arxiv 2023*

This paper summarizes existing methods for editing LLMs, including parameter-preserving and -modifying methods, compares their performance, and proposes a new dataset for evaluating them, especially in terms of generalization and efficiency.


*2023-09-04*

#### [Automatically Auditing Large Language Models via Discrete Optimization](https://proceedings.mlr.press/v202/jones23a.html)

*Erik Jones, Anca D. Dragan, Aditi Raghunathan, Jacob Steinhardt*

*ICML 2023*

This paper formulates auditing LLMs as a discrete optimizing problem, and proposes an algorithm for automatically searching for input-output pairs that match a target behavior (e.g., toxic output). To improve the search efficiency, it further decomposes the optimizing target as a linear approximation and an autoregressive part.


*2023-09-02*

#### [Exploring the Benefits of Training Expert Language Models over Instruction Tuning](https://proceedings.mlr.press/v202/jang23a.html)

*Joel Jang, Seungone Kim, Seonghyeon Ye, Doyoung Kim, Lajanugen Logeswaran, Moontae Lee, Kyungjae Lee, Minjoon Seo*

*ICML 2023*

This paper argues that instead of using large amount of multi-task data for prompt-tuning a general LM, fine-tuning an "expert" LM on a single task provides better performance. Following this, it also proposes separately training LMs on a single task brings benefits include (1) avoiding negative task transfer, (2) being able to continually learn new tasks without having to re-train on previous tasks to avoid catastrophic forgetting, and (3) showing compositional capabilities when merging individual experts together.


*2023-08-24*

#### [Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought](https://openreview.net/pdf?id=qFVVBzXxR2V)

*Abulhair Saparov, He He*

*ICLR 2023*

This paper proposes a synthetic question-answering dataset for evaluating the reasoning capabilities of LLMs. To generate each example, it firstly samples an ontology as a tree, and secondly produces a symbolic proof from the ontology. Thirdly, it converts the ontology into a natural language context, and finally, the proof is converted into a natural language question, chain-of-thought, and an answer label (True or False).


*2023-08-21*

#### [Neuro-Symbolic Procedural Planning with Commonsense Prompting](https://openreview.net/pdf?id=iOc57X9KM54)

*Yujie Lu, Weixi Feng, Wanrong Zhu, Wenda Xu, Xin Eric Wang, Miguel P. Eckstein, William Yang Wang*

*ICLR 2023*

Procedural planning aims to implement complex high-level goals by decomposition into simpler low-level steps. This paper proposes to leverage an external knowledge graph to automatically produce causal prompts for LLMs to conduct procedural planning.


*2023-08-20*

#### [Automatically Correcting Large Language Models: Surveying the landscape of diverse self-correction strategies](https://arxiv.org/abs/2308.03188)

*Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, William Yang Wang*

*Arxiv 2023*

Self-correction for LLM is to prompt the model to fix problems in its own output. This survey paper categorizes existing methods for self-correction into three kinds, namely, training-time, generation-time and post-hoc correction, as well as their applications.


*2023-08-13*

#### [Ask Me Anything: A simple strategy for prompting language models](https://openreview.net/pdf?id=bhUPJnS2g0X)

*Simran Arora, Avanika Narayan, Mayee F. Chen, Laurel J. Orr, Neel Guha, Kush Bhatia, Ines Chami, Christopher Ré*

*ICLR 2023*

Observed that question-answering (QA) prompts, which encourage open-ended generation (“Who went to the park?”) tend to outperform those that restrict the model outputs (“John went to the park. True or False?”), this paper proposes a simple prompting method named Ask Me Anything (AMA). It recursively uses the LLM to transform task inputs to the effective QA format, generates multiple questions per input, and applies these prompts to collect several noisy votes for the input’s true label.


*2023-08-10*

#### [Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs](https://openreview.net/pdf?id=SMa9EAovKMC)

*Albert Qiaochu Jiang, Sean Welleck, Jin Peng Zhou, Timothée Lacroix, Jiacheng Liu, Wenda Li, Mateja Jamnik, Guillaume Lample, Yuhuai Wu*

*ICLR 2023*

This paper investigates the task of automatic formal proof generation based on informal sketches. The informal drafts are firstly produced either by ordinary human user or LLMs. Then a mapping model is prompted to match the parts of informal sketches to a high-level structure of the formal proof. Finally, an off-the-shelf automated prover is executed to fill-in the gaps of the formal proof.


*2023-08-09*

#### [Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning](https://openreview.net/pdf?id=3Pf3Wg6o-A4)

*Antonia Creswell, Murray Shanahan, Irina Higgins*

*ICLR 2023*

This paper investigates the ability of LLMs to perform multi-step logical reasoning tasks. Motivated by the fact that LLMs usually perform well on single step inference and entailments, this paper further proposes a selection-inference module to guide the model with an interpretable, causal reasoning chain leading to the final answer. In the experiment, it uses pre-trained, frozen language models in a 5-shot generalisation setting with prompt engineering to implement the selection and inference modules.


*2023-07-02*

#### [Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning](https://arxiv.org/abs/2301.13808)

*Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, Yongbin Li*

*SIGIR 2023*

This paper proposes to use LLM for decomposing both large tables and complex questions for table-based reasoning for QA. On the one hand, it applies LLM to decompose large tables into small sub-tables while retaining the useful evidences and giving up the redundancy for a given question. On the other hand, it also  uses a LLM to decompose complex questions into chain of steps (as sub-questions), which is more convenient for SQL query generation.


*2023-06-29*

#### [Can ChatGPT Write a Good Boolean Query for Systematic Review Literature Search?](https://arxiv.org/abs/2302.03495)

*Shuai Wang, Harrisen Scells, Bevan Koopman, Guido Zuccon*

*SIGIR 2023*

Systematic reviews are comprehensive reviews of the literature for a highly focused research question. This paper investigates using prompt engineering with ChatGPT to generate boolean queries for systematic reviews. By analyzing the results, it reveals that (1) ChatGPT give good performance with better precision than recall, (2) the type of prompts used has considerable effects on the effectiveness of the queries produced by ChatGPT, (3) guided prompts lead to higher effectiveness than single prompt strategies for both precision and recall, (4) two main issues, i.e., incorrect MeSH terms, and high variability in query effectiveness across multiple requests.


*2023-06-25*

#### [Unifying Large Language Models and Knowledge Graphs: A Roadmap](https://arxiv.org/abs/2306.08302)

*Shirui Pan, Linhao Luo,Yufei Wang, Chen Chen, Jiapu Wang, Xindong Wu*

*Arxiv 2021*

This paper discusses three methods as a roadmap for unifying LLMs and KGs including: (1) KG-enhanced LLMs, which incorporates KGs during the pre-training and inference phases of LLMs, or for the purpose of enhancing understanding of the knowledge learned by LLMs, (2) LLM-augmented KGs, that leverages LLMs for different KG tasks such as embedding, completion, construction, graph-to-text generation, and question answering, (3) Synergized LLMs + KGs, in which LLMs and KGs play equal roles and work in a mutually beneficial way to enhance both LLMs and KGs for bidirectional reasoning driven by both data and knowledge.


*2023-06-24*

#### [WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences](https://arxiv.org/pdf/2306.07906.pdf)

*Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan Zeng, Zhengxiao Du, Peng Zhang, Yuxiao Dong, Jie Tang*

*KDD 2023*

This paper proposes WebGLM, a LLM-based system with Web search abilities for question answering. The system is compared with WebGPT. This paper also proposes systematic criteria for evaluating Web-enhanced QA systems.


*2023-06-01*

#### [Can Language Models Solve Graph Problems in Natural Language?](https://arxiv.org/abs/2305.10037)

*Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov*

*Arxiv 2023*

This paper proposes a benchmark of graph-based reasoning tasks designed for evaluating LLMs. The evaluation results on GPT-3/4 demonstrate that: (1) LLMs do possess preliminary graph reasoning abilities. (2) The benefit of advanced prompting methods diminishes with complex problems. (3) Learning from examples did not happen on complex graph reasoning problems. (4) LLMs are (un)surprisingly brittle to spurious correlations in problem settings.


*2023-05-31*

#### [From chocolate bunny to chocolate crocodile: Do Language Models Understand Noun Compounds?](https://arxiv.org/pdf/2305.10568.pdf)

*Jordan Coil, Vered Shwartz*

*ACL 2023*

Noun compound interpretation is the task of expressing a noun compound (e.g. chocolate bunny) in a free-text paraphrase that makes the relationship between the constituent nouns explicit (e.g. bunny-shaped chocolate). This paper investigates this task by modifying the noun compounds with rare or novel compounds, e.g., chocolate crocodile, and re-evaluates the performance of LLMs (e.g., GPT-3). The result shows that the outputs from GPT-3 often have significant overlap with a large Web corpus, but the parroting strategy is less beneficial for novel noun compounds.


*2023-05-29*

#### [What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning](https://arxiv.org/pdf/2305.09731.pdf)

*Jane Pan, Tianyu Gao, Howard Chen, Danqi Chen*

*ACL 2023*

This paper investigates the in-context learning ability of LLMs by dividing it into task recognition and task learning. It mainly reveals that (1) models can achieve non-trivial performance with only TR, and TR does not further improve with larger models or more demonstrations; (2) LLMs acquire TL as the model scales, and TL’s performance consistently improves with more demonstrations in context.


*2023-05-25*

#### [Are Machine Rationales (Not) Useful to Humans? Measuring and Improving Human Utility of Free-Text Rationales](https://arxiv.org/abs/2305.07095)

*Brihi Joshi, Ziyi Liu, Sahana Ramnath, Aaron Chan, Zhewei Tong, Shaoliang Nie, Qifan Wang, Yejin Choi, Xiang Ren*

*ACL 2023*

As LLMs being able to generate free-text rationales (i.e., chain-of-thoughts as explanation), this paper investigates the ability of LMs to help humans answer questions based on the generated rationales. To achieve this, it proposes a metric Gen-U to estimate the rationale’s helpfulness in answering similar but unseen instances. It also shows that the LMs can be trained to have better human utility by optimizing Gen-U while retaining similar performance on other tasks.


*2023-04-18*

#### [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/pdf/2304.05302.pdf)

*Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang*

*Arxiv 2023*

InstructGPT implements RLHF through several stages, including Supervised Fine-Tuning (SFT), reward model training, and Proximal Policy Optimization (PPO). To address the problem that PPO is sensitive to parameter and hard to train, this paper proposes another learning method by aligning model response with human feedback through ranking loss.


*2023-04-09*

#### [How well do Large Language Models perform in Arithmetic tasks](https://arxiv.org/pdf/2304.02015.pdf)

*Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang*

*Arxiv 2023*

This paper evaluates the ability of LLMs handling arithmetic tasks, such as Euler equation and decimal calculation. It also analyzes the failed cases of ChatGPT and the effectiveness of different prompts.


*2023-04-06*

#### [Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!](https://arxiv.org/abs/2303.08559)

*Yubo Ma, Yixin Cao, YongChing Hong, Aixin Sun*

*Arxiv 2023*

Although LLMs exhibit attractive in-context learning ability, their performance is sometimes limited due to the constrained input length, and increasing the number of demonstrations cannot significantly improve the performance of LLMs. Therefore, this paper proposes a filter-then-rerank pipeline, which firstly utilizes small LMs (SLMs) to handle the easy cases, then passes the complex part to LLMs as a reranking task. The comparison shows that LLMs perform better than SLMs on complex (i.e., "harder") tasks.


*2023-04-04*

#### [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/pdf/2303.17580.pdf)

*Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang*

*Arxiv 2023*

This paper proposes to combine ChatGPT with AI models available in HuggingFace to solve complex tasks. Specifically, it uses ChatGPT to conduct task planning when receiving a user request, selects models according to their function descriptions available in HuggingFace, executes each subtask with the selected
AI model, and summarizes the response according to the execution results.

