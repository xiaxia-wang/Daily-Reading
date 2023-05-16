






*2023-05-16*

#### [Solving Math Word Problems via Cooperative Reasoning induced Language Models](https://arxiv.org/pdf/2210.16257)

*Xinyu Zhu, Junjie Wang, Lin Zhang, Yuxiang Zhang, Ruyi Gan, Jiaxing Zhang, Yujiu Yang*

*ACL 2023*

This paper proposes a reasoning framework containing cooperative training and inference. By making a PLM output both the answer and corresponding reasoning process for a given question, and also including 2 verifiers at the token-level and sentence-level, the model is able to provide feedback in the whole solution generation process.


*2023-05-15*

#### [Faithful Question Answering with Monte-Carlo Planning](https://arxiv.org/pdf/2305.02556.pdf)

*Ruixin Hong, Hongming Zhang, Hong Zhao, Dong Yu, Changshui Zhang*

*ACL 2023*

This paper proposes a model for question answering based on Monte-Carlo planning to conduct step-by-step reasoning. It produces the reasoning steps in the form of an entailment tree, and the answer following from the steps. The entailment tree contains basic facts and novel intermediate conclusions connected by entailment steps, and the entailment step are decided by PLM-based scoring model.


*2023-05-14*

#### [Entity Tracking in Language Models](https://arxiv.org/pdf/2305.02363.pdf)

*Najoung Kim, Sebastian Schuster*

*ACL 2023*

This paper proposes a task of tracking the states of an entity given an English description of its initial state and following state change operations. By experimenting over several PLMs, the results reveal that only models in the GPT3.5 series, which have been trained on both text and code, are able to perform non-trivial entity tracking. Besides, a smaller language model (i.e., T5) can learn to perform nontrivial entity tracking. These results suggest that language models can learn to track entities but pretraining on text corpora alone does not make this capacity surface.


*2023-05-11*

#### [A Unified Generative Retriever for Knowledge-Intensive Language Tasks via Prompt Learning](https://arxiv.org/pdf/2304.14856.pdf)

*Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yiqun Liu, Yixing Fan, Xueqi Cheng*

*SIGIR 2023*

This paper proposes a unified generative retriever model for different knowledge-intensive retrieval tasks, including document retrieval, passage retrieval, sentences retrieval and entity retrieval. It firstly proposes a n-gram-based identifier to identify relevant contexts, and then tries several prompt learning strategy (i.e., discrete prompts, continuous prompts and hybrid prompts) for each of the tasks.


*2023-05-06*

#### [Context Generation Improves Open Domain Question Answering](https://arxiv.org/pdf/2210.06349)

*Dan Su, Mostofa Patwary, Shrimai Prabhumoye, Peng Xu, Ryan Prenger, Mohammad Shoeybi, Pascale Fung, Anima Anandkumar, Bryan Catanzaro*

*EACL 2023*

To improve the performance of closed-book QA (i.e., the model directly answers the question without access to any external knowledge), this paper proposes a two-stage framework, which firstly uses a PLM to generate related contexts for the given question, and then prompts the same LM to generate the answer based on the contexts. Besides, it also marginalizes over the generated contexts to further improve the accuracy and reduce context uncertainty.


*2023-05-05*

#### [Controlled Text Generation with Natural Language Instructions](https://arxiv.org/pdf/2304.14293)

*Wangchunshu Zhou, Yuchen Eleanor Jiang, Ethan Wilcox, Ryan Cotterell, Mrinmaya Sachan*

*ICML 2023*

This paper is about training a LLM for the controlled text generation task. It considers 5 kinds of constraints: lexical constraints, syntax constraints, semantic constraints, style constraints and length constraints. For training data, it collects a very large corpus containing 1M constraint-text pairs for each kind of constraints, and designs templates to transform the training pair, especially the constraint, to natural language instructions to be fed to the model.


*2023-04-23*

#### [Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study](https://arxiv.org/pdf/2304.06762.pdf)

*Boxin Wang, Wei Ping, Peng Xu, Lawrence McAfee, Zihan Liu, Mohammad Shoeybi, Yi Dong, Oleksii Kuchaiev, Bo Li, Chaowei Xiao, Anima Anandkumar, Bryan Catanzaro*

*Arxiv 2023*

This paper proposes a comprehensive comparison between LMs with or without retrieval as augmentation. It shows that, smaller pretrained LMs with retrieval augmentation generally outperforms the standard GPT on text generation and other tasks. The findings highlight the direction of pretraining autoregressive LMs with retrieval being a promising model in future work.


*2023-03-31*

#### [ChatGPT is a Knowledgeable but Inexperienced Solver: An Investigation of Commonsense Problem in Large Language Models](https://arxiv.org/abs/2303.16421)

*Ning Bian, Xianpei Han, Le Sun, Hongyu Lin, Yaojie Lu, Ben He*

*Arxiv 2023*

This paper proposes an empirical work to evaluate the ability of LLMs, especially GPTs to answer commonsense questions and exploit commonsense knowledge. This paper firstly introduces 11 commonsense datasets, and categorizes them into 8 different kinds of knowledge (e.g., physical, scientific). Then from each dataset it samples several natural language questions and asks the GPTs for an answer. It also investigates the ability of GPTs to understand and leverage the commonsense knowledge.


*2023-03-27*

#### [JiuZhang: A Chinese Pre-trained Language Model for Mathematical Problem Understanding](https://doi.org/10.1145/3534678.3539131)

*Wayne Xin Zhao, Kun Zhou, Zheng Gong, Beichen Zhang, Yuanhang Zhou, Jing Sha, Zhigang Chen, Shijin Wang, Cong Liu, Ji-Rong Wen*

*KDD 2022*

This paper proposes a pre-trained language model for solving mathematical problems. It designs 3 pre-training tasks based on a model architecture consisting of a shared encoder, a decoder for understanding problems and another decoder for generation. The 3 pre-training tasks are (from easy to hard): (1) masked token prediction, (2) mathematical logic recovering, and (3) solution checking.


*2023-03-25*

#### [A Logic Aware Neural Generation Method for Explainable Data-to-text](https://dl.acm.org/doi/10.1145/3534678.3539082)

*Xiexiong Lin, Huaisong Li, Tao Huang, Feng Wang, Linlin Chao, Fuzhen Zhuang, Taifeng Wang, Tianyi Zhang*

*KDD 2022*

This paper proposes a data-to-text application for anti-money laundering system based on input tabular data and expert rules. It firstly builds a logic graph based on expert rules, and extracts the meta paths from the graph. Then the tabular data and meta paths are input into a transformer encoder with an attention-based retriever to extract related statements. The target of the system is to generate descriptive text of risky behaviors of customers, which is implemented as a LSTM-based decoder.


*2023-02-06*

#### [Link-BERT: Pretraining a Language Model with Document Links](https://knowledge-nlp.github.io/aaai2023/papers/007-LinkBERT-poster.pdf)

*Michihiro Yasunaga, Jure Leskovec, Percy Liang*

*KnowledgeNLP-AAAI 2023*

This paper proposes a pretrained BERT model named LinkBERT to capture links between documents. It regards a text corpus as a graph of documents and uses linked documents in the same context as inputs. Then it applies two pretraining tasks to the model: masked language modeling and document relation prediction. The experiment shows that this new model performs better especially in multi-hop reasoning and few-shot QA tasks.


*2023-02-05*

#### [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://doi.org/10.18653/v1/D19-1410)

*Nils Reimers, Iryna Gurevych*

*EMNLP 2019*

This paper proposes SentenceBERT, a model for computing semantic relevance between pair of sentences. The original network structure of BERT handles this as a regression problem, which requires $n^2$ time complexity thus being inefficient in practice. To address this problem, this paper proposes a "Siamese" network with two BERT  sharing same weights, which is pretrained with pairs of sentences and the goal is to minimize the gap of embedding vectors. In practice,  SentenceBERT directly generates embeddings for sentences whose semantic relevance can be measured by cosing similarity.


*2023-02-03*

#### [Knowledge Relevance BERT: Integrating Noisy Knowledge into Language Representation](https://knowledge-nlp.github.io/aaai2023/papers/005-KRBERT-oral.pdf)

*Karan Samel, Jun Ma, Zhengyang Wang, Tong Zhao, Irfan Essa*

*KnowledgeNLP-AAAI 2023*

This paper proposes a model to integrate noisy real-world data for entity linking task. It is based on the relevance between candidate entity labels and existing labels of the training data. The relevance is computed using a language model K-BERT, which performed well in the experiments.


*2023-01-30*

#### [DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining](https://knowledge-nlp.github.io/aaai2023/papers/001-Dragon-oral.pdf)

*Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D Manning, Percy Liang, Jure Leskovec*

*KnowledgeNLP-AAAI 2023*

This paper proposes a self-supervised pretrained model that jointly combines texts and KGs. It takes pairs of text segments and relevant KG subgraphs as input, and fuses the LM and GNN layer to generate the output. The model is pre-trained with a masked LM task and a KG link predication task. The experiments show that this jointly pretrained model can achieve better performance on question answering tasks.


*2023-01-18*

#### [NeuroLogic A*esque Decoding: Constrained Text Generation with Lookahead Heuristics](https://aclanthology.org/2022.naacl-main.57/)

*Ximing Lu, Sean Welleck, Peter West, Liwei Jiang, Jungo Kasai, Daniel Khashabi, Ronan Le Bras, Lianhui Qin, Youngjae Yu, Rowan Zellers, Noah A. Smith, Yejin Choi*

*NAACL 2022*

Motivated by the idea of A* search, this paper proposes a lookahead heuristic to improve the performance of neural text generation based on left-to-right decoder. The model incorporates a length-constrained "lookahead" mechanism to predict the best "next-word" generation, which should be close to the overall generation target. It aims to approximately optimize the future cost. In the experiments, the heuristics are evaluated over several tasks including commonsense generation, constrained machine translation, table-to-text generation and constrained question answering. 


*2023-01-17*

#### [Reframing Human-AI Collaboration for Generating Free-Text Explanations](https://aclanthology.org/2022.naacl-main.47/)

*Sarah Wiegreffe, Jack Hessel, Swabha Swayamdipta, Mark O. Riedl, Yejin Choi*

*NAACL 2022*

This paper proposes a new framework to make large PLMs generate human-acceptable explanations for free-text question answering. It firstly uses GPT-3 with few-shot prompts to (over-)generate potential explanation candidates. Then based on a set of binary crowdsourcing labels, it trains a filter to select high-quality explanations for evaluation. The result shows this over-generation & filteration pipeline works well for generating explanations on CommonsenseQA and SNLI. 


*2023-01-16*

#### [Understanding Dataset Difficulty with V-Usable Information](https://proceedings.mlr.press/v162/ethayarajh22a.html)

*Kawin Ethayarajh, Yejin Choi, Swabha Swayamdipta*

*ICML 2022*

This paper proposes a new measure called V-usable information, which can be used to interpret the difficulty of datasets w.r.t a given model V (e.g., BERT-base). Besides, it also introduces pointwise V-information (PVI) for measuring the difficulty of individual instances (in the dataset) w.r.t. a given distribution.


*2023-01-15*

#### [Is GPT-3 Text Indistinguishable from Human Text? Scarecrow: A Framework for Scrutinizing Machine Text](https://aclanthology.org/2022.acl-long.501/)

*Yao Dou, Maxwell Forbes, Rik Koncel-Kedziorski, Noah A. Smith, Yejin Choi*

*ACL 2022*

This paper proposes a scrutinizing framework to evaluate the texts generated by pre-trained language model. It proposes 10 kinds of errors which involve language errors, factual errors and reader issues. Then it summarizes 4 key insights based on 41k crowdsourcing annotated error spans. 


*2023-01-14*

#### [Generated Knowledge Prompting for Commonsense Reasoning](https://aclanthology.org/2022.acl-long.225/)

*Jiacheng Liu, Alisa Liu, Ximing Lu, Sean Welleck, Peter West, Ronan Le Bras, Yejin Choi, Hannaneh Hajishirzi*

*ACL 2022*

This paper proposes a knowledge prompting method to improve commonsense reasoning. The method is mainly divided into two steps. Firstly, it generates relevant knowledge statements as prompts using a pre-trained language model. Then in the second step, it uses these prompts with another language model to predict the final answer to the question. It shows the accuracy is improved a lot with the generated knowledge prompts. 


*2022-12-31*

#### [TIARA: Multi-grained Retrieval for Robust Question Answering over Large Knowledge Bases](https://arxiv.org/abs/2210.12925)

*Yiheng Shu, Zhiwei Yu, Yuhan Li, Börje F. Karlsson, Tingting Ma, Yuzhong Qu, Chin-Yew Lin*

*EMNLP 2022*

This paper proposes a KBQA method based on PLM. It consists of entity retrieval, schema retrieval, logical form retrieval parts, and feeds them into a PLM to generate the final results. 


*2022-12-14*

#### [MultPAX: Keyphrase Extraction Using Language Models and Knowledge Graphs](https://doi.org/10.1007/978-3-031-19433-7_18)

*Hamada M. Zahera, Daniel Vollmers, Mohamed Ahmed Sherif, Axel-Cyrille Ngonga Ngomo*

*ISWC 2022*

Keyphrase extraction aims at identifying a small set of phrases which represent the content of a document. This paper proposes a method named MultPAX for extracting existing keyphrases from the document and adding absent phrases based on external KGs. It firstly identifies existing phrases using a PLM. Then it links these phrases to the KG and gets relevant ones. Finally it ranks all the phrases and uses the top-k as the output.


*2022-12-04*

#### [Revisiting Pre-Trained Models for Chinese Natural Language Processing](https://doi.org/10.18653/v1/2020.findings-emnlp.58)

*Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, Guoping Hu*

*EMNLP Findings 2020*

This paper proposes an improved version of BERT especially for Chinese. It uses Chinese whole-word-masking to replace the original token-level mask. Besides, it also applies similar words (based on word2vec) as the mask instead of the special token [MASK], to enhance the pre-training performance. 


*2022-12-02*

#### [Neural Module Networks for Reasoning over Text](https://openreview.net/forum?id=SygWvAVFPr)

*Nitish Gupta, Kevin Lin, Dan Roth, Sameer Singh, Matt Gardner*

*ICLR 2020*

This paper proposes a model for question answering with reasoning over texts. Given an input question, it applies a question parser to divide it into several execution modules. Each module is associated with a reasoning task, and implemented based on different attention mechanism. 


*2022-12-01*

#### [RoBERTa: A Robustly Optimized BERT Pretraining Approach](http://arxiv.org/abs/1907.11692)

*Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov*

*Arxiv 2019*

This paper proposes a new training strategy to apply on BERT which achieves better model performance. 


*2022-11-29*

#### [Few-Shot NLG with Pre-Trained Language Model](https://doi.org/10.18653/v1/2020.acl-main.18)

*Zhiyu Chen, Harini Eavani, Wenhu Chen, Yinyin Liu, William Yang Wang*

*ACL 2020*

This paper proposes a NLG model which relies on a few training data and a pre-trained language model. Gives a table as input, the pre-trained language model (GPT-2 in this paper) serves as the generator, which switches between copying/pasting the labels in the table, and composing fluent, coherent sentences.


*2022-11-26*

#### [DyRRen: A Dynamic Retriever-Reranker-Generator Model for Numerical Reasoning over Tabular and Textual Data](https://arxiv.org/abs/2211.12668)

*Xiao Li, Yin Zhu, Sichen Liu, Jiangzhou Ju, Yuzhong Qu, Gong Cheng*

*AAAI 2023*

This paper optimizes the FinQANet model for numerical reasoning. Its input is a textual question, a table with data and a set of sentences. Its output is a numerical expression targeting to answer the question. This paper proposes a retriever-generator model with a re-ranker module which adjusts the ranking score according to the decoder output in each step. 


*2022-11-21*

#### [BERTMap: A BERT-Based Ontology Alignment System](https://ojs.aaai.org/index.php/AAAI/article/view/20510)

*Yuan He, Jiaoyan Chen, Denvar Antonyrajah, Ian Horrocks*

*AAAI 2022*

This paper proposes an ontology matching approach using BERT and rule-based refinement. It firstly extracts pairs of synonyms and non-synonyms (labels of classes) from multiple sources. Then it implements sub-word indexes, string-based matching and a fine-tuned BERT for prediction. Finally, it evaluates neighboring classes of the predicted results, adds them into matched pairs, and deletes false positive pairs based on rules.


*2022-11-20*

#### [Approximated Doubly Robust Search Relevance Estimation](https://doi.org/10.1145/3511808.3557145)

*Lixin Zou, Changying Hao, Hengyi Cai, Shuaiqiang Wang, Suqi Cheng, Zhicong Cheng, Wenwen Ye, Simiu Gu, Dawei Yin*

*CIKM 2022*

This paper proposes a PLM-based learning model to avoid the biases (e.g., positional bias, trust bias) in users' click-through logs and estimate the real query-document relevance. The model is based on ERNIE 2.0, and fine-tuned with unbiased data to minimize the bias and variance loss. This paper also proposes a robust relevance estimator based on the model and implements it in an online system.


*2022-10-30*

#### [QEN: Applicable Taxonomy Completion via Evaluating Full Taxonomic Relations](https://doi.org/10.1145/3485447.3511943)

*Suyuchen Wang, Ruihui Zhao, Yefeng Zheng, Bang Liu*

*TheWebConf 2022*

Taxonomy completion is to find a pair of $\langle$ hypernym, hyponym $\rangle$ in the existing taxonomy for inserting the query concept. This paper firstly incorporates the similarities between the query and two potential siblings, and utilizes the term descriptions instead of embeddings as input features. Besides, it also applies a code attention module with PLM to reduce the online computation efforts. 


*2022-10-29*

#### [EventBERT: A Pre-Trained Model for Event Correlation Reasoning](https://doi.org/10.1145/3485447.3511928)

*Yucheng Zhou, Xiubo Geng, Tao Shen, Guodong Long, Daxin Jiang*

*TheWebConf 2022*

This paper proposes a pre-trained model to conduct event correlation reasoning. It firstly collects a set of training examples (natural language paragraphs and events) from a large book corpus. Then it proposes three self-supervised event-based and correlation-based learning objectives to pre-train the model, including correlation-based relation ranking, contradiction event tagging and discourse relation ranking. The former two train the model to distinguish the correct event against negative ones, while the latter helps the model identify subtle difference among discourse relations. 


*2022-10-23*

#### [Enhancing Knowledge Bases with Quantity Facts](https://doi.org/10.1145/3485447.3511932)

*Vinh Thinh Ho, Daria Stepanova, Dragan Milchevski, Jannik Strötgen, Gerhard Weikum* (MPI)

*TheWebConf 2022*

This paper proposes a recall-oriented knowledge base augmentation method named QL, to add missing quantity facts into the existing KB. It extracts facts from external text corpus, and divides them into high-confidence and low-confidence groups. Then it iteratively consolidates the facts using distribution-based denoising method and expends the query with more equivalent properties. In the experiments, QL is compared with other question answering methods including RoBERTa, QSearch and GPT-3 on precision, recall and novelty. 


*2022-10-23*

#### [Unified Question Generation with Continual Lifelong Learning](https://dl.acm.org/doi/10.1145/3485447.3511930)

*Wei Yuan, Hongzhi Yin, Tieke He, Tong Chen, Qiufeng Wang, Lizhen Cui:*

*TheWebConf 2022*

This paper proposes a unified model for natural language question generation (QG) tasks. It is based on *T5* for natural language generation. Firstly, it unifies four formats of QG (i.e., answer-extraction, answer-abstraction, multi-choice and boolean QG) by concatenating their different components (i.e., the answer, passage, distractor, etc. ) as input, respectively. Then it applies life-long learning by keeping and re-playing difficult examples with similarity regularization to reduce the negative effect of forgetting history. 


*2022-10-21*

#### [Ontology-enhanced Prompt-tuning for Few-shot Learning](https://doi.org/10.1145/3485447.3511921)

*Hongbin Ye, Ningyu Zhang, Shumin Deng, Xiang Chen, Hui Chen, Feiyu Xiong, Xi Chen, Huajun Chen*

*TheWebConf 2022*

This paper proposes to use ontology information to enhance prompt-tuning in few-shot learning tasks. It evaluates three tasks including relation extraction, event extraction and knowledge graph completion in this paper. The ontology information of target entities used in this paper is extracted from the knowledge graph, and used as plain texts as auxiliary prompts for PLM. 
