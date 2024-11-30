








*2024-11-30*

#### [Do PLMs Know and Understand Ontological Knowledge?](https://aclanthology.org/2023.acl-long.173/)

*Weiqi Wu, Chengyue Jiang, Yong Jiang, Pengjun Xie, Kewei Tu*

*ACL 2023*

This paper investigates how well PLMs are able to memorize: (1) types of entities; (2) hierarchical relationships among classes and properties; (3) domain and range constraints of properties. To probe whether PLMs truly understand ontological knowledge beyond memorization, it evaluates whether PLMs can reliably perform logical reasoning with given knowledge according to ontological entailment rules. Results show that, although they can memorize certain ontological knowledge and utilize implicit knowledge in reasoning, both performances are less than perfect.


*2024-10-20*

#### [Knowledge Circuits in Pretrained Transformers](https://arxiv.org/abs/2405.17969)

*Yunzhi Yao, Ningyu Zhang, Zekun Xi, Mengru Wang, Ziwen Xu, Shumin Deng, Huajun Chen*

*NeurIPS 2024*

This paper investigates the computation graph of language models, focusing on knowledge circuits, a critical subgraph in the language model to view the knowledge mechanism of Transformers. Specifically, it analyzes the information flow in the connected directed acyclic graph (DAG) representing the model, where individual nodes represent various components involved in the forward pass, such as neurons, attention heads, and embeddings. The edges symbolize interactions between these components, including residual connections, attention mechanisms, and projections.


*2024-07-25*

#### [Learning In-context Learning for Named Entity Recognition](https://aclanthology.org/2023.acl-long.764/)

*Jiawei Chen, Yaojie Lu, Hongyu Lin, Jie Lou, Wei Jia, Dai Dai, Hua Wu, Boxi Cao, Xianpei Han, Le Sun*

*ACL 2023*

This paper proposes an in-context learning-based NER approach, which effectively injects in-context NER ability into PLMs and recognizes entities of novel types on-the-fly using only a few demonstrative instances.  Specifically, it models the PLM as a meta-function, and a new entity extractor can be implicitly constructed by applying new instruction and demonstrations to the PLM.


*2024-07-19*

#### [It Ain't That Bad: Understanding the Mysterious Performance Drop in OOD Generalization for Generative Transformer Models](https://arxiv.org/abs/2308.08268)

*Xingcheng Xu, Zihao Pan, Haipeng Zhang, Yanqing Yang*

*IJCAI 2024*

This paper investigates the performance drop of LLMs for OOD generalization. Specifically, by training smaller models with similar attention mechanism, the authors discover that the strong in-distribution generalization stems from structured representations. While behind the unsatisfying OOD performance, the models still exhibit clear learned algebraic structures, by mapping unseen OOD inputs to outputs with learned equivalence relations in the in-distribution domain.


*2024-06-10*

#### [Successor Heads: Recurring, Interpretable Attention Heads In The Wild](https://openreview.net/forum?id=kvcbV8KQsi)

*Rhys Gould, Euan Ong, George Ogden, Arthur Conmy*

*ICLR 2024*

This paper presents empirical findings of the successor heads (essentially attention heads that capture the "increment" relationship in some natural ordering) in LLMs based on extensive experiments, which is interesting and gives insights to understand LLMs better through mechanical interpretability.


*2024-04-29*

#### [The Integration of Semantic and Structural Knowledge in Knowledge Graph Entity Typing](https://arxiv.org/abs/2404.08313)

*Muzhi Li, Minda Hu, Irwin King, Ho-fung Leung*

*NAACL 2024*

The Knowledge Graph Entity Typing (KGET) task aims to predict missing type annotations for entities in knowledge graphs. This paper proposes an end-to-end model for the task, including a semantic knowledge encoding module that encodes factual knowledge in the KG with a masked entity typing task, a structural knowledge aggregation module that aggregates knowledge from the multi-hop neighborhood of entities to infer missing types, and an unsupervised type re-ranking module that utilizes the inference results from previous modules to generate type predictions that are robust to false-negative samples.


*2024-04-16*

#### [NoteLLM: A Retrievable Large Language Model for Note Recommendation](https://arxiv.org/abs/2403.01744)

*Chao Zhang, Shiwei Wu, Haoxin Zhang, Tong Xu, Yan Gao, Yao Hu, Di Wu, Enhong Chen*

*WWW 2024*

This paper proposes a model named NoteLLM for item-to-item (I2I) note recommendation. Specifically, it uses a unified prompt for I2I note recommendation and hashtag/category generation, where notes are compressed via the Note Compression Prompt and processed by pre-trained LLMs. It utilizes the co-occurrences to construct the related note pairs and train the I2I recommendation task using Generative-Contrasting Learning. Besides, NoteLLM also extracts each note’s key concepts for hashtag/category generation to enhance the I2I recommendation task.


*2024-04-11*

#### [Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs](https://arxiv.org/abs/2307.03393)

*Zhikai Chen, Haitao Mao, Hang Li, Wei Jin, Hongzhi Wen, Xiaochi Wei, Shuaiqiang Wang, Dawei Yin, Wenqi Fan, Hui Liu, Jiliang Tang*

*KDD 2024*

This paper explores two potential pipelines to incorporate LLMs to node classification tasks on text-attributed graphs. (1) LLMs-as-Enhancers: LLMs are adopted to enhance the textual information; subsequently, GNNs utilize refined textual data to generate predictions. (2) LLMs-as-Predictors: LLMs are adapted to generate the final predictions, where structural and attribute information is present completely through natural languages.


*2024-03-25*

#### [GraphTranslator: Aligning Graph Model to Large Language Model for Open-ended Tasks](https://arxiv.org/abs/2402.07197)

*Mengmei Zhang, Mingwei Sun, Peng Wang, Shen Fan, Yanhu Mo, Xiaoxiao Xu, Hong Liu, Cheng Yang, Chuan Shi*

*WWW 2024*

Pretrained graph models are usually able to handle pre-defined tasks but not for open-ended ones, while LLMs offer the capability to address open-ended tasks. This paper proposes a model that can solve both pre-defined and open-ended tasks by combining both paradigms, where the graph model is leveraged for pre-defined tasks, and the LLM is extended as the interface of the graph model for open-ended tasks.


*2024-03-17*

#### [On the Stability of Iterative Retraining of Generative Models on their own Data](https://openreview.net/forum?id=JORAfH2xFd)

*Quentin Bertrand, Joey Bose, Alexandre Duplessis, Marco Jiralerspong, Gauthier Gidel*

*ICLR 2024 Spotlight*

This paper provides a theoretical understanding of when generative models iteratively re-trained on their own data diverge from the original target distribution. It develops a framework to study the impact of training generative models on mixed datasets (of real and synthetic data). It proves the stability of iterative training under the condition that the initial generative models approximate the data distribution well enough, and the proportion of clean training data (w.r.t. synthetic data) is large enough.


*2024-03-12*

#### [Evaluating the Zero-shot Robustness of Instruction-tuned Language Models](https://openreview.net/forum?id=g9diuvxN6D)

*Jiuding Sun, Chantal Shaib, Byron C Wallace*

*ICLR 2024 Spotlight*

This paper answers two questions: (1) How sensitive are instruction-tuned models to the particular phrasings of instructions, (2) How can we make them more robust to such natural language variation? To answer the former, it collects a set of instructions manually written by NLP practitioners and evaluates the variance and average performance of these instructions as compared to instruction phrasings observed during instruction fine-tuning. The result shows novel (unseen) instructions typically lead to performance degredation. To solve this problem, it proposes a soft-embedding approach to maximize the similarity between representations of semantically equivalent instructions.


*2024-03-10*

#### [SNIP: Bridging Mathematical Symbolic and Numeric Realms with Unified Pre-training](https://openreview.net/forum?id=KZSEgJGPxu)

*Kazem Meidani, Parshin Shojaee, Chandan K. Reddy, Amir Barati Farimani*

*ICLR 2024 Spotlight*

The authors propose pre-training for mathematical expressions via multi-modal contrastive learning that jointly encodes the symbolic and numerical parts of the expressions via a dual-encoder scheme. The tasks considered are property prediction and symbolic regression.    The proposed method is then used for a qualitative assessment of the learned embeddings and for downstream symbolic regression tasks (with a added a decoder). It results competitive w.r.t. recent baselines on these benchmarks.


*2024-03-08*

#### [Making Pre-trained Language Models Great on Tabular Prediction](https://openreview.net/forum?id=anzIzGZuLi)

*Jiahuan Yan, Bo Zheng, Hongxia Xu, Yiheng Zhu, Danny Chen, Jimeng Sun, Jian Wu, Jintai Chen*

*ICLR 2024 Spotlight*

This paper presents TP-BERTa, a specifically pre-trained LM model for tabular data prediction. Concretely, a relative magnitude tokenization converts scalar numerical feature values to finely discrete, high-dimensional tokens, and an intra-feature attention approach integrates feature values with corresponding feature names.


*2024-02-28*

#### [Memorization Capacity of Multi-Head Attention in Transformers](https://openreview.net/forum?id=MrR3rMxqqv)

*Sadegh Mahdavi, Renjie Liao, Christos Thrampoulidis*

*ICLR 2024 Spotlight*

This paper presents a lower bound on the memorization capacity of multi-head attention layers under a set of input-data assumptions. The results demonstrate that memorization increases with the number of heads, monotonically with the context size, and monotonically with the head dimension up to the context size. These findings are supported by experimental results obtained using synthetic data.


*2024-01-24*

#### [A Branching Decoder for Set Generation](https://openreview.net/forum?id=riNuqYiD66)

*Zixian Huang et al.*

*ICLR 2024*

Typical approaches usually handle set generation as a sequential generation problem, such as generating a set of keywords for summarizing a paper. In contrast, this paper proposes a parallel approach that generates each element in the set in an independent manner, which is closer to the unordered nature of sets, and also improves the generation efficiency.


*2023-12-17*

#### [Make Your Decision Convincing! A Unified Two-Stage Framework: Self-Attribution and Decision-Making](https://arxiv.org/abs/2310.13610)

*Yanrui Du, Sendong Zhao, Haochun Wang, Yuhan Chen, Rui Bai, Zewen Qiang, Muzhen Cai, Bing Qin*

*EMNLP 2023 Findings*

This paper proposes a unified two-stage prompt-based framework known as Self-Attribution and Decision-Making (SADM), to improve the causal relations between the predictions and explanations given by LLMs, i.e., to mitigate the problem of LLMs giving right answer with wrong explanation, or right explanation with wrong answer.


*2023-11-29*

#### [Harnessing the Power of Large Language Models for Natural Language to First-Order Logic Translation](https://arxiv.org/abs/2305.15541)

*Yuan Yang, Siheng Xiong, Ali Payani, Ehsan Shareghi, Faramarz Fekri*

*Arxiv 2023*

This paper introduces LOGICLLAMA, a LLaMA-7B model fine-tuned for NL-FOL translation using LoRA on a single GPU. LOGICLLAMA is capable of directly translating natural language into FOL rules, which outperforms GPT-3.5. LOGICLLAMA is also equipped to correct FOL rules predicted by GPT-3.5, and can achieve similar performance as GPT-4 with a fraction of the cost.


*2023-11-21*

#### [LEMON: Language-Based Environment Manipulation via Execution-Guided Pre-training](https://aclanthology.org/2022.findings-emnlp.33/)

*Qi Shi, Qian Liu, Bei Chen, Yu Zhang, Ting Liu, Jian-Guang Lou*

*EMNLP Findings 2022*

This paper investigates the task of language-based environment manipulation, by formulating it as a conversational process which starts and ends in some specific states. It proposes an execution-guided pre-training strategy based on a synthetic corpus for the base model of BART.


*2023-11-15*

#### [Relevance-based Infilling for Natural Language Counterfactuals](https://dl.acm.org/doi/10.1145/3583780.3615029)

*Lorenzo Betti, Carlo Abrate, Francesco Bonchi, Andreas Kaltenbrunner*

*CIKM 2023*

In the context of an NLP classification task, a counterfactual is a piece of text that, while being as similar as possible to the original text, is classified differently by the model. This paper introduces RELITC, an approach to generate counterfactuals for text classifiers that first masks important words for the classifier and then infills them through a masked language model conditioned on the counterfactual class.


*2023-11-09*

#### [GripRank: Bridging the Gap between Retrieval and Generation via the Generative Knowledge Improved Passage Ranking](https://dl.acm.org/doi/10.1145/3583780.3614901)

*Jiaqi Bai, Hongcheng Guo, Jiaheng Liu, Jian Yang, Xinnian Liang, Zhao Yan, Zhoujun Li*

*CIKM 2023*

To improve the retrieval-enhanced text generation model performance, this paper proposes a generative passage estimator for ranking the candidate passages. Motivated by the problem that sometimes the most similar passage to the passage retriever does not able to answer the query, the proposed generative passage estimator is a generative language model that measures how likely a candidate passage can be used to guide the answer generation to the given query.


*2023-10-24*

#### [The Wisdom of Hindsight Makes Language Models Better Instruction Followers](https://proceedings.mlr.press/v202/zhang23ab.html)

*Tianjun Zhang, Fangchen Liu, Justin Wong, Pieter Abbeel, Joseph E. Gonzalez*

*ICML 2023*

This paper proposes to view the process of RLHF in a more direct reinforcement learning manner. It converts the feedback to instruction by relabeling the original instruction and trains the model for better alignment in a supervised manner. It proposes a two-phase algorithm for this approach, with the labels for the reasoning task consist of only true or false.


*2023-10-19*

#### [Why do Nearest Neighbor Language Models Work?](https://proceedings.mlr.press/v202/xu23a.html)

*Frank F. Xu, Uri Alon, Graham Neubig*

*ICML 2023*

kNN-LM extends a trained base LM by linearly interpolating the output distribution with a kNN model. The nearest neighbors are retrieved according to the distances between the current context embedding of the base LM and all the context embeddings in the datastore. This paper investigates the reason of the superior performance of kNN-LMs, including (1) ensembling the  output of softmax, (2) using approximate nearest neighbor search, and (3) adding a temperature term.


*2023-10-16*

#### [HyperTuning: Toward Adapting Large Language Models without Back-propagation](https://proceedings.mlr.press/v202/phang23a.html)

*Jason Phang, Yi Mao, Pengcheng He, Weizhu Chen*

*ICML 2023*

This paper introduces a new fine-turning approach for LMs based on the notion of hyper network. Specifically,  a hypernetwork is an auxiliary model being used to generate (a part of) parameters for a primary network. Motivated by the idea, this paper proposes a set of T5-based hypermodels that output soft prefixes or LoRA parameters for a frozen T5 model from few-shot examples.


*2023-10-12*

#### [Tuning Language Models as Training Data Generators for Augmentation-Enhanced Few-Shot Learning](https://proceedings.mlr.press/v202/meng23b.html)

*Yu Meng, Martin Michalski, Jiaxin Huang, Yu Zhang, Tarek F. Abdelzaher, Jiawei Han*

*ICML 2023*

This paper proposes a new approach by using the PLM as a sample generator to enhance few-shot learning. Specifically, it firstly tunes a PLM on the original few-shot samples and uses it as a generator to produce more synthetic training samples. In this process it applies a weighted maximum likelihood based on a discriminative meta learning objective. Finally, another classification PLM is tuned on the augmented sample set to address the specific task.


*2023-10-09*

#### [A Kernel-Based View of Language Model Fine-Tuning](https://proceedings.mlr.press/v202/malladi23a.html)

*Sadhika Malladi, Alexander Wettig, Dingli Yu, Danqi Chen, Sanjeev Arora*

*ICML 2023*

This paper proposes to use the Neural Tangent Kernel (NTK)—which originated as a model to study the gradient descent dynamics of infinitely wide networks with suitable random initialization—to describe fine-tuning of pre-trained LMs. It demonstrates that formulating the downstream task as a masked word prediction problem through prompting often induces kernel-based dynamics during fine-tuning.


*2023-10-04*

#### [Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise](https://proceedings.mlr.press/v202/lin23d.html)

*Zhenghao Lin, Yeyun Gong, Yelong Shen, Tong Wu, Zhihao Fan, Chen Lin, Nan Duan, Weizhu Chen*

*ICML 2023*

A diffusion language model consists of an encoder and a diffusion-based decoder, which simulates a (reverse) discrete-time Markov process by gradually subtracting a Gaussian noise from the noisy data, thus recovering the target text. Therefore, diffusion language models are non-autoregressive language models. This paper proposes a pre-trianed diffusion model with a continuous paragraph denoise objective, in which the training objective contains the squared error between the predicted and true noise, the reconstruction error and the target embeddings.


*2023-09-30*

#### [LongCoder: A Long-Range Pre-trained Language Model for Code Completion](https://proceedings.mlr.press/v202/guo23j.html)

*Daya Guo, Canwen Xu, Nan Duan, Jian Yin, Julian J. McAuley*

*ICML 2023*

This paper proposes a sparse-transformer based model for long code completion task. Specifically, it employs a sliding window mechanism for self-attention and uses two new tokens, i.e., bridge tokens for aggregating local information, and memory tokens for highlighting and memorizing important statements. The sparse mechanism makes it time-efficient.


*2023-09-27*

#### [Transformers Meet Directed Graphs](https://proceedings.mlr.press/v202/geisler23a.html)

*Simon Geisler, Yujia Li, Daniel J. Mankowitz, Ali Taylan Cemgil, Stephan Günnemann, Cosmin Paduraru*

*ICML 2023*

This paper studies the transformer models for directed graphs by proposing a two direction- and structure-aware positional encodings for directed graphs, i.e., (1) the eigenvectors of the Magnetic Laplacian – a direction-aware generalization of the combinatorial Laplacian, and (2) directional random walk encodings.


*2023-09-26*

#### [Cramming: Training a Language Model on a single GPU in one day](https://proceedings.mlr.press/v202/geiping23a.html)

*Jonas Geiping, Tom Goldstein*

*ICML 2023*

This paper investigates the downstream performance achievable with a transformer-based LM trained from scratch with masked language modeling for a single day on a single consumer GPU. By analyzing why scaling down is hard, and which modification actually improve the performance, it provides evidence that the performance closely follows scaling laws observed in large-compute settings.


*2023-09-24*

#### [Specializing Smaller Language Models towards Multi-Step Reasoning](https://proceedings.mlr.press/v202/fu23d.html)

*Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, Tushar Khot*

*ICML 2023*

This paper investigates the reasoning ability for small-scale language models given few-shot chain-of-thought prompts. It demonstrates that (1) balancing LM's performance on multiple tasks is a delicate matter, as improvements on one task may compromise others, and (2) by intentionally paying the price of decreased generic ability, the model's ability for a specific task, e.g., multi-step math reasoning, can be clearly improved.


*2023-09-16*

#### [Pretraining Language Models with Human Preferences](https://proceedings.mlr.press/v202/korbak23a.html)

*Tomasz Korbak, Kejian Shi, Angelica Chen, Rasika Vinayak Bhalerao, Christopher L. Buckley, Jason Phang, Samuel R. Bowman, Ethan Perez*

*ICML 2023*

This paper investigates the problem of pretraining LMs to be aligned with human preferences. It proposes 5 objectives for pretraining, including conditional training, dataset filtering, unlikelihood loss, reward-weighted regression (RWR) and advantage-weighted regression (AWR, the last 2 are offline RL algorithms). It evaluates on three tasks, namely, generating non-toxic text, test without personally identifiable information and PEP8-compliant Python. The results show that conditional training seems more promising in aligning with human preferences.


*2023-09-01*

#### [Efficient Training of Language Models using Few-Shot Learning](https://proceedings.mlr.press/v202/j-reddi23a.html)

*Sashank J. Reddi, Sobhan Miryoosefi, Stefani Karp, Shankar Krishnan, Satyen Kale, Seungyeon Kim, Sanjiv Kumar*

*ICML 2023*

This paper proposes a generic framework for training few-shot learners in a staged manner. The key component of the framework (i.e., seq2seq network with K layers, just like standard Transformer) is to stack a good few-shot learner on a good small language model to provide a good initializer for the larger language model.


*2023-08-02*

#### [Weakly Supervised Explainable Phrasal Reasoning with Neural Fuzzy Logic](https://openreview.net/forum?id=Hu4r-dedqR0)

*Zijun Wu, Zi Xuan Zhang, Atharva Naik, Zhijian Mei, Mauajama Firdaus, Lili Mou*

*ICLR 2023*

Natural language inference (NLI) aims to determine the relationship between two sentences, such as Entailment, Contradiction, and Neutral. This paper proposes a reasoning method which detects relationships between phrases in the two sentences, and induces an overall label for the sentence pair. The model firstly obtains phrases as semantic units, and aligns corresponding phrases by embedding similarity. The NLI labels are then predicted for the aligned phrases. Finally, sentence-level labels are induced from phrasal labels in a fuzzy logic manner.


*2023-07-24*

#### [Can Pre-trained Language Models Understand Chinese Humor](https://dl.acm.org/doi/10.1145/3539597.3570431)

*Yuyan Chen, Zhixu Li, Jiaqing Liang, Yanghua Xiao, Bang Liu, Yunwen Chen*

*WSDM 2023*

To investigate the capability of PLMs to understand humor, this paper conducts three evaluation steps with four tasks. The tasks include humor recognition, humor type classification, humor level classification and punchline detection. The evaluation steps include evaluating original PLM, knowledge-enhanced PLM and humor interpretation. It also proposes a Chinese humor datasets w.r.t. the four tasks.


*2023-07-01*

#### [Sequence-to-Sequence Knowledge Graph Completion and Question Answering](https://aclanthology.org/2022.acl-long.201.pdf)

*Apoorv Saxena, Adrian Kochsiek, Rainer Gemulla*

*ACL 2022*

Unlike typical KGE methods that represent each entity with an embedding, this paper formulates KG link prediction as a sequence-to-sequence task and exchanges the triple scoring approach taken by prior KGE methods with autoregressive decoding.


*2023-06-27*

#### [Incorporating Graph Information in Transformer-based AMR Parsing](https://arxiv.org/abs/2306.13467)

*Pavlo Vasylenko, Pere-Lluís Huguet Cabot, Abelardo Carlos Martínez Lorenzo, Roberto Navigli*

*ACL 2023*

Abstract Meaning Representation (AMR) is a Semantic Parsing formalism that aims at providing a semantic graph abstraction representing a given text. The task is formulated as a seq2seq task with the input being a sentence and the output being a linearized graph. It incorporates the structural information from a word-aligned graph (WAG) to enrich the encoder's hidden representation. The model is built based on BART.


*2023-06-15*

#### [Learning Multi-step Reasoning from Arithmetic Task](https://arxiv.org/pdf/2306.01707.pdf)

*Tianduo Wang, Wei Lu*

*ACL 2023*

This paper proposes to enable relatively small LMs with the capability of multi-step mathematical reasoning by continually pre-training (i.e., before fine-tuning) the LM on a synthetic dataset named MSAT, short for Multi-step Arithmetic Task.


*2023-06-14*

#### [Brainformers: Trading Simplicity for Efficiency](https://arxiv.org/pdf/2306.00008.pdf)

*Yanqi Zhou, Nan Du, Yanping Huang, Daiyi Peng, Chang Lan, Da Huang, Siamak Shakeri, David So, Andrew Dai, Yifeng Lu, Zhifeng Chen, Quoc Le, Claire Cui, James Laundon, Jeff Dean*

*Arxiv 2023*

This paper proposes a complex computation block for transformer-shaped networks called Brainformer. Compared with vanilla transformer block, it changes the way of stacking the feed-forward and attention layers, and also adds some sparsely gated feed-forward layers.


*2023-06-06*

#### [Direct Fact Retrieval from Knowledge Graphs without Entity Linking](https://arxiv.org/pdf/2305.12416.pdf)

*Jinheon Baek, Alham Fikri Aji, Jens Lehmann, Sung Ju Hwang*

*ACL 2023*

This paper introduces a simple fact retrieval model from KG, by embedding the KG into a dense space using a language model trained by the texts and facts. The fact with closest representation to the query is selected as answer in the prediction phase.


*2023-06-05*

#### [Prefix Propagation: Parameter-Efficient Tuning for Long Sequences](https://arxiv.org/pdf/2305.12086.pdf)

*Jonathan Li, Will Aitken, Rohan Bhambhoria, Xiaodan Zhu*

*ACL 2023*

This paper proposes a prefix propagation architecture for generative models by adjusting the representations of the training prompts (i.e., prefixes) in each transformer layer. It shows that prefix propagation approach is especially effective for long documents, while only requires half of the trainable parameters.


*2023-06-03*

#### [Interpretable Word Sense Representations via Definition Generation: The Case of Semantic Change Analysis](https://arxiv.org/pdf/2305.11993.pdf)

*Mario Giulianelli, Iris Luden, Raquel Fernandez, Andrey Kutuzov*

*ACL 2023*

This paper proposes to generate sense labels for a given target word based on a set of its usage examples and usage clusters. Such a generative model is especially useful for interpreting polysemous words. The obtained sense labels can be further used for semantic change analyses.


*2023-05-30*

#### [Massively Multi-Lingual Event Understanding: Extraction, Visualization, and Search](https://arxiv.org/pdf/2305.10561.pdf)

*Chris Jenkins, Shantanu Agarwal, Joel Barry, Steven Fincke, Elizabeth Boschee*

*ACL 2023*

This paper proposes a system for multi-lingual event extraction and search. It is based on XLM-RoBERTa for multi-lingual extraction. It is compatible with any event ontology that identifies a set of event types and argument roles. The system expects sentence-level English training data that identifies, for each event, one or more anchor spans and zero or more argument spans (with roles). It also supports the identification of time and location extraction.


*2023-05-20*

#### [Revisiting Relation Extraction in the era of Large Language Models](https://arxiv.org/pdf/2305.05003.pdf)

*Somin Wadhwa, Silvio Amir, Byron C. Wallace*

*ACL 2023*

This paper investigates the problem of relation extraction using LLMs by regarding it as a sequence-to-sequence generation task. The key findings include: (1) Few-shot prompting with GPT-3 achieves roughly equivalent performance to existing fully supervised models. (2) Flan-T5 is not as capable in the few-shot setting, but supervising and fine-tuning it with Chain-of-Thought style explanations (generated via GPT-3) yields SOTA results.


*2023-05-19*

#### [Unlearning Bias in Language Models by Partitioning Gradients](https://blender.cs.illinois.edu/paper/debiasinggradient2023.pdf)

*Charles Yu, Sullam Jeoung, Anish Kasi, Pengfei Yu, Heng Ji*

*ACL 2023*

This paper investigates the problem of biased information captured by PLMs. To mitigate the problem, it proposes a method of partitioned contrastive gradient unlearning to optimize the weights that contribute most to a specific domain of bias, using a first-order approximation based on the gradients of contrastive sentence pairs.


*2023-05-18*

#### [Open-Domain Hierarchical Event Schema Induction by Incremental Prompting and Verification](https://blender.cs.illinois.edu/paper/hierarchicalschema2023.pdf)

*Sha Li, Ruining Zhao, Manling Li, Heng Ji, Chris Callison-Burch, Jiawei Han*

*ACL 2023*

Instead of extracting event instances and learning to generalize their schema, this paper proposes to treat event schema as commonsense knowledge that can be derived from LLMs. It proposes an incremental prompting and verification method for doing so, which consists of 3 major steps, event skeleton construction, event expansion, and event-event relation verification.


*2023-05-17*

#### [Reasoning with Language Model Prompting: A Survey](https://arxiv.org/pdf/2212.09597.pdf)

*Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei Huang, Huajun Chen*

*ACL 2023*

Taxonomy, comparison and resources of reasoning methods based on language model prompting.


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
