










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

