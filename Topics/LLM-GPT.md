




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

