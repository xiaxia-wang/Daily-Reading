







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

