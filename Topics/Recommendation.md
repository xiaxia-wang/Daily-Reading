





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

