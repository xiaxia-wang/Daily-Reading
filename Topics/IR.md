



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
