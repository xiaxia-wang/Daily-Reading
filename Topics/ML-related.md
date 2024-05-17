









*2024-05-17*

#### [What Knowledge Gets Distilled in Knowledge Distillation?](https://papers.nips.cc/paper_files/paper/2023/hash/2433fec2144ccf5fea1c9c5ebdbc3924-Abstract-Conference.html)

*Utkarsh Ojha, Yuheng Li, Anirudh Sundara Rajan, Yingyu Liang, Yong Jae Lee*

*NeurIPS 2023*

This paper investigates the process of knowledge distillation with the question: in what ways does the student become similar to the teacher, Does it start to localize objects in the same way? Does it get fooled by the same adversarial samples? Does its data invariance properties become similar? To answer them, it analyzes three popular knowledge distillation methods by (i) mimicking of class probabilities, (ii) mimicking of features at an intermediate layer, and (iii) features from the student and teacher for the same image constitute a positive pair, and those from different classes make up a negative pair. Some interesting findings: by simply mimicking the teacher’s output using method (i), the student can inherit many implicit properties of the teacher. It can gain the adversarial vulnerability that the teacher has. If the teacher is invariant to color, the student also improves its invariance to color. To understand why these properties get transferred without an explicit objective to do so, it studies the distillation process through a geometric lens, where the features from a teacher are regarded as relative positions of an instance (i.e., distances) from its decision boundary. Mimicking those features can therefore help the student inherit the decision boundary and (consequently) the implicit properties of the teacher.


*2024-05-16*

#### [Zero-shot causal learning](https://papers.nips.cc/paper_files/paper/2023/hash/15ddb1773510075ef44981cdb204330b-Abstract-Conference.html)

*Hamed Nilforoshan, Michael Moor, Yusuf Roohani, Yining Chen, Anja Šurina, Michihiro Yasunaga, Sara Oblak, Jure Leskovec*

*NeurIPS 2023*

This paper proposes a meta-learning framework for predicting how an intervention will causally affect a specific individual under a zero-shot context. It trains a simple MLP-based meta-model across thousands of tasks, each constructed by sampling an intervention, its recipients, and its nonrecipients. By leveraging both intervention information (e.g., a drug’s attributes) and individual features (e.g., a patient’s history), it is able to predict the personalized effects of novel interventions that do not exist at the time of training.


*2024-04-24*

#### [ExplainableFold: Understanding AlphaFold Prediction with Explainable AI](https://dl.acm.org/doi/10.1145/3580305.3599337)

*Juntao Tan, Yongfeng Zhang*

*KDD 2023*

This paper proposes a counterfactual learning framework inspired by biological principles to generate counterfactual explanations for protein structure prediction. It provides insights about which residue(s) of a sequence is crucial (or indecisive) to the protein’s structure and how certain changes on the residue(s) will change the structure, which helps to understand, e.g., what are the most impactful amino acids on the structure, and what are the most radical (or safe) substitutions when modifying a protein structure.


*2024-03-24*

#### [Faithful and Efficient Explanations for Neural Networks via Neural Tangent Kernel Surrogate Models](https://openreview.net/forum?id=yKksu38BpM)

*Andrew William Engel, Zhichao Wang, Natalie Frank, Ioana Dumitriu, Sutanay Choudhury, Anand Sarwate, Tony Chiang*

*ICLR 2024 Spotlight*

This paper investigates to use surrogate models to kernel general linear models. Specifically, it defines and evaluates new kernel functions for faithful approximation of an underlying neural network, and shows that approximate empirical neural tangent kernel surrogate models are consistently correlated to the underlying neural network across experiments.


*2024-03-20*

#### [Equivariant Matrix Function Neural Networks](https://openreview.net/forum?id=yrgQdA5NkI)

*Ilyes Batatia, Lars Leon Schaaf, Gabor Csanyi, Christoph Ortner, Felix Andreas Faber*

*ICLR 2024 Spotlight*

This paper proposes Matrix Function Neural Networks (MFNs), a new framework of Graph Neural Networks using the concept of Matrix Functions. They are good in modelling non-local interactions through analytic matrix equivariant functions, which are useful especially for scientific and physical data such as large conjugated molecules, metals, or amorphous materials.


*2024-03-19*

#### [Pre-Training and Fine-Tuning Generative Flow Networks](https://openreview.net/forum?id=ylhiMfpqkm)

*Ling Pan, Moksh Jain, Kanika Madan, Yoshua Bengio*

*ICLR 2024 Spotlight*

Generative Flow Networks (GFlowNets) are amortized samplers that learn stochastic policies to sequentially generate compositional objects from a given unnormalized reward distribution. This paper introduces an approach for reward-free pre-training of GFlowNets. By framing the training as a self-supervised problem, it proposes an outcome-conditioned GFlowNet (OC-GFN) that learns to explore the candidate space.


*2024-03-13*

#### [From Sparse to Soft Mixtures of Experts](https://openreview.net/forum?id=jxpsAj7ltE)

*Joan Puigcerver, Carlos Riquelme Ruiz, Basil Mustafa, Neil Houlsby*

*ICLR 2024 Spotlight*

While the router in sparse MoE layers learns to assign individual input tokens to each of the available slots, in soft MoE layers each slot is the result of a weighted average of all the input tokens. Learning to make discrete assignments introduces several optimization and implementation issues that soft MoE sidesteps. Therefore, this paper explores the soft MoE approach with the image classification task. The result suggests the superiority of soft MoE compared to sparse MoE.


*2024-03-11*

#### [Fantastic Gains and Where to Find Them: On the Existence and Prospect of General Knowledge Transfer between Any Pretrained Model](https://openreview.net/forum?id=m50eKHCttz)

*Karsten Roth, Lukas Thede, A. Sophia Koepke, Oriol Vinyals, Olivier Hénaff, Zeynep Akata*

*ICLR 2024 Spotlight*

This paper demonstrates the existence of complementary knowledge between different pretrained models, as evidenced by their distinct predictions on each sample, and further investigates the potential for transferring this knowledge without performance degradation. Given the limitations of existing knowledge distillation approaches, the authors frame the problem as a continual learning task and introduce a data partitioning scheme to maintain the useful knowledge of the student while learning from a teacher model. Empirical results show the effectiveness of the proposed approach across a variety of pretrained model pairs, including both strong and weak models.


*2024-03-04*

#### [On the Provable Advantage of Unsupervised Pretraining](https://openreview.net/forum?id=rmXXKxQpOR)

*Jiawei Ge, Shange Tang, Jianqing Fan, Chi Jin*

*ICLR 2024 Spotlight*

This paper studies a generic framework where the unsupervised representation learning task is specified by an abstract class of latent variable models Φ and the downstream task is specified by a class of prediction functions Ψ. It considers a natural approach of using Maximum Likelihood Estimation (MLE) for unsupervised pretraining and Empirical Risk Minimization (ERM) for learning downstream tasks. It proves that with some informative condition, unsupervised learning is able to achieve better excess risk than supervised learning, thus showing its advantage.


*2024-02-27*

#### [Predictive, scalable and interpretable knowledge tracing on structured domains](https://openreview.net/forum?id=NgaLU2fP5D)

*Hanqi Zhou, Robert Bamler, Charley M Wu, Álvaro Tejero-Cantero*

*ICLR 2024 Spotlight*

Knowledge tracing aims to track learning progress by predicting a learner’s performance on different knowledge components based on past learning interactions. This paper proposes a intelligent tutoring system that models both individual cognitive traits and the prerequisite structure of knowledge influence learning dynamics.


*2024-02-25*

#### [MT-Ranker: Reference-free machine translation evaluation by inter-system ranking](https://openreview.net/forum?id=Rry1SeSOQL)

*Ibraheem Muhammad Moosa, Rui Zhang, Wenpeng Yin*

*ICLR 2024 Spotlight*

This paper proposes a reference-free MT evaluation approach by formulating it as a pairwise ranking problem. Given the source sentence and a pair of translations, the system predicts which translation is better. In addition to the formulation, it shows that this paradigm demonstrates superior correlation with human judgments by merely using indirect supervision from natural language inference and weak supervision from synthetic data.


*2024-02-21*

#### ["What Data Benefits My Classifier?" Enhancing Model Performance and Interpretability through Influence-Based Data Selection](https://openreview.net/forum?id=HE9eUQlAvo)

*Anshuman Chhabra, Peizhao Li, Prasant Mohapatra, Hongfu Liu*

*ICLR 2024 Oral*

Unlike typical approaches focusing on model architecture or learning algorithms to improve classification performance, this paper explores the direction of estimating the influence from the perspective of the data feature space. Additionally, it proposes data selection approaches based on influence that enhance model utility, fairness, and robustness.


*2023-11-27*

#### [A Simple Interpretable Transformer for Fine-Grained Image Classification and Analysis](https://arxiv.org/abs/2311.04157)

*Dipanjyoti Paul, Arpita Chowdhury, Xinqi Xiong, Feng-Ju Chang, David Carlyn, Samuel Stevens, Kaiya Provost, Anuj Karpatne, Bryan Carstens, Daniel Rubenstein, Charles Stewart, Tanya Berger-Wolf, Yu Su, Wei-Lun Chao*

*Arxiv 2023*

This paper investigates a transformer-based proactive approach for interpretable image classification, which learns "class-specific" queries as input to the decoder, and uses the cross-attention weights as the interpretation.


*2023-11-10*

#### [Deep Integrated Explanations](https://dl.acm.org/doi/10.1145/3583780.3614836)

*Oren Barkan, Yehonatan Elisha, Jonathan Weill, Yuval Asher, Amit Eshel, Noam Koenigstein*

*CIKM 2023*

This paper introduces Deep Integrated Explanations (DIX), a comprehensive approach aimed at explaining vision models, which finds applicability across both CNN and ViT architectures. DIX employs integration over the internal model representations and their gradients, facilitating the extraction of insights from any activation (or attention) map within the network.


*2023-10-25*

#### [On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network](https://proceedings.mlr.press/v202/zhang23ad.html)

*Shijun Zhang, Jianfeng Lu, Hongkai Zhao*

*ICML 2023*

This paper investigates the expressive power of neural networks. Specifically, it demonstrates that the repeated composition of a single fixed-size ReLU network can approximate 1-Lipschitz continuous function on $[0, 1]^d$ with an arbitrarily small error.


*2023-10-10*

#### [Cross-Entropy Loss Functions: Theoretical Analysis and Applications](https://proceedings.mlr.press/v202/mao23b.html)

*Anqi Mao, Mehryar Mohri, Yutao Zhong*

*ICML 2023*

This paper analyzes the theoretical guarantee of cross-entropy loss functions when being used as a surrogate loss. Specifically, it analyzes the comp-sum loss family, including cross-entropy, generalized cross-entropy, the mean absolute error and other cross-entropy-like losses. It presents H-consistency bounds for these loss functions, and presents a new family of loss functions, i.e., smooth adversarial comp-sum losses which are derived from the comp-sum counterparts by adding a related smooth term.


*2023-10-07*

#### [A Closer Look at Few-shot Classification Again](https://proceedings.mlr.press/v202/luo23e.html)

*Xu Luo, Hao Wu, Ji Zhang, Lianli Gao, Jing Xu, Jingkuan Song*

*ICML 2023*

Few-shot classification consists of a training phase where a model is learned on a relatively large dataset and an adaptation phase where the learned model is adapted to previously-unseen tasks with limited labeled samples. This paper observes that the performance of few-shot classification models does not increase with the size of training datasets, instead, it is related to the number of training classes. It also proposes to split the training and adaptation process, and optimizes them in a disentangled manner.


*2023-10-02*

#### [Does a Neural Network Really Encode Symbolic Concepts?](https://proceedings.mlr.press/v202/li23at.html)

*Mingjie Li, Quanshi Zhang*

*ICML 2023*

This paper investigates the "real ability" of deep neural networks to encode concepts (i.e., as labels to be predicted for the input data sample). Specifically, it examines the DNNs from 4 perspectives: sparsity of the encoded concepts, transferability over different samples, transferability across different DNNs, and discrimination power of concepts.


*2023-09-28*

#### [Dividing and Conquering a BlackBox to a Mixture of Interpretable Models: Route, Interpret, Repeat](https://proceedings.mlr.press/v202/ghosh23c.html)

*Shantanu Ghosh, Ke Yu, Forough Arabshahi, Kayhan Batmanghelich*

*ICML 2023*

This paper proposes a post-hoc method to explain for a black-box ML model. Given such a model, it iteratively carves out a mixture of interpretable experts (MoIE) and a residual network, where each interpretable model handles a subset of examples and explains them in FOL.


*2023-09-22*

#### [Explainable Data-Driven Optimization: From Context to Decision and Back Again](https://proceedings.mlr.press/v202/forel23a.html)

*Alexandre Forel, Axel Parmentier, Thibaut Vidal*

*ICML 2023*

This paper investigates the problem of explaining data-driven optimization. It firstly define two classes of explanations, i.e., relative and absolute explanations. Then it presents algorithms to find relative and absolute explanations for the random forest and NN predictors.


*2023-09-21*

#### [Hyperparameters in Reinforcement Learning and How To Tune Them](https://proceedings.mlr.press/v202/eimer23a.html)

*Theresa Eimer, Marius Lindauer, Roberta Raileanu*

*ICML 2023*

This paper investigates the best practice for hyperparameter optimization in a reinforcement learning process. The results show that compared to tuning hyperparameters by hand, existing HPO tools are capable of producing better performing, more stable, and more easily comparable RL agents, while using fewer computational resources.


*2023-09-18*

#### [Towards Bridging the Gaps between the Right to Explanation and the Right to be Forgotten](https://proceedings.mlr.press/v202/krishna23a.html)

*Satyapriya Krishna, Jiaqi Ma, Himabindu Lakkaraju*

*ICML 2023*

This paper investigates the problem of generating explanations by the model under the circumstance of required data removal. Specifically, it first defines the problem of finding robust counterfactual explanations in the presence of training data removal, and provides an algorithm to solve it.


*2023-09-17*

#### [TabDDPM: Modelling Tabular Data with Diffusion Models](https://proceedings.mlr.press/v202/kotelnikov23a.html)

*Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, Artem Babenko*

*ICML 2023*

This paper introduces a diffusion model that can be applied to tabular data with mixed data types. To sum up, it uses the multinomial diffusion to model the categorical and binary features, and the Gaussian diffusion to model the numerical ones.


*2023-09-13*

#### [Learnability and Algorithm for Continual Learning](https://proceedings.mlr.press/v202/kim23x.html)

*Gyuhak Kim, Changnan Xiao, Tatsuya Konishi, Bing Liu*

*ICML 2023*

Class Incremental Learning (CIL) learns a sequence of tasks consisting of disjoint sets of concepts or classes. At any time, a single model is built that can be applied to predict or classify test instances of any classes learned thus far without providing any task related information for each test instance. This paper shows CIL is learnable.


*2023-09-12*

#### [PAC Prediction Sets for Large Language Models of Code](https://proceedings.mlr.press/v202/khakhar23a.html)

*Adam Khakhar, Stephen Mell, Osbert Bastani*

*ICML 2023*

This paper investigates the task of constructing PAC sets for code generation. It introduces a notion of partial programs as prediction sets for code generation, and proposes an algorithm for construction PAC prediction sets under this setting.


*2023-09-11*

#### [Trainability, Expressivity and Interpretability in Gated Neural ODEs](https://proceedings.mlr.press/v202/kim23b.html)

*Timothy Doyeon Kim, Tankut Can, Kamesh Krishnamurthy*

*ICML 2023*

Neural ordinary differential equations (nODEs) is a class of dynamical models with a velocity field parametrized by a deep neural network, which can potentially implement more complex computations in lower dimensions than classical RNNs. This paper introduces a gating interaction for nODEs, designs a new measure of expressivity related to the network capacity, and demonstrates its superiority.


*2023-09-10*

#### [On the Relationship Between Explanation and Prediction: A Causal View](https://proceedings.mlr.press/v202/karimi23a.html)

*Amir-Hossein Karimi, Krikamol Muandet, Simon Kornblith, Bernhard Schölkopf, Been Kim*

*ICML 2023*

This paper investigates the relationship between the model prediction (Y) and the explanation (E) with a casual influence view. It measures the treatment effect when intervening on their casual ancestors, i.e., the hyperparameters and the inputs used to generate saliency-based Es or Ys.


*2023-09-09*

#### [Leveraging Proxy of Training Data for Test-Time Adaptation](https://proceedings.mlr.press/v202/kang23a.html)

*Juwon Kang, Nayeong Kim, Donghyeon Kwon, Jungseul Ok, Suha Kwak*

*ICML 2023*

Test-time adaptation (TTA) is the task of adapting a trained model to an arbitrary test domain using unlabeled input data on-the-fly during testing. This paper proposes two lightweight proxies of the training data and a TTA method that fully exploits them.


*2023-09-03*

#### [Learning Unnormalized Statistical Models via Compositional Optimization](https://proceedings.mlr.press/v202/jiang23g.html)

*Wei Jiang, Jiayu Qin, Lingyu Wu, Changyou Chen, Tianbao Yang, Lijun Zhang*

*ICML 2023*

Existing noise contrastive estimation methods for learning unnormalized statistical models suffer from bad performance and slow convergence. This paper proposes a direct approach for optimizing negative log-likelyhood of models by converting it to a stochastic compositional optimization (SCO) problem.


*2023-08-31*

#### [On the Impact of Knowledge Distillation for Model Interpretability](https://proceedings.mlr.press/v202/han23b.html)

*Hyeongrok Han, Siwon Kim, Hyun-Soo Choi, Sungroh Yoon*

*ICML 2023*

This paper argues that knowledge distillation of models can not only enhance the performance but also improve the interpretability. It measures the model's interpretability by the number of concept detectors introduced in network dissection, and attributes the improvement of interpretability to the class similarity transferred from the teacher model to the student model.


*2023-08-28*

#### [Do We Really Need Complicated Model Architectures For Temporal Networks?](https://openreview.net/pdf?id=ayPPc0SyLv1)

*Weilin Cong, Si Zhang, Jian Kang, Baichuan Yuan, Hao Wu, Xin Zhou, Hanghang Tong, Mehrdad Mahdavi*

*ICLR 2023*

This paper proposes a simple but effective network architecture for temporal graph learning. Specifically, it only includes a link encoder based on MLP, a node encoder of mean-pooling of neighbors, and an MLP-based link classifier for link predictions. It demonstrates good performance in practice.


*2023-08-27*

#### [On the duality between contrastive and non-contrastive self-supervised learning](https://openreview.net/pdf?id=kDEL91Dufpa)

*Quentin Garrido, Yubei Chen, Adrien Bardes, Laurent Najman, Yann LeCun*

*ICLR 2023*

This paper investigates the theoretical and empirical similarities of the so-called contrastive and non-constrastive learning methods, especially the covariance regularization-based non-contrastive methods. It proposes to unify both kinds of approaches with a pair of contrastive and non-contrastive criteria based on F-norm and embedding normalizations.


*2023-08-26*

#### [In-context Reinforcement Learning with Algorithm Distillation](https://openreview.net/pdf?id=hy0a5MMPUv)

*Michael Laskin, Luyu Wang, Junhyuk Oh, Emilio Parisotto, Stephen Spencer, Richie Steigerwald, DJ Strouse, Steven Stenberg Hansen, Angelos Filos, Ethan A. Brooks, Maxime Gazeau, Himanshu Sahni, Satinder Singh, Volodymyr Mnih*

*ICLR 2023*

This paper proposes a method for simulating reinforcement learning using in-context learning by regarding it as a sequential prediction problem. A dataset of learning histories is generated by a source RL algorithm, and then a causal transformer is trained by autoregressively predicting actions given their preceding learning histories as context.


*2023-08-23*

#### [No Reason for No Supervision: Improved Generalization in Supervised Models](https://openreview.net/pdf?id=3Y5Uhf5KgGK)

*Mert Bülent Sariyildiz, Yannis Kalantidis, Karteek Alahari, Diane Larlus*

*ICLR 2023*

This paper proposes a supervised training setup that incorporates multi-crop data augmentation and an expendable projector that can produce models with favorable performance both on the training task and transfer tasks.


*2023-08-19*

#### [Relational Attention: Generalizing Transformers for Graph-Structured Tasks](https://openreview.net/pdf?id=cFuMmbWiN6)

*Cameron Diao, Ricky Loynd*

*ICLR 2023*

This paper proposes an adapted version of transformer blocks for relational graphs, named Relational Transformer. In addition to accepting node vectors representing entity features (as do all transformers), RT also accepts edge vectors representing relation features, which may include edge-presence flags from an adjacency matrix. (But overall it operators on a fully-connected graph, unconstrained by any input adjacency matrix.)


*2023-08-18*

#### [Effects of Graph Convolutions in Multi-layer Networks](https://openreview.net/pdf?id=P-73JPgRs0R)

*Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath*

*ICLR 2023*

This paper theoretically explores the effects of graph convolution in multi-layer networks with a node classification problem. It shows that a single graph convolution enables a multi-layer network to classify the nodes with a larger threshold (by a factor) of the distance between the means of node features. In a graph with a higher density, two graph convolutions will further improve the factor.


*2023-08-17*

#### [Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://openreview.net/pdf?id=8aHzds2uUyB)

*Rajkumar Ramamurthy, Prithviraj Ammanabrolu, Kianté Brantley, Jack Hessel, Rafet Sifa, Christian Bauckhage, Hannaneh Hajishirzi, Yejin Choi*

*ICLR 2023*

This paper proposes to solve NLP tasks in a reinforcement learning view. It uses an open-sourced library for optimizing language generators with RL, and presents a benchmark consisting of 6 natural language generation tasks with reward functions from human preference. Besides, it also provides an RL algorithm named NLPO that learns to effectively reduce the combinatorial action space in language generation.


*2023-08-14*

#### [Sparse Mixture-of-Experts are Domain Generalizable Learners](https://openreview.net/pdf?id=RecZ9nB9Q4)

*Bo Li, Yifei Shen, Jingkang Yang, Yezhen Wang, Jiawei Ren, Tong Che, Jun Zhang, Ziwei Liu*

*ICLR 2023*

To address the problem of domain generalization, this paper proposes a model built upon vision transformers, in which the network's robustness to distribution shifts is characterized by the architecture's alignment with the correlations in the dataset.


*2023-08-12*

#### [Learning with Logical Constraints but without Shortcut Satisfaction](https://openreview.net/pdf?id=M2unceRvqhh)

*Zenan Li, Zehua Liu, Yuan Yao, Jingwei Xu, Taolue Chen, Xiaoxing Ma, Jian Lü*

*ICLR 2023*

Noticing that existing methods with loss function including logical constraints may be bypassed with shortcuts thus not fully exploiting intrinsic knowledge, this paper designs a new loss function for logical constraints. It introduces an additional random variable for the logical constraint indicating its satisfaction degree, and formulates it as a distributional loss which is compatible with the neural network’s original training loss under a variational framework.


*2023-08-11*

#### [A Survey on the Explainability of Supervised Machine Learning](https://jair.org/index.php/jair/article/view/12228)

*Nadia Burkart, Marco F. Huber*

*JAIR 2021*

This paper discusses the essential definitions, an overview of different principles and methodologies of explainable supervised machine learning.


*2023-08-07*

#### [What learning algorithm is in-context learning? Investigations with linear models](https://openreview.net/pdf?id=0g0X4H8yN4I)

*Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, Denny Zhou*

*ICLR 2023*

This paper investigates the hypothesis that transformer-based in-context learners implement standard learning algorithms implicitly by encoding smaller models in their activations, and updating these implicit models as new examples appear in the context. The hypothesis is evaluated with the problem of linear regression, and demonstrated with several source of evidence.


*2023-08-05*

#### [Encoding Recurrence into Transformers](https://openreview.net/pdf?id=7YfHla7IxBJ)

*Feiqing Huang, Kexin Lu, Yuxi Cai, Zhen Qin, Yanwen Fang, Guangjian Tian, Guodong Li*

*ICLR 2023*

This paper proposes to equivalently replace a RNN layer with a set of simple RNNs, and further by a multi-head self-attention block. It further proposes a new module named Self-Attention with Recurrence, which can incorporate the recurrent dynamics into a transformer.


*2023-07-03*

#### [Normalizing Flow-based Neural Process for Few-Shot Knowledge Graph Completion](https://arxiv.org/abs/2304.08183)

*Linhao Luo, Reza Haffari, Yuan Fang Li, Shirui Pan*

*SIGIR 2023*

Neural Processes (NPs) combine the stochastic process and neural networks to define a distribution over prediction functions with limited observed data. Normalizing flows (NFs) [36] employ a sequence of bijective mapping functions to transform a simple distribution into a complex target distribution. This paper applies the NPs and normalized flows in an encoder-decoder model for KGC.


*2023-05-21*

#### [KGA: A General Machine Unlearning Framework Based on Knowledge Gap Alignment](https://arxiv.org/pdf/2305.06535.pdf)

*Lingzhi Wang, Tong Chen, Wei Yuan, Xingshan Zeng, Kam-Fai Wong, Hongzhi Yin*

*ACL 2023*

Machine unlearning refers to the ability of the learned model to forget information about specific training data as if they never existed in the training set. This paper follows the idea of approximate unlearning, whose goal is to forget the data to be forgotten while maintaining the performance. To achieve this, it optimizes the model to have similar behaviors on the data to be forgotten as unseen data, while maintaining the performance on the rest of data.
