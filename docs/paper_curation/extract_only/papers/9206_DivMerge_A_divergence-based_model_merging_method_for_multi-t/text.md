## DivMerge: A divergence-based model merging method for multi-tasking

Brahim Touayouch1,2 Loïc Fosse1,3 Géraldine Damnati1 Gwénolé Lecorvé1 1Orange Research, Lannion, France 2École polytechnique, Institut polytechnique de Paris, Palaiseau, France 3CNRS, LIS, Aix Marseille Université, France Contact: first.last@orange.com

# arXiv:2509.02108v3[cs.LG]12Sep2025

### Abstract

Multi-task learning (MTL) is often achieved by merging datasets before fine-tuning, but the growing availability of fine-tuned models has led to new approaches such as model merging via task arithmetic. A major challenge in this setting is task interference, which worsens as the number of tasks increases. We propose a method that merges models trained on different tasks into a single model, maintaining strong performance across all tasks. Our approach leverages Jensen-Shannon divergence to guide the merging process without requiring additional labelled data, and automatically balances task importance. Unlike existing methods, our approach remains robust as the number of tasks grows and consistently outperforms prior work.

### 1 Introduction

Current transformer based language models (Vaswani et al., 2017; Brown et al., 2020) have demonstrated remarkable efficiency in handling a wide range of tasks within a single unified architecture. This led to the creation of the so-called Instruct models (Shengyu et al., 2023), which are now state-of-the-art in almost all NLP tasks. However, creating these models is very expensive and relies on large data collections and models. With this goal of reducing costs in mind, a paradigm has been revived: model merging. This paradigm rooted in ensemble methods consists in combining parameters of models specialized on specific tasks in order to create a new one that has new properties such as multi-tasking — to only cite this one. This paradigm is all the more interesting given the large number of specialized models for specific tasks available on collaborative platforms such as HuggingFace (Wolf et al., 2019). Thus, since the work of Ilharco et al. (2022) a whole series of studies on model merging have been published, proposing many different methods to combine model parameters in order

to produce a multi-tasking type model (Goddard et al., 2024; Yang et al., 2024b). The motivation behind the creation of these different methods is to answer the following question: how can models trained on different tasks be combined without losing performance on each task? In the literature this problem is also mentioned as interference between models. However, the wide variety of methods available and the lack of consensus in the production of methods seem to suggest that this question remains unanswered. In this paper, we propose a new model merging method that allows us to create a so-called multi-task model. Our method formally addresses the interference problem mentioned above, which can adversely affect the model resulting from the combination. More formally, the contributions of this paper are the following three points:

- 1. A novel merging method. We propose a merging method that is grounded in information theory (Cover and Thomas, 1991). This method automatically learns how to combine parameters of different models in a data driven but reference free setting to achieve the multi-task property. We formally demonstrate that our method is linked to classical multi-task learning (Caruana,

1997) and is constructed in a way that it minimizes interferences between models by respecting weight disentanglement defined by OrtizJimenez et al. (2023).

- 2. Better performances. On a classical set-up with only two tasks to merge we illustrate that our method is the best one in average among classical state of the art methods.
- 3. Better scalability. We empirically show that our method scales better when the number of tasks to merge increases. This clearly illustrates that our method is more effective at limiting the possible effects of interference between the different models we wish to combine.

### 2 Related Work

Multi-Task Learning. Multi-task learning (Caruana, 1997) refers to methods that produce a model capable of solving several tasks. Classical multitask learning generally consists in combining datasets from multiple tasks and training a model on this union, which is justified by classical results such as Stein’s paradox (Stein, 1956). This philosophy is at the core of the production of most current models such as T5 (Raffel et al., 2020) and more recently Instruct type models (Shengyu et al., 2023). Although the results of these models are now considered state-of-the-art in NLP, this approach is not without its issues. Indeed, several studies (Baxter, 2000; Ben-David and Schuller, 2003; Fifty et al., 2021; Standley et al., 2020; Jeong and Yoon, 2025; Maurer et al., 2016) show that the choice of tasks is important in order to limit effects such as interference (Yu et al., 2020): some tasks may have a negative effect on others. In this study we propose a novel method, grounded in Information Theory (Cover and Thomas, 1991), to produce a multi-task model. Our method is theoretically connected to the classical multi-task learning, and model merging.

Model Merging. While the idea of combining several models has its roots in ensemble methods originally developed for variance reduction (Dietterich, 2000), model merging refers to methods that combine models in the parameter space to produce a new model that has new properties that go beyond variance reduction. While most applications of model merging are described by Yang et al. (2024c), we recall here the most popular ones. One objective of model merging is of course variance reduction (i.e. better generalization) (Jin et al., 2022; Matena and Raffel, 2022; Ferret, 2025; Izmailov et al., 2018; Wortsman et al., 2022), with the goal of producing a more reliable model on a single target task by merging models trained on the same task with different initializations or hyperparameter settings. The most popular objective in model merging is the multi-task one (Ilharco et al., 2022; Yang et al., 2024d; Yu et al., 2024; Pfeiffer et al., 2021): by merging models specialized on different tasks, the resulting model should achieve good performance on all tasks. Zhou et al. (2024) and (Ortiz-Jimenez et al., 2023) provide theoretical insights into why model merging can work well for this multi-task objective. Another objective is task unlearning (Ilharco et al., 2022; Kuo et al.,

2025; Kim et al., 2024): by merging certain models (via addition and negation), we seek to remove or "forget" specific components. Finally, modular learning (Ballard, 1987; Pfeiffer et al., 2023; Chronopoulou et al., 2023) is an interesting but less explored objective, which consists of creating a model that performs well on a task by merging models trained on other (possibly unrelated) tasks, leveraging the notion of transfer between tasks. In this study, we propose a new model merging method with the goal of multi-task learning.

Merging Methods. While the literature offers a wide range of merging methods, some of them stand out with interesting results and properties. Since this study is focused on multi-task learning, we give a quick overview of methods designed for this. The most straightforward and simple approach is model averaging (Wortsman et al., 2022), also known as isotropic merging, which simply consists in taking the uniform average of the models’ parameters. In (Ilharco et al., 2022) the notion of task vector is introduced which is the shift in the parameter space from a pre-trained model to a finetuned one1. This concept has led to the framework of task arithmetic (TA) which is now extensively used (Ortiz-Jimenez et al., 2023) and has proven its efficiency across a wide range of applications. To enhance TA, SLERP (Spherical Linear Interpolation) (Jang et al., 2024) proposes a merging method that preserves certain geometric properties of the task vectors, thereby helping to mitigate interference as described earlier. This notion of interference is also at the heart of a wide range of methods that attempt to address this problem by following a two-step process: first, task vectors are modified using techniques such as masking or singular value decomposition (SVD) (Wang et al., 2024; Yadav et al., 2023; Stewart, 1993); second, the preprocessed task vectors are interpolated to produce the merged model. This is the case of methods such as TIES (Yadav et al., 2023), AdaMerging++ (Yang et al., 2024d), or DARE (Yu et al., 2024). Among these methods, AdaMerging stands out as it is data-driven, i.e. this method automatically learns the best way to combine each model’s parameters. It requires a learning algorithm that uses data from the different tasks. In this study, we propose a novel multi-task merging method rooted in task arithmetic and, like AdaMerging, leverages reference-free data-driven optimization.

1We provide a more formal definition in Sec. 3

### 3 Formalism

Notations. Random variables are denoted by capital letters (e.g., X), their spaces by calligraphic letters (e.g., X), and elements by lowercase letters (e.g., x ∈ X). P(X) is the set of probability measures on X, and P(Y|X) the set of conditional probabilities on Y given X. For X ∈ X, PX ∈ P(X) is its law, and SX ⊂ X its support. A task t is a probability measure PXt,Yt ∈ P(X ×Y), following the formalism of (Fosse et al., 2025). For sake of simplicity, we hypothesis that all tasks share the same space X × Y, and that X = Y, which is true in most generative tasks: both inputs and outputs are texts. We note a language model with parameters θ with M(·|·;θ) ∈ P(Y|X). For task t, the specialized model on t is M(·|·;θt) or Mt(·|·).

- Remark 1. We identify a language model with a conditional probability: M(·|x;θ) is a probability distribution over texts. For sake of simplicity we will refer to a language model by M(x;θ) when the model is given the data x as input. In other words a language model is a communication channel.

##### 3.1 Task vectors and model merging

Task vectors have become essential objects in modern machine learning, given the vast number of finetuned models available on collaborative platforms such as HuggingFace (Wolf et al., 2019). These objects were first defined in (Ilharco et al., 2022) as follows: if we denote by θ0 ∈ Rd the parameters of a pre-trained model (d being the number of parameters in the model), and by θt ∈ Rd the parameters of the same model after fine-tuning on a task t, the task vector is given by,

τt ≜ θt − θ0 ∈ Rd. (1)

The main approach in model merging is to combine such task vectors to create a new one that has new properties. In the following, we denote any task vector-based merging algorithm as,

f (θ0,{τt},Γ) ∈ Rd, (2)

where θ0 are the pre-trained model parameters, {τt} is the set of task vectors to be merged, and Γ is the set of parameters for the merging method. For example, task arithmetic (Ilharco et al., 2022; Ortiz-Jimenez et al., 2023) (TA) can be formulated as,

f (θ0,{τt},Γ) = θ0 +

t

Γt × τt, (3)

where Γt are real values (possibly negative). In the following we will refer to the TA method as

ΦΓn, with n the number of task vectors, and Γ the merging coefficients. Estimating the parameters Γ can be challenging, computationally intensive, and it strongly depends on the objective we aim to achieve with model merging (e.g., multi-task learning, modularity, task unlearning, etc.). In this study, we focus on the multi-task setup in model merging: the merged model must be capable of solving all the tasks on which the individual component models have been fine-tuned. In this set-up, as stated in (Ortiz-Jimenez et al., 2023), one of the main properties we hope to achieve is weight disentanglement, which is defined in Definition 1.

Definition 1 (Weight Disentanglement (Ortiz-Jimenez et al., 2023)). Let M(.;θ) be a model parametrized by θ. Consider a set of tasks, {(Xt,Yt), t ∈ T }, and their corresponding task vectors {τt, t ∈ T } relatively to the model M. If tasks have non overlapping supports i.e. SXt ∩ SXt′ = ∅ for all t ̸= t′, we say that a merging method f satisfies weight disentanglement iff,

M (x;f (θ0,{τt},Γ)) =

M(x;θ0 + τt) if x ∈ SXt, M(x;θ0) if x ∈/ Tt=1 SXt.

In other words, a merging method satisfies Definition 1 if adding τt does not affect model’s output outside the corresponding task support SXt. A merging method satisfying weight disentanglement assures that the performance on all the merged tasks will be preserved, assuring thus good behaviour in terms of multi-tasking. In this study we will focus on TA merging algorithm denoted as ΦΓn. We choose this method because, as explained in App. E, most current model merging approaches are based on task arithmetic. However, as we will demonstrate, our theoretical framework is not limited to task arithmetic, and most of our results can be extended to more general methods.

##### 3.2 Our Method

Based on the Kullback and Leibler (KL) (Kullback and Leibler, 1951) and Jensen-Shannon (JS) (Wong and You, 1985) divergences, we propose a method to automatically estimate the merging coefficients in Eq. 3 in a way that forces weight disentanglement (c.f. Definition 1), which translates into preserving performance (as much as possible) on the different task components. First, we recall some basic properties of the KL divergence.

Given two discrete probability distributions µ and ν over some discrete space I we have,

µ(i) ν(i)

KL(µ∥ν) ≜

.

µ(i)log

i∈I

In this definition, we refer to µ as the reference distribution. From this, we also define the JS divergence, which is a symmetric version of the KL divergence,

#### JS(µ,ν) ≜ 21 KL µ µ+2ν + KL ν µ+2ν .

Both divergences are positive real numbers that quantify a notion of distance between the measures µ and ν (lower values indicate closer distributions). We can extend these definitions to define divergence between language models. Given two LMs, M1 and M2, and a reference input dataset X, we define the divergence D (either KL or JS) between these models as,

###### DX(M1∥M2) ≜ EX D M1(· | X)∥M2(· | X) .

This is one possible expression for the divergence between transition probabilities. Since, in NLP tasks, the model generates sequences of text, we can naturally extend the formula above to sequencelevel distributions. See Sec. B.1 for more details. Since this work focuses on task vectors and model parameters, in the following, we will denote a language model (LM) by its parameters, i.e. θt ≡ Mt. Given a divergence D (either KL or JS), we provide in Eq. 4 an optimization problem to automatically find coefficients in task arithmetic.

Γ∗ ≜ arg min

Γ

n

DXt θt ∥ΦΓn . (4)

t=1

- Remark 2. (Nielsen, 2020) provides intuition for the solution of Eq. 4. In the case where D = JS, the solution corresponds to the probabilistic centroid of the different output distributions generated by the

models {θt}, which is a concept firstly introduced in the framework of Information Geometry (Amari and Nagaoka, 2000). In App. E we give more details about properties of some centroids in the case of model merging.

- Remark 3. Eq. 4 is defined for the task arithmetic

merging function. However, we can replace ΦΓn with any merging method f(θ0,{τt},Γ) (not only task arithmetic).

Despite the simplicity of the optimisation problem in Eq. 4, we propose several results that connect with weight disentanglement and multi-task learning theory.

- Proposition 1. For either DX = KL or DX = JS, the objective function defined in Eq. 4 has a minimum value of 0 if and only if the merging method

satisfies weight disentanglement around θ0 on the merged tasks.

The proof of this result is given in Sec. A.1. Therefore, the optimization problem in Eq. 4 admits a minimum value of 0 iff there exists a choice of merging parameters Γ that respects the weight disentanglement property, giving added weight to this property. Furthermore, the following proposition establishes an interesting link between Eq. 4 and classical multi-task learning setups (i.e., when datasets are merged before fine-tuning), further emphasizing the relevance of the objective function defined in Eq. 4.

- Proposition 2. For D = KL, the optimization problem defined in Eq. 4 is a Moment projection (M-projection) approximation of the multi-task objective when tasks are merged before training.

We give the proof of Proposition 2 in Sec. A.2. This proposition is noteworthy as it links our method to the classical multi-task learning objective considered so far as the strongest baseline. In other words our method is an approximation of the classical multi-task learning objective. Moreover, the demonstration of Proposition 2 indicates that the effectiveness of this approximation depends on the performance of the individual fine-tuned models. Finally, we give an illustration of our procedure for merging two models in Figure 1, and we provide the general algorithm for our approach in Algorithm 1.

### 4 Experimental Protocol

Models and datasets. As stated in Sec. 3, our proposed merging method is designed for generative language models. Therefore, we evaluate our approach on two architectures: a decoderonly architecture using the Qwen2.5-0.5B (Yang et al., 2024a) model and an encoder decoder architecture using the T5-Base (Raffel et al., 2020) model. We apply our method on both classification and generative tasks. For classification tasks, we used the GLUE benchmark (Wang et al., 2019), as is common in recent studies. We fine-tuned

[Figure 1]

Figure 1: Illustration of the divergence-based model merging method. This figure shows the merging loss associated to one task to be merged, which is the task t. Our method consists in doing this procedure for every task t and, as it is written on the left side, to sum all the associated loss. For more details, we refer to Algorithm 1.

Description Data Prompt

Is the following sentence linguistically acceptable or unacceptable in english?

Sodium is a little too peppy for me to want to try mixing and water in a teacup.

Data

Answer with acceptable or unacceptable. Answer:

Labels

Answer unacceptable<|endoftext|>

Table 1: Data example for the fine-tuning on the GLUE Benchmark. This example is derived from the CoLA dataset. Fine-tuning is done in a completion only fashion (SFT) after the pattern Answer:

Qwen2.5-0.5B on 7 classification tasks, resulting in 7 distinct checkpoints for our experiments. Finetuning was performed using standard supervised fine-tuning (SFT), where the model generates the classification labels. An example of the fine-tuning data is shown in Table 1. For generative tasks, we used the T5-Base model and existing checkpoints on several tasks, including IMDB (Maas et al., 2011), QASC (Khot et al., 2020), SQuAD (Rajpurkar et al., 2016), and CommonGen (Lin et al., 2020)2. This setup allows us to evaluate our method in a more realistic scenario, where we do not control the fine-tuning process and only have access to the fine-tuned models. We demonstrate that our method is effective in this setting as well.

Evaluation Metrics. To assess the quality of a merging method, we define a metric to quantify how the merged model performs compared to each fine-tuned model. This metric is the Average Normalized Performance (ANP), defined as:

1 n

ANP ≜

n

PERF(f (θ0,{τi},Γ);t) PERF(θt;t)

, (5)

t=1

2All checkpoints are available here

where n is the number of merged tasks, PERF(f (θ0,{τi},Γ);t) is the performance of the merged model on task t using the merging method f, and PERF(θt;t) is the performance of the fine-tuned model on task t. In our work, PERF is measured by classical accuracy for classification tasks, and by the ROUGE1 score (Lin, 2004) for generation tasks. Each performance metric is computed on a separate test set. This metric quantifies the performance of a specific merging experiment. In practice, multiple merging experiments can be conducted (for example, by sampling different sets of k tasks to merge). In such cases, we compute the ANP metric for each merging experiment and report the average. It is worth noting that the ANP metric is well suited for the multitasking setup, as it automatically provides a value relative to the baselines.3

Merging method. As stated previously, in this study we focus on the task arithmetic merging function ΦΓn. We define two versions of this merging method. The first, described in Eq. 3, is what we call Task Level (TL) since one coefficient Γi is given for each task. The granularity of the method can be further refined: since the models we use are deep neural networks organized in layers, we can define a different parameter Γil for each layer for task l, resulting in n × L merging coefficients (where L is the number of layers). We refer to this method as the Layer Level (LL) approach. In the following, we present results for both methods and always specify which one is used. Moreover, our method requires input data for each task; that is, when merging tasks i and j, we need data from Xi and Xj. For this purpose, we use the validation set of each task.

3If the merging method verifies Definition 1 ⇒ ANP = 1.

Algorithm 1 Model Merging via Divergence-Based Optimization. In this procedure, for a sequence y and a model M, Logits(y, M) denotes the logits (soft probs) of the sequence y given by the model M. From a technical point of view, this is only a forward pass of y through M. In Sec. B.1 we propose more details about the computation of divergences.

Require: : {Xt | t ∈ T } fΓ (Merging method) θ0 (Pre-trained model) {θt | t ∈ T } (Fine-tuned models) // Get logits of the data by generation for t ∈ T do

for x ∼ Xt do yˆtx ← {y1, . . . , ym, eos} ∼ M(x; θt) ℓ(t, x) ← Logits(ˆytx, M(x, θt))

end for end for // Compute coefficients

Γi ← |T1| ∀i (Init. of coefficients) for each epoch do

for b ∼ ∪tXt do // Batch sampling for x ∈ b do

ℓ(f, x) ← Logits(ˆytx, M(x, fΓ)) end for LΓ ← 0 for each t ∈ T do

for x ∈ b do // Choose right task // Compute pointwise Loss LΓ ← LΓ + D(ℓ(t, x)∥ℓ(f, x))

end for end for Γ ← Γ − ∇ΓLΓ // Gradient update

end for end for return Γ

Baselines. Since we propose a new merging method, we compare our method with existing ones that are widely used such as model averaging (Wortsman et al., 2022), Multi-SLERP (Goddard et al., 2024), TIES (Yadav et al., 2023), and AdaMerging (Yang et al., 2024d) in its two variants: Task Level and Layer Level. For TIES, we followed the recommended recipe from (Yadav et al., 2023). For Multi-SLERP, the weights associated to each tasks were set to n1. For optimization-based methods (AdaMerging and ours), for each method we used the same hyper-parameters across all merging experiments, with a batch size of 4×n for each iteration. In Table 9 we provide more training details.

### 5 Results

##### 5.1 Divergence justification

As stated in Sec. 3, our method is based on computing divergences between the merged model and the various fine-tuned ones. Before applying this method, we decided to illustrate an interesting re-

Task (θi) KL JS

CoLA 0.812 0.925 SST-2 0.920 0.887 QQP 0.313 0.340 QNLI 0.774 0.796 MNLI 0.947 0.990 RTE 0.769 0.915 MRPC 0.877 0.875

Avg. 0.773 0.818

Table 2: Spearman’s correlation between {−DX

(θi∥θj), ∀j ∈ T } and {PERF(θj,i), ∀j ∈ T } for all i ∈ T (The negation sign is added to have positive correlations). The "Avg." row reports the mean correlation across all tasks.

i

sult: divergence and performance are well correlated. Given a set of tasks T , for each pair of tasks (i,j) ∈ T × T , we compute DXi(θi∥θj) on a development set, and the performance of the model θj on task i on a test set, denoted as PERF(θj,i). Then, for all tasks i we compute the correlation between {−DXi(θi∥θj), ∀j ∈ T } and {PERF(θj,i), ∀j ∈ T }. We report Spearman’s correlations in Table 2 for classification tasks. We clearly observe that we obtain high correlations indicating that KL and JS are interesting proxies for performance: if DXi(θi∥θj) is low, it may suggest that PERF(θj,i) will be high (without stating that this is a causation relation). We also observe that the JS divergence achieves the highest correlation; therefore, unless stated otherwise, we use the JS divergence criterion in our experiments. Although the JS divergence consistently outperforms KL divergence, the difference is not always significant. Since our method involves minimizing

n i=1 DXi(θi∥ΦΓn), it thus can be viewed as aiming for a merging model that performs well on each task. This aligns directly with Proposition 2, which states that our method is an approximation of the classical multi-task learning objective. For more details, we refer to Sec. B.2.

##### 5.2 Effective Model Merging

We first illustrate our method in a pairwise model merging setup i.e. we merge only two tasks at a time and compute ANP as defined in Eq. 5. We computed this metric for every possible pairwise merging experiment and for each task types (classification and generation). For example, for classification tasks, we have 7 distinct tasks, meaning that we can perform 72 = 21 pairwise merging experiments and thus compute 21 distinct values

Merging Method Classif. Gen. Model Averaging 88.48 (± 3.17) 94.38 (± 2.6) Multi-SLERP 91.54 (± 2.98) 76.39 (± 21.04) TIES 94.06 (± 1.81) 95.53 (± 4.44) TL Adamerging 93.62 (± 2.53) 93.42 (± 10.08) LL Adamerging 94.06 (± 2.95) 83.20 (± 9.94) TL KL (ours) 97.68 (± 1.94) 93.97 (± 3.46) LL KL (ours) 99.16 (± 0.50) 97.50 (± 1.73) TL JS (ours) 97.73 (± 2.01) 97.29 (± 1.94) LL JS (ours) 99.18 (± 0.51) 98.93 (± 1.05)

Table 3: Average ANP (%) for various merging methods across GLUE and T5 task pairs. The best results per benchmark are boldfaced, and the second best is underlined. See Table 6 and Table 7 in the appendix for more detailed results.

of ANP (for generation tasks we have 42 = 6).

- Table 3 presents the average ANP metric across all pairwise merging experiment. We can clearly observe that our method achieves the best average performance. Moreover we can see that the Layer Level variant outperforms the Task Level variant, which is expected since the former has a greater number of merging coefficients (one for each layer of each task), giving a higher degree of granularity.

##### 5.3 Robustness to Number of Tasks

A major limitation of existing merging methods, as noted in (Yadav et al., 2023), is their lack of robustness as the number of merged models increases. We thus decided to empirically assess whether our method is robust to an increasing number of merged tasks by varying this number in our experiments and testing all possible combinations of tasks. For example, in the classification setup, we have 7 different tasks. We test our merging method by merging between 2 and 7 tasks. Moreover, to ensure the reliability of our conclusions, we follow the same procedure as before: for each number of merged tasks, we perform all possible merging combinations. For instance, when merging three tasks, we consider all possible combinations, i.e., 73 = 35 merging experiments. We then compute the average ANP metric. In this analysis, we focus on the use of the JS divergence, as it demonstrated slightly better results. Figure 2 shows the average ANP metric across all possible merging experiments as a function of the number of tasks, for both the classification setup (Figure 2a) and the generation setup (Figure 2b). First, we observe that regardless of the used method, increasing the number of tasks generally leads to a degrada-

100

90

80

Avg.ANP

70

60

Model Averaging

AdaMerging Task Level

JS Task Level

JS Layer Level

TIES

AdaMerging Layer Level

Multi-SLERP

50

2 3 4 5 6 7

# Tasks

(a) Classification

100

90

80

Avg.ANP

70

60

50

Model Averaging

AdaMerging Task Level

JS Task Level

JS Layer Level

TIES

AdaMerging Layer Level

Multi-SLERP

40

2 3 4

# Tasks

(b) Generation

Figure 2: Evolution of the average ANP metric as a function of the number of merged tasks. For each number of tasks k on the x-axis, several merging experiments were conducted ( nk in total), and we report the 95% confidence interval.

tion in the average ANP, which is consistent with the fact that task interference becomes more apparent. We can also observe that, regardless of the number of merged tasks, our method (both tasklevel and layer-level) consistently provides better results, with curves that remain higher throughout the graph. Moreover, the drop in performance as the number of tasks increases is less pronounced for our method, illustrating its robustness with respect to the number of tasks.

In addition to the average ANP, Figure 2 presents confidence intervals (CI) across experiments for various tasks with a fixed number of tasks to merge. Our method demonstrates strong stability, both in classification and generation tasks. For generation tasks, some SOTA methods show large, overlapping confidence intervals, so we re-

###### Merging Method 2 Tasks 3 Tasks

Model Averaging ±2.60 ±2.96 Multi-SLERP ±21.04 ±27.97 TIES ±4.44 ±19.01 TL Adamerging ±10.08 ±23.94 LL Adamerging ±9.94 ±19.03

TL JS (ours) ±1.94 ±5.37 LL JS (ours) ±1.05 ±1.09

- Table 4: Confidence interval (CI) margins for different merging methods when merging 2 or 3 tasks. For each method, the margin is measured across multiple merging experiments performed on different task combinations.

port CI margins in Table 4. Notably, our method exhibits greater stability compared to Adamerging, another optimization-based approach.

##### 5.4 Method Behaviour Analysis

In this section, we analyse our method in greater depth by examining its convergence behaviour. We focus here exclusively on classification tasks.

Performance Convergence. Since our method is data-driven, it requires a training procedure, which we described in Algorithm 1. We decided to further investigate its training dynamics by studying the evolution of the ANP metric over different training iterations. Here, we focus on pairwise merging, and Figure 3 shows the evolution of the ANP metric across training iterations for both the Task Level and Layer Level variants. First we can notice once again that Task level and Layer level provide similar results which is in line with our previous findings. Then, in all cases, the merging process converges smoothly without signs of over-fitting (i.e., a sudden drop in performance), indicating that the proposed methods are effective and stable, consistently merging task-specific representations over training iterations.

Dataset Size Influence. As described in Sec. 3, when merging a set of task vectors {τi}, we require some data derived from {Xi}. A natural and important question is how much data is needed for our method to achieve strong performance. To investigate this, we studied the evolution of the ANP metric as a function of the amount of data used by our method. For this experiment, we focused on classification tasks and considered three merging scenarios: (CoLA, SST-2), (QNLI, MNLI), and (RTE, MRPC). In Figure 4, we plot the evolution of the ANP metric. We observe that our method

100

95

90

Avg.ANP

85

80

sst2 + qnli (Task Level)

sst2 + qnli (Layer Level)

mnli + mrpc (Task Level)

mnli + mrpc (Layer Level)

sst2 + rte (Task Level)

sst2 + rte (Layer Level)

75

cola + sst2 (Task Level)

cola + sst2 (Layer Level)

0 20 40 60 80

Iterations

- Figure 3: ANP metric of merged task pairs with Task Level and Layer Level JS Divergence as a function of training iterations.

0 50 100 150 200 250 300 350 400

Dataset Size

- 75

- 76

- 77

- 78

- 79

- 80

- 81

- 82

- 83

- 84

- 85

- 86

- 87

- 88

- 89

- 90

- 91

- 92

- 93

- 94

- 95

- 96

- 97

- 98

- 99

- 100

Avg.ANP

Model Averaging

Multi-SLERP

TIES

JS Task Level

JS Task Level 95% CI

JS Layer Level

JS Layer Level 95% CI

- Figure 4: Impact of dataset size on the performance of our approach compared to data-free baselines. The average ANP is computed across three task pairs.

outperforms state-of-the-art methods with as few as 25 samples, which corresponds to only 0.4% of the training corpus used for fine-tuning the merged models and 5% of the validation dataset.

### 6 Conclusion

In this work, we propose a new, data-driven but reference free, merging method which consists in finding the probabilistic centroid of fine-tuned models in order to produce a multi-task model. After showing that, theoretically, our method is directly linked to multi-task learning and the concept of weight disentanglement, we demonstrate, empirically, that our method consistently outperforms most state-of-the-art methods on a pairwise model merging set-up. Furthermore, we show that our method seems to better handle interference issues considering that it is the best one when the number of merged tasks increases. Finally, we show that our method has high training stability and requires a relatively small amount of data to work.

### Limitations

Despite interesting results, our method present several limitations that we tempt to address here.

Other fine-tuning methods. Our method has been extensively tested when the specialized models were constructed using full-finetuning. In this setup, the task vectors are sparse, and as a consequence, the interference problem is more limited. However, in low rank adaptation (LoRA) (Hu et al., 2022) fine-tuning, task vectors (i.e., LoRA matrices) affect the task arithmetic paradigm and are responsible for significant performance loss when merging. A limitation of our work is that we have not experimented within this constrained setup.

Dataset influence. Our method assumes that for each task t, we have access to a sample of the distribution PXt corresponding to the input data for task t. However, in some setups, we may not have access to such a distribution, but only to an approximation of it (PX˜

). We have not addressed this case here. On the other hand, we propose an initial theoretical analysis of such a case in Sec. A.3. We believe that pushing in this direction will provide more robust results for model merging.

t

### References

Shun-ichi Amari and Hiroshi Nagaoka. 2000. Methods of information geometry, volume 191. American Mathematical Soc.

Dana H Ballard. 1987. Modular learning in neural networks. In Proceedings of the sixth National conference on Artificial intelligence-Volume 1, pages 279–284.

J. Baxter. 2000. A Model of Inductive Bias Learning. Journal of Artificial Intelligence Research, 12:149– 198.

Shai Ben-David and Reba Schuller. 2003. Exploiting Task Relatedness for Multiple Task Learning. In Gerhard Goos, Juris Hartmanis, Jan Van Leeuwen, Bernhard Schölkopf, and Manfred K. Warmuth, editors, Learning Theory and Kernel Machines, volume 2777, pages 567–580. Springer Berlin Heidelberg, Berlin, Heidelberg.

- Yochai Blau and Tomer Michaeli. 2018. The perceptiondistortion tradeoff. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6228–6237.
- Yochai Blau and Tomer Michaeli. 2019. Rethinking lossy compression: The rate-distortion-perception tradeoff. In International Conference on Machine Learning, pages 675–685. PMLR.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Rich Caruana. 1997. Multitask Learning. Machine Learning, 28(1):41–75.

Alexandra Chronopoulou, Jonas Pfeiffer, Joshua Maynez, Xinyi Wang, Sebastian Ruder, and Priyanka Agrawal. 2023. Language and Task Arithmetic with Parameter-Efficient Layers for Zero-Shot Summarization. Preprint, arXiv:2311.09344.

T. M. Cover and J. A. Thomas. 1991. Elements of Information Theory. John Wiley & Sons, Inc.

Imre Csiszár. 1975. I-divergence geometry of probability distributions and minimization problems. The annals of probability, pages 146–158.

Thomas G. Dietterich. 2000. Ensemble methods in machine learning. In Multiple Classifier Systems, pages 1–15, Berlin, Heidelberg. Springer Berlin Heidelberg.

Olivier Ferret. 2025. Projeter pour mieux fusionner: une histoire de bandit et de lit.

Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. 2021. Efficiently Identifying Task Groupings for Multi-Task Learning. In Advances in Neural Information Processing Systems,

volume 34, pages 27503–27516. Curran Associates, Inc.

Loïc Fosse, Frédéric Béchet, Benoît Favre, Géraldine Damnati, Gwénolé Lecorvé, Maxime Darrin, Philippe Formont, and Pablo Piantanida. 2025. Statistical deficiency for task inclusion estimation. arXiv preprint arXiv:2503.05491.

Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vlad Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz. 2024. Arcee’s mergekit: A toolkit for merging large language models. arXiv preprint arXiv:2403.13257.

Karsten Grove and Hermann Karcher. 1973. How to conjugate c 1-close group actions. Mathematische Zeitschrift, 132(1):11–20.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2022. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089.

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. 2018. Averaging weights leads to wider optima and better generalization. arXiv preprint arXiv:1803.05407.

Young Kyun Jang, Dat Huynh, Ashish Shah, Wen-Kai Chen, and Ser-Nam Lim. 2024. Spherical linear interpolation and text-anchoring for zero-shot composed image retrieval. Preprint, arXiv:2405.00571.

Wooseong Jeong and Kuk-Jin Yoon. 2025. Selective task group updates for multi-task optimization. arXiv preprint arXiv:2502.11986.

Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. 2022. Dataless knowledge fusion by merging weights of language models. arXiv preprint arXiv:2212.09849.

Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen, and Ashish Sabharwal. 2020. Qasc: A dataset for question answering via sentence composition. arXiv:1910.11473v2.

Hyoseo Kim, Dongyoon Han, and Junsuk Choe. 2024. Negmerge: Consensual weight negation for strong machine unlearning. arXiv preprint arXiv:2410.05583.

Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Solomon Kullback and Richard A Leibler. 1951. On information and sufficiency. The annals of mathematical statistics, 22(1):79–86.

Kevin Kuo, Amrith Setlur, Kartik Srinivas, Aditi Raghunathan, and Virginia Smith. 2025. Exact unlearning of finetuning data via model merging at scale. arXiv preprint arXiv:2504.04626.

Bill Yuchen Lin, Wangchunshu Zhou, Ming Shen, Pei Zhou, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. 2020. CommonGen: A constrained text generation challenge for generative commonsense reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1823–1840, Online. Association for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics.

Michael S Matena and Colin A Raffel. 2022. Merging models with fisher-weighted averaging. Advances in Neural Information Processing Systems, 35:17703– 17716.

Andreas Maurer, Massimiliano Pontil, and Bernardino Romera-Paredes. 2016. The benefit of multitask representation learning. Journal of Machine Learning Research, 17(81):1–32.

Frank Nielsen. 2020. On a generalization of the jensen– shannon divergence and the jensen–shannon centroid. Entropy, 22(2):221.

Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard. 2023. Task arithmetic in the tangent space: Improved editing of pre-trained models. Advances in Neural Information Processing Systems, 36:66727–66754.

Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. 2021. AdapterFusion: Non-Destructive Task Composition for Transfer Learning. Preprint, arXiv:2005.00247.

Jonas Pfeiffer, Sebastian Ruder, Ivan Vuli´c, and Edoardo Maria Ponti. 2023. Modular Deep Learning. Preprint, arXiv:2302.11529.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of

the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.

Henry Scheffé. 1947. A Useful Convergence Theorem for Probability Distributions. The Annals of Mathematical Statistics, 18(3):434–438.

Zhang Shengyu, Dong Linfeng, Li Xiaoya, Zhang Sen, Sun Xiaofei, Wang Shuhe, Li Jiwei, Runyi Hu, Zhang Tianwei, Fei Wu, and 1 others. 2023. Instruction tuning for large language models: A survey. arXiv preprint arXiv:2308.10792.

Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. 2020. Which tasks should be learned together in multi-task learning? In International conference on machine learning, pages 9120–9132. PMLR.

Charles Stein. 1956. Inadmissibility of the usual estimator for the mean of a multivariate normal distribution. In Proceedings of the third Berkeley symposium on mathematical statistics and probability, volume 1: Contributions to the theory of statistics, volume 3, pages 197–207. University of California Press.

Gilbert W Stewart. 1993. On the early history of the singular value decomposition. SIAM review, 35(4):551– 566.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019. Glue: A multi-task benchmark and analysis platform for natural language understanding. Preprint, arXiv:1804.07461.

Ke Wang, Nikolaos Dimitriadis, Guillermo OrtizJimenez, François Fleuret, and Pascal Frossard. 2024. Localizing task information for improved model merging and compression. arXiv preprint arXiv:2405.07813.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and 1 others. 2019. Huggingface’s transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771.

Andrew KC Wong and Manlai You. 1985. Entropy and distance of random graphs with application to structural pattern recognition. IEEE transactions on pattern analysis and machine intelligence, (5):599– 609.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt.

2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. Preprint, arXiv:2203.05482.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. 2023. Ties-merging: Resolving interference when merging models. Preprint, arXiv:2306.01708.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 22 others. 2024a. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. 2024b. Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities. Preprint, arXiv:2408.07666.

Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. 2024c. Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities. arXiv preprint arXiv:2408.07666.

Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, and Dacheng Tao. 2024d. Adamerging: Adaptive model merging for multi-task learning. Preprint, arXiv:2310.02575.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2024. Language models are super mario: Absorbing abilities from homologous models as a free lunch. Preprint, arXiv:2311.03099.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient surgery for multi-task learning. Advances in neural information processing systems, 33:5824– 5836.

Yuefeng Zhang. 2023. A rate-distortion-classification approach for lossy image compression. Digital Signal Processing, 141:104163.

Yuyan Zhou, Liang Song, Bingning Wang, and Weipeng Chen. 2024. Metagpt: Merging large language models using model exclusive task arithmetic. arXiv preprint arXiv:2406.11385.

### A Theoretical results

##### A.1 Proof of Proposition 1

Proof. Since the divergences we use are non-negative, we have the following equivalence:

n

DXt θt ∥ΦΓn = 0 ⇔ ∀t, DXt θt ∥ΦΓn = 0. Moreover, by the properties of the KL and JS divergences, we have:

t=1

DXt θt ∥ΦΓn = 0 ⇔ ∀x ∈ SXt, M(x;θt) = M(x;ΦΓn),

where the last equality is understood in the sense of equality of measures. By transitivity, we obtain:

n

DXt θt ∥ΦΓn = 0 ⇔ ∀t, ∀x ∈ SXt, M(x;θt) = M(x;ΦΓn), which concludes the proof.

t=1

| |
|---|

- Remark 4. In this demonstration, we stated that this was due thanks to some properties of the KL or JS divergence. However, we have the same result if we use any f-divergence, any divergence than can be expressed as following,

dµ dν

Df(µ∥ν) = f

dν,

which is of course the case of the Jensen Shannon and the Kullback ones. In fact this proof is valid for any divergence D which satisfies the following property,

Df(µ∥ν) = 0 ⇔ µ = ν

##### A.2 Proof of Proposition 2

Definition 2 (Multi task objective). Let {(Xt,Yt) | t ∈ T } be a set of tasks, H the cross-entropy loss function, and M(·;θ) a model parameterized by θ. We define the multi-task loss function as follows:

1 |T | t∈T

LMT(θ) ≜

H PYt|Xt,M(Xt;θ) ,

Lemma 1. Let (Xt,Yt) be a task, H the cross-entropy loss function, and M(·;θ) a model parameterized by θ. Then, the following relation holds:

H(PYt|Xt,M(Xt;θ)) = H(Yt|Xt) + KL(PYt|Xt∥M(Xt;θ)), where H denotes Shannon’s entropy.

We now provide the proof of Proposition 2: Proof. By hypothesis, we have

H(PYt|Xt,M(Xt;θ)). By Lemma 1, this is equivalent to

θt = arg min

θ

KL(PYt|Xt∥M(Xt;θ)).

θt = arg min

θ

Thus, M(Xt;θt) is the moment projection (M-projection) (Csiszár, 1975) of PYt|Xt onto the set {M(Xt;θ) | θ ∈ Rd}. Based on this, we define the M-projection multi-task objective as follows:

1 |T | t∈T

LMMT(θ) ≜

H (M(Xt;θt),M(Xt;θ)).

Again, by Lemma 1, we have

LMMT(θ) = arg min

KL(M(Xt;θt)∥M(Xt;θ)). This concludes the proof.

arg min

θ

θ

| |
|---|

- A.3 Distribution shift The objective function we proposed in Eq. 4 supposed that for each task we have access to the input data distribution denoted as PXt. However, in some cases we can have no access to PXt but to an

approximation of it, denoted as PX˜

. For example, we have a model trained on sentiment analysis and we do not have access the true data. We can thus use existing data for such task as an approximation. We show in the following that we can in fact control the behaviour of our method with respect to the quality of the approximation.

t

- Proposition 3. Considering a set of approximated distribution PX˜

, for D = JS, our method will converge in a uniform way with PX˜

t

. Proof. We recall that in Eq. 4 for a given task t we have the following,

t

D θt(.|x)∥ΦΓn(.|x) PXt(dx). Then,

DXt θt ∥ΦΓn =

x

DXt θt ∥ΦΓn − DX˜

t

θt ∥ΦΓn =

⩽

D θt(.|x)∥ΦΓn(.|x) (PXt(dx) − PX˜

(dx))

t

x

D θt(.|x)∥ΦΓn(.|x) PXt(dx) − PX˜

(dx)

t

x

If we use the Jensen Shannon divergence we then have,

JSXt θt ∥ΦΓn − JSX˜

t

θt ∥ΦΓn ⩽ log(2)

Then by Scheffe’s Theorem (Scheffé, 1947), we have:

x

PXt(dx) − PX˜

(dx)

t

θt ∥ΦΓn | ⩽ 2log(2)TV(PXt,PX˜

|JSXt θt ∥ΦΓn − JSX˜

), where TV stands for total variation distance. Then we have,

t

t

t

DXt θt ∥ΦΓn − DX˜

t

θt ∥ΦΓn ⩽ 2log(2)

t

TV(PXt,PX˜

),

t

which concludes the proof.

| |
|---|

- Remark 5. In Proposition 3 we state that the convergences is uniform in the sense that if the approximations we have converge uniformally to the true distribution i.e. in the sense of the total variation, then we have a convergence of our objective function.

### B Divergence Details

- B.1 Divergence Between Models on Sequence Outputs In this section, we give more details on the computation of a divergence D between autoregressive models. In this study, we recall that we defined the divergence between two LMs M1 and M2 as following,

###### DX(M1∥M2) ≜ EX D M1(· | X)∥M2(· | X) .

Since a model prompted with input x generates a sequence of symbols in an auto-regressive set-up, we propose here to detail more the way divergence is computed between models. For each input x in X, we generate the next tokens in a greedy manner using the reference model M1, while storing the softened logits (probability distributions) of the generated tokens at each step, until the end-of-sequence (EOS) token is produced. We then append the generated token sequence y to the original input x, forming

[Figure 2]

colasst2qqpqnlimnlirtemrpc TargetTask

0.0000 0.8130 0.7687 0.3687 0.5292 0.4525 0.7547

3.0

0.4731 0.0000 0.6529 0.4850 0.5909 0.8322 0.5886

2.5

0.5077 0.3708 0.0000 0.5293 0.9362 1.2963 0.3393

2.0

- 0.6821 0.6900 1.2664 0.0000 0.8675 1.1930 1.2186

2.7304 3.2225 2.9976 3.0643 0.0000 2.0251 3.2550

- 1.0662 1.2483 1.2769 0.9011 0.8865 0.0000 1.4644

1.5

1.0

0.5

0.2438 0.2322 0.1801 0.2809 0.2307 0.3320 0.0000

0.0

cola sst2 qqp qnli mnli rte mrpc Reference Task

(a) KL

[Figure 3]

colasst2qqpqnlimnlirtemrpc TargetTask

0.0000 0.1605 0.1815 0.1024 0.1392 0.1258 0.1777

0.4

0.1448 0.0000 0.1795 0.1468 0.1751 0.2380 0.1662

0.1466 0.1111 0.0000 0.1511 0.2574 0.3443 0.0953

0.3

0.1786 0.1831 0.2625 0.0000 0.2162 0.3102 0.2575

0.2

0.4689 0.4966 0.4921 0.4810 0.0000 0.3974 0.4909

0.2343 0.2592 0.2688 0.1977 0.2114 0.0000 0.2395

0.1

0.0694 0.0649 0.0490 0.0686 0.0654 0.0941 0.0000

0.0

cola sst2 qqp qnli mnli rte mrpc Reference Task

(b) JS

- Figure 5: DX

(θi∥θj) values for different divergences. (i corresponds to the row index, while j corresponds to the column index.)

i

the extended sequence x + y. Next, we forward propagate this extended sequence through the second model M2 once, and obtain the probability distributions of the tokens generated by M1. This allows us to compare the token-level distributions of M1 and M2 on the same generated sequence. Using this procedure, we can measure the divergence DX(M1∥M2) over sequences generated by M1. In a more formal way we propose to compute divergence in a recurrent. Let x ∈ X be a sequence, t ∈ N be an index, and we suppose that we can sample

D M1(· | x,y<t)∥M2(· | x,y<t) . Then the final divergence is given by,

Tx

1 |X| x

1 Tx

D M1(· | x,y<t)∥M2(· | x,y<t) ,

t=1

Where for each x, Tx is the maximum number of token to generate before the end of sequence token (it can be viewed as some sort of stopping time). For greater accuracy, this calculation should be performed as follows,

Tx

1 |X| x

1 Tx

D M1(· | x,y<t)∥M2(· | x,y<t) ,

t=1 y<t∼M1(·|x)

where the sum over y<t ∼ M1(· | x), would correspond to a sampling procedure of sequences of size less than t with respect to the model M1(· | x). In our work, for sake of simplicity we stick to some greedy procedure.

- B.2 Correlation between Divergence Variants and Model Relatedness. In Sec. 5.1, we proposed an experiment to investigate links between the divergences we used and the

notion of performance on the different tasks. In Figure 5, we propose the heat map defined by DXi(θi∥θj), and in Table 5, we proposed the matrix of values PERF(θj,i). Correlations computed in Table 2 in this study, correspond to correlations compute between rows of Figure 5 and rows of Table 5.

### C Additional experiments

##### C.1 Task Vectors Cosine Similarities

As is well known in the model merging literature, and more specifically within the task arithmetic framework, cosine similarities between task vectors are typically close to zero. This indicates that the

Eval Dataset → CoLA SST-2 QQP QNLI MNLI RTE MRPC Model ↓

CoLA 82.20 77.80 50.60 44.60 8.80 28.88 70.00 SST-2 44.00 92.80 71.20 37.00 10.20 26.71 66.75 QQP 33.00 50.20 84.60 40.80 8.60 28.88 69.75 QNLI 46.20 76.40 39.20 86.40 9.20 43.68 69.00 MNLI 33.80 61.40 65.40 45.00 78.00 28.16 67.50 RTE 34.80 48.20 63.60 50.20 14.00 75.81 69.00 MRPC 32.80 57.40 66.20 41.20 11.60 45.85 85.00

- Table 5: Accuracies (%) of each model checkpoint (rows) evaluated on the seven GLUE tasks (columns). Each row corresponds to a model fine-tuned on a specific task. The highest accuracy for each task is highlighted in bold, and corresponds each time to the specialized model.

Classification

Generation

1.0

1.0

[Figure 4]

[Figure 5]

cola

1.000 0.051 0.019 0.016 0.022 0.016 0.015

imdb

1.000 -0.002 0.004 0.006

0.8

0.8

sst2

0.051 1.000 0.033 0.031 0.023 0.006 0.031

qqp

0.019 0.033 1.000 0.042 0.020 0.002 0.089

qasc

-0.002 1.000 0.004 0.004

0.6

0.6

qnli

0.016 0.031 0.042 1.000 0.021 0.031 0.031

0.4

0.4

squad

0.004 0.004 1.000 0.010

mnli

0.022 0.023 0.020 0.021 1.000 0.031 0.022

rte

0.016 0.006 0.002 0.031 0.031 1.000 0.014

0.2

0.2

common_gen

0.006 0.004 0.010 1.000

mrpc

0.015 0.031 0.089 0.031 0.022 0.014 1.000

0.0

0.0

imdb qasc squad common_gen

cola sst2 qqp qnli mnli rte mrpc

- Figure 6: Cosine similarity matrices of task vectors between different tasks. Left: Similarity between GLUE benchmark tasks (CoLA, SST2, QQP, QNLI, MNLI, RTE, MRPC). Right: Similarity between diverse generative tasks tasks (IMDB, QASC, SQUAD, commonGen). Lower similarity values indicate greater orthogonality between task vectors, suggesting less interference when merging models fine-tuned on these tasks.

tasks are sufficiently disentangled and can be effectively merged using task arithmetic methods. In Figure 6, we present the cosine similarity matrices of the task vectors used in our experiments.

##### C.2 Details on Figure 2

On Table 6, we propose the values that are plotted on Figure 2a, and on Table 7 we propose the values that are plotted on Figure 2b.

##### C.3 Parameter convergence

In Sec. 5 we provided an analysis of the convergence of our method by displaying the evolution of our loss function through training iterations and we concluded that our method smoothly converges to an local optimum value. We decided to go further and analyse the evolution of the coefficients associated to each task. As a recall we used the framework of task arithmetic and in this framework the merged model is given by the following,

Γi × τi,

θ0 +

i

and we are here interested into the evolution of the coefficients Γi. In Figure 7, we provide the evolution of Γ1 (left) and Γ2 (right) through training iterations, on different pairwise merging set-up on the benchmark GLUE. We can mainly observe that the dynamic of our method is also smooth in the coefficients Γi, with an interesting convergence of the parameters. We can also go further by observing in some settings

###### # Tasks Model Averaging Multi-SLERP TIES Task Level Layer Level

Adamerging KL (ours) JS (ours) Adamerging KL (ours) JS (ours)

- 2 93.89 94.20 96.02 91.16 98.18 98.28 92.41 98.85 98.85

- 3 89.10 92.18 94.08 90.86 97.40 97.39 90.26 98.42 98.26

- 4 75.12 79.31 79.37 78.73 80.12 79.21 78.47 95.20 95.19

- 5 60.92 66.86 73.61 66.84 83.82 85.37 64.93 95.60 95.50

- 6 56.98 70.48 67.23 67.97 83.45 84.45 62.85 93.11 93.42

- 7 60.51 68.89 68.39 67.26 83.45 84.70 63.05 92.53 93.06

Average 72.75 78.65 79.78 77.14 87.73 88.23 75.33 95.62 95.71

- Table 6: ANP for merged tasks obtained via different merging methods. Values are normalized as percentages, with separate evaluations for KL and JS Divergence variants.

# Tasks Model Averaging Multi-SLERP TIES Task Level Layer Level

Adamerging Forward (ours) JS (ours) Adamerging Forward (ours) JS (ours)

- 2 97.92 98.96 98.43 99.74 96.34 98.96 100.00 98.96 99.48

- 3 82.62 67.89 87.87 79.92 92.99 96.36 89.26 95.91 97.79

- 4 53.38 64.85 56.42 60.88 87.52 91.75 83.15 94.97 97.44

Average 77.97 77.23 80.91 80.18 92.28 95.69 90.80 96.61 98.24

- Table 7: ANP for merged tasks obtained via different merging methods. Values are normalized as percentages, with separate evaluations for KL and JS Divergence variants.

that the values of Γ1 and Γ2 seem to be independent meaning that when merging two tasks the merging coefficient associated to one task seems to strongly depend on the task itself and not the task with which we merge. To better support this fact, we decided to add a visualization. In the framework of task arithmetic, each merging experiment can be represented by a point in an euclidean space defined by the following coordinates (Γ1,Γ2,...,Γn)t. In the case of pairwise merging experiments, these points are in a plan and we decided to visualize this plan on Figure 8, for classification tasks, and Figure 9 for generative tasks. On these figures, we can mainly observe that we have different scenarios. For tasks such as QNLI, the factor associated with the QNLI task seems not to depend on the other tasks, while for some other tasks such as MRPC and CoLA we have another scenario where the value of the coefficient associated to the task seems to depend on the value associated to the other tasks. This seems to be an interesting observation, to be considered alongside the fact that some tasks may be independent, while others may have a statistical dependency, i.e., completing one task may have a positive or negative impact on another.

### D Training Settings

##### D.1 Data details

As explained in Sec. 4, we used the GLUE Benchmark (Wang et al., 2019) to perform our experiments. We recall on Table 8 the description of tasks from this benchmark.

CoLA detection of the linguistic acceptability of a sentence MNLI natural language inference MRPC paraphrase detection QNLI question answering converted into natural language inference QQP detection of equivalence between questions RTE natural language inference SST2 sentiment analysis

Table 8: Description of the GLUE Benchmark

###### First Task Coefficient Evolution

###### Second Task Coefficient Evolution

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

|Task Pairs<br><br>cola+mrpc<br><br>cola+qnli sst2+qqp qnli+rte<br><br>sst2+rte<br><br>sst2+mnli<br><br>qqp+mrpc<br><br>qnli+mnli<br><br>qqp+qnli mnli+rte cola+qqp<br><br>cola+rte<br><br>cola+mnli<br><br>sst2+mrpc<br><br>qqp+rte<br><br>qnli+mrpc<br><br>qqp+mnli sst2+qnli rte+mrpc mnli+mrpc<br><br>cola+sst2|
|---|

0.9

0.9

0.8

0.8

0.7

CoefficientValue

CoefficientValue

0.7

0.6

0.6

0.5

0.5

0.4

0.4

0.3

0 20 40 60 80

0 20 40 60 80

Iteration

Iteration

- Figure 7: Evolution of the task coefficients across training iterations. The first graph shows the coefficient assigned to the first task in each task pair (as indicated in the legend), while the second graph shows the coefficient assigned to the second task.

##### Method Level Batch Size Epochs Dataset Size Scheduler (LR) Init. Param.

KL/JS (Classif) Task 4× # tasks 4 200 1e−2 0.5 KL/JS (Classif) Layer 4× # tasks 4 400 1e−2 0.5 KL/JS (Gen.) Task 4× # tasks 4 200 1e−2 0.5 KL/JS (Gen.) Layer 4× # tasks 4 400 1e−2 0.5 AdaMerging (Classif) Task 4× # tasks 5 200 1e−3 0.5 AdaMerging (Classif) Layer 4× # tasks 5 400 1e−3 0.5 AdaMerging (Gen.) Task 4× # tasks 5 200 1e−2 0.5 AdaMerging (Gen.) Layer 4× # tasks 5 400 1e−2 0.5

Table 9: Training configurations.

##### D.2 Training details

We propose in Table 9 training hyper-parameters we chose for our method, as well as for the Adamerging one since it also requires a training procedure. All the optimizations were done using the Adam optimizer (Kingma and Ba, 2014) with default moments hyper-parameters. From a practical standpoint, the hyperparameters we choose, both for the Adamerging method and for our own, allow us to maximize multi-task performance on evaluation sets. While the choice of hyperparameters is relatively sensitive in the Adamerging method, our method appears to be more robust in terms of hyperparameter selection. The other methods we used that had hyper-parameters were TIES and Multi-SLERP, for which we basically used the recommended recipes:

- • TIES: We used the recommended recipe from (Yadav et al., 2023), with λ = 1 and a mask rate of 0.2 (i.e., 80% zeros in the mask).
- • Multi-SLERP: The weights were set to 1/N, where N is the number of tasks.

### E Everything is task arithmetic

Many different merging methods have emerged in the landscape of machine learning. Among them, task arithmetic (Ilharco et al., 2022) is probably the most widely used. As a reminder, the merging function in

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
|SST2| | | | |

1.0

1.0

1.0

PairedTaskFactor

###### PairedTaskFactor

PairedTaskFactor

0.8

0.8

0.8

0.6

0.6

0.6

0.4

0.4

0.4

###### COLA

###### QQP

0.2

0.2

0.2

0.2 0.4 0.6 0.8 1.0 1.2 COLA Factor

0.2 0.4 0.6 0.8 1.0 1.2 SST2 Factor

0.2 0.4 0.6 0.8 1.0 1.2 QQP Factor

1.2

1.2

1.2

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
|QNLI| | | | |

1.0

1.0

1.0

###### PairedTaskFactor

PairedTaskFactor

PairedTaskFactor

0.8

0.8

0.8

0.6

0.6

0.6

0.4

0.4

0.4

###### MNLI

###### RTE

0.2

0.2

0.2

0.2 0.4 0.6 0.8 1.0 1.2 QNLI Factor

0.2 0.4 0.6 0.8 1.0 1.2 MNLI Factor

0.2 0.4 0.6 0.8 1.0 1.2 RTE Factor

1.2

1.0

PairedTaskFactor

0.8

0.6

0.4

###### MRPC

0.2

0.2 0.4 0.6 0.8 1.0 1.2 MRPC Factor

COLA

SST2 QQP QNLI MNLI RTE MRPC

| | | |
|---|---|---|
| | | |

- Figure 8: Visualization of coefficient values for a fixed reference task versus coefficient values for the remaining GLUE tasks. Each subplot corresponds to a different reference task.

the case of task arithmetic is defined as follows:

f (θ0,{τt},Γ) = θ0 +

t

Γt × τt.

An interesting question that naturally arises is the following: Given a merging method g (θ0,{τt},∆), can we find coefficients Γ such that g (θ0,{τt},∆) = f (θ0,{τt},Γ)? If this is true, then we can state that

min

Γ

t

DXt (θt ∥f(θ0,{τt},Γ)) ⩽ min

∆

t

DXt (θt ∥g(θ0,{τt},∆)).

This inequality would highlight the strength of our method, as it encompasses the entire range of task arithmetic. As stated in Remark 3, our method can be applied to any hyper-parameter differentiable merging approach. As pointed out in (Goddard et al., 2024), a wide range of merging methods are based on task arithmetic, with the main differences lying in the estimation of the merging coefficients. In the following, we provide an analysis of other merging methods to show that they can be expressed as model merging. This demonstrates that our method has the potential to achieve better results.

SLERP. Spherical linear interpolation (SLERP) (Wortsman et al., 2022) is a classical method used to combine vectors on a spherical manifold. For this method, we introduce a hyperparameter t ∈ [0,1], and define SLERP as follows:

sin((1 − t)Ω) sinΩ

τ1 ∥τ1∥

sin(tΩ) sinΩ

τ2 ∥τ2∥

f(θ0,{τ1,τ2},t) ≜ θ0 +

+

,

where Ω is the angle between τ1 and τ2.

Fisher Weight Averaging. The Fisher weight averaging method, introduced in (Matena and Raffel, 2022), is a merging technique that reduces to task arithmetic in the case of linear interpolation between specialized models. The merging coefficients for each model are based on the Fisher Information matrix

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| |IMD|B| | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| |QAS|C| | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| |SQUA|D| | |

1.0

1.0

1.0

###### PairedTaskFactor

###### PairedTaskFactor

PairedTaskFactor

0.8

0.8

0.8

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.2 0.4 0.6 0.8 1.0 1.2 IMDB Factor

0.2 0.4 0.6 0.8 1.0 1.2 QASC Factor

0.2 0.4 0.6 0.8 1.0 1.2 SQUAD Factor

1.2

1.0

PairedTaskFactor

0.8

0.6

0.4

###### COMMON_GEN

0.2

0.2 0.4 0.6 0.8 1.0 1.2 COMMON_GEN Factor

IMDB

QASC SQUAD COMMON_GEN

| | | |
|---|---|---|
| | | |

- Figure 9: Coefficient values for each task at different T5 checkpoints. Each plot fixes a reference task and compares its coefficient to those of the other tasks.

and are determined by solving an optimization problem related to finding a centroid between models. One limitation, as pointed out in the original paper, is the high computational cost of estimating the Fisher Information matrix to obtain the merging coefficients. This estimation can also be numerically unstable, as the coefficients in the matrix can be close to zero. Additionally, this method introduces extra scaling hyperparameters that must be tuned.

RegMean. The RegMean merging method, proposed in (Jin et al., 2022), also reduces to task arithmetic, as it performs a linear interpolation between specialized models. This interpolation aims to minimize the L2 distance between the merged model and the individual models, whereas our method is designed to minimize the JS (or KL) divergence between models. The L2 distance is a restrictive measure. Moreover, as stated in (Blau and Michaeli, 2018, 2019; Zhang, 2023), L2 distance is a distortion measure, while KL and JS are perception measures. Minimizing perception distance appears to be more suitable for downstream applications, such as performing other tasks.

Kracher Mean. The Kracher mean (or Riemannian centroid), originally formulated in (Grove and Karcher, 1973) can be used as a merging method which consists in finding some sort of centroid of a finite set of task vectors, denoted as {τt}. To do so, we suppose that task vectors lies in a Finite dimension Hilbert Space (H,< ·,· >), where < ·,· > is the standard dot product onto this space and thus ∥ · ∥ is the associated norm. The Kracher mean is defined as following,

τF ≜ arg min τ∈H

t

∥τ − τt∥2.

The following proposition holds,

- Proposition 4. Let (H,< ·,· >) be a finite dimension Hilbert space. Then for all set of point {τt} ⊂ H, representing task vectors, the solution of the Kracher mean (or equivalently the centroid) can be expressed in the task arithmetic framework. Proof. Let {τt} ⊂ H. Let F ≜ Span({τt}). Let τ ∈ H. We have the following result,

τ = p1 + p2, s.t. p1 ∈ F,p2 ∈ F⊥.

Then we have,

ψ(τ) ≜

=

=

=

t

t

t

t

∥τ − τt∥2,

< τ − τt,τ − τt >,

∥τ∥2 − 2 < τ,τt > +∥τt∥2,

∥p1∥2 + ∥p2∥2 − 2 < p1,τt > +∥τt∥2.

Then by taking, τ′ = p1, we have ψ(τ′) ⩽ ψ(τ), which leads to the following statement: ∀τ ∈ H, ∃τ′ ∈ F, such that,

###### ψ(τ′) ⩽ ψ(τ). Then arg minτ∈H ψ(τ) ∈ F, which concludes the proof.

| |
|---|

- Remark 6. As stated in Remark 2, the method we proposed in this study can also be viewed as a centroid. However the framework we used does not allow to connect directly to the theory of Kracher mean. In fact, if one would want to formulate our method as a Kracher mean, the "distance" defined over the space of task vectors would be the following,

###### d(τi,τj) = DXi (θi∥θj).

However, even in the case of the Jensen Shannon divergence this "distance" is not a mathematical one as it does not respect the property of the distance. Consequently it does not define a metric space and therefore even less a Hilbert space. Then an interesting line of research would be to identify the possible distances one could define over the space of task vectors. From the result we just demonstrated, if we can verify that the distance can be derived from a dot product and thus induce a Hilbert Space, then we can conclude that the optimal solution lies in the framework of task arithmetic, giving thus added weight to this method and possibly offering more theoretical explanations as to why this method is in many cases the state of the art.

