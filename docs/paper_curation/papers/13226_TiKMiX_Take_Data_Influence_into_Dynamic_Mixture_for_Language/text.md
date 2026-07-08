## TiKMiX: Take Data Influence into Dynamic Mixture for Language Model Pre-training

### Yifan Wang, Binbin Liu, Fengze Liu, Yuanfan Guo, Jiyao Deng, Xuecheng Wu Weidong Zhou, Xiaohuan Zhou*, Taifeng Wang

ByteDance yifanyfwang@gmail.com, zhouxiaohuan@bytedance.com

# arXiv:2508.17677v1[cs.LG]25Aug2025

##### Abstract

The data mixture used in the pre-training of a language model is a cornerstone of its final performance. However, a static mixing strategy is suboptimal, as the model’s learning preferences for various data domains shift dynamically throughout training. Crucially, observing these evolving preferences in a computationally efficient manner remains a significant challenge. To address this, we propose TiKMiX, a method that dynamically adjusts the data mixture according to the model’s evolving preferences. TiKMiX introduces Group Influence, an efficient metric for evaluating the impact of data domains on the model. This metric enables the formulation of the data mixing problem as a search for an optimal, influence-maximizing distribution. We solve this via two approaches: TiKMiX-D for direct optimization, and TiKMiX-M, which uses a regression model to predict a superior mixture. We trained models with different numbers of parameters, on up to 1 trillion tokens. TiKMiX-D exceeds the performance of state-of-the-art methods like REGMIX while using just 20% of the computational resources. TiKMiX-M leads to an average performance gain of 2% across 9 downstream benchmarks. Our experiments reveal that a model’s data preferences evolve with training progress and scale, and we demonstrate that dynamically adjusting the data mixture based on Group Influence, a direct measure of these preferences, significantly improves performance by mitigating the “under digestion” of data seen with static ratios.

[Figure 1]

Figure 1: Performance Comparisons of our TiKMiX versus SOTA Data Mixing Strategies for Pre-training a 1B Parameter Language Model with 1T Tokens.

### Introduction

The availability of large-scale public datasets has been a key factor in the creation of Large Language Models (LLMs). The pre-training data for LLMs is predominantly sourced from the internet (Wettig et al. 2025; Yu, Liu, and Xiong 2025), encompassing a wide range of materials such as academic papers (Tirumala et al. 2023), books (Tirumala et al. 2023), and more. The mixture ratio of data from these different domains has a significant impact on the capabilities of an LLM (Zhang et al. 2025b; Liu et al. 2025b; Bai et al. 2024a). For instance, the authors of GPT-3 (Floridi and Chiriatti 2020) considered Wikipedia to have very high-quality data and consequently decided to increase its proportion in the training dataset. REGMIX (Liu et al. 2024) uses results from small-scale experiments to automatically set its mixing ratios, but it does not account for dynamic changes in the state of the

*Corresponding author.

model (Yu, Das, and Xiong 2024; Zhang et al. 2025a). This leads to a critical research question: How can we dynamically select the training data for a model based on its evolving data preferences in a manner that is both scalable and efficient?

Prior research (Xie et al. 2023; Fan, Pagliardini, and Jaggi 2023; Team 2024; Albalak et al. 2023) has leveraged small proxy models to determine domain weights for large-scale language models. This approach is computationally expensive, as it requires training these proxy models on massive datasets, often exceeding 100 billion tokens. Some methods assume that the relative performance of data mixtures is stable across different model scales and training durations (Liu et al. 2024), they neglect the dynamic nature of a model’s data preferences as training progresses. Approaches such as ODM (Albalak et al. 2023) attempt to address this by monitoring training dynamics to guide data allocation, but their iterative nature proves inefficient when dealing with the ever-

increasing scale of pre-training data (Jin et al. 2024; Wang et al. 2025). A significant gap exists in current practices: leading LLMs (Yang et al. 2025; Team et al. 2025; Dubey et al. 2024) utilize multi-stage pre-training, but they lack a mechanism for rapid, dynamic data re-weighting between stages that aligns with the model’s evolving preferences.

We propose a data mixing strategy that dynamically adjusts data proportions during training with minimal computational overhead. To this end, we introduce Group Influence, which efficiently evaluates each domain’s collective impact on validation performance at low computational cost by leveraging gradient accumulation. This allows us to quantify the model’s data preferences at any training stage. Building upon this, we introduce TiKMiX, a method that dynamically adjusts the data mixing strategy by framing it as an optimization problem: finding the data combination that maximizes positive influence. We devise two approaches to solve this: TiKMiXD, which directly optimizes a weighted sum of influences from individual domains to find the best mixing ratios; and the more sophisticated TiKMiX-M, which uses TiKMiX-D’s output as a starting point, conducts perturbation experiments in its vicinity, and then uses a regression model to fit the relationship between mixing ratios and performance, thereby predicting a globally optimal mixture for subsequent largescale training.

With the proposed TiKMiX, we can dynamically adjust the data mixture strategy throughout the model’s entire pretraining cycle, adapting to changes in model scale and training stage. Following previous work (Bai et al. 2024b; Kang et al. 2024; Diao et al. 2025; Tao et al. 2025), we trained models with varying parameter sizes, scaling up to 1 trillion tokens. TiKMiX-D outperforms state-of-the-art methods such as REGMIX while requiring only 20% of the computational resources. TiKMiX-M achieves an average performance improvement of 2% across 9 downstream benchmarks as shown in Fig. 1. Moreover, we further discuss the feasibility and implications of employing TiKMiX in even largerscale models. Our experiments also revealed several key phenomena:(1)A model’s data preferences change as training progresses;(2)Models of different scales exhibit different patterns of preference change;(3) Dynamically adjusting the data mixture promotes a more thorough learning of the data by the model. In conclusion, the main contributions of this paper can be summarized as follows:

- • We propose Group Influence, a novel and efficient method for observing and quantifying the dynamic preferences of Large Language Models for different data domains during the pre-training process.
- • We designed TiKMiX, a dynamic data mixture framework that leverages the observations from Group Influence to adaptively adjust data ratios, aiming to balance the model’s performance across multiple tasks.
- • Extensive experiments demonstrate that our method not only significantly enhances model performance but also provides profound insights into how a model’s data preferences evolve with the training process and model scale, thereby validating the effectiveness of dynamically adjusting data proportions.

### Related Work

#### Influence Function

Influence Functions offer a mathematically grounded method to estimate the effect of training data on model predictions without costly retraining (Koh and Liang 2017). Their application to high-dimensional models like Large Language Models (LLMs) has been hampered by the computational challenge of inverting the Hessian matrix. Recent work has overcome this barrier through scalable approximation techniques. Notably, the work by Anthropic (Grosse et al. 2023) adapted EK-FAC (George et al. 2018), an efficient Hessian approximation, to successfully apply influence functions to 50B-parameter Transformer models. This breakthrough established influence functions as a viable tool for performing data attribution at the scale of modern LLMs, enabling the identification of specific pre-training data that drives model outputs (Kou et al. 2025; Choe et al. 2024; Lin et al. 2024a). However, computation at the sample level incurs prohibitive overhead in large-scale pre-training scenarios. Therefore, we propose Group Influence, a method that extends influence functions to groups of data. By leveraging gradient accumulation techniques, Group Influence can efficiently evaluate the collective impact of an entire data domain with relatively low computational cost. This allows us to quantify the model’s current data preferences.

#### Data Selection and Mixing

Strategic curation of training data significantly enhances model performance (Koh and Liang 2017; Albalak et al.

- 2023). For pre-training Large Language Models (LLMs), data curation methods are commonly categorized by granularity: Token-level Selection: The most fine-grained approach, which filters individual tokens according to specific criteria (Lin et al. 2024b). Sample-level Selection: Methods include heuristic-based approaches (Sharma et al. 2024; Soldaini et al. 2024) and learning-based techniques employing optimization algorithms (Chen et al. 2024; Shao et al.
- 2024). Additionally, approaches such as MATES (Yu, Das, and Xiong 2024) utilize model-derived signals (e.g., perplexity or loss) to inform selection (Marion et al. 2023; Ankner et al. 2024). Group-level Selection: This method partitions data into groups or domains and seeks optimal mixing ratios. Earlier work relied on manually defined ratios, while recent advances favor learning-based strategies. Offline methods like REGMIX (Liu et al. 2024) and DoReMi (Xie et al. 2023) use proxy models to assign static group weights, whereas dynamic methods such as Quad (Zhang et al. 2025a) and ODM (Albalak et al. 2023) iteratively adjust weights during training. Current mainstream pre-training pipelines are typically divided into multiple stages but often lack a mechanism to dynamically adjust the data mixture ratio based on the model’s state in different stages. Our proposed method, TiKMiX, is a semi-offline, group-level selection approach that dynamically adjusts the data mixture ratio across multiple training stages. Unlike fully dynamic methods that require repeated iterative updates, TiKMiX directly optimizes the mixture ratio based on the model’s current data preferences, enabling efficient adaptation without multiple rounds of adjustment.

###### Training Stage N Language Model Language Model

###### Training Stage N+1

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Multi-validation sets

Data ratio from direct optimization

TiKMiX-D: Directly maximize influence

[Figure 6]

[Figure 7]

Observe domainlevel data influence

[Figure 8]

TiKMiX-M: Mix influence model

Data ratio from Objective: Optimize the ratio to mix model

enhanced the data influence across multiple validation sets while maintaining data diversity.

[Figure 9]

[Figure 10]

Figure 2: The process involves periodically measuring domain contributions via Group Influence and adjusting the data mixture to maximize learning efficiency.

### Methodology

In this section, we introduce TiKMiX, a framework for dynamically optimizing the data mixture during large language model pre-training as shown in Fig. 2. Our approach is centered on a novel metric, Group Influence, designed to efficiently measure the real-time contribution of each data domain to the model’s learning. We formulate the dynamic data mixture problem as an optimization task aimed at maximizing this Group Influence. To solve this, we propose two distinct methods : TiKMiX-D, which directly optimizes the mixture based on influence scores, and TiKMiX-M, which leverages a regression model for a computationally efficient approximation. We first define the problem setup and Group Influence, then elaborate on these two optimization strategies.

#### Group Influence

Group Influence function extends the classical influence function framework from individual data points to cohesive groups of data. We first establish the theoretical motivation for this extension, then provide a rigorous mathematical derivation of Group Influence, and finally, discuss its computational properties.

Influence functions offer a principled and computationally efficient method for estimating the effect of a single training instance on a model’s parameters or predictions (Koh and Liang 2017). By approximating the change in model parameters resulting from upweighting a training point z, they provide valuable insights into model behavior without the need for retraining. However, many complex model behaviors, such as systemic bias, factual recall, or vulnerability to specific adversarial attacks, are not attributable to a single, isolated training example. Instead, they often emerge from the collective effect of a group of semantically related instances. A linear summation of individual influence scores, i.e., z

i∈S I(zi), is insufficient as it fails to capture the nontrivial interactions between data points during optimization.

The collective gradient of a group can shape the loss landscape in a manner distinct from the sum of its constituent parts. To quantify the consolidated impact of a data subset S

- as a single entity, we define the Group Influence function. Let a model, parameterized by θ ∈ Rd, be trained on a dataset

D = {z1,...,zN} by minimizing an empirical risk objective J(θ):

θ∗ = arg min

θ

J(θ) = arg min

θ

1 N

N

i=1

L(zi,θ), (1)

where L(zi,θ) is the loss function for instance zi. To measure the influence of a subset S ⊆ D, we introduce a perturbed objective where every member of S is simultaneously upweighted by an infinitesimal positive value ϵ. The new optimal parameters θϵ∗ are found by minimizing this perturbed objective:

θϵ∗ = arg min

θ

  1

N

N

i=1

L(zi,θ) + ϵ

zj∈S

L(zj,θ)

 . (2)

This formulation models a scenario where the training process is nudged to place greater emphasis on the group S. For ϵ = 0, we recover the original optimal parameters, θϵ∗=0 = θ∗. The influence of group S on the model parameters is then defined as the rate of change of θϵ∗ with respect to ϵ, evaluated

- at ϵ = 0. A closed-form expression for this quantity can be derived using the implicit function theorem. The first-order optimality condition for any ϵ requires that the gradient of

the perturbed objective at its minimum θϵ∗ is zero, which can be formulated as:

1 N

∇θJϵ(θϵ∗,S) =

N

∇θL(zi,θϵ∗)+ϵ

∇θL(zj,θϵ∗) = 0.

zj∈S

i=1

(3)

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Figure 3: The impact of different pre-training data domains on the validation set as training progresses.

Differentiating this entire equation with respect to ϵ via the chain rule yields:

The scalar value If(S) quantifies the extent to which upweighting the group S during training would increase

(If(S) > 0) or decrease (If(S) < 0) the value of the function f. A significant computational advantage of Equation 8

dθϵ∗ dϵ

d dϵ

∂ ∂ϵ

[∇θJϵ(θϵ∗,S)] = ∇2θJϵ(θϵ∗,S)

(∇θJϵ(θϵ∗,S)) = 0.

+

is its structure. The term z

j∈S ∇θL(zj,θ∗) is the accumulated gradient of the group S. This allows for an efficient implementation where the gradients for all samples within the group are first computed and aggregated. Subsequently, the computationally intensive Hessian-inverse-vector product is performed only once. This structure ensures the computation of Group Influence is scalable, as its cost is not dominated by the cardinality of the group |S|.

(4)

Evaluating this expression at ϵ = 0 (where θϵ∗=0 = θ∗), the Hessian ∇2θJϵ(θϵ∗,S) simplifies to the Hessian of the original objective, Hθ∗ ≜ ∇2θJ(θ∗). The partial derivative term becomes z

j∈S ∇θL(zj,θ∗). Substituting these into Equation 4 gives:

dθϵ∗ dϵ ϵ=0

∇θL(zj,θ∗) = 0. (5)

+

Hθ∗

zj∈S

#### TiKMiX-D: Directly maximize influence

Assuming the Hessian Hθ∗ is positive definite and thus invertible, we can solve for the influence of group S on the model parameters:

Building upon the Group Influence metric, which quantifies the impact of data domains on the model’s state as shown in Fig. 3, our objective is to determine an optimal data mixture, represented by a weight vector w, that maximizes the aggregate benefits of this influence. To this end, we introduce TiKMiX-D, a method that formulates this task as a multiobjective optimization problem. This approach dynamically adjusts the data mixture for the subsequent training stage to achieve balanced performance improvement, maximize overall gains, and maintain data diversity. The Group Influence scores, as computed in the previous section, are first organized into an n × m Influence Matrix, S, where n is the number of validation tasks and m is the number of data domains. Each element Sij represents the influence of data domain dj on validation task vi. Given a data mixture weight vector w = [w1,w2,...,wm]T, the expected total influence on each validation task is captured by the vector P = S · w. To facilitate fair comparison across tasks of varying scales, we normalize this influence vector. The normalized influence

 

 .

dθϵ∗ dϵ ϵ=0

Iparam(S) ≜

= −Hθ−∗1

∇θL(zj,θ∗)

zj∈S

(6) A common practical application is to measure the influence of S on a scalar-valued function of the parameters, f(θ), such as the loss on a test sample, f(θ) = L(ztest,θ). By applying the chain rule, the influence of S on f is given by:

df(θϵ∗) dϵ ϵ=0

dθϵ∗ dϵ ϵ=0

If(S) ≜

= ∇θf(θ∗)T

. (7)

Substituting Equation 6 into Equation 7 yields the final expression for the Group Influence function:

 

 . (8)

If(S) = −∇θf(θ∗)THθ−∗1

∇θL(zj,θ∗)

zj∈S

Pˆi for task vi can be computed as: Pˆi =

Pi maxj Sij + ϵ

, (9)

where ϵ is a small constant (e.g., 10−8) to prevent division by zero. The optimization objective for TiKMiX-D is formulated as a unified function L(w) that integrates three distinct goals:

L(w) = α · std(Pˆ) − β ·

n

Pˆi − γ · H(w). (10)

i=1

This function balances three key components. First, influence uniformity, measured by the standard deviation of the normalized influence, std(Pˆ), encourages balanced capability gains across all tasks. Second, overall influence gain, represented by the total sum of normalized influences, P ˆi, aims to maximize the model’s aggregate performance improvement; hence, its negative is minimized. Third, data diversity, measured by the information entropy of the weight vector, H(w) = − mj=1 wj log(wj), promotes a more uniform weight distribution to ensure robust generalization. The hyperparameters α,β, and γ control the trade-offs among these objectives; in our experiments, they are set to 1 to assign equal importance. The complete optimization problem is subject to several constraints to ensure a valid and beneficial solution. The weights must be non-negative (wj ≥ 0) and sum to one ( wj = 1). Furthermore, to guarantee continuous improvement, we enforce a Pareto improvement constraint, ensuring that the influence generated by the new mixture w is no less than that of the prior mixture wprior for any task, i.e., S · w ≥ S · wprior. This leads to the final constrained non-linear optimization problem:

n

Pˆi − γ · H(w)

α · std(Pˆ) − β ·

minimize

w

i=1

m

(11)

subject to

wj = 1

j=1

wj ≥ 0, ∀j ∈ {1,...,m} S · w ≥ S · wprior.

We employ the Sequential Least Squares Quadratic Programming algorithm (Gupta and Gupta 2018) to solve this problem, initializing the weights with a uniform distribution. The resulting optimal vector, wbest, serves as the dynamic data mixture for the subsequent training stage.

#### TiKMiX-M: Mix influence model

While TiKMiX-D provides an efficient strategy for data mixing through direct optimization, it operates on the assumption that the influences of data domains are linearly additive. This simplification may overlook the mix of different domain, non-linear cross-domain interactions that arise when different data sources are combined. We introduce TiKMiX-M, optimize mixture proportions by modeling the interactions within domain mixtures To more accurately capture these mixture effects.

To explore the model’s performance across a diverse range of domain weightings, we generate a set of N candidate mixture vectors. Our approach is anchored by an empirically determined prior weight vector, worig ∈ RD, where D is the number of domains. For each domain i, we define a plausible sampling interval by scaling the original weight. We employ Latin Hypercube Sampling (Loh 2021) within this D-dimensional hyperrectangle to efficiently generate candidate vectors, ensuring a uniform and non-collapsing coverage of the parameter space.

Each candidate vector wcand produced by Latin Hypercube Sampling is subsequently normalized to satisfy the sum-toone constraint ( Di=1 wi = 1), yielding a normalized vector wnorm = wcand/ Dj=1 wcand,j. However, this normalization can shift components outside their predefined intervals. Therefore, we implement a rejection sampling scheme, where a normalized vector wnorm is accepted into our final set only if it satisfies the boundary constraints for all dimensions, i.e., wnorm,i ∈ [li,hi] for all i ∈ {1,...,D}. This iterative process is repeated until N valid weight vectors that meet both the summation and boundary conditions have been collected, resulting in a robust and well-distributed set of weights for subsequent analysis.

For each generated candidate mixture wi, we calculate its true aggregate influence score, yi, across all validation sets using the Group Influence evaluation method. This score

can be the sum of normalized influences, P ˆi, or another comprehensive metric.

Following these steps, we obtain a training set Dtrain =

{(wi,yi)}Ni=1. We select LightGBM (Ke et al. 2017), an efficient gradient boosting decision tree model, as our re-

gression surrogate. This model, fLGBM, is trained to predict the aggregate influence y for given data mixture w, i.e.,

y = fLGBM(w).We leverage it to efficiently explore the mixture space without performing expensive, true influence evaluations. We design an iterative search algorithm that balances exploration and exploitation to find the optimal mixture.

Algorithm 1: Iterative Search via TiKMiX-M

Input: Surrogate fsur, initial mix w(0), iters T, samples N,

exploration [αmin,αmax], top-k. Output: Optimized mixture w∗.

wbest ← w(0) Generate exploration strengths {αt}Tt=1 logarithmically from αmax to αmin. for t = 1 to T do

Sample candidates {wi}Ni=1 ∼ Dirichlet(αt · wbest). Predict scores yi = fsur(wi) for each wi. Let Itop-k be the indices of the top k scores.

wbest ← k1 i∈Itop-k wi. end for return wbest

The process is detailed in Algorithm 1. We start from the ratio from TiKMiX-D, wbest-D. At each step, we sample candidate mixtures on the current best solution. The distribution’s concentration parameter is annealed over step, beginning with

a large value to encourage global exploration and gradually decreasing to promote local exploitation near the optimum. We employ the surrogate model to evaluate all sampled candidates. The center for the next iteration is then updated to be the average of the top-k candidates with the highest predicted scores. This procedure is repeated until convergence or a maximum number of iterations is reached.

TiKMiX-M not only accounts for non-linear cross-domain interactions but also significantly enhances search efficiency through the surrogate model, enabling it to discover superior solutions within the vast mixture space.

### Experiments

This section presents a comprehensive set of experiments designed to validate the effectiveness of our TiKMiX framework. We first outline the experimental setup, including evaluation benchmarks, datasets, and baseline methods. Subsequently, we demonstrate that: (1) the pre-training data mixture significantly impacts downstream task performance; (2) our proposed Group Influence is an effective predictor of downstream performance; and (3) the TiKMiX framework, particularly TiKMiX-D and TiKMiX-M, markedly improves model performance and surpasses existing SOTA methods.

#### Experimental Setup

Datasets and Models Optimizing the data mixture of webscale corpora is a crucial and highly impactful step in pretraining performant Large Language Models (LLMs). While the diversity of web data presents unique challenges, effective mixing strategies can unlock significant performance gains. To systematically investigate this, we conduct our experiments on the RefinedWeb dataset (Penedo et al. 2023), which comprises 26 distinct data domains. Our models, ranging in size from 1B to 7B parameters, are trained on up to 1 trillion tokens. The training process is divided into two distinct stages, each consisting of 500 billion tokens, with a strategic adjustment of the data mixture ratio at the transition point between stages. We compare TiKMiX against several representative data mixing strategies: Pile-CC (Gao et al. 2020): The original data mixture proposed by the authors of The Pile based on heuristics. REGMIX (Liu et al. 2024): SOTA method that uses a regression model to predict and optimize validation loss for determining the mixture. DoReMi (Xie et al. 2023): a classic dynamic data mixing method that relies on a proxy model. QUAD (Zhang et al. 2025a): a method for dynamic selection during training after clustering data We use the best-reported mixture from their paper, re-normalized to the domains available in our setup.

Downstream Task Evaluation To comprehensively evaluate our proposed method, we curated a diverse set of 9 widely recognized downstream benchmarks, which were strategically divided into two categories: in-domain and outof-domain. This division allows for a rigorous assessment of both the model’s core capabilities and its generalization prowess. Our in-domain evaluation suite was designed to cover a wide spectrum of reasoning and knowledge-based tasks. It includes MMLU (Massive Multitask Language Understanding) (Hendrycks et al. 2020), a challenging bench-

[Figure 23]

Figure 4: Analysis of the Group Influence and actual performance on the benchmark.

mark measuring knowledge across 57 diverse subjects; HellaSwag (Zellers et al. 2019), a commonsense reasoning task that involves choosing the most plausible continuation for a given context; ARC (Clark et al. 2018), which we evaluate on both the Easy (ARC-E) and the more difficult Challenge (ARC-C) sets of grade-school science questions; and TriviaQA (Joshi et al. 2017), a reading comprehension benchmark requiring models to locate answers within lengthy documents. To evaluate the generalization capabilities of our method, we selected a set of out-of-domain benchmarks. These include PiQA (Bisk et al. 2020), a commonsense benchmark focused on physical interactions; OpenBookQA (Mihaylov et al. 2018), a question-answering task requiring reasoning over a given set of science facts; BoolQ (Clark et al. 2019), a dataset of naturally occurring yes/no questions; and MathQA (Amini et al. 2019), a mathematical reasoning benchmark with multi-step word problems.

#### Group Influence as an Effective Predictor of Performance

The core hypothesis of our introduced TiKMiX framework is that maximizing Group Influence can effectively enhance overall downstream task performance. To validate this hypothesis, we calculated the impact of 10 different data mixtures on various benchmarks. As validation, we trained a 1B-parameter model on 500B data using the corresponding mixtures. The normalized scores are shown in Fig. 4. We can clearly observe a strong positive correlation (i.e., Pearson correlation coefficient ρ = 0.789) between the total Group Influence and the average downstream scores. This indicates that mixtures generating higher total influence almost invariably lead to better downstream performance. This finding not only confirms the validity of Group Influence as an optimization target but also provides a solid theoretical foundation for the design of our TiKMiX-D and TiKMiX-M.

Benchmark Human DoReMi Average QUAD Pile-CC REGMiX TiKMiX-D TiKMiX-M In-Domain Benchmarks

MMLU (Hendrycks et al. 2020) 31.3 31.2 30.9 31.7 31.2 31.5 32.2 31.8 HellaSwag (Zellers et al. 2019) 55.5 55.3 55.9 56.5 55.6 56.0 57.4 56.6 ARC Easy (Clark et al. 2018) 64.4 65.7 64.1 62.8 63.2 66.2 69.3 70.7 ARC Challenge (Clark et al. 2018) 33.7 33.6 32.1 33.5 32.7 33.2 37.0 38.3 Triviaqa (Joshi et al. 2017) 17.6 15.5 17.3 17.6 16.3 15.8 17.7 17.3 Out-of-Domain Benchmarks

PiQA (Bisk et al. 2020) 73.5 73.1 71.5 72.4 69.2 73.3 74.1 74.5 OpenBookQA (Mihaylov et al. 2018) 35.8 36.5 34.6 36.6 37.1 37.0 37.4 37.4 Boolq (Clark et al. 2019) 56.3 59.2 58.3 60.5 58.7 58.9 61.3 62.2 MathQA (Amini et al. 2019) 22.7 23.1 23.7 23.9 22.5 23.3 23.5 24.2

Estimated FLOPs 0 4.2e19 0 2.3e18 0 3.7e18 7.2e17 3.2e18 Average Perf. 43.4 43.7 43.2 43.9 42.9 43.9 45.5 45.9 Best On 0/9 0/9 0/9 0/9 0/9 0/9 4/9 6/9

Table 1: Comparison of 1B Parameter Models Trained on 1T Tokens Across Various Benchmarks. The best-performing model on each benchmark is highlighted in bold.

#### TiKMiX Improves Downstream Performance

Building on the preceding findings, we formally evaluate the two implementations of our TiKMiX framework: TiKMiX-D and TiKMiX-M. During a 1T-token pre-training process, we dynamically adjusted the data mixture every 200B tokens using TiKMiX. As shown in Table 1, both of our methods significantly outperform all baselines. On average, across 9 benchmarks, TiKMiX-D and TiKMiX-M improved performance by 1.6% and 2.0%, respectively, over the strongest baseline, REGMIX. Notably, on challenging tasks like ARC Easy and ARC Challenge, TiKMiX-M achieved a performance advantage of over 4.8%.

#### Analysis of Computational Efficiency

TiKMiX also excels in computational efficiency. Unlike methods such as MATES(Yu, Das, and Xiong 2024),GroupMATES(Yu et al. 2025) and REGMIX, which require training small proxy models, the Group Influence calculation and optimization process of TiKMiX is highly efficient. In our 1B model experiments, the total computational overhead for TiKMiX-D to determine the next-stage mixture (including influence calculation and regression model inference) was only about 20% of that required by the RegMix method, while achieving comparable or even superior performance. This high efficiency makes TiKMiX a practical and powerful tool for large scale LLM training.

#### Ablation Study

We conduct a series of ablation studies, with the results presented in Table 2. Our primary investigation focused on the efficacy of using group influence and TiKMiX for preference observation and data mixture adjustments. As shown in Table 2, our approach allows for the accurate observation of model preferences using only 0.1B tokens and requires no model training, leading to a significant performance improvement over the loss. This highlights the superiority of our method in efficiently identifying and correcting data biases. We further discuss the effectiveness of our model on a larger scale in the appendix.

Loss TiKMiX-D

Benchmark

5B 10B 0.1B 0.5B In-Domain Benchmarks

MMLU (Hendrycks et al. 2020) 31.4 31.2 32.2 32.1 HellaSwag (Zellers et al. 2019) 56.3 56.4 57.4 57.6 ARC Easy (Clark et al. 2018) 67.3 65.6 69.3 69.1 ARC Challenge (Clark et al. 2018) 34.4 33.4 37.0 37.1 TriviaQA (Joshi et al. 2017) 16.5 16.9 17.7 17.9 Out-of-Domain Benchmarks

PiQA (Bisk et al. 2020) 73.2 73.5 74.1 74.2 OpenBookQA (Mihaylov et al. 2018) 36.4 36.6 37.4 37.3 BoolQ (Clark et al. 2019) 59.4 59.7 61.3 61.5 MathQA (Amini et al. 2019) 23.9 23.7 23.5 23.6

Average Perf. 44.3 44.1 45.5 45.6

Table 2: Ablation study of Loss and TiKMiX on different data sizes.

### Conclusion and Discussions

In this work, we address the suboptimality of static data mixing strategies in language model pre-training, demonstrating that a model’s learning preferences for different data domains evolve dynamically with its training progress. To tackle this, we introduce TiKMiX, a novel framework that dynamically adjusts the data mixture based on Group Influence, a highly efficient metric to evaluate the contribution of data domains to the model’s performance. By framing data mixing as an influence-maximization problem, we developed two approaches: TiKMiX-D, which directly optimizes the mixture and surpasses state-of-the-art methods like REGMIX using only 20% of the computational resources, and TiKMiX-M, which uses a regression model to predict superior mixtures, achieving an average performance gain of 2% across 9 downstream benchmarks. Our experiments confirm that dynamically adjusting the data mixture based on Group Influence significantly improves performance by mitigating the ”under-digestion” of data seen with static ratios. We plan to conduct further experiments on larger-scale models and more diverse datasets to further validate the effectiveness of Group Influence and TiKMiX.

### References

Albalak, A.; Pan, L.; Raffel, C.; and Wang, W. Y. 2023. Efficient online data mixing for language model pre-training. arXiv preprint arXiv:2312.02406.

Amini, A.; Gabriel, S.; Lin, P.; Koncel-Kedziorski, R.; Choi, Y.; and Hajishirzi, H. 2019. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. arXiv preprint arXiv:1905.13319.

Ankner, Z.; Blakeney, C.; Sreenivasan, K.; Marion, M.; Leavitt, M. L.; and Paul, M. 2024. Perplexed by perplexity: Perplexity-based data pruning with small reference models. arXiv preprint arXiv:2405.20541.

Bai, T.; Liang, H.; Wan, B.; Xu, Y.; Li, X.; Li, S.; Yang, L.; Li,

- B.; Wang, Y.; Cui, B.; et al. 2024a. A survey of multimodal large language model from a data-centric perspective. arXiv preprint arXiv:2405.16640. Bai, T.; Yang, L.; Hao Wong, Z.; Peng, J.; Zhuang, X.; Zhang,
- C.; Wu, L.; Qiu, J.; Zhang, W.; Yuan, B.; et al. 2024b. Multiagent collaborative data selection for efficient llm pretraining. arXiv e-prints, arXiv–2410.

Bisk, Y.; Zellers, R.; Gao, J.; Choi, Y.; et al. 2020. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 7432–7439.

Chen, X.; Wang, Z.; Sow, D.; Yang, J.; Chen, T.; Liang,

- Y.; Zhou, M.; and Wang, Z. 2024. Take the bull by the horns: Hard sample-reweighted continual training improves llm generalization. arXiv preprint arXiv:2402.14270.

Choe, S. K.; Ahn, H.; Bae, J.; Zhao, K.; Kang, M.; Chung, Y.; Pratapa, A.; Neiswanger, W.; Strubell, E.; Mitamura, T.; et al.

- 2024. What is your data worth to gpt? llm-scale data valuation with influence functions. arXiv preprint arXiv:2405.13954.

Clark, C.; Lee, K.; Chang, M.-W.; Kwiatkowski, T.; Collins, M.; and Toutanova, K. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044.

Clark, P.; Cowhey, I.; Etzioni, O.; Khot, T.; Sabharwal, A.; Schoenick, C.; and Tafjord, O. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Diao, S.; Yang, Y.; Fu, Y.; Dong, X.; Su, D.; Kliegl, M.; Chen, Z.; Belcak, P.; Suhara, Y.; Yin, H.; et al. 2025. Climb: Clustering-based iterative data mixture bootstrapping for language model pre-training. arXiv preprint arXiv:2504.13161.

Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.; et al. 2024. The llama 3 herd of models. arXiv e-prints, arXiv–2407.

Fan, S.; Pagliardini, M.; and Jaggi, M. 2023. Doge: Domain reweighting with generalization estimation. arXiv preprint arXiv:2310.15393.

Floridi, L.; and Chiriatti, M. 2020. GPT-3: Its nature, scope, limits, and consequences. Minds and machines, 30(4): 681– 694.

Gao, L.; Biderman, S.; Black, S.; Golding, L.; Hoppe, T.; Foster, C.; Phang, J.; He, H.; Thite, A.; Nabeshima, N.; et al. 2020. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027.

George, T.; Laurent, C.; Bouthillier, X.; Ballas, N.; and Vincent, P. 2018. Fast approximate natural gradient descent in a kronecker factored eigenbasis. Advances in neural information processing systems, 31.

Grosse, R.; Bae, J.; Anil, C.; Elhage, N.; Tamkin, A.; Tajdini, A.; Steiner, B.; Li, D.; Durmus, E.; Perez, E.; et al. 2023. Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296.

Gupta, M.; and Gupta, B. 2018. An ensemble model for breast cancer prediction using sequential least squares programming method (slsqp). In 2018 eleventh international conference on contemporary computing (IC3), 1–3. IEEE.

He, P.; Gao, J.; and Chen, W. 2023. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. arXiv preprint arXiv:2111.09543.

Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Jin, X.; Zhu, H.; Li, S.; Wang, Z.; Liu, Z.; Tian, J.; Yu, C.; Qin, H.; and Li, S. Z. 2024. A survey on mixup augmentations and beyond. arXiv preprint arXiv:2409.05202.

Joshi, M.; Choi, E.; Weld, D. S.; and Zettlemoyer, L. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551.

Kang, F.; Sun, Y.; Wen, B.; Chen, S.; Song, D.; Mahmood, R.; and Jia, R. 2024. AutoScale: Automatic Prediction of Compute-optimal Data Compositions for Training LLMs.

Ke, G.; Meng, Q.; Finley, T.; Wang, T.; Chen, W.; Ma, W.; Ye, Q.; and Liu, T.-Y. 2017. Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30.

Koh, P. W.; and Liang, P. 2017. Understanding black-box predictions via influence functions. In International conference on machine learning, 1885–1894. PMLR.

Kou, S.; Tian, Q.; Xu, H.; Zeng, Z.; and Deng, Z. 2025. Which Data Attributes Stimulate Math and Code Reasoning? An Investigation via Influence Functions. arXiv preprint arXiv:2505.19949.

Lin, X.; Wang, W.; Li, Y.; Yang, S.; Feng, F.; Wei, Y.; and Chua, T.-S. 2024a. Data-efficient Fine-tuning for LLM-based Recommendation. In Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, 365–374.

Lin, Z.; Gou, Z.; Gong, Y.; Liu, X.; Shen, Y.; Xu, R.; Lin, C.; Yang, Y.; Jiao, J.; Duan, N.; et al. 2024b. Rho-1: Not all tokens are what you need. arXiv preprint arXiv:2404.07965. Liu, F.; Zhou, W.; Liu, B.; Yu, Z.; Zhang, Y.; Lin, H.; Yu, Y.; Zhang, B.; Zhou, X.; Wang, T.; et al. 2025a. Quadmix: Quality-diversity balanced data selection for efficient llm pretraining. arXiv preprint arXiv:2504.16511.

Liu, J.; Zhu, D.; Bai, Z.; He, Y.; Liao, H.; Que, H.; Wang,

- Z.; Zhang, C.; Zhang, G.; Zhang, J.; et al. 2025b. A comprehensive survey on long context language modeling. arXiv preprint arXiv:2503.17407.

Liu, Q.; Zheng, X.; Muennighoff, N.; Zeng, G.; Dou, L.; Pang, T.; Jiang, J.; and Lin, M. 2024. Regmix: Data mixture as regression for language model pre-training. arXiv preprint arXiv:2407.01492.

Loh, W.-L. 2021. On Latin hypercube sampling. The annals of statistics.

Marion, M.; Ust¨¨ un, A.; Pozzobon, L.; Wang, A.; Fadaee, M.; and Hooker, S. 2023. When less is more: Investigating data pruning for pretraining llms at scale. arXiv preprint arXiv:2309.04564.

Mihaylov, T.; Clark, P.; Khot, T.; and Sabharwal, A. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789.

Penedo, G.; Malartic, Q.; Hesslow, D.; Cojocaru, R.; Alobeidli, H.; Cappelli, A.; Pannier, B.; Almazrouei, E.; and Launay, J. 2023. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data only. Advances in Neural Information Processing Systems, 36: 79155–79172.

Shao, Y.; Li, L.; Fei, Z.; Yan, H.; Lin, D.; and Qiu, X. 2024. Balanced data sampling for language model training with clustering. arXiv preprint arXiv:2402.14526.

Sharma, V.; Padthe, K.; Ardalani, N.; Tirumala, K.; Howes, R.; Xu, H.; Huang, P.-Y.; Li, S.-W.; Aghajanyan, A.; Ghosh, G.; et al. 2024. Text quality-based pruning for efficient training of language models. arXiv preprint arXiv:2405.01582.

Soldaini, L.; Kinney, R.; Bhagia, A.; Schwenk, D.; Atkinson,

- D.; Authur, R.; Bogin, B.; Chandu, K.; Dumas, J.; Elazar, Y.; et al. 2024. Dolma: An open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159. Tao, Z. S.; Vinken, K.; Yeh, H.-W.; Cooper, A.; and Boix, X.

- 2025. Merge to Mix: Mixing Datasets via Model Merging. arXiv preprint arXiv:2505.16066.

Team, K.; Bai, Y.; Bao, Y.; Chen, G.; Chen, J.; Chen, N.; Chen, R.; Chen, Y.; Chen, Y.; Chen, Y.; et al. 2025. Kimi K2: Open Agentic Intelligence. arXiv preprint arXiv:2507.20534.

Team, Q. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671.

Tirumala, K.; Simig, D.; Aghajanyan, A.; and Morcos, A. 2023. D4: Improving llm pretraining via document deduplication and diversification. Advances in Neural Information Processing Systems, 36: 53983–53995.

Wang, K.; Zhang, G.; Zhou, Z.; Wu, J.; Yu, M.; Zhao, S.; Yin, C.; Fu, J.; Yan, Y.; Luo, H.; et al. 2025. A comprehensive survey in llm (-agent) full stack safety: Data, training and deployment. arXiv preprint arXiv:2504.15585.

Wettig, A.; Lo, K.; Min, S.; Hajishirzi, H.; Chen, D.; and Soldaini, L. 2025. Organize the Web: Constructing Domains Enhances Pre-Training Data Curation. arXiv preprint arXiv:2502.10341.

Xie, S. M.; Pham, H.; Dong, X.; Du, N.; Liu, H.; Lu, Y.; Liang, P. S.; Le, Q. V.; Ma, T.; and Yu, A. W. 2023. Doremi: Optimizing data mixtures speeds up language model pretraining. Advances in Neural Information Processing Systems, 36: 69798–69818.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu,

- B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yu, S.; Liu, Z.; and Xiong, C. 2025. Craw4LLM: Efficient Web Crawling for LLM Pretraining. arXiv preprint arXiv:2502.13347.

Yu, Z.; Das, S.; and Xiong, C. 2024. Mates: Model-aware data selection for efficient pretraining with data influence models. Advances in Neural Information Processing Systems, 37: 108735–108759.

Yu, Z.; Peng, F.; Lei, J.; Overwijk, A.; Yih, W.-t.; and Xiong,

- C. 2025. Data-efficient pretraining with group-level data influence modeling. arXiv preprint arXiv:2502.14709.

Zellers, R.; Holtzman, A.; Bisk, Y.; Farhadi, A.; and Choi, Y. 2019. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830.

Zhang, C.; Zhong, H.; Zhang, K.; Chai, C.; Wang, R.; Zhuang, X.; Bai, T.; Jiantao, Q.; Cao, L.; Fan, J.; et al. 2025a. Harnessing Diversity for Important Data Selection in Pretraining Large Language Models. In The Thirteenth International Conference on Learning Representations.

Zhang, X.; Wang, D.; Dou, L.; Zhu, Q.; and Che, W. 2025b. A survey of table reasoning with large language models. Frontiers of Computer Science, 19(9): 199348.

### Appendix

#### Experimental Setup

Datasets and Models Web data serves as one of the core sources for pre-training large language models (LLMs), playing a crucial role in enhancing model capabilities due to its broad coverage and diversity. However, precisely because web data encompasses a wide range of domains—including news, encyclopedias, forums, and academic content—its highly diverse origins make it extremely challenging to achieve a balanced mixture across different domains. We follow the same experimental setup as prior studies on web data mixture (Wettig et al. 2025; Liu et al. 2025a), utilize the RefinedWeb dataset (Penedo et al. 2023), and employ the domain classifier (He, Gao, and Chen 2023) to categorize the data into 26 distinct domains. Our models, ranging in size from 1B to 7B parameters, are trained on up to 1 trillion tokens. The training process is divided into two distinct stages, each consisting of 500 billion tokens, with a strategic adjustment of the data mixture ratio at the transition point between stages. We compare TiKMiX against several representative data mixing strategies: Pile-CC (Gao et al. 2020): The original data mixture proposed by the authors of The Pile based on heuristics. REGMIX (Liu et al. 2024): SOTA method that uses a regression model to predict and optimize validation loss for determining the mixture. DoReMi (Xie et al. 2023): a classic dynamic data mixing method that relies on a proxy model. QUAD (Zhang et al. 2025a): a method for dynamic selection during training after clustering data We use the best-reported mixture from their paper, re-normalized to the domains available in our setup.

Our proposed TiKMiX method achieves a balance between dynamic adaptability and computational efficiency in data mixture strategies. Similar to other dynamic approaches such as DoReMi and QUAD, TiKMiX adjusts the data mixture ratios according to the current state of the model. However, unlike these methods, TiKMiX does not require multiple iterations, which significantly improves training efficiency. Furthermore, TiKMiX simplifies the data mixing process and reduces engineering complexity without sacrificing model performance.

To systematically evaluate the effectiveness of different data mixing strategies, we conduct large-scale experiments on the RefinedWeb dataset. Our models range in size from 1B to 7B parameters and are trained on up to 1 trillion tokens. The training process is divided into two distinct stages, each consisting of 500 billion tokens. At the transition point between these two stages, we strategically adjust the data mixture ratios to further assess the impact of mixing strategies on model performance.

#### Downstream Task Evaluation

To conduct a comprehensive and rigorous evaluation of our proposed method, we curated a diverse suite of nine widelyrecognized downstream benchmarks. This evaluation matrix is strategically divided into two categories: in-domain and out-of-domain. This bifurcation allows for a dual-faceted assessment of our model’s capabilities: on one hand, to measure its proficiency on tasks closely aligned with its training

objectives, and on the other, to critically examine its ability to generalize learned skills to novel tasks and knowledge domains. The consistent performance gains observed across both categories underscore our method’s ability to enhance the model’s foundational capabilities and foster robust generalization.

In-Domain Evaluation Our in-domain evaluation suite is designed to probe the model’s core competencies in complex reasoning, commonsense understanding, and knowledgeintensive applications. These benchmarks are thematically aligned with our method’s primary optimization goals and serve to quantify the depth of improvement in these critical areas.

- • MMLU (Massive Multitask Language Understanding) (Hendrycks et al. 2020): A highly challenging multitask benchmark that assesses knowledge across 57 disparate subjects, ranging from elementary mathematics and U.S. history to computer science and law. MMLU demands not only a vast repository of knowledge but also the ability to perform precise, domain-specific reasoning, making it a key indicator of a model’s comprehensive intellectual and academic capabilities.
- • HellaSwag (Zellers et al. 2019): A commonsense reasoning benchmark that tasks the model with selecting the most plausible continuation for a given context. HellaSwag is distinguished by its use of adversariallygenerated distractors, which are designed to be highly confusable for models that rely on superficial statistical cues. It therefore serves as a robust test of a model’s deeper understanding of causality and everyday situations.
- • ARC (AI2 Reasoning Challenge) (Clark et al. 2018): This benchmark evaluates reasoning and comprehension on grade-school science questions. We assess performance on both its subsets: ARC-Easy (ARC-E), which contains questions often solvable via information retrieval, and the more difficult ARC-Challenge (ARC-C), which requires multi-step reasoning and synthesis of knowledge. Evaluating on both allows for a fine-grained analysis of the model’s capabilities, from basic knowledge retrieval to complex scientific inference.
- • TriviaQA (Joshi et al. 2017): A large-scale reading comprehension benchmark where questions are authored by trivia enthusiasts, leading to a high degree of diversity and complexity. The task requires models to locate answers within lengthy, evidence-rich documents, often amidst significant distractor information. It primarily evaluates the model’s proficiency in long-context processing, precise information retrieval, and fact verification.

Out-of-Domain Evaluation To rigorously assess the generalization power of our method, we selected a set of outof-domain benchmarks that are distinct from the in-domain tasks in terms of subject matter, format, or required reasoning skills. Performance on these benchmarks directly reflects the model’s ability to transfer its learned meta-skills to new and unseen challenges.

• PiQA (Physical Interaction QA) (Bisk et al. 2020): A commonsense benchmark focused on physical reasoning.

Table 3: Ablation study of REGMIX and TiKMiX on 1B and 7B models.

1B Model 7B Model Benchmark REGMIX TiKMiX-D REGMIX TiKMiX-D In-Domain Benchmarks

MMLU (Hendrycks et al. 2020) 31.5 32.2 40.7 41.5 HellaSwag (Zellers et al. 2019) 56.0 57.4 76.6 76.4 ARC Easy (Clark et al. 2018) 66.2 69.3 78.5 78.4 ARC Challenge (Clark et al. 2018) 32.2 37.0 49.4 50.2 TriviaQA (Joshi et al. 2017) 15.8 17.7 46.4 45.3

###### Out-of-Domain Benchmarks

PiQA (Bisk et al. 2020) 73.3 74.1 79.1 79.2 OpenBookQA (Mihaylov et al. 2018) 37.0 37.4 43.2 45.4 MathQA (Amini et al. 2019) 23.2 23.5 28.8 29.9

Average Perf. 43.9 45.5 55.3 56.0

Presented in a question-answering format, it requires the model to understand the properties and affordances of everyday objects (e.g., ”How can you cool a cup of water faster?”). PiQA probes the model’s intuitive grasp of how the physical world operates, a domain of commonsense distinct from academic knowledge, making it an excellent test of generalization.

- • OpenBookQA (Mihaylov et al. 2018): This benchmark simulates an ”open-book” exam, requiring the model to answer questions using a given set of elementary science facts. Success demands not only reading comprehension but, more importantly, the ability to reason over and combine these facts to answer questions whose solutions are not explicitly stated. It critically evaluates the model’s capacity for multi-step reasoning and knowledge application within a constrained context.
- • BoolQ (Boolean Questions) (Clark et al. 2019): A dataset of naturally occurring yes/no questions, sourced from real user search queries. The challenge lies in the fact that the relationship between the question and the provided evidence passage is often implicit, requiring sophisticated syntactic and semantic analysis to arrive at a correct Boolean judgment. BoolQ effectively measures the model’s fine-grained comprehension of natural, conversational language.
- • MathQA (Amini et al. 2019): A mathematical reasoning benchmark featuring multi-step word problems. The task requires models to parse natural language descriptions, formulate a correct sequence of operations, and execute them to find a solution. Covering a diverse range of mathematical reasoning categories, MathQA is a crucial benchmark for evaluating a model’s symbolic reasoning and logical chain-of-thought capabilities, representing a significant test of higher-order cognitive skills.

By systematically evaluating our method across this dualcategory, nine-benchmark matrix, we demonstrate that our approach not only enhances performance in core competency areas (as shown by MMLU and ARC-C) but also significantly improves the transfer of these abilities to novel contexts (as evidenced by PiQA and MathQA). This comprehensive improvement across both in-domain and out-of-domain tasks provides strong evidence for the effectiveness and generalizability of our method.

To further investigate the impact of model scale on data

utilization, we present a supplementary analysis in Figures 5 to 11. Our key finding is that models of different scales (1B and 7B) exhibit significantly different learning responses and form distinct preferences, even when trained on the exact same data. This phenomenon reveals a complex interplay between data utility and model scale. It provides a solid theoretical foundation for understanding and optimizing the data mixture for models of varying sizes.

#### Experiments on models of different sizes

Considering computational overhead, for the 7-byte model, we adopted an experimental design similar to REGMIX(Liu et al. 2024), training with 500B tokens in the first stage and 200B tokens in the second stage. Table 3 presents the experimental results of our method on models of different scales. It can be observed that our proposed method significantly outperforms the current state-of-the-art approach, REGMIX, on both in-domain and out-of-domain benchmarks. The performance on the 7B model effectively demonstrates the scalability of our approach. Furthermore, we note that unlike the 1B model, the 7B model’s performance on the benchmarks consistently improves throughout the training process. This suggests that the advantage of TiKMiX could be even more pronounced with additional training data.

#### Observation of data mixing with Group Influence

To conduct a rigorous analysis of inter-domain interactions during mixed training, we designed an experiment to test the principle of influence additivity. Our hypothesis was that the influence of a mixed dataset on a validation set could be accurately predicted by a weighted sum of the influences from its individual constituent domains. To verify this, we first established a baseline mixing recipe using our TiKMiXD method. We then systematically explored the local space around this recipe by generating 256 perturbed configurations, created by applying a random scaling factor between 0.5 and 2.0 to each domain’s original proportion. After filtering out two sampling outliers, we proceeded with 254 unique data mixture configurations. For each of these 254 points, we sampled a corresponding 0.1B token dataset and measured its direct influence. We then compared this empirical influence value against a predicted influence, which was calculated by summing the pre-computed influences of each individual domain, weighted by their respective proportions in the mixture. As depicted in Fig 13 , this comparison revealed a strong linear correlation. Specifically, the Pearson correlation coefficients on the ARC(Clark et al. 2018), Hellaswag(Zellers et al. 2019), and TriviaQA(Joshi et al. 2017) benchmarks reached 0.845, 0.848, and 0.931, respectively, all of which are statistically highly significant (p < 0.0001). This result provides compelling evidence that the outcome of data mixing is highly predictable and can be modeled as a linear combination of inter-domain influences. Consequently, this finding offers a solid empirical justification for the theoretical soundness of our proposed two-stage optimization framework, encompassing both TiKMiX-D and TiKMiX-M.

Figure 5: The impact of domains on a 1B model’s performance on the ARC benchmark as training progresses.

[Figure 25]

Figure 6: The impact of domains on a 1B model’s performance on the HELLASWAG benchmark as training progresses.

[Figure 26]

- Figure 7: The impact of domains on a 1B model’s performance on the MMLU benchmark as training progresses.

- Figure 8: The impact of domains on a 1B model’s performance on the TRIVIAQA benchmark as training progresses.

[Figure 28]

Figure 9: The impact of domains on a 7B model’s performance on the ARC benchmark as training progresses.

[Figure 29]

Figure 10: The impact of domains on a 7B model’s performance on the HELLASWAG benchmark as training progresses.

[Figure 30]

- Figure 11: The impact of domains on a 7B model’s performance on the MMLU benchmark as training progresses.

[Figure 31]

- Figure 12: The impact of domains on a 7B model’s performance on the TRIVIAQA benchmark as training progresses.

[Figure 32]

[Figure 33]

Figure 13: A Group Influence-based Analysis of Data Mixing Effects on Various Benchmarks.

