## DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling

Pala Tej Deep1, Rishabh Bhardwaj1, Soujanya Poria1 1 Singapore University of Technology and Design

# arXiv:2406.11617v1[cs.CL]17Jun2024

Abstract

With the proliferation of domain-specific models, model merging has emerged as a set of techniques that combine the capabilities of multiple models into one that can multitask without the cost of additional training. In this paper, we propose a new model merging technique, Drop and rEscaLe via sampLing with mAgnitude (DELLA-Merging), that employs a novel pruning technique, MAGPRUNE, which shows significant advantages over DARE and TIES. MAGPRUNE first ranks the parameters in order of their magnitude and assigns higher dropout probabilities (p) to parameters with lower ranks corresponding to lower magnitudes. To approximate the original embeddings, MAGPRUNE employs a rescaling operation on the parameters that survive the random dropping by 1/(1 − p). On three different expert models considered for merging (LM, Math, Code) and corresponding benchmark datasets (AlpacaEval, GSM8K, MBPP), DELLA shows an average improvement of

- 2.4 points over baseline methods employing delta parameter pruning (an improvement of
- 3.6 points over TIES, 1.2 points over DARE), and 11.1 points over the no-pruning baseline (TA). We release the source code at: https: //github.com/declare-lab/della.

### 1 Introduction

Interactive systems based on general-purpose LLMs have become widely popular due to their impressive instruction-following capabilities (OpenAI, 2023). Furthermore, tuning these models on downstream tasks has been shown to transform them into domain experts (Rozière et al., 2023; Luo et al., 2023).

Maintaining separate fine-tuned models for each task presents several limitations, such as a significantly higher memory footprint and the inability to leverage information across tasks, which could enhance both in-domain and out-of-domain performance. As a result, merging different homologous

models (models fine-tuned from the same backbone) is gaining traction for its cost-effectiveness, knowledge sharing, and space efficiency (Yadav et al., 2024; Yu et al., 2023). The homologous models differ from each other in terms of delta parameters, i.e., the difference between the fine-tuned model and backbone model parameters.

In this paper, we introduce a novel approach for merging homologous models, termed Drop and rEscaLe via sampLing with mAgnitude (DELLA). This approach consists of three steps: (Step-1) involves delta parameter drops to reduce interference among model parameters. We propose MAGPRUNE, a novel pruning method that samples delta parameters based on their magnitudes; (Step-2) further reduces interference through sign-based delta parameter selection; and (Step-3) fuses the selected delta parameters.

On three different homologous (expert) models considered for merging (LM, Math, Code) and their corresponding benchmark datasets (AlpacaEval, GSM8K, MBPP), DELLA outperforms baseline methods (DARE, TIES, TA) in three out of four merges. It achieves an average improvement of 2.4 points over baseline methods employing delta parameter pruning (3.6 points over TIES, and 1.2 points over DARE), and 11.1 points over the method without pruning (TA).

Crucial to DELLA merging, we demonstrate the importance of MAGPRUNE. We observe that DELLA outperforms the baselines (RANDOM in DARE, TOPK in TIES, and NODROP in TA) in 5 out of 9 domain-specific performance metrics for merged models, as well as in 2 out of 4 aggregated benchmark scores (Avg.), establishing it as the most effective pruning approach.

We also demonstrate the importance of scaling the unpruned delta parameters. Scaling in MAGPRUNE improves the performance of DELLA by 7.6 points on the Math+Code model. When the impact of scaling is studied on individual model per-

Generalist Model

| | | |
|---|---|---|
| | | |
| | | |

LMMathCode

SFT

Math+Code+LM

| | | |
|---|---|---|
| | | |
| | | |

SFT

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |
| | | |

SFT

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Drop (MagPrune)

Fuse

Elect

Merged model

Experts Pruned-Experts

Elected Deltas

Figure 1: Methodology: Three Steps involved in DELLA. First step performs magnitude-based sampling of delta parameters (MAGPRUNE), second step elects the parameters that will undergo merging operation, and the final step (Fuse) performs merging.

[Figure 1]

[Figure 2]

formance, we observe an average improvement of 28.4 and 28.7 points on RANDOM (in MAGPRUNE) and MAGPRUNE (in DELLA), respectively.

delta parameter pruning taking into account their magnitude. For each node in the neural architecture, we map its delta parameters {δ1,...,δn} to drop probabilities Pd = {p1,...,pn} in the inverse order of their magnitudes as depicted follows:

[Figure 3]

### 2 Methodology

We denote the expert (homologous) models by Mt with parameters θt where t signifies their task expertise. Mt is obtained from task-specific supervised fine-tuning (SFT) of a base model M which is common across experts. Let θ denote its parameters. We define delta parameters as the difference between expert and base model parameters, δt := θt − θ.

[Figure 4]

#### 2.1 DELLA

Figure 2: Mapping weights to inversely proportional to drop probabilities.

The model merging task is to combine multiple experts so that without an additional training phase, the obtained model inherits the qualities of all the experts. DELLA consists of three steps, 1) Drop: a delta parameter dropping method, that includes a novel magnitude-based pruning approach MAGPRUNE, 2) Elect: Elect the delta parameters that will take part in merging, and 3) Fuse: Perform element-wise addition of delta parameters. The merging process is depicted by Figure 1.

where ∆ is a hyperparameter such that pi+1 − pi = ∆, we further discuss the choice of hyperparameters in Section 2.2.

(Scaling)—To compensate for the effect of pruning on the model’s embedding, we rescale the remaining delta parameters by 1/(1 − pi) where pi is the probability of drop of weight δi. We relegate the theoretical analysis of magnitude-based stochastic dropping and scaling to Section 2.2. We provide empirical evidence proving the importance of scaling in Table 3.

Step-1: Drop. In this step, we set a proportion of delta parameters to zero thus nullifying their role in the expert model 1. Drop is a crucial step of DELLA that employs pruning, aimed at lessening the interference between the expert models while preserving the task-specific performance. In this work, we propose a new approach MAGPRUNE that performs

Step-2: Elect. Elect refers to selecting delta weights that will undergo the merging operation to minimize further interference caused by merging. Unlike stochastic Drop, Elect reduces the interference by minimizing directional discrepancy in delta parameters. First, we identify the dominating direc-

1Dropping a parameter δit would mean the parameter value θit equals base value θi.

tion by noticing the sign of the sum of all the delta parameters S = sgn( Tt=1 δkt) at position k. Thus, the elected delta parameters for position k are ones carrying the same sign as S:

C = {t ∈ [T] | sgn(δkt) = S},

where C is the index of such delta parameters.

Step-3: Fuse. We obtain the merged delta parameter at position k by computing the average of elected delta parameters, i.e.,

δkavg =

δki

i∈C

However, similar to Step 1, to compensate for the drop in model parameters, we rescale the delta values to obtain the final merged model θm:

θm = θ + λ ∗ δavg, (1)

where λ is a scalar value. We provide an analysis on λ in Figure 4.

- 2.2 MAGPRUNE: Stochastic Magnitude-based Pruning

As a critical step of merging, we detail the sampling step. For a set of delta parameters in an expert model δ1,δ2,...δn, first we rank them followed by assigning them to corresponding drop probabilities 2:

{r1,r2,...,rn} = rank({δ1,δ2,...δn}) ∆i =

ϵ n ∗ ri

pi = pmin + ∆i

where ri ∈ {0,1,...,n − 1} and pi is the rank, and drop probability of ith delta; pmin is the minimum probability of dropping assigned to maximum magnitude delta parameter such that

ϵ 2

pmin = p −

where p is the average drop probability. Next, we perform the sampling step:

mi ∼ Bernoulli(pi), δ˜i = (1 − mi) ⊙ δi, δˆi =

δ˜i 1 − pi

.

2We perform the ranking of the delta parameters row-wise fashion, as it performed better than layer-wise ranking as seen in Figure 6.

Here mi value 1 would denote δi is dropped, 1 − pi denotes rescaling which is effective to all the undropped delta parameters. Next, we provide the theoretical justification of the scaling factor.

DARE and TIES as configuration of DELLA. Notably, in MAGPRUNE, while we define Ps with step probabilities ∆, one can freely choose any set of valid probabilities p′is to map δ′s, to explore more possibilities of dropping. To obtain pruning used in DARE, one can choose all pi’s to be the same. To obtain pruning used in TIES 3, one can assign 1 to K elements in Pd and 0 to others.

Algorithm 1 Drop-Elect-Merge with DELLA. Input: Fine-tuned model parameters {θt}Tt=1, Pre-

trained model parameters θ, p, λ and ϵ. Output: Merged Model Parameters θm

for all t in 1,...,T do

- ▷ Step 0 Get Delta parameters δt = θt − θ
- ▷ Step 1: Drop δˆt ← MAGPRUNE(δt,p,ϵ) γt ← sign(δˆt) µt ← |δˆt| end for
- ▷ Step 2: Elect γm = sign( Tt=1 δˆt)
- ▷ Step 3: Merge for all k in 1,...,d do

Ck = {t ∈ [T]|γkt = γkm} δkm = |C1

k| t∈Ck δˆkt end for

- ▷ Step 4: Obtain merged checkpoint θm ← θ + λ ∗ δm return θm

#### 2.3 Theoretical Analysis

We analyze the impact of Drop step on the linear transformations. Let h denote a node in the network with weights w1,...,wn and corresponding delta parameters δ1,...,δn, the input embedding vector be {x1,...,xn}. We can define expected output

3Keep top-K magnitudes.

embedding as:

E[h] = E

n

=

i=1

n

=

i=1

n

(wi + δi)xi

i=1

n

xiE[wi] +

xiE[δi]

i=1

n

xiwi +

xiδi = h + ∆h

i=1

In Step 1 of DELLA, we drop the delta parameters with a probability pi and rescale others by γ, the expected output of the node can be identified as

n

E[hˆ] = E

wi + δiˆ xi

i=1

n

n

xiE[δiˆ ]

xiE[wi] +

=

i=1

i=1

n

n

xi(1 − pi) ∗ γ ∗ δi

=

xiwi +

i=1

i=1

n

xi ∗ (1 − pi) ∗ γ ∗ δi

= h +

i=1

If we put γ = 1/(1 − pi), we get,

n

E[hˆ] = h +

xi ∗ δi

i=1

E[hˆ] = h + ∆h

Thus, similar to DARE, DELLA can preserve model performance by approximating the original embeddings, yet being a more generic (superset) pruning approach. Algorithm 1 outlines the steps of DELLA with the Drop-Elect-Merge approach.

### 3 Experimental Setup

Expert Models. We experiment with the Wizard model-based experts. WizardLM (LM-expert) (Xu et al., 2024) for instruction following, WizardMath (Luo et al., 2023) for math (Math-expert), and WizardCoder-Python for code (Code-expert). Specific to model merging, a primary requirement is to have experts obtained from the same backbone model. Since WizardCoder-Python (Luo et al., 2024) uses a different pre-trained backbone (CodeLlama), we use llama-2-13b-code-alpaca 4 as code expert which shares a common backbone with WizardLM and WizardMath i.e. LLaMA-2.

4https://huggingface.co/layoric/llama-2-13b-code-alpaca

Baselines. We compare DELLA with two state-ofthe-art merging approaches: DARE and TIES. For valid comparisons, we keep Step-2 (Elect) and Step-3 (Merge) consistent across all baselines and compare them in Step-1, which involves interference resolution in delta parameters.

- • TA (Ilharco et al., 2023)—Task Arithmetic (TA) is a simple delta weight merging without Step-1 and 2. However, to provide a valid comparison, we employ Step-2 i.e. Elect. We refer to the absence of dropping by NODROP.
- • DARE (Yu et al., 2023)— DARE randomly drops the delta parameter at a uniform drop rate, p, and rescales the remaining values by 1/(1 − p), referred to as RANDOM.
- • TIES (Yadav et al., 2024)— TIES retains only the top-K parameters with the highest magnitudes in each layer, referred to as TOPK.
- • DELLA (Ours)— In Step-1, DELLA employs MAGPRUNE that first ranks delta magnitudes (row-wise), followed by assigning them to a predefined set of probabilities.

(Hyperparameter Search)—To achieve the best performance with all the baseline approaches, we conducted a hyperparameter search to determine the suitable values of the drop rate p and rescaling factor, λ, wherever applicable. We performed merging by varying drop rates p in [0.1, 0.3, 0.5, 0.7, 0.9] and selected the optimal p for each approach to compare the final performance. We evaluated the merged models’ performance by evaluating them on all the tasks of its base models. We set the temperature parameter to 0.0 for greedy decoding and set the maximum number of generated tokens to 512. We used the mergekit library (Goddard et al., 2024) to perform the pruning and merging operations on the model parameters.

(Casting DELLA as DARE and TIES)Essentially, DELLA encompasses NoDrop, DARE, TIES, and rank-proportional probability distribution. DELLAp=0 is essentially no dropping of parameters given the probability of dropping each parameter sampled is 0; DELLAϵ=0 boils down to DARE with all the parameters having the same probability pa of getting sampled (mentioned in Section 2.2). DELLAp=0/1 refers to mapping the ranked delta magnitudes to probabilities where k probabilities are 1s and rest are (n − k) are 0s.

[Figure 5]

[Figure 6]

[Figure 7]

Random MegPrune NoDrop

Random MegPrune NoDrop

Random MegPrune NoDrop

- Figure 3: Performance vs Drop Rate comparison of DELLA (magnitude-based random drop) against baselines DARE (random drop) and TIES (magnitude-based deterministic drop).

Merging Models Method AlpacaEval GSM8K MBPP Avg.

DELLAp∝r is magnitude proportional sampling of delta parameters.

LM - 80.8 45.9 31.4 / Math - / 63.5 / / Code - / / 27.0 /

Single

TA (DELLAp=0)

Evaluation Metrics. We used AlpacaEval (Li et al., 2023) to evaluate the performance of models on the instruction-following task, GSM8K to evaluate a model’s performance on mathematical reasoning (Cobbe et al., 2021), and MBPP (Austin et al., 2021) for evaluation of code generation. For AlpacaEval, we compare the model-generated answer against the reference answer using GPT4 as an evaluator to determine the win rate of the model against the reference solution. For MBPP, we use the Pass@1 score to measure the performance of the model’s code generation capabilities. Lastly, for GSM8K, we use GPT4-as-a-judge to evaluate the mathematical correctness i.e. the final answer as well as the intermediate steps (reasoning).

79.1 60.6 / 69.9

DARE (DELLAϵ=0)

81.3 63.3 / 72.3

LM + Math

TIES (DELLAp=0/1)

77.6 66.8 / 72.2

DELLAp∝r 81.8 64.1 / 72.9

TA (DELLAp=0)

73.3 / 0 36.7

DARE (DELLAϵ=0)

79.4 / 32.8 56.1

LM + Code

TIES (DELLAp=0/1)

77.4 / 9.6 43.5

Merge

##### DELLAp∝r 79.7 / 35.2 57.5

TA (DELLAp=0)

/ 63.2 15.6 39.4

DARE (DELLAϵ=0)

/ 61.9 20.0 41.0

Math + Code

TIES (DELLAp=0/1)

##### / 65.9 20.8 43.4

DELLAp∝r / 64.9 20.0 42.4

TA (DELLAp=0)

72.1 48.3 0 40.1

Experiments (w/o Elect). We also cast TA, DARE, TIES, and DELLA in a setup where we remove the participation of Step-2 i.e. Elect. This setting is the standard setting used in TA (Ilharco et al., 2023) and DARE(Yu et al., 2023).

DARE (DELLAϵ=0)

LM + Math + Code

77.5 62.7 29.4 56.5

TIES (DELLAp=0/1)

77.5 66.9 27.2 57.2

DELLAp∝r 80.4 61.8 31.4 57.9

Table 1: DELLA against baseline approaches in DropElect-Fuse Setup.

Experiments (only Drop). We also compare different pruning methods, RANDOM, TOPK, and MAGPRUNE, directly on expert models.

the inference from other experts (LM and Code), despite considerable performance retention even after pruning 90% of the delta parameters.

### 4 Results

Merging. Shown in Table 1, DELLAp∝r outperforms the baselines DARE, TIES, and TA in three out of four merge settings: LM+Math (↑ 0.6 points), LM+Code (↑ 1.4 points), and LM+Math+Code (↑ 0.7 points). It is on the GSM8K (Math) task where merging models using DELLAp∝r, as well as DARE, do not tend to work as well as TIES. We posit this is due to the randomness in the delta parameter dropping that causes performance drops. This is likely because the coding expert does not have enough task-prominent parameters to sustain

Using DELLAp∝r, merging the LM-expert with other experts (LM+<other-experts>) is shown to maintain performance on the AlpacaEval task, scoring within approximately ±1 point of the LMexpert’s score on the task (80.8), with an average drop of ↓0.2 points. The average drop in performance using TIES and DARE merging is observed to be higher, i.e., ↓3.3 points and ↓1.4 points, respectively. Without employing pruning Step-1, i.e., TA setting observes the highest performance degra-

Method GSM8K MBPP Avg. DELLA(w/o scaling) 42.2 25.6 33.9 DELLA(w/ scaling) 63.8 (↑ 21.6) 19.2 (↓ −6.4) 41.5 (↑ 7.6)

Table 2: Effect of scaling on merging (at λ = 1).

[Figure 8]

- Figure 4: Performance vs lambda for the math+code merge combination

dation of ↓5.9 across baselines. Merging the Mathexpert with others (Math+<other-experts>) generally improves performance on GSM8K (DELLA shows an average increase of ↑0.1 points), whereas the three-expert setting is observed to face interference, diminishing the scores post-merging. Merging the Code-expert with others (Code+<otherexperts>) shows an increase in performance in two out of three merges, with an average improvement of ↑2.1 points on MBPP Python problems compared to the Code-only expert. Whereas DARE observes a smaller increase of ↑0.4 points and TIES faces a decrease in score by ↓-7.8 points. TA’s scored zero on the MBPP task whenever Codeexpert is merged with LM-expert.

A similar case occurs with TIES, which shows a significant drop of 17.4 on the MBPP coding task in the LM+Code merge setting. Upon closer examination, we observe a greater influence of the LM-expert on the model, making the outputs more chat-like rather than structured code-like. We believe scaling (not present in TIES) plays an important role here; not scaling (up) the parameters after dropping can cause dominating effects, where one expert overshadows the qualities of the other.

Effect of scaling. To understand the importance of 1/(1 − p) scaling in Step-1, we prune expert models5 using different drop methods and under-

5Unless mentioned, DELLA refers to DELLAp∝r.

Sparsity (p)

DARE (w/o scaling)

DARE (w/ scaling)

DELLA (w/o scaling)

DELLA (w/ scaling)

0.3 39.7 48.5 (↑ 8.8) 42.1 47.3 (↑ 5.2) 0.5 33.7 47.3 (↑ 13.6) 34.9 48.5 (↑ 13.6)

- 0.7 13.0 43.3 (↑ 30.3) 19.7 48.8 (↑ 29.1)
- 0.8 2.4 45.5 (↑ 43.1) 2.4 49.1 (↑ 46.7)
- 0.9 0 46.4 (↑ 46.4) 0.3 49.1 (↑ 48.8) Table 3: Effect of scaling on expert’s performance.

stand the scaling effect on individual models’ performance as well as merged models. Table 2 shows the effect of scaling after Step-1 on the Math+Code model, we observe the scaling raises the score on the coding task (GSM8K) with an increase in score by ↑21.6 points while a ↑6.4 point reduction was observed on MBPP math task. Overall, scaling improved the average score of the merged experts by ↑7.6 points. A similar observation is made when we analyse the importance of scaling on pruned expert models with at drop rate p Table 3. We observe an average of improvement by ↑28.4 and ↑28.7 points on DARE and DELLA, respectively.

Results on w/o Elect. In Table 4, we focus on comparing the effect of different pruning approaches on merging, thus skipping Step 2, i.e., Elect. On average scores, we observe that DELLA outperforms the baselines in 5 out of 9 merged model-benchmark task settings, as well as 2 out of 4 times on aggregated benchmark scores (Avg.). We also notice that the performance drop due to omitting Step 2 (Elect) is the least for DELLA compared to the other baselines. This shows MAGPRUNE drop in DELLA intrinsically facilitates electing the parameters.

How effective is MAGPRUNE? We study the effect of MAGPRUNE on individual expert models (Figure 3). We find MAGPRUNE consistently outperforms TOPK on math task GSM8K with a wider performance gap at higher pruning ratios. A similar observation is made on the AlpacaEval task where except for the pruning ratio 0.7, MAGPRUNE outperforms TOPK. MAGPRUNE outperforms RANDOM on AlpacaEval consistently except for p = 0.8. On math task MBPP, we observe an inconsistent trend, making it hard to identify a comprehensive winner. One thing to note from GSM8K and AlpacaEval results is that MAGPRUNE becomes more effective at higher ratios p ≥ 0.8, preserving more expert-specific capabilities while identifying more crucial parameters to solve the task.

[Figure 9]

- Figure 5: Adaptive Scaling vs Constant lambda Scaling DELLA

[Figure 10]

Figure 6: Layer-Wise vs Row-Wise Ranking

Adaptive vs Constant λ? Figure 5 shows that applying constant lambda scaling after merging the delta weights (after Fusion step 3) results in greater performance improvements for the merged model compared to adaptively scaling each model’s deltas based on the proportion of parameters elected in Step 2 (effective p after Step 2). Using Constant λ scaling leads to a notable improvement across all merging approaches: DELLA (↑ 1.5 points), DARE (↑ 2.9 points), TIES (↑ 2.3 points). To find the optimal value of the constant λ, we perform a hyperparameter search and select the best performing λ as shown in Figure 4.

Deciding pruning groups. For the ranking in Drop Step 1, we experimented with two different types of ranking: row-wise and layer-wise. In rowwise ranking, we rank each delta parameter of a node (rows of the matrix layer matrix), similarly for layer-wise, we rank it within the entire layer.

- Figure 6 shows that row-wise ranking achieves a higher average score than the layer-wise approach across the 4 merge combinations using DELLAp∝r.

Importance of GPT4-as-a-judge for Math tasks. We perform prompt-based evaluation by prompting GPT4 with the question, reference solution, and model-generated answer to evaluate the correctness

Merging Models Method AlpacaEval GSM8K MBPP Avg.

LM - 80.8 45.9 31.4 / Math - / 63.5 / / Code - / / 27.0 /

Single

TA (DELLAp=0)

73.7 65.2 / 69.5

DARE (DELLAϵ=0)

72.7 64.8 / 68.9

LM + Math

TIES (DELLAp=0/1)

72.1 65.7 / 68.9

DELLAp∝r 75.0 65.5 / 70.2

TA (DELLAp=0)

67.2 / 33.2 50.2

DARE (DELLAϵ=0)

###### 67.6 / 33.0 50.3

LM + Code

TIES (DELLAp=0/1)

64.9 / 34.6 49.8

Merge

DELLAp∝r 65.7 / 34.8 50.2

TA (DELLAp=0)

/ 51.2 7.8 29.6

DARE (DELLAϵ=0)

/ 48.8 7.0 27.9

Math + Code

TIES (DELLAp=0/1)

/ 52.9 5.6 29.3

DELLAp∝r / 48.7 8.6 28.6

TA (DELLAp=0)

62.2 55.8 0 40.1

DARE (DELLAϵ=0)

LM + Math + Code

63.1 55.0 30.2 49.4

TIES (DELLAp=0/1)

63.6 57.7 32.2 51.2

DELLAp∝r 77.5 66.8 29.8 58.0

- Table 4: Comparison of DELLA against baselines in merging w/o Elect setting.

Merging Models Method AlpacaEval GSM8K MBPP Avg.

DropElectMerge

LM + Math

WANDAp=0/1 78.2 63.4 / 70.8 WANDAp∝r 81.3 62.5 / 71.9 DELLAp∝r 81.8 64.1 / 72.9

LM + Code

WANDAp=0/1 54.8 / 5 29.9 WANDAp∝r 80.7 / 30.6 55.6 DELLAp∝r 79.7 / 35.2 57.5

Math + Code

WANDAp=0/1 / 58.6 24 41.3 WANDAp∝r / 63.8 17.8 40.8 DELLAp∝r / 64.9 20.0 42.4

LM + Math + Code

WANDAp=0/1 67.6 57.8 28 51.2 WANDAp∝r 75.4 60.9 28.6 55.0 DELLAp∝r 80.4 61.8 31.4 57.9

- Table 5: DELLA against WANDA in Drop-Elect-Fuse Setup.

(Li et al., 2024). This is a more comprehensive automatic evaluation as compared to the hard-coded parsing approaches that are suboptimal math evaluators. Refer to Appendix A.1 for an example showcasing the effectiveness of using GPT4-as-a-judge for evaluating math tasks.

Using activations to rank delta parameters. Inspired by WANDA, (Sun et al., 2024), we used the product of delta parameter magnitudes and input activations, the higher score of which corresponds to a higher rank of delta parameters in Step 1. The activations are computed for each layer using 128 data samples from GSM8K for the Math-expert (Cobbe et al., 2021), MBPP train

for the Code-expert (Austin et al., 2021), and C4 train dataset (Raffel et al., 2019) for the LM-expert. Table 5 shows the comparison of DELLA with MAGPRUNE pruning with WANDA with TOPK and MAGPRUNE pruning. We observe the DELLA outperforms WANDA in all the task performances and average scores. We also note that WANDA with MAGPRUNE outperforms WANDA with TOPK in three of the four merges but performs 0.5 points worse on the math+code merging.

### 5 Related Work

Supervised Fine-tuning (SFT) is a widely adopted technique in natural language processing, used to equip pre-trained LMs with expert capabilities by optimizing them on task-specific data (Dodge et al., 2020; Zhao et al., 2023). This paradigm has emerged as the de facto standard approach. Generally, Supervised Fine-tuning (SFT) of LMs can be categorized into two distinct methodologies:

- 1. Full Fine-tuning: All parameters of the pretrained LM are updated during the fine-tuning process, allowing the model to adapt extensively to the target task or domain (Radford et al., 2018; Kenton and Toutanova, 2019).
- 2. Parameter-efficient Fine-tuning: updates only a subset of the pre-trained LM’s parameters, thereby preserving the knowledge acquired during the initial pre-training phase while enabling efficient adaptation to the target task or domain (Houlsby et al., 2019; Li and Liang, 2021; Hu et al., 2021; Lester et al., 2021; Liu et al., 2023).

Model pruning techniques are a form of model compression that reduces the number of parameters in a model while maintaining the model’s performance (Zhu and Gupta, 2017; Liu et al., 2018; Frankle and Carbin, 2019; Zhu et al.,

- 2023). Pruning methods employ various strategies to select parameters to retain, including using magnitude-based pruning (Li et al., 2018; Lee et al., 2021), leveraging the Hessian matrix (Frantar and Alistarh, 2023), combining weight magnitudes with input activations (Sun et al.,
- 2024), minimizing discrepancies between source and pruned model parameters (Zhang et al., 2023b), and introducing nonuniform layered sparsity for higher pruning rates (Yin et al., 2023). While these pruning techniques share the goal of parameter reduction, there is a notable difference compared

to DELLA. DELLA focuses on pruning delta parameters instead of fine-tuned model parameters.

Model merging techniques focus on combining multiple task-specific models into a single model with diverse capabilities without requiring the original training data (Zhang et al., 2023a). Widely recognised methods in model merging include Average Merging (Wortsman et al., 2022), which constructs merged models using averaged SFT parameters; Task Arithmetic (Ilharco et al., 2023), which uses domain-specific offsets and pre-defined scaling terms to distinguish the importance of each model being merged; Fisher Merging (Matena and Raffel, 2022), utilizing the Fisher information matrix for weighted parameter fusion (Fisher, 1922); RegMean (Jin et al., 2023), which optimizes a linear regression problem for model merging; and TIES-Merging (Yadav et al., 2024), which resolves conflicts by trimming low-magnitude parameters, resolving sign disagreements, and disjointly merging parameters with consistent signs. Recent approaches such as DARE (Yu et al., 2023) and DPPA (Zhu et al., 2024) aim to reduce interference during merging by dropping delta parameters and rescaling remaining delta parameters to maintain each model’s performance before merging.

This paper focuses its contribution on merging homologous models, where model merging is achieved by manipulating delta parameters. The baselines we use in this paper primarily involve manipulating delta parameters of homologous models.

### 6 Conclusion

We introduced a novel approach for merging homologous models, DELLA. This approach consists of three steps: Drop, Elect and Merge. For the Drop step, We proposed MAGPRUNE, a novel pruning method that samples delta parameters based on their magnitudes. We showed that DELLA encompasses NODROP, DARE and TIES and we can cast DELLA as these methods by adjusting hyperparameters. We found that DELLA achieves an average improvement of 2.4 points over baseline methods using pruning techniques(DARE, TIES) and 11.1 points over the method without pruning(TA). DELLA achieved the highest average score for 5 out of 8 merges and the second-best score for 2 out of 3 remaining merges (Tables 1 and 4).

- 7 Limitations

DELLA has more hyperparameters compared to other baselines like DARE and TIES. While this allows our approach to be more customized to each model merging, it adds additional complexity to finding the optimal merging. Due to computational constraints and a limited number of on-the-shelf models available, we could not extensively evaluate models with different backbones and sizes. Similar to DARE, TIES and TA, our approach is only effective for models with the same backbone model. Some potential future work could involve exploring applications of these techniques for merging models with different pre-trained backbones.

- 8 Potential Risks

LLMs can be used for harmful content generation and for spreading misinformation.

- 9 Ethical Considerations Not applicable.

References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. 2020. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305.

Ronald A Fisher. 1922. On the mathematical foundations of theoretical statistics. Philosophical transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character, 222(594-604):309–368.

Jonathan Frankle and Michael Carbin. 2019. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations.

Elias Frantar and Dan Alistarh. 2023. SparseGPT: Massive language models can be accurately pruned in one-shot. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of

Proceedings of Machine Learning Research, pages 10323–10337. PMLR.

Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vlad Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz. 2024. Arcee’s mergekit: A toolkit for merging large language models. arXiv preprint arXiv:2403.13257.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2023. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations.

Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. 2023. Dataless knowledge fusion by merging weights of language models. In The Eleventh International Conference on Learning Representations.

Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of naacL-HLT, volume 1, page 2.

Jaeho Lee, Sejun Park, Sangwoo Mo, Sungsoo Ahn, and Jinwoo Shin. 2021. Layer-adaptive sparsity for the magnitude-based pruning. In International Conference on Learning Representations.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691.

Guiying Li, Chao Qian, Chunhui Jiang, Xiaofen Lu, and Ke Tang. 2018. Optimization based layer-wise magnitude-based pruning for dnn compression. In International Joint Conference on Artificial Intelligence.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval.

Zhen Li, Xiaohan Xu, Tao Shen, Can Xu, Jia-Chen Gu, and Chongyang Tao. 2024. Leveraging large language models for nlg evaluation: A survey. arXiv preprint arXiv:2401.07103.

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. 2023. Gpt understands, too. AI Open.

Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. 2018. Rethinking the value of network pruning. arXiv preprint arXiv:1810.05270.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2024. Wizardcoder: Empowering code large language models with evolinstruct. In The Twelfth International Conference on Learning Representations.

Michael S Matena and Colin A Raffel. 2022. Merging models with fisher-weighted averaging. In Advances in Neural Information Processing Systems, volume 35, pages 17703–17716. Curran Associates, Inc.

OpenAI. 2023. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training.

Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1–140:67.

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2023. Code llama: Open foundation models for code.

Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. 2024. A simple and effective pruning approach for large language models. In The Twelfth International Conference on Learning Representations.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. 2022. Model soups: averaging weights of multiple fine-tuned models improves

accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. 2024. WizardLM: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raffel, and Mohit Bansal. 2024. Ties-merging: Resolving interference when merging models. Advances in Neural Information Processing Systems, 36.

Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, Yaqing Wang, Yiling Jia, Mykola Pechenizkiy, Yi Liang, Zhangyang Wang, and Shiwei Liu. 2023. Outlier weighed layerwise sparsity (owl): A missing secret sauce for pruning llms to high sparsity. arXiv preprint arXiv:2310.05175.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2023. Language models are super mario: Absorbing abilities from homologous models as a free lunch. arXiv preprint arXiv:2311.03099.

Jinghan Zhang, shiqi chen, Junteng Liu, and Junxian He. 2023a. Composing parameter-efficient modules with arithmetic operation. In Advances in Neural Information Processing Systems, volume 36, pages 12589–12610. Curran Associates, Inc.

Yuxin Zhang, Lirui Zhao, Mingbao Lin, Yunyun Sun, Yiwu Yao, Xingjia Han, Jared Tanner, Shiwei Liu, and Rongrong Ji. 2023b. Dynamic sparse no training: Training-free fine-tuning for sparse llms. arXiv preprint arXiv:2310.08915.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

Michael Zhu and Suyog Gupta. 2017. To prune, or not to prune: exploring the efficacy of pruning for model compression. arXiv preprint arXiv:1710.01878.

Xunyu Zhu, Jian Li, Yong Liu, Can Ma, and Weiping Wang. 2023. A survey on model compression for large language models. arXiv preprint arXiv:2308.07633.

Yaochen Zhu, Rui Xia, and Jiajun Zhang. 2024. Dppa: Pruning method for large language model to model merging. arXiv preprint arXiv:2403.02799.

### A Appendix

A.1 Importance of GPT4-as-a-judge for Math tasks - Example

Question: Each person in a certain household consumes 0.2 kg of rice every meal. Supposing 5 members of the household eat rice every lunch and dinner, how many weeks will a 42 kg bag of rice last?

Generated Answer: 1. First, we need to determine the total amount of rice consumed by the household in one day. Since each person consumes 0.2 kg of rice per meal and there are 5 people in the household, the total amount of rice consumed in one day is:

0.2 kg/person/meal * 5 people * 2 meals/day = 2 kg/day

2. Next, we need to calculate the total amount of rice consumed in a week. There are 7 days in a week, so the total amount of rice consumed in a week is:

- 2 kg/day * 7 days/week = 14 kg/week
- 3. Finally, we need to determine how many weeks a 42 kg bag of rice will last. To do this, we will divide the total weight of the bag by the total amount of rice consumed in a week: 42 kg / 14 kg/week = 3 weeks

So, a 42 kg bag of rice will last for 3 weeks for a household of 5 people who eat rice at every meal.

GPT4 Evaluation Comment: The model’s response is correct. It followed the correct process of determining the daily consumption of rice, converting that to weekly consumption, and then determining the number of weeks a 42 kg bag of rice would last under those conditions. The computation aligns correctly with the reference solution.

Given the example above, a hard-coded parser would take the last number in the generated answer, 5, as the final answer and mistakenly label the answer as incorrect. In contrast, GPT4 acts as a smart parser and correctly identifies the final answer as

- 3 and labels it as correct. We also note that GPT-
- 4 evaluated the intermediate steps in the solution

to ensure their correctness. This demonstrates the limitations of inflexible parsing methods and the benefits of using language models that understand the context for automatic evaluation.

#### A.2 Pruning Experiments

We compare the performance of DELLA against the DARE baseline to show that magnitude sampling improves the selection of delta parameters to retain and better maintain the model’s task performance. We vary the drop rate p in [0.3, 0.5, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94] and apply the DARE and DELLA to get models after removing the proportion of delta parameters. We then evaluate the model’s performance on its corresponding SFT task. Table 6 shows the comparison between DARE, random ranking and MAGPRUNE. We performed experiments with random ranking where we assign ranks to parameters randomly in step1. Comparing MAGPRUNE with random ranking would enable us to see the effectiveness of using delta parameter magnitudes as a ranking method. Overall, MAGPRUNE performs better than DARE and random ranking on average by 1.3 points and 1.6 points. Similar to the results from the merging experiment, row-wise MAGPRUNE performs better than layerwise by 1 point.

|Sparsity<br><br>|DARE<br><br>MAGPRUNE (row)<br><br>MAGPRUNE (layer)<br><br>random rank (row)<br><br>random rank (layer)|
|---|---|
|0.3 0.5 0.7 0.8 0.9 0.91 0.92 0.93 0.94|0.636 0.652 0.642 0.630 0.642 0.627 0.648 0.633 0.655 0.621 0.615 0.642 0.612 0.636 0.615 0.658 0.645 0.624 0.655 0.621 0.618 0.639 0.621 0.621 0.618 0.633 0.618 0.639 0.633 0.609 0.615 0.630 0.618 0.588 0.652 0.652 0.648 0.642 0.591 0.642 0.597 0.645 0.642 0.555 0.600|

- Table 6: Wizardmath-13B-V1.0 Row-Wise vs LayerWise ranking Pruning Results

A.3 Pruning Rate Hyperparameter Search For Model Merging

- Table 7 shows the results of the pruning rate hyperparameter search for each merging combination. While both MAGPRUNE and DARE can maintain the performance of individual expert model performance up to a high drop rate of 0.9, our findings indicate that a drop rate of 0.5, works best for LM+Math, Math+Code and LM+Math+Code. For LM+Code, a drop rate of 0.7 is optimal. Thus, we can infer that while dropping delta parameters helps reduce interference during merging, dropping too many parameters may lead to the loss of

information useful for effective merging.

|Models|Drop rate|AlpacaEval<br><br>|GSM8K|MBPP|Average|
|---|---|---|---|---|---|
|LM + Math|0.1 0.3 0.5 0.7 0.9|0.805 0.812 0.804 0.787 0.683|0.599 0.629 0.645 0.611 0.455|/ / / / /<br><br>|0.702 0.721 0.724 0.699 0.570|
|LM + Code<br><br>|0.1 0.3 0.5 0.7 0.9|0.741 0.770 0.802 0.798 0.737|/ / / / /|0 0 0.152 0.34 0.262|0.370 0.385 0.477 0.569 0.500|
|Math + Code|0.1 0.3 0.5 0.7 0.9|/ / / / /|0.619 0.618 0.626 0.633 0.622|0.166 0.184 0.206 0.19 0.128|0.393 0.401 0.416 0.412 0.375<br><br>|
|LM + Math + Code|0.1 0.3 0.5 0.7 0.9|0.732 0.766 0.794 0.770 0.688|0.545 0.623 0.630 0.622 0.446|0.114 0.302 0.3 0.23 0.128|0.464 0.564 0.575 0.541 0.421|

Table 7: Drop Rate of parameters against Task performance

