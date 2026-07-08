# arXiv:2605.07850v1[cs.CL]8May2026

## MATRYOSHKALORA: Learning Accurate Hierarchical Low-Rank Representations for LLM Fine-Tuning

Ionut-Vlad Modoranu∗

ISTA

Mher Safaryan Lancaster University, UK

Dan Alistarh ISTA

### Abstract

With the rise in scale for deep learning models to billions of parameters, the computational cost of fine-tuning remains a significant barrier to deployment. While Low-Rank Adaptation (LoRA) has become the standard for parameter-efficient fine-tuning, the need to set a predefined, static rank r requires exhaustive grid searches to balance efficiency and performance. Existing rank-adaptive solutions such as DyLoRA mitigate this by sampling ranks during the training from a predefined distribution. However, they often yield sub-optimal results at higher ranks due to lack of consistent gradient signals across the full hierarchy of ranks, thus making these methods data-inefficient. In this paper, we propose MATRYOSHKALORA, a general, Matryoshka-inspired training framework for LoRA that learns accurate hierarchical low-rank representations by inserting a fixed, carefully crafted diagonal matrix P between the existing LoRA adapters to scale their sub-ranks accordingly. By introducing this simple modification, our general framework recovers LoRA and DyLoRA only by changing P and ensures all sub-ranks embed the available gradient information efficiently. Our MATRYOSHKALORA supports dynamic rank selection with minimal degradation in accuracy. We further propose Area Under the Rank Accuracy Curve (AURAC), a metric that consistently evaluates the performance of hierarchical low-rank adapters. Our results demonstrate that MATRYOSHKALORA learns more accurate hierarchical low-rank representations than prior rank-adaptive approaches and achieves superior accuracyperformance trade-offs across ranks on the evaluated datasets. Our code is available at https://github.com/IST-DASLab/MatryoshkaLoRA.

### 1 Introduction & Related Work

The research advancements in the Deep Learning community allowed researchers and practitioners to train models with billions of parameters. Building and managing pipelines for such a complex system requires a significant engineering effort, from preparing the dataset to scaling the model and the optimizer states via multi-dimensional parallelization techniques across many GPUs in a cluster or even multiple physical clusters - a process spanning over months. Given the costs, these large models are rarely deployed to production-ready environments as they are; instead, they serve as a knowledge base for downstream adaptation, such as fine-tuning, which still remains computationally prohibitive for many applications and settings.

[Figure 1]

To alleviate this overhead, LoRA [9] has emerged as the defacto standard for parameter-efficient fine-tuning (PEFT).

Figure 1: MATRYOSHKALORA

∗Institute of Science and Technology Austria (ISTA). Correspondence to ionut-vlad.modoranu@ista.ac.at

Preprint.

However, it introduces a significant structural constraint: the rank r must be predefined. Finding the optimal balance between parameter efficiency and model performance currently requires exhaustive search across multiple training runs, each with a different static rank.

One line of work studies adaptive-rank LoRA methods, which optimize how rank capacity is allocated across layers or modules under a given or learned parameter budget. These methods improve parameter efficiency, but they typically produce a single specialized adapter configuration rather than a nested adapter family that can be sliced at inference time [23, 6, 14, 24, 22, 17, 5, 11].

- A second direction studies dynamic-rank LoRA methods, which aim to train a single adapter whose prefixes are independently usable as lower-rank adapters. Unlike adaptive allocation methods, these approaches are designed to provide multiple sub-adapters from one checkpoint [18, 16]. A conceptually related, but not directly comparable, line of work is slimmable [20] and once-for-all networks [2]. These are not LoRA methods, but they provide the broader once-for-all training paradigm: a single set of weights is trained so that many nested sub-networks are valid deployment choices [19].

Our work can be interpreted as translating this paradigm from channel width to LoRA rank and belongs primarily to the second cluster. It differs from adaptive-rank allocation methods because it does not merely identify one rank configuration; it trains a nested adapter family. It differs from general slimmable networks because the nested structure is imposed on low-rank adapter factors rather than full-model channels. Its closest comparison is DyLoRA [18], which also targets dynamic rank usage but enforces rank hierarchy through a different training mechanism discussed next.

DyLoRA [18] is an alternative solution to grid-searching for the rank r for LoRA. At each training step, a rank k ∈ N is randomly sampled from a predefined distribution and the forward pass is performed using only the first k rows and first k columns of adapters A and B, respectively, denoted by Ak and Bk. The loss is computed with respect to the output of the network that was generated using only Ak and Bk for all linear layers. The purpose of this approach is to train adapters A and B that still yield high accuracy when using only a slice of size k on the fly. DyLoRA solves the issue of adaptive rank r only partially because, as we will show in our experimental section, the accuracies of the models fine-tuned with DyLoRA do not exhibit a true hierarchical pattern in the context of reasoning tasks, such as math datasets. We empirically show that DyLoRA strategy is sub-optimal because it constrains the learning only to a fixed rank k sampled at random, while the ranks r > k do not receive any gradient signal.

Our work proposes Matryoshka-style training framework for LoRA, inspired from Matryoshka Representation Learning [12], which we call MATRYOSHKALORA. It is motivated by the drawback of DyLoRA, which fails to learn what we call hierarchical low-rank features. Instead of using only one randomly generated rank k per forward pass, we use all possible slices Ar and Br, with r ∈ {1,2,4,...,R}, where R is the maximum rank of A and B. This way, the lower-dimensional representations are contained in higher-dimensional representations as prefixes, thus building a nested hierarchy of features. We present the schematic of MATRYOSHKALORA in Figure 1.

We see three advantages of having an accurate technique to train hierarchical low-rank adapters: (1) it eliminates the need for multiple training runs to grid-search for different ranks, implicitly translating to a reduction in costs; (2) we benefit from a high-performance adapter at each rank r by deploying targeted ranks on different devices depending on their computational power and (3) we enable dynamic rank selection under varying cluster loads to serve the requests at the same rate with minimal accuracy drop.

Contribution. We summarize our contribution as follows:

- • We introduce a general framework inspired by Matryoshka Representation Learning for training LoRA adapters whose lower-rank prefixes remain accurate and independently usable at inference time, which views LoRA, DyLoRA, and Matryoshka-style adapters as different parameterizations of a shared rank-weighting vector;
- • Starting from the goal of learning hierarchical low-rank representations at every layer, we show that this naturally leads to a simple diagonal weighting between the standard LoRA adapters A and B, making the hierarchy explicit while keeping the implementation and exposition close to standard LoRA;

- • We propose MATRYOSHKALORA, a concrete instance of this framework that learns accurate nested low-rank prefixes via the new diagonal matrix. The resulting adapter can be evaluated or deployed at multiple ranks from a single checkpoint, without retraining separate adapters;
- • We introduce Area Under the Rank Accuracy Curve (AURAC), a metric for evaluating adapter performance across a set of ranks. AURAC summarizes the rank-performance tradeoff while weighting each rank according to its magnitude, reflecting the expectation that larger ranks should generally achieve stronger performance.

### 2 Method

In this section, we introduce the notation we will use throughout the paper and provide details about our main baselines in the literature. Then we describe MATRYOSHKALORA, our approach to learn accurate hierarchical low-rank representations for fine-tuning and AURAC, the metric we propose to assess the performance of LoRA approaches when evaluated on multiple ranks.

#### 2.1 Notation

We consider the weights of a fully connected layer W0 ∈ Rm×n and the LoRA adapters A ∈ Rm×R and B ∈ RR×n, where R is the maximum possible rank of A and B (bottleneck dimension), which is usually a power of 2. Given the adapters A and B and an integer k, we extract the subsets of these adapters, denoted by Ak = A[∗,1 : k] (first k columns of A) and Bk = B[1 : k,∗] (first k rows of B) by indexing into the R-dimension of each adapter.

We denote by S the set of ranks used for training/inference. If not specified otherwise, the set S is the same for both training and inference. Note that in our work we restrict S to contain only to powers of two to be in line with most of the settings that already exist, as the ranks for LoRA are also chosen to be powers of 2.

#### 2.2 Preliminaries

In this section, we consider the default low-rank adaptation for a pretrained layer W0 as W = W0 + sR · AB, where A ∈ Rm×R and B ∈ RR×n are the rank-R adapters and the scaling factor sR, with sz ∈ {1, z1, √1z},∀z ∈ N\{0}. Given the input activation x ∈ Rd×m for the current layer, the forward pass for LoRA is presented in Equation 1. The fundamental difference between LoRA and DyLoRA is in the forward pass. Given an integer k sampled from S uniformly at random, the forward pass of DyLoRA is presented in Equation 2.

LORA: Y = x(W0 + sRAB) (1) DYLORA: Y = x(W0 + skAkBk) (2)

#### 2.3 MATRYOSHKALORA

Our approach stores the same LoRA adapters A ∈ Rm×R and B ∈ RR×n and instead of using the forward passes for LoRA and DyLoRa in Equations 1 and 2, we include all slices Ar,Br into the forward pass as in Equation 3:

MATRYOSHKALORA: Y = x W0 + r∈S srArBr (3) The goal of MATRYOSHKALORA is to train accurate hierarchical low-rank representations for different ranks inside the same LoRA adapters A and B such that the accuracy drop for ranks r < R is minimized, as the loss will contain contributions from all ranks r ∈ S. Our goal is to update each slice Ak and Bk using gradient signal computed for the current batch of data. In contrast, DyLoRA uses the gradient from the current batch to update only the first k columns/rows, while the rest R − k do not receive any update, making it data-inefficient.

Equation 3 cannot be implemented as it is in PyTorch because the framework does not allow propagating gradients only through a slice of parameter. Therefore, we must use the full adapters A and B and mask the gradients accordingly:

Y = x W0 + r∈S sr · (A ⊙ MrA)(B ⊙ MrB) (4)

where MrA and MrB are binary masks of the same shape as A and B where we set to 1 only the first r columns and first r rows of MA and MB, respectively. For example, for m = n = R = 3 and S = {1,2}, we would have the following masks:

1 0 0 1 0 0 1 0 0

1 1 0 1 1 0 1 1 0

1 1 1 0 0 0 0 0 0

1 1 1 1 1 1 0 0 0

M1A =

;M2A =

;M1B =

;M2B =

.

If we ignore the overhead of the element-wise multiplications between adapters A and B and their corresponding masks MA and MB, then the Equation 4 would require |S| matmuls for each layer with inner dimensions r ∈ S, compared to one matmul for LoRA and DyLoRA with inner dimensions R and k, which is clearly an undesired overhead. During training, the boolean masks MrA and MrB must be stored and applied to the gradients computed for A and B at each forward pass. Even though they are boolean values, storing two masks per linear layer is against the simplicity of LoRA. Next, our goal is to simplify the formulation of MATRYOSHKALORA. A careful inspection of Equation 3 suggests there exist two matrices CA,CB with the same shape as A and B such that:

(A ⊙ CA)(B ⊙ CB) = r∈S(A ⊙ MrA)(B ⊙ MrB) (5)

In other words, we do not have to store individual boolean masks MrA,B for r ∈ S and potentially share these masks among layers whenever the dimensions allow. Instead, we can simply use two

matrices CA,CB per layer instead of 2 · |S| matrices, which completely remove the need for masks MrA,B, as well as the need the loop in Equation 3. At this stage, we removed the boolean masks MrA,B and we are left with the floating point matrices CA,CB, which still increase memory usage, even when stored in half precision. We go one step further towards simplifying the forward pass for MATRYOSHKALORA and observe that CA,CB can be replaced with a diagonal matrix P. We provide more details in Appendix A. Therefore, we obtain the simplest form as:

(A ⊙ CA)(B ⊙ CB) = A · diag(P) · B = (A ∗ P) · B, (6) where P is an R-dimensional vector. Note we do not need to explicitly create the matrix diag(P) ∈ RR×R. Instead, we can simply multiply the row i of A by Pi to obtain the same effect at the lowest possible cost. The vector P is the same for all layers and therefore the memory overhead of our MATRYOSHKALORA is only the additional vector P with R elements. The final formulation of the forward pass for MATRYOSHKALORA is based on the observation that there exists an R-dimensional vector Pk such that Ak · Bk = A · diag(Pk) · B, with Pk = diag(1,··· ,1

) such that:

##### , 0,··· ,0

k values

R−k values

Y = xW = x(W0 + (A ∗ P)B), with P = r∈S sr · Pr (7) Figure 1 shows the schematic of MATRYOSHKALORA: the diagonal matrix P is used during the training using the forward pass in Equation 7; during evaluation, we discard the matrix P, choose a specific rank k and perform dynamic inference according to the default LoRA formula described in Equation 1. In Appendix B we provide a theoretical view of our formulation.

#### 2.3.1 How to compute P?

In Algorithm 1 we show the steps to create the vector P with R components. For each rank r from R down to 1, we use the integer c to count how many times the sub-rank k is employed in the sum in Equation 3. After that, the value for P at component r (denoted by pr) is obtained as the sum of sr

Algorithm 1 Algorithm to compute vector P in Equation 6 for our MATRYOSHKALORA

- 1: inputs:

- • S - set of training ranks
- • R - maximum rank of A,B

- 2: P ← 0R // R zeros
- 3: c ← 0
- 4: for r = R down to 1 do
- 5: if r ∈ S then
- 6: c ← c + 1
- 7: end if
- 8: pr ← |kS=||S|−c sr

k

- 9: end for
- 10: return P ∈ RR

, where rk is the rank at index k in set S containing training ranks, as shown in Equations 8, 9, 10 and 11 as an example. Note that the summation loops through indices of S indexed from 1 to |S| and not its elements. For example, R = 8 and S = {1,2,4,8} will generate the vector P with the following components pi:

k

- p1 = s1+s2+s4+s8 (8)
- p2 = s2+s4+s8 (9)

p3 = p4 = s4+s8 (10) p5 = p6 = p7 = p8 = s8 (11)

Specifically, pr will have contributions from all ranks r that appear as subscripts in its expression

above. When we use sr = 1,∀r ∈ S, we will obtain P = [4,3,2,2,1,1,1,1], which has the effect of scaling the first column of A and first row of B by 4 and so on, thus increasing the contribution of the gradient. Observe the last components of P are all 1, meaning that the columns/rows 5,6,7,8 from A and B respectively will get gradient signal only from the rank-8 component. In short, the vector P embeds the global contribution for each column/row in Equation 3. The vector P induces the desired effect of learning hierarchical low-rank features in our MATRYOSHKALORA.

#### 2.3.2 Adapter Scaling

The default forward pass for LoRA uses sR = 1/R in Equation 1, which accounts for the multiplication of A and B that have both the the inner dimension R. Prior work in the literature [10] suggests the scaling should be 1/√r instead of 1/r. However, in the context of MATRYOSHKALORA, this scaling requires a much larger learning rate to learn the desired hieararchical low-rank features because larger ranks will have a smaller contribution. This is justified by the inequalities 1 ≤

√r < r which implies 1/r < 1/√r ≤ 1. Therefore, to minimize the size of our learning rate grid, we choose sk = 1 instead of sk ∈ {1/r,1/√r} in the experiments for MATRYOSHKALORA. We provide an empirical example of this phenomenon in Section 3.4.

#### 2.3.3 General Framework for Recovering Existing LoRA Approaches

In this section we show our Matryoshka framework is general enough to obtain both LoRA and DyLoRA by manipulating the expression of vector P. To recover LoRA described in Equation 1, we have to choose P = 1/r · 1R, an R-dimensional vector that contains only the values 1/r. To recover DyLoRA described in Equation 2 for a sampled rank k S, we have to choose P = 1/k · [1,··· ,1 k values

], where the first k components are equal to 1/k and the rest are zero.

, 0,··· ,0

R−k values

#### 2.4 Gradient Computation for MATRYOSHKALORA

In this section we explain how inserting the constant matrix diag(P) between A and B scales the gradients for A and B, even though in the implementation we multiply the columns of A with elements of vector P for efficiency. Let W ∈ Rm×n be the pretrained layer and ∆ ∈ Rm×n be the upstream gradient for the current linear layer, for which we compute the gradients ∇A and ∇B for

- A and B, respectively. Then, the gradients used in the optimizer for A and B have the following expression:

- ∇A = ∆ · B⊤ · diag(P) ∈ Rm×R (12)
- ∇B = diag(P) · A⊤ · ∆ ∈ RR×n (13)

Given the structure of diag(P) described in previous sections, Equations 12 and 13 illustrate how the matrix diag(P) scales the gradients ∇A and ∇B, thus learning the hierarchical, low-rank representations we are targeting in MATRYOSHKALORA.

#### 2.5 Evaluation Metrics

Given a model fine-tuned with a LoRA variant (can be LoRA, DyLoRA or MATRYOSHKALORA) and a set of evaluation ranks S, we define the set of accuracies AS = {ak | k ∈ S} obtained when evaluating the model with rank k ∈ S using the forward pass Y = x(W0 + sk · AkBk), regardless the underlying LoRA-type.

For example, if A and B have the bottleneck dimension R = 16 and we choose the evaluation ranks k ∈ S = {1,2,4,8,16} on a specific dataset D, we obtain a set of fine-tuning accuracies AS = {a(1),a(2),a(4),a(8),a(16)}, where a(k) is the accuracy on dataset D for rank k.

Given S and AS, we denote ri as the i-th rank in S and a(ri) as the accuracy obtained with rank ri, we compute two versions for an aggregate metric called the Area Under the Rank Accuracy Curve

(AURAC), which employs the trapezoidal rule:

|S|−1

1 r|S| − r1

a(ri) + a(ri+1) 2 · (ri+1 − ri) (14)

AURAC =

i=1

|S|−1

1 log2(r|S|) − log2(r1)

a(ri) + a(ri+1) 2 · (log2(ri+1) − log2(ri)) (15)

log-AURAC =

i=1

The AURAC metric takes into account the distance between consecutive ranks in the set S containing training ranks. For our example mentioned above, the accuracy area between ranks 8 and 16 will have the largest proportion in the resulting AURAC metric, equivalent to (16 − 8)/(16 − 1) = 8/15 = 53.3%, which would bias the final AURAC value towards the accuracy of the largest interval between ranks 8 and 16, which is in line with the expectation that the LoRA adapters should learn hierarchical, low-rank features: a large value for the AURAC metric means all intermediary ranks achieve high accuracy. In our experiments we use AURAC by default.

The log-AURAC metric is designed only for the case when S contains ranks that are power of two, which would result in all ranks being equally distant, therefore all intervals will contribute uniformly to the log-AURAC metric. For our example, all intervals will weighted by 1/4 = 25%. For the datasets we tested on, we haven’t seen significant differences between AURAC and log-AURAT and therefore we chose the best runs based on the AURAC metric.

### 3 Experimental Results

Setting. To prove the effectiveness of our MATRYOSHKALORA in learning hierarchical low-rank features, we fine-tune Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct models on GSM-8k and OpenPlatypus [13] and test on GSM-8K (3/8−shot settings) and Open LLM Leaderboard [1], such as ARC-C [3] and HellaSwag [21], respectively.

Ranks used. We reiterate that the LoRA adapters in our work have shapes A ∈ Rm×R and

- B ∈ RR×n, where R is the maximum rank (bottleneck size), which we ablate over in our experiments to assess the quality of each sub-rank. We choose the set S that contains all powers of 2 up to the maximum rank R (such as 32 or 256). Then, the ranks used for training and evaluation are determined by the set SR = {1,2,4,...,R} (all powers of 2).

Hyper-params & Adapter Type. We consider the best performing runs the ones achieving largest average AURAC metric, which we also report alongside evaluation accuracies for each particular rank. Averaging is performed across 3 seeds (42,666,2408) and we omit the error bars or standard deviation for clarity reasons. We use 3 epochs for all datasets in our study. To keep the compute costs for the evaluation low, we mention the grid we used to ablate over the learning rate in each section separately. All experiments in this section use Equation 1 for LoRA, Equation 2 for DyLoRA and Equation 7 for MATRYOSHKALORA. We use the default AdamW [15] optimizer for fine-tuning.

Hardware & Software. All our experiments are performed in single-GPU setting on Nvidia H100 GPUs with 80GB of RAM under PyTorch v2.8.0, lm-eval v0.4.9.1, transformers v4.57.6 and datasets v3.6.0.

Overhead. Since our approach adds one simple operation (multiplying the columns of adapter A with vector P), the required resources in terms of running time and memory of MATRYOSHKALORA are identical to LoRA and DyLoRA.

#### 3.1 Fine-Tuning Llama-3.2-1B-Instruct on GSM-8K for different ranks R

The first set of experiments focuses on fine-tuning Llama-3.2-1B-Instruct [8] on GSM-8k [4] using LoRA, DyLoRA and MATRYOSHKALORA. We fix the bottleneck size to R = 32 and use the ranks in the set S32 = {1,2,4,8,16,32} to train the models and evaluate them using 8-shot strategy via lm-evaluation-harness [7].

We perform grid search over learning rates η ∈ {1e − 5,2e − 5,4e − 5,6e − 5,8e − 5} and report the results in Table 3.1. For the discussion that follows, we mention that the baseline accuracy of the pre-trained model (before any fine-tuning) is 34.7%.

Rank-wise Accuracy. Our MATRYOSHKALORA dominates across the entire rank spectrum. While LoRA and DyLoRA exhibit relatively modest gains as rank increases, typically pleateauing in the mid 34% range, our approach scales more effectively with rank and exhibits a significant increase in evaluation accuracy starting with R = 2. At higher ranks, such as r ∈ {16,32}, MATRYOSHKALORA achieves accuracies up to 39% and 38%, respectively, clearly surpassing the best LoRA (≈ 35%) and DyLoRA (≈ 35.5%).

We would like to emphasize that the evaluation accuracy obtained for the sub-rank r = 1 obtained when trained with bottleneck size R = 32 is better already better than any other sub-rank for LoRA and DyLoRA. When we compare against the evaluation accuracy of 34.7% for the pre-trained model, we observe that fine-tuning with LoRA and DyLoRA does not improve accuracy significantly. Moreover, only deploying k = 32 obtained for R = 32 would lead to better performance compared to the pre-trained model.

AURAC. In terms of AURAC, which captures the performance aggregated across ranks, our method provides again a clear advantage. Both LoRA and DyLoRA achieve peak AURAC values of ≈ 35%, whereas MATRYOSHKALORA reaches 38.4%, representing a substantial improvement. This gap highlights both stronger peak performance and more consistent gains across all ranks.

Key takeaway. Selecting rank r ∈ {4,8} with accuracies 37 − 39%) from MATRYOSHKALORA trained with bottleneck size R = 32 (the very last row of Table 1) yields higher accuracy compared to any sub-rank chosen from either LoRA or DyLoRA.

- Table 1: [Results for Section 3.1] AURAC metric and per-rank, evaluation accuracies for Llama-3.21B-Instruct on GSM-8k (8-shot) across multiple ranks. We mark in bold the best AURAC per row and emphasize the accuracy of pre-trained model (before finetuning) of 34.7%.

ADAPTER TYPE

EVAL ACCURACY PER RANK AURAC 1 2 4 8 16 32

- 32.4 32.4
- 32.4 33.6 33.0

- 33.5 34.8 34.3 34.4
- 34.2 34.1 34.6 34.9 34.5

LORA

- 32.5 33.6 34.5 34.8 34.9 34.6
- 33.3 34.3 34.3 34.3 34.5 34.6 34.5

- 32.4 32.4

- 32.9 33.5 33.2
- 33.4 34.0 34.3 34.0

- 33.1 33.8 34.6 34.7 34.4 32.9 33.9 34.6 34.8 34.8 34.6
- 34.4 34.5 34.1 34.7 34.9 35.4 34.9

MATRYOSHKALORA

32.3 32.3

- 35.9 34.7 35.3
- 36.5 36.8 35.4 36.3

DYLORA

- 35.2 36.0 36.4 37.3 36.5
- 35.3 35.9 37.4 37.7 37.1 37.2 35.8 35.6 37.2 38.8 39.1 38.3 38.4

#### 3.2 Fine-Tuning Llama-3.1-8B-Instruct on GSM-8K

The second set of experiments focuses on fine-tuning Llama-3.1-8B-Instruct [8] on GSM-8k [4] using the same LoRA, DyLoRA and MATRYOSHKALORA and evaluated using 3/8-shots strategies. We fix the bottleneck size to R = 256 and use the ranks in the set S256 = {1,2,4,8,16,32,64,128,256} for both training and evaluation via the same lm-evaluation-harness [7].

- We perform grid search over learning rates η ∈ {5e − 6,6e − 6,7e − 6,8e − 6,9e − 6,1e − 5,2e −

- 5,3e − 5,4e − 5,5e − 5} and report the results in the upper half of Table 2. We mention the accuracy of the pre-trained model (before any fine-tuning) in the PT column of the table.

3-shot setting. Our method consistently outperforms both LoRA and DyLoRA across nearly all ranks. While the evaluation accuracy for these two baselines fluctuate in the range 74 − 75%, MA-

TRYOSHKALORA achieves higher and more stable per-rank performance, reaching up to 78.6% for rank r = 128, an improvement of over 4% compared to both pre-trained model and baselines. Moreover, our improvements are visible from low to high ranks, which is the purpose of the hierarchical low-rank representations. The improvements in per-rank accuracies translate to the AURAC score, where MATRYOSHKALORA reaches more than 3% compared to baselines. 8-shot setting. The advantage of MATRYOSHKALORA remains consistent, reaching 80% accuracy for ranks r ∈ {32,64,128}. However the gap decreases when r increases. Our explanation is that more in-context information helps learning overall and shadows the influence of the adapter.

Drops from 128 to 256. In both 3/8-shot settings, there is a drop in accuracy when going from rank 128 to 256 for both LoRA and MATRYOSHKALORA. We believe this happens because the bottleneck size R = 256 might require more than 3 epochs to fix the drop.

Key takeaway. MATRYOSHKALORA closes the gap between 3-shot and 8-shot: if the goal is to achieve 77% accuracy with 3-shots, one should pick the adapters trained with MATRYOSHKALORA for R = 256 and use r = 32 or r = 64 with 3-shots. In principle, would translate to shorter contexts compared to 8-shots and therefore lower inference costs.

- Table 2: [Results for Section 3.2 and 3.3] AURAC metric and per-rank evaluation accuracies for GSM-8k (3/8-shot) and Open LLM Leaderboard on ARC-C (25-shot) and Hellaswag (10-shot) for Llama-3.1-8B-Instruct on GSM-8k across multiple ranks. PT stands for the accuracy of the Pre-Trained model (before fine-tuning) with N-shots.

ADAPTER TYPE PT

EVAL ACCURACY PER RANK AURAC 1 2 4 8 16 32 64 128 256

N

LORA 74.9 74.8 74.1 74.2 75.0 73.6 74.7 74.6 73.6 74.3 GSM-8K 3-SHOT

DYLORA 74.0 74.7 75.0 74.7 74.5 73.5 74.1 74.5 74.5 74.5 74.4 MATRYOSHKA 73.5 73.5 74.7 75.0 75.9 76.8 77.6 78.6 76.4 77.4

LORA 78.6 78.4 79.1 78.6 78.5 78.7 78.9 79.5 79.1 79.1 GSM-8K 8-SHOT

DYLORA 77.8 78.5 78.5 77.9 78.6 78.5 78.4 78.8 79.5 79.8 79.2 MATRYOSHKA 77.9 77.7 79.2 79.1 79.8 80.2 80.5 80.2 78.1 79.7

LORA 56.7 56.9 56.6 56.9 56.9 56.9 57.1 57.1 57.0 57.0 ARC-C 25-SHOT

DYLORA 56.7 56.9 56.7 57.0 57.0 56.9 57.0 57.0 57.0 57.0 57.0 MATRYOSHKA 56.9 56.8 56.9 57.0 57.2 57.3 57.6 58.4 58.1 58.0

LORA 59.0 59.1 59.2 59.1 59.2 59.2 59.2 59.2 59.1 59.2 HELLASWAG 10-SHOT

DYLORA 59.3 59.2 59.2 59.2 59.2 59.1 59.2 59.2 59.2 59.2 59.2 MATRYOSHKA 59.4 59.3 59.3 59.4 59.6 59.9 60.5 61.4 62.8 61.4

#### 3.3 Fine-Tuning Llama-3.1-8B-Instruct on Open Platypus

The third set of experiments focuses on fine-tuning Llama-3.1-8B-Instruct [8] on Open Platypus [13] and evaluating on two datasets from Open LLM Leaderboard [1]: ARC-Challenge [3] (25-shots) and Hellaswag [21] (10-shots). The setting for R is identical to Section 3.2.

- We perform grid search over learning rates η ∈ {6e − 6,7e − 6,8e − 6,9e − 6,1e − 6,2e − 5,3e −

- 5,4e − 5} and report the results in the lower half of Table 2.

ARC-C. LoRA and DyLoRA exhibit an identical behavior, with both per-rank accuracies and AURAC clustered around 57%, a small improvement over the pre-trained accuracy of 56.7%. In contrast, MATRYOSHKALORA achieves more than 58% per-rank accuracy and 58% AURAC, showing consistent imprvements as the rank increases.

HellaSwag. The advantage of our method is more pronounced for this dataset. While LoRA and DyLoRA remain effectively flat around the accuracy of the pre-trained model of 59.2%, MATRYOSHKALORA exhibits steady gains with increasing rank, achieving 60.5% for r = 64, 61.4% for r = 128 and peaking at 62.8% for r = 256, which represents an increase of more than 3% over the rank r = 256 for LoRA and DyLoRA. In line with the increased rank-wise accuracy, AURAC score reaches 61.4%.

#### 3.4 Ablation for scaling parameters sk ∈ {1,1/k,1/√k}

Our last set of experiments focuses on understanding the effect of the scale sk ∈ {1,1/k,1/√k} over the learning rate and evaluation accuracy/AURAC for our MATRYOSHKALORA for a fixed bottleneck

size R = 32. Our hypothesis is that sk = 1/√r requires larger learning rate to achieve highest accuracy over a learning rate grid Gη.

We fine-tune Llama-3.2-1B-Instruct on GSM-8k with R = 32 using three seeds and ablating over a learning rate grid Gη ∈ {5e − 5,··· ,1e − 3}, containing 14 values. We would like to emphasize that changing R directly influences the behavior of this experiment and because of that we decide to keep R fixed and ablate only over the learning rate η.

We show the results of this experiment in Table 3, where we can see that our hypothesis is confirmed: compared to scaling sk = 1, we need a learning rate that is 9× larger when we use scaling sk = 1/√k and 4× larger when we use scaling sk = 1/k.

- Table 3: [Results for Section 3.4] Scaling ablation for MATRYOSHKALORA on Llama-3.2-1B-Instruct on GSM-8k(8-shot). The accuracy of the pre-trained model (before fine-tuning) is 34.7%.

SCALING TYPE

EVAL ACCURACY PER RANK AURAC η 1 2 4 8 16 32

sk = 1/r 36.0 37.2 36.8 36.5 38.3 38.8 37.8 9 · 10−4 sk = 1/√r 36.4 36.9 38.2 38.4 39.1 39.0 38.7 3 · 10−4 sk = 1 35.5 36.2 37.7 38.7 39.2 37.8 38.4 1 · 10−4

- 4 Conclusion, Limitations and Broader Impact

Conclusion. We propose MATRYOSHKALORA, a framework that modifies standard LoRA by inserting a carefully crafted diagonal matrix diag(P) between adapters A and B that enables learning hierarchical, low-rank features inside the same adapters A and B as embedded prefixes. We demonstrate the effectiveness of this simple approach on Llama models with 1 billion and 8 billions of parameters on a few popular datasets in the literature. We also show that we can obtain LoRA and DyLoRA in our framework by simply changing the expression of the vector P. In addition propose a metric to assess the performance of a LoRA-like model when evaluated on multiple ranks to facilitate choosing a best model during the hyper-parameter tuning process.

Limitations. Despite this method being simple and having the same overhead as LoRA, the preparation of this work requires many model evaluations for each rank, which increases the actual runtime to assess the performance. Therefore, we believe our results can be improved by further hyper-parameter tuning. Specifically, one might experiment with, just to name a few, a wider grid for the learning rate, different number of epochs or weight decay. One more aspect worth mentioning is that our evaluations are performed with the same rank k for the entire network. However, in our paper we haven’t experimented with settings where different layers get different ranks for the same forward pass, as it is the case for studies in the literature that measure sensitivity of each layer which affects its LoRA rank.

Broader Impact. Our MATRYOSHKALORA introduces a general framework in the LoRA-like literature that allows learning accurate, hierarchical low-rank features that live inside the same adapters, allowing more efficient deployment. However, it is important to note that while the purpose of our our technique is to reduce overall deployment costs, we do not have control over the applications where our method might be used in.

### Acknowledgments

We would like to thank the Scientific Computing Department at ISTA for providing access to computational resources to develop this work. MS’s work was supported by Research England under the Expanding Excellence in England (E3) funding stream, which was awarded to MARS:

Mathematics for AI in Real-world Systems in the School of Mathematical Sciences at Lancaster University.

### References

- [1] Edward Beeching et al. Open LLM Leaderboard. https://huggingface.co/spaces/ HuggingFaceH4/open_llm_leaderboard. 2023.
- [2] Han Cai et al. Once-for-All: Train One Network and Specialize it for Efficient Deployment.

2020. arXiv: 1908.09791 [cs.LG]. URL: https://arxiv.org/abs/1908.09791.

- [3] Peter Clark et al. “Think you have solved question answering? try arc, the ai2 reasoning challenge”. In: arXiv preprint arXiv:1803.05457 (2018).
- [4] Karl et al. Cobbe. “Training Verifiers to Solve Math Word Problems”. In: arXiv preprint arXiv:2110.14168 (2021).
- [5] Xuan Cui et al. IGU-LoRA: Adaptive Rank Allocation via Integrated Gradients and Uncertainty-Aware Scoring. 2026. arXiv: 2603.13792 [cs.LG]. URL: https://arxiv. org/abs/2603.13792.
- [6] Ning Ding et al. Sparse Low-rank Adaptation of Pre-trained Language Models. 2023. arXiv: 2311.11696 [cs.CL]. URL: https://arxiv.org/abs/2311.11696.
- [7] Leo Gao et al. A framework for few-shot language model evaluation. Version v0.4.0. Dec. 2023. DOI: 10.5281/zenodo.10256836. URL: https://zenodo.org/records/10256836.
- [8] Aaron Grattafiori et al. The Llama 3 Herd of Models. 2024. arXiv: 2407.21783 [cs.AI]. URL: https://arxiv.org/abs/2407.21783.
- [9] Edward J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021. arXiv: 2106.09685 [cs.CL]. URL: https://arxiv.org/abs/2106.09685.
- [10] Damjan Kalajdzievski. A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. 2023. arXiv: 2312.03732 [cs.CL]. URL: https://arxiv.org/abs/2312.03732.
- [11] Vishnuprasadh Kumaravelu, Sunil Gupta, and P. K. Srijith. Post-Optimization Adaptive Rank Allocation for LoRA. 2026. arXiv: 2604.27796 [cs.AI]. URL: https://arxiv.org/abs/ 2604.27796.
- [12] Aditya Kusupati et al. Matryoshka Representation Learning. 2024. arXiv: 2205.13147 [cs.LG]. URL: https://arxiv.org/abs/2205.13147.
- [13] Ariel N Lee, Cole J Hunter, and Nataniel Ruiz. “Platypus: Quick, cheap, and powerful refinement of llms”. In: arXiv preprint arXiv:2308.07317 (2023).
- [14] Zequan Liu et al. ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models. 2024. arXiv: 2403.16187 [cs.CL]. URL: https://arxiv.org/abs/2403. 16187.
- [15] Ilya Loshchilov and Frank Hutter. “Decoupled weight decay regularization”. In: arXiv preprint arXiv:1711.05101 (2017).
- [16] Dan Luo et al. “ERAT-DLoRA: Parameter-efficient tuning with enhanced range adaptation in time and depth aware dynamic LoRA”. In: Neurocomputing 614 (2025), p. 128778. ISSN: 0925-2312. DOI: https://doi.org/10.1016/j.neucom.2024.128778. URL: https: //www.sciencedirect.com/science/article/pii/S0925231224015492.
- [17] Raul Singh et al. L1RA: Dynamic Rank Assignment in LoRA Fine-Tuning. 2025. arXiv: 2509.04884 [cs.CL]. URL: https://arxiv.org/abs/2509.04884.
- [18] Mojtaba Valipour et al. DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation. 2023. arXiv: 2210.07558 [cs.CL]. URL: https://arxiv.org/abs/2210.07558.
- [19] Jiahui Yu and Thomas Huang. Universally Slimmable Networks and Improved Training Techniques. 2019. arXiv: 1903.05134 [cs.CV]. URL: https://arxiv.org/abs/1903. 05134.
- [20] Jiahui Yu et al. Slimmable Neural Networks. 2018. arXiv: 1812.08928 [cs.CV]. URL: https://arxiv.org/abs/1812.08928.
- [21] Rowan Zellers et al. “Hellaswag: Can a machine really finish your sentence?” In: Proceedings of the 57th annual meeting of the association for computational linguistics. 2019, pp. 4791– 4800.

- [22] Feiyu Zhang et al. IncreLoRA: Incremental Parameter Allocation Method for ParameterEfficient Fine-tuning. 2023. arXiv: 2308.12043 [cs.CL]. URL: https://arxiv.org/abs/ 2308.12043.
- [23] Qingru Zhang et al. AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.

2023. arXiv: 2303.10512 [cs.CL]. URL: https://arxiv.org/abs/2303.10512.

- [24] Ruiyi Zhang et al. AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning. 2024. arXiv: 2403.09113 [cs.CL]. URL: https://arxiv.org/ abs/2403.09113.

### Appendix Contents

- 1 Introduction & Related Work 1
- 2 Method 3

- 2.1 Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 2.2 Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 2.3 MATRYOSHKALORA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

- 2.3.1 How to compute P? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3.2 Adapter Scaling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.3.3 General Framework for Recovering Existing LoRA Approaches . . . . . . 5

- 2.4 Gradient Computation for MATRYOSHKALORA . . . . . . . . . . . . . . . . . . . 5
- 2.5 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 3 Experimental Results 6

- 3.1 Fine-Tuning Llama-3.2-1B-Instruct on GSM-8K for different ranks R . . . . . . . 6
- 3.2 Fine-Tuning Llama-3.1-8B-Instruct on GSM-8K . . . . . . . . . . . . . . . . . . 7
- 3.3 Fine-Tuning Llama-3.1-8B-Instruct on Open Platypus . . . . . . . . . . . . . . . . 8
- 3.4 Ablation for scaling parameters sk ∈ {1,1/k,1/√k} . . . . . . . . . . . . . . . . . 9

- 4 Conclusion, Limitations and Broader Impact 9

- A Simplification of MATRYOSHKALORA 13
- B DyLoRA vs MatryoshkaLoRA: Theory View 13

- B.1 DyLoRA as stochastic optimization of the multi-rank objective . . . . . . . . . . . 14
- B.2 From DyLoRA’s stochastic objective to a MatryoshkaLoRA surrogate . . . . . . . 15
- B.3 Summary of the connection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

### A Simplification of MATRYOSHKALORA

In this section we explain how we simplify the expression of MATRYOSHKALORA presented in Equation 3, where we choose scaling sr = 1 for simplicity. Let’s take an example of two LoRA adapters, where A ∈ (3,3) and B ∈ (3,3), from which we want to use ranks r ∈ {1,2,3} and denote by Ar,Br the first r columns for A and first r rows for B.

A = A3 =

a11 a12 a13 a21 a22 a23 a31 a32 a33

A1 =

a11 a21 a31

A2 =

a11 a12 a21 a22 a31 a32

B = B3 =

b11 b12 b13 b21 b22 b23 b31 b32 b33

B1 = (b11 b12 b13) B2 =

b11 b12 b13 b21 b22 b23

We want to find matrices CA,CB of the same shape as A and B such that (A ⊙ CA)(B ⊙ CB) =

r ArBr. First, let’s look at the result of multiplications Ar · Br for each r ∈ {1,2} by explicitly writing the scalar products between rows of Ar and columns of Br

- A1·B1 =

a11b11 a11b12 a11b13 a21b11 a21b12 a21b13 a31b11 a31b12 a31b13

A2·B2 =

a11b11 + a12b21 a11b12 + a12b22 a11b13 + a12b23 a21b11 + a22b21 a21b12 + a22b22 a21b13 + a22b23 a31b11 + a32b21 a31b12 + a32b22 a31b13 + a32b23

##### A3 · B3 =

a11b11 + a12b21 + a13b31 a11b12 + a12b22 + a13b32 a11b13 + a12b23 + a13b33 a21b11 + a22b21 + a23b31 a21b12 + a22b22 + a23b32 a21b13 + a22b23 + a23b33 a31b11 + a32b21 + a33b31 a31b12 + a32b22 + a33b32 a31b13 + a32b23 + a33b33

3a11b11 + 2a12b21 + 1a13b31 3a11b12 + 2a12b22 + 1a13b32 3a11b13 + 2a12b23 + 1a13b33 3a21b11 + 2a22b21 + 1a23b31 3a21b12 + 2a22b22 + 1a23b32 3a21b13 + 2a22b23 + 1a23b33 3a31b11 + 2a32b21 + 1a33b31 3a31b12 + 2a32b22 + 1a33b32 3a31b13 + 2a32b23 + 1a33b33

ArBr =

r∈{1,2,3}

By visual inspection, we observe the elements from the first row of A and first column of B are scaled by 3, which is the total number of ranks we test for. The second row/column is scaled by 2 and finally, the third row/column is scaled by 1. Since we need to make sure we learn hierarchical features for both A and B, we can use the square root of coefficients 3,2,1 for CA and CB:

√√√333 √√√222 111

  CB =

 

 

 

√√32 √√32 √√32 1 1 1

CA =

Moreover, instead of using matrices CA and CB, we can obtain the matrix r∈{1,2,3} ArBr by multiplying column i of A by Pi, thus yielding the final MATRYOSHKALORA forward pass during training as in Equation 7. Specifically, the product A ∗ P yields the following matrix:

3a11 2a12 1a13 3a21 2a22 1a23 3a31 2a32 1a33

A ∗ P =

### B DyLoRA vs MatryoshkaLoRA: Theory View

Consider a linear layer with pretrained weights W0 ∈ Rm×n. Standard LoRA introduces trainable adapters A ∈ Rm×R and B ∈ RR×n, and replaces the layer by

W = W0 + sAB,

where s = sR is a rank-dependent scaling factor, for example sR = α/R or sR = α/√R. Given a fine-tuning dataset and loss function ℓ, the fine-tuning objective is

min

ℓ(W0 + sAB).

A,B

The limitation of the standard approach is that fine-tuning is tied to a specific rank R. If, after training, we want to use a smaller rank r < R, then in general we would need to fine-tune again. A naive alternative is to take the first r columns of A and the first r rows of B, but there is no reason for these truncated adapters to perform well unless the training procedure explicitly encourages this structure.

Let R be the maximum rank, and let

S = {1 ≤ r1 < r2 < ··· < rk ≤ R}

be the collection of smaller ranks we would like to support. Our goal is to fine-tune a single pair of rank-R adapters A,B such that, for every r ∈ S, the truncated adapters

Ar = A[:,1:r], Br = B[1:r,:]

define a useful rank-r adaptation W0 + srArBr. This leads naturally to the family of objectives

Lr(A,B) = ℓ(W0 + srArBr), r ∈ S,

which we would like to optimize simultaneously. A natural way to train one adapter pair for all ranks is the weighted multi-rank objective

Lmulti(A,B), where

min

A,B

λr ℓ(W0 + srArBr), with normalized weights

Lmulti(A,B) =

λrLr(A,B) =

r∈S

r∈S

λr ≥ 0,

λr = 1.

r∈S

The weights λr specify how much training emphasis is placed on each supported rank. Now define the diagonal truncation matrix

##### ) ∈ RR×R.

##### ,0,...,0

##### Pr = diag(1,...,1

r

R−r

Then ArBr = APrB, and therefore

Lr(A,B) = ℓ(W0 + srAPrB). Hence the multi-rank objective can also be written as

Lmulti(A,B) =

λr ℓ(W0 + srAPrB).

r∈S

#### B.1 DyLoRA as stochastic optimization of the multi-rank objective

DyLoRA studies the stochastic version of the multi-rank objective above. In DyLoRA, one chooses a rank range

S = {rmin,rmin + 1,...,rmax},

and samples a rank b ∼ pB(·) at each training step. The LoRA matrices are then truncated to

Ab = A[:,1:b], Bb = B[1:b,:],

and the forward pass uses the rank-b perturbation ∆b = sbAbBb in the usual DyLoRA scaling sb = αb . Thus the sampled DyLoRA loss at one training step is

Lb(A,B) = ℓ(W0 + sbAbBb). Taking expectation over the sampled rank gives

Eb∼p

Lb(A,B) =

pB(r)ℓ(W0 + srArBr).

B

r∈S

Therefore DyLoRA corresponds to the multi-rank objective with λr = pB(r). That is, Lmulti(A,B) = Eb∼p

[Lb(A,B)].

B

Equivalently, using the truncation matrices Pr,

Lb(A,B) = ℓ(W0 + sbAPbB), and

Lmulti(A,B) = Eb∼p

[ℓ(W0 + sbAPbB)] =

λrℓ(W0 + srAPrB).

B

r∈S

For DyLoRA update, the sampled gradient is an unbiased estimator of the full multi-rank gradient:

Eb∼p

[∇A,BLb(A,B)] = ∇A,BLmulti(A,B).

B

So DyLoRA does not need to compute the full sum over all ranks at each step. Instead, it solves the same expected objective by sampling one rank-loss term at a time, exactly like SGD solves a full-data objective by sampling minibatches.

#### B.2 From DyLoRA’s stochastic objective to a MatryoshkaLoRA surrogate

We now connect the stochastic DyLoRA objective to a deterministic MatryoshkaLoRA-style forward pass. Define the loss as a function of the LoRA perturbation:

f(∆) = ℓ(W0 + ∆).

For each r ∈ S, define ∆r = srArBr = srAPrB. Then the multi-rank objective is

Lmulti(A,B) =

λrf(∆r).

r∈S

Assume now that f is differentiable and L-smooth, i.e., there exists a constant L > 0 such that

∥∇f(∆) − ∇f(∆′)∥F ≤ L∥∆ − ∆′∥F for all perturbations ∆,∆′. This implies the first-order expansion

f(∆) = f(0) + ⟨∇f(0),∆⟩ + R(∆), where |R(∆)| ≤ L2 ∥∆∥2F. Therefore,

λr f(0) + ⟨∇f(0),∆r⟩ + R(∆r)

λrf(∆r) =

r∈S

r∈S

On the other hand,

= f(0) + ∇f(0),

λr∆r +

r∈S

λrR(∆r).

r∈S

λr∆r = f(0) + ∇f(0),

f

r∈S

r∈S

Subtracting the two identities gives

λr∆r + R

λr∆r .

r∈S

Hence,

λrf(∆r) − f

r∈S

λr∆r =

r∈S

λrf(∆r) − f

r∈S

r∈S

By convexity of the squared norm,

λr∆r ≤

r∈S

≤

λr∆r

r∈S

2

≤

F

λrR(∆r) − R

r∈S

λr|R(∆r)| + R

L 2 r∈S

L 2 r∈S

λr∥∆r∥2F +

λr∥∆r∥2F.

r∈S

λr∆r .

r∈S

λr∆r

r∈S

λr∆r

2

.

F

Therefore,

λr∥∆r∥2F. Equivalently,

λrf(∆r) − f

λr∆r ≤ L

r∈S

r∈S

r∈S

λrf(∆r) = f

r∈S

Using ArBr = APrB, we obtain

λr∆r + O

r∈S

λr∥∆r∥2F .

r∈S

ℓ W0 +

λrsrArBr = ℓ W0 +

r∈S

r∈S

= ℓ W0 + A

Then the deterministic surrogate objective is

λrsrAPrB

λrsrPr B .

r∈S

ℓ(W0 + APB), P =

λrsrPr.

r∈S

This is the MatryoshkaLoRA-style surrogate: instead of sampling one rank per step as in DyLoRA, we use a single forward pass with a weighted combination of all nested rank components.

- B.3 Summary of the connection The connection can be summarized as follows. DyLoRA defines a sampled-rank training procedure:

b ∼ pB(·), Lb(A,B) = ℓ(W0 + sbAbBb). With λr = pB(r), this is stochastic optimization of

Lmulti(A,B) = Eb∼p

[ℓ(W0 + sbAbBb)]. MatryoshkaLoRA instead uses the deterministic first-order surrogate

B

which becomes

λrf(∆r) ≈ f

Lmulti(A,B) =

r∈S

λr∆r ,

r∈S

ℓ W0 + A

λrsrPr B . This gives the surrogate perturbation

r∈S

APB, with P =

λrsrPr.

r∈S

Hence, under the smoothness and small-perturbation assumptions above, the multi-rank training problem can be approximated by optimizing a single LoRA-style objective with an intermediate diagonal weighting matrix P:

##### ℓ(W0 + APB).

This reduction is local rather than exact, and its error is quadratic in the size of the perturbations. Still, it suggests a useful perspective: training a single pair of adapters for multiple nested ranks can be viewed as learning a shared rank-R factorization whose rank components are weighted according to how strongly different truncation levels are emphasized during training.

