## Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization

# arXiv:2510.21583v3[cs.CV]16Jun2026

#### Yifu Luo*‡1 Haoyuan Sun*1 Xinhao Hu*1 Penghui Du*2 Keyu Fan1 Bo Li†2 Sinan Du1 Xu Wan3 Zhiyu Chen1 Bo Xia1 Yongzhe Chang1 Changqian Yu2 Kun Gai2 Tiantian Zhang1 Xueqian Wang1

[Figure 1]

Figure 1. Compared to GRPO, Group Chunking Policy Optimization (GCPO) yields a significant improvement in image quality, particularly regarding structure, lighting, and fine-grained details, demonstrating the superiority of chunk-level policy optimization. Notably, Notably, the contrast between columns 3 and 4 underscores the importance of chunking with the inherent temporal dynamics of flow matching, which substantially elevates aesthetic style and structure of images. Additional details are provided in Section B.1.

*Equal contribution †Project lead. ‡Work done during internship in Kolors Team, Kuaishou Technology. 1Tsinghua University 2Kolors Team, Kuaishou Technology 3Zhejiang University. Correspondence to: Changqian Yu <yuchangqian@kuaishou.com>, Tiantian Zhang <zhang.tt@sz.tsinghua.edu.cn>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

### Abstract

Recent Progress in post-training flow matching for text-to-image (T2I) generation with Group Relative Policy Optimization (GRPO) has demonstrated strong potential. However, it is hindered by a critical limitation: inaccurate advantage attribution. In this work, we argue that aggregating consecutive steps into a coherent ‘chunk’ and

shifting the policy optimization paradigm from GRPO’s step level to the chunk level can effectively mitigate the negative impact of this issue. Building on this insight, we propose Group Chunking Policy Optimization (GCPO), the first chunk-level reinforcement learning approach for post-training flow matching. Extensive experiments demonstrate that GCPO achieves superior performance on both standard T2I benchmarks and preference alignment, with up to 43% relative gains over GRPO, highlighting the promise of chunk-level policy optimization. The code is available on https://github.com/ xingzhejun/GCPO.

### 1. Introduction

Reinforcement learning (RL) (Sutton et al., 1998; Schulman et al., 2017) has recently achieved remarkable success beyond traditional domains, most notably in the post-training of Large Language Models (LLMs) (Jaech et al., 2024; Guo et al., 2025; Wang et al., 2026a;b). Inspired by these advances, recent works (Xue et al., 2025; Liu et al., 2026; Wang & Yu, 2025; Luo et al., 2026) have explored adapting RL to text-to-image (T2I) generation. In this context, Group Relative Policy Optimization (GRPO) (Shao et al., 2024; Guo et al., 2025) has emerged as a promising approach for post-training flow matching (Lipman et al., 2023; Liu

- et al., 2023; Esser et al., 2024) for T2I generation. Typically, GRPO methods generate a group of images from the same prompt, obtain final rewards using reward models, convert them into group relative advantages, and assign these advantages uniformly across all steps in flow matching generations for policy optimization.

Despite its effectiveness, GRPO methods suffer from a critical limitation: inaccurate advantage attribution. The equal attribution of final reward to all steps rests on a strong assumption: A superior final outcome implies a superior policy at every step. However, this assumption is not always guaranteed. Consider two generation trajectories from the same prompt in Figure 2, each consisting of three steps. Although Trajectory1 yields a better final reward, Trajectory2 may have a superior policy at an intermediate step t = 1. In such cases, GRPO provides erroneous optimization signals, resulting in training instability. To quantify it more generally, we utilized a step-aware preference model from (Liang et al., 2025) to determine how often this occurs. As shown in Table 1, approximately half of the steps are affected by this inaccurate attribution, further demonstrating its severity. A straightforward solution to this issue would be to employ a precise, process (step-aware) reward model, which is capable of evaluating not only final clear images but also

[Figure 2]

Figure 2. An illustration of the inaccurate advantage attribution. Although Trajectory1 yields a better final reward, Trajectory2 demonstrates a superior policy at step t = 1. This conflicts with GRPO, which uniformly assigns the final advantage across all steps.

Table 1. Frequency of inaccurate advantage attribution. Approximately half of the steps are affected by it. Statistics are reported over 400 prompts from HPDv2.1 (Wu et al., 2023). Further details are provided in Section B.2.

FINAL REWARD BETTER WORSE STEPADVANTAGE

###### BETTER 63% 37% WORSE 44% 56%

noisy intermediate images during the flow matching generation process. However, training such models requires vast amounts of preference data for noisy images, which is currently unavailable. While some approaches (Liang et al., 2025; Liao et al., 2025) attempt to obtain process rewards using estimations like one-step diffusion, their effectiveness is limited due to estimation bias and the lack of groundtruth data (Zhang et al., 2025). Our comparison results in Section B.3 further corroborate this.

In this work, instead of introducing process rewards, we mitigate the negative impact of this issue from a different perspective. We draw inspiration from action chunking (Zhao et al., 2023; Li et al., 2025b) in robotics, which predicts a sequence of consecutive actions jointly as a single action rather than treating them independently. In a similar spirit, we propose to aggregate consecutive flow matching steps into a single “action chunk”, and optimize the policy at the chunk level rather than the step level. This effectively smooths out the gradient fluctuations caused by inaccurate advantage attribution, thereby stabilizing training. Furthermore, we leverage the distinct temporal dynamics

(Wimbauer et al., 2024; Liu et al., 2025) in flow matching to naturally guide the chunking process.

Building on these insights, we introduce Group Chunking Policy Optimization (GCPO), the first chunk-level RL approach in post-training flow matching for T2I generation. As illustrated in Figure 3, our key innovation lies in shifting the policy optimization paradigm from the step level to the chunk level and treating each chunk as a single action with a principled chunk-level importance ratio. In addition, we incorporate an optional weighted sampling strategy to further enhance GCPO’s performance.

Our contributions can be summarized as follows:

- • We pioneer chunk-level RL for T2I generation. By treating consecutive flow matching steps as a single action, we mitigate the adverse effects of inaccurate advantage attribution presented in GRPO methods, leading to better training stability.
- • We propose GCPO, a novel chunk-level policy optimization approach in post-training flow matching for T2I generation, which integrates chunk-level importance ratio with temporal-dynamic-guided chunking. An optional weighted sampling strategy is introduced to further boost performance.
- • Extensive experiments demonstrate that GCPO achieves superior performance on both preference alignment and standard T2I benchmarks, significantly surpassing prior RL approaches.

### 2. Related Work

#### 2.1. Action Chunking

Action chunking (Zhao et al., 2023; Lai et al., 2025) has been widely applied to robotics (Chi et al., 2025). This approach mitigates compounding error and non-Markovian noise in human demonstrations by jointly predicting a sequence of future actions as a single action, which enables smoother and more stable rollouts. Recently, it has also proven effective in vision-language-action models (Black et al., 2024a; 2025) and in RL (Li et al., 2025b). These successes suggest that action chunking accelerates value propagation and stabilizes long-horizon prediction.

#### 2.2. Reinforcement Learning for Flow-matching-based Image Generation

Diffusion and flow matching models (Ho et al., 2020; Rombach et al., 2022; Podell et al., 2024; Lipman et al., 2022; Batifol et al., 2025; Wu et al., 2025; Wang et al., 2026c) have become one of the dominant paradigms for T2I generation. Early works (Xu et al., 2023; Black et al., 2024b; Fan et al., 2023) introduced RL into diffusion models through

policy gradient optimization. Preference-based methods (Wallace et al., 2024; Sun et al., 2025a;c) were later developed, achieving competitive alignment without explicit reward modeling.

More recently, GRPO (Shao et al., 2024; Sun et al., 2025b) has attracted attention as an efficient alternative. DanceGRPO (Xue et al., 2025) and Flow-GRPO (Liu et al., 2026) pioneered the use of GRPO for T2I generation, unifying diffusion and flow matching through an SDE-based reformulation. MixGRPO (Li et al., 2025a) further improved efficiency via a mixed ODE–SDE paradigm. TempFlowGRPO (He et al., 2025) introduced temporal-aware weighting across denoising steps. Pref-GRPO (Wang et al., 2025) identified the issue of illusory advantage and reformulated the optimization objective as pairwise preference fitting. BranchGRPO (Li et al., 2025c) restructured the rollout process into a branching tree, amortizing computation across shared prefixes.

In contrast to these works, our approach aims to mitigate the adverse effects of one key issue in GRPO methods: inaccurate advantage attribution. By introducing chunk-level policy optimization guided by the inherent temporal dynamics of flow matching, we enhance GRPO from the perspective of policy optimization granularity and stabilize its training. Note that the key difference of our approach from some process-reward methods, such as (Deng et al., 2026), lies in that we do not introduce any process rewards. Instead, we still assign the same outcome reward but shift the optimization from the step level to chunk level. Additional empirical results are provided in Section C.1.

### 3. Preliminary

#### 3.1. Flow Matching

Suppose that x0 ∼ X0 is a data sample from the true distribution, and x1 ∼ X1 is a noise sample. Following (Liu et al., 2023), the intermediate noisy samples xt can be expressed as:

xt = (1 − t)x0 + tx1, (1)

where t ∈ [0,1] denotes the noise level. Then, flow matching aims to directly regress the estimated velocity field vˆθ(xt,t) by minimizing the objective function (Lipman et al., 2023):

0∼X0,x1∼X1[∥v − vˆθ(xt,t)∥22], (2)

LFM(θ) = Et,x

where v = x1 − x0 represents the target velocity field. Furthermore, a deterministic Ordinary Differential Equation (ODE) is utilized to model the forward process of flow matching:

dxt = vˆθ(xt,t)dt. (3)

[Figure 3]

Figure 3. Overall framework of GCPO. GCPO shifts the policy optimization from the step level to the chunk level based on temporaldynamic-guided chunking, using a principled chunk-level importance ratio r. It also introduces an optional weighted sampling strategy that assigns a sampling weight w to each chunk.

#### 3.2. GRPO on Flow Matching

As an RL algorithm, GRPO (Guo et al., 2025; Shao et al., 2024) effectively eliminates the need for an additional critic model by estimating the baseline through group-wise relative rewards. In line with the settings of DDPO (Black et al., 2024b), GRPO is also applied in flow matching. Given a group of G images {xi0}Gi=1 generated from the same prompt c, the advantage corresponding to the i-th sample is formulated as:

r(xi0,c) − mean({r(xj0,c)}Gj=1) std({r(xj0,c)}Gj=1)

Ait =

. (4)

Notice that Ait always keeps the same value for any step t. For simplicity, we neglect the subscript and denote it as Ai. The policy is updated by maximizing the following GRPO objective:

G

T

1 T

1 G

J(θ) = Ec,{xi}Gi=1

t=1

i=1

min rti(θ)Ai,clip rti(θ),1 − ϵ,1 + ϵ Ai − βDKL (πθ||πref) ,

(5)

where rti denotes the step-level importance ratio:

pθ(xit−1|xit,c) pold(xit−1|xit,c)

rti(θ) =

. (6)

Furthermore, to meet the exploration requirement of RL, Flow-GRPO (Liu et al., 2026) and Dance-GRPO (Xue et al.,

2025) introduce stochasticity into flow matching by transforming the deterministic ODE into an equivalent Stochastic Differential Equation (SDE):

σt2 2t

(xt + (1 − t)vθ(xt,t)) dt + σtdwt,

dxt = vθ(xt,t) +

(7) where dwt represents the increments of the Wiener process and σt controls the stochasticity.

### 4. Method

In this section, we begin by reformulating the RL objective from the step level to chunk level and demonstrating why this approach improves GRPO in Section 4.1. Next, we explain how to guide chunking process using the distinct temporal dynamics of flow matching in Section 4.2. Finally, we present our proposed GCPO along with an optional weighted sampling strategy in Section 4.3.

4.1. From Step-level to Chunk-level Policy Optimization We begin by revisiting the two-trajectory example in Figure 2. With the step-level GRPO objective defined in Equation (5), the optimization objective for this example is (omitting the KL term for simplicity):

2

2

- 1

- 2

- 1

- 2

min

J(θ) =

(8)

t=1

i=1

rti (θ)Ai,clip rti (θ),1 − ϵ,1 + ϵ Ai .

As previously discussed, GRPO’s uniform advantage assignment introduces inaccurate advantage attribution, resulting in training instability. To address this issue, the core princi-

ple of chunk-level policy optimization is to group consecutive steps into a chunk and treat it as a single, atomic action. In the example case, where the steps t = 1 and t = 2 form a chunk, the optimization objective becomes to:

2

- 1

- 2

J(θ) =

min

(9)

i=1

ri (θ)Ai,clip ri (θ),1 − ϵ,1 + ϵ Ai ,

where the importance ratio r is redefined over the chunklevel likelihood:

- 1

- 2

2

pθ xit−1|xit,c pold xit−1|xit,c

ri(θ) =

. (10)

t=1

Comparing Equation (9) and Equation (10) with Equation (8) and Equation (6), we observe that the policy optimization paradigm shifts to the chunk level via a principled chunk-level importance ratio. This smooths the misleading fluctuations caused by the inaccurate advantage attribution. A detailed theoretical analysis is provided in Section A.

Building on this insight, we formally formulate chunk-level policy optimization as follows. Given an image generation trajectory:

(xT,xT−1,··· ,x2,x1,x0)i, (11) we partition it into K distinct chunks: 1

{ch1,··· ,chK}i ={(xT,··· ,xT−cs

(12)

,··· ,x1)}i,

1+1),··· ,(xcs

K

where csj denotes the chunk size (the number of consecutive steps within a chunk) of the j-th chunk chj, and:

k

csij = T. (13)

j=1

The chunk-level optimization objective is then:

G

K

1 G

1 K

J(θ) = Ec,{xi}Gi=1

i=1

j=1

min rji (θ)Ai,clip rji (θ),1 − ϵ,1 + ϵ Ai − βDKL (πθ||πref) ,

(14)

where we redefine the importance ratio rji(θ) based on the chunk-level likelihood:

 

 

1 csj

pθ xit−1|xit,c pold xit−1|xit,c

rji(θ) =

. (15)

t∈chj

1we neglect xi0 as it has no preceding transition to xi−1.

[Figure 4]

Figure 4. Performance varies with different chunkings. The chunking with temporal dynamics is our method.

Note that we normalize the importance ratio by the chunk size to ensure stability, following (Zheng et al., 2025).

It is evident that various chunkings exist. For example, setting K = 1 treats the entire generation trajectory as a single chunk, and the policy optimization further shifts from chunk-level to sequence-level, similar to (Zheng et al., 2025). Conversely, setting K = T forces csj = 1, reverting the policy optimization back to step-level GRPO. Consequently, in the next section, we will discuss the principles that guide chunking.

4.2. Chunking Guided by Inherent Temporal Dynamics Before diving into the detailed analysis, we conducted a preliminary experiment where all chunks shared an equal size cs1 = cs2 ··· = csk. As shown in Figure 4, performance varies with chunkings, indicating that the choice of chunking is non-trivial.

We attribute this variation to the inherent temporal dynamics of flow matching. Unlike autoregressive LLMs, flow matching involves time-dependent dynamics during generation, where different steps contribute unequally to the final image. To better understand this, following (Wimbauer et al., 2024; Liu et al., 2025), we illustrate the relative L1 distance L1rel(x,t) throughout the generation process:

L1rel(x,t) = ∥xt − xt−1∥1 ∥xt∥1

. (16)

The relative L1 distance quantifies the rate of change in the latent space during generation. As shown in Figure 5, L1rel(x,t) exhibits prompt-invariant yet step-dependent patterns. Crucially, it naturally segments the generation trajectory into meaningful chunks, where a large L1rel(x,t) indicates rapid latent changes, while a small value implies that adjacent latents are similar. Based on this observation, we argue that steps with similar temporal dynamics (e.g., L1rel(x,t)) should be grouped into the same chunk,

[Figure 5]

- Figure 5. Illustration of the prompt-invariant yet step-dependent temporal dynamics of flow matching, where y-axis denotes the relative L1 distance L1rel(x, t) in Equation (16).

whereas those with distinct dynamics should be separated into different chunks.

Consequently, we guide the chunking process using temporal dynamics (e.g., L1rel(x,t)), aligning the chunk-level policy optimization with the intrinsic temporal dynamics of flow matching. In the next section, we will discuss the chunking implementation details.

#### 4.3. Group Chunking Policy Optimization

We now present Group Chunking Policy Optimization (GCPO), which integrates chunk-level policy optimization with temporal-dynamic-guided chunking.

Specifically, for a given image generation trajectory in Equation (11), we first capture temporal dynamics (e.g., L1rel(x,t) following Equation (16)), and then partition the trajectory into chunks according to them. Treating each chunk as a single action, the policy optimization follows the chunk-level objective in Equation (14). The whole framework is shown in Figure 3.

In practice, we employ an adaptive chunking strategy by iteratively analyzing the derivative of the relative L1 distance for each generation trajectory. Specifically, we first compute the first-order derivatives for each step and group consecutive steps sharing the same sign into chunks. In cases where the derivative signs are uniform across the entire trajectory, we split the sequence at its midpoint. We then recursively apply this process to each resulting chunk using progressively higher-order derivatives (second-order, third-order, etc.). The recursion terminates for a given chunk when its size becomes sufficiently small, and the process concludes once all chunks meet this termination condition.

It is important to note that, unlike other methods that attempt to resolve inaccurate advantage attribution by introducing process rewards, our approach retains the equal assignment of final rewards across all steps (chunks). However, we aim to mitigate the negative influence of this issue by optimizing

the policy at the chunk level with the principled chunk-level importance ratio. This effectively smooths the unexpected misleading gradient fluctuations caused by inaccurate advantage attribution.

Furthermore, we propose an optional weighted sampling strategy to further enhance GCPO. Following Dance-GRPO (Xue et al., 2025), in practice, we select only a subset of chunks (e.g., with fraction 0.5) from each trajectory for training. However, instead of uniform sampling, we assign a sampling weight w to each chunk:

1

csj t∈chj L1rel (x,t)

. (17)

w(chj) =

K j=1

1

csj t∈chj L1rel (x,t)

The motivation stems from our ablation studies in Section 5.3, where we observed varying training contributions from different chunks. As shown in Figure 5, this strategy biases the sampling toward high-noise regions. However, although this strategy enhances GCPO in terms of preference alignment, its overall impact on image quality is mixed and nuanced, as discussed in Section 5.3.

### 5. Experiments

#### 5.1. Experiment Setup

Training Settings. We adopt Dance-GRPO (Xue et al., 2025) and Flow-GRPO (Liu et al., 2026) as baselines, conducting experiments with FLUX.1 Dev (Labs, 2024) as our base model. Comparison of additional baselines is provided in Section C.1. HPDv2.1 (Wu et al., 2023) serves as the dataset, while HPSv3 (Ma et al., 2025) and CLIP (Radford et al., 2021) are used as the primary reward models for the preference alignment task and the standard T2I benchmarks, respectively. In our ablation studies Section 5.3, we additionally validate our approach using Pick Score (Kirstain et al., 2023) as the reward model. Further training details are provided in Section B.4.

Table 2. Performance on GenEval and DPG.

GenEval DPG

Methods

Single Obj. Two Obj. Counting Colors Position Color Attr. Overall Overall

Flux 0.99 0.83 0.71 0.75 0.24 0.44 0.66 84.00 Dance-GRPO 1.00 0.86 0.71 0.78 0.22 0.46 0.67 85.17 Flow-GRPO 0.99 0.84 0.71 0.79 0.21 0.46 0.67 85.05 GCPO w/o ws 0.99 0.85 0.75 0.81 0.21 0.51 0.69 86.60 GCPO w/ ws 0.98 0.82 0.73 0.76 0.27 0.48 0.67 85.14 1 Here and in following tables, the ‘ws’ refers to the weighted sampling strategy.

Evaluation Details. We evaluate performance on both preference alignment and standard T2I benchmarks. For preference alignment, we use HPSv3 (Ma et al., 2025) and ImageReward (Xu et al., 2023) as in-domain and out-ofdomain evaluation metrics, respectively, on the HPDv2.1 (Wu et al., 2023) test set. For the standard T2I benchmark, we report results on GenEval (Ghosh et al., 2023) and DPG (Hu et al., 2024). All evaluations utilize the hybrid inference strategy from (Li et al., 2025a), which has proven effective in mitigating reward hacking. Additionally, we conduct a user study to assess human preferences. More evaluation details are provided in Section B.5.

#### 5.2. Main Results

Table 3 presents results for preference alignment, and Table 2 details performance on the GenEval and DPG benchmarks. GCPO consistently outperforms both the base model and the baselines. In preference alignment, our approach achieves significant relative gains of up to 43% over the baseline across both in-domain and out-of-domain metrics. On GenEval and DPG, our approach achieves the strongest performance, with relative gains up to three times larger than those of the baselines, despite GenEval being a notoriously challenging benchmark when relying solely on CLIP rewards. Notably, while the weighted sampling strategy enhances preference alignment, it yields mixed effects on standard T2I benchmarks, a phenomenon we further analyze in Section 5.3.

Qualitative comparisons in Figure 1 and Section D.1 further highlight GCPO’s improvements in image quality. Our method generates outputs that align more closely with human aesthetic preferences, exhibiting stronger lighting contrast, more vivid colors, and finer details. Notably, the contrast between columns 3 and 4 underscores the importance of chunking in relation to the inherent temporal dynamics of flow matching, which substantially elevates the aesthetic style and structure of the images.

We also conducted a user study to assess human preferences. Reviewers were tasked with selecting the best image from

- 3 alternatives generated by Dance-GRPO, GCPO without weighted sampling strategy, and GCPO with weighted sam-

Table 3. Performance on Preference Alignment. Methods HPSv3 ImageReward

Flux 13.804 1.086 Dance-GRPO 15.080 1.141 Flow-GRPO 14.900 1.135 GCPO w/o ws 15.236 1.147 GCPO w/ ws 15.373 1.149

Table 4. Results of User Study. Methods Win Rate

Dance-GRPO 0.275 GCPO w/o ws 0.350 GCPO w/ ws 0.375

pling strategy, across 40 prompts. The results in Table 4 show that the GCPO variants were preferred by human reviewers 72.5% of the time, which further demonstrates the significant superiority of our approach in aligning with human preferences. More details are provided in Section B.6.

5.3. Ablation Study Additional ablation studies are provided in Section C.2.

Training on Specific Chunks. We analyze the training contributions of distinct temporal regions by training GCPO on individual chunks only. For simplicity, we fix the chunk size and divide the generation trajectory to K = 4 chunks. Results in Figure 7 indicate that while high-noise chunks yield larger improvements than low-noise chunks, they suffer from training instability (e.g., after 60 steps). This observation motivated the weighted sampling strategy in Equation (17), which adaptively emphasizes high-noise chunks to accelerate training while retaining low-noise chunks to ensure stability.

Chunk Setting. In general, there are two ways of constructing chunks. One is the adaptive chunking strategy described in Section 4.3, which we adopt as our default implementation. The other is keeping the chunking fixed during the training, despite different prompts and tempo-

[Figure 6]

- Figure 6. A failure case of the weighted sampling strategy. The strategy erroneously alters the image structure within high-noise regions, resulting in the poorest variant.

[Figure 7]

- Figure 7. Results of training on specific chunks. Lower chunk indices correspond to high-noise regions.

- Table 5. Ablation Results of Different Chunk Settings. Methods Chunking HPSv3 Flux - 13.804

Dance-GRPO - 15.080 GCPO (default) adaptive 15.236

GCPO (fixed)

[2, · · · , 2] 15.115 [4, 4, 4, 4] 15.078

[8, 8] 15.173 [16] 15.142

- Table 6. Ablation Results on Different Reward Models.

###### Model PickScore HPSv3 ImageReward

Flux 22.643 13.804 1.086 Dance-GRPO 23.427 14.612 1.208 Flow-GRPO 23.335 14.610 1.186 GCPO w/o ws 23.442 14.810 1.222 GCPO w/ ws 23.476 14.913 1.233

ral dynamics. Results in Table 5 show that chunk-level optimization consistently outperforms standard step-level GRPO. Moreover, temporal-dynamics-guided chunking outperforms fixed chunk size, underscoring the importance of aligning the optimization process with the intrinsic temporal structure of flow matching. Note that the weighted sampling strategy is disabled here for fairness.

Pick Score (Shukor et al., 2025) as our reward model. Results in Table 6 confirm that our method consistently outperforms step-level GRPO regardless of the reward model used, demonstrating its generalization capabilities.

Weighted Sampling Strategy. As shown in Tables 3 and 2, the optional weighted sampling strategy improves preference alignment but slightly reduces standard T2I benchmark performance. Careful qualitative analysis reveals a tradeoff: while this strategy accelerates policy optimization, it can destabilize image structures in high-noise regions, occasionally leading to semantic collapse. A failure example is depicted in Figure 6. Although all methods struggle with this challenging prompt (e.g., Dance-GRPO fails to generate the attribute ‘sleeveless’), the weighted sampling strategy further degrades the overall image structure, producing the worst outcome by omitting the item ‘black loafers’ entirely and only partially rendering the ‘capris’. This demonstrates the mixed and nuanced effects of the strategy. More qualitative comparison is provided in Section D.2.

### 6. Conclusion

In this paper, we propose GCPO, the first chunk-level policy optimization approach for post-training flow matching in T2I generation. Guided by the temporal dynamics of flow matching, GCPO aggregates consecutive steps into a chunk and optimizes shifts the policy optimization to the chunk level, achieving consistent improvements over GRPO.

Despite its strong performance, several limitations remain. For example, exploring how to combine heterogeneous rewards across different chunks (e.g., employing different reward models for high- vs. low-noise regions) could unlock further improvements.

Reward Models. Finally, we test GCPO’s robustness across different reward models. We replace HPSv3 with

### Acknowledgements

This work was partially supported by the China Postdoctoral Science Foundation under Grant No. 2025M781490, the Natural Science Foundation of Shenzhen under Grant Nos. JCYJ20230807111604008 and JCYJ20240813112007010, the Natural Science Foundation of Guangdong Province under Grant No. 2024A1515010003 and National Key Research and Development Program of China under Grant No. 2022YFB4701400.

### Impact Statement

This paper presents GCPO, a chunk-level reinforcement learning approach designed to enhance the visual quality and preference alignment of text-to-image flow matching models. Our work aims to advance the capabilities of generative AI as a tool for creative expression and visual synthesis. However, we acknowledge that the significant improvements in structural coherence, lighting, and photorealism demonstrated by GCPO could potentially be misused to generate misleading content or deepfakes that are difficult to distinguish from reality. Furthermore, since our optimization relies on preference reward models, there is an inherent risk of amplifying biases present in these evaluators or the underlying training data. We encourage the community to deploy such advanced post-training methods alongside robust safety filters, watermarking technologies, and diverse reward modeling to mitigate these risks.

### References

Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., et al. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter,

- B., et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024a.

Black, K., Janner, M., Du, Y., Kostrikov, I., and Levine, S. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pp. 4965–4987, 2024b.

Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M. R., Finn, C., Fusai, N., Galliker, M. Y., et al. π0.5: a vision-language-action model with open-world generalization. In Proceedings of The 9th Conference on Robot Learning, volume 305, pp. 17–40, 2025.

Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11): 1684–1704, 2025.

Deng, H., Yan, K., Mao, C., Wang, X., Liu, Y., Gao, C., and Sang, N. Densegrpo: From sparse to dense reward for flow matching model alignment. arXiv preprint arXiv:2601.20218, 2026.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for highresolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pp. 12606–12633, 2024.

Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., and Lee, K. DPOK: Reinforcement learning for fine-tuning text-toimage diffusion models. In Advances in Neural Information Processing Systems, volume 36, pp. 79858–79885, 2023.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. In Advances in Neural Information Processing Systems, volume 36, pp. 52132–52152, 2023.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, X., Fu, S., Zhao, Y., Li, W., Yang, J., Yin, D., Rao, F., and Zhang, B. TempFlow-GRPO: When timing matters for grpo in flow models. arXiv preprint arXiv:2508.04324, 2025.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851, 2020.

Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., and Yu, G. ELLA: Equip diffusion models with LLM for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. OpenAI o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., and Levy, O. Pick-a-Pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems, volume 36, pp. 36652–36663, 2023.

Labs, B. F. Flux. https://github.com/ black-forest-labs/flux, 2024.

Lai, L., Huang, A. Z., and Gershman, S. J. Action chunking as conditional policy compression. Cognition, 264: 106201, 2025.

Li, J., Cui, Y., Huang, T., Ma, Y., Fan, C., Yang, M., and Zhong, Z. MixGRPO: Unlocking flow-based GRPO efficiency with mixed ODE-SDE. arXiv preprint arXiv:2507.21802, 2025a.

Li, Q., Zhou, Z. P., and Levine, S. Reinforcement learning with action chunking. In Advances in Neural Information Processing Systems, volume 38, pp. 55518–55553, 2025b.

Li, Y., Wang, Y., Zhu, Y., Zhao, Z., Lu, M., She, Q., and Zhang, S. BranchGRPO: Stable and efficient GRPO with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025c.

Liang, Z., Yuan, Y., Gu, S., Chen, B., Hang, T., Cheng, M., Li, J., and Zheng, L. Aesthetic post-training diffusion models from generic preferences with step-by-step preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13199–13208, 2025.

Liao, X., Wei, W., Qu, X., and Cheng, Y. Step-level reward for free in RL-based T2I diffusion model fine-tuning. arXiv preprint arXiv:2505.19196, 2025.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2022.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In The 11th International Conference on Learning Representations, 2023.

Liu, F., Zhang, S., Wang, X., Wei, Y., Qiu, H., Zhao, Y., Zhang, Y., Ye, Q., and Wan, F. Timestep embedding tells: It’s time to cache for video diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7353–7363, 2025.

Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-GRPO: Training flow matching models via online RL. In Advances in Neural Information Processing Systems, volume 38, pp. 40783–40818, 2026.

Liu, X., Gong, C., et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The 11th International Conference on Learning Representations, 2023.

Luo, Y., Hu, X., Fan, K., Sun, H., Chen, Z., Xia, B., Zhang, T., Chang, Y., and Wang, X. Reinforcement learning meets masked generative models: Mask-GRPO for textto-image generation. In Advances in Neural Information Processing Systems, volume 38, pp. 108460–108485, 2026.

Ma, Y., Wu, X., Sun, K., and Li, H. HPSv3: Towards widespectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15086–15095, 2025.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The 12th International Conference on Learning Representations, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139, pp. 8748–8763, 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., et al. SmolVLA: A vision-languageaction model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

Sun, H., Liang, B., Xia, B., Wu, J., Zhao, Y., Qin, K., Chang, Y., and Wang, X. Diffusion-RainbowPA: Improvements integrated preference alignment for diffusion-based textto-image generation. Transactions on Machine Learning Research, 2025a.

Sun, H., Wu, J., Xia, B., Luo, Y., Zhao, Y., Qin, K., Lv, X., Zhang, T., Chang, Y., and Wang, X. Reinforcement fine-tuning powers reasoning capability of multimodal large language models. arXiv preprint arXiv:2505.18536, 2025b.

Sun, H., Xia, B., Chang, Y., and Wang, X. Generalizing alignment paradigm of text-to-image generation with preferences through f-divergence minimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 27644–27652, 2025c.

Sutton, R. S., Barto, A. G., et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Tong, Y., Liu, M., Zhao, C., He, W., Zhang, S., Zhang, H., Zhang, P., Liu, J., Huang, J., Wang, J., et al. Alleviating sparse rewards by modeling step-wise and long-term sampling effects in flow-based GRPO. arXiv preprint arXiv:2602.06422, 2026.

Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Wang, F. and Yu, Z. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025.

Wang, H., Ma, G., Cui, S., Kong, Y., Luo, H., Shen, L., Gao, M., Wu, Y., Wang, X., and Tao, D. Language-based trial and error falls behind in the era of experience. arXiv preprint arXiv:2601.21754, 2026a.

Wang, H., Zhao, Y., Qin, Z., Du, C., Lin, M., Wang, X., and Pang, T. Lifelong safety alignment for language models. In Advances in Neural Information Processing Systems, volume 38, pp. 19908–19938, 2026b.

Wang, J., Lai, Z., Chen, J., Guo, J., Guo, H., Li, X., Yue, X., and Guo, C. Elastic diffusion transformer. arXiv preprint arXiv:2602.13993, 2026c.

Wang, Y., Li, Z., Zang, Y., Zhou, Y., Bu, J., Wang, C., Lu, Q., Jin, C., and Wang, J. Pref-GRPO: Pairwise preference reward-based GRPO for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025.

Wimbauer, F., Wu, B., Schoenfeld, E., Dai, X., Hou, J., He, Z., Sanakoyeu, A., Zhang, P., Tsai, S., Kohler, J., et al. Cache me if you can: Accelerating diffusion models through block caching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6211–6220, 2024.

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., and Li, H. Human preference score v2: A solid benchmark for

evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. ImageReward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, volume 36, pp. 15903–15935, 2023.

Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al. DanceGRPO: Unleashing GRPO on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Zhang, Z., Zheng, C., Wu, Y., Zhang, B., Lin, R., Yu, B., Liu, D., Zhou, J., and Lin, J. The lessons of developing process reward models in mathematical reasoning. In Findings of the Association for Computational Linguistics, pp. 10495– 10516, 2025.

Zhao, T. Z., Kumar, V., Levine, S., and Finn, C. Learning fine-grained bimanual manipulation with low-cost hardware. In ICML Workshop on New Frontiers in Learning, Control, and Dynamical Systems, 2023.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

### A. Theoretical Analysis

Here, we present the superiority of chunk-level policy optimization from the perspective of gradient weights, which demonstrates why it offers a more stable training.

To begin with, the gradient of the GRPO’s objective in Equation (5) can be derived as follows (clipping and KL are omitted for brevity)2:

G

T

1 T

1 G

rti (θ)Ai

∇θJGRPO(θ) = ∇θEc,x

t=1

i=1

(18)

Given Equation (6), this simplifies to:

= ∇θEc,x

G

T

1 T

1 G

Ai

rti (θ) · ∇θlog rti (θ) .

t=1

i=1

G

T

pθ(xit−1|xit,c) pold(xit−1|xit,c) · ∇θlog pθ(xit−1|xit,c) . (19)

1 G

1 T

Ai

∇θJGRPO(θ) = ∇θEc,x

t=1

i=1

In comparison, the gradient of the chunk-level policy optimization objective in Equation (14) is derived below. For convenience, we denote the chunk-level importance ratio from Equation (15) as:

 

 

1 csj

pθ xit−1|xit,c pθ

sij(θ) =

, (20)

xit−1|xit,c

old

t∈chj

Then the gradient becomes:

 

  1

K

G

1 K

sij (θ)Ai

∇θJGCPO(θ) = ∇θEc,x

G

j=1

i=1

  1

 

G

K

1 K

Ai

sij (θ) · ∇θlog sij (θ)

= ∇θEc,x

G

i=1

j=1

(21)

G

1 G

Ai

= ∇θEc,x

i=1

  

  

 

 

1 csj

K

pθ xit−1|xit,c pθ

1 csj t∈ch

1 K

∇θlog pθ(xit−1|xit,c)

·

xit−1|xit,c

old

j=1

t∈chj

j

Therefore, the fundamental distinction between chunk-level policy optimization and step-level GRPO lies in how they weight the gradients of the log likelihoods. In step-level GRPO, they are weighted individually according to their respective importance weight pθ(x

i t−1|xit,c)

pold(xit−1|xit,c). However, these unequal weights, which can vary among (0,1 + ϵ] for A ≥ 0 or [1 − ϵ,∞) for A ≤ 0, are not negligible, and their impact can accumulate and lead to unstable consequences. In contrast, chunk-level policy optimization applies a unified weight t∈ch

1 csj

pθ(xit−1|xit,c) pθold(xit−1|xit,c)

to all timesteps within a chunk, effectively smoothing these fluctuations and eliminating training instability.

j

2The omission of clipping and KL terms makes no difference to the gradient derivation. When the clipping occurs, the gradient of the clipped term is. Regarding KL, it is identical for both GCPO and GRPO, therefore its gradient contribution is the same.

### B. Details

#### B.1. Details for Figure 1

The “GCPO w/o Temporal Dynamics” variant in Figure 1 is with the setting that all steps are treated as a single chunk (k = 1).

#### B.2. Details for the Frequency of Inaccurate Advantage Attribution

We used Flux (Labs, 2024) as the test model and followed its default parameters for generation. Specifically, the generation timestep is T = 50. All final rewards and step advantages are obtained by the step-aware preference model from (Liang et al., 2025).

#### B.3. Details for the Process Reward Comparison Table 7. Results of Process Reward

To validate the limitations of current process reward models, we replaced the reward in GRPO (Shao et al., 2024; Guo et al., 2025) with the process reward from (Liang et al., 2025), and compared it to the standard GRPO. The evaluation metric is HPSv2 (Wu et al., 2023). The results in Table 7 demonstrates the limitations of current process reward models.

Method HPSv2

Flux 0.304 Dance-GRPO 0.365 Process Reward 0.348

#### B.4. Details for Training

All experiments were conducted on 8 Nvidia GPUs. The hyperparameters are summarized in Table 8, remaining the same for all baselines and our approach.

Table 8. Hyperparameter Settings

Parameter Value Parameter Value

Learning rate 1 × 10−5 Weight decay 1 × 10−4 Train batch size 2 SP size 1 SP batch size 2 Max grad norm 0.01 Resolution 720 × 720 Sampling steps 17 Eta 0.7 Num. generations 12 Grad. accum. steps 12 Shift (branch offset) 3 Clip range 5 × 10−5 Training steps 150 Timestep fraction 0.5

#### B.5. Details for Evaluation Table 9. Performance on Preference Alignment. Methods HPSv3 ImageReward Flux 0.304 1.086

We set the generation timestep T = 50 during evaluation. Following (Li et al., 2025a), the first 30 steps are sampled with the trained model, while the remaining 20 steps are sampled with the base model. This hybrid inference strategy and corresponding settings, also used in (Li et al., 2025a), have proven effective in mitigating reward hacking.

Dance-GRPO 15.080 1.141 Flow-GRPO 14.900 1.135 Pref-GRPO 14.868 1.139 MixGRPO 15.128 1.146 TP-GRPO 15.206 1.148 GCPO 15.236 1.149

#### B.6. Details for User Study

During the test phase, ten reviewers possessing good aesthetic tastes were recruited. We extracted 40 prompts from our evaluation dataset - HPDv2.1 test set(Wu et al., 2023).

### C. Additional Experiments

C.1. Additional Baselines

We provide performance comparison with additional baselines: Pref-GRPO (Wang et al., 2025), MixGRPO (Li et al., 2025a), and TP-GRPO (Tong et al., 2026), where TP-GRPO is a process-reward approach. Note that the weighted sampling strategy is disabled here. Table 9 demonstrates that, our method outperforms the step-level approach by solving the sparse reward issue from the unified chunk-level importance ratio perspective.

C.2. Additional Ablation Studies

We used relative L1 distance L1rel(x,t) to capture temporal dynamics in the implementation. However, there are also other options. Additionally, we conduct ablation studies comparing L1, L2, and cosine similarity. Table 10 shows that despite L2 even achieves better performance in some cases, the consistent improvement over various distance matrices highlights the effectiveness of chunk-level optimization.

Table 10. Ablation Results of Different Distance Metrix.

Methods HPSv3 ImageReward

Flux 0.304 1.086 L1 15.236 1.147 L2 15.268 1.149 Cosine 15.220 1.147

D. Additional Visualization

#### D.1. Main Visualization

Figure 8, Figure 9, Figure 10, and Figure 11 present qualitative comparisons among FLUX, Dance-GRPO, GCPO without temporal dynamics, and GCPO with temporal dynamics. Overall, GCPO generates outputs that align more closely with human aesthetics, exhibiting stronger lighting contrast, more vivid colors, and finer details. Notably, the contrast between columns 3 and 4 in Figure 8 and Figure 9 underscores the importance of chunking with the inherent temporal dynamics of flow matching, which substantially elevates aesthetic style and structure of images.

#### D.2. Visualization Comparison of the Weighted Sampling Strategy

Figure 12 presents additional qualitative comparisons of the weighted sampling strategy (the last column). While this strategy improves preference alignment, it destabilizes image structures in high-noise regions.

[Figure 8]

[Figure 9]

[Figure 10]

###### Figure 10. Additional visualization comparison between the FLUX, DanceGRPO, and GCPO. The GCPO here is with temporal dynamics.

[Figure 11]

###### Figure 11. Additional visualization comparison between the FLUX, DanceGRPO, and GCPO. The GCPO here is with temporal dynamics.

[Figure 12]

###### Figure 12. Additional visualization comparison of the Weighted Sampling Strategy (the last column).

