## Training-Free Reasoning and Reflection in MLLMs

Hongchen Wei1 and Zhenzhong Chen*1 1School of Remote Sensing and Information Engineering, Wuhan University

# arXiv:2505.16151v1[cs.CV]22May2025

###### Abstract

Recent advances in Reasoning LLMs (e.g., DeepSeek-R1 and OpenAI-o1) have showcased impressive reasoning capabilities via reinforcement learning. However, extending these capabilities to Multimodal LLMs (MLLMs) is hampered by the prohibitive costs of retraining and the scarcity of high-quality, verifiable multimodal reasoning datasets. This paper introduces FRANK Model, a training-FRee ANd r1-liKe MLLM that imbues off-the-shelf MLLMs with reasoning and reflection abilities, without any gradient updates or extra supervision. Our key insight is to decouple perception and reasoning across MLLM decoder layers. Specifically, we observe that compared to the deeper decoder layers, the shallow decoder layers allocate more attention to visual tokens, while the deeper decoder layers concentrate on textual semantics. This observation motivates a hierarchical weight merging approach that combines a visual-pretrained MLLM with a reasoning-specialized LLM. To this end, we propose a layer-wise, Taylor-derived closed-form fusion mechanism that integrates reasoning capacity into deep decoder layers while preserving visual grounding in shallow decoder layers. Extensive experiments on challenging multimodal reasoning benchmarks demonstrate the effectiveness of our approach. On the MMMU benchmark, our model FRANK-38B achieves an accuracy of 69.2, outperforming the strongest baseline InternVL2.5-38B by +5.3, and even surpasses the proprietary GPT-4o model. Our project homepage is at: http://iip.whu.edu.cn/frank/index.html

###### 1 INTRODUCTION

Recent reasoning-focused large language models (LLMs) [1, 2, 3, 4] such as DeepSeek-R1 [1] and OpenAI-o1 [5] have demonstrated strong performance in tasks requiring complex logic, including math reasoning, symbolic manipulation, and program synthesis. These models leverage mechanisms like reinforcement learning to perform multi-step problem-solving and iterative self-correction, often surpassing even human experts.

In real-world scenarios, numerous tasks demand sophisticated multimodal reasoning capabilities. For instance, solving visual mathematics problems, interpreting diagrams, and understanding code snippets embedded within images require the integration of visual perception with logical reasoning. Inspired by the successes of reasoning-augmented LLMs, researchers have begun exploring methods to endow multimodal large language models (MLLMs) [6, 7, 8, 9] with similar reasoning abilities. A prevalent approach [10, 11, 12] involves adapting reinforcement learning techniques, such as GRPO [1], to the multimodal context. However, this strategy encounters significant challenges. Firstly, the reinforcement learning training of large-scale MLLMs demands substantial computational resources, making it resource-intensive. Secondly, there is a notable scarcity of high-quality, verifiable multimodal reasoning datasets, which are essential for effective training. This paucity of suitable data severely impedes the development and scalability of reasoningcapable MLLMs.

In this paper, we introduce FRANK, a training-free and R1like MLLM that is designed to endow existing MLLMs with advanced reasoning and reflection capabilities without any additional training or supervision. Figure 1 visualizes this pipeline. Our method is built upon two key insights:

• Homologous Model Merging: We conceptualize MLLMs as base LLMs fine-tuned on visual-text data, while reasoning-

Corresponding author: Zhenzhong Chen, E-mail:zzchen@ieee.org

specialized LLMs represent the same base LLM fine-tuned on reasoning tasks. According to the task arithmetic hypothesis [13, 14, 15], the difference in weights between a fine-tuned model and its base model encapsulates the task-specific adaptations. By merging task vectors from models fine-tuned on different tasks, we can integrate multiple capabilities into a single model without additional training.

• Layer-wise Functional Specialization in MLLMs: Drawing inspiration from the hierarchical processing observed in the human brain [16, 17], where sensory inputs are initially processed in primary sensory areas and progressively integrated into higher-order cognitive functions in association cortices, we observe a similar pattern in MLLMs. As shown in Figure 2, compared to the deeper decoder layers, the shallow decoder layers allocate more attention to visual tokens, facilitating perceptual grounding, while the deeper decoder layers concentrate on textual semantics.

Based on these two key insights, we design a hierarchical weight merging strategy to effectively integrate a vision-text pretrained MLLM with a reasoning-specialized LLM. Specifically, we develop a layer-wise, Taylor-derived closed-form fusion mechanism that enables fine-grained control over the contribution of each model at different depths of the decoder. This mechanism builds on the task vector formulation and refines it with a layer-wise optimization strategy: For each decoder block, we derive a closed-form solution for the optimal task vector fusion weights by minimizing the Taylor-approximated task loss difference. This allows precise control over how visual and reasoning adaptations are combined at each layer. Furthermore, guided by the empirical prior that shallow decoder layers in MLLMs attend more to visual inputs while deeper layers focus on symbolic reasoning, we incorporate layer-dependent fusion weights to align with the distinct functional roles across the model hierarchy. This design enables FRANK to embed reasoning capabilities into deeper layers, responsible for abstraction and reasoning, while preserving the visual grounding in shallower layers, which

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Answer: C

<think> Okay, so I need to find the measure of angle H in this figure.

Non-Reasoning MLLM

[Figure 5]

The figure is a pentagon

<think> Okay, so I need to

Answer the question

with vertices labeled E, F,G, H, J … </think>

find the measure of angle H in this figure. But I can't see the image. … </think>

based on the provided

Reasoning MLLM (Ours)

image. Question: Find ∠H Choices: (A) 97 (B) 102 (C) 107 (D) 122

| |
|---|

Answer: A

Answer: I can't see the image.

Reasoning LLM

- Figure 1: Non-reasoning MLLMs lack reasoning and reflection abilities, while reasoning LLMs are unable to perceive visual information. We propose a training-free, closed-form layerwise fusion method that combines visual perception and language reasoning strengths, substantially enhancing overall reasoning capability in multimodal settings.

are more sensitive to perceptual signals.

the model to generate structured outlines before deriving final answers. In concurrent work, methods like MM-Eureka [26], Vision-R1 [10], and LMM-R1 [11] extend reinforcementlearning-based reasoning (e.g., GRPO) to the multimodal setting, using multimodal datasets to fine-tune MLLMs. However, all of these approaches require costly secondary training and depend on the availability of large, high-quality multimodal reasoning corpora, which remain extremely scarce, thus limiting their scalability and generalization.

We validate the effectiveness of FRANK through comprehensive evaluations on challenging multimodal reasoning benchmarks. Notably, our model FRANK-38B achieves an accuracy of 69.2 on the MMMU benchmark, outperforming the strongest baseline InternVL2.5-38B by +5.3, and even surpasses the proprietary GPT-4o model. These results underscore FRANK’s ability to enhance reasoning capabilities in MLLMs without additional training or supervision.

In summary, our contributions are threefold:

2.2 Model Merging

- • The layer-wise functional specialization in MLLMs, where shallow decoder layers focus on visual perception and deeper layers on textual reasoning, is identified and leveraged.
- • A novel, training-free hierarchical weight fusion mechanism is proposed, integrating reasoning capabilities into existing MLLMs by merging task vectors at each layer, guided by a Taylor-derived closed-form solution.
- • It is demonstrated that FRANK effectively enhances the reasoning and reflection abilities of MLLMs, achieving superior performance on multimodal reasoning tasks without the need for additional training or supervision.

Model merging [27, 13, 14, 28, 29, 30] has emerged as an efficient alternative to multi-task fine-tuning, enabling the fusion of multiple task-specific models into a single model without accessing original training data. Early work on weight averaging [13] showed that simple parameter interpolation can improve robustness across tasks. Task Arithmetic [14] generalized this idea by introducing task vectors that represent the differences between fine-tuned and base model weights, and linearly combining them to incorporate new capabilities. To address interference between tasks, TIES-Merging [28] trims low-magnitude updates and aligns parameter signs before merging, while DARE [31] further sparsifies and rescales delta parameters to reduce redundancy. MetaGPT [29] formalizes model merging as a multi-task learning problem, achieving balanced task performance by computing the norms of different model parameters. Although these methods achieve strong results for text-only LLMs, they do not exploit the unique structure of multimodal models. In concurrent work, VLM-Merging [32] employs fixed fusion weights (with an MLLM weight of 0.9 and an LLM weight of 0.1) to merge MLLM and LLM, enabling the MLLM to acquire reasoning capabilities. However, such manually designed fusion weights are often suboptimal and struggle to transfer the LLM’s reflective abilities effectively. In this work, we extend the model merging paradigm to MLLMs by incorporating multimodal priors, specifically, layer-wise perception-to-cognition specialization, into the fusion process, thereby enabling training-free integration of visual perception and reasoning capabilities.

###### 2 Related Work

- 2.1 Multimodal Large Language Models

Multimodal large language models (MLLMs) have recently advanced the integration of visual and linguistic understanding through large-scale vision-language pretraining and instruction tuning. Early works [18, 19, 20, 21] enable zero- or few-shot visual question answering, image captioning, and multimodal instruction following by augmenting an LLM with a pretrained visual encoder. While these models excel at aligning visual features with text, their performance on tasks requiring multi-step logical reasoning or self-reflection remains limited.

Building on chain-of-thought (CoT) [22] prompting in pure-text LLMs, several recent works [6, 7, 8, 9, 10, 11, 12] attempt to endow MLLMs with explicit reasoning capabilities. LLaVAReasoner [23] shows that naively appending CoT prompts often yields marginal gains or can even degrade MLLM accuracy. To better guide intermediate reasoning, CCoT [24] and TextCoT [25] introduce plan-based CoT prompting, prompting

###### 3 Approach

In this section, we introduce FRANK, a training-free and R1-like MLLM that endows off-the-shelf MLLMs with advanced reasoning and reflection by merging them with reasoning-specialized

LLMs, entirely without gradient updates or extra supervision. At its core, FRANK relies on two principles: (1) homologous model merging, which treats both models as task-fine-tuned variants of the same base LLM and fuses them via task-vector arithmetic; and (2) hierarchical layer-wise fusion, which exploits the observation that shallow decoder layers chiefly process visual inputs while deep layers focus on text, allowing us to inject reasoning only where it is most effective.

- 3.1 Preliminary: Task Arithmetic for Homologous Model Merging

We build on the paradigm of task arithmetic [14], which provides a simple yet effective mechanism to merge multiple fine-tuned models, so-called homologous models, that share the same base architecture. Let f(·;θ0) denote a pre-trained base model, and let

θt = argmin

Lt f(·;θ),Dt (1)

θ

be the parameters obtained by fine-tuning on task t with loss Lt over dataset Dt. We define the task vector for task t as τt = θt − θ0 . Under the homologous assumption that all θt lie in the same parameter space as θ0, we can form a merged model by a linear combination of these task vectors:

λt τt , (2)

###### θf = θ0 +

t∈{V,R}

where {λt} are non-negative fusion weights controlling each task’s contribution.

Early approaches choose all λt heuristically (e.g., a constant 0.3) or via grid search on held-out data, but these methods either under-utilize model capacity or incur prohibitive search costs as T grows.

- 3.2 FRANK: a Training-free and R1-like MLLM

Building on the task-arithmetic preliminary, FRANK introduces two key innovations: Layer-wise Fusion, which respects the functional specialization of each decoder layer (for clarity, we define a decoder’s block as a “layer”); Modality Priors, which steer shallow layers toward visual grounding and deep layers toward symbolic reasoning. We next detail (i) how we decompose the MLLM decoder into per-layer task vectors; (ii) the Taylor-based derivation of closed-form fusion weights at each layer; and (iii) the incorporation of layer-dependent modality priors.

- 3.2.1 Layer-Wise Fusion Setup for MLLMs

FRANK bridges these two paradigms by viewing the vision-finetuned decoder and a reasoning-specialized decoder as “homologous” variants of the same base model checkpoint. Leveraging their shared architecture, we can merge their strengths without retraining. The decoder consists of L stacked transformer layers, indexed by l = 1,..., L. Figure 2 shows that, compared to the deeper decoder layers, the shallow decoder layers allocate more attention to visual tokens, facilitating perceptual grounding, while the deeper decoder layers concentrate on textual semantics. To preserve this functional hierarchy and prevent interference between vision and reasoning, FRANK performs

layer-wise weight fusion: each decoder layer is merged independently, preserving its specialized role.

Concretely, let θ0(l),θV(l),θR(l) denote the parameters of layer l for the pre-trained base model, the vision-fine-tuned MLLM, and

the reasoning-fine-tuned LLM, respectively. We define the layerwise task vectors:

τ(Vl) = θV(l) − θ0(l), τ(Rl) = θR(l) − θ0(l). (3)

These differences capture how each task (vision vs. reasoning) shifts the model weights at each depth.

To measure the impact of fusing these shifts, we introduce two metrics. First, the Layer-Wise Task Loss Difference (LTLD) compares the fused layer’s loss against each branch’s own finetuned layer:

- Definition 1 (Layer-Wise Task Loss Difference, LTLD). Let

L(tl)(θ, x) be the loss of branch t ∈ {V,R} at layer l on input x. For fusion weights (λV,λR), define the fused parameter

θ(fl) = θ0(l) + λV τ(Vl) + λR τ(Rl), (4)

where, θ(fl) represents the parameters of the fusion model at layer l, while τ(Vl) and τ(Rl) denote the task vectors at layer l of the nonreasoning MLLM and reasoning-specialized LLM, respectively. Then, we define the layer-wise task loss difference (LTLD) as

LTLD(tl)(λV,λR) = L(tl)(θ(fl), x) − L(tl)(θb(l), x), (5)

quantifies the degradation when using the fused weights in place of branch b’s own.

- Definition 2 (Layer-Wise Average Loss Difference, LALD). The LALD averages LTLD across both branches,

LALD(l)(λV,λR) = 12 LTLD(Vl) + LTLD(Rl) , (6)

and serves as our per-layer fusion objective. By minimizing LALD independently at each layer, we derive fusion weights that optimally balance visual grounding and reasoning without cross-layer interference.

We emphasize that LTLD is a theoretical sensitivity measure, not an operation we perform in implementation. Specifically, LTLD represents the second-order Taylor approximation of the loss increase induced by small perturbations in the parameters of layer l, holding all other layers fixed. Under the Neural Tangent Kernel (NTK) [33] linearization and task-vector orthogonality assumptions, this localized view yields a tractable quadratic

form in the norms |τ(Vl)|2 and |τ(Rl)|2, from which we derive closedform fusion weights (see Section 3.2.2). Importantly, we do

not compute per-layer losses on the fly; LTLD merely guides our analytical derivation. In subsequent sections, we will show how to approximate LTLD via a second-order Taylor expansion, invoke NTK linearization and vector orthogonality, and obtain a

data-agnostic closed-form for (λ(Vl),λ(Rl)). 3.2.2 Taylor-Based Approximation and Closed-Form Fusion

Weights

To obtain efficient, data-agnostic fusion weights at each decoder layer, we approximate the layer-wise average loss difference (LALD) using a second-order Taylor expansion. Intuitively,

0.35

0.30

Visionattentionweights

0.25

0.20

0.15

0.10

0.05

0.00

0 5 10 15 20 25 30 35 40 45

Number of layer

- Figure 2: Layer-wise visual attention of NVIL-15B. Each curve shows the average attention from a text token to all visual tokens across layers. Shallow layers assign significantly higher attention to visual tokens, while attention in deeper layers approaches zero and rapidly descends indicating a shift from perception to language reasoning. This supports our use of an exponential decay prior to the fusion process.

0.5

0.4

CosineSimilarity

0.3

0.2

0.1

0.0

0 4 8 12 16 20 24 28 32 36 40 44

Decoder Block Index

Figure 3: Cosine similarity between task vectors of vision-finetuned (NVIL-15B) and reasoning-finetuned (DeepSeekDistil-Qwen2.5-14B) models at each decoder block. The task vector at each block is computed by flattening the weight deltas with respect to the base model. The similarity remains close to 0 across all layers, indicating strong nearorthogonality.

this expansion captures how a small perturbation—arising from merging vision and reasoning task vectors—affects the loss. Under two standard assumptions:

NTK Linearization. In the infinite-width regime, neural networks evolve under training according to a fixed NTK [33], which implies that small weight perturbations produce locally linear changes in the model output [33, 29]. The appendix A.5 provides additional details. Previous study [29] has empirically validated this NTK-linear behavior for LLMs. They evaluated LLaMA-2-7b-chat-hf [34] on the AGIEval benchmark [35], sampling three random prompts and measuring model outputs for interpolation coefficients α ∈ {0,0.1,...,1.0}. The output trajectories scale almost perfectly linearly with α, confirming that LLM operates in the NTK regime during fine-tuning, which is specifically suitable for the LLM’s arithmetic scenario.

Task-Vector Orthogonality. Although both vision and reasoning fine-tuning update the same decoder weights, their resulting task vectors often lie in nearly orthogonal subspaces. As shown in Figure 3, we verify this by computing the cosine similarity

between τ(Vl) and τ(Rl) at each layer, which remains close to zero across all layers. The appendix A.6 provides additional details.

Under the NTK linearization and task-vector orthogonality assumptions, we can now quantify how merging vision and reasoning updates perturbs the layer loss. In the next step, we expand each task’s loss around its fine-tuned parameters to derive a closed-form bound on the loss increase.

Let θ0(l) be the initialization at layer l, and θt(l) = θ0(l) + τ(tl) be the fine-tuned weights for task t ∈ {V,R}. Define the fused weights

θ(fl) = θ0(l) +

λ(tl) τ(tl) (7)

t∈{V,R}

and the fusion residual

h(tl) = θ(fl) − θt(l) =

λ(kl) τ(kl) − (1 − λ(tl))τ(tl). (8)

k t∈{V,R}

Around θt(l), a second-order Taylor expansion of the layer loss

L(tl) gives L(tl)(θ(fl)) ≈ L(tl)(θt(l)) + ∇L(tl)(θt(l))⊤h(tl)

+21 h(tl)⊤∇2L(tl)(θt(l))h(tl),

≈0

(9) where the first-order term vanishes under near-convergence, as guaranteed by the NTK theory for wide networks. This NTK regime further implies two key properties for the second-order term: 1) The Hessian is dominated by the Jacobian Gram matrix due to the approximate linearity of the network in parameter space, and 2) Its eigenvalue distribution becomes approximately isotropic. Concretely, the Hessian admits the following approximation:

∇2L(tl)(θt(l)) ≈ δ(tl) Id

, δ(tl) = d1

tr ∇f(l)(xt;θ0(l))∇f(l)(xt;θ0(l))⊤ ,

l

l

(10)

where, δ(tl) is a data-dependent constant, dl represents the parameter dimension of the l-th layer, Id

denotes the dl-dimensional identity matrix, which is used to simplify the structure of the Hessian. Consequently,

l

(l) t

LTLD(tl) = L(tl)(θ(fl)) − L(tl)(θt(l)) ≤ δ

2 ∥h(tl)∥2. (11)

Summing the above bound for t ∈ {V,R} and using ⟨τ(Vl),τ(Rl)⟩ ≈ 0 yields a layer-wise bound on the average loss increase:

δ(tl) (1−λ(tl))2∥τ(tl)∥2+

(λ(kl))2∥τ(kl)∥2 ∥τ(tl)∥2.

LALD(l) ≤ 21

t∈{V,R}

k t∈{V,R}

(12) To derive the optimal fusion weights, we minimize the LALD bound under the constraint λV + λR = 1. Setting the derivative to zero and solving yields the closed-form solution:

λ(tl) = ∥τ(tl)∥2 ∥τ(Vl)∥2 + ∥τ(Rl)∥2

, t ∈ {V,R}. (13)

This compact expression emphasizes FRANK’s key advantage: fusion weights depend solely on the observed parameter shifts of each task, eliminating the need for held-out data, grid search, or additional training. For full statements of the Taylor expansion lemma, formal properties, and proofs leading to this closed form, please refer to Appendix A.7 and A.8.

- Table 1: Comparison of FRANK variants (8B, 15B, 38B) and state-of-the-art baselines across five multimodal reasoning benchmarks: MMMU val, MMMU-Pro standard (10 opts), MathVista testmini, MathVision testmini, and WeMath testmini. * indicates the baseline model.

Methods Size MMMU MMMU-Pro MathVista MathVision WeMath

LLaVA-1.5 [36] [2023.10.05] 7B 35.7 19.7 25.6 10.2 7.0 LLaVA-NeXT [37] [2024.01.30] 7B 35.3 19.4 24.9 10.0 3.3

LLaVA-LLaMA3 [37] [2024.05.10] 8B 39.2 - 40.0 - VILA1.5-LLaMA3 [38] [2024.05.16] 8B 38.6 - 36.7 - -

Samll

Idefics3-LLaMA3* [39] [2024.08.07] 8B 43.9 32.6 58.4 20.1 12.3 FRANK (Ours) 8B 48.3 34.7 50.7 27.6 11.7

LLaVA-1.5 [36] [2023.10.05] 13B 37.0 - 27.7 13.1 7.4

ShareGPT4V [40] [2023.11.21] 13B 36.6 - 29.3 13.9 LLaVA-NeXT [37] [2024.01.30] 13B 36.2 19.8 - - -

Medium

VILA-1.5 [38] [2024.05.16] 13B 37.9 - 42.7 15.2 11.4 NVILA* [41] [2024.12.05] 15B 53.2 36.2 67.6 23.2 31.1 FRANK (Ours) 15B 61.3 49.4 55.4 37.2 32.3

LLaVA-NeXT [37] [2024.01.30] 34B 48.1 30.3 46.5 - VILA-1.5 [38] [2024.05.16] 40B 51.9 35.9 49.5 - -

LLaVA-OneVision [42] [2024.08.06] 72B 56.8 38.0 67.5 25.3 32.0 Qwen2-VL [43] [2024.09.18] 72B 64.5 49.2 70.5 26.6 36.0 GPT-4o [44] [2024.05.13] - 69.1 54.0 63.8 29.9 -

Large

InternVL2.5* [45] [2024.12.06] 38B 63.9 48.0 71.9 32.2 38.3 FRANK (Ours) 38B 69.2 56.8 73.1 39.7 47.0

- 3.2.3 Incorporating Modality Priors

While the closed-form fusion weights from Section 3.2.2 balance visual and reasoning shifts purely by their magnitudes, MLLM decoders exhibit a clear functional hierarchy: Shallow layers (small l) predominantly attend to visual tokens, anchoring the model in perceptual representations. Deep layers (large l) focus on textual tokens, supporting abstraction and symbolic reasoning.

To encode this prior knowledge, we introduce layer-dependent modality priors w(Vl),w(Rl) > 0 and reformulate the per-layer fusion objective:

wV(l) LTLD(Vl)(λV,λR) + w(Rl) LTLD(Rl)(λV,λR). (14) The modality prior-guided closed-form is

min

λV,λR

w(tl) ∥τ(bl)∥2 wV(l) ∥τ(Vl)∥2 + w(Rl) ∥τ(Rl)∥2

λt(l) =

, t ∈ {V,R}. (15)

Attention-Guided Decay Priors During our experiments, we observed that compared to deeper decoder layers, shallower decoder layers in the MLLM allocate more attention to visual tokens, thereby facilitating visual perceptual grounding, while deeper layers focus primarily on textual semantics, as illustrated in Figure 2. To introduce reasoning and reflection capabilities while preserving visual perception, and to maintain methodological simplicity without introducing additional supervision, we derive modality priors from the model’s layer-wise attention patterns and fit them with an exponential decay function (prioritizing simplicity, though more complex quadratic functions could alternatively be used for fitting). This approach naturally captures the non-uniform transition from visual grounding to

symbolic reasoning observed in practice. Specifically, we first collect the visual attention weights al from each decoder layer in the MLLM. Then, we posit al ≈ C e−αˆ l, and obtain α,ˆ logC via linear regression on (l,logal) l L=1. Finally, we set

exp(−αˆ l)

w(Vl) =

, w(Rl) = 1 − w(Vl). (16)

L j=1 exp(−αˆ j)

This attention-guided exponential schedule requires no labels and ensures the modality priors faithfully mirror the model’s intrinsic shift from visual grounding to reasoning across the decoder hierarchy. Please refer to Appendix A.9 for details.

To conclude, FRANK delivers a training-free, interpretable, and efficient framework for the per-layer fusion of visual grounding and logical reasoning in MLLMs, requiring only task-vector norms and simple priors. Its closed-form weights avoid any extra labeling data or optimization, making FRANK practical for scaling multimodal intelligence.

###### 4 Experiments

4.1 Datasets and Model Variants

We evaluate FRANK on five widely-used multimodal reasoning benchmarks: MMMU val [46], MMMU-Pro standard (10 opts) [47], MathVista testmini [48], MathVision testmini [49], and WeMath testmini [50]. These datasets encompass a broad spectrum of task formats (e.g., diagram interpretation, symbolic math problems) and difficulty levels.

To study the effects of model scale and architecture, we instantiate three FRANK variants by applying our layer-wise fusion to pairs of non-reasoning MLLMs and reasoning-specialized LLMs. Specifically, FRANK-8B merges Idefics3-8B [39]

- Table 2: Perceptual performance of FRANK-15B on MME compared to vision-only NVIL-15B. “w/o MP” denotes the ablation without Modality Prior. Sub-task abbreviations: “Comm.” = Commonsense Reasoning, “Num.” = Numerical Calculation, “Text.”

= Text Translation, “Code.” = Code Reasoning.

MME Existence Count Position Color OCR Poseter Celebrity

NVIL-15B (upper bound) 100.0 80.0 86.7 91.7 90.0 94.2 83.2 FRANK-15B w/o MP 96.7 78.3 68.3 90.0 75.0 91.2 76.2 FRANK-15B 96.7 80.0 60.0 91.7 85.0 91.8 77.7

MME Scene Landmark Artwork Comm. Num. Text. Code.

NVIL-15B (upper bound) 83.5 90.5 82.3 82.9 72.5 67.5 85.0 FRANK-15B w/o MP 80.0 83.7 76.5 84.3 75.0 57.5 82.5 FRANK-15B 83.0 84.3 74.3 85.0 72.5 60.0 85.0

(non-reasoning MLLM) with DeepSeekDistil-LLaMA3-8B [1] (reasoning LLM), FRANK-15B merges NVIL-15B [41] with DeepSeekDistil-Qwen2.5-14B [1], and FRANK-38B merges InternVL2.5-38B [45] with QwQ-32B [51]. These variants test our fusion across LLaMA- and Qwen-based architectures and varying capacities. These three FRANK variants enable analysis of our approach across small, medium, and large model scales. We apply FRANK’s closed-form fusion weights layer-by-layer without any additional fine-tuning. The attention-guided exponential decay prior is fit on 1000 randomly sampled validation examples from MSCOCO dataset [52] (note: only the image is needed, no annotations) using least-squares regression to obtain decay parameter α and normalization constant C.

- 4.2 Quantitative Evaluation

Table 1 presents the performance of FRANK variants and stateof-the-art baselines on five multimodal reasoning benchmarks. Below we analyze results by dataset, highlighting both absolute gains and relative improvements to demonstrate the efficacy and scaling behavior of our layer-wise fusion.

MMMU/MMMU-Pro. On MMMU (college-level image-text questions), FRANK-8B achieves 48.3, a 4.4-point gain over its vision branch. This improvement indicates that even at 8B parameters, our fusion effectively combines visual grounding and reasoning. On the more stringent MMMU-Pro, FRANK-8B reaches 34.7. Increasing model capacity yields further gains: FRANK-15B attains 61.3 on MMMU and 49.4 on MMMU-Pro, demonstrating that additional parameters enable richer reasoning adaptations. FRANK-38B further improves to 69.2 (MMMU) and 56.8 (MMMU-Pro), outperforming InternVL2.5-38B by

- 5.3 and 8.8 points, respectively, and confirming a strong scaling trend. Math Benchmarks (MathVista, MathVision, WeMath). We evaluate FRANK variants on three math-focused datasets (Table 1). FRANK-8B and FRANK-15B underperform on MathVista (50.7 and 55.4, -7.7 and -12.2 vs. Idefics3-LLaMA3 and NVIL-15B). In contrast, FRANK-38B achieves 73.1 on MathVista (+1.2 vs. InternVL2.5), 39.7 on MathVision (+7.5), and 47.0 on WeMath (+8.7), demonstrating that larger capacity better absorbs fusion weights, mitigating model merging interference, while still enhancing deeper-layer symbolic reasoning.

Scale and Architecture Analysis. Aggregating results across all three benchmarks reveals a clear scaling law: FRANK variants continue to reap sustained reasoning benefits and exhibit greater robustness as model size increases. Both Qwen-based

- Table 3: Ablation study results of different model mergeing methods on the MMMU.

Method MMMU Acc.

NVIL-15B (baseline) 53.2 VLM-Merging 53.6 Task Arithmetic 56.1 MetaGPT 57.9 FRANK-15B w/o Modality Prior 58.4 FRANK-15B 61.3

- Table 4: The number of reflection tokens in MMMU responses across models.

Reflection NVIL-15B FRANK-15B

Wait 0 20125 Hmm 0 1733 Mistake 0 1663 Alternatively 0 7712 Check 0 1586

FRANK-15B and LLaMA-based FRANK-8B showed consistent improvements in reasoning capabilities, indicating that our layer-wise fusion is independent of the backbone architecture. Moreover, FRANK-38B’s strong performance on the most challenging benchmarks underscores that larger models can more fully exploit the injected reasoning capabilities, all achieved without any additional gradient-based training.

4.3 Visual Perception Evaluation

We assess visual perception capability on the MME benchmark [53] (Table 2). NVIL-15B, being a non-reasoning multimodal model, achieves the strong perceptual performance across all 14 sub-tasks, representing the upper bound on visual grounding accuracy. As expected, fusing in decoder layers incurs some degradation; however, by incorporating our attention-guided Decay Prior, FRANK-15B only experiences very slight drops relative to NVIL-15B, substantially outperforming the no-prior ablation. By contrast, the ablated FRANK-15B w/o MP falls further behind, indicating that the Modality Prior is crucial for preserving visual perception when fusing the model. Moreover, in sub-tasks requiring commonsense reasoning (“Comm.”), FRANK-15B even slightly surpasses NVIL-15B (85.0 vs 82.9), suggesting that our fusion maintains and can enhance performance in scenarios where light reasoning complements visual

FRANK-8B

3519.7

3500

###### AverageOutputLength

FRANK-15B FRANK-38B

3241.1

3000

2500

2288.6

2019.6

2000

1634.5

1566.8

1500

1290.2

1216.0

1000

841.3

Easy Medium Hard

Difficulty Level

- Figure 4: Average output length of the FRANK on the MMMU benchmark, stratified by task difficulty.

understanding.

- 4.4 Ablation Study

- Table 3 evaluates each fusion component on MMMU using the same NVIL-15B and DeepSeekDistil-Qwen2.5-14B backbones without fine-tuning. The results confirm that model merging injects reasoning into a pretrained MLLM: Traditional fusion baselines (VLM-Merging [32], Task Arithmetic [14], and MetaGPT [29]) achieve 53.6%, 56.1%, and 57.9%, respec-

tively. Task Arithmetic follows a fixed fusion weight λV of 0.3, shown in prior work [14, 29] to perform robustly across diverse tasks, which nonetheless yields only modest gains (3.6-4.7 points), indicating limited integration of reasoning capabilities. Layer-only fusion (FRANK-15B w/o Modality Prior) leverages closed-form weights for per-layer merging and improves to 58.4%, outperforming all traditional methods by 1.7-3.6 points. Full FRANK-15B adds the attention-guided exponential decay modality prior and attains 61.3%, a further 2.9-point gain. This confirms that the learned prior successfully balances visual grounding and symbolic reasoning across layers.

- 4.5 Reflection Frequency and Output Length Analysis

We quantify FRANK’s self-correction capability by analyzing reflection tokens (Wait, Hmm, Mistake, Alternatively, Check) in MMMU val set responses (N=900). As Table 4 shows: 1) NVIL-15B (non-reasoning) produces zero reflection tokens. 2) FRANK-15B generates multiple reflection cycles per example This demonstrates that our fusion method intrinsically enables iterative self-correction during reasoning. Figure 4 reveals two scaling trends on MMMU: 1) Output length increases with task difficulty (Easy→Hard). 2) Larger models produce longer responses at each level. This demonstrates our method’s inference scaling capability, which automatically expands reasoning for harder problems while benefiting from increased model capacity.

- 4.6 Case Study: Exemplary Model Outputs

To illustrate the qualitative improvements brought by our FRANK fusion, we present three representative examples—one for each model size (8B, 15B, 38B)—comparing a nonreasoning baseline against the corresponding FRANK variant. In each case, the baseline model provides a direct (and incorrect) answer, whereas the FRANK model produces a detailed

chain-of-thought with reflection tokens, arriving at the correct solution.

As shown in Figures 5, 6, and 7, across all three model scales, the FRANK variants consistently produce step-by-step reasoning chains and explicit reflection checks. This enables them to correct intermediate arithmetic or logical reasoning and arrive at the correct results. In contrast, the non-reasoning baselines directly output an incorrect answer without any justification.

###### 5 Conclusions

In this paper, we presented FRANK, a training-free and R1like MLLM that endows off-the-shelf MLLMs with advanced reasoning and self-reflection capabilities. By decomposing the decoder into per-layer task vectors and deriving closed-form fusion weights under NTK linearization and task-vector orthogonality, FRANK seamlessly integrates visual grounding from vision-fine-tuned MLLMs with logical reasoning from reasoning-specialized LLMs, guided by attention-driven modality priors. Extensive experiments across five multimodal reasoning benchmarks demonstrate that FRANK consistently outperforms state-of-the-art baselines at 8B, 15B, and 38B scales, with particularly strong gains on reasoning tasks.

Future work includes extending FRANK to support more diverse modalities (e.g., audio, video), exploring dynamic fusion strategies for real-time tasks, and investigating theoretical guarantees under broader neural architectures. We believe FRANK offers a practical and interpretable path toward scalable multimodal intelligence without the overhead of task-specific retraining.

###### References

- [1] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [2] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi K1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [3] Ruotian Ma, Peisong Wang, Cheng Liu, Xingyan Liu, Jiaqi Chen, Bang Zhang, Xin Zhou, Nan Du, and Jia Li. S 2 R: Teaching llms to self-verify and self-correct via reinforcement learning. arXiv preprint arXiv:2502.12853, 2025.
- [4] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In ICLR, 2024.
- [5] OpenAI. OpenAI o1. https://openai.com/o1/, 2024.
- [6] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. LLaVA-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024.

|[Figure 6]<br><br>Question: Janet Ludlow's firm requires all its analysts to use a two-stage dividend discount model (DDM) and the capital asset pricing<br><br>model (CAPM) to value stocks. Using the CAPM and DDM, Ludlow has valued QuickBrush Company at $63 per share. She now must value<br><br>SmileWhite Corporation. Calculate the required rate of return for SmileWhite by using the information in the following table:. Options: (A) 14% (B) 15% (C) 16% Ground Truth: C<br><br>Idecifics3-8B: Answer: B<br><br>FRANK-8B (Ours): <think> Okay, so I need to figure out the required rate of return for SmileWhite Corporation using the information given in the table. Let's break this down step by step.\n\nFirst, I see there's a table comparing QuickBrush and SmileWhite with their Beta, Market price, Intrinsic value, and some notes about risk-freerate and expected market return. The question is asking for the requiredrate of return for SmileWhite.\n\nI remember that the required rate of return can be calculated using the Capital Asset Pricing Model (CAPM), which is:\n\nRequired Rate of Return = Risk-Free Rate + (Beta * Expected Market Return)\n\nLooking at the table, I see that the risk-free rate is 4.50%, and the expected market return is 14.50%. But wait, which Beta should I use? The table shows Beta for both companies, but it's the same for both, 1.35 for QuickBrush and 1.15 for SmileWhite. Hmm, so I think I need to use SmileWhite's Beta for their required rate of return.\n\nSo plugging in the numbers:\n\nRequired Rate of Return = 4.50% + (1.15 * 14.50%)\n\nLet me calculate that. First, 1.15 times 14.50% is... 1.15 * 0.145 = 0.1665, or 16.65%. Wait, the Beta is given for each company.\n\nWait, maybe I need to calculate the required rate of return for SmileWhite as:\n\nRequired Rate = Risk-Free Rate + (Beta * (Market Return - Risk-Free Rate))\n\nBut that's the same as CAPM. So it should be 4.50% + (1.15 * (14.50% - 4.50%)) = 4.50% + (1.15 * 10%) = 4.50% + 11.5% = 16%.\n\nAh, that's one of the options. So maybe I did a miscalculation earlier.\n\nWait, let me recalculate that:\n\n1.15 * (14.50% - 4.50%) = 1.15 * 10% = 11.5%\n\nThen add the risk-free rate: 4.50% + 11.5% = 16%. So the requiredrate of return is 16%.\n\nThat's option C, which is 16%. So the answer should be C.\n</think> \n\nThe required rate of return for SmileWhite is calculated using the CAPM formula, considering the given Beta and market return. \n\n**Step-bystep explanation:**\n\n1. **Identify the given values:**\n - Risk-free rate (Rf) = 4.50%\n - Expected market return (Rm) = 14.50%\n - Beta (SmileWhite) = 1.15\n\n2. **Apply the CAPM formula:**\n \\[\n R_e = Rf + \\beta \\times (Rm - Rf)\n \\]\n \\[\n R_e = 4.50\\% + 1.15 \\times (14.50\\% - 4.50\\%)\n \\]\n \\[\n R_e = 4.50\\% + 1.15 \\times 10\\% = 4.50\\% + 11.50\\% = 16\\%\n \\]\n\n**Answer:** (C) 16%<br><br>[Figure 7]<br><br>……|
|---|

- Figure 5: Output examples from FRANK-8B and the non-reasoning baseline model Idecifics3-8B. Here, <think> and </think> denote R1-like reasoning processes, while blue text indicates reflection tokens.

|[Figure 8]<br><br>The jet engine in Fig. P3.50 admits air at 20\u00b0C and 1 atm at (1), where A_1 = 0.5 m^2 and V_1 = 250 m/s. The fuel-air ratio is 1:30. The air leaves section (2) at 1 atm, V_2 = 900 m/s, and A_2 = 0.4 m^2. Compute the test stand support reaction R_x needed. Options: (A) 99,000 N (B) 102,000 N (C) 121,000 N Ground Truth: B<br><br>NVIL-15B: (C)<br><br>FRANK-15B (Ours): <think> Okay, so I have this problem about a jet engine, and I need to find the test stand support reaction R_x. Hmm, let me try to visualize the setup based on the description. There's a combustion chamber with air entering at point 1 and exiting at point 2. The engine is on a test stand, so it's probably stationary, and we need to calculate the force it exerts on the stand.\n\nFirst , I remember that in fluid dynamics, when dealing with forces on a control volume, we can use the momentum equation. The general form is \u03a3F = m_dot*(V2 - V1) + (pA)(V2 - V1) terms, but since the areas are different,<br><br>maybe it's better to use the integral form of the momentum equation.\n\nWait, the problem gives me the areas A1 and A2, and the velocities V1 and V2.<br><br>Also, the fuel-air ratio is 1:30, which I think means that for every 30 units of air, there's 1 unit of fuel. But since fuel is burned in the combustion chamber, the mass flow rate of the air will increase because fuel is added. Hmm, but I'm not sure if I need to consider the fuel mass flow here. Let me think.\n\nThe support reaction R_x is the force that the engine exerts on the test stand. By Newton's third law, the reaction force would be equal and opposite to the thrust force produced by the engine. So, if I can find the thrust, that would give me R_x.\n\nThrust in a jet engine is typically calculated as the mass flow rate of the air multiplied by the velocity of the exhaust gases minus the incoming air velocity. But wait, in this case, the engine is on a test stand, so maybe the thrust is just the difference in momentum flow rates at the inlet and outlet.\n\nLet me write down the given data:\n\nAt inlet<br><br>(1):\n- Temperature T1 = 20\u00b0C, which is 293.15 K.\n- Pressure P1 = 1 atm.\n- Area A1 = 0.5 m\u00b2.\n- Velocity V1 = 250 m/s.\n\nAt<br><br>outlet (2):\n- Pressure P2 = 1 atm.\n- Velocity V2 = 900 m/s.\n- Area A2 = 0.4 m\u00b2.\n\nFuel-air ratio is 1:30, so for every 30 kg of air, 1 kg of fuel is burned. So, the total mass flow rate at the exit would be the mass flow rate of air plus the mass flow rate of fuel.\n\nBut wait, the mass flow rate of air at the inlet is m_dot_air = rho1 * A1 * V1. Then, the mass flow rate of fuel is m_dot_fuel = m_dot_air / 30. So, the total mass flow rate at the exit is m_dot_total = m_dot_air + m_dot_fuel = m_dot_air * (1 + 1/30) = m_dot_air * (31/30).\n\ Wait, but the pressure at both inlet and outlet is 1 atm, so the pressure forces would be p*A1*V1 and p*A2*V2, but I'm not sure if they contribute to the net force. Alternatively, maybe I should consider the momentum equation as:\n\nR_x = m_dot2 * V2 - m_dot1 * V1.\n\n But wait, that's what I did earlier, and I got approximately<br><br>102,234 N, which is close to option B, 102,000 N.\n\nAlternatively, maybe I should consider the specific heat or the energy balance, but I think the<br><br>momentum equation approach is sufficient here.\n\nWait, another thought: The mass flow rate at the exit includes both the incoming air and the fuel. So, m_dot2 = m_dot_air + m_dot_fuel = m_dot_air + (m_dot_air / 30) = (31/30)*m_dot_air.\n\nWhich is what I did earlier. So, I think my calculation is correct.\n\nSo, the support reaction R_x is equal to the thrust, which is approximately 102,234 N, so the closest option is B) 102,000 N.\n</think> \n\nTo determine the test stand support reaction \\( R_x \\), we apply the momentum equation. The thrust \\( T \\) is calculated as the difference in momentum flow rates at the inlet and outlet:\n\n\\[\nT = \\dot{m}_2 V_2 - \\dot{m}_1 V_1\n\\]\n\nGiven the fuel-air ratio of 1:30, the total<br><br>mass flow rate at the exit is \\( \\dot{m}_2 = \\frac{31}{30} \\dot{m}_1 \\). Using the ideal gas law, the density \\( \\rho_1 \\) at the inlet is<br><br>calculated, and the mass flow rates \\( \\dot{m}_1 \\) and \\( \\dot{m}_2 \\) are determined. Substituting these into the thrust equation yields \\( T \\approx 102,234 \\, \\text{N} \\), which rounds to the closest option:\n\n**Answer:(B) 102,000 N**<br><br>[Figure 9]<br><br>……|
|---|

Figure 6: Output examples from FRANK-15B and the non-reasoning baseline model NVIL-15B.

|[Figure 10]<br><br>Which reaction in the following reaction sequence <image 1> is regioselective?<br><br>Ground Truth: B<br><br>InternVL2.5-38B:Step A<br><br>FRANK-38B (Ours): <think> Okay, let me try to work through this problem step by step. The question is asking which reaction in the given reaction sequence is regioselective. The image shows a series of four steps labeled A, B, C, D. I need to figure out which of these steps involves regioselectivity.\n\nFirst, I should recall what regioselective means. reaction is one where the major product forms in a specific region of the molecule, favoring one structural form over another possible isomer. This is different from stereoselectivity, which is about the spatial arrangement.\n\nLooking at the reaction steps in the image:\n\nStep A: The reactants start with a compound that has an OMs (maybe mesylate?) group. The reagent used here is<br><br>NaHCO3. Wait, the image might be a bit hard to interpret, but the reaction might be an acid-base reaction or maybe deprotonation. Alternatively, if it's<br><br>a substitution reaction, like maybe an esterification? But with NaHCO3? Hmm. Let me sketch the structure based on the details given.\n\nStep A shows a molecule with an OMS group attached to a benzene ring. When treated with NaHCO3 (which is a weak base), perhaps it results in deprotonation somewhere. But the product after Step A has an ester group (COOEt) attached. Maybe in Step A, there's a substitution where OMs leaves, and EtCOcomes in? Wait, but sodium bicarbonate might be used in some way to deprotonate and form a carboxylate.\n\n Alternatively, perhaps Step A is an saponification or some ester hydrolysis? The OMS group might be a leaving group, and the base (NaHCO3) could facilitate substitution of that leaving group with something else. But I'm a bit confused here.\n\nWait, maybe the first step is a Swern oxidation? No, the reagents are NaHCO3. Wait,<br><br>looking at the structure after Step A, maybe there's an ester group on the same benzene ring as where OMs was. Did the OMS get replaced by an ester<br><br>group? If so, perhaps it's nucleophilic substitution at the benzylic position.\n \n\n Alternatively, maybe the Staudinger reaction in Step C produces an amine in a certain position, but that's part of the reduction step. Since reduction of azide to amine doesn't alter the position, it's not regioselective there.\n\nTherefore, I think the regioselective step is Step B. Because in aromatic substitution with sodium azide, the position where azide adds depends on the directing groups present on the molecule. So if there are activating groups like OH or other substituents directing the azide to a specific position (ortho or para to those groups), then Step B is regioselective.\n</think> \n\nThe reaction that is regioselective is **Step B**. Here's the analysis:\n\n### Key Observations:\n1. **Step A**: The displacement of the mesylate<br><br>(OMs) by an ethyl ester (COEt) occurs at the same position where the leaving group (OMs) was located. Since the nucleophile (EtCO\u207b) attacks the<br><br>carbon bearing the leaving group, this step is not regioselective as it does not depend on existing directing groups.\n\n2. **Step B**: This involves introducing an azide group (N\u2083\u207b) via nucleophilic aromatic substitution. The position where azide attacks will be directed by activating or deactivating groups already present on the benzene ring. For example, if the benzene ring has an electron-donating group like a hydroxyl (\u2013OH), it will direct the azide to the **ortho** or **para** positions relative to \u2013OH. This directed substitution introduces regioselectivity.\n\n3. **Step C**: Reduction of azide to amine (via PPh\u2083 / NEt\u2083) does not alter regiochemistry\u2014the position of the new amine group is fixed from the prior substitution (Step B).\n\n4. **Step D**: This step modifies an amine group (N) with methanesulfonyl chloride, producing an N-sulfonyl derivative.<br><br>The substitution occurs at the nitrogen (not the aromatic ring), so this is not regioselective in an aromatic sense.\n\n### Conclusion:\n**Step B** is the<br><br>only step where the regiochemistry is influenced by existing substituents (e.g., activating groups like \u2013OH) on the aromatic ring, making it regioselective.\n\n Answer: **Step B** is regioselective.<br><br>……<br><br>[Figure 11]|
|---|

###### Figure 7: Output examples from FRANK-38B and the non-reasoning baseline model InternVL2.5-38B.

- [7] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-V: Exploring long-chain visual reasoning with multimodal large language models. In CVPR, 2025.
- [8] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319, 2024.
- [9] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. VisualRFT: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [10] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, and Shaohui Lin. VisionR1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [11] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. LMM-R1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.
- [12] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-VL: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025.
- [13] Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In ICML, volume 162, pages 23965–23998, 2022.
- [14] Gabriel Ilharco, Marco Túlio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In ICLR, 2023.
- [15] Jinghan Zhang, Shiqi Chen, Junteng Liu, and Junxian He. Composing parameter-efficient modules with arithmetic operation. In NeurIPS, New Orleans, LA, USA, 2023.
- [16] Scott L Brincat, Markus Siegel, Constantin von Nicolai, and Earl K Miller. Gradual progression from sensory to task-related processing in cerebral cortex. Proceedings of the National Academy of Sciences, 115(30):E7202–E7211, 2018.
- [17] Haruka Kawasaki, Satoshi Nishida, and Ichiro Kobayashi. Hierarchical processing of visual and language information in the brain. In AACL-IJCNLP, pages 405–410, 2022.
- [18] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, and Jifeng Dai. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. In NeurIPS, 2023.
- [19] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

- [20] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, New Orleans, LA, USA, 2023.
- [21] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In ICLR, Vienna, Austria, 2024.
- [22] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, New Orleans, LA, USA, 2022.
- [23] Ruohong Zhang, Bowen Zhang, Yanghao Li, Haotian Zhang, Zhiqing Sun, Zhe Gan, Yinfei Yang, Ruoming Pang, and Yiming Yang. Improve vision language model chain-of-thought reasoning. arXiv preprint arXiv:2410.16198, 2024.
- [24] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of-thought prompting for large multimodal models. In CVPR, pages 14420– 14431, Seattle, WA, USA, 2024.
- [25] Bozhi Luan, Hao Feng, Hong Chen, Yonghui Wang, Wengang Zhou, and Houqiang Li. TextCoT: Zoom in for enhanced multimodal text-rich image understanding. arXiv preprint arXiv:2404.09797, 2024.
- [26] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. MM-Eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.
- [27] Michael Matena and Colin Raffel. Merging models with fisher-weighted averaging. In NeurIPS, New Orleans, LA, USA, 2022.
- [28] Prateek Yadav, Derek Tam, Leshem Choshen, Colin A. Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. In NeurIPS, New Orleans, LA, USA, 2023.
- [29] Yuyan Zhou, Liang Song, Bingning Wang, and Weipeng Chen. Metagpt: Merging large language models using model exclusive task arithmetic. In EMNLP, pages 1711– 1724, Miami, FL, USA, 2024.
- [30] Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. Dataless knowledge fusion by merging weights of language models. In ICLR, Kigali, Rwanda, 2023.
- [31] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In ICML, Vienna, Austria, 2024.
- [32] Shiqi Chen, Jinghan Zhang, Tongyao Zhu, Wei Liu, Siyang Gao, Miao Xiong, Manling Li, and Junxian He. Bring reason to vision: Understanding perception and reasoning through model merging. arXiv preprint arXiv:2505.05464, 2025.
- [33] Arthur Jacot, Clément Hongler, and Franck Gabriel. Neural Tangent Kernel: Convergence and generalization in

- neural networks. In NeurIPS, pages 8580–8589, Montreal, Canada, 2018.
- [34] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [35] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. AGIEval: A human-centric benchmark for evaluating foundation models. In NAACL, pages 2299– 2314, Mexico City, Mexico, 2024.
- [36] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, pages 26286–26296, Seattle, WA, USA, 2024.
- [37] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext. https://llava-vl.github.io/blog/2024-01-30-llava-next/, 2024.
- [38] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: on pre-training for visual language models. In CVPR, pages 26679–26689, Seattle, WA, USA, 2024.
- [39] Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding visionlanguage models: insights and future directions. In Workshop on RBFM, 2024.
- [40] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In ECCV, volume 15075, pages 370–387, Milan, Italy, 2024.
- [41] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. arXiv preprint arXiv:2412.04468, 2024.
- [42] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual task transfer. TMLR, 2024.
- [43] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [44] OpenAI. GPT-4o. https://openai.com/index/hello-gpt-4o/, 2024.
- [45] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [46] Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and

- Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In CVPR, pages 9556–9567, Seattle, WA, USA, 2024.
- [47] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. MMMU-Pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.
- [48] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, Vienna, Austria, 2024.
- [49] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with mathvision dataset. In NeurIPS, Vancouver, BC, Canada, 2024.
- [50] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.
- [51] Qwen. Qwq. https://qwenlm.github.io/blog/qwq-32bpreview/, 2024.
- [52] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In ECCV, volume 8693, pages 740–755, Zurich, Switzerland, 2014.
- [53] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. A survey of multimodel large language models. In CAICE, pages 405–409, 2024.

###### A Appendix

- A.1 Implementation Details

As shown in Table 5, we summarize the precise fusion configurations used in our FRANK variant. For each model size, we list the vision-finetuned model, the reasoning-finetuned model, and the underlying base model that we linearly combine via our layer-wise Taylor-derived weights. All our experiments were conducted on NVIDIA GPUs.

In each case, the “Non-Reasoning MLLM” provides the task vector encoding of the vision-adapted decoder updates, the “Reasoning LLM” supplies the task vector for pure language reasoning, and the “Base Model” is the pre-trained backbone into which these vectors are fused according to our attention-guided exponential decay schedule.

- A.2 Ablation of the Multimodal Prior

To validate the contribution of our attention-driven multimodal prior (MP), we perform an ablation study on FRANK-15B across five standard benchmarks. Table 6 compares the full FRANK15B against a variant with the MP component removed (w/o MP). On MMMU, accuracy drops from 61.3% to 58.4% when MP is disabled, a 2.9-point decrease that underscores the prior’s role in balancing visual and textual signals. The impact is even more pronounced on MathVision, where performance falls from

- 37.2% to 28.7%, reflecting a 8.5-point degradation in complex visual-mathematical reasoning. We observe consistent drops on the remaining benchmarks (MMMU-Pro, MathVista, and WeMath), indicating that MP systematically guides layer-wise fusion towards more effective cross-modal integration. These results confirm that our multimodal prior is essential for robust reasoning: by weighting visual and language contributions according to per-layer attention, it ensures that FRANK-15B can harness both modalities optimally across diverse tasks.

A.3 Ablation of the Attention-Guided Exponential Decay Prior

To further assess the effectiveness of our attention-guided exponential decay strategy, we compare against two manually chosen exponential decay factors (0.3 and 0.5) and a simple layer-indexed linear decay. As shown in Table 7, both fixed-rate exponentials (59.1% and 56.3%) and the linear schedule (59.6%) improve over the NVIL-15B baseline (53.2%), but remain significantly below our full FRANK-15B (61.3%). This clear margin demonstrates that leveraging per-layer visual attention to drive the decay adapts the modality weighting more precisely than uniform schedules and achieves the best trade-off between visual perception and language reasoning.

A.4 Validation of Layer-Wise Multimodal Prior and Task-Vector Orthogonality

To verify the two key assumptions underpinning our fusion strategy—namely that (1) shallow decoder layers focus predominantly on visual perception while deeper layers prioritize language reasoning (the multimodal prior), and (2) task vectors from vision-finetuned and reasoning-finetuned models are mutually orthogonal—we analyze both Idefics3-8B and InternVL2.5-

- 38B.

Layer-Wise Attention Patterns Figures 8 and 10 plot the average per-layer attention weights that each model’s decoder assigns to visual tokens when processing vision inputs. In both the 8B and 38B variants, we observe a clear decay: the first few layers exhibit high visual attention, and then steadily decline toward deeper layers. This consistent decay profile confirms our multimodal prior hypothesis across both scales, justifying the use of an exponential decay schedule guided by these attention statistics.

Task-Vector Cosine Similarity Figures 9 and 11 report the cosine similarity between the task vectors extracted from vision-finetuned models (Idefics3-8B / InternVL2.5-38B) and those from reasoning-finetuned counterparts (DeepSeekDistilLLaMA3-8B / QwQ-32B) at each decoder block. In both cases, the similarity values remain close to zero across almost all layers, confirming that the vision and reasoning task vectors occupy nearly orthogonal subspaces. This orthogonality underpins our linear fusion derivation via Taylor expansion, ensuring that the combined update remains a meaningful superposition of the two modalities without destructive interference.

- A.5 Proof of NTK Linearization (Property)

Under the Neural Tangent Kernel (NTK) regime for wide transformers, the network behavior near initialization θ0 admits:

f(x;θ0 + ∆θ) = f(x;θ0) + ∇θ f(x;θ0)⊤∆θ + O(∥∆θ∥2/

√

width).

(17)

For task-specific loss L(tl), the Hessian exhibits two key properties: Jacobian dominance: Residual terms vanish due to

∇2 f(l) ∼ O(1/

√width). Isotropy: Gradient directions become nearly orthogonal in high dimensions. This leads to the layerwise Hessian approximation:

∇2L(tl)(θ) ≈ Ex ∇f(l)∇f(l)⊤ ≈ δ(tl)Id

l

, (18) where the curvature scalar:

δ(tl) =

1 dl

Tr ∇2L(tl)(θ0(l)) =

1 dl

Ex ∥∇f(l)(x;θ0(l))∥2 , (19) captures the average gradient magnitude at initialization.

- A.6 Proof of Task-Vector Orthogonality (Property)

Figure 2 shows that, vision-fine-tuning and reasoning-finetuning update disjoint aspects of the LLM’s representations. Concretely, let

τ(Vl) = θV(l) − θ0(l), τ(Rl) = θR(l) − θ0(l). (20) We observe that

(τ(Vl))⊤τ(Rl) = O(ε), (21) where ε is a small constant, indicating near-orthogonality.

- A.7 Proof of Taylor Expansion of Layer Task Loss Difference (Lemma)

To quantify the impact of parameter fusion on task performance, we analyze the Layer-wise Task Loss Difference (LTLD) through a rigorous Taylor expansion approach. The fused parameters combine both task updates through convex combination:

###### θ(fl) = θ0(l) +

λ(tl)τ(tl), where λ(tl) ∈ [0,1], (22)

t∈{V,R}

Table 5: Detailed fusion components for FRANK configurations

FRANK Variant Non-Reasoning MLLM Reasoning LLM Base Model

FRANK-8B Idefics3-8B DeepSeekDistil-LLaMA3-8B LLaMA3.1-8B FRANK-15B NVIL-15B DeepSeekDistil-Qwen2.5-14B Qwen2.5-14B FRANK-38B InternVL2.5-38B QwQ-32B Qwen2.5-32B

- Table 6: Ablation study results of FRANK-15B across five multimodal reasoning benchmarks: MMMU val, MMMU-Pro standard (10 opts), MathVista testmini, MathVision testmini, and WeMath testmini. * indicates the baseline model.

Methods MMMU MMMU-Pro MathVista MathVision WeMath

NVILA* 53.2 36.2 67.6 23.2 31.1 FRANK-15B w/o MP 58.4 41.8 52.5 28.7 31.4 FRANK-15B 61.3 49.4 55.4 37.2 32.3

- Table 7: Ablation study results of attention-guided exponential decay prior on the MMMU.

A.8 Derivation of Closed-Form Fusion Weights

To enable efficient model merging while preserving task performance, we derive theoretically-grounded fusion weights through layer-wise NTK and Task-Vector Orthogonality analysis. The key insight is that LLMs exhibit approximately quadratic loss landscapes under NTK, permitting closed-form solutions. Under the NTK linearization regime for LLMs, we first analyze the

###### Method MMMU Acc.

NVIL-15B (baseline) 53.2 0.3 59.1 0.5 56.3 linear 59.6 FRANK-15B 61.3

layer-wise behavior. For any layer l with initialization θ0(l) and perturbation τ(l) = θ(l) − θ0(l), the parameter admits the first-order approximation:

f(l)(x;θ(l)) ≈ f(l)(x;θ0(l)) + ∇θ(l) f(l)(x;θ0(l))⊤τ(l)

+O(∥τ(l)∥2) (27)

where, θ0(l) denotes the pre-trained initialization at layer l, θt(l) = θ0(l) + τt(l) denotes the fine-tuned parameters for task t ∈ {V,R}, τ(tl) denotes the task-vector update from θ0(l) to θt(l).

Linear term

This linearity emerges in wide transformers where the network’s output changes linearly with parameter perturbations.

Fusion Residual Vector: Measures deviation from optimal task parameters:

For the quadratic loss L(tl)(θ(l), xt) = 12∥f(l)(xt;θ(l)) − yt∥2, we compute the Hessian:

ht(l) = θ(fl) − θt(l) =

λ(kl)τ(kl) − (1 − λ(tl))τ(tl). (23)

∇2L(tl) = ∇f(l)(xt;θ(l))∇f(l)(xt;θ(l))⊤

k t∈{V,R}

+ (f(l)(xt;θ(l)) − yt)⊤∇2 f(l)(xt;θ(l))

(28)

Interpolation Path: Defines a linear trajectory in parameter space from the fine-tuned parameters θt(l) to the fused parameters θ(fl), enabling exact Taylor expansion along the fusion direction:

Vanishes under NTK regime

Under the NTK regime, the second term becomes negligible due to two intrinsic properties of wide neural networks: 1) the output residual ∥f(l) −yt∥ vanishes with near-optimal fine-tuning (guaranteed by NTK’s convex-like optimization landscape), and 2) the Hessian ∇2 f(l) shrinks as O(1/

γt(l)(β) = θt(l) + βh(tl), β ∈ [0,1]. (24) Taylor Expansion Analysis: Applying second-order expansion along γt(l):

√width) (a direct conse-

quence of NTK’s linearization effect). Under NTK conditions:

L(tl)(θ(fl), xt) = L(tl)(θt(l) + h(tl), xt)

= L(tl)(θt(l), xt) + ∇L(tl)(θt(l), xt)⊤h(tl)

∇2L(tl)(θ(l)) ≈ ∇f(l)(xt;θ0(l))∇f(l)(xt;θ0(l))⊤ (29) Taking the isotropic approximation for layers:

(25)

1

- 1

- 2

∇2L(tl)(γt(l)(β))dβ h(tl)

h(tl)⊤

+

1 dl

0

∇2L(tl) ≈ δ(tl)Id

, δ(tl) =

tr ∇f(l)(xt;θ0(l))∇f(l)(xt;θ0(l))⊤

Under fine-tuning convergence ∇L(tl)(θt(l), xt) ≈ 0, we obtain: LTLD(tl) = L(tl)(θ(fl), xt) − L(tl)(θt(l), xt)

l

(30)

where dl is the parameter dimension at layer l. This follows from the observation that gradient directions become nearly orthogonal in high dimensions.

(26)

1

- 1

- 2

h(tl)⊤

∇2L(tl)(γt(l)(β))dβ h(tl).

=

0

Substituting the Hessian approximation into the Taylor remainder:

This derivation establishes the theoretical foundation for our layer-wise fusion analysis, connecting parameter perturbations to task performance through differentiable geometry.

1

∇2L(tl)(γt(l)(β))dβ ≈ δ(tl)I, (31) 13

0

0.8

0.7

Visionattentionweights

0.6

0.5

0.4

0.3

0.2

0.1

0.0

0 5 10 15 20 25 30

Number of layer

Figure 8: Layer-wise visual attention of Idefics3-8B.

0.8

CosineSimilarity

0.6

0.4

0.2

0.0

0 4 8 12 16 20 24 28

Decoder Block Index

Figure 9: Cosine similarity between task vectors of vision-finetuned (Idefics3-8B) and reasoning-finetuned (DeepSeekDistil-LLaMA3-8B) models at each decoder block.

0.7

Visionattentionweights

0.6

0.5

0.4

0.3

0.2

0.1

0.0

0 5 10 15 20 25 30 35 40 45 50 55 60 65

Number of layer

- Figure 10: Layer-wise visual attention of InternVL2.5-38B.

0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60

Decoder Block Index

0.0

0.2

0.4

0.6

0.8

CosineSimilarity

- Figure 11: Cosine similarity between task vectors of visionfinetuned (InternVL2.5-38B) and reasoning-finetuned (QwQ32B) models at each decoder block.

collapses the quadratic form to a scalar multiple of ∥h(tl)∥2, yielding a tractable bound and closed-form solution for fusion

weights. The second-order Taylor remainder is bounded by the extremal eigenvalues of the integral Hessian. Substituting our isotropic approximation:

2LTLD(tl) ≤ h(tl)⊤ δ(tl)Id

h(tl)

l

= δ(tl)∥h(tl)∥2

2

λ(kl)τ(kl) − (1 − λ(tl))τ(tl)

###### = δ(tl)

(32)

k t∈{V,R}

where we used the identity ∥a + b∥2 = ∥a∥2 + ∥b∥2 + 2a⊤b. The cross-term vanishes due to task vector orthogonality. This demonstrates that the layer-wise task loss difference is governed

by the product of the initialization curvature δ(tl) and the squared fusion residual norm ∥h(tl)∥2, justifying our layer-wise analysis.

Next, we show the optimal layer-wise fusion weights. The layerwise analysis enables independent optimization at each layer. The LALD decomposes into layer-specific terms:

LALD(l) ≤

 (1 − λ(tl))2∥τ(tl)∥2 +

 

δ(tl)

(33)

2 ∥τt(l)∥2

(λ(kl))2∥τ(kl)∥2

k t∈{V,R}

t∈{V,R}

This additive structure permits layer-wise optimization by solving L (where L is the total number of layers) independent problems:

{λ(tl)}t∈{V,R} = argmin J(l), ∀l = 1,..., L (34)

Focusing on a single layer (omitting superscript (l) for clarity), we reformulate the objective:

###### δt 2 ∥τt∥2 (1 − λt)2∥τt∥2 +

λ2k∥τk∥2

###### J =

t∈{V,R}

k t∈{V,R}

- 1

- 2 t∈{V,R} k t∈{V,R}

- 1

- 2 t∈{V,R}

δt∥τt∥4(1 − λt)2 +

δtλ2k∥τt∥2∥τk∥2

=

(35)

Under task orthogonality τ⊤VτR = 0, the cross-derivative terms vanish, simplifying the gradient to:

∂J ∂λt

δk∥τk∥2

= −δt∥τt∥4(1 − λt) + λt∥τt∥2

k t∈{V,R}

0 = −δt∥τt∥4 + λt 

 

(36)

δk∥τk∥2

δt∥τt∥4 + ∥τt∥2

k t∈{V,R}

Solving the linear system yields:

δt∥τt∥4 δt∥τt∥4 + ∥τt∥2 k t δk∥τk∥2

δt∥τt∥2

(37)

λt =

=

k∈{V,R} δk∥τk∥2

Under the uniform curvature assumption δk ≈ δ0 (arising from NTK’s layer-wise gradient statistics in wide networks, where

1 dl Tr(∇f(l)∇f(l)⊤) becomes task-invariant as dl → ∞), the solution simplifies to:

λ(tl) = ∥τ(tl)∥2

⇒ λ(tl) ∝ ∥τ(tl)∥2 (38)

k∈{V,R} ∥τ(kl)∥2

This closed-form solution adaptively suppresses interfering task vectors while preserving target task information at each layer.

A.9 Derivation of Attention-Guided Exponential Decay Priors In Section 3.2.3 we introduce an attention-guided exponential schedule for modality priors. Here, we provide a step-by-step derivation.

Modeling the decay. We observe the layer-wise visualattention ratio

1 N x,h

Attnl→vis(x,h) Attnl→vis(x,h) + Attnl→text(x,h) ∀l = 1,..., L,

al =

(39) where x indexes inputs, h indexes attention heads, and N is a normalization factor. We posit an exponential decay model:

al ≈ C e−αl, (40) where C > 0 and α > 0 are unknown constants. Logarithmic linearization. Taking natural logarithm on both sides of Eq. ( 40), we obtain

lnal ≈ lnC − αl. (41)

Define yl = lnal, xl = l, b0 = lnC, b1 = −α. Then Eq. ( 41) becomes a linear regression problem:

yl ≈ b0 + b1 xl. (42)

Least-squares estimation. We collect the dataset {(xl,yl)}lL=1 and solve the normal equations for the least-squares fit:

L l=1(xl − x¯)(yl − y¯)

b1 =

, b0 = y¯ − b1 x¯,

L l=1(xl − x¯)2

(43)

where x¯ = L1 l xl and y¯ = L1 l yl. We then recover

αˆ = −b1, Cˆ = eb

. (44)

0

Constructing modality priors. Finally, we define the normalized exponential priors

e−αˆ l

wV(l) =

, w(Rl) = 1 − w(Vl). (45)

L j=1 e−αˆ j

These priors smoothly interpolate from strong visual emphasis in early layers to strong reasoning emphasis in late layers, and can be replaced by alternative data-driven schedules if desired.

