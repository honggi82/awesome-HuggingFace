# arXiv:2602.01541v1[cs.CV]2Feb2026

## Toward Cognitive Supersensing in Multimodal Large Language Model

Boyi Li1,∗, Yifan Shen1,∗,§, Yuanzhe Liu1,∗, Yifan Xu1, Jiateng Liu1, Xinzhuo Li1, Zhengyuan Li1, Jingyuan Zhu2, Yunhan Zhong1, Fangzhou Lan2, Jianguo Cao2, James M. Rehg1, Heng Ji1, Ismini Lourentzou1,†, Xu Cao1,2,†

1University of Illinois Urbana-Champaign, 2PediaMed AI ∗Equal contribution, §Project lead, †Corresponding author

Multimodal Large Language Models (MLLMs) have achieved remarkable success in open-vocabulary perceptual tasks, yet their ability to solve complex cognitive problems remains limited, especially when visual details are abstract and require visual memory. Current approaches primarily scale Chain-of-Thought (CoT) reasoning in the text space, even when language alone is insufficient for clear and structured reasoning, and largely neglect visual reasoning mechanisms analogous to the human visuospatial sketchpad and visual imagery. To mitigate this deficiency, we introduce Cognitive Supersensing, a novel training paradigm that endows MLLMs with human-like visual imagery capabilities by integrating a Latent Visual Imagery Prediction (LVIP) head that jointly learns sequences of visual cognitive latent embeddings and aligns them with the answer, thereby forming vision-based internal reasoning chains. We further introduce a reinforcement learning stage that optimizes text reasoning paths based on this grounded visual latent. To evaluate the cognitive capabilities of MLLMs, we present CogSense-Bench, a comprehensive visual question answering (VQA) benchmark assessing five cognitive dimensions. Extensive experiments demonstrate that MLLMs trained with Cognitive Supersensing significantly outperform state-of-the-art baselines on CogSense-Bench and exhibit superior generalization on out-of-domain mathematics and science VQA benchmarks, suggesting that internal visual imagery is potentially key to bridging the gap between perceptual recognition and cognitive understanding. We will open-source the CogSense-Bench and our model weights.

[Figure 1]

GitHub: https://github.com/PediaMedAI/Cognition-MLLM HuggingFace: https://huggingface.co/datasets/PediaMedAI/CogSense-Bench Correspondence to: xucao@pediamed.ai

1 Introduction

Multimodal Large Language Models (MLLMs) have rapidly advanced open-vocabulary visual understanding, enabling strong performance in recognition, grounding, and many compositional perception tasks (Yang et al., 2025a). Yet, existing models can describe what is present but struggle to mentally operate on visual information, e.g., by explaining abstract layouts, simulating future transformations, or inferring visual rules in diagrams (Schulze Buschoff et al., 2025). These failures expose a gap between high-level cognitive reasoning and low-level perception, and are becoming key bottlenecks in modern VQA and visual reasoning benchmarks (Lu et al., 2024; Cao et al., 2025).

Current approaches often attempt to narrow this gap by eliciting Chain-of-Thought (CoT) reasoning as natural-language reasoning traces, encouraging models to articulate intermediate steps explicitly (Wei et al., 2022; Zhang et al., 2024c), or relying on help from additional tools (Hu et al., 2024). However, even in multimodal settings, the intermediate computation is externalized in text, which is a poor interface for many visuospatial operations. Many subroutines underlying abstract visual reasoning, such as mentally rotating shapes, simulating dynamics, or inducing rules in pattern matrices, are most naturally expressed as geometric transformations, continuous states, or structured visual relations rather than as a sequence of discrete tokens. Expressing these intermediate states solely in linear text can introduce representational

bottlenecks, where spatial relations are compressed into discrete tokens, increasing the risk of information loss and brittle reasoning.

To quantify this cognitive gap, we introduce CogSense-Bench, a comprehensive benchmark that operationalizes visual cognition along five core dimensions: fluid intelligence, crystallized intelligence, visuospatial cognition, mental simulation, and visual routines, where these dimensions are the foundation of intuitive theories of physics and psychology (Lake et al., 2017). Using CogSense-Bench, we systematically re-evaluate SoTA MLLMs and find that substantial weaknesses persist across all dimensions, even when models are equipped with CoT prompting, suggesting that text-only reasoning is indeed often a brittle interface for tasks that require multi-step visual transformation and manipulation.

We therefore seek to shift part of the intermediate reasoning from discrete tokens to a representation space that better preserves geometry, continuity, and structured visual relations. This direction is consistent with cognitive science accounts of a visuospatial sketchpad (the “mind’s eye”) that supports maintaining and transforming internal visual representations during problem solving (Ganis & Schendan, 2011; Xie et al., 2020; Tabi et al., 2022), and with recent progress showing that latent visual states can serve as effective substrates for prediction and world modeling (Yang et al., 2025b). Motivated by these insights, we explore whether equipping MLLMs with an internal visual reasoning substrate can better support multi-step, non-linguistic manipulation of visual information. To this end, we introduce Cognitive Supersensing, a training paradigm that encourages MLLMs to construct latent image-based internal reasoning chains. Concretely, we augment an MLLM with a Latent Visual Imagery Prediction (LVIP) head that predicts a sequence of visual imagery latent embeddings, latent “imagery” states that encode intermediate visual reasoning, and aligns them with image representations derived from the target answer and supervision signal, thereby grounding the latent chain toward the correct solution. This design enables multi-step reasoning in a representation space that better preserves visual structure, while still allowing language to provide high-level semantic guidance and final explanations.

To train models under this paradigm, we further curate CogSense-Dataset, a targeted training set spanning five categories and eleven sub-tasks, and adopt a multi-stage training pipeline. First, in a supervised fine-tuning stage, we optimize a LVIP objective that provides an auxiliary learning signal encouraging intermediate latent visual states to stay predictive of the final answer’s visual representation, tightly coupling the reasoning process with the solution. Second, we introduce a reinforcement learning stage that optimizes rollout trajectories in a latent-space conditioned on these visual cognitive latents, encouraging coherent latent dynamics and discouraging brittle text-only reasoning paths. Across extensive experiments, CogSense-8B achieves SoTA performance in all tasks and the strongest overall accuracy (73.8%), surpassing GPT-5.2 by +33.5. It also has improvement in out-of-domain (OOD) generalization, outperforming strong baselines on challenging broader science VQA benchmarks. Taken together, our results suggest that latent visual imagery can serve as an effective internal scaffold that bridges perceptual recognition and cognitive-level multimodal reasoning.

Overall, our contributions are:

- • We propose Cognitive Supersensing, a latent-space reasoning-and-learning framework with supervised finetuning and reinforcement learning stages that endows MLLMs with visual imagery capability by aligning semantic reasoning with latent visual world modeling.
- • We introduce CogSense-Bench, a comprehensive benchmark spanning five cognitive dimensions, fluid intelligence, crystallized intelligence, visuospatial cognition, mental simulation, and visual routines, providing a systematic testbed for evaluating visual cognition beyond perceptual recognition and supporting future research in this direction.
- • Through extensive experiments, we show that reasoning and planning purely in text space is often insufficient for visual cognition: CogSense-8B achieves SoTA performance on CogSense-Bench and exhibits strong OOD generalization on challenging mathematics and science VQA benchmarks, outperforming competitive baselines that rely on text-only planning.

###### Fluid Intelligence Crystallized Intelligence Visuospatial Cognition Mental Simulation Visual Routines

Based on the pattern shown in the question image, which option completes the sequence?

Select the image that belongs to the same category as the images in the question set.

Select the image that belongs to the same category as the images in the question set.

[Figure 2]

Find the odd one out.

- A
- B
- C

A B

A B

A B

B

A

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

D

C D

C

C D

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

F

E

E F

C D

E F

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

H

G

G H

G H

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Select the image that belongs to the same category as the images in the question set.

Based on the pattern shown in the question image, which option completes the sequence?

Select the image that belongs to the same category as the images in the question set.

Based on the pattern, what should the output look like?

Find the odd one out.

[Figure 47]

A B

A B

A B

B

A

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

A

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

C D

C D

D

C

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

C D

E F

E F

F

- B C D

Based on the pattern, what should the output look like?

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

A

- B C D

E

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

H

G

G H

G H

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Based on the pattern shown in the question image, which option completes the sequence?

Select the image that belongs to the same category as the images in the question set.

Select the image that belongs to the same category as the images in the question set.

Find the odd one out.

A B

A B

- A
- B
- C
- D

A B

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

C D

C D

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

E F

C D

E F

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

G H

G H

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Select the image that belongs to the same category as the images in the question set.

Based on the pattern shown in the question image, which option completes the sequence?

Apply the transformation shown in the example and select the correct answer from the choices.

Select the image that belongs to the same category as the images in the question set.

Find the odd one out.

A B

- A
- B
- C
- D

- A
- B
- C

A B

A B

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

C D

C D

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

C D

E F

E F

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

G H

G H

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

###### Figure 1 CogSense-Dataset Examples. Samples across each category from the CogSense-Dataset. CogSense-Dataset comprises various visual cognitive questions classified into five categories: Fluid Intelligence, Crystallized Intelligence, Visuospatial Cognition, Mental Simulation, and Visual Routines, which require visual imagery and cognitive supersensing with deep thinking and reasoning.

[Figure 177]

[Figure 178]

Fluid Intelligence 28% MaRs-VQA (1.4K), PGM(10K), RAVEN (18K)

[Figure 179]

10%

28%

Crystallized Intelligence 37%

Bongard-RWR+ (16K), Bongard-HOI (23K)

14%

Bongard-LOGO (12K)

Visuospatial Cognition 11%

CogSense-Dataset 105K

[Figure 180]

[Figure 181]

11%

Mental Simulation 14%

KiVA (1.4K), STARE (4K), ARC-AGI (1.6K), ARC-AGI-2 (8K)

CVR (10K)

Visual Routines 10%

37%

- Figure 2 CogSense-Dataset Distribution. The data distribution of our CogSense-Dataset-105K.

- 2 CogSense Dataset and Benchmark

While numerous VQA benchmarks evaluate MLLMs across various tasks, they predominantly focus on semantic recognition or description (Li et al., 2025e; Chia et al., 2024). There remains a lack of systematic evaluation protocols specifically designed to assess high-level visual cognition, i.e. the ability to reason over visual inputs through distinct cognitive mechanisms such as abstract reasoning, spatial structuring, mental simulation, and attention-driven operations. To address this gap, we introduce CogSense-Dataset-105K, a comprehensive and large-scale dataset, together with a unified multi-task benchmark, CogSense-Bench.

Visual cognition supports high-level visual reasoning and inference processes, such as causal reasoning, intuitive physics, and intuitive psychology (Schulze Buschoff et al., 2025; Battaglia et al., 2013). These capacities reflect central dimensions through which humans interpret, predict, and reason about visual environments. To systematically evaluate whether MLLMs exhibit analogous competencies, we assess visual cognitive intelligence across five theory-grounded cognitive categories. These categories are selected to align with established constructs in cognitive science and psychology, and to capture complementary mechanisms underlying human visual reasoning as articulated in prior theoretical and empirical work.

First, Fluid Intelligence (Gf) evaluates the capacity to solve unseen reasoning problems independently of prior knowledge (Cattell, 1963). Grounded in Structure Mapping Theory (Gentner, 1983), this requires the model to transcend surface attributes and map high-order logical rules. Crystallized Intelligence (Gc) targets the utilization of learned world knowledge (Cattell, 1963), relying on Inductive Reasoning and Prototype Theory to abstract semantic concepts from visual variance (Rosch, 1973). To assess 3D spatial understanding, VisuospatialCognition tests the reconstruction of structural relationships, requiring the ability to group discrete visual elements into coherent, holistic structures based on laws (Wertheimer, 1923) and the decomposition of shapes into geometric primitives (geons) consistent with Recognition-by-Components Theory (Biederman, 1987). Distinct from static recognition, Mental Simulation demands that models act as "simulation engines," employing Hypothetico-Deductive Reasoning to synthesize programs and infer hidden dynamics (Battaglia et al., 2013). Finally, Visual Routines evaluates efficiency in visual search, requiring the composition of elementary operations and Focused Attention to bind features and exercise inhibitory control (Ullman, 1984).

Fig. 1 depicts examples of each category, and Fig. 2 shows the data distribution of CogSense-Dataset. Detailed statistics are provided in Appendix A. Appendix B provides the implementation details for the data pipeline.

CogSense Dataset and Benchmark are designed to probe high-level visual cognition beyond recognition, including abstract rule induction, spatial structuring, mental simulation, and attention control. These categories often require (i) explicit multi-step reasoning that composes elementary operations, and (ii) maintaining and manipulating answer-oriented internal visual states during inference. Motivated by these requirements, we propose Cognitive Supersensing that couples chain-of-thought style rationale generation with latent visual imagery prediction, enabling MLLMs to better support simulation-like and stateful reasoning demanded by CogSense-Bench.

#### 3 Method

- 3.1 Preliminaries

Let V and Q denote the visual input and the textual prompt, respectively, and denote the multimodal input as X = (V,Q). Our goal is to learn a multimodal model parameterized by θ, which induces a conditional generation distribution qθ(· | X) over an output consisting of a reasoning rationale Z (a token sequence) and a final answer y.

Specifically, V is processed by a pre-trained visual encoder Encvis(·) to extract visual features, denoted as VV = {vi}Ti=1, where T is the number of images and vi corresponds to the i-th image. These features are mapped into the language embedding space via a projection layer P(·), yielding projected visual tokens hV = P(VV). Simultaneously, the prompt Q is tokenized and embedded using the LLM embedding layer to obtain textual tokens hQ. The LLM backbone Enctxt(·) takes the concatenated token sequence [hV,hQ] as input and produces hidden states used by (i) a text decoder to generate (Z,y) autoregressively and (ii) a Latent Visual Imagery Prediction (LVIP) head to predict answer-oriented latent imagery.

The training follows a three-stage pipeline, shown in Fig. 3: (1) Reasoning Chain Generation to synthesize high-quality rationales, (2) SFT with Latent Visual Imagery Prediction to jointly learn text generation and latent imagery prediction, and (3) RL with Latent Rationales to further optimize rationale sampling.

- 3.2 Stage I: Reasoning Chain Generation

To address the scarcity of high-quality cognitive training data for visual CoT reasoning, we synthesize rationales using a powerful MLLM as a teacher model, denoted as MT. Given a multimodal input pair (V,Q), MT generates a reasoning rationale Z that demonstrates the logical deduction steps, along with a predicted answer yˆ, which we then filter to ensure alignment with the ground-truth answer y.

Specifically, for each piece of data, we construct a task-specific generation prompt Pgen. The teacher model is instructed to analyze the visual input and produce a step-by-step reasoning chain based on (V,Q,Pgen):

(Z,yˆ) ∼ MT(· | V,Q,Pgen). (1)

We filter out generated reasoning chains that fail to reach the correct conclusion (ˆyi ≠ yi) or contain hallucinated content. This yields an augmented dataset Dchain = {(Vi,Qi,Zi,yi)}Ni=1 for SFT. Additional details regarding Pgen are available in Section B.3.

- 3.3 Stage II: SFT with Latent Visual Imagery Prediction

We propose Cognitive Supersensing as an approach for acquiring visual cognitive capabilities. Motivated by the human constructive matching process, we introduce an auxiliary module, Latent Visual Imagery Prediction (LVIP), to predict hy, the latent representation of the ground-truth answer option image. LVIP is trained jointly with the standard supervised fine-tuning objective.

LVIP Head. The LVIP head gψ(·) is a two-layer MLP attached to the shared LLM backbone, in addition to the text decoder. It predicts the latent representation of the answer option using the backbone hidden states corresponding to visual tokens.

We assume the visual input V contains the question image together with candidate option images. Let HV ∈ RN×d be the sequence of hidden states for all visual tokens output by Enctxt(·). We extract the subset corresponding to the option images, denoted as Hopt ∈ RM×d with M ≤ N. We apply average pooling over Hopt to obtain an aggregated representation h¯opt, and compute the predicted latent imagery hˆy = gψ(h¯opt).

Learning Objectives. Let Vy denote the candidate option image indexed by the ground-truth answer y. The supervision target for LVIP is the embedding of Vy extracted by the (frozen) visual encoder, where hy = Encvis(Vy). We optimize LVIP with an MSE loss between hˆy and hy, jointly with the standard

[Figure 182]

###### Stage III: RL with Latent Rationales

###### Stage I: Reasoning Chain Generation

[Figure 183]

Select the image that belongs to the same category as the images in

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

𝑲𝑲𝑳𝑳(𝒒𝒒𝜽𝜽(𝒁𝒁|𝑿𝑿)|| 𝒑𝒑∗(𝒁𝒁|𝑿𝑿, 𝒚𝒚))

[Figure 189]

[Figure 190]

[Figure 191]

the question set.

Target posterior: 𝒑𝒑∗(𝒁𝒁|𝑿𝑿,𝒚𝒚)

[Figure 192]

[Figure 193]

[Figure 194]

𝒛𝒛𝒏𝒏 𝒛𝒛𝒏𝒏 𝒛𝒛𝒏𝒏

[Figure 195]

Question Options Prompts

𝒛𝒛𝟎𝟎

𝒛𝒛𝒏𝒏

Vision Encoder

𝒁𝒁𝒓𝒓𝒓𝒓𝒓𝒓

<reasoning> Let’s think step by step: ### Step 1, <picture1> describes that ...### Step2, <picture2> describes that... </reasoning> In conclusion, ...

𝒛𝒛𝟎𝟎

###### Large Language Model

𝑹𝑹𝒂𝒂𝒏𝒏𝒂𝒂

𝑹𝑹𝒍𝒍𝒍𝒍𝒍𝒍𝒑𝒑

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 196]

[Figure 197]

[Figure 198]

Text Decoder

LVIP Head

LVIP Head

FrozenScorer 𝒒𝒒𝜽𝜽𝟎𝟎

[Figure 199]

The final answer is A.

[Figure 200]

𝒒𝒒𝜽𝜽

[Figure 201]

[Figure 202]

SFT

Answer

Stage II: SFT with Latent Visual Imagery Prediction

Latent Visual Imagery for the Ground Truth

CogSense-8B

Visual Tokens Text Tokens

| |
|---|

| |
|---|

- Figure 3 The framework of Cognitive Surpersensing. Left: Architecture Overview. CogSense-8B is a VLM that takes images and prompts as input with a text decoder to generate the answer and a Latent Visual Imagery Prediction (LVIP) head to generate a latent visual imagery of the option-image in parallel. Right: Method Overview. To train CogSense-8B, we

(1) generate reasoning paths via LLMs, (2) implement SFT to jointly optimize the LVIP head and the model weights, and (3) implement RL to further optimize reasoning paths with Latent Rationales.

autoregressive cross-entropy loss over the target text sequence x = (Z,y) conditioned on X = (V,Q):

|x|

log qθ(xt | X,x<t) + β · MSE(hˆy,hy), (2)

LSFT = −

t=1

where β balances the two objectives. At inference time, the final answer is generated by the text decoder; the LVIP head can be kept as a frozen auxiliary module to produce hˆy for optional answer-oriented grounding.

- 3.4 Stage III: RL with Latent Rationales

In this stage, we leverage diversity-seeking RL by performing amortized variational inference over latent rationales using a Generative Flow Network (Bengio et al., 2021, 2023; Lahlou et al., 2023; Zhang et al., 2022). Concretely, we refine the rationale policy qθ(Z | X) to sample rationale trajectories Z in proportion to an unnormalized trajectory score, rather than committing to a single deterministic chain. We define the corresponding reward-induced target posterior over rationales as

p∗(Z | X,y) ∝ exp(R(Z;X,y)) (3)

and train qθ(Z | X) to approximate this posterior via the flow-matching objective. We define the trajectory score as a weighted combination of answer evidence and LVIP-based representation grounding:

R(Z;X,y) = α Rans(Z;X,y) + γ Rlvip(Z;X,y). (4)

Importantly, the LVIP term provides answer-oriented grounding in representation space, which complements discrete answer supervision and facilitates exploration over long rationale trajectories.

###### Table 1 Cognitive Ability Results. Performance comparison on CogSense-Bench with CogSense-8B and different MLLMs and VLMs. We bold the best results and underline the runner-ups. Human baseline is also listed on the first row.

Model Fluid Intel. Crystallized Intel. Visuospatial Cog. Mental Simu. Visual Rout. Avg.

Human 82.7 91.3 88.5 97.9 78.7 88.4 Gemini 2.5 Flash 23.2 40.2 31.0 40.2 45.3 36.3 GPT-o3 4.7 51.4 20.4 38.7 43.0 32.3 GPT-5.2 29.4 35.9 57.5 60.0 37.6 40.3 Claude Sonnet 4 22.5 31.3 26.6 58.0 34.4 32.6 Grok 4 Fast 13.0 45.4 41.6 21.3 37.6 31.7 Llama-4-Scout-17B 20.3 29.9 35.4 48.7 41.9 31.8 Gemma-3-27B 18.5 29.4 39.8 55.3 43.0 32.7 Ministral-14B 9.42 10.6 10.6 39.3 31.2 16.5 Intern-S1 6.6 21.0 12.4 17.5 42.6 17.4 Qwen3-VL-30B 30.8 34.0 37.2 56.0 40.9 37.4 CogSense-8B (Ours) 63.8 91.0 69.0 68.0 50.5 73.8

Token-WiseMarginalRewardEstimation. Let Z = (z1,...,zn,⊤) be an autoregressive rationale, with ⊤ denoting end of sequence. We define the answer-evidence component using a frozen scorer to avoid a moving-target reward:

##### (y | X,Z), (5) where qθ

Rans(Z;X,y) = log qθ

0

is a frozen scorer (a fixed copy of the supervised model). To incorporate LVIP grounding during RL while keeping it stationary, we keep the LVIP head gψ frozen and compute a representation-level reward that depends on Z through the backbone conditioning on [X;Z]. Let Vy be the option image indexed by the ground-truth answer y, and let hy = Encvis(Vy) be its embedding from a frozen visual encoder. Let h¯opt(X,Z;θ) denote the average-pooled final-layer hidden state of the option-image visual tokens produced by the backbone when conditioned on [X;Z]. We define

0

Rlvip(Z;X,y) = − gψ h ¯opt(X,Z;θ) − hy 22 . (6)

To provide prefix-level training signals required by flow-based objectives, we define the prefix state at step t as τt = (z1:t,⊤), where ⊤ indicates the availability of a termination action rather than an emitted token. Instead of evaluating the scorer at every token, we compute R(τt) only at sparse anchor indices with stride λ and linearly interpolate within each segment. Specifically, let anchors be t ∈ {0,λ,2λ,...} with t < n, and define t+ = min(t + λ,n). For any integer i ∈ {0,1,...,t+ − t},

i t+ − t

R(τt+) − R(τt) . (7)

R(τt+i) = R(τt) +

At anchor positions, we set R(τt) ≡ R(τt;X,y) with R(τt;X,y) = α log qθ

(y | X,z1:t) + γ Rlvip(τt;X,y), (8)

0

where Rlvip(τt;X,y) is computed analogously by conditioning the backbone on [X;z1:t]. We then plug R into a standard SubTB loss (Madan et al., 2023) to train qθ(Z | X) toward the reward-induced target distribution.

Reference-Guided GFlowNet Fine-tuning. To reduce variance from low-quality samples, we anchor exploration with a reference rationale Zref . For each X, we sample m candidates {Zi}mi=1 ∼ qθ(· | X) and keep only those that meet a relative evidence threshold:

I(Zi) = 1[R(Zi;X,y) ≥ R(Zref;X,y) + log δs ], (9)

- Table 2 General Ability Results. Performance comparison in general tasks with CogSense-8B and the base model. CogSense-8B demonstrates similar general ability performance to the base model.

Model HallusionBench AI2D GQA ScienceQA RealWorldQA ChartQA BLINK MMStar

Qwen3-VL-8B(base) 61.1 85.4 71.4 92.6 71.5 88.6 64.7 70.9 CogSense-8B (Ours) 60.5 85.1 71.8 92.6 71.9 84.7 65.3 66.8

- Table 3 Ablation Study Results. We compare the base model and three variants: SFT w/o LVIP, SFT w/ LVIP, SFT w/o LVIP + GRPO, SFT w/ LVIP + GRPO and CogSense-8B. We bold the best results and underline the runner-ups.

Variant Fluid Intel. Crystallized Intel. Visuospatial Cog. Mental Simu. Visual Rout. Avg.

Qwen3-VL-8B (base) 31.2 34.8 31.0 45.3 40.9 35.5 Qwen3-VL-8B SFT w/o LVIP 51.1 76.6 63.7 59.3 41.9 62.3 Qwen3-VL-8B SFT w/ LVIP 55.4 88.6 61.1 61.3 44.1 68.0 Qwen3-VL-8B SFT w/o LVIP + GRPO 55.8 79.9 63.7 63.3 43.0 65.5 Qwen3-VL-8B SFT w/ LVIP + GRPO 59.1 89.9 64.6 65.3 46.2 70.8 CogSense-8B (Ours) 63.8 91.0 69.0 68.0 50.5 73.8

where δs ∈ (0,1] and s is the index of the current training step (so log δs ≤ 0 controls the allowed slack in log-space relative to the reference). We optimize SubTB only on accepted trajectories:

L(θ) =

m

i=1

I(Zi) · LsubTB(Zi;θ), (10)

where LsubTB is instantiated with the densified prefix scores R.

Bayesian Posterior over Latent Rationales. At inference time, we treat Z as latent and aggregate evidence across multiple sampled rationales. We sample N rationales {Zi}Ni=1 ∼ qθ(Z | X), decode an answer yi conditioned on (X,Zi), and compute a length-normalized evidence score with the frozen scorer

Si =

1 |Zi| + |yi|

log qθ

0

(yi | X,Zi). (11)

We then output yˆ = yi∗, where i∗ = arg maxi Si, which serves as a simple MAP-style selection over sampled latent rationales and reduces sensitivity to any single brittle chain under reasoning ambiguity.

- 4 Experiments

- 4.1 Experimental Settings

Baselines and Metrics. We compare CogSense-8B with several mainstream open and closed-source MLLMs and VLMs with strong multimodal performance: Gemini 2.5 Flash (Comanici et al., 2025), GPT-o3 (OpenAI, 2025b), GPT-5.2 (OpenAI, 2025a), Claude-Sonnet-4 (Anthropic, 2025), Grok 4 Fast (xAI, 2025), Llama-4Scout-17B (Meta, 2025), Gemma-3-27B (Team, 2025), Ministral-3-14B (MistralAI, 2025), Intern-S1 (Bai et al., 2025a), and Qwen3-VL-30B (Bai et al., 2025b). The accuracy rate is used as the metric to evaluate the performance in all tasks. We also let 20 participants complete 100 questions selected from CogSense-Bench through stratified sampling, reporting the comparison results of human-accurate rates for CogSense-Bench in Tab. 1 as a human baseline. This study has been approved by the Institutional Review Board (IRB). The detailed information of the human study can be found in Appendix C.

Implementation Details. All experiments are conducted with 8 NVIDIA H200 GPUs. We use Qwen3-VL-8B as the backbone to train CogSense-8B. We use Adam as the optimizer with η = 10−5 and weight decay 10−5.

- 4.2 Cognitive Ability Results

QuantitativeResults. As Tab. 1 illustrates, despite being an 8B model, CogSense-8B achieves SoTA performance in all tasks and delivers the strongest overall accuracy (73.8%), surpassing GPT-5.2 by +33.5. Notably, when

|Q: Based on the pattern shown in the question image, which choice completes the pattern?|
|---|

Visual Input Response from Gemini 2.5 Flash

###### A

B

Step 1: By examining the properties across the rows, three rules emerge: All items in Row 3 are Pentagons. Each row contains Small, Medium, and Large. Row 3 is missing Large. Each row

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

C D

contains Light, Medium, and Dark. Row 3 is missing Dark/Black.

[Figure 207]

[Figure 208]

- Step 2: Comparing Row 2 and Row 3 reveals a shift pattern: the elements of Row 2 are moved one position to the left to create Row 3, with the first element wrapping around to the end. This confirms the missing piece corresponds to the first element of Row 2: a Large, Black Pentagon.

- Step 3: No option perfectly matches the derived "Large, Black Pentagon.“ G is the closest one. Answer: G

E F

[Figure 209]

[Figure 210]

G H

Errors occur during reasoning Response from CogSense-8B

[Figure 211]

[Figure 212]

[Figure 213]

Response from GPT-5

Step 1: The first row consists of diamonds, and the second row consists of pentagons. Since the third row starts with pentagons, the missing figure must also be a

[Figure 214]

Step 1: View the question image as a 3×3 grid: each row keeps the same shape type and follows the same left→right progression in size and shading.

[Figure 215]

###### pentagon.

- Step 2: From rows 1–2, the pattern is large & darkest → small & lightest → medium & dark (same structure for each row).

- Step 3: In row 3, the missing cell must therefore be a mediumsized dark pentagon, which matches choice F. So, the final answer is F.

- Step 2: There is a subtraction relationship between the columns for both size and color shade (Value 1 - Value 2 = Value 3). Size: Large

(3) - Small (1) = Medium (2). Color: Black (3) - Light Grey (1) = Dark Grey (2).

- Step 3: Apply Logic to the Third Row: Size: Medium (2) - Small (1) = Small (1).Color: Dark Grey (2) - Light Grey (1) = Light Grey (1). The missing figure must be a Small Light-Grey Pentagon.

Errors occur during reasoning

[Figure 216]

[Figure 217]

Looking at the options, B is the correct answer.

- Figure 4 Qualitative Example of Visual Cognition Reasoning Across Models. We underline decisive sentences in the reasoning chain. CogSense-8B demonstrates a coherent, multi-step logical chain that closely matches the ground truth, while other models exhibit less precise or less interpretable reasoning paths

compared against human performance, CogSense-8B substantially narrows the performance gap relative to existing MLLMs, outperforming the next strongest baseline by a large margin across all cognitive categories. This reduction in the human–model gap suggests that CogSense-8B more effectively leverages visual imagery to support cognitive reasoning, rather than relying primarily on superficial or text-biased heuristics.

Qualitative Comparison. Fig. 4 shows a qualitative comparison between CogSense-8B and mainstream models, demonstrating that Gemini 2.5 Flash and GPT-5.2 both fail in extracting the correct underlying rule of the pattern, leading to a wrong answer. In contrast, CogSense-8B demonstrates high-quality reasoning with a concise and clear expression of the underlying pattern. This comparison suggests that CogSense-8B more effectively captures the underlying visual regularities required for abstract pattern reasoning. Additional qualitative examples can be found in Appendix D.

- 4.3 General Ability Results

We also evaluate the general vision-language understanding ability of CogSense-8B. As shown in Tab. 2, we evaluate on vision-language datasets such as HallusionBench (Guan et al., 2024), AI2D (Kembhavi et al., 2016), GQA (Hudson & Manning, 2019), ScienceQA (Lu et al., 2022), RealWorldQA (xAI, 2024), ChartQA (Masry et al., 2022), BLINK (Fu et al., 2024), MMStar (Chen et al., 2024). CogSense-8B maintains robust performance on these general benchmarks, comparable to the base model. This balance proves that the visual cognitive abilities of Cognitive Supersensing are not due to overfitting of data, but successfully injects high-level visual cognitive knowledge while preserving the foundational knowledge and instruction-following capabilities inherent in the pre-trained backbone.

- 4.4 Ablation Study

As summarized in Tab. 3, to isolate the contributions of LVIP and our proposed reinforcement learning mechanism, we perform a hierarchical evaluation across five variants: the standard SFT without LVIP, SFT with LVIP, SFT without LVIP plus GRPO, SFT with LVIP plus GRPO, and CogSense-8B that combines LVIPsupervised SFT and the RL method we proposed. Compared to the vanilla Qwen3-VL-8B, applying standard

SFT nearly doubled the average performance. The integration of LVIP introduces a further improvement, raising the average accuracy to 68.0%. This suggests that LVIP helps the model align semantic reasoning and abstract world modeling, enabling the model to gain initial “supersensing" capabilities. To validate the necessity of our specific RL design, we introduce GRPO as a reinforcement learning baseline. Although applying GRPO to the SFT models (w/ or w/o LVIP) yields observable gains, achieving improvement by +3.2 and +2.8 compared to models without GRPO, it serves primarily to refine the policy within a standard optimization scope. Finally, CogSense-8B achieves the SoTA performance of 73.8% on average, outperforming the strongest baseline (SFT w/ LVIP + GRPO) by a distinct margin. This result highlights that our elaborately designed RL method is not only an optimization trick, but a specialized reasoning refiner that is more effective than general GRPO in leveraging the visual supersensing established by LVIP.

- 4.5 Out-of-Domain Evaluation Q: Complete the matrix.

Q: The kangaroo wants to visit the koala. On its way, it is not allowed to jump through a square with water. Each arrow shows one jump onto a neighbouring field. Which path is the kangaroo allowed to take?

To further verify the generalization capability of CogSense-8B, we extend our evaluation to out-of-domain scenarios using the Chemistry and Math subsets of the EMMA benchmark (Hao et al., 2025), with both the question and the options having images. Examples for selected data from EMMA are displayed in Fig. 5. As shown in Tab. 4, CogSense-8B achieves substantial gains in the Chemistry and Math subsets, respectively, by +6.2 and +8.8, confirming that our method learns generalized visual cognition patterns rather than overfitting to specific training data.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Pattern Inference

Path Tracing/Change of view simulation

[Figure 222]

Q: The %yield of ammonia as a function of time in the reaction N2 g + 3H2 g ⇌ 2NH3 g ,Δ𝐻𝐻 < 0 at P, T1 is given below. If this reaction is conducted at P, T2 with T2 > T1, the %yield of ammonia as a function of time is represented by:

Q: An 'arrow-pushing' diagram is a common type of chemical image used to illustrate electron flow in mechanistic steps. The molecule undergoes changes after the electron has been relocated or reacted. Which of the following options shows the molecule after the change?

[Figure 223]

#### 5 Related Work

[Figure 224]

Our work is closely related to Abstract Reasoning, Visual Cognition, VLMs for Visual Reasoning, and Latent Visual Reasoning. A comprehensive discussion is provided in Appendix E.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

A B C D

Reaction Simulation-Pro Graph Reasoning

Figure 5 EMMA Benchmark Sample Problems.

#### 6 Conclusion

Table 4 Out-of-Domain Evaluation. We evaluate the generalization ability of CogSense-8B compared with the backbone on the EMMA benchmark.

In this work, we propose Cognitive Supersensing, a paradigm designed to bridge the gap between perceptual processing and complex cognitive reasoning in MLLMs. By integrating a Latent Visual Imagery Prediction (LVIP) head and employing a training strategy that combines SFT and RL with Latent Rationales, we enable CogSense-8B to simulate internal visual imagery aligned with semantic reasoning chains. We further introduce CogSense-Bench, a comprehensive evaluation suite targeting five dimensions of visual cognition. Our experiments demonstrate that CogSense-8B achieves superior performance compared to SoTA MLLM baselines on these cognitive tasks. These results indicate that modeling the interaction between visual simulation and logical deduction is an effective direction for advancing the reasoning abilities of multimodal systems.

Model Chemistry Mathematics Qwen3-VL-8B (base) 39.2 26.0 CogSense-8B (Ours) 45.4↑6.2 34.8↑8.8

Appendix

- A Statistics and Samples of Dataset and Benchmark. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- B Implementation Details for Data Pipeline. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- B.1 Data Extraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- B.2 Data Reformatting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- B.3 Reasoning Chain Generation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- C Human Study Design and Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- D More Qualitative Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- E Related Works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- A Statistics and Samples of Dataset and Benchmark

We conclude CogSense Dataset and Benchmark statistics in Tab. A1 and Tab. A2. We ensured zero data leakage by randomly sampling instances from each category at the same proportion as in the CogSense-Dataset and subsequently removing the sampled data from it.

- Table A1 CogSense-Dataset Descriptions and Statistics. We have listed the number of VQA pairs corresponding to each category.

Category Dataset #Test License

Fluid Intelligence

MaRs-VQA 1.4K CC-BY-NC-3.0 PGM 10K Unspecified RAVEN 18K GPL-3.0 License

Crystallized Intelligence

Bongard-RWR+ 16K CC-BY-4.0 Bongard-HOI 23K NVIDIA Source Code License

Visuospatial Cognition Bongard-LOGO 12K MIT license

Mental Simulation

KiVA 1.4K Apache-2.0 License STARE 4K Unspecified ARC-AGI 1.6K Apache-2.0 License ARC-AGI-2 8K Apache-2.0 License

Visual Routines CVR 10K Apache-2.0 License Sum 105.4K

- Table A2 CogSense-Bench descriptions and statistics. We have listed the number of VQA pairs corresponding to each category.

Category Dataset #Test License

MaRs-VQA 13 CC-BY-NC-3.0 PGM 94 Unspecified RAVEN 169 GPL-3.0 License

Fluid Intelligence

Bongard-RWR+ 152 CC-BY-4.0 Bongard-HOI 216 NVIDIA Source Code License

Crystallized Intelligence

Visuospatial Cognition Bongard-LOGO 113 MIT license

KiVA 13 Apache-2.0 License STARE 26 Unspecified ARC-AGI 31 Apache-2.0 License ARC-AGI-2 80 Apache-2.0 License

Mental Simulation

Visual Routines CVR 93 Apache-2.0 License Sum 1000

- B Implementation Details for Data Pipeline

- B.1 Data Extraction

We searched for valuable raw data related to visual cognition from a large number of datasets with various tasks, such as Bongard-OpenWorld (Wu et al., 2024), Bongard-HOI (Jiang et al., 2022), Bongard-LOGO (Nie

- et al., 2020), Bongard-RWR+ (Pawlonka et al., 2025), CVR (Zerroug et al., 2022), KiVA (Yiu et al., 2025), MaRs-VQA (Cao et al., 2025), PGM (Barrett et al., 2018), RAVEN (Zhang et al., 2019), I-RAVEN (Hu
- et al., 2021), RAVEN-FAIR (Benny et al., 2021), A-I-RAVEN (Małkiński & Mańdziuk, 2025), I-RAVENMesh (Małkiński & Mańdziuk, 2025), I-RAVEN-X (Camposampiero et al., 2025), STARE (Li et al., 2025e), ARC-AGI (Chollet, 2019), ARC-AGI-2 (Chollet et al., 2025), etc, and manually selected valuable data that fit pre-defined five categories from datasets mentioned above. These sources were strictly selected to ensure high relevance to the downstream tasks. Building upon these seed sources, we developed automated extraction and cleaning scripts to perform a large-scale expanded collection and finally got the raw dataset.

- B.2 Data Reformatting

In order to standardize the format of the dataset, we reformatted those questions where the metric was not originally multiple-choice to become multiple-choice.

Reformatting for Bongard Problems. We randomly selected one image from the positive side and mixed it with the negative samples, shuffling them as options. The remaining positive samples were used as the question.

Reformatting for ARC-AGI Problems. We perform data augmentation on the ground-truth images, including methods such as color modification. Both the ground-truth images and the augmented images are provided as options.

- B.3 Reasoning Chain Generation

As stated in the main paper, we designed to transform traditional short QA pairs into high-quality reasoning data containing an explicit reasoning chain. We customized different prompts for different types of original questions and used LLMs to generate corresponding reasoning chains based on these different prompts. The model was required not only to output the final answer but also to elaborate on its reasoning process. Specifically, the model was guided to analyze visual cues within the images, infer underlying rules, and explain causal relationships, finally generating detailed reasoning process text. To ensure quality, we filter out generated reasoning chains that fail to reach the correct conclusion or exhibit hallucinated content. Fig. B1 shows the customized prompts.

- C Human Study Design and Setup

Participant Recruitment. To establish a human performance baseline, we recruited a total of 20 participants through online platforms using a random sampling strategy. Participation in the study was entirely voluntary, and no financial compensation or monetary incentives were provided to the subjects.

Questionnaire Design. The questionnaire design involved a rigorous selection process to ensure representativeness. We constructed a test set comprising 100 multiple-choice questions derived from the CogSense-Bench. These questions were manually selected by stratified sampling to strictly maintain the category proportions of the original benchmark. The study was implemented and distributed digitally via the Google Forms platform, allowing for remote administration.

Compliance. Regarding experimental standards and ethical compliance, all protocols were reviewed and approved by the Institutional Review Board (IRB) prior to the study.

###### Reasoning Chain Generation Prompts for CogSense-Dataset

You are a careful visual reasoning assistant. Your task is to generate CoT and the final answer for the provided questions. Your response must consist of ONLY the following two parts in JSON format:

- 1. ‘reasoning’: reasoning steps (3-12 concise steps from high-level to details).
- 2. ‘final_answer’: the final answer of the question. Keep reasoning concise (3-12 steps). Do NOT include internal deliberation beyond the steps.

Output must be in JSON format and ONLY with the reasoning steps and final answer with nothing else. For Crystallized Intelligence and Visuospatial Cognition Questions: You will be given a set of images that share a common pattern. Analyze these examples to understand the pattern. Then, select the image that belongs to the same category as the images in the question set. Question images (these images share a common pattern): [question_images] Option images: [option_images] For Fluid Intelligence Questions: You will be given an image that contains eight subimages and a blank space. Then, given some option images, predict which one is the right option, following the same pattern to complete the sequence based on the pattern shown in the question image. Question image: <question_image> Option images: [option_images] For ARC-AGI in Mental Simulation: You will be given some pairs of input-output images as examples. The output is obtained by coloring the grid according to a specific pattern based on the input. Analyze these examples to understand the underlying grid color pattern. Then, given a new input image and option images, predict which image is generated by applying the same pattern to the input. Examples: [<input_image, output_image>] Question image: <new_input_image> Option images: [option_images] For KiVA in Mental Simulation: You will be given some pairs of input-output images or transform patterns as examples. Analyze the pattern and select the correct answer from the given options. Then, given a new input image and some options that indicate possible outputs, predict the right option following the same pattern. Examples: [pattern_image] Question image: <new_input_image> Option images: [option_images] For STARE in Mental Simulation: You will be given some images and a blank space. Then, given some option images, predict which one is the right option, following the same pattern to complete the sequence based on the pattern shown in the question image. Question image: <question_image> Option images: [option_images] For Visual Routines Questions: You will be given some images, in which one of them differs from the rest of the images. Find the odd one out. Question images: [images] Option images: [images]

Input: [images] and <prompt> Output: <reasoning_chain> and <answer>

###### Figure B1 Reasoning Chain Generation Prompts. We customize different prompts for different types of original questions and use LLMs to generate corresponding reasoning chains based on these different prompts.

### D More Qualitative Examples More qualitative examples are shown in Fig. D1.

###### Q: Select the image that belongs to the same category as the images in the question set.

Visual Input A B

Response from Gemini 2.5 Pro

Step 1: The common category is people actively riding motorized vehicles on the ground. The set features various types (monowheel, scooters, police, racing), all moving on roads or

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

C D

tracks.

[Figure 235]

[Figure 236]

[Figure 237]

- Step 2: Eliminate B, D, E, and H because the vehicles are stationary or on display. Retain A, C, F, and G as they depict riders in motion.

- Step 3: Images A and F show aerial stunts, which differ from the groundbased riding in the examples. Image G is too similar to existing images. Image C is the best choice: it fits the active riding rule while adding diversity (vintage style) to the set. So, the final answer is C.

E F

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

G H

Errors occur during analogy analysis

[Figure 242]

[Figure 243]

[Figure 244]

Response from Intern-S1

Response from CogSense-8B

Step 1: The images depict various motorcycle-related

Step 1: All images depict people actively riding motorcycles (or similar motorized two/one-wheeled vehicles) on a ground surface (road or track).

[Figure 245]

[Figure 246]

scenes. The task is to identify which image belongs to the same category as the others.

- Step 2: Upon analysis, most images (A, B, C, E, G, H) show motorcycles in motion or in use (e.g., stunts, racing, commuting, touring).
- Step 3: However, Image D stands out as it features a customized chopper motorcycle on display in a showroom or exhibition setting, which is a static, non-action-oriented context. This makes Image D the outlier. Answer: D

- Step 2: Images B and D are excluded because they show static bikes with no riders. Images A and F are excluded because they show aerial stunts rather than ground transportation/racing. Image C is excluded because while it fits the action, the style is inconsistent with the modern color images in the question.

- Step 3: Image G is the best fit as it clearly depicts the action of riding a motorcycle on a road/track, which is the defining category of the question set. The image that belongs to the same category is G.

Errors occur while understanding the question

[Figure 247]

[Figure 248]

###### Q: Based on the pattern shown in the question image, which choice completes the pattern?

Visual Input Response from Gemini 2.5 Pro

A B

Step 1: Compare the first and last columns. In Row 1, the Diamond Frame in the first cell transforms into a Cross with a Central Circle in the third cell.

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

C D

- Step 2: The first cell of Row 3 also features a Diamond Frame. Following the analogy from Row 1, the missing cell must contain a Cross with a Central Circle.

- Step 3: Look at the inner shapes. The pattern involves retaining a core element (the Star) and introducing a specific polygon. Option C fits this by combining the Star with a Pentagon, mirroring the complexity of previous rows. Option C is the correct answer.

[Figure 253]

[Figure 254]

E F

[Figure 255]

[Figure 256]

G H

Errors occur during reasoning Response from CogSense-8B

[Figure 257]

[Figure 258]

[Figure 259]

Response from InternVL3-78B

Step 1: The puzzle involves outer frames (Square, Circle, Diamond) and inner line patterns: Diagonal (X), Orthogonal (+), or Combined (*).

[Figure 260]

[Figure 261]

Step 1: The core pattern relies on a structure of intersecting lines (like crosses or diamonds) that establish the image's symmetry.

- Step 2: Row 1: The middle is Diagonal. The inner pattern changes from Diagonal (X) to Orthogonal (+). Row 2: The middle is Straight/Vertical. The inner pattern changes from Orthogonal (+) to Diagonal (X). Rule: The operator converts lines of its own type into the opposite type (Diagonal -> Orthogonal).

- Step 3: Following the rule, the Diagonal part (X) of the star converts to Orthogonal (+). The existing Orthogonal part stays the same. Correct Answer: F

- Step 2: The pattern requires a diverse mix of geometric shapes (stars, hexagons, squares) arranged specifically along these lines or at intersections.
- Step 3: Option D is the correct choice because it uniquely matches the high level of structural complexity and shape variety found in the pattern. The other options are either too simple or lack the correct framework. Answer: D

Errors occur during reasoning

[Figure 262]

[Figure 263]

###### Figure D1 More Qualitative Examples of Visual Cognition Reasoning Across Models. CogSense-8B demonstrates a coherent, multi-step logical chain that closely matches the ground truth, while other models exhibit less precise or less interpretable reasoning paths

- E Related Works

Our work is closely related to Abstract Reasoning, Visual Cognition, VLMs for Visual Reasoning, and Latent Visual Reasoning.

Bongard Problems. The Bongard Problems (BPs) are introduced by M. M. Bongard (Bongard, 1968, 1970) as a novel challenge to machine vision. A typical BP includes two sets, positive and negative, with each side consisting of six images that share a common pattern. It requires a system to induce the logical rule that distinguishes a set of positive examples from negative ones (Bongard, 1968, 1970). This task evaluates the core properties of human cognition, identifying the underlying rule that differentiates the sides and articulating it in natural language (Małkiński et al., 2025). To modernize this benchmark for deep learning, the Bongard-LOGO dataset was introduced, transforming the concept learning challenge into a few-shot binary classification problem (Nie et al., 2020). Recent works have expanded the field of BPs to include real-world images, such as Bongard-HOI (Jiang et al., 2022), Bongard-OpenWorld (Wu et al., 2024), Bongard-RWR (Małkiński et al., 2025), Bongard-RWR+ (Pawlonka et al., 2025). These works expand the range of presented objects, attributes, and relationships, illustrating human-object interactions and incorporating free-form concepts in the real world, thus increasing the diversity of featured scenes.

Matrix Reasoning. Matrix reasoning tasks play a critical role in assessing human intelligence, particularly in relation to visual cognition and working memory. (Salthouse, 1993; Jaeggi et al., 2010; Fleuret et al., 2011). These tasks are widely used in the psychology field through Raven’s Progressive Matrices (RPMs) (Raven, 1936; Raven & Court, 1998; John & Raven, 2003) and the Wechsler Intelligence Scale (WISC) (Wechsler, 1949, 2008; Kaufman et al., 2015) to evaluate human fluid intelligence and abstract visual reasoning. Early works, such as PGM (Barrett et al., 2018) and RAVEN (Zhang et al., 2019), are proposed to test if neural networks can learn abstract reasoning and to lift machine intelligence by associating vision with structural, relational, and analogical reasoning in a hierarchical representation. The claim that deep learning models can be trained to solve simple matrix reasoning was then proved by other works (Małkiński & Mańdziuk, 2025; Xu et al., 2023; Małkiński & Mańdziuk, 2024). Several datasets and benchmarks are also introduced to enlarge this field, such as I-RAVEN (Hu et al., 2021), RAVEN-FAIR (Benny et al., 2021), and CVR (Zerroug et al., 2022). However, these studies tend to overlook the ability of humans to solve such puzzles in a zero-shot manner, without the need for explicit training on extensive datasets. Therefore, some useful zero-shot visual reasoning inference datasets such as RAVEN-IQ (Huang et al., 2023) and Visual Reasoning Benchmark (Zhang et al., 2024b) are proposed to deal with the issue. The latest MaRs-VQA (Cao et al., 2025) overcomes previous works’ limitations of lacking rigorous human experiments as a reference and conducting experiments on relatively small datasets without psychometric validation.

Abstraction and Reasoning Corpus. Abstraction and Reasoning Corpus for Artificial General Intelligence (ARCAGI) is a novel general AI benchmark designed to measure a human-like form of general fluid intelligence and enable fair general intelligence comparisons between AI systems and humans (Chollet, 2019). Its measurement standard is based on the skill-acquisition efficiency and out-of-domain generalization over mere task performance. The tasks require systems to solve grid-based visual program synthesis problems by abstracting the underlying rules from only a minimal set of input-output examples (typically three to five pairs). A robust estimate of human performance (LeGris et al., 2024) on the full ARC public evaluation set places the average human accuracy at 64.2%, which greatly exceeds the current state-of-the-art AI methods. Its updated version, ARC-AGI-2 (Chollet et al., 2025), is more challenging. Even the most recent and powerful models at that time, such as OpenAI’s o3 (OpenAI, 2025b), achieve only negligible performance on the newly released, adversarially selected ARC-AGI-2 challenge. This reinforces that modern frontier reasoning models fundamentally lack the structural inductive biases that are necessary for human-like skill-acquisition and generalization.

VLMs for Vision Reasoning. Visual recognition-related Vision Language Model (VLM) studies have made great progress since the development of CLIP (Radford et al., 2021). It has now been proven that VLMs have the ability to address vision reasoning tasks (Zellers et al., 2019; Bordes et al., 2024). VLMs are the dominant architecture for bridging perception and language, typically integrating a visual encoder (e.g., CLIP) with pretrained LLMs via a cross-modal connector to align visual features with text space (Radford et al., 2021; Gupta & Kembhavi, 2023; Li et al., 2023; Liu et al., 2023; Zhang et al., 2024a; Shao et al., 2024; Fu

- et al., 2024). The training pipeline for VLMs often involves pre-training for modality alignment, followed

by instruction tuning or Supervised Fine-Tuning (SFT) to enhance general multimodal capabilities (Zhu et al., 2023; Li et al., 2023; Liu et al., 2023; Ye et al., 2024). Methodologies such as Qwen-VL (Bai et al., 2023; Wang et al., 2024a; Bai et al., 2025c,b), LLaVA (Liu et al., 2023, 2024), MiniGPT-4 (Zhu et al., 2023), InstructBLIP (Dai et al., 2023), CogVLM (Wang et al., 2024b), etc., highlight the significance of employing high-quality visual instruction tuning data. However, current VLMs still face challenges in adapting to high-resolution (Carvalho & Martins, 2025; Li et al., 2025d) and visually complex images in vision reasoning. This is because most current vision reasoning approaches are primarily text-level, with the LLM exploring textual tokens while the visual input remains static (Wu & Xie, 2024; Izadi et al., 2025; Li et al., 2025a; Bai et al., 2025d; Liu et al., 2025). This absence of a dynamic visual search mechanism limits the model’s ability to selectively acquire fine-grained visual cues. Other reasons stem from few-shot reasoning (Guo et al., 2023), compositional understanding (Yuksekgonul et al., 2023), and the constrained visual grounding capabilities inherent in CLIP (Tong et al., 2024), etc. Our work overcomes these problems, providing a novel approach for high-performing vision reasoning.

Latent Visual Reasoning. While standard VLMs rely on static visual encoding, a new paradigm of Latent Visual Reasoning has emerged, shifting the focus from passive perception to active, internal simulation. This approach draws inspiration from World Models (Zhu et al., 2025; Sun et al., 2025), enabling systems to "think" in a compressed latent space before generating a response. Rather than mapping pixels directly to text, these models perform intermediate reasoning steps, like System-2 cognition, by predicting future states or manipulating visual abstractions (Zhu et al., 2025; Sun et al., 2025). For instance, recent frameworks introduce "visual scratchpads" or latent tokens that allow the model to sketch out reasoning traces implicitly (Zhang et al., 2025; Li et al., 2025b). Technologies such as Mirage (Yang et al., 2025c) and Latent Sketchpad (Zhang

- et al., 2025) exemplify this by empowering models to generate and refine mental imagery during the inference process, effectively bridging the gap between visual grounding and abstract logic. Further advancements include Latent Visual Reasoning (LVR) (Li et al., 2025a) and implicit reasoning tokens (Li et al., 2025c), which allow models to perform autoregressive planning in the visual embedding space without explicit supervision. These methods overcome the limitations of text-centric reasoning by maintaining rich, high-dimensional visual information throughout the decision-making chain (Li et al., 2025a,c).

References

Anthropic. System card: Claude opus 4 & claude sonnet 4. System card (technical report), May 2025. URL https://www.anthropic.com/claude-4-system-card. Document shows changelog updates on July 16, 2025 and September 2, 2025.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308.12966.

Lei Bai, Zhongrui Cai, Yuhang Cao, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, et al. Intern-s1: A scientific multimodal foundation model, 2025a. URL https://arxiv.org/abs/ 2508.15763.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report, 2025b. URL https://arxiv.org/abs/2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report, 2025c. URL https://arxiv.org/abs/2502.13923.

Tianyi Bai, Zengjie Hu, Fupeng Sun, Jiantao Qiu, Yizhen Jiang, Guangxin He, Bohan Zeng, Conghui He, Binhang Yuan, and Wentao Zhang. Multi-step visual reasoning with visual tokens scaling and verification, 2025d. URL https://arxiv.org/abs/2506.07235.

David Barrett, Felix Hill, Adam Santoro, Ari Morcos, and Timothy Lillicrap. Measuring abstract reasoning in neural networks. In International Conference on Machine Learning, pp. 511–520. PMLR, 2018.

Peter W Battaglia, Jessica B Hamrick, and Joshua B Tenenbaum. Simulation as an engine of physical scene understanding. Proceedings of the National Academy of Sciences, 110(45):18327–18332, 2013.

Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. Advances in Neural Information Processing Systems, 34: 27381–27394, 2021.

Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J Hu, Mo Tiwari, and Emmanuel Bengio. Gflownet foundations. Journal of Machine Learning Research, 24(210):1–55, 2023.

Yaniv Benny, Niv Pekar, and Lior Wolf. Scale-localized abstract reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12557–12565, 2021.

Irving Biederman. Recognition-by-components: a theory of human image understanding. Psychological Review, 94(2): 115, 1987.

Mikhail Moiseevich Bongard. The recognition problem. Technical report, Foreign Technology Div Wright-Patterson

AFB Ohio, 1968. Mikhail Moiseevich Bongard. Pattern Recognition. Spartan Books, 1970. ISBN 978-0-87671-118-7. Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C. Li, Adrien Bardes, Suzanne Petryk, Oscar Mañas,

Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, et al. An introduction to vision-language modeling, 2024. URL https://arxiv.org/abs/2405.17247.

Giacomo Camposampiero, Michael Hersche, Roger Wattenhofer, Abu Sebastian, and Abbas Rahimi. I-raven-x: Benchmarking generalization and robustness of analogical and mathematical reasoning in large language and reasoning models, 2025. URL https://arxiv.org/abs/2510.17496.

Xu Cao, Yifan Shen, Bolin Lai, Wenqian Ye, Yunsheng Ma, Joerg Heintz, Jintai Chen, Meihuan Huang, Jianguo Cao, Aidong Zhang, and James M. Rehg. What is the visual cognition gap between humans and multimodal llms?, 2025. URL https://arxiv.org/abs/2406.10424.

Miguel Carvalho and Bruno Martins. Efficient architectures for high resolution vision-language models. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 10520–10530, Abu Dhabi, UAE, January

2025. Association for Computational Linguistics. Raymond B Cattell. Theory of fluid and crystallized intelligence: A critical experiment. Journal of Educational Psychology, 54(1):1, 1963.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.

Yew Ken Chia, Vernon Toh, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. Puzzlevqa: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 16259–16273, 2024.

François Chollet. On the measure of intelligence, 2019. URL https://arxiv.org/abs/1911.01547. François Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, and Henry Pinkard. Arc-agi-2: A new challenge for

frontier ai reasoning systems, 2025. URL https://arxiv.org/abs/2505.11831.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 36:49250–49267, 2023.

François Fleuret, Ting Li, Charles Dubout, Emma K Wampler, Steven Yantis, and Donald Geman. Comparing machines and humans on a visual categorization test. Proceedings of the National Academy of Sciences, 108(43): 17621–17625, 2011.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pp. 148–166. Springer, 2024.

Giorgio Ganis and Haline E Schendan. Visual imagery. Wiley Interdisciplinary Reviews: Cognitive Science, 2(3): 239–252, 2011.

Dedre Gentner. Structure-mapping: A theoretical framework for analogy. Cognitive Science, 7(2):155–170, 1983.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14375–14385, 2024.

Qing Guo, Prashan Wanigasekara, Jian Zheng, Jacob Zhiyuan Fang, Xinwei Deng, and Chenyang Tao. How do large multimodal models really fare in classical vision few-shot challenges? a deep dive. In R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models, 2023.

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14953–14962, 2023.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark, 2025. URL https: //arxiv.org/abs/2501.05444.

Sheng Hu, Yuqing Ma, Xianglong Liu, Yanlu Wei, and Shihao Bai. Stratified rule-aware network for abstract visual reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 1567–1574, 2021.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024.

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. Advances in Neural Information Processing Systems, 36:72096–72109, 2023.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6700–6709, 2019.

Amirmohammad Izadi, Mohammad Ali Banayeeanzade, Fatemeh Askari, Ali Rahimiakbar, Mohammad Mahdi Vahedi, Hosein Hasani, and Mahdieh Soleymani Baghshah. Visual structures helps visual reasoning: Addressing the binding problem in vlms, 2025. URL https://arxiv.org/abs/2506.22146.

Susanne M Jaeggi, Barbara Studer-Luethi, Martin Buschkuehl, Yi-Fen Su, John Jonides, and Walter J Perrig. The relationship between n-back performance and matrix reasoning—implications for training and transfer. Intelligence, 38(6):625–635, 2010.

Huaizu Jiang, Xiaojian Ma, Weili Nie, Zhiding Yu, Yuke Zhu, and Anima Anandkumar. Bongard-HOI: Benchmarking few-shot visual reasoning for human-object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19056–19065, 2022.

John and Jean Raven. Raven progressive matrices. In Handbook of nonverbal assessment, pp. 223–237. Springer, 2003. Alan S Kaufman, Susan Engi Raiford, and Diane L Coalson. Intelligent testing with the WISC-V. John Wiley & Sons,

2015. Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European Conference on Computer Vision, pp. 235–251. Springer, 2016.

Salem Lahlou, Tristan Deleu, Pablo Lemos, Dinghuai Zhang, Alexandra Volokhova, Alex Hernández-Garcıa, Léna Néhale Ezzine, Yoshua Bengio, and Nikolay Malkin. A theory of continuous generative flow networks. In International Conference on Machine Learning, pp. 18269–18300. PMLR, 2023.

Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. Behavioral and Brain Sciences, 40:e253, 2017.

Solim LeGris, Wai Keen Vong, Brenden M. Lake, and Todd M. Gureckis. H-arc: A robust estimate of human performance on the abstraction and reasoning corpus benchmark, 2024. URL https://arxiv.org/abs/2409.01374.

Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, Jialian Wu, Xiaodong Yu, Hao Chen, Emad Barsoum, Muhao Chen, and Zicheng Liu. Latent visual reasoning, 2025a. URL https://arxiv.org/abs/2509.24251.

Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vulić, and Furu Wei. Imagine while reasoning in space: Multimodal visualization-of-thought, 2025b. URL https://arxiv.org/abs/2501.07542.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, pp. 19730–19742. PMLR, 2023.

Kelvin Li, Chuyi Shang, Leonid Karlinsky, Rogerio Feris, Trevor Darrell, and Roei Herzig. Latent implicit visual reasoning, 2025c. URL https://arxiv.org/abs/2512.21218.

Kevin Y. Li, Sachin Goyal, Joao D. Semedo, and J. Zico Kolter. Inference optimal vlms need fewer visual tokens and more parameters, 2025d. URL https://arxiv.org/abs/2411.03312.

Linjie Li, Mahtab Bigverdi, Jiawei Gu, Zixian Ma, Yinuo Yang, Ziang Li, Yejin Choi, and Ranjay Krishna. Unfolding spatial cognition: Evaluating multimodal models on visual simulations, 2025e. URL https://arxiv.org/abs/2506. 04633.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pp. 34892–34916, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26296–26306, 2024.

Zhining Liu, Ziyi Chen, Hui Liu, Chen Luo, Xianfeng Tang, Suhang Wang, Joy Zeng, Zhenwei Dai, Zhan Shi, Tianxin Wei, Benoit Dumoulin, and Hanghang Tong. Seeing but not believing: Probing the disconnect between visual attention and answer correctness in vlms, 2025. URL https://arxiv.org/abs/2510.17771.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts, 2024. URL https://arxiv.org/abs/2310.02255.

Kanika Madan, Jarrid Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Cristian Nica, Tom Bosc, Yoshua Bengio, and Nikolay Malkin. Learning gflownets from partial episodes for improved convergence and stability. In International Conference on Machine Learning, pp. 23467–23483. PMLR, 2023.

Mikołaj Małkiński and Jacek Mańdziuk. One self-configurable model to solve many abstract visual reasoning problems. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 14297–14305, 2024.

Mikołaj Małkiński and Jacek Mańdziuk. Deep learning methods for abstract visual reasoning: A survey on raven’s progressive matrices. ACM Computing Surveys, 57(7):1–36, 2025.

Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 2263–2279, 2022.

Mikołaj Małkiński and Jacek Mańdziuk. A-i-raven and i-raven-mesh: Two new benchmarks for abstract visual reasoning, 2025. URL https://arxiv.org/abs/2406.11061.

Mikołaj Małkiński, Szymon Pawlonka, and Jacek Mańdziuk. Reasoning limitations of multimodal large language models. a case study of bongard problems, 2025. URL https://arxiv.org/abs/2411.01173.

Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. Online, 2025. URL

https://ai.meta.com/blog/llama-4-multimodal-intelligence/. MistralAI. Introducing mistral 3. Online, 2025. URL https://mistral.ai/news/mistral-3. Weili Nie, Zhiding Yu, Lei Mao, Ankit B. Patel, Yuke Zhu, and Animashree Anandkumar. Bongard-LOGO: A new

benchmark for human-level concept learning and reasoning. In Advances in Neural Information Processing Systems, volume 33, pp. 16468–16480, 2020.

OpenAI. Introducing gpt-5.2. Online, 2025a. URL https://openai.com/index/introducing-gpt-5-2/. OpenAI. Introducing gpt-o3 and gpt-o4-mini. Online, 2025b. URL https://openai.com/index/

introducing-o3-and-o4-mini/. Szymon Pawlonka, Mikołaj Małkiński, and Jacek Mańdziuk. Bongard-rwr+: Real-world representations of fine-grained concepts in bongard problems, 2025. URL https://arxiv.org/abs/2508.12026.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pp. 8748–8763. PmLR, 2021.

James C Raven. Mental tests used in genetic studies: The performance of related individuals on tests mainly educative and mainly reproductive. Master’s thesis, University of London, 1936.

John C Raven and John Hugh Court. Raven’s progressive matrices and vocabulary scales. Oxford pyschologists Press

Oxford, England, 1998. Eleanor H Rosch. Natural categories. Cognitive Psychology, 4(3):328–350, 1973. Timothy A Salthouse. Influence of working memory on adult age differences in matrix reasoning. British Journal of

Psychology, 84(2):171–199, 1993. Luca M Schulze Buschoff, Elif Akata, Matthias Bethge, and Eric Schulz. Visual cognition in multimodal large language models. Nature Machine Intelligence, 7(1):96–106, 2025.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning, 2024. URL https://arxiv.org/abs/2403.16999.

Guohao Sun, Hang Hua, Jian Wang, Jiebo Luo, Sohail Dianat, Majid Rabbani, Raghuveer Rao, and Zhiqiang Tao. Latent chain-of-thought for visual reasoning, 2025. URL https://arxiv.org/abs/2510.23925.

Younes Adam Tabi, Maria Raquel Maio, Bahaaeddin Attaallah, Shannon Dickson, Daniel Drew, Mohamad Imran Idris, Annika Kienast, Verena Klar, Lisa Nobis, Olivia Plant, et al. Vividness of visual imagery questionnaire scores and their relationship to visual short-term memory performance. Cortex, 146:186–199, 2022.

Gemma Team. Gemma 3 technical report, 2025. URL https://arxiv.org/abs/2503.19786. Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the

visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9568–9578, June 2024.

Shimon Ullman. Visual routines. Cognition, 18(1):97–159, 1984. ISSN 0010-0277. doi: https://doi.org/10.1016/ 0010-0277(84)90023-4. URL https://www.sciencedirect.com/science/article/pii/0010027784900234.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv:2409.12191, 2024a.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2024b. URL https://arxiv.org/abs/2311.03079.

David Wechsler. Wechsler intelligence scale for children. Psychological corporation, 1949. David Wechsler. WAIS-IV Administration and Scoring Manual. PsychCorp, 2008. URL https://books.google.co.il/

books?id=Bf-DswEACAAJ.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

Max Wertheimer. Untersuchungen zur Lehre von der Gestalt. II. Psychologische Forschung, 4(1):301–350, January

1923. ISSN 1430-2772. doi: 10.1007/BF00410640. URL https://doi.org/10.1007/BF00410640. Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13084–13094, 2024.

Rujie Wu, Xiaojian Ma, Zhenliang Zhang, Wei Wang, Qing Li, Song-Chun Zhu, and Yizhou Wang. BongardOpenWorld: Few-shot reasoning for free-form visual concepts in the real world. In International Conference on Learning Representations, 2024.

xAI. Grok-1.5 vision preview, 2024. URL https://x.ai/blog/grok-1.5v. xAI. Grok 4 fast model card. System card (technical report), September 2025. URL https://data.x.ai/

2025-09-19-grok-4-fast-model-card.pdf. Siying Xie, Daniel Kaiser, and Radoslaw M Cichy. Visual imagery and perception share neural representations in the alpha frequency band. Current Biology, 30(13):2621–2627, 2020.

Jingyi Xu, Tushar Vaidya, Yufei Wu, Saket Chandra, Zhangsheng Lai, and Kai Fong Ernest Chong. Abstract visual reasoning: An algebraic approach for solving raven’s progressive matrices. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6715–6724, 2023.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10632–10643, 2025a.

Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video, 2025b. URL https://arxiv.org/abs/ 2511.04670.

Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, and Chuang Gan. Machine mental imagery: Empower multimodal reasoning with latent visual tokens, 2025c. URL https://arxiv.org/abs/2506.17218.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality, 2024. URL https://arxiv.org/abs/2304.14178.

Eunice Yiu, Maan Qraitem, Anisa Noor Majhi, Charlie Wong, Yutong Bai, Shiry Ginosar, Alison Gopnik, and Kate Saenko. KiVA: Kid-inspired visual analogies for testing large multimodal models. In International Conference on Learning Representations, 2025.

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it?, 2023. URL https://arxiv.org/abs/2210.01936.

Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6720–6731, 2019.

Aimen Zerroug, Mohit Vaishnav, Julien Colin, Sebastian Musslick, and Thomas Serre. A benchmark for compositional visual reasoning. In Advances in Neural Information Processing Systems, volume 35, pp. 29776–29788, 2022.

Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. RAVEN: A dataset for relational and analogical visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5317–5327, 2019.

Dinghuai Zhang, Nikolay Malkin, Zhen Liu, Alexandra Volokhova, Aaron Courville, and Yoshua Bengio. Generative flow networks for discrete probabilistic modeling, 2022. URL https://arxiv.org/abs/2202.01361.

Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, and Dong Yu. Mm-llms: Recent advances in multimodal large language models, 2024a. URL https://arxiv.org/abs/2401.13601.

Huanyu Zhang, Wenshan Wu, Chengzu Li, Ning Shang, Yan Xia, Yangyu Huang, Yifan Zhang, Li Dong, Zhang Zhang, Liang Wang, Tieniu Tan, and Furu Wei. Latent sketchpad: Sketching visual thoughts to elicit multimodal reasoning in mllms, 2025. URL https://arxiv.org/abs/2510.24514.

Yizhe Zhang, He Bai, Ruixiang Zhang, Jiatao Gu, Shuangfei Zhai, Josh Susskind, and Navdeep Jaitly. How far are we from intelligent visual deductive reasoning?, 2024b. URL https://arxiv.org/abs/2403.04732.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models, 2024c. URL https://arxiv.org/abs/2302.00923.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023. URL https://arxiv.org/abs/2304.10592.

Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, et al. A survey on latent reasoning, 2025. URL https://arxiv.org/abs/2507.06203.

