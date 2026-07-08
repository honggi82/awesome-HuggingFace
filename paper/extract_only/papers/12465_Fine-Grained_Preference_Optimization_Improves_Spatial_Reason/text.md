# arXiv:2506.21656v3[cs.CV]5Jan2026

[Figure 1]

## Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs

Yifan Shen1, Yuanzhe Liu2, Jingyuan Zhu2, Xu Cao1, Xiaofeng Zhang3, Yixiao He1, Wenming Ye4, James M. Rehg1, Ismini Lourentzou1 1University of Illinois Urbana-Champaign 2University of Pennsylvania

3Shanghai Jiao Tong University 4Google {yifan26,lourent2}@illinois.edu https://plan-lab.github.io/spatialreasoner

[Figure 2]

[Figure 3]

What is the distance between and ?

What is the vertical distance height between and ?

[Figure 4]

[Figure 5]

[Figure 6]

- Step 1: Select the yellow car, which is ,and adopt its average height of approximately 1.5 meters.
- Step 2: Using the height of the car to estimate the vertical distance, the distance is estimated to be about 5 car

heights.

- Step 3: Compute the vertical distance between and using this scale: 5 × 1.5m = 7.5 m

- Step 1: Define the average length of , and it is about 12 meters.
- Step 2: Visually assess that the gap between and spans about 2.2 times length of .
- Step 3: Calculate the distance by multiplying the length of : 2.2 × 12m

= 26.4m

[Figure 7]

How much higher is the highest point of than the highest point of ?

[Figure 8]

From the perspective of ,

[Figure 9]

at which clock-face directions are ?

[Figure 10]

- Step 1: Set ground level of as zero.
- Step 2: According to the standard billboard next to road, the height of

is about 3 meters.

- Step 3: The height of billboard is about the height of 5 adults, about 8.5 meters.
- Step 4: Calculate the height difference between and : 8.5m − 3m = 5.5m

- Step 1: Take the front of car as the forward direction, which is 12 o'clock.
- Step 2: Car is to the left of car and also slightly forward. This means it is in the quadrant between the 9 o'clock and 12 o'clock positions.
- Step 3: After careful analysis, Car should be at the 11 o'clock position of Car .

[Figure 11]

Impact of fDPO on Spatial Quantity Estimation

[Figure 12]

SpatialReasoner-R1-8B DPO SpatialReasoner-R1-8B fDPO

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Below/Above

DirectDistance

Accuracy(%)

Tall/Short

Width

| | | |
|---|---|---|
| | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

Gemini 2.0 Flash Llama 4 Maverick InternVL2.5-78B Width Height SpatialRGPT-8B SpatialReasoner-R1-8B

Direct Distance

Horizontal

Vertical

Distance

Distance

### Abstract

Current Vision-Language Models (VLMs) struggle with fine-grained spatial reasoning, particularly when multi-step logic and precise spatial alignment are required. In this work, we introduce SpatialReasoner-R1, a vision-language reasoning model designed to address these limitations. To construct high-quality supervision for spatial reasoning, we design a Multi-Model Monte Carlo Tree Search (M3CTS) method that generates diverse, logically consistent Long Chain-of-Thought (LongCoT) reasoning trajectories. In addition, we propose a fine-grained Direct Preference Optimization (fDPO) method that introduces segment-specific preference

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

granularity for descriptive grounding and logical reasoning, guided by a spatial reward mechanism that evaluates candidate responses based on visual consistency, spatial grounding, and logical coherence. Experimental results demonstrate that fDPO achieves relative performance gains of 4.1% and 9.0% over standard DPO on spatial qualitative and quantitative tasks, respectively. SpatialReasoner-R1, trained with fDPO, sets a new SoTA on SPATIALRGPT-BENCH, outperforming the strongest baseline by 9.4% in average accuracy, while maintaining competitive performance on general vision-language tasks.

### 1 Introduction

Vision-Language Models (VLMs) have demonstrated significant advancements in multimodal understanding tasks, such as image captioning, visual question answering, object detection, and video interpretation [2, 31, 53, 62, 95]. However, their ability to perform spatial reasoning remains limited, especially in scenarios involving complex object arrangements and occlusions [7, 13, 18, 81]. This gap poses a significant challenge for applications such as robotics, autonomous driving, and augmented reality, where robust spatial understanding is essential for effective decision-making [50].

Historically, early VLMs predominantly employed direct-response paradigms [2, 53], i.e., producing immediate answers without explicit reasoning, which often leads to shallow understanding. Recent advances in Chain-of-Thought (CoT) prompting have introduced step-by-step reasoning [72], but standard CoT traces are often too brief or abstract to capture fine-grained spatial logic. In contrast, Long Chain-of-Thought (LongCoT) prompting produces richer, more interpretable reasoning paths that better support comprehension [10, 43, 73]. Still, such prompting must go beyond simple depth estimation, as accurate spatial reasoning requires understanding occlusions, relative orientations, and positional ambiguity, all of which are difficult to capture without structured, fine-grained supervision.

To address these challenges, we introduce SpatialReasoner-R1, a novel VLM designed to perform spatial reasoning directly from 2D images. SpatialReasoner-R1 employs structured, interpretable LongCoT reasoning to systematically parse and solve spatial queries without relying on additional modalities or external sensor data. To optimize the training process for multi-step reasoning, we introduce a new fine-grained Direct Preference Optimization (fDPO) method that applies differentiated learning updates tailored to two semantically distinct components, descriptive grounding and logical reasoning. Unlike traditional DPO, fDPO introduces segment-specific preference granularity, allowing SpatialReasoner-R1 to adjust its optimization for each generation phase, emphasizing spatial localization during descriptive grounding and enhancing multi-step logical inferences during reasoning.

To curate diverse high-quality spatial reasoning data for training, we propose a Multi-Model Monte Carlo Tree Search (M3CTS) that generates high-quality LongCoT responses by leveraging collaborative exploration across multiple VLMs, and a fine-grained spatial reward mechanism that evaluates candidate responses across three dimensions: descriptive accuracy, spatial grounding precision, and logical coherence, which are then used to construct positive and negative sample pairs for DPO and fDPO training. Empirical results across several challenging spatial reasoning tasks demonstrate that SpatialReasoner-R1 achieves state-of-the-art performance, significantly outperforming existing VLMs and CoT-based methods, particularly on complex, multi-step spatial reasoning. Specifically, SpatialReasoner-R1 surpasses the best baseline by 9.4% in average accuracy on spatial understanding. Our fDPO improves by 4.1% and 9.0% on average over standard DPO on spatial qualitative and quantitative tasks, respectively. Our contributions are as follows:

- (1) We introduce SpatialReasoner-R1, a LongCoT spatial reasoning VLM that effectively generates interpretable, step-by-step explanations directly from 2D images. SpatialReasonerR1 establishes a new SoTA in spatial understanding, while maintaining robust performance on general vision-language benchmarks.
- (2) To enhance training stability and precision, we propose a new fine-grained Direct Preference Optimization (fDPO) method that employs segment-specific learning updates tailored explicitly for descriptive grounding and logical reasoning.
- (3) To address the scarcity of high-quality spatial reasoning data, we introduce a data generation pipeline that combines Multi-Model Monte Carlo Tree Search (M3CTS) with fine-grained spatial rewards, enabling the creation of diverse, logically consistent LongCoT trajectories for fine-grained preference training.

### 2 Related Work

Vision Language Models and Spatial Reasoning. Recent advances in VLMs have significantly enhanced the ability of multimodal models to understand and generate descriptive text grounded in visual contexts [31, 39, 40, 49, 66, 95]. Models such as Flamingo [1], BLIP-2 [32], and QwenVL [39] use high-capacity vision encoders [53] paired with LLMs [5, 64] to achieve state-of-the-art performance in various multimodal tasks, such as visual question answering, image captioning, and instruction following [2, 15, 37, 65, 76, 100]. Current trends involve scaling models to improve general understanding [12, 25, 62] and using large-scale instruction tuning datasets [40, 56, 93]. Both proprietary [21, 26, 25] and open-source VLMs [12, 17, 89] have shown impressive results.

While VLMs show promise in visual understanding, accurately perceiving and reasoning about spatial arrangements remains a challenge [13]. Recent efforts to improve spatial understanding include fine-tuning VLMs on spatial VQA datasets [7, 8, 13, 41, 75, 28, 55], and zero-shot frameworks that leverage external 3D foundation models for geometric priors [44]. Region-aware models have also been proposed for better grounding and finer spatial queries [24, 87, 91]. These advances extend to scenarios such as video understanding [81] and 3D generation [46, 50]. To track progress, specialized benchmarks like Q-Spatial Bench [36], SpatialRGPT-Bench [13], VSI-Bench [81], and 3DSRBench [45] have been introduced to assess spatial skills. However, current models still struggle with complex, multi-step spatial reasoning. SpatialReasoner-R1 addresses this gap by introducing fine-grained preference optimization and multi-level reward mechanisms.

Aligning VLMs using Preference Optimization. Preference-based learning methods, particularly DPO [54], have become standard techniques for aligning models with human intentions. These methods bypass the need for explicit reward model training and have often demonstrated strong performance compared to earlier Reinforcement Learning with Human Feedback (RLHF) approaches [3, 19, 48, 98]. In the multimodal domain, DPO and its variants have been adapted to address specific challenges such as reducing hallucinations and improving visual grounding [70, 78, 88]. The adaptability of DPO is further highlighted by its recent application in aligning generative models beyond language, such as text-to-image diffusion models [22, 33, 67, 82, 90]. Adaptation methods often involve constructing preference pairs based on human corrections, AI feedback, or contrasting inputs to guide the model towards desired behaviors [11, 14, 18, 20, 61, 68, 74, 77, 79, 85].

Standard DPO methods treat the reasoning process as a single structure. To address this, preference granularity in DPO has been explored at the token [38, 57, 94, 97, 99], step [29, 96], sentence [51, 54, 58], and turn [59, 60, 80] levels. While effective in certain domains, these approaches overlook the semantic roles of different segments in LongCoT, where descriptive grounding and logical reasoning require distinct optimization. In contrast, our proposed fDPO introduces functional-level preference granularity.

Multi-LLM Guided Reasoning Recent work has explored leveraging multiple LLMs to collaboratively solve complex reasoning tasks, often integrated with Monte Carlo Tree Search (MCTS). Methods such as MoA [69], MoSA [84], AlphaLLM-CPL [71], and LE-MCTS [52] enhance multiagent text-based reasoning using ensemble methods and stepwise search. CoMCTS (Mulberry) [86] extends multi-LLM MCTS to multimodal reasoning, primarily targeting collaborative reflection and error correction. In contrast, our method, M3CTS, addresses the challenge of spatial reasoning in VLMs, introducing fine-grained preference learning and multi-level spatial rewards that allow for coherent, visually-grounded reasoning paths across multimodal data.

### 3 Method

#### 3.1 Spatial Reasoning from Images

Spatial reasoning is a core vision-language challenge, requiring models to understand visual layouts and perform logical inference over spatial relationships. We define spatial reasoning as a multimodal understanding problem where the goal is to generate accurate reasoning paths based on visual and textual inputs. Formally, a spatial reasoning instance can be represented as a tuple T =(I,Q,P) −→πθ R where I represents the input image containing the visual content, Q is the textual query specifying the spatial reasoning task, P denotes the visual prompt tokens pointing to a specific object or region in the image, and R the textual response, providing the answer or step-by-step reasoning path. The

Stage 1. Generating LongCoT based on M3CTS

[Figure 17]

[Figure 18]

[Figure 19]

<description>

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Details the input context, such as the visual scene

[Figure 29]

Q M3CTS

[Figure 30]

or scenario description. </description> <reasoning> Let's think step by step: ### Step 1, <region1> is located at ... ### Step2, <region2> is located at ...

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

LongCoT

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Generation

</reasoning>

[Figure 61]

[Figure 62]

[Figure 63]

Preference Optimization

[Figure 64]

Description Reasoning

[Figure 65]

Stage 1. Generating LongCoT based on M3CTS

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Positive Response

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

<description>

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Details the input context, such as the visual scene

[Figure 89]

Q M3CTS

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

or scenario description. </description> <reasoning> Lets think step by step: ### Step 1, <region1> is located at ... ### Step2, <region2> is located at ...

Responses Generation Reward

Description Reasoning

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Negative Response

[Figure 105]

[Figure 106]

LongCoT

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Stage 2. Preference Pair Construction Stage 3. Fine-grained Direct Preference Optimization

[Figure 123]

Generation

</reasoning>

[Figure 124]

[Figure 125]

[Figure 126]

Preference Optimization

Figure 1: Method Overview. To train SpatialReasoner-R1, we (1) generate reasoning paths using M3CTS, (2) construct fine-grained preference pairs via reward-based selection, and (3) train with fine-grained DPO (fDPO) to optimize descriptive and logical reasoning separately.

[Figure 127]

Description Reasoning

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Positive Response

[Figure 139]

[Figure 140]

Image

Image

[Figure 141]

[Figure 142]

[Figure 143]

Responses Generation Reward

Description Reasoning

Encoder

Embd

Response

Positive Response

[Figure 144]

Image

primary objective of a spatial reasoning model, denoted as πθ, is to map the multimodal input T to a logically sound and spatially grounded response R.

Stage 2. Preference Pair Construction Stage 3. fine-grained Direct Preference Optimization

Prompt

Prompt Encoder

Embd

Visual Prompt (Masks)

[Figure 145]

Unlike typical direct-response VQA tasks, SpatialReasoner-

Text

Text

Text

Encoder

Embd

Decoder

[Figure 146]

| |
|---|

- R1 is designed to output LongCoT reasoning traces that decompose spatial reasoning into clear, verifiable steps. To train the model, we introduce a finegrained preference objective that optimizes descriptive and reasoning responses separately (§3.2) and a spatial reward mechanism that evaluates candidate reasoning paths based on spatial and logical understanding (§3.3). Finally, to address the lack of LongCoT supervision for spatial reasoning, we propose a multi-model collaborative tree search method that generates diverse, reward-aligned reasoning trajectories to enable preference-based training (§3.4). An overview of the proposed training framework is depicted in Figure 1 and the SpatialReasoner-R1 architecture is shown in Figure 2.

Image

Image

Text

Encoder

Embd

Response

[Figure 147]

Image

Prompt

Prompt Encoder

Embd

Visual Prompt (Masks)

[Figure 148]

Text

Text

Text

Encoder

Embd

Decoder

Text

Figure 2: Architecture Overview. SpatialReasoner-R1 is a VLM that takes as input a text instruction, visual prompts, and an image, and generates LongCoT reasoning responses.

#### 3.2 Fine-grained Direct Preference Optimization (fDPO)

We propose fDPO as a novel fine-grained off-policy preference learning algorithm to optimize LongCoT spatial reasoning. Traditional DPO methods apply a single global trade-off parameter β uniformly across all reasoning steps [68, 74, 85], implicitly treating all response segments as equally learnable. However, this can lead to degenerate solutions, as the model may overfit to simpler descriptive responses while under-optimizing the more complex reasoning paths. This observation motivates the design of our fine-grained preference mechanism, which introduces segment-level preference granularity.

To facilitate fine-grained preference optimization, we first segment each LongCoT response R into its constituent description Rdesc and reasoning Rreason components, represented as R=[Rdesc,Rreason]. We then quantify the preference signal for each segment by calculating the score difference between the corresponding segments derived from the positive Rp and negative responses Rl, yielding segment-wise preference differentials:

∆Rdesc = score Rdescp − score Rdescl , ∆Rreason = score Rreasonp − score Rreasonl , (1)

where the differentials ∆Rdesc and ∆Rreason quantify the preference margin for description and reasoning segments based on the preference pair, and score(·) composite scores are introduced in Section 3.3. The design of fDPO is guided by two key principles:

- Principle 1: Preference optimization strength should be dynamically balanced according to the intrinsic complexity and quality disparity between description and reasoning components.

Our analysis of the fine-grained reward signals reveals that descriptive segments (Rdesc) are easier to optimize while models struggle with reasoning segments (Rreason) that are typically longer and require multi-hop logic. Thus, a unified optimization parameter β may lead to reasoning under-optimization. To address this, fDPO introduces separate, adaptively-tuned trade-off parameters, βdesc and βreason, which dynamically control the learning signals for each segment independently, to allow the model to prioritize deeper logical inference while maintaining visual and attribute accuracy.

- Principle 2: The choice of segment-specific optimization parameters (βdesc and βreason) should prioritize the component that exhibits a larger preference differential, such that learning focuses on harder-to-learn segments.

We further empirically observe that the preference score differential for descriptive components ∆Rdesc is consistently smaller than ∆Rreason. To account for this, fDPO computes dynamic segment weights wdesc and wreason to adaptively adjust the learning signals for the description and reasoning components, respectively:

exp λ · ∆Rs exp λ · ∆Rdesc + exp λ · ∆Rreason

, s ∈ {desc,reason}, (2)

ws =

where ∆Rs is the preference score differential for segment s (either description or reasoning), λ > 0 controls the sensitivity of weights, and {wdesc,wreason} reflect the relative importance of each segment.

These weights are then mapped to adjustment factors centered around 1 and applied to the base optimization parameter β to yield segment-specific trade-off parameters βdesc and βreason for description and reasoning, respectively:

βs = β × f(ws) = β 1 + α (2ws − 1) , s ∈ {desc,reason}, (3)

where ws ∈ [0,1] is the respective segment-specific weight (either description or reasoning), α is a hyperparameter that controls the maximum scaling amplitude, and β is the base hyperparameter value. This design implements a dynamic learning strategy: segments with larger preference differentials (higher relative importance ws) receive a higher effective βs, amplifying the learning signal and prioritizing those components. Conversely, smaller preference differentials yield lower βs, enabling finer-grained updates. This adaptive mechanism allows fDPO to balance optimization based on segment-specific learning difficulty, promoting better alignment for complex reasoning steps while preserving descriptive accuracy. The final optimization objective for each segment is defined as

πθ Rsl | T πref Rsl | T

πθ Rsp | T πref Rsp | T

Fs(T ,Rp,Rl) = log

, s ∈ {desc, reason}. (4)

− log

Here, T represents the multimodal input (image, text query, visual prompt), πθ is the model’s learned policy, and Fs measures the segment-specific preference margin in log-likelihood ratios relative to a reference policy πref. The overall optimization objective for fDPO is

##### LfDPO(θ) = −E(T ,Rp,Rl)∼D log σ βdesc Fdesc(T ,Rp,Rl) + βreason Freason(T ,Rp,Rl) , (5)

where D={(T (i),R(p,i),R(l,i))}Ni=1 and σ(·) sigmoid activation function.

#### 3.3 Fine-Grained Spatial Rewards

To optimize spatial reasoning paths effectively, we introduce a fine-grained reward mechanism that evaluates candidate reasoning paths across visual, spatial, and logical dimensions. Rewards capture alignment with image content, spatial relationships, and logical inference. Figure 3 illustrates the proposed fine-grained spatial rewards for fDPO. Specifically, we define four scalar rewards; details about their formulations and rationale behind each reward are provided in Appendix A.

⋄ Visual Consistency Reward (Rvc) evaluates the description Rdesc to ensure spatial grounding and fidelity. The reward verifies key aspects of the quality and alignment of the description with the visual scene, such as whether all referenced objects are present and identifiable, whether the

###### Resp1 Resp2

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

The image shows a modern kitchen island with a dark gray base cabinet. The question

asks for the height of this kitchen island. Let's think step by step. Step 1: The

Resp1

cabinet looks slightly lower than the kitchen counter behind it. Step 2: Standard kitchen is about 90 centimeters tall. Step 3: Thus the height of the kitchen island is

Question: What is the

[Figure 156]

height of the kitchen island?

about 45 centimeters.

[Figure 157]

[Figure 158]

###### Reward Score Breakdown

Fine-grained Direct

[Figure 159]

Negative Sample [Descriptive Scoring]:3.4 [Spatial Scoring]:3.5 [Reasoning Scoring]:2.8

Preference

Optimization

Fine-grained

[Figure 160]

Spatial

Positive Sample [Descriptive Scoring]:3.8 [Spatial Scoring]:7.6 [Reasoning Scoring]:3.8

Reference

Reward

Model

[Figure 161]

[Figure 162]

[Figure 163]

The image shows a modern kitchen with a large island in the center. Two white bar

stools are positioned at the corner of the island. The question asks for the height of the dark gray cabinet of the kitchen island. Let's think step by step. Step 1: The

###### bar stools appear to be standard bar height, which is around 75 centimeters from

Resp2

Question: What is the height of the kitchen

the floor to the seat. Step 2: The top of the island appears to be about 20 centimeters higher than the seats of the bar stools. Step 3: Adding the bar stool

island?

[Figure 164]

height and the additional height above the stools: 75 + 20 = 95 cm.

- Figure 3: Fine-Grained Spatial Rewards. Candidate reasoning paths are decomposed into three aspects, descriptive, spatial, and reasoning, scored separately; the higher value in each row is marked by and the lower by . Explanation of Scoring: Descriptive: Negative response omits the two bar-stools and uses generic “modern kitchen” wording, whereas the positive response lists every salient object; Spatial: Negative response wrongly claims the island is lower than the rear counter and ignores the 20cm offset revealed by the stool reference, whereas the positive response provides its estimate to the 75cm stool height plus that offset; Reasoning: Negative response uses an illogical “half-height” heuristic 90cm → 45cm without intermediate computation, whereas the positive response explicitly adds reference height and gap (75cm + 20cm = 95cm). These per-category deficits yield lower composite reward, designating the upper response as negative sample.

[Figure 165]

[Figure 166]

stated properties (such as color, size, and shape) match the visual content, whether the description includes all necessary details prompted by the query, and whether it remains contextually appropriate and free from extraneous information.

⋄ Depth-Guided Spatial Reward (Rsp) measures fine-grained spatial understanding by leveraging depth information. This reward is independently computed for the description Rdesc and reasoning Rreason components, with two adaptive weighting mechanisms: an uncertainty weight that adjusts the score for spatial expressions with qualifiers (e.g., "approximately","possibly") to account for reduced confidence, and a context-aware weight that emphasizes spatial relations directly relevant to the query. The final spatial rewards are computed as the uncertainty and context-aware weighted correctness scores across all spatial assertions in the corresponding description and reasoning components, validated against both the RGB image and its corresponding depth map. This ensures that more confident and contextually aligned relations have stronger influence on the reward.

⋄ Logical Coherence Reward (Rlc) evaluates the reasoning Rreason for structural integrity and logical correctness. This reward captures multi-hop inference and factual alignment by verifying that premises are consistent with the image, depth map, and preceding descriptions, reasoning steps maintain spatial and causal logic, the application of physical, spatial, and logical principles remains coherent throughout, and the conclusion is fully supported by the reasoning chain.

The preference differentials in fDPO are computed from composite rewards that aggregate fine-grained evaluations across segments: score(Rdesc)=Rvc + Rsp,desc and score(Rreason)=Rlc + Rsp,reason.

#### 3.4 Multi-Model MCTS (M3CTS)

We introduce a Multi-Model Monte Carlo Tree Search (M3CTS) framework for generating highquality LongCoT data D={(T ,Rp,Rl)}Ni=1 tailored to spatial reasoning. Inspired by DeepSeekR1-Zero [16] and prior multimodal MCTS methods [86, 52, 84], M3CTS explores diverse reasoning trajectories across multiple VLMs to effectively search for logical, spatially-consistent explanations

The image shows a modern kitchen island with a dark gray base cabinet. The question asks for the height of this kitchen island. Let's think step by step. Step 1: The

cabinet appears to be about the same height as the kitchen counter behind it. Step

2: Based on common kitchen counter heights, I estimate the height of the island cabinet is around 90 centimeters. Step 3: Thus the height of the kitchen island is

also around 90 centimeters.

that satisfy the query. Formally, the reasoning process is defined as a sequence of reasoning states:

- S = {s0,...,st,...,sT}, where st ∈ S represents a partial reasoning state, s0 is the initial state derived from T , and sT is a terminal state corresponding to a fully reasoned path. M3CTS operates through four key stages: Expand, Simulate, Backprop, and Select.

Expand. At each step t, M3CTS expands the current state st by generating diverse candidate reasoning states Sc using multiple VLMs {πk}Kk=1 concurrently, i.e.,

K

πk st | T ,Parent(st) , (6)

Sc =

k=1

where πk is the k-th VLM, T multimodal input, and Parent(st) ancestor reasoning states of st. To ensure consistency, we enforce a structured output format across all VLMs (Appendix B).

Simulate. Each candidate sk,t ∈ Sc generated during expansion is evaluated based on three distinct criteria: (i) visual description accuracy against the original image, (ii) spatial correctness of inferred spatial relationships utilizing both original and depth-derived images, and (iii) logical coherence of the textual reasoning steps. The evaluation score R(sk,t) is computed as:

M

1 M

R(sk,t) =

m=1

I(visualm) (sk,t) + I(spatialm) (sk,t) + I(logicalm) (sk,t) , (7)

where M is the number of evaluation models, and each I(evalm)(sk,t) indicator function defined as:

I(evalm)(sk,t) = {+1 (fully accurate), 0 (neutral), −1 (inaccurate)}. (8)

We preserve high-quality paths by pruning the candidate set according to the evaluation score, i.e., Sc∗={sk,t | R(sk,t) ≥ 0}. Appendix C provides detailed descriptions of the evaluation.

Backprop. To perform credit assignment, scores from the simulation phase are recursively propagated upwards through the search tree. The objective is to update the value estimates V (sk,t) and visit counts N(sk,t) for each parent node sk,t based on the performance of its children Sc∗=Child(sk,t),

N(sk,t)V (sk,t) + s

N(sc)R(sc) N(sk,t) + s

c∈Sc∗

N(sc). (9)

V (sk,t) ←

, N(sk,t) ← N(sk,t) +

N(sc)

c∈Sc∗

sc∈Sc∗

Select. This phase is responsible for choosing the most promising candidate state for further exploration in the next iteration of tree expansion. We use the Upper Confidence Bound (UCB) strategy to select the next state s⋆k′,t+1 to traverse, based on updated values and visitation statistics. UCB ensures that high-value paths are prioritized, while also exploring less-visited nodes to discover new reasoning trajectories. The candidate selected maximizes the UCB objective, i.e.,

log N(sk,t) 1 + N(sc)

s⋆k′,t+1 = arg max

, (10)

V (sc) + α˜

sc∈Sc∗

where V (sc) is the value estimate of the candidate state sc, N(sc) its visit count, and α˜ > 0 is a hyperparameter that balances exploration vs. exploitation.

### 4 Experiments

#### 4.1 Experimental Setup

We evaluate SpatialReasoner-R1 across diverse spatial reasoning and general vision-language established benchmarks to assess the model’s fine-grained spatial understanding and logical reasoning capabilities. Implementation details are provided in Appendix D.

Spatial Reasoning Benchmarks. Our primary benchmark is SPATIALRGPT-BENCH [13], comprising image-based spatial reasoning questions and their corresponding ground truth answers. Detailed descriptions of benchmarks and evaluation protocols are provided in Appendix E.

Table 1: Spatial Reasoning Success Rates (↑) on SPATIALRGPT-BENCH. Classification (top) and numeric distance/direction (bottom). are General Large VLMs, are Customized VLMs, are SpatialReasoner-R1 variants. “/” indicates the model refuses to provide a response for that metric.

Below/ Above

Left/ Right

Big/ Small

Tall/ Short

Wide/ Thin

Behind/ Front

Qual. Acc.

Gemini 2.0 Flash [21] 58.33 68.57 16.98 50.00 15.38 53.63 44.29 Llama 4 Maverick [47] 54.17 61.90 33.02 50.89 25.96 55.45 47.18

- Gemini 1.5 Pro [63] 85.83 56.19 58.49 71.42 55.76 60.00 65.14 ChatGPT-4o [25] 87.50 80.00 53.77 63.39 51.92 60.90 66.67

SpatialBot-3B [6] 52.50 62.86 57.54 49.11 49.04 62.73 55.56 SpaceThinker Qwen2.5VL-3B [4] 89.16 63.81 76.41 56.25 56.73 70.91 69.25 InternVL2.5-78B [12] 94.16 94.28 64.15 65.17 55.76 58.18 72.29 Sa2VA 4B [89] 22.50 25.71 25.47 16.07 27.88 30.91 24.65 Sa2VA 8B [89] 50.00 39.04 45.28 26.78 45.19 53.63 43.37 SpatialRGPT-8B [13] 99.17 100.00 84.90 89.28 91.34 90.90 92.69

SpatialReasoner-R1 SFT 4B 79.16 78.09 55.66 66.96 59.61 75.45 69.41 SpatialReasoner-R1 SFT 8B 81.66 81.90 75.47 75.89 79.80 83.63 79.75 SpatialReasoner-R1 DPO 4B 91.66 91.42 69.81 65.17 71.15 85.45 79.29 SpatialReasoner-R1 DPO 8B 94.16 93.33 89.62 90.18 88.64 92.27 91.48 SpatialReasoner-R1 fDPO 4B 95.83 93.33 83.96 74.10 87.50 89.09 87.37 SpatialReasoner-R1 fDPO 8B 98.33 98.10 95.28 96.43 91.34 93.64 95.59

Direct Distance

Horizontal Distance

Vertical Distance

Width Height Direction

Quan. Acc.

- Gemini 2.0 Flash [21] 9.45 10.65 26.41 10.52 30.82 54.20 22.43 Llama 4 Maverick [47] 24.48 28.68 34.28 35.71 44.61 58.09 36.72 Gemini 1.5 Pro [63] 14.18 17.21 14.15 19.54 36.09 30.84 21.90 ChatGPT-4o [25] / / / / / 60.75 /

SpatialBot-3B [6] 6.00 15.51 8.00 10.52 18.75 39.00 15.62 SpaceThinker Qwen2.5VL-3B [4] 24.32 17.21 59.43 23.27 23.62 32.35 28.97 InternVL2.5-78B [12] 27.70 22.13 41.50 29.32 34.58 62.61 35.25 Sa2VA 4B [89] 13.51 15.57 19.81 13.53 12.03 10.28 14.02 Sa2VA 8B [89] 14.18 14.75 9.43 14.28 19.54 14.18 14.55 SpatialRGPT-8B [13] 45.90 68.00 56.60 48.90 61.70 95.30 61.42

SpatialReasoner-R1 SFT 4B 22.29 27.86 31.13 25.56 33.80 47.66 30.71 SpatialReasoner-R1 SFT 8B 28.43 20.49 44.05 33.59 51.63 46.72 37.12 SpatialReasoner-R1 DPO 4B 47.97 46.72 60.37 45.11 55.63 91.58 56.61 SpatialReasoner-R1 DPO 8B 62.83 56.55 60.37 70.45 68.42 93.45 68.22 SpatialReasoner-R1 fDPO 4B 60.13 59.01 71.70 65.41 57.89 92.52 66.76 SpatialReasoner-R1 fDPO 8B 70.95 72.13 74.52 80.45 74.43 94.39 77.30

General Vision-Language Benchmarks. To validate the robustness of SpatialReasoner-R1 beyond purely spatial tasks, we evaluate on broader vision-language datasets such as MME, POPE, SEEDBENCH, AI2D, SQA-TEST, MMMUV, MMSTAR, and HALLUSIONBENCH [35, 34, 30, 27, 42, 92, 9, 23]. These datasets cover fundamental vision-language tasks such as object grounding, hierarchical scene parsing, multimodal understanding, and multi-turn reasoning in diverse multimodal contexts.

Baselines. We benchmark SpatialReasoner-R1 against two categories of baseline models: General Large VLMs. This includes powerful, widely-accessible models such as Gemini 2.0 Flash [21], Llama 4 Maverick [47], Gemini 1.5 Pro [63], and ChatGPT-4o [25]. These are evaluated in zero-shot/few-shot settings as reference of standard VLM capabilities w/o task-specific fine-tuning. Specialized VLMs. This baseline set comprises models specifically developed, adapted, or finetuned for spatial understanding tasks, allowing us to assess our contributions relative to other specialized approaches. The models included are: SpatialBot-3B [6], SpaceThinker Qwen2.5VL-3B [4], InternVL2.5-78B [12], Sa2VA (4B, 8B) [89], and SpatialRGPT-8B [13].

SpatialReasoner-R1 VLMs. We also include SpatialReasoner-R1 4B and 8B variants with different training strategies, such as SpatialReasoner-R1 SFT, SpatialReasoner-R1 DPO, trained with standard DPO, and SpatialReasoner-R1 fDPO trained with the proposed fine-grained DPO method.

#### Table 2: General Vision-Language Understanding Results. Best performance in bold.

Models MME POPE SEED-BENCH AI2D SQA-TEST MMMUV MMSTAR HALLUSIONBENCH

SpatialRGPT-8B [13] 1667/348 85.50 67.00 67.42 81.81 41.40 43.98 40.80 SpatialReasoner-R1 fDPO 8B 1667/503 89.71 76.21 78.85 93.85 48.11 55.43 51.10

###### Q: Can you give me an estimation of the horizontal distance between center of <Region1> and <Region 2>?

Ground Truth Response from SpatialRGPT 8B

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

The horizontal distance of Region 1 from Region 2 is 1.08 meters.

Response from InternVL2.5-78B

[Figure 171]

[Figure 172]

- Step 1: Estimate the width of Region 1 (fireplace) as 1.2 meter.
- Step 2: Estimate the width of Region 2 (TV cabinet) as 1.5 meters.
- Step 3: the distance between the fireplace and the TV stand seems to be about the width of one object (either the fireplace or the TV stand)

- Step 4: We can estimate that the total horizontal distance between the centers of Region1 and Region2 is approximately the sum of half the width of the fireplace, the distance between the objects, and half the width of the TV stand.

Center of <Region1> and <Region2> are 1.6 meters apart horizontally

Step 5: Distance = 1.2/2 + 1.5 + 1.5/2 = 2.85 meters

Response from Gemini 1.5 Pro Response from 8B

[Figure 173]

[Figure 174]

[Figure 175]

Step1: The right edge of the fireplace seems aligned with the left edge of the TV, and the TV is centered above the TV stand, which is slightly wider than the TV.

[Figure 176]

[Figure 177]

Step 1: Estimate the width of the fireplace. The fireplace looks like a standard size, about 1.2 meters wide.

- Step2: Let's assume the width of fireplace is 1 meter. The distance between the center of the fireplace and the left edge of the TV can be estimated as half the fireplace's width, which is 0.5 meters.

- Step3: Let's assume the width of the TV stand is about 1 meter. The distance from the left edge of the TV to the center of the TV stand can be estimated as 1/4 of the stand's width, which is 0.25 meters.

- Step 2: Estimate the width of the TV stand. The TV stand looks slightly wider than the TV, about 1.4 meters wide.

- Step 3: Estimate the gap between the fireplace and the TV stand. There's a small gap between the fireplace and the TV stand, which appears to be about 0.2 meters.

- Step 4: Add the estimated widths and the gap. 1.2m/2 (fireplace) + 0.2m (gap) + 1.4m/2 (TV) = 1.5 meters. So, the region1 and region2 are 1.5 meters apart horizontally.

Step4: Combining steps 2 and 3, the total horizontal distance is

roughly 0.75 meters.

- Figure 4: Qualitative Examples of Spatial Reasoning Across Models. SpatialReasoner-R1 demonstrates a coherent, multi-step logical chain that closely matches the ground truth, while other models exhibit less precise or less interpretable reasoning paths.

#### 4.2 Experimental Results

Spatial Reasoning. As shown in Table 1, SpatialReasoner-R1 models achieve substantial improvements over both general-purpose and spatial-specialized VLMs across all spatial tasks. Notably, SpatialReasoner-R1 fDPO 8B sets a new benchmark for average accuracy with 2.9% and 15.8% gains over SpatialRGPT-8B on spatial qualitative and quantitative tasks, respectively. Our parameter-efficient SpatialReasoner-R1 fDPO 4B outperforms larger models like InternVL2.5-78B, highlighting the effectiveness of our fine-tuning strategy. Finally, when compared to its predecessor DPO 8B, our optimized variant fDPO 8B boosts average accuracy by 4.1% across qualitative tasks and by 9.0% in quantitative tasks.

General Vision-Language Understanding. Beyond achieving state-of-the-art performance in spatial reasoning tasks, our SpatialReasoner-R1 fDPO 8B also demonstrates significant gains in general vision-language benchmarks compared to SpatialRGPT-8B, as presented in Table 2.

#### 4.3 Qualitative Examples

- Figure 4 provides qualitative examples that demonstrate SpatialReasoner-R1’s advanced capability for coherent, multi-step spatial reasoning. SpatialReasoner-R1 first estimates the fireplace and TV-stand

widths at 1.2m and 1.4m, then computes 12.2 +0.2(gap)+ 12.4 = 1.5m, a value that nearly matches the reference while transparently tying each term to an observed feature. In contrast, InternVL2.5-78B

adopts similar width guesses (1.2m, 1.5m) but assumes “the distance between the fireplace and the TV stand seems to be about the width of one object”. This assumption is inconsistent with what is shown in the image. Gemini 1.5 Pro aligns the fireplace’s right edge with the TV’s left edge, assigns both objects a 1m width, and combines only half a fireplace width (0.5m) with one quarter of the stand width (0.25m). These two estimates are not accurate and ignore the gap between the two regions. SpatialRGPT-8B yields a more accurate estimate than Gemini 1.5 Pro and InternVL2.5-78B. However, since it is not designed as a reasoning model, it cannot generate step-by-step reasoning

#### Table 3: Effect of Alpha (α).

Metric 10% 20% 30% 40% Direct Dist. 53.38 56.76 60.13 58.11 Horiz. Dist. 52.46 55.74 59.01 56.55 Vert. Dist. 65.09 67.92 71.75 69.81 Width 51.88 57.89 65.41 63.16 Height 56.39 57.14 57.89 58.64 Direction 91.58 92.23 92.52 94.39

#### Table 4: Effect of Lambda (λ) at α=30%.

Metric λ=0.2 λ=0.4 λ=0.6 λ=0.8 Direct Dist. 54.05 57.38 60.13 59.45 Horiz. Dist. 53.27 57.37 59.01 58.19 Vert. Dist. 65.09 68.86 71.75 69.81 Width 53.38 60.15 65.41 54.67 Height 56.39 57.14 57.89 57.89 Direction 91.58 91.58 92.53 93.45

traces, i.e., does not explicitly reveal the logical chain of spatial deductions or intermediate calculations leading to that estimate. These qualitative examples show that SpatialReasoner-R1 has more accurate spatial awareness. Additional examples can be found in Appendix F.

#### 4.4 Ablations

Table 3 illustrates the impact of varying the Alpha (α) parameter, which modulates the magnitude of segment-specific learning adjustments during fDPO optimization. When α is set too high, the model may overly focus on the reasoning part at the expense of the other, introducing instability and degraded performance, as observed when α reaches 40%. Conversely, if α is too low, both description and reasoning segments are optimized equally. A moderate value of α = 30% allows the model to effectively amplify learning signals for fine-grained spatial distinctions, leading to substantial improvements across all spatial metrics. Furthermore, Table 4 presents the impact of varying the Lambda (λ) parameter, which modulates the sensitivity of segment-specific weights to preference differentials, controlling how responsively the model shifts learning focus based on the observed preference margins. As λ increases, the model becomes more sensitive to segment-specific preference differences, leading to noticeable changes in performance across spatial metrics. We observe that while a moderate value of λ = 0.6 achieves the best overall results, setting λ too high can introduce slight performance degradation in some spatial metrics, likely due to overly aggressive re-weighting.

### 5 Conclusion

In this work, we introduce SpatialReasoner-R1, a novel VLM with state-of-the-art spatial reasoning capabilities, trained with a proposed fine-grained DPO (fDPO) method that decomposes LongCoT paths into description and reasoning components, allowing for targeted preference-based learning and enhanced logical reasoning. fDPO is guided by a set of comprehensive rewards that evaluate reasoning paths across visual consistency, spatial alignment, logical coherence, and depth-based verification. Additionally, we propose a Multi-Model Monte Carlo Tree Search (M3CTS) strategy that leverages multiple VLMs to generate high-quality, diverse LongCoT data. Our comprehensive evaluations demonstrate SpatialReasoner-R1 achieves state-of-the-art performance, outperforming significantly larger models. Moving forward, we plan to evaluate fDPO on additional VLM tasks such as GUI navigation and reasoning segmentation.

### Acknowledgments

This research was supported by a gift from Google AI, the Google TPU Research Cloud (TRC) program, and the U.S. Defense Advanced Research Projects Agency (DARPA) under award numbers HR00112390062 and HR001125C0303. We gratefully acknowledge the cloud TPU credits from the Google TPU Research Cloud (TRC) program and the Google Tunix (Tune-in-JAX) team for their feedback. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of Google, DARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [2] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In International Conference on Computer Vision (ICCV), 2015.
- [3] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In Annual Conference on Artificial Intelligence and Statistics (AISTATS), 2024.
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv:2502.13923, 2025.
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems (NeurIPS), 2020.
- [6] Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In International Conference on Robotics and Automation (ICRA), 2025.
- [7] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [8] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv:2306.15195, 2023.
- [9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large visionlanguage models? In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [10] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wanxiang Che. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. arXiv:2503.09567, 2025.
- [11] Zhangquan Chen, Xufang Luo, and Dongsheng Li. Visrl: Intention-driven visual perception via reinforced reasoning. In International Conference on Computer Vision (ICCV), 2025.
- [12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [13] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [14] Chenhang Cui, An Zhang, Yiyang Zhou, Zhaorun Chen, Gelei Deng, Huaxiu Yao, and Tat-Seng Chua. Fine-grained verifiers: Preference modeling as next-token prediction in vision-language alignment. In International Conference on Learning Representations (ICLR), 2024.
- [15] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: towards general-purpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

- [16] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv 2501.12948, 2025.
- [17] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- [18] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- [19] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model alignment as prospect theoretic optimization. In International Conference on Machine Learning (ICML), 2024.
- [20] Yuhan Fu, Ruobing Xie, Xingwu Sun, Zhanhui Kang, and Xirong Li. Mitigating hallucination in multimodal large language model via hallucination-targeted direct preference optimization. In Findings of the Association for Computational Linguistics (ACL), 2025.
- [21] Google Deepmind. Gemini 2.0 is now available to everyone. https://blog.google/ technology/google-deepmind/gemini-model-updates-february-2025/, 2025.
- [22] Yi Gu, Zhendong Wang, Yueqin Yin, Yujia Xie, and Mingyuan Zhou. Diffusion-rpo: Aligning diffusion models through relative preference optimization. arXiv:2406.06382, 2024.
- [23] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [24] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

- [25] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o System Card. arXiv:2410.21276, 2024.
- [26] Amazon Artificial General Intelligence. The amazon nova family of models: Technical report and model card. 2024.
- [27] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European Conference on Computer Vision (ECCV), 2016.
- [28] Yangzhe Kong, Daeun Song, Jing Liang, Dinesh Manocha, Ziyu Yao, and Xuesu Xiao. Autospatial: Visual-language reasoning for social robot navigation through efficient spatial reasoning learning. arXiv:2503.07557, 2025.
- [29] Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv:2406.18629, 2024.
- [30] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
- [31] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. In International Conference on Learning Representations (ICLR), 2025.
- [32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In International Conference on Machine Learning (ICML), 2023.
- [33] Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Yusuke Kato, and Kazuki Kozuka. Aligning diffusion models by optimizing human utility. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [34] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.
- [35] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. A survey of multimodel large language models. In International Conference on Computer, Artificial Intelligence and Control Engineering, 2024.
- [36] Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.
- [37] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision (ECCV), 2014.
- [38] Aiwei Liu, Haoping Bai, Zhiyun Lu, Yanchao Sun, Xiang Kong, Simon Wang, Jiulong Shan, Albin Madappally Jose, Xiaojiang Liu, Lijie Wen, et al. Tis-dpo: Token-level importance sampling for direct preference optimization with estimated weights. In International Conference on Learning Representations (ICLR), 2024.
- [39] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
- [40] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

- [41] Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yaochen Hu, Lingfeng Zhang, Yingxue Zhang, Shuang Wu, Tongtong Cao, Guowei Huang, et al. Spatialcot: Advancing spatial reasoning through coordinate alignment and chain-of-thought for embodied task planning. arXiv:2501.10074, 2025.
- [42] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [43] Haotian Luo, Haiying He, Yibo Wang, Jinluan Yang, Rui Liu, Naiqiang Tan, Xiaochun Cao, Dacheng Tao, and Li Shen. Ada-R1: From long-cot to hybrid-cot via bi-level adaptive reasoning optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2025.
- [44] Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [45] Wufei Ma, Haoyu Chen, Guofeng Zhang, Celso M de Melo, Jieneng Chen, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In International Conference on Computer Vision (ICCV), 2025.
- [46] Wufei Ma, Yu-Cheng Chou, Qihao Liu, Xingrui Wang, Celso de Melo, Jieneng Chen, Jianwen Xie, and Alan Yuille. Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. In Advances in Neural Information Processing Systems (NeurIPS), 2025.
- [47] Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai.meta.com/blog/llama-4-multimodal-intelligence/, April 2025.
- [48] Remi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Côme Fiegel, et al. Nash learning from human feedback. In International Conference on Machine Learning (ICML), 2024.
- [49] Kiet A Nguyen, Adheesh Juvekar, Tianjiao Yu, Muntasir Wahed, and Ismini Lourentzou. Calico: Part-focused semantic co-segmentation with large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- [50] Zhenyu Pan and Han Liu. Metaspatial: Reinforcing 3d spatial reasoning in vlms for the metaverse. arXiv:2503.18470, 2025.
- [51] Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [52] Sungjin Park, Xiao Liu, Yeyun Gong, and Edward Choi. Ensembling large language models with process reward-guided tree search for better complex reasoning. In Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), 2025.
- [53] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021.
- [54] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
- [55] Kanchana Ranasinghe, Satya Narayan Shukla, Omid Poursaeed, Michael S Ryoo, and TsungYu Lin. Learning to localize objects improves spatial reasoning in visual-llms. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

- [56] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [57] Ruichen Shao, Bei Li, Gangao Liu, Yang Chen, Xiang Zhou, Jingang Wang, Xunliang Cai, and Peng Li. Earlier tokens contribute more: Learning direct preference optimization from temporal decay perspective. In International Conference on Learning Representations (ICLR), 2025.
- [58] Shuaijie She, Wei Zou, Shujian Huang, Wenhao Zhu, Xiang Liu, Xiang Geng, and Jiajun Chen. Mapo: Advancing multilingual reasoning through multilingual-alignment-as-preference optimization. In Annual Meeting of the Association for Computational Linguistics (ACL), 2024.
- [59] Wentao Shi, Mengqi Yuan, Junkang Wu, Qifan Wang, and Fuli Feng. Direct multi-turn preference optimization for language agents. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.
- [60] Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li, and Bill Yuchen Lin. Trial and error: Exploration-based trajectory optimization of llm agents. In Annual Meeting of the Association for Computational Linguistics (ACL), 2024.
- [61] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. In Annual Meeting of the Association for Computational Linguistics (ACL), 2023.
- [62] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv:2312.11805, 2023.
- [63] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv:2403.05530, 2024.
- [64] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv:2302.13971, 2023.
- [65] Shangqing Tu, Yucheng Wang, Daniel Zhang-Li, Yushi Bai, Jifan Yu, Yuhao Wu, Lei Hou, Huiqin Liu, Zhiyuan Liu, Bin Xu, et al. Longwriter-v: Enabling ultra-long and high-fidelity generation in vision-language models. In ACM International Conference on Multimedia (ACM MM), 2025.
- [66] Muntasir Wahed, Kiet A Nguyen, Adheesh Sunil Juvekar, Xinzhuo Li, Xiaona Zhou, Vedant Shah, Tianjiao Yu, Pinar Yanardag, and Ismini Lourentzou. Prima: Multi-image visionlanguage models for reasoning segmentation. arXiv:2412.15209, 2024.
- [67] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [68] Fei Wang, Wenxuan Zhou, James Y Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and Muhao Chen. mdpo: Conditional preference optimization for multimodal large language models. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.
- [69] Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. Mixture-of-agents enhances large language model capabilities. In International Conference on Learning Representations (ICLR), 2025.

- [70] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv:2411.10442, 2024.
- [71] Xiyao Wang, Linfeng Song, Ye Tian, Dian Yu, Baolin Peng, Haitao Mi, Furong Huang, and Dong Yu. Towards self-improvement of llms via mcts: Leveraging stepwise knowledge with curriculum preference learning. arXiv:2410.06508, 2024.
- [72] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [73] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, et al. Light-R1: Curriculum sft, dpo and rl for long cot from scratch and beyond. In Annual Meeting of the Association for Computational Linguistics (ACL), 2025.
- [74] Junkang Wu, Yuexiang Xie, Zhengyi Yang, Jiancan Wu, Jinyang Gao, Bolin Ding, Xiang Wang, and Xiangnan He. beta-dpo: Direct preference optimization with dynamic beta. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [75] Wenshan Wu, Shaoguang Mao, Yadong Zhang, Yan Xia, Li Dong, Lei Cui, and Furu Wei. Mind’s eye of llms: Visualization-of-thought elicits spatial reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [76] Xun Wu, Shaohan Huang, Guolong Wang, Jing Xiong, and Furu Wei. Multimodal large language models make text-to-image generative models align better. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [77] Yuxi Xie, Guanzhen Li, Xiao Xu, and Min-Yen Kan. V-dpo: Mitigating hallucination in large vision language models via vision-guided direct preference optimization. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.
- [78] Shuo Xing, Yuping Wang, Peiran Li, Ruizheng Bai, Yueqi Wang, Chengxuan Qian, Huaxiu Yao, and Zhengzhong Tu. Re-align: Aligning vision language models via retrieval-augmented direct preference optimization. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025.
- [79] Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llava-critic: Learning to evaluate multimodal models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- [80] Wei Xiong, Chengshuai Shi, Jiaming Shen, Aviv Rosenberg, Zhen Qin, Daniele Calandriello, Misha Khalman, Rishabh Joshi, Bilal Piot, Mohammad Saleh, et al. Building math agents with multi-turn iterative preference learning. In International Conference on Learning Representations (ICLR), 2025.
- [81] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv:2412.14171, 2024.
- [82] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [83] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [84] Sen Yang, Yafu Li, Wai Lam, and Yu Cheng. Multi-llm collaborative search for complex problem solving. arXiv:2502.18873, 2025.
- [85] Zhihe Yang, Xufang Luo, Dongqi Han, Yunjian Xu, and Dongsheng Li. Mitigating hallucinations in large vision-language models via dpo: On-policy data hold the key. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

- [86] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al. Mulberry: Empowering mllm with o1like reasoning and reflection via collective monte carlo tree search. In Advances in Neural Information Processing Systems (NeurIPS), 2025.
- [87] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. In International Conference on Learning Representations (ICLR), 2023.
- [88] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [89] Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv:2501.04001, 2025.
- [90] Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning of diffusion models for text-to-image generation. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [91] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [92] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [93] Xiang Yue, Tianyu Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [94] Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. Tokenlevel direct preference optimization. In International Conference on Machine Learning (ICML), 2024.
- [95] Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, and Dong Yu. Mm-llms: Recent advances in multimodal large language models. In Annual Meeting of the Association for Computational Linguistics (ACL), 2024.
- [96] Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, and Min Lin. Chain of preference optimization: Improving chain-of-thought reasoning in llms. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [97] Qi Zhao, Haotian Fu, Chen Sun, and George Konidaris. Epo: Hierarchical llm agents with environment preference optimization. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.
- [98] Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. Slic-hf: Sequence likelihood calibration with human feedback. arXiv:2305.10425, 2023.
- [99] Han Zhong, Zikang Shan, Guhao Feng, Wei Xiong, Xinle Cheng, Li Zhao, Di He, Jiang Bian, and Liwei Wang. Dpo meets ppo: Reinforced token optimization for rlhf. In ICML 2024 Workshop on Models of Human Feedback for AI Alignment, 2024.
- [100] Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang, Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Linjun Zhang, and Huaxiu Yao. Calibrated self-rewarding vision language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

### A Fine-Grained Spatial Reward Details

This appendix details design and hyperparameter choices for the fine-grained spatial rewards introduced in Section 3.3. The prompt template for estimating rewards is shown in Figure 10.

#### A.1 Visual Consistency Reward (Rvc)

The Visual Consistency Reward quantifies alignment between the generated description and the visual scene across four continuous criteria: Existence, Attribute accuracy, Completeness, and Appropriateness. Each component yields a score in the range [0.0,1.0], where its continuous range enables fine-grained assessment, permitting fractional scores when descriptions partially satisfy evaluation criteria. Scores near 0.0 indicate misalignment, while scores near 1.0 denote perfect alignment. Intermediate values reflect varying degrees of partial correctness or uncertainty. The total reward, Rvc ∈ [0,4.0], distinguishes varying degrees of alignment across responses.

#### A.2 Depth-Guided Spatial Reward (Rsp)

We introduce a depth-guided reward to evaluate the spatial accuracy of model outputs using groundtruth depth maps. The reward is computed independently for the description Rdesc and reasoning Rreason components, yielding two sub-scores: Rsp,desc and Rsp,reason, each ranging from 0 to 4. These scores capture the alignment of spatial expressions with geometric cues in the image. The final spatial reward is given by Rsp=Rsp,desc + Rsp,reason.

Uncertainty Weight (Wu). Spatial expressions in model outputs often include uncertain qualifiers. Wu ranges from 0.8 to 1.0, with 1.0 indicating complete certainty in spatial assertions, and the lower bound of 0.8 representing cautious but plausible uncertainty. Setting the lower bound at 0.8 balances cautious language (e.g., “approximately,” “possibly”) without overly penalizing reasonable uncertainty. Lower values (below 0.8) would overly penalize reasonable, conservative predictions and discourage the model from producing cautious but informative reasoning.

Context-aware Weight (Wc). The context-aware weight Wc ∈ [0.8,1.0] reflects the relevance of spatial statements to the question. Explicitly asked spatial relationships are assigned Wc=1.0, while auxiliary or indirect spatial references are assigned Wc =0.8. This distinction prioritizes primary spatial relations explicitly required by the query, ensuring the model emphasizes essential spatial assertions more significantly. Scores below 0.8 would disproportionately underemphasize auxiliary information, degrading the model’s ability to handle broader contextually relevant details.

Given a response, we extract all spatial relationship statements from the description and reasoning response sections. Each statement is then evaluated using GPT-4o by comparing the original image and its corresponding depth image, which is generated using Depth Anything [83], to obtain a

correctness score ri ∈ [0,1]. Every statement is also assigned an associated Wu(i) and Wc(i). The spatial reward scores are computed as

1 n

Rsp,desc =

n

1 m

Wu(i) · Wc(i) · ri, Rsp,reason =

i=1

m

Wu(i) · Wc(i) · ri, (11)

i=1

where n and m denote the number of spatial statements in the description and reasoning components, respectively, and ri represents the correctness of each spatial relationship, validated against both the RGB image and its corresponding depth map.

#### A.3 Logical Coherence Reward (Rlc)

This reward quantifies the logical robustness of a response by aggregating four components: Factual Consistency, Logical Coherence, Correct Rule Application, and Conclusion Validity. Each component is scored in the range [0.0,1.0], with fractional values capturing partial correctness, e.g., from minor gaps in logical sequences to partial inaccuracies in applying physical, spatial, or logical rules. Scores of 0.0 and 1.0 indicate complete logical coherence failure or perfect logical chains, respectively. The final reward, Rlc ∈ [0,4.0], reflects the overall logical quality of the reasoning.

### B Structured Output Format Specification for M3CTS

To enable reliable parsing and downstream analysis, M3CTS requires reasoning paths from VLMs to follow a standardized structured format. This format uses Markdown-style headings to clearly segment key components of the reasoning trace. Each section begins with a line prefixed by ###, followed by a descriptive heading. The defined sections are:

### Description: Details the input context, such as the visual scene or scenario description. ### Rationale: Summarizes the overall reasoning strategy or justification. ### Let’s think step by step: An optional phrase before detailed step-by-step breakdown. ### Step N (e.g., ### Step 1, ### Step 2, ...): Enumerates the sequential steps involved in the reasoning procedure. Multiple steps are typically present. ### In Conclusion: States the final derived conclusion of the reasoning process.

Figure 5 shows the reasoning tree example produced by M3CTS.

### C Node Evaluation Protocol for M3CTS

To ensure the semantic and visual quality of each candidate reasoning step sk,t within the M3CTS framework, we employ a structured multi-criteria evaluation. Individual steps that form a reasoning

path are segmented by ###. Each candidate sk,t ∈ Sc is independently evaluated by two multimodal models, Gemini 1.5 Pro and Qwen2.5VL-72B, along key distinct dimensions:

⋄ Visual Description Accuracy: Assesses whether the entities, attributes, and contextual cues

described in sk,t correctly reflect the visual content of the input image. This includes references to objects, colors, spatial layouts, and contextual cues.

⋄ Spatial Consistency: Evaluates whether the spatial relations expressed in sk,t (e.g., “above,” “to the left of,” “behind”) are consistent with both the RGB image and depth map generated via the Depth Anything model [83]. Errors such as inversion of relations (e.g., stating “behind” instead of “in front”) are penalized.

⋄ Logical Reasoning Coherence: For steps within the “think step-by-step” chain-of-thought reasoning block, this component checks whether the logical flow of inferences is coherent and justified. This includes identifying unsupported jumps in logic or contradictions.

Each criterion I(evalm)(sk,t) is rated as follows:

 

+1, if the content is entirely accurate according to model m;

I(evalm)(sk,t) =

0, if the content is ambiguous or partially accurate;



−1, if there is any clear inaccuracy.

We preserve high-quality paths by pruning the candidate set. Specifically, we retain any node sk,t whose aggregated score across all evaluators and criteria is non-negative, i.e., Sc∗={sk,t | R(sk,t) ≥

- 0}. This threshold is chosen empirically to balance filtering out incorrect steps while maintaining adequate reasoning diversity.

### D Training Details

#### D.1 Implementation Details

SpatialReasoner-R1 is built upon the Sa2VA architecture [89], which is based on InternVL2.5. We train the 8B-parameter model in two stages on two NVIDIA H100 GPUs, each stage taking approximately 2.5 days. For supervised fine-tuning, we employ AdamW optimizer with a learning rate of 4 × 10−5, weight decay of 0.05, and a 5% linear warm-up schedule, using a batch size of 2 per device with gradient accumulation over 4 steps. For Direct Preference Optimization, we similarly use AdamW with learning rate of 1 × 10−7, weight decay of 0.05, and a 5% warm-up, training with a batch size of 1 per device.

[Figure 178]

[Figure 179]

Response from Qwen2.5VL-72B

Question: <image>\n Can

you provide an estimation

[Figure 180]

of the horizontal distance between the centers of

Response from Qwen2.5VL-7B Ground Truth: 30cm

Region2

<Region1> and <Region2>?

[Figure 181]

Response from Gemini1.5 Pro

Region1

Observe the bell peppers in the

The medium bell pepper in the background is taken to span eight

Check the cucumbers in between the 2 regions. Assume the width of the cucumbers are 3 cm.

background. Assume a medium-sized bell pepper typically has a width of

###Description

centimeters and serves as a

approximately 8 cm.

convenient scale.

The distance between <Region

<Region 1> appears to occupy almost

The red pepper's horizontal width is around 8 cm, either pepper lying on its

The distance between the two

Four cucumbers lie between

1> and <Region2> is about the

###Rationale

the same horizontal width as that

center marks spans ten

the two center marks; at 3 cm

width of orange peppers plus a small gap.

side is about 10 cm, and the larger

pepper, adding another eightcentimeter segment.

graphic units, each unit representing one centimeter.

per cucumber, the horizontal distance spans 12 cm.

dark-green pepper is roughly 12 cm.

The center marks align with the right edges of the 2nd and 6th

The <Region1> is at the start of the third cucumber and the

The cloth gap between the right edge of <Region1> and the left edge of <Region2>

Assuming <Region 2> is roughly twice the bell-pepper width, its span is set

Add roughly 2 cm to account for the gap.

###Let's Think step

by step ###Step 1

cucumbers, a separation of four

second is at the end of the

looks about half that reference length,

at sixteen centimeters and it begins

cucumber widths, equating to 12 cm.

sixth giving four full widths.

contributing roughly 4 cm.

exactly where <Region 1> ends.

###Step N

The final

The final

The final

The final

The final

The final answer is 30cm

The final answer is 24cm

The final

###In Conclusion

The final answer is 12cm

answer is 18cm

answer is 12cm

answer is 12cm

answer is 40cm

answer is 20cm

answer is 12cm

- Figure 5: Example Reasoning Tree from the M3CTS Data Generation Pipeline. Diverse candidate reasoning paths are sampled from multiple models. Each path follows a structured LongCoT format with markdown-style section headers that decompose the answer into interpretable reasoning stages.

#### D.2 Training Data

For SFT, we convert samples from the OPEN SPATIAL dataset [13] to reasoning chains using the M3CTS pipeline. While the original OPEN SPATIAL dataset provides single-sentence answers, we transform 400K samples, grounded in distinct images, into structured LongCoT reasoning chains, where examples are used to teach the model to generate high-quality, step-by-step spatial reasoning responses. For Direct Preference Optimization (DPO) training, the goal is to train the model to distinguish high-quality spatial reasoning from suboptimal or subtly flawed alternatives. To this end, we utilize our OPEN SPATIAL REASONING dataset, described below, that consists of spatial reasoning preference pairs.

An additional set of 100K challenging negative pairs is meticulously crafted by perturbing only the conclusion keywords of high-quality positive samples. Each original response represents a coherent and accurate reasoning path with a factually correct outcome. To create the corresponding negative sample, we retain the exact description and reasoning segments and alter only the final conclusion value. This yields tightly controlled preference pairs that isolate correctness at the conclusion level. For example, a positive sample may assert “The distance between region1 and region2 is 11 meters.”, while its negative perturbed counterpart is “The distance between region1 and region2 is 10 meters.”

Our method adopts a data-centric strategy that emphasizes high-quality supervision and reasoning diversity. Instead of collecting large volumes of weakly aligned or noisy data, we curate training examples using the M3CTS sampling strategy guided by structured reward evaluations. By applying reward-based filtering, we reduce noise and enforce a consistent output structure. In parallel, using multiple VLMs during generation introduces variation in reasoning styles, improving coverage of diverse spatial patterns and edge cases. The effectiveness of this approach is evident in the substantial performance gains of DPO-trained models over their simpler SFT counterparts (Table 1) and the reasoning improvements and diversity depicted in Figure 5.

#### D.3 Open Spatial Reasoning Dataset

We curate the OPEN SPATIAL REASONING dataset, a collection of 400K Vision Question Answering (VQA) preference pairs (yp,yl), to support training of preference-based spatial reasoning models. This dataset is derived from the OPEN SPATIAL dataset [13], which provides image-based spatial questions paired with ground-truth answers and offers 10 question variations per image-grounding scenario. To construct each preference pair, we randomly sample a question instance from the source dataset, and generate a diverse pool of eight candidate answers using four distinct sources: our M3CTS pipeline, Gemini 1.5 Pro, GPT-4o, and our SpatialReasoner-R1 Supervised Fine-Tuned

###### If you are at <region2>, where will you find <region1>?

[Figure 182]

Rejected Response: The image shows a street scene with buildings on both sides. There are two labeled regions: <region2>, which is a car parked on the left side of the street, and <region1>, which is a person walking

Chosen Response: The image shows a street intersection with <region2> marked on a vehicle to the left and <region1> marked on a person to the right. <region2> is located on the left side of the image, while <region1> is on the right side. Let's think step by step. Step 1: Identify the position of <region2> on the left side of the street. Step 2: Locate <region1> on the right side of the street, and further down the road. Step 3: Compare this direction to a clock face, aligning it with the 4 o'clock direction. The final answer is region [2] will find region [1] around the 4 o'clock direction.

on the sidewalk on the right side of the street. Let's think step by step. Step 1. Imagine standing at the

[Figure 183]

location of <region2>, which is the car on the left side of the street. Step 2. Look around for <region2>, which is the person walking on the sidewalk. Step 3. The person is at the right of car. The final answer is region [2] will find region [1] around the 3 o'clock direction.

[Figure 184]

###### Between Region [1] and Region [2], which one has less height?

[Figure 185]

Chosen Response: The image shows a collection of bottles on a table. Region[1] is a green bottle

Rejected Response: The image shows a collection of bottles on a table. Region[1] is a green bottle

[Figure 186]

on the right, and Region[2] is a blue bottle on the

on the right, and Region[2] is a blue bottle on the left. Region[1] appears shorter than Region[2].

left. The bottles' heights can be compared by their relative positions and surrounding context. Let's think step by step. Step 1: The middle bottles have a similar height to Region[2]. Step 2: Region[1] is taller than the middle bottles. The final answer is Region[2].

Let's think step by step. Step 1: Compare the heights visually. Region[1] is noticeably shorter than the surrounding bottles. The final answer is Region[1].

[Figure 187]

###### What is the distance between <region1> and <region2>?

[Figure 188]

Chosen Response: The image shows two taxis in front of Trump Tower. Region[1] is a yellow taxi on the right, and Region[2] is a blue taxi on the left. Typical road width is about 3 meters, with each taxi approximately 1.5 meters wide. Let's think step by step. Step 1: The taxis are aligned with their fronts and backs on the same line. Step 2: With each taxi being 1.5 meters wide and the road 3 meters wide, the space between them is about 1.5 meters. The final answer is: 1.5 meters.

Rejected Response: The image shows two taxis in front of Trump Tower. Region[1] is a yellow taxi on the right, and Region[2] is a blue taxi on the left. The taxis are parked closely side by side. Let's think step by step. Step 1: Region[1] and Region[2] are adjacent with minimal visible space. Step 2: Given typical urban parking, assume a standard spacing. The final answer is: Approximately 0.5 meters.

[Figure 189]

[Figure 190]

###### Is <region1> positioned at right of <region2>?

[Figure 191]

Chosen Response: The image shows a side mirror of a vehicle reflecting a road scene. Region[1] is the highlighted road, and region[2] is the blue car. Let’s think step by step. Step 1: Identify the positions of Region[1] and Region[2] in the mirror. Step2:

Rejected Response: The image shows a vehicle reflecting a road scene. Region[1] is the highlighted road, and Region[2] is the blue car. Let's think step by step. Step 1: Identify the positions of Region[1] and Region[2]. Region[1] is on the left side. Step2: Region[2] is on the right side. The final answer is No <region1> positioned at left <region2>.

[Figure 192]

- Region[1] is on the left side. Since the image is shown in mirror, so Region[1] is to the right of
- Region[2]. The final answer is Yes <region1> positioned at right <region2>.

[Figure 193]

- Figure 6: Example DPO Pairs of our OPEN SPATIAL REASONING Dataset, constructed from M3CTS-generated reasoning trajectories. Each pair consists of a preferred and a rejected response to the same spatial question. The examples highlight differences in descriptive accuracy, spatial alignment, and reasoning coherence, which guide preference optimization during training.

(SFT) model, with each method contributing two response variants. All eight candidate responses are independently evaluated by our fine-grained spatial reward mechanism (Appendix A). The highestscoring response is selected as the preferred answer (yp), while the response with the lowest score is designated as the less-preferred (yl), ensuring that each preference pair is anchored in meaningful fine-grained spatial reasoning quality. Figure 6 shows dataset examples.

### E Evaluation Details

We evaluate on SPATIALRGPT-BENCH [13], a benchmark specifically designed to assess the 3D spatial reasoning abilities of VLMs, featuring 657 qualitative and 749 quantitative VQA pairs, covering 88 object classes across diverse environments. We employ the same GPT-4 evaluation proposed in SPATIALRGPT-BENCH [13] for evaluating the free-form responses generated by the models. For qualitative questions, GPT-4o assesses the semantic alignment between the model’s response and the ground-truth answer, assigning a binary score (1 for correct, 0 for incorrect). For quantitative questions (e.g., distance, size), GPT-4o first extracts numerical values from both the prediction and the ground truth, standardizing them to a common unit (meters). We then compute accuracy (e.g., success rate defined as predictions within ±25% of the ground truth).

We also evaluate on several general vision-language benchmarks to provide a comprehensive assessment of SpatialReasoner-R1’s capabilities. Specifically, we use MME [34] to assess multimodal models on perception and cognition tasks across a wide range of domains. POPE [34] is employed

###### Q: Can you give me an estimation of the horizontal distance between Region [1] and Region [2]?

Ground Truth Response from SpatialRGPT 8B

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

The horizontal distance of Region [1] from Region [2] is 12.0 meters.

Response from InternVL2.5-78B

[Figure 198]

- Step 1:To estimate the horizontal distance between Region [1] and Region [2], we can use the trucks as reference points.
- Step 2: The average length of a rental truck is around 4 meters.
- Step 3: The space between the two trucks appears to be slightly more than the length of one truck. By adding the space to the length of the trucks, we can approximate the distance.

- Step 4: Therefore, the horizontal distance between Region [1] and Region [2] is approximately 10 meters.

[Figure 199]

Region2

Region1

Region [1] and Region [2] are 16.58 meters apart horizontally.

Response from Gemini 1.5 Pro Response from 8B

[Figure 200]

[Figure 201]

[Figure 202]

Step 1: Consider the width of the street and the sidewalk. The person (Region 2) is on the sidewalk, and the truck (Region 1) is

[Figure 203]

[Figure 204]

Step1: Region 1 (truck) and Region 2

(pedestrian) are separated by the Penske truck and some sidewalk.

on the street.

- Step 2:To estimate the horizontal distance, we can sum the widths of the main traffic lanes. Let's consider three distinct traffic lanes, each approximated as 4 meters wide, which total is 12 meters

- Step 3: Estimate the distance of roadside or shoulder area between the person to road. And it is about 4 meters.
- Step 4: Therefore, combining the 12 meters from the traffic lanes with this 4meter roadside area yields a total estimated distance of 16 meters.

- Step 2: The Penske truck looks about 4 meters width. And one lane plus the truck is about 8 meters.

- Step 3: The sidewalk adds another 1.5-3 meters.
- Step 4: Thus, Region 1 and Region 2 are roughly 10 meters apart horizontally.

- Figure 7: Qualitative Examples of Spatial Reasoning Across Models. SpatialReasoner-R1 demonstrates coherent, step-by-step spatial reasoning that closely aligns with ground truth estimates. In contrast, baseline models produce less precise or partially incorrect reasoning steps, often neglecting key visual cues or misestimating spatial references.

to evaluate object hallucination in testing the ability of VLMs to ground responses to visual content, while SEED-BENCH [30] offers a multi-dimensional evaluation, covering aspects from image understanding to complex reasoning across various modalities and tasks. We further utilize AI2D [27], a benchmark focusing on diagram understanding and reasoning, which requires parsing visual elements and their relationships within schematic representations. SQA [42] is used to measure the model’s ability to answer science-related questions based on visual context, often requiring domain-specific knowledge and reasoning. MMMU [92] evaluates massive multi-disciplinary multimodal understanding and reasoning across diverse college-level subjects. Moreover, MMSTAR [9] provides a challenging benchmark with meticulously curated, multimodal instances that require advanced reasoning, low hallucination, and resistance to leading questions. Finally, HALLUSIONBENCH [23] is specifically designed to quantitatively measure and analyze the hallucination phenomena in VLMs, probing for both object-level and attribute-level inconsistencies.

F Qualitative Experiment Examples

In this section, we provide additional qualitative experiment examples. Figure 7 shows a question that requires estimation of the horizontal distance between a truck and a pedestrian. SpatialReasoner-R1 demonstrates a clear advantage by decomposing the scene into semantically meaningful components, explicitly reasoning over the widths of multiple traffic lanes, the roadside, and the sidewalk. This results in an estimated distance that closely matches the ground truth and provides full transparency into the model’s stepwise deductions. In contrast, InternVL2.5-78B bases its answer primarily on the width of the trucks and the space between them, omitting the crucial step of accounting for the distance from the pedestrian to the roadway, which leads to significant underestimation. Gemini1.5Pro correctly recognizes that the separation includes the truck, traffic lane, and sidewalk, but substantially underestimates the width of the sidewalk, causing a notable error in its final answer. Meanwhile, SpatialRGPT-8B provides a more accurate estimate than Gemini or InternVL2.5-78B, but still has a gap compared to the ground truth. Most importantly, it cannot generate step-by-step reasoning traces.

- Figure 8 presents another illustrative example evaluating spatial reasoning capabilities of various models, specifically focusing on size comparison between two highlighted image regions. The question is whether Region 1 (a computer monitor) appears smaller than Region 2 (a computer tower). SpatialReasoner-R1 accurately identifies Region 2 as a computer tower and explicitly reasons by comparing Region 1 with the closest computer tower positioned adjacent to the monitor. This

###### Q: Does <region1> have a smaller size compared to <region2>?

Ground Truth Response from SpatialRGPT 8B

[Figure 205]

[Figure 206]

[Figure 207]

No, <region1> is not smaller than <region2>.

[Figure 208]

Region2

Response from InternVL2.5-78B

[Figure 209]

[Figure 210]

Step 1: The image shows an office environment with several computer monitors and towers. Region1 is highlighted around a monitor, while Region2 is highlighted around a computer tower.

- Step 2: The monitor's width is around 16-24 inches, and its height is about 10-15 inches.
- Step 3: A standard computer tower might be around 10-14 inches wide and 16-20 inches tall.

Region1

Step 4: No, Region1 does not have a smaller size compared to Region2.

Yes, Region1 has a smaller size

compared to Region2

Response from Gemini 1.5 Pro Response from 8B

[Figure 211]

[Figure 212]

[Figure 213]

Step 1: The image shows an office setting with computer workstations. Region1 highlights a blue computer monitor on a desk in the foreground. Region2

[Figure 214]

[Figure 215]

Step 1: The image shows a computer lab with several desks and computer setups. Region1 is labeled on a computer monitor, and Region2 is labeled on a computer tower. The monitor appears smaller than the tower.

highlights a green, translucent object, possibly a plant

or decorative item, on a different desk further back and to the right.

- Step 2:Compare the sizes of the monitor (Region1) and the computer towers (Region2 and the one next to Region1). Monitors are typically smaller than computer towers.

- Step 3: Yes, Region1 has a smaller size compared to the computer towers next to it, including Region2.

Step 2:Compare the apparent size of the object within Region1 to the apparent size of the object within Region2. The computer monitor in Region1 appears larger than the decorative object in Region2.

Step 3: No, Region1 does not have a smaller size

compared to Region2.

Figure 8: Qualitative Examples of Spatial Reasoning Across Models. SpatialReasoner-R1 correctly recognizes Region2 as a computer tower and compares it clearly with the nearby monitor, reaching an accurate conclusion. InternVL2.5-78B relies on general object size knowledge but provides incorrect reasoning, Gemini1.5Pro fails to identify Region2 clearly and draws incorrect visual conclusions, while SpatialRGPT-8B directly provides a wrong answer.

systematic visual grounding and clear comparative reasoning enable SpatialReasoner-R1 to correctly conclude that the monitor is indeed smaller than the tower. By contrast, the baseline models exhibit varying degrees of errors and reasoning inadequacies. InternVL2.5-78B relies significantly on prior general knowledge about typical object dimensions and incorrectly concludes the monitor is not smaller, without effectively validating this against the visual evidence provided. textbfGemini1.5Pro fails entirely to recognize what object Region 2 represents, causing it to inaccurately rely purely on the objects’ visual proximity and perspective, leading to an incorrect conclusion. Lastly, the SpatialRGPT-8B model directly presents an incorrect judgment (“Region1 is not smaller”) without providing any interpretable reasoning steps or visual grounding.

We also provide a failure example in this section. Figure 9 illustrates a representative failure case on vertical size estimation in an indoor setting. The query is “How tall is Region 1?”. Here, Region 1 corresponds to the dresser mirror on the right, adjacent to a sleigh-style bed headboard on the left. SpatialReasoner-R1 produces an estimate of approximately 2.0m, while the ground truth is closer to

- 1.5m. Our model’s reasoning proceeds as follows: (1) segments Region 1 (the mirror) and searches for a nearby object of familiar scale, (2) identifies the bed headboard and assumes a typical headboard height of 1.5m, further estimating that the mirror extends about 0.5m above the headboard, and (3) sums these values to obtain about 2.0m. This error arises from overreliance on default furniture priors rather than fully grounding the estimate in image evidence, such as the mirror’s vertical extent relative to the floor plane and its contact points with the dresser. To mitigate this, training incorporates fine-grained reward signals that explicitly reward consistency between predicted measurements and image-derived cues, encouraging verification of intermediate steps (e.g., floor contact, vanishing-line alignment) before finalizing a measurement.

### G Broader Impacts

This work aims to improve the spatial reasoning capabilities of vision-language models through fine-grained preference optimization. Accurate spatial understanding is critical for downstream applications such as robotics, autonomous navigation, assistive technologies, and visual analytics. By introducing more interpretable and structured reasoning mechanisms, our method can contribute to building AI systems that are safer, more transparent, and more aligned with human expectations in

###### Q: How tall is Region 1?

Ground Truth

Response from 8B

[Figure 216]

[Figure 217]

[Figure 218]

Step 1: The image shows a furnished bedroom with a dresser mirror on the right (Region1) and a sleigh-style bed headboard on the left. Region1 is labeled on the dresser mirror, which appears adjacent to the bed.

[Figure 219]

[Figure 220]

- Step 2:Compare the height of the mirror (Region1) with the bed headboard nearby. Bed headboards typically have a standard height of around 1.5 m. The mirror appears to extend approximately 0.5 m taller than the headboard.

- Step 3: Region1 is approximately 1.5 m headboard height + 0.5 m additional height = 2.0 m total.

Yes, the height of region1 is about 1.5m.

Figure 9: Failure case on height estimation in a furnished bedroom.

spatially grounded tasks. However, as with other vision-language systems, potential risks remain. If deployed in safety-critical domains, incorrect spatial inferences, especially in edge cases, could lead to unintended consequences. Additionally, reward scoring and generation rely on foundation models that may encode hidden biases, which can propagate through the training pipeline. Although we attempt to mitigate these risks via multi-source sampling and structured evaluation, future work should explore robustness to distribution shifts, adversarial spatial prompts, and the inclusion of human-in-the-loop verification for high-stakes use cases.

### H Limitations

While our work demonstrates strong improvements in spatial reasoning, a limitation of our approach is its reliance on explicit region representations provided as input to disambiguate object references within the spatial queries. Enabling the model to implicitly ground entities solely based on natural language descriptions remains an avenue for future investigation, which would enhance the model’s flexibility in real-world scenarios. Future work could focus on integrating implicit linguistic context understanding to alleviate this constraint. Finally, our focus is limited to 2D spatial reasoning; extending this framework to 3D or embodied contexts would require structural adjustments left for future work.

###### System Prompt for LongCoT Reward Evaluation

The following is a spatial reasoning task, and this is the question: question and the ground truth is: ground_truth. The response is divided into different sections. There are 4 dimensions to evaluate, and I will provide you with the corresponding image and text for reference. You will need to evaluate the response based on the following criteria:

The first task: Descriptive Scoring (Total 0–4.0 points) Evaluate the "Description" section based on:

- • Existence: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly confidently incorrect. Does the description correctly identify objects that actually appear in the image?
- • Attribute Accuracy: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly confidently incorrect. Are the object’s attributes (color, shape, size, etc.) described accurately?
- • Completeness: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly confidently incorrect. Does the description include all key objects and necessary details relevant to the question?
- • Appropriateness: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly confidently incorrect. Does the description focus on the core aspects of the question?

Clearly state the score for each sub-category and sum them to obtain the final descriptive score. You need to give the score with the following format: {"task1_score": your score}

The second task: Depth-Guided Spatial Relationship Scoring – Description (Total 0–4.0 points) Evaluate all spatial statements within the "Description" section using the provided depth image as ground truth.

• For each spatial claim in the description:

- – Correctness score: Assign 1 if the spatial claim is correct based on the depth image, and 0 if not correct.
- – Uncertainty score: For claims expressed with uncertainty (using words like "approximately", "roughly", "possibly"), assign a score from 0.8 to 1.0, where 1.0 means the statement is expressed with high certainty.
- – Relationship score: Assign a weight from 0.8 to 1.0 based on whether the relationship is explicitly emphasized by the question (1.0) or is extra/irrelevant information (0.8).

Provide a detailed breakdown for each spatial claim. Calculate the final score as: (Sum of (Correctness score × Uncertainty score × Relationship score)) / (Number of claims), then scale to 4.0. You need to give the score with the following format:

- {"task2_claim_score": [Correctness score, Uncertainty score, Relationship score]}

The third task: Depth-Guided Spatial Relationship Scoring – Reasoning (Total 0–4.0 points) Apply the same evaluation method as in Task 2 to the spatial statements within the "Reasoning" section.

• For each spatial claim in the reasoning:

- – Correctness score: Assign 1 if the spatial claim is correct based on the depth image, and 0 if not correct.
- – Uncertainty score: For claims expressed with uncertainty, assign a score from 0.8 to 1.0, where 1.0 means the statement is expressed with high certainty.
- – Relationship score: Assign a weight from 0.8 to 1.0 based on whether the relationship is explicitly emphasized by the question (1.0) or is extra/irrelevant information (0.8).

Provide a detailed breakdown for each spatial claim. Calculate the final score as: (Sum of (Correctness score × Uncertainty score × Relationship score)) / (Number of claims), then scale to 4.0. You need to give the score with the following format:

- {"task3_claim_score": [Correctness score, Uncertainty score, Relationship score]}

The fourth task: Reasoning Scoring (Total 0–4.0 points) Evaluate the "Reasoning" section (the chain-of-thought) based on:

- • Factual Consistency: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly confidently incorrect. Are the claims consistent with the image, depth image, and the earlier description?
- • Logical Coherence: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly confidently incorrect. Do the reasoning steps flow logically without gaps or contradictions?
- • Correct Application of Rules: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly confidently incorrect. Are physical, spatial, and logical rules applied correctly?
- • Conclusion Validity: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly confidently incorrect. Does the reasoning properly support the final answer?

Clearly state the score for each sub-category and sum them to obtain the final reasoning score. You need to give the score with the following format: {"task4_score": your score}

Figure 10: System Prompt for Evaluating LongCoT Spatial Reasoning w.r.t. descriptive accuracy, spatial alignment, and logical consistency of reasoning steps.

