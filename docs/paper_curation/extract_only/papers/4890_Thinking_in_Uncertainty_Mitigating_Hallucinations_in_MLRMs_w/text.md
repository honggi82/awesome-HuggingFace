# arXiv:2603.13366v1[cs.CV]9Mar2026

## Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding

Zhongxing Xu1* Zhonghua Wang1* Zhe Qian1* Dachuan Shi2 Feilong Tang1 Ming Hu1 Shiyan Su1 Xiaocheng Zou4 Wei Feng1 Dwarikanath Mahapatra5 Yifan Peng3 Minquan Lin6 Zongyuan Ge1

1Monash University 2Georgia Tech 3Cornell University 4Northeastern University 5Khalifa University 6University of Minnesota

{zhongxing.xu,zongyuan.ge}@monash.edu

#### Abstract

Recent advancements in multimodal large reasoning models (MLRMs) have significantly improved performance in visual question answering. However, we observe that transition words (e.g., because, however, and wait) are closely associated with hallucinations and tend to exhibit high-entropy states. We argue that adequate contextual reasoning information can be directly extracted from the token probability distribution. Inspired by superposed representation theory, we propose leveraging latent superposed reasoning to integrate multiple candidate semantics and maintain latent reasoning trajectories. The hypothesis is that reliance on discrete textual inputs may drive the model toward sequential explicit reasoning, underutilizing dense contextual cues during high-entropy reasoning stages. Therefore, we propose constructing rich semantic representations from the token probability distributions to enhance in-context reasoning. With this goal, we present Latent Entropy-Aware Decoding (LEAD), an efficient plug-andplay decoding strategy that leverages semantic context to achieve reliable reasoning. The heart of our method lies in entropy-aware reasoning mode switching. The model employs probability-weighted continuous embeddings under high-entropy states and transitions back to discrete token embeddings as entropy decreases. Moreover, we propose a prior-guided visual anchor injection strategy that encourages the model to focus on visual information. Extensive experiments show that LEAD effectively mitigates hallucinations across various MLRMs on multiple benchmarks.

#### 1. Introduction

Large reasoning models [2, 16, 45] enhance their complex reasoning capabilities by scaling up the computational

* Equal contribution. https://mlrm-LEAD.github.io/

[Figure 1]

[Figure 2]

Describe this image in detail.

[Figure 3]

<think> … At first glance, the stage sits in a shallow bowl of rust-red rock walls… The crowd at the front rows clusters tightly, watching on the stage. Therefore, they should be watching the band that is playing on the stage … The daylight is not so strong, the rocks on the far left ridge are sun-washed. So the time should be approaching the evening as they are playing right now … A few people in the nearest rows face away from the stage, as they are taking photos of the view at…

70

Hallucinations Co-occur with Transition Words (within 10 tokens)

NumberofCasesin200Samples

60

All Hallucinations

50

40

30

20

10

0

OpenVLThinker VL-Rethinker Vision-R1 R1-Onevision

Figure 1. Illustrations of the correlation between hallucinations and transition words. In MLRMs, hallucinations tend to emerge more frequently after transition words, and these cases constitute a significant proportion of the overall hallucination occurrences.

budget during inference. This allows them to generate extended reasoning chains that incorporate causal, contrastive, and self-reflective logic before arriving at a final answer. Recently, this paradigm has been expanded to the multimodal setting. Multimodal reasoning models (MLRMs) [12, 58, 82, 86] integrate visual understanding with linguistic reasoning by constructing explicit reasoning chains, trained via reinforcement learning with verifiable rewards. However, despite these advances and their strong multimodal reasoning capabilities, MLRMs remain highly prone to hallucinations [9, 14, 33, 37, 55].

Recent studies have primarily aimed to mitigate hallucinations in multimodal reasoning models through visual reward designs [13, 50, 85] and data augmentation strategies [4, 26], but these methods often incur substantial additional costs. Conversely, training-free decoding strategies [21, 23, 24, 84], such as contrastive decoding, mitigate

20

[Figure 4]

1.0

[Figure 5]

Describe this image in detail.

[Figure 6]

Position - dependent decay Variation Range

Mask Other Tokens Mask High-Entropy Tokens

|<think> Okay, let‘s see. The photo shows a densely built hillside city filled with white and beige houses, tightly packed together … But maybe just beyond the last row of buildings, you can see the blue surface of the sea shimmering faintly … Now, the green-tiled roofs in the foreground and the tall minaret confirm an Islamic architectural style … Actually, the soft light and distant haze make it look like sea mist is drifting inland from the coast ... So the whole city seems to rise gently from the shoreline, with sunlight reflecting..|
|---|

PerformanceRatio()Δ↓

0.8

Performance(%)Δ↓

15

0.6

10

0.4

5

0.2

Exploratory Phase

0

0.0

3B 7B 32B

[Figure 7]

0 0.2 0.4 0.6 0.8 1.0

High Entropy

Model Parameter

[Figure 8]

Token Position

[Figure 9]

[Figure 10]

###### (a) (b)

[Figure 11]

[Figure 12]

[Figure 13]

###### TokenEntropy

36

| |High-entropy High-entropy|
|---|---|
| | |

tokens with hallucination tokens without hallucination

Low-Entropy State

High-Entropy State

24

Density

12

Low Entropy

0

0.00 0.05 0.10 0.15 0.20

Visual Attention Ratio

(c) (d)

|Reasoning Steps|
|---|

Convergent Phase

Figure 3. (a) Performance gap when masking different types of token during reasoning. Masking high-entropy tokens produces a larger performance drop than other tokens. (b) Token masking impact across reasoning steps. Earlier tokens tend to have stronger influence on the final answer, while the influence of later ones gradually diminishes. (c) Schematic depiction of reasoning paths at different states. (d) Token density comparisons. On average, high-entropy tokens without hallucinations exhibit higher visual attention ratios compared to hallucinated ones.

Figure 2. Visualizations of token entropy during the reasoning phase show that tokens with higher entropy often correspond to transition words, consistent with our previous findings.

hallucinations during generation by perturbing token-level samples to adjust output distributions. Though previous works have shown effectiveness, they lack analysis of the behavioral characteristics unique to reasoning models. In our analysis, we observe that MLRMs employ causal, contrastive, and reflective transition words (e.g., because, however, wait) at significantly higher frequencies during generation. These markers help structure multimodal reasoning chains and organize semantic relations through linguistic logic, a pattern consistent with recent findings in language models [7, 61]. Furthermore, as shown in Fig. 1, the content that follows such transition words often exhibits hallucinatory descriptions.

ther divide the explicit reasoning chains of MLRMs into five segments and perturb high-entropy tokens in each segment. As illustrated in Fig. 3 (b), token masking applied early in the reasoning chain results in the most severe performance degradation. This finding demonstrates that early high-entropy tokens exert stronger directional influence on the overall reasoning trajectory and play a pivotal role in guiding the model toward (or away from) correct reasoning paths. Therefore, our findings suggest that maintaining semantic diversity and visual grounding during high-entropy phases is key to mitigating reasoning-related hallucinations.

In this study, we investigate the intrinsic relationship between transition words and hallucinations from the perspective of token-level uncertainty, measured by entropy. As illustrated in Fig.2, transition words consistently exhibit higher entropy, indicating high-uncertainty stages within the reasoning chain. During these high-entropy phases, the model faces greater semantic divergence and increased competition among potential reasoning paths, thereby heightening the likelihood of hallucination. We hypothesize that reliance on discrete textual inputs encourages sequential, explicit reasoning, limiting its ability to effectively leverage dense contextual cues when uncertainty is high. In this work, we argue that the construction of richer semantic representations from token probability distributions enhances the model’s contextual reasoning capability.

In this work, we propose Latent Entropy-Aware Decoding (LEAD), a lightweight plug-and-play decoding strategy that enables reasoning reliability by leveraging contextual semantics. Specifically, when the model enters a highentropy state, LEAD enriches the input representation by combining the discretely sampled token with its predicted probability distribution. This fuses diverse semantic cues while preserving model’s inherent uncertainty. The core idea of LEAD is entropy-aware reasoning mode switching. Under high entropy, LEAD replaces the collapsed one-hot token vector with a probability-weighted combination of all token embeddings, implicitly preserving multiple reasoning hypotheses. As entropy decreases, the model naturally reverts to discrete token embeddings, achieving adaptive semantic convergence. Moreover, as illustrated in Fig.3(d), high-entropy tokens associated with hallucinations typically exhibit lower visual attention, suggesting a reduced reliance on visual information under high-uncertainty conditions. To address this, LEAD introduces a visual guidance vec-

To verify the role of high-entropy tokens in the reasoning chain, we conduct token masking ablation experiments. As illustrated in Fig. 3 (a), masking high-entropy tokens leads to a significant drop in reasoning performance, whereas masking low-entropy tokens causes only minor degradation. This indicates that high-entropy tokens serve as critical informational nodes in the reasoning process. We fur-

tor derived from pretrained visual embeddings during highentropy phases, encouraging the model to refocus on visual content and thus mitigating multimodal hallucinations.

With extensive experiments, LEAD demonstrates significant hallucination-mitigating performance across different MLRMs on both general and scientific multimodal reasoning benchmarks, validating its effectiveness. Our contributions are as follows:

- • We analyze the relationship between transition words and hallucinations in multimodal reasoning from the perspective of token-level uncertainty.
- • We propose LEAD, a plug-and-play decoding approach that effectively mitigates hallucinations in high-entropy reasoning states through an entropy-aware reasoning and visual injection mechanism.
- • Extensive evaluations on both general and scientific tasks show the superior performance of LEAD, offering an effective solution for multimodal reasoning hallucinations.

#### 2. Related Work

Multimodal Large reasoning models. Recent multimodal large language models (MLLMs) have achieved substantial progress in multimodal reasoning, largely driven by innovations in post-training techniques. Among these, supervised fine-tuning (SFT) [31, 42, 44, 68, 69, 89] and reinforcement learning (RL) [30, 32, 55, 66, 78] remain the two most common and fundamental approaches. A number of recent works [15, 26, 47, 53, 63, 92] primarily focus on enhancing long-chain reasoning in MLLMs through SFT. Meanwhile, the Group Relative Policy Optimization algorithm has emerged as a standard paradigm for training multimodal large reasoning models [35, 36, 58, 62, 77, 87]. Among these, some approaches [1, 3, 4, 18, 57, 70, 71, 74] adopt a two-stage training paradigm, while others directly employ reward-optimized RL strategies on largescale datasets [5, 6, 60, 65, 94].

Multimodal reasoning Hallucinations. Despite improvements from chain-of-thought reasoning, multimodal reasoning models remain prone to hallucinations, including contradictions with visual evidence [9, 14, 29, 33, 37, 55, 81] and logical inconsistencies in reasoning [8, 19, 27, 38, 46, 48, 49, 52, 83]. One solution is to optimize the reward-function paradigm [13, 67, 70, 76, 85] to improve perception and stabilize multimodal reasoning. Existing multimodal hallucination mitigation methods include contrastive decoding [23, 24, 64, 84, 88] and self-corrective attention [21, 34, 41, 54, 79], which reduce reliance on biases and priors. Inspired by superposed representation theory [11, 17, 72, 93, 95], we propose a latent superposed reasoning approach for reasoning models, which uses the token probability distribution to extract sufficient contextual information and effectively mitigates hallucinations.

#### 3. Methodology

Figure 4 provides an overview of the proposed strategy, which builds upon the MLRM decoding paradigm introduced in Section 3.1. Section 3.2 elaborates on the entropyaware reasoning mode switching, designed to optimize embedding representations under high-entropy states and guide the model toward semantically enriched contextual information. Meanwhile, Section 3.3 introduces a guidance vector derived from the pretrained visual modality to strengthen the model’s focus on visual content during uncertain reasoning phases. For clarity, Algorithm 1 exhibits the pseudocode for the decoding process of LEAD.

##### 3.1. MLRMs Generation

Vision and Language Inputs. A Multimodal Large Reasoning Model (MLRM) accepts both image and text as input. The raw image is first processed by a vision encoder to extract semantic features, which are then projected into the language model’s input space through a cross-modal projection module, forming a sequence of N vision tokens xv = {xv,1,xv,2,...,xv,N}. Meanwhile, the textual input is tokenized and embedded to form a sequence of M text tokens xt = {xt,1,xt,2,...,xt,M}. These vision and text tokens are concatenated to form the complete multimodal input sequence x = xv⊕xt = {xt}Tt=1, where T = N+M, serving as the input for subsequent reasoning and enabling the model to jointly process and infer over visual and linguistic information.

MLRMs Forward. The backbone of the MLRMs, denoted as Rθ, is a pre-trained LLM parameterized by θ, which generates responses autoregressively. Given a multimodal input x, the model predicts the next token distribution at each time step t as:

pt = Rθ · | x,y<t ∈ ∆|V|−1, (1)

where y<t = (y1,y2,...,yt−1) denotes all previously generated tokens, V is the vocabulary of the model, and ∆|V|−1 denotes the (|V| − 1)-dimensional probability simplex.

Discrete Reasoning Decoding. Reasoning models achieve test-time scaling by explicitly separating the intermediate reasoning phase from the final answering phase. Given a multimodal input x, the model first generates a reasoning trajectory r1:m = (r1,r2,...,rm) and then produces the final answer sequence a1:n = (a1,a2,...,an), thereby structuring generation into two distinct stages.

At each intermediate reasoning step t, the model first computes a probability distribution pt over the vocabulary based on the multimodal input embeddings e(x) and the embeddings of all previously generated reasoning tokens

### ... as it

|[Figure 14]|
|---|

Sampled token

there is

shows

###### However

TextualtokensVisualtokens

### ...

Token distribution

So However But And

as with that be

[Figure 15]

Latent decoding

𝐻 𝐻

Multimodal Large Reasoning Models

|Textual Input<br><br>Please describe this image<br><br>Ongoing Response<br><br><think>, [...] None of these explicitly mention cups.| |
|---|---|
| | |

Discrete decoding

[Figure 16]

...

None [...] cups. as it

<image _pad>

[Figure 17]

𝑒 𝑒 𝑒

𝑒

[Figure 18]

𝐻

𝐻

Discrete embedding

Weighted embedding

Figure 4. Illustration of multimodal reasoning and entropy-aware decoding. The model receives both visual and textual tokens (left) and generates responses by integrating contextual information. During reasoning, token-level entropy Ht measures model confidence and is compared with the reference entropy Hˆ. High-entropy states (orange) trigger latent decoding, using probability-weighted embeddings to preserve semantic diversity, while low-entropy states (blue) activate discrete decoding, using sampled tokens for precise semantic convergence. This adaptive switching mechanism balances exploration and commitment in multimodal reasoning.

that can easily trigger hallucinations. In contrast, the lowentropy phase reflects a converging reasoning chain with more stable outputs. However, existing models typically operate under a fixed discrete reasoning mode and are unable to adapt to these dynamic states. To address this limitation, we propose an entropy-aware dynamic reasoning switch mechanism that uses token-level entropy as a confidence indicator. During high-entropy phases, it activates latent reasoning decoding to maintain semantic diversity; as entropy decreases, it switches back to discrete decoding to ensure stable convergence. This adaptive mechanism allows the reasoning mode to dynamically respond to uncertainty.

e(r<t), and sample the token rt in current step: pt = Rθ e(x),e(r<t) , rt ∼ pt, rt ∈ V. (2)

Decoding continues until the special end-of-thinking token ⟨/think⟩ is generated. The model then enters the answering phase, where a1:n is decoded in the same manner.

Latent Reasoning Decoding. Although discrete reasoning improves reliability by exposing intermediate reasoning steps, its decoding strategy collapses the full predictive distribution pt into a single sampled token at each step, thereby discarding crucial distributional information that may be needed to navigate uncertain reasoning states. To address this limitation, latent reasoning decoding replaces the discrete choice with a continuous representation that retains the entire predictive distribution. At reasoning step t, the model outputs a probability distribution pt over the vocabulary, and forms a probability-weighted embedding for the next step as:

Mode Switch Criterion. We use token-level entropy H to measure the model’s uncertainty at each generation step. Formally, at step t, the entropy is defined as:

pt[v]log pt[v], (4)

Ht = −

v

[e(v)], (3)

e˜t = Ev∼p

where pt[v] denotes the predicted probability of token v.Intuitively, high entropy arises when several candidate tokens have similar probabilities, e.g., pt[v1] ≈ pt[v2] ≈ ··· ≈ pt[vm], indicating competition among multiple potential reasoning paths in the semantic space. Conversely, when a single token dominates, i.e., pt[v∗] ≫ pt[v] for all v ̸= v∗, the model’s uncertainty decreases and its reasoning process progressively converges toward a single deterministic trajectory.

t

where E denotes the expectation under the distribution pt, and e(v) denotes the embedding of token v. This continuous embedding, representing a mixture of all possible tokens, is fed back into the model as input for the next step, rather than the one-hot embedding of a sampled token. Such a formulation allows the model to propagate contextual uncertainty across reasoning steps and mitigates information loss inherent in discrete sampling.

Let Hˆ be the reference entropy threshold for the current reasoning mode, which is initialized at the beginning of each mode and updated after every transition. This allows the model to adjust its reasoning behavior adaptively according to the evolving uncertainty state. The model dynamically switches between reasoning modes based on the

##### 3.2. Entropy-Aware Reasoning Mode Switching

As shown in Figure 3(c), multimodal reasoning models exhibit distinct reasoning states during generation. The high-entropy phase corresponds to increased semantic uncertainty and competition among potential reasoning paths

local trend of entropy variation. Specifically, the next-step input embedding e˜t is defined as:

e(rt), if Ht < Hˆ (Uncertainty drops), Ev∼pt[e(v)], otherwise (Uncertainty rises).

(5)

e˜t =

where pt is the probability distribution at current step and rt is the token sampled from pt. In low-entropy states, the model employs discrete token embeddings for deterministic reasoning, while in high-entropy states, it utilizes probability-weighted embeddings to preserve semantic diversity. This entropy-aware mechanism enables a continuous, self-regulated transition between discrete and latent reasoning, with entropy serving as an internal signal.

Persistence Window. To avoid rapid oscillation between the two reasoning modes, we introduce a persistence window into the switching rule. Let mt ∈ {D, L} denote the reasoning mode at step t, where D and L correspond to the discrete and latent modes, respectively. We define two gating variable for mode transition as:

###### gtD = [H t < Hˆ], (6)

gtL = [(H t > Hˆ) ∧ (ρt ≥ WD→L)], (7) where [·] denotes the indicator function, ρ t denotes the number of consecutive steps the model has remained in its current mode, and WD→L is the minimum number of steps the model must remain in the discrete mode before switching to the latent mode. The mode transition rule is defined as:

mt+1 = gtDD + gtLL + (1 − gtD − gtL)mt. (8) When a mode transition occurs, the reference entropy is updated as Hˆ ← Ht, and the persistence counter ρt is reset to 0. Otherwise, the counter is incremented as ρt ← ρt + 1. In practice, we enforce a persistence window only for the discrete-to-latent transition, i.e., WD→L > 0. This allows a L → D transition to occur immediately when confidence rises. In contrast, a D → L transition is permitted only after the model has remained in the discrete mode for at least WD→L steps. This asymmetric design ensures that the model stays in discrete reasoning long enough to consolidate a coherent reasoning trajectory before returning to latent exploration.

Switch Count Regulation. Although the model can dynamically switch between reasoning modes based on uncertainty, it may still exhibit overthinking, leading to unnecessary mode transitions even after the reasoning process has largely converged. To mitigate this, we introduce a global switch counter Ct with an upper bound Cmax to limit the total number of allowed mode transitions. Once this limit is exceeded, the model halts further reasoning and proceeds directly to generate the final answer.

Algorithm 1 Pseudocode of LEAD in Python Style

# logits: raw scores before softmax # E: embedding matrix # tau: entropy threshold tracked in state # c: maximum switch budget # vis_injected: visual embedding injected already or not # vis_emb/ter_emb: special embeddings via overrides

def LEAD_step(logits, E): # probability geometry p = torch.softmax(logits) H = -(p * (p + eps).log()).sum()

# mode transition with threshold update mode = torch.where(H>=tau, LATENT, DISCRETE).where(prev) switched = (mode != prev) tau = torch.where(switched, H, tau)

# latent embedding construction p = p / (p**2).sum().sqrt() + eps base = LATENT * (p.unsqueeze(-1) @ E).sum(dim=0)

+ (1 - LATENT) * E[argmax_token(p)]

# visual injection on latent embedding inject = base + vis_injected * vis_emb.unsqueeze(-1)

# last embedding based on termination condition last_embedding = K(switch_count, c, ter_emb, inject) return last_embedding

##### 3.3. Entropy-Aware Visual Anchor Injection

To strengthen visual grounding during uncertain reasoning states, we introduce an entropy-aware visual anchor injection mechanism. Unlike continuous anchor blending, this strategy performs an injection at the first token of each highentropy phase (i.e., at the onset of latent reasoning). This design supplies a visual initialization cue that orients the model toward the visual semantic space without interfering with subsequent adaptive reasoning.

Let evis denotes the averaged embedding of pretrained visual special tokens (i.e., <|vision start|>, <|image pad|>, <|vision end|>). When the model detects an entropy rise above the threshold Hˆ and enters the first latent step t⋆ in this phase, the visual anchor is injected into the weighted embedding as:

t⋆[e(v)] + λ evis, (9)

e˜t⋆ = (1 − λ) Ev∼p

where λ ∈ [0,1] controls the strength of visual guidance. This one-time injection provides a visual grounding signal that helps stabilize the model’s reasoning trajectory in the multimodal semantic space. The model injects the visual anchor each time it enters a high-entropy phase to reinforce visual guidance.

#### 4. Experiments 4.1. Experimental Setup

Baselines. We evaluate LEAD on a set of representative MLRMs, including R1-Onevision-7B [82], Vision-R17B [22], VL-Rethinker-7B [58], VL-Cogito-7B [86], and OpenVLThinker-7B [12]. Additional results for different model scales are provided in Appendix A.

Evaluation Benchmarks. We conduct evaluations on both general and domain-specific multimodal reason-

- Table 1. Effect of visual anchor injection strength λ on overall performance. Scores are reported for MMHalu (ranging from 0 to

6) and Bingo (ranging from 1 to 5), while accuracy is reported for VStar and MMEval-Pro. Best results are highlighted in Bold.

Model λ VStar MMEval-Pro MMHalu Bingo

0 67.5 71.9 3.59 3.74 0.2 69.6 72.0 3.66 3.73 0.4 71.2 73.9 3.80 3.84 0.6 68.1 73.3 3.77 3.76

R1-Onevision-7B

0 79.1 72.7 3.69 3.68 0.2 80.1 73.9 3.78 3.70 0.4 81.7 75.1 3.89 3.77 0.6 79.6 74.5 3.83 3.75

Vision-R1-7B

ing benchmarks. For general evaluation, we consider two categories: (1) General Reasoning & Understanding (MMEval-Pro [20], MMVP [56], RealWorldQA [75], VMCBench [91], and VStar [73]) and (2) Hallucination Assessment (Bingo [10], MMHalu [51], and POPE [28]). For domain-specific evaluation, we assess performance on (1) Mathematical Reasoning (MathVision [59], MathVista [40], MathVerse [90], VisuLogic [80], Geometry3K [39] and Mathematics subset of MMK12 [43]) and (2) Scientific Reasoning (Physics, Chemistry and Biology subsets of MMK12).

Implementation Details. LEAD samples tokens in the output stage using the conventional discrete manner, with the examples illustrated using the greedy decoding strategy. Details of other methods are provided in Appendix B. For the Switch Count, we set the switching number Ct with a default maximum value of 5. Extensive experiments indicate that Cmax = 5 ensures stable and consistent generation.

##### 4.2. Ablation Study

Effect of Entropy Threshold. We experiment with different entropy thresholds to evaluate the effectiveness of the discrete–latent reasoning switching mechanism. As shown in Fig.5, dynamic thresholding consistently yields the best performance, improving MMHalu scores by +4.7% and +4.1% for R1-Onevision and Vision-R1, respectively, showing the advantage of LEAD’s adaptive switching strategy. In contrast, a large threshold forces the model to remain in discrete CoT reasoning, preventing it from leveraging exploratory latent reasoning. Conversely, a small threshold keeps the model in latent reasoning for too long, weakening the discrete convergence and increasing the risk of hallucination.

Effect of Switching Window Size. We examine the influence of the discrete reasoning window size on final performance. Fig.6 shows that performance improves as the window size grows up to 128, after which it begins to decline. A moderate window size encourages the model to remain briefly in discrete reasoning before switching, thereby

R1-Onevision-7B OpenVLThinker-7B VL-Cogito-7B Vision-R1-7B VL-Rethinker-7B

MMHalu

Bingo

3.9

4.4

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

3.8

4.2

3.7

4.0

Score

Score

3.6

3.8

3.5

3.6

3.4

3.4

0 0.3 0.5 1.0 2.0 ∞ Δ

0 0.3 0.5 1.0 2.0 ∞ Δ

Entropy Threshold

Entropy Threshold

- Figure 5. Comparisons of average score on MMHalu and Bingo datasets under different entropy thresholds. ∆ denotes the dynamic thresholding strategy in LEAD. ∞ keeps the model in standard discrete CoT reasoning, while 0 keeps it in latent reasoning.

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

- 3.4 R1-Onevision-7B Vision-R1-7B
- 3.5
- 3.6
- 3.7
- 3.8
- 3.9

4.0

(a)

Score

MMHalu

| | |
|---|---|
| | |
| | |
| | |
| | |

- 3.4 R1-Onevision-7B Vision-R1-7B
- 3.5
- 3.6
- 3.7
- 3.8
- 3.9

(b)

Score

Bingo

Size=64 Size=128 Size=256 Size=∞

- Figure 6. Comparisons of model performance under different persistence window sizes. (a) and (b) show model performance with varying window values on the MMHalu and Bingo datasets.

avoiding excessively frequent transitions. However, when the window size is too large, the model remains in discrete CoT-style reasoning for most of the inference process, reducing the benefits of latent reasoning. In the extreme case where the window size is set to ∞, the model switches back to discrete reasoning after its first latent reasoning turn and then remains in discrete mode permanently, causing performance to regress toward the level of standard CoT.

Effect of Visual Anchor Injections. We evaluate the impact of visual anchor injection strength on hallucination mitigation. Table 1 presents performance across different injection strengths. Performance improves as injection strength increases, reaching its peak at 0.4 across all datasets. By injecting a moderate amount of visual information during high-entropy reasoning steps, the model is encouraged to ground its latent reasoning process in visual evidence, helping maintain consistency between generated content and the underlying image. However, when the injection strength is too high, visual embedding begins to dominate the representation, diminishing the influence of linguistic context and leading to a slight performance drop.

Qualitative Analysis. We visualize the response of R1Onevision across different methods. Fig. 7 (a) shows that LEAD allocates relatively higher visual attention to queryrelevant regions compared to Baseline and MemVR. This

1.5

|When<br><br>0.35<br><br><think><br><br>1.00<br><br>observing<br><br>0.55<br><br>the<br><br>0.88<br><br>image 1.00 … straight<br><br>1.00<br><br>river<br><br>1.00<br><br>.<br><br>1.00<br><br>While<br><br>0.39<br><br>skyscraper<br><br>0.64<br><br>density<br><br>0.82<br><br>suggests<br><br>0.91<br><br>that<br><br>1.00<br><br>a<br><br>1.00|
|---|

| | |
|---|---|
| | |
| | |
| | |
| | |

Query: Describe this image. What’s the name of the city?

Answer: São Paulo

Then

corporate

tower

repetition

The

lower

part

1.2

[Figure 19]

0.39

0.21

0.15

0.09

0.35

0.31

0.12

Although

the

bridge

###### I

notice

0.21

0.15

0.03

0.29

0.24

###### TokenEntropy

|the<br><br>1.00<br><br>business 1.00 … bridge<br><br>1.00<br><br>structure<br><br>1.00<br><br>yet<br><br>0.40<br><br>familiar<br><br>0.55<br><br>spatial<br><br>0.83<br><br>cues<br><br>1.00<br><br>confirm<br><br>1.00<br><br>that 1.00 … indicates<br><br>1.00<br><br>less<br><br>1.00<br><br>structure<br><br>1.00<br><br>since<br><br>0.89<br><br>layout<br><br>1.00|
|---|

0.9

and

small

details

even

0.40

0.45

0.17

0.11

0.6

even

0.20

|</think><br><br>1.00<br><br>.<br><br>1.00<br><br>Paulo<br><br>1.00<br><br>São possible 1.00 <Answer> The city …<br><br>1.00<br><br>locations<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>0.68<br><br>should<br><br>1.00<br><br>be<br><br>1.00<br><br>the<br><br>0.87<br><br>stable<br><br>1.00<br><br>solution<br><br>1.00|
|---|

[Figure 20]

[Figure 21]

[Figure 22]

0.3

As

discussed

Base MemVR LEAD (Ours) (a) (b)

0.32

0.13

0.0

Figure 7. Qualitative visualization of LEAD under discrete and latent reasoning. (a) Comparisons of the average visual attention allocation across reasoning steps among Base, MemVR and our LEAD. (b) Example visualization of LEAD’s token-level probability distribution and entropy across reasoning steps. The token probabilities and corresponding entropies are shown at each step. The tokens highlighted in orange box correspond to those sampled in the final output sequence. More detailed visualizations are provided in Appendix D.

aligns with the injection of visual anchors, which reallocates the attention to task-related visual information and reduces attention to irrelevant tokens. Fig. 7 (b) presents the token probability distribution and token-level entropy across reasoning steps for LEAD. For clarity, we highlight the top three tokens. During latent reasoning, the token distribution appears to be more dispersed, corresponding to higher token entropy. In contrast, during discrete reasoning, the token distribution approaches a one-hot pattern with lower entropy, indicating deterministic reasoning.

Base VCD MemVR SID LEAD (Ours)

Fluency ↑

Fluency ↑

8.69

8.84

8.77

8.6 8.51

8.65

8.34

8.48

Grammar ↑

Natural ↑

Grammar ↑

Natural ↑

9.53

7.72

9.51

7.89

7.67

9.46

7.79

7.59

9.5

9.38

9.36 7.68

8.09

7.46 9.24

8.21

9.28

7.56

9.1 7.2

9.12 7.35

7.74

8.25

7.58

8.12

7.41 7.23

7.55

7.95

8.02

7.81

7.86

7.32

7.05

7.66

7.71

7.16

7.08 7.63

6.86 7.57

PPL1 ↓

PPL2 ↓

PPL2 ↓

PPL1 ↓

(a) R1-Onevision-7B

###### (b) Vision-R1-7B

- Figure 8. The average performance is evaluated on MMHalu using R1-Onevision-7B and Vision-R1-7B. PPL1 and PPL2 are calculated using gpt2, while the ratings for Grammar, Fluency and Naturalness are provided by GPT-5.

400 420 440 460 480 500

29.0

29.8

30.6

31.4

32.2

33.0

Average Token Length

Accuracy

LEAD (Ours)

VCD

Base

SID MemVR

LEAD (Ours) SID MemVR VCD Base

- Figure 9. Comparisons of accuracy and reasoning length across multiple hallucination mitigation methods. The x-axis represents the average reasoning length computed on the MathVision dataset with R1-Onevision-7B.

##### 4.3. Comparisons to State-of-the-Arts

Benchmark Evaluation. To evaluate the general image understanding, we compare models with the LEAD extension against several decoding methods, including VCD [25], MemVR [96], and SID [23], as shown in Table 2. Integrating LEAD as a plugin into R1-onevision results in an average improvement of +3.6% in the General reasoning and understanding tasks. It also achieves significant gains in hallucination metrics, with MMHalu and Bingo scores and increasing by +4.7% and +3.8%, respectively. These results indicate that LEAD is effective at reducing hallucinations in unstructured environments. As shown in Table 3, in domain-specific reasoning tasks, LEAD improves average accuracy by +2.0% on mathematics benchmarks and +3.2% on scientific benchmarks, demonstrating its effectiveness in structured and symbolic reasoning scenarios. Furthermore, the benefits of LEAD extend beyond the R1Onevision model, as other models also experience considerable enhancements.

GPT-5 Assisted Evaluation. To comprehensively assess the quality of the generated text, we employ the Perplexity (PPL) metric and utilize GPT-5 to evaluate grammar, fluency, and naturalness of the text. We conduct evaluations on the MMHalu dataset using R1-OneVision and Vision-R1. As demonstrated in Fig. 8, LEAD consistently preserves the quality of the generated text across multiple dimensions.

Reasoning Efficiency. We evaluate reasoning efficiency on MathVision using R1-Onevision, as shown in Fig.9. LEAD generates shorter reasoning length than the baselines while maintaining the highest accuracy. This efficiency gain is attributed to the latent reasoning phase, which allows the

- Table 2. Comparisons of different MLRMs with LEAD across general reasoning and hallucination benchmarks. Scores are reported for MMHalu (ranging from 0 to 6) and Bingo (ranging from 1 to 5), while accuracy is reported for all other benchmarks.

General Reasoning & Understanding Hallucination Benchmark Method

VStar ↑ RealWorldQA ↑ MMVP ↑ MMEval-Pro ↑ VMCBench ↑ MMHalu ↑ Bingo ↑ POPE-R ↑ POPE-P ↑ POPE-A ↑

R1-Onevision-7B 66.5 62.5 43.0 69.4 65.2 3.52 3.65 84.6 84.0 82.5 + VCD 67.1 62.6 42.9 69.8 66.0 3.55 3.61 84.4 83.8 82.3 + MemVR 69.6 64.3 44.5 71.3 67.5 3.69 3.68 82.3 85.0 83.5 + SID 70.2 65.2 43.2 71.0 67.8 3.70 3.65 85.0 84.7 81.9 + LEAD (Ours) 71.2 (+4.7) 66.4 (+3.9) 45.0 (+2.0) 73.9 (+4.5) 67.9 (+2.7) 3.80 (+4.7) 3.84 (+3.8) 85.9 (+1.3) 85.3 (+1.3) 83.9 (+1.4)

Vision-R1-7B 78.5 64.3 44.0 72.2 80.3 3.64 3.61 88.0 85.2 84.0

+ LEAD (Ours) 81.7 (+3.2) 67.5 (+3.2) 46.3 (+2.3) 75.1 (+2.9) 82.1 (+1.8) 3.89 (+4.1) 3.77 (+3.2) 91.4 (+3.4) 88.3 (+3.1) 87.7 (+3.7) VL-Rethinker-7B 67.6 69.3 42.0 73.2 73.9 4.06 3.67 85.5 81.8 82.8

- + LEAD (Ours) 70.1 (+2.5) 71.2 (+1.9) 46.6 (+4.6) 75.7 (+2.5) 75.2 (+1.3) 4.27 (+3.5) 3.85 (+3.6) 86.2 (+0.7) 85.1 (+3.3) 84.9 (+2.1) VL-Cogito-7B 79.6 68.1 40.0 73.0 73.2 3.95 3.63 85.0 85.0 84.1

+ LEAD (Ours) 81.7 (+2.1) 69.2 (+1.1) 42.0 (+2.0) 75.6 (+2.6) 75.6 (+2.4) 4.13 (+3.0) 3.80 (+2.8) 86.3 (+1.3) 86.6 (+1.6) 86.1 (+2.0) OpenVLThinker-7B 68.1 62.3 46.5 71.5 80.3 3.59 3.50 82.4 82.5 79.1

- + LEAD (Ours) 70.2 (+2.1) 65.3 (+3.0) 47.2 (+0.7) 73.5 (+2.0) 81.3 (+1.0) 3.76 (+2.8) 3.71 (+4.2) 84.1 (+1.7) 83.5 (+1.0) 80.2 (+1.1)

Table 3. Comparisons of different MLRMs with LEAD across mathematical and scientific visual reasoning benchmarks.

Mathematical Reasoning Scientific Reasoning Method

MathVision↑ MathVista↑ MathVerse↑ VisuLogic↑ Geometry3K↑ MMK12-Math↑ MMK12-Phys↑ MMK12-Chem↑ MMK12-Bio↑ R1-Onevision-7B 29.9 64.1 46.4 24.9 57.9 44.8 33.8 39.8 40.8

- + LEAD (Ours) 32.4 (+2.5) 66.4 (+2.3) 47.3 (+0.9) 26.1 (+1.2) 61.2 (+3.3) 46.7 (+1.9) 36.1 (+2.3) 43.2 (+3.4) 44.8 (+4.0) Vision-R1-7B 27.2 73.5 52.4 26.4 67.0 52.1 47.3 55.4 57.9

+ LEAD (Ours) 29.7 (+2.5) 74.9 (+1.4) 54.5 (+2.1) 27.9 (+1.5) 68.3 (+1.3) 53.9 (+1.8) 49.2 (+1.9) 56.6 (+1.2) 58.6 (+0.7) VL-Rethinker-7B 32.3 74.9 54.2 27.3 67.7 51.3 47.2 57.4 64.8

- + LEAD (Ours) 33.1 (+0.8) 75.6 (+0.7) 54.9 (+0.7) 28.5 (+1.2) 68.9 (+1.2) 52.4 (+1.1) 49.1 (+1.9) 60.6 (+3.2) 65.6 (+0.8) VL-Cogito-7B 30.7 74.8 53.3 28.2 68.7 63.7 43.2 57.5 61.3

+ LEAD (Ours) 32.4 (+1.7) 76.3 (+1.5) 55.1 (+1.8) 28.9 (+0.7) 69.1 (+0.4) 65.1 (+1.4) 44.6 (+1.4) 58.4 (+0.9) 64.6 (+3.3)

Base VCD MemVR SID

LEAD (Ours)

RealWorldQA

MathVista

78

75

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

74

72

Pass@kAccuracy

Pass@kAccuracy

70

69

66

66

62

63

58

60

4 8 16 32

4 8 16 32

k

k

Figure 10. Pass@k accuracy evaluation of R1-Onevision-7B on sampled data of RealworldQA and MathVista, illustrating results for k ∈ [4, 32].

model to retain multiple reasoning hypotheses at each step and reach the solutions with fewer generated tokens.

Pass@k Performance. In addition to Pass@1, we evaluate Pass@k performance for k ∈ [1,64] on R1-Onevision and compare it with other methods. We show results for k ∈ [4,32] for better illustration (See Appendix C for the full results). As shown in Figure 10, LEAD reaches its peak accuracy at smaller k values than the baselines, in-

dicating higher sample efficiency. In addition to requiring fewer samples to reach peak accuracy, LEAD also shows a steeper increase in Pass@k at small k and attains a higher final accuracy than VCD and MemVR. This indicates greater diversity of LEAD in reasoning and greater correctness.

#### 5. Conclusion

In this work, we examine token-level uncertainty and reveal that transition words frequently coincide with highentropy reasoning states, which exhibit a strong association with hallucination-prone behaviors. Additionally, we find that high-entropy tokens linked to hallucinations tend to receive markedly lower visual attention, indicating that the model tends to overlook visual information under uncertainty. Motivated by these observations, we present LEAD, a lightweight and plug-and-play decoding framework that adaptively alternates between discrete and latent semantic representations, while incorporating visual guidance during high-uncertainty phases to enhance reasoning stability. Extensive evaluations on both general-purpose and scientific benchmarks demonstrate that LEAD consistently strengthens reasoning reliability and significantly reduces multimodal hallucinations.

#### Acknowledgments

This research was supported by the Australian Government Research Training Program (RTP) Scholarship.

#### References

- [1] Inclusion AI, Fudong Wang, Jiajia Liu, Jingdong Chen, Jun Zhou, Kaixiang Ji, Lixiang Ru, Qingpei Guo, Ruobing Zheng, Tianqi Li, et al. M2-reasoning: Empowering mllms with unified general and spatial reasoning. arXiv preprint arXiv:2507.08306, 2025. 3
- [2] Edward Y. Chang, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. CoRR, abs/2502.03373, 2025. 1
- [3] Jierun Chen, Tiezheng Yu, Haoli Bai, Lewei Yao, Jiannan Wu, Kaican Li, Fei Mi, Chaofan Tao, Lei Zhu, Manyi Zhang, et al. The synergy dilemma of long-cot sft and rl: Investigating post-training techniques for reasoning vlms. arXiv preprint arXiv:2507.07562, 2025. 3
- [4] Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. arXiv preprint arXiv:2506.04207, 2025. 1, 3
- [5] Yang Chen, Yufan Shen, Wenxuan Huang, Sheng Zhou, Qunshu Lin, Xinyu Cai, Zhi Yu, Jiajun Bu, Botian Shi, and Yu Qiao. Learning only with images: Visual reinforcement learning with reasoning, rendering, and visual feedback. arXiv preprint arXiv:2507.20766, 2025. 3
- [6] Zhangquan Chen, Ruihui Zhao, Chuwei Luo, Mingze Sun, Xinlei Yu, Yangyang Kang, and Ruqi Huang. Sifthinker: Spatially-aware image focus for visual reasoning. arXiv preprint arXiv:2508.06259, 2025. 3
- [7] Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758, 2025. 2
- [8] Jiahao Cheng, Tiancheng Su, Jia Yuan, Guoxiu He, Jiawei Liu, Xinqi Tao, Jingwen Xie, and Huaxia Li. Chain-ofthought prompting obscures hallucination cues in large language models: An empirical evaluation. arXiv preprint arXiv:2506.17088, 2025. 3
- [9] Jiwan Chung, Neel Joshi, Pratyusha Sharma, Youngjae Yu, and Vibhav Vineet. What mllms learn about when they learn about multimodal reasoning: Perception, reasoning, or their integration? arXiv preprint arXiv:2510.01719, 2025. 1, 3
- [10] Chenhang Cui, Yiyang Zhou, Xinyu Yang, Shirley Wu, Linjun Zhang, James Zou, and Huaxiu Yao. Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges. arXiv preprint arXiv:2311.03287, 2023. 6
- [11] Jingcheng Deng, Liang Pang, Zihao Wei, Shichen Xu, Zenghao Duan, Kun Xu, Yang Song, Huawei Shen, and Xueqi Cheng. Latent reasoning in llms as a vocabulary-space superposition. arXiv preprint arXiv:2510.15522, 2025. 3
- [12] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: Complex visionlanguage reasoning via iterative sft-rl cycles. In The Thirty-

- ninth Annual Conference on Neural Information Processing Systems, 2025. 1, 5
- [13] Yizhuo Ding, Mingkang Chen, Zhibang Feng, Tong Xiao, Wanying Qu, Wenqi Shao, and Yanwei Fu. Vtperception-r1: Enhancing multimodal reasoning via explicit visual and textual perceptual grounding. arXiv preprint arXiv:2509.24776,

2025. 1, 3

- [14] Bowen Dong, Minheng Ni, Zitong Huang, Guanglei Yang, Wangmeng Zuo, and Lei Zhang. Mirage: Assessing hallucination in multimodal reasoning chains of mllm. arXiv preprint arXiv:2505.24238, 2025. 1, 3
- [15] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. arXiv preprint arXiv:2411.14432, 2024. 3
- [16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1
- [17] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024. 3
- [18] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025. 3
- [19] Chen Huang, Wei Lu, and Wenxuan Zhang. Pear: Phase entropy aware reward for efficient reasoning. arXiv preprint arXiv:2510.08026, 2025. 3
- [20] Jinsheng Huang, Liang Chen, Taian Guo, Fu Zeng, Yusheng Zhao, Bohan Wu, Ye Yuan, Haozhe Zhao, Zhihui Guo, Yichi Zhang, et al. Mmevalpro: Calibrating multimodal benchmarks towards trustworthy and efficient evaluation. arXiv preprint arXiv:2407.00468, 2024. 6
- [21] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multimodal large language models via over-trust penalty and retrospection-allocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13418–13427, 2024. 1, 3
- [22] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025. 5
- [23] Fushuo Huo, Wenchao Xu, Zhong Zhang, Haozhao Wang, Zhicheng Chen, and Peilin Zhao. Self-introspective decoding: Alleviating hallucinations for large vision-language models. arXiv preprint arXiv:2408.02032, 2024. 1, 3, 7
- [24] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13872–13882, 2024. 1, 3

- [25] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13872–13882, 2024. 7
- [26] Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Yuming Jiang, Hang Zhang, Xin Li, Lidong Bing, et al. Mmr1: Enhancing multimodal reasoning with variance-aware sampling and open resources. arXiv preprint arXiv:2509.21268, 2025. 1, 3
- [27] Junyi Li and Hwee Tou Ng. The hallucination dilemma: Factuality-aware reinforcement learning for large reasoning models. arXiv preprint arXiv:2505.24630, 2025. 3
- [28] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, 2023. 6
- [29] Zejun Li, Yingxiu Zhao, Jiwen Zhang, Siyuan Wang, Yang Yao, Runzhou Zhao, Jun Song, Bo Zheng, and Zhongyu Wei. Mixture-of-visual-thoughts: Exploring context-adaptive reasoning mode selection for general visual reasoning. arXiv preprint arXiv:2509.22746, 2025. 3
- [30] Qian Liang, Yujia Wu, Kuncheng Li, Jiwei Wei, Shiyuan He, Jinyu Guo, and Ning Xie. Mm-r1: Unleashing the power of unified multimodal large language models for personalized image generation. arXiv preprint arXiv:2508.11433, 2025. 3
- [31] Yiqing Liang, Jielin Qiu, Wenhao Ding, Zuxin Liu, James Tompkin, Mengdi Xu, Mengzhou Xia, Zhengzhong Tu, Laixi Shi, and Jiacheng Zhu. Modomodo: Multi-domain data mixtures for multimodal llm reinforcement learning. arXiv preprint arXiv:2505.24871, 2025. 3
- [32] Ming Lingfeng, Li Yadong, Chen Song, Xu Jianhua, Zhou Zenan, and Chen Weipeng. Ocean-r1: An open and generalizable large vision-language model enhanced by reinforcement learning, 2025. Accessed: 2025-04-03. 3
- [33] Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, and Sheng Liu. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. arXiv preprint arXiv:2505.21523, 2025. 1, 3
- [34] Shi Liu, Kecheng Zheng, and Wei Chen. Paying more attention to image: A training-free method for alleviating hallucination in lvlms. arXiv preprint arXiv:2407.21771, 2024. 3
- [35] Yue Liu, Shengfang Zhai, Mingzhe Du, Yulin Chen, Tri Cao, Hongcheng Gao, Cheng Wang, Xinfeng Li, Kun Wang, Junfeng Fang, Jiaheng Zhang, and Bryan Hooi. Guardreasonervl: Safeguarding vlms via reinforced reasoning. arXiv preprint arXiv:2505.11049, 2025. 3
- [36] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025. 3
- [37] Haolang Lu, Bolun Chu, WeiYe Fu, Guoshun Nan, Junning Liu, Minghui Pan, Qiankun Li, Yi Yu, Hua Wang,

- and Kun Wang. Mitigating hallucination in multimodal reasoning via functional attention control. arXiv preprint arXiv:2510.10285, 2025. 1, 3
- [38] Haolang Lu, Yilian Liu, Jingxin Xu, Guoshun Nan, Yuanlong Yu, Zhican Chen, and Kun Wang. Auditing metacognitive hallucinations in reasoning large language models. arXiv preprint arXiv:2505.13143, 2025. 3
- [39] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021. 6
- [40] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR),

2024. 6

- [41] Fan Ma, Xiaojie Jin, Heng Wang, Yuchen Xian, Jiashi Feng, and Yi Yang. Vista-llama: Reducing hallucination in video language models via equal distance to visual tokens. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13151–13160, 2024. 3
- [42] Weijia Mao, Zhenheng Yang, and Mike Zheng Shou. Unirl: Self-improving unified multimodal models via supervised and reinforcement learning. arXiv preprint arXiv:2505.23380, 2025. 3
- [43] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 6
- [44] Minheng Ni, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Kevin Lin, Wangmeng Zuo, and Lijuan Wang. Point-rft: Improving multimodal reasoning with visually grounded reinforcement finetuning. arXiv preprint arXiv:2505.19702,

2025. 3

- [45] OpenAI. Learning to reason with LLMs. 2024. 1
- [46] Chen Qian, Dongrui Liu, Haochen Wen, Zhen Bai, Yong Liu, and Jing Shao. Demystifying reasoning dynamics with mutual information: Thinking tokens are information peaks in llm reasoning. arXiv preprint arXiv:2506.02867, 2025. 3
- [47] Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan, Sitong Zhou, Guanting Dong, Yuchen Zeng, Yida Xu, et al. We-math 2.0: A versatile mathbook system for incentivizing visual mathematical reasoning. arXiv preprint arXiv:2508.10433, 2025. 3
- [48] Dachuan Shi, Abedelkadir Asi, Keying Li, Xiangchi Yuan, Leyan Pan, Wenke Lee, and Wen Xiao. Swireasoning: Switch-thinking in latent and explicit for pareto-superior reasoning llms. arXiv preprint arXiv:2510.05069, 2025. 3
- [49] Linxin Song, Taiwei Shi, and Jieyu Zhao. The hallucination tax of reinforcement finetuning. arXiv preprint arXiv:2505.13988, 2025. 3
- [50] Guohao Sun, Hang Hua, Jian Wang, Jiebo Luo, Sohail Dianat, Majid Rabbani, Raghuveer Rao, and Zhiqiang Tao. Latent chain-of-thought for visual reasoning. arXiv preprint arXiv:2510.23925, 2025. 1

- [51] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. In Annual Meeting of the Association for Computational Linguistics,

2024. 6

- [52] Zhongxiang Sun, Qipeng Wang, Haoyu Wang, Xiao Zhang, and Jun Xu. Detection and mitigation of hallucination in large reasoning models: A mechanistic perspective. arXiv preprint arXiv:2505.12886, 2025. 3
- [53] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025. 3
- [54] Feilong Tang, Chengzhi Liu, Zhongxing Xu, Ming Hu, Zile Huang, Haochen Xue, Ziyang Chen, Zelin Peng, Zhiwei Yang, Sijin Zhou, et al. Seeing far and clearly: Mitigating hallucinations in mllms with attention causal decoding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26147–26159, 2025. 3
- [55] Xinyu Tian, Shu Zou, Zhaoyuan Yang, Mengqi He, Fabian Waschkowski, Lukas Wesemann, Peter Tu, and Jing Zhang. More thought, less accuracy? on the dual nature of reasoning in vision-language models. arXiv preprint arXiv:2509.25848, 2025. 1, 3
- [56] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024. 6
- [57] Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, et al. Srpo: Enhancing multimodal llm reasoning via reflection-aware reinforcement learning. arXiv preprint arXiv:2506.01713, 2025. 3
- [58] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025. 1, 3, 5
- [59] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024. 6
- [60] Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, and Feng Zhao. Vrag-rl: Empower vision-perception-based rag for visually rich information understanding via iterative reasoning with reinforcement learning. arXiv preprint arXiv:2505.22019, 2025. 3
- [61] Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025. 2
- [62] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Sheng-

- long Ye, Xizhou Zhu, et al. Visualprm: An effective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291, 2025. 3
- [63] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 3
- [64] Xintong Wang, Jingheng Pan, Liang Ding, and Chris Biemann. Mitigating hallucinations in large vision-language models with instruction contrastive decoding. arXiv preprint arXiv:2403.18715, 2024. 3
- [65] Xiyao Wang, Zhengyuan Yang, Chao Feng, Yongyuan Liang, Yuhang Zhou, Xiaoyu Liu, Ziyi Zang, Ming Li, Chung-Ching Lin, Kevin Lin, et al. Vicrit: A verifiable reinforcement learning proxy task for visual perception in vlms. arXiv preprint arXiv:2506.10128, 2025. 3
- [66] Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. arXiv preprint arXiv:2504.07934, 2025. 3
- [67] Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, et al. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448, 2025. 3
- [68] Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, and Lichao Sun. First sft, second rl, third upt: Continual improving multi-modal llm reasoning via unsupervised post-training. In The Thirty-ninth Annual Conference on Neural Information Processing Systems. 3
- [69] Lai Wei, Yuting Li, Kaipeng Zheng, Chen Wang, Yue Wang, Linghe Kong, Lichao Sun, and Weiran Huang. Advancing multimodal reasoning via reinforcement learning with cold start. arXiv preprint arXiv:2505.22334, 2025. 3
- [70] Yana Wei, Liang Zhao, Jianjian Sun, Kangheng Lin, Jisheng Yin, Jingcheng Hu, Yinmin Zhang, En Yu, Haoran Lv, Zejia Weng, et al. Open vision reasoner: Transferring linguistic cognitive behavior for visual reasoning. arXiv preprint arXiv:2507.05255, 2025. 3
- [71] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965,

2025. 3

- [72] Junhong Wu, Jinliang Lu, Zixuan Ren, Gangqiang Hu, Zhi Wu, Dai Dai, and Hua Wu. Llms are single-threaded reasoners: Demystifying the working mechanism of soft thinking. arXiv preprint arXiv:2508.03440, 2025. 3
- [73] Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084–13094, 2024. 6
- [74] Zijian Wu, Jinjie Ni, Xiangyan Liu, Zichen Liu, Hang Yan, and Michael Qizhe Shieh. Synthrl: Scaling visual reasoning with verifiable data synthesis. arXiv preprint arXiv:2506.02096, 2025. 3

- [75] X.AI. Grok-2 beta release, 2024. Accessed: 2024. 6
- [76] Tong Xiao, Xin Xu, Zhenya Huang, Hongyu Gao, Quan Liu, Qi Liu, and Enhong Chen. Advancing multimodal reasoning capabilities of multimodal large language models via visual perception reward. arXiv preprint arXiv:2506.07218, 2025. 3
- [77] Wenyi Xiao, Leilei Gan, Weilong Dai, Wanggui He, Ziwei Huang, Haoyuan Li, Fangxun Shu, Zhelun Yu, Peng Zhang, Hao Jiang, et al. Fast-slow thinking for large vision-language model reasoning. arXiv preprint arXiv:2504.18458, 2025. 3
- [78] Zhiyou Xiao, Qinhan Yu, Binghui Li, Geng Chen, Chong Chen, and Wentao Zhang. M2io-r1: An efficient rl-enhanced reasoning framework for multimodal retrieval augmented multimodal generation. arXiv preprint arXiv:2508.06328,

2025. 3

- [79] Yun Xing, Yiheng Li, Ivan Laptev, and Shijian Lu. Mitigating object hallucination via concentric causal attention. arXiv preprint arXiv:2410.15926, 2024. 3
- [80] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025. 6
- [81] Zhongxing Xu, Feilong Tang, Zhe Chen, Yingxue Su, Zhiyi Zhao, Ge Zhang, Jionglong Su, and Zongyuan Ge. Toward modality gap: Vision prototype learning for weaklysupervised semantic segmentation with clip. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9023–9031, 2025. 3
- [82] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025. 1, 5
- [83] Zijun Yao, Yantao Liu, Yanxu Chen, Jianhui Chen, Junfeng Fang, Lei Hou, Juanzi Li, and Tat-Seng Chua. Are reasoning models more prone to hallucination? arXiv preprint arXiv:2505.23646, 2025. 3
- [84] Hao Yin, Guangzong Si, and Zilei Wang. Clearsight: Visual signal enhancement for object hallucination mitigation in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 1, 3
- [85] En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954,

2025. 1, 3

- [86] Ruifeng Yuan, Chenghao Xiao, Sicong Leng, Jianyu Wang, Long Li, Weiwen Xu, Hou Pong Chan, Deli Zhao, Tingyang Xu, Zhongyu Wei, et al. Vl-cogito: Progressive curriculum reinforcement learning for advanced multimodal reasoning. arXiv preprint arXiv:2507.22607, 2025. 1, 5
- [87] Liu Yuqi, Peng Bohao, Zhong Zhisheng, Yue Zihao, Lu Fanbin, Yu Bei, and Jia Jiaya. Seg-zero: Reasoning-chain guided segmentation via cognitive reinforcement, 2025. 3

- [88] Ce Zhang, Zifu Wan, Zhehan Kan, Martin Q Ma, Simon Stepputtis, Deva Ramanan, Russ Salakhutdinov, LouisPhilippe Morency, Katia Sycara, and Yaqi Xie. Selfcorrecting decoding with generative feedback for mitigating hallucinations in large vision-language models. arXiv preprint arXiv:2502.06130, 2025. 3
- [89] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025. 3
- [90] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision. Springer, 2024. 6
- [91] Yuhui Zhang, Yuchang Su, Yiming Liu, Xiaohan Wang, James Burgess, Elaine Sui, Chenyu Wang, Josiah Aklilu, Alejandro Lozano, Anjiang Wei, et al. Automated generation of challenging multiple-choice questions for vision language model evaluation. arXiv preprint arXiv:2501.03225, 2025. 6
- [92] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025. 3
- [93] Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, and Xin Eric Wang. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space. arXiv preprint arXiv:2505.15778,

2025. 3

- [94] Linghao Zhu, Yiran Guan, Dingkang Liang, Jianzhong Ju, Zhenbo Luo, Bin Qin, Jian Luan, Yuliang Liu, and Xiang Bai. Shuffle-r1: Efficient rl framework for multimodal large language models via data-centric dynamic shuffle. arXiv preprint arXiv:2508.05612, 2025. 3
- [95] Yufan Zhuang, Liyuan Liu, Chandan Singh, Jingbo Shang, and Jianfeng Gao. Mixture of inputs: Text generation beyond discrete token sampling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems. 3
- [96] Xin Zou, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Kening Zheng, Sirui Huang, Junkai Chen, Peijie Jiang, Jia Liu, Chang Tang, et al. Look twice before you answer: Memory-space visual retracing for hallucination mitigation in multimodal large language models. arXiv preprint arXiv:2410.03577, 2024. 7

