# arXiv:2601.10129v1[cs.CV]15Jan2026

## LaViT: Aligning Latent Visual Thoughts for Multi-modal Reasoning

### Linquan Wu*1, Tianxiang Jiang*2, Yifei Dong3, Haoyu Yang4, Fengji Zhang1, Shichang Meng1, Ai Xuan1, Linqi Song 1, Jacky Keung1

1City University of Hong Kong, 2University of Science and Technology of China, 3Utrecht University, 4University of Electronic Science and Technology of China

https://github.com/Svardfox/LaViT

### Abstract

Current multimodal latent reasoning often relies on external supervision (e.g., auxiliary images), ignoring intrinsic visual attention dynamics. In this work, we identify a critical Perception Gap in distillation: student models frequently mimic a teacher’s textual output while attending to fundamentally divergent visual regions, effectively relying on language priors rather than grounded perception. To bridge this, we propose LaViT, a framework that aligns latent visual thoughts rather than static embeddings. LaViT compels the student to autoregressively reconstruct the teacher’s visual semantics and attention trajectories prior to text generation, employing a curriculum sensory gating mechanism to prevent shortcut learning. Extensive experiments show that LaViT significantly enhances visual grounding, achieving up to +16.9% gains on complex reasoning tasks and enabling a compact 3B model to outperform larger open-source variants and proprietary models like GPT-4o.

### 1 Introduction

Multimodal Large Language Models (MLLMs) have advanced rapidly in recent years (Bai et al., 2025a; Comanici et al., 2025; Team, 2025), early multimodal reasoning models primarily thinking about images, captioning visual inputs into text and reasoning mainly in the language space via explicit chains of thought (Huang et al., 2025; Yang et al., 2025a; Shen et al., 2025a). Recent work instead emphasizes thinking with images, more tightly integrating visual evidence into the reasoning process (Hu et al., 2024; Zheng et al., 2025; Su et al., 2025b), leading to improved performance on complex visual reasoning tasks (Fu et al., 2024; Wu and Xie, 2024; Zhang et al., 2024).

* Equal contribution.

Subsequently, latent reasoning (Hao et al., 2024) has emerged as a complementary direction, compressing intermediate reasoning into continuous hidden states rather than explicit CoT. This paradigm has been extended to MLLMs to model abstract visual thoughts within latent tokens (Yang et al., 2025b; Li et al., 2025a). While effective, existing methods largely rely on manually designed visual supervision, such as auxiliary images or annotated regions, leaving intrinsic visual attention dynamics during reasoning unexplored.

These limitations motivate the use of knowledge distillation (Hinton et al., 2015) as a lens to analyze and transfer visual reasoning behaviors in MLLMs. Distilling a high-capacity teacher into a compact student enables us to probe not only what knowledge is transferred, but also how visual reasoning is internally realized. However, existing multimodal distillation methods mainly align final textual outputs or distributions (Cai et al., 2025; Shu et al., 2024), implicitly assuming that reproducing answers or static representations suffices to inherit multimodal reasoning ability.

To examine this assumption, we conduct empirical analyses, revealing a pronounced mismatch between textual alignment and visual reasoning: (I) Correct multimodal reasoning is causally constrained by focused visual attention: when models fail to attend to relevant regions, hallucinated or unreliable responses emerge. (II) Even when student models closely match teacher outputs under standard distillation, their visual attention trajectories can diverge substantially, particularly for tasks requiring fine-grained visual grounding. Together, these findings expose a fundamental Perception Gap in multimodal distillation: students often learn what to say without learning where to look, instead relying on language priors rather than grounded visual evidence.

Motivated by this insight, we propose LaViT, a distillation framework that aligns latent visual

[Figure 1]

Previous Training

Previous Training

[Figure 2]

- 1. Identify Visual Contexts （𝑽sem）

[Figure 3]

| | |
|---|---|
| | |

[Figure 4]

- 2. Look at A and B （𝑽traj）

[Figure 5]

1. Answer: B

[Figure 6]

Teacher Model

3. Answer：B

Teacher Model

[Figure 7]

Question: Which point is closer to the camera, Point

[Figure 8]

[Figure 9]

[Figure 10]

Answer: B

[Figure 11]

[Figure 12]

Answer: A

A or Point B?

Aligning Latent Visual Thoughts

Text-Only Mimicry

Figure 1: Conceptual Illustration of Our Proposed Method LaViT.

thoughts rather than static visual embeddings. LaViT trains the student to autoregressively generate continuous latent tokens that reconstruct the teacher’s internal visual semantics and attention trajectories prior to textual response generation, explicitly transferring both what visual concepts to encode and where to attend during reasoning.

generation, exemplified by Chameleon’s unified tokens (Team, 2024), MVoT’s multimodal trajectories (Li et al., 2025b), and other general-purpose frameworks (Tong et al., 2025b; Deng et al., 2025; Gu et al., 2025).

##### 2.2 Latent Reasoning

To prevent shortcut learning through direct access to visual features, we introduce Curriculum Sensory Gating, which progressively restricts and then relaxes visual input during training. This strategy enforces a latent bottleneck early on, compelling reliance on latent visual reasoning while avoiding training–inference mismatch.

Recent advances have shifted the reasoning paradigm from discrete token sequences to continuous hidden states, effectively enhancing both computational efficiency and flexibility (Hao et al., 2024; Shen et al., 2025b; Wei et al., 2025). Extending this concept to multimodal learning, current MLLMs align specialized latent tokens with visual embeddings derived from auxiliary supervision signals, such as helper images (Yang et al., 2025b) or annotated bounding boxes (Li et al., 2025a). To further improve grounding, CoVT (Qin et al., 2025) integrates fine-grained perceptual priors from models like DINO (Oquab et al., 2023) and SAM (Kirillov et al., 2023), while other approaches explore interleaved patterns to mimic internal visual imagination (Tong et al., 2025a; Wang et al., 2025b). However, these existing methods primarily constrain latent tokens using static encoder features, critically overlooking the dynamic guidance offered by attention maps.

Extensive experiments demonstrate that LaViT substantially improves both visual grounding and multimodal reasoning. LaViT-3B achieves up to +5.0% gains on fine-grained perception benchmarks MMVP (Tong et al., 2024) and substantial improvement on BLINK (Fu et al., 2024), outperforming strong baselines and rivaling or surpassing 7B models and proprietary GPT-4o.

### 2 Related Work

##### 2.1 Visual Chain-of-Thought

Originating from text-only LLMs (Wei et al., 2022), Chain-of-Thought (CoT) has expanded to multimodal contexts (Shao et al., 2024). Following DeepSeek-R1 (Guo et al., 2025), recent works enhance multi-step visual reasoning via RL-style optimization (Huang et al., 2025; Yang et al., 2025a; Shen et al., 2025a; Feng et al., 2025a; Jiang et al., 2025); however, these methods primarily rely on indirect textual proxies rather than intrinsic visual understanding. Conversely, a parallel stream shifts to thinking with images by orchestrating tools (Zhang et al., 2025a; Wu et al., 2025; Su et al., 2025a; Wang et al., 2025a; Zhang et al., 2025b), utilizing executable programs (Hu et al., 2024) or iterative region grounding (Zheng et al., 2025). Furthermore, unified architectures now support interleaved

##### 2.3 Knowledge Distillation

Knowledge distillation (Hinton et al., 2015), which transfers capabilities from a high-capacity teacher to a compact student, has been widely adopted in LLMs via logit matching (Sun et al., 2019; Jiao et al., 2020). Extending this paradigm to multimodal models, DistillVLM (Fang et al., 2021) performs transformer distillation by using an MSE loss to match the teacher and student’s hidden attention distributions and feature maps. In contrast, MAD (Wang et al., 2022) emphasizes aligning visual and textual token features between teacher and student, leveraging token selection to guide

the matching. More recent research explores distillation tailored to MLLMs for specific downstream tasks, including visual grounding (Cai et al., 2025; Feng et al., 2025b) and compositional learning (Kim et al., 2025).

### 3 Empirical Analysis of Perception Gap

We conduct a pilot study to quantify the misalignment between textual generation and visual attention, centered on two research questions: (RQ1) Is correct visual reasoning causally linked to focused visual attention? (i.e., Does looking at the right place precondition the right answer?) (RQ2) Does a significant “alignment gap” exist in the visual trajectories between teacher and student models, even when their textual outputs are similar?

- 3.1 Visual Attention Dictates Reasoning Bounds

Definition 3.1 (Visual Focusing Score). Let I

be the image and Bgt the target bounding box. Given the model’s aggregated attention trajectory Atraj ∈ RH×W, which accumulates attention weights across all layers and heads, the visual focusing score Sfocus is defined as:

Atraj(u,v) (u,v)∈I Atraj(u,v)

Sfocus = (u,v)∈Bgt

(1)

where Atraj(u,v) denotes the attention intensity at spatial coordinate (u,v). The denominator represents the total attention mass distributed across the entire image I. A larger Sfocus indicates a stronger dependency of the reasoning process on the verified visual evidence, implying that the model is actively “looking” at the semantically correct region rather than relying on language priors or blind guessing.

Building upon the above metric, we analyze the attention trajectories of Qwen2.5-VL-32B (Bai et al., 2025b) on 1,000 randomly sampled instances from the Visual-CoT (Shao et al., 2024). As illustrated in Figure 2, reasoning outcomes are strictly constrained by the intensity of visual attention:

• Monotonic Performance Gain: We observe that reasoning accuracy improves monotonically as the Sfocus threshold increases. Statistical analysis confirms that correct samples maintain a significantly higher average Sfocus (15.89%) compared to incorrect ones (11.84%), indicating a substantial relative gap of ∼34%. This validates that higher visual energy is a strong predictor of reasoning success.

[Figure 13]

Figure 2: Impact of Visual Attention on Reasoning Accuracy. The monotonic increase in accuracy with higher Visual Focusing Score (Sfocus) thresholds validates that effective visual grounding is a prerequisite for correct reasoning.

##### • Visual Absence and Hallucination: Con-

versely, in samples with negligible Sfocus (< 1%), we predominantly observe responses that are completely irrelevant to the visual content or contain severe hallucinations. This pattern suggests that without active visual grounding, the model relies on language priors to blindly guess, which proves to be a highly unreliable strategy for complex visual tasks.

|Observation 1: Visual attention is determinative, not merely interpretative. The strict positive correlation confirms that focused visual grounding (Sfocus) is a necessary condition for reasoning success. Models cannot reason correctly without “looking” at the right evidence, effectively ruling out blind guessing as a viable strategy.|
|---|

3.2 Perception Gap between Teachers and Students

Given the critical role of visual attention, we investigate whether standard Supervised FineTuning (SFT) enables student models to inherit the teacher’s visual thinking process. This analysis is conducted on the same 1,000 samples following the experimental setup in Section 3.1.

Analysis of Attention Divergence. We fed identical reasoning prompts to both the Teacher (MT) and the Student (MS). We categorized the generated tokens into three groups based on their semantic reliance on visual evidence: Functional (e.g., stop words), Object (nouns), and Attribute (adjectives, spatial relations). We then computed the Kullback-Leibler (KL) divergence between their normalized attention maps AT and AS, alongside the Cosine Distance of their hidden states.

[Figure 14]

Figure 3: The Perception-Reasoning Gap. While the student aligns closely with the teacher in textual representations (stable Cosine Distance), their visual attention trajectories diverge significantly on attribute-heavy tokens (rising KL Divergence). This reveals that textual mimicry does not imply visual grounding.

As shown in Figure 3, we observe a decoupling between textual alignment and visual attention:

- • Attention Drift on Visual Concepts: The attention divergence exhibits a distinct monotonic increase as the semantic reliance on vision deepens. While Functional tokens maintain a lower average divergence (µ = 1.11), Attribute tokens, which require precise visual grounding, show the highest misalignment (µ ≈ 1.39). This indicates that when describing fine-grained details (e.g., color, texture), the student struggles to focus on the same regions as the teacher.
- • The “Blind Guessing” Phenomenon: Crucially, while the attention divergence surges, the Cosine Distance of the hidden states remains relatively stable across categories (ranging from 0.52 to 0.55). This implies that the student can mimic the teacher’s textual representations (learning what to say) without correctly aligning its visual attention (learning where to look), effectively relying on language priors rather than active observation.

|Observation 2: Textual mimicry does not guarantee visual understanding. SFT trains the student to reproduce the teacher’s words but fails to transfer the underlying visual trajectory. This “Perception Gap” suggests that the student model is often “guessing” based on language context rather than actively “observing” the image.|
|---|

### 4 Method

##### 4.1 LaViT-SFT-15K We construct LaViT-SFT-15K, comprising 15K

tuples ⟨I,Q,A,Atraj,Vsem⟩, to distill the Internal Cognitive States of the teacher (Qwen2.5-VL32B (Bai et al., 2025b)). Unlike tool-based approaches, we directly extract intrinsic reasoning traces. Data quality is enforced via a Three-Stage Filtering pipeline: (1) Correctness: retaining samples matching ground truth; (2) Difficulty: removing instances solvable by a text-only model; and (3) Alignment: rejecting samples with < 20% aggregated attention mass falling within the target regions delineated by the ground-truth bounding box annotations in Visual-CoT (Shao et al., 2024), thereby excluding non-visually grounded hallucinations.

We extract two white-box signals. First, Dynamic Visual Gaze (Atraj) represents the attention trajectory. Given text sequence Ttext, we aggregate cross-attention weights A(i,jl,h) across layers L and heads H:

L

H

1 L · H · |Ttext|

A(i,jl,h) (2)

Sj =

h=1 i∈Ttext

l=1

We further apply Min-Max normalization to obtain the final gaze probability:

Sj − min(S) max(S) − min(S) + ϵ

(3)

Atraj(j) =

Crucially, unlike static visual features extracted from a frozen vision encoder, our target Vsem is derived from the Teacher’s last transformer layer. Due to the self-attention mechanism, these image token representations have effectively interacted with the textual instructions Q. Therefore, Vsem represents contextualized visual thoughts—reflecting not just what is in the image, but how the Teacher interprets the visual content specifically in response to the given query. To distill the most salient visual cues, we subject Atraj to Top-K (k = 8) sparsification, thereby ensuring sparse and noise-free supervision.

##### 4.2 White-box Trajectory Distillation

We establish the foundational reasoning capability of our model through a Supervised Fine-tuning (SFT) stage defined as Latent Teacher Forcing. Unlike standard SFT that maps inputs directly to text, we train the model to autoregressively generate a sequence of K continuous latent tokens,

V = {< v − trace1 >,...,< v − tracek >}, prior to producing the textual response.

These latent tokens serve as Visual Information Containers. By leveraging a white-box distillation approach, we force V to explicitly capture and compress the teacher’s high-dimensional visual semantics and gaze patterns. Formally, given an input [I,Xq], the model generates the latent and textual sequences to form the complete trajectory X = [I,Xq,V,Xans], where V acts as the indispensable cognitive bridge supplying visual evidence for the subsequent response Xans.

- 4.2.1 Curriculum Sensory Gating A naive attention mechanism allows response to-

kens Xans to attend directly to image patches I, enabling the model to bypass V (shortcut learning). Conversely, a permanent hard mask creates a training-inference distribution shift. To resolve this, we propose Curriculum Sensory Gating, which modulates the direct visual perception path via a time-dependent scalar γ(t) ∈ [ϵ,1].

We implement gating within the attention bias of the Transformer. Let Qtxt denote queries from response tokens and Kimg denote keys from image patches. The attention scores are computed as:

QtxtKimg⊤ √

Attn(Qtxt,Kimg) = Softmax

+ Bgate(t) ,

d

s.t. Bgate(t) = ln(γ(t)).

(4) To ensure a structured internalization process mentioned above, we introduce a warm-up period Tw. The gating scalar γ(t) is defined as:

ϵ + 1−2ϵ 1 − cos T πt

#### , t < Tw 1, t ≥ Tw

γ(t) =

w

(5) where Tw denotes the warm-up steps. This schedule defines two distinct operational phases governed by the training progress:

• Phase 1: Sensory Warm-up (t < Tw). The direct visual path opens gradually, following the cosine curve. Initially, γ ≈ ϵ (set to 1e-6 for numerical stability), resulting in a large negative bias Bgate ≪ 0. This creates a strict Latent Bottleneck, mathematically compelling the model to compress necessary visual information into V. The subsequent smooth relaxation prevents optimization shock while establishing a strong dependency on latent reasoning.

• Phase 2: Fully Observable (t ≥ Tw). The gate becomes fully open (γ = 1), reducing the bias Bgate to zero. The direct visual path functions as a Residual Perception connection, allowing the model to attend to fine-grained pixel details that complement the high-level reasoning encoded in V. This configuration matches the standard inference topology, ensuring zero distribution shift.

##### 4.2.2 Optimization Objectives

To ensure the student strictly internalizes the teacher’s cognition, we employ a dual-stream distillation scheme with explicit gradient flow controls.

- 1. Semantic Reconstruction (Lconcept). We align the student’s latent hidden states hz with the teacher’s holistic visual concepts Vsem. Since the teacher’s representations encapsulate high-quality visual semantics, we treat them as fixed semantic anchors. We employ a projection

head ϕmlp to map the student’s latent manifold into the teacher’s semantic space:

Lconcept = 1 −

1 B

B

i=1

CosSim ϕmlp(h(zi)),Vsem(i)

(6) This objective compels the student’s latent tokens V to act as informative containers, actively capturing and compressing the visual information necessary to reconstruct the teacher’s visual semantics.

- 2. Trajectory Alignment (Ltraj). We define the reasoning trajectory as the distribution of attention weights over visual patches. We

treat the teacher’s attention map Atraj as the target, following prior observations that distilling teacher attention maps can effectively guide student visual alignment (Fang et al., 2021). We constrain the student’s attention Astudent (originating from V) to match this target via KL Divergence:

Ltraj =

1 B

B

i=1

K

j=1

DKL A(traji,j)∥A(studenti,j) . (7)

This ensures the latent tokens learn where to look, inheriting the teacher’s visual search strategy.

- 3. Response Generation with Dynamic Gradient

##### Transition (Lntp).

The Next-Token Prediction loss drives the response generation. Crucially, the gradient flow from this loss is intrinsically modulated by our gating mechanism. By chain rule, the sensitivity of the loss with

respect to direct visual features is proportional to the attention weight:

∂Lntp ∂I ∝ Attn(Qtxt,Kimg) ≈ γ(t). (8)

During Phase 1, gradients are fully channeled through V, establishing the bottleneck. As γ → 1, the gradients transition to a synergistic flow, optimizing both the latent abstraction and the residual perception paths jointly.

- 4.2.3 Joint Training Dynamics Unlike complex multi-stage pipelines (Li et al., 2025a; Wang et al., 2025b) that require careful hyperparameter scheduling to avoid collapse, our Curriculum Sensory Gating provides a robust structural constraint. This physically enforced bottleneck naturally regulates the learning difficulty, eliminating the need for dynamic loss weighting.

We employ a streamlined Joint Training paradigm with a fixed distillation weight. The total loss is defined as:

Ltotal = Lntp + λ · (Lconcept + Ltraj), (9)

By maintaining a constant but moderate alignment pressure, we ensure the latent tokens V remain semantically consistent with the teacher in the curriculum, while allowing the NTP loss to primarily drive the generation quality as the sensory gate opens.

5 Experiment

- 5.1 Experiment Setup

SFT Settings. In Phase 1, we train the model for 400 steps with the sensory gating scalar γ initialized at 1e-6. In Phase 2, we continue training for an additional 600 steps. We set λ = 0.3 across all experiments. The model is initialized from Qwen2.5VL-3B and finetuned on the LaViT-SFT-15K. We set the number of latent visual tokens V to 4. More details are shown in the appendix.

Evaluated Benchmarks. We evaluate LaViT on diverse benchmarks. For subtle visual details, we use MMVP (Tong et al., 2024) and the Attribute Recognition subset of V∗ (Wu and Xie, 2024), which target CLIP-blind patterns and object attributes. For higher-level cognition, we adopt BLINK (Fu et al., 2024) tasks on Relative Depth, IQ-Test, Relative Reflectance, and Spatial Relation, which require mental manipulation and geometric reasoning. We also include MMStar (Chen et al.,

2024) to test robustness against language priors and ensure genuine visual understanding.

##### 5.2 Main Results

Table 1 presents the comparative performance of LaViT against state-of-the-art MLLMs. As shown, our method achieves consistent improvements across all benchmarks, demonstrating the efficacy of latent visual thinking.

Cross-Scale Superiority and Efficiency. LaViT significantly enhances its backbone, achieving substantial gains of +15.67% on Relative Reflectance and +16.94% on Relative Depth. Despite its compact 3B scale, it demonstrates remarkable crossscale competitiveness, outperforming the larger Qwen2.5-VL-7B on five out of seven benchmarks. Moreover, LaViT surpasses SOTA 7B models, beating LVR-7B on fine-grained spatial tasks (e.g., Relative Depth: 78.23% vs. 76.61%) and R1OneVision-7B on MMStar. These results confirm that optimizing latent thinking is a more parameterefficient strategy than simply scaling up model size. Advantage in Complex Visual Reasoning. LaViT excels in perception-intensive BLINK benchmarks by leveraging continuous latent reasoning to preserve spatial structures, effectively addressing the limitations of standard models in abstract geometric manipulation. Consequently, LaViT-3B achieves 78.23% on Relative Depth and 32.0% on IQ-Test, outperforming proprietary models like GPT-4o (64.52% and 30.0%) and reasoningenhanced baselines. Notably, it even surpasses the SOTA latent reasoning model LVR-7B on Relative Depth (+1.62%) and Relative Reflectance (+3.0%), validating its superior capability in capturing structural visual semantics.

Fine-Grained Perception and Robustness. LaViT effectively mitigates “CLIP-blindness,” achieving 67.33% on MMVP and substantially outperforming DMLR (61.33%) and PAPO (50.0%). By refining visual features to correct encoding errors rather than trading perception for reasoning, LaViT ensures robust visual grounding. This is further evidenced by its 54.07% score on MMStar (vs. 50.2% baseline), confirming that performance gains stem from genuine visual understanding rather than language hallucinations.

##### 5.3 In-depth Analysis of Attention Dynamics

To investigate the underlying mechanisms of LaViT, we conduct a deep dive into the attention distribution on the BLINK Relative Depth. We

[Figure 15]

Fine-grained Perception Visual Reasoning Robustness Method Params MMVP V* Attrib. Rel. Depth IQ-Test Rel. Ref. Spatial MMStar

Proprietary Model

GPT-4o (Hurst et al., 2024) - 58.33 72.5 64.52 30.0 38.8 76.9 63.9

###### Open Source Model

Qwen2.5-VL (Bai et al., 2025b) 7B 66.7 77.39 71.77 26.0 38.8 87.4 58.9 Qwen2.5-VL (Bai et al., 2025b) 32B 75.33 82.61 75.81 30.0 55.97 85.31 69.8

Open Source Reasoning MLLMs Naive SFT 3B 65.33 80.87 70.16 28.0 34.33 81.82 55.53 PAPO (Wang et al., 2025c) 3B 50.0* 22.61* 59.68 31.33* 33.6 76.92 52.7 DMLR (Liu et al., 2025) 3B 61.33 46.96 54.84 26.0 23.88 72.03 51.2 LVR_RL (Li et al., 2025a) 3B 55.3* 69.6* 64.52 30.7* 42.54 77.62 53.73 R1-OneVision (Huang et al., 2025) 7B 67.0* - - - - - 52.1* LVR (Li et al., 2025a) 7B 72.0 84.4 76.61 28.7 42.5 89.5 59.4

Baseline & Our Model Qwen2.5-VL (Baseline) 3B 62.33 81.74 61.29 24.0 29.85 81.12 50.2* LaViT 3B 67.33 ↑5.00 82.61 ↑0.87 78.23 ↑16.94 32.0 ↑8.00 45.52 ↑15.6781.82 ↑0.70 54.07 ↑3.87

Table 1: Performance comparison on multimodal benchmarks across three categories:

Fine-grained Perception , Visual Reasoning , and Multimodal Robustness . The best and second-best results are marked in bold and underlined, respectively. Green values indicate the absolute gains of LaViT over the Qwen2.5-VL-3B baseline. Asterisks (*) denote results reported by papers.(Fu et al., 2024)(Li et al., 2025a)(Liu et al., 2025)

Figure 4: Attention entropy distribution.

combine quantitative metrics with qualitative visualizations to analyze Concentration and Stability.

Quantifying Attention Concentration (Entropy). We employ Information Entropy (H) to measure the “sharpness” of the model’s visual focus. Formally, for a given image, the attention entropy is defined as:

N

H = −

i=1

pi log(pi) (10)

where pi is the normalized attention weight of the i-th patch. As visualized in Figure 4 and detailed in Table 2, the Base 3B model exhibits a heavy tail towards high entropy (H = 4.870), suggesting it lacks a clear visual target. LaViT significantly shifts this distribution leftward, reducing the mean entropy to 4.686. This improvement not only approximates the Teacher’s focused state (H = 4.284) but is also visually corroborated by Figure 5, where LaViT exhibits pinpoint focus on critical depth markers compared to the scattered gaze of the Base model.

##### Takeaway 1: Sharpening Visual Focus

LaViT effectively transitions the student from a “diffuse” observation mode to a “focused” reasoning mode. By reducing attention entropy (Figure 4), the student learns to ignore irrelevant background noise and concentrate on task-relevant regions.

Achieving Superior Stability via Sparsification. While the Teacher model is highly focused, it ex-

Table 2: Statistical analysis of attention distribution on BLINK Relative Depth. Salient Regions refers to patches with attention weights > 1.5× the mean. CV denotes the Coefficient of Variation (σ/µ).

Salient Regions Count Attention Entropy (H) Mean Std (σ) Range CV Mean Std (σ) CV

Model

Qwen2.5-32B 39.9 15.6 8-65 0.392 4.284 0.787 0.1837 Qwen2.5-3B 53.8 9.0 36-77 0.191 4.870 0.216 0.0444 LaViT 47.3 5.6 43-69 0.102 4.686 0.204 0.0435

hibits significant variance in its viewing strategy across samples (Salient Regions CV = 0.392). We observe that LaViT not only inherits the Teacher’s focus but actually achieves a much more stable attention pattern (CV = 0.102), surpassing the teacher in consistency.

We attribute this distilled stability directly to our White-box Trajectory Distillation design:

- • Top-K Sparsification: By explicitly supervising the student with only the Top-K (K = 8) strongest attention points from the teacher, we enforce a hard constraint that filters out the teacher’s low-confidence “attentional noise” or hesitation.
- • Data Filtering: Our preprocessing pipeline excludes samples where attention does not align with ground-truth regions, ensuring only high-quality traces are learned.

Consequently, LaViT does not merely mimic the Teacher’s raw output; it distills the most robust visual cues, resulting in a student model that is consistently focused on critical regions without the variance observed in the large teacher model.

- (1)
- (2)

Question: Which point is closer to the camera, Point A or Point B?

|[Figure 16]<br><br>Input 1<br><br>[Figure 17]<br><br>[Figure 18]|[Figure 19]<br><br>|[Figure 20]<br><br>[Figure 21]|[Figure 22]|
|---|---|---|---|
|[Figure 23]<br><br>[Figure 24]<br><br>Input 2<br><br>[Figure 25]<br><br>[Figure 26]|[Figure 27]|[Figure 28]<br><br>[Figure 29]|[Figure 30]<br><br>[Figure 31]|

Figure 5: Visualization of attention distributions across Qwen2.5-VL-3B, 32B (Teacher), and LaViT on two representative samples from the BLINK. ▲ indicates the task-relevant critical regions required for correct reasoning.

firms explicit supervision on “where to look” and “what to see” is essential for the student to inherit the teacher’s visual cognitive patterns effectively.

Table 3: Ablation study of LaViT on MMVP and BLINK subsets. w/o Traj. Align: Removes trajectory alignment loss. w/o Sem. Recon: Removes semantic reconstruction loss. w/o Curr. Gate: Removes progressive gating, setting γ(t) = 0 in Phase 1 and γ(t) = 1 in Phase 2. w/o Latent Tokens: Masking Latent Tokens at Inference Single Stage: Trains in one stage where visual tokens are always visible (γ(t) = 1).

Effectiveness of Curriculum Sensory Gating. The performance degradation observed without this module (e.g., MMVP accuracy drops to 59.33%) indicates that a simple hard switch of visual visibility is insufficient. Our progressive gating strategy is crucial for preventing shortcut learning and forcing the model to rely on deep latent reasoning.

Method MMVP Rel. Depth IQ-test Rel. Ref Spatial

LaViT-3B 67.33 78.23 32 45.52 81.82 w/o Trajectory Alignment 64.33 75 30.67 44.03 78.32 w/o Semantic Reconstruction 65.33 75.81 30.67 42.54 76.92 w/o Curriculum Sensory Gating 59.33 71.77 27.33 48.51 79.72 w/o Latent Tokens 64.33 77.42 25.33 81.12 25.33 Single Stage Training 65.67 71.77 28 38.81 79.02

Impact of Training Strategy. The Single Stage Training baseline (γ(t) = 1) consistently underperforms the full model, particularly on Relative Reflectance (38.81% vs. 45.52%). This demonstrates that progressively exposing visual information is key to avoiding sub-optimal convergence and building robust reasoning capabilities.

##### Takeaway 2: The Denoising Effect

LaViT acts as a semantic filter. Instead of inheriting the teacher’s instability, LaViT leverages Top-K sparsification to retain only the core attention patterns, reducing the Coefficient of Variation (CV) from 0.392 to 0.102. This results in a student that is surprisingly more decisive and stable than its teacher.

### 6 Conclusion

In this work, we identify the “Perception Gap” in multimodal distillation, where student models often mimic textual outputs without inheriting the teacher’s visual attention patterns. To bridge this, we propose LaViT, a framework that aligns latent visual thoughts via White-box Trajectory Distillation and Curriculum Sensory Gating. By utilizing latent tokens as cognitive containers, LaViT compels the student to reconstruct the teacher’s visual semantics and gaze before response generation. Extensive experiments demonstrate that LaViT-3B significantly outperforms SFT baselines and rivals 7Bscale models on reasoning-intensive benchmarks like BLINK and MMVP.

##### 5.4 Ablation Study

We evaluate LaViT on MMVP and BLINK subsets, comparing it against variants including single-stage training and the removal of curriculum gating.

Impact of Alignment Components. As shown in Table 3, removing either Trajectory Alignment or Semantic Reconstruction leads to significant performance drops across all tasks. Furthermore, masking latent tokens during inference precipitates a marked decline, verifying the model’s genuine reliance on the generated visual thoughts. This con-

### References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025a. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025b. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Yuxuan Cai, Jiangning Zhang, Haoyang He, Xinwei He, Ao Tong, Zhenye Gan, Chengjie Wang, Zhucun Xue, Yong Liu, and Xiang Bai. 2025. Llava-kd: A framework of distilling multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 239–249.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. 2024. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, and 3416 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. 2025. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683.

Zhiyuan Fang, Jianfeng Wang, Xiaowei Hu, Lijuan Wang, Yezhou Yang, and Zicheng Liu. 2021. Compressing visual-linguistic model via knowledge distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1428– 1438.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. 2025a. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776.

Qianhan Feng, Wenshuo Li, Tong Lin, and Xinghao Chen. 2025b. Align-kd: Distilling cross-modal alignment knowledge for mobile vision-language large

model enhancement. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4178–4188.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, WeiChiu Ma, and Ranjay Krishna. 2024. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer.

Jiawei Gu, Yunzhuo Hao, Huichen Will Wang, Linjie Li, Michael Qizhe Shieh, Yejin Choi, Ranjay Krishna, and Yu Cheng. 2025. Thinkmorph: Emergent properties in multimodal interleaved chain-of-thought reasoning. arXiv preprint arXiv:2510.27492.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024. Haotraining large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. 2024. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. 2025. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749.

Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Ma˛dry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, and 399 others. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Tianxiang Jiang, Sheng Xia, Yicheng Xu, Linquan Wu, Xiangyu Zeng, Limin Wang, Yu Qiao, and Yi Wang. 2025. Vknowu: Evaluating visual knowledge understanding in multimodal llms. arXiv preprint arXiv:2511.20272.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2020.

Tinybert: Distilling bert for natural language understanding. In Findings of the association for computational linguistics: EMNLP 2020, pages 4163–4174.

Jiwan Kim, Kibum Kim, Sangwoo Seo, and Chanyoung Park. 2025. Compodistill: Attention distillation for compositional reasoning in multimodal llms. arXiv preprint arXiv:2510.12184.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. 2023. Segment anything. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3992–4003.

Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, Jialian Wu, Xiaodong Yu, Hao Chen, Emad Barsoum, Muhao Chen, and Zicheng Liu. 2025a. Latent visual reasoning. arXiv preprint arXiv:2509.24251.

Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vuli´c, and Furu Wei. 2025b. Imagine while reasoning in space: Multimodal visualization-of-thought. arXiv preprint arXiv:2501.07542.

Chengzhi Liu, Yuzhe Yang, Yue Fan, Qingyue Wei, Sheng Liu, and Xin Eric Wang. 2025. Reasoning within the mind: Dynamic multimodal interleaving in latent space. arXiv preprint arXiv:2512.12623.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, ShangWen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, and 7 others. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

Yiming Qin, Bomin Wei, Jiaxin Ge, Konstantinos Kallidromitis, Stephanie Fu, Trevor Darrell, and Xudong Wang. 2025. Chain-of-visual-thought: Teaching vlms to see and think better with continuous visual tokens. arXiv preprint arXiv:2511.19418.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. 2024. Visual cot: Unleashing chain-of-thought reasoning in multi-modal language models. CoRR.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, and 1 others. 2025a. Vlm-r1: A stable and generalizable r1style large vision-language model. arXiv preprint arXiv:2504.07615.

Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. 2025b. Codi: Compressing chain-of-thought into continuous space via selfdistillation. arXiv preprint arXiv:2502.21074.

Fangxun Shu, Yue Liao, Le Zhuo, Chenning Xu, Lei Zhang, Guanghao Zhang, Haonan Shi, Long Chen, Tao Zhong, Wanggui He, Siming Fu, Haoyuan Li, Bolin Li, Zhelun Yu, Si Liu, Hongsheng Li, and Hao Jiang. 2024. Llava-mod: Making llava tiny via moe knowledge distillation. arXiv preprint arXiv:2408.15881.

Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, and 1 others. 2025a. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint

- arXiv:2505.08617.

Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, Linjie Li, Yu Cheng, Heng Ji, Junxian He, and Yi R. Fung. 2025b. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint

- arXiv:2506.23918.

Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. 2019. Patient knowledge distillation for bert model compression. arXiv preprint arXiv:1908.09355.

ByteDance Seed Team. 2025. Seed1.5-vl technical report. arXiv preprint arXiv:2505.07062.

Chameleon Team. 2024. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818.

Jintao Tong, Jiaqi Gu, Yujing Lou, Lubin Fan, Yixiong Zou, Yue Wu, Jieping Ye, and Ruixuan Li. 2025a. Sketch-in-latents: Eliciting unified reasoning in mllms. arXiv preprint arXiv:2512.16584.

Shengbang Tong, David Fan, Jiachen Li, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. 2025b. Metamorph: Multimodal understanding and generation via instruction tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17001–17012.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578.

Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. 2025a. Pixel reasoner: Incentivizing pixel-space reasoning with curiositydriven reinforcement learning. arXiv preprint arXiv:2505.15966.

Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, and Yisen Wang. 2025b. Monet: Reasoning in latent visual space beyond images and language. arXiv preprint arXiv:2511.21395.

Zhecan Wang, Noel Codella, Yen-Chun Chen, Luowei Zhou, Xiyang Dai, Bin Xiao, Jianwei Yang, Haoxuan You, Kai-Wei Chang, Shih fu Chang, and Lu Yuan. 2022. Multimodal adaptive distillation for leveraging unimodal encoders for vision-language tasks. arXiv preprint arXiv:2204.10496.

Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, and Heng Ji. 2025c. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. 2025. Sim-cot: Supervised implicit chain-ofthought. arXiv preprint arXiv:2509.20317.

Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, and Klara Nahrstedt. 2025. Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use. arXiv preprint arXiv:2505.19255.

Penghao Wu and Saining Xie. 2024. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084– 13094.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, Bo Zhang, and Wei Chen. 2025a. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615.

Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, and Chuang Gan. 2025b. Machine mental imagery: Empower multimodal reasoning with latent visual tokens. arXiv preprint arXiv:2506.17218.

Fengji Zhang, Linquan Wu, Huiyu Bai, Guancheng Lin, Xiao Li, Xiao Yu, Yue Wang, Bei Chen, and Jacky Keung. 2024. Humaneval-v: Evaluating visual understanding and reasoning abilities of large multimodal models through coding tasks.

Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, and Qing Li. 2025a. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv preprint arXiv:2505.15436.

Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi

Liu, Tianke Zhang, Haonan Fan, Kaibing Chen, Jiankang Chen, Haojie Ding, Kaiyu Tang, Zhang Zhang, Liang Wang, Fan Yang, Tingting Gao, and Guorui Zhou. 2025b. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630.

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. 2025. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362.

### A More Implementation Details

##### A.1 Baselines

To comprehensively evaluate the effectiveness of LaViT, we benchmark it against a diverse spectrum of state-of-the-art MLLMs. We categorize these baselines into three distinct paradigms to isolate the contributions of our latent reasoning mechanism from model scale and data exposure.

General-Purpose Foundation Models. We first establish a performance lower bound using our backbone model, Qwen2.5-VL-3B (Bai et al., 2025b), to quantify the specific gains attributed to our architecture. Crucially, to prove that our improvements stem from the proposed latent thinking mechanism rather than merely domain-specific data exposure, we construct a controlled baseline named Naive-SFT. This model is fine-tuned on the identical LaViT-15k dataset but utilizes standard text-only supervision, serving as a rigorous control variable. Furthermore, we include Qwen2.5VL-7B to assess cross-scale competitiveness and employ GPT-4o (Hurst et al., 2024) as a proprietary upper-bound reference to gauge how close our compact 3B model is to industrial state-of-theart performance.

Explicit Reasoning Frameworks (Thinkingabout-Images). This category includes methods that enhance reasoning via explicit textual Chainsof-Thought (CoT) or Reinforcement Learning. We compare against R1-OneVision (Yang et al., 2025a), which explicitly generates a “think-beforeanswer” trajectory in the language space. Additionally, we include PAPO (Wang et al., 2025c), an RL-based approach that optimizes image-grounded descriptions through verifiable rewards. For a fair comparison, we reproduce PAPO on the same 3B backbone to evaluate the efficiency of latent versus explicit alignment strategies.

Latent Visual Reasoning Competitors. Finally, we compare LaViT against direct competitors that also operate within the latent space. We benchmark against LVR (Li et al., 2025a) (and its RL variant), which operates on a larger 7B backbone. This serves as a cross-scale benchmark to demonstrate LaViT’s parameter efficiency. We also compare with DMLR (Liu et al., 2025), a framework utilizing test-time latent optimization. To ensure a fair comparison regarding computational cost and inference latency, we re-implement DMLR on the

Qwen2.5-VL-3B backbone with the number of latent optimization steps restricted to 4, matching the computational budget of our method.

A.2 Hyperparameters The Table A1 details the hyperparameters used for the 1000-step training run.

### B Analysis of numbers of K

To investigate the optimal capacity of the latent bottleneck, we conducted an ablation study on the number of latent visual tokens K ∈ {4,6,8} and monitored performance variations across different training steps. As detailed in Table A2, we observe that K = 4 yields the superior balance between visual grounding and reasoning capabilities.Specifically, the model with K = 4 achieves peak performance at 1,000 steps, recording the highest scores on MMVP (67.33) and IQTest (32.0), while matching the best Relative Reflectance performance (45.52). Interestingly, increasing the number of latent tokens to K = 6 or K = 8 does not translate to performance gains; instead, it leads to a slight degradation in reasoning tasks (e.g., IQ-Test and Relative Reflectance). This suggests that a compact set of 4 latent tokens is sufficient to encapsulate the necessary high-level visual semantics guided by the teacher’s attention, whereas a larger K may introduce redundancy or noise into the reasoning process. Furthermore, regarding training dynamics, we observe that performance generally peaks around 1,000 steps before stabilizing or slightly declining, indicating that the model reaches optimal alignment at this stage. Consequently, we adopt K = 4 and the 1,000-step checkpoint for all main experiments reported in this paper.

### C Training Data Construction C.1 Data Enrichment

To support the proposed distillation framework, the training dataset was enriched with pre-computed visual features serving as teacher supervision signals:

- 1. V-top Tensors: High-dimensional feature vectors (5120-dim) extracted from the final layer of the base model’s visual encoder, representing holistic semantic concepts.
- 2. Attention Maps: Compressed attention weights aggregated across heads and layers,

Table A1: Hyperparameters used for the 1000-step training run.

Category Parameter Value Description

Learning Rate 5e-6 Initial learning rate LR Scheduler linear Linear decay with warmup Warmup Ratio 0.03 Default transformer warmup Optimizer AdamW β1 = 0.9, β2 = 0.999 Weight Decay 0.0 No weight decay applied

Optimization

Total Steps 1000 Fixed step training Batch Size 16 Per-device training batch size Grad. Accum. 1 No gradient accumulation

Training Scale

Num Latent Tokens 4 Number of latent tokens V-top Dim 5120 Dimension of reconstruction target Freeze Vision True ViT encoder weights are locked Freeze LLM False LLM backbone is fine-tuned

Model Config

Max Pixels 1,003,520 1280×28×28 equivalent Min Pixels 200,704 256×28×28 equivalent

Data Config

Table A2: Ablation study on the number of latent tokens (K) and training steps. We report the performance on MMVP and BLINK subsets (IQ-Test, Relative Reflectance, and Spatial Relation). The selected configuration (K = 4, 1000 steps) is highlighted in bold.

Method K Steps MMVP IQ-Test Rel. Ref. Spatial

Qwen2.5-VL-3B - - 62.33 24.00 29.85 81.12 Naive SFT - - 65.33 28.00 34.33 81.82

600 58.33 26.00 45.52 83.92 800 63.33 32.00 44.78 77.62

1000 67.33 32.00 45.52 81.82 1200 67.00 28.67 42.54 76.92 1400 67.67 26.67 43.28 78.32

LaViT 4

600 62.00 26.67 42.54 80.42 800 58.67 28.67 43.28 81.82

1000 61.33 27.33 43.28 77.62 1200 63.00 26.00 42.54 76.92

LaViT 6

- 1400 65.67 28.67 44.78 75.52

LaViT 8

600 63.33 22.67 30.60 84.62 800 66.00 28.00 42.54 77.62

1000 65.67 28.00 38.81 79.02 1200 66.00 28.67 40.30 76.92

- 1400 66.67 29.33 40.30 76.22

used to generate the trajectory supervision signal for explicit visual grounding.

##### C.2 Data Processing and Alignment

To ensure precise synchronization between the student’s latent states and the teacher’s signals, we implemented the following processing strategies:

Adaptive Scaling. Given the dynamic resolution characteristics of Qwen2.5-VL, feature maps often vary in spatial dimensions. We record the critical step of applying Bilinear Interpolation to align the spatial resolution of the pre-computed v_top feature maps with the attention_maps. This step

is essential for maintaining pixel-level correspondence across inputs with varying aspect ratios.

Latent Supervision Strategy. For the optimization of latent visual thoughts, we explicitly designate the hidden state of the last latent token (e.g., <v-trace4>) as the anchor for supervision. By computing the loss only at the end of the latent sequence, we force the visual information to flow fully through the bottleneck, compelling the preceding latent tokens to compress and structure the visual data effectively.

##### C.3 System Prompt for LVR Alignment

During the Supervised Fine-Tuning (SFT) stage, we utilized a specialized system prompt to prime the model for latent visual reasoning. The prompt is presented below:

System Prompt

“You are an expert multimodal large language model. You process visual information through specialized latent tokens to ensure precise alignment between visual perception and textual reasoning.”

##### C.4 LaViT-15k Dataset Statistics

We constructed LaViT-15k, a high-quality multimodal dataset specifically designed to train the latent visual thought capabilities of the model. The dataset comprises 14,567 samples, aggregating diverse visual scenarios from 10 mainstream visionlanguage benchmarks.

Data Distribution. The dataset ensures diversity by incorporating samples from captioning, VQA,

and document understanding tasks. As shown in Table A3, Flickr30k (32.13%) and GQA (20.05%) constitute the majority of the data, providing a strong foundation for general visual grounding, while datasets like DocVQA and TextCap enhance the model’s fine-grained perception capabilities.

- Table A3: Distribution of data sources in LaViT-15k. Source Dataset Samples Percentage

Flickr30k 4,681 32.13% GQA 2,921 20.05% DocVQA 1,647 11.31% TextCap 1,383 9.50% Visual7W (V7W) 1,110 7.62% OpenImages 1,016 6.97% TextVQA 839 5.76% InfographicsVQA 651 4.47% CUB (Birds) 174 1.19% VSR 145 1.00%

Total 14,567 100.00%

Statistical Properties. Table A4 summarizes the key statistical properties of the text and visual components. The dataset features high-resolution inputs (avg. 957 × 882) and rich visual semantics, represented by an average of 697.73 visual tokens per image. The high-dimensional visual features (D = 5120) serve as the supervision target for the latent thought process.

- Table A4: Statistical properties of textual and visual components in LaViT-15k.

Statistic Value Textual Statistics Avg. Question Length 13.58 words Max Question Length 34 words Avg. Answer Length 4.92 words Visual Statistics Avg. Image Resolution 957 × 882 pixels Avg. Visual Tokens 697.73 Feature Dimension 5,120

