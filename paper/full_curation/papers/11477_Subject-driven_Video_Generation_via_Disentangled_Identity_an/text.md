# arXiv:2504.17816v3[cs.CV]5May2026

## Learning Zero-Shot Subject-Driven Video Generation Using 1% Compute

Daneul Kim1,3,§ Jingxu Zhang3,§ Wonjoon Jin2,§ Sunghyun Cho2 Qi Dai3 Jaesik Park1 Chong Luo3

1Seoul National University 2POSTECH 3Microsoft Research Asia

Abstract. Subject-driven video generation (SDV-Gen) aims to produce videos of a specific subject by adapting a pretrained video model, enabling personalized and application-driven content creation. To achieve this goal, per-subject tuning methods require approximately 200 A100 GPU hours to generate a customized video, whereas zero-shot methods avoid per-subject tuning but typically rely on millions of subjectvideo pairs for the supervision, incurring massive network fine-tuning costs (10K–200K A100 GPU hours). We propose a data- and computeefficient zero-shot SDV-Gen framework that avoids test-time per-subject tuning and the use of large-scale subject-video pairs. Our key idea decomposes SDV-Gen into (i) identity injection learned from subject-image pairs and (ii) motion-awareness preservation maintained by a small set of arbitrary videos. We optimize the two tasks with stochastic switching, using random reference-frame sampling and image-token dropout to prevent trivial first-frame copying. Our gradient analysis shows that the two objectives rapidly evolve toward nearly orthogonal update subspaces, explaining the stable optimization. Using CogVideoX-5B, we adapt a single model with 200K subject-image pairs and 4,000 arbitrary videos in 288 A100 GPU hours. This yields about 1% of compute compared to prior zero-shot baselines (i.e., 0.4% of VACE and 2.8% of Phantom) while using no subject-video pairs, yet remaining competitive in subject fidelity and motion quality. We show that the same recipe transfers to Wan 2.2-5B.

Keywords: Zero-Shot · Video Customization · Video Personalization · Subject-Driven Video Generation

### 1 Introduction

Recent advances in video diffusion models [4,21,49,56] have expanded controllable text-to-video (T2V) synthesis and video customization, enabling conditioning on signals such as keypoints, edges, and reference images [1, 24, 32, 46, 57].

§Work done during internship at Microsoft Research Asia.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

+ “A pig surfing on a blue wave on a surfing board”

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

###### + “A blue car

skids through a snowy mountain”

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

+ “A musician, wearing a Hawaiian shirt, strums a ukulele on a boardwalk, island tunes”

[Figure 15]

Fig. 1: Our results. We extend video generation models to enable zero-shot subjectdriven video generation (SDV-Gen). We use small datasets and compute for the supervision. (First row) Results of extended CogVideoX-5B [56] supervised with 200K subject-image pairs and 4,000 arbitrary videos. (Second row) The same model extended with only 4,000 subject-image pairs and 4,000 videos. (Third row) Results of extended Wan 2.2-5B [49] with 200K subject-image pairs and 4,000 arbitrary videos.

Building on this progress, subject-driven video generation [25, 27, 52, 53, 60, 63] (also known as subject-to-video customization or video personalization, i.e., SDV-Gen) seeks to synthesize videos that preserve a target subject’s identity while allowing variations in scene, motion, and context, an ability essential for customized content creation. In practice, this is typically achieved by adapting a pretrained T2V backbone to the target subject.

Early SDV-Gen approaches fine-tune a separate model or LoRA per subject [7,43,52,55], limiting scalability. More recent methods train a single model that generalizes to unseen subjects [12,23,25,27,28,34,60], but this implicitly assumes that both identity and motion-awareness must be learned from subjectvideo pairs. More importantly, these zero-shot methods rely on large subjectvideo datasets and substantial compute (Table 1; 10K–210K A100 GPU-hours for VACE [28] and Phantom [34]), raising the barrier to entry in this field. Reducing dependence on subject-video pairs and lowering computational cost are thus crucial for making subject-driven, customized video generation more accessible.

A potential alternative is to learn identity from subject-driven image generation (SDI-Gen) data (i.e., subject-image pairs) while inheriting motion priors from the pretrained T2V backbone. In principle, subject-image pairs provide strong identity cues, whereas the backbone encodes motion priors (i.e., temporal dynamics) from large-scale pretraining. However, naïve fine-tuning on subjectimages pairs severely degrades inter-frame temporal coherence, often causing motion to collapse into inconsistent dynamics or near-static (frozen) movement. This suggests that identity and motion modeling interfere with each other, indicating that treating SDV-Gen as a single objective has a fundamental issue.

We consider subject adaptation into two complementary components: identity injection from subject-image pairs and motion-awareness preservation from

- Table 1: Comparison of subject-driven video generation methods. We compare per-subject tuning methods’ required inputs and per-subject tuning time, along with zero-shot methods’ dataset size and required finetuning time.

Per-subject Tuning Zero-shot Required Inputs Base Model (Param.) Steps A100 hours∗

CustomCrafter [55] ✗ 200 regularizing imgs per subject VideoCrafter2 [9] (1.4B) 10K 200‡ Still-Moving [7] ✗ Few reference images + 40 videos Lumiere [3] (1.2B) 500 –

###### Zero-shot Methods Zero-shot Dataset Size Base Model (Param.) Steps A100 hours∗

VideoBooth [27] △ 48,724 subject-video pairs SD-based VDM [20,42] (1.08B) 400K‡ 775–1,938‡ VACE [28] ✓ 53M source videos LTX [18] & Wan [49] (1.3–14B) 200K 70K–210K‡ Phantom [34] ✓ 1M subject-video pairs Wan [49] (1.3–14B) + Seed [50] 30K 10K–30K‡

2,000/4,000 subject-image pairs + 4,000 arbitrary videos

Ours-tiny/mini ✓

CogVideoX [56] (5B) 4,000 288

200K subject-image pairs

CogVideoX [56] (5B) & Wan [49] (2.2-5B)

Ours ✓

4,000 288

+ 4,000 arbitrary videos

∗Total GPU hours. ‡Estimated from reported batch sizes and wall-clock training time. Please see the supplement for details.

a small set of arbitrary videos. Based on this, we propose to use a stochastic task-switching schedule that alternates between these objectives, predominantly sampling subject-image pairs while using arbitrary videos only to maintain temporal coherence. This separation allows identity and motion to be learned from different types of data sources, avoiding the collapse observed in naïve SDI-Gen fine-tuning and removing the need for subject-video pairs.

Across CogVideoX-5B [56] and Wan 2.2-5B [49], we observe that the conflict between the two gradients from the two-objectives diminishes rapidly and becomes nearly orthogonal. To understand why this two-objective training remains stable under task switching, we analyze the gradients of the two objectives during fine-tuning. The gradient analysis suggests that identity and motion updates concentrate on largely disjoint parameter subspaces, reducing interference.

This yields a data- and compute-efficient recipe to extend a T2V backbone into a zero-shot (i.e., tuning-free at test time) SDV-Gen model without subjectvideo pairs. We fine-tune CogVideoX-5B on 200K subject-image pairs and 4,000 Pexels videos [30] for 4,000 steps (288 A100-hours), using ∼1% of the compute (2.8% of Phantom [34]; 0.4% of VACE [28]) while remaining competitive. By showing that SDV-Gen can be learned with substantially less computation than prior baselines, we outline a practical path toward scalable video customization. Our approach even applies in a strong low-data regime (4,000 subject-image pairs with 4,000 arbitrary videos) and transfers to Wan 2.2-5B.

Our contributions are summarized in threefold:

- – We introduce a recipe to extend T2V models into a compelling zero-shot subject-driven video generation framework. We separate identity injection from motion preservation, enabling training without any subject-video pairs.
- – We provide a gradient analysis showing that the identity and motion-aware objectives update independent parameter subspaces, explaining why the proposed formulation remains stable without subject-video pairs.
- – We fine-tune multiple T2V backbones using only 200K subject-image pairs and 4,000 arbitrary videos, trained for 288 A100-hours, which is 1% of compute compared to other zero-shot baselines.

### 2 Related Work

#### 2.1 Subject-driven Image Generation

Diffusion models have substantially advanced text-to-image synthesis [11,15,42], and extensive work studies how to inject novel subjects while preserving identity. Early approaches introduce explicit spatial control (e.g., pose or edge guidance) [39, 62]. More recent methods employ cross-attention-based visual encoders such as IP-Adapter and SSR-Encoder [57, 64], as well as personalization techniques including DreamBooth [43], Textual Inversion [17], CustomDiffusion [31], DisenBooth [10], and subsequent multi-subject or zero-shot variants [5, 14, 35, 48, 51, 61]. These methods typically learn subject-specific representations from subject-image pairs, enabling subject-driven image generation (SDI-Gen). We follow this line by using subject-image pairs (Subject-200K from OminiControl [46]) as the primary source of identity supervision, and additionally leverage a small number of arbitrary videos to preserve temporal coherence.

#### 2.2 Subject-driven Video Generation

Subject-driven video generation (SDV-Gen) extends SDI-Gen to the temporal domain. Per-subject tuning methods such as DreamVideo [52], MotionBooth [54], Still-Moving [7], and CustomCrafter [55] adapt a model (or LoRA) for each identity, incurring non-trivial test-time compute per subject. In contrast, zero-shot approaches including VideoBooth [27], Concept-Master [25], DreamVideo-2 [53], MagicMirror [63], ID-Animator [19], and Consis-ID [60] train a single model but often rely on subject-video pairs and may target restricted domains (e.g., faces). Large-scale DiT-based systems such as VACE [28], Phantom [34], HunyuanCustom [23], and VideoAlchemist [12] further improve quality by training on million-scale datasets (e.g., Phantom-Data [13] and OpenS2V-Nexus [59]), but require substantial compute budgets (Table 1). In contrast, we study a data- and compute-efficient alternative that avoids subject-video pairs during adaptation by decomposing identity learning and motion preservation, adapting pretrained backbones using subject-image pairs and a small set of arbitrary videos. We demonstrate efficient zero-shot SDV-Gen on modern video diffusion backbones, such as CogVideoX-5B [56] and Wan 2.2-5B [49].

#### 2.3 Multi-task Optimization

Our approach trains a single model with two objectives: (1) identity injection on subject-image pairs and (2) motion preservation on arbitrary videos. This relates to multi-task learning, where gradient conflicts can hurt performance. Methods such as Gradient Surgery [58] mitigate interference by modifying updates. Related issues such as catastrophic forgetting [37] motivate continual learning approaches that revisit past samples via memory buffers [6,36] or replay [16, 38, 41, 44]. In contrast, since we keep a fixed, small pool of videos, we proposed to adopt a lightweight stochastic task-switching schedule between

subject-to-image and image-to-video updates. This design is computationally more efficient than Gradient Surgery and continual learning approaches because it updates a single objective per step, avoiding the simultaneous computation of multiple task gradients, explicit gradient projections, and memory expansion. We analyze training dynamics and show that gradient conflict diminishes rapidly. This supports stable optimization under limited data and compute.

### 3 Method

#### 3.1 Preliminaries

[Figure 16]

###### Dual-task Learning 🔥

We fine-tune a pretrained multimodal diffusion transformer (MMDiT)) [40] for zero-shot subjectdriven video generation. We instantiate our framework with Cog VideoX-5B [56] and show that the same adaptation recipe transfers to Wan 2.2-5B [49].

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

GT

Stochastically

VideoReplay

ℒ

Switching

Generated

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Frames

[Figure 26]

Generator🔥

ID Injection

[Figure 27]

|[Figure 28]|
|---|

Penguins under a sunset …

+

At each denoising step, MM-DiT processes noisy visual tokens X ∈ RN×d and text tokens CT ∈ RM×d that share embedding dimension d. Each block consists of LayerNorm followed by multi-modal attention (MMA) over the concatenated sequence [X;CT]: MMA([X;CT]) = softmax QK

Randomly Selected Frame Prompt

Reference

|[Figure 29]|
|---|

[Figure 30]

[Figure 31]

[Figure 32]

🔥Generator

Image Prompt

ℒ

[Figure 33]

+

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Bikers and Carriages Driving …

withinan

Generated GT Image

Nestled

urban…

<CLS>

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Blurred Crowd of People …

[Figure 42]

|[Figure 43]|
|---|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Sailboat Sailing During…

Subject Image Pairs

|[Figure 49]|
|---|

[Figure 50]

[Figure 51]

|[Figure 52]|
|---|

Arbitrary Videos

Fig. 2: Dual-task learning strategy. We formulate SDV-Gen as a dual-task problem: (i) identity injection (bottom) from subjectimage pairs, and (ii) motion-awareness preservation (left) using arbitrary videos via stochastically switched learning.

⊤

d V , where Q,K,V

√

are the usual query, key, and value projections. Spatial and temporal positions are encoded with RoPE [45], applied to visual tokens as Xi,j ← Xi,j · R(i,j).

Because attention cost scales quadratically with the number of tokens, videocentric fine-tuning is substantially more expensive than image-only training. Our goal is to adapt the backbone without any subject-video pairs by using subjectimage pairs as the primary identity supervision and a small set of arbitrary videos to preserve temporal dynamics.

Let Dimg denote a dataset of subject-image pairs {(Iref,Iout),P}, where Iref and Iout depict the same subject under different poses, viewpoints, or contexts, and P is the text prompt. Let Dvid denote a text-video dataset {(Pvid,V )}, where Pvid is the text prompt and V is the corresponding video. We define two training objectives, Limg = Lsubject-image(Iref,Iout,P) for identity injection and Lvid = Ltext-video(Pvid,V ) for motion-awareness preservation. The corresponding gradients are gimg = ∇θLimg and gvid = ∇θLvid with respect to the trainable parameters θ.

[Figure 53]

Fig. 3: Training and Inference Details. Left: During training, we stochastically alternate between two objectives: identity injection using subject-image pairs and motion-awareness preservation using a small set of arbitrary videos. Right: At inference time, no additional per-subject tuning is required. The model generates a video conditioned on the reference image and text prompt in a zero-shot manner.

#### 3.2 Dual-Task Formulation

We formulate a dual-task problem (Fig. 2): (i) identity injection from subjectimage pairs to learn subject appearance, and (ii) motion-awareness preservation from arbitrary videos. We optimize via stochastic task switching (Fig. 3). At each iteration, we draw u ∼ U(0,1) and select:

Ltotal = Lvid, if u < p, Limg, otherwise.

(1)

yielding expected update E[g] = (1 − p)gimg + pgvid. We use p = 0.2, so 80% of updates come from subject-image pairs and 20% from arbitrary videos during fine-tuning. p is chosen empirically. We demonstrate more in detail in supplementary material.

#### 3.3 Identity Injection with Subject-Image Pairs

Following OminiControl [46], given subject-image pairs (Iref,Iout) of the same subject and prompt P, we encode Xref = VAE(Iref), Xout = VAE(Iout), CT = T5(P), and train MM-DiT for Xout conditioned on (Xref,CT). For single-frame input, we truncate RoPE embeddings to match the token count of one frame.

We apply LoRA [22] to a subset of attention and normalization layers that interact with reference-image tokens, while keeping the rest of the backbone frozen. To anchor identity in the conditioning stream, we prepend a special <CLS> token to prompts (e.g., “An <CLS> armchair in the living room”). We provide ablations in the supplement showing that <CLS> improves identity metrics (CLIP-I, DINO-I) without degrading motion quality.

#### 3.4 Motion Awareness with Arbitrary Videos

Fine-tuning a pretrained video model on SDI-Gen pairs alone often erodes temporal modeling and yields near-static generations at test time. To preserve motion, we introduce an image-to-video (I2V) objective on a small set of arbitrary videos (Pvid,V ) ∈ Dvid. For each video, we sample a reference frame index i and set Iref = Vi as conditioning, while the full video V serves as the reconstruction target. We encode

Xref = VAE(Iref), Xvid = VAE(V ), CTvid = T5(Pvid), (2)

and train MM-DiT for Xvid conditioned on (Xref,CTvid). We adopt I2V finetuning (instead of pure T2V) since it matches the inference-time modality (reference image→video) and isolates motion preservation from identity supervision. Mitigating reference-frame copying. Conditioning on a single frame can admit a degenerate solution that simply replicates the reference appearance across time with minimal motion. We propose to apply two simple regularizers to discourage this behavior: (i) random reference-frame sampling, where i is sampled uniformly over frames rather than fixed to the first frame; (ii) image-token dropout, where we randomly drop a subset of tokens in Xref with probability pdrop, encouraging the model to rely on temporal priors instead of copying. We ablate these regularizers in the supplement.

#### 3.5 Training Procedure

Putting everything together, fine-tuning proceeds as follows. At each training step we sample u ∼ U(0,1) and

- 1. if u < p, draw a mini-batch of arbitrary videos (T,V ) from Dvid, sample a random reference frame and apply image-token dropping, and update θ with Lvid(T,V );
- 2. otherwise, draw subject-image pairs (I(1),I(2),P) from Dimg and update θ with Limg(I(1),I(2),P).

This stochastic proxy replay maintains motion-awareness through occasional video updates while injecting identity from subject-image pairs in most steps.

[Figure 54]

SDI-Gen→I2V. An alternative way to generate subject-driven video generation is to sequentially apply SDI-Gen model and I2V model. However, this pipeline relies on clear subject visibility in the initial frame for identity propagation, which often fails in dynamic scenes (e.g., “a toy car comes closer to the camera on a wet pavement,” Figure 4). When the subject is small or occluded, the image-to-video model

| |
|---|
| |

[Figure 55]

[Figure 56]

###### OursI2V

[Figure 57]

As Start Frame

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

+ “Under the glow of a streetlamp,a toy carcomes closer to camera on a wet pavement, reflecting the flickering city lights.”

Fig. 4: Limitation of SDI-Gen→I2V. When the subject is small in the first frame, I2V fails to produce consistent results because it cannot reliably interpret lowresolution subjects.

may misidentify it and introduce inconsistent or fabricated details, causing a progressive loss of identity in later frames. We provide additional comparison with SDI-Gen→I2V as baselines. Please refer to the result in Sec. 5.

Computational Efficiency. We quantify the computational efficiency of our stochastic task-switching scheme by analyzing the expected per-step cost. Let Cimg and Cvid denote the per-step cost of SDI-Gen and I2V updates, respectively. Under MM-DiT attention, if a video contains T latent frames, the total token count grows approximately linearly with T, so the attention cost grows approximately as T2. For T=13, we empirically observe Cvid ≈ 169Cimg. With task-switch probability p=0.2, the expected cost per step is

E[C] = (1 − p)Cimg + pCvid ≈ 0.8Cimg + 0.2 · 169Cimg ≈ 34.6Cimg,

which is substantially lower than video-only fine-tuning. Using only 4,000 arbitrary videos, we fine-tune T2V models within 288 A100-hours while achieving competitive zero-shot SDV-Gen performance (Table 1).

### 4 Gradient Dynamics in Dual-Task Learning

In Sec. 3.2, we formulated SDV-Gen as a dual-task learning problem with two losses, Limg and Lvid, and their corresponding gradients gimg(t) and gvid(t). To analyze gradient dynamics, we freeze the model at each checkpoint θt and compute gimg(t) and gvid(t) on mini-batches from Dimg and Dvid with respect to the trainable parameters θtrain (LoRA and task-specific normalization layers), matching the training setup. We log their cosine similarity ϕ(t) and the gradient norms ∥gimg(t)∥2 and ∥gvid(t)∥2. Measurement protocols are provided in the supplement.

#### 4.1 Do the Gradients Become Orthogonal?

Figure 5 plots the evolution of ϕ(t) under stochastic dual-task learning. We fine-tune pretrained CogVideoX-5B and Wan 2.2-5B using Subject-200K [46] as subject-image supervision and a small set of arbitrary videos from Pexels [30]. Starting from initialization, the cosine similarity rapidly converges to a narrow band around zero and remains there for the rest of fine-tuning regardless of the model used. This supports the hypothesis that, after a short transient, identity injection and motion-awareness preservation update nearly orthogonal directions in the trainable parameter subspace.

To rule out the trivial explanation that ϕ(t) is close to zero only because both gradients vanish as training converges, we additionally track gradient norms. As shown in Fig. 5, both ∥gimg(t)∥2 and ∥gvid(t)∥2 remain well above the numerical noise floor throughout fine-tuning and stay within the same order of magnitude after the initial drop. The decay of ϕ(t) therefore reflects a genuine change in gradient direction rather than an artifact of vanishing magnitudes.

CogVideoX

- 0
- 1

GradNorm

Alignment

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

Image

0.5

Video

- 0 200 400 600 800 1000 Step
- 1

0.0

0 200 400 600 800 1000

Step

Wan

- 0
- 1

GradNorm

Alignment

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

0.1

Image

Video

- 0 200 400 600 800 1000 Step
- 1

0.0

0 200 400 600 800 1000

Step

- Fig. 5: Gradient analysis on alignment and norms during fine-tuning on CogVideoX (Top) and Wan (Bottom). (Left) Cosine similarity ϕ(t) between gimg and gvid (over trainable parameters) quickly collapses to a narrow band near zero under dual-task training, indicating emergent near-orthogonality. (Right) ℓ2 norms ∥gimg(t)∥2 and ∥gvid(t)∥2 remain non-negligible and similar in scale after 100-step.

#### 4.2 Why Does the Gradient Conflict Diminish?

The empirical behavior above suggests that stochastic switching between Limg and Lvid tends to reduce their gradient inner product over training. For intuition, we provide a simplified analysis in the supplement (Appendix C) based on a local second-order (quadratic) approximation.

Proposition 4.2.1 (Local decay of gradient inner product). Under the local second-order assumptions, let

L¯(θ) = (1 − p)L1(θ) + pL2(θ), (3)

and consider gradient descent on the mixture loss. Then there exist constants C > 0 and ρ ∈ (0,1) such that

|⟨∇L1(θt),∇L2(θt)⟩| ≤ Cρt. (4) In particular,

⟨∇L1(θt),∇L2(θt)⟩ = 0. (5)

lim

t→∞

#### 4.3 Relation to Gradient-Surgery Method

Several works in multi-task learning explicitly manipulate gradients to resolve conflicts, such as Gradient Surgery [58], which projects each task gradient to remove components that conflict with other tasks. In our two-task setting, Gra-

dient Surgery would replace gimg by gimgGS = gimg− ⟨g∥imgg ,gvid⟩

gvid if ⟨gimg,gvid⟩ < 0

vid∥22

and analogously for gvid. This guarantees ⟨gimgGS ,gvid⟩ ≥ 0 at each step, but requires computing both gradients and performing additional projections at every iteration, roughly doubling the gradient-computation cost compared to sampling a single task.

- Table 2: VBench benchmark results. We utilize publicly available versions for Phantom [34] and VACE [28] based on Wan-2.1 1.3B [49]. We denote the first, the second, and the third best approaches.

SDI-Gen → I2V methods Motion Smooth. Dynamic Degree CLIP-T CLIP-I DINO-I

OminiControl [46] 98.21 51.67 31.89 72.58 54.16 BLIP [32] 97.53 49.17 28.19 79.29 56.58 IP-Adapter [57] 97.21 55.83 26.97 73.86 45.18

###### SDV-Gen Methods Motion Smooth. Dynamic Degree CLIP-T CLIP-I DINO-I

VideoBooth [27] 96.95 51.67 29.59 66.06 34.54 Phantom-1.3B [34] 98.93 54.90 33.51 72.49 51.94 VACE-1.3B [28] 98.68 40.00 33.60 73.35 52.68

Ours (CogVideoX-5B) 98.45 69.64 32.69 77.14 62.88 Ours (Wan 2.2-5B) 98.53 66.67 32.38 78.28 60.18

Our empirical analysis shows that the simpler stochastic switching scheme drives the gradient cosine ϕ(t) toward zero while keeping both norms nonvanishing, achieving a similar end effect (reducing gradient conflict) without explicit projections or extra gradient evaluations. For this reason, we use dual-task learning with stochastic switching as our default, and regard Gradient Surgerystyle techniques as unnecessary refinements in this setting. We provide qualitative and quantitative comparisons between stochastic switching and Gradient Surgery in the supplement.

### 5 Experiments

#### 5.1 Setup

Implementation. We build our approach on CogVideoX-5B and Wan 2.2-5B. CogVideoX-5B uses a 3D MM-DiT backbone [56] with a DDIM scheduler [42], while Wan 2.2-5B uses an X-DiT backbone [49] trained with flow matching. Finetuning with the two objectives is performed using AdamW with a learning rate of 5×10−5 and a cosine-with-restarts schedule. We fine-tune the CogVideoX-5B and Wan 2.2-5B for 4,000 steps (∼288 A100 GPU hours) using BF16 mixed precision, with batch sizes of 256 for images and 32 for videos. We apply LoRA (rank 128, dropout 0.2) to the designated trainable layers and use stochastic switching with probability p = 0.2, based on the empirical choice. We provide an ablation study of LoRA rank and p in the supplement.

We use OminiControl’s Subject200K dataset as our subject-image pairs. To study data efficiency, we additionally evaluate 2,000 and 4,000 subsets of Subject200K. We sample ∼4,000 arbitrary videos (1% of Pexels 400K [30]), providing diverse real-world motion patterns. We train two sampling-rate variants: an 8 FPS model aligned with the original CogVideoX setting, and a 16 FPS model further trained on Pexels videos to improve motion smoothness and temporal consistency at higher frame rates. All ablations are conducted on CogVideoX, with additional implementation details deferred to the supplementary material. Baseline. For zero-shot baselines, we use the SDV-Gen models of VideoBooth [27],

Phantom [34] and VACE [28] with Wan 2.1-1.3B and additionally compare with state-of-the-art SDI-Gen models OminiControl [46], BLIP-Diffusion [32], and IP

###### Table 3: OpenS2V benchmark results. Total score is the normalized weighted sum of other scores. We denote the first, the second, and the third best approaches.

Method Training Cost Total Score↑ Aesthetics↑ Motion↑ FaceSim↑ GmeScore↑ NexusScore↑ NaturalScore↑

Vidu 2.0 - 48.67 34.78 24.40 36.20 65.56 45.20 72.60 Pika 2.1 - 48.93 38.64 31.90 32.94 62.19 47.34 70.60 Kling 1.6 - 53.12 35.63 36.40 39.26 61.99 48.24 81.40

VACE-P1.3B [28] ≈70K hrs 44.28 42.58 18.00 18.02 65.93 36.26 76.00 VACE-1.3B [28] ≈70K hrs 47.33 41.81 33.78 22.38 65.35 38.52 76.00 VACE-14B [28] ≈210K hrs 58.00 41.30 35.54 64.65 58.55 51.33 77.33 Phantom-1.3B [34] ≈10K hrs 49.95 42.98 19.30 44.03 65.61 37.78 76.00 Phantom-14B [34] ≈30K hrs 53.17 47.46 41.55 51.82 70.07 35.35 69.35 SkyReels-A2-P14B [8] - 51.64 33.83 21.60 54.42 61.93 48.63 70.60 HunyuanCustom [23] - 51.64 34.08 26.83 55.93 54.31 50.75 68.66

Ours (CogVideoX-5B) [56] 288 hrs 50.05 45.40 19.38 18.05 70.53 41.23 68.52 Ours (Wan 2.2-5B) [49] 288 hrs 53.26 42.87 15.98 36.77 66.31 40.72 72.89

-Adapter [57], combined with the CogVideoX-5B I2V model: each first performs subject-driven image generation with its original setup, and we then generate videos via CogVideoX-5B. For per-subject tuning, we additionally comapre with Still-Moving [7] and CustomCrafter [55] qualitatively, using the results reported in their paper due to unavailable code or costly per-subject tuning.

Evaluation Details. We randomly picked 30 reference images from state-of-theart SDI-Gen sets [2,7] and the traditional DreamBooth dataset [43]. We utilize GPT to generate 4 prompts for each image comprising of a total 120 pairs for video generation. Utilizing 120 videos, we report VBench metrics [26]: Motion Smoothness (temporal consistency), Dynamic Degree (amount of motion), CLIPT (text-video alignment), CLIP-I (CLIP-based image similarity), and DINO-I (DINO-based subject fidelity). Additionally, we provide OpenS2V [59] evaluation benchmark result.

- 5.2 Comparison with Baselines

Quantitative Analysis.

Table 2 reports VBench metrics for zero-shot SDV-Gen (Phantom and VACE use the publicly available Wan-2.1 1.3B versions). On VBench, our CogVideoX variant achieves the best Dynamic Degree (69.64) and the best identity consistency (DINO-I, 62.88), while maintaining high motion smoothness (98.45). Our Wan variant shows consistent gains with Dynamic Degree 66.67 and DINOI 60.18, and also attains strong CLIP-I (78.28). While Phantom and VACE achieve slightly higher CLIP-T (33.51/33.60), our method remains competitive on text alignment (32.69/32.38) and substantially improves dynamics and identity. Among SDI-Gen→I2V pipelines, BLIP+I2V attains highest CLIP-I (79.29) but exhibits weaker dynamics (49.17) and lower DINO-I (56.58), whereas our approach provides a more balanced trade-off between motion and subject fidelity.

Table 3 further evaluates Open S2V, where our Wan model achieves a total score of 53.26% with only 288 training hours, surpassing Phantom-14B (53.17%, about 30K hours) and remaining close to stronger large-scale baselines. Our CogVideoX model reaches 50.05% total score with the same 288-hour budget

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

CustomCrafterStill-Moving

PhantomVideoBoothVidu2.0

[Figure 69]

[Figure 70]

[Figure 71]

OursCogVideoX

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

OursOursOursWanWanCogVideoX

[Figure 78]

[Figure 79]

[Figure 80]

VACE

[Figure 81]

+ “teddy bear running on a country road”

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

OursOursWanCogVideoX

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

+ “A man holds the backpack at his hand at metro, while he sits on the metro's seat.”

+ “cat running through the fallen leaves in an autumn forest”

- Fig. 6: Qualitative comparison with zero-shot methods (left) and persubject tuning methods (right). Ours CogvideoX denotes our model fine-tuned on CogVideoX-5B, and Ours Wan denotes our model fine-tuned on Wan 2.2-5B. Note that ours is zero-shot, requiring no per-subject tuning at inference time.

and achieves the best GmeScore (70.53%) among the listed methods. VACE14B attains the highest total score (58.00%) but requires about 210K training hours, highlighting the favorable compute-performance trade-off of our approach.

Qualitative Analysis. As shown on the left of Fig. 6, our method produces sharper subject details and more consistent identity than tuning-free baselines such as Vidu 2.0 and VideoBooth, while remaining competitive with recent SDV-Gen methods (Phantom and VACE). In the backpack example, VideoBooth largely fails to follow the prompt, collapsing to unrelated close-up frames. Vidu 2.0 preserves the backpack appearance but shows less stable motion across frames. In contrast, our CogVideoX and Wan variants better retain fine backpack textures and maintain coherent foreground motion, consistent with the VBench gains in Tab. 2.

Table 4: Ablation study on training strategy.

Motion Dynamic

Method

CLIP-T CLIP-I DINO-I Smoothness Degree

Image-only 99.60 0.84 32.67 71.15 43.19 Two-stage 96.04 81.51 28.96 84.73 76.13

Ours 98.45 69.64 32.69 77.14 62.88

We also compare against per-subject tuning methods CustomCrafter [55] and Still-Moving [7] using their official samples (Fig. 6, right). For the teddy-bear sequence, CustomCrafter exhibits noticeable shape/texture drift across frames, whereas our results are more identity-faithful. For the cat sequence, Still-Moving tends to lose fine facial details (e.g., whiskers) and shows weaker detail preser-

vation, while our results better maintain subject appearance under motion. We further find that ours-mini, trained with only 4,000 SDI-Gen pairs, remains competitive against these baselines. We discuss this further in the ablation below.

- 5.3 Ablation Study

Training Strategy.

We compare three training strategies: (1) alternating optimization (Ours), (2) image-only, and (3) sequential two-stage (i.e., imageonly → image-to-video) finetuning approaches. Image-only fine-tuning produced high motion smoothness (99.60) because static videos (0.84 dynamic degree) lead to smooth motion, indicating a need for improved dynamics. (Tab. 4) Two-stage fine-tuning improved dynamics (81.51) and similarity (CLIP-I: 84.73) but introduced severe artifacts and identity forgetting. Our dualtask learning excelled, achieving a balance between motion smoothness (98.72) and dynamic degree (60.19).

Table 5: Ablation study on the number of subject-image pairs.

# of Pairs Motion Dynamic

CLIP-T CLIP-I DINO-I Subject-Images Smoothness Degree

Ours-tiny 2,000 98.72 60.71 33.38 74.72 57.03 Ours-mini 4,000 97.99 78.33 33.22 75.87 58.86

Ours 200K 98.45 69.64 32.69 77.14 62.88

Comparison with Gradient Surgery. We also apply Gradient Surgery [58] on top of our dual-task learning, but observe only minor visual differences compared to our default stochastic switching. Given the similar performance but roughly doubled gradient computation, we adopt the simpler stochastic schedule as our default and regard Gradient Surgery as an optional refinement. We provide quantitative and qualitative comparisons in the supplement.

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

OursOurs-tiny

[Figure 107]

[Figure 108]

✅ ❌

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

###### Ours-mini

[Figure 113]

[Figure 114]

✅ ✅

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

✅ ✅

[Figure 121]

+ “A man holds the backpack at his hand at metro, while he sits on the metro's seat.”

+ “A pig surfing on a blue wave"

[Figure 122]

Fig. 7: Ablation on varying the number of subject-image pairs. When it reduces to 2,000 pairs (Ours-tiny), it shows less dynamic motions and failure cases.

Effect of Scaling the Image Dataset. While our method remains robust with only 4,000 subject-image pairs (Ours mini), increasing the dataset to the full 200K consistently improves fine-grained subject fidelity and temporal realism. With smaller subject-image pairs, we occasionally observe a copy-paste artifact where the subject appears overly rigid across time (e.g., a near-identical appearance/layout being reused), even though identity is preserved. Importantly, this issue is not well captured by identity metrics (CLIP-I, DINO-I), as these scores largely reward consistency in subject identity and can remain high even when motion-dependent pose changes are under-realized.

Using the full 200K pairs mitigates this copy-paste behavior by encouraging more natural, motion-consistent identity rendering. For example, in the pig

###### Ours CogVideoX Ours Wan

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

+ “A man gently clutching a bouquet of vibrant flowers ….” + “The video features a young man, wearing green sleeveless top and red headphones. Background vibrant neon light….”

- Fig. 8: Human-driven Video Generation. Left: CogVideoX. Right: Wan. Without training on human-centric datasets, our model still preserves facial identity well.

surfing case, the full-set model is more likely to reflect viewpoint and pose variations (e.g., head turns and body articulation) instead of reusing a static subject template, aligning with the qualitative trend in Fig. 7 and the stronger identity scores in Table 5. Overall, these observations suggest that identity injection scales: a small subject-image pairs is sufficient to establish the identity mapping, while larger subject-image pairs coverage reduces copy-paste artifacts and better preserves high-frequency details under complex motions.

#### 5.4 Limitations

Our primary fine-tuning dataset, Subject-200K [46], is largely object-centric and includes very few human faces. Despite this limited face coverage, our model generalizes reasonably well to facial subjects, preserving recognizable identity (Fig. 8). FaceSim scores on OpenS2V that are comparable to commercial models (Table 3). However, failures can still occur under strong stylization/extreme appearance changes, and motion can be challenged by rare high-dynamic trajectories that are under-represented in our current video pool.

Regarding the Motion score in Table 3, we verify that our extended CogVideoX

model preserves the motion magnitude of the vanilla model, indicating that motion variation is not degraded even after our recipe is applied (details in the supplement). However, it is lower than state-of-the-art subject-driven methods, suggesting room for improvement.

### 6 Conclusion

We recast subject-driven video generation as a dual-task problem that combines identity injection from subject-image pairs with motion-awareness preservation from a small set of arbitrary videos. Using a simple stochastic task-switching scheme, we train with 200K subject-image pairs and 4,000 arbitrary videos in 288

- A100-hours, achieving competitive subject fidelity and motion quality while generalizing to unseen subjects in a zero-shot manner. Across backbones, gradient measurements show that the image and video objectives rapidly become nearly orthogonal, enabling stable task switching without explicit gradient surgery and delivering strong identity fidelity, motion quality, and text alignment compared to both per-subject-tuned and zero-shot SDV-Gen baselines.

Ethics Statement. This work is purely a research project. Currently, we have no plans to incorporate the developed method into a product or expand access to the public. Our research paper accounts for the ethical concerns associated with video generation research. To mitigate issues associated with training data, we have implemented a rigorous filtering process to remove inappropriate content, such as explicit imagery and offensive language, from our training data, thereby minimizing the likelihood of generating inappropriate content.

### References

- 1. Atzmon, Y., Gal, R., Tewel, Y., Kasten, Y., Chechik, G.: Multi-shot character consistency for text-to-video generation (2024)
- 2. Avrahami, O., Hertz, A., Vinker, Y., Arar, M., Fruchter, S., Fried, O., Cohen-Or, D., Lischinski, D.: The chosen one: Consistent characters in text-to-image diffusion models. In: ACM SIGGRAPH 2024 conference papers. pp. 1–12 (2024)
- 3. Bar-Tal, O., Chefer, H., Tov, O., Herrmann, C., Paiss, R., Zada, S., Ephrat, A., Hur, J., Liu, G., Raj, A., et al.: Lumiere: A space-time diffusion model for video generation. In: SIGGRAPH Asia 2024 Conference Papers. pp. 1–11 (2024)
- 4. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127

(2023)

- 5. Chan, K.C., Zhao, Y., Jia, X., Yang, M.H., Wang, H.: Improving subject-driven image synthesis with subject-agnostic guidance. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6733–6742 (2024)
- 6. Chaudhry, A., Ranzato, M., Rohrbach, M., Elhoseiny, M.: Efficient lifelong learning with a-gem. arXiv preprint arXiv:1812.00420 (2018)
- 7. Chefer, H., Zada, S., Paiss, R., Ephrat, A., Tov, O., Rubinstein, M., Wolf, L., Dekel, T., Michaeli, T., Mosseri, I.: Still-moving: Customized video generation without customized video data. ACM Transactions on Graphics (TOG) 43(6), 1–11 (2024)
- 8. Chen, G., Lin, D., Yang, J., Lin, C., Zhu, J., Fan, M., Zhang, H., Chen, S., Chen, Z., Ma, C., et al.: Skyreels-v2: infinite-length film generative model (2025). URL https://arxiv. org/abs/2504.13074
- 9. Chen, H., Zhang, Y., Cun, X., Xia, M., Wang, X., Weng, C., Shan, Y.: Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7310–7320 (2024)
- 10. Chen, H., Zhang, Y., Wu, S., Wang, X., Duan, X., Zhou, Y., Zhu, W.: Disenbooth: Identity-preserving disentangled tuning for subject-driven text-to-image generation. In: The Eleventh International Conference on Learning Representations (2023)
- 11. Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., Li, Z.: Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis (2023)
- 12. Chen, T.S., Siarohin, A., Menapace, W., Fang, Y., Lee, K.S., Skorokhodov, I., Aberman, K., Zhu, J.Y., Yang, M.H., Tulyakov, S.: Multi-subject open-set personalization in video generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6099–6110 (2025)

- 13. Chen, Z., Li, B., Ma, T., Liu, L., Liu, M., Zhang, Y., Li, G., Li, X., Zhou, S., He, Q., et al.: Phantom-data: Towards a general subject-consistent video generation dataset. arXiv preprint arXiv:2506.18851 (2025)
- 14. Ding, G., Zhao, C., Wang, W., Yang, Z., Liu, Z., Chen, H., Shen, C.: Freecustom: Tuning-free customized image generation for multi-concept composition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9089–9098 (2024)
- 15. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)
- 16. French, R.M.: Catastrophic forgetting in connectionist networks. Trends in cognitive sciences 3(4), 128–135 (1999)
- 17. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022)
- 18. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., et al.: Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103 (2024)
- 19. He, X., Liu, Q., Qian, S., Wang, X., Hu, T., Cao, K., Yan, K., Zhang, J.: Idanimator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275 (2024)
- 20. He, Y., Yang, T., Zhang, Y., Shan, Y., Chen, Q.: Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221 (2022)
- 21. Hong, W., Ding, M., Zheng, W., Liu, X., Tang, J.: Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868

(2022)

- 22. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022)
- 23. Hu, T., Yu, Z., Zhou, Z., Liang, S., Zhou, Y., Lin, Q., Lu, Q.: Hunyuancustom: A multimodal-driven architecture for customized video generation. arXiv preprint arXiv:2505.04512 (2025)
- 24. Hu, Z., Xu, D.: Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073

(2023)

- 25. Huang, Y., Yuan, Z., Liu, Q., Wang, Q., Wang, X., Zhang, R., Wan, P., Zhang, D., Gai, K.: Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698 (2025)
- 26. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21807–21818 (2024)
- 27. Jiang, Y., Wu, T., Yang, S., Si, C., Lin, D., Qiao, Y., Loy, C.C., Liu, Z.: Videobooth: Diffusion-based video generation with image prompts. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6689– 6700 (2024)
- 28. Jiang, Z., Han, Z., Mao, C., Zhang, J., Pan, Y., Liu, Y.: Vace: All-in-one video creation and editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17191–17202 (2025)

- 29. Jin, W., Dai, Q., Luo, C., Baek, S.H., Cho, S.: Flovd: Optical flow meets video diffusion model for enhanced camera-controlled video synthesis. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2040–2049 (2025)
- 30. jovianzm: Pexels-400k. https://huggingface.co/datasets/jovianzm/Pexels400k (2025), accessed: 2026-03-05
- 31. Kumari, N., Zhang, B., Zhang, R., Shechtman, E., Zhu, J.Y.: Multi-concept customization of text-to-image diffusion. In: CVPR (2023)
- 32. Li, D., Li, J., Hoi, S.: Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems 36, 30146–30166 (2023)
- 33. Liu, J., Qu, Y., Yan, Q., Zeng, X., Wang, L., Liao, R.: Fr\’echet video motion distance: A metric for evaluating motion consistency in videos. arXiv preprint arXiv:2407.16124 (2024)
- 34. Liu, L., Ma, T., Li, B., Chen, Z., Liu, J., Li, G., Zhou, S., He, Q., Wu, X.: Phantom: Subject-consistent video generation via cross-modal alignment. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14951–14961

(2025)

- 35. Liu, Z., Zhang, Y., Shen, Y., Zheng, K., Zhu, K., Feng, R., Liu, Y., Zhao, D., Zhou, J., Cao, Y.: Customizable image synthesis with multiple subjects. Advances in neural information processing systems 36, 57500–57519 (2023)
- 36. Lopez-Paz, D., Ranzato, M.: Gradient episodic memory for continual learning. Advances in neural information processing systems 30 (2017)
- 37. McCloskey, M., Cohen, N.J.: Catastrophic interference in connectionist networks: The sequential learning problem. In: Psychology of learning and motivation, vol. 24, pp. 109–165. Elsevier (1989)
- 38. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., et al.: Human-level control through deep reinforcement learning. nature 518(7540), 529–533 (2015)
- 39. Mou, C., Wang, X., Xie, L., Wu, Y., Zhang, J., Qi, Z., Shan, Y.: T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In: Proceedings of the AAAI conference on artificial intelligence. vol. 38, pp. 4296–4304 (2024)
- 40. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)
- 41. Robins, A.: Catastrophic forgetting, rehearsal and pseudorehearsal. Connection Science 7(2), 123–146 (1995)
- 42. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 43. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine-tuning text-to-image diffusion models for subject-driven generation. In: CVPR (2023)
- 44. Shin, H., Lee, J.K., Kim, J., Kim, J.: Continual learning with deep generative replay. Advances in neural information processing systems 30 (2017)
- 45. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024)
- 46. Tan, Z., Liu, S., Yang, X., Xue, Q., Wang, X.: Ominicontrol: Minimal and universal control for diffusion transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14940–14950 (2025)
- 47. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: European conference on computer vision. pp. 402–419. Springer (2020)

- 48. Tewel, Y., Kaduri, O., Gal, R., Kasten, Y., Wolf, L., Chechik, G., Atzmon, Y.: Training-free consistent text-to-image generation. ACM Transactions on Graphics (TOG) 43(4), 1–18 (2024)
- 49. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)
- 50. Wang, J., Lin, Z., Wei, M., Zhao, Y., Yang, C., Loy, C.C., Jiang, L.: Seedvr: Seeding infinity in diffusion transformer towards generic video restoration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2161–2172 (2025)
- 51. Wang, X., Fu, S., Huang, Q., He, W., Jiang, H.: Ms-diffusion: Multi-subject zeroshot image personalization with layout guidance. arXiv preprint arXiv:2406.07209

(2024)

- 52. Wei, Y., Zhang, S., Qing, Z., Yuan, H., Liu, Z., Liu, Y., Zhang, Y., Zhou, J., Shan, H.: Dreamvideo: Composing your dream videos with customized subject and motion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6537–6549 (2024)
- 53. Wei, Y., Zhang, S., Yuan, H., Wang, X., Qiu, H., Zhao, R., Feng, Y., Liu, F., Huang, Z., Ye, J., et al.: Dreamvideo-2: Zero-shot subject-driven video customization with precise motion control. arXiv preprint arXiv:2410.13830 (2024)
- 54. Wu, J., Li, X., Zeng, Y., Zhang, J., Zhou, Q., Li, Y., Tong, Y., Chen, K.: Motionbooth: Motion-aware customized text-to-video generation. Advances in Neural Information Processing Systems 37, 34322–34348 (2024)
- 55. Wu, T., Zhang, Y., Wang, X., Zhou, X., Zheng, G., Qi, Z., Shan, Y., Li, X.: Customcrafter: Customized video generation with preserving motion and concept composition abilities. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 8469–8477 (2025)
- 56. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024)
- 57. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023)
- 58. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.: Gradient surgery for multi-task learning. Advances in neural information processing systems 33, 5824–5836 (2020)
- 59. Yuan, S., He, X., Deng, Y., Ye, Y., Huang, J., Lin, B., Luo, J., Yuan, L.: Opens2vnexus: A detailed benchmark and million-scale dataset for subject-to-video generation. arXiv preprint arXiv:2505.20292 (2025)
- 60. Yuan, S., Huang, J., He, X., Ge, Y., Shi, Y., Chen, L., Luo, J., Yuan, L.: Identitypreserving text-to-video generation by frequency decomposition. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 12978–12988 (2025)
- 61. Zeng, Y., Patel, V.M., Wang, H., Huang, X., Wang, T.C., Liu, M.Y., Balaji, Y.: Jedi: Joint-image diffusion models for finetuning-free personalized text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6786–6795 (2024)
- 62. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3836–3847 (2023)

- 63. Zhang, Y., Liu, Y., Xia, B., Peng, B., Yan, Z., Lo, E., Jia, J.: Magicmirror: Idpreserved video generation in video diffusion transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14464–14474 (2025)
- 64. Zhang, Y., Song, Y., Liu, J., Wang, R., Yu, J., Tang, H., Li, H., Tang, X., Hu, Y., Pan, H., et al.: Ssr-encoder: Encoding selective subject representation for subjectdriven generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8069–8078 (2024)

## Supplementary Material

Learning Zero-Shot Subject-Driven Video Generation Using 1% Compute

Supplementary Contents

Notes. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- A Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21 A.1 Training Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21 A.2 Backbone-Specific Configurations . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- B Gradient Measurement and Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- B.1 Measurement Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- B.2 Image-only Finetuning and Comparison . . . . . . . . . . . . . . . . . . . . . . 25
- B.3 Gradient Surgery Details and Comparison . . . . . . . . . . . . . . . . . . . . 26

- C Theoretical Analysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- C.1 Setup and Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- C.2 Local Second-Order Mixture Model . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- C.3 Dual-Task Learning and Gradient Interaction . . . . . . . . . . . . . . . . . 29
- C.4 Image-Only Fine-Tuning in the Same Model . . . . . . . . . . . . . . . . . . 31
- C.5 Effect of Gradient Surgery in the Local Model . . . . . . . . . . . . . . . . 32

- D Training Strategies & Ablation Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

- D.1 Effect of the Reference Token . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- D.2 T2V vs. I2V Fine-tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- D.3 Effect of Varying the Video Dataset Size . . . . . . . . . . . . . . . . . . . . . 34
- D.4 Effect of Varying the Switching Probability . . . . . . . . . . . . . . . . . . . 35
- D.5 Effect of Random Initial Frame Selection & Dropping. . . . . . . . . . 36
- D.6 Effect of LoRA Rank on Training. . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- D.7 Effect of Training Strategies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
- D.8 Exploration of Various Training Strategies with Penalties . . . . . . . 38

- E Additional Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

- E.1 Human Preference Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- E.2 Temporal Modeling Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

- F Estimation of Other Methods’ Training Cost . . . . . . . . . . . . . . . . . . . . . . 42

- F.1 CustomCrafter. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- F.2 VACE. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- F.3 VideoBooth. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

- G Limitation Analysis & Additional Result . . . . . . . . . . . . . . . . . . . . . . . . . . 43

### Notes

We use green color to refer to figures, tables in the main manuscript (e.g., Figure 2) and use black color to refer to figures, tables in this supplementary material (e.g., Figure 1).

#### Algorithm 1: Dual-task Learning for Subject-Driven Video Generation

Input : Pretrained CogVideoX-5B fθ with LoRA; paired image set Dimg; proxy video set Dvid; replay probability p = 0.2; RandomFrameSelect and TokenDrop(pdrop = 0.5).

Output: Fine-tuned fθ balancing identity fidelity and motion coherence Problem Decomposition:

- – Identity learning: inject subject features from Dimg with random crop/jitter and view drop
- – motion-awareness preservation: maintain motion priors via proxy replay from Dvid

foreach iteration t ← 1, . . . , Tmax do Sample u ∼ U(0, 1) for stochastic task selection if u < p then

/* Motion-awareness preservation: video path */ Decode batch {(Ti, Vi)} from Dvid to 720 × 480, 8 fps (max 49 frames) fref ← RandomFrameSelect(Vi) // mitigate frame bias Xref ← TokenDrop(VAE(fref), pdrop) // drop ref tokens Compute Lvid(Ti, Vi, Xref) via v-prediction

gvid ← ∇θLvid; θ ← θ − ηgvid else

/* Identity learning: image path */ Decode paired views {(Iisrc, Iitgt)} from Dimg with crop jitter and

optional view drop Apply LoRA-only updates on subject-token pathways Compute Limg(Iisrc, Iitgt) via v-prediction gimg ← ∇θLimg; θ ← θ − ηgimg

/* Optional: monitor gradient interactions (off by default) */ if t mod 100 = 0 then

ϕt ← cos(gimg, gvid) // conflict tracking / orthogonality return fθ

### A Implementation Details

#### A.1 Training Protocol

Our subject-driven video generation (SDV-Gen) framework follows Algorithm 1. We initialize a pre-trained multi-modal diffusion transformer fθ (e.g., CogVideo X-5B [56]), which already provides strong text–visual alignment and motion priors.

Training uses two data sources: a subject-image pairs Dimg, consisting of image pairs (I(1),I(2)) of the same subject under different poses or contexts, and an arbitrary video dataset Dvid, consisting of text–video pairs (T,V ). Our key idea is to decompose SDV-Gen into two complementary objectives, identity injection

and motion-aware preservation, and optimize them by stochastic interleaving during training.

At each iteration, we sample u ∼ U(0,1). If u < p, we enter the motion-aware phase and sample a mini-batch of arbitrary videos (Ti,Vi). We then optimize the video reconstruction loss Lvid(Ti,Vi) to preserve or recover realistic motion dynamics. In this phase, RandomFrameSelect and TokenDrop are applied to prevent the model from overfitting to a single reference frame.

Otherwise, if u ≥ p, we enter the identity phase and sample a mini-batch of paired subject images (Ii(1),Ii(2)) from Dimg. We optimize the identity injection loss Limg(Ii(1),Ii(2)) while updating only the LoRA parameters associated with the subject-conditioning pathway Xin. This preserves the model’s original capacity for text conditioning CT and output generation Xout.

Unless otherwise noted, we use p = 0.2. This stochastic alternation balances subject fidelity and temporal consistency throughout training, avoiding the drawbacks of purely sequential fine-tuning or single-objective optimization. After Tmax iterations, the resulting model acts as a zero-shot SDV-Gen generator without requiring a large-scale annotated SDV-Gen dataset.

#### A.2 Backbone-Specific Configurations

We validate the generality of our method on two backbones, CogVideoX-5B and Wan 2.2-5B. The overall training recipe is shared across backbones, while several design choices are adapted to match each architecture. CogVideoX is based on MM-DiT, whereas Wan uses a cross-attention-based DiT variant.

Shared Configuration We insert rank-128 LoRA adapters with scaling α = 64 into the transformer. The adapters are applied to the attention projections (to_q, to_k, to_v, to_out.0) and selected MLP and normalization-related linear projections (e.g., proj, text_proj, norm1.linear, norm2.linear, ff.net.2). All base weights remain frozen, and only the LoRA parameters are updated. Normalization parameters themselves are not directly optimized; any normalizationrelated adaptation is induced through the attached LoRA modules. We denote the union of all trainable parameters by θtrain, which is also the set used for gradient logging in Sec. B. Gradient checkpointing is enabled for memory efficiency.

CogVideoX-5B We train CogVideoX-5B in bfloat16 with TF32 enabled, and use tiling/slicing to improve memory efficiency. Training sequences contain up to 49 frames at 720 × 480 resolution and 8 or 16 FPS. In practice, videos are first decoded at a higher frame rate and then sub-sampled to 8 or 16 FPS for training.

We optimize the model using AdamW with learning rate 5 × 10−5, β1 = 0.9, β2 = 0.95, cosine-with-restarts scheduling with 200 warmup steps, and gradient clipping at 1.0. We train for 30 epochs with a global batch size of 32, and save checkpoints every 50 steps. In the identity phase, we apply per-view random

crop jitter and randomly drop one reference view with probability 0.5 to reduce pose overfitting.

Wan 2.2-5B We further evaluate our method on Wan 2.2-5B using the same overall training protocol. Compared with CogVideoX, several backbone-specific adjustments are required. First, because Wan 2.2-5B is a TI2V model, timestep conditioning must be factorized between the reference frame and the latent video frames through masking. Second, the positional encoding treatment differs across backbones. While CogVideoX works well with a continuous extension of RoPE [45], we found that directly extending RoPE in Wan does not perform well. Third, to better match the native capacity of Wan 2.2-5B, we use sequences of up to 81 frames at 1280 × 704 resolution and 16 FPS.

Unless otherwise noted, we keep the same AdamW optimizer, learning-rate schedule, warmup, gradient clipping, and batch-size configuration as in CogVideoX.

### B Gradient Measurement and Analysis

#### B.1 Measurement Protocol

Measuring full gradients of a multi-billion-parameter model is memory-intensive. We therefore follow multi-task learning practice and compute on the LoRA parameters actually updated during fine-tuning, denoted θtrain.

Trained Parameters. We compute statistics over the same trainable set θtrain defined in Sec. A.1 (LoRA adapters on attention and selected projection layers; base weights and normalization scale/bias remain frozen).

At a checkpoint θt, we freeze parameters and assemble two validation minibatches, using the same preprocessing as training unless noted:

- – Subject-image pairs: ((I(1),P),I(2)) sampled from the validation split, disjoint from training subjects. Images are decoded to 720×480 with the same preprocessing as training (random crop jitter enabled; no view drop during measurement). Prompts P are tokenized exactly as in training.
- – Arbitrary videos: (T,V ) sampled from the validation split of the unapired video corpus. Videos are decoded on-the-fly, resized to 720×480, and truncated to at most 49 frames rendered at 8 or 16 fps. A reference frame is selected uniformly at random (RandomFrameSelect).

Timestep Sampling and Objective. We use the same diffusion scheduler as in Sec. A.1. For each sample, draw a timestep τ ∼ Unif{0,...,N−1} and add Gaussian noise according to the scheduler; the target is formed via velocity parameterization. We then compute gradients with respect to θtrain:

gimg(t) = ∇θtrain Limg I(1),I(2),P , gvid(t) = ∇θtrain Lvid(T,V ).

1.0

2.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Image

GradientAlignment

2.0

Video

0.5

GradNorm

1.5

0.0

1.0

0.5

0.5

1.0

0.0

0 250 500 750 1000

0 250 500 750 1000

Step

Step

- Fig. 9: Gradient alignment in CogVideoX-5B between image and video batches during image-only finetuning and gradient norms of two objectives. Recorded gradient alignment and norm every 50 steps.

DDP Pre-sync Capture and Aggregation. In distributed data parallel training, we record pre-allreduce gradients to avoid distortion from cross-rank synchronization. For each dataset (Subject-image pairs/Arbitrary videos) and before any optimizer step:

- 1. Register backward hooks on modules containing θtrain to copy per-parameter

.grad into a preallocated flat buffer in a fixed parameter order.

- 2. Capture the flat gradient on each rank before the framework performs allreduce.
- 3. Reconstruct the global gradient by reducing the sum of the flat buffers across ranks and dividing by the world size (equivalent to an average pre-sync gradient at the current step).
- 4. Move the aggregated flat gradient to the CPU for logging to minimize device memory pressure. When gradient accumulation is used, we first accumulate local micro-batches, then capture the pre-sync aggregate.

Flattening and Layer-wise Grouping. Let g˜img(t) and g˜vid(t) be the aggregated flat vectors formed by concatenating per-parameter gradients from θtrain in a deterministic module/parameter order, skipping None entries. We also maintain indices to recover layer-wise blocks (e.g., attention vs. MLP-associated projections) for per-group statistics.

Cosine Similarity and Norms. We report the cosine similarity and L2 norms:

ϕ(t) = ⟨g˜img(t),g˜vid(t)⟩ ∥g˜img(t)∥2 ∥g˜vid(t)∥2 + ε

, ∥g˜img(t)∥2,∥g˜vid(t)∥2,

with a small ε to avoid division by zero. When desired, we compute the same statistics per layer group by slicing the flat vectors using the stored indices.

Cadence and Compared Strategies. We repeat the above every K training steps (we use K=5 to 50 depending on the experiment.), holding the evaluation seed fixed and reusing the same validation batches over time for comparability. We compare: (a) SDI-Gen-only fine-tuning (p=0), (b) our stochastic dual-task learning (p=0.2).

#### B.2 Image-only Finetuning and Comparison

We now probe the gradient behaviour of a pure SDI-Gen set fine-tuning regime (p=0), where all optimizer steps are taken on Limg and the video loss Lvid is never used for training. At checkpoints between steps 50 and 1000, we freeze θ and measure gimg(t) and gvid(t) on fixed held-out image and video mini-batches following the protocol in Sec. B; the resulting norms and cosine similarities are shown in Fig. 9.

Norms vs. the Dual-task Schedule. In the main paper, Fig. 5 shows that under our stochastic dual-task schedule the norms of gimg(t) and gvid(t) rapidly settle to a similar and relatively small scale after the first ∼100 steps and remain nonnegligible but stable thereafter. In contrast, the image-only run in Fig. 9 exhibits persistently large and highly fluctuating image gradients: ∥gimg(t)∥2 grows well beyond the range observed in Fig. 5 and does not decay to a steady plateau. At the same time, the video gradient norm stays clearly non-zero and is substantially larger than in the dual-task case, even though Lvid is never optimized. This indicates that image gradient updates continue to inject strong gradients into the shared LoRA subspace along directions that the video objective remains sensitive to, rather than gradually stabilizing that subspace, as in the dual-task setting.

Alignment vs. Emergent Orthogonality. Figure 5 in the main paper shows that, under dual-task training, the cosine similarity ϕ(t) = cos∠(gimg(t),gvid(t)) rapidly collapses into a narrow band around zero and stays there, revealing an emergent near-orthogonality between identity and temporal gradients while their norms remain non-vanishing. By contrast, in the image-only run, the cosine in Fig. 9 never approaches this orthogonal regime; after an initial transient, it stays strongly positive, typically in the 0.4–0.75 range across checkpoints. Pure image fine-tuning thus does not lead to the decoupling observed in Fig. 5 and instead keeps gimg and gvid acting through highly overlapping directions in the LoRA parameter subspace.

Implications for Temporal Behavior. At first glance, a large positive cosine might seem desirable, since it suggests that the two gradients would cooperate if both were optimized simultaneously. In the image-only setting, however, the optimizer follows only −gimg(t). The strong and persistent alignment in Fig. 9, combined with the large and highly variable gradient norms, implies that each identity step makes a substantial update in directions to which the video loss is also sensitive, yet no video gradient is applied to regularize or correct this drift.

Empirically, this matches the behavior of the “Image-only” baseline in Tab. 4 of the main manuscript, which achieves high motion smoothness but almost zero dynamic degree, indicating near zero-motion (“frozen”) videos despite strong identity scores. Comparing Fig.5 with Fig. 9 thus suggests that our stochastic dual-task schedule is crucial for decoupling identity and temporal directions in parameter space, allowing identity injection to proceed without continuously eroding the pretrained motion prior.

#### B.3 Gradient Surgery Details and Comparison

Gradient Surgery in Our Dual-task Setting. We consider the two objectives used in the main paper (identity injection and motion-awareness preservation) and apply Projected Conflicting Gradient (PCGrad, Gradient Surgery) at the level of task gradients computed on the trainable set θtrain (Sec. A.1). At a given iteration we first obtain pre-synchronization gradients gimg and gvid as in Sec. B. Gradient Surgery then removes components responsible for pairwise conflicts:

g˜img = gimg − 1[⟨gimg,gvid⟩ < 0] ⟨gimg,gvid⟩ ∥gvid∥22 + ε

gvid,

g˜vid = gvid − 1[⟨gvid,gimg⟩ < 0] ⟨gvid,gimg⟩ ∥gimg∥22 + ε

gimg,

where ε stabilizes divisions for small norms. The final update direction is g⋆ = ˜(1 − p)gimg + pg˜vid with changing probability of p. In practice, we implement Gradient Surgery in a buffered fashion: as training proceeds, we maintain a short FIFO buffer of recent pre-sync gradients annotated by domain and apply the projection once both domains have appeared within the buffer. This

avoids doubling the per-iteration cost and lets us trade off temporal smoothing of gradients against recency by changing the buffer size.

[Figure 129]

Gradient Alignment under Gradient Surgery. Figure 10 reports the cosine similarity between gimg and gvid when Gradient Surgery is used during training with two buffer sizes (4 and 8). Both variants quickly push the gradient cosine into a narrow band around zero and keep it there, with only small differences between buffer sizes. The resulting curves are very close to those obtained with our default stochastic dual-task schedule in blue dashed line in

Fig. 10: Gradient alignment of the two objectives with and without Gradient Surgery. The red and orange curves show the alignment obtained with Gradient Surgery using different buffer sizes, while blue dashed curve shows the result without Gradient Surgery, corresponding to Fig. 5 in the manuscript.

###### Table 6: Quantitative comparison on applying Gradient Surgery.

Motion Dynamic

Training Method

CLIP-T CLIP-I DINO-I Smoothness Degree

w/ Gradient Surgery + buffer 8 98.86 50.83 31.28 82.17 74.32 w/ Gradient Surgery + buffer 4 97.74 70.83 33.44 74.52 56.15

w/o Gradient Surgery 98.45 69.64 32.69 77.14 62.88

- Fig. 10 (Equivalent to that of Fig. 5 in the main manuscript), indicating that stochastic interleaving plus proxy replay already induces near-orthogonality between image and video-batch gradients. In particular, increasing the buffer length from 4 to 8 yields only minor changes in the alignment trajectory, suggesting that most of the benefit comes from the dual-task geometry itself rather than from explicit projections over a longer history.

[Figure 130]

[Figure 131]

[Figure 132]

Why We Keep the Simpler Schedule. Although Gradient Surgery slightly accelerates the convergence in early iterations, we observe only marginal differences in downstream metrics and qualitative samples (Fig. 11) compared to our default method. This is consistent with the observation that (i) the stochastic taskswitching with mixing probability p already reduces gradient conflict,

GradientSurgery

Ours Ours+

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

+ As lanterns float into the sky, a fox trots through a quiet village, its ears twitching at distant sounds.

Fig. 11: Qualitative Comparison with Gradient Surgery.

(ii) updates are confined to low-rank LoRA adapters where the representational overlap between tasks is limited, and (iii) the replay ratio biases most steps toward image gradient updates, so genuine conflicts are relatively rare. Given this, the additional complexity of maintaining gradient buffers and performing projections offers limited practical gains in our setting. We therefore adopt the stochastic dual-task schedule without Gradient Surgery as our main training recipe and regard Gradient Surgery as an optional refinement that produces similar gradient-alignment behavior, as illustrated in Fig. 10.

Quantitative Comparison on VBench. We evaluate default stochastic dual-task training (without Gradient Surgery) and its Gradient Surgery-augmented counterpart on VBench. We report text alignment, subject consistency, motion smoothness, and overall quality metrics using the same evaluation protocol and sampling setup as in Sec. A.1. Across settings, we observe that headline VBench scores are similar between the two, with differences within experimental variance; see

- Table 6 for full results.

### C Theoretical Analysis

In this section, we provide a stylized analysis that complements Proposition 4.1 in the main paper and the empirical gradient measurements in Figs. 5 and Figs. 9– 10. We work with a local second-order model for both objectives and show that, under dual-task learning, the gradient inner product decays to zero. In contrast, under image-only fine-tuning, it typically converges to a non-zero constant. We then use the same framework to explain why Gradient Surgery yields only marginal gains once gradients are already nearly orthogonal. Throughout this section, the analysis is intended purely as intuition for the observed gradient dynamics, not as a rigorous description of large diffusion transformers.

#### C.1 Setup and Notation

Let L1 : Rn → R (identity loss) and L2 : Rn → R (temporal loss) be two objectives with gradients

g1(θ) = ∇θL1(θ), g2(θ) = ∇θL2(θ).

Gradient Conflict. We measure the interaction between the two tasks by the cosine of the angle between their gradients:

ϕ(θ) = cos∠ g1(θ),g2(θ) = ⟨g1(θ),g2(θ)⟩ ∥g1(θ)∥∥g2(θ)∥

. (6)

Stochastic Task Switching. The training rule in the main paper alternates between updates on L1 and L2:

θt+1 = θt − ηgt, gt =

- g1(θt), w.p. 1 − p,
- g2(θt), w.p. p,

(7)

so that the expected update direction equals the gradient of the mixture loss

L¯(θ) = (1 − p)L1(θ) + pL2(θ), E[gt | θt] = ∇θL¯(θt). (8)

The dual-task schedule in Sec. 4 of the main paper can therefore be viewed as stochastic gradient descent on L¯, with L1 and L2 sampled according to the mixing probability p.

#### C.2 Local Second-Order Mixture Model

We focus on the behavior in a neighborhood of a point θ⋆ where both tasks are approximately stationary.

- Assumption C.2.1 (Local second-order behavior). There exists a point θ⋆ and symmetric matrices H1,H2 ∈ Rn×n such that, in a neighborhood of θ⋆, the losses are twice differentiable with Hessians Hi(θ⋆) and

Li(θ) ≈ Li(θ⋆) +

- 1

- 2

(θ − θ⋆)⊤Hi(θ − θ⋆), i ∈ {1,2}, (9) so that the corresponding gradients satisfy the local linearization

gi(θ) ≈ Hi(θ − θ⋆), i ∈ {1,2}. (10)

We assume H1 and H2 are positive semi-definite and approximately commute, i.e., they are (up to a small perturbation) simultaneously diagonalizable in this local region.

This second-order approximation is standard around critical points. Commutativity means that near θ⋆ the two tasks share a common set of principal directions, which lets us express the dynamics in a joint eigenbasis.

- Assumption C.2.2 (Positive curvature of the mixture along the trajectory). Let

M := (1 − p)H1 + pH2. (11) We assume that, restricted to the subspace explored by training (the span of θ0 − θ⋆), M is positive definite, i.e., all eigenvalues µk of M on this subspace satisfy

0 < µmin ≤ µk ≤ µmax < ∞. (12)

Note that Assumption C.2.2 only requires positive curvature for the mixture L¯ along the trajectory we care about, rather than global strong convexity of each task.

#### C.3 Dual-Task Learning and Gradient Interaction

Under Assumptions C.2.1–C.2.2, we can make the statement of Proposition 4.1 concrete in this local model.

Proposition C.3.1 (Local model behind Proposition 4.1). Assume the local second-order model (Assumption C.2.1) and the positive-curvature condition on the mixture (Assumption C.2.2). Consider deterministic gradient descent on L¯ with step size 0 < η < 1/µmax,

θt+1 = θt − η∇L¯(θt) = θt − η (1 − p)g1(θt) + pg2(θt) . (13) In this local model, the gradients evolve linearly, and the inner product

At ≡ g1(θt),g2(θt) decays exponentially:

|At| ≤ C (1 − ηµmin)2t, for some constant C > 0, (14) and in particular lim

At = 0.

t→∞

Proof. Let zt := θt − θ⋆. Under the local linearization,

g1(θt) ≈ H1zt, g2(θt) ≈ H2zt, and the gradient of the mixture loss is

∇L¯(θt) ≈ (1 − p)H1zt + pH2zt = Mzt. The gradient descent update (13) therefore becomes

zt+1 = θt+1 − θ⋆ ≈ θt − θ⋆ − ηMzt = (I − ηM)zt.

By Assumption C.2.1, H1 and H2 commute and hence are simultaneously diagonalizable with M (ignoring the small perturbation in this toy model). Let U be the orthonormal matrix of shared eigenvectors and write

H1 = UΛ1U⊤, H2 = UΛ2U⊤, M = UMU⊤,

where Λ1 = diag(λ(1)k ), Λ2 = diag(λ(2)k ) and M = diag(µk). In these coordinates zt = Uct for some coefficients ct ∈ Rn and the update simplifies to

ct+1 = (I − ηM)ct, ct = (I − ηM)tc0, so that each component evolves as

ct,k = (1 − ηµk)tc0,k. The gradients in this basis are

g1(θt) ≈ UΛ1ct, g2(θt) ≈ UΛ2ct, and their inner product is

At ≈ ⟨g1(θt),g2(θt)⟩ = c⊤t Λ1Λ2ct

λ(1)k λ(2)k 1 − ηµk 2tc20,k.

λ(1)k λ(2)k c2t,k =

=

k

k

Using Assumption C.2.2, 0 < µmin ≤ µk ≤ µmax on the subspace reached by training, and since η < 1/µmax we have 0 ≤ 1 − ηµk ≤ 1 − ηµmin < 1. Therefore

|λ(1)k λ(2)k | ∥c0∥2(1 − ηµmin)2t = C(1 − ηµmin)2t,

|At| ≤ max

k

for some constant C > 0 depending only on the initial point and the local Hessians. This proves the exponential decay (14) and hence limt→∞ At = 0 in this local model.

Connection to Proposition 4.1 and Empirical Trends. In the main manuscript we measure the cosine similarity between the image and video gradients, ϕ(t) = cos∠ gimg(t),gvid(t) , and empirically observe that it rapidly converges to a narrow band around zero while the gradient norms remain non-negligible (Figs. 5).

- Proposition C.3.1 shows that, in a local second-order model where the two tasks share a common eigenbasis and the mixture has positive curvature along the trajectory, gradient descent on the mixture loss naturally drives the inner product

⟨g1,g2⟩ (and hence its cosine) toward zero. This toy result should be read as one concrete instance consistent with Proposition 4.1, not as a formal guarantee for deep diffusion transformers.

C.4 Image-Only Fine-Tuning in the Same Model

We now show that in the same framework, pure image-only fine-tuning (p = 0) is not expected to drive the cosine toward zero. This matches the behavior observed in Fig. 9, where the cosine remains strongly positive throughout training.

- Proposition C.4.1 (Image-only fine-tuning). Assume the setting of Assumption C.2.1 and suppose that H1 is positive definite on the subspace explored by training. Consider gradient descent on L1 alone,

θt+1 = θt − ηg1(θt), 0 < η < 2/λmax(H1), (15) and define At = ⟨g1(θt),g2(θt)⟩ and ϕ(t) as in (6). Let k⋆ be an index of an eigenvalue of H1 with maximal magnitude on the trajectory, i.e., |λ(1)k⋆ | = maxk |λ(1)k | and c0,k⋆ ̸= 0. Then in the local model,

ϕ(t) = sign λ(1)k⋆ λ(2)k⋆ , (16)

lim

t→∞

and in particular the cosine converges to a non-zero constant whenever λ(1)k⋆ λ(2)k⋆ ̸= 0.

Proof. Under the image-only update zt+1 = (I − ηH1)zt with zt = θt − θ⋆, the same eigenbasis U as in the previous proof diagonalizes H1 and H2. Writing zt = Uct gives ct+1,k = (1 − ηλ(1)k )ct,k and hence

ct,k = (1 − ηλ(1)k )tc0,k. Because H1 is positive definite on the relevant subspace and 0 < η < 2/λmax(H1), we have |1−ηλ(1)k | < 1 for all k, and the component with largest |λ(1)k | dominates as t → ∞. More precisely, for k ̸= k⋆,

1 − ηλ(1)k 1 − ηλ(1)k⋆

t c0,k c0,k⋆

ct,k ct,k⋆

→ 0.

=

Thus zt aligns with vk⋆, the k⋆th column of U, and the gradients satisfy g1(θt) ≈ λ(1)k⋆ ct,k⋆vk⋆, g2(θt) ≈ λ(2)k⋆ ct,k⋆vk⋆,

up to asymptotically negligible contributions from other modes. Therefore

⟨g1(θt),g2(θt)⟩ ∥g1(θt)∥∥g2(θt)∥

= sign λ(1)k⋆ λ(2)k⋆ ,

lim

ϕ(t) = lim

t→∞

t→∞

which is non-zero whenever λ(1)k⋆ λ(2)k⋆ ̸= 0.

This proposition explains why, in Fig. 9, the cosine between image and video gradients remains strongly positive under image-only fine-tuning. In the local model, the optimization path aligns with a dominant eigen-direction of H1, and if

- H2 has positive curvature along the same direction, then g1 and g2 stay positively aligned instead of becoming orthogonal. Because the optimizer follows only −g1 in this regime, every identity update moves parameters along directions to which the video objective is also highly sensitive, without any counter-balancing video gradient, which is consistent with the near-static yet identity-faithful behavior of the image-only baseline in Table 4.

#### C.5 Effect of Gradient Surgery in the Local Model

We finally use the same framework to explain why Gradient Surgery yields gradient alignment curves similar to our default dual-task schedule (Fig. 10) and only marginal gains in downstream metrics (Table 7).

Gradient Surgery Update for Dual-tasks. For two tasks with gradients g1 and g2, Gradient Surgery projects away conflicting components when ⟨g1,g2⟩ < 0. In the two-task case, the projected gradients are

- g˜1 = g1 − 1[⟨g1,g2⟩ < 0]⟨g1,g2⟩ ∥g2∥2

g2, (17)

- g˜2 = g2 − 1[⟨g2,g1⟩ < 0]⟨g2,g1⟩ ∥g1∥2

g1. (18)

Given mixing probability p, the Gradient Surgery update direction is gPC = (1 − p)˜g1 + pg˜2, whereas the dual-task schedule without projection uses the mixture gradient gmix = (1 − p)g1 + pg2.

Lemma C.5.1 (Difference between Gradient Surgery and mixture update). Let A = ⟨g1,g2⟩ and suppose A < 0 so that Gradient Surgery modifies both gradients. Then the difference between the Gradient Surgery and mixture updates satisfies

∥gPC − gmix∥ ≤ |A|

and in particular

1 − p ∥g2∥

p ∥g1∥

+

≤ 2|A|max

1 ∥g1∥

,

1 ∥g2∥

, (19)

∥gPC − gmix∥ ∥gmix∥

≤ Crel |cos∠(g1,g2)|, (20)

for some constant Crel depending only on the ratio of ∥g1∥ and ∥g2∥.

Proof. When A < 0 we have

and therefore

A ∥g1∥2

A ∥g2∥2

g2, g˜2 − g2 = −

g˜1 − g1 = −

g1,

gPC − gmix = (1 − p)(˜g1 − g1) + p(˜g2 − g2) = −A

Taking norms gives

1 − p ∥g2∥2

p ∥g1∥2

g2 +

g1 .

1 − p ∥g2∥

p ∥g1∥

∥gPC − gmix∥ ≤ |A|

,

+

and the second inequality in (19) follows by bounding each coefficient by the maximum of 1/∥g1∥ and 1/∥g2∥. Dividing by ∥gmix∥ and using |A| = ∥g1∥∥g2∥|cos∠(g1,g2)| yields (20) for a suitable constant Crel.

Lemma C.5.1 shows that the Gradient Surgery update is a small perturbation of the mixture gradient whenever the cosine is already close to zero. Combined with Proposition C.3.1, which predicts that At (and hence the cosine) decays exponentially under dual-task mixing, this indicates that Gradient Surgery can only have a significant effect during an early transient phase when gradients are strongly conflicting. Once the dynamics enter the near-orthogonal regime, Gradient Surgery and the unmodified dual-task schedule follow almost identical update directions.

Connection to Empirical Gradient Surgery Results. In Sec. B.3 and Fig. 10, we empirically compare gradient alignment with and without Gradient Surgery. Both buffered variants of Gradient Surgery quickly drive the cosine into a narrow band near zero and then track the unprojected dual-task curve closely, which is consistent with Lemma C.5.1 and Proposition C.3.1. Because our stochastic dualtask schedule already reduces gradient conflict rapidly and updates are confined to a low-rank LoRA subspace, the additional projections performed by Gradient Surgery mainly act as a small early-time correction and yield only marginal gains in downstream metrics (Table 6). This theoretical picture supports our choice to adopt the simpler stochastic schedule without Gradient Surgery as the default training recipe.

### D Training Strategies & Ablation Studies

In this section, we present extensive ablation studies on training strategies and various techniques used. Due to the cost of training, we use the model trained with a subset of 4,000 of subject-image pairs [46] unless otherwise stated, since they tend to converge a little faster for the comparison in this section, with less degraded performance according to the Table 4 in the main manuscript.

- Table 7: Ablation on the reference token. Adding <CLS> yields improved subject identity scores (CLIP-I, DINO-I) and a higher dynamic degree.

Training Method

Motion Dynamic

CLIP-T CLIP-I DINO-I Smoothness Degree

w/o Ref. token 98.84 54.55 32.87 73.36 57.51 w/ Ref. token 97.99 78.33 33.22 75.87 58.86

- Table 8: Comparison with T2V-only stochastically-switched finetuning, with image drop probability of 1. Switching to text-only input (T2V) moderately boosts CLIP-T but hurts subject fidelity and dynamic degree.

Motion Dynamic

Training Method

CLIP-T CLIP-I DINO-I Smoothness Degree

T2I + T2V (joint) 98.85 44.14 33.41 72.71 48.68 Ours 97.99 78.33 33.22 75.87 58.86

#### D.1 Effect of the Reference Token

Tab. 7 demonstrates how adding a dedicated <CLS> token to the prompt affects our model’s performance. Without this reference token, the model attains slightly higher motion smoothness and marginally better CLIP-T scores, but it underperforms on dynamic degree and identity-focused metrics. Introducing <CLS> evidently improves subject fidelity and fosters more diverse motion. We attribute these gains to the reference token guiding the alignment of subject tokens (Xin) with the textual prompt more explicitly, leading to stronger identity preservation and more coherent motion variations.

#### D.2 T2V vs. I2V Fine-tuning

We also ablate replacing our image-to-video (I2V) fine-tuning with a text-tovideo (T2V) setup. In Tab. 9, the T2I+T2V (joint) attains a slightly better CLIP-T but lower dynamic degree and subject alignment (CLIP-I, DINO-I). This suggests that T2V fine-tuning struggles when introducing a novel subject identity solely through text, resulting in weaker overall identity preservation. By contrast, our I2V approach strikes a better balance, preserving subject details (CLIP-I: 73.70, DINO-I: 59.29) and maintaining sufficient motion (dynamic degree: 60.19).

#### D.3 Effect of Varying the Video Dataset Size

Tab. 9 reports how our method’s performance changes when using different amounts of arbitrary videos for I2V fine-tuning (1K, 2K, 3K, and 4K videos). Notably, with only 1K videos, we already obtain relatively strong results, suggesting that even a small unlabeled corpus can restore temporal consistency to

Table 9: Ablation study on using a different number of videos.

Video Motion Dynamic

CLIP-T CLIP-I DINO-I Count Smoothness Degree

- 1K 99.03 59.66 32.16 72.69 56.98
- 2K 98.96 52.25 32.49 72.13 54.42
- 3K 98.79 55.46 32.04 72.79 55.57
- 4K 97.99 78.33 33.22 75.87 58.86

Table 10: Abalation study on using different ’p’ to choose I2V finetuning.

Video Motion Dynamic

CLIP-T CLIP-I DINO-I Ratio Smoothness Degree

0.0 99.60 0.84 32.67 71.15 43.19 0.2 (Ours) 97.99 78.33 33.22 75.7 58.86 0.4 98.31 59.12 31.94 73.53 56.60 0.6 98.33 60.87 31.31 76.71 62.84

some extent. However, increasing the video count to 4K (our default setting) steadily improves dynamic degree from 59.66 to 78.33 and also boosts identity fidelity (DINO-I) from 56.98 up to 58.56, indicating more consistent subject representation across frames.

We also observe a modest variation in CLIP-T and CLIP-I scores when moving from 1K to 4K videos, suggesting that a larger video dataset helps balance the preservation of subject detail and temporal motion, without overfitting to specific frames or motion patterns. In short, while our method is fairly robust to smaller unlabeled datasets, using around 4K (i.e., 1% of Pexels [30] dataset) videos offers the best trade-off between data efficiency and stable motion/appearance results.

#### D.4 Effect of Varying the Switching Probability

In Tab. 10, we examine how different values of p—the probability of sampling unlabeled video data (I2V fine-tuning)—affect overall performance. When p = 0.0, the model relies solely on subject-driven image generation (SDI-Gen) fine-tuning and achieves a relatively high dynamic degree but moderate identity scores. Increasing p to 0.2 or 0.4 yields balanced improvements across most metrics, reflecting better coordination between identity and motion. At p = 0.6, the model dedicates a greater share of updates to I2V fine-tuning, strengthening identity alignment (CLIP-I: 76.71, DINO-I: 62.84) while keeping the dynamic degree stable (60.87). Although different p values trade off between motion smoothness and identity fidelity to varying degrees, our chosen p = 0.2 demonstrates a strong overall balance, as highlighted in the main paper.

w/o random initial frame

w/o random image drop

Ours

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

+"On the beach, a lady sits under a beach umbrella. She's wearing thishawaiianshirtand has a big smile on her face, with her surfboard hehind her. The sun is setting.

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

+ “A close up view.A bowl of orangesplaced on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show.”

Fig. 12: Effect of random initial frame and image token dropping.

#### D.5 Effect of Random Initial Frame Selection & Dropping.

We further investigate the effects of random frame selection and image-token dropping during I2V fine-tuning through qualitative analysis. As shown in Figure 12, when neither technique is applied, the first reference image exhibits excessive dominance over subsequent frames, resulting in limited temporal variation. Enabling random frame selection alone partially restores temporal dynamics, yet introduces visual artifacts in certain regions (e.g., shirts, oranges). In contrast, the joint application of both techniques achieves a more favorable balance: artifacts are substantially reduced while the reference image blends more naturally into the generated sequence. These qualitative observations underscore the importance of mitigating over-reliance on the first frame and preventing excessive conditioning on a single reference image to achieve smoother, more temporally coherent video generation.

#### D.6 Effect of LoRA Rank on Training

We further study the effect of LoRA rank on training with CogVideoX-5B in Table 11. Quantitatively in Figure 13, the differences across LoRA ranks are relatively small, and the standard evaluation metrics do not show a large gap, as the identity injection capability is not affected a lot with LoRA rank. However, the fidelity of the injected identity is affected a lot, showing clear qualitative trend. When the LoRA rank is reduced below 32, the generated videos exhibit noticeable degradation in visual quality, with weaker textures, blurrier structures, and the loss of fine-grained details. In particular, subtle appearance cues that are important for subject fidelity tend to disappear as the rank becomes

###### Table 11: Abalation study on LoRA Rank on Training.

LoRA Motion Dynamic

CLIP-T CLIP-I DINO-I Rank Smoothness Degree

8 97.80 73.68 32.89 74.65 57.17 32 98.20 71.43 33.31 75.97 59.27 128 (Ours) 97.99 78.33 33.22 75.7 58.86 512 98.44 69.64 33.75 73.97 55.67

too small. When we increase the LoRA rank to 512, the fidelity gets improved in qualitative manner, we chose rank 128 for the memory and compute balance.

#### D.7 Effect of Training Strategies

[Figure 145]

+ “A pig surfing on a blue wave”

We conduct an ablation on three training strategies: image-only, two-stage, and our alternating (stochastically switched) approach. Qualitative comparisons in Figure 14 reveal distinct failure modes for each strategy. The image-only approach produces videos with minimal subject motion, effectively generating near-static sequences that artificially inflate motionsmoothness metrics while sacrificing temporal dynamics, as demonstrated in Table 4 of the main manuscript. Twostage training, which first fine-tunes on subject-image pairs before switching to arbitrary videos, introduces pronounced visual artifacts and exhibits catastrophic forgetting of the base model’s text-tovideo capability. These artifacts manifest as inconsistent textures and structural distortions that compromise overall video quality, despite preserving certain fine-grained image details. In contrast, our proposed alternating strategy successfully navigates these failure modes by interleaving identity injection and motion-aware objectives throughout training. As illustrated in Figure 14, this approach generates videos with both natural motion dynamics and coherent visual appearance, avoiding the static collapse observed in image-only training and the artifact accumulation in two-stage training. The qualitative results demonstrate that stochastic alternation between objectives effectively maintains the model’s generative capabilities across both modalities.

[Figure 146]

[Figure 147]

Rank 8 Rank 32

[Figure 148]

[Figure 149]

Rank 512

Rank 128 (Ours)

Fig. 13: Ablation study on LoRA rank. Identity injection capability is trained even with rank of 8, but the details such as where the eye is located for ‘pig’ differs when rank goes below 32. When rank is above 128, they show good fidelity in identity injection.

Time 6 sec

OursTwo-stageImage-only

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

+ ”Catis rollerblading in the park.”

- Fig. 14: Qualitative result on ablation study of our component in temporal awareness preservation.

- Table 12: Exploration on various training strategies with penalties on imageonly.

Training Motion Dynamic

CLIP-T CLIP-I DINO-I Strategy Smoothness Degree

Ours 97.99 78.33 33.22 75.7 58.86 image-only 99.60 0.84 32.67 71.15 43.19 image-only + L2 99.36 31.67 32.91 73.12 48.23 image-only + L2 anchoring 99.26 38.33 32.64 72.15 44.71

#### D.8 Exploration of Various Training Strategies with Penalties

We further explore whether standard regularization strategies can mitigate the failure modes of the image-only and two-stage training schemes in Table 12 and Figures 15- 16 In particular, we apply commonly used penalties, including L2 regularization, and anchoring to the pretrained weights, during both image-only and two-stage fine-tuning. However, these penalties do not meaningfully alleviate catastrophic forgetting in our setting.

Although such regularizers can slightly constrain parameter drift, they do not recover the pretrained model’s temporal modeling capability once optimization becomes overly biased toward the image objective. As a result, image-only training still tends to produce near-static videos, while two-stage training continues to suffer from degraded video quality and forgetting of the original motion prior. Overall, we find that adding these practical penalties is not sufficient to resolve the core instability of these training strategies, which supports the need for our stochastic alternating schedule.

[Figure 163]

+ “On the beach, a lady sits under a beach umbrella. She's wearing this hawaiian shirt and has a big smile on her face, with her surfboard hehind her. The sun is setting”

[Figure 164]

[Figure 165]

[Figure 166]

##### L2 L2 + anchoring Ours

- Fig. 15: Qualitative result on comparison with image-only approaches with penalty.

+ “On the beach, a lady sits under a beach umbrella. She's wearing this hawaiian shirt and has a big smile on her face, with her surfboard hehind her. The sun is setting”

L2 L2 + anchoring Ours

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

- Fig. 16: Qualitative result on comparison with two-stage approaches with penalty.

### E Additional Evaluation

#### E.1 Human Preference Study

While benchmark metrics offer quantitative insights, they can sometimes be misled by “cheating” behaviors, such as static outputs that yield artificially high scores. To complement our objective measurements, we conducted a human preference study using 20 randomly chosen samples from each baseline and our approach, without cherry-picking. A total of 30 participants were asked to rate the generated videos on a five-point Likert scale across dimensions of ID consistency, Prompt alignment, Motion quality, and Overall visual appeal. Our method consistently outperformed the baselines, suggesting that our balanced approach to identity preservation and temporal awareness best aligns with human judgments of video realism and quality when viewed holistically.

#### E.2 Temporal Modeling Evaluation

We evaluate temporal modeling performance using videos sampled from the Pexels dataset, ensuring no overlap with those used during training. Following

###### Table 13: Exploration on various training strategies with penalties on twostage.

Training Motion Dynamic

CLIP-T CLIP-I DINO-I STrategy Smoothness Degree

Ours 97.99 78.33 33.22 75.7 58.86 two-stage 96.04 81.51 28.96 84.73 76.13 two-stage + L2 97.06 81.67 31.10 77.14 62.15 two-stage + L2 anchoring 96.60 78.33 31.16 77.71 61.27

Table 14: Result on human preference study in Likert scale of 1-5.

ID Prompt Motion Overall Consistency Alignment Quality Quality

Method

Omini+I2V 3.80 3.78 3.62 3.44 VideoBooth 3.25 3.20 3.08 2.91 Vidu 2.0 3.42 3.24 3.22 3.03

Ours 4.08 3.82 3.88 3.71

the protocol of FloVD [29], we categorize videos into three groups based on optical flow magnitude (small: ≤ 25, medium: 25 ∼ 50, large: ≥ 50) to enable detailed analysis. For object-motion evaluation in particular, we focus on videos with minimal or no camera motion and construct benchmark subsets by applying the same motion-based grouping to a separate collection of Pexels videos not used in our stochastically switched fine-tuning.

#### Evaluation Protocol

- 1) Foreground–Background Segmentation. For each video, we use an offthe-shelf segmentation model (e.g., Grounded-SAM2) on the first frame to separate foreground and background regions. This allows us to measure object (foreground) motion independently from any camera-induced background shifts.
- 2) Optical Flow Computation. We estimate optical flow between the first frame and each subsequent frame using a standard flow estimator (e.g., RAFT [47]).

Let uf(x) and ub(x) denote the per-pixel flow vectors for the foreground and background pixels, respectively, at position x. We record:

FlowMagf =

1

Nf x∈fg∥uf(x)∥, FlowMagb =

1 Nb x∈bg∥ub(x)∥,

where Nf and Nb are the respective pixel counts in the foreground and background masks.

- 3) Dataset Filtering. To ensure negligible camera motion, we discard any video whose average magnitude of background flow FlowMagb exceeds 10 pixels.

Table 15: Temporal Evaluation following FloVD [29], assessing whether motion dynamics improve compared to image-only or two-stage training. Small - Medium - Large - with each number representing FVD. † denotes Pexels [30]-finetuned version of CogVideoX [56].

Method Small↓ Medium↓ Large↓ CogVideoX-T2V† 597.54 594.26 573.86 Image-only 641.92 636.42 680.34 Two-stage 801.97 872.30 824.03 Ours 512.30 511.66 550.14

This filtering step excludes scenes with significant global shifts, retaining only those with primarily object-centric motion.

- 4) Category Assignment. Based on the average magnitude of foreground flow FlowMagf (averaged over all frames), we categorize videos into:

- – Small: 0 ≤ FlowMagf ≤ 25
- – Medium: 25 < FlowMagf ≤ 50
- – Large: FlowMagf > 50

Each category contains 300 videos, ensuring a balanced evaluation of low-, moderate-, and high-motion scenarios.

- 5) Evaluation Protocol. Within each subset, we use only the first frame (including any textual or reference cues, if required) to generate a video of the same length. We then compute FVD [33] between the generated outputs and the ground-truth videos. By comparing FVD across small, medium, and large motion classes, we obtain a clearer picture of how each model (ours vs. baselines) adapts to varying object-motion intensities.

Discussion. We compared image-only, two-stage, and our dual-task learning strategies. Image-only training produced a video with poor FVD (Tab. 15. Twostage fine-tuning leads to high FVD, which aligns with the result of artifacts as in Fig. 14. Our strategy (Ours) excelled, with lower FVD scores confirming superior motion realism and temporal consistency, on par with CogVideoX [56].

One thing to note is that motion-focused split highlights each method’s strengths and weaknesses. For example, a model might produce near-static outputs for low-motion data, cheating on metrics such as smoothness in VBench metrics, yet fail to track fast-moving objects in high-motion videos. As observed in FloVD [29], categorizing by foreground flow magnitude reveals these nuances more effectively than aggregated scores alone.

Additional Details. To ensure a fair comparison with our method, we additionally fine-tune the original CogVideoX [56] on a subset of the Pexels [30] dataset that is equivalent to the one used in our training. Since our model is trained for 4K steps with a sampling ratio p = 0.2, we match this by finetuning CogVideoX

for 800 steps (0.2 × 4000). As a result, we achieved performance comparable to the original CogVideoX in motion dynamics evaluation.

### F Estimation of Other Methods’ Training Cost

We estimate the train time of CustomCrafter [55] and VACE [28] based on the given implementation details in the manuscript.

#### F.1 CustomCrafter.

The estimated GPU hours for training a single subject are in the range of 100– 300 A100-hours, with a median estimate of around 200 A100-hours.

10000 × t 3600

wall-clock time (hours) =

total GPU-hours = 4 × wall-clock time where t ∈ [10,30] is the estimated time per iteration in seconds.

#### F.2 VACE.

The estimated GPU hours for training VACE on LTX-Video are in the range of 9,000–27,000 A100-hours, with a median estimate of around 18,000 A100-hours.

number of training steps = 200000

200000 × t 3600 GPU-hours = 16 ×

wall-clock time per training (hours) =

200000 × t 3600

3200000t 3600

8000t 9

=

=

where t ∈ [10,30] is the estimated time per iteration in seconds. The estimated GPU hours for training VACE on Wan-T2V are in the range of 70,000–210,000 A100-hours, with a median estimate of around 140,000 A100-hours.

number of training steps = 200000

200000 × t 3600 GPU-hours = 128 ×

wall-clock time per training (hours) =

200000 × t 3600

64000t 9 where t ∈ [10,30] is the estimated time per iteration in seconds.

25600000t 3600

=

=

Unlearned Style Leading to Awkward Video Blurry Artifacts

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

+ “The video features a man sitting at a table in a café, sipping his coffee….”

+ “The video features a man jogging along a grassy path next to …”

+ “In the video, a man is hiking through a dense forest. He is wearing….”

Fig. 17: Limitation in human-driven generation.

#### F.3 VideoBooth.

VideoBooth trains two stages (coarse and fine), each for 218,000 steps, totaling 436,000 training steps. We estimate the total compute assuming an 8×A100 setup.

number of training steps = 218,000 + 218,000 = 436,000 wall-clock time per training (hours) =

436,000 × t 3,600 GPU-hours = 8 ×

436,000 × t 3,600

3,488,000t 3,600

- 8,720t

- 9

= where t ∈ [0.8,2.0] is the estimated time per iteration in seconds.

=

Therefore, the estimated GPU hours for training VideoBooth are in the range of 775–1,938 A100-hours, with a median estimate of around 1,356 A100-hours (at t = 1.4).

### G Limitation Analysis & Additional Result

Limitation & Analysis. As discussed in the main manuscript, strong stylization can occasionally hinder faithful generation, leading to failure cases such as those shown in Fig. 17, and in some cases, to blurry outputs. However, this phenomenon is less pronounced when using Wan 2.2-5B as the backbone. We conjecture that this is because Wan 2.2-5B provides stronger identity preservation, as evidenced by its substantially higher FaceSim score in the OpenS2V [59] results reported in Tab. 3 of the main paper.

Additional Qualitative Results. We present additional qualitative comparisons against per-subject tuning baselines in Figs. 18 and 19, as well as zero-shot baselines in Figs. 20 and 21, including state-of-the-art methods [28,34]. For these comparisons, we use Phantom-Wan 2.1-1.3B and VACE-Wan 2.1-1.3B, which are denoted as Phantom-1.3B and VACE-1.3B in the supplemental video for brevity. Note that mini refers to our model trained on a 4,000-sample subset of subjectimage pairs [46].

We further include additional qualitative video results in the supplemental video, along with benchmark results from Open-S2V-Evaluation [59]. In addition, to better evaluate performance on deformable subjects, highly dynamic scenes, and human-driven generation, we compare against Phantom-Wan 14B and VACE-Wan 14B, denoted as Phantom-14B and VACE-14B in the supplemental video. Because 14B-scale models are computationally demanding and require four A100 GPUs for inference, we report the 14B results provided by Open-S2V-Evaluation.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

StillMovingOursWan OursCogVideoX

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

OursWanOursCogVideoXmini

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

+ “pig skiing down a slope”

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

StillMoving OursCogVideoX

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

OursCogVideoXmini

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

+ “a boy riding a race cart”

###### Fig. 18: Additional qualitative comparison with Still-Moving [7]. Note that mini denotes our method trained with a 4K subset of Subject 200K [46].

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

CustomCrafter OursCogVideoX

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

CustomCrafterOursWanOursCogVideoXmini OursCogVideoX

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

+ “a teddy bear is playing guitar”

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

OursCogVideoXOursWanmini

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

+ “anime girl selfie standing under the pink blossoms of a cherry tree”

###### Fig. 19: Additional qualitative comparison with CustomCrafter [55]. Note that mini denotes our method trained with 4K subset of Subject 200K [46].

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

VideoBooth OursCogVideoX

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

OursCogVideoXVidu2.0mini OminiControl

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

VACEPhantom+I2V

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

OursWan

[Figure 301]

+ ”Under the glow of a streetlamp, a toy car comes closer to camera on a wet pavement, reflecting the flickering city lights.”

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

VideoBooth OursCogVideoX

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

OursCogVideoXVidu2.0mini OminiControl

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

VACEPhantom+I2V

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

OursWan

[Figure 342]

+ “A cat cautiously steps through a rainy alleyway, its fur slightly damp as puddles reflect the city lights.”

