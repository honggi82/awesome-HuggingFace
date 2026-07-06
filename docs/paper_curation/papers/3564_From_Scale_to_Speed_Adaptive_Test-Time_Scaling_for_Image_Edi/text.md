## From Scale to Speed: Adaptive Test-Time Scaling for Image Editing

### Xiangyan Qu12∗,†Zhenlong Yuan3∗, Jing Tang3‡, Rui Chen3, Datao Tang3, Meng Yu3, Lei Sun3, Yancheng Bai3, Xiangxiang Chu3, Gaopeng Gou12§, Gang Xiong12, Yujun Cai4,

1Institute of Information Engineering, Chinese Academy of Sciences 2School of Cyber Security, University of Chinese Academy of Sciences

3AMAP, Alibaba Group 4University of Queensland ∗ Equal contribution. ‡ Project lead. § Corresponding author.

# arXiv:2603.00141v3[cs.CV]26Mar2026

### Abstract

- (a) Text-to-Image Generation
- (b) Image Editing

Diverse plausibleoutputs

|[Figure 1]<br><br>[Figure 2]<br><br>…<br><br>[Figure 3]<br><br>[Figure 4]|
|---|

Image-CoT

A cat running on the ground.

Image Chain-of-Thought (Image-CoT) is a test-time scaling paradigm that improves image generation by extending inference time. Most Image-CoT methods focus on text-toimage (T2I) generation. Unlike T2I generation, image editing is goal-directed: the solution space is constrained by the source image and instruction. This mismatch causes three challenges when applying Image-CoT to editing: inefficient resource allocation with fixed sampling budgets, unreliable early-stage verification using general MLLM scores, and redundant edited results from large-scale sampling. To address this, we propose ADaptive Edit-CoT (ADE-CoT), an on-demand test-time scaling framework to enhance editing efficiency and performance. It incorporates three key strategies: (1) a difficulty-aware resource allocation that assigns dynamic budgets based on estimated edit difficulty; (2) editspecific verification in early pruning that uses region localization and caption consistency to select promising candidates; and (3) depth-first opportunistic stopping, guided by an instance-specific verifier, that terminates when intentaligned results are found. Extensive experiments on three SOTA editing models (Step1X-Edit, BAGEL, FLUX.1 Kontext) across three benchmarks show that ADE-CoT achieves superior performance-efficiency trade-offs. With comparable sampling budgets, ADE-CoT obtains better performance with more than 2× speedup over Best-of-N.

Text Prompt

Redundantcorrectoutputs

|[Figure 5]<br><br>…<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|
|---|

Text Prompt Image

[Figure 13]

Image-CoT

Add a cherryeating action.

+

Figure 1. Impact of Image-CoT on different generative tasks. (a) T2I generation is an open-ended task benefiting from largescale sampling. (b) Image editing is a goal-directed task where outputs are constrained by the prompt and source image, leading to redundant correct outputs after large-scale sampling.

Thought (Image-CoT) [48, 90, 105], a test-time scaling strategy, offers a promising approach to tackle this challenge. It provides a plug-and-play, training-free solution that extends the inference time to boost the generation quality.

Most Image-CoT studies focus on text-to-image (T2I) generation. The standard approach samples multiple candidates via noise perturbations [48, 90] and uses Best-of-N (BoN) selection. However, the computational cost scales linearly with the number of samples, making it challenging to generate higher-quality images under a limited inference budget [103]. To address this, some methods incorporate prompt-level intervention through rewriting or reflective updates [38, 111] to increase candidate diversity and image-text alignment. Other work involves path search and pruning during the generation process. These methods utilize MLLMs as verifiers [12, 43, 48, 60, 79, 105] to score intermediate denoising states and select promising candidates. By pruning low-potential samples early, they reduce computational cost without degrading final image quality.

### 1. Introduction

Recent image-editing methods have shown impressive progress by latent-level fusion between multimodal large language models (MLLMs) and diffusion decoders, such as Step1X-Edit [44], FLUX.1 Kontext [34], BAGEL [15], and Qwen-Image [88]. However, their performance remains challenging on complex edits, such as large pose changes, multi-object edits, or multi-turn edits. Image Chain-of-

However, directly applying these T2I-centric Image-CoT methods to image editing is suboptimal due to fundamental differences between the tasks. T2I generation is an openended task that benefits from large-scale sampling and post hoc selection (see Fig. 1(a)). In contrast, image editing is a goal-directed task. The solution space is constrained

†Work done during the internship at AMAP, Alibaba Group.

(b) Misjudgement in Pruned Samples (c) High Redundancy in Large-scale Sampling.

(a) Score Gain in Different Initial Score.

[Figure 14]

[Figure 15]

[Figure 16]

60% 40%

Minimal Gain

| |
|---|

| |
|---|

- Figure 2. Why existing Image-CoT methods are suboptimal for editing. (a) Inefficient resource allocation: Fixed budgets waste computation on simple edits (high initial scores, red box) that show minimal improvement. (b) Unreliable early-stage verification: 40% of samples with low early scores achieve high final scores, but are pruned incorrectly by general MLLM scores. (c) Redundant edited results: Large-scale sampling produces redundant correct outputs with identical best scores, though only one is sufficient.

by the source image and instruction, even when varying noise or rewriting prompts (see Fig. 1(b)). This mismatch exposes three issues when transferring Image-CoT to editing: (1) Inefficient resource allocation. Existing methods [48, 90, 105, 107] use a fixed sampling budget for all edits (e.g., 32 samples). However, Fig. 2(a) shows that simple edits (high initial scores, obtained by MLLM evaluation before applying Image-CoT) see minimal improvement, while difficult edits (low initial scores) benefit more from ImageCoT. This fixed budget wastes computation on simple cases. (2) Unreliable early-stage verification. Current methods [105, 107] rely on general MLLM scores to evaluate intermediate denoising states for early pruning. However, editing often modifies subtle, localized regions of the source image, making these changes hard to distinguish in early denoising stages. This causes general scores to misjudge sample quality: 40% of samples scoring low at early stages achieve high final scores(see Fig. 2(b)). Such misjudgement leads to incorrect pruning of high-potential candidates, degrading final performance. (3) Redundant edited results. Large-scale sampling in editing often generates multiple correct results with identical best scores. For instance, most edit cases with best scores in [7,9) obtain over 15 candidates that share the same best score (see Fig. 2(c)). However, one intent-aligned result is sufficient for the editing task. Existing pruning strategies [48, 105, 107] typically adopt breadth-first search, generating all candidates in parallel before a final best-of-N selection. This leads to unnecessary computational cost on redundant correct results.

ful edited images. We preview partially denoised images at intermediate timesteps to check edited-region localization accuracy and instruction–caption consistency. Moreover, samples with high visual similarity are discarded to avoid redundancy. (3) Depth-first opportunistic stopping. To reduce redundant correct results, we incorporate a depthfirst generation process. Candidates are generated sequentially by their early-stage scores. The process terminates when sufficient intent-aligned images are found. A twostage instance-specific verifier guides this decision to confirm fine-grained correctness. Our key contributions are:

- • We identify three issues when applying Image-CoT to editing: inefficient resource allocation, unreliable earlystage verification, and redundant edited results. Based on this analysis, we propose ADE-CoT, an on-demand testtime scaling algorithm that enhances editing efficiency and correctness during large-scale sampling.
- • We address these challenges with three mechanisms: (1) difficulty-aware resource allocation that assigns a dynamic budget for each instance; (2) edit-specific verification using region localization and caption consistency for accurate early pruning; and (3) depth-first opportunistic stopping guided by an instance-specific verifier.
- • Extensive experiments on three SOTA editing models (Step1X-Edit [44], BAGEL [15], and FLUX.1 Kontext [34]) demonstrate that ADE-CoT achieves superior performance-efficiency trade-offs across three benchmarks. With comparable sampling budgets, it delivers better performance with more than 2× speedup over BoN.

To address these issues, we propose ADaptive EditCoT (ADE-CoT), an on-demand test-time scaling framework that shifts focus from scale to speed. It improves efficiency while preserving editing correctness through three core strategies: (1) Difficulty-aware resource allocation. Instead of a fixed sampling budget, ADE-CoT dynamically adjusts the budget based on estimated edit difficulty. Simple edits receive a minimal budget, whereas complex ones expand the search. This allocates computation to challenging edits. (2) Edit-specific verification in early pruning. To mitigate the misjudgement of general scores, we introduce edit-specific metrics to discover potentially success-

### 2. Related Work

Image editing with diffusion models has advanced rapidly. Early methods [2, 6, 22, 33, 39, 54, 80, 82, 109] are often training-free, relying on prompt guidance, attention modulation, or inversion editing. However, they lack precise control in fidelity and controllability. To address this issue, later work [5, 17, 27, 69, 86, 92, 96, 97, 102, 108] fine-tunes on high-quality, large-scale datasets with modified architectures. Recent approaches [16, 20, 26, 34, 44, 45, 47, 50, 53, 62, 74, 75, 88, 94] combine multimodal

large language models with diffusion decoders. This enables instruction-following edits by fusing modalities at the latent level. Inspired by Gemini [19] and GPT-4o [58], subsequent work [1, 7, 13, 15, 18, 25, 41, 61, 101, 110] has improved generation quality by jointly training understanding and generation tasks. However, their performance in complex edits still faces challenges. In this paper, we adapt Image-CoT to editing to enhance generation while improving efficiency.

Test-time scaling in image generation enhances quality by extending inference time. Inspired by Chain-of-Thought (CoT) in LLMs [14, 40, 49, 87], Image-CoT has emerged as a promising paradigm. Noise scaling [48, 90] generates multiple samples by perturbing the noise and selects the best as the final result. However, its computational cost scales linearly. Subsequent work aims to improve this trade-off. Some methods enhance sample diversity through prompt-level intervention, such as rewriting [29, 56] or reflective updates [38, 105, 111]. Others adapt search algorithms [21, 28, 48, 66, 70, 106] (e.g., MCTS) to treat the reverse diffusion chain as a search trajectory and change the noise based on verifier scores. Recent methods [43, 60, 79, 105, 107] use MLLMs as verifiers to prune low-potential trajectories early. However, most Image-CoT methods target T2I generation and are suboptimal for editing. To this end, we propose ADE-CoT, an edit-specific test-time scaling method, to address issues of inefficient resource allocation, unreliable early-stage verification, and redundant edited results during Image-CoT.

Path pruning in Image-CoT aims to remove low-potential samples by scoring intermediate states. Existing methods typically adopt breadth-first search. PRM [83] evaluates whether each denoising step meets quality requirements to prune candidates. PARM [105] introduces two verifiers to judge which step is clear and assess whether it has highquality potential. VideoTTS [43] proposes Tree-of-Frames to leverage the feedback from multi-verifiers to guide the generation. ICEdit [107] is the first work to incorporate Image-CoT into editing. It proposes the early filter strategy to generate preliminary images with a few additional denoising steps and select the optimal initial noise by a general MLLM verifier (i.e., VIE-Score [32]). However, general scores may incorrectly remove high-potential candidates. In contrast, we propose edit-specific metrics, including edited-region and caption verification, to mitigate this issue. Moreover, an early preview mechanism is also introduced to obtain preliminary images without extra denoising steps. Besides, we incorporate depth-first opportunistic stopping to terminate the search early.

### 3. Method

Our ADaptive Edit-CoT (ADE-CoT) framework is shown in Fig. 3(c) and detailed in Supp. Alg.3. We first propose a

difficulty-aware resource allocation strategy to dynamically adjust sampling budgets based on edit difficulty (Sec. 3.1). Then, an edit-specific verification is introduced to search promising candidates and discard errors in the early pruning stage (Sec. 3.2). Subsequently, we adopt a depth-first sequential generation that stops adaptively when sufficient intent-aligned results are found. (Sec. 3.3).

Preliminaries. Given a source image Isrc and an edit instruction c, the goal of image editing is to generate an edited image I that semantically aligns with the guidance provided by c. In Image-CoT, the standard Best-of-N method (see Fig. 3(a) and Supp. Alg.1) consists of two stages: (1) Generation: A set of N candidate images {I1,I2,··· ,IN} is generated by varying initial noise or rewriting prompts. (2) Selection: A verifier assigns a scalar score S to each candidate, reflecting its semantic alignment with c:

Vrf : Isrc × I × c → R. (1) The common verifier is the general score Sgen, which uses an MLLM with prompts (e.g., VIE-Score [32]) to evaluate images on instruction adherence and aesthetic quality. The final output I∗ is the image with the highest score:

I∗ = arg max

Vrf(Isrc,Ii,c). (2)

Ii∈{I1,I2,··· ,IN}

Early Pruning [43, 105, 107] is a standard method to improve efficiency (see Fig. 3(b) and Supp. Alg.2). It generates intermediate previews by sampling at timestep te instead of the full denoising steps T, where te < T. Candidates scoring below the rejection threshold Srj are pruned, avoiding the full generation of low-potential samples.

#### 3.1. Difficulty-aware Resource Allocation

To mitigate the inefficiency of fixed sampling budgets in previous methods [43, 48, 105, 107, 111], we introduce a difficulty-aware resource allocation strategy (summarized in Supp. Alg.4), which dynamically adjusts sampling budgets based on a preliminary estimation of edit difficulty.

Given the source image Isrc and instruction c, we generate a single candidate and evaluate it with the verifier Vrf. This yields an initial score S, which acts as a proxy for edit difficulty. The adaptive budget Na is formulated as:

Na = Nmin + ⌈(N − Nmin) × (1 − S/Smax)γ⌉, (3) where Nmin and N respectively denote the minimal and original budgets, Smax is the maximum score, and γ is a hyperparameter to control sensitivity. This formulation ensures that for easy edits where S → Smax, Na converges to Nmin. Conversely, for difficult edits where S → 0, Na approaches N. As a result, it allocates more computation to difficult cases and saves resources on easy ones.

#### 3.2. Edit-specific Verification in Early Pruning

To address the misjudgement by general scores at early denoising stage, we introduce three key components for early

###### (b) Early Pruning

Breadth-first Search

###### (a) Best-of-N (BoN)

Breadth-first Search

𝑇

0

0

𝑇

𝑡

Edit Instruction

| |[Figure 17]| |
|---|---|---|
| | | |

Edit Instruction

[Figure 18]

[Figure 19]

[Figure 20]

| |[Figure 21]| |
|---|---|---|
| | | |

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

ScalingPromptorNoise

ScalingPromptorNoise

… … … … …

… … … … …

Add a cherryeating action.

Add a cherryeating action.

GeneralVerifier

GeneralVerifier

GeneralVerifier

| |[Figure 27]| |
|---|---|---|
| | | |

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

(fixedbudget)

(fixedbudget)

[Figure 33]

[Figure 34]

+

+

| |[Figure 35]| |
|---|---|---|
| | | |

𝑁

| |[Figure 36]| |
|---|---|---|
| | | |

𝑁

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

| |[Figure 49]| |
|---|---|---|
| | | |

| |[Figure 50]| |
|---|---|---|
| | | |

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

| |[Figure 59]| |
|---|---|---|
| | | |

| |[Figure 60]| |
|---|---|---|
| | | |

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Source Image

Source Image

[Figure 67]

[Figure 68]

###### Depth-first Search in Late Denoising Stage

###### Breadth-first Search in Early Denoising Stage

###### (c) ADE-CoT (Ours)

###### 𝑡

0

𝑡

𝑇 𝑡

Edit Instruction

|[Figure 69]| |
|---|---|
| | |

[Figure 70]

[Figure 71]

[Figure 72]

|I:GenerateQuestion|
|---|

|II:Answer&Score|
|---|

Instance-SpecificVerifier

(Difficulty-awareBudget)

Edit-SpecificVerification

SortbyEvaluatedScore

5

ScalingPromptorNoise

- 1
- 2
- 3
- 4
- 5

- 1
- 2
- 3
- 4
- 5

Stop when obtaining ( ) results.

|𝑁| |
|---|---|
| | |
| | |

Add a cherryeating action.

:GenerateQuestion

SimilarCandidates

ByEvaluatedScore

(Region&Caption)

RetainTopResults

One-StepPreview

Inswer&Score

𝑁

Region error

|[Figure 73]| |
|---|---|
| | |

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

FilterVisually

[Figure 79]

𝑡 0

+

|[Figure 80]|
|---|

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

1

1

1

[Figure 88]

Hand error

|[Figure 89]| |
|---|---|
| | |

[Figure 90]

[Figure 91]

[Figure 92]

|[Figure 93]| |
|---|---|
| | |

|[Figure 94]|
|---|

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

3

3

3

Source Image

###### Intent-aligned Result Failed/Filtered Result

Denoising Sequential Generation

Opportunistic Stopping

[Figure 102]

[Figure 103]

| |
|---|

[Figure 104]

Latent Representation 𝑥 Final Result

[Figure 105]

Decoded Image

[Figure 106]

[Figure 107]

- Figure 3. Pipeline comparison of Image-CoT methods for editing. (a) Best-of-N employs a breadth-first search with a fixed sampling budget, producing all N candidates before verification. (b) Previous pruning strategies improve efficiency by pruning with general MLLM scores. (c) Our ADE-CoT employs three strategies for improved quality and efficiency: difficulty-aware resource allocation to dynamically adjust the sampling budget (orange, Sec. 3.1), edit-specific verification to identify promising candidates in early denoising stage (blue, Sec. 3.2), and depth-first opportunistic stopping to terminate search when intent-aligned results are obtained (green, Sec. 3.3).

pruning (see blue in Fig. 3(c) and Supp. Alg.5): (1) a onestep preview mechanism to obtain approximate previews of the final output; (2) a unified score using edit-specific verifiers to find high-potential candidates; and (3) a visual similarity filter to remove redundant results. Finally, the retained candidates are ranked to guide the subsequent generation.

image I and source image Isrc, we compute the per-pixel change map ∆ ∈ RH×W by averaging the absolute RGB differences across image channel dimension C:

C

1 C

|I(c) − Isrc(c)|. (5)

∆ =

c=1

We normalize ∆ using a pixel-wise softmax to weight pixels by relative change magnitude. The score Sreg is computed by aggregating changes within the mask M:

One-step preview mechanism. Directly evaluating the noisy latent xt

at an early timestep te is challenging, where te ≪ T. Since recent editing models [15, 34, 44] are mainly trained with flow matching [42], we estimate the approximate clean latent x0|t

e

(∆), (6)

M ⊙ softmax

Sreg =

H,W

in a single step: x0|t

from xt

H,W

e

e

where ⊙ denotes the element-wise product. A higher Sreg indicates that the changes are better concentrated within the intended region. More details are in Supp. Sec. B.2.3.

). (4) Here, σt

e − σt

= xt

ϵθ(xt

,Tt

e

e

e

e

) is the predicted noise. x0|t

is the noise scale and ϵθ(xt

,Tt

e

e

e

is the predicted clean latent, which is then decoded into the preview image I0|t

e

. In Supp. B.2.1, we show that this preview reflects the correctness of the final output, which provides the basis for edit-specific verifiers.

(2) Instruction-caption consistency. A standard metric in image editing is the semantic similarity between the edited image and a ground-truth caption. However, a key challenge is that such captions are unavailable at test time. To address this, we instruct an MLLM with prompt Pcap, conditioned on the source image Isrc and the instruction c, to generate a targeted caption ccap. This allows us to compute an image-caption consistency score using CLIP [65]:

e

Edit-specific verifiers. To complement the general score Sgen, we introduce two verifiers that assess edited-region correctness and instruction-caption consistency:

(1) Edited-region correctness. A primary failure in image editing is the mislocalization of edits. To assess this, we first prompt the MLLM with Preg to identify the object to modify or keep unchanged. This object is then fed into Grounded SAM2 [67] to generate a binary mask M ∈ {0,1}H×W for the expected edit region. We hypothesize that correct edits primarily affect pixels within this region. Given the edited

Scap = CLIPScore(I,ccap), (7)

where a higher Scap indicates better semantic alignment to the instruction. Detailed prompts are in Supp. B.2.4.

##### Filtering error by evaluated score. We combine the

above metrics into a unified evaluation score S:

S = Sgen + λregSreg + λcapScap, (8) where λreg and λcap are weighting factors. We compute the score for each preview image I0|t

and prune low-potential candidates using a rejection threshold Srj. In Sec. 4.2, we show that this strategy significantly reduces misjudgement of high-potential candidates in early pruning. As a result, our method improves efficiency while maintaining performance. Moreover, computing Sreg and Scap requires only a single MLLM query per edit case. This design minimizes the additional MLLM overhead.

e

Filtering visually similar candidates. Goal-directed image editing often yields multiple redundant edited results during large-scale sampling. Notably, this redundancy has been apparent in the early preview images (see Supp.B.2.5). To remove similar images, we extract visual embeddings from each preview using DINOv2 [59] and compute the pairwise similarity between them. If the similarity score between two preview images exceeds a threshold τsim, we discard the one with the lower evaluation score. This step ensures that visually distinct and high-potential candidates are retained for the subsequent generation stage.

Sorting by evaluated score. Finally, the remaining candidates are sorted by S in descending order. We empirically find that candidates with higher early scores tend to achieve higher final scores (see Supp. C.3). This enables the opportunistic stopping stage to terminate search earlier.

#### 3.3. Depth-first Opportunistic Stopping

To avoid unnecessary computation on redundant yet correct results, we introduce a depth-first opportunistic stopping mechanism. The remaining candidates are processed sequentially based on early-stage scores. The search stops upon finding intent-aligned results (see green in Fig. 3(c) and Supp. Alg.6). It consists of two components: (1) a latestage filter to retain the most promising candidates; (2) an instance-specific verifier to guide the stopping decision.

Retaining top results. Motivated by the stronger correlation between final image quality and preview scores at later denoising stages (see Supp. C.3), we use an additional check at a later timestep tl, where te < tl < T. Similar to the early pruning stage, we generate a preview for each candidate and compute its unified score at timestep tl. Instead of a fixed threshold, we apply an adaptive filter to retain candidates with scores comparable to the current highest score. This dynamically prunes suboptimal samples.

Instance-specific verifier. While general scores Sgen are effective for coarse-grained rank, they often assign the same top scores to many candidates (see Fig. 2(c)), even when some contain editing errors. This makes the final selection unreliable. To address this, we introduce an instancespecific verifier for fine-grained assessment. We found that a two-stage inquiry effectively guides the MLLM to notice

critical details (see Sec. 4.2). Given Isrc and c, it first generates a set of yes-no questions about the current edit via prompt Pq, covering aspects such as instruction adherence and aesthetics. Then, it answers them using prompt Pa to produce an instance-specific score Sspec by counting yes responses, where yes indicates a correct edit of one aspect. We combine Sspec with the unified score to penalize error candidates. The search process stops after Nhigh candidates are intent-aligned (i.e., receiving yes answers). Finally, the highest-scoring candidate is selected as the output.

### 4. Experiments

Evaluation settings. We evaluate on three popular benchmarks: (1) GEdit-Bench contains real-world user edits. We use GPT4.1 [57] with VIE-Score [32] to measure semantic consistency (G SC), perceptual quality (G PQ), and overall score (G O). (2) AnyEdit-Test covers a range of tasks, such as local, global, and implicit editing tasks. Following [96], we report average semantic similarity (CLIPim and CLIPout [23, 65]) and visual similarity (DINO distance [8, 59]). (3) Reason-Edit involves complex understanding and reasoning scenarios. We follow [26] to evaluate with PSNR (dB) [24], LPIPS [104], and CLIP Score.

Metrics for efficiency. Following [43, 48, 107], we measure computational cost via the Number of Function Evaluations (NFE), i.e., the total denoising steps in generation. To evaluate the balance between quality and computational cost after applying pruning strategies, we introduce the rea-

(i)

soning efficiency, η = M1 Mi=1 σi · S

. Here, σi = 1 if the final result achieves non-degraded performance compared to BoN, and σi = 0 otherwise. S(i) is the final score for instance i, and Smax is the maximum score. M is the number of test instances. A higher η means a better trade-off between performance and efficiency. To assess generation redundancy, we propose outcome efficiency

Smax · NFENT(i)

(i) min

ξ = M1 Mi=1 σiNFE

NFE(i). Here, NFE(mini) denotes the NFE to reach the first image that achieves a non-degraded result compared to BoN. A higher ξ means less redundancy.

Implementation details. Our ADE-CoT is evaluated on three SOTA, open-sourced image editing models: Step1XEdit [44], FLUX.1 Kontext [34], and BAGEL [15]. We use their default denoising steps, i.e., T = 28,28,50. In the default setting, the early step te is set to 8,8,16 and the retain step tl is set to 16,16,36. We adopt Qwen-VL-MAX [3] for MLLM queries and VIE-Score [32] as the general score Sgen. We generate five instance-specific yes-no questions per edit. To ensure robustness, all results are averages of three runs. More details are in Supp. C.1.

#### 4.1. Comparison with SOTA Methods

Compared methods. We compare ADE-CoT with SOTA Image-CoT methods. Best-of-N (BoN) [48] serves as the

- Table 1. Comparison of ADE-CoT with SOTA Image-CoT methods. η measures performance-efficiency trade-off and ξ measures generation redundancy. Except for LPIPS, higher values are better for other metrics. The best results within a model are in bold. Nondegraded performances compared to the BoN method after pruning strategies are highlighted . All results are the average of three runs.

GEdit-Bench-EN [44] (Full set) AnyEdit-Test [96] Reason-Edit [26]

Model N

G SC G PQ G O η ξ CLIPim CLIPout DINO η ξ PSNR LPIPS↓ CLIP η ξ

FLUX.1 Kontext [34] 1 6.517 7.548 6.021 - - 0.874 0.302 0.774 - - 25.135 0.066 21.361 - w/ BoN [48] 32 7.132 7.721 6.641 0.66 0.12 0.882 0.307 0.784 0.66 0.21 25.657 0.054 21.635 0.24 0.24 w/ PRM [105] 32 7.018 7.713 6.517 1.13 0.27 0.880 0.306 0.782 0.99 0.37 25.633 0.058 21.603 0.31 0.44 w/ PARM [105] 32 7.087 7.716 6.563 0.77 0.29 0.881 0.307 0.783 0.85 0.36 25.498 0.056 21.656 0.26 0.39

- w/ TTS-EF [107] 32 6.866 7.657 6.376 0.98 0.57 0.878 0.305 0.779 0.72 0.33 25.509 0.057 21.639 0.44 0.46 w/ TTS-EF (modified) 32 7.142 7.699 6.643 0.79 0.51 0.882 0.307 0.785 0.62 0.35 25.657 0.054 21.637 0.25 0.49 w/ ADE-CoT (ours) 32 7.225 7.719 6.695 1.47 0.66 0.883 0.308 0.784 1.61 0.58 25.755 0.053 21.663 0.50 0.70 Speedup (vs. BoN) - - - - ↑ 2.2× ↑ 5.5× - - - ↑ 2.4× ↑ 2.8× - - - ↑ 2.1× ↑ 2.9× BAGEL [15] 1 7.124 6.664 6.372 - - 0.881 0.305 0.778 - - 26.820 0.061 23.241 - w/ BoN [48] 32 7.725 7.016 6.908 0.69 0.14 0.891 0.310 0.794 0.67 0.21 27.668 0.050 23.393 0.26 0.22 w/ PRM [105] 32 7.496 6.854 6.685 1.17 0.33 0.888 0.308 0.785 1.12 0.36 27.231 0.051 23.408 0.37 0.41 w/ PARM [105] 32 7.566 6.899 6.765 1.21 0.22 0.890 0.309 0.789 1.08 0.40 27.483 0.050 23.324 0.44 0.56

- w/ TTS-EF [107] 32 7.402 6.834 6.660 1.15 0.43 0.886 0.307 0.788 1.12 0.30 27.387 0.054 23.230 0.30 0.27 w/ TTS-EF (modified) 32 7.749 6.986 6.910 1.04 0.43 0.891 0.310 0.793 0.88 0.40 27.673 0.049 23.409 0.39 0.53 w/ ADE-CoT (ours) 32 7.823 6.987 6.972 1.27 0.62 0.893 0.311 0.796 1.64 0.53 27.849 0.045 23.399 0.58 0.62

- Speedup (vs. BoN) - - - - ↑ 1.8× ↑ 4.4× - - - ↑ 2.4× ↑ 2.5× - - - ↑ 2.2× ↑ 2.8× Step1X-Edit [44] 1 7.002 7.085 6.403 - - 0.865 0.302 0.742 - - 21.443 0.106 22.463 - w/ BoN [48] 32 7.732 7.485 7.157 0.72 0.13 0.877 0.308 0.765 0.65 0.21 23.301 0.087 22.750 0.23 0.22 w/ PRM [105] 32 7.647 7.405 7.031 0.94 0.22 0.874 0.307 0.762 1.03 0.40 23.118 0.090 22.791 0.31 0.41 w/ PARM [105] 32 7.692 7.446 7.072 0.94 0.23 0.876 0.307 0.764 0.84 0.36 23.299 0.088 22.803 0.29 0.44 w/ TTS-EF [107] 32 7.296 7.301 6.777 0.96 0.51 0.873 0.306 0.758 0.70 0.33 22.273 0.095 22.502 0.34 0.24 w/ TTS-EF (modified) 32 7.743 7.478 7.162 0.93 0.54 0.877 0.308 0.766 0.61 0.35 23.297 0.087 22.760 0.23 0.49 w/ ADE-CoT (ours) 32 7.821 7.465 7.196 1.45 0.62 0.878 0.309 0.766 1.34 0.56 23.405 0.086 22.834 0.46 0.63

- Speedup (vs. BoN) - - - - ↑ 2.0× ↑ 4.8× - - - ↑ 2.1× ↑ 2.7× - - - ↑ 2.0× ↑ 2.9×

7.0

7.2

6.6

7.0

| |
|---|

6.8

| |
|---|

| |
|---|

6.4

###### G_O

###### G_O

###### G_O

| |
|---|

6.8

| |
|---|

| |
|---|

6.6

| |
|---|

| |
|---|

| |
|---|

6.2

BoN

TTS-EF

BoN

TTS-EF

BoN

TTS-EF

6.6

| |
|---|

PRM

ADE-CoT

PRM

ADE-CoT

PRM

ADE-CoT

| |
|---|

PARM

PARM

PARM

| |
|---|

6.4

6.4

6.0

50 100 200 400 800 1600

28 56 112 224 448 896

28 56 112 224 448 896

NFE

NFE

NFE

(a) FLUX.1 Kontext.

(b) BAGEL.

(c) Step1X-Edit.

- Figure 4. Scaling curves on GEdit-Bench across different editing models. We show overall performance (G O, y-axis) versus computational cost (NFE, x-axis) for sampling budgets of N = 1, 2, 4, 8, 16, 32. The shaded regions indicate error bars. Our ADE-CoT (purple star) consistently surpasses SOTA Image-CoT methods across all models and budgets, achieving a better performance-efficiency trade-off.

baseline. PRM [105] and PARM [105] assess intermediate denoised images using general MLLM scores to prune lowpotential samples. TTS-EF [107] generates early preview images by additional denoising steps and selects the best initial noise to continue generation. For fair comparison, we modify TTS-EF by increasing the number of retained samples to maintain performance comparable to BoN.

Results under fixed sampling budget. We first evaluate all methods under a fixed sampling budget (N = 32) across three editing models and three benchmarks. Our analysis reveals three key findings. ❶ ADE-CoT achieves comparable or superior performance to BoN while providing significant speedups. Specifically, it improves reasoning efficiency η by over 2× compared to BoN, indicating a superior performance–efficiency trade-off. Moreover, ADECoT increases outcome efficiency ξ by an average of 4.9×, 2.7×, and 2.9× on GEdit-Bench, AnyEdit, and ReasonEdit, respectively. This shows low redundancy through our proposed strategies. The scaling curves in Fig. 4 further

demonstrate this advantage across different sampling budgets (N = 2,4,8,16,32). ADE-CoT consistently achieves higher performance with lower computational cost than all baselines at every budget level. ❷ PRM and PARM show limited performance compared to BoN. This is primarily because general scores struggle to accurately judge early preview images, causing many high-potential samples to be incorrectly discarded. ❸ TTS-EF shows high efficiency scores but suffers from poor performance. This is because selecting only a single best sample from early previews becomes less reliable when scaling the number of samples.

Results under comparable performance. We next compare methods that achieve non-degraded quality relative to BoN. We observe three key findings. ❶ BoN, TTSEF (modified), and ADE-CoT achieve similar results across datasets. Notably, Tab. 1 shows that ADE-CoT achieves the highest η and ξ, demonstrating superior efficiency. ❷ Although TTS-EF (modified) maintains BoN-level performance, it shows lower efficiency than our method, even

NFE

G_O

###### Input Image Output-1 Output-2

###### + Prompt

###### ReasoningEfficiency()

| |
|---|

OutcomeEfficiency()

900

0.8

6.6

[Figure 108]

[Figure 109]

[Figure 110]

Change the person‘s movements to look forward.

800

0.2

###### G_O

NFE

6.4

0.7

700

6.2

600

0.1

###### General Score:

6.0

0.6

0 0.05 0.1 0.15 0.2 0.25

[Figure 111]

0 0.05 0.1 0.15 0.2 0.25

- Ø Output-1: Gaze meets the prompt while keeping other scenes unchanged.
- Ø Output-2: The person‘s gaze has been changed to look forward.

[Figure 112]

(a) NFE and performance versus γ.

(b) Efficiency metrics versus γ.

###### Instance-specific Verifier:

###### Figure 5. Effect of γ in difficulty-aware resource allocation.

Ø Question: (1) Are shoulders aligned to indicate a forward-facing posture?

300

(2) Are the person‘s head and body positioned to face directly forward?

Low Score Region High Score Region

[Figure 113]

###### NumberofSamples

- Ø Output-1: NO (Head turned sideways), NO (Shoulders remain sideways)
- Ø Output-2: YES, YES.

250

236

6.6

226

[Figure 114]

| |
|---|

General Score

200

6.5

| |
|---|

w/ Edit-Specific Scores

###### G_O

Figure 7. Instance-specific verification detects subtle errors. Both candidates receive similarly high general scores. In contrast, our instance-specific verifier reveals the critical flaw in the left image (“head turned sideways”), improving final selection accuracy.

150

| |
|---|

6.4

88

100

72 81 82

73

BoN

| |
|---|

Early filter w/ Sgen

6.3

50

32 35

33

30

19

Early filter w/ S

0

[0,2) [2,4) [4,6) [6,7) [7,8) [8,9)

56 112 224 448 896

Final Score of Early Low-Scoring Samples

NFE

(a) Distribution of pruned candidates.

(b) Scaling curves.

6.6

460

6.70

Figure 6. Effect of edit-specific verification on early pruning.

440

6.68

| |
|---|

###### G_O

###### NFE

###### G_O

lower than the original TTS-EF. This is due to an increase in retained samples and additional denoising steps for early previews. ❸ The scaling curves in Fig. 4 further confirm this advantage. When performance is comparable (same yaxis values), ADE-CoT requires less computation (lower xaxis values). This validates that our three strategies effectively allocate computation to challenging edits.

6.4

420

| |
|---|

6.66

| |
|---|

BoN

BoN w/ stop

NFE

| |
|---|

| |
|---|

PRM

PRM w/ stop

400

6.2

G_O 6.64

PARM

PARM w/ stop

1 2 4 6 8 16 32

56 112 224 448 896

Nhigh

NFE

(a) Scaling curves. Shaded regions are efficiency gains from opportunistic stopping.

(b) Effect of Nhigh.

Figure 8. Impact of opportunistic stopping strategy.

#### 4.2. Performance and Efficiency Analysis

N > 8, accumulated misjudgement causes its performance to drop below BoN. In contrast, our unified score S with edit-specific verifiers maintains superior results. ❸ Rows b) and c) in Tab. 2 quantify the NFE reduction when maintaining comparable performance to BoN. By mitigating misjudgement, early pruning with S enables a higher rejection threshold, achieving greater NFE reduction than Sgen.

In this section, we analyze the key reasons for the performance and efficiency improvement of our ADE-CoT.

Difficulty-aware resource allocation reduces costs on easy edits. To validate this, we study the impact of the threshold γ in Eq. (3) on both performance and efficiency: ❶ In Fig. 5(a), we show that increasing γ from 0 (equivalent to BoN Baseline) steadily reduces NFE while performance remains nearly unchanged until γ exceeds 0.15. This confirms that reducing budgets for simple edits enhances efficiency while preserving image quality. Row a) in Tab. 2 further confirms this finding. ❷ We further examine the efficiency gains in Fig. 5(b). Both efficiency metrics (η and ξ) increase as γ grows. Growth slows beyond γ = 0.15, mainly because the performance starts to decline. To balance efficiency and quality, we set the default γ to 0.15.

Opportunistic stopping reduces redundancy in late denoising. We analyze its performance and efficiency gains from three aspects: ❶ Instance-specific verification improves final selection. As shown in Fig. 7, general scores often fail to distinguish correct from flawed outputs, assigning similarly high values to both. In contrast, our verifier generates targeted questions (e.g., “Are shoulders aligned?”, “Is the head facing forward?”) that correctly identify errors. Row f) in Tab. 2 quantitatively confirms this improvement across all models. ❷ Depth-first opportunistic stopping reduces redundant computation. Our strategy exploits inherent sampling redundancy in image editing (Fig. 2(c)). When integrated with breadth-first pruning methods (BoN, PRM, PARM), it consistently reduces their NFE while maintaining comparable performance, as shown in Fig. 8(a). Row g) in Tab. 2 confirms at least a 10% NFE reduction with minimal quality drop. ❸ Optimal Nhigh balances performance and efficiency. Fig. 8(b) shows that performance (orange) saturates when Nhigh ≥ 4, while NFE (blue) increases linearly. To balance these factors, we set Nhigh = 4 as the default. Although stopping after the first intent-aligned im-

Edit-specific verifiers reduce misjudgement in early pruning. We demonstrate this with three key findings: ❶ Fig. 6(a) shows that using the general score alone causes significant misjudgement in the high score region [6,9). After utilizing edit-specific scores, misjudgement drops from 235 to 86 (a 63% reduction). Meanwhile, pruned low-score samples remain nearly unchanged (357 → 329). This confirms that assessing edited-region correctness and caption consistency effectively improves pruning accuracy. ❷ Fig. 6(b) shows performance across different budgets. Under an identical pruning threshold, the general score achieves lower performance as sampling increases. When

###### Table 2. Effect of the three proposed strategies on efficiency and performance. We evaluate our method on GEdit-Bench [44].

Kontext BAGEL Step1X-Edit

Model

G O↑ NFE↓ G O↑ NFE↓ G O↑ NFE↓ Baseline (BoN) 6.641 896 6.908 1600 7.157 896

- a) +difficulty-aware budgets 6.641 797 6.909 1391 7.157 778

- b) +early pruning (w/ Sgen) 6.642 719 6.912 1351 7.157 719

- c) +early pruning (w/ S) 6.647 673 6.916 1290 7.161 638

- d) +filtering similar samples 6.651 508 6.915 1087 7.162 522

- e) +late retaining 6.652 464 6.935 972 7.163 462

- f) +instance-specific verifier 6.702 464 6.984 972 7.206 462

- g) +opportunistic stopping (full) 6.695 418 6.972 882 7.196 434 Table 3. Ablation of obtaining early preview images.

Kontext BAGEL Step1X-Edit

Model

G O↑ NFE↓ G O↑ NFE↓ G O↑ NFE↓

- a) w/ additional steps 6.678 523 6.952 1008 7.188 525

- b) w/ noisy latents 6.648 790 6.945 1334 7.153 765 ADE-CoT (full) 6.695 418 6.972 882 7.196 434

###### Table 4. Ablation of different search ways for pruning strategy.

Kontext BAGEL Step1X-Edit

Model

G O↑ NFE↓ η ↑ G O↑ NFE↓ η ↑ G O↑ NFE↓ η ↑

- a) w/ BFS [55] 6.702 464 1.37 6.984 972 1.12 7.206 462 1.36

- b) w/ DFS [78] 6.644 574 1.32 6.966 1073 1.08 7.162 554 1.29

- c) w/o sorting 6.694 433 1.42 6.972 905 1.23 7.194 445 1.41 ADE-CoT (full) 6.695 418 1.47 6.972 882 1.27 7.196 434 1.45

age could reduce computation, collecting four high-quality candidates improves robustness across diverse edits.

#### 4.3. Ablation Study

Does filtering visually redundant images degrade performance? Row d) of Tab. 2 shows that removing similar candidates reduces NFE by 24%,16%,18% for Kontext, BAGEL, and Step1X-Edit, respectively, with almost no change in G O. This indicates that many candidates are redundant and can be discarded without harming quality.

What is the optimal way to obtain early previews? We compare three methods in Tab. 3: (1) extra denoising steps (as in TTS-EF [107]), (2) directly decoding noisy latents, and (3) our one-step preview. Additional denoising steps produce high-quality previews but significantly increase NFE. Decoding noisy latents provides clear previews only at later stages, leading to the highest NFE. In contrast, our one-step preview generates clear images in early steps with no extra iterations, achieving comparable G O with the lowest NFE across all models. Detailed analysis is in Supp. C.3.

Which search strategy is most effective for pruning? Tab. 4 compares Breadth-First Search (BFS) and DepthFirst Search (DFS) with our hybrid search: BFS in the early denoising stage and DFS in the late denoising stage. While BFS (row a) yields the highest G O, our ADE-CoT achieves a superior performance-efficiency balance with the highest

725

500

6.65

6.65

700

475

6.64

6.63

###### G_O

###### G_O

###### NFE

###### NFE

675

450

6.63

6.60

650

NFE G_O

NFE

425

G_O 6.62

625

6.58

12 14 16 20 24

4 6 8 10 12 14

tl

te

(a) early step te

(b) retain step tl

###### Figure 9. Effect of timesteps in early prune and late retain. Table 5. Effect of MLLMs on performance and efficiency.

Kontext BAGEL Step1X-Edit

Model MLLM

G O↑ NFE↓ G O↑ NFE↓ G O↑ NFE↓ BoN

###### 6.568 896 7.001 1600 7.155 896 ADE-CoT 6.637 436 7.042 897 7.193 446 BoN

Qwen2.5-VL-72B [3]

###### 6.641 896 6.908 1600 7.157 896 ADE-CoT 6.695 418 6.972 882 7.196 434 BoN

Qwen-VL-MAX [3]

###### 6.691 896 7.034 1600 7.158 896 ADE-CoT 6.719 403 7.109 806 7.240 414

Qwen3-VL-32B [93]

η. This validates that early BFS preserves high-potential candidates, while late DFS avoids unnecessary computation once aligned results are found. Row c) shows that sequential generation further reduces NFE without quality drop.

What are the optimal timesteps for early pruning and late retaining? We analyze the impact of te and tl for FLUX.1 Kontext in Fig. 9. Increasing the early step te improves quality up to te = 8, but higher values only increase cost. This is because preview images become clearer and enable more accurate pruning. For the retain step tl, performance increases until 16 and then saturates, since previews are already high-fidelity. We set te = 8 and tl = 16 as the optimal trade-off. More results are in Supp. Sec.C.3.

How do MLLM verifiers affect performance? We evaluate ADE-CoT with three MLLMs in Tab. 5, including two open-source models (Qwen2.5-VL-72B [3] and Qwen3-VL-32B [93]) and one proprietary model (QwenVL-MAX [3]). We demostrate that ADE-CoT is robust to different verifiers. ❶ It achieves over 2× speedup across all MLLMs compared to BoN. ❷ The benefits also scale with MLLM capability. Stronger MLLMs, such as Qwen3-VL, yield higher G O and larger efficiency gains. Additional discussion on the effect of MLLMs are in Supp. Sec.C.7.

### 5. Conclusion

In this work, we propose ADE-CoT, an on-demand testtime scaling algorithm to enhance quality and efficiency in image editing. Our difficulty-aware resource allocation adjusts computational budget based on edit difficulty, avoiding waste on simple edits. To improve selection accuracy, we introduce edit-specific verification for early pruning, effectively finding high-potential candidates. Finally, our depthfirst opportunistic stopping mechanism terminates generation once intent-aligned results are found, reducing redundancy without quality drop. Extensive experiments on three

SOTA editing models and three benchmarks show ADECoT achieves over a 2× speedup while maintaining performance. We hope our work provides new insights into efficient test-time scaling for goal-directed generation.

### References

- [1] Inclusion AI, Biao Gong, Cheng Zou, Dandan Zheng, Hu Yu, Jingdong Chen, Jian Sun, Junbo Zhao, Jun Zhou, Kaixiang Ji, Lixiang Ru, Libin Wang, Qingpei Guo, Rui Liu, Weilong Chai, Xinyu Xiao, and Ziyuan Huang. Ming-liteuni: Advancements in unified architecture for natural multimodal interaction. CoRR, abs/2505.02471, 2025. 3
- [2] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel Cohen-Or. Stable flow: Vital layers for training-free image editing. In CVPR, pages 7877–7888, 2025. 2
- [3] Shuai Bai, Keqin Chen, and Xuejing Liu et al. Qwen2.5-vl technical report. CoRR, abs/2502.13923, 2025. 5, 8, 25, 29
- [4] Sule Bai, Mingxing Li, Yong Liu, Jing Tang, Haoji Zhang, Lei Sun, Xiangxiang Chu, and Yansong Tang. Univgr1: Reasoning guided universal visual grounding with reinforcement learning. arXiv preprint arXiv:2505.14231,

2025. 30

- [5] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 2
- [6] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, pages 22503–22513, 2023. 2
- [7] Siyu Cao, Hangting Chen, and Peng et al. Chen. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025. 3
- [8] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, pages 9630–9640, 2021. 5, 31
- [9] Chubin Chen, Jiashu Zhu, Xiaokun Feng, Nisha Huang, Meiqi Wu, Fangyuan Mao, Jiahong Wu, Xiangxiang Chu, and Xiu Li. S2-Guidance: Stochastic Self Guidance for Training-Free Enhancement of Diffusion Models. arXiv preprint arXiv:2508.12880, 2025. 30
- [10] Jiefeng Chen, Jie Ren, Xinyun Chen, Chengrun Yang, Ruoxi Sun, and Sercan O.¨ Arik. SETS: leveraging selfverification and self-correction for improved test-time scaling. CoRR, abs/2501.19306, 2025. 30
- [11] Rui Chen, Lei Sun, Jing Tang, Geng Li, and Xiangxiang Chu. Finger: Content aware fine-grained evaluation with reasoning for ai-generated videos. In ACM MM, pages 3517–3526, 2025. 30
- [12] Zhekai Chen, Ruihang Chu, Yukang Chen, Shiwei Zhang, Yujie Wei, Yingya Zhang, and Xihui Liu. TTS-VAR: A test-time scaling framework for visual auto-regressive generation. CoRR, abs/2507.18537, 2025. 1
- [13] Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified self-supervised pretraining for image generation and understanding. arXiv preprint arXiv:2503.06132, 2025. 3

- [14] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. 3, 30
- [15] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Shi Guang, and Haoqi Fan. Emerging properties in unified multimodal pretraining. CoRR, abs/2505.14683, 2025. 1, 2, 3, 4, 5, 6, 13, 15, 25, 28, 30
- [16] Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. In ICLR,

2024. 2

- [17] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. CoRR, abs/2405.04007, 2024. 2
- [18] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. SEED-X: multimodal models with unified multi-granularity comprehension and generation. CoRR, abs/2404.14396, 2024. 3
- [19] Google Gemini2. Experiment with gemini 2.0 flash native image generation, 2025. 3
- [20] Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chen-Wei Xie, Yu Liu, and Jingren Zhou. ACE: all-round creator and editor following instructions via diffusion transformer. In ICLR, 2025. 2
- [21] Haoran He, Jiajun Liang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Ling Pan. Scaling image and video generation via test-time evolutionary search. CoRR, abs/2505.17618, 2025. 3, 31
- [22] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross-attention control. In ICLR, 2023. 2
- [23] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In EMNLP, pages 7514– 7528, 2021. 5, 31
- [24] Alain Hor´e and Djemel Ziou. Image quality metrics: PSNR vs. SSIM. In ICPR, pages 2366–2369, 2010. 5
- [25] Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, and Hang Xu. ILLUME+: illuminating unified MLLM with dual visual tokenization and diffusion refinement. CoRR, abs/2504.01934, 2025. 3
- [26] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, and Ying Shan. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In CVPR, pages 8362–8371,

2024. 2, 5, 6

- [27] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Cihang Xie, and Yuyin Zhou. Hq-edit: A high-quality dataset for instruction-based image editing. In ICLR, 2025. 2, 31

- [28] Vineet Jain, Kusha Sareen, Mohammad Pedramfar, and Siamak Ravanbakhsh. Diffusion tree sampling: Scalable inference-time alignment of diffusion models. CoRR, abs/2506.20701, 2025. 3, 31
- [29] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2I-R1: reinforcing image generation with collaborative semantic-level and token-level cot. CoRR, abs/2505.00703, 2025. 3, 30
- [30] Dongyang Jin, Ryan Xu, Jianhao Zeng, Rui Lan, Yancheng Bai, Lei Sun, and Xiangxiang Chu. Semantic context matters: Improving conditioning for autoregressive models,

2026. 30

- [31] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. 30
- [32] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation. In ACL, pages 12268– 12290, 2024. 3, 5, 18, 19, 25, 31
- [33] Vladimir Kulikov, Matan Kleiner, Inbar HubermanSpiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. CoRR, abs/2412.08629, 2024. 2
- [34] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. FLUX.1 kontext: Flow matching for in-context image generation and editing in latent space. CoRR, abs/2506.15742, 2025. 1, 2, 4, 5, 6, 13, 15, 25, 28, 30
- [35] Rui Lan, Yancheng Bai, Xu Duan, Mingxing Li, Dongyang Jin, Ryan Xu, Lei Sun, and Xiangxiang Chu. Flux-text: A simple and advanced diffusion transformer baseline for scene text editing. arXiv preprint arXiv:2505.03329, 2025.
- [36] Huaqiu Li, Yong Wang, Tongwen Huang, Hailang Huang, Haoqian Wang, and Xiangxiang Chu. Ld-rps: Zero-shot unified image restoration via latent diffusion recurrent posterior sampling. arXiv preprint arXiv:2507.00790, 2025. 30
- [37] Mingxing Li, Rui Wang, Lei Sun, Yancheng Bai, and Xiangxiang Chu. Next token is enough: Realistic image quality and aesthetic scoring with multimodal large language model. arXiv preprint arXiv:2503.06141, 2025. 30
- [38] Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Arsh Koneru, Yusuke Kato, Kazuki Kozuka, and Aditya Grover. Reflect-dit: Inference-time scaling for text-to-image diffusion transformers via in-context reflection. CoRR, abs/2503.12271, 2025. 1, 3, 14, 30
- [39] Yaowei Li, Yuxuan Bian, Xuan Ju, Zhaoyang Zhang, Ying Shan, Yuexian Zou, and Qiang Xu. Brushedit: All-inone image inpainting and editing. CoRR, abs/2412.10316,

- 2024. 2

- [40] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schul-

- man, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In ICLR, 2024. 3, 30
- [41] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, Yatian Pang, and Li Yuan. Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation. CoRR, abs/2506.03147, 2025. 3
- [42] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023. 4
- [43] Fangfu Liu, Hanyang Wang, Yimo Cai, Kaiyan Zhang, Xiaohang Zhan, and Yueqi Duan. Video-t1: Test-time scaling for video generation. CoRR, abs/2503.18942, 2025. 1, 3, 5, 14, 18, 31
- [44] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 1, 2, 4, 5, 6, 8, 13, 15, 25, 28, 30
- [45] Chunhao Lu, Qiang Lu, Meichen Dong, and Jake Luo. End-to-end multi-modal diffusion mamba. In ICCV, pages 20529–20540, 2025. 2
- [46] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. Mach. Intell. Res., 22(4):730–751, 2025. 30
- [47] Jian Ma, Qirong Peng, Xu Guo, Chen Chen, Haonan Lu, and Zhenyu Yang. X2I: seamless integration of multimodal understanding into diffusion transformer via attention distillation. CoRR, abs/2503.06134, 2025. 2
- [48] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi S. Jaakkola, Xuhui Jia, and Saining Xie. Scaling inference time compute for diffusion models. In CVPR, pages 2523–2534, 2025. 1, 2, 3, 5, 6, 14, 18, 28, 30, 31
- [49] Qianli Ma, Haotian Zhou, Tingkai Liu, Jianbo Yuan, Pengfei Liu, Yang You, and Hongxia Yang. Let’s reward step by step: Step-level reward model as the navigators for reasoning. CoRR, abs/2310.10080, 2023. 3, 30
- [50] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. ACE++: instructionbased image creation and editing via context-aware content filling. CoRR, abs/2501.02487, 2025. 2
- [51] Fangyuan Mao, Aiming Hao, Jintao Chen, Dongxia Liu, Xiaokun Feng, Jiashu Zhu, Meiqi Wu, Chubin Chen, Jiahong Wu, and Xiangxiang Chu. Omni-effects: Unified and spatially-controllable visual effects generation. arXiv preprint arXiv:2508.07981, 2025. 30
- [52] Kou Misaki, Yuichi Inoue, Yuki Imajuku, So Kuroki, Taishi Nakamura, and Takuya Akiba. Wider or deeper? scaling LLM inference-time compute with adaptive branching tree search. CoRR, abs/2503.04412, 2025. 30
- [53] Sicheng Mo, Thao Nguyen, Xun Huang, Siddharth Srinivasan Iyer, Yijun Li, Yuchen Liu, Abhishek Tandon, Eli

Shechtman, Krishna Kumar Singh, Yong Jae Lee, Bolei Zhou, and Yuheng Li. X-fusion: Introducing new modality to frozen large language models. CoRR, abs/2504.20996,

- 2025. 2

- [54] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, pages 6038– 6047, 2023. 2
- [55] Edward F Moore. The shortest path through a maze. In Proc. of the International Symposium on the Theory of Switching, pages 285–292. Harvard University Press, 1959. 8
- [56] Yoonjin Oh, Yongjin Kim, Hyomin Kim, Donghwan Chi, and Sungwoong Kim. Object-centric self-improving preference optimization for text-to-image generation. CoRR, abs/2506.02015, 2025. 3, 30
- [57] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774,

2023. 5

- [58] OpenAI. Introducing 4o image generation, 2025. 3
- [59] Maxime Oquab, Timoth´ee Darcet, and Th´eo Moutakanni et al. Dinov2: Learning robust visual features without supervision. TMLR, 2024, 2024. 5, 25
- [60] Yuta Oshima, Masahiro Suzuki, Yutaka Matsuo, and Hiroki Furuta. Inference-time text-to-video alignment with diffusion latent beam search. CoRR, abs/2501.19252, 2025. 1, 3, 31
- [61] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. CoRR, abs/2504.06256, 2025. 3
- [62] Yuandong Pu, Le Zhuo, Kaiwen Zhu, Liangbin Xie, Wenlong Zhang, Xiangyu Chen, Peng Gao, Yu Qiao, Chao Dong, and Yihao Liu. Lumina-omnilv: A unified multimodal framework for general low-level vision. CoRR, abs/2504.04903, 2025. 2
- [63] Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, and Akash Srivastava. A probabilistic inference approach to inference-time scaling of llms using particle-based monte carlo methods. CoRR, abs/2502.01618, 2025. 30
- [64] Xiangyan Qu, Gaopeng Gou, Jiamin Zhuang, Jing Yu, Kun Song, Qihao Wang, Yili Li, and Gang Xiong. Proapo: Progressively automatic prompt optimization for visual classification. In CVPR, pages 25145–25155, 2025. 30
- [65] Alec Radford, Jong Wook Kim, and Chris Hallacy et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 4, 5, 31
- [66] Vignav Ramesh and Morteza Mardani. Test-time scaling of diffusion models via noise trajectory search. CoRR, abs/2506.03164, 2025. 3
- [67] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded SAM: assembling open-world models for diverse visual tasks. CoRR, abs/2401.14159, 2024. 4
- [68] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 30

- [69] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In CVPR, pages 8871–8879, 2024. 2
- [70] Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of diffusion models. CoRR, abs/2501.06848, 2025. 3, 31
- [71] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 30
- [72] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. In ICLR. OpenReview.net, 2021.
- [73] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, pages 32211– 32252, 2023. 30
- [74] Qianqian Sun, Jixiang Luo, Dell Zhang, and Xuelong Li. Smartfreeedit: Mask-free spatial-aware image editing with complex instruction understanding. CoRR, abs/2504.12704, 2025. 2
- [75] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. CoRR, abs/2411.15098,

2024. 2

- [76] Datao Tang, Xiangyong Cao, Xingsong Hou, Zhongyuan Jiang, Junmin Liu, and Deyu Meng. Crs-diff: Controllable remote sensing image generation with diffusion model. IEEE Transactions on Geoscience and Remote Sensing,

2024. 30

- [77] Datao Tang, Xiangyong Cao, Xuan Wu, Jialin Li, Jing Yao, Xueru Bai, Dongsheng Jiang, Yin Li, and Deyu Meng. Aerogen: Enhancing remote sensing object detection with diffusion-driven data generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3614–3624, 2025. 30
- [78] Robert Tarjan. Depth-first search and linear graph algorithms. SIAM journal on computing, 1(2):146–160, 1972. 8
- [79] Rui Tian, Mingfei Gao, Mingze Xu, Jiaming Hu, Jiasen Lu, Zuxuan Wu, Yinfei Yang, and Afshin Dehghan. Unigen: Enhanced training & test-time strategies for unified multimodal understanding and generation. CoRR, abs/2505.14682, 2025. 1, 3, 31
- [80] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, pages 1921–1930,

2023. 2

- [81] Jiyuan Wang, Chunyu Lin, Lang Nie, Kang Liao, Shuwei Shao, and Yao Zhao. Digging into contrastive learning for robust depth estimation with diffusion models. In ACM MM, page 4129–4137, 2024. 30
- [82] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. CoRR, abs/2411.04746, 2024. 2

- [83] Jiyuan Wang, Chunyu Lin, Cheng Guan, Lang Nie, Jing He, Haodong Li, Kang Liao, and Yao Zhao. Jasmine: Harnessing diffusion prior for self-supervised depth estimation,

2025. 30

- [84] JiYuan Wang, Chunyu Lin, Lei Sun, Rongying Liu, Lang Nie, Mingxing Li, Kang Liao, Xiangxiang Chu, and Yao Zhao. From editor to dense geometry estimator, 2025.
- [85] Jiyuan Wang, Chunyu Lin, Lei Sun, Zhi Cao, Yuyang Yin, Lang Nie, Zhenlong Yuan, Xiangxiang Chu, Yunchao Wei, Kang Liao, and Guosheng Lin. Geometry-guided reinforcement learning for multi-view consistent 3d scene editing,

2026. 30

- [86] Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. In ICLR, 2025. 2
- [87] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022. 3, 30
- [88] Chenfei Wu, Jiahao Li, and Jingren Zhou et al. Qwenimage technical report. CoRR, abs/2508.02324, 2025. 1, 2, 13, 30
- [89] Meiqi Wu, Jiashu Zhu, Xiaokun Feng, Chubin Chen, Chen Zhu, Bingze Song, Fangyuan Mao, Jiahong Wu, Xiangxiang Chu, and Kaiqi Huang. Imagerysearch: Adaptive testtime search for video generation beyond semantic dependency constraints. arXiv preprint arXiv:2510.14847, 2025. 31
- [90] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, Han Cai, Bingchen Liu, Daquan Zhou, and Song Han. SANA 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. CoRR, abs/2501.18427, 2025. 1, 2, 3, 14, 31
- [91] Ryan Xu, Dongyang Jin, Yancheng Bai, Rui Lan, Xu Duan, Lei Sun, and Xiangxiang Chu. Scalar: Scale-wise controllable visual autoregressive learning. arXiv preprint arXiv:2507.19946, 2025. 30
- [92] Yingjing Xu, Jie Kong, Jiazhi Wang, Xiao Pan, Bo Lin, and Qiang Liu. Insightedit: Towards better instruction following for image editing. In CVPR, pages 2694–2703, 2025. 2
- [93] An Yang, Anfeng Li, and Baosong Yang et al. Qwen3 technical report. CoRR, abs/2505.09388, 2025. 8, 29
- [94] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In ICML, 2024. 2
- [95] Meng Yu and Kun Zhan. Frequency regulation for exposure bias mitigation in diffusion models. In ACM MM, pages 10370–10378, 2025. 30
- [96] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In CVPR, pages 26125–26135, 2025. 2, 5, 6

- [97] Yongsheng Yu, Ziyun Zeng, Hang Hua, Jianlong Fu, and Jiebo Luo. Promptfix: You prompt and we fix the photo. In NeurIPS, 2024. 2
- [98] Zhenlong Yuan, Xiangyan Qu, Chengxuan Qian, Rui Chen, Jing Tang, Lei Sun, Xiangxiang Chu, Dapeng Zhang, Yiwei Wang, Yujun Cai, and Shuo Li. Video-star: Reinforcing open-vocabulary action recognition with tools. arXiv preprint arXiv:2510.08480, 2025. 30
- [99] Zhenlong Yuan, Jing Tang, Jinguo Luo, Rui Chen, Chengxuan Qian, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, and Shuo Li. Autodrive-r2: Incentivizing reasoning and self-reflection capacity for vla model in autonomous driving. arXiv preprint arXiv:2509.01944, 2025.
- [100] Zhenlong Yuan, Xiangyan Qu, Jing Tang, Rui Chen, Lei Sun, Ruidong Chen, Hongwei Yu, Chengxuan Qian, Xiangxiang Chu, Shuo Li, and Yuyin Zhou. What if agents could imagine? reinforcing open-vocabulary hoi comprehension through generation, 2026. 30
- [101] Hong Zhang, Zhongjie Duan, Xingjun Wang, Yingda Chen, Yuze Zhao, and Yu Zhang. Nexus-gen: A unified model for image understanding, generation, and editing. CoRR, abs/2504.21356, 2025. 3
- [102] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. In NeurIPS, 2023. 2
- [103] Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Zhihan Guo, Yufei Wang, Irwin King, Xue Liu, and Chen Ma. What, how, where, and how well? A survey on test-time scaling in large language models. CoRR, abs/2503.24235, 2025. 1
- [104] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586– 595, 2018. 5
- [105] Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Ziyu Guo, Haoquan Zhang, Manyuan Zhang, Jiaming Liu, Peng Gao, and Hongsheng Li. Let’s verify and reinforce image generation step by step. In CVPR, pages 28662–28672,

2025. 1, 2, 3, 6, 14, 17, 18, 28, 30, 31

- [106] Xiangcheng Zhang, Haowei Lin, Haotian Ye, James Y. Zou, Jianzhu Ma, Yitao Liang, and Yilun Du. Inference-time scaling of diffusion models through classical search. CoRR, abs/2505.23614, 2025. 3, 31
- [107] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. CoRR, abs/2504.20690, 2025. 2, 3, 5, 6, 8, 14, 17, 18, 19, 28, 31
- [108] Haozhe Zhao, Xiaojian (Shawn) Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instructionbased fine-grained image editing at scale. In NeurIPS,

2024. 2

- [109] Tianrui Zhu, Shiyi Zhang, Jiawei Shao, and Yansong Tang. Kv-edit: Training-free image editing for precise background preservation. CoRR, abs/2502.17363, 2025. 2
- [110] Xianwei Zhuang, Yuxin Xie, Yufan Deng, Dongchao Yang, Liming Liang, Jinghan Ru, Yuguo Yin, and Yuexian Zou.

Vargpt-v1.1: Improve visual autoregressive large unified model via iterative instruction tuning and reinforcement learning. CoRR, abs/2504.02949, 2025. 3

[111] Le Zhuo, Liangbing Zhao, Sayak Paul, Yue Liao, Renrui Zhang, Yi Xin, Peng Gao, Mohamed Elhoseiny, and Hongsheng Li. From reflection to perfection: Scaling inferencetime optimization for text-to-image diffusion models via reflection tuning. CoRR, abs/2504.16080, 2025. 1, 3, 30

In this appendix, we provide comprehensive supplementary material to offer a more complete understanding of our ADE-CoT framework. The appendix is organized as follows:

- • Sec. A revisits and expands upon the motivation behind our work.
- • Sec. B details the technical components of ADE-CoT.
- • Sec. C presents additional experimental results and ablation studies.
- • Sec. D outlines the limitations and directions for future research.
- • Sec. E reviews extended related work in the field of testtime scaling and image editing.

Detailed contents are listed as follows:

- A. Further Discussion on Motivation 13

- A.1. Limitations of SOTA Models on Complex Edits 13
- A.2. Image Chain-of-Thought Methods . . . . . . 14
- A.3. Issues of Image-CoT Methods for Editing . . 14

- B. Details of the Proposed Method ADE-CoT 18

- B.1. Difficulty-aware Resource Allocation . . . . 18
- B.2. Edit-specific Verification in Early Pruning . . 18

- B.2.1. One-Step Preview Mechanism . . . . 19
- B.2.2. General Score by MLLM . . . . . . 19
- B.2.3. Edited-Region Correctness . . . . . 20
- B.2.4. Instruction-Caption Consistency . . . 22
- B.2.5. Filtering Visually Similar Candidates 22

- B.3. Depth-first Opportunistic Stopping . . . . . . 22

- B.3.1. Details of Retaining Top Results . . 24
- B.3.2. Details of Instance-Specific Verifier . 25

- C. Additional Experimental Results 25

- C.1. Experimental Details . . . . . . . . . . . . . 25
- C.2. Details of Evaluation setting . . . . . . . . . 26
- C.3. More Ablation Studies . . . . . . . . . . . . 27
- C.4. More Hyperparameter Analysis . . . . . . . . 28
- C.5. More Analysis of Cost Computation . . . . . 28
- C.6. More Qualitative Results . . . . . . . . . . . 29
- C.7. Critical Analysis and Discussion . . . . . . . 29

- D. Limitations and Future Work 29
- E. Extended Related Work 30

### A. Further Discussion on Motivation

#### A.1. Limitations of SOTA Models on Complex Edits

Recent image editing models [15, 34, 44, 88] based on latent-level fusion between MLLMs and diffusion decoders have demonstrated impressive capabilities on standard editing tasks. However, their performance remains challenging when faced with complex editing scenarios:

Large pose changes. As shown in Fig. 10, baseline models (i.e., default single-pass inference setting) struggle with edits requiring significant pose or action modifica-

tions. For instance, when asked to “change the man’s gesture to raising his hands”, the baseline unintentionally alters the surrounding context, including the position of the chair and the man’s location in the scene. Similarly, the instruction “make the action of the plane to taking off” results in the baseline replacing the original plane with a completely different aircraft model and color scheme. The instruction “change the bird’s action to flapping its wings and flying” often yields anatomically incorrect wing positions.

Multi-object modification. Complex edits involving multiple objects pose additional challenges for baseline models. In Fig. 10, we observe failures in instructions such as “remove the woman standing next to the lady in white”, where the model fails to remove the correct person and leaves visible artifacts. Similarly, “remove the foreground snow-covered trees” results in incomplete removal, with only some trees being eliminated while others remain.

Fine-grained regional edits. Baseline models often fail to perform precise localized modifications. As shown in Fig. 10, instructions such as “change clothes of the person in lower right corner to green” frequently result in incorrect region selection and color bleeding to adjacent areas. The edit “change the hair of the green-eyed toy figure to blonde” shows similar issues with precise attribute modification. Fine-grained color changes such as “alter the color of the front bus to lime” or “change the color of couch to yellow” commonly affect unintended areas.

Multi-turn editing. As illustrated in Fig. 11, multi-turn edits are particularly susceptible to cascading errors, where mistakes in early turns propagate and accumulate through subsequent editing steps. For instance, in the first example, the initial instruction is “Turn 1: Change the pants color to black.” However, the model fails to execute this correctly, leaving the pants unchanged. This initial mistake cascades: subsequent edits are performed on an already flawed image. Consequently, the final result fails to reflect the cumulative user intent.

These limitations demonstrate that single-pass inference is insufficient for complex editing scenarios.

#### A.2. Image Chain-of-Thought Methods

Image Chain-of-Thought (Image-CoT) [48, 90, 105] offers a promising approach to address these challenges. As a testtime scaling strategy, Image-CoT generates multiple candidates through extended inference time. By selecting the best one from diverse candidates, it improves editing quality on complex scenarios without requiring additional training.

Best-of-N (BoN) [48, 90] is the standard method for

Image-CoT. As shown in Fig. 3(a) and summarized in Alg. 1, it consists of two primary stages: ❶ Generation. This stage produces a diverse set of N candidate images. This is achieved through a loop that iterates N times (Line 2). Within each iteration, diversity is introduced by sam-

pling a unique initial noise x(Ti) (Line 3) and optionally rewriting the text prompt c into a variant c(i) (Line 4). The

core of this stage is the Sampler function (Line 5), which performs a complete denoising process from timestep T

down to 0 to generate a clean latent representation x(0i). This latent is then decoded into the final image I(i) (Line 6). ❷ Selection. For each image I(i), a general MLLM verifier Vrfg computes a score S(i) that reflects its quality and adherence (Line 7). The (I(i),S(i)) pair is added to the set U (Line 8). After all N candidates are generated, the candidate with the highest score is chosen as the final output I∗ (Line 10). Computational cost. While BoN improves generation quality, its computational cost scales linearly with the number of samples N. Since every candidate must complete the full T denoising steps before selection, the total cost is N ×T function evaluations (NFE). This inefficiency makes BoN impractical for large-scale sampling.

Early pruning strategies in Image-CoT. To address the inefficiency of BoN, early pruning [43, 105, 107] is a standard approach. The core idea is to identify and discard lowpotential candidates at an early stage, avoiding the cost of their full generation. A unified framework for two common strategies is shown in Alg. 2, controlled by the mode. ❶ Early preview via additional denoising steps. This variant, proposed by TTS-EF [107], generates preview images by performing additional denoising from te to 0 (Line 7). The Sampler function produces a complete denoised latent x(ti)

that is decoded into a clear preview image It(i)

e

e

(Line 12). This provides high-quality previews for reliable verification. However, it introduces extra computational cost of te denoising steps per candidate. After pruning, passing candidates resume denoising denoising from T to 0 to generate the final output (Line 17). ❷ Early pruning on intermediate states. This variant, used by PRM [105] and PARM [105], samples from T to te to obtain an intermediate latent x(ti)

(Line 10). The latent is directly decoded into a preview image. This approach requires no extra steps for preview generation. After pruning, passing candidates resume denoising from te to 0 to complete generation (Line 19). However, the preview may be noisy due to incomplete denoising, which may affect verification accuracy.

e

#### A.3. Issues of Image-CoT Methods for Editing

Most Image-CoT methods [38, 48, 105, 107] are developed for text-to-image generation. However, directly applying them to editing is suboptimal. As discussed in the Introduction, this mismatch causes three issues:

##### Inefficient resource allocation. Image-CoT methods

###### Input Image Baseline w/ ADE-CoT Input Image Baseline w/ ADE-CoT

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Prompt: Change the man's gesture to raising his hands. Prompt: Make the action of the plane to taking off.

###### FLUX.1Kontext

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Prompt: Remove the slice of pizza that has been cut out. Prompt: Remove the bananas in the background.

|[Figure 131]|
|---|

[Figure 132]

| | |
|---|---|
| | |
| | |

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Prompt: Change clothes of the person in lower right corner to green. Prompt: Transform the tallest building into a TV tower.

###### Input Image Baseline w/ ADE-CoT Input Image Baseline w/ ADE-CoT

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

Prompt: Remove the woman standing next to the lady in white. Prompt: Change the hair of the green-eyed toy figure to blonde.

|[Figure 147]|
|---|

|[Figure 148]|
|---|

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

###### BAGEL

Prompt: Change the bird's action to flapping its wings and flying. Prompt: Change the weather to snowy.

|[Figure 153]|
|---|

|[Figure 154]|
|---|

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Prompt: Adjust the background to the ocean. Prompt: Alter the color of the front bus to lime.

###### Input Image Baseline w/ ADE-CoT Input Image Baseline w/ ADE-CoT

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

|[Figure 167]|
|---|

|[Figure 168]|
|---|

Prompt: Remove the peanuts. Prompt: Remove the foreground snow-covered trees.

###### Step1X-Edit

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

|[Figure 173]|
|---|

|[Figure 174]|
|---|

Prompt: Replace the text 'WOOD' with 'LAND'. Prompt: Change the color of couch to yellow.

|[Figure 175]|
|---|

|[Figure 176]|
|---|

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Prompt: Adjust the background to a glass wall. Prompt: Replace the laptop in front of the girl with a book.

- Figure 10. Qualitative comparison on complex edits. We compare three SOTA editing models (FLUX.1 Kontext [34], BAGEL [15], and Step1X-Edit [44]) on challenging edits (large pose changes, multi-object modifications, and fine-grained regional edits). Baseline models often fail, while our ADE-CoT produces correct results via adaptive test-time scaling. Zoom in for detailed view.

###### Input Image Baseline w/ ADE-CoT

###### Baseline w/ ADE-CoT Baseline w/ ADE-CoT

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Turn2: Add a leather clothing on the bed.

Turn3: Add another leather bag.

Turn1: Modify the pillow's material to leather.

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

|[Figure 197]|
|---|

|[Figure 198]|
|---|

Turn2: Remove the hat.

Turn3: Change the color of the clothes to black.

Turn1: Change the pants color to black.

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Turn3: Change the color of the flag on the city wall to milky white.

Turn1: Place a milky white sculpture in the open area.

Turn2: Change the color of the flower bed to milky white.

|[Figure 206]|
|---|

|[Figure 207]|
|---|

[Figure 208]

[Figure 209]

|[Figure 210]|
|---|

[Figure 211]

|[Figure 212]|
|---|

Turn3: Change the plate's color to green.

Turn2: Replace the tomato with some green fruit.

Turn1: Add some green elements in the empty area of the plate.

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Turn1: Change text 'Fast' to 'Slow'. Turn2: Replace the color of the

Turn3: Add a toolbox on the ground with black.

car in front with red.

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Turn2: Change the car body color to yellow.

Turn1: Change the license plate color to yellow.

- Figure 11. Qualitative comparison on multi-turn edits. The baseline model often fails to preserve the context from previous edits, leading to accumulated errors in subsequent turns. Our ADE-CoT maintains consistency across multiple sequential instructions, producing correct final images that reflect all requested changes. Zoom in for detailed view.

- Algorithm 1 Best-of-N (BoN) algorithm for image editing. Require: source image Isrc, text prompt c, number of samples N, and total steps T

- 1: U ← {} ▷ Initialize empty set for (I,S) pairs
- 2: for i = 1 to N do
- 3: x(Ti) ∼ N(0,I) ▷ Sample initial noise
- 4: c(i) ← Rewrite(c) ▷ (optional) Rewrite prompt
- 5: x(0i) ← Sampler(Isrc,x(Ti),c(i),T,0) ▷ Sample from T to 0, i.e., full denoising process
- 6: I(i) ← VAE Decoder(x(0i))

- 7: S(i) ← Vrfg(Isrc,I(i),c) ▷ Compute general MLLM score
- 8: U ← U ∪ {(I(i),S(i))} ▷ Update set
- 9: end for
- 10: I∗ ← arg max(I,S)∈U S
- 11: return I∗

- Algorithm 2 Early Pruning algorithm for image editing.

Require: source image Isrc, text prompt c, number of samples N and steps T, early step te, reject threshold Srj, mode ∈

{‘additional steps’, ‘intermediate state’}

- 1: U ← {} ▷ Initialize empty set for (I,S) pairs
- 2: for i = 1 to N do
- 3: x(Ti) ∼ N(0,I) ▷ Sample initial noise
- 4: c(i) ← Rewrite(c) ▷ (optional) Rewrite prompt
- 5: if mode == ‘additional steps’ then

- 6: ▷ TTS-EF [107] method
- 7: x(ti)

e

← Sampler(Isrc,x(Ti),c(i),te,0) ▷ Sample from te to 0, i.e., early preview by additional denoising steps

- 8: else if mode == ‘intermediate state’ then

- 9: ▷ PRM [105] and PARM [105] methods
- 10: x(ti)

e

← Sampler(Isrc,x(Ti),c(i),T,te) ▷ Sample from T to te, i.e., partially denoising process

- 11: end if
- 12: It(i)

e

← VAE Decoder(x(ti)

e

)

- 13: St(i)

e

← Vrfg(Isrc,It(i)

e

,c) ▷ Compute general MLLM score

- 14: ▷ Prune sample below Srj
- 15: if St(i)

e

>= Srj then

- 16: if mode == ‘additional steps’ then

- 17: x(0i) ← Sampler(Isrc,x(Ti),c(i),T,0) ▷ Sample from T to 0, i.e., full denoising process
- 18: else if mode == ‘intermediate state’ then

- 19: x(0i) ← Sampler(Isrc,x(ti)

e

,c(i),te,0) ▷ Sample from te to 0, i.e., resume denoising process

- 20: end if
- 21: I(i) ← VAE Decoder(x(0i))

- 22: S(i) ← Vrfg(Isrc,I(i),c) ▷ Compute general MLLM score
- 23: U ← U ∪ {(I(i),S(i))} ▷ Update set
- 24: end if
- 25: end for
- 26: I∗ ← arg max(I,S)∈U S
- 27: return I∗

typically use a fixed sampling budget for all edits, which may be inefficient. To validate this, we measure the performance gain relative to edit difficulty. First, we defined edit difficulty by generating a single image for each editing instance and using its MLLM score as an initial score.

We then grouped the edit tasks into bins based on this initial score. For all tasks, we applied a standard Best-of-32 (BoN) sampling strategy. We calculated the average score gain between the initial score and the final score after adding Image-CoT for each bin. As shown in Fig. 12, edits with

high initial scores (simple edits) showed minimal improvement from large-scale sampling. In contrast, edits with low initial scores (complex edits) benefited significantly. This confirms that a fixed budget wastes computational resources on simple edits that do not require extensive sampling.

Unreliable early-stage verification. Current ImageCoT methods [43, 105, 107] use general MLLM scores to prune candidates at early denoising stages. We examine whether these scores correctly identify high-potential candidates. In our experiment, we generate N = 32 candidates per edit case and evaluate each at an early timestep te = 8 using VIE-Score [32]. Candidates scoring below a rejection threshold Srj are pruned. We then complete the full denoising process for all candidates to obtain final scores. As shown in Fig. 13, this misjudgement occurs consistently across all tested models and datasets. On average, 40% of the pruned samples ultimately achieve high final scores (≥ 6). This indicates that general scores incorrectly discard many high-potential candidates during early pruning. This misjudgement leads to degraded final performance.

Redundant edited results. The goal-directed nature of image editing suggests that large-scale sampling may produce many similar, correct results. To quantify this redundancy, we apply a Best-of-32 (BoN) strategy to each editing instance. For each case, we identify the best score achieved among the 32 candidates and then count how many candidates share the same best score. As shown in Fig. 14, we observe this redundancy across three models and three datasets. For edit cases with high final scores (e.g., in the range [7,9)), a large number of candidates, often more than 8, achieve the identical best score. Since only one intent-aligned result is sufficient for editing, this redundancy reflects unnecessary computation. Existing breadthfirst search strategies [48, 105, 107] generate all candidates before selection. This leads to wasted denoising steps on redundant correct outputs.

To address these issues, we propose ADE-CoT, an adaptive test-time scaling framework for image editing. Our method improves editing performance while maintaining computational efficiency. Specifically, we introduce three key strategies: (1) difficulty-aware resource allocation to dynamically adjust sampling budgets based on edit difficulty, (2) edit-specific verification to accurately identify high-potential candidates during early pruning, and (3) depth-first opportunistic stopping to reduce redundant computation on correct results. As shown in Fig. 10 and Fig. 11, our method significantly enhances baseline performance on complex editing scenarios discussed in Sec. A.1.

### B. Details of the Proposed Method ADE-CoT

In this section, we provide implementation details of three components in ADE-CoT: difficulty-aware resource allocation (Sec. B.1), edit-specific verification in early pruning (Sec. B.2), and depth-first opportunistic stopping (Sec. B.3).

#### B.1. Difficulty-aware Resource Allocation

- As described in Sec. 3.1 of the main paper, our difficultyaware resource allocation strategy dynamically adjusts the sampling budget to improve computational efficiency. The process is summarized in Alg. 4. It first generates a single candidate to estimate the edit difficulty, which then deter-

mines the final sampling budget, Na.

❶ The process begins by generating one preliminaryestimation image (Lines 1-3). First, we sample an initial noise vector xT from a standard normal distribution (Line 1). A diffusion Sampler then generates a clean latent x0 from this noise, conditioned on the source image Isrc and instruction c (Line 2). A VAE decoder subsequently converts the latent x0 into a pixel-space image I (Line 3).

❷ Next, we estimate the edit difficulty using this initial image (Line 4). A general MLLM verifier, Vrfg, evaluates the image I to produce an initial score S. This score acts as a proxy for difficulty, where a high score suggests an easy edit and a low score indicates a difficult one.

❸ The adaptive budget Na is then calculated based on this score (Line 5), following Eq. 3 from the main text. For an easy edit where the score S is high, the budget Na is reduced towards the minimum budget Nmin. Conversely, for a difficult edit where S is low, the budget Na increases towards the original budget N. The hyperparameter γ controls the sensitivity of this adjustment, and the ceiling function ⌈·⌉ ensures the budget is an integer. Finally, the algorithm returns the calculated budget Na (Line 6). This strategy effectively allocates more computational resources to difficult cases and saves them on easy ones.

B.2. Edit-specific Verification in Early Pruning

- As described in Sec. 3.2 of the main paper, our edit-specific verification strategy addresses the misjudgement issue of general MLLM scores in early denoising stages. The detailed process is summarized in Alg. 5.

The algorithm operates as follows. ❶ We first initialize empty sets for intermediate latents Xte

, prompts C, and scores Ste

(Line 1). ❷ For each of the Na samples (Lines

2-13), we sample random noise x(Ti) ∼ N(0,I) (Line 3) and optionally rewrite the prompt c(i) (Line 4). We then

perform denoising from timestep T to early timestep te (Line 5) and apply the one-step preview mechanism to ob-

tain x(0i|)t

(Line 6), which is decoded into preview image I0(i|t)

e

(Line 7). The unified score S0(i|t)

is computed by combining general MLLM score, edited-region correctness, and

e

e

[Figure 225]

[Figure 226]

[Figure 227]

Minimal Gain

Minimal Gain

Minimal Gain

| |
|---|

| |
|---|

| |
|---|

(a) FLUX.1 Kontext in GEdit-Bench.

(b) BAGEL in GEdit-Bench. (c) Step1X-Edit in GEdit-Bench.

[Figure 228]

[Figure 229]

[Figure 230]

Minimal Gain

Minimal Gain

Minimal Gain

| |
|---|

| |
|---|

| |
|---|

(d) FLUX.1 Kontext in AnyEdit. (e) BAGEL in AnyEdit.

(f) Step1X-Edit in AnyEdit.

[Figure 231]

[Figure 232]

[Figure 233]

Minimal Gain

Minimal Gain

Minimal Gain

| |
|---|

| |
|---|

| |
|---|

(g) FLUX.1 Kontext in Reason-Edit.

(h) BAGEL in Reason-Edit.

(i) Step1X-Edit in Reason-Edit.

- Figure 12. Inefficient resource allocation with fixed sampling budgets. This figure extends the analysis from Fig. 2(a) to three SOTA models (FLUX.1 Kontext, BAGEL, and Step1X-Edit) and three benchmarks (GEdit-Bench, AnyEdit, and Reason-Edit). Edits with high initial scores (red boxes, [8,9) and [9,10)) show minimal improvement across all three models and three benchmarks. Edits with low initial scores (indicating complex tasks) benefit significantly from Image-CoT, achieving substantial performance gains. This demonstrates that fixed sampling budgets waste computation on simple edits, motivating our difficulty-aware resource allocation strategy.

- Algorithm 3 Our ADaptive Edit-CoT (ADE-CoT) algorithm for image editing. Require: source image Isrc, text prompt c, number of samples N and steps T, early step te and retain step tl

- 1: ▷ Adapt N by edit difficulty
- 2: Na ← Adapt Num(Isrc,c,N,T)

- 3: ▷ Sample from T to te and prune samples
- 4: Xte

,C ← Early Prune(Isrc,c,Na,T,te)

- 5: ▷ Sample from te to tl and retain top results; Sample from tl to 0 and select intent-aligned results
- 6: U ← Adaptive Stop(Isrc,c,Xte

,C,te,tl)

- 7: I∗ ← arg max(I,S)∈U S
- 8: return I∗

instruction-caption consistency (Line 8). Candidates with scores below the rejection threshold Srj are pruned (Lines 10-12), while others are retained. ❸ After processing all samples, we remove visually similar candidates using DINOv2 features and the threshold τsim (Lines 14-15). ❹ Finally, the remaining candidates are sorted by their unified scores in descending order (Lines 16-17), which guides the subsequent depth-first generation stage.

##### B.2.1. One-Step Preview Mechanism

Lines 6-7 of Alg. 5 implement the one-step preview mechanism, which obtains approximate previews without additional denoising steps. To validate that these early previews reliably reflect the final output, we visualize the pro-

cess in Fig. 15. The figure compares the noisy latent It with our corresponding one-step preview I0|t across various timesteps for three models. As shown, our preview generates a clear and high-fidelity approximation of the final result, even at very early stages where the noisy latent is uninterpretable (e.g., t = 8). This observation confirms that our one-step preview provides a sufficiently clear signal for early-stage evaluation, enabling accurate pruning.

##### B.2.2. General Score by MLLM

Following prior work [107], we use VIE-Score [32] as our general MLLM verifier. VIE-Score evaluates an edited image based on two criteria: Semantic Consistency (SC), which measures instruction adherence and preservation of

[Figure 234]

[Figure 235]

[Figure 236]

63% 37% 60% 40% 58% 42%

(a) FLUX.1 Kontext in GEdit-Bench. (b) BAGEL in GEdit-Bench. (c) Step1X-Edit in GEdit-Bench.

[Figure 237]

[Figure 238]

[Figure 239]

67% 33%

66% 34%

68% 32%

(d) FLUX.1 Kontext in AnyEdit. (e) BAGEL in AnyEdit. (f) Step1X-Edit in AnyEdit.

[Figure 240]

[Figure 241]

[Figure 242]

58% 42% 61% 39%

61% 39%

(g) FLUX.1 Kontext in Reason-Edit. (h) BAGEL in Reason-Edit. (i) Step1X-Edit in Reason-Edit.

- Figure 13. Misjudgement by general MLLM scores in early pruning. This figure extends the analysis from Fig. 2(b) to three SOTA editing models (FLUX.1 Kontext, BAGEL, and Step1X-Edit) evaluated on three benchmarks (GEdit-Bench, AnyEdit-Test, and ReasonEdit). Samples with low early scores are categorized by their final scores (x-axis): low score region (red, [0,6)) and high score region (green, [6,9)). On average, 37% of early low-scoring samples eventually achieve high final scores, yet would be incorrectly discarded by general MLLM scores. This demonstrates unreliable early-stage verification in editing, motivating our edit-specific verification strategy.

- Algorithm 4 AdaptNum algorithm - Lines 1-2 of Alg. 3.

Require: source image Isrc, text prompt c, number of samples N and steps T Require: number of minimum samples Nmin, maximum possible score Smax, sensitivity factor γ

- 1: xT ∼ N(0,I)
- 2: x0 ← Sampler(Isrc,xT,c,T,0) ▷ Sample from T to 0
- 3: I ← VAE Decoder(x0)

- 4: S ← Vrfg(Isrc,I,c) ▷ Compute general MLLM score
- 5: Na ← Nmin + ⌈(N − Nmin) × (1 − S/Smax)γ⌉ ▷ Adapt N based on S (cf. Sec 3.1 Eq. 3)
- 6: return Na

unedited regions, and Perceptual Quality (PQ), which assesses visual realism and aesthetics. The final overall score is calculated as the geometric mean of these two components: Sgen = SSC × SPQ. While this general score provides a coarse-grained assessment, it struggles to detect subtle errors such as mislocalized edits or semantic misalignment in early denoising stages. To address this, we complement it with edit-specific metrics.

##### B.2.3. Edited-Region Correctness

- In Fig. 16, we present the prompt Preg used to identify the edited or kept objects. When the MLLM successfully identifies the edited object, the mask M corresponds to that ob-

ject’s region. When the MLLM identifies the kept object, the mask M is set to the inverted region (i.e., the complement of the kept object). If the MLLM cannot accurately determine either the edited or kept objects, we skip this verifier. Since directly computing RGB differences across the entire image is computationally expensive, we employ a sliding window approach to aggregate the change map ∆. However, the obtained mask may not perfectly align with the true editing region. To address this, we apply an adaptive mask refinement strategy. If all early preview candidates yield Sreg = 0, we iteratively expand the mask M by padding additional pixels around its boundary. The expansion process continues until at least one candidate achieves

[Figure 243]

[Figure 244]

[Figure 245]

(a) FLUX.1 Kontext in GEdit-Bench. (b) BAGEL in GEdit-Bench. (c) Step1X-Edit in GEdit-Bench.

[Figure 246]

[Figure 247]

[Figure 248]

(d) FLUX.1 Kontext in AnyEdit. (e) BAGEL in AnyEdit. (f) Step1X-Edit in AnyEdit.

[Figure 249]

[Figure 250]

[Figure 251]

(g) FLUX.1 Kontext in Reason-Edit. (h) BAGEL in Reason-Edit. (i) Step1X-Edit in Reason-Edit.

- Figure 14. Redundant edited results in large-scale sampling. This figure extends the analysis from Fig. 2(c) to three SOTA editing models (FLUX.1 Kontext, BAGEL, and Step1X-Edit) evaluated on three benchmarks (GEdit-Bench, AnyEdit-Test, and Reason-Edit). For most edit cases, a large number of candidates (green bars, left y-axis) share identical best scores. The number of edit cases exhibiting such redundancy (orange bars, right y-axis) increases significantly in high score regions, particularly in [7,8) and [8,9) (x-axis). This demonstrates that Image-CoT produces redundant correct outputs in goal-directed editing, motivating our opportunistic stopping strategy.

- Algorithm 5 Early Prune algorithm - Lines 3-4 of Alg. 3.

Require: source image Isrc, text prompt c, number of samples Na and steps T, early step te Require: reject threshold Srj, similarity threshold τsim

- 1: Xte ← {},C ← {},Ste ← {} ▷ Initialize empty set
- 2: for i = 1 to Na do
- 3: x(Ti) ∼ N(0,I)
- 4: c(i) ← Rewrite(c) ▷ (optional) Rewrite prompt
- 5: x(ti)

e

← Sampler(Isrc,x(Ti),c(i),T,te) ▷ Sample from T to te

- 6: x(0i|)t

e

← One Step Preview(x(ti)

e

,te) ▷ Preview image from intermediate latent (cf. Sec 3.2 Eq. 4)

- 7: I0(i|t)

e

← VAE Decoder(x(0i|)t

e

)

- 8: S0(i|t)

e

← Vrf(Isrc,I0(i|t)

e

,c) ▷ Compute unified score (cf. Sec 3.2, Eq. 8)

- 9: ▷ Filter error by evaluated score
- 10: if S0(i|t)

e

>= Srj then

- 11: Xte ← Xte ∪ {x(ti)

e

},C ← C ∪ {c(i)},Ste ← Ste ∪ {S0(i|t)

e

} ▷ Update set

- 12: end if
- 13: end for
- 14: ▷ Filter visually similar candidates
- 15: Xte

,C,Ste ← Remove Similar(Xte

,C,Ste

,τsim)

- 16: ▷ Sort by evaluated score
- 17: Xte

,C ← Sort by Score(Xte

,C,key = Ste

) ▷ Sort Xte

,C by Ste

in descending order

- 18: return Xte

,C

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

𝑡 = 4 𝑡 = 8 𝑡 = 12 𝑡 = 16 𝑡 = 20 𝑡 = 24 𝑡 = 28

FLUX.1KontextBAGELStep1X-Edit

𝐼

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

𝐼 | 

[Figure 266]

[Figure 267]

[Figure 268]

𝑡 = 8 𝑡 = 16 𝑡 = 24 𝑡 = 32 𝑡 = 36 𝑡 = 40 𝑡 = 50

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

𝐼

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

𝐼 | 

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

𝑡 = 4 𝑡 = 8 𝑡 = 12 𝑡 = 16 𝑡 = 20 𝑡 = 24 𝑡 = 28

𝐼

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

𝐼 | 

Denoising

- Figure 15. Effectiveness of the one-step preview mechanism. We compare the noisy latent It (top row for each model) with our corresponding one-step preview image I0|t (bottom row) at various denoising timesteps. Across all three models, our one-step preview generates a clear and high-fidelity approximation of the final output, even at very early stages (e.g., t = 8). This demonstrates that the preview accurately reflects the final image’s content and quality, providing a solid basis for our edit-specific verifiers.

both these checks is it deemed reliable.

Sreg > 0. This ensures that the mask adequately covers the actual edited region. This refinement step improves the robustness of region correctness evaluation.

##### B.2.5. Filtering Visually Similar Candidates

To eliminate redundancy, we filter out visually similar candidates. Goal-directed image editing often yields multiple redundant results during large-scale sampling. Notably, this redundancy is already apparent in the early preview images, as illustrated in Fig. 18. To address this, we extract visual embeddings from each preview using DINOv2 and compute pairwise similarity. If the similarity between two candidates exceeds a threshold τsim, the one with the lower unified score is discarded. This step ensures that only visually distinct and high-potential candidates are retained.

##### B.2.4. Instruction-Caption Consistency

- In Fig. 17, we present the prompt Pcap, which instructs an MLLM to generate a target caption for the ideally edited image. However, for subtle edits like local object modifications, this caption-based score may not be sufficiently discriminative. To ensure the reliability of the generated caption, we introduce a two-stage filtering process. First, we confirm that the MLLM correctly understands the source image. We check if its generated original caption has a CLIP score above a threshold (default 0.27) with the source image. Second, we verify that the edited caption actually reflects a change. We ensure its textual similarity to the original caption is below a threshold (default 0.9). Only if a caption passes

#### B.3. Depth-first Opportunistic Stopping

As described in Sec. 3.3 of the main paper, our depth-first opportunistic stopping strategy avoids unnecessary computation on redundant yet correct results. It is summarized in

### Prompt Preg for identifying edited and keep objects

System Role: You are an assistant in an image-editing pipeline. Your sole task is to determine, from a textual edit instruction and the original image, (1) which object(s) in the image must be edited and (2) which object(s) are explicitly required to remain unchanged (“keep”).

INPUT The user will always provide both keys:

- • original image – the image to be edited.
- • edit instruction – the user’s textual instruction that specifies how the image should be edited.

OUTPUT (must follow exactly) Return one of the two following JSON structures:

- Case A. The edit instruction clearly identifies at least one concrete visual object to be edited:

{"edit_object": ["<the object(s) in the image that must be edited>"], "keep_object": ["<the object(s) that must NOT be edited>"]}

- • Both keys must be present.
- • If no “keep” object is mentioned, output an empty array for keep object.

- • If multiple distinct objects are explicitly requested, list each object as a separate string in the array.
- • Regardless of whether the edit instruction is in English or Chinese, ALWAYS return object names in English.

- Case B. No concrete visual object is to be edited (ambiguous, global change, object absent, or the instruction is about modifying, adding, or deleting TEXT in the image):

{"edit_object": null, "keep_object": null}

##### DECISION RULES

- • “Clearly identified” means the instruction contains explicit nouns or noun phrases that unambiguously map to elements in the image description (e.g., “cat,” “red cup,” “man’s face,” “background sky”).
- • If the instruction applies to the entire image without pointing to a specific object (e.g., “add a vintage filter,” “increase brightness”), return Case B.
- • If the instruction references an object absent from the image description, return Case B.
- • If the instruction’s main purpose is to change, remove, or add TEXT in the image, return Case B (both values null).
- • Do NOT perform the edit and do NOT add explanations—output the JSON only.

User Input: edit instruction: ⟨edit prompt⟩ original image: ⟨image⟩

Figure 16. Prompt Preg for identifying edited and keep objects.

Alg. 6, which consists of two key components: (1) a latestage filter to retain the most promising candidates; and (2) an instance-specific verifier to guide the stopping decision.

The algorithm operates as follows. ❶ We first initialize an empty set U to store image-score pairs (I,S), a retain threshold Srt, and a high-score count Ncnt (Lines 13). ❷ For each candidate (xt

,cr) from the sorted sets (Xte

e

,C) (Line 5), we perform depth-first sequential generation. We first sample from early timestep te to late timestep tl (Line 6) and apply the one-step preview mechanism to obtain x0|t

(Line 7), which is decoded into preview image

l

is computed using the same verifiers as in early pruning (Line 9). ❸ We apply an adaptive late-stage filter. If S0|t

(Line 8). The unified score S0|t

I0|t

l

l

l ≥ Srt − δ, we update the retain threshold Srt (Line 12) and continue sampling from tl to 0 to generate the final image I (Lines 13-14). Otherwise, we skip this candidate and proceed to the next one. ❹ For each retained candidate, we compute the unified score S (Line 15) and apply the instance-specific verifier to obtain Sspec (Line 17). We update the evaluated score as S ← S + Sspec (Line 18). If Sspec ≥ Shigh (indicating all yes-no questions are answered “yes”), we increment

### Prompt Pcap for generating output caption

System Role: You are “Dual-Caption,” an expert vision-language assistant. Your job is to describe (a) the user-supplied original image, and (b) the image that would exist after perfectly applying the user’s editing instruction.

INPUT The user will always provide both keys:

- • edit instruction:
- • original image:

OUTPUT (must follow exactly) Return a single JSON object with two keys, in this order:

{"original_caption": "<1-2 sentences describing the original image>", "edited_caption": "<1-2 sentences describing the image after the edit>"}

##### STYLE & CONTENT RULES

- • Base captions solely on visible content; no unfounded guesses.
- • Mention dominant objects, attributes, actions, and setting.
- • In edited caption, describe the resulting scene only—do NOT mention the edit process or instruction.

- – (Bad: “The image is edited to....” Good: “A red balloon now floats above the dog....”)
- – (Bad: “An apple, no pear....” Good: “An apple....”)

- • Each caption ≤ 40 English words.
- • Do not quote the user’s instruction verbatim; convey its visual effect instead.
- • If the instruction is impossible or unsafe, produce the best policy-compliant visual outcome.

##### FINAL CHECKLIST

✓ Return exactly one JSON object—no extra text, blank lines, or comments.

✓ Keys must be original caption and edited caption, in that order.

✓ Follow length, tense, tone, and truthfulness requirements.

User Input: edit instruction: ⟨edit prompt⟩ original image: ⟨image⟩

Figure 17. Prompt Pcap for generating output caption.

|[Figure 294]<br><br>[Figure 295]|
|---|

|[Figure 296]<br><br>[Figure 297]|
|---|

|[Figure 298]<br><br>[Figure 299]|
|---|

Early Previews

|[Figure 300]<br><br>[Figure 301]|
|---|

|[Figure 302]<br><br>[Figure 303]|
|---|

|[Figure 304]<br><br>[Figure 305]|
|---|

###### Final Images

Figure 18. Redundancy appears in early preview images. The figure displays clusters of visually similar images, grouped by colored boxes. We observe that candidates which are highly similar in the final output (bottom row) are also highly similar in their early previews (top row). This confirms that visual redundancy can be detected and filtered at an early stage.

the high-score count Ncnt (Lines 19-21). The image-score pair (I,S) is added to U (Line 22). ❺ The search terminates when Ncnt = Nhigh (Lines 24-27), meaning sufficient intent-aligned results have been found. Finally, we re-

turn the set U (Line 29), and the candidate with the highest score is selected as the output.

##### B.3.1. Details of Retaining Top Results

Lines 10-23 of Alg. 6 implement the late-stage filter to retain the most promising candidates. Unlike the early pruning stage that uses a fixed rejection threshold Srj, we adopt an adaptive filtering strategy at the late timestep tl. This is motivated by the observation that preview scores at later denoising stages exhibit stronger correlation with final image quality. For each candidate, we generate a preview image I0|t

(Lines 6-9). We maintain a retain threshold Srt that dynamically updates to the maximum score observed so far (Line 12). A candidate is retained if its score S0|t

and compute its unified score S0|t

l

l

is within a tolerance δ of the current threshold: S0|t

l

l ≥ Srt − δ (Line 11). This ensures that only candidates with scores comparable to the current best are fully denoised. This adaptive filter dynamically adjusts

- Algorithm 6 Adaptive Stop algorithm - Lines 5-6 of Alg. 3.

Require: source image Isrc, text prompt c, intermediate latent set Xte

, prompt set C, early step te and retain step tl Require: high-score number Nhigh, high-score threshold Shigh, tolerance factor δ

- 1: U ← {} ▷ Initialize empty set for (I,S) pairs
- 2: Srt ← 0 ▷ Initialize retain threshold
- 3: Ncnt ← 0 ▷ Initialize high-score count
- 4: ▷ Depth-first generation
- 5: for (xt

e

,cr) in (Xte

,C) do

- 6: xt

l ← Sampler(Isrc,xt

e

,cr,te,tl) ▷ Sample from te to tl

- 7: x0|t

l ← One Step Preview(xt

l

,tl) ▷ Preview image from intermediate latent (cf. Sec 3.2 Eq. 4)

- 8: I0|t

l ← VAE Decoder(x0|t

l

)

- 9: S0|t

l ← Vrf(Isrc,I0|t

l

,c) ▷ Compute unified score (cf. Sec 3.2, Eq. 8)

- 10: ▷ Retain top results by evaluated score
- 11: if S0|t

l

>= Srt − δ then

- 12: Srt ← max(Srt,S0|t

l

) ▷ Update retain threshold

- 13: x0 ← Sampler(Isrc,xt

l

,cr,tl,0) ▷ Sample from tl to 0

- 14: I ← VAE Decoder(x0)

- 15: S ← Vrf(Isrc,I,c) ▷ Compute unified score
- 16: ▷ Instance-specific verification
- 17: Sspec ← Specific Verifier(Isrc,I,c) ▷ Compute instance-specific score

- 18: S ← S + Sspec ▷ Update evaluated score
- 19: if Sspec ≥ Shigh then
- 20: Ncnt ← Ncnt + 1 ▷ Update high-score count
- 21: end if
- 22: U ← U ∪ {(I,S)}
- 23: end if
- 24: ▷ Stop when intent-aligned results suffice
- 25: if Ncnt == Nhigh then
- 26: break
- 27: end if
- 28: end for
- 29: return U

to the quality distribution of candidates. It avoids wasting computation on samples that are unlikely to be optimal.

##### B.3.2. Details of Instance-Specific Verifier

The instance-specific verifier provides a fine-grained assessment to distinguish high-quality results from subtly flawed ones. It employs a two-stage process. First, using prompt Pq (shown in Fig. 19), the MLLM generates a set of five specific yes/no questions tailored to the edit instruction. These questions cover aspects such as instruction adherence and aesthetics. Second, using prompt Pa (shown in Fig. 20), the MLLM answers these questions for the fully generated image. As implemented in Alg. 6, the instancespecific score, Sspec, is calculated by the verifier (Line 17). This score is then added to the candidate’s unified score to reward correctness (Line 18). We also track the number of intent-aligned candidates using a counter Ncnt, which is incremented if Sspec ≥ Shigh (Lines 19–21). This counter is used to trigger the final opportunistic stopping condition.

### C. Additional Experimental Results

#### C.1. Experimental Details

Our method, ADE-CoT, is built upon three open-sourced, state-of-the-art image editing models: Step1X-Edit [44], FLUX.1 Kontext [34], and BAGEL [15]. We adhere to their default configurations for the total number of denoising steps, which are T = 28,28,50, respectively. The early pruning timestep te and the late retaining timestep tl are set based on the total steps for each model. The specific hyperparameter settings used in our experiments are detailed in Tab. 6. We use Qwen-VL-MAX [3] as the MLLM for all queries and VIE-Score [32] as our general verifier Sgen. To ensure the robustness of our findings, we conduct each scaling experiment three times with different random seeds for sampling noise. The results reported in the main paper are the average of these three runs. When multiple candidates achieve the same maximum score, we compute their pairwise visual similarity using DINOv2 [59] embeddings.

### Prompt Pq for generating instance-specific questions

System Role: You are an expert AI assistant specializing in Image Editing Quality Assurance (QA). Your primary role is to act as a meticulous verifier of digital image edits. Your task is to analyze an Original Image and a corresponding Edit Instruction, and then generate a set of exactly 5 specific, verifiable questions. The fundamental rule is this: if a human reviewer answers “yes” to all 5 of your questions, it must unequivocally confirm that the edit was successfully and perfectly executed according to the instruction.

##### INPUT

- • Original Image: [The initial image before editing]
- • Edit Instruction: [A text description of the desired change] OUTPUT (must follow exactly)

- • Every question MUST be phrased so that a “yes” answer confirms a positive outcome.
- • Return a single JSON object with the following structure: {"questions": [

- "Question 1",
- "Question 2",
- "Question 3",
- "Question 4",
- "Question 5" ]}

##### CORE PRINCIPLES FOR QUESTION GENERATION

- • Instruction-Centric: Every question must directly derive from the Edit Instruction. Deconstruct the instruction into its core components (e.g., object, action, style, location).
- • Binary & Objective: Frame each question to be answerable with a simple and objective “Yes” or “No”. Avoid subjective questions like “Does it look better?” and instead focus on verifiable facts, such as “Has the color of the car been changed from blue to red?”.
- • Comprehensive Coverage: Your 5 questions must collectively cover all aspects of the instruction. If the instruction is “Make the man taller and add a hat,” you must have questions that verify both his height and the presence of the hat.
- • Negative Verification (No Collateral Damage): At least one question must check for unintended side effects. This includes verifying that parts of the image not mentioned in the instruction remain unchanged, and that no new artifacts, blurs, or distortions have been introduced.
- • Holistic Quality Check: At least one question must assess the overall integration and realism of the edit. It should check if the edit blends seamlessly with the rest of the image, maintaining consistent lighting, shadows, and texture.

User Input: Edit Instruction: ⟨edit instruction⟩ Original Image: ⟨image⟩

Figure 19. Prompt Pq for generating instance-specific questions.

We select the candidate with the highest average similarity to others as the centroid, representing the visual consensus among top-scoring outputs.

#### C.2. Details of Evaluation setting

Proposed efficiency metrics. We introduce two metrics to measure efficiency from different perspectives. ❶ Reasoning Efficiency (η): This metric is designed to measure the overall trade-off between performance and computational cost. An effective pruning strategy must not only reduce the Number of Function Evaluations (NFE) but also main-

tain high image quality. The design of η rewards methods that achieve high final scores with low NFE. The binary factor σi ensures that only methods achieving non-degraded performance are considered, which prevents strategies from gaining high efficiency scores by producing poor results. ❷ Outcome Efficiency (ξ): This metric is designed to quantify generation redundancy. Large-scale sampling in image editing often yields multiple correct outputs. An ideal method should find a satisfactory result quickly. ξ measures this by comparing the total NFE spent against the minimum NFE required to generate the first acceptable image.

### Prompt Pa for answering instance-specific questions

System Role: You are an “Image-Edit Compliance Judge.” For every dialogue turn you will receive:

- • EDIT INSTRUCTION – A text description of the desired change.

- • QUESTION LIST – exactly five yes/no questions, each asking whether a certain visual condition is true after the edit.

- • ORIGINAL IMAGE – The initial image before editing.

- • EDITED IMAGE – an edited version of the initial image. YOUR TASK

- • Imagine the edit is carried out exactly as written and reason about the resulting image.
- • For each of the five questions decide “yes” (the condition is satisfied) or “no” (the condition is not satisfied or cannot be inferred). When unsure, answer “no.”
- • Return nothing except a JSON object with five keys: "Q1", "Q2", "Q3", "Q4", "Q5".
- • The value of every key must be the lowercase string “yes” or “no”.
- • Do not output any explanations, comments, or additional keys.

OUTPUT (must follow exactly) Return a single JSON object with the following structure:

{"Q1": "yes|no",

- "Q2": "yes|no",
- "Q3": "yes|no",
- "Q4": "yes|no",
- "Q5": "yes|no"}

User Input: EDIT INSTRUCTION: ⟨edit instruction⟩ QUESTION LIST: ⟨instance-specific question⟩ ORIGINAL IMAGE: ⟨original image⟩ EDITED IMAGE: ⟨edited image⟩

Figure 20. Prompt Pa for answering instance-specific questions.

Table 6. Default hyperparameters used in our experiments.

Hyper. Value Description T 28, 28, 50 Total denoising steps for each model. te 8, 8, 16 Timestep for early preview and pruning. tl 16, 16, 36 Timestep for late retaining. Nmin 1 Minimum sampling budget. γ 0.15 Sensitivity for difficulty-aware allocation. Smax 10 Maximum score for normalization. λreg 1 Weight for the region correctness score. λcap 3 Weight for the caption consistency score. Srj 5 Rejection threshold for early pruning. τsim 0.98 Similarity threshold for filtering candidates. Nhigh 4 Number of intent-aligned results to stop.

A higher ξ indicates that the method wastes less computation on producing redundant correct images.

Two comparison settings. We analyze the performance of all methods from two different settings to provide a comprehensive comparison. ❶ Results under fixed sampling

budget. In this setting, all methods start with the same initial sampling budget (e.g., N = 32). This comparison aims to identify which method achieves the best performance and efficiency when allocated a fixed amount of computational resources. It directly shows the effectiveness of different pruning and search strategies. ❷ Results under comparable performance. In this setting, we compare the computational cost of methods that achieve a similar quality level, specifically a non-degraded performance relative to the Best-of-N (BoN) baseline. The goal is to measure the actual speedup a method provides while maintaining a target performance. This highlights the practical value of a strategy in terms of saving time and resources.

#### C.3. More Ablation Studies

We present a comprehensive ablation study of our key components in Tab. 7. Most of these results have been analyzed in the main paper. Here, we provide additional analyses to further validate the effectiveness of our design choices.

Are both edited-region correctness and instructioncaption consistency effective? To isolate the impact of

###### Table 7. Effect of the three proposed strategies on efficiency and performance. We evaluate our method on GEdit-Bench [44].

FLUX.1 Kontext [34] BAGEL [15] Step1X-Edit [44]

Model N

G O ↑ NFE ↓ η ↑ ξ ↑ G O ↑ NFE ↓ η ↑ ξ ↑ G O ↑ NFE ↓ η ↑ ξ ↑ Baseline (BoN) 32 6.641 896 0.66 0.11 6.908 1600 0.69 0.14 7.157 896 0.72 0.13

- a) + adaptive sampling 32 6.641 797 0.74 0.26 6.909 1391 0.76 0.23 7.157 778 0.81 0.27

- b) + early filter by general verifier 32 6.642 719 0.81 0.40 6.912 1351 0.79 0.39 7.157 719 0.89 0.42

- c) + early filter (+ Sreg) 32 6.645 687 0.84 0.42 6.915 1321 0.84 0.43 7.158 678 0.95 0.45

- d) + early filter (+ Scap) 32 6.647 673 0.87 0.44 6.916 1290 0.88 0.45 7.161 638 1.02 0.47

- e) + removing visually similar samples 32 6.651 508 1.26 0.58 6.915 1087 1.02 0.50 7.162 522 1.22 0.54

- f) + retaining top results at late stage 32 6.652 464 1.34 0.61 6.935 972 1.08 0.54 7.163 462 1.34 0.58

- g) + instance-specific verifier 32 6.702 464 1.37 0.63 6.984 972 1.12 0.57 7.206 462 1.36 0.60

- h) + opportunistic stopping (full model) 32 6.695 418 1.47 0.66 6.972 882 1.27 0.62 7.196 434 1.45 0.62

our edit-specific verifiers, we start from a baseline that uses only early filtering with a general verifier (Sgen) (row b). As shown in Tab. 7, adding the edited-region correctness score (Sreg) (row c) consistently reduces NFE across all models while maintaining performance. This indicates that Sreg is effective at pruning candidates with incorrect edit localization. We then further incorporate the instructioncaption consistency score (Scap) (row d). This step yields additional efficiency gains, reducing the NFE for Step1XEdit from 678 to 638 and for FLUX.1 Kontext from 687 to 673. These results confirm that both Sreg and Scap are effective and complementary. They target distinct failure modes—localization and semantic alignment, respectively—and their combined use significantly enhances the precision of our early pruning strategy.

Why fixed thresholds for early pruning but dynamic thresholds for late retaining? Our choice of pruning thresholds is based on the correlation between intermediate and final scores at different denoising stages, as shown in Fig. 21. In the early stage, the correlation between preview scores and final scores is moderate. A low preview score does not guarantee a low final score. An aggressive pruning strategy is therefore risky, as it could discard high-potential candidates. We use a fixed, conservative threshold to safely remove only clear failures. Conversely, the correlation becomes much stronger in the late stage. Fig. 21(b) shows that late-stage scores are highly predictive of the final quality. This strong correlation allows for a more aggressive strategy. We apply a dynamic threshold to filter out candidates that are unlikely to surpass the current best, which improves efficiency without performance loss.

#### C.4. More Hyperparameter Analysis

What are the optimal hyperparameters Srj and τsim? We analyze the impact of the rejection threshold Srj and the similarity threshold τsim in Fig. 22. Increasing Srj improves reasoning efficiency by pruning low-potential samples, while performance (G O) remains stable up to a threshold of 5. However, a higher Srj may remove potentially correct candidates, causing both metrics to decline.

###### Early-stageDenoisingScore

[0-2) [2-4) [4-6)

[0-2) [2-4) [4-6)

###### Late-stageDenoisingScore

80%

80%

60%

60%

40%

40%

- [6-7)

- [7-8)

- [8-9)

- [6-7)

- [7-8)

- [8-9)

20%

20%

0%

0%

[0-2) [2-4) [4-6) [6-7) [7-8) [8-9)

[0-2) [2-4) [4-6) [6-7) [7-8) [8-9)

Final Denoised Output Score

Final Denoised Output Score

(a) Early denoising stage.

(b) Late denoising stage.

Figure 21. Correlation between preview scores and final performance across denoising stages. We show correlation matrixs between intermediate scores (y-axis) and final scores (x-axis) at (a) the early and (b) late denoising stages.

ReasoningEfficiency()

ReasoningEfficiency()

6.60

6.64

0.85

1.20

6.50

6.62

0.80

###### G_O

###### G_O

1.10

6.40

6.60

0.75

1.00

G_O

G_O

6.30

6.58

0.70

0.90

0 2 4 5 6 7

0.95 0.96 0.97 0.98 0.99 1.0

Srj

sim

(a) Srj

(b) τsim

Figure 22. More hyperparameter analysis.

For τsim, decreasing it removes redundant images, which improves both performance and reasoning efficiency, peaking at τsim = 0.98. This may be because discarded similar images also tend to have low potential scores. A further decrease causes a sharp drop in both metrics. Thus, we set Srj = 5 and τsim = 0.98 as our default values.

#### C.5. More Analysis of Cost Computation

Beyond the Number of Function Evaluations (NFE), we analyze the computational cost of MLLM queries in our framework. Tab. 8 reports the average number of MLLM calls per editing case on GEdit-Bench under a sampling budget of N = 32 across three models. We observe that the general MLLM verifier, which is also required by all prior Image-CoT methods [48, 105, 107], accounts for the majority of queries in our framework. This component evaluates multiple candidates during early pruning and late retaining stages. In contrast, verifiers introduced by our method, including edited-region localization (via prompt

Table 8. Average MLLM queries per editing case on GEditBench. We report the average number of calls for each MLLM component under a sampling budget of N = 32.

MLLM Component Kontext BAGEL Step1X-Edit General Verifier 65.1 71.4 66.1 Region Generation 1.0 1.0 1.0 Caption Generation 1.0 1.0 1.0 Instance-Specific Questions (Generate) 1.0 1.0 1.0 Instance-Specific Questions (Answer) 14.9 17.6 15.5 Total Avg. 83.0 92.0 84.6

Preg), instruction-caption consistency (via prompt Pcap), and the instance-specific verifier, contribute minimal overhead. Specifically, region and caption generation require only 1.0 query per case. The instance-specific verifier generates questions once per case and answers them for a small number of top candidates, resulting in an average of 14.9, 17.6, and 15.5 queries for Kontext, BAGEL, and Step1X-Edit, respectively. The total average MLLM calls per case range from 83.0 to 92.0 across different models. This demonstrates that our edit-specific verification strategies introduce limited additional MLLM overhead while achieving significant NFE reduction compared to baseline methods.

#### C.6. More Qualitative Results

We present additional qualitative results to demonstrate the effectiveness of ADE-CoT across different editing scenarios. Fig. 10 and Fig. 11 show that our ADE-CoT significantly improves baseline model performance on complex edits and multi-turn edits. In Fig. 10, baseline models often fail on challenging tasks such as large pose changes, multiobject modifications, and fine-grained regional edits. Our method successfully handles these cases through adaptive test-time scaling. Fig. 11 demonstrates that baseline models struggle to maintain consistency across multiple sequential instructions, leading to accumulated errors in subsequent turns. In contrast, ADE-CoT preserves context from previous edits and produces correct final images that reflect all requested changes. Fig. 23 illustrates the advantage of our instance-specific verifier in final selection. Both baseline and Best-of-N methods fail to detect subtle errors in edited results. Our ADE-CoT generates targeted questions to examine critical details, enabling more accurate identification of correct outputs and producing superior final results compared to Best-of-N selection.

#### C.7. Critical Analysis and Discussion

How much do MLLMs impact our framework? Our framework, ADE-CoT, relies on MLLMs for key verification steps. It uses MLLMs to identify the edit region, generate a target caption, and compute the general score. Due to hallucinations, these models inevitably introduce incorrect judgments. To verify the impact of MLLM capabil-

- Table 9. Effect of MLLMs on edited-region correctness. We compare three MLLMs for edited-region localization (via prompt Preg), while keeping Qwen-VL-72B for other components.

Model MLLM for Region

Kontext BAGEL Step1X-Edit

G O↑ NFE↓ G O↑ NFE↓ G O↑ NFE↓ BoN - 6.641 896 6.908 1600 7.157 896

ADE-CoT

Qwen2.5-VL-72B [3] 6.637 436 7.042 897 7.193 446 Qwen-VL-MAX [3] 6.661 428 6.993 901 7.194 445 Qwen3-VL-32B [93] 6.673 424 7.048 869 7.198 436

- Table 10. Effect of MLLMs on instruction-caption consistency. We compare three MLLMs for instruction-caption consistency (via prompt Pcap), while keeping Qwen-VL-72B for others.

Kontext BAGEL Step1X-Edit

Model MLLM for Caption

G O↑ NFE↓ G O↑ NFE↓ G O↑ NFE↓ BoN - 6.641 896 6.908 1600 7.157 896

Qwen2.5-VL-72B [3] 6.637 436 7.042 897 7.193 446 Qwen-VL-MAX [3] 6.651 419 7.021 899 7.186 450 Qwen3-VL-32B [93] 6.664 423 7.052 883 7.195 440

ADE-CoT

ity on our framework, we conduct experiments with three different MLLMs. We find that ADE-CoT demonstrates strong robustness across varying MLLM capacities. ❶ As shown in Tab. 5 of the main paper, our method consistently achieves over 2× speedup compared to BoN across all tested MLLMs. ❷ We observe that even when using a weaker MLLM such as Qwen2.5-VL-72B, replacing specific components with stronger MLLMs leads to consistent improvements. Tab. 9 shows that replacing the MLLM for region localization improves edited-region correctness and overall performance. Similarly, Tab. 10 demonstrates that using a stronger MLLM for caption generation enhances instruction-caption consistency. These results show that more accurate predictions from better MLLMs lead to consistent gains in both performance and efficiency.

Can Image-CoT improve all editing cases? As illustrated in Fig. 14, we find that some samples still receive low scores even after applying Image-CoT. These cases typically represent scenarios where the model inherently lacks editing capability. Even after extensive sampling or prompt modification, their performance shows minimal improvement. It also demonstrates that Image-CoT can serve as a diagnostic method to identify model capability boundaries. By incorporating these cases into training data, we can further enhance the capabilities of existing models.

### D. Limitations and Future Work

Limitations. Despite achieving superior trade-offs between performance and efficiency, our method suffers from the following limitations: ❶ MLLM computational overhead. Our verification relies on large-scale MLLMs (e.g., Qwen-VL 72B), which increases inference latency and limits in resource-constrained scenarios. ❷ Hallucination in

###### Input Image Baseline BoN ADE-CoT Input Image Baseline BoN ADE-CoT

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

Prompt: Make me look 10 pounds thinner. Prompt: Make the person jump.

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

|[Figure 327]|
|---|

Prompt: Add a book in her hand. Prompt: Make the person in the photo dance.

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

|[Figure 334]|
|---|

|[Figure 335]|
|---|

Prompt: Adjust the background to a beach. Prompt: Dress the person in the image in a suit.

Figure 23. Instance-specific verifier improves fine-grained selection. We compare the baseline model, Best-of-N (BoN), and our ADECoT across challenging editing scenarios. Baseline models and BoN often produce results with subtle errors (marked in red boxes), such as incorrect pose changes, background adjustments, and clothing modifications. In contrast, our instance-specific verifier generates targeted questions to examine critical details, enabling accurate detection of editing errors and selecting correct results (marked in green boxes).

verification. MLLMs may generate hallucinations during the verification process. This can compromise the accuracy of generated captions, region masks, and instancespecific questions, leading to incorrect quality assessments. While our experiments demonstrate that different MLLMs show minimal variance when ranking multiple candidates (Sec. 4.3, Tab. 5), accurately determining whether the edited image fully satisfies user intent remains challenging.

Future work. We identify two primary directions for future research based on these limitations. ❶ Efficient and accurate verification models. A key direction is to utilize smaller, specialized models for evaluation. Lightweight models (e.g., 7B parameters) could be trained to provide fast and accurate assessments of edited images. Additionally, an accurate evaluation model would be particularly effective for evaluating intermediate preview images during the denoising process. This could enhance the proposed edit-specific verification and opportunistic stopping strategies, further improving overall efficiency. ❷ Broader applications. Our ADE-CoT framework could be extended to other goal-directed generation tasks. 1 Its core strategies of difficulty-aware resource allocation and opportunistic stopping are applicable to domains like video editing and multiturn conversational generation. These strategies may also be beneficial for standard text-to-image and text-to-video

generation, enabling a more efficient Image-CoT process.

### E. Extended Related Work

Test-time scaling in image generation. Recent years have seen rapid development in generative models [9, 15, 30, 34– 36, 44, 51, 76, 77, 81, 83–85, 88, 91, 95]. Test-time scaling, as a training-free approach, aims to improve generation quality by extending the inference time. Early work on diffusion-based models investigates this by scaling the number of denoising steps [31, 46, 68, 71–73]. More recently, following the success of Chain-of-Thought (CoT) [4, 10, 11, 14, 37, 40, 49, 52, 63, 64, 87, 98–100], Image-CoT has emerged as a promising paradigm. The standard Image-CoT method is noise scaling [48], which generates multiple candidates by perturbing the noise and selects the best image as the final output. While effective, its computational cost scales linearly with the number of candidates. Subsequent work aims to generate higherquality candidates within a fixed budget. Some methods enhance candidate diversity through prompt-level intervention, which enhances prompts by rewriting [29, 56] or reflective updates for iterative refinement [38, 105, 111]. Another line of work adapts search algorithms to the diffusion process. These methods treat the reverse diffusion chain as

a search trajectory [21, 28, 48, 70, 106]. They change the noise based on verifier scores to select promising sampling directions. Recent methods [43, 60, 79, 89, 90, 105, 107] utilize MLLMs as a verifier to prune low-potential trajectories at early denoising stages. However, most ImageCoT methods focus on text-to-image generation. They are inefficient for image editing due to three key issues: (1) fixed sampling budgets waste computation on simple edits, (2) general MLLM scores cause misjudgement during early pruning, and (3) large-scale sampling produces redundant correct outputs. To address these challenges, we propose ADE-CoT, an edit-specific test-time scaling method that incorporates difficulty-aware resource allocation, editspecific verification, and depth-first opportunistic stopping. Verifiers for image editing can be divided into two primary approaches. The first approach comprises metrics requiring ground-truth edited images and corresponding output captions. These methods typically utilize CLIP Score [23] for image-text alignment, CLIP [65] and DINO [8] similarity for image-image comparison, and L1/L2 distance for pixel-level similarity. However, such metrics are difficult to apply in test-time scaling (TTS) scenarios, as ground-truth data are unavailable during inference. The second approach leverages MLLMs as verifiers. Existing methods, such as VIEScore [32] and HQScore [27], use general verifiers that use instance-agnostic prompts to evaluate aesthetic quality and semantic consistency for each instance. However, general verifiers face two limitations in editing Image-CoT. In early denoising stages, they may incorrectly pruning highpotential candidates that exhibit low preview quality. In the final selection, they struggle to distinguish subtle differences between candidates, assigning identical high scores to images with minor errors. To address these limitations, we introduce two complementary verification strategies. For early-stage misjudgement, we propose edit-specific verification that queries MLLMs to generate ground-truth captions and expected edit regions, enabling more accurate assessment of caption consistency and region localization. For final-stage selection, we introduce instance-specific verification that first generates targeted yes-no questions about specific editing aspects, then provides answers based on these questions. This two-stage inquiry guides MLLMs to notice critical details and provides the decision for opportunistic stopping, improving fine-grained selection accuracy, and reducing redundant outputs.

