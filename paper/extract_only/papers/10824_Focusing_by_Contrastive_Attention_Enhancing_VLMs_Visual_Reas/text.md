# arXiv:2509.06461v2[cs.CV]11Sep2025

## FOCUSING BY CONTRASTIVE ATTENTION: ENHANCING VLMS’ VISUAL REASONING

Yuyao Ge1 Shenghua Liu1∗ Yiwei Wang2 Lingrui Mei1 Baolong Bi1 Xuanshan Zhou1 Jiayu Yao1 Jiafeng Guo1 Xueqi Cheng1 1Institute of Computing Technology, Chinese Academy of Sciences 2University of California, Merced {geyuyao24z, liushenghua}@ict.ac.cn

ABSTRACT

Vision-Language Models (VLMs) have demonstrated remarkable success across diverse visual tasks, yet their performance degrades in complex visual environments. While existing enhancement approaches require additional training, rely on external segmentation tools, or operate at coarse-grained levels, they overlook the innate ability within VLMs. To bridge this gap, we investigate VLMs’ attention patterns and discover that: (1) visual complexity strongly correlates with attention entropy, negatively impacting reasoning performance; (2) attention progressively refines from global scanning in shallow layers to focused convergence in deeper layers, with convergence degree determined by visual complexity. (3) Theoretically, we prove that the contrast of attention maps between general queries and task-specific queries enables the decomposition of visual signal into semantic signals and visual noise components. Building on these insights, we propose Contrastive Attention Refinement for Visual Enhancement (CARVE), a trainingfree method that extracts task-relevant visual signals through attention contrasting at the pixel level. Extensive experiments demonstrate that CARVE consistently enhances performance, achieving up to 75% improvement on open-source models. Our work provides critical insights into the interplay between visual complexity and attention mechanisms, offering an efficient pathway for improving visual reasoning with contrasting attention.

1 INTRODUCTION

Vision-Language Models (VLMs) have achieved remarkable success across diverse tasks (Radford et al., 2021; Jia et al., 2021; Alayrac et al., 2022). However, in human vision, complex visual features frequently divert attention from task-relevant regions (Treisman & Gelade, 1980). Given this cognitive parallel, a question arises: Similarly, do complex images interfere with VLMs’ attention mechanisms, making it difficult for them to focus on task-relevant regions?

To answer this question, we investigate the relationship between visual complexity and attention patterns via quantitative experiments. Specifically, we define visual complexity as texture and color dimensions, revealing a significant positive correlation between both factors and attention entropy. Furthermore, our analysis shows that attention entropy negatively correlates with accuracy on visual reasoning tasks. Through this two-stage analysis, we establish that complex visual information impairs VLMs’ reasoning performance via attention distribution (detailed in Section 3).

Based on these findings, we conduct a preliminary experiment on TextVQA (Singh et al., 2019) by first applying progressive masking to obscure background regions, then cropping to retain only task-relevant regions and adaptively magnifying them to the original image size. Figure 1 presents two representative samples where cluttered visual environments initially cause incorrect predictions. While incorrect token probability initially prevails in both samples, correct token probability surpasses incorrect probability at mask ratios of approximately 0.02 and 0.65 respectively. These results provide initial validation that masking visual noise can improve correct token probability.

∗ Corresponding author.

|Q: What shape is seen through the cups handle?|
|---|

|Q: How many crayons are in the box?|
|---|

[Figure 1]

[Figure 2]

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

- Figure 1: The effect of manually progressive masking on candidate tokens probabilities predicted by QWEN2.5-VL-3B. The x-axis represents mask ratio and y-axis shows log10 probability.

To automate the visual noise masking process, we leverage contrasting attention maps between general instructions and task-specific questions to distinguish semantic signal from visual noise. To this end, we propose Contrastive Attention Refinement for Visual Enhancement (CARVE), a contrastive method for visual extraction. By masking with contrastive attention maps, CARVE crops and magnifies semantic regions to focus on essential semantic signal (detailed in Section 4).

- 2 RELATED WORK

Contrastive Learning in LLMs. Liu et al. (2023b) pioneered contrastive objectives for aligning LLMs with human preferences, establishing the foundation for RECIPE (Chen et al., 2024b) which trains a Knowledge Sentinel to determine when queries trigger knowledge updates. Building on alignment challenges, Jiang et al. (2024) employ hallucinated text as hard negatives while Zhang

- et al. (2024b) apply contrastive learning in hidden representations to suppress hallucinations. Zhai
- et al. (2025) traces critical transmission paths across all layers, treating less important pathways as negatives, which Pan et al. (2024) further adapted to multimodal LLMs through UniKE’s semantictruthfulness space disentanglement. Departing from embedding-space modifications, DeCK (Bi et al., 2025a) shifts contrastive logic to the decoding stage by comparing token probabilities with and without injected knowledge, while parallel applications emerged in DistiLLM-2 (Ko et al., 2025) for knowledge distillation and Zhu et al. (2024) for factual consistency enhancement.

Attention-based LLM Optimization. Alayrac et al. (2022) established multimodal foundations through perceiver resampler architecture, which Li et al. (2023a) refined via a lightweight Querying Transformer for parameter-efficient visual extraction. Extending attention optimization to text modality, Chen et al. (2025) exploit attention scores for dynamic prompt compression through importance sampling at both token and sentence levels. Ma et al. (2024a) eliminated redundant visual token computations while Acharya et al. (2024) introduced block-sparse mechanisms for parallel processing, and Liu et al. (2025b) bypassed attention bottlenecks through sequential chunk processing in Recurrent LLMs. Yao et al. (2025b) identified position bias in multimodal RAG where models over-focus on boundary items. Zhang et al. (2025) discovered that models consistently know where to look, even when they provide the wrong answer. Our method eliminates visual noise by contrasting attention maps to distinguish semantic pixels from noise pixels without requiring training.

- 3 FAILURE TO FOCUS: PHENOMENON, MECHANISM, AND CONSEQUENCE

- 3.1 PHENOMENON: UNDER WHAT CONDITIONS VISUAL FOCUS FAILS

Building upon the question in Section 1, we aim to investigate the underlying causes of VLMs’ answering failures. We conduct experiments on TextVQA dataset using QWEN2.5-VL-3BINSTRUCT. As shown in Figure 2, we visualize attention maps during inference and find two interesting phenomena. Attention progressively refines from broad global scanning in shallow layers to regional localization in the middle layers, culminating in focused convergence in deep layers. The degree of convergence varies based on input images.

Moreover, visual complexity critically influences attention convergence. In simple scenes with clear targets and minimal distractors, the high-attention regions successfully narrow as layers deepen,

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

29

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Q: What is the player's number?

Focus on Object

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

jim beam

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Q: What brand is the bottle with red label?

Confused where to look

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

infuser

[Figure 48]

[Figure 49]

[Figure 50]

Global Scanning

Finding Object

Q: What brand of vinegar is in the bottles on the left?

[Figure 51]

Shallow Layers

###### Middle Layers Deep Layers

- Figure 2: Attention maps across different layers during inference. Each row represents a visual question-answering task: row 1 shows a simple scene with clear targets, while rows 2-3 depict complex scenes with dense textures and multiple similar objects. From left to right, the columns display: input images, shallow layer attention, middle layer attention, and deep layer attention.

aligning with task-relevant regions. Conversely, in complex scenes with rich textures and colors, the high-attention regions still attempt to narrow as layers deepen, yet the resulting attention weights remain more diffused compared to simple scenes. As indicated by the annotation “Confused where to look”, this attention dispersion resembles human hesitation when confronting crowded shelves and ultimately manifests as reasoning failures. These observations lead us to formulate a question: Does visual complexity affect VLMs’ attention distribution, and does this attention distribution further influence VLMs’ performance?

To answer it, we conduct two deeper experiments: First, we quantify the relationship between visual complexity and attention entropy to establish whether complex inputs produce dispersed attention (Section 3.2); Second, we examine the correlation between attention entropy and model performance to determine whether dispersed attention contributes to reasoning failures (Section 3.3).

- 3.2 MECHANISM: THE EFFECT OF VISUAL COMPLEXITY ON ATTENTION ENTROPY

In Figure 2, images in rows 2-3 differ from row 1 by displaying numerous colorful bottles, containing significantly more textures and colors. Therefore, we decompose visual complexity into two dimensions: texture and color, and investigate their respective impacts on attention.

We define the texture complexity and the color complexity as follows:

Texture Complexity. Let I ∈ RH×W×3 denote an input image. We define the texture complexity Tc(I) using Canny edge detection (Canny, 1986), where E(I) ∈ {0,1}H×W represents the resulting binary edge map. The texture complexity is then defined as:

1 HW

Tc(I) =

H

W

E(I)ij = ∥E(I)∥1 HW ∈ [0,1] (3.1)

i=1

j=1

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

(a) Original (b) Edge detection

179

[Figure 56]

|[Figure 57]| |
|---|---|
| | |
| | |

120

HueValue

60

0

179

[Figure 58]

|[Figure 59]| |
|---|---|
| | |
| | |

120

HueValue

60

0

(c) Hue distribution

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.05

0.04

0.03

0.02

0.01

0.00

0 25 50 75 100 125 150 175

0.14

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.12

0.10

0.08

0.06

0.04

0.02

0.00

0 25 50 75 100 125 150 175

(d) Hue histogram

- Figure 3: Visualization of texture and color complexity analysis. Each row represents a sample

image: (a) Original image I, (b) Canny edge map E(I) for texture complexity Tc, (c) Spatial hue distribution ζ in HSV space, and (d) Hue statistic (x-axis: hue value, y-axis: ratio).

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Texture Complexity

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

AttentionEntropy

0.38 0.41 0.44

0.53 0.56 0.58 0.60 0.61 0.63

0.67

(a) Texture-entropy correlation

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Color Complexity

0.0

0.1

0.2

0.3

0.4

0.5

0.6

AttentionEntropy

0.37 0.39 0.40 0.41 0.42 0.45 0.46

0.55 0.57 0.60

(b) Color-entropy correlation

- Figure 4: Correlation analysis between visual complexity and attention entropy. Both attention entropy and complexity are normalized to the [0,1], divided into intervals of 0.1, and the average attention entropy is calculated within each interval.

Color Complexity. Let ζij = Hue(ΨRGB→HSV (Iij)) denote the hue value at pixel (i,j) after applying the RGB to HSV transformation operator Ψ. The color complexity is then defined as:

B−1

1 lnB

nb HW

ρb lnρb, where ρb =

, nb = |{(i,j) : ζij = b}| (3.2)

Cc(I) = −

b=0

with B = 180 hue bins, yielding Cc ∈ [0,1] where higher values indicate greater color diversity.

- Figure 3 visually validates the effectiveness of our complexity measurement approach. The first row demonstrates high texture complexity with dense edge networks in E(I) and diverse color distribution across the hue spectrum. In contrast, the second row exhibits minimal edge density and concentrated hue values, indicating lower complexity scores. We therefore proceed to quantitatively investigate the correlation between complexity metrics and attention distribution.

For measuring the distribution of attention across visual tokens, inspired by Yao et al. (2025b), we employ Shannon entropy (Shannon, 1948) as our quantification metric. Let Nv denote the number of visual tokens in the model’s representation. We denote the attention map as A(l,tQ) ∈ RN

v, where l indicates the layer index, t the generation time step, and Q the input question. For entropy analysis, we focus on the final generation step tend and define the overall attention entropy H as:

H =

1 |L| l∈L

H(A(l,tQ)

end

) =

1 |L| l∈L

−

Nv

i=1

al,t

end,i lnal,t

end,i (3.3)

where L = [Lstart,Lend] represents the layer range under consideration, and al,t,i denotes the contrasted attention weight for the i-th visual token. Higher entropy indicates more dispersed attention, while lower entropy indicates more concentrated focus.

- Figure 4 presents the correlation analysis between our defined complexity metrics and computed attention entropy. Both texture complexity (Figure 4a) and color complexity (Figure 4b) exhibit strong positive linear relationships with attention entropy. This monotonic trend indicates that complex visual features lead to dispersed attention patterns in VLMs.

90

Average Performance

85

95% CI

80

Performance(%)

75

70

65

60

55

50

5.2 5.4 5.6 5.8 6.0 6.2 6.4 6.6 6.8 Attention Entropy

(a) Entropy-accuracy correlation

Average Performance 95% CI

- 2

- 3

- 4

- 5

- 6

- 7

- 8

AttentionEntropy

5 10 15 20 25 30 Layers

(b) Layer-wise entropy correlation

- Figure 5: Attention entropy’s correlation with accuracy and its evolution across layers. Shaded regions indicate 95% confidence intervals computed as x¯ ± t0.975,n−1 · s/√n. (a) Shows accuracy for samples grouped by overall attention entropy H. (b) Displays mean entropy across N samples

### per layer as N1 Ni=1 H(A(l,tQ,i)

### ), where A(l,tQ,i)

is sample i’s attention at the final generation step.

end

end

- 3.3 CONSEQUENCE: HOW DISPERSED ATTENTION IMPAIRS PERFORMANCE

Figure 5(a) reveals a strong negative correlation between attention entropy and accuracy. As attention entropy increases from 5.1 to 6.8, performance decreases from approximately 76% to 65%, confirming that increased attention dispersion directly impairs visual reasoning capabilities in VLMs.

To investigate the hierarchical evolution of attention entropy, we present mean entropy and its distribution across layers in Figure 5(b). The results reveal two notable characteristics: (1) attention entropy monotonically decreases with layer depth, consistent with Figure 2. (2) The 95% confidence intervals progressively widen with increasing depth, indicating enhanced inter-sample variability. For samples with clear visual targets, deep layers achieve highly concentrated attention. In contrast, for noisy samples, the model maintains dispersed attention patterns even in deep layers.

- 4 CONTRASTIVE ATTENTION REFINEMENT FOR VISUAL ENHANCEMENT

- 4.1 THEORETICAL FOUNDATION: NOISE SUPPRESSION AND VISUAL REFINEMENT

Based on our findings in Section 3, where we demonstrated that visual complexity causes attention dispersion and performance degradation, we seek to extract pure task-related semantic signal. Therefore, we first formally define the attention signal decomposition mechanism.

- Definition 1 (Attention Decomposition): Attention distributions are influenced by inherent visual noise (detailed in Appendix A.2) of the image and task-related semantic signal. The attention map A(l,tQ)(I) decomposes as:

A(l,tQ)(I) = Fvis(I) ⊗ Fsem(Q,I) (4.1) where Fvis(I) ∈ RN

v captures image-inherent visual noise, Fsem(Q,I) ∈ RN

v captures task-related

semantic signal, and ⊗ denotes the Hadamard product. When using general instructions G, due to the absence of specific tasks to introduce semantic information, the semantic signal function reduces to uniform distribution (Fsem(G,I) ≈ 1N

v

), making general instruction attention predominantly capture visual noise:

A(l,tG)(I) ≈ Fvis(I) ⊗ 1N

v

= Fvis(I) (4.2)

- Definition 2 (Semantic Extraction Based on Attention Decomposition): To extract semantic sig-

nal function Fsem(Q,I) from A(Q), we define estimated semantic attention Aˆ ∈ RN

+ as our estimate of Fsem(Q,I), which is the solution to the following optimization problem:

v

J (A˜;A(Q),A(G)) (4.3) where the objective function is constructed based on Definition 1’s decomposition:

Aˆ = arg min

A˜∈A

Attention Maps

[Figure 60]

[Figure 61]

Output

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

<General Instruction>

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

|Cup|
|---|

[Figure 76]

The high-attention area should be focus on.

𝐴𝐴(𝐺𝐺)𝑖𝑖

|Write a of the image.<br><br>|general description|
|---|
<br><br>[Figure 77]|
|---|

|What shape is seen through the cups hand?<br><br><Question>|
|---|

[Figure 78]

[Figure 79]

[Figure 80]

𝐴𝐴̂𝑖𝑖

[Figure 81]

[Figure 82]

[Figure 83]

<Original Image>

Contrast

| ||[Figure 84]<br><br>[Figure 85]<br><br>What shape is seen through the<br><br>|cups hand?|
|---|
|
|---|
<br><br><Question>|
|---|---|
| | |

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

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

𝐴𝐴(𝑄𝑄)𝑖𝑖

Output

[Figure 102]

[Figure 103]

Too much visual noise, but I know which area to look in…

|[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>Circle<br><br>|
|---|

Mask

[Figure 107]

[Figure 108]

<Masked Image>

<Question>

- Star 1

[Figure 109]

- Star 2

Output VLM

[Figure 110]

The visual noise has been masked, so I can focus on the object easily.

[Figure 111]

[Figure 112]

[Figure 113]

|[Figure 114]<br><br>[Figure 115]<br><br>Star<br><br>|
|---|

- Star 1

[Figure 116]

- Star 2

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

- Figure 6: CARVE comprises three stages: Stage 1 generates general attention distribution A(iG) with general instructions; Stage 2 extracts task-specific attention A(iQ); Stage 3 applies contrasted attention Aˆi to generate enhanced masked images for noise suppression.

J (A˜) =

Nv

2

A ˜i · Fvis,i(I) − [Fvis,i(I) · Fsem,i(Q,I)]

i=1

Semantic reconstruction error

Nv

A˜2i · Fvis,i(I)

+ λ

i=1

Visual suppression regularization

(4.4)

Theorem 3 (Closed-form Solution for Semantic Extraction): Substituting Definition 1’s relationships A(iQ) ≈ Fvis,i · Fsem,i and A(iG) ≈ Fvis,i into the optimization objective yields:

Nv

J (A˜) =

i=1

A ˜i · A(iG) − A(iQ)

Nv

2

A˜2i · A(iG) (4.5)

+ λ

i=1

where λ > 0 is a regularization parameter that controls the strength of visual noise suppression. Solving the first-order optimality conditions yields the closed-form solution:

A(iQ) A(iG) + λ

= Fvis,i · Fsem,i Fvis,i + λ ≈ Fsem,i when Fvis,i ≫ λ (4.6)

Aˆi =

Equation 4.6 demonstrates that normalization suppresses the influence of Fvis,i when it dominates (i.e., Fvis,i ≫ λ), approximating the semantic signal Fsem,i (detailed analysis in Appendix C).

- 4.2 CONTRASTIVE ATTENTION-BASED VISUAL ENHANCEMENT

Having obtained the semantically refined attention maps {Aˆ}, as shown in Algorithm 1, we now proceed to generate attention masks that physically remove visual noise from the input image.

Attention Maps Fusion. Since different layers and time steps capture complementary information, we fuse attention maps across the layer range L and generation time steps T = [tstart,tend] through weighted aggregation. Later tokens encode richer contextual information by accessing complete preceding sequences during inference, thus receiving higher fusion weights.

Mask Generation and Visual Extraction. Task-relevant regions are identified by applying the top-p percentile threshold τ = Qp(S), which retains the top p ∈ (0,1] proportion of pixels from attention map S. Connected component analysis extracts coherent regions from the thresholded map.

##### Model Step: T A-OKVQA POPE V∗ TextVQA

w/o CARVE 73.0(−) 86.9(−) 50.3(−) 72.8(−)

tstart 76.5(↑4.79) 87.1(↑0.23) 56.0(↑11.33) 76.1(↑4.53) tend 79.2(↑8.49) 88.4(↑1.73) 57.1(↑13.52) 76.4(↑4.95) Tfull 78.3(↑7.26) 87.9(↑1.15) 56.5(↑12.33) 76.3(↑4.81)

QWEN2.5-VL-3B

w/o CARVE 75.0(−) 87.0(−) 50.8(−) 75.0(−)

tstart 77.0(↑2.67) 87.9(↑1.03) 58.6(↑15.35) 80.7(↑7.60) tend 78.3(↑4.40) 89.7(↑3.10) 59.7(↑17.52) 81.9(↑9.20) Tfull 78.0(↑4.00) 88.6(↑1.84) 58.1(↑14.37) 81.7(↑8.93)

QWEN2.5-VL-7B

w/o CARVE 71.5(−) 83.6(−) 38.7(−) 47.8(−)

tstart 73.9(↑3.36) 86.8(↑3.83) 57.1(↑47.55) 57.9(↑21.13) tend 78.2(↑9.37) 89.0(↑6.46) 66.5(↑71.83) 58.2(↑21.76)

LLAVA1.5-7B

- Tfull 75.4(↑5.45) 89.0(↑6.46) 66.5(↑71.83) 57.9(↑21.13)

LLAVA1.5-13B

w/o CARVE 75.7(−) 84.6(−) 42.4(−) 57.1(−)

tstart 76.2(↑0.66) 90.0(↑6.38) 65.4(↑54.25) 59.2(↑3.68) tend 76.9(↑1.59) 90.7(↑7.21) 74.3(↑75.24) 61.2(↑7.18)

- Tfull 76.5(↑1.06) 90.1(↑6.50) 70.0(↑65.09) 61.2(↑7.18)

- Table 1: Accuracy comparison of CARVE across VLMs on four datasets. We evaluate three tempo-

ral configurations: tstart uses attention from initial generated tokens, tend from final tokens, and Tfull applies weighted fusion across all tokens. We use layer range L = [20,25] for attention fusion.

Algorithm 1 CARVE: Contrastive Attention Refinement for Visual Enhancement

Notation: M: VLM model; Ξ: attention extraction; πH×W: spatial reshape;Qp: top-p threshold; Φ: visual extraction (mask, crop, resize); G: general instruction; τ: threshold; R: connected regions; K: max regions to keep

Require: I ∈ RH×W×3, Q, M, Θ = {L,T ,p,λ,K}

- 1: Inference: AQ ← {A(l,tQ)}l∈L,t∈T = Ξ(M,I,Q) ▷ Question-specific attention
- 2: Inference: AG ← {A(l,tG)}l∈L,t∈T = Ξ(M,I,G) ▷ General attention
- 3: Contrast: Aˆl,t ← A

(Q) l,t

A(l,tG)+λ

for all l ∈ L,t ∈ T ▷ following Eq. 4.6

- 4: Fuse: S ← t∈T wt l∈L πH×W(Aˆl,t), wt = t − tstart + 1 ▷ Weighted attention fusion
- 5: Threshold: τ ← Qp(S) ▷ Compute threshold to retain top p percentile
- 6: Mask: M∗ = Kk=1 Rk∗ where Rk∗ = arg maxR∈R (i,j)∈R S(i,j) with R from S ≥ τ
- 7: Extract: Irefined ← Φ(I,M∗) ▷ Visual extraction
- 8: Inference: return M(Irefined,Q) ▷ Final inference

We select the top-K regions ranked by cumulative attention scores and generate the enhanced image through Irefined = Φ(I,M∗), where Φ applies masking, cropping, and resizing, and K controls the maximum number of regions to preserve. This refinement eliminates visual noise while magnifying task-relevant content, enabling focused attention on task-related areas.

As shown in Figure 6, we propose Contrastive Attention Refinement for Visual Enhancement (CARVE), a method that contrasts attention maps to distinguish semantic pixels from noise, preserving only task-relevant regions for enhanced model focus (detailed in Appendix B).

- 5 METHOD ANALYSIS

- 5.1 EXPERIMENTAL SETUP

Datasets. We conduct our experiments on four datasets: A-OKVQA (Schwenk et al., 2022), POPE (Li et al., 2023b), V∗ (Wu & Xie, 2023), and TextVQA (Singh et al., 2019), which cover multiple task dimensions including visual reasoning, visual understanding, and visual knowledge reasoning. For TextVQA, we evaluate the models’ intrinsic visual text recognition capabilities by providing only images and questions without external OCR augmentation (detailed in Appendix E).

Models. We conduct experiments on four VLMs: QWEN2.5-VL-3B-INSTRUCT, QWEN2.5-VL7B-INSTRUCT (Qwen, 2025), LLAVA-1.5-7B, and LLAVA-1.5-13B (Liu et al., 2023a). The

###### Model Layer(s): L A-OKVQA POPE V∗ TextVQA

w/o CARVE 73.0(−) 86.9(−) 50.3(−) 72.8(−)

14 74.3(↑1.78) 87.1(↑0.23) 53.9(↑7.16) 73.6(↑1.10) 20 76.5(↑4.79) 87.4(↑0.58) 56.0(↑11.33) 74.7(↑2.61) 25 76.7(↑5.07) 87.5(↑0.69) 56.0(↑11.33) 75.9(↑4.26)

Single Layer

QWEN2.5-VL-3B

[10, 15] 74.0(↑1.37) 86.9(0.00) 53.4(↑6.16) 73.0(↑0.27) [15, 20] 76.8(↑5.21) 87.7(↑0.92) 56.0(↑11.33) 76.0(↑4.40) [20, 25] 78.3(↑7.26) 87.9(↑1.15) 57.1(↑13.52) 76.3(↑4.81)

Multi-Layers

w/o CARVE 75.0(−) 87.0(−) 50.8(−) 75.0(−)

14 75.2(↑0.27) 87.5(↑0.57) 54.5(↑7.28) 75.2(↑0.27) 20 76.9(↑2.53) 87.9(↑1.03) 56.5(↑11.22) 77.9(↑3.87) 25 77.0(↑2.67) 88.2(↑1.38) 57.0(↑12.20) 78.4(↑4.53)

Single Layer

QWEN2.5-VL-7B

[10, 15] 75.0(0.00) 87.0(0.00) 51.3(↑0.98) 75.0(0.00)

Multi-Layers

[15, 20] 77.1(↑2.80) 88.4(↑1.61) 57.6(↑13.39) 79.5(↑6.00) [20, 25] 78.0(↑4.00) 88.6(↑1.84) 58.1(↑14.37) 81.7(↑8.93)

w/o CARVE 71.5(−) 83.6(−) 38.7(−) 47.8(−)

14 71.7(↑0.28) 85.1(↑1.79) 63.4(↑63.82) 54.0(↑12.97) 20 74.0(↑3.50) 87.2(↑4.31) 65.4(↑68.99) 56.1(↑17.36) 25 74.1(↑3.64) 87.1(↑4.19) 65.4(↑68.99) 56.2(↑17.57)

Single Layer

LLAVA1.5-7B

[10, 15] 71.5(0.00) 84.5(↑1.08) 48.2(↑24.55) 49.2(↑2.93) [15, 20] 74.2(↑3.78) 87.5(↑4.67) 65.4(↑68.99) 56.4(↑17.99) [20, 25] 75.4(↑5.45) 89.0(↑6.46) 66.5(↑71.83) 58.2(↑21.76)

Multi-Layers

w/o CARVE 75.7(−) 84.6(−) 42.4(−) 57.1(−)

14 75.8(↑0.13) 86.1(↑1.77) 66.5(↑56.84) 58.2(↑1.93) 20 76.2(↑0.66) 88.2(↑4.26) 68.6(↑61.79) 59.1(↑3.50) 25 76.2(↑0.66) 88.1(↑4.14) 69.0(↑62.74) 59.2(↑3.68)

Single Layer

LLAVA1.5-13B

[10, 15] 75.7(0.00) 85.0(↑0.47) 52.9(↑24.76) 57.4(↑0.53) [15, 20] 76.8(↑1.45) 88.6(↑4.73) 69.1(↑62.97) 59.4(↑4.03) [20, 25] 76.9(↑1.59) 90.1(↑6.50) 70.0(↑65.09) 61.2(↑7.18)

Multi-Layers

- Table 2: We investigate CARVE’s accuracy using both single-layer and multi-layer intervention strategies at shallow, middle, and deep model depths, where single-layer interventions use attention

maps Aˆi from individual layers, while multi-layer interventions fuse maps across multiple layers to guide masking decisions. We employ Tfull as the time step configuration.

Qwen family processes images at 448 × 448 resolution, while the LLaVA-1.5 family operates at 336 × 336 resolution. All models employ greedy decoding.

- 5.2 RESULTS

CARVE Enhances VLMs’ Visual QA Performance. Tables 1 and 2 demonstrate CARVE’s consistent performance enhancement across all evaluated models and datasets. Earlier-generation models exhibit substantially greater improvements than their more recent counterparts. For instance, LLAVA1.5-7B achieves a 71.83% relative improvement on V∗, whereas QWEN2.5-VL-7B shows a 17.52% gain. This pattern indicates that limited-capability models suffer more from visual complexity interference and benefit more from contrastive attention-guided focusing mechanisms.

Ablation Study on the Time Step. Table 1 reveals a consistent performance hierarchy across various time step selection strategies. Specifically, tend generally outperforms Tfull, which in turn surpasses tstart across most experimental configurations. This pattern is exemplified by QWEN2.5VL-7B’s performance on TextVQA, where tend achieves 81.9% accuracy, followed by Tfull at 81.7% and tstart at 80.7%. This phenomenon aligns with architectural principles. Later tokens encode richer contextual information by accessing complete preceding sequences during inference. Consequently, the final token’s attention maps accurately localize target objects, providing prerequisite conditions for CARVE’s noise masking mechanism.

Ablation Study on the Layer Selection. To investigate layer selection effects on attention pattern extraction, we conduct systematic experiments as shown in Table 2. Across all tested model architectures, the layer-wise performance demonstrates the following general ordering from best to worst: [20,25], [15,20], single layer 25, single layer 20, single layer 14, and [10,15]. This pattern is exemplified by LLAVA1.5-7B’s performance on TextVQA, where the multi-layer [20,25] achieves a 21.76% improvement, the [15,20] reaches a 17.99% improvement, while the early-layer [10,15]

|72.80| |72.80| |72.80| |72.80| |72.80| |
|---|---|---|---|---|---|---|---|---|---|
|74.40| |[Figure 123]<br><br>75.60| |74.10| |74.00| |73.80| |
|75.70| |76.40| |76.10| |75.20| |74.00| |
|76.80| |77.40| |76.40| |75.80| |74.90| |
|76.70| |77.00| |76.60| |75.90| |74.20| |
| | | | | | | | | | |

100%

ThresholdPercentile

80%

60%

40%

20%

1 2 3 4 5

Max Keep Regions

|[Figure 124]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

- 73
- 74
- 75
- 76
- 77

(a) Accuracy for QWEN2.5-VL-3B on TextVQA

|75.00| |[Figure 125]<br><br>75.00| |75.00| |75.00| |75.00| |
|---|---|---|---|---|---|---|---|---|---|
|76.50| |78.20| |77.00| |76.00| |75.50| |
|78.00| |80.50| |79.00| |77.50| |76.50| |
|79.50| |82.10| |80.00| |78.50| |77.00| |
|77.50| |79.00| |78.00| |76.50| |75.50| |
| | | | | | | | | | |

100%

ThresholdPercentile

80%

60%

40%

20%

1 2 3 4 5

Max Keep Regions

- 75
- 76
- 77
- 78
- 79
- 80
- 81
- 82

|[Figure 126]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

(b) Accuracy for QWEN2.5-VL-7B on TextVQA

- Figure 7: Impact of mask generation hyperparameters on TextVQA accuracy for QWEN family. Results show performance across varying top-p threshold and maximum keep regions K.

ViCrop

Method Original SAM YOLO CLIP

CARVE rel-att grad-att pure-grad

Accuracy 47.80 49.42 48.84 48.55 55.17 56.06 51.67 58.2 GPU Time 0.17 3.33 0.35 1.07 1.16 0.89 2.36 1.34

- Table 3: Performance comparison of CARVE against external tool-based approaches and ViCrop on TextVQA: accuracy (%) and inference time overhead per sample (seconds).

attains only a 2.93% improvement. Multi-layer fusion outperforms single-layer alternatives by capturing complementary information and providing robustness against individual layer randomness. This phenomenon aligns with our findings in Figure 5(b): early layers perform global scanning with high entropy, while middle-to-deep layers focus on task-relevant patterns.

Sensitivity Analysis of Mask Generation. We examine a 1,000-instance subset randomly sampled from TextVQA. As shown in Figure 7, when p = 1.0, corresponding to no masking intervention, performance remains at original levels. However, when p is set within [0.2,0.6] combined with K ∈ {2,3}, the model achieves optimal performance, as these settings maintain a balance between preserving object representations and suppressing visual noise. In contrast, aggressive masking strategies manifest detrimental effects: retention ratios set to 20% and single-region constraints lead to degradation, since such aggressive configurations discard essential visual information.

Comparative Analysis with Alternative Methods. As shown in Table 3, CARVE substantially outperforms external tool-based approaches: SAM (Kirillov et al., 2023), YOLO (Redmon et al., 2016), CLIP (Radford et al., 2021) and recent ViCrop (Zhang et al., 2025) variants across diverse baseline methodologies (conducted on NVIDIA RTX A6000). External tools rely on generic segmentation algorithms that lack question-image context awareness. While ViCrop effectively reduces visual noise through strategic cropping, it lacks pixel-level noise masking. Regarding computational efficiency, CARVE requires 1.34 seconds of GPU processing time, exceeding simpler approaches such as YOLO (0.35 seconds) but remaining within practical deployment constraints.

- 6 CONCLUSION

In this work, we demonstrate that visual complexity correlates with attention entropy, which in turn negatively impacts VLMs’ performance. Theoretically, we prove that contrasting attention maps between general and specific instructions enables effective decomposition of visual signal into semantic signal and visual noise components. To this end, we propose Contrastive Attention Refinement for Visual Enhancement (CARVE), a training-free method that leverages this theoretical insight to extract task-relevant signal through attention contrasting and pixel-level masking. Our work provides critical insights into the interplay between visual complexity and attention mechanisms, offering an efficient pathway for improving visual reasoning without training.

REFERENCES

Shantanu Acharya, Fei Jia, and Boris Ginsburg. Star attention: Efficient llm inference over long sequences. arXiv preprint arXiv:2411.17116, 2024.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716– 23736, 2022.

Baolong Bi, Shaohan Huang, Yiwei Wang, Tianchi Yang, Zihan Zhang, Haizhen Huang, Lingrui Mei, Junfeng Fang, Zehao Li, Furu Wei, et al. Context-dpo: Aligning language models for context-faithfulness. arXiv preprint arXiv:2412.15280, 2024.

Baolong Bi, Shenghua Liu, Lingrui Mei, Yiwei Wang, Junfeng Fang, Pengliang Ji, and Xueqi Cheng. Decoding by contrasting knowledge: Enhancing large language model confidence on edited facts. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 17198–17208, Vienna, Austria, July 2025a. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.841. URL https://aclanthology.org/2025.acl-long.841/.

Baolong Bi, Shenghua Liu, Xingzhang Ren, Dayiheng Liu, Junyang Lin, Yiwei Wang, Lingrui Mei, Junfeng Fang, Jiafeng Guo, and Xueqi Cheng. Refinex: Learning to refine pre-training data at scale from expert-guided programs. arXiv preprint arXiv:2507.03253, 2025b.

John Canny. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, 8(6):679–698, 1986.

Lizhe Chen, Yan Hu, Yu Zhang, Yuyao Ge, Haoyu Zhang, and Xingquan Cai. Frequency-importance gaussian splatting for real-time lightweight radiance field rendering. Multimedia Tools and Applications, 83(35):83377–83401, 2024a.

Lizhe Chen, Binjia Zhou, Yuyao Ge, Jiayi Chen, and Shiguang Ni. Pis: Linking importance sampling and attention mechanisms for efficient prompt compression. arXiv preprint arXiv:2504.16574, 2025.

Qizhou Chen, Taolin Zhang, Xiaofeng He, Dongyang Li, Chengyu Wang, Longtao Huang, and Hui Xue’. Lifelong knowledge editing for LLMs with retrieval-augmented continuous prompt learning. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 13565–13580, Miami, Florida, USA, November 2024b. Association for Computational Linguistics. doi: 10.18653/ v1/2024.emnlp-main.751. URL https://aclanthology.org/2024.emnlp-main. 751/.

Zenghao Duan, Wenbin Duan, Zhiyi Yin, Yinghan Shen, Shaoling Jing, Jie Zhang, Huawei Shen, and Xueqi Cheng. Related knowledge perturbation matters: Rethinking multiple pieces of knowledge editing in same-subject. arXiv preprint arXiv:2502.06868, 2025.

Honghao Fu, Yufei Wang, Wenhan Yang, Alex C Kot, and Bihan Wen. Dp-iqa: Utilizing diffusion prior for blind image quality assessment in the wild. arXiv preprint arXiv:2405.19996, 2024.

Honghao Fu, Junlong Ren, Qi Chai, Deheng Ye, Yujun Cai, and Hao Wang. Vistawise: Building cost-effective agent with cross-modal knowledge graph for minecraft. arXiv preprint arXiv:2508.18722, 2025.

Haonan Ge, Yiwei Wang, Ming-Hsuan Yang, and Yujun Cai. Mrfd: Multi-region fusion decoding with self-consistency for mitigating hallucinations in lvlms, 2025a. URL https://arxiv. org/abs/2508.10264.

Yuyao Ge, Zhongguo Yang, Lizhe Chen, Yiming Wang, and Chengyang Li. Attack based on data: a novel perspective to attack sensitive points directly. Cybersecurity, 6(1):43, 2023.

Yuyao Ge, Shenghua Liu, Baolong Bi, Yiwei Wang, Lingrui Mei, Wenjie Feng, Lizhe Chen, and Xueqi Cheng. Can graph descriptive order affect solving graph problems with llms? ACL 2025, pp. 6404–6420, 2025b.

Yuyao Ge, Shenghua Liu, Yiwei Wang, Lingrui Mei, Lizhe Chen, Baolong Bi, and Xueqi Cheng. Innate reasoning is not enough: In-context learning enhances reasoning large language models with less overthinking. arXiv preprint arXiv:2503.19602, 2025c.

Yan Hu, Lizhe Chen, Hanna Xie, Yuyao Ge, Shun Zhou, and Xingquan Cai. Real-time nonphotorealistic rendering method for black and white comic style in games and animation. Journal of System Simulation, 36(7):1699–1712, 2024.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pp. 4904–4916. PMLR, 2021.

Chaoya Jiang, Haiyang Xu, Mengfan Dong, Jiaxing Chen, Wei Ye, Ming Yan, Qinghao Ye, Ji Zhang, Fei Huang, and Shikun Zhang. Hallucination augmented contrastive learning for multimodal large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 27036–27046, 2024.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015–4026, 2023.

Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, and SeYoung Yun. Distillm-2: A contrastive approach boosts the distillation of llms. arXiv preprint arXiv:2503.07067, 2025.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PMLR, 2023a.

Tianhao Li, Jingyu Lu, Chuangxin Chu, Tianyu Zeng, Yujia Zheng, Mei Li, Haotian Huang, Bin Wu, Zuoxian Liu, Kai Ma, et al. Scisafeeval: a comprehensive benchmark for safety alignment of large language models in scientific tasks. AAAI 2025 AI for Cybersecurity, 2024.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023b.

Zhecheng Li, Guoxian Song, Yujun Cai, Zhen Xiong, Junsong Yuan, and Yiwei Wang. Texture or semantics? vision-language models get lost in font recognition. In Conference on Language Modeling COLM, 2025., 2025.

Chang Liu, Hongkai Chen, Yujun Cai, Hang Wu, Qingwen Ye, Ming-Hsuan Yang, and Yiwei Wang. Structured attention matters to multimodal llms in document understanding. arXiv preprint arXiv:2506.21600, 2025a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023a.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26296–26306, June 2024.

Kai Liu, Zhan Su, Peijie Dong, Fengran Mo, Jianfei Gao, ShaoTing Zhang, and Kai Chen. Smooth reading: Bridging the gap of recurrent llm to self-attention llm on long-context tasks. arXiv preprint arXiv:2507.19353, 2025b.

Yixin Liu, Kejian Shi, Katherine S He, Longtian Ye, Alexander R Fabbri, Pengfei Liu, Dragomir Radev, and Arman Cohan. On learning to summarize with large language models as references. arXiv preprint arXiv:2305.14239, 2023b.

Feipeng Ma, Yizhou Zhou, Zheyu Zhang, Shilin Yan, Hebei Li, Zilong He, Siying Wu, Fengyun Rao, Yueyi Zhang, and Xiaoyan Sun. Ee-mllm: A data-efficient and compute-efficient multimodal large language model. arXiv preprint arXiv:2408.11795, 2024a.

Weizhi Ma, Yujia Zheng, Tianhao Li, Zhengping Li, Ying Li, and Lijun Wang. A comprehensive review of deep learning in eeg-based emotion recognition: classifications, trends, and practical implications. PeerJ Computer Science, 10:e2065, 2024b.

Weizhi Ma, Ying Li, Tianhao Li, Haowei Yang, Zhengping Li, Lijun Wang, and Junyu Xuan. Sfswts: A spatial-frequency shifted windows and time self-attention network for eeg emotion recognition. Neurocomputing, pp. 130309, 2025.

Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, and Xueqi Cheng. Slang: New concept comprehension of large language models. arXiv preprint arXiv:2401.12585, 2024a.

Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, Jiayi Mao, and Xueqi Cheng. ” not aligned” is not” malicious”: Being careful about hallucinations of large language models’ jailbreak. arXiv preprint arXiv:2406.11668, 2024b.

Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, Ruibin Yuan, and Xueqi Cheng. Hiddenguard: Fine-grained safe generation with specialized representation router, 2024c. URL https://arxiv.org/abs/2410.02684.

Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, Yuyao Ge, Jun Wan, Yurong Wu, and Xueqi Cheng. a1: Steep test-time scaling law via environment augmented generation. arXiv preprint arXiv:2504.14597, 2025a.

Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, et al. A survey of context engineering for large language models. arXiv preprint arXiv:2507.13334, 2025b.

Shiyu Ni, Keping Bi, Jiafeng Guo, and Xueqi Cheng. When do llms need retrieval augmentation? mitigating llms’ overconfidence helps retrieval augmentation. arXiv preprint arXiv:2402.11457, 2024a.

Shiyu Ni, Keping Bi, Lulu Yu, and Jiafeng Guo. Are large language models more honest in their probabilistic or verbalized confidence? In China Conference on Information Retrieval, pp. 124–

135. Springer, 2024b.

Kaihang Pan, Zhaoyu Fan, Juncheng Li, Qifan Yu, Hao Fei, Siliang Tang, Richang Hong, Hanwang Zhang, and Qianru Sun. Towards unified multimodal editing with enhanced knowledge collaboration. Advances in Neural Information Processing Systems, 37:110290–110314, 2024.

Qwen. Qwen2.5-vl: A powerful vision-language model for seamless computer interaction. arXiv preprint arXiv:2409.12191, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 779–788, 2016.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VIII, pp. 146–162. Springer, 2022.

Claude E Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.

V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. URL https://arxiv.org/abs/2507.01006.

Anne M Treisman and Garry Gelade. A feature-integration theory of attention. Cognitive psychology, 12(1):97–136, 1980.

Yuanyuan Wei, Xianxian Liu, Yao Mu, Changran Xu, Guoxun Zhang, Tianhao Li, Zida Li, Wu Yuan, Ho-Pui Ho, and Mingkun Xu. From droplets to diagnosis: Ai-driven imaging and system integration in digital nucleic acid amplification testing. Biosensors and Bioelectronics, pp. 117741, 2025.

Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.

Qian Xiong, Yuekai Huang, Ziyou Jiang, Zhiyuan Chang, Yujia Zheng, Tianhao Li, and Mingyang Li. Butterfly effects in toolchains: A comprehensive analysis of failed parameter filling in llm tool-agent systems. arXiv preprint arXiv:2507.15296, 2025.

Jiayu Yao, Shenghua Liu, Yiwei Wang, Lingrui Mei, Baolong Bi, Yuyao Ge, Zhecheng Li, and Xueqi Cheng. Who is in the spotlight: The hidden bias undermining multimodal retrievalaugmented generation. arXiv preprint arXiv:2506.11063, 2025a.

Jiayu Yao, Shenghua Liu, Yiwei Wang, Lingrui Mei, Baolong Bi, Yuyao Ge, Zhecheng Li, and Xueqi Cheng. Who is in the spotlight: The hidden bias undermining multimodal retrievalaugmented generation, 2025b. URL https://arxiv.org/abs/2506.11063.

Songlin Zhai, Yuan Meng, Yuxin Zhang, and Guilin Qi. Parameter-aware contrastive knowledge editing: Tracing and rectifying based on critical transmission paths. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 28189–28200, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1367. URL https://aclanthology. org/2025.acl-long.1367/.

Guangzi Zhang, Lizhe Chen, Yu Zhang, Yan Liu, Yuyao Ge, and Xingquan Cai. Translating words to worlds: zero-shot synthesis of 3d terrain from textual descriptions using large language models. Applied Sciences, 14(8):3257, 2024a.

Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. Mllms know where to look: Training-free perception of small visual details with multimodal llms. arXiv preprint arXiv:2502.17422, 2025.

Shaolei Zhang, Tian Yu, and Yang Feng. Truthx: Alleviating hallucinations by editing large language models in truthful space. arXiv preprint arXiv:2402.17811, 2024b.

Yujia Zheng, Tianhao Li, Haotian Huang, Tianyu Zeng, Jingyu Lu, Chuangxin Chu, Yuekai Huang, Ziyou Jiang, Qian Xiong, Yuyao Ge, et al. Are all prompt components value-neutral? understanding the heterogeneous adversarial robustness of dissected prompt in large language models. arXiv preprint arXiv:2508.01554, 2025.

Rongxin Zhu, Jey Han Lau, and Jianzhong Qi. Factual dialogue summarization via learning from large language models. arXiv preprint arXiv:2406.14709, 2024.

- A DEFINITION AND EXPLANATION

- A.1 DEFINITION

Symbol Definition Description I ∈ RH×W×3 Input image Image with height H and width W H,W Image dimensions Height and width in pixels Q Task-specific question Task-specific question

- G General instruction General instruction

A(l,tQ) ∈ RN

v Question attention map Attention map at layer l, step t A(l,tG) ∈ RN

v General attention map General question attention map Nv Visual tokens Number of visual tokens L Layer range Layer indices T Time range Generation time step tend Final step Final generation step H(·) Shannon entropy Attention distribution entropy

- H Overall entropy Layer-averaged attention entropy Tc(I) Texture complexity Edge density from Canny detection Cc(I) Color complexity Hue diversity measure E(I)∈{0,1}H×W Edge map Binary edge map from Canny ΨRGB→HSV Color transform RGB to HSV transformation operator ζij Hue value Hue value at pixel (i,j) ρb Hue proportion Fraction of pixels in hue bin b

B Hue bins Number of hue bins Fvis(I) ∈ RN

v

+ Visual noise factor Image-inherent visual noise component Fsem(Q,I) ∈ RN

v

+ Semantic signal factor Task-related semantic signal component 1N

v

Uniform vector Vector of ones

- Aˆ ∈ RN

v

+ Estimated attention Estimated semantic attention λ > 0 Regularization Regularization parameter p ∈ (0,1] Top-p percentile Percentile threshold for masking K ∈ N Max regions Maximum regions to preserve wt Temporal weights Later token weighting with wt = t − tstart + 1 S ∈ RH×W Fused map Spatially reshaped attention map Qp(·) Percentile function Top-p percentile operator τ Threshold Computed threshold value M∗ ⊆ {1..H} × {1..W} Final mask Union of top-K regions Rk Connected region Connected component from thresholding Φ(I,M) Visual extraction Masking, cropping and resizing πH×W Spatial reshape Token to image projection Ξ Attention extractor Function to extract attention maps M VLM Vision-language model

Ltotal Total layers Number of model layers Nq Text tokens Number of query tokens

- A.2 EXPLANATION

- • Time Step (t): In the autoregressive generation process of vision-language models, a time step denotes the sequential position index in the output token sequence. The model generates responses token-by-token, where t = 1 corresponds to the first generated token and t = tend represents the final token. At each time step, the model produces an attention

distribution A(l,tQ) ∈ RN

v over visual tokens.

- • Visual Complexity (Tc(I), Cc(I)): Visual complexity quantifies the inherent characteristics of an image that can interfere with VLMs’ attention mechanisms, decomposed into

two orthogonal dimensions. Texture complexity Tc(I) ∈ [0,1] measures the density of edge information using Canny edge detection, where higher values indicate more intricate

patterns, object boundaries, and structural details. Color complexity Cc(I) ∈ [0,1] captures the diversity of hue distribution in HSV color space through Shannon entropy, where higher values reflect greater chromatic variation.

- • Visual Tokens (Nv): Visual tokens constitute the discrete representational units obtained after processing an input image through a visual encoder. An image of dimensions H ×W is partitioned and encoded into Nv visual tokens, which form the fundamental units for visual information processing. The attention mechanism allocates weights across these Nv tokens to determine which image regions to attend to.
- • Semantic Signal Factor (Fsem(Q,I)): The semantic signal factor represents the question-

specific component in the attention decomposition framework, valued in RN

v

+ . This factor quantifies the semantic signal between each visual token and the given question Q. Under general instructions G (e.g., ”describe this image”), this factor approximates a uniform distribution (Fsem(G,I) ≈ 1N

v

), whereas task-specific questions yield elevated values in semantically relevant regions.

- • Visual Noise Factor (Fvis(I)): The visual noise factor captures the image-inherent,

question-independent attention component, valued in RN

+ . This factor, determined by texture complexity and color diversity of the image, reflects the influence of visual content characteristics on attention distribution. Under general instructions, the attention distribu-

v

tion is predominantly governed by this factor: A(l,tG)(I) ≈ Fvis(I).

- B IMPLEMENTATION DETAILS

We conduct our experiments on a server with 4 × NVIDIA RTX A6000 GPUs. τ is set to 0.05. In practical implementation, CARVE requires three inference passes; however, the first two passes (extracting general instruction and task-specific question) can be terminated early. Specifically, when we require attention maps only from layers L = [Lstart,Lend], the first two inference processes can halt upon completing layer Lend computation, eliminating the need for full Ltotal layer forward propagation. The third inference must run completely to generate the final answer.

- C PROOFS AND ADDITIONAL THEOREMS

- C.1 MATHEMATICAL BASIS OF ATTENTION DECOMPOSITION

- Theorem C.1 (Existence of Attention Decomposition): For any attention distribution A(l,tQ)(I) ∈

RN

v

+ , there exists a unique decomposition: A(l,tQ)(I) = Fvis(I) ⊗ Fsem(Q,I) (C.1)

Proof: Define a logarithmic space mapping ϕ : R+ → R where ϕ(x) = log(x). Under this transformation, the decomposition becomes additive in logarithmic space:

ϕ(A(l,tQ)(I)) = ϕ(Fvis(I)) + ϕ(Fsem(Q,I)) (C.2) Given the boundary condition that Fsem(G,I) = 1N

v

when Q = G (general instruction), we obtain: ϕ(Fvis(I)) = ϕ(A(l,tG)(I)) (C.3)

Consequently, through substitution:

ϕ(Fsem(Q,I)) = ϕ(A(l,tQ)(I)) − ϕ(A(l,tG)(I)) (C.4) The unique solution is obtained via the inverse mapping ϕ−1(x) = exp(x). □

C.2 CONVEXITY ANALYSIS OF THE OPTIMIZATION PROBLEM

- Theorem C.2 (Strict Convexity of Objective Function): The optimization objective

Nv

J (A˜) =

i=1

A ˜i · A(iG) − A(iQ)

Nv

2

A˜2i · A(iG) (C.5)

+ λ

i=1

is strictly convex with respect to A˜. Proof: Computing the Hessian matrix reveals its structure. Since the objective function is separable across components A˜i, the Hessian is diagonal with elements:

∂2J ∂A˜2i

= 2(A(iG))2 + 2λA(iG) = 2A(iG)(A(iG) + λ) (C.6)

Hii =

Given that A(iG) > 0 and λ > 0, all diagonal elements are positive, thus H ≻ 0 (positive definite). According to convex optimization theory, a twice continuously differentiable function with positive

definite Hessian everywhere is strictly convex. □

- C.3 DERIVATION AND UNIQUENESS OF CLOSED-FORM SOLUTION

- Theorem C.3 (Closed-form Expression of Optimal Solution): The optimization problem admits a unique global optimum:

Aˆi =

A(iQ) A(iG) + λ

(C.7)

Proof: Applying first-order optimality conditions (KKT conditions):

∇A˜iJ = 2(A˜i · A(iG) − A(iQ)) · A(iG) + 2λA˜i · A(iG) = 0 (C.8) Rearranging terms yields:

A˜i · A(iG) · (A(iG) + λ) = A(iQ) · A(iG) (C.9)

Solving for A˜i:

A˜i =

A(iQ) A(iG) + λ

(C.10)

By Theorem C.2’s strict convexity, this solution represents the unique global optimum. □

C.4 ERROR BOUNDS AND CONVERGENCE ANALYSIS

- Theorem C.4 (Approximation Error Bound): Let Fsem(G,I) = 1N

+ ϵ where ∥ϵ∥∞ ≤ δ. Then the estimation error satisfies:

v

δ · ∥Fsem(Q,I)∥∞ 1 − δ

∥Aˆ − Fsem(Q,I)∥∞ ≤

Proof: Under perturbation A(iG) = Fvis,i · (1 + ϵi), the estimate becomes:

Aˆi = Fvis,i · Fsem,i(Q,I) Fvis,i · (1 + ϵi) + λ ≈

Fsem,i(Q,I) 1 + ϵi

when Fvis,i ≫ λ

Using the Taylor expansion Aˆi = Fsem,i(Q,I) · ∞k=0(−ϵi)k and truncating to first order yields:

|ϵi| 1 − |ϵi| Taking the infinity norm completes the proof. □

|Aˆi − Fsem,i(Q,I)| ≤ Fsem,i(Q,I) ·

- C.5 THEORETICAL SELECTION OF REGULARIZATION PARAMETER

Proposition C.5 (Optimal Regularization Parameter): The optimal regularization parameter λ∗ that minimizes the expected mean squared error satisfies:

E ∥Aˆ(λ) − Fsem(Q,I)∥22 (C.11)

λ∗ = arg min

λ

Proof: From Theorem C.3, the estimator takes the form:

A(iQ) A(iG) + λ

= Fvis,i · Fsem,i Fvis,i + λ

Aˆi(λ) =

(C.12)

The mean squared error decomposes as: MSE(λ) = Bias2(λ) + Variance(λ) (C.13)

where Bias(λ) = E[Aˆ(λ)] − Fsem(Q,I) and Variance(λ) = E[(Aˆ(λ) − E[Aˆ(λ)])2]. For the bias term, assuming E[Fvis,i] = µi:

λ · Fsem,i µi + λ

Biasi(λ) = E Fvis,i · Fsem,i

Fvis,i + λ − Fsem,i ≈ −

Thus |Biasi(λ)| = O(λ) as λ → 0. For the variance term, let Fvis,i = µi + ϵi with Var(ϵi) = σi2. Taylor expansion yields:

Fsem2 ,iµ2iσi2 (µi + λ)4

Var(Aˆi(λ)) ≈

Therefore Var(Aˆi(λ)) = O(1/λ2) as λ → 0. The component-wise MSE becomes:

λ2 · Fsem2 ,i (µi + λ)2

+ Fsem2 ,iµ2iσi2 (µi + λ)4

MSEi(λ) =

Setting dMSEi

dλ = 0 and solving yields:

σi2 µi

λ∗i = µi 1 + 2σi2/µ2i − 1 ≈

(C.14)

(C.15)

(C.16)

(C.17)

for small noise-to-signal ratio. The global optimum requires minimizing Ni=1v MSEi(λ). □ Corollary C.5.1 (Numerical Stability): For any λ > 0, the condition number of the regularized problem satisfies:

maxi A(iG) + λ λ

maxi(A(iG) + λ) mini(A(iG) + λ)

(C.18)

≤

κ(λ) =

Proof: The bound follows directly from the definition of condition number and the positivity of A(iG) and λ. The regularization ensures κ(λ) < ∞, guaranteeing numerical stability. □

Remark: The regularization parameter λ serves dual purposes: controlling the bias-variance tradeoff and ensuring numerical stability. As λ → 0, the estimator becomes unbiased but exhibits high

variance and potential numerical instability when A(iG) ≈ 0. Conversely, as λ → ∞, the estimator becomes increasingly biased toward zero but achieves maximum stability. The optimal choice λ∗ ∝

σ2/µ balances these competing objectives, where σ2 represents the noise variance and µ the signal mean. In practice, cross-validation on a held-out set provides robust estimation of λ∗.

- C.6 HIERARCHICAL EVOLUTION OF ATTENTION ENTROPY

Theorem C.6 (Monotonicity of Entropy): For a layer sequence l1 < l2 < ... < ln, attention entropy satisfies:

H(A(lQ)

1,t) ≥ H(A(lQ)

2,t) ≥ ... ≥ H(A(lQ)

n,t) (C.19)

Proof: Applying the Data Processing Inequality, we treat each layer as an information processing channel. Since deeper networks progressively extract high-level features and focus on task-relevant regions, information entropy decreases monotonically. This aligns with the principle of maximum entropy: systems tend toward maximum entropy states under constraints, where deeper layers impose stronger task constraints. □

- C.7 COMPUTATIONAL OPTIMIZATION POTENTIAL OF CARVE

This section analyzes the computational optimization potential of the CARVE algorithm. While CARVE requires three inference passes, its structural properties enable significant optimization opportunities.

The key observation is that the first two inference passes (general instruction and task-specific question) only require extracting attention maps from intermediate layers, without completing full forward propagation or generating complete responses. This characteristic enables early termination strategies. Furthermore, the general attention maps A(G) depend solely on the input image and are independent of specific questions, creating opportunities for caching and reuse.

v at layer l have computational cost cl = Θ(Nv2). The baseline complexity without optimization is:

Let the forward propagation P : RN

→ RN

v

### Cbaseline = 3Ltotal · Θ(Nv2) + Θ(|L| · |T | · Nv)

Early Termination Strategy. Since only attention maps from layers L = [Lstart,Lend] are required, the first two inference passes can terminate after layer Lend:

Cearly = (2Lend + Ltotal) · Θ(Nv2) + Θ(|L| · |T | · Nv) The relative computational savings rate is:

η1 = Cbaseline − Cearly Cbaseline

2(Ltotal − Lend) 3Ltotal

2(1 − α) 3

=

=

where α = Lend/Ltotal. For practical configurations with L = [20,25] and Ltotal = 28, we have α = 25/28 ≈ 0.89, yielding theoretical savings of η1 ≈ 7.3%.

Attention Caching Mechanism. The general attention maps A(G) depend only on the image I and can be reused across multiple questions. Define a cache mapping H : I → {A(lG)}l∈L. For n different questions {Q1,...,Qn} on the same image, the total computational cost is:

Ccached(n) = Lend · Θ(Nv2) + n · (Lend + Ltotal) · Θ(Nv2) compared to 3n · Ltotal · Θ(Nv2) for the baseline approach. The average cost per question becomes:

Lend n · Θ(Nv2) + (Lend + Ltotal) · Θ(Nv2)

C¯cached =

As n → ∞, the average cost approaches (Lend + Ltotal) · Θ(Nv2), yielding a speedup ratio relative to baseline:

3Ltotal Lend + Ltotal

3 1 + α

=

Scache =

For α = 0.89, this gives Scache ≈ 1.59, representing approximately 37% computational savings.

Combined Optimization Analysis. When processing batches containing repeated images, combining both strategies yields:

Ccombined = (1 − ρ)Lend · Θ(Nv2) + (Lend + Ltotal) · Θ(Nv2) where ρ ∈ [0,1] denotes the cache hit rate. The relative speedup becomes:

3Ltotal (2 − ρ)Lend + Ltotal

3 (2 − ρ)α + 1

Scombined =

=

Under practical scenarios with α = 0.89 and ρ = 0.3, we obtain Scombined ≈ 1.24, corresponding to approximately 19% computational savings.

The space complexity remains S(CARVE) = Θ(|L| · |T | · Nv). For typical configurations (|L| =

- 5, |T | = 10, Nv = 1024), this requires approximately 200KB of additional memory, which is negligible on modern hardware.

- D PROMPT DESIGN

General Instruction Accuracy (%) Std Dev (%) Relative Gain (%)

w/o CARVE 72.4 0.8 – “Write a general description of the image.” 77.2 0.6 +6.63 “Describe this image in detail.” 75.8 0.9 +4.70 “Provide a comprehensive overview of the image.” 75.2 1.4 +3.87 “What do you see in this image?” 74.9 1.2 +3.45 “Explain what appears in the image.” 74.8 1.7 +3.31

Table 4: Comparison across general instructions.

To identify the optimal general instruction for inducing uniform attention distributions, we conducted experiments on a randomly sampled subset of 1000 instances from the TextVQA dataset using the QWEN2.5-VL-3B. Our objective was to identify prompts that encourage global image scanning without focusing on specific semantic regions. To assess stability, we performed ten independent trials and computed standard deviations across runs. To avoid discrepancies arising from layer and time step variations, we conduct experiments using Tfull and L = [20,25] as hyperparameters. As shown in Table 4, “Write a general description of the image” achieves both the highest accuracy (77.2%) and the lowest standard deviation (0.6%), indicating superior stability. Meanwhile, “What do you see in this image?” and “Explain what appears in the image.” are excluded due to their poor stability. Beyond considering accuracy and stability, we also need to consider the number of tokens generated by the VLM. Specifically, “Describe this image in detail.” and “Provide a comprehensive overview of the image.” are excluded because they output significantly more tokens than “Write a general description of the image.”. Based on the above considerations, we ultimately adopt “Write a general description of the image.” as the general instruction for CARVE.

- E DATASETS

For A-OKVQA (Schwenk et al., 2022), we utilize the validation split containing 1,145 questions across 1,122 images that require integrating visual perception with commonsense reasoning, evaluated using VQA-score accuracy. For POPE (Li et al., 2023b), we employ 500 distinct images paired with 9,000 binary questions systematically designed to detect hallucination phenomena through polling-based object probing. For V∗ (Wu & Xie, 2023), we evaluate on 191 image-question pairs that demand fine-grained visual reasoning capabilities. For TextVQA (Singh et al., 2019), we test on 3,166 images with 5,000 questions focusing on text comprehension abilities.

For TextVQA evaluation, we adopt the protocol established by Zhang et al. (2025), deliberately excluding OCR-extracted tokens from model inputs. We treat TextVQA identically to other visual reasoning benchmarks, providing only the image and question without auxiliary text annotations. While this configuration yields marginally reduced accuracy compared to OCR-augmented baselines in original implementations, it enables unbiased assessment of models’ intrinsic visual text

recognition capabilities, eliminating confounding factors from external OCR systems. This evaluation strategy ensures that performance metrics genuinely reflect the visual perception and text understanding abilities inherent to the vision-language models.

- F VISUALIZATIONS

This section presents visual analysis of masked images generated by CARVE across different threshold values τ from 1.0 (no masking) to 0.1 (aggressive masking). Figures 8 and 9 show two representative TextVQA samples where visual complexity initially causes incorrect predictions.

Figure 8 shows a street scene where the model fails to detect the Bridgestone sign at τ = 1.0. Progressive masking removes background buildings and vehicles, enabling correct recognition at τ = 0.3. In Figure 9, multiple decorative mugs cause shape misidentification through the cup handle. At τ = 0.2, only the relevant mug remains, yielding the correct “star” answer. Across both samples, optimal performance occurs within τ ∈ [0.2,0.4], where contrastive attention effectively preserves semantic signal while eliminating visual distractors.

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

(a) τ = 1.0 (b) τ = 0.9 (c) τ = 0.8 (d) τ = 0.7 (e) τ = 0.6

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

(f) τ = 0.5 (g) τ = 0.4 (h) τ = 0.3 (i) τ = 0.2 (j) τ = 0.1

- Figure 8: Images masked with CARVE. The caption of each subfigure shows the computed threshold value τ.

[Figure 137]

(a) τ = 1.0 (b) τ = 0.9 (c) τ = 0.8 (d) τ = 0.7 (e) τ = 0.6

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

(f) τ = 0.5 (g) τ = 0.4 (h) τ = 0.3 (i) τ = 0.2

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

(j) τ = 0.1

- Figure 9: Images masked with CARVE. The caption of each subfigure shows the computed threshold value τ.

