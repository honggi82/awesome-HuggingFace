### DRAGFLOW: UNLEASHING DIT PRIORS WITH REGION BASED SUPERVISION FOR DRAG EDITING

###### Zihan Zhou1,∗, Shilin Lu1,∗, Shuli Leng1, Shaocong Zhang1, Zhuming Lian1, Xinlei Yu2, Adams Wai-Kin Kong1 1Nanyang Technological University, 2National University of Singapore

{zihan010, shilin002, nie25.ls3409, zhan0711, zhuming001}@e.ntu.edu.sg xinlei.yu@u.nus.edu adamskong@ntu.edu.sg

Point Input DragLoRA GoodDrag Region Input RegionDrag OURS

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

## arXiv:2510.02253v3[cs.CV]1Mar2026

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

[Figure 18]

Point-based Methods Region-based Methods

Figure 1: Comparison of drag-editing results between baselines and our method, DragFlow. DragFlow successfully unleashes FLUX’s stronger generative prior, removing the distortions that previous methods produced on challenging scenarios.

ABSTRACT

Drag-based image editing has long suffered from distortions in the target region, largely because the priors of earlier base models, Stable Diffusion, are insufficient to project optimized latents back onto the natural image manifold. With the shift from UNet-based DDPMs to more scalable DiT with flow matching (e.g., SD3.5, FLUX), generative priors have become significantly stronger, enabling advances across diverse editing tasks. However, drag-based editing has yet to benefit from these stronger priors. This work introduces DragFlow, the first framework to effectively harness FLUX’s rich prior via region-based supervision, enabling full use of its finer-grained, spatially precise features for drag-based editing and achieving substantial improvements over existing baselines. We first show that directly applying point-based drag editing to DiTs performs poorly: unlike the highly compressed features of UNets, DiT features are insufficiently structured to provide reliable guidance for point-wise motion supervision. To overcome this limitation, DragFlow introduces a region-based editing paradigm, where affine transformations enable richer and more consistent feature supervision. Additionally, we integrate pretrained open-domain personalization adapters (e.g., IP-Adapter) to enhance subject consistency, while preserving background fidelity through gradient mask-based hard constraints. Multimodal large language models (MLLMs) are further employed to resolve task ambiguities. For evaluation, we curate a novel Region-based Dragging benchmark (ReD Bench) featuring region-level dragging

∗Equal Contribution.

instructions. Extensive experiments on DragBench-DR and ReD Bench show that DragFlow surpasses both point-based and region-based baselines, setting a new state-of-the-art in drag-based image editing. Code and dataset are available at https://github.com/Edennnnnnnnnn/DragFlow.

- 1 INTRODUCTION

Text-driven image editing (Labs et al., 2025) has made impressive progress, but natural language often underspecifies geometry and locality, leading to unintended changes. Drag-based image editing (Pan et al., 2023; Jiang et al., 2025; Xia et al., 2025) bridges this gap by enabling users to specify finer-grained, spatially localized motions through interactive drag instructions, yielding more controllable edits. Despite their success, however, these methods often introduce unnatural deformations and distortions, especially in images with intricate details or complex structures.

We attribute this limitation to the insufficient generative prior of Stable Diffusion (SD) (Rombach et al., 2022a), the predominant base model, which struggles to constrain optimized latents back onto the natural image manifold. Past findings align with this view: applying nearly identical loss functions for drag editing yields far fewer unnatural distortions when using GAN-based priors compared to SD (Shi et al., 2024b). In parallel, recent advances in generative modeling have shifted from UNet-based DDPMs to more scalable Diffusion Transformers (DiTs) (Peebles & Xie, 2023) trained with flow matching (Lipman et al., 2022) (e.g., SD 3.5 (Esser et al., 2024a), FLUX.1-dev (Black Forest Labs, 2024)), yielding substantially stronger priors that have propelled progress across various editing tasks (Lu et al., 2025; Yan et al., 2025; Wei et al., 2025; Deng et al., 2024; Wang et al., 2024; Rout et al., 2024). Yet, drag-based editing has not capitalized on these enhanced priors.

In this work, we pioneer the exploration of leveraging a stronger generative prior for drag editing. We first observe that directly applying previous drag editing methods to DiTs yields suboptimal results. Through a detailed analysis of features extracted from U-Nets and DiTs, we identify two core obstacles. First, point-based objectives used by prior drag methods mismatch DiT representations. U-Net bottlenecks produce spatially compact, highly compressed features that aggregate high-level semantics over broad receptive fields; supervising a single feature-map location therefore carries strong semantic evidence. DiTs, in contrast, yield finer-grained, spatially precise features with narrower receptive fields. Directly applying point-wise motion or tracking losses to DiTs provides weak semantic supervision and degrades editing effectiveness. Second, modern DiT models like FLUX are classifier-free-guidance (CFG)–distilled, which exacerbates inversion drift. As a result, standard key-value (KV) injection is insufficient to preserve subject identity consistency during drag edits.

To harness the potent priors of DiT-based models for drag-based editing, we introduce DragFlow, a novel region-based editing framework. DragFlow departs from point supervision and rethinks inversion and background handling to align with DiT feature geometry and the realities of CFG-distilled models. DragFlow advances the state of the art through three key innovations: (i) region-level motion supervision, which delivers richer and more consistent feature guidance via affine transformations; (ii) a replacement of traditional background consistency losses with hard constraints that preserve the background while updating only the editable region; and (iii) adapter-enhanced inversion, which injects subject representations from a pretrained open-domain personalization adapter (e.g., IP-Adapter (Ye et al., 2023)) into the base model’s prior, achieving markedly superior subject fidelity under edits. Together, these components make drag-based editing practical with DiT backbones: they harness the stronger generative prior without sacrificing controllability, reducing deformation artifacts and improving faithfulness on complex, detail-rich images.

For evaluation, we introduce the Region-based Dragging Benchmark (ReD Bench). Each sample in ReD Bench is equipped with point-to-region alignment, explicit task tags spanning relocation, deformation, and rotation, and contextual descriptions that clarify user intent. We validate DragFlow extensively on both DragBench-DR (Lu et al., 2024a) and ReD Bench. Results demonstrate that DragFlow consistently outperforms state-of-the-art (SOTA) baselines.

- 2 RELATED WORK

Recent advances in diffusion models have led to a surge of interactive image-editing techniques, enabling users to intuitively reposition or deform specific regions of an image through drag-based interactions. Existing methods for drag-based editing can be broadly grouped into three categories.

(i) Optimization-based methods (Xia et al., 2025; Ling et al., 2024; Liu et al., 2024; Zhang et al.,

- 2024c; Jiang et al., 2025; Shi et al., 2024b; Karras et al., 2022; Mou et al., 2023; 2024; Lin et al.,
- 2025; Hou et al., 2024; Cui et al., 2024; Luo et al., 2024; Choi et al., 2024), which is the most prevalent category, iteratively refine inverted noisy latents during inference. These techniques are predominantly point-based: they accept point-wise drag instructions as input and employ motion supervision and point tracking, both of which operate at the point level. However, they often yield unnatural deformations or distortions in the edited images, primarily because the optimized latents deviate from the natural image manifold learned by the base model, residing in out-of-distribution regions. Thus, many studies have focused on more judicious optimization strategies to ensure that the resulting latents can be more readily mapped back to plausible natural images. (ii) Finetuningbased methods (Shin et al., 2024; Shi et al., 2024a), which adapt a base text-to-image (T2I) diffusion model using curated video datasets. Yet, the inherent mismatch between video data and the precise instructions required for drag editing, coupled with the scarcity of high-quality, large-scale training data, limits their generalization. These methods usually fail to achieve complete drag effects and are prone to distortions. (iii) Methods that avoid both finetuning and optimization (e.g., RegionDrag (Lu et al., 2024a) and FastDrag (Zhao et al., 2024d)), instead directly copying and pasting noisy latent patches to target locations computed via predefined mapping functions during inference. While this approach significantly enhances efficiency, it heavily relies on handcrafted priors for the mapping functions, often resulting in edited images that lack faithfulness and realism.

Our method falls within the optimization-based paradigm but innovates by replacing point-based motion supervision with region-level supervision, thereby enabling drag capabilities in DiTs. Like RegionDrag, our approach accepts region-based inputs; however, whereas RegionDrag requires users to manually predefine the target region mask (a challenging task in non-rigid scenarios), we only necessitate specifying a target point serving as the region’s center. Moreover, RegionDrag performs point-wise copy-pasting within noisy latents, which, due to the handcrafted nature of its mappings, often fails to preserve internal region structures. In contrast, we treat the region as a cohesive unit, extracting holistic regional features to serve as supervision signals during latent optimization, thereby ensuring the integrity of internal structures.

- 3 METHODOLOGY

###### Sources

###### UNet Feature Maps

###### DiT Feature Maps

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

In this section, we begin by elucidating the inherent limitations of previous point-based drag editing frameworks when adapted to DiTs (Sec. 3.1). Building on this analysis, we introduce a region-level affine supervision strategy (Sec. 3.2). To further enhance fidelity, we incorporate hard-constrained background preservation (Sec. 3.3) and adapterenhanced subject consistency mechanisms (Sec. 3.4), addressing inversion drifts in DiTs.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Double

Double

Mid

Down-samp.

Up-samp. block

17th block

18th block

block

block

Figure 2: Comparison of feature maps extracted from UNet and DiT at the same denoising step. UNet produces spatially compact, highly compressed features that capture high-level semantic information, whereas DiT generates finer-grained, spatially precise representations.

- 3.1 WHY POINT-BASED DRAG FAILS ON DIT

To harness the robust prior of FLUX for drag-based image editing, we initially applied established point-based drag editing frameworks, including image inversion, motion supervision, point tracking, and key-value injection, directly to FLUX. Surprisingly, as shown in Fig. 6, this straightforward adaptation offers only limited improvements compared to its counterpart in SD.

IP Embed

IP Embed

IP Adapter

[Figure 37]

VAEDecoder

[Figure 38]

###### VAEEncoder

🔥

z0 z1 ... zt z0'

zt-1 ...

DiT DiT

KV Cache

KV Cache

[Figure 39]

Dragging Procedure

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

M(0)

###### Module Inputs:

❄

[Figure 45]

Zt(0)

- - 1 [IMG] source image
- - 2 [IMG] operation image

IP

[Figure 46]

|[Figure 47]|
|---|

[Figure 48]

Feat(Zt(0))

###### Loss

###### DiT

[Figure 49]

Module Outputs:

[Figure 50]

M(k)

|[Figure 51]|
|---|

- - 1 [TXT] possible intents
- - 2 [CLASS] operation types Gradient Backward

🔥

MLLM

KV

[Figure 52]

Zt(k)

RELOCATION

Feat(Zt(k))

Gradient Mask B

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

|[Figure 57]|
|---|

###### B ∇ Loss

|ROTATION: Rotate to close the beck|
|---|

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

M(0) M(k=20) M(k=40) M(K)

ROTATION

|RELOCATION: Raise the lower beck|
|---|

|DEFORMATION: Extend the beck to left|
|---|

DEFORMATION

Subtasks

Progressive Affine Control

Figure 3: Overview of the DragFlow framework. The original image is inverted into a noisy latent space and iteratively optimized under the proposed region-level affine supervision. Subject consistency is reinforced through key-value (KV) injection and our adapter-enhanced inversion, while background fidelity is maintained via gradient mask-based hard constraints. In addition, a multimodal large language model (MLLM) is employed to better interpret and clarify user intents.

We attribute this performance gap to fundamental differences in the feature granularity extracted by the DiT and UNet, which significantly impact the effectiveness of point-wise motion supervision and tracking methods. As illustrated in Fig. 2, the UNet architecture, due to its bottleneck design, produces spatially compact and highly compressed features that encapsulate high-level semantic representations. In contrast, DiT generates finer-grained, spatially precise features. Consequently, in UNet, each point on the feature map aggregates semantic information from a broad receptive field in the input image, whereas in DiT, each point corresponds to a narrower region.

This difference directly impacts point-based methods, which rely on computing losses for motion supervision and point tracking using individual points on the feature map. In UNet, the broader receptive field of each feature point provides rich semantic context, enabling effective motion supervision and tracking. However, in DiT, the finer-grained features, with their narrower receptive fields, capture less semantic information per point, undermining the efficacy of these point-based methods when applied directly to DiT.

- 3.2 REGION-LEVEL AFFINE SUPERVISION

To leverage the powerful prior of FLUX for drag-based image editing, we introduce DragFlow, a region-based framework.

User Input Specification. In our framework, the user designates source region masks {Mi}Ni=1, each paired with a corresponding target point ti = (xi,yi). The target point serves as the centroid of the target region. The expected target region mask can be obtained via an affine transformation (see Appendix C.1 for details). As illustrated in Fig. 3, we harness the capabilities of a multimodal large language model (MLLM), GPT-5 (OpenAI, 2025), to infer users’ underlying intentions and thereby facilitate drag-based editing. The model receives as input the original image together with userprovided drag instructions (i.e., source region masks and target points). We then prompt the MLLM with carefully designed in-context examples, which guide it to produce two outputs: (i) a class label indicating the type of editing operation, and (ii) a textual description articulating the inferred editing intent (see Appendix D.3 for details on the prompting strategy). The class label determines which affine transformation is applied, while the textual description serves as a natural-language prompt for the generative model during the drag-editing process.

Iterative Latent Optimization. Given an input image x, it is first encoded by the VAE to produce the latent z, which is then inverted to obtain the noisy latent zt where t ∈ [0,T]. We optimize zt over k iterations where k ∈ [0,K], denoted as zt(k), such that subsequent denoising of zt(k) produces

an output image that fulfills the user-specified drag operations. This optimization is guided by the following loss function:

LDrag =

N

γi · Mi(k) ⊙ F zt(k) − sg Mi(0) ⊙ F zt(0)

i=1

N

, where

γi = 1. (1)

1

i=1

Here, zt(0) ≜ zt is the initial unoptimized latent, while zt(k) represents the latent after k optimization iterations. The function F(·) extracts features from DiT, with the specific feature layers detailed

in Appendix C.5. The operator sg[·] denotes stop-gradient. The weighting coefficient γi balances multiple drag operations within the same image, determined adaptively according to the relative size of the corresponding manipulated regions (see Appendix D.1 for the formulation in details). Finally,

Mi(0) ≜ Mi is the user-provided mask for the source region, and Mi(k) specifies the corresponding target region, where we enforce similarity to the features of the source region.

Affine Transformation for Mask Propagation. The target mask Mi(k) is derived from the source mask Mi(0) via an affine transformation:

 

k K

(ti − bi), (Relocation & Deformation)

Mi(k) = Ω Mi(0), ξi(k) , ξi(k) =

(2)

k K

∠biaiti, ai , (Rotation)



where Ω applies the affine transformation (affine computation detailed in Appendix C) to Mi(0) with parameters governed by ξi(k). Different drag types influence the affine matrix parameters distinctly: for relocation and deformation, these are determined by the vector from the target point ti to the centroid bi of the source region, where bi = (1/|Mi(0)|) q∈M(0)

q; for rotation, they are governed

i

by the angle ∠biaiti formed by ti, bi, and the user-specified anchor point ai. The linear schedule k/K moves the mask smoothly from the source configuration toward the target over K iterations.

Why Region-level Supervision. This formulation causes the target mask Mi(k) to translate (or rotate) progressively from the source centroid toward the target point as k increases. Intuitively, it

transports the object’s features from the source region to the destination step by step. Although this echoes the high-level idea of point-based dragging, there are two crucial differences:

Feature granularity. Point-based methods compare features only at a handle point, either against its previous location or against the original source point. In contrast, we match features between entire source and target regions. Region-level supervision provides richer semantic context and mitigates myopic gradients, which leads to more effective latent updates on FLUX.

Tracking requirement. Point-based approaches require handle point tracking to keep the extracted features aligned with the moving content. Without tracking, naively advancing the handle along a straight line is brittle: even a slight deviation of the optimized content from that line causes subsequent local features to mismatch the intended structure, causing error accumulation and eventual drag failure. Our region-level supervision avoids this failure mode. Because we compare features over regions rather than at a single point, we do not need to pinpoint a feature extraction location on the object at each step. This makes the procedure substantially more stable and robust, eliminating the need for explicit tracking; it suffices to shift the source region mask along the path from the source centroid to the target point.

- 3.3 BACKGROUND PRESERVATION Prior work typically enforces fidelity in non-editable regions via an auxiliary consistency loss:

LBG = zt(−k)1 − sg[zt(0)−1] ⊙ (1 − B)

, (3)

1

where B denotes the mask that specifies the editable region. In practice, this term competes with the feature-matching objective, making performance highly sensitive to its weight. The issue is exacerbated in FLUX, a CFG–distilled model that exhibits larger image-inversion drift than non-distilled

counterparts. Compounding this, the consistency loss is evaluated against the inverted latent zt(0)−1,

which is treated as ground truth; when the inversion is biased, this target is unreliable, and the loss misguides optimization rather than helping it.

Instead of balancing competing losses, we hard constrain the background and update only the editable region:

∂LDrag ∂zt(k)

###### + (1 − B) ⊙ ztorig, (4)

zt(k+1) = B ⊙ zt(k) − α ·

where B denotes the mask that specifies the editable region (see Appendix D.2 for extraction details), and ztorig is obtained from a pure reconstruction path. Implementationally, this requires an additional reconstruction branch that starts from the inverted latent zt; the overhead is modest and, critically, it yields substantially better background preservation in challenging FLUX settings.

- 3.4 SUBJECT CONSISTENCY ENHANCEMENT Table 1: Inversion Performance (3,000 images).

While our proposed region-level affine supervision enables drag-based editing using the prior of FLUX, it still suffers from subject inconsistency between the source and edited images. A natural remedy is the KV injection, which is widely used when SD serves as the base model. In FLUX, however, KV injection underperforms, as shown in Fig. 4 (left). We attribute this gap to FLUX being a CFG-distilled model, which exhibits more pronounced inversion drift compared to non-distilled counterparts, as evidenced in Tab. 1.

Method LPIPS ↓ SSIM ↑ PSNR ↑

DPM-Solver (Lu et al., 2022) Inv. (SD) 0.167 0.799 26.31 Fireflow Inv. w/o adapter (FLUX) 0.283 0.703 20.43 Fireflow Inv. w/ adapter (FLUX) 0.173 0.784 25.87

To address this, we introduce adapter-enhanced inversion for DiT-based models. Specifically, pretrained open-domain personalization adapters (e.g., IP-Adapter (Ye et al., 2023), PuLID (Guo et al., 2024), and InstantCharacter (Tao et al., 2025)) are trained to extract a subject’s representation from a reference image, enabling its seamless integration into a T2I base model for generation across varied contexts. Leveraging this insight, we propose employing such adapters as auxiliary subject representation extractors. Without any additional fine-tuning, we inject the adapter’s subject representation into the model prior, which substantially improves inversion quality (Tab. 1) and yields visibly better subject consistency under edits (Fig. 4 (right)).

- 4 EXPERIMENTS

- 4.1 IMPLEMENTATION DETAILS

Sources w/ KV w/ KV & IP-Adap.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Figure 4: Visualization of the effect of adapterenhanced inversion on subject consistency, compared with KV injection alone.

We implement DragFlow using FLUX.1dev (Black Forest Labs, 2024) as the base model. We adopt FireFlow (Deng et al., 2024) as the inversion algorithm for FLUX, employing 25 diffusion steps, with 6 steps skipped and drag editing commencing at the 19th step. We perform optimization at the 7th denoising step over 70 iterations, using a learning rate of 1000 for the first 50 iterations and 1200 for the final 20. The adapter employed is InstantCharacter (Tao et al., 2025). Additional implementation details are provided in Appendix C.5.

- 4.2 EXPERIMENTAL SETUP

Benchmark. To facilitate systematic evaluation of region-based image drag-editing methods, we introduce a new Region-based Dragging Bench (ReD Bench). Existing datasets are often limited in scope. For example, DragBench (Shi et al., 2024b) primarily provides

Table 2: Comparison of composition performance across two benchmarks. Optimal results are bolded, where the second-best own underlines. Abbreviation: OPT – optimization-based; FT – fine-tuning-based; NFT – neither fine-tuning nor optimization-based.

Image Fidelity Mean Distance IFbg ↑ IFs2t ↑ IFs2s ↓ MD1 ↓ MD2 ↓

Benchmark Method Category #Params (M)

RegionDrag (Lu et al., 2024a) NFT 0 1.000 0.957 0.957 33.69 6.38 FastDrag (Zhao et al., 2024d) NFT 0 0.928 0.950 0.941 23.37 5.00 InstantDrag (Shin et al., 2024) FT 914 0.930 0.949 0.946 24.38 4.54 DragLoRA (Xia et al., 2025) OPT 3.19 0.927 0.950 0.938 26.04 4.86 FreeDrag (Ling et al., 2024) OPT 0.07 0.941 0.947 0.956 30.31 6.08 DragNoise (Liu et al., 2024) OPT 0.33 0.942 0.932 0.975 45.46 8.85 GoodDrag (Zhang et al., 2024c) OPT 0.07 0.935 0.956 0.942 20.38 4.50 CLIPDrag (Jiang et al., 2025) OPT 0.07 0.952 0.942 0.965 33.84 6.98 DragDiffusion (Shi et al., 2024b) OPT 0.07 0.944 0.948 0.947 32.15 5.65 DragFlow (Ours) OPT 0 0.992 0.958 0.934 19.46 4.48

ReD Bench

RegionDrag (Lu et al., 2024a) NFT 0 1.000 0.942 0.960 32.32 6.31 FastDrag (Zhao et al., 2024d) NFT 0 0.938 0.947 0.952 35.96 6.60 InstantDrag (Shin et al., 2024) FT 914 0.944 0.945 0.966 36.26 6.99 DragLoRA (Xia et al., 2025) OPT 3.19 0.942 0.941 0.952 42.03 6.77 FreeDrag (Ling et al., 2024) OPT 0.07 0.955 0.946 0.967 34.77 6.81 DragNoise (Liu et al., 2024) OPT 0.33 0.956 0.943 0.977 39.31 7.69 GoodDrag (Zhang et al., 2024c) OPT 0.07 0.948 0.946 0.956 37.87 6.91 CLIPDrag (Jiang et al., 2025) OPT 0.07 0.962 0.945 0.972 38.06 7.45 DragDiffusion (Shi et al., 2024b) OPT 0.07 0.954 0.944 0.958 39.41 7.05 DragFlow (Ours) OPT 0 0.969 0.948 0.941 31.59 5.93

DragBench-DR

point-to-point guidance for dragging operations, which fails to capture the complexity of regionlevel manipulation. Extensions of DragBench with coarse region annotations also fall short, as they lack explicit point-to-region alignment and operation-specific instructions, such as task tags and anchor points.

In ReD, each sample not only includes point-level operations, but also offers the translated regionto-region correspondences, together with explicit task labels covering the three most common categories of dragging (i.e., relocation, deformation, and rotation). Moreover, ReD is enriched with detailed contextual descriptions and intent prompts. This richer supervision allows ReD to serve as a more reliable testbed for assessing our approach and comparing it with a range of SOTA baselines.

Evaluation Metrics. Following prior work (Liu et al., 2024; Zhang et al., 2024c; Shi et al., 2024b), we evaluate drag-based editing using a combination of Mean Distance (MD) and Image Fidelity (IF). The standard MD metric (Shi et al., 2024b) quantifies the spatial correspondence of dragged content. We employ a masked variant, denoted as MD1, which computes correspondences only within the edited region, providing a more focused evaluation of alignment quality. In addition, we also adopt the variant proposed by Lu et al. (2024a), denoted as MD2. IF assesses visual consistency between the original and edited images via LPIPS (Zhang et al., 2018). We employ three variants for a finegrained analysis: IFbg: LPIPS computed on non-edited regions, capturing how well background content is preserved. IFs2t: LPIPS between the original source region and the edited target region, indicating how faithfully source content is transferred. IFs2s: LPIPS between the original and edited source regions, measuring how effectively the source is cleared after transfer.

- 4.3 COMPARISON WITH BASELINES

Quantitative Analysis. As shown in Tab. 2, our method achieves the lowest MD across both benchmarks, demonstrating the strongest spatial correspondence between user instructions and the resulting drag operations. The superior performance on IFs2s highlights the method’s ability to deliver precise and reliable content manipulation with high structural consistency and completeness. Although our approach ranks second on IFbg, the margin is marginal and largely attributable to the inherent inversion limitations of the CFG-distilled model. Nevertheless, our gradient mask–based hard constraints ensure robust background integrity, allowing our method to outperform most baselines despite these limitations.

Qualitative Analysis. Fig. 5 presents side-by-side comparisons with representative baselines. Our method consistently produces edits that accurately follow the specified dragging operations while preserving global scene coherence. In contrast, RegionDrag and InstantDrag often introduce struc-

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

FastDragInstantDragRegionDragInstructionDragLoRAGoodDragCLIPDragFreeDragOURS

##### InputsRegion-basedPoint-based

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

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

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

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Figure 5: Qualitative comparison of our method with multiple baselines in challenging scenarios.

tural distortions, while FreeDrag and FastDrag struggle with complex transformations such as rotations. CLIPDrag and DragLoRA frequently misinterpret relocation as deformation, leading to unintended artifacts. Across all these scenarios, our approach demonstrates superior structural preservation, faithful intent alignment, and robust performance over a diverse range of editing tasks.

- 4.4 ABLATION STUDY

To assess the contribution of each component, we perform ablation studies (results shown in Fig. 6 and Tab. 3) by progressively incorporating modules into our framework. Specifically, we examine the impact of adopting (a) region-based manipulation, introducing (b) mask-led background preservation, and applying (c) adapter-enhanced inversion. In (a), transitioning from points to regionbased control leads to consistent gains across all evaluation metrics. Compared with the baseline, regional manipulation reduces MD1 by 19.95 and increases IFs2t by 0.027, highlighting its ability to provide semantically richer and more coherent feature guidance. These results support our hypothesis that the regional affine strategy constitutes a more principled paradigm for modern generative

Point-based FLUX Dragger

+ Region-Level Affine Supervision

+ Background Preservation

+ Adapter Enhanced Inversion

Region Input

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

[Figure 149]

Figure 6: Qualitative ablation study comparing different variants of our framework. Table 3: Ablation study on ReD Bench examining the impact of some key components.

Image Fidelity Mean Distance IFbg ↑ IFs2t ↑ IFs2s ↓ MD1 ↓ MD2 ↓

Configs

Baseline (Point-based FLUX) 0.765 0.932 0.962 51.21 9.38 + Region-Level Affine Supervision 0.757 0.946 0.936 31.26 5.88 + Background Preservation 0.925 0.948 0.943 29.67 5.39 + Adapter-Enhanced Inversion 0.991 0.959 0.938 20.15 4.48

priors, effectively alleviating the intrinsic limitations of sparse point-level supervision. For (b), our gradient mask design substantially improves background consistency, with IFbg rising from 0.757 to 0.925, underscoring its effectiveness in preserving global backdrop integrity. Finally, the adapterenhanced inversion in (c) significantly strengthens subject fidelity, as reflected by an increase in IFs2t from 0.948 to 0.959, thereby confirming its capability to maintain foreground consistency under drag-editing. Taken together, these results clearly demonstrate that each component is effective on its own while synergizing to validate the overall design of our method.

- 5 CONCLUSION

We propose DragFlow, the first drag-based image editing framework tailored for DiTs. By rethinking supervision, inversion, and background handling in light of the unique representational and training properties of DiTs, DragFlow unlocks their powerful generative priors for controllable, fine-grained drag editing. Our three contributions, namely region-level motion supervision, background hard constraints, and adapter-enhanced inversion, collectively address the weaknesses of prior drag approaches, mitigating deformation artifacts while preserving subject identity and image realism. To support rigorous evaluation, we introduce ReD Bench, a benchmark designed around region-aware annotations and explicit task categories. Across both ReD Bench and DragBench-DR, DragFlow consistently surpasses existing state-of-the-art methods, demonstrating stronger faithfulness, better controllability, and higher-quality outputs.

Limitations and Future Work. Since the FLUX model we employ for drag-based editing is a CFG-distilled variant, its inversion drift is notably larger than that of non-distilled counterparts. Although we mitigate this issue through adapter-enhanced inversion, images with highly intricate structures still exhibit detail loss in the reconstruction. Consequently, the drag-editing results inherit these artifacts, leading to degraded visual quality. Future research could benefit from techniques

or advanced adapter architectures that further strengthen inversion fidelity, thereby reducing such artifacts and enhancing overall editing performance.

ACKNOWLEDGMENTS

This research is supported by the National Research Foundation, Prime Minister’s Office, Singapore, and the Ministry of Digital Development and Information, under its Online Trust and Safety (OTS) Research Programme (MDDI-OTS-001). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Prime Minister’s Office, Singapore, the Ministry of Digital Development and Information, or the Centre for Advanced Technologies in Online Safety.

REFERENCES

Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM transactions on graphics (TOG), 42(4):1–11, 2023.

Black Forest Labs. Flux.1 [dev]. https://huggingface.co/black-forest-labs/ FLUX.1-dev, 2024.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402, 2023.

Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.

Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zeroshot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6593–6602, 2024.

Gayoon Choi, Taejin Jeong, Sujung Hong, and Seong Jae Hwang. Dragtext: Rethinking text embedding in point-based image editing, 2024. URL https://arxiv.org/abs/2407.17843.

Yutao Cui, Xiaotong Zhao, Guozhen Zhang, Shengming Cao, Kai Ma, and Limin Wang. Stabledrag: Stable dragging for point-based image editing, 2024. URL https://arxiv.org/abs/ 2403.04437.

Yingying Deng, Xiangyu He, Changwang Mei, Peisong Wang, and Fan Tang. Fireflow: Fast inversion of rectified flow for image semantic editing, 2024. URL https://arxiv.org/abs/ 2412.07517.

Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890–16902, 2022.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning,

- 2024a.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

- 2024b.

Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. arXiv preprint arXiv:2309.17102, 2023.

Daiheng Gao, Shilin Lu, Shaw Walters, Wenbo Zhou, Jiaming Chu, Jie Zhang, Bang Zhang, Mengxi Jia, Jian Zhao, Zhaoxin Fan, et al. Eraseanything: Enabling concept erasure in rectified flow transformers. arXiv preprint arXiv:2412.20413, 2024.

Daiheng Gao, Nanxiang Jiang, Andi Zhang, Shilin Lu, Yufei Tang, Wenbo Zhou, Weiming Zhang, and Zhaoxin Fan. Revoking amnesia: Rl-based trajectory optimization to resurrect erased concepts in diffusion models. arXiv preprint arXiv:2510.03302, 2025.

Zinan Guo, Yanze Wu, Chen Zhuowei, Peng Zhang, Qian He, et al. Pulid: Pure and lightning id customization via contrastive alignment. Advances in neural information processing systems, 37: 36777–36804, 2024.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Xingzhong Hou, Boxiao Liu, Yi Zhang, Jihao Liu, Yu Liu, and Haihang You. Easydrag: Efficient point-based manipulation on diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8404–8413, 2024.

Ziqi Jiang, Zhen Wang, and Long Chen. Clipdrag: Combining text-based and drag-based instructions for image editing, 2025. URL https://arxiv.org/abs/2410.03097.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. Advances in neural information processing systems, 35:26565–26577, 2022.

Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2426–2435, June 2022.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Leyang Li, Shilin Lu, Yan Ren, and Adams Wai-Kin Kong. Set you straight: Auto-steering denoising trajectories to sidestep unwanted concepts. arXiv preprint arXiv:2504.12782, 2025.

Xiaojian Lin, Hanhui Li, Yuhao Cheng, Yiqiang Yan, and Xiaodan Liang. Gdrag: Towards generalpurpose interactive editing with anti-ambiguity point diffusion. In The Thirteenth International Conference on Learning Representations, 2025.

Pengyang Ling, Lin Chen, Pan Zhang, Huaian Chen, Yi Jin, and Jinjin Zheng. Freedrag: Feature dragging for reliable point-based image editing. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6860–6870. IEEE, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Haofeng Liu, Chenshu Xu, Yifei Yang, Lihua Zeng, and Shengfeng He. Drag your noise: Interactive point-based editing via diffusion semantic propagation, 2024. URL https://arxiv.org/ abs/2404.01050.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775–5787, 2022.

Jingyi Lu and Kai Han. Inpaint4drag: Repurposing inpainting models for drag-based image editing via bidirectional warping. arXiv preprint arXiv:2509.04582, 2025.

Jingyi Lu, Xinghui Li, and Kai Han. Regiondrag: Fast region-based image editing with diffusion models, 2024a. URL https://arxiv.org/abs/2407.18247.

Shilin Lu, Yanzhu Liu, and Adams Wai-Kin Kong. Tf-icon: Diffusion-based training-free crossdomain image composition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2294–2305, 2023.

Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, and Adams Wai-Kin Kong. Mace: Mass concept erasure in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6430–6440, 2024b.

Shilin Lu, Zihan Zhou, Jiayou Lu, Yuanzhi Zhu, and Adams Wai-Kin Kong. Robust watermarking using generative priors against image editing: From benchmarking to advances. arXiv preprint arXiv:2410.18775, 2024c.

Shilin Lu, Zhuming Lian, Zihan Zhou, Shaocong Zhang, Chen Zhao, and Adams Wai-Kin Kong. Does flux already know how to perform physically plausible image composition? arXiv preprint arXiv:2509.21278, 2025.

Minxing Luo, Wentao Cheng, and Jian Yang. Rotationdrag: Point-based image editing with rotated diffusion features, 2024. URL https://arxiv.org/abs/2401.06442.

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421, 2023.

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8488–8497, 2024.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

Shen Nie, Hanzhong Allan Guo, Cheng Lu, Yuhao Zhou, Chenyu Zheng, and Chongxuan Li. The blessing of randomness: Sde beats ode in general diffusion-based image editing, 2024. URL https://arxiv.org/abs/2311.01410.

OpenAI. Introducing gpt-5, 2025. URL https://openai.com/index/ introducing-gpt-5/.

Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Proceedings, SIGGRAPH ’23, pp. 1–11. ACM, July 2023. doi: 10.1145/3588432.3591500. URL http://dx.doi.org/10.1145/3588432.3591500.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Yan Ren, Shilin Lu, and Adams Wai-Kin Kong. All that glitters is not gold: Key-secured 3d secrets within 3d gaussian splatting. arXiv preprint arXiv:2503.07191, 2025.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022a.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models, 2022b. URL https://arxiv.org/ abs/2112.10752.

Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22500– 22510, 2023.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.

Yujun Shi, Jun Hao Liew, Hanshu Yan, Vincent YF Tan, and Jiashi Feng. Lightningdrag: Lightning fast and accurate drag-based image editing emerging from videos. arXiv preprint arXiv:2405.13722, 2024a.

Yujun Shi, Chuhui Xue, Jun Hao Liew, Jiachun Pan, Hanshu Yan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8839–8849, 2024b.

Joonghyuk Shin, Daehyeon Choi, and Jaesik Park. Instantdrag: Improving interactivity in dragbased image editing. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–10, 2024.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. URL https://arxiv.org/abs/2010.02502.

Jiale Tao, Yanbing Zhang, Qixun Wang, Yiji Cheng, Haofan Wang, Xu Bai, Zhengguang Zhou, Ruihuang Li, Linqing Wang, Chunyu Wang, et al. Instantcharacter: Personalize any characters with a scalable diffusion transformer framework. arXiv preprint arXiv:2504.12395, 2025.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1921–1930, June 2023.

Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024.

Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J. Fleet, Radu Soricut, Jason Baldridge, Mohammad Norouzi, Peter Anderson, and William Chan. Imagen editor and editbench: Advancing and evaluating textguided image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18359–18369, June 2023.

Yanghao Wang and Long Chen. Inversion circle interpolation: Diffusion-based image augmentation for data-scarce classification. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 25560–25569, 2025a.

Yanghao Wang and Long Chen. Noise matters: Optimizing matching noise for diffusion classifiers. arXiv preprint arXiv:2508.11330, 2025b.

Yanghao Wang, Zhen Wang, and Long Chen. Target-aware image editing via cycle-consistent constraints. arXiv preprint arXiv:2510.20212, 2025.

Tianyi Wei, yifan Zhou, Dongdong Chen, and Xingang Pan. Freeflux: Understanding and exploiting layer-specific roles in rope-based mmdit for versatile image editing. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Siwei Xia, Li Sun, Tiantian Sun, and Qingli Li. Draglora: Online optimization of lora adapters for drag-based image editing in diffusion model, 2025. URL https://arxiv.org/abs/ 2505.12427.

Zexuan Yan, Yue Ma, Chang Zou, Wenteng Chen, Qifeng Chen, and Linfeng Zhang. Eedit: Rethinking the spatial and temporal redundancy for efficient image editing. arXiv preprint arXiv:2503.10270, 2025.

Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18381– 18391, 2023.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for contentrich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022.

Xinlei Yu, Zhangquan Chen, Yudong Zhang, Shilin Lu, Ruolin Shen, Jiangning Zhang, Xiaobin Hu, Yanwei Fu, and Shuicheng Yan. Visual document understanding and question answering: A multi-agent collaboration framework with test-time scaling. arXiv preprint arXiv:2508.03404, 2025a.

Xinlei Yu, Chengming Xu, Guibin Zhang, Yongbo He, Zhangquan Chen, Zhucun Xue, Jiangning Zhang, Yue Liao, Xiaobin Hu, Yu-Gang Jiang, et al. Visual multi-agent system: Mitigating hallucination snowballing via visual flow. arXiv preprint arXiv:2509.21789, 2025b.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36, 2024a.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.

Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9026–9036, 2024b.

Zewei Zhang, Huan Liu, Jun Chen, and Xiangyu Xu. Gooddrag: Towards good practices for drag editing with diffusion models, 2024c. URL https://arxiv.org/abs/2404.07206.

Chen Zhao, Weiling Cai, Chenyu Dong, and Chengwei Hu. Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8281–8291, 2024a.

Chen Zhao, Weiling Cai, Chengwei Hu, and Zheng Yuan. Cycle contrastive adversarial learning with structural consistency for unsupervised high-quality image deraining transformer. Neural Networks, pp. 106428, 2024b.

Chen Zhao, Zhizhou Chen, Yunzhe Xu, Enxuan Gu, Jian Li, Zili Yi, Qian Wang, Jian Yang, and Ying Tai. From zero to detail: Deconstructing ultra-high-definition image restoration from progressive spectral perspective. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 17935–17946, 2025a.

Chen Zhao, En Ci, Yunzhe Xu, Tiehan Fan, Shanyan Guan, Yanhao Ge, Jian Yang, and Ying Tai. Ultrahr-100k: Enhancing uhr image synthesis with a large-scale high-quality dataset. arXiv preprint arXiv:2510.20661, 2025b.

Chen Zhao, Jiawei Chen, Hongyu Li, Zhuoliang Kang, Shilin Lu, Xiaoming Wei, Kai Zhang, Jian Yang, and Ying Tai. Luve: Latent-cascaded ultra-high-resolution video generation with dual frequency experts. arXiv preprint arXiv:2602.11564, 2026.

Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. arXiv preprint arXiv:2407.05282, 2024c.

Xuanjia Zhao, Jian Guan, Congyi Fan, Dongli Xu, Youtian Lin, Haiwei Pan, and Pengming Feng. Fastdrag: Manipulate anything in one step. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024d.

Dewei Zhou, Zongxin Yang, and Yi Yang. Pyramid diffusion models for low-light image enhancement. arXiv preprint arXiv:2305.10028, 2023.

Dewei Zhou, You Li, Fan Ma, Zongxin Yang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. arXiv preprint arXiv:2402.05408, 2024a.

Dewei Zhou, You Li, Fan Ma, Zongxin Yang, and Yi Yang. Migc++: Advanced multi-instance generation controller for image synthesis. arXiv preprint arXiv:2407.02329, 2024b.

Dewei Zhou, Ji Xie, Zongxin Yang, and Yi Yang. 3dis: Depth-driven decoupled instance synthesis for text-to-image generation. arXiv preprint arXiv:2410.12669, 2024c.

Dewei Zhou, Mingwei Li, Zongxin Yang, and Yi Yang. Dreamrenderer: Taming multi-instance attribute control in large-scale text-to-image models. arXiv preprint arXiv:2503.12885, 2025.

Yuanzhi Zhu, Ruiqing Wang, Shilin Lu, Junnan Li, Hanshu Yan, and Kai Zhang. Oftsr: Onestep flow for image super-resolution with tunable fidelity-realism trade-offs. arXiv preprint arXiv:2412.09465, 2024.

Yuanzhi Zhu, Xi Wang, St´ephane Lathuili`ere, and Vicky Kalogeiton. Dimo: Distilling masked diffusion models into one-step generator. arXiv preprint arXiv:2503.15457, 2025.

Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting, 2023.

# Appendix

#### Table of Contents

- A Additional Related Work 17
- B Preliminary Information 17 B.1 Diffusion Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17 B.2 Point-based Image Drag-editing . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C Implementation Details 19

- C.1 Progressive Transformation in Subtasks . . . . . . . . . . . . . . . . . . . . . . 20
- C.2 Relocation Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.3 Deformation Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.4 Rotation Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.5 Details about Experimental Settings . . . . . . . . . . . . . . . . . . . . . . . . 22
- C.6 Quantitative Analysis for DiT Supervision Granularity . . . . . . . . . . . . . . 24

- D Adaptive Input Processing 24

- D.1 Region Weight Regularization for Multi-Operations . . . . . . . . . . . . . . . 25
- D.2 Adaptive Gradient Mask Generation . . . . . . . . . . . . . . . . . . . . . . . 25
- D.3 Leveraging MLLM for Prompt and Tag Generation . . . . . . . . . . . . . . . . 26
- D.4 Effectiveness of MLLM-driven Intent Parsing . . . . . . . . . . . . . . . . . . . 26

- E Criterion Details 27

- E.1 Computation of Image Fidelity (IF) . . . . . . . . . . . . . . . . . . . . . . . . 27
- E.2 Computation of Mean Distance (MD) . . . . . . . . . . . . . . . . . . . . . . . 28

- F Additional Baseline Information 29
- G Benchmark Details 29

- G.1 Formation of the ReD Benchmark . . . . . . . . . . . . . . . . . . . . . . . . . 29
- G.2 Adoption of the DragBench-DR Benchmark . . . . . . . . . . . . . . . . . . . 29
- G.3 Demonstration of Data Samples . . . . . . . . . . . . . . . . . . . . . . . . . . 30

- H Extra Qualitative Results 31

- H.1 Additional Qualitative Samples . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- H.2 Challenging Scenarios . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- H.3 Failure Scenarios . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- I Runtime and Memory Comparison 32
- J Framework and Module Generalizability 33

- J.1 Generalizability of Dragging Framework . . . . . . . . . . . . . . . . . . . . . 33
- J.2 Generalizability of ID-preservation Design . . . . . . . . . . . . . . . . . . . . 33

- K LLM Usage Statement 34

- A ADDITIONAL RELATED WORK

Generative Model-based Image Editing. Recent and significant advancements in generative models (Chang et al., 2023; Ding et al., 2022; Nichol et al., 2021; Ramesh et al., 2022; Rombach et al., 2022a; Saharia et al., 2022; Yu et al., 2022; Peebles & Xie, 2023; Esser et al., 2024a) have enhanced many applications (Avrahami et al., 2023; Ruiz et al., 2023; Hertz et al., 2022; Kim et al.,

- 2022; Tumanyan et al., 2023; Zhou et al., 2025; 2023; 2024a;b;c; Zhao et al., 2024a; 2025b;a; 2024b; 2026; Wang et al., 2025; Wang & Chen, 2025a;b; Lu et al., 2024b;c; Li et al., 2025; Ren et al., 2025; Gao et al., 2024; 2025; Zhu et al., 2024; 2025; Yu et al., 2025a;b). In this study, we focus on real image editing, which allows users to freely modify actual photographs, producing highly realistic results. Typically, the inputs for image editing include an image and various conditions that help users accurately describe their desired changes. These conditions can encompass text prompts using natural language to specify the edits (Brooks et al., 2023; Zhang et al., 2024b; Fu et al., 2023; Zhang et al., 2024a), region masks to designate areas for modification (Zhao et al., 2024c; Zhuang et al.,
- 2023; Wang et al., 2023), additional images to provide desired styles or objects (Chen et al., 2024; Lu et al., 2023; Yang et al., 2023), and drag points (Lu & Han, 2025; Nie et al., 2024) that enable users to interactively move specific points in the image to target positions.

- B PRELIMINARY INFORMATION

This section provides essential background knowledge on diffusion models and point-based image drag-editing techniques. We begin by discussing the evolution of diffusion methods, tracing the transition from early frameworks of Stable Diffusion (SD) Series with Denoising Diffusion Implicit Models (DDIM) (Song et al., 2022) to more advanced rectified flow models and ODE solutions. These advancements enhance the generative process, facilitating more efficient and deterministic sampling. In the latter part, we explore point-based drag-editing methods, highlighting how they allow users to manipulate key points for image modification directly. This approach contrasts with region-based methods by leveraging specific user-given control points for supervision and tracking.

- B.1 DIFFUSION METHODS

Recent advances in text-to-image generative models have driven a rapid evolution in architectural paradigms, providing a strong impetus for advancing existing image editing methods. Innovations in newer architectures and learning objectives enrich prior knowledge and exhibit significantly improved capacity in data comprehension and semantic alignment. Early UNet–based diffusion frameworks, such as SD 1.5 and SD 2 from Rombach et al. (2022b), have been increasingly supplanted by DiT-based rectified flow designs with more robust pre-trained priors (e.g., FLUX.1 (Black Forest Labs, 2024) and SD 3 (Esser et al., 2024b)).

Stable Diffusion with DDIM Inversion. Early drag-based image editing techniques primarily use SD as the foundational framework. They leverage SD’s UNet architecture for noise prediction and rely on DDIM inversion to derive noisy latents zt from the clean latent encoding z0 of an input image x. DDIM operates through two core processes: a forward inversion process that iteratively adds noise to transform clean latents into noisy ones, and a backward denoising process that removes noise to generate edited outputs.

To formalize these processes, we first define α¯t = ts=1(1 − βs), where βs is the noise schedule at step s, and t ∈ [0,T] indexes the diffusion steps, with t = 0 representing the clean state and t = T

the fully noisy state. Starting from the clean latent z0 (i.e., the latent output of the VAE encoder), the forward inversion process computes zt from zt−1 by first using the UNet noise predictor ϵθ(·,·) to estimate the noise in zt−1, which reconstructs an approximation of the original clean latent zˆ0. This estimated clean latent is then perturbed to produce the next noisy latent zt:

###### zt = √α¯t · zˆ0 + √1 − α¯t · ϵθ(zt−1,t − 1), (5)

where the term √α¯t · zˆ0 retains a scaled version of the estimated clean signal, and √1 − α¯t · ϵθ(·) adds controlled noise with scaling factors aligning with the pre-defined schedule α¯t.

Oppositelly, the backward denoising process iteratively refines zt to zt−1, gradually reducing noise. From the current noisy latent zt, the model first estimates the clean latent zˆ0 by inverting the noise

addition—subtracting the predicted noise ϵθ(zt,t) (scaled by the schedule √1 − α¯t):

√1 − α¯t · ϵθ(zt,t) √α¯t

zt −

. (6)

zˆ0 =

Using this estimated clean latent zˆ0, the denoised latent zt−1 is computed by re-adding a controlled amount of noise to zˆ0, with the noise level reduced compared to zt, such as

###### zt−1 = √α¯t−1 · zˆ0 + 1 − α¯t−1 · ϵθ(zt,t). (7)

This re-noising step ensures a gradual transition toward the clean state: as t decreases, α¯t−1 increases to approaching 1, so the weight on the estimated clean structure zˆ0 grows, while the weight on the noise term shrinks. This stepwise denoising is critical for editing, as the iterative formulation enables conditional guidance (e.g., text prompts or drag constraints) to be seamlessly injected at each stage, thereby ensuring precise and progressively refined control over the final output.

Rectified Flow with ODE Solver. In contrast, rectified flow models, as adopted in our DragFlow framework leveraging the DiT prior, reformulate the generative process through approximate straight-line trajectories between data and noise distributions, enabling more efficient and deterministic sampling with fewer steps. For simplicity, we normalize the time parameter to t ∈ [0,1].

The forward process linearly interpolates the latent as

###### zt = (1 − t)z0 + tϵ, (8)

where z0 is the clean VAE-encoded latent (same as the UNet’s), and ϵ ∼ N(0,I) is standard Gaussian noise. The associated velocity field is constant, defined as dzt

dt = ϵ − z0. In this way, the model can model a parameterized velocity vθ(zt,t) ≈ ϵ − z0 via objectives like conditional flow matching, by minimizing

t,ϵ ∥vθ(zt,t) − (ϵ − z0)∥22 . (9)

L = Et,z

For the backward denoising process, we solve the ordinary differential equation (ODE) dz/dt = vθ(z,t) backward in time from t = 1 to t = 0, starting from z1 ≈ ϵ. In discrete steps (e.g., via Fireflow Solver (Deng et al., 2024)), this approximates:

###### zt−∆t = zt + (−∆t) · vθ(zt,t), (10)

where ∆t > 0 is the step size. Conversely, to obtain the noisy latent zt during inversion (from clean z0 at t = 0 to z1 at t = 1), we integrate the ODE forward:

###### zt+∆t = zt + ∆t · vθ(zt,t). (11)

This straight-path formulation in rectified flow supports fewer function evaluations compared to the curved trajectories with more complicated noise schedules in DDIM, enabling our region-level affine supervision in DragFlow to leverage more robust priors for drag-based editing.

- B.2 POINT-BASED IMAGE DRAG-EDITING

Point-based drag editing was first introduced by DragGAN (Pan et al., 2023), as an interactive paradigm for image manipulation, enabling users to directly move key points on an image to achieve desired transformations. Unlike text-guided methods, which often struggle with ambiguity in complex scenes, point-based dragging encodes editing intentions through spatially localized control points. This approach aligns well with users’ intuitive interaction patterns, providing a straightforward yet effective means of specifying editing goals.

User Input. Each editing task requires the following basic inputs:

- • An original image x (converted to the initial latent z0 through VAE encoding).
- • A set of handle points {hi}ni=1 indicating locations to be manipulated.
- • Corresponding target points {ti}ni=1 representing desired positions after dragging.
- • Mask B to protect or constrain regions that should remain unchanged.

Core Components. The workflow typically consists of two interconnected modules:

- 1. Motion Supervision (MS). MS is designed to ensure the model preserves image features while enforcing alignment between source and target points. MS computes losses based on:

- • Alignment Loss: Measures feature differences between patches around original handle points and patches around their current predicted locations.

Lalign =

i p∈Ω(h0i ,r) q∈Ω(hki ,r)

Fq(ztk) − sg(Fp(zt0)) 1 , (12)

where F(·) extracts local features, Ω(,r) denotes a patch of radius r, and sg(·) is the stop-gradient operator. q and p define the source patch and the manipulated patches, respectively.

- • Smoothness Loss: Encourages gradual changes in the feature space.

Lsmooth =

i q∈Ω(hki ,r)

Fq(ztk) − sg(Fq(ztk)) 1 , (13)

- • Mask Loss: Penalizes unintended modifications outside user-defined regions.

Lmask = ∥(ztk − sg(zt0)) ⊙ (1 − B)∥1, (14) The overall motion supervision loss is:

LMS = βLalign + (1 − β)Lsmooth + λLmask, (15) which is backpropagated to iteratively update the latent code:

ztk+1 = ztk − lr ·

∂LMS ∂ztk

. (16)

- 2. Point Tracking (PT). PT updates handle point locations across diffusion steps to ensure

they follow intended trajectories. For each handle point hki, the new location is determined via nearest-neighbor feature matching:

hki+1 = argminq∈Ω(hk

i ,r2)∥Fq(ztk+1) − Fh0

(zt0)∥1. (17) Workflow Summary. The point-based editing process proceeds iteratively:

i

- 1. Apply several MS steps to align features toward target positions while preserving image consistency during the updating processes.
- 2. Perform PT to update handle points based on feature tracking.
- 3. Repeat the MS and PT cycle until the handle points converge to their targets.

To sum up, classic point-based drag editing leverages feature-level supervision and PT to enable localized manipulations, ensuring that the edited image remains coherent with the original while reflecting user-specified modifications.

However, this pipeline inherently suffers from several limitations: nearest-neighbor search and PT introduce high uncertainty during optimization; the explicit influence of each control point is confined to a narrow feature neighborhood, which restricts the scope of effective guidance. By contrast, region-based dragging naturally extends these ideas by operating over semantically coherent masks, thereby offering more stable, interpretable, and semantically meaningful editing outcomes.

- C IMPLEMENTATION DETAILS

This section highlights the core motion strategy of region-based drag operations through progressive affine transformations. We define multiple types of drag operations, each designed to enhance user control and precision in image editing. Subsequent subsections detail the implementations of these operations, all leveraging progressively interpolated affine transformations for seamless transitions. Additionally, the final subsection offers supplementary settings related to the experimental section.

- C.1 PROGRESSIVE TRANSFORMATION IN SUBTASKS

Operation in Subtasks. As introduced in the main text, we define three types of drag operations: relocation, local deformation, and rotation. Recall these definitions, the source masks Mi denote the initial regions, with centroids bi guiding relocation and local deformation toward the target region indicated by the centroid ti. For rotation, the anchor ai specifies the pivot. Each transformation is realized through affine updates, with parameters ξi(k) interpolated over K iterations to gradually propagate the source mask toward its target configuration. In the following subsections, we detail how these subtask settings facilitate the affine transformation in region-based image dragging.

Progressive Motion Schedule. In practice, for each region-specific drag operation, we only compute the full motion schedule ξi(K) during the initial update iteration:

ξi(K) =

(ti − bi), (Relocation & Deformation) (∠biaiti, ai), (Rotation),

(18)

where bi, ti, and ai denote the begin, target, and anchor points, respectively. Rather than recomputing the dragged and left distances at every iteration, subsequent steps obtain their progressive schedules directly via a linear interpolation:

ξi(k) = Kk · ξi(K), Kk ∈ [0,1]. (19) This design ensures that the motion evolves smoothly from the initial to the target state. The resulting progressive motion schedule is then injected into the affine transformation operator Ω(·,ξi(k)), to enforce consistent supervision over the sequence of incremental updates.

- C.2 RELOCATION TASKS

Relocation involves shifting an entire region to a new position while preserving its original geometry and scale. Recall our definition in Subsec. 3.2 This operation is parameterized by a displacement vector derived from the difference between the target point ti and the source centroid bi, scaled linearly as ξi(k) = Kk (ti − bi).

To apply this, for example, we can consider a point p = (u,v) within the source region mask Mi(0). The transformed point p′ = (u′,v′) can be computed via homogeneous coordinates:

- u′
- v′ 1

=

1 0 du 0 1 dv 0 0 1

- u
- v 1

, (20)

where (du,dv) is the displacement vector from ξi(k). Breaking down the matrix:

- • The top-left 2 × 2 submatrix is the identity, ensuring no rotation or scaling;
- • The relocation components du and dv in the first two rows of the third column directly add to the coordinates: u′ = u + du, v′ = v + dv;
- • The bottom row maintains homogeneity.

This matrix is applied across the whole operation patch, resulting in an efficient shift of the entire region in dragging steps.

- C.3 DEFORMATION TASKS

Deformation enables localized adjustments by selectively displacing subregions, effectively altering the overall shape without global rigidity. In our setup, this is treated similarly to relocation, utilizing the same affine transformation method as described in the relocation operation. The key difference lies in its application: it targets only the edge areas of the object to be edited based on scene requirements, achieving effects such as elongation or shortening.

This selective application distinguishes deformation from full-region relocation, allowing for intuitive shape modifications as visualized in our method’s overview.

SINGLE-00 SINGLE-01 SINGLE-02 SINGLE-03 SINGLE-04 SINGLE-05

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Source Image

SINGLE-06 SINGLE-07 SINGLE-08 SINGLE-09 SINGLE-10 SINGLE-11

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

DOUBLE-00 DOUBLE-01 DOUBLE-02 DOUBLE-03

SINGLE-12 SINGLE-13 SINGLE-14 SINGLE-15 SINGLE-16 SINGLE-17

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

SINGLE-18 SINGLE-19 SINGLE-20 SINGLE-21 SINGLE-22 SINGLE-23

DOUBLE-04 DOUBLE-05 DOUBLE-06 DOUBLE-07

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

DOUBLE-08 DOUBLE-09 DOUBLE-10 DOUBLE-11

SINGLE-24 SINGLE-25 SINGLE-26 SINGLE-27 SINGLE-28 SINGLE-29

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

DOUBLE-12 DOUBLE-13 DOUBLE-14 DOUBLE-15

SINGLE-30 SINGLE-31 SINGLE-32 SINGLE-33 SINGLE-34 SINGLE-35

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

DOUBLE-16 DOUBLE-17 DOUBLE-18

SINGLE-36 SINGLE-37

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

- Figure 7: Visualization of DiT latent features (Sample 1 out of 2) based on PCA using the top 5 principal components.

- C.4 ROTATION TASKS

Rotation pivots a region around a fixed anchor, reorienting it by a progressively interpolated angle. The rotation angle is determined geometrically from the triangle formed by bi (i.e., the source region centroid), ai (i.e., the user-specified anchor), and ti (i.e., the target region centroid). Over K iterations, the interpolated parameters are defined as

ξi(k) = K k ∠biaiti, ai ,

which guarantees a smooth progression from the original orientation toward the desired angular displacement.

For a point r = (w,z) in Mi(k), rotated around anchor c = (wc,zc) by an angle ϕ derived from ξi(k), the updated coordinates r′ = (w′,z′) are obtained as

w′ z′ 1

=

1 0 wc 0 1 zc 0 0 1

cosϕ −sinϕ 0 sinϕ cosϕ 0

0 0 1

1 0 −wc 0 1 −zc 0 0 1

w z 1

. (21)

This composite transformation can be interpreted step by step:

- • The rightmost matrix translates the point so that the anchor ai coincides with the origin, removing bias from the global coordinate system.
- • The central rotation matrix applies the angular displacement ϕ, producing the intermediate rotated coordinates (w′′,z′′) where w′′ = w cosϕ − z sinϕ and z′′ = w sinϕ + z cosϕ.

SINGLE-00 SINGLE-01 SINGLE-02 SINGLE-03 SINGLE-04 SINGLE-05

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

Source Image

SINGLE-06 SINGLE-07 SINGLE-08 SINGLE-09 SINGLE-10 SINGLE-11

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

DOUBLE-00 DOUBLE-01 DOUBLE-02 DOUBLE-03

SINGLE-12 SINGLE-13 SINGLE-14 SINGLE-15 SINGLE-16 SINGLE-17

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

SINGLE-18 SINGLE-19 SINGLE-20 SINGLE-21 SINGLE-22 SINGLE-23

DOUBLE-04 DOUBLE-05 DOUBLE-06 DOUBLE-07

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

DOUBLE-08 DOUBLE-09 DOUBLE-10 DOUBLE-11

SINGLE-24 SINGLE-25 SINGLE-26 SINGLE-27 SINGLE-28 SINGLE-29

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

DOUBLE-12 DOUBLE-13 DOUBLE-14 DOUBLE-15

SINGLE-30 SINGLE-31 SINGLE-32 SINGLE-33 SINGLE-34 SINGLE-35

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

DOUBLE-16 DOUBLE-17 DOUBLE-18

SINGLE-36 SINGLE-37

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

- Figure 8: Visualization of DiT latent features (Sample 2 out of 2) based on PCA using the top 5 principal components.

• The leftmost matrix translates the rotated point back into the original coordinate frame, re-centering the result around ai.

This decomposition not only clarifies the geometric intuition behind rotation but also integrates seamlessly with our iterative framework: at each step k, the patch masks Mi(k) are updated under this rotation transformation, ensuring gradual, controlled reorientation. Compared with relocation and deformation, rotation requires an explicit anchor to define the pivot, highlighting its distinct interaction design while still being governed by the same affine transformation principles.

- C.5 DETAILS ABOUT EXPERIMENTAL SETTINGS

Layers for Feature Manipulation. We empirically identify the 17th and 18th double-stream blocks of the DiT backbone as the most representative features to facilitate drag optimization. To substantiate this choice, we conduct a visualization-based analysis of the main DiT modules (refer to Fig. 7 and Fig. 8 for details). Our selection is guided by the principle that effective feature blocks should retain high representational fidelity to the input. While certain blocks may encode large amounts of information, much of it can be dominated by noise. By examining PCA visualizations, we qualitatively assess whether the leading components align with the appearance of the original image, thereby confirming that the selected layers capture the essential visual characteristics required for precise and robust editing.

As illustrated in the figures, the latent representations from “DOUBLE-17” and “DOUBLE-18” retain rich and flexible features, which can be effectively manipulated during editing. In contrast, certain blocks (e.g., “SINGLE-20” and “SINGLE-30”) exhibit overly “clean” latents, making it difficult to perform meaningful edits. Other blocks (e.g., “DOUBLE-04” and “SINGLE-06”) con-

表格 1

Published as a conference paper at ICLR 2026

| |CLIP ⬆|IF_glb ⬆|IF_unm ↔|IF_msk ⬆|IF_b2t ↑|IF_b2b ↓|IF_t2t ⬇|MD_glb ↓|**MD_unm ⬇|DS_glb ⬇|DS_b2t ⬇|
|---|---|---|---|---|---|---|---|---|---|---|---|
|D0|0.9902|0.9760|0.9823|0.9921|0.9300|0.9867|0.9881|54.2866|53.6015|0.0179|0.7366|
|D1| | | | |0.9362|0.9649| |60.6548| | | |
|D2|0.9670|0.9119|0.9199|0.9895|0.9417|0.9503|0.9425|62.3858|44.1256|0.0745|0.7752|
|D3| | | | |0.9487|0.9489| |43.3012| | | |
|D4|0.9723|0.9049|0.9128|0.9886|0.9489|0.9463|0.9398|38.9764|32.4896|0.0693|0.8197|
|D5| | | | |0.9499|0.9469| |37.5541| | | |
|D6|0.9759|0.9092|0.9174|0.9898|0.9515|0.9478|0.9417|35.2521|29.4525|0.0593|0.8320|
|D7| | | | |0.9513|0.9468| |35.0182| | | |
|D8|0.9740|0.9021|0.9099|0.9895|0.9526|0.9432|0.9374|33.0196|27.7077|0.0641|0.8385|
|D9| | | | |0.9533|0.9430| |32.8782| | | |
|D10|0.9704|0.9003|0.9086|0.9893|0.9539|0.9424|0.9368|31.2791|26.7950|0.0666|0.8457|
|D11| | | | |0.9537|0.9412| |33.1572| | | |
|D12|0.9706|0.8970|0.9048|0.9899|0.9539|0.9404|0.9364|32.5503|26.9787|0.0674|0.8444|
|D13| | | | |0.9542|0.9400| |33.1270| | | |
|D14|0.9701|0.8952|0.9030|0.9901|0.9548|0.9397|0.9349|33.2684|26.7753|0.0682|0.8491|
|D15| | | | |0.9548|0.9395| |31.9174| | | |
|D16|0.9729|0.8999|0.9068|0.9895|0.9550|0.9410|0.9376|32.6700|26.0195|0.0589|0.8488|
|D17|0.9678|0.8937|0.9013|0.9894|0.9559|0.9380|0.9351|27.5244|23.7475|0.0690|0.8529|
|D18|0.9681|0.8913|0.8991|0.9887|0.9559|0.9365|0.9341|28.3916|23.2467|0.0688|0.8545|
|S0|0.9857|0.9538|0.9605|0.9918|0.9404|0.9739|0.9708|48.2325|46.0935|0.0293|0.7814|
|S1| | | | |0.9376|0.9831| |49.8347| | | |
|S2|0.9909|0.9780|0.9845|0.9918|0.9297|0.9893|0.9904|53.1380|53.2201|0.0151|0.7400|
|S3| | | | |0.9293|0.9889| |51.4738| | | |
|S4|0.9908|0.9777|0.9842|0.9918|0.9295|0.9893|0.9904|53.0743|53.1336|0.0149|0.7405|
|S5| | | | |0.9312|0.9806| |50.1831| | | |
|S6|0.9875|0.9590|0.9652|0.9925|0.9373|0.9764|0.9755|45.1789|45.1502|0.0263|0.7677|
|S7| | | | |0.9374|0.9761| |46.7583| | | |
|S8|0.9869|0.9572|0.9636|0.9926|0.9380|0.9756|0.9746|44.5410|44.4219|0.0272|0.7761|
|S9| | | | |0.9382|0.9748| |45.3253| | | |
|S10|0.9870|0.9580|0.9641|0.9926|0.9371|0.9742|0.9751|47.7709|45.0795|0.0285|0.7676|
|S11| | | | |0.9375|0.9744| |46.8467| | | |
|S12|0.9875|0.9581|0.9644|0.9926|0.9374|0.9743|0.9759|44.1075|44.5282|0.0271|0.7688|
|S13| | | | |0.9369|0.9748| |47.8562| | | |
|S14|0.9879|0.9606|0.9666|0.9926|0.9363|0.9751|0.9767|48.2122|45.3721|0.0262|0.7641|
|S15| | | | |0.9300|0.9856| |50.1526| | | |
|S16|0.9910|0.9777|0.9843|0.9918|0.9301|0.9891|0.9901|53.2020|53.0729|0.0145|0.7405|
|S17| | | | |0.9381|0.9824| |49.5279| | | |
|S18|0.9864|0.9643|0.9705|0.9925|0.9356|0.9781|0.9787|47.5378|47.7791|0.0245|0.7590|
|S19| | | | |0.9355|0.9783| |48.8562| | | |
|S20|0.9885|0.9648|0.9708|0.9925|0.9350|0.9780|0.9794|50.4383|48.2626|0.0221|0.7597|
|S21| | | | |0.9301|0.9842| |52.9673| | | |
|S22|0.9890|0.9753|0.9826|0.9917|0.9297|0.9886|0.9889|53.1101|53.0228|0.0174|0.7398|
|S23| | | | |0.9310|0.9831| |50.5489| | | |
|S24|0.9816|0.9612|0.9708|0.9884|0.9317|0.9793|0.9805|49.5741|50.1751|0.0309|0.7431|
|S25| | | | |0.9319|0.9792| |51.2489| | | |
|S26|0.9879|0.9689|0.9748|0.9926|0.9324|0.9805|0.9827|50.6373|50.4120|0.0234|0.7474|
|S27| | | | |0.9317|0.9798| |53.8482| | | |
|S28|0.9742|0.9555|0.9642|0.9893|0.9318|0.9766|0.9766|54.5720|50.9768|0.0351|0.7425|
|S29| | | | |0.9297|0.9865| |53.0572| | | |
|S30|0.9867|0.9731|0.9805|0.9915|0.9291|0.9876|0.9881|53.0489|52.9016|0.0215|0.7376|
|S31| | | | |0.9298|0.9805| |52.1048| | | |
|S32|0.9806|0.9594|0.9670|0.9914|0.9330|0.9780|0.9785|50.2635|49.9257|0.0319|0.7434|
|S33| | | | |0.9296|0.9842| |53.9480| | | |
|S34|0.9880|0.9738|0.9813|0.9913|0.9293|0.9876|0.9883|53.4740|53.4046|0.0195|0.7377|
|S35| | | | |0.9292|0.9870| |55.9870| | | |
|S36|0.9874|0.9722|0.9802|0.9914|0.9294|0.9869|0.9878|57.3963|54.8566|0.0217|0.7363|

70

52.5

35

###### MIN

17.5

0

D0D1D2D3D4D5D6D7D8D9D10D11D12D13D14D15D16D17D18S0S1S2S3S4S5S6S7S8S9S10S11S12S13S14S15S16S17S18S19S20S21S22S23S24S25S26S27S28S29S30S31S32S33S34S35S36

- (A) MD_1↓
- (B) IF_s2s↓
- (C) IF_s2t↑

0.99

0.97

0.96

###### MIN

0.94

0.92

D0D1D2D3D4D5D6D7D8D9D10D11D12D13D14D15D16D17D18S0S1S2S3S4S5S6S7S8S9S10S11S12S13S14S15S16S17S18S19S20S21S22S23S24S25S26S27S28S29S30S31S32S33S34S35S36

0.96

MAX

0.95

0.94

0.92

0.91

D0D1D2D3D4D5D6D7D8D9D10D11D12D13D14D15D16D17D18S0S1S2S3S4S5S6S7S8S9S10S11S12S13S14S15S16S17S18S19S20S21S22S23S24S25S26S27S28S29S30S31S32S33S34S35S36

- Figure 9: Comparison of dragging effectiveness across feature manipulation layers in the FLUX model. The analysis is evaluated on ReD Bench, demonstrating that double-stream layers (prefixed with “D”, in blue) consistently outperform single-stream layers (prefixed with “S”, in purple).

tain too few semantically informative features, where optimization tends to struggle in preserving source identity and achieving precise dragging control. Taken together, these comparisons suggest that our default setting, by using “DOUBLE-17” and “DOUBLE-18”, offers a favorable balance: it preserves stable feature representations with sufficient semantic and spatial information, while also maintaining rich identity-related features to support effective editing. This justifies its adoption as the backbone choice in DragFlow.

Layer-based Performance Comparison. To further validate the effectiveness of our selected feature manipulation layers, we conducted an extensive ablation study focused on layer-wise feature selection. Specifically, under the same configuration, we compute the loss each time using the features from a particular layer. This procedure was repeated across all 57 layers from both the doublestream and single-stream branches of the FLUX model, enabling a systematic comparison of editing behaviors across various layers.

The results are summarized in Fig. 9. We evaluate three metrics of dragging effectiveness: MD1,

- IFs2s, and IFs2t. As shown, operations performed on the double-stream layers (i.e., layers prefixed with “D”) consistently outperform those on the single-stream layers (i.e., layers prefixed with “S”) under the same dragging configuration. Notably, the 17th and 18th double-stream blocks (i.e., D17 and D18), which we ultimately adopt in our framework, exhibit the optimal performance robustly across all these metrics. This indicates that our chosen layers yield the highest-effectiveness solutions in terms of spatial displacement precision, preservation of post-dragging local features, and effective suppression of original-location features.

Affine Transformation Steps. During the dragging process, the motion from the source region to the target region is primarily executed over the first 50 steps (k = 0 to 49), where each iteration progressively applies affine transformations to the latent representations, thereby facilitating the drag procedure through gradient updates. After this stage, DRAGFLOW performs an additional 20 optimization steps (k = 50 to 69) by repeating the final affine transformation, which further refines the feature expression of the dragged object, helps it better adapt to the new semantic context, and enhances the consistency between the pre- and post-dragging regions.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

###### DeformationRelocationRotation

[Figure 274]

[Figure 275]

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

Task Preview

k=0 k=9 k=19 k=29 k=39 k=49 K=50 ~ K=69

- Figure 10: Region operation masks M(ik) created by progressive affine transformations at each step k across multiple subtasks. Each dragging process consists of 50 steps (k = 0 to 49), followed by 20 additional steps (k = 50 to 69) that repeat the final motion iteration to further refine the feature quality of the post-dragging region.

As illustrated in Fig. 10, the patch operation mask M(ik), obtained through progressive transformations, evolves smoothly with increasing k, thereby driving the gradient optimization process to

ensure effective dragging.

C.6 QUANTITATIVE ANALYSIS FOR DIT SUPERVISION GRANULARITY

- Table 4: Quantitative comparison on DragBench-DR between Point-based and Region-based strategies for the DiT architecture. Both methods utilize the same backbone and auxiliary modules.

Image Fidelity Mean Distance IFbg ↑ IFs2t ↑ IFs2s ↓ MD1 ↓ MD2 ↓

Attempts

Point-based variant 0.961 0.943 0.958 37.02 6.94 Region-based variant (DragFlow) 0.969 0.948 0.941 31.59 5.93

To further validate the analysis in Sec. 3.1, that DiTs’ fine-grained features render sparse point supervision ineffective, we conducted a controlled quantitative comparison.

By evaluating two distinct supervision paradigms: a baseline Point-based variant (i.e., applying standard point to the DiT features) and our proposed Region-based variant (DragFlow). To ensure a rigorous evaluation of the supervision mechanisms in isolation, both variants were equipped with identical auxiliary modules, specifically Background Preservation and Adapter-Enhanced Inversion.

The comparative results are summarized in Tab. 4. These quantitative findings empirically corroborate our assertion that sparse point tracking is insufficient for the high-frequency feature landscape of DiTs, whereas region-level affine supervision effectively bridges this gap to fully unleash the model’s generative potential.

- D ADAPTIVE INPUT PROCESSING

To ensure robust and user-aligned editing, DragFlow incorporates an adaptive input processing pipeline that systematically addresses three key challenges. First, when multiple operations are applied to a single image, we introduce an adaptive weighting scheme to balance region-level optimization and prevent dominance by large areas (see Appendix D.1). Second, to protect uneditable content during optimization, we design an adaptive gradient mask generation strategy that provides strict spatial constraints while maintaining flexibility for complex transformations (refer to Appendix D.2). Finally, to further reduce user effort and improve precision, we leverage a multimodal large language model (MLLM) to automatically generate candidate prompts and semantic tags, which are subsequently refined through minimal human intervention (detailed in Appendix D.3).

Together, these components form a cohesive input processing framework that enhances both the accuracy and practicality of DragFlow in real-world interactive editing scenarios.

- D.1 REGION WEIGHT REGULARIZATION FOR MULTI-OPERATIONS

When multiple drag operations are specified for a single image (i.e., N > 1), the loss function in Eq. 1 incorporates weighting coefficients {γi}Ni=1 to balance the influence of each region, preventing larger regions from dominating the optimization. These weights are computed adaptively based on the relative sizes of the manipulated regions, ensuring equitable gradient contributions.

Formally, for each operation i, let Si denote the relative size of the source mask Mi(0), defined as:

Si =

Mi(0) |Mi(0)|

, (22)

where Mi(0) is the number of non-zero (unmasked) pixels, and |Mi(0)| is the total number of elements in the mask. Under this setup, a raw weight wi can then be calculated as:

wi = clamp 1.0 +

0.5 Si + 0.1

, 1.0, 5.0 , (23)

where clamp(·,a,b) restricts the value to the interval [a,b]. This formulation assigns higher weights to smaller regions to amplify their impact. And the total raw weight sum is W = Ni=1 wi. The normalized weights are:

γi =

 



1 N

if W = 0,

wi W

otherwise.

(24)

If N = 1, we set γ1 = 1 directly. This adaptive scheme ensures Ni=1 γi = 1, promoting balanced optimization across diverse region scales.

- D.2 ADAPTIVE GRADIENT MASK GENERATION

To sufficiently protect uneditable regions in our procedure, we generate an adaptive bounding mask B that only uncovers the edited areas. This mask B is derived from the user-provided source masks

{Mi(0)}Ni=1 and their target points {ti}Ni=1, providing a static envelope for feature preservation during optimization. It differs from the iterative patch masks Mi(k) by serving as a holistic boundary for the entire dragging process.

Compare to Mask Loss. Compared to the conventional mask loss strategy, gradient masking offers a more precise mechanism for constraining gradient flow, thereby preventing optimization from introducing unintended perturbations into uneditable regions. A key limitation of mask loss is that it can only partially restrict future updates, while inadvertently retaining undesired changes that have already been introduced. In contrast, gradient masking directly blocks gradient propagation to these regions, ensuring that they remain unaffected throughout the optimization process.

Automatic Mask Creation. For each source mask Mi(0), we first apply the appropriate affine transformation (as defined in Eq. 3.2) with full interpolation (i.e., k=K) to obtain the final target

mask Mi(K) = Ω(Mi(0),ξi(K)), where ξi(K) = ti − bi for relocation/deformation or (∠biaiti,ai) for rotation.

Next, for each i, we compute the union mask Ui = Mi(0) ∪ Mi(K). To ensure comprehensive coverage, we enclose Ui with a minimal rotated bounding rectangle. The points in Ui are collected as Pi = {p | Ui(p) = 1}. The rectangle parameters are then derived as

###### (ci,(wi,hi),ϕi) = minAreaRect(Pi), (25)

where ci is the center, (wi,hi) are the dimensions, and ϕi is the rotation angle. The vertices of this rectangle are obtained via

Vi = boxPoints((ci,(wi,hi),ϕi)). (26)

Since multiple operations may exist, each yielding an independent rectangle, we merge them into a single adaptive mask B by filling all convex polygons from an image-shape initiated canvas ∅ :

N

fillConvexPoly(∅,Vi). (27)

B =

i=1

Finally, B is binarized to [0,1], ensuring no excessive expansion while covering all transformed regions for effective integration into the latent optimization.

- D.3 LEVERAGING MLLM FOR PROMPT AND TAG GENERATION

In practical usage, the DragFlow framework provides a user-friendly interactive interface, enhanced by a multimodal large language model (MLLM) to support intuitive and precise image editing. The system emphasizes high automation to reduce user effort, while a two-step confirmation workflow ensures that user intentions are accurately communicated and ambiguities are minimized.

The interaction begins with the user providing an input image and roughly indicating the target region via a simple scribble, along with a click specifying the desired target position. The system converts the scribbled region into an initial operational mask representing the affected area. Leveraging an automatic mask generation algorithm (Appendix D.2), users do not need to redraw the target region for task specification. The original image and operational mask are then processed by the MLLM, which infers the user’s editing intent and proposes a set of candidate prompts. Specifically, the MLLM generates ten potential prompts paired with corresponding task classes, from which users select the one that best reflects their intended operation.

After confirming the operational intent and task class, the interface produces a preview of the expected outcome based on the current operation parameters. For rotation operations, users specify an additional anchor point as the rotation center, which can be interactively adjusted while observing real-time updates in the preview. This iterative adjustment allows for fine-grained control, ensuring the final result closely aligns with user expectations. By combining automated inference with user-in-the-loop refinement, this design streamlines the editing process while faithfully translating high-level user intentions into precise image manipulations.

Here we present the prompt used for the prompt and tag generation procedure:

Refer to the original image, and the “dragged” image with the blue starting region, estimated green target region, and the arrow direction. You need to describe the content and the object for editing of the picture in English, in terms of “background details” and “editing changes”. Then you should guess the editing intents from the user by selecting one label for each answer, where the label classes have {relocation, deformation, rotation}.

Your tasks:

- - (a) You should first provide a detailed description about the original image (e.g., include, but are not limited to objects, spatial relationship, color, style, structure). Then try to describe the motion/editing in short words.
- - (b) You should provide the ten most possible guesses about the static condition of the after-dragged image, and at most 60 words for each. See if you can provide more details to facilitate the editing.

- D.4 EFFECTIVENESS OF MLLM-DRIVEN INTENT PARSING

To quantitatively assess the contribution and robustness of the MLLM-driven intent parsing module, we conducted an extensive ablation study on the DragBench-DR dataset. This analysis compares the performance of DragFlow under three distinct experimental scenarios:

- Table 5: To validate the effectiveness of prompts automatically generated by MLLM, we conducted an additional test. First, we examined whether using accurate and matching prompts is effective for drag operations. We considered three scenarios: Null Prompt, Incorrect Prompt (i.e., intentionally misinterpreted prompts), and Matched Prompt generated by MLLMs based on image and operation inputs. For the last scenario, we examine the effectiveness of two MLLMs (GPT-5 vs. QWen-VL). This experiment was conducted and scored on DragBench-DR.

Image Fidelity Mean Distance IFbg ↑ IFs2t ↑ IFs2s ↓ MD1 ↓ MD2 ↓

DragFlow w/

Null Prompt 0.966 0.944 0.948 33.81 6.81 Incorrect Prompt 0.955 0.936 0.955 36.87 7.73 Matched Prompt (by QWem-VL) 0.968 0.947 0.945 32.42 6.25 Matched Prompt (by GPT-5) 0.969 0.948 0.941 31.59 5.93

- • (1) Null Prompt: The optimization is driven solely by image and mask inputs, removing the text guidance entirely;
- • (2) Incorrect Prompt: The model is provided with intentionally misinterpreted prompts (e.g., describing a rotation task as deformation) to simulate worst-case MLLM failure;
- • (3) Matched Prompt: The standard configuration using prompts generated by MLLMs (comparing both GPT-5 and QWen-VL) based on the image and operation inputs.

The averaged quantitative results are presented in Tab. 5. As evidenced by the data, the framework remains robust even without textual guidance (i.e., Null Prompt), demonstrating the efficacy of the core region-level affine supervision.

In contrast, including Matched Prompts provides a valuable performance boost, whereas Incorrect Prompts degrade results due to conflicting guidance. However, thanks to the “Double Confirmation” workflow (detailed in Appendix D.3) we implemented, it helps prevent such misinterpretations in practice: rather than autonomous execution, the system requires users to select the optimal intent from generated candidates before optimization, ensuring the process is driven strictly by accurate, user-validated instructions.

- E CRITERION DETAILS

To systematically evaluate the effectiveness of dragging operations, we introduce a set of quantitative criteria that assess both visual fidelity and spatial accuracy. These criteria fall into two complementary families: Image Fidelity (IF), which measures the extent to which semantic content, source regions, and background integrity are preserved or altered as intended (detailed in Appendix E.1); and Mean Distance (MD), which provides geometric and feature-based assessments of the dragging process (see Appendix E.2). Together, these metrics capture different yet complementary aspects of editing quality, enabling a fair and comprehensive evaluation of diverse dragging approaches.

- E.1 COMPUTATION OF IMAGE FIDELITY (IF)

- IFs2t Evalutaion. The identity fidelity from the source to the target (i.e., IFs2t) evaluates how well original source features are preserved in the target regions after moving, promoting semantic

consistency in dragged content. A higher IFb2t signifies superior fidelity, indicating minimal perceptual loss during the feature transfer. For each region i, we first affine-transform the masked original image to align with the target configuration using the full interpolation parameter of k = K:

xaff = Ω Mi(0) ⊙ x, ξi(K) , (28)

followed by masking both sides with the target region mask Mi(K). Then the score is averaged as:

1 N

IFs2t = 1 −

N

LPIPS Mi(K) ⊙ xaff, Mi(K) ⊙ x′ . (29)

i=1

|[Figure 290]<br><br>Regpurple => ( Mi(K) ⊙ xaff )|
|---|

|[Figure 291]<br><br>Regred => ( Mi(K) ⊙ x’ )|
|---|

|[Figure 292]|
|---|

IFs2t = 1 – LPIPS(

vs.

)

|vs.<br><br>[Figure 293]<br><br>Regpurple => ( Mi(0) ⊙ x )|
|---|

|[Figure 294]<br><br>Regpurple => ( Mi(0) ⊙ x’ )|
|---|

Initial Region Mask [Mi(0)]

IFs2s = 1 – LPIPS( vs.

)

|[Figure 295]|
|---|

|vs.<br><br>[Figure 296]<br><br>Regbg => ( (1-B) ⊙ x )|
|---|

|[Figure 297]<br><br>Regbg => ( (1-B) ⊙ x’ )|
|---|

###### IFbg = 1 – LPIPS(

vs.

###### )

Target Region Mask [Mi(K)]

|[Figure 298]|
|---|

|[Figure 299]<br><br>[Figure 300]|
|---|

|[Figure 301]|
|---|

[Figure 302]

###### Gradient Mask [B]

###### Source Image [x]

###### Affined Source Image [xaff]

Dragged Image [x’]

(Black => Uneditable Region [1-B])

(w/ task preview)

(w.r.t., target centroid)

- Figure 11: For IFs2t and IFs2s, the criterion computation considers only the feature discrepancies within the labeled blocks. In IFs2t, the purple region on the left image denotes (Mi(K) ⊙ xaff), corresponding to the source region of (Mi(0) ⊙x), while the red patch on the right image represents the post-drag target (Mi(K)⊙x′). By contrast, the criterion IFs2s compares the same purple original

region Mi(0) across two images: the source x (left) and the dragged result x′ (right). Lastly, IFbg evaluates all uneditable areas, as indicated by the black areas on the gradient mask as (1 − B).

IFs2s Evalutaion. To assess the extent to which original features are effectively removed from the source regions after editing, we define the identity fidelity from source region to source region

(i.e., IFs2s), which measures the perceptual dissimilarity between the source region features in the original image x and the edited image x′. A lower IFs2s indicates better performance, as it reflects greater divergence and implies successful “moving-out” of the selected features. Formally, for each source region i, we compute the LPIPS distance on region-masked image tensors and get the final mean score over all task regions via

N

1 N

LPIPS Mi(0) ⊙ x, Mi(0) ⊙ x′ , (30)

IFs2s = 1 −

i=1

where ⊙ indicates the element-wise multiplication, and Mi(0) indicates the original region mask (i.e., the operation region given by the user);

IFbg Evalutaion. To ensure background preservation outside edited regions, we introduce the background identity fidelity IFbg, quantifying feature consistency in protected areas defined by the complement of the adaptive gradient mask B. A higher IFbg denotes better integrity, with minimal changes to non-targeted zones. Using the protection mask (1 − B), we yield the score:

IFbg = 1 − LPIPS((1 − B) ⊙ x, (1 − B) ⊙ x′). (31)

To aid interpretation, Fig. 11 presents a visual example demonstrating how the three proposed criteria are applied in practice.

- E.2 COMPUTATION OF MEAN DISTANCE (MD)

MD1 Implementation. We implement MD1 following the existing criteria established by Xia et al. (2025). MD1 enables precise feature matching within the uneditable region, allowing us to

validate the effectiveness of the dragging procedure by measuring the distance between the centroid of the source feature region and its most similar corresponding feature.

MD2 Implementation. Building upon the original MD2 design proposed in Lu et al. (2024a), we introduce an enhanced version that provides more precise and informative feedback for region-based drag operations. Unlike the original method, which computes feature matching distances based on manually annotated sample points, our approach automatically evaluates feature differences around the centroid scope of the pre- and post-drag regions. By leveraging this centroid-based formulation, we eliminate the inaccuracies and subjective biases inherent in manual annotations and also ensure a more consistent metric for assessing the effectiveness of various dragging strategies. This improvement allows for a more faithful reflection of the actual feature transformations induced by the dragging process and facilitates fairer comparisons across different methods.

- F ADDITIONAL BASELINE INFORMATION

Here we provide the official project pages for the baseline methods used in our comparisons. All implementations follow the default configurations and instructions provided by their authors:

- 1 CLIPDrag: https://github.com/HKUST-LongGroup/CLIPDrag
- 2 DragDiffusion: https://github.com/Yujun-Shi/DragDiffusion
- 3 DragLoRA: https://github.com/Sylvie-X/DragLoRA
- 4 DragNoise: https://github.com/haofengl/DragNoise
- 5 FastDrag: https://github.com/XuanjiaZ/FastDrag
- 6 FreeDrag: https://github.com/LPengYang/FreeDrag
- 7 GoodDrag: https://github.com/zewei-Zhang/GoodDrag
- 8 InstantDrag: https://github.com/SNU-VGILab/InstantDrag
- 9 RegionDrag: https://github.com/Visual-AI/RegionDrag

And some encapsulated modules applied in our framework or experiments:

- 1 InstantCharacter: https://github.com/Tencent-Hunyuan/InstantCharacter
- 2 SD3.5-Large-IP-Adapter: https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter

- G BENCHMARK DETAILS

- G.1 FORMATION OF THE ReD BENCHMARK

To evaluate model performance on the regional drag-based image editing task, we introduce a new benchmark, the Regional-based Dragging (ReD) Bench, consisting of 120 images annotated with precise drag instructions at both point and region levels. Each manipulation in the dataset is associated with an intention label, selected from relocation, deformation, or rotation.

For every image, we provide two complementary instruction sets corresponding to point-based and region-based dragging. The region-based annotations are supplied as multiple PNG masks, with each region uniquely represented by its centroid for cross-reference. The drag annotations include multiple start-to-target point pairs, which can be directly aligned with the region annotations, ensuring consistency in task intention. Additionally, we provide background prompts and editing intention prompts for each image to facilitate multimodal tasks, along with masks generated using the DragFlow automatic masker Appendix D.2. These design choices enable a more faithful representation of user intents underlying the provided drag instructions.

- G.2 ADOPTION OF THE DragBench-DR BENCHMARK

To further assess the effectiveness of DragFlow on a broader spectrum of images, we adapt and evaluate it on DragBench-DR (Lu et al., 2024a). DragBench-DR extends the classic point-based

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

SAMPLE(A)SAMPLE(B)

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Task Preview Original Image Operation Mask

- Figure 12: Real samples from ReD Bench: the first colume shows the dragging preview, where the green region is estimated from the user-specified target centroid; the second colume presents the source images, while the third colume highlights the user-marked operation regions in the form of masks, which may include multiple valid regions; and the last colume depicts the adaptive masks generated (detailed in Appendix D.2). Other matched instructions are provided in Appendix G.3.

dragging benchmark DragBench to region-based operations. Unlike the original benchmark, which relies on sparse point guidance, DragBench-DR formulates edits over regions, thereby providing a clearer reflection of user intentions. For evaluation, the accompanying metrics compare the predrag source region with the post-drag target region by computing differences over pre-annotated correspondence points. Despite this extension, as noted by the authors, DragBench-DR remains consistent with its point-based counterpart, while more effectively capturing region-level semantics in interactive editing tasks.

While DragBench-DR extends the benchmark to region-based operations, its evaluation protocol (MD2) still relies on point comparisons: differences are computed between pre- and post-drag regions using pre-annotated correspondence points. This design can introduce mismatches, as regionlevel edits are not faithfully captured by sparse point feature correspondences, leading to potentially unfair assessments for region methods. To better align the evaluation with region-based editing and integrate existing datasets into our experiment, we update the feature comparison criterion by replacing point-based annotations with an automatic centroid-based formulation (see Appendix E.2).

- G.3 DEMONSTRATION OF DATA SAMPLES

We present two real data samples (i.e., SAMPLE (A) and SAMPLE (B)) from the ReD Bench. The corresponding instructions are provided as follows, and the images are shown in Fig. 12.

- 1 { % SAMPLE (A) %

- 2 "region_operations": {

- 3 "0": {

- 4 "task": "rotation",

- 5 "centroids": [[337, 175], [379, 179]],

- 6 "anchors": [351, 256]

- 7 }

- 8 },

- 9 "point_operations": {

- 10 "begin_points": [[326, 111], [342, 190]],

- 11 "target_points": [[400, 116],[376, 198]]

- 12 },

- 13 "background_prompt": "From a rear view, a student in a blue denim jacket raises their hand in a classroom. Wooden desks, large windows (letting in light), and a distant teacher form the backdrop. The scene captures an engaged learning moment, with a realistic, observational style.",

- 14 "editing_prompt": " The student in a blue denim jacket moves his arm rightward, with his hand closer to the right side on this image."

- 15 }

- 1 { % SAMPLE (B) %

- 2 "region_operations": {

- 3 "0": {

- 4 "task": "deformation",

- 5 "centroids": [[251, 52], [357, 52]],

- 6 "anchors": null

- 7 },

- 8 "1": {

- 9 "task": "deformation",

- 10 "centroids": [[281, 200], [192, 195]],

- 11 "anchors": null

- 12 },

- 13 "2": {

- 14 "task": "deformation",

- 15 "centroids": [[221, 335], [307, 335]],

- 16 "anchors": null

- 17 }

- 18 },

- 19 "point_operations": {

- 20 "begin_points": [[284, 11], [244, 96], [280, 165], [287, 235], [243, 305], [244, 365]],

- 21 "target_points": [[392, 11], [356, 97], [193, 162], [199, 233], [332, 306], [335, 367]]

- 22 }

- 23 "background_prompt": "The image is an aerial view of a coastal scene. There’s a beach with light - colored sand between a dense forest (with green, yellow, and orange foliage) and a turquoise - blue

sea. The forest covers the left side, the beach runs along the middle, and the sea is on the right.",

- 24 "editing_prompt": "The top and bottom sections of the beach are narrowed to the outside, and the middle part is narrowed inside, altering the coastline shape to form a bay."

- 25 }

- H EXTRA QUALITATIVE RESULTS

- H.1 ADDITIONAL QUALITATIVE SAMPLES

In addition to the qualitative studies reported in the main experiments, we provide further examples in Fig. 15 and Fig. 16. These additional visualizations help to illustrate the advantages of our approach across diverse editing scenarios.

- H.2 CHALLENGING SCENARIOS

To further assess the robustness of DragFlow, we evaluate its performance under extreme conditions. As illustrated in Fig. 14, our analysis focuses on two distinct dimensions: (a) complex operational requirements and (b) unusual feature structures. Following this, in Subsec. H.3, we are also advised to provide some illustrations about the failure cases. We believe these parts may inspire and offer insights for future research about image drag-editing.

- (a) Cases with Complex Instructions. Fundamentally, we conceptualize the defined operations as elementary “building blocks.” Consequently, sophisticated instructions that involve composite movements (e.g., do both rotation and relocation) or non-affine warps (e.g., bending) are effectively executed by composing these basic units. Among the composite movements, we consider two scenarios, where multiple distinct operations are processed in parallel or in sequence. The lower section of Fig. 14 demonstrates how our framework effectively handles these complex instructions and overcomes the challenging scenarios.
- (b) Cases with Complex Structures. Regarding the non-rigid deformations inherent to complex textures such as hair and cloth, our empirical results confirm that the affine assumption remains valid.

Instruction Dragged Dragged

Instruction

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

Failure Scenarios Success Scenarios

- Figure 13: Extra qualitative results for failure scenarios. Left: The incorrect operation mask and failure outcomes; Right: The suitable operation mask with failure outcomes.

By applying flexible single operations or iteratively composing multiple blocks to approximate the target motion, the framework maintains high editing effectiveness and structural fidelity even on these intricate features. Representative samples validating this capability are presented in the upper part of Fig. 14.

- H.3 FAILURE SCENARIOS

Dependency on Input Precision. A primary limitation of the proposed framework is its reliance on the quality of user-provided inputs. Given that both the region-level affine supervision and the adaptive gradient masking mechanisms hinge directly on the initial mask annotation, ambiguous or inaccurate inputs can inevitably introduce generation artifacts.

Empirically, we observe that while the framework exhibits considerable tolerance for slightly oversized masks, ensuring complete coverage of the targeted semantic object is critical. Specifically, undersized or incomplete masks tend to compromise the editing quality. Fig. 13 illustrates this phenomenon by comparing outcomes derived from precise versus careless annotations. To relax these strict precision requirements, future iterations could explore integrating auxiliary techniques such as superpixel segmentation or implementing error-tolerant peripheral buffer zones to robustly handle manual variances.

- I RUNTIME AND MEMORY COMPARISON

To provide a comprehensive understanding of DragFlow’s practical performance, especially as an optimization-based method, a quantitative analysis of its computational cost is necessary. This discussion is intended to help users understand the quality versus speed trade-offs inherent in the approach. Tab. 6 presents a detailed comparison of the average inference time and GPU memory consumption for DragFlow alongside various baseline methods. The results were recorded as an average over all samples contained in DragBench-DR using one NVIDIA H100 GPU.

In our implementation, we applied qint8 quantization to the FLUX backbone to accelerate inference and reduce memory usage during the optimization. Additionally, by enabling CPU Offloading, the framework is capable of running on a consumer-grade 24 GB GPU. Beyond the current level, we believe there is still plenty of room to further reduce memory costs and increase speed through future improvements in both the algorithm and the code.

As the data illustrates, DragFlow’s resource requirements are higher than the baseline methods. This is an anticipated result and a direct consequence of the underlying model architecture. DragFlow is the only method in this comparison engineered to operate on the FLUX.1 model, a high-parameter, DiT-based architecture. In contrast, all other baselines are built upon the significantly smaller and less complex Stable Diffusion models.

- Table 6: This table collects and summarizes the inference time and memory consumption depending on different image drag-editing methods. Among them, DragFlow, as the only DiT-based dragger, shows the highest resource demand due to its more complex backbone architecture. The results are recorded as averages on DragBench-DR.

Methods Prep + Edit Time (s) Peak Memory (GB)

RegionDrag 9.2 4.3 FastDrag 5.4 5.2 InstantDrag 1.2 4.1 DragLoRA 58.6 14.6 FreeDrag 113.2 13.2 DragNoise 103.4 12.2 GoodDrag 104.6 12.8 CLIPDrag 101.0 19.4 DragDiffusion 82.1 12.1

DragFlow 158.7 23.5

- Table 7: The editing performance of our DragFlow framework on different DiT-based generative priors (SD3.5 vs. FLUX.1). The results are averaged on DragBench-DR.

Image Fidelity Mean Distance IFbg ↑ IFs2t ↑ IFs2s ↓ MD1 ↓ MD2 ↓

Method

DragFlow (based on Stable Diffusion 3.5) 0.962 0.945 0.952 35.21 6.58 DragFlow (based on FLUX.1-dev) 0.969 0.948 0.941 31.59 5.93

- J FRAMEWORK AND MODULE GENERALIZABILITY

- J.1 GENERALIZABILITY OF DRAGGING FRAMEWORK

To validate the generalization capability of DragFlow beyond the specific architecture of FLUX, we conducted an additional evaluation applying our framework to Stable Diffusion 3.5 (SD3.5), another prominent text-to-image model based on the DiT architecture. Notably, since the default InstantCharacter adapter is FLUX-specific, we employed SD3.5-Large-IP-Adapter to ensure a fair comparison. Tab. 7 presents the comparative editing performance averaged over all samples from the DragBench-DR dataset. As the results indicate, DragFlow maintains robust performance when transferred to the SD3.5 backbone.

Specifically, the SD3.5-based implementation achieves an MD1 of 35.21 and an IFs2t of 0.945, versus 31.59 and 0.948 of the FLUX edition, respectively. The FLUX-based version demonstrates a slight performance advantage, likely due to its substantially larger parameter count and enhanced generative prior, while the performance on SD3.5 remains competitive and consistent. These findings provide empirical evidence that the key contributions of DragFlow, including region-level affine supervision, gradient mask constraints, and adapter-enhanced inversion, are agnostic to the architecture used. The framework effectively generalizes across different DiT-based backbones, confirming that its efficacy is not limited to the specific training paradigm of FLUX but rather serves as a generalized solution for modern DiT generative models.

- J.2 GENERALIZABILITY OF ID-PRESERVATION DESIGN

To further evaluate the design choices behind our method, we conducted an experiment in which the IP-Adapter was replaced with a subject-specific LoRA. In this setup, a LoRA was first trained for the target subject and then applied during the drag-editing stage. Performance on DragBench-DR is summarized in Table 8.

The LoRA-based variant exhibits slightly stronger performance, which aligns with expectations since the LoRA is expressly optimized for the specific subject rather than functioning as a generic,

- Table 8: The editing performance of our DragFlow framework on different ID-preservation modules (IP-Adapter vs. LoRA). The results are averaged on DragBench-DR.

Image Fidelity Mean Distance IFbg ↑ IFs2t ↑ IFs2s ↓ MD1 ↓ MD2 ↓

Method

DragFlow (w/ LoRA) 0.970 0.949 0.940 30.98 5.88 DragFlow (w/ IP-Adapter) 0.969 0.948 0.941 31.59 5.93

open-domain adapter. However, this improvement comes with the overhead of training a new LoRA for each subject. In contrast, many modern text-to-image models now include high-quality, readily usable IP-Adapters, making the IP-Adapter approach more practical and broadly applicable in typical workflows.

- K LLM USAGE STATEMENT

We used large language models for text polishing and grammar correction during manuscript preparation. No LLMs were involved in the design of the method, experiments, or analysis. All content has been carefully verified and validated by the authors.

Instruction DragNoise GoodDrag FreeDrag InstantDrag RegionDrag OURS

| |[Figure 319]<br><br>[Figure 320]<br><br>[Figure 321]<br><br>[Figure 322]|[Figure 323]<br><br>[Figure 324]<br><br>[Figure 325]<br><br>[Figure 326]<br><br>[Figure 327]<br><br>[Figure 328]<br><br>[Figure 329]<br><br>[Figure 330]<br><br>[Figure 331]<br><br>[Figure 332]<br><br>[Figure 333]<br><br>[Figure 334]<br><br>[Figure 335]<br><br>[Figure 336]<br><br>[Figure 337]<br><br>[Figure 338]<br><br>[Figure 339]<br><br>[Figure 340]<br><br>[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]<br><br>[Figure 345]<br><br>[Figure 346]<br><br>|
|---|---|---|
|(Sequential)(Sequential)(Parallel)<br><br>|[Figure 347]<br><br>[Figure 348]<br><br>[Figure 349]|[Figure 350]<br><br>[Figure 351]<br><br>[Figure 352]<br><br>[Figure 353]<br><br>[Figure 354]<br><br>[Figure 355]<br><br>[Figure 356]<br><br>[Figure 357]<br><br>[Figure 358]<br><br>[Figure 359]<br><br>[Figure 360]<br><br>[Figure 361]<br><br>[Figure 362]<br><br>[Figure 363]<br><br>[Figure 364]<br><br>[Figure 365]<br><br>[Figure 366]<br><br>[Figure 367]<br><br>[Figure 368]<br><br>[Figure 369]<br><br>[Figure 370]<br><br>[Figure 371]<br><br>[Figure 372]<br><br>[Figure 373]<br><br>[Figure 374]<br><br>[Figure 375]<br><br>[Figure 376]<br><br>[Figure 377]<br><br>[Figure 378]<br><br>[Figure 379]|
| |[Figure 380]<br><br>[Figure 381]| |

Clothe[1]Clothe[2]Hair[1]Hair[2]

Composite[2b]Composite[2a]Composite[1]Bending[1]Bending[2]

###### ComplexStructures

ComplexInstructions

- Figure 14: Extra qualitative comparison for challenging scenarios, with complex feature structures (e.g., clothes and hair) or complex operational instructions (e.g., bending and composite operations). Two cases of composite operations are provided: the parallel and the sequential.

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

DragDiffusionInstantDragRegionDragDragNoiseDragLoRAGoodDragCLIPDragFreeDragFastDragOURSInstruction

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

- Figure 15: Extra qualitative comparison (Part 1 out of 2) of DragFlow with multiple baselines.

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

DragDiffusionInstantDragRegionDragDragNoiseDragLoRAGoodDragCLIPDragFreeDragFastDragOURSInstruction

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

- Figure 16: Extra qualitative comparison (Part 2 out of 2) of DragFlow with multiple baselines.

