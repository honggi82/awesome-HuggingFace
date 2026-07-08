# arXiv:2507.08396v2[cs.CV]1Feb2026

## CODI: SUBJECT-CONSISTENT AND POSE-DIVERSE TEXT-TO-IMAGE GENERATION

Zhanxin Gao1 Beier Zhu2 Liang Yao3 Jian Yang1 Ying Tai1∗ 1Nanjing University, 2Nanyang Technological University 3Vipshop zxgao@smail.nju.edu.cn yingtai@nju.edu.cn

ABSTRACT

Subject-consistent generation (SCG)—aiming to maintain a consistent subject identity across diverse scenes—remains a challenge for text-to-image (T2I) models. Existing training-free SCG methods often achieve consistency at the cost of layout and pose diversity, hindering expressive visual storytelling. To address the limitation, we propose subject-Consistent and pose-Diverse T2I framework, dubbed as CoDi, that enables consistent subject generation with diverse pose and layout. Motivated by the progressive nature of diffusion, where coarse structures emerge early and fine details are refined later, CoDi adopts a two-stage strategy: Identity Transport (IT) and Identity Refinement (IR). IT operates in the early denoising steps, using optimal transport to transfer identity features to each target image in a pose-aware manner. This promotes subject consistency while preserving pose diversity. IR is applied in the later denoising steps, selecting the most salient identity features to further refine subject details. Extensive qualitative and quantitative results on subject consistency, pose diversity, and prompt fidelity demonstrate that CoDi achieves both better visual perception and stronger performance across all metrics. The code is provided in https://github.com/NJU-PCALab/CoDi.

1 INTRODUCTION

While text-to-image (T2I) (Ramesh et al., 2022; Saharia et al., 2022; Rombach et al., 2022; Blattmann

- et al., 2023) models excel in high-quality image generation (Rombach et al., 2022; Mou et al., 2024), they struggle to maintain subject consistency across multiple scenes. Subject-consistent generation (SCG) aims to synthesize images of the same subject across diverse contextual prompts with three key objectives: (1) ensuring subject consistency across generated instances, (2) promoting layout and pose diversity across different instances to avoid repetitive or overly similar compositions, and (3) maintaining prompt fidelity to accurately reflect the semantics of each prompt. The capability enables numerous practical applications including multi-scene narrative for visual storytelling, customizable character design for animation and gaming, and coherent illustration sequences for graphic novels.

Current SCG methods (Kopiczko et al., 2024; Ye et al., 2023) primarily rely on training-intensive optimization (Avrahami et al., 2024) or mapping networks (Ruiz et al., 2024; Gal et al., 2023b) to bind subjects to latent representations. These approaches often require computationally expensive finetuning per subject or depend on domain-specific encoders, limiting scalability and generalizability. Training-free methods (Tewel et al., 2024; Zhou et al., 2024) have gained significant attention due to their elimination of parameter tuning, strong generalization capabilities, and broad compatibility with diverse diffusion architectures. Current training-free methods—ConsiStory (Tewel et al., 2024) and StoryDiffusion (Zhou et al., 2024)—enhance subject consistency by sharing self-attention keys and values across generated images. However, as noted in their limitations (Tewel et al., 2024; Hertz

- et al., 2024) and evident in Fig. 1, these methods often achieve high consistency at the cost of severely reduced layout and pose diversity, making it challenging to balance all three objectives.

To better balance the three objectives, we propose a training-free framework—subject-Consistent and pose-Diverse generation, dubbed CoDi—that achieves strong subject consistency while preserving diverse poses. Motivated by the progressive nature of diffusion models (Yue et al., 2024)—which

∗indicates corresponding author.

creating seasonal drink menus

arranging flowers …

braiding ribbons …

… asleep in a sunbeam on the window seat

roasting coffee beans

sourcing ethical coffee beans

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

VanillaSDXLConsiStoryStoryDiffusionCoDi(ours)

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | |BeU|tterpperRight| |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

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

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

(b) A delicate porcelain miniature of a thoughtful princess in lace.

(a) A warm cafe sketch of a skilled barista with espresso machine.

(c) Subject consistency vs. pose diversity

Figure 1: Comparison of subject-consistent generation methods: Vanilla SDXL (Podell et al., 2023), ConsiStory (Tewel et al., 2024), StoryDiffusion (Zhou et al., 2024) and CoDi (ours). (a&b) Existing methods sacrifice pose diversity for subject consistency, e.g., ConsiStory produces similar poses in

- Figure 1 (a); and the lower right with hands placed in front in Figure 1 (b). In contrast, CoDi generates consistent subjects, while matching the pose diversity of Vanilla SDXL. (c) Subject consistency vs. pose diversity. Current methods struggle to balance the two, whereas CoDi achieves both effectively.

shows that low-frequency attributes like pose and layout are formed in early denoising steps, while high-frequency details such as facial features emerge later—our CoDi adopts a two-stage strategy: Identity Transport (IT) and Identity Refinement (IR). During the early denoising steps, IT uses optimal transport to align each target image’s features with the reference identity features. Intuitively, this resembles mosaicking: assembling the subject using visual pieces from the reference image, rearranged to match the target pose—thus naturally preserving identity and keeping the original pose. In the later denoising steps, IR further refines subject consistency by guiding each target image to attend to the most salient identity attributes via cross-attention. As shown in Figure 1, CoDi achieves superior visual results in both subject consistency and pose diversity, and quantitatively demonstrates advantages over existing methods in balancing this trade-off.

We evaluate our method on the existing T2I SCG benchmark ConsiStory+ (Liu et al., 2025). Compared to other training-free approaches, both quantitative and qualitative results validate that our framework achieves better subject consistency while preserving richer layout and pose diversity. It demonstrates a superior trade-off among subject consistency, pose diversity, and prompt fidelity. Further analysis is also provided to demonstrate CoDi’s advantages in pose diversity.

- 2 RELATED WORK

To steer T2I generation with diffusion models (Rombach et al., 2022; Podell et al., 2023; Nan et al., 2025; Esser et al., 2024; Zhu et al., 2025a; Wang et al., 2025a; Hu et al., 2023; Lin et al., 2025a; Zhou et al., 2025; Lin et al., 2025b; Wang et al., 2025b), various methods have been proposed to incorporate control signals such as depth maps, edge maps, and segmentation (Mei et al., 2025; Zhang et al., 2023; Yang et al., 2023; Lei et al., 2025; Chen et al., 2025). Among them, subject consistency (a.k.a identity preservation) has attracted growing attention, aiming to generate a set of images conditioned on a specified subject. Existing subject-consistent generation (SCG) methods can be broadly categorized into two groups: training-based and training-free.

Training-based SCG. Training-based methods require either (1) fine-tuning on additional training data (Yang et al., 2024; Li et al., 2024; 2019; Betker et al., 2023; Liu et al., 2024) or (2) test-time optimization using reference images (Roich et al., 2022; Gal et al., 2023b; Kumari et al., 2023; Xiao et al., 2024). The first line of work, represented by StoryDALL-E (Maharana et al., 2022) and Make-AStory (Rahman et al., 2023), incorporates additional modules to capture subject information, followed by fine-tuning on large datasets to enable direct control over the subject given a reference image. The second line of work, exemplified by DreamBooth (Ruiz et al., 2023) and Textual Inversion (Gal

- et al., 2023a), optimizes model parameters or token embeddings on the given test images to inject subject identity. Despite their success in maintaining subject consistency, training-based methods suffer from high training costs or significant test-time latency. In contrast, our CoDi is training-free and introduces only mild additional latency.

Training-free SCG. Training-free methods circumvent the need for iterative tuning of model parameters. For instance, 1Prompt1Story (Liu et al., 2025) improves consistency by aligning prompt embeddings across generations. However, textual embedding control alone is insufficient to to enforce consistency, often resulting in subject drift. The current leading methods, ConsiStory (Tewel et al., 2024) and StoryDiffusion (Zhou et al., 2024), adopt attention-based mechanisms to promote subject consistency by sharing self-attention keys and values. However, as noted in their limitation discussions (Tewel et al., 2024; Hertz et al., 2024), applying attention across a set of images reduces pose diversity. To address this issue, our CoDi explicitly preserves diversity and promotes consistency by aligning early-stage features between the target and reference images via optimal transport.

3 METHOD

Our CoDi consists of two stages: Identity Transport (IT) and Identity Refinement (IR). Our IT operates in the early denoising stage to transport identity features from the reference image while preserving the pose and background of the target images. IR is applied in later denoising stages to refine subject consistency in fine-grained details. This two-stage design is inspired by (Yue et al., 2024), which reveals that low-frequency attributes such as pose and layout are determined early in the denoising timesteps, whereas high-frequency components like facial details emerge in later steps. We begin with the setup of subject-consistent generation (SCG), a review of attention-based SCG methods and a brief introduction of optimal transport.

- 3.1 PRELIMINARIES

Setup. SCG aims to synthesize a batch of images that share the same subject identity across diverse scenes. Formally, given a set of N textual prompts {tn}Nn=1, where each prompt is composed of a shared identity prompt tid and a unique attribute prompt an, i.e., tn = [tid,an]. For instance, given t1 =“A hyper-realistic digital painting of a fairy giggling in a grove of enchanted crystals” and t2 =“A hyper-realistic digital painting of a fairy lost in a maze of giant sunflowers”, the identity prompt is tid =“a hyper-realistic digital painting of a fairy”, and the attribute prompts are a1 = “giggling in a grove of enchanted crystals” and a2 = “lost in a maze of giant sunflowers”. We refer to the image generated from the identity prompt tid as the reference image, denoted as xid. The objective is to generate target images {xn}Nn=1 that depict a visually consistent subject with xid, while capturing the scene-specific attributes described in an. See Figure 2 for a concrete example.

Review of cross-image attention SCG. The current leading training-free SCG methods, ConsiStory (Tewel et al., 2024) and StoryDiffusion (Zhou et al., 2024), adopt attention-based strategies that extend the standard self-attention to cross-image attention mechanism. Formally, let {Xn}Nn=1 denote the features of the target images {xn}Nn=1. For generating i-th image, standard self-attention first projects Xi to queries Qi, keys Ki, and values Vi, then compute

Zi = Attn(Qi,Ki,Vi) = softmax

QiKi⊤

√

d

Vi, (1)

where d is the feature dimension. Let ⊕ denote matrix concatenation. We compute the concatenated keys and values as K1:N = [K1 ⊕ ... ⊕ KN] and V1:N = [V1 ⊕ ... ⊕ VN], respectively. To enhance consistency, cross-image attention mechanism allows the feature of the i-th image, Xi, to attend to the values V1:N of other images using their corresponding keys K1:N.

Zi = Attn(Qi,K1:N,V1:N) = softmax

QiK1:⊤N

√

d

V1:N. (2)

While both SCG methods adopt cross-image attention, they differ slightly in implementation: ConsiStory (Tewel et al., 2024) limits attention to masked subject regions, whereas StoryDiffusion (Zhou

- et al., 2024) randomly samples tokens from all regions without subject constraints.

: a hyper-realistic digital painting of a fairy. : a hyper-realistic digital painting of a fairy giggling in a grove of enchanted crystals.

(a) Extract Subject Masks

(b) Compute OT Plan

[Figure 25]

[Figure 26]

[Figure 27]

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

| | | |
|---|---|---|
| | | |
| | | |

Diffusion

[Figure 33]

[Figure 34]

[Figure 35]

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

(c) Identity Transport (d) Identity Refinement

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 46]

[Figure 47]

[Figure 48]

Background Concat Multiplication Softmax

Early Denoising Steps Late Denoising Steps

- Figure 2: Illustration of our CoDi. (a) Extract subject masks (Mid and Mn) by averaging the imagetext cross-attention at the final denoising timestep for subject-related tokens (e.g., “fairy”). (b)

Compute the OT plan Tn using the cost matrix C and the probability masses a and b (detailed in Sec. 3.2). (c) Identity transport (IT) operates in the early denoising steps to transfer reference subject features to targe images in a pose-aware manner. (d) Identity refinement (IR) operates in the late denoising steps to refine subject details using selective cross-image attention mechanism.

As discussed in their limitations (Tewel et al., 2024; Hertz et al., 2024), attention-based methods significantly reduce layout and pose diversity. We conjecture that attending to a shared pool of keys and values entangles feature updates across images, implicitly aligning spatial layouts and poses. To mitigate this, prior work (Tewel et al., 2024) introduces components such as attention dropout and query blending. However, these additions increase computational overhead and still fail to recover pose diversity (as shown in Figure 1). In this paper, we draw inspiration from structural learning to simultaneously preserve subject consistency and pose diversity by transporting identity features to each target image via optimal transport.

Optimal transport. Optimal transport (OT) (Villani et al., 2008; Monge, 1781; Zhu et al., 2025b) provides a framework for measuring the distance between two distributions. Specifically, given two sets of support features {vm}Mm=1 and {un}Nn=1, we define two discrete distributions P and Q as:1

N

M

bnδ(un − x) (3)

amδ(vm − x), Q(x) =

P(x) =

n=1

m=1

where δ(·) denotes the Dirac function, and am, bn denote the associated probabilities that sum to 1, respectively. Let a = [a1,...,aM]⊤ and b = [b1,...,bN]⊤. Given a cost matrix C ∈ RM×N, where each entry C(m,n) denotes the transport cost between vm and un (typically defined by their similarity), the OT distance between P and Q is defined as:

⟨T,C⟩, s.t. T1M = a,T⊤1N = b, (4)

dOT(P,Q;C) = min T≥0

where T ∈ RM×N is the transport plan, with T(m,n) ≥ 0 representing the amount of mass moved from vm to un, ⟨,⟩ denotes the Frobenius inner product, 1M is M-dimensional all-one vector.

1We slightly abuse the notations x and N, which here do not refer to an image or the number of target images.

- 3.2 IDENTITY TRANSPORT

Our IT operates in the early denoising steps (e.g., the first 10 of 50 total steps) to independently transport identity features from the reference image xid to each target image xn for all n ∈ [1,N]. Our IT begins by extracting subject features from masked regions.

Extract subject features. Masking out background regions offers two benefits for subject consistency: it reduces background interference and computational cost by focusing on the subject alone. We adopt a similar strategy to that of previous methods (Hertz et al., 2023; Tewel et al., 2024), using image-text cross-attention to extract subject masks. Specifically, let Xid denote the features of the reference image xid generated from the identity prompt tid. When generating xid, we average the cross-attention maps at the final denoising timestep for subject-related tokens (e.g., “fairy”), followed by applying Otsu’s method (Otsu et al., 1975) to produce a binary mask Mid. This mask highlights the subject-relevant regions, from which we extract the subject features as:

Sid = Xid ⊗ Mid ∈ Rs

id×d (5)

where ⊗ applies the binary mask to retain subject features, sid denotes the number of ones in the binary mask Mid, and d is the feature dimension. Similarly, for each target image xn, we extract subject features as Sn = Xn ⊗ Mn ∈ Rs

n×d. The process is visualized in Figure 2 (a). Transport between Sid and Sn. Given the subject feature pairs Sid = [s1id,...,ss

id

id ]⊤ and Sn = [s1n,...,ss

nn]⊤, we first derive an optimal transport plan T that aligns the reference features set {siid}s

id

i=1 with the target features {sin}s

n

i=1 (See Figure 2 (b)). Using this plan T, we compose the target subject features by transporting features from the reference image. Intuitively, this process resembles mosaicking: we assemble the target subject using pieces from the reference image, rearranged to match the target pose. Since the visual pieces originate from the reference image, subject identity is naturally preserved. To solve the OT problem in Eq. (4), we first define the cost matrix C and the associated probability masses a and b.

Definition of the cost matrix C. The cost matrix is typically defined based on the pairwise distances between features: smaller distances imply lower transport costs. For a pair siid and sjn from final denoising step (where features contain minimal noise), the cost is defined as:

C(i,j) = 1 − cos(siid,sjn) = 1 −

siid⊤sjn ∥siid∥2∥sjn∥2

. (6)

Definition of the probability masses a and b. Intuitively, a = [a1,...,as

id

]⊤ represents the importance weights of the subject features, where a larger ai indicates that feature siid is more relevant to the subject tid. We reuse the average cross-attention maps for generating the subject-relevant mask as the feature importance and apply softmax function to ensure the sum i ai equals to 1. The importance weights b for the target subject features Sn are derived analogously.

With the cost matrix C and the probability masses a and b, we solve the OT plan Tn in Eq. (4) using network simplex algorithm (Orlin, 1997). With the derived Tn, the subject target features composed by reference subject features are computed as

SnOT = Tn⊤Sid. (7)

To form the final representation XnOT, we combine SnOT with the non-subject features (masked out by Mn) from Xn. The representation is then passed through the diffusion network to produce the output. The IT process is illustrated in Figure 2 (c).

- 3.3 IDENTITY REFINEMENT

The motivation behind this stage is that the IT module performs a coarse transport between Sid and Sn. However, since the binary subject masks are imprecise and the foreground of target images evolves during denoising—while our transport plan Tn remains fixed—further refinement of subject details becomes necessary.

Our IR operates in the later denoising steps (e.g., the last 40 of 50 total steps) to reinforce subject details in the target images. IR resembles cross-image attention-based SCG methods, except that

swaying barefoot on a cloudlit stage

floating in a dream bubble …

hovering mid-air while whispering …

analyzing samples under a microscope

writing a research paper

experimenting with chemicals

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Vanilla SDXL

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

ConsiStory

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

StoryDiffusion

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

1Prompt1Story

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

CoDi (ours)

(b) A soft dreamy illustration of a gentle lullaby singer in a night-blue dress.

(a) A focused scene of a scientist with a lab coat.

- Figure 3: Qualitative comparison among Vanilla SDXL (Podell et al., 2023), ConsiStory (Tewel et al., 2024), StoryDiffusion (Zhou et al., 2024), and 1Prompt1Story (Liu et al., 2025). ConsiStory and StoryDiffusion generate similar poses across examples, while 1Prompt1Story preserves pose diversity but struggles with subject consistency. In contrast, our CoDi achieves both.

each target image attend only to the most relevant reference features to avoid entangled feature update across target images. Specifically, to generate the n-th image, we first construct the concatenated keys and values as Kn,id = [Kn ⊕ Kid] and Vn,id = [Vn ⊕ Vid], respectively. The cross-image attention scores are compute as

An = softmax

QnKn,⊤id √

d

. (8)

For each query, we retain only the top-α attention scores of the reference tokens (i.e., Kid). Specifically, for each row Ai of A, we define the top-α index set Ii (see Appendix A for details) and zero out all other entries of Kid:

A˜ij =

Aij, if j ∈ Ii 0, otherwise

A˜i

and Aˆi =

. (9)

A˜ij

j∈Ii

The final cross-attention output is then computed as:

### Attnα(Qn,Kn,id,Vn,id) = AVˆ n,id (10)

This filtering mechanism ensures that only the most relevant identity features from the reference image contribute to the attention update. The IR process is demonstrated in Figure 2 (d).

- Table 1: Quantitative comparison of subject consistency, pose diversity and prompt fidelity. Best results are marked in bold. ↑ indicates higher is better, and ↓ indicates lower is better.

###### Subject Consistency Pose Prompt

Method

CLIP-I (↑) DINO-v2 (↑) DreamSim (↓) Diversity (↑) Fidelity (↑)

Vanilla SDXL (Podell et al., 2023) 0.8417 0.8010 0.3139 0.0772 0.9082 1Prompt1Story (Liu et al., 2025) 0.8627 0.8233 0.2959 0.0662 0.8814 ConsiStory (Tewel et al., 2024) 0.8751 0.8428 0.2336 0.0621 0.9148 StoryDiffusion (Zhou et al., 2024) 0.8776 0.8471 0.2356 0.0605 0.9038

CoDi (ours) 0.8809 0.8514 0.2136 0.0758 0.9041

- 4 EXPERIMENTS

- 4.1 SETUP

Benchmark. We evaluate our CoDi on the standard SCG benchmark, ConsiStory+(Liu et al., 2025), which comprises nearly 200 prompt sets and supports the generation of over 1,100 images. Each prompt set includes a subject described in a specific style, with multiple frame-specific descriptions.

Baselines and implementation details. We compare our CoDi with SoTA training-free SCG methods, including ConsiStory (Tewel et al., 2024), StoryDiffusion (Zhou et al., 2024) and 1Prompt1Story (Liu et al., 2025). We reproduce all baselines using their official released code. All methods are implemented using the same backbone model, Stable Diffusion XL 1.0 (Podell et al., 2023), with an image resolution of 1024 × 1024, except for StoryDiffusion, which is evaluated at 768 × 768 due to its high memory consumption, following its original setting. To ensure fairness, identical noise seeds are enforced for all methods, ensuring that each prompt is initialized with the same random noise input. Hyperparameter α in Eq. (10) selects the top 50% of reference features.

Evaluation metrics. Our evaluation framework assesses the quality of generated images from three aspects: (1) subject consistency, (2) pose diversity, and (3) prompt fidelity. Subject consistency is evaluated by computing the average pairwise cosine similarity (or distance) between image embeddings within each target image set. We use three image encoders for this evaluation: CLIPI (Hessel et al., 2021), DINO-v2 (Oquab et al., 2023), and DreamSim (Fu et al., 2023). To evaluate pose diversity, we extract 2D human joint coordinates using ViTPose’s pose estimation model (Xu

- et al., 2022). To eliminate global variations in translation, rotation, and scale, we align poses using Procrustes analysis (Schönemann, 1966), inspired by standard practices in face alignment (Lin et al., 2021). The pose diversity score is then computed as the average Euclidean distance between corresponding keypoints across aligned image pairs. A higher score indicates greater pose diversity. For prompt fidelity, we use CLIP-Score (Hessel et al., 2021) to measure the cosine similarity between image and textual prompt embeddings. See Appendix B for more details.

4.2 EXPERIMENTAL RESULTS

Qualitative comparison. As shown in Figure 3, our CoDi achieves superior visual quality in terms of pose diversity, subject consistency, and prompt fidelity. Our CoDi preserves the pose diversity of Vanilla SDXL (Podell et al., 2023) while overcoming its limitation in subject consistency. In comparison, ConsiStory (Tewel et al., 2024) and StoryDiffusion (Zhou et al., 2024) achieve subject consistency at the cost of pose diversity. For example, in the scientist scenario, the man exhibits nearly identical body poses. Although 1Prompt1Story (Liu et al., 2025) maintains strong layout and pose diversity in both cases, its subject consistency remains limited.

Quantitative comparison. Table 1 presents a quantitative comparison. (1) Across all three subject consistency metrics—CLIP-I (Hessel et al., 2021), DINO-v2 (Oquab et al., 2023), and DreamSim (Fu

- et al., 2023)—our CoDi achieves the best performance, demonstrating superior identity preservation across instances. In particular, our method obtains the lowest DreamSim score (0.2136), indicating closer alignment with human perceptual similarity than competing methods. (2) In terms of pose diversity, CoDi achieves the highest score (0.0758), closely matching Vanilla SDXL (0.0772). This demonstrates its ability to preserve the inherent pose diversity of the diffusion model while maintaining subject consistency. (3) For prompt fidelity, CoDi performs competitively—ranking

mapping underwater currents …

collecting glowing jellyfish

decorating underwater caves …

basking on a sunlit rock

playing with colorful fish

drifting through the ocean

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Vanilla SDXL

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

+ IT

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

+ IR

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

CoDi (+ IT + IR)

A dreamy underwater illustration of a beautiful mermaid with a shimmering tail.

Figure 4: Main component analysis (qualitative) on identity transport (IT) and identity refinement (IR). IT enhances subject consistency in the coarse-grained level and preserves pose diversity. IR enhances subject consistency in the fine-grained level reduces pose diversity. Their combination yields the best consistency and preserves diversity.

- Table 2: Main component analysis (quantitative) on identity transport (IT) and identity refinement (IR). IT enhances subject consistency and preserves pose diversity. IR enhances subject consistency while reduces pose diversity. Their combination yields the best consistency and preserves diversity.

Subject Consistency Pose Prompt

IT IR

CLIP-I (↑) DINO-v2 (↑) DreamSim (↓) Diversity (↑) Fidelity (↑) 0.8417 0.8010 0.3139 0.0772 0.9082

✓ 0.8576 0.8207 0.2707 0.0800 0.9090 ✓ 0.8859 0.8618 0.2044 0.0675 0.8975 ✓ ✓ 0.8809 0.8514 0.2136 0.0758 0.9041

second only to ConsiStory and comparable to Vanilla SDXL. These results demonstrate CoDi’s ability to achieve subject consistency without compromising pose diversity or prompt alignment.

- 4.3 ABLATION STUDIES

Main component analysis. The contribution of each module (IT and IR) to subject consistency and pose diversity are evaluated through quantitative and qualitative ablations, as shown in Table 2 and Figure 4. Table 2 shows that both IT and IR improve subject consistency, while IT also enhances pose diversity. However, using IR alone reduces pose diversity—for example, the score drops from 0.0772 to 0.0675 compared to the SDXL baseline. When both modules are applied, subject consistency further improves due to their synergistic effect, while pose diversity is preserved.

Figure 4 visualizes the effect of each module. Compared to Vanilla SDXL, applying IT preserves the original pose and improves subject consistency, but some details, such as facial identity, remain suboptimal. In contrast, IR alone enhances fine-grained consistency, but produces nearly identical poses across images, substantially reducing diversity. As shown in the bottom row, combining IT and IR improves both coarse and fine-grained consistency without compromising pose diversity.

0.080

0.080

0.8600

0.8600

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | |DIN|O-v2| | |Pose|Diversity| | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

0.075

0.075

0.8575

0.8575

0.070

0.070

0.8550

0.8550

PoseDiversity

PoseDiversity

0.065

0.065

DINO-v2

DINO-v2

0.8525

0.8525

0.060

0.060

0.8500

0.8500

0.055

0.055

0.8475

0.8475

0.050

0.050

0.8450

0.8450

0.045 DINO-v2 Pose Diversity

0.045

0.8425

0.8425

2 4 6 8 10 12 14 16 18 20 30

30 40 50 60 70

Transition point

#### Figure 5: Ablation studies on (a) stage transition point, and (b) the effect of α.

A magical winter illustration with contrasting colors of a brave girl and her fox companion wandering the snow, the fox leads her through the shadows of the trees; a quiet stillness blankets the frozen forest; the sun rises slowly over the frosted hills; the girl rests against the trunk of a frosty pine; the snow catches the pale morning light; the girl tosses a stick into the snow.

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Figure 6: Multi-subject generation. CoDi consistently preserves identities of multiple subjects while maintaining diverse poses and spatial layouts.

Study on stage transition point. Our CoDi adopts a two-stage strategy: identity transport (IT) in the early denoising steps and identity refinement (IR) in the later ones. By default, we set the stage transition point at step t = 10 out of a total of 50 denoising steps (IT is applied when t ≤ 10, and IR afterward). In this study, we investigate how the choice of transition point affects generation quality. As shown in Figure 5 (a), we vary t from 2 to 30 and evaluate subject consistency (DINO-v2) and pose diversity. We find that our default choice t = 10 achieves a favorable trade-off between consistency and diversity.

Effect of α. In the IR stage, we select the top-α percent of reference features to inject into the target subject features. In this study, we examine how varying α affects subject-consistent generation. Specifically, we vary α from 30% to 70% and report subject consistency (DINO-v2) and pose diversity in Figure 5 (b). We observe that increasing α improves subject consistency but reduces pose diversity. Setting α = 50% provides a favorable trade-off.

- 4.4 MULTI-SUBJECT GENERATION

Both IT and IR are independently applied to each subject and are easily extendable, enabling our CoDi to naturally support multi-subject generation. For each subject, we perform IT and extract its most salient features for IR , which effectively prevents feature interference across subjects and enhances subject consistency. As shown in Fig. 6, our CoDi preserves multi-subject consistency while maintaining their pose diversity.

- 4.5 DIFFERENT STYLE GENERATION

CoDi first transports the identity features from the reference image during the IT stage, and in the IR stage, the diffusion model refines the subject with a specific style. As shown in Fig. 7, CoDi generates images with consistent subject appearance and diverse styles.

- 4.6 USER STUDY

We conducted a user study to compare our method with state-of-the-art approaches. A total of 30 prompt sets were randomly sampled, each consisting of four fixed-length prompts. Thirty-nine participants were asked to evaluate which method demonstrated the best overall performance of the generated images in terms of subject consistency, pose/layout diversity, and prompt fidelity. As

minimalist, geometric abstraction … wearing a shimmering gown …

retro-futuristic neon glow style … strolling through a moonlit forest

whimsical, watercolor dreamscape style … riding a white horse …

steampunk mechanical style … playing with a pet horse in a secret garden

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

##### Identity Prompt: A beautiful princess with a kind smile

retro-futuristic neon glow style … jumping over a puddle

retro, comic book pop art style … wearing a collar

minimalist, geometric abstraction … sticking head out of the car

sleek and fast depiction … wearing a small sweater

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Identity Prompt: A puppy

Figure 7: CoDi can generate images with consistent subject appearance and different style.

- Table 3: User study with 39 participants evaluating T2I SCG methods based on human preference.

Method ConsiStory StoryDiffusion 1Prompt1Story CoDi (ours) Percent (%) ↑ 19.06 21.20 14.53 45.21

shown in Table 3, CoDi achieved the highest overall preference, surpassing the second-best method (StoryDiffusion) by 24.01%.

- 5 CONCLUSION

In this paper, we propose CoDi, a novel training-free framework that addresses the trade-off between subject consistency and pose diversity. CoDi comprises two key components: identity transport (IT) and identity refinement (IR). During early denoising steps, IT aligns features across instances by optimally transporting the identity subject’s features, while preserving pose diversity. IR further refines subject consistency by aligning instance features with the salient attributes of the identity subject in the later denoising steps. The effectiveness of our CoDi is demonstrated by its state-of-theart performance in achieving subject consistency and maintaining pose diversity.

ACKNOWLEDGMENTS

This work was supported by the Gusu Innovation and Entrepreneur Leading Talents: No. ZXL2024362, Natural Science Foundation of Jiangsu Province: BK20241198, and Natural Science Foundation of China: No. 62406135.

REFERENCES

Omri Avrahami, Amir Hertz, Yael Vinker, Moab Arar, Shlomi Fruchter, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. The chosen one: Consistent characters in text-to-image diffusion models. In ACM SIGGRAPH, 2024.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023.

Lifeng Chen, Jiner Wang, Zihao Pan, Beier Zhu, Xiaofeng Yang, and Chi Zhang. Detail++: Trainingfree detail enhancer for text-to-image diffusion models, 2025. URL https://arxiv.org/ abs/2507.17853.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In NeurIPS, 2023.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In ICLR, 2023a.

Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Designing an encoder for fast personalization of text-to-image models. arXiv preprint arXiv:2302.12228, 2(3), 2023b.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Promptto-prompt image editing with cross attention control. In ICLR, 2023.

Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In CVPR, 2024.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A referencefree evaluation metric for image captioning. In EMNLP, 2021.

Xiantao Hu, Bineng Zhong, Qihua Liang, Shengping Zhang, Ning Li, Xianxian Li, and Rongrong Ji. Transformer tracking via frequency fusion. TCSVT, 2023.

Dawid J Kopiczko, Tijmen Blankevoort, and Yuki M Asano. Vera: Vector-based random matrix adaptation. In ICLR, 2024.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023.

Mingkun Lei, Xue Song, Beier Zhu, Hao Wang, and Chi Zhang. Stylestudio: Text-driven style transfer with selective control of style elements. In CVPR, 2025.

Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, and Jianfeng Gao. Storygan: A sequential conditional gan for story visualization. In CVPR, 2019.

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In CVPR, 2024.

Chunze Lin, Beier Zhu, Quan Wang, Renjie Liao, Chen Qian, Jiwen Lu, and Jie Zhou. Structurecoherent deep feature learning for robust face alignment. IEEE Transactions on Image Processing, 30:5313–5326, 2021.

Kun-Yu Lin, Hongjun Wang, Weining Ren, and Kai Han. Panoptic captioning: An equivalence bridge for image and text. In The Thirty-Ninth Annual Conference on Neural Information Processing Systems, 2025a.

Xiaotong Lin, Tianming Liang, Jian-Fang Hu, Kun-Yu Lin, Yulei Kang, Chunwei Tian, Jianhuang Lai, and Wei-Shi Zheng. Coopdiff: Anticipating 3d human-object interactions via contact-consistent decoupled diffusion. arXiv preprint arXiv:2508.07162, 2025b.

Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. Intelligent grimm-open-ended visual storytelling via latent diffusion models. In CVPR, 2024.

Tao Liu, Kai Wang, Senmao Li, Joost van de Weijer, Fahad Shahbaz Khan, Shiqi Yang, Yaxing Wang, Jian Yang, and Ming-Ming Cheng. One-prompt-one-story: Free-lunch consistent text-to-image generation using a single prompt. In ICLR, 2025.

Adyasha Maharana, Darryl Hannan, and Mohit Bansal. Storydall-e: Adapting pretrained text-toimage transformers for story continuation. In ECCV. Springer, 2022.

Kangfu Mei, Hossein Talebi, Mojtaba Ardakani, Vishal M Patel, Peyman Milanfar, and Mauricio Delbracio. The power of context: How multimodality improves image super-resolution. In CVPR, 2025.

Gaspard Monge. Mémoire sur la théorie des déblais et des remblais. Mem. Math. Phys. Acad. Royale Sci., pp. 666–704, 1781.

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI, 2024.

Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. In ICLR, 2025.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

James B Orlin. A polynomial time primal network simplex algorithm for minimum cost flows. Mathematical Programming, 78:109–129, 1997.

Nobuyuki Otsu et al. A threshold selection method from gray-level histograms. Automatica, 11 (285-296):23–27, 1975.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, and Leonid Sigal. Make-a-story: Visual memory conditioned consistent story generation. In CVPR, 2023.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Transactions on graphics (TOG), 42(1):1–13, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In CVPR, 2024.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.

Peter H Schönemann. A generalized solution of the orthogonal procrustes problem. Psychometrika, 31(1):1–10, 1966.

Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon.

Training-free consistent text-to-image generation. SIGGRAPH, 2024. Cédric Villani et al. Optimal transport: old and new, volume 338. Springer, 2008. Ruoyu Wang, Ziyu Li, Beier Zhu, Liangyu Yuan, Hanwang Zhang, Xun Yang, Xiaojun Chang,

and Chi Zhang. Parallel diffusion solver via residual dirichlet policy optimization, 2025a. URL https://arxiv.org/abs/2512.22796.

Ruoyu Wang, Beier Zhu, Junzhi Li, Liangyu Yuan, and Chi Zhang. Adaptive stochastic coefficients for accelerating diffusion sampling. In NeurIPS, 2025b.

Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025.

Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. IJCV, 2024.

Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. Vitpose: Simple vision transformer baselines for human pose estimation. In NeurIPS, 2022.

Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683, 2024.

Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, et al. Reco: Region-controlled text-to-image generation. In CVPR, 2023.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Zhongqi Yue, Jiankun Wang, Qianru Sun, Lei Ji, Eric I Chang, Hanwang Zhang, et al. Exploring diffusion time-steps for unsupervised representation learning. In ICLR, 2024.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In CVPR, 2023.

Jiaming Zhou, Ke Ye, Jiayi Liu, Teli Ma, Zifan Wang, Ronghe Qiu, Kun-Yu Lin, Zhilin Zhao, and Junwei Liang. Exploring the limits of vision-language-action manipulations in cross-task generalization. In The Thirty-Ninth Annual Conference on Neural Information Processing Systems, 2025.

Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. In NeurIPS, 2024.

Beier Zhu, Ruoyu Wang, Tong Zhao, Hanwang Zhang, and Chi Zhang. Distilling parallel gradients for fast ode solvers of diffusion models. In ICCV, 2025a.

Xingyu Zhu, Shuo Wang, Beier Zhu, Miaoge Li, Yunfan Li, Junfeng Fang, Zhicai Wang, Dongsheng Wang, and Hanwang Zhang. Dynamic multimodal prototype learning in vision-language models. In ICCV, 2025b.

APPENDIX. This appendix presents supplementary materials that extend the methodological details, experimental evaluations, and analytical discussions introduced in the main body of the paper.

- A ADDITIONAL IMPLEMENTATION DETAILS

Extracting subject masks. We extract subject masks (Mid and Mn) by averaging the image-text cross-attention maps over all layers at the final denoising timestep, focusing specifically on subject-

related tokens. Let Qimg denote the keys of image features and Ksub the keys of the subject-related tokens. For each cross-attention layer l, the unnormalized attention weights are computed as:

Wl =

QimgKsub⊤

√

d

, (11)

where d is the feature dimension. We then average the attention weights across all L layers:

W =

1 L

L

l=1

Wl. (12)

We apply Otsu’s thresholding (Otsu et al., 1975) to obtain the binary subject mask M:

M = Otsu(W). (13)

Selection of the most salient identity features. Our IR refines target images using the most salient reference features, which are determined by the OT plan. Specifically, the saliency score of the i-th identity feature is computed as:

sOTi =

N

n=1

⟨Tn(i,:), 1 − C(i,:)⟩, (14)

The top-α index set Ii in Eq. 9 contains indices with the α highest saliency scores.

0.2 0.3 0.4 0.5 0.6 0.7 0.8

0.06

0.08

0.10

0.12

PoseDiversity

ConsiStory

StoryDiffusion 1Prompt1Story CoDi

- Figure 8: Pose diversity scores across different confidence thresholds τ. Our CoDi consistently outperforms other SCG methods by a clear margin under all τ settings.

- B ADDITIONAL EVALUATION DETAILS

Unified evaluation protocol. We adopt a unified evaluation protocol across all metrics. Specifically, for each target image set k with N generated images {xn}Nn=1, we compute the average pairwise evaluation score as follows:

N−1

N

2 N(N − 1)

f(xn,xj), (15)

uk =

n=1

j=n+1

where f(·,·) denotes the metric-specific similarity or distance function between two images, depending on the evaluation objective. The final evaluation score is then obtained by averaging uk over all

target image sets:2

K

1 K

uk. (16)

u =

k=1

Pose diversity score. We begin by extracting normalized 2D human keypoints and their confidence scores from each target image using ViTPose (Xu et al., 2022), a SoTA transformer-based model known for its high accuracy and robustness in human pose estimation. Each image x is represented by a set of H keypoint locations p and their confidences β.

### p = [(px1,py1)⊤,...,(pxK,pyK)⊤]⊤ ∈ RH×2, β = [β1,...,βK]⊤ ∈ RK (17)

where each keypoint pi = (pxi ,pyi ) is normalized by the image width and height and βi ∈ [0,1] denotes its confidence score. To ensure robustness, we discard keypoints with confidence scores

below a threshold τ. For a pair of target images xi and xj, we retain only the indices of keypoints that are valid in both images. We then perform Procrustes method (Schönemann, 1966) to remove

global variations in translation, rotation, and scale by aligning pi to pj. Specifically, we first compute the centroids of the keypoints which are denoted as µi and µj. We then center both keypoint sets by subtracting their respective centroids and normalize their ℓ2 norm:

pi − µi ∥pi − µi∥2

pj − µj ∥pj − µj∥2

(18) Next, we compute the optimal rotation matrix using singular value decomposition (SVD):

p¯i =

, p¯j =

### U, Σ, V⊤ = SVD(p¯⊤i p¯j), R = V⊤U⊤. (19)

The resulting R is an orthogonal rotation matrix that minimizes the Frobenius norm between the aligned keypoint sets, ensuring the best rigid alignment in the least-squares sense. The optimal scaling factor is given by:

γ = ∥p¯j∥2 ∥p¯i∥2

· tr(Σ). (20) The aligned keypoints are then obtained by applying the computed scale, rotation, and translation:

### pˆi = γ · p¯iR + µj. (21)

The pose diversity score between a pair of images xi and xj is computed as the average Euclidean distance between pˆi and pj. To analyze pose diversity under different confidence thresholds τ, we compare the pose diversity scores of various methods across a range of τ values. As shown in the Fig. 8, our CoDi demonstrates a clear advantage over other SCG methods across all τ settings. We use τ = 0.7 in our experiments to balance keypoint reliability and coverage.

- C LIMITATIONS

A heroic nature illustration of a centaur with the body of a horse and the torso of a warrior, drinking from a crystal-clear stream; forging weapons in a woodland forge; practicing swordsmanship in a field.

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

- Figure 9: Limitations. Our method relies on the quality of cross-attention from the pre-trained diffusion model to accurately localize the subject.

Similar to prior subject-mask-based methods such as ConsiStory (Tewel et al., 2024), our CoDi framework relies on cross-attention scores to extract subject masks and estimate image

2We slightly abuse the notations K, which here do not refer to keys in transformer.

Identity Prompt: A mystical and powerful illustration of A wise wizard with a long, ﬂowing beard

summoning creatures in a circle

brewing potions in a dark cave

wandering through a mystical forest

studying celestial maps under the stars

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

UNO

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

###### UNO(face)

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

CoDi

Figure 10: CoDi shows superior results compared to SOTA training models.

token importance in the OT Plan. Occasionally, the pre-trained diffusion model assigns higher attention to background regions than to the subject, as shown in Fig. 9, which hinders the effective transport of identity features Xid from the reference image xid to target image xn, resulting in subject inconsistency. However,such failures are rare in practice (under 5%) and can be solved by simply changing the seed.

- D INFERENCE TIME AND MEMORY USAGE

- Table 4: Inference Time and Memory Usage. We report the inference time (in seconds) and peak GPU memory usage on a single A6000 GPU for generating a set of five images from a prompt set with a resolution of 1024 × 1024. StoryDiffusion (Zhou et al., 2024) is excluded due to excessive GPU memory consumption beyond the A6000’s limit.

Method Inference time (s) GPU memory (GB)

Vanilla SDXL (Podell et al., 2023) 77.67 35.58 1Prompt1Story (Liu et al., 2025) 115.85 17.13 ConsiStory (Tewel et al., 2024) 113.88 46.60 CoDi (ours) 154.89 45.20

We measure the inference time and memory usage of different SCG methods on a single A6000 GPU, as shown in Table 4. We report the wall-clock time for generating a set of five images from a prompt set (since the baseline method ConsiStory (Tewel et al., 2024) performs cross-image attention across a batch of images) at a resolution of 1024 × 1024. Based on Table 4, our method CoDi exhibits slightly higher inference time (154.89s) and comparable GPU memory usage (45.20GB) relative to ConsiStory. While 1Prompt1Story (Liu et al., 2025) is the most memory-efficient, it compromises subject consistency. Note that we exclude StoryDiffusion (Zhou et al., 2024) due to excessive GPU memory usage beyond the A6000’s limit at 1024 × 1024 resolution (its original setting uses 768 × 768).

practicing painting in an art studio

sitting on a rooftop, watching the city lights

in an art gallery, admiring paintings

decorating her room with fairy lights

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

###### Reference Image

Vanilla SDXL

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

CoDi (Cosine)

Identity Prompt: A pixel art portrait of a girl with wavy locks and brown eyes

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

CoDi (OT)

- Figure 11: CoDi (Cosine) fails to transport the corresponding features when there is a significant difference between the reference and target images. In contrast, CoDi (OT) achieves overall structural alignment, thereby preserving pose diversity.

- E COMPARISON WITH SOTA TRAINING METHOD

We compare CoDi with UNO (Wu et al., 2025), as shown in Fig. 10. While UNO achieves subject consistency, its identical layout across outputs shows poor action prompt adherence and aesthetics. In contrast, CoDi demonstrates improved performance, highlighting the effectiveness of our trainingfree approach, which shows competitive results compared to SOTA company-level models.

- F THE EFFECTIVENESS OF OPTIMAL TRANSPORT

To assess the effectiveness of OT, we compare it with a simpler cosine similarity approach. CoDi (OT) achieves 0.0758, significantly outperforming CoDi (Cosine) at 0.0704 in pose diversity. As shown in Fig. 11, when there is a large feature difference between the reference and target images, especially across styles, CoDi (Cosine) tends to transport the mismatched features, failing to preserve the reference image’s pose. In contrast, CoDi (OT) effectively maintains the pose while enabling diverse variations, resulting in a failure to preserve the reference image’s pose. In contrast, CoDi (OT) effectively preserves the pose of the reference image while maintaining diverse pose variations.

- G GENERALIZATION TO DIT-BASED ARCHITECTURES

We adapt CoDi to DiT-based models (Flux), with the generation results shown in Fig. 13, demonstrating richer details, subject consistency, and pose diversity.

- H USER STUDY DETAILS

We conducted a user study comparing our method with state-of-the-art approaches, including ConsiStory, StoryDiffusion, and 1Prompt1Story. Thirty prompt sets, each containing four fixed-length prompts, were randomly sampled. Thirty-nine participants evaluated which method achieved the best overall image quality in terms of subject consistency, pose/layout diversity, and prompt fidelity.

Participants were instructed to select the set that best satisfied three evaluation criteria: subject consistency, pose/layout diversity, and prompt fidelity. Fig. 14 illustrates these criteria at the start

making a tense phone call while hurrying through the crowded city street

Identity Prompt: A fashion illustration style of a shorthaired man wearing a black suit and a red tie

2. talking with a passerby on the corner

1. walking down the street

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

3. handling paperwork at the office

4. feeling weary, he heads to the edge of a rooftop to take a break

5. walking back home at sunrise, reflecting on the day’s events

- Figure 12: CoDi can maintain both subject and clothing consistency while preserving pose diversity by adding the clothing description in the prompt and adjusting the subject mask threshold to include clothing in the foreground mask.

[Figure 155]

A sleek and fast depiction of a cheetah with sharp eyes, sprinting across the savannah; stalking a gazelle in the grass; relaxing in the shade under a tree; marking territory with its scent.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

A 3D animation of a black and white dog with yellow collar, wearing a bandana; biting a bone; wearing a birthday hat; sitting by a ﬁreplace.

- Figure 13: We have extended CoDi to the DiT-based architecture (Flux), with the generated results maintaining subject consistency while achieving diverse pose diversity.

of the questionnaire. To facilitate informed selections, a representative example was provided, accompanied by a performance comparison and rationale, highlighting the reasoning behind choosing the optimal set.

- I ADDITIONAL RESULTS

We present additional qualitative comparisons in Fig. 15, along with more results generated by our CoDi in Fig. 16. These examples further demonstrate that our method achieves state-of-the-art performance in subject consistency, pose diversity, and prompt fidelity. In contrast, existing SCG methods remain limited, often excelling in only one or two of these aspects—typically at the expense of pose diversity or subject consistency.

Long story generation. As each target image xn relies solely on reference image xid for subject identity, our CoDi enables extended visual storytelling. As demonstrated in Fig. 17, it maintains subject consistency across diverse prompt semantics, supporting the generation of varied layouts and poses. This makes CoDi effective for long-form generation, where both prompt fidelity and visual diversity are essential.

- J THE USE OF LARGE LANGUAGE MODELS (LLMS)

In our work, large language models (LLMs) were employed primarily for general writing assistance. Specifically, we used LLMs to refine sentence expressions, check for grammatical errors, and convert tables into LATEX format.

Evaluation Criteria

The questionnaire includes 30 questions, each showing four image sets generated by diﬀerent methods. For each question, please choose the set that best meets the following criteria (evaluated within each row of images):

- 1. Subject Consistency: the subject should maintain a consistent appearance across the row.
- 2. Pose/layout Diversity: the subject should exhibit varied poses and spatial arrangements across the row.
- 3. Prompt Fidelity: Alignment with given text descriptions.

Example

at a science lab reading a book eating ice cream in a library

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

- A
- B
- C
- D

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

A puppy

Performance Comparison and Selection Rationale

- Row A: Although it achieves subject consistency, the poses and layouts are nearly identical.
- Row B: Performs well across subject consistency, pose/layout diversity, and prompt ﬁdelity.
- Row C: Shows poor subject consistency.
- Row D: Shows poor subject consistency. Overall, Row B represents the best choice.

- Figure 14: Questionnaire of user study. Evaluation Criteria outlines the standards for selecting image sets. Example illustrates a representative question for demonstration purposes. Performance Comparison and Selection Rationale demonstrates how the best choice is determined based on the example. These sections visually convey the evaluation criteria and guide participants’ selections.

teaching pinch techniques

stacking kiln shelves methodically

polishing stoneware with smoothing stones

organizing seashell collections

painting coral reefs with natural dyes

braiding seahorses' manes

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Vanilla SDXL

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

ConsiStory

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

StoryDiffusion

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

1Prompt1Story

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

CoDi (ours)

(a) An earthy terracotta relief of a focused potter with clay-smeared hands.

(b) A shimmering aquatic fresco of a curious mermaid with rainbow scales.

playing with colorful fish

decorating underwater caves …

teaching synchronized swimming

singing to woodland animals in a forest

wearing a shimmering gown at a royal ball

gazing at the stars from a castle balcony

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

Vanilla SDXL

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

ConsiStory

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

StoryDiffusion

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

1Prompt1Story

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

CoDi (ours)

(c) A dreamy underwater illustration of a beautiful mermaid with a shimmering tail.

(d) A dreamy illustration of a beautiful princess.

#### Figure 15: Additional qualitative comparisons. Our CoDi achieves the best trade-off among subject consistency, pose diversity, and prompt fidelity.

A pastoral countryside painting of a ﬂour-dusted miller in apron, stacking ﬂour sacks; sifting golden grains; reading wind signs; restoring waterwheel mechanisms; testing ﬂour quality; bagging fresh products.

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

A haunting painting of a seductive siren, weaving sea fog; resting on a sunken ship’s mast; whispering to dolphins in the moonlight; weaving seaweed into magical charms; ﬂoating among glowing jellyﬁsh in the deep sea; playing an enchanted harp on a rocky shore.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

A peaceful and rustic illustration of a hobbit with large feet and a warm smile, enjoying a feast at home; lounging by a cozy ﬁreplace; walking through the rolling hills; baking bread in a stone oven; reading a book in a sunlit nook; harvesting vegetables from the ﬁeld.

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

A terrifying and wild illustration of a werewolf with glowing yellow eyes, snarling beside a ﬂickering campﬁre; resting with a pack under moonlight; glaring through the cracks of a cabin wall; tracking footprints through fresh snow; standing atop a hill under a blood moon; crouching behind tall reeds by a lake.

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

A graceful and elegant illustration of a beautiful princess in a ﬂowing gown, befriending a dragon atop a cliﬀ; contemplating her destiny in a candlelit chapel; feeding swans by the lake; reading in the palace library; walking through a royal garden; lighting lanterns during a peace ceremony.

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

A 3D animation of a happy hedgehog, dressed in a festive outﬁt; hiding inside a boot; in a cozy nest; dressed in a miniature jacket; nibbling on a strawberry; wearing round glasses.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

#### Figure 16: Additional qualitative results generated by our CoDi demonstrate strong subject consistency and pose diversity.

brewing a potion in a bubbling cauldron

preparing a magical brew in a forest

reading spells in an ancient book

meditating in a circle of mushrooms

collecting dew from cursed roses

ﬂoating above a haunted graveyard

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

summoning shadows from a mirror

gazing into a crystal ball under candlelight

binding spells with threads of moonlight

drawing protective runes around a cabin

speaking to spirits through wind chimes

carving charms into tree bark

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

feeding a talking black cat potions

collecting bones under a blood-red moon

wandering through a fogﬁlled swamp

hiding in shadows near a sacred well

sketching forbidden glyphs …

taming a serpent with her voice

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

conjuring illusions to deceive travelers

weaving spells with strands of hair

levitating stones in a moonlit clearing

whispering curses into the wind

disguising herself with shifting shadows

creating potions with stardust and mist

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

feeding a ﬂame with enchanted paper

commanding vines to entangle intruders

taming a ﬁery elemental …

camouﬂaging herself within tree bark

guarding a cursed relic in her lair

crafting talismans from fallen stars

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

tending to cursed dolls on a shelf

gazing into a mirror of smoke

drawing power from an eclipse

hiding secrets in glowing jars

collecting whispers in vials

sketching omens in the ashes

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

taming ﬂames with her hands

etching moon phases into stone

trapping time in a glass orb

hiding memories in tree roots

summoning light from a dark well

walking through walls with a chant

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

Identity Prompt: A dark drawing of a mysterious witch.

- Figure 17: Long Story Generation. CoDi supports extended visual storytelling by generating diverse scene compositions while consistently preserving subject identity throughout the sequence.

