## Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling

Hayeon Kim1,∗ Ji Ha Jang1,∗ Se Young Chun1,2,† 1 Dept. of Electrical and Computer Engineering, 2 INMC & IPAI Seoul National University, Republic of Korea

{khy5630, jeeit17, sychun}@snu.ac.kr

# arXiv:2507.11061v3[cs.CV]30Jun2026

[Figure 1]

DGE [4] GaussianEditor [5]

GaussCtrl [44] RoMaP (Ours)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Instance editing

Segmentation

“Person”

[Figure 9]

Editing

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

“turn him into Albert einstein”

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

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Part editing

Segmentation

“Eyes”

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Editing

[Figure 36]

“turn his left eye blue and his right eye green”

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

3D Gaussian Splatting scene

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Drastic part editing

Segmentation

“Nose”

[Figure 49]

Editing

[Figure 50]

[Figure 51]

[Figure 52]

“turn his nose into green emerald”

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Figure 1. Enhanced controllability in 3D Gaussian part-level editing achieved with RoMaP, surpassing prior arts. RoMaP enables highly controllable and localized part-level edits, allowing even for unconventional modifications such as ‘emerald nose’ or modifications requiring a high-level controllability such as ‘blue left eye, right green eye’ while maintaining global consistency. In contrast, existing baselines perform well for instance-level editing, but struggle with part-level editing, especially with drastic changes.

##### Abstract

across viewpoints. Second, we propose a regularized SDS loss that combines the standard SDS loss with additional regularizers. In particular, an L1 anchor loss is introduced via our Scheduled Latent Mixing and Part (SLaMP) editing method, which generates high-quality part-edited 2D images and confines modifications only to the target region while preserving contextual coherence. Additional regularizers, such as Gaussian prior removal, further improve flexibility by allowing changes beyond the existing context, and robust 3D masking prevents unintended edits. Experimental results demonstrate that our RoMaP achieves state-of-the-art local 3D editing on both reconstructed and generated Gaussian scenes and objects qualitatively and quantitatively, making it possible for more robust and flexible part-level 3D Gaussian editing. Code is available at https://janeyeon.github.io/romap.

Recent advances in 3D neural representations and instancelevel editing models have enabled the efficient creation of high-quality 3D content. However, achieving precise local 3D edits remains challenging, especially for Gaussian Splatting, due to inconsistent multi-view 2D part segmentations and inherently ambiguous nature of Score Distillation Sampling (SDS) loss. To address these limitations, we propose RoMaP, a novel local 3D Gaussian editing framework that enables precise and drastic part-level modifications. First, we introduce a robust 3D mask generation module with our 3D-Geometry Aware Label Prediction (3DGALP), which uses spherical harmonics (SH) coefficients to model view-dependent label variations and soft-label property, yielding accurate and consistent part segmentations

*Authors contributed equally. † Corresponding author.

##### 1. Introduction

Recent advances in 3D neural representations [25, 39, 50, 61, 66] and generative models [19, 53] have enabled efficient, high-quality 3D content creation, increasingly vital for industries such as mixed reality and robotics. Unlike traditional, labor-intensive methods, text-to-image diffusion models [10, 38, 47, 48] generate contents from text prompts, potentially reducing production costs and effort significantly. Enhancing controllability in 3D content generation is crucial for customizing these assets. Text-guided editing methods [5, 6, 9, 40, 62, 64] enhance this by enabling flexible expression of abstract and specific concepts while enabling edits at various levels of detail.

Local 3D editing involves modifying part-level attributes like texture and color, or replacing parts. While previous works [5, 6, 9, 12, 17, 60, 62] have achieved excellent performance in instance-level 3D editing, local 3D editing remains challenging (See Fig. 1). Prior methods [5, 6, 60, 62] often use 2D segmentation [28] to localize changes and apply 2D multi-view editing [2] for 3D modifications. However, these approaches face two major challenges for partlevel modifications, often leading to inaccurate or no edit.

First, achieving consistent 3D editing across multiple views requires precise masking to preserve unchanged regions, typically relying on 2D multi-view image segmentation. However, compared to instance segmentation, part segmentation is challenging due to occlusions and variations in appearance across viewpoints. Existing approaches [5, 6, 60, 62] leverage language-based SAM [28] to segment target parts in multi-view images and re-project them onto 3D for editing. While 2D instance segmentation remains consistent across views, part-level segmentation is much less reliable (e.g., some views may capture only one eye, merge both, or miss them entirely), resulting in unstable and incomplete masks, as shown in Fig. 2. Additionally, assigning a hard segmentation label to each Gaussian from a 2D map may be inappropriate, as Gaussians at part boundaries could represent different parts depending on the view, thus resulting in mixed soft-labels.

Second, part-level 3D editing remains challenging as existing models struggle to isolate and modify specified parts [2] or handle semantically low-probability edits [58]. Learned part-instance correlations often cause unintended changes or failures when the target attribute deviates from the original context. As shown in Fig. 2, InstructPix2Pix [2], widely used for 2D editing in prior works [5, 6, 12, 17], excels in instance edits but struggles with part edits. Instead of applying precise direct changes to the eyes, the model alters the background to green and turns the eyes blue, as odd-eye coloration is rare in human faces, making the edit statistically more likely. Moreover, achiev-

ing such fine-grained control remains highly challenging.

To address this challenge, we introduce RoMaP, a novel part-level 3D editing framework that enables precise and substantial local modifications for Gaussian. RoMaP comprises two core components: (1) A robust 3D mask generation module with 3D-Geometry Aware Label Prediction (3D-GALP): 3D-GALP leverages spherical harmonics (SH) coefficients to explicitly model view-dependent label variations, effectively capturing the mixed-label property of Gaussians. This results in accurate and consistent part segmentations across viewpoints, enabling reliable local edits. (2) A regularized Score Distillation Sampling (SDS) loss: Our regularized SDS combines the standard SDS loss with additional regularizers, including an L1 anchor loss from Scheduled Latent Mixing and Part (SLaMP) edited images. SLaMP generates 2D multi-view images with drastic changes strictly confined to the target region, guiding SDS optimization toward the intended modification. Additionally, robust 3D masking prevents unintended changes. Gaussian prior removal allows flexible adjustments, and together they enable precise local 3D editing, even along rare or unconventional directions. Our RoMaP enables local 3D Gaussian editing, allowing diverse changes in specific areas. As seen in Fig. 1, our RoMaP achieved even drastic local edits, enabling unlikely or unconventional modifications while preserving the original identity, thereby enhancing controllability in 3D content editing. Our contributions are summarized as:

- • Proposing RoMaP for precise and consistent local 3D Gaussian editing, enabled through our robust full 3D mask using our 3D-geometry aware label prediction, exploiting the uncertainty in soft-label Gaussians.
- • Proposing regularized SDS loss, enabling drastic part edits with scheduled latent mixing part editing and robust masks, along with Gaussians prior removal.
- • Experiments show that RoMaP enhances 3D Gaussian editing quality both qualitatively and quantitatively across reconstructed and generated Gaussian scenes and objects, improving controllability in 3D content generation.

##### 2. Related Works

###### 2.1. Diffusion and Rectified Flow based generation

Recent advances in Diffusion Models (DMs) [13, 48] have greatly enhanced image generation, excelling in tasks like image editing, stylization [18, 24, 44, 63]. Rectified Flows (RFs) [36], a flow-based approach [11], streamline diffusion by linearizing the its path, enabling more efficient training, faster sampling, and more accurate latent space inversion. Recent combinations of RF and Diffusion Transformer (DiT) [42] models, like FLUX and Stable Diffusion 3 (SD3) [13], have advanced high-quality image gen-

#### Instance level editing results

Segmentation prompt : “Person” Editing prompt : “Turn him into a Albert einstein”

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

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

[Figure 72]

[Figure 73]

2D segment results 2D editing results 3D editing results 3D segment results 2D editing results 3D editing results

[Figure 74]

Baselines RoMaP (Ours)

[Figure 75]

#### Part level editing results

Segmentation prompt : “Eyes” Editing prompt : “Turn his eyes into a left blue and right green eyes”

[Figure 76]

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

2D segment results 2D editing results 3D editing results 3D segment results 2D editing results

3D editing results

[Figure 92]

[Figure 93]

Baselines RoMaP (Ours)

- Figure 2. Limitations of prior local 3D editing methods leveraging 2D part level segmentation and edits. Although existing 3D editing methods excel in instance level editing, they struggle with part level editing as part segmentation [28] (for ‘eye’) lacks view consistency, and 2D editing [2] often misplaces changes, turning a wall green instead of the left eye. In contrast, our method achieves accurate 3D eye segmentation with geometric awareness and clearly defines modification direction, enabling successful 3D Gaussian editing.

eration, benefiting applications like text-to-3D and image editing. These models enhance prompt-faithful editing and inversion by refining noise [65] and utilizing RF’s linearity [49] but still lack part-level controllability. Similarly, prior works [31, 33, 65] employ these models in text-to-3D, achieving high-fidelity and faster convergence. Notably, SD3 has been applied to part-level controllable text-to-3D generation [33] but is limited to animals, leaving broader applications unexplored. Our approach enables previously unattainable drastic local 3D edits by leveraging the SD3’s part-awareness and RF’s linearity, allowing flexible edits across various reconstructed and generated Gaussians.

###### 2.2. Editing of 3D Gaussian Splatting

Editing 3D neural fields has advanced 3D generation by enhancing controllability, attracting research interest [17, 34]. Early works focused on Neural Radiance Field (NeRF) [12, 17, 29], but recent works have shifted toward 3D Gaussians for better local control and efficient rendering. Editing Gaussians requires both an editing and a masking strategy to target specific parts. In editing strategy, some methods [5, 6, 62] edit 2D-rendered Gaussian images from multiple views using image editing models [2, 5, 56] and project them back onto Gaussians. However, this approach is limited by the constraints of the 2D editing model and causes inconsistencies in 3D projection. Others [40] directly update Gaussians using Score Distillation Sampling (SDS)

loss, but struggle to make significant modifications due to its implicit characteristic [4, 6, 15, 23]. We first remove priors and set the modification direction with the SLaMPedited image, then refine for greater control beyond the original context. For masking strategies, most works utilize

- 2D masks for localized edits, projecting them onto Gaussians [6, 60, 62]. However, noisy multi-view 2D masks introduce inconsistencies, affecting unintended regions or preventing proper transfer of 2D changes to 3D. Also, Gaussians at part boundaries can represent different parts depending on the view. However, assigning 3D Gaussian labels based on a 2D map overlooks this, resulting in inaccurate segmentation at part boundaries. To address this, our
- 3D-GALP selects anchors based on view-dependent label prediction consistency and enforces neighbor consistency in 3D, refining 3D masks to correct 2D imperfections.

###### 2.3. Local editing of 3D representations

Most 3D editing methods discussed in Sec. 2.2 focus on instance level modifications or scene wide style changes. Some extend this to local edits, enabling precise adjustments to specific parts for finer control and prompt adaptability. A key challenge in local editing is effectively selecting specific areas. Some methods [9, 64] use bounding boxes from users or Large Language Models (LLMs) to make local changes, but these restrict selection to simple shapes, and their fixed nature prevents deformable edits.

###### Input

###### Local 3D Segmentation

###### Local 3D Editing

###### Segment Prompt (𝒑 ) : “A photo of woman with lips and hair and face” Edit Prompt (𝒑 ) : “A photo of woman with with a jellyfish tentacles for hair”

[Figure 94]

[Figure 95]

[Figure 96]

ℳ Ω

[Figure 97]

[Figure 98]

[Figure 99]

|Gaussian prior removal| |
|---|---|
| | |

𝑪 ,𝑹 ,𝒑

[Figure 100]

Regularized SDS loss

,𝒑

[Figure 101]

|[Figure 102]|
|---|

Rendered views 𝑪

,ℳ

Attention map extraction

|[Figure 103]|
|---|

|[Figure 104]|
|---|

𝑪

view point 𝜙

[Figure 105]

[Figure 106]

[Figure 107]

Ω = {𝒄 ,𝒓,..}

|[Figure 108]|
|---|

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

{𝒓}

SLaMP(𝑪 ) 𝑪

[Figure 113]

[Figure 114]

|[Figure 115]<br><br>[Figure 116]<br><br>3D-GALP|
|---|

[Figure 117]

Segmented 3D mask ℳ

3D Gaussian Splatting Ω

[Figure 118]

Final editing result

[Figure 119]

| | |
|---|---|
|SLaMP editing| |

[Figure 120]

,ℳ

[Figure 121]

Prior removed Gaussian Ω

,𝒑

Rendered label predictions 𝑹

|[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>Scheduled Latent Mixing and Part editing (SLaMP)<br><br>[Figure 126]<br><br>[Figure 127]<br><br>∗<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>∗<br><br>[Figure 131]<br><br>[Figure 132]<br><br>Scheduled weight mask<br><br>… …<br><br>at timestep t<br><br>[Figure 133]<br><br>[Figure 134]| |
|---|---|
| | |

|3D-Geometry Aware Label Prediction (3D-GALP)<br><br>|[Figure 135]|
|---|
<br><br>Gaussian label prediction {𝒓}<br><br>Label softness for Gaussian i across all view 𝜙<br><br>[Figure 136]<br><br>[Figure 137]<br><br>𝜙<br><br>𝜙<br><br>[Figure 138]<br><br>Noisy map<br><br>Noisy map<br><br>[Figure 139]<br><br>[Figure 140]<br><br>Selected Gaussian Ω<br><br>Considered as 𝒍 at 𝜙<br><br>[Figure 141]<br><br>Sampled anchors A<br><br>Neighboring loss Considered as<br><br>𝒍 at 𝜙<br><br>[Figure 142]|
|---|

[Figure 143]

Extracted 2D Mask

[Figure 144]

ℳ

{𝒓}

Renderd view

[Figure 145]

Edited image

[Figure 146]

𝑪 Edit prompt 𝒑

SLaMP(𝑪 )

- Figure 3. Overall pipeline of RoMaP. RoMaP first segments 3D Gaussian using 3D-GALP, leveraging the soft-label properties of Gaussians to address the intricacies of part-level segmentation. With anchors consisting of both label-consistent and inconsistent Gaussians, we refine 3D segmentation considering locality with neighboring Gaussians. Then, in local 3D editing, we first remove Gaussian priors and introduce a new modification direction using SLaMP-edited images, followed by refinement via a regularized SDS loss.

###### 3.1. Preliminary: 3D Gaussian Splatting

Another work [22] utilizes a pre-trained 3D GAN [3] with CLIP [46] to select local areas and generate changes in human and cat faces. While effective for some edits, it remains limited to specific targets and struggles with more drastic edits. Our model is the first to achieve part-level 3D editing for general objects in both reconstructed and generated Gaussians. By fully utilizing SD3 and Gaussian properties, RoMaP enables faithful and drastic 3D local edits.

Gaussian Splatting [25] is a point-based method that represents a 3D scene using Gaussian properties. Let Ω be a set of Gaussians composing the scene, where each Gaussian Ωi is defined as Ωi = {pi,si,qi,αi,ci}, where pi, si, qi, αi, and ci represent the centroid, standard deviations, rotational quaternion, opacity, and spherical harmonics (SH) coefficients, respectively. The projected RGB color of Gaussians varies by viewpoint ϕ and is computed as cϕ = SH(c,ϕ), where SH(c,ϕ) evaluates the SH coefficients c at ϕ. The rendered image Cϕ for view ϕ is obtained by projecting Ω onto a 2D plane using the differentiable rasterization D:

##### 3. Method

We propose RoMaP, a novel method for locally editing 3D Gaussians with text prompts, enabling targeted regional modifications. Existing approaches struggle with part edits because (1) projecting 2D segmentations to 3D is unreliable due to inconsistent part-aware models and ambiguous part boundaries, and (2) isolating specific parts is difficult due to entanglements in 2D diffusion models.

Ω = {p,s,q,α,c} ϕ cϕ = SH(c,ϕ) D Cϕ.

###### 3.2. Local 3D segmentation: 3D-GALP

To address these challenges, RoMaP first performs explicit local 3D segmentation by adopting view-dependent segmentation labels and resolving 2D segmentation inconsistencies using 3D Geometry-Aware Label Prediction (3DGALP), as discussed in Sec. 3.2. To enable drastic part edits beyond pre-existing contexts, we introduce a new modification direction using regularized score distillation sampling, guided by regularizers: anchored L1 with Scheduled Latent Mixing and Part (SLaMP) editing, Gaussian prior removal and masking. This process is detailed in Sec. 3.3. The full pipeline of RoMaP is shown in Fig. 3.

This section describes the ‘Local 3D Segmentation’ on the left side of the Fig. 3. To localize changes in the target region, we create a 3D segmentation M3D given Ω. The goal is to predict which regions of M3D correspond to each predefined part label lj. This involves two steps: attention map extraction and 3D geometry-aware label prediction (3D-GALP). Given a segmentation prompt, we extract the attention map A(Cϕ) from a randomly rendered view Cϕ and treat it as a pseudo 2D segmentation map to guide 3D-GALP. More details on attention map extraction are in the supplementary material.

Attention-based pseudo segmentation for 3D Gaussians In this step, we obtain the explicit 3D segmentation M3D using 3D-GALP, guided by A(Cϕ). Once constructed, M3D provides segmentation information for all Gaussians. To represent these labels, we introduce a new parameter ri and incorporate it into the Gaussian representation: Ωi = {pi,si,qi,αi,ci,ri}. Since a single Gaussian may correspond to different labels depending on the viewpoint, it exhibits mixed-label property. To model this view-dependent labeling, we represent each Gaussian’s label as SH coefficients. We interpret Rϕ, the 2D projection of Gaussians obtained via r at view ϕ, as a segmentation map:

Ω = {p,s,q,α,c,r}ϕ rϕ = SH(r,ϕ) D Rϕ.

The learnable parameter r is then optimized via L1(A(Cϕ),Rϕ) loss, encouraging the rendered map to align appropriately with the pseudo 2D attention map in the given view. While this process aligns Gaussians with the attention map across multiple views, the alignment may remain imperfect. To further refine segmentation, we apply an anchor-based neighbor consistency loss, with anchors sampled by considering label softness.

Label-softness based anchor sampling Occlusions and view-dependent shape complexity can lead A(Cϕ) to produce incomplete segmentation maps (See Fig. 3). To achieve complete and view-consistent 3D segmentation, we refine the segmentation by leveraging the view-dependent label softness of Gaussians. Here, ri is treated as an SH color, and a Gaussian is considered to exhibit label softness if rϕi varies with the viewpoint ϕ. To quantify the label softness, we measure vi, the variance of rϕ across ϕ. Then, we calculate the cosine similarity between ¯ri, the mean color observed from all directions and lj, the label assigned to each part. We then compute the entropy as follows:

¯ri·lj ∥¯ri∥∥lj∥

e

,Hi = −

pij =

¯ri·lj ∥¯ri∥∥lj∥

l e

j

pij log(pij) (1)

where pij denotes the probability obtained from the cosine similarity between predicted label ri and ground truth label lj, while Hi denotes the entropy of pij. We define the softness of the label of each Gaussian as the product of Hi and vi, given by Si = Hi ·vi. As visualized in Fig. 4, Si is high at part boundaries, where Gaussians inherently exhibit softlabel properties. This is due to the 2D part segmentation map classifying Gaussians noisly around these boundaries. All Gaussians are sorted based by Si, then K anchors are selected: the top ⌊K/2⌋ from those with the highest softness values and the bottom ⌊K/2⌋ from those with the lowest. This sampling method selects anchors from both Gaussians

Random sampling loss

Label softness based anchor sampling loss

Label softness map

Original scene

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Figure 4. Effectiveness of label softness-based anchor sampling. By applying 3D loss with anchors sampled based on label softness, we observe that differentiation of boundaries between parts is much more precise compared to random sampling.

with high soft-label properties and those with consistent labels, enabling refinement of 3D segmentation while preserving locality and effectively handling part boundaries. Fig. 4 shows that part boundaries can be segmented precisely with label-softness based sampling compared to random sampling.

Anchor-based neighboring loss Given the selected anchors A, we enforce neighbor consistency by incorporating segmentation information from nearby Gaussians. For each anchor Ωi ∈ A, we find its K nearest neighbors, where NK(i) denotes the top-k nearest neighbors of the i-th anchor. We then compute the L1 between the segmentation label rj of neighboring points and the ri of the anchor point:

 . (2)

  1

∥ri − rk∥1

LGALP =

K

i∈A

k∈NK(i)

As shown in Fig. 5, 3D-GALP effectively can segment various parts of diverse objects. Additional 3D segmentation results in various scenes are provided in the supplementary.

###### 3.3. Local 3D Editing: Regularized score distillation sampling

Regularized score distillation sampling We can now explicitly select Gaussian regions for editing using M3D. Since the SDS loss primarily serves as an implicit objective but has limited direct impact on 3D Gaussians [4, 6, 15, 23], we enable more effective modifications by introducing a regularized SDS loss. This loss combines two regularizers: Gaussian prior removal and masking, and an anchoredbased L1 loss using SLaMP edited image. The regularized SDS loss is defined as:

LR-SDS = λ1LˆSDS(cϕpr,pedit) + λ2Lˆ1(cϕpr,SLaMP(cϕpr)).

(3) Here, λ1 and λ2 are hyperparameters that balance the contribution of LˆSDS and Lˆ1 during training. Lˆ denotes a

Original scene Segmentation results

Original scene Segmentation results Original scene Segmentation results

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

baseball

hair

head

body

face

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

arm

body

[Figure 168]

seam

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

window

nose

ears

collar

[Figure 176]

[Figure 177]

[Figure 178]

shirts body

wheel body

sheep

- Figure 5. 3D Gaussian segmentation results of 3D-GALP. With our 3D-GALP, 3D Gaussian segmentation accurately captures diverse object parts, addressing the limitations of 2D part segmentation and the inherent mixed nature of 3D Gaussian segmentation labels.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Original 𝑡 = 1 𝑡 = 5 𝑡 = 9

𝑡 = 11 𝑡 = 15 𝑡 = 17 𝑡 = 19

(a) Change in CLIP and Prompt : “A man with a croissant nose” SSIM by 𝑡 selection (b) Edited image results depending on 𝑡

[Figure 187]

: CLIP : SSIM

Sweet spot

𝑡

- Figure 6. Experiments on the effect of ts. ts controls the extent of deviation from the original. We set ts to induce drastic changes in the target region while preserving the surrounding identity.

key aspect of SLaMP is the scheduled blending of latents over time, enabling fine-grained control over the influence of the original image. Effective part-level editing requires isolating the target region while guiding it toward the desired change without compromising global identity. SLaMP achieves this by scheduling a sharp transition in the blending ratio between the target latent zt and the original latent zt,orig. The resulting latent zt+1 is expressed as follows:

zt+1 = zt(1−Ft·(1−M2D))+zt,origFt·(1−M2D). (4)

Here, Ft is a time-dependent blending coefficient. We begin with a low Ft to generate new context without strong influence from the original image. At timestep ts, we increase Ft sharply to preserve the alignment with original. As shown in Fig. 6, setting ts too low disrupts the original image context, while setting it too high hinders new content generation. To balance preservation and editing, we set ts to where SSIM is stable while CLIPdir remains high. More details are in the supplementary.

masked loss leveraging M3D and M2D to restrict changes only to intended regions. cϕpr refers to the 2D projection of prior-removed Gaussians in view ϕ and SLaMP refers to our 2D part editing method that enables clear directional change that SDS loss cannot achieve. These two components will be discussed in following section.

- Regularizer 1: Gaussian prior removal and masking Due to the inherent ambiguity of SDS loss and the localized nature of Gaussians, applying SDS alone limits modification extent [4, 15]. To address this, we introduce an

L1 loss on explicitly edited images to provide more targeted and controllable guidance. However, directly combining L1 with LSDS often induces overly broad changes, since LSDS operates in all directions and biases the optimization toward preserving strong appearance priors. To mitigate this, we perform Gaussian prior removal by replacing dominant color priors with neutral colors (e.g., white or gray), producing cϕpr to discourage fixation on original appearances. Additionally, we explicitly prevent gradient updates to Gaussians on M3D, avoiding unintended changes and ensuring that edits are confined to the target regions.

- Regularizer 2: Anchored L1 with SLaMP edited image To generate an anchor image for the L1 loss, we propose SLaMP editing, a part level editing strategy that balances localized modification with global identity preservation. A

##### 4. Experiments

###### 4.1. Experimental setting

Dataset and evaluation metrics To evaluate editing performance on reconstructed Gaussians, we use scenes from IN2N [17] and NeRF-Art [17], testing 75 editing prompts targeting different parts and changes in each scene. For evaluation metrics, we used two CLIP-based metrics, CLIP Similarity [46] and CLIPdir Similarity [14], to measure the overall fidelity between the input text and the edited scene. Furthermore, we used BLIP-VQA [21] and TIFA [20] to assess how well edits align with specific text prompt components via visual question answering.

Baselines We compared RoMaP with three state-of-theart 3D Gaussian editing methods (DGE [5], GaussianEditor [6], and GaussCtrl [62]) and three NeRF editing methods (Instruct-Nerf2Nerf (IN2N) [17], ViCA-NeRF (ViCA) [12], and Posterior Distillation Sampling (PDS) [29]). All baselines perform 2D edits before lifting them to 3D. [5, 6, 12]

###### Original 3DGS Diverse and Precise Part-level Editing Results

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

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

“A beautiful woman” “with cloudy hair” “with chick’s beak” “with butterfly hair” “with gold bell nose”

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

| |
|---|

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

| |
|---|

| |
|---|

“with lava bulb”

“with sparkling crystals”

“with stained glass”

“made of wooven bamboo”

“with Blue flame”

“covered with spider webs”

“A standing lamp”

- Figure 7. Enhanced controllability in 3D asset generation with RoMaP. Our approach enables precise manipulation of specific 3D parts. As shown above, RoMaP provides diverse control over multiple narrow regions within a single 3D object, allowing deformations in targeted areas like a ‘duck’s beak’ or ‘jellyfish hair’ and facilitating various modifications in targeted area such as a lamp’s lampshade.

comparing RoMaP with 3DGS generation and editing methods. Ours enables drastic local changes, such as butterfly lips and goat’s head, while others fail. Its enhanced controllability also enables text-aligned generation that other models struggle with. As shown in Fig. 7, RoMaP enables diverse 3D creations, such as a lamp with different bulbs and lampshades, simplifying 3D asset customization.

Editing Methods Metrics

CLIP ↑ CLIPdir ↑ B-VQA ↑ TIFA ↑ NeRF baselines

IN2N [17] 0.248 0.072 0.142 0.634 ViCA [12] 0.223 0.048 0.241 0.427 PDS [29] 0.167 -0.005 0.237 0.212

###### Gaussian Splatting baselines

GaussCtrl [62] 0.182 0.044 0.190 0.432 GaussianEditor [6] 0.179 0.087 0.370 0.571 DGE [5] 0.201 0.095 0.497 0.565 RoMaP (Ours) 0.277 0.205 0.723 0.674

Metrics Baseline + Mask + Mask & Lˆ1 Full (Ours)

CLIP ↑ 0.218 0.228 0.267 0.277 CLIPdir ↑ 0.060 0.162 0.205 0.205

- Table 1. Quantitative comparison with 3D editing methods. Our method outperforms various baselines in multiple metrics.

employ InstructPix2Pix [2], while [62] utilizes ControlNet [68], and [29] applies posterior distillation sampling. In a user study, we compared RoMaP against generation models [7, 65, 67], assessing how editing improves controllability in generating previously difficult samples.

4.2. Experimental results

Quantitative comparisons Tab. 1 presents a quantitative comparison of RoMaP against 3DGS and NeRF editing models, where it outperforms all baselines across metrics. As shown in Tab. 2, user study further validates RoMaP’s superior performance. Statistical significance of user study is confirmed by Friedman and pairwise Wilcoxon tests.

Editing Method User Study ↑ Generation Method User Study ↑

GaussCtrl [62] 0.201 GSGEN [7] 0.203 GaussianEditor [6] 0.201 GaussianDreamer [67] 0.198 DGE [5] 0.224 RFDS [65] 0.234 RoMaP (Ours) 0.372 RoMaP (Ours) 0.365

- Table 2. User study results. Quantitative comparison of user study results for editing and generation methods. Qualitative comparisons Fig. 8 shows qualitative results

Table 3. Ablation study results The ablation study shows results from sequentially adding key methods.

###### 4.3. Ablation study

Tab. 3 presents an ablation study validating each step of RoMaP. In Tab. 3, ‘Mask’ refers to results using masks (M2D & M3D) generated from 3D-GALP. ‘Lˆ1’ is the result of a regularized SDS loss, by only employing the second term. The ‘Full’ represent our full regularized SDS loss, enabling modification in the desired direction. This confirms the necessity of all steps in RoMaP.

##### 5. Conclusion

In this work, we introduce RoMaP, a novel approach for local 3D Gaussian editing that enables precise and consistent part-level edits. To localize part accurately, we employ robust segmentation with geometry-aware label prediction, utilizing the soft-label properties of Gaussians. We also propose the regularized SDS loss using scheduled latent mixing and Gaussian prior removal, enabling drastic part-level edits while preserving remaining areas. Experimental results demonstrate RoMaP’s significant improvements in 3D Gaussian editing quality across various scenes even in challenging scenarios.

(a) Reconstruction based editing method comparison

Original GaussCtrl [44] GaussianEditor [5] DGE [4] RoMaP (Ours)

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

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

[Figure 231]

[Figure 232]

[Figure 233]

“A man with butterfly lips”

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

“A man with golden bell nose”

[Figure 247]

[Figure 248]

[Figure 249]

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

[Figure 261]

“A man with mechanical hair”

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

“A man with goat’s head”

(b) Generation based method comparison

GSGEN [6] GaussianDreamer [49] RFDS [47] Original RoMaP (Ours)

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

“A cat with white angle’s wing instead of ear”

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

“A woman with chicken’s beak instead of mouth”

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

[Figure 320]

[Figure 321]

“A vase with yellow tulip and stained glass textured rose”

- Figure 8. Comparison results The results of comparing our methodology with various reconstruction-based 3D editing methods and text-to-3D generation approaches are presented. In the reconstructed scene, our method enables drastic changes in very narrow regions, breaking the existing priors that other approaches have been unable to overcome. This allows for diverse transformations, such as replacing a human face with a goat’s face or substituting hair with butterflies. In the text-to-3D generation scenario, our approach achieves success in examples where naive text prompts alone fail, demonstrating its ability to generate a wider range of 3D assets.

##### Acknowledgements

This work was supported in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)], the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. RS-2025-02263628) and AI-Bio Research Grant through Seoul National University. Also, the authors acknowledged the financial support from the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers, Seoul National University.

## Supplementary Material for Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling

##### S.1. Additional details on quantitative results

###### S.1.1. Experimental setting

###### S.1.1.1. Comparison with 3D Gaussian editing models

We collected human face scenes from the IN2N [17] and Nerf-Art [58] datasets. For each facial part: ‘eyes’, ‘nose’, ‘mouth’ and ‘hair’, we applied five editing prompts: ‘silvertextured’, ‘gold-textured’, ‘diamond’, ‘green’, ‘pink’ to evaluate editing success. Additionally, we designed five prompts requiring drastic changes: ‘delicious croissant nose’, ‘hair made of metallic gears, steampunk style’, ‘hair on fire, red and blue flame’, ‘hair covered with beautiful butterfly’, ‘left blue and right green eye’, and categorized them as ‘hard’ to assess extreme editing performance. For models incorporating InstructPix2Pix [2] in their pipelines, we adapted the prompts to the format: “Turn ... into ...”.

###### S.1.1.2. Comparison with 3D Gaussian generation models

To prove that our local 3D editing method enhances controllability in 3D content generation, we designed prompts for samples that were challenging for previous 3D generation methods to create. The prompts included: ‘A beautiful woman with a cheek’s beak’, ‘A woman with cloudy hair’, ‘A beautiful woman with butterfly hair’, ‘A snail with skyscapes inside its shell’, and ‘A vase with a yellow tulip and a stained glass-textured rose’. We tasked 3D generation models with directly generating 3D content from these prompts. In our approach, we first generated the base objects, such as ‘A snail’, then applied these prompts as editing instructions to assess whether our method could successfully produce the desired samples.

User Study We conducted a user study across three categories: (1) Alignment - Is the 3D Gaussian edited to match the text? (2) Fidelity - Does the image look visually appleaing? (3) Accuracy - Were only the specified parts edited correctly?. Users were asked to score a 4-point scale, and we averaged it for mean opinion score (MOS). For reconstructed scene, participants evaluated all three criteria, collecting 4,680 responses from 260 respondents. For generated 3D, evaluations were based on alignment and fidelity, yielding 2,600 responses from 260 respondents.

###### S.1.1.3. Metrics

CLIP and CLIP directional score The CLIP-based metrics calculate the cosine similarity between text and image features extracted using CLIP [46]. CLIP scores are commonly utilized in evaluating text-to-3D [34, 43, 55]. CLIP directional scores are specifically employed to evaluate whether the changes occurred in the desired direction, first introduced by [14] and adopted mostly by editing models [5, 6, 9, 62]. We used the ViT-L/14 version of the model, with images cropped to 512 pixels and resized to 336 pixels before being input into the model.

TIFA and BLIP score While CLIP-based metrics effectively evaluate coarse similarity between image and text, they have limitations in assessing fine-grained correspondences [1, 9, 20, 21, 52]. To address this, we adopted two additional evaluation metrics focused on fine-grained visual-textual alignment, based on visual question answering (VQA). The TIFA score, introduced in [20], measures the faithfulness of generated image to text input by generating questions with LLaMA2 [54], answering with UnifiedQA-v2 [27]. BLIP-VQA, proposed in [21] breaks down a prompt into multiple questions, assigning a score based on the probability of answering ‘yes’ to each question, leveraging the vision-language understanding and generation capabilities of BLIP [32].

###### S.1.1.4. Implementation details

Our method is implemented in PyTorch [41], based on Threestudio [16]. We employ Stable Diffusion 3 [13]. All experiments are conducted on a single A100.

###### S.1.2. Experimental Results

Quantitative results Detailed quantitative results are shown in Tab. S.1, Tab. S.2, Tab. S.3 and Tab. S.4. The tables present quantiative results for each part editing. Our approach outperformed all other baselines in NeRF and Gaussian Splatting editing across all parts and metrics [5, 6, 12, 17, 29, 62]. Notably, considering that our models achieve strong performance on both CLIP-based and VQA-based scores, we can conclude that our models perform well in editing at both coarse and fine levels. Detailed results of user study for each evaluation criterion are provided in Table. S.5 and Table. S.6. Validity of the user

part eye nose mouth hair hard avg

method

CLIP CLIPdir CLIP CLIPdir CLIP CLIPdir CLIP CLIPdir CLIP CLIPdir

GaussCtrl [62] 0.191 0.042 0.183 0.035 0.173 0.056 0.195 0.060 0.168 0.026 0.182 0.044 GaussianEditor [6] 0.190 0.068 0.130 0.057 0.140 0.086 0.232 0.144 0.202 0.083 0.179 0.087

DGE [5] 0.193 0.076 0.190 0.058 0.182 0.070 0.232 0.161 0.211 0.110 0.201 0.095 RoMaP(ours) 0.246 0.150 0.263 0.210 0.311 0.265 0.277 0.211 0.291 0.188 0.277 0.205

Table S.1. Comparison with GS editing methods. CLIP score and CLIP directional score value for each method and part.

part eye nose mouth hair hard avg B-VQA TIFA B-VQA TIFA B-VQA TIFA B-VQA TIFA B-VQA TIFA B-VQA TIFA

method

GaussCtrl [62] 0.194 0.422 0.195 0.561 0.223 0.389 0.239 0.494 0.098 0.292 0.190 0.432 GaussianEditor [6] 0.361 0.561 0.301 0.633 0.448 0.572 0.593 0.722 0.148 0.368 0.370 0.571

DGE [5] 0.517 0.539 0.427 0.717 0.512 0.5 0.774 0.683 0.255 0.388 0.497 0.565 RoMaP(ours) 0.700 0.667 0.797 0.733 0.935 0.711 0.796 0.717 0.399 0.543 0.723 0.674

Table S.2. Comparison with GS editing methods. BLIP-VQA score and TIFA score value for each method and part.

part eye nose mouth hair hard avg

method

CLIP CLIPdir CLIP CLIPdir CLIP CLIPdir CLIP CLIPdir CLIP CLIPdir CLIP CLIPdir

iN2N [17] 0.247 0.067 0.257 0.071 0.258 0.084 0.253 0.079 0.227 0.060 0.248 0.072 VICA [12] 0.224 0.050 0.225 0.040 0.219 0.052 0.229 0.049 0.217 0.051 0.223 0.048

PDS [29] 0.162 -0.033 0.171 0.014 0.177 0.007 0.176 0.008 0.152 -0.020 0.167 -0.005 RoMaP(ours) 0.246 0.150 0.263 0.210 0.311 0.265 0.277 0.211 0.291 0.188 0.277 0.205

Table S.3. Comparison with NeRF editing methods. CLIP score and CLIP directional score value for each method and part.

part eye nose mouth hair hard avg B-VQA TIFA B-VQA TIFA B-VQA TIFA B-VQA TIFA B-VQA TIFA B-VQA TIFA iN2N [17] 0.168 0.589 0.168 0.489 0.163 0.471 0.139 0.671 0.072 0.623 0.142 0.565 VICA [12] 0.277 0.436 0.204 0.507 0.292 0.387 0.228 0.396 0.205 0.41 0.241 0.427

method

PDS [29] 0.267 0.2 0.287 0.173 0.264 0.147 0.333 0.160 0.034 0.380 0.237 0.212 RoMaP(ours) 0.700 0.667 0.797 0.733 0.935 0.711 0.796 0.717 0.399 0.543 0.723 0.674

Table S.4. Comparison with GS editing methods. BLIP-VQA score and TIFA score value for each method and part.

study result is evaluated using pairwise Wilcoxon tests and the Friedman test, as shown in Fig.S.3. The test results confirm that our method significantly outperforms other editing and generation methods with strong statistical significance

and validating the effectiveness of our method.

Qualitative results We included more qualitative results of our approach in Fig. S.1, Fig. S.2, Fig. S.8, Fig. S.9, and

Open-vocabulary Part segmentation

Open-vocabulary Part segmentation Editing results

Scene-level reconstruction

[Figure 322]

Editing results

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

Silver spoon

Ears

Collar Handle

[Figure 327]

| |
|---|

| |
|---|

[Figure 328]

Cup

Sheep

“a white cup with pink handle” “a sheep with purple ears”

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

Pirate hat

Eyes

| |
|---|

| |
|---|

Beak

[Figure 334]

[Figure 335]

Body

Rubber duck

“a rubber duck with white hat” “a dog figurine with yellow eyes”

- Figure S.1. 3DGS part editing results in complex 3DGS scenes. We performed RoMaP editing on complex 3DGS scenes from the LERF dataset. As shown above, our RoMaP achieved precise open-vocabulary part segmentation for parts of varying sizes, such as the collar, eyes, body, and rubber duck. Additionally, we achieved accurate part editing based on prompts like ‘a sheep with purple ears’ and ‘a rubber duck with a white hat’.

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

“with blue hair” “with purple dress”

“with ‘Hi’ name tag” “with yellow collar”

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

Original complex 3D scene Editied complex 3D scene

- Figure S.2. 3DGS part editing results in complex scenes. We demonstrate RoMaP editing results on complex 3D Gaussian Splatting (3DGS) scenes from both the 3D-OVS and LERF datasets. As shown above, RoMaP achieves high-quality normal editing, effectively handling diverse and practical edits such as ‘with blue hair’ or ‘with a ‘Hi’ name tag’. These results highlight RoMaP’s ability to generalize across various scene complexities.

Fig. S.10. As shown in Fig. S.8, Fig. S.9 and Fig. S.10, our RoMaP can generate diverse 3D assets by editing the original 3D Gaussian Splatting (3DGS). Also, Fig. S.1 and Fig. S.2 show part-editing of our RoMaP in complex scenes. The results demonstrate that our 3D-GALP and editing

strategies achieve high precision in 3D segmentation and enable precise modifications to the targeted regions, highlighting the scalability of our method to more complex and cluttered 3D scenes.

Method Alignment Fidelity Accuracy GaussCtrl [62] 19.70% 19.98% 20.6% GaussianEditor [6] 19.61% 19.98% 20.72% DGE [5] 23.18% 23.62% 20.24% RoMaP (Ours) 36.73% 36.31% 38.43%

- Table S.5. User study results on comparison with 3D Gaussian editing models.

Method Alignment Fidelity GSGEN [7] 20.48% 20.09% GaussianDreamer [67] 19.61% 19.98% RFDS [65] 23.18% 23.62% RoMaP (Ours) 36.73% 36.31%

- Table S.6. User study results on comparison with 3D Gaussian generation models.

Editing methods Generation methods

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

DGE [3] GC [35]

GD [37]

GS [5] RFDS [36]

GE [4] Ours

Ours DGE GC GE Ours GD GS RFDS Ours

(a)Pairwise wilcoxon test with editing and generation methods

(b) Friedman test on both results

- Figure S.3. Statistical results from user study. (a) Pairwise Wilcoxon test results for editing and generation methods. (b) Friedman test p-values for fidelity, accuracy, and alignment. Our approach (Ours) achieves significantly better performance in both reconstruction and generation compared to existing methods.

Qualitative results of baselines We visualized qualitative results of Gaussian and NeRF-editing baselines in Fig. S.15 and Fig. S.16. For the NeRF baseline model, we present result from IN2N [17]. Due to the implicit nature of NeRF, precisely selecting the target region is challenging, often resulting in unintended global changes. For example, when applying the prompt ‘Turn his hair into silvertextured hair’, the entire scene shifts to a silver hue S.15. Similarly, prompts such as ‘hair on fire’ or ‘left eye blue and right eye green’ lead to incorrect region selection, causing widespread color alterations across the scene. For the Gaussian Splatting baseline, we show results from GaussianEditor [6]. Inconsistencies in 2D part segmentation lead to unreliable 3D part segmentation, as shown in Fig. S.16. Additionally, 2D editing results demonstrate difficulties in precisely modifying the desired regions. For instance, a croissant appears in the background instead of the intended edit, or the entire scene turns pink rather than just his eyes.

##### S.2. Additional results in complex scene

To further validate the robustness and generalizability of RoMaP, we present additional editing results on complex 3DGS scenes from both the 3D-OVS [35] and LERF [26] datasets. These scenes contain multiple objects with intricate part-level structures and diverse contextual settings.

As illustrated in Fig. S.1, RoMaP demonstrates precise open-vocabulary part segmentation and editing across a wide range of object types and part granularity. Examples include edits guided by prompts such as a ‘white cup with pink handle’, ‘a rubber duck with white hat’, and ‘a dog figurine with yellow eyes’. RoMaP effectively identifies and modifies fine-grained parts such as handles, beaks, collars, and ears, even under cluttered backgrounds and occlusions.

In addition, Fig. S.2 further showcases our model’s ability to perform practical part editing tasks involving realistic human and animal figures. Prompts such as ‘with blue hair’, ‘with purple dress’, and ‘with ‘Hi’ name tag’ illustrate RoMaP’s capability to generalize beyond common categories and execute attribute-level modifications across highly complex scenes. These results collectively highlight RoMaP’s strength in both semantic understanding and finegrained spatial localization, making it a versatile tool for open-vocabulary 3D scene editing.

##### S.3. Additional validation and details of pipeline

###### S.3.1. Attention map extraction

Unlike the naive reverse flow-matching process used in textto-3D generation, we adopted a controlled forward ODE to extract more accurate attention maps for real images, thereby enhancing robustness. Controlled forward ODE, proposed in [49], helps maintain consistency with the given image while aligning with the distribution of typical images. This balancing mechanism allows for effective inversion and editing across various inputs, especially real images, even when the given image is corrupted or atypical. Additionally, we adopted the approach proposed in [59] for dense prediction. This method allows for faster and more accurate extraction of attention maps.

Post-processing We post-processed extracted attention maps by normalizing them with a softmax temperature and utilizing a refiner [8]. Adjusting softmax temperature allowed us to segment regions with varying granularity, while the refiner, by incorporating the original image features, enabled segmentation of parts with more precise edges, as shown in Fig. S.4.

Input Image w.o / softmax normalization w.o / refiner Full

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

Segment Prompt : “A photo of joker with mouth and hair and face and clothes”

- Figure S.4. Ablation study of attention map post-processing procedure By adjusting the softmax temperature, we achieved segmentation with varying levels of granularity, while the refiner, leveraging the original image features, facilitated the segmentation of parts with sharper and more defined edges.

###### S.3.2. 3D-geometry aware label prediction

###### S.3.2.1. Details of 3D-geometry aware label prediction

The detailed algorithm for 3D-Geometry Aware Label Prediction (3D-GALP) is provided in Algo. 1. 3D-GALP produces high-quality 3D segmentation maps even when part segmentation maps from multiple views are noisy, by applying a neighbor consistency loss that considers the softlabel property of Gaussian segmentation. Label softness is typically higher at part boundaries due to abrupt shape changes, which can lead to substantial variation in segmentation results across different views. Moreover, in practice, the Gaussians at these part boundaries may simultaneously represent pixels belonging to multiple parts depending on the viewpoint, further complicating consistent segmentation. To address this, Gaussians with both high and low softness are sampled, enabling continuous refinement of ambiguous as well as more view-invariant regions while taking surrounding information into account.

###### S.3.2.2. Part segmentation performance of 3D-GALP compared with other language-embedded 3DGS model in complex scenes

Experimental setting To evaluate how effectively 3DGALP performs part segmentation in complex scenes, we annotated part segmentation for every object in all scenes of the 3D-OVS dataset [35]. We compared 3D-GALP with two text-aligned segmentation models for 3D Gaussians, LangSplat [45] and LeGaussian [51]. We kept hyperparameter, the softmax value for our 2D attention map extraction, to 0.2 during segmentation. We then evaluated part-segmentation results for each object from three different views, comparing them against ground truth using the mean Intersection over Union (mIoU). Examples of partsegmentation annotation are presented in Fig. S.5.

Experimental results As shown in Tab. S.7, our 3D segmentation method, 3D-GALP, achieves the highest mIoU,

Original Annotated part Original Annotated part

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

“body, hair, face of a barbie” “wheels, body of a car”

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

“shirts, head, body of a pooh” “face, body, ears of a rabbit”

Figure S.5. Examples of part segmentation annotation in 3DOVS dataset.

outperforming other 3DGS segmentation baselines across all scenes. Furthermore, 3D-GALP successfully performs open-vocabulary 3DGS segmentation for parts of varying sizes in complex scenes, as illustrated in Fig. S.11.

Scene Bench Blue sofa Cov.desk Room Average

LangSplat [45] 0.005 0.076 0.093 0.129 0.076 LeGaussian [51] 0.320 0.312 0.264 0.257 0.288 3D-GALP (Ours) 0.607 0.580 0.546 0.502 0.559

Table S.7. Comparison of 3D-GALP with part segmentation on complicated 3D scenes.

###### S.3.2.3. Ablation study on SH degree

Experimental setting We ablated the SH order to analyze its effect on part-level segmentation. While low-order SH is typically sufficient for modeling lighting in color representation, part-level segmentation requires sharper spatial transitions, particularly around object boundaries. To evaluate this, we conducted experiments using the same experimental settings as in S.3.2.2 with different SH degree settings.

Experimental results As shown in Tab. S.8 and Fig. S.6, SH=3 consistently provides the best average mIoU across

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

SH = 1 SH = 2 SH = 3 SH = 4

[Figure 387]

[Figure 388]

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

Figure S.6. Part-level segmentation visusalizations with different SH orders.

scenes and captures fine-grained parts more clearly than lower orders. Although SH=4 performs best in some scenes, it introduces more noise and higher memory usage, leading to slightly worse overall performance. Based on these observations, we fix SH=3 for all segmentation experiments, as it provides the best trade-off between detail preservation and stability.

Order of SH 1 2 3 4 mIoU 0.4777 0.5306 0.5587 0.5506

Table S.8. mIoU average scores across the scenes per SH degree. Best per scene is in bold.

###### S.3.3. Scheduled latent mixing and part editing

###### S.3.3.1. Scheduled latent mixing and part editing

The detailed algorithm is provided in Algo. 2. This method leverages the property of rectified flow that is more faithful to the original image. During the editing process, αbase is multiplied by the mask to ensure that regions outside the target editing area retain their original information. This introduces weak conditioning at intermediate steps of image generation, guiding the generated regions to align with the original context. At the timestep ts, αlast is applied to ensure that most of the Minv regions are replaced with ztarget, preserving the majority of the reference image’s information in the final output. Further results on the selection of ts are shown in Fig. S.14. A low ts induces dramatic changes based on the prompt, while a high ts ensures faithful adherence to the mask, taking into account the original content and its context. In the ts selection described in the main paper, we randomly selected 100 person images from the CelebAMaskHQ [30] dataset, performed part-level editing using 25 prompts, and evaluated the results using CLIPdir [14] and SSIM to assess the direction of change while preserving the original content. The full experimental results with 25 prompts are shown in Fig. S.7.

###### S.3.3.2. Comparison of SLaMP with other image editing models

Experimental setting To evaluate the effectiveness of our SLaMP in preserving non-target regions while accurately modifying only the specified parts compared to other

[Figure 399]

Sweet spot

Figure S.7. Statistical result for finding sweet spot using CLIP and SSIM results.

models, we randomly selected 15 male and female images from the CelebAMaskHQ [30] dataset. For each image, we performed image editing using 25 prompts as described in Sec. S.1.1.1. For comparison, we selected SD3based models (SD3-inpainting [65], Plug&Play [13], RFinversion [49]), as well as an editing model based on naive latent mixing (RePaint [37]), in contrast to our scheduled latent mixing approach. Additionally, we include a trainingbased model, InstructPix2Pix (IP2P [2]), which is commonly adopted in 3DGS and NeRF editing approaches. For RePaint, we used a Stable Diffusion-integrated variant from HuggingFace Diffusers [57] library since RePaint is not originally designed for text-based image editing. We evaluated how well the changes aligned with the prompts using the CLIPdir [14] and B-VQA [21] metrics.

Metrics RePaint iP2P SD3-inp. Plug&Play RF-inv. SLaMP

[37] [2] [13] [65] [49] (Ours)

CLIPdir ↑ 0.111 0.117 0.147 0.044 0.089 0.165 B-VQA ↑ 0.439 0.668 0.693 0.564 0.740 0.758

Table S.9. Quantitative comparison of SLaMP with other 2D part editing baselines.

Experimental results The quantitative experimental results are presented in Tab. S.9, and the qualitative results in Fig. S.12. SLaMP outperforms all other 2D image editing baselines across all metrics, including CLIPdir [14] and BLIP-VQA [21]. Unlike baselines that either fail to reflect the prompt or fail to preserve the original context, SLaMP produces significant changes in the target part while accurately maintaining the untouched regions, achieving strong alignment with the text prompt.

As shown in Fig. S.12, the widely used 2D image editing baseline for 3D editing research, iP2P [2], struggles to perform meaningful part edits and often deviates from the original image context. This helps explain why existing 3D editing models often produce no visible changes in part editing tasks. RePaint [37] employs a fixed blending ratio for harmonized inpainting, making it unsuitable for strong, prompt-driven part-level edits. In contrast, SLaMP adopts

a scheduled blending strategy that enables bold edits early on and gradually preserves global context, achieving both precise modifications and faithful preservation. Additional results of SLaMP editing can be found in Fig. S.13.

##### S.4. Social Impact and Limitations

In our methodology, we utilized existing datasets from prior works [2, 58]. These datasets include information about real individuals, and if the results of our editing approach are misused, it could lead to concerns regarding negative societal impacts. Therefore, we strongly advocate for the responsible use of our methodology in adherence to ethical guidelines and relevant laws. In perspective on limitation, our approach relies on 3D segmentation based on attention maps observed from 360-degree viewpoints. Consequently, it may not perform well when dealing with objects with highly complex geometries (e.g., a Klein bottle), leading to unintended editing results. Additionally, if the Gaussian Splatting scene is inherently blurry or poorly reconstructed, it becomes difficult to distinguish individual components. This can cause SD3 to fail in accurately interpreting the scene, resulting in incorrect 3D segmentation or undesired editing outcomes.

Algorithm 1: Algorithm of 3D-geometry aware label prediction (3D-GALP).

Input: Gaussian Representation Ω, Camera Parameters C, Number of Anchors K, Nearest Neighbors k, Segmentation Labels slabels

Output: Segmentation Loss L3D // Initialize multi-view camera

dataset

- 1 Dtest ← LoadMultiviewDataset(C) // Compute SH consistency
- 2 S ← Ω.get sh objects()

- 3 T ← ∅ // Store SH values for different views
- 4 foreach b in Dtest do

- 5 d ← ComputeViewDirection(b,C)
- 6 sb ← EvalSH(Ω,S,d)
- 7 T ← T ∪ sb // Compute variance and entropy for

each Gaussian

- 8 foreach Gaussian i in Ω do

- 9 Compute variance: vi ← |T1| r∈T ∥r − ¯r∥2, where ¯r = |T1| r∈T r

- 10 Compute entropy: sim ← ¯r·R

labels

∥¯r∥∥Rlabels∥

- 11 pi ← e

sim esim

- 12 Hi ← − pi log(pi + ϵ) // Compute entropy
- 13 Compute label softness: Ui ← Hi · vi // Anchor Selection Based on label

softness

- 14 Sort all Gaussians by Ui in descending order
- 15 Select ⌊K/2⌋ anchors with highest Ui
- 16 Select ⌊K/2⌋ anchors with lowest Ui
- 17 Define set of selected anchors: S // Compute Anchor-Based Neighbor

Consistency Loss

- 18 foreach anchor i ∈ S do

- 19 Find nearest neighbors Nk(i) = {j1,...,jk} using Euclidean distance
- 20 Compute L1 loss:

L3D ← i∈S

1 k j∈Nk(i) ∥ri − rj∥1

- 21 return L3D

Algorithm 2: Scheduled latent mixing and part editing Algorithm

Input: Latents z, Text Embeddings E, Camera Condition C, Timestep T, Noise ntarget, Cfg scale c, γ, ηvalues, αbase, αlast, Mask M, , Mix timestep ts

Output: Model Prediction mpred // Latent Initialization and Noise

Target

- 1 for tcurr,tprev in timesteps[: −1],timesteps[1 :] do

- 2 t ← tcurr × 1000
- 3 vpred ← transformer(znoisy,t,Euncond)
- 4 vtarget ← (ntarget − znoisy)/(1 − tcurr)
- 5 vinterp ← γ · vtarget + (1 − γ) · vpred
- 6 znoisy ← znoisy + (tprev − tcurr) · vinterp

- 7 ztarget ← z.clone
- 8 for t in timesteps do

- 9 t ← t/1000
- 10 vpred ← transformer(znoisy,t,Emix)
- 11 vtarget ← −(ztarget − znoisy)/t
- 12 η ← ηvalues[i]
- 13 vinterp ← vpred + η · (vtarget − vpred)
- 14 znoisy ← scheduler.step(vinterp,t,znoisy)
- 15 F ← αlast if i > |timesteps| − ts else αbase
- 16 Minv ← F × (1 − M) znoisy ← znoisy × (1 − Minv) + ztarget × Minv

- 17 mpred ← znoisy
- 18 return mpred

A superman with face of a skeleton with face of a joker

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

with face of a clown with face of an alien with face of a batman

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

A bird with dragon’s wings with mechanical wings

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

with butterfly wings with flames as wings with bat’s wings

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

A man with fluffy pink cottoncandy hair with sharp, silver metallic spikes hair

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

A cup of coffee with rainbow straw with gold & silver straw

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

- Figure S.8. Additional qualitative results of RoMaP. Our approach, RoMaP, enables editing across a wide range of parts, objects, and prompts in generated 3D Gaussians, further providing users with enhanced controllability over 3D content generation.

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

Original 3D asset

“~ with cloudy hair”

“~ with tomato nose”

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

“~ with blue fire hair” “~ with gold bell nose＂

“~ with duck’s beak”

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

“~ with closed eyes”

“~ with rounded nose”

“~ with yellow hair”

“~ with slight smile”

“A photo of man”

(a) Additional results for enhanced controllability in 3D asset generation

GSGEN [5] GaussianDreamer [37] RFDS [36] Original RoMaP(Ours)

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

”A snail with beautiful skyscape inside of shell”

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

”A woman with golden bell nose”

(b) Additional qualitative comparison with 3DGS generation models

- Figure S.9. Additional qualitative results of RoMaP. Our approach, RoMaP, enables editing across a wide range of parts, objects, and prompts in generated 3D Gaussians, further providing users with enhanced controllability over 3D content generation.

###### GaussCtrl [35] GaussianEditor [4] DGE [3]

###### Original GaussCtrl GaussianEditor DGE RoMaP(Ours)

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

”A man with flowered lips”

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

[Figure 536]

”A man with green lips”

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

”A man with delicious blueberry nose”

- Figure S.10. Additional comparison results of RoMaP. Our approach, RoMaP, enables editing across a wide range of parts, objects, compare to other methods in 3D scene reconstruction settings.

“Petals, center of a flower”

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

Original Annotated part Part-segmentation result of LangSplat [25]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

Part-segmentation result of LeGaussian [28]

Part-segmentation result of 3D-GALP (Ours)

“Body, head of a shampoo”

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

Original Annotated part Part-segmentation result of LangSplat [25]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

Part-segmentation result of LeGaussian [28]

Part-segmentation result of 3D-GALP (Ours)

“Body, display of a remote controller”

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

Original Annotated part Part-segmentation result of LangSplat [25]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

Part-segmentation result of LeGaussian [28]

Part-segmentation result of 3D-GALP (Ours)

“Body, head of a chicken”

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

Original Annotated part Part-segmentation result of LangSplat [25]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

Part-segmentation result of LeGaussian [28]

Part-segmentation result of 3D-GALP (Ours)

###### Figure S.11. Open-voca part segmentation results comparison in complicated 3DGS scenes of 3D-OVS dataset.

Original IP2P Plug & Play SD3-inpainting

Original IP2P [2] Plug & Play [36] SD3-inpainting [9] SLaMPSLaMP (Ours)(ours)

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

Prompt : A woman with croissant nose

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

Prompt : A woman with left blue, right green eyes

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

Prompt : A woman with pink-colored eyes

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

Prompt : A man with green lips

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

Prompt : A man with gold-textured nose

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

Prompt : A man with diamond lips

Figure S.12. Local editing results between SLaMP and 2D image editing methods. SLaMP editing employs rectified flow inversion to achieve effective modifications while maintaining the original context in unedited regions. This contrasts with 2D image editing baselines, which struggle to edit the specified part in alignment with the text prompt.

Original SLaMP edited images

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

###### Figure S.13. More 2D part editing results with SLaMP.

Original 𝑡 = 0 𝑡 = 1 𝑡 = 3 𝑡 = 5 𝑡 = 7

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

𝑡 = 9 𝑡 = 11 𝑡 = 13 𝑡 = 15 𝑡 = 17 𝑡 = 19

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

Prompt : A man with diamond nose

Original 𝑡 = 0 𝑡 = 1 𝑡 = 3 𝑡 = 5 𝑡 = 7

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

𝑡 = 9 𝑡 = 11 𝑡 = 13 𝑡 = 15 𝑡 = 17 𝑡 = 19

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

Prompt : A man with delicious croissant nose

###### Figure S.14. Effect of different ts in SLaMP editing.

### Original Timestep 1000 Timestep 2000 Timestep 3500

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

Prompt : “Turn his hair into silver-textured hair”

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

Prompt : “Turn his hair into hair on fire, red and blue flames”

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

Prompt : “Turn his left eye blue and right eye green”

###### Figure S.15. Qualitative results of nerf baseines [17] in 3D part editing.

1) Prompt : “Turn his nose into a delicious croissant” Part segmentation (“nose”) results

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

2D multi view edit results

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

Original Final edited 3D result

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

| |
|---|

| |
|---|

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

2) Prompt : “Turn his eyes pink” Part segmentation (“eyes”) results

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

2D multi view edit results

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

Original Final edited 3D result

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

###### Figure S.16. Qualitative results of 3DGS baseine [6] in 3D part editing.

##### References

- [1] Saba Ahmadi and Aishwarya Agrawal. An examination of the robustness of reference-free image captioning evaluation metrics. ACL Anthology, 2023. 10
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 2, 3, 7, 10, 15, 16
- [3] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In CVPR, 2022. 4
- [4] Honghua Chen, Yushi Lan, Yongwei Chen, Yifan Zhou, and Xingang Pan. Mvdrag3d: Drag-based creative 3d editing via multi-view generation-reconstruction priors. arXiv preprint arXiv:2410.16272, 2024. 3, 5, 6
- [5] Minghao Chen, Iro Laina, and Andrea Vedaldi. Dge: Direct gaussian 3d editing by consistent multi-view editing. ECCV,

2024. 2, 3, 6, 7, 10, 11, 13

- [6] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In CVPR, 2024. 2, 3, 5, 6, 7, 10, 11, 13, 26
- [7] Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. Text-to-3d using gaussian splatting. In CVPR, 2024. 7, 13
- [8] Ho Kei Cheng, Jihoon Chung, Yu-Wing Tai, and Chi-Keung Tang. Cascadepsp: Toward class-agnostic and very highresolution segmentation via global and local refinement. In CVPR, 2020. 13
- [9] Xinhua Cheng, Tianyu Yang, Jianan Wang, Yu Li, Lei Zhang, Jian Zhang, and Li Yuan. Progressive3d: Progressively local editing for text-to-3d content creation with complex semantic prompts. ICLR, 2023. 2, 3, 10
- [10] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. NeurIPS, 2021. 2
- [11] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. ICLR, 2016. 2
- [12] Jiahua Dong and Yu-Xiong Wang. Vica-nerf: Viewconsistency-aware 3d editing of neural radiance fields. NeurIPS, 2023. 2, 3, 6, 7, 10, 11
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2, 10, 15
- [14] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clipguided domain adaptation of image generators. 2022. 6, 10, 15
- [15] Pengsheng Guo, Hans Hao, Adam Caccavale, Zhongzheng Ren, Edward Zhang, Qi Shan, Aditya Sankar, Alexander G Schwing, Alex Colburn, and Fangchang Ma. Stabledreamer: Taming noisy score distillation sampling for textto-3d. arXiv preprint arXiv:2312.02189, 2023. 3, 5, 6

- [16] Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen, ZiXin Zou, Chen Wang, Yan-Pei Cao, and Song-Hai Zhang. threestudio: A unified framework for 3d content generation. https://github.com/threestudio-project/ threestudio, 2023. 10
- [17] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. In CVPR, 2023. 2, 3, 6, 7, 10, 11, 13, 25
- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. ICLR, 2022. 2
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 2
- [20] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In CVPR, 2023. 6, 10
- [21] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. 2023. 6, 10, 15
- [22] Junha Hyung, Sungwon Hwang, Daejin Kim, Hyunji Lee, and Jaegul Choo. Local 3d editing via 3d distillation of clip knowledge. In CVPR, 2023. 4
- [23] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. Noise-free score distillation. ICLR, 2023. 3, 5
- [24] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023. 2
- [25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 2023. 2, 4
- [26] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19729–19739,

2023. 13

- [27] Daniel Khashabi, Yeganeh Kordi, and Hannaneh Hajishirzi. Unifiedqa-v2: Stronger generalization via broader crossformat training. 2022. 10
- [28] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 2, 3
- [29] Juil Koo, Chanho Park, and Minhyuk Sung. Posterior distillation sampling. In CVPR, 2024. 3, 6, 7, 10, 11
- [30] Cheng-Han Lee, Ziwei Liu, Lingyun Wu, and Ping Luo. Maskgan: Towards diverse and interactive facial image manipulation. In CVPR, 2020. 15
- [31] Hangyu Li, Xiangxiang Chu, and Dingyuan Shi. Dreamcouple: Exploring high quality text-to-3d generation via rectified flow. arXiv preprint arXiv:2408.05008, 2024. 3

- [32] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,

2022. 10

- [33] Runjia Li, Junlin Han, Luke Melas-Kyriazi, Chunyi Sun, Zhaochong An, Zhongrui Gui, Shuyang Sun, Philip Torr, and Tomas Jakab. Dreambeast: Distilling 3d fantastical animals with part-aware knowledge transfer. 3DV, 2025. 3
- [34] Yuhan Li, Yishun Dou, Yue Shi, Yu Lei, Xuanhong Chen, Yi Zhang, Peng Zhou, and Bingbing Ni. Focaldreamer: Textdriven 3d editing via focal-fusion assembly. In AAAI, 2024. 3, 10
- [35] Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, and Shijian Lu. Weakly supervised 3d openvocabulary segmentation. NeurIPS, 2023. 13, 14
- [36] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. ICLR, 2023. 2
- [37] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11461–11471, 2022. 15
- [38] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2
- [39] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. ECCV, 2020. 2
- [40] Francesco Palandra, Andrea Sanchietti, Daniele Baieri, and Emanuele Rodol`a. Gsedit: Efficient text-guided editing of 3d objects via gaussian splatting. arXiv preprint arXiv:2403.05154, 2024. 2, 3
- [41] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019. 10
- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In CVPR, 2023. 2
- [43] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. 2023. 10
- [44] Tianhao Qi, Shancheng Fang, Yanze Wu, Hongtao Xie, Jiawei Liu, Lang Chen, Qian He, and Yongdong Zhang. Deadiff: An efficient stylization diffusion model with disentangled representations. In CVPR, 2024. 2
- [45] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In CVPR, 2024. 14
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 4, 6, 10

- [47] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 2

- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2
- [49] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 3, 13, 15
- [50] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. NeurIPS, 2021. 2
- [51] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and ShaoHua Guan. Language embedded 3d gaussians for openvocabulary scene understanding. In CVPR, 2024. 14
- [52] Yaya Shi, Xu Yang, Haiyang Xu, Chunfeng Yuan, Bing Li, Weiming Hu, and Zheng-Jun Zha. Emscore: Evaluating video captioning via coarse-grained and fine-grained embedding matching. In CVPR, 2022. 10
- [53] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 2
- [54] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. 2023. 10
- [55] Christina Tsalicoglou, Fabian Manhardt, Alessio Tonioni, Michael Niemeyer, and Federico Tombari. Textmesh: Generation of realistic 3d meshes from text prompts. In 3DVS,

2024. 10

- [56] Cyrus Vachha and Ayaan Haque. Instruct-gs2gs: Editing 3d gaussian splats with instructions, 2024. 3
- [57] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models, 2022. 15
- [58] Can Wang, Ruixiang Jiang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Nerf-art: Text-driven neural radiance fields stylization. TVCG, 2023. 2, 10, 16
- [59] Feng Wang, Jieru Mei, and Alan Yuille. Sclip: Rethinking self-attention for dense vision-language inference. In ECCV,

2025. 13

- [60] Junjie Wang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, and Qi Tian. Gaussianeditor: Editing 3d gaussians delicately with text instructions. In CVPR, 2024. 2, 3
- [61] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021. 2
- [62] Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian Reid, Philip Torr, and Victor Adrian Prisacariu. Gaussctrl:

- multi-view consistent text-driven 3d gaussian splatting editing. ECCV, 2024. 2, 3, 6, 7, 10, 11, 13
- [63] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, 2023. 2
- [64] Yunqiu Xu, Linchao Zhu, and Yi Yang. Gg-editor: Locally editing 3d avatars with multimodal large language model guidance. In ACM International Conference on Multimedia,

2024. 2, 3

- [65] Xiaofeng Yang, Cheng Chen, Xulei Yang, Fayao Liu, and Guosheng Lin. Text-to-image rectified flow as plug-and-play priors. arXiv preprint arXiv:2406.03293, 2024. 3, 7, 13, 15
- [66] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. NeruIPS, 2021. 2
- [67] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In CVPR,

2024. 7, 13

- [68] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In CVPR, 2023. 7

