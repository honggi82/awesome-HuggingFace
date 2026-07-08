arXiv:2505.23907v1[cs.CV]29May2025

Cora: Correspondence-aware image editing using few step diffusion

AMIRHOSSEIN ALIMOHAMMADI∗, Simon Fraser University, Canada ARYAN MIKAEILI∗, Simon Fraser University, Canada SAURADIP NAG, Simon Fraser University, Canada NEGAR HASSANPOUR, Huawei, Canada ANDREA TAGLIASACCHI, Simon Fraser University, University of Toronto, Google Deepmind, Canada ALI MAHDAVI-AMIRI, Simon Fraser University, Canada

[Figure 1]

- Fig. 1. Cora supports diverse edits, including object insertion, subject and background changes, and non-rigid deformations (e.g., jumping). Our novel correspondence-aware method provides strong control and flexibility for both appearance and structure editing.

Project Page: cora-edit.github.io 1

Image editing is an important task in computer graphics, vision, and VFX, with recent diffusion-based methods achieving fast and high-quality results. However, edits requiring significant structural changes, such as non-rigid deformations, object modifications, or content generation, remain challenging. Existing few step editing approaches produce artifacts such as irrelevant texture or struggle to preserve key attributes of the source image (e.g., pose). We introduce Cora , a novel editing framework that addresses these limitations by introducing correspondence-aware noise correction and interpolated attention maps. Our method aligns textures and structures between the source and target images through semantic correspondence, enabling accurate texture transfer while generating new content when necessary. Cora offers control over the balance between content generation and preservation. Extensive experiments demonstrate that, quantitatively and qualitatively, Cora excels in maintaining structure, textures, and identity across diverse edits, including pose changes, object addition, and texture refinements. User studies confirm that Cora delivers superior results, outperforming alternatives.

1 INTRODUCTION

Image editing is an important task in fields such as computer graphics, computer vision, and VFX. Recent diffusion-based, few-step image editing techniques have significantly improved this process, enabling fast and effective edits with impressive results across diverse scenarios [10, 58, 60].

Despite these impressive advancements, edits that require structural changes that go beyond pixel color alteration (e.g., non-rigid edits, object change) remain a challenging task for diffusion models. TurboEdit [10] that relies on noise correction to perform edits, often produces artifacts and cannot necessarily preserve identity or important properties of the source image (e.g., pose); see Fig. 2. This is because these corrections do not account for the fact that the generated and source image may not any longer be pixel-aligned after the

Authors’ addresses: Amirhossein Alimohammadi∗, Simon Fraser University, Canada; Aryan Mikaeili∗, Simon Fraser University, Canada; Sauradip Nag, Simon Fraser University, Canada; Negar Hassanpour, Huawei, Canada; Andrea Tagliasacchi, Simon Fraser University, and University of Toronto, and Google Deepmind, Canada; Ali MahdaviAmiri, Simon Fraser University, Canada.

1∗ Authors have contributed equally

edit. We resolve this shortcoming by introducing “correspondenceaware” noise corrections that connect source to target pixels by matching their diffusion features.

Edits involving significant deformation of the subject in the image (Fig. 2), often require the generation of new parts, or the exposure of regions not present in the source image. Some approaches that aim to respect the source for such edits primarily rely on the source image for texture information to maintain the subject’s identity [8]. While this strategy is somewhat effective, since they inject the intermediate features of the diffusion model from the source image into the self-attention modules [8], their edits copy undesired texture from the source into regions of the target image with no clear correspondence (Fig 3:b).

One of our technical contributions is to combine the keys and values that carry texture information from both source and target. This enables the network to generate content when needed while accurately copying textures when relevant information is available in the source image. However, simple methods for combining the source and target, such as concatenation, fail to achieve the desired results (see Sec.4.2). We show that interpolating attention maps enhances performance while offering flexibility and control in both generating new content and preserving existing content.

To achieve the right textures while respecting the structure of the source image, it is also necessary to align attentions by establishing a semantic correspondence. Therefore, we incorporated a correspondence technique (DIFT) into our method whenever source information is available. This technique aligns the attention maps (i.e., keys and values) of the source and target, enabling a more accurate and effective transfer of relevant textures. In the early stages of generation, the model’s output is primarily noise, making correspondence infeasible. Therefore, in a four-step diffusion process, we initiate the correspondence process at the last two steps, where the image structure is established, but textures are still being refined. To align the structure of the source and target images, we apply a permutation on queries that are obtained using a matching algorithm. This alignment is performed in the first step of generation, as the image structure takes shape during this phase.

[Figure 2]

- Fig. 2. Comparison between TurboEdit [10] and our correspondence-aware editing approach. Due to misalignment between the source and target images, artifacts are visible in TurboEdit results, such as texture inconsistencies in (a), silhouette artifacts in the legs and fins in (b, d), and undesired elements in (c). Please zoom in for a clearer view of these artifacts.

By combining these strategies, we present Cora, a novel editing method built upon a few-step text-to-image model, SDXL-Turbo [47].

We demonstrate that Cora delivers improved visual results for various edits,thanks toourinnovative attention mixing and correspondenceaware techniques. Cora not only excels at preserving the structure of the image and maintaining textures but also supports a wide range of edits, including non-rigid edits (e.g., pose changes), object addition and removal, and texture modifications; see Figure 1. Furthermore, quantitative and qualitative experiments and user studies demonstrate the effectiveness of each component of our method and confirm that Cora surpasses other alternatives.

2 RELATED WORKS

Image editing with diffusion models. Diffusion-based text-toimage models are known for their ability to produce high-quality and diverse outputs [4, 41, 42, 44]. These models start with Gaussian noise, and iteratively denoise it to generate the final image [19, 48]. Beyond generation, they can also be leveraged as a powerful prior for visual content editing. This is achieved by adding noise to an image and iteratively denoising it with a new text prompt [32], but this method often struggles to balance content preservation with prompt alignment. To address these challenges, follow-up works tweaked the architecture, for example by including editing masks, to better preserve the content in the original image [3, 16, 33, 35, 38, 43]. Another approach involves training diffusion models on purposebuilt datasets with images, instructional prompts, and ground truth edits [6], or fine-tuning models on the source image for improved content preservation [22]. However, since training these methods can be both time-consuming and resource-intensive, recent works have focused on leveraging the features of diffusion models for zero-shot editing.

Diffusion models’ features. The features of diffusion models have been shown to contain rich spatial and semantic information. These features can be used for tasks such as point correspondence [14, 15, 30, 51], image and video segmentation [2, 23], and conditional image generation [5, 29]. Some approaches leverage the cross-attention maps of diffusion models for localized text-based image editing [17, 34]. [54] inject self-attention and intermediate features of the source image into the generation process of the target image for better content preservation. MasaCtrl [8] observes that using the keys and values of the self-attention modules from the source image allows copying its appearance into the target image while enabling nonrigid editing. However, as we will show, only using features from the source image is not sufficient for generating new content in the edited image.

Later works use this observation for style transfer [1], styleconsistent generation [18], image editing [24, 28, 37] and video editing [12, 39]. ConsiStory [52] employs attention sharing, and similar to us, employs DIFT-aligned [51] feature maps for consistent image generation. Several studies have explored the use of diffusion features for image interpolation, such as [46] that investigates various interpolation strategies in the noise space of diffusion models, or [13] that examines linear interpolation within the attention space to achieve realistic inbetweening. In Cora, we apply correspondence to the inverted latents of an image to enable edits involving significant structural changes. Also, interpolation is performed on aligned attentions to balance content preservation and editing flexibility.

[Figure 3]

- Fig. 3. Attention mixing strategies. (b): Using only the source image’s keys and values causes artifacts and misalignment with the edit text-prompt. (c): Concatenating keys and values leads to undesired appearance bleeding between source and target images. (d, e): Aligning and interpolating keys and values produces the best results, with slerp providing more realistic and natural outcomes compared to lerp.

Inversion in diffusion models. One way to edit images involves inverting the diffusion model: determining the noise that, when fed to the diffusion model, produces the source image [49]. Some methods focus on improving this inversion process for more effective editing. DDIM Inversion reverses the deterministic DDIM denoising process [49], but accumulates small errors during inversion; these errors can lead to significant content drift, especially when using classifier-free guidance [20] with high guidance [34]. To address this issue, some works rely on iteratively optimizing the intermediate noisy images [11, 26, 36]. Another line of inversion algorithms [21, 57] predicts both the initial as well as intermediate noise maps in the DDPM process, enabling better reconstruction and higher editing quality. In our work, we build upon the DDPM inversion proposed by [21] for its speed, high reconstruction quality, and versatility across editing tasks.

Few step diffusion models. Because of the iterative nature of the diffusion process, these methods are usually slow, requiring 20-100 forward passes of the model. Recent research have focused on designing few-step frameworks using distillation [45, 61], consistency constraint [31, 50, 59] and adversarial training [47], enabling image generation in 1-8 steps. While these methods are faster, adapting

- them for editing is not trivial. Several recent works attempted to adapt few-step diffusion to editing tasks. [58] trains an encoder for fast inversion, [60] proposes a novel inversion-free framework, and [10] adapts editing-friendly DDPM inversion to few-step models. These methods focus on appearance changes with minimal structural edits, while the proposed Cora supports both.

3 PRELIMINARIES

Noise-inversion. [21] maps an input image 𝑥0 to

{𝑥𝑇,𝑧𝑇−1, . . .,𝑧1,𝑧0}, (1)

where 𝑥𝑇 represents the inverted noise at the final timestep 𝑇, and 𝑧𝑡 denotes correction terms. Given 𝑥0, the algorithm first computes its noisy versions across all timesteps in forward (fw) diffusion:

√1 − 𝛼¯𝑡𝜖˜𝑡, 𝑡 = 1, . . .,𝑇, (2)

𝑥𝑡fw = √𝛼¯𝑡𝑥0 +

where 𝜖˜𝑡 is independent standard Gaussian noise, and 𝛼¯𝑡 is a parameter of the diffusion scheduler. In the backward (bw) process, the

scheduler calculates 𝑥𝑡bw−1 as:

𝑥𝑡bw−1 = 𝜇𝑡 (𝑥𝑡bw,𝑐) + 𝜎𝑡𝑧𝑡, (3) where 𝑐 is the text prompt used for inversion, and

1 √𝛼𝑡

1 − 𝛼𝑡 √1 − 𝛼𝑡¯

𝜇𝑡 (𝑥𝑡bw,𝑐) =

𝑥𝑡bw −

𝜖𝜃 (𝑥𝑡bw,𝑡,𝑐) (4)

is the predicted 𝑥𝑡−1 given 𝑥𝑡bw, 𝜖𝜃 (𝑥𝑡bw,𝑡,𝑐) is the output of the diffusion model, and 𝜎𝑡 is the variance of the scheduler.

At first glance, it is evident that if 𝑧𝑡 is standard Gaussian noise,

as in the typical denoising process, then 𝑥𝑡bw−1 and 𝑥𝑡fw−1 will not necessarily match. However, by setting 𝑧𝑡 as:

𝑥𝑡fw−1 − 𝜇𝑡 (𝑥𝑡bw,𝑐) 𝜎𝑡

, (5)

𝑧𝑡 =

we ensure 𝑥𝑡bw−1 ≡ 𝑥𝑡fw−1. Therefore, using the same text prompt 𝑐 for generation leads to perfect reconstruction. Modifying the text

prompt to a desired edit 𝑐ˆ enables editing, provided the edit does not involve significant structural changes.

TurboEdit. [10] introduce a series of modifications to the inversion algorithm to adapt it for the few-step setting, including timeshifted inversion, norm-clipping at the final denoising step, and text-guidance to improve prompt alignment. Our method incorporates these modifications.

4 METHOD

We build Cora on top of the few-step diffusion model (i.e., SDXLTurbo[47]),enablingboth appearanceediting and structural changes.

Outline. In Section 4.1 we observe that using noise inversion for structural editing introduces artifacts (Fig. 2). We explain how and why a hierarchical correspondence-aware latent correction can resolve these issues. Second, edits that involve significant structural changes or object additions require generating entirely new content or revealing regions absent in the source image. Directly using the keys and values of the source image in the self-attention modules of the diffusion model, as done in MasaCtrl [8], often results in unwanted artifacts or misalignment between the target image and appearance editing instructions in the text-prompt (Fig. 3:b). To

mitigate this, it is necessary to combine the keys and values of both the source and target images. In Section 4.2, we explore various strategies for this combination and introduce a novel method called correspondence-aware attention interpolation. We propose two strategies for interpolation: spherical (SLERP) and linear, and show that SLERP is beneficial. Finally, in Section 4.3, we demonstrate that by matching the queries of the source and target images, we enable control over the extent of structural change in the target image.

- 4.1 Correspondence-aware latent correction As explained in Section 3, noise-inversion maps the input image

𝑥0 to {𝑥𝑇,𝑧𝑇−1, . . .,𝑧1,𝑧0}, where {𝑧𝑡} serve as corrections, ensuring perfect reconstruction when the text prompt 𝑐 used for inversion matches the one used for generation. However, when the text prompt requires a large deformation of the source image, the corrections {𝑧𝑡} are not pixel-aligned with respect to the generated image, leading to severe artifacts (Fig. 2). To address this issue, we align the corrections {𝑧𝑡} with the spatial transformation introduced in the edited image in the final two steps of denoising. This involves constructing a correspondence map𝐶𝑇→𝑆 between the source image 𝐼𝑆 and the target image 𝐼𝑇 . We achieve this via their DIFT features [51], denoted as 𝐷𝑆 and 𝐷𝑇 :

𝐶𝑇→𝑆 (𝑝) = argmax 𝑞∈𝐼𝑆

sim(𝐷𝑇 (𝑝),𝐷𝑆 (𝑞)), (6)

where 𝑝 and 𝑞 are pixels from the target and source images, respectively, and sim is the cosine similarity between the two feature vectors. We construct an aligned correction term 𝑧𝑡aln as:

𝑧𝑡aln(𝑝) = 𝑧𝑡 (𝐶𝑇→𝑆 (𝑝)). (7) Patch correspondence. Since DIFT features might be noisy and inaccurate, the resulting target image often contains artifacts(inset left). Therefore, we propose a patch-wise correspondence approach. The DIFT features 𝐷𝑆 and 𝐷𝑇 are divided into small, overlapping patches, and correspondence is computed for each patch rather than individual pixels. For each patch, we concatenate its pixelwise DIFT features and calculate cosine similarity between these concatenated features. Due to overlapping patches, multiple patches may contribute to the alignment of a single pixel 𝑝. To obtain the final aligned correction latent 𝑧𝑡aln(𝑝), we average the contributions from all overlapping patches at pixel 𝑝.

As denoising progresses and the features become less noisy at later timesteps, the size of the patches is gradually reduced. This ensures that the correspondence becomes more precise, adapting dynamically to the evolving reliability of the features (see inset -right).

[Figure 4]

4.2 Correspondence-aware attention interpolation

Achieving high-quality image editing requires balancing the preservation of key aspects of the source image (e.g., appearance or identity) with the introduction of new elements or modifications. Recent

approaches often achieve this by injecting the attention features of the source image into the denoising process of the target image [8]. While this method is effective, it overlooks the fact that editing often involves generating new content, and this content may lack clear correspondence to content stored within the source image. We now consider existing methods, and present a novel strategy for combining attentions between source and target images.

Mutual self-attention. MasaCtrl [8] uses the source image’s keys and values in the self-attention modules of the diffusion model. This ensures that the context of the source image, such as its appearance and identity are preserved:

𝑓 𝑒 = Attention(𝑄𝑇,𝐾𝑆,𝑉𝑆), (8)

where 𝑄𝑇 is the query of the target image, 𝐾𝑆 and 𝑉𝑆 are the keys and values of the source image, and 𝑓 𝑒 represents the output of the self-attention module. While this strategy effectively retains the appearance and identity of the original content, it also limits the model’s ability to generate new content, such as adding objects or significantly altering appearances (see Fig. 3:b).

Concatenation. An approach to incorporate appearance editing is concatenating the keys and values of the source and target images as done by [18], [9], and [52]:

𝑓 𝑒 = Attention 𝑄𝑇, [𝜆 · 𝐾𝑆,𝐾𝑇 ], [𝜆 ·𝑉𝑆,𝑉𝑇 ] , (9) where [ , ] denotes concatenation, and 𝜆 is a scaling factor that controls how much the source appearance affects the target image. While this enables appearance changes, it often fails to achieve a smooth interpolation between the two appearances. This can result in unnatural “appearance leakage”, such as elements of the red car blending into the white bus (see Fig. 3:c).

Linear interpolation. Another approach is to linearly interpolate the keys and values of the source and target images [13]. Differently from [13], we linearly interpolate after matching features between source and target features (Sec. 4.1), as otherwise this may cause artifacts due to the mis-alignment of source and target (see the inset):

[Figure 5]

L(𝑣1,𝑣2,𝛼) = (1 − 𝛼) · 𝑣1 + 𝛼 · 𝑣2, (10)

𝑓 𝑒 = Attention(𝑄𝑇, L(𝐾𝑆aln,𝐾𝑇,𝛼), L(𝑉𝑆aln,𝑉𝑇,𝛼)). (11) While this approach is somewhat effective, it sometimes causes unwanted artifacts when interpolating between features that are significantly different (see the mirrors in Figure 3:d). To address this limitation, we explore the use of spherical linear interpolation (SLERP) for interpolating between the keys and values, which takes vector directions into account for smoother blending:

𝑣1 |𝑣1|

𝑣2 |𝑣2|

+ sinsin((𝛼𝜃𝜃)) ·

Q(𝑣1,𝑣2;𝛼) = sin((sin1−(𝜃𝛼))𝜃) ·

, (12)

where 𝜃 is the angle between the vectors 𝑣1 and 𝑣2, and 𝛼 ∈ [0, 1] is the blending weight. The output of this operation is a unit vector.

[Figure 6]

Fig. 4. Adjusting 𝛼 Provides control over the appearance transition between the source image and the target appearance. When 𝛼 = 0, the appearance is entirely derived from the source image, and when 𝛼 = 1, it fully reflects the editing prompt. Intermediate values of 𝛼 allow for a gradual blend, enabling fine-grained control between these two extremes.

To preserve the magnitude information of the vectors, we also interpolate their magnitudes as M(𝑣1,𝑣2;𝛼) = (1 − 𝛼) · |𝑣1| + 𝛼 · |𝑣2|. Combining these, the full interpolation formula becomes:

MQ(𝑣1,𝑣2;𝛼) = M(𝑣1,𝑣2;𝛼) · Q(𝑣1,𝑣2;𝛼). (13) In the self-attention modules, this results in:

𝑓 𝑒 = Attention 𝑄𝑡, MQ(𝐾𝑆aln,𝐾𝑇 ;𝛼), MQ(𝑉𝑆aln,𝑉𝑇 ;𝛼) . (14) This ensures that transitions between source and target vectors respect their angular relationships and provides smoother and more reliable blending between the appearance of the source and target images (see Fig. 3:e). Furthermore, adjusting 𝛼 enables control over the extent of appearance changes in the target image relative to the source image (see Fig. 4).

Content-adaptive interpolation. In situations where the prompt suggests the insertion of a new object or a deformation that causes significant disocclusion, expecting that every generated/target pixel matches a pixel in the source image is not reasonable. In such scenarios, naively aligning the keys and values of the source and target images leads to text misalignment and/or visual artifacts (see Fig. 5). To address this issue, we propose to classify pixels in the target image that do not have any correspondence in the source image (i.e. “new” pixels). We achieve this by a bidirectional comparison between source and target image patches. Specifically, for each patch in the source image 𝑠 ∈ S, we compute the set 𝐾(𝑠) of k-nearest target patches:

𝐾(𝑠) = argtop𝑘

sim(𝑠,𝑡), (15)

𝑡∈T

where sim(𝑠,𝑡) is the cosine similarity from Section 4.1. Similarly, for each patch in the target image 𝑡 ∈ T, we define its top-𝑘 nearest patches as 𝐾(𝑡). We call a pair of patches (𝑠,𝑡) bidirectionally matched if 𝑡 ∈ 𝐾(𝑠) and 𝑠 ∈ 𝐾(𝑡). For bidirectionally matched patches, we use the hyper-parameter 𝛼 specified by the user, while by setting 𝛼 = 1 we could let patches being purely driven by the text,

[Figure 7]

- Fig. 5. Effect of content-adaptive interpolation. Interpolating the keys and values for target image regions without clear correspondence in the source image results in undesirable edits (b). Classifying these regions and using only the target keys and values mitigates this issue (c).

[Figure 8]

- Fig. 6. Structure Alignment. DIFT-aligned queries produce unnatural edits, while our matching algorithm with adjustable blending weights (𝛽) enables transitions between full structure alignment and new layout generation.

rather than the source image. In particular, for unmatched target patches 𝑡 ∈ U we set 𝛼 = 1 if we deem the correspondence to be particularly “weak”. We measure this weakness by computing the score of the best available match:

sim𝑚𝑎𝑥 (𝑡) = max 𝑠∈S

sim(𝑠,𝑡) (16)

and determining which subset of 𝑡 ∈ U have the worst matches by computing the 𝛾-quantile of the scores in this set:

𝜏𝛾 = quantile𝛾 {simmax(𝑡) | 𝑡 ∈ U} . (17)

finally setting 𝛼=1 whenever 𝑠𝑖𝑚𝑚𝑎𝑥 (𝑡)<𝜏𝛾. We typically set 𝛾=3% in our experiments.

4.3 Structural alignment

Preserving the overall layout of the image (i.e. preserving structure) is important when editing images. Recent works [1, 8] have demonstrated that it is the queries in the self-attention modules of diffusion models that specify the structure of the generated image. Hence, while in Section 4.2 we described key/value mixing, we now incorporate queries from the source image during the denoising process to retain the overall image structure. Our key idea is that reproducing the structure of the original image, up to a non-rigid deformation, implies that we want to find all the local structures

[Figure 9]

Fig. 7. Structural Alignment. In the first denoising step, we extract selfattention queries from both source and target images. We then define two cost matrices: 𝐶𝑆𝐴, which promotes structural alignment between source and target, and𝐶𝑇𝐶, which preserves target structure. By linearly combining these matrices, we can control the strength of alignment. The resulting cost matrix is then used in the Hungarian matching algorithm to permute the target queries, aligning them with the source’s structure.

of the source image within the generated target. We achieve this via Hungarian matching [25] between source and target queries, as this provides us with a one-to-one matching (i.e. every target query should match one source query). In particular, Hungarian matching computes the optimal permutation given a weight matrix 𝐶, which

- then shuffles our generation queries:

𝜋∗ = argmin

𝜋∈Perm(𝑁 )

∑︁𝑁

𝑛=1

𝐶[𝑛,𝜋(𝑛)], 𝑄𝑇𝜋 [𝑛] ← 𝑄𝑇 [𝜋∗(𝑛)] (18)

We define our weight matrix 𝐶 as a linear interpolation controlled by a blending weight 𝛽 between two matrices (described next):

𝐶 = (1 − 𝛽) · normalize(𝐶SA) + 𝛽 · normalize(𝐶TC), (19) where normalize(.) rescales the matrices to ensure comparability. The first matrix𝐶SA encourages the target queries to remain similar

- to the source queries, hence promoting Source Alignment: 𝐶SA[𝑖, 𝑗] = 1 − sim(𝑞𝑠 [𝑖],𝑞𝑡 [𝑗]). (20)

The second matrix𝐶TC attempts to penalize index differences among the target queries, hence promoting Target Consistency:

𝐶TC[𝑖, 𝑗] = √︁|𝑖 − 𝑗 |. (21) Varying 𝛽 provides us with control over the structure of the target image, transitioning between preserving the source structure (𝛽≈0) or adhering more to the text prompt while remaining self-consistent (𝛽≈1). An illustration of the effect of 𝛽 can be found in Figure 6. Note that this process is limited to the first step of denoising when the coarse structure of the generated image is established.

- 5 EXPERIMENTS

In this section, we present various editing results on real images, demonstrating the versatility of our method. We then compare our

- Table 1. User study. Our method has received a significantly higher score than the alternatives.

Method Average Ranking (↑)

MasaCtrl [8] 1.02 DDPM Inversion [21] 1.78 InfEdit [60] 1.67 TurboEdit [10] 2.24 Ours 3.29

- Table 2. User study. Ablation user study on different attention mixing strategies mentioned in Sec. 4.2. It is evident that correspondence-aligned SLERP interpolation yields the best results.

Method Average Ranking (↑)

Mutual 0.40 Concatenation 1.71 LERP 2.08 LERP+DIFT 2.58 SLERP+DIFT 3.23

approach with both multi-step and few-step baselines to highlight its advantages in terms of visual quality and speed. Finally, we conduct ablation studies to analyze the contribution of each component.

Qualitative Results. Fig. 8 showcases several edits generated by our 4-step denoising procedure. These examples include non-rigid deformations, inserting new objects, and replacing existing objects. Our method generally preserves the overall structure of the input image while accurately reflecting the requested edits.

To compare with existing approaches, Figure 9 illustrates that our approach is more successful at maintaining the subject’s identity and reducing artifacts. We focus on few-step baselines such as TurboEdit [58] and InfEdit [60] as they operate in a similar fewstep regime, as well as multi-step frameworks such as MasaCtrl [8] and Edit friendly DDPM inversion [21]. Our results exhibit fewer distortions and better fidelity, particularly upon closer inspection.

We further expand the comparison to multi-step methods, including Prompt-to-Prompt (P2P) [17], plug-and-play (PnP) [53], instructpix2pix [7], and StyleDiffusion [27] (see Figure 10). Despite being significantly faster, our few-step approach achieves comparable or superior results in preserving details and adhering to the edits.

User studies. Although we quantitatively show in Supplementary that across different metrics Cora is comparable or superior to alternatives, standard metrics (e.g., PSNR and LPIPS) often fail to capture the visual quality of edits with significant deformation—the key focus of our paper. Therefore, we also conducted a user study. Participants were shown the original image, the editing prompt, and outputs from our method and various baselines. They ranked the images based on (i) alignment with the prompt and (ii) preservation of the subject in the source image, using a scale from 1 (worst) to 4 (best). The average rankings are summarized in Tab.1. Feedback from 51 participants strongly favored our method over other few-step approaches and found it comparable to computationally intensive multi-step techniques. Additionally, a separate user study on attention mixing strategies revealed that correspondence-aligned SLERP interpolation produced the best results, as shown in Tab.2.

[Figure 10]

Fig. 8. Qualitative results. We demonstrate the ability of our method to perform various types of edits on multiple images.

Ablation Studies. We now examine the contributions of the main components in our framework:

Structure Alignment. Disabling structure alignment reduces background fidelity,although the edited object remains well-aligned to the text prompt. Visual comparisons (see Supplementary) confirm that structure alignment is crucial for preserving scene details.

Correspondence-Aware Latent Correction. Removing this module causes significant distortions in the edited region. Hence, the latent correction is essential for producing coherent edits. For visual results, see Supplementary.

SLERP vs. LERP. While switching from SLERP to LERP often produces similar outcomes, SLERP can yield more consistent transitions in certain challenging cases. For visual results, see Supplementary. Removing correspondence alignment From Attention. As seen in Sec. 4.2, this leads to more artifacts, as it helps enforce alignment between the modified content and the background.

- 6 CONCLUSION, LIMITATIONS, FUTURE WORK

We have introduced Cora, a novel diffusion-based image editing method that addresses the challenges of structural edits, such as

non-rigid transformations and object insertions. By leveraging innovative attention mixing and correspondence-aware techniques, our approach enables accurate texture preservation and structural alignment. Unlike existing methods, Cora effectively generates new content when required while maintaining fidelity to the source image where relevant. Our results demonstrate significant improvements in visual quality and flexibility across a wide range of editing tasks, including pose alterations, object manipulation, and texture modifications. Quantitative evaluations further validate that Cora consistently outperforms state-of-the-art methods in both quality and versatility. However, our method still suffers from some limitations. For example, prompts may change unintended parts of the image (e.g., changing a car’s color might also affect the background). This can be resolved by using masks automatically obtained via cross and self-attention. While this is a promising direction, it is challenging with only four steps denoising and can be considered as a future work. Another future directions could be to extend Cora to video editing or evaluate alternative non-linear interpolation techniques for attentions.

REFERENCES

- [1] Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar Averbuch-Elor, and Daniel Cohen-Or. 2023. Cross-Image Attention for Zero-Shot Appearance Transfer. arXiv:2311.03335 [cs.CV]
- [2] Amirhossein Alimohammadi, Sauradip Nag, Saeid Asgari Taghanaki, Andrea Tagliasacchi, Ghassan Hamarneh, and Ali Mahdavi Amiri. 2024. SMITE: Segment Me In TimE. arXiv:2410.18538 [cs.CV] https://arxiv.org/abs/2410.18538
- [3] Omri Avrahami, Dani Lischinski, and Ohad Fried. 2022. Blended Diffusion for Text-Driven Editing of Natural Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 18208–18218.
- [4] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. 2022. eDiff-I: Text-to-Image Diffusion Models with Ensemble of Expert Denoisers. arXiv preprint arXiv:2211.01324 (2022).
- [5] Shariq Farooq Bhat, Niloy J. Mitra, and Peter Wonka. 2023. LooseControl: Lifting ControlNet for Generalized Depth Conditioning. arXiv:2312.03079 [cs.CV]
- [6] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. InstructPix2Pix: Learning to Follow Image Editing Instructions. In CVPR.
- [7] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. InstructPix2Pix: Learning to Follow Image Editing Instructions. arXiv:2211.09800 [cs.CV] https: //arxiv.org/abs/2211.09800
- [8] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 22560–22570.
- [9] Yingying Deng, Xiangyu He, Fan Tang, and Weiming Dong. 2023. 𝑍∗: Zeroshot Style Transfer via Attention Rearrangement. arXiv:2311.16491 [cs.CV] https://arxiv.org/abs/2311.16491
- [10] Gilad Deutch, Rinon Gal, Daniel Garibi, Or Patashnik, and Daniel Cohen-Or.

2024. TurboEdit: Text-Based Image Editing Using Few-Step Diffusion Models. arXiv:2408.00735 [cs.CV] https://arxiv.org/abs/2408.00735

- [11] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. 2024. ReNoise: Real Image Inversion Through Iterative Noising. arXiv:2403.14602 [cs.CV]
- [12] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. 2023. TokenFlow: Consistent Diffusion Features for Consistent Video Editing. arXiv preprint arxiv:2307.10373 (2023).
- [13] Qiyuan He, Jinghao Wang, Ziwei Liu, and Angela Yao. 2024. AID: Attention Interpolation of Text-to-Image Diffusion. arXiv preprint arXiv:2403.17924 (2024).
- [14] Eric Hedlin, Gopal Sharma, Shweta Mahajan, Hossam Isack, Abhishek Kar, Helge Rhodin, Andrea Tagliasacchi, and Kwang Moo Yi. 2024. Unsupervised Keypoints from Pretrained Diffusion Models.
- [15] Eric Hedlin, Gopal Sharma, Shweta Mahajan, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 2023. Unsupervised Semantic Correspondence Using Stable Diffusion.
- [16] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. 2023. Delta Denoising Score.

(2023).

- [17] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control.

(2022).

- [18] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. 2023. Style Aligned Image Generation via Shared Attention. (2023).
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. In NeurIPS.
- [20] Jonathan Ho and Tim Salimans. 2022. Classifier-Free Diffusion Guidance. arXiv:2207.12598 [cs.LG] https://arxiv.org/abs/2207.12598
- [21] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. 2024. An Edit Friendly DDPM Noise Space: Inversion and Manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- [22] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. 2023. Imagic: Text-Based Real Image Editing with Diffusion Models. In Conference on Computer Vision and Pattern Recognition 2023.
- [23] Aliasghar Khani, Saeid Asgari Taghanaki, Aditya Sanghi, Ali Mahdavi Amiri, and Ghassan Hamarneh. 2023. SLiMe: Segment Like Me. arXiv:2309.03179 [cs.CV]
- [24] Gwanhyeong Koo, Sunjae Yoon, Ji Woo Hong, and Chang D Yoo. 2024. FlexiEdit: Frequency-Aware Latent Refinement for Enhanced Non-Rigid Editing. arXiv preprint arXiv:2407.17850 (2024).
- [25] H. W. Kuhn. 1955. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly 2, 1-2 (1955), 83–97. https://doi.org/10.1002/nav.3800020109 arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/nav.3800020109
- [26] Ruibin Li, Ruihuang Li, Song Guo, and Lei Zhang. 2024. Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models. (2024).
- [27] Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, and Jian Yang. 2023. StyleDiffusion: Prompt-Embedding Inversion

- for Text-Based Editing. arXiv preprint arXiv:2303.15649 (2023).
- [28] Kuan Heng Lin, Sicheng Mo, Ben Klingher, Fangzhou Mu, and Bolei Zhou. 2024. Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance. In Advances in Neural Information Processing Systems.
- [29] Grace Luo, Trevor Darrell, Oliver Wang, Dan B Goldman, and Aleksander Holynski. 2024. Readout Guidance: Learning Control from Diffusion Features. CVPR.
- [30] Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, and Trevor Darrell.

2023. Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence. In Advances in Neural Information Processing Systems.

- [31] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. 2023. Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference. arXiv:2310.04378 [cs.CV]
- [32] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2022. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In International Conference on Learning Representations.
- [33] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, and Igor Gilitschenski. 2024. Watch Your Steps: Local Image and Scene Editing by Text Instructions. In ECCV.
- [34] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or.

2023. Null-text Inversion for Editing Real Images using Guided Diffusion Models.

(2023).

- [35] Hyelin Nam, Gihyun Kwon, Geon Yeong Park, and Jong Chul Ye. 2024. Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 9192–9201.
- [36] Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, and Stephen Huang. 2023. Effective real image editing with accelerated iterative diffusion inversion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 15912–15921.
- [37] Or Patashnik, Rinon Gal, Daniel Cohen-Or, Jun-Yan Zhu, and Fernando De La Torre. 2024. Consolidating Attention Features for Multi-view Image Editing. In SIGGRAPH Asia 2024 Conference Papers (SA ’24). Association for Computing Machinery, New York, NY, USA, Article 40, 12 pages. https://doi.org/10.1145/ 3680528.3687611
- [38] Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, and Daniel CohenOr. 2023. Localizing Object-level Shape Variations with Text-to-Image Diffusion Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
- [39] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023. FateZero: Fusing Attentions for Zero-shot Textbased Video Editing. arXiv:2303.09535 (2023).
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020 [cs.CV] https: //arxiv.org/abs/2103.00020
- [41] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical textconditional image generation with clip latents. arXiv:2204.06125 [cs.CV]
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2021. High-Resolution Image Synthesis with Latent Diffusion Models. arXiv preprint arXiv:2112.10752 (2021). arXiv:2112.10752
- [43] Mehdi Safaee, Aryan Mikaeili, Or Patashnik, Daniel Cohen-Or, and Ali MahdaviAmiri. 2024. CLiC: Concept Learning in Context. CVPR (2024).
- [44] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. https://doi.org/10.48550/ARXIV.2205.11487
- [45] Tim Salimans and Jonathan Ho. 2022. Progressive Distillation for Fast Sampling of Diffusion Models. In International Conference on Learning Representations. https://openreview.net/forum?id=TIdIXIpzhoI
- [46] Dvir Samuel, Rami Ben-Ari, Nir Darshan, Haggai Maron, and Gal Chechik. 2024. Norm-guided latent space exploration for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’23). Curran Associates Inc., Red Hook, NY, USA, Article 2522, 13 pages.
- [47] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. 2024. Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation. arXiv:2403.12015 [cs.CV] https://arxiv.org/ abs/2403.12015
- [48] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli.

2015. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 37), Francis Bach and David Blei (Eds.). PMLR, Lille, France, 2256–2265. https://proceedings.mlr.press/v37/sohl-dickstein15.html

- [49] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2022. Denoising Diffusion Implicit Models. arXiv:2010.02502 [cs.LG] https://arxiv.org/abs/2010.02502
- [50] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency models. In Proceedings of the 40th International Conference on Machine Learning. 32211–32252.
- [51] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. 2023. Emergent Correspondence from Image Diffusion. In Thirtyseventh Conference on Neural Information Processing Systems. https://openreview. net/forum?id=ypOiXjdfnU
- [52] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. 2024. Training-Free Consistent Text-to-Image Generation. arXiv:2402.03286 [cs.CV] https://arxiv.org/abs/2402.03286
- [53] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. 2022. Plugand-Play Diffusion Features for Text-Driven Image-to-Image Translation. arXiv:2211.12572 [cs.CV] https://arxiv.org/abs/2211.12572
- [54] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. 2023. Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 1921–1930.
- [55] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. 2022. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/diffusers.
- [56] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13, 4 (2004), 600–612. https://doi.org/10.1109/TIP.2003.819861
- [57] Chen Henry Wu and Fernando De la Torre. 2023. A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance. In ICCV.
- [58] Zongze Wu, Nicholas Kolkin, Jonathan Brandt, Richard Zhang, and Eli Shechtman.

2024. TurboEdit: Instant text-based image editing. ECCV (2024).

- [59] Jie Xiao, Kai Zhu, Han Zhang, Zhiheng Liu, Yujun Shen, Yu Liu, Xueyang Fu, and Zheng-Jun Zha. 2023. CCM: Adding Conditional Controls to Text-to-Image Consistency Models. arXiv preprint arXiv:2312.06971 (2023).
- [60] Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. 2024. InversionFree Image Editing with Natural Language. (2024).
- [61] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. 2024. One-step Diffusion with Distribution Matching Distillation. CVPR (2024).
- [62] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang.

2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. arXiv:1801.03924 [cs.CV] https://arxiv.org/abs/1801.03924

[Figure 11]

## Fig. 9. Qualitative comparison. We compare the visual quality of our edits with several few-step and multi-step frameworks. our method strikes the perfect balance between respecting the editing text-prompt and preserving the content of the input image.

[Figure 12]

## Fig. 10. Additional Qualitative Comparison. We compare our method against several editing frameworks, including Null-Text Inversion [34], InstructPix2Pix [6], Plug-and-Play [53], and StyleDiffusion [27].

- 7 IMPLEMENTATION DETAILS

We use SDXL-Turbo [47] from the Diffusers library [55] as our fewstep diffusion model. Both the inversion of the source image and the generation of the target image are performed using four denoising steps. Following FlexiEdit [24], we observed that high-frequency components in the latent space during the initial timesteps of denoising can hinder flexible editing by imposing rigid constraints, limiting significant deviations from the original image’s structure. To facilitate larger and more flexible changes, we apply high-frequency suppression to the latent representation at the first timestep. Furthermore, while our method works without masking, incorporating a mask to specify regions of the image that should remain unchanged can enhance background preservation during editing. Following [3], we use masked latent blending to preserve these regions by blending them from the source image.

We applied our latent correction in timesteps 3 and 4 of denoising in which the patch sizes are 5 × 5 and 3 × 3 respectively. Moreover, in our content-adaptive interpolation, we found that setting 𝑘 = 3 or 𝑘 = 4 typically produces better results.

- 8 QUANTITATIVE RESULTS

We perform a quantitative comparison of Cora with several fewstep and multi-step editing methods. for content preservation, we measure PSNR, LPIPS [62], mean squared error (MSE), and structural similarity [56] losses between the backgrounds of the source and target images. For text alignment we measure CLIP similarity [40] between the target image with and without subject masking and the editing text-prompt. The results are presented in Tab. 3.

- 9 ADDITIONAL EXPERIMENTS

- 9.1 Ablation on Structure alignment

- In Fig. 12, we show five examples to illustrate how our structural alignment step preserves the original pose and layout. Each row compares results generated without alignment (left) to those generated with alignment (right). Without alignment, the target images tend to deviate significantly from the source structure. In contrast, applying our one-to-one query matching ensures that the coarse spatial arrangement of the source image is retained, resulting in edited images that faithfully reproduce the same pose while incorporating the desired changes.

9.2 Ablation on latent correction

- In Fig. 13, we compare our editing results with and without applying latent correction. When the correction is skipped, we frequently observe misalignment artifacts and unnatural deformations, especially for edits involving large shape changes. In contrast, incorporating our proposed patchwise correspondence alignment effectively mitigates these artifacts, leading to higher-quality edits that better preserve both the overall structure and fine details of the source image. This highlights the importance of the latent correction mechanism in achieving coherent and visually appealing results.

9.3 Ablation on different attention mixing strategies

In Fig. 15, we provide visual comparison between different attention mixing strategies. These include mutual self-attention [8], Concatenating source and target image attentions [18, 52], and our linear and spherical interpolation with and without DIFT alignment. As evident, the aligned spherical interpolation yields the most realistic and natural results.

- 10 GENERALIZATION THROUGH CONTROLLABLE PARAMETERS

When edits rely solely on text instructions, users often find it hard to specify how much of the source’s appearance or structure should remain intact. We address this by introducing two user-defined parameters, 𝛼 and 𝛽, and, because our approach only uses a few denoising steps, users can quickly adjust𝛼 and 𝛽 to reach the desired outcome with minimal time overhead. As shown in Figure 16 to Figure 21, tuning 𝛼 determines how much of the original look is preserved versus newly generated, while 𝛽 refines how strictly the layout follows the source. These parameters provide clear, finegrained control, reducing guesswork and enabling a broad range of edits, from slight refinements to large-scale transformations, while preserving stable and predictable outcomes.

- 11 LATENT CORRECTION VIA PATCH-BASED CORRESPONDENCE

- 11.1 Why the Original Corrections Fail Noise–inversion of the input image returns a sequence of latent

corrections {𝑧𝑡}𝑇𝑡=1 such that, if we inject the same 𝑧𝑡’s during the backward pass and keep the conditioning text unchanged, the model

reconstructs the source image pixel-perfectly. Each 𝑧𝑡 therefore assumes that every object will still occupy the exact spatial location it had in the source.

Editing, however, breaks this assumption: for instance, if a dog is asked to jump, the pixels affected by the edit are now displaced, and no longer correspond to the locations for which the original 𝑧𝑡 corrections were computed. Reusing those unmodified corrections forces the model to inject noise that is no longer spatially aligned, leading to mismatched textures, silhouette glitches, and the reappearance of content that no longer belongs.

- 11.2 Solution: Correspondence-aware Latent Corrections

Fig. 11 contrasts the standard TurboEdit pipeline (middle) with our correspondence-aware variant (bottom). After inversion (top) we still obtain the original noise residual 𝑧𝑡−1, but before re-injecting it we realign it so that it follows the geometry of the edited frame 𝑥𝑡−1.

Our patch-based correspondence approach divides DIFT feature maps into overlapping patches, matches each target patch to its most similar source patch, and then reconstructs the aligned target representation by reassembling the matched patches back into their original spatial locations and averaging overlapping regions.

Table 3. Quantitative comparisons among text-based editing baselines. Bold indicates the best scoring method, underline indicates the second best, blue indicates the third best.

PSNR (Background) ↑

LPIPS (Background) ↓

MSE (Background) ↓

SSIM (Background) ↑

CLIP Sim. (Whole) ↑

CLIP Sim. (Edited) ↑

Prompt2prompt [17] 19.9 153.98 188.94 79.58 24.87 22.35 Plug-and-play [53] 24.88 80.86 72.53 86.25 25.13 22.08 NTI [34] 29.92 36.68 25.18 91.02 24.18 21.09 StyleDiffusion [27] 28.65 45.16 34.53 89.6 24.02 21.28 InstructPix2Pix [6] 22.47 145.50 241.69 80.38 21.54 19.80 DDPM inversion [21] 24.09 110.68 220.50 84.65 23.35 20.40 MasaCtrl [8] 24.47 82.81 78.44 87.01 23.99 21.00 TurboEdit [10] 27.95 44.98 37.87 89.82 24.79 22.44 InfEdit [60] 25.51 123.76 370.24 80.21 23.34 20.08 Cora (Ours) 28.02 50.57 38.03 89.39 24.91 22.69

- 11.3 Patch Extraction and Matching

Let 𝐷𝑆 ∈ R𝐶×𝐻×𝑊 and 𝐷𝑇 ∈ R𝐶×𝐻×𝑊 be the feature maps of the source and target images, respectively. We extract overlapping patches of size 𝑘 × 𝑘 from both maps with stride 𝑠. This turns each map into a set of flattened patch vectors:

patches𝑆 = {𝑝𝑆,1, . . .,𝑝𝑆,𝑁𝑆 }, 𝑝𝑆,𝑖 ∈ R𝐶𝑘2, patches𝑇 = {𝑝𝑇,1, . . .,𝑝𝑇,𝑁𝑇 }, 𝑝𝑇,𝑗 ∈ R𝐶𝑘2.

(22)

Here, 𝑁𝑆 and 𝑁𝑇 denote the total numbers of patches extracted from the source and target, respectively.

We then compute cosine similarity between every pair (𝑝𝑇,𝑗,𝑝𝑆,𝑖): sim 𝑝𝑇,𝑗,𝑝𝑆,𝑖 =

𝑝𝑇,𝑗 · 𝑝𝑆,𝑖 ∥𝑝𝑇,𝑗 ∥ ∥𝑝𝑆,𝑖∥

. (23)

For each target patch 𝑝𝑇,𝑗, we select the index of the best matching source patch:

sim 𝑝𝑇,𝑗, 𝑝𝑆,𝑖 . (24)

𝐶𝑇→𝑆 (𝑗) = arg max

1≤𝑖≤𝑁𝑆

- 11.4 Reassembling Matched Patches After finding the best-match index for each target patch, we re-

place each target patch 𝑝𝑇,𝑗 with its matched source patch 𝑝𝑆,I(𝑗), yielding an updated patch set:

𝑝ˆ𝑇,𝑗 = 𝑝𝑆, 𝐶𝑇→𝑆 (𝑗). (25) We then reassemble these patches into a tensor of shape (𝐶,𝐻,𝑊 ) by placing each 𝑝ˆ𝑇,𝑗 at the corresponding spatial location of the 𝑗-th patch in the target. Since patches overlap, every pixel can receive contributions from multiple patches. Specifically, we denote by 𝐷˜𝑇 (u) the sum of all matched-patch contributions at spatial location u. Let Ω𝑗 be the set of pixel coordinates covered by the 𝑗-th patch. Then:

# 𝐷˜𝑇 (u) = ∑︁

𝑝ˆ𝑇,𝑗 (u), (26)

𝑗: u∈Ω𝑗

# 𝑊 (u) = ∑︁

1, (27)

𝑗: u∈Ω𝑗

where 𝑝ˆ𝑇,𝑗 (u) is the feature value of patch 𝑝ˆ𝑇,𝑗 for pixel u. To normalize overlaps, we compute the final aligned map 𝐷ˆ𝑇 by dividing each spatial location by its total overlap:

𝐷˜𝑇 (u) 𝑊 (u) + 𝜀

𝐷ˆ𝑇 (u) =

, (28) with 𝜀 a small constant to prevent division by zero.

12 CONTENT-ADAPTIVE INTERPOLATION

When a prompt leads to large deformations or introduces new objects, not every pixel in the edited (target) image should be forced to match a pixel in the original (source) image. Over-enforcing alignment often creates visual artifacts or incorrect texture transfers. To address this, we propose a two-step strategy that checks whether each target patch has a reliable counterpart in the source before blending.

- 12.1 1. Bidirectional Matching

Let 𝑆 be the set of source patches and 𝑇 be the set of target patches. For each source patch 𝑠 ∈ 𝑆, we define its top-𝑘 most similar target patches as:

K(𝑠) = argtopk

𝑡∈𝑇

{sim(𝑠,𝑡)}, (29)

where sim(·, ·) is the cosine similarity between feature vectors. Similarly, for each 𝑡 ∈ 𝑇, define K(𝑡). A pair (𝑠,𝑡) is said to be bidirectionally matched if 𝑠 ∈ K(𝑡) and 𝑡 ∈ K(𝑠). These are considered strong correspondences, and we blend source and target information using a user-defined weight 𝛼.

- 12.2 2. Weak Matches → Treat as New

Some target patches remain unmatched. For each such target patch 𝑡 ∈ 𝑇, we compute the strength of its best match in the source as:

sim(𝑠,𝑡). (30) Let 𝑈 ⊂ 𝑇 be the set of unmatched patches. We compute the

simmax(𝑡) = max

𝑠∈𝑆

𝛾-quantile threshold 𝜏𝛾 over simmax(𝑡) values:

[Figure 13]

Fig. 11. Correspondence-aware latent correction. Top: inversion turns the source image into noise 𝑥𝑇 and extracts residuals {𝑧𝑡 } that exactly reconstruct the original pose. Middle: TurboEdit re-injects the unaligned residual (correction term) 𝑧𝑡−1 while editing the pose, causing textures to snap back to old positions. Bottom: our method aligns 𝑧𝑡−1 via DIFT-based patch correspondences, producing a geometry-aware correction 𝑧𝑡aln−1 and a clean, artifact-free result.

𝜏𝛾 = quantile𝛾 ({simmax(𝑡) | 𝑡 ∈ 𝑈 }) . (31) If simmax(𝑡) < 𝜏𝛾, we classify 𝑡 as new and set 𝛼 = 1, meaning it

is fully guided by the prompt and not the source image.

- 12.3 Practical Insights

This content-adaptive interpolation balances preservation and generation: we reuse source appearance where reliable correspondences exist, and rely on prompt-driven generation in new regions. This avoids artifacts from over-aligning unrelated content. While the

matching process is not perfectly accurate, we found it to be sufficient in most examples. In practice, it significantly reduces artifacts caused by mistakenly interpolating between unmatched patches.

Interestingly, even when only a portion of a new region, such as 20% of the pixels corresponding to a newly generated object like a hat, is correctly identified as "new," the outcome is often satisfactory. Since we avoid interpolation for those pixels, they remain purely prompt-driven. During denoising, the rest of the image tends to adapt around these pixels, effectively completing the structure and appearance of the new content. Nonetheless, improving the accuracy of identifying new regions would further enhance the quality and reliability of the generated results.

[Figure 14]

## Fig. 12. Ablation on structure alignment. By applying our structure alignment, we can preserve the structural layout of the source image.

[Figure 15]

## Fig. 13. Ablation on latent correction. Without latent correction, multiple misalignment artifacts and unnatural deformations occur. Applying correction produces cleaner and more realistic results.

[Figure 16]

## Fig. 14. Ablation on latent correction. Without latent correction, multiple misalignment artifacts and unnatural deformations occur. Applying correction produces cleaner and more realistic results.

[Figure 17]

## Fig. 15. Ablation on attention mixing strategies. With these visual results, we demonstrate that DIFT-aligned SLERP yields the best results.

[Figure 18]

## Fig. 16. Additional results showcasing our correspondence-aware attention interpolation and structural alignment. Adjusting 𝛼 smoothly shifts the appearance from the source to the target, while varying 𝛽 progressively alters structural elements. The grid shows how appearance and structure can be controlled independently to achieve diverse transformations.

[Figure 19]

## Fig. 17. Additional results showcasing our correspondence-aware attention interpolation and structural alignment. Adjusting 𝛼 smoothly shifts the appearance

[Figure 20]

## Fig. 18. Additional results showcasing our correspondence-aware attention interpolation and structural alignment. Adjusting 𝛼 smoothly shifts the appearance from the source to the target, while varying 𝛽 progressively alters structural elements. The grid shows how appearance and structure can be controlled independently to achieve diverse transformations.

[Figure 21]

## Fig. 19. Additional results showcasing our correspondence-aware attention interpolation and structural alignment. Adjusting 𝛼 smoothly shifts the appearance

[Figure 22]

## Fig. 20. Additional results showcasing our correspondence-aware attention interpolation and structural alignment. Adjusting 𝛼 smoothly shifts the appearance from the source to the target, while varying 𝛽 progressively alters structural elements. The grid shows how appearance and structure can be controlled independently to achieve diverse transformations.

[Figure 23]

## Fig. 21. Additional results showcasing our correspondence-aware attention interpolation and structural alignment. Adjusting 𝛼 smoothly shifts the appearance

