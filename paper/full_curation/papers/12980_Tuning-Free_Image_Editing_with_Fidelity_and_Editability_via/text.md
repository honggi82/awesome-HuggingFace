## Tuning-Free Image Editing with Fidelity and Editability via Unified Latent Diffusion Model

Qi Mao, Lan Chen, Yuchao Gu, Mike Zheng Shou, Ming-Hsuan Yang

### arXiv:2504.05594v1[cs.CV]8Apr2025

Abstract—Balancing fidelity and editability is essential in text-based image editing (TIE), where failures commonly lead to over- or under-editing issues. Existing methods typically rely on attention injections for structure preservation and leverage the inherent text alignment capabilities of pre-trained text-to-image (T2I) models for editability, but they lack explicit and unified mechanisms to properly balance these two objectives. In this work, we introduce UnifyEdit, a tuning-free method that performs diffusion latent optimization to enable a balanced integration of fidelity and editability within a unified framework. Unlike direct attention injections, we develop two attention-based constraints: a self-attention (SA) preservation constraint for structural fidelity, and a cross-attention (CA) alignment constraint to enhance text alignment for improved editability. However, simultaneously applying both constraints can lead to gradient conflicts, where the dominance of one constraint results in over- or under-editing. To address this challenge, we introduce an adaptive time-step scheduler that dynamically adjusts the influence of these constraints, guiding the diffusion latent toward an optimal balance. Extensive quantitative and qualitative experiments validate the effectiveness of our approach, demonstrating its superiority in achieving a robust balance between structure preservation and text alignment across various editing tasks, outperforming other state-of-the-art methods. The source code will be available at https://github.com/CUC-MIPG/UnifyEdit.

Index Terms—Text-based image editing, diffusion model, latent optimization, attention-based constraint, tuning-free.

✦

1 INTRODUCTION

# N

ATURAL language is one of the most intuitive and effective ways for people to express their thoughts. Recent advance-

ments in large-scale text-to-image (T2I) diffusion models [1]– [3] have successfully bridged the gap between textual and visual modalities, facilitating the generation of high-quality images from free-form text prompts. However, in addition to creating images from scratch, there is an increasing need to modify existing images based on textual descriptions. This has led to the emergence of text-based image editing (TIE) [4]–[18], which aims to manipulate input images according to given text prompts while preserving the integrity of other content. Over the past years, diffusion models for TIE have been extensively developed, categorized into: instruction-based training [4], [5], fine-tuning [6], [7], and tuningfree [8]–[14] methods. This work focuses on tuning-free editing approaches, which adapt existing T2I models for manipulations without extensive retraining or fine-tuning.

Two critical concepts in TIE, distinct from T2I generation, are “fidelity” and “editability”. Fidelity concerns preserving the original image’s content in areas that are not intended to be changed. Editability refers to the effectiveness of an editing method in making the desired changes specified by the text prompt. In the realm of diffusion models for TIE, a dual-branch editing paradigm such as P2P [8] is commonly adopted, as demonstrated in Fig. 2(a). This approach involves a source branch that reconstructs the original image based on the source prompt and a target branch

Qi Mao and Lan Chen are with the State Key Laboratory of Media Convergence and Communication, Communication University of China. (E-mail: qimao@cuc.edu.cn, chenlaneva@mails.cuc.edu.cn). Yuchao Gu and Mike Zheng Shou are with Show Lab, National University of Singapore. (E-mail: yuchaogu@u.nus.edu, mikeshou@nus.edu.sg). Ming-Hsuan Yang is with the University of California at Merced and Yonsei University. (E-mail: mhyang@ucmerced.edu). (Corresponding Author: Qi Mao)

that generates the target image guided by the target prompt. Within this framework, fidelity is achieved through shared inverted noise latents [17], [19], [20] and structural information provided by the source branch [8], [9], [12], [16], [21]. Meanwhile, editability originates from the inherent ability of T2I models to align target text descriptions with visual outputs in the target branch.

The fundamental challenge in achieving this balance stems from the varying trade-offs required by different types of edits. For instance, color edits (e.g., Fig. 1(a)) demand a high degree of structural consistency to maintain the integrity of the original image. In contrast, object replacements (e.g., Fig. 1(c)) allow for greater editability, requiring only that the pose of the original elements be preserved. As such, a poor balance can lead to two undesirable issues in Fig. 1:

- • Over-editing (editability > fidelity): This occurs when the editing method makes excessive changes, prioritizing the text prompt over the original image’s content. For example, in the second row of Fig. 1(c), although the tiger aligns visually with the target prompt, its posture is heavily altered compared to the original.
- • Under-editing (editability < fidelity): This situation arises when the method fails to sufficiently apply the desired changes in the edited regions, maintaining too much of the original image. As a result, the edited image does not accurately reflect the changes specified in the text prompt. For instance, in the last row of Fig. 1(b), while the structure of the coat is well-preserved, its appearance does not align with the modifications required by the target prompt.

The existing dual-branch editing paradigm, which mainly utilizes attention injections [8], [9], [12], [21] for structure preservation, lacks an explicit method to balance both fidelity and editability. However, adjustments can only be achieved through hyperparameters such as attention injection timesteps. To address these limitations, we introduce UnifyEdit to explicitly bal-

[Figure 1]

- Fig. 1: Illustration of balancing fidelity and editability. We demonstrate examples of over-, balanced, and under-editing across six types of edits: (a) color change, (b) texture modification (c) object replacement (d) background editing, (e) global style transfer, and (f) human face attribute editing. Over-editing occurs when excessive changes distort the original image, while under-editing results in changes too subtle to meet the text prompt’s requirements. In contrast, our UnifyEdit balances fidelity and editability within a unified framework, ensuring edits align with the text prompt while preserving the essential integrity.

ance fidelity and editability through a unified diffusion latent optimization framework, enabling adaptive adjustments to meet the specific requirements of various editing types. Specifically, UnifyEdit differs from direct attention injections by employing two attention-based constraints derived from the pre-trained T2I models: the self-attention (SA) preservation constraint, which ensures structural fidelity by measuring discrepancies between SA maps of the source and target branches, and the cross-attention (CA) alignment constraint, which boosts editability by promoting higher CA values in areas corresponding to the target edited token.

We note that directly combining these two constraints to guide diffusion latent optimization can produce conflicting gradients, causing one constraint to dominate and skew the guidance direction. This imbalance may lead to either over- or under-editing failures. To address this issue, we propose an adaptive time-step scheduler that dynamically adjusts the weighting parameters of each constraint according to the denoising timestep. At the initial denoising stage, when the target diffusion denoising trajectory is close to the source’s, emphasis is placed on the CA alignment constraint to enhance editability. As the denoising process progresses and the target diffusion latent increasingly aligns well with the new prompt, the importance of the SA preservation constraint is heightened to ensure structural fidelity. Interestingly, we also find that visualizing the gradients of the proposed constraints can pinpoint the causes of over- or under-editing, enabling users to tailor the fidelity-editability trade-off to their preferences.

The main contributions are summarized as follows:

• We introduce UnifyEdit, a novel tuning-free framework that takes the first step toward achieving a balance between

fidelity and editability within a unified diffusion latent optimization framework.

- • We propose an adaptive time-step scheduler that balances two attention-based constraints, one focused on maintaining structural fidelity and the other on enhancing editability. This approach effectively optimizes the diffusion latent toward a balanced direction, accommodating various types of edits.
- • To validate the efficiency of our proposed method in balancing various editing types, we develop a dataset named UnifyBench, which includes a wide range of edits across different scopes of editing regions, such as foreground modifications (changes in color, texture, or material, and object replacements), background editing, global style transfer, and human face attribute editing. Both quantitative and qualitative experimental results demonstrate that our method significantly improves the trade-off between structure fidelity and editing efficiency compared to existing state-of-the-art approaches.

2 RELATED WORK

2.1 Text-based Image Editing Using Diffusion Models

Text-based image editing (TIE) involves modifying input images based on specific text prompts while preserving the integrity of the original content. With the emergence of diffusion models [19], [22], numerous approaches have been developed to leverage their capabilities for this task. Existing TIE methods based on diffusion modes can be broadly divided into three categories: training [4], [5], [23], [24], fine-tuning [6], [7], [25], and tuning-free methods [8]–[13], [16]–[18], [26].

###### (a) Existing Methods : Source Branch

###### (b) Our Method :

Source Branch

- • Fidelity: SA Preservation Constraint
- • Editability: CA Alignment Constraint

- • Fidelity: Attention Injection
- • Editability: Text

𝐼መ 𝐼መ

𝑃

𝑃

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Pre-Trained Diffusion Model

Pre-Trained Diffusion Model

𝑧𝑡

[Figure 6]

[Figure 7]

𝑧𝑡

graffitied

Constrain Attention

Inversion

Inject Attention

Inversion

Latent Optimization

+

[Figure 8]

[Figure 9]

𝑧Ƹ𝑡∗

|[Figure 10]<br><br>[Figure 11]|
|---|

[Figure 12]

𝑧𝑡

Pre-Trained Diffusion Model

Pre-Trained Diffusion Model

𝐼

𝑃: a black car

𝑧𝑡∗

𝑧𝑡∗

𝑃∗ 𝐼∗

𝑃∗

𝐼∗

𝑃∗: a graffitied car

Target Branch

Target Branch

- Fig. 2: UnifyEdit vs. dual-branch editing paradigm. (a) The typical dual-branch editing paradigm consists of source and target branches, using attention injection to maintain fidelity while relying on the text prompt to achieve editability. (b) In contrast, our method explicitly models the fidelity and editability using two attention-based constraints and performs latent optimization within a unified framework, facilitating an adaptive balance across various editing types.

Training-based methods focus on training a model specifically for a given task using a substantial amount of data to transform a source image into a target image. Early works [23], [24] such as CLIPDiffusion [23] mainly target domain-specific image transformations, for instance, transforming a “dog” into a “bear”. In particular, it leverages CLIP loss to train the diffusion model that aligns the generated image with the target text. However, these methods are constrained by their reliance on short phrases to define domains, which limits their ability to fully utilize the flexibility offered by free-form text. To address this limitation and reduce the need for complex descriptions, recent methods [4], [5], [25], [27], [28] such as InstructPix2Pix [4] introduce editing driven by natural language instructions such as “add a flower”. InstructPix2Pix [4] employs a fully supervised approach, utilizing training datasets of paired source and edited images with corresponding instructions. It enhances the backbone T2I model with an additional input channel for incorporating image conditions during denoising. This allows it to produce images that adhere to the instructions while maintaining the original’s integrity. Furthermore, it adapts Classifier-Free Guidance (CFG) [29] to balance alignment with the input image and edit instructions.

merge the noisy image in the edited region, which is guided by text prompts, with the area outside of the mask using the noisy source image. Recently, the dual-branch P2P [8] model extracts self-attention and cross-attention maps from the source image and injects them into the target branch for editing. In this work, we focus on tuning-free methods, eliminating the need for extensive retraining and thus saving time and computational resources.

###### 2.2 Tuning-Free Text-based Image Editing

In TIE, achieving a balance between fidelity and editability is important to generate high-quality results. Fidelity involves preserving the original content that should not be changed, while editability ensures the desired changes that satisfy the text prompt. As categorized in Section 2.1, recent tuning-free TIE methods fall into two representative categories: inpaintingbased and dual-branch-based methods. These approaches utilize distinct mechanisms to balance fidelity and editability. Inpaintingbased methods [13], [15], [30]–[32] prioritize preserving fidelity in non-edited regions by introducing advanced mask-grounding and mask-blending techniques. They aim to accurately identify the target region and seamlessly integrate the generated foreground object with the background latent of the source image, ensuring a natural and cohesive result. In particular, Blended Latent Diffusion [15] directly generates a foreground object based on text prompts and introduces an improved blending operation to seamlessly integrate the object with the background latent of the source image. DiffEdit [13], [32] proposes an unsupervised maskpredicting method and utilizes DDIM inversion [19] to integrate latent features alongside the target prompt, thereby generating the foreground image. However, these methods often result in significant structural alterations in the target foreground objects due to the inadequate structural information provided by the source image.

To reduce the computational cost associated with training a full diffusion model, fine-tuning methods [6], [7], [25] focus on refining specific layers or embeddings rather than the entire denoising network. UniTune [25] fine-tunes the diffusion model on a single base image during the tuning phase, ensuring that the generated images closely resemble the base image. Imagic [6] optimizes the text embedding and fine-tunes the T2I model by minimizing the discrepancy between the reconstructed and original images. The final edited image is then generated by linearly interpolating the optimized text embedding with the target text using the fine-tuned diffusion model.

Unlike training or fine-tuning diffusion models, numerous tuning-free methods that directly adapt existing pre-trained T2I models for image manipulations have recently gained substantial attention. The core idea behind these methods is to use the diffusion denoising process of the T2I model to preserve the fidelity of parts of the original image while simultaneously leveraging the inherent editability of the original text-visual alignment. These approaches can be broadly categorized into two representative methods: inpainting-based methods [13], [15], [30]–[32] and dual-branch-based methods [8], [9], [12], [14], [18], [33], [34]. Inpainting-based methods, such as DiffEdit [13], utilize a mask to

To maintain the overall fidelity of edited images, dual-branchbased methods [8], [9], [12] such as P2P [8] leverage selfattention and cross-attention attention injection from the source branch to guide the target branch. Attention-injection-based methods [8], [9], [12], [16] emphasize extracting and injecting highly expressive features into the target branch, thereby enhancing structure preservation of the edited images. Recent advancements in inversion-based methods [11], [17], [20] refine the inversion process to enable a more precise source branch of the original

image, thereby generating enhanced feature injection sets compared to DDIM inversion [19]. Motivated by inpainting-based methods, recent dual-branch approaches [14], [33], [34] further utilize masks to focus on preserving fidelity in the non-edited regions and effectively prevent unintended attribute leakage.

Despite their success, existing dual-branch-based methods mainly regulate balance by adjusting attention injection timestep hyperparameters. However, how to balance both fidelity and editability within a unified framework has been overlooked in the literature. In this work, we concentrate on explicitly balancing fidelity and editability within a unified diffusion latent optimization framework. The two works most closely related to ours are Guide-and-Rescale (G-R) [18] and MAG-Edit [14], which both employ gradient-based methods to formulate either fidelity or editability using attention-based constraints explicitly. G-R [18] proposes to model the structure preservation using the SA constraint and performs gradient optimization using noise guidance. However, it merely leverages the text’s inherent editability within the CFG without explicitly modeling. On the other hand, MAGEdit [14] proposes amplifying the CA values within the mask to locally enhance the text alignment, and improve editability through diffusion latent optimization. Nevertheless, it still relies on attention injection for structure preservation. In contrast, our UnifyEdit explicitly integrates two powerful constraints for both fidelity and editability to guide the diffusion latent in a unified manner adaptively.

###### 2.3 Diffusion Latent Optimization

Diffusion latent optimization iteratively refines latent variables during denoising by minimizing specified constraints to align the latent trajectory with the target distribution and guide outcomes toward desired results. This technique has been effectively used in training-free T2I generation [35]–[40], for improving semantic alignment and enabling training-free control. Specifically, Attendand-Excite [35] and Linguistic Binding [36] utilize latent optimization with cross-attention constraints to address attribute leakage and incorrect binding. Additionally, latent optimization facilitates training-free condition control, such as color and layout, during the image generation process. For instance, Rich-Text-toImage [38] employs an objective function that minimizes the discrepancy between the predicted initial latent and a predefined RGB triplet, thereby enabling precise control over the color of generated objects. Similarly, training-free layout generation methods [37], [39], [40] leverage latent optimization by formulating objectives based on cross-attention maps and bounding boxes, effectively positioning objects within designated regions.

While latent feature optimization has shown its effectiveness in T2I, its application in TIE has received comparatively less attention. In TIE, Pix2Pix-Zero [10] leverages latent optimization to minimize discrepancies between the CA maps of the source and target branches, effectively preserving the fidelity of edited images. Most recently, MAG-Edit [14] utilizes latent optimization to enhance the alignment between textual prompts and latent features, significantly improving editability. These advancements highlight the potential of diffusion latent optimization in TIE. Compared to diffusion latent optimization, many editing-based methods focus on utilizing noise guidance [18], [41]. From the perspective of score-based diffusion [42], both noise guidance and latent optimization construct an energy function g(zt,y) and compute the gradient ∇zt

g(zt,y). However, the key difference lies in

their application of the gradient: noise guidance uses it to update the predicted noise ϵθ, guiding the sampling of zt−1, whereas latent optimization directly adjusts zt itself and recomputes the ϵθ based on the new zt. Compared to noise guidance, under this framework, the sampling of zt can be viewed as a fixed-point problem, solving it iteratively to reach the optimal solution. In this work, we adopt the diffusion latent optimization technique and present additional comparisons with noise guidance in Section 5.7 using the same constraints.

##### 3 DUAL-BRANCH TUNING-FREE IMAGE EDITING

The goal of text-based image editing is to transform the source image I into a target image I∗ that aligns with the target prompt P∗ while preserving the content of I that is not intended to be changed. To achieve this goal, the dual-branch editing paradigm [8]–[12], [14], [17], [18] is widely adopted in the literature. As illustrated in Fig. 2(a), this paradigm includes two branches: the source branch, generated by the original prompt P, and the target branch, generated by the target prompt P∗. The set of new target tokens present in P∗ against P is defined as S∗ = {s∗1,s∗i ,...,s∗I}, for instance, “graffitied”. Both branches begin with the same initial noise latent feature zT and end with different outputs: a reconstructed image Iˆ in the source branch and an edited image I∗ in the target branch. Existing methods typically focus on improving the following three operations to enhance fidelity and editability.

Attention Injection. The layer of denoising U-Net in the T2I model, such as Stable Diffusion [2] contains an SA block and a CA block. The SA block captures long-range interactions between image features, and the CA block integrates visual features with the text prompt. Both can be uniformly expressed as:

QK⊤ √

Attention(Q,K,V ) = SoftMax(

)V, (1)

d

where Q is the projected from spatial features, and K,V of SA and CA are projected from either the text embedding or spatial features, respectively. To ensure the overall fidelity of edited images, P2P [8] and PnP [9] first propose attention injection, which involves copying the SA maps Aself and CA maps Across generated in the source branch to the target branch. Formally, the SA maps A∗self and the CA maps A∗cross in the target branch can be formulated as:

A∗self ← Aself, A∗cross ← Across.

(2)

With this formulation, numerous methods [9], [12], [16] have been proposed to identify the most semantically rich features for injection. However, directly copying these features imposes overly strict conditions, limiting these approaches to achieving balance solely by modulating the attention injection timesteps.

Advanced Inversion. To obtain the initial noise latent feature zT, the DDIM inversion scheme [19] is widely adopted, defined as:

z¯t+1 =

αt+1 αt

z¯t + √αt+1(

1 αt+1 − 1 −

1 αt − 1)ϵθ(¯zt, t, P), (3)

where t = 0, . . . , T-1. Using DDIM sampling, the reconstructed latent feature z0 is obtained through the following process:

zt−1 =

αt−1 αt

zt + √αt−1(

1 αt−1 − 1 −

1 αt − 1)ϵθ(zt, t, P), (4)

SA Injection SA Constraint Source Image SA Injection SA Constraint

Leakage Low

###### Algin

Source Image

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

###### Cat

Tabby cat Green Tree Red T-shirt

Ice flower Pink wall Black cat

Blue dress

SourceImageOutputCA Visualization

Dress

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

Horse

Colorful dress

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

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Glass vase

Monkey

(a) Experiments on SA constraint (b) Experiments on CA visualization

- Fig. 3: Experiments with self-attention and cross-attention. (a) Compared to SA injection, the SA constraint offers greater flexibility in editing. (b) When the CA map accurately focuses on the target region with a strong response, the resulting edits align effectively with the text prompt. However, attention leakage or low attention values can lead to misalignment or ineffective editing outcomes.

where t = T, . . . , 1 and zT = z¯T. However, the CFG [43] in T2I models amplifies accumulated errors from DDIM inversion, resulting in a significant discrepancy between z¯t and zt. This deviation shifts the denoising sampling trajectory of the source branch away from the inversion path and propagates to the target branch through attention injection, thereby adversely impacting fidelity. To address this issue, several approaches [11], [17], [20] either mathematically enhance the DDIM inversion [17], [20] or introduce additional supervisory controls [11], [44] during sampling for more accurate shared features and greater flexibility in achieving editability.

Editing Area Grounding and Blending. For localized editing, to preserve the fidelity outside the foreground region, the mask grounding and blending operation is typically employed [8], [30] to integrate the edited object with the background. This operation combines the latent noise feature zt∗ from the target branch within the grounded editing area M with zt from the source branch outside M, given by:

zt∗ = M ⊙ zt∗ + (1 − M) ⊙ zt (5)

To seamlessly integrate the content inside and outside the targeted foreground region, automatic mask-grounding methods and better blending operations have been significantly proposed [13], [15], [26], [31]–[34], [45], [46].

- 4 UNFIY-EDIT VIA LATENT OPTIMIZATION

For global editing, M is set to a matrix of all ones. This formulation enables precise adjustments within specified areas or across the entire image, depending on the requirements.

To formulate L that guides zt∗ from both fidelity and editability perspectives, we begin by rethinking the roles of SA and CA in TIE, conducting two experiments described in Section 4.1. Specifically, we propose two attention-based constraints to model fidelity and editability, detailed in Section 4.2. Finally, leveraging the adaptive time-step scheduler introduced in Section 4.3, our framework dynamically balances these constraints to meet specific requirements of various editing tasks.

###### 4.1 Rethinking Self- and Cross-Attention for TIE

For fidelity, previous works [9], [21] have demonstrated that SA maps play a more significant role in preserving the layout and structure of images than CA maps. Regarding editability, existing studies [14], [27] have demonstrated that the CA maps are crucial for aligning the editing effects with the text prompt. In this section, we conduct two experiments within the commonly used dualbranch editing paradigm, P2P [8], to better understand how SA and CA influence the editing process in TIE. Note that all experiments are conducted without CA injection.

Experiments on Self-Attention: We replace SA injection in P2P [8] by optimizing the diffusion latent feature zt∗ to minimize the L2 loss between SA maps from the source and target branches.

Optimizing zt∗ with the SA constraint effectively preserves the layout and structural fidelity of the original image while unleashing greater editing flexibility compared to direct SA injection. Both the SA injection and the SA constraint effectively preserve the structure and layout fidelity of the original image without requiring CA injection. However, using the SA constraint allows for greater flexibility in appearance editing. For example, it results in the texture of the original “zebra” fading completely, while the shirt successfully transitions to a “colorful” appearance, as illustrated in Fig. 3(a).

Although the dual-branch editing paradigm uses attention injection to preserve fidelity, it lacks a systematic method for balancing fidelity and editability. Existing schemes are primarily limited to tuning hyperparameters [8], [9], [12], [17], such as attention injection timesteps [8], [9], [17]. In this work, we explicitly balance fidelity and editability through a unified framework that allows for adaptable modifications tailored to the diverse needs of different editing scenarios. In contrast to attention injections of the dual-branch paradigm shown in Fig. 2(a), we propose UnifyEdit that optimizes the diffusion latent zt∗ in the target branch guided by two attention-based constraints to achieve a balance between fidelity and editability, as illustrated in Fig. 2(b). Furthermore, we introduce a mask M for localized editing to target and balance the edited region specifically. The mask-guided optimization is formalized as follows:

Experiments on Cross-Attention: We visualize the average of all CA maps corresponding to the target token (e.g., “Blue” in the first column of Fig. 3(b)) at a resolution of 16 × 16 for both successful and failed editing examples in P2P [8].

High-response CA values indicate strong alignment between text and image features, resulting in pronounced editing effects. As demonstrated in the first two columns of Fig. 3(b), when the CA map accurately focuses on the intended region, the editing output

zˆt∗ = zt∗ − M ⊙ ∇z∗

L, (6)

t

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

UnifyEdit

…

…

- Step 1: § Ⅳ.B Calculate Constraints
- Step 2: § Ⅳ. C

###### Source Prompt 𝑷:a black car

𝓛𝑪𝑨𝑨

𝓛𝑺𝑨𝑷

[Figure 60]

𝑧𝑇

𝑧𝑡

𝑧𝑡−1

𝛁𝒛𝒕

𝛁𝒛𝒕

𝐴𝑡𝑠𝑒𝑙𝑓

𝛁𝒛𝒕𝓛𝑺𝑨𝑷/||𝛁𝒛𝒕𝓛𝑺𝑨𝑷||𝟐

𝛁𝒛𝒕𝓛𝑪𝑨𝑨/||𝛁𝒛𝒕𝓛𝑪𝑨𝑨||𝟐

𝑧0

UnifyEdit

[Figure 61]

[Figure 62]

###### § Ⅳ.B

Frozen U-net

§ Ⅳ.B Self-Attention Preservation Constraint

§ Ⅳ.C

Inversion

𝝀𝟏∗ ×

× 𝝀𝟐∗

+

Cross-Attention

|UnifyEdit|
|---|

[Figure 63]

CA Map

Alignment Constraint

𝑮𝒃𝒍𝒄

#### −

SA Map

Adaptive Time-Step Scheduler

𝐼

Latent Optimization

[Figure 64]

[Figure 65]

Latent Optimization

Target Prompt 𝑷∗:

[Figure 66]

𝐴𝑡∗𝑠𝑒𝑙𝑓 𝐴𝑡∗𝑐𝑟𝑜𝑠𝑠

𝑧𝑡∗

𝑧Ƹ𝑡∗ = 𝑧𝑡∗ − 𝑴 ⊙ 𝑮𝒃𝒍𝒄

a graffitied car

[Figure 67]

[Figure 68]

[Figure 69]

…

…

𝜖𝑡∗(𝑧Ƹ𝑡∗,𝑡,𝑃∗)

Deterministic

DDIM Sampling

𝑧𝑡−1∗ 𝑧0∗

𝑧𝑡∗

𝑧𝑇∗

𝑧𝑡−1∗

- Fig. 4: Illustration of UnifyEdit. UnifyEdit is applied to the diffusion latent feature zt∗ in the target branch, involving two key steps:

1) calculating LSAP and LCAA for fidelity and editability, and 2) applying an adaptive time-step scheduler for latent optimization.

aligns effectively with the textual prompt. Conversely, misaligned or weak values of CA maps lead to editing leakage or underediting issues. For instance, in the “green tree” scenario, excessive attention leakage to the ground areas in the CA map of token “green” results in edits mistakenly targeting the ground rather than the tree itself. Additionally, the low CA responses for the “red” token in the T-shirt area result in minimal changes to the T-shirt’s color to red. Thus, CA maps inherently reflect the degree of text-visual alignment, and controlling them offers a promising approach to enhancing editability.

constraint to maximize the ratio of CA values for the targeted token within the predefined editing region M relative to those outside M. Consider the CA map (A∗tcross)i corresponding to the i-th editing token Si∗ (e.g., “graffitied” in Fig. 4(a)) within a mask region M. The constraint emphasizes increasing the proportion Rl of the Si∗’s CA values within M relative to those outside mask 1 − M:

1 |M| j∈M

(A∗tcross)li 1

, (9)

Rl =

(A∗cross

t )li

|1−M| j /∈M

###### 4.2 Deriving Attention-Based Constraints

where j denotes the spatial index of CA maps, l represents the CA maps at the l-th layer, and |∗| calculates the total number inside or outside of the mask. All CA maps are computed in the resolution of 16 × 16 of the U-Net, recognized for containing the most semantically rich information [35]. Furthermore, our experiments reveal that promoting the ratio Rl in each layer at this resolution can further encourage alignment. Consequently, the CA aligned constraint is defined as:

Based on the experimental results discussed above, we introduce two constraints leveraging SA and CA to explicitly model fidelity and editability, respectively, as shown in Fig. 4.

Self-Attention Preservation Constraint. As demonstrated in Section 4.1, using the constraint to reduce the discrepancy between the SA maps Aselft generated from the source branch and A∗tself from the target branch successfully preserves the structural fidelity and offers more flexibility than direct SA injection. The SA preservation constraint is defined as:

2

L

, (10)

LCAA = −

Rl

LSAP = (Aselft − A∗tself)2. (7)

l=1

where L = 5, representing the total number of CA layers at a resolution of 16 × 16.

Furthermore, editing small objects within complex scenarios requires more precise control to preserve high-frequency details. In such cases, a region-based SA preservation constraint proves more effective:

###### 4.3 Balancing via Adaptive Time-Step Scheduler

###### LR−SAP = (Mˆ ⊙ Aselft − Mˆ ⊙ A∗tself)2, (8)

After obtaining LSAP and LCAA in Section 4.2, the simplest formulation of L in Eq. (6) is:

where the mask Mˆ for the SA maps is defined as Mˆ = MM⊤, with M representing the flattened vector of M. Although fullresolution SA maps are generally used to construct this constraint, for tasks requiring significant shape variation (e.g., object replacement), we specifically employ SA maps at 16 × 16 and 8 × 8 resolutions. Notably, our method is compatible with various inversion techniques. Furthermore, the SA maps can be directly derived from DDIM inversion [19] rather than from a dedicated denoising sampling process in the source branch. We discuss them in Section 5.7 and present the experimental results in Fig. 12.

###### L = λ1LSAP + λ2LCAA, (11)

where λ1 and λ2 are static balancing weights. However, this naive combination of constraints frequently produces unsatisfied editing results and causes image collapse when using consistent λ∗ values.

We visualize the gradient of L as follows and analyze the trends of the two gradients:

###### LCAA. (12)

Gnaive = λ1∇z∗

LSAP + λ2∇z∗

Cross-Attention Alignment Constraint. Larger and more aligned CA values signify stronger alignment of text-visual features, thereby enhancing editability in intended regions, as demonstrated in Section 4.1. Accordingly, we design the CA alignment

t

t

As illustrated in Fig. 5(a), since the ∇LCAA is substantially larger than ∇LSAP, the impact of the latter is significantly diminished, leading to over-editing or image collapse issues. To mitigate this

###### Source Image (a) Optimize 𝒛𝒕 using Eq.(12) (b) Optimize 𝒛𝒕 using Eq.(13)

###### (c) Optimize 𝒛𝒕 using Eq.(15)

[Figure 70]

[Figure 71]

[Figure 72]

×

× ×

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Purple pear

[Figure 77]

[Figure 78]

×

×

[Figure 79]

×

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Robot bird

[Figure 84]

×

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

×

[Figure 89]

×

[Figure 90]

Chicken

- Fig. 5: Editing and visualization results of different gradients. (a) Using Eq. (12) alone results in a significantly stronger influence

of LCAA, disabling LSAP and causing an unbalanced guidance on zt. (b) Although calculating their norms as in Eq. (13) brings the magnitudes of the constraints closer, the irregular dynamics lead to either under-editing or over-editing failures. (c) In contrast, applying the adaptive time-step scheduler in Eq. (14) shapes the gradient trends in Eq. (15) such that ∇z∗

LSAP starts small and gradually increases, whereas ∇z∗

t

LCAA exhibits the opposite trend, facilitating fidelity-editability balance. Algorithm 1: A Denoising Step Using UnifyEdit

t

in successful editing cases, ∇z∗

LSAP initially starts small and gradually increases throughout the denoising process, whereas

t

Input: An original and edited prompt P, P∗; a timestep t and corresponding noise latent features of source and target branches zt, zt∗; a maximum iteration step MAX IT and the hyperparameters β1,β2, k1, k2; a function F1(·) and a a function F2(·) for computing the proposed constraint LSAP and LCAA; a pre-trained Stable Diffusion model SD.

LCAA exhibits the opposite trend. These results can be explained as follows. During the early stages of denoising, the target and source diffusion trajectories remain relatively close, as they originate from the same last latent feature zT. This proximity requires a small ∇z∗

∇z∗

t

LSAP and a large ∇z∗

LCAA to enhance editability. As the diffusion denoising stage moves forward, the latent feature of the target branch progressively aligns with the new target prompt, necessitating an increase in ∇z∗

t

t

Output: the noisy latent feature zt∗−1 for the next

LSAP to preserve structural fidelity.

timestep of the target branch.

t

- 1 λ1 = β1e−k

1t ;

- 2 λ2 = β2(1 − e−k

2t) ;

- 3 for i = 1 to MAX IT do

- 4 ,Aselft , ← SD(zt,P,t) ;

- 5 ,A∗tself,A∗tcross ← SD(zt∗,P∗,t) ;

- 6

LSAP ← F1(Aselft ,A∗tself); LCAA ← F2(A∗tcross);

Gblc = λ∗1 ∇z

t∗

LSAP ||∇zt∗

LSAP||2 + λ∗2 ∇z

t∗

LCAA ||∇zt∗

LCAA||2 ; zˆt∗ = zt∗ − M ⊙ Gblc;

// UnifyEdit.

- 7 end
- 8 zt∗−1 ← SD(ˆzt∗,P∗,t) ;
- 9 Return zt∗−1

To enforce this desired gradient behavior, we propose an Adaptive Time-Step Scheduler which replaces the constants λ1 and λ2 with dynamic values λ∗1 and λ∗2:

- λ∗1 = β1(1 − e−k

1(T−t)),

- λ∗2 = β2e−k

(14)

2(T−t),

where the scaling factors β1 and β2 control the baseline values of the gradients, influencing the magnitude of LSAP at the endpoint and LCAA at the starting point of the optimization process. The rate factors k1 and k2 determine the rates at which the gradients rise and decay, respectively. The variable t ∈ {T,...,1}, indicates the timestep of the denoising sampling process in the T2I diffusion model, allowing the weighting of each constraint to be dynamically adjusted based on the timestep t. We define the adaptive time-step gradient Gblc as:

imbalance, we first propose normalizing the two gradients using their L2 norm as:

+ λ∗2 ∇z∗

LCAA ||∇z∗

Gblc = λ∗1 ∇z∗

LSAP ||∇z∗

. (15)

t

t

LSAP||2

LCAA||2

t

t

Gnorm = λ1 ∇z∗

LSAP ||∇z∗

+ λ2 ∇z∗

LCAA ||∇z∗

. (13)

Consequently, Eq. (6) can be re-written as follows:

t

t

LSAP||2

LCAA||2

t

t

###### zˆt∗ = zt∗ − M ⊙ Gblc. (16)

Although Eq. (13) brings the effects of both constraints to a similar magnitude, the dynamics of the normalized gradients remain irregular, resulting in unstable editing outcomes. Consequently, both under-editing and over-editing occur, as illustrated in Fig. 5(b).

Fig. 10(c) demonstrates that applying the adaptive time-step scheduler, implemented with Eq. (15), effectively shapes gradient trends as intended, thereby achieving a better balance between fidelity and editability. The main steps of UniyEdit are summarized in Algorithm 1. This tuning-free, inference-stage optimization

Furthermore, we manually adjust λ1 and λ2 and observe that

###### Edit Type Source Image Source Prompt Target Prompt Mask

[Figure 91]

[Figure 92]

Color a black bird a white bird Change there is a girl wearing

there is a girl wearing a yellow beanie

[Figure 93]

[Figure 94]

a purple beanie

[Figure 95]

[Figure 96]

Foreground

Texture a man in a beige suit a man in a lace suit Modification there is a woman in green

there is a woman in wool pants sitting at a table

[Figure 97]

[Figure 98]

Editing

pants sitting at a table

[Figure 99]

[Figure 100]

Object a dog a fox Replacement there is a dog standing

there is a cat standing on a pink background

[Figure 101]

[Figure 102]

on a pink background

[Figure 103]

[Figure 104]

Background Editing \ a man falls on snow a man falls on grass

[Figure 105]

Global Style Transfer \ a photo of a fox a drawing of a fox None

###### Human Face

[Figure 106]

[Figure 107]

Attribute Editing \ a photo of a smiling man a photo of a crying man

- TABLE 1: Examples in Unify-Bench. Each image in Unify-Bench is annotated with a source prompt, a target edit prompt, and an edit region mask. Complex scenarios within the dataset are distinctly highlighted with a grey.

method is adaptable and can seamlessly apply to any pre-trained T2I models. It is important to note that parameters β1, β2, k1, and k2 can be adjusted to balance fidelity and editability, accommodating different editing requirements and tailoring to users’ preferences.

- 5 EXPERIMENTS

- 5.1 Benchmark Dataset

To facilitate comprehensive evaluations of our method’s ability to balance fidelity and editability across different editing types, we develop a benchmark dataset named Unify-Bench. This dataset comprises 181 images sourced from TEd-Bench [6], PIEBench [20], Magicbrush [5], and the Internet. Unify-Bench is designed to assess the editing capabilities of various methods across different editing regions. It includes a diverse range of edits such as foreground modifications, background alterations, global style transfers, and specialized human face attribute editing tasks:

- • Foreground editing: it encompasses color change, texture modification, and object replacement. These edits are applied to simple scenarios, which feature a single prominent object, and complex scenarios, which are characterized by multiple objects of the same kind arranged in intricate layouts. In complex scenarios, edits are specifically targeted at a single object.
- • Background editing: it focuses on replacing or modifying the scene behind the foreground subjects.
- • Global style transfer: it aims to globally change the visual style of an image while preserving its underlying content.
- • Human face attribute editing: it includes changing facial expressions, hair color, gender, age, and etc.

For generating source and target prompts, we initially utilized GPT-4 [47], followed by manual refinement to ensure the accuracy

and relevance of these prompts. For conciseness, we use the simplest possible prompts, employing the format “a XX” for simple scenarios and “there is XX in/on XX” for complex scenarios. For localized editing, we generate the corresponding editing masks using the Segment Anything method [48]. Thus, each image in Unify-Bench is annotated with a source prompt, a target edit prompt, and an edit region mask, as detailed in Table 1.

###### 5.2 Implementation Details

All experiments are conducted on a single NVIDIA A100 GPU. We utilize the official pre-trained Stable Diffusion v1.4 model [49] as our base model. We choose Null-Text inversion (NTI) [11] as the inversion method to obtain zT, and the denoising sampling process employs the DDIM method [19] over T = 50 steps, maintaining a constant CFG scale of 7.5. Our approach is compatible with various inversion methods (see Section 5.7). The LCAA, LSAP are applied during diffusion steps within the ranges [T,τ1] and [T,τ2], respectively, with both τ1 and τ2 typically set at 25. However, τ1 is specifically set at 5 for color editing. We generally set the scaling factors β1 and β2 to 5. For rate factors, we set k1 and k2 as follows: 0.05 for color change, 0.08 for texture modification or background editing, 0.15 for object replacement, 0.1 for global style transfer and 0.25 for human face attribute editing. The optimization process is further defined by the maximum number of iterations, empirically set to MAX IT = 1. For localized editing, we employ the mask blending operation in P2P [8] with the annotated masks to better preserve the original information in areas outside the mask.

###### 5.3 Evaluation Metrics

We quantitatively evaluate our proposed method against baseline models using both automatic metrics and human evaluations.

Source Image DiffEdit P2P PnP+Blend SPDInv G-R MAG-Edit Ours

P2P+Blend PnP

###### Purple pear

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

White prairie

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

Editing Type: Color Change

###### Floral jacket

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

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Bohemian carpet

Editing Type: Texture Modification

Eggplant

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Orangutan

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Editing Type: Object Replacement

Frozen water

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

Muddy street

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Cartoon photo

Editing Type: Background Editing

###### Cartoon

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

Claymation

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

Editing Type: Global Style Transfer

Angry woman

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

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Surprised woman

Editing Type: Human Face Attribute Editing

- Fig. 6: Qualitative comparisons across various editing types. We use white dashed outlines to highlight the target object in foreground editing. Our proposed method achieves a superior balance compared to other baseline methods, demonstrating enhanced editing effects while more effectively maintaining structural consistency.

[Figure 228]

(a) (b) (c)

[Figure 229]

[Figure 230]

× × ×

[Figure 231]

[Figure 232]

(d) (e) (f)

[Figure 233]

× × ×

- Fig. 7: Quantitative comparisons of baselines and our UnifyEdit across various editing types. We quantify editability and fidelity using CLIP sore (righter is better) and DINO similarity distance (lower is better), respectively. Balancing the aspects requires a high CLIP score and relatively low DINO similarity. Therefore, points closer to the pink region of the background represent better performance, while those closer to the blue region indicate poorer performance.

Automatic Metrics. We assess the efficiency of our method in terms of fidelity and editability using the following automatic metrics: 1) fidelity: We calculate the DINO-ViT self-similarity (DINO Similarity) [50] between the source and edited images to analyze structure preservation. 2) editability: We compute the CLIP score [51] with the CLIP ViT-L/14 model by evaluating the similarity between text and image embeddings in a shared space to measure image-text alignment. We adapt the reference code for localized editing to crop the target regions in both source and edited images using bounding boxes, as described in [31].

[Figure 234]

Fig. 8: Average human preferences across various editing types. The values indicate the proportion of users who preferred our proposed method over comparative approaches.

User Study. We conduct a user study on the Amazon MTurk platform [52] to evaluate our proposed method. In each questionnaire, participants are presented with a source image and two edited images: one generated by our proposed method and the other by a randomly selected baseline method. The presentation order of the edited images is randomized to avoid bias. To enhance their visibility, we outline the desired edited regions with white dashed lines in both the source and edited images. Additionally, a simplified version of the target edit prompt is displayed beneath the comparison images to facilitate direct comparisons. Following the evaluation approach of [14], participants are asked to respond to three questions:

- • Inpainting-based methods: DiffEdit [13] is a typical inpainting method that employs an implicitly predicted mask to preserve the non-editing region.
- • Attention-injection-based methods: P2P [8] and PnP [9] utilize attention injection to maintain fidelity across the entire image. P2P+Blend and PnP+Blend are enhanced P2P [8] and PnP [9] with blending operations to ensure fidelity outside the editing mask remains unchanged.
- • Inversion-based methods: SPD Inversion (SPD Inv) [17] is the latest inversion-based method that focuses on improving the DDIM inversion [19] to achieve more accurate reconstruction results.
- • Gradient-based methods: Guide-and-Rescale (G-R) [18] is a recent method that leverages noise guidance in TIE.

- • Structure Preservation: In the dashed region, which image preserves structures more similarly to the source image?
- • Text Alignment: Which image aligns better with the “edit prompt” in the dashed region?
- • Overall: In the dashed region, which image performs better overall?

Similar to our LSAP, it aims at reducing discrepancies between SA maps generated during reconstruction and editing, thereby achieving fidelity. MAG-Edit [14] builds on attentioninjection-based backbones like P2P [8] to maintain fidelity while introducing two constraints to enhance text alignment

###### 5.4 Comparisons with state-of-the-art Methods

Baselines. We evaluate our approach against existing state-of-theart TIE methods, categorized as follows:

Unify Edit Color Change DINO Similarity ↓ 0.124 0.077 0.127 0.088 0.084

w/o SAP

w/o

Source Image w/o SAP w/o CAA UnifyEdit

Method

CAA Gnaive Gnorm

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

White car

CLIP Score ↑ 21.83 20.87 21.13 21.82 22.49

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

Texture Modification DINO Similarity ↓ 0.099 0.082 0.097 0.090 0.083

CLIP Score ↑ 21.48 20.39 19.93 22.30 22.71

Wooden chair

Object Replacement DINO Similarity ↓ 0.109 0.082 0.096 0.100 0.077

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

CLIP Score ↑ 23.27 22.76 22.78 23.16 23.51

Background Editing DINO Similarity ↓ 0.146 0.104 0.157 0.122 0.102

Sculpture cat

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

CLIP Score ↑ 18.33 16.36 16.67 18.88 18.36

Global Style Transfer DINO Similarity ↓ 0.113 0.053 0.091 0.081 0.066

Vangogh-style

CLIP Score ↑ 22.77 17.24 17.19 23.04 20.83

Fig. 9: Qualitative results of ablation study on attention-based constraints. White dashed outlines are used to highlight the target object in foreground editing. Combining both terms is crucial for achieving a good balance between fidelity and editability.

Human Face Attribute Editing DINO Similarity ↓ 0.083 0.053 0.099 0.071 0.054

CLIP Score ↑ 19.53 17.75 18.89 19.55 18.30

outside the mask. While SPDInv [17] enhances DDIM inversion to generate more accurate features, its reliance on attention injection still limits its effectiveness in achieving a balanced outcome. The gradient-based method, G-R [18], which optimizes noise ϵ∗θ to minimize the gap between Aself and A∗self, demonstrates greater flexibility in structure preservation compared to attentioninjection-based methods, particularly in human face attribute editing. However, due to its limited focus on text alignment, its editing effects are constrained in scenarios that require higher editability, as demonstrated in examples such as the “cartoon” style. Although MAG-Edit [14] focuses on enhancing text alignment, its reliance on attention injection to preserve fidelity limits its editability, resulting in under-editing issues in cases like the “purple pear”. In contrast, our method demonstrates superior editing adaptability, effectively balancing fidelity and editability across a wide range of editing types.

- TABLE 2: Quantitative results of ablation study. Bold and underline indicate the best and second best value, respectively.

for editability.

We use the official codes released by the authors for P2P [53], PnP [54], SPDInv [55], G-R [56], and MAG-Edit [57]. For DiffEdit [13], we adopt the implementation of InstructEdit [58]. To facilitate fair comparisons, all methods use the identical masks provided in our benchmark dataset and the Stable Diffusion v1.4 model as the backbone. Notably, for DiffEdit [13], P2P [8] + Blend, PnP [9] + Blend and SPDInv [17], we utilize ground-truth masks instead of those generated through unsupervised learning or derived from average CA maps. In the case of P2P [8] and MAGEdit [14], we also integrate NTI [11] as our approach to encode real images.

Qualitative Results. Fig. 6 shows that DiffEdit [13], which employs DDIM inversion [19] for foreground generation, significantly alters the structure fidelity across all editing types. For example, while it successfully generates an “angry woman”, it loses critical identity information from the source image. Attentioninjection-based methods like P2P [8] and PnP [9] effectively preserve the original structure in editing scenarios that do not require shape variations. However, as discussed in Section 4.1, directly copying attention maps restricts flexibility in editability, leading to under-editing issues such as insufficient color change and texture modification. As shown in the first four rows of Fig. 6, these methods fail to generate the corresponding colors and textures specified by the text prompts. On the other hand, for significant shape variations, such as changing a “dog” to an “orangutan,” these methods may alter the posture too drastically compared to the source image, leading to over-editing issues. Combined with the blending operation, P2P+Blend and PnP+Blend still exhibit the same issues, although they effectively preserve the regions

Quantitative Results. Fig. 7 shows quantitative results of the evaluation methods with CLIP score on the x-axis and DINO similarity on the y-axis. The points on the bottom-right (high CLIP score and low DINO similarity) represent better balance performance. We use a colormap to visualize the performance of the compared methods and ours, ranging from blue to pink. As shown in Fig. 7(a)(b)(d), our method performs favorably against the baselines by achieving better alignment with the target text for edits involving minimal shape variations, such as those related to color and texture. Although object replacement and global style transfer entail significant shape or texture changes, maintaining structural consistency is vital to preserving the source image’s visual integrity. Our approach records the lowest DINO similarity for human facial attribute editing, as shown in Fig. 7(f). While DiffEdit [13] and G-R [18] obtain higher CLIP scores, these methods do not preserve the subject’s identity, as demonstrated in the last two rows of Fig. 6. These results show that our method

Source Image 𝐺𝑛𝑎𝑖𝑣𝑒 UnifyEdit

𝐺𝑛𝑜𝑟𝑚

Ours+ NTI [11]

DiffEdit [13]

P2P [8]

PnP [9]

SPDInv [17]

G-R [18]

MAGEdit [14]

Method

Yellow carpet

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Inversion Time (s)

- 4.2 87.9 46.5 21.5 3.0 87.9 87.9 Denoising Time (s)

- 5.3 10.7 93.2 10.6 30.2 83.9 24.2 Memory (GB)

Painting cup

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

10.8 12.8 17.6 16.2 25.7 19.5 21.4

TABLE 3: Runtime and GPU memory requirements for the baselines and our proposed method.

lose structural integrity in the w/o SAP examples, causing noticeable artifacts. This is reflected in a relatively high CLIP score but a very low DINO similarity, as presented in Table 2. In contrast, structural fidelity is maintained without the influence of LCAA, but the lack of sufficient text alignment results in unsatisfactory edits. For instance, the “sculpture cat” retains excessive structure information from the source images, compromising texture changes’ editability. As a result, both the CLIP score and DINO similarity are low, as presented in Table 2. As shown in the last column of Fig. 9, utilizing both constraints together yields the best results.

White bird

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Watercolor-painting

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

- Fig. 10: Qualitative results of ablation study on different gradients. The target object is accentuated with white dashed outlines in the foreground editing. Gnaive in Eq. (12) can lead to over-editing and, in some cases, image collapse. While Gnorm in Eq. (13) mitigates these issues, it still encounters both underediting and over-editing failures. In contrast, our method, which employs Gblc in Eq. (15), successfully achieves a balanced result.

5.6 Ablation Study on Adaptive Time-Step Scheduler We conduct ablation studies from two perspectives to explore the effectiveness of the adaptive time-step scheduler:

Impact on the Different Gradients. We compare with optimization using Gnaive in Eq. (12), Gnorm in Eq. (13) and Gblc in Eq. (15) (i.e., UnifyEdit), respectively. As discussed in Section 4.3, the direct combination of LSAP and LCAA for optimizing zt results in the predominance of LCAA’s gradient. This dominance can lead to a significant loss of structural fidelity and image collapse demonstrated in Fig. 10. Utilizing the L2 norm to balance the two constraints mitigates the risk of image collapse, yet it still results in over- or under-editing issues. As shown in Fig. 10, the “painting cup” exhibits weak editing effects, while the structure of the “white bird” deviates from the original image. In contrast, the proposed adaptive time-step scheduler effectively balances the influence of the two constraints, achieving optimal editing results across a diverse range of editing scenarios.

performs well with a robust balance compared to other baselines across various editing types.

User Study. To ensure reliability, we invite Amazon MTurk workers with ‘Master’ status and a Human Intelligence Task (HIT) Approval Rate exceeding 90% across all Requesters’ HITs. We collect 1,750 completed questionnaires from these subjects.

- As shown in Fig. 8, the percentages indicate the proportion of participants who preferred our proposed method over baseline approaches. For fidelity, a significant majority, ranging from 66% to 84%, indicates that our method demonstrates superior structure preservation compared to existing methods. Regarding editability, our method is preferred for improved text alignment, with preference rates ranging from 68% to 89%. Overall, our proposed method is favored by 68% to 86% of participants due to its effective balance between editability and fidelity.

5.5 Ablation Study on Attention-Based Constraints

We conduct ablation studies on the following variations to validate the role of two attention-based constraints,

- 1) w/o SAP: diffusion latent feature zt is optimized without the gradient of LSAP, meaning that the Gblc in Eq. (15) is

replaced with GCAA = λ∗2 ∇z

t∗

LCAA ||∇zt∗

LCAA||2 .

- 2) w/o CAA: diffusion latent feature zt is optimized without the gradient of LCAA, so Gblc in Eq. (15) is replaced with

GSAP = λ∗1 ∇z

t∗

LSAP ||∇zt∗

LSAP||2 .

- As shown in Fig. 9, the absence of LSAP results in strong editing effects that align closely with the prompt but introduce significant structural discrepancies. For example, both the “wooden chair”

Impact on the Hyper-Parameters. As discussed in Section 4.3, the scaling factors β1, β2, and rate factors k1, k2 are essential for adjusting the weights of LSAP and LCAA in the adaptive time-step scheduler. As outlined in Section 5.2, we define the standard parameter settings for various editing tasks as baseline parameters, represented as Pbase = (β1,β2,k1,k2)j, where j refers to a specific editing type. To investigate the impact of β1, we adjust its baseline value by adding or subtracting 3.0, denoted as Pbase(β1±3), while maintaining the other parameters constant. Similarly, for the rate factors, we modulate k1 by modifying its baseline value by ±0.04, expressed as Pbase(k1±0.04). The same procedure is applied to β2 and k2. We visualize the mean absolute value of GSAP and GCAA over timesteps under different settings across various editing types. The gradient variations induced by the parameters and the resulting differences in editing effects are consistent across these types. We present the average values across all editing types in Fig. 11 for simplicity. As illustrated in Fig. 11(a)(b), the scaling factors determine the overall magnitude of the gradients, represented by the vertical shift of the curve. For instance, Pbase(β1 +3) raises the GSAP curve, causing the final gradient guidance on zt to shift away from LCAA. As

[Figure 267]

[Figure 268]

× ×

Source Image 𝑃𝑏𝑎𝑠𝑒(𝛽1 − 3) 𝑃𝑏𝑎𝑠𝑒 𝑃𝑏𝑎𝑠𝑒(𝛽1 + 3) 𝑃𝑏𝑎𝑠𝑒(𝛽2 − 3) 𝑃𝑏𝑎𝑠𝑒 𝑃𝑏𝑎𝑠𝑒(𝛽2 + 3)

Source Image

Chinese-painting Written book

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

(a) Impact of 𝜷𝟏 (b) Impact of 𝜷𝟐

[Figure 277]

[Figure 278]

× ×

Source Image 𝑃𝑏𝑎𝑠𝑒(𝑘1 − 0.04) 𝑃𝑏𝑎𝑠𝑒 𝑃𝑏𝑎𝑠𝑒(𝑘1 + 0.04) Source Image 𝑃𝑏𝑎𝑠𝑒(𝑘2 − 0.04) 𝑃𝑏𝑎𝑠𝑒 𝑃𝑏𝑎𝑠𝑒(𝑘2 + 0.04)

Brown chair Starry sky

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

(c) Impact of 𝒌𝟏 (d) Impact of 𝒌𝟐

- Fig. 11: Ablation study on hyper-parameters in adaptive time-step scheduler. The scaling factors β1 and β2, along with the rate factors k1 and k2, regulate the magnitude and changing rate, influencing the editing outcomes.

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

Source Image Output Source Image Output

Salmon suit

Floral dress Striped cat

Golden hair

- Fig. 12: Editing results using DDIM inversion. The proposed method maintains effectiveness by employing SA constraints derived from the SA maps generated during the DDIM inversion.

Source Image Noise Guidance Latent Optimization

[Figure 295]

[Figure 296]

[Figure 297]

Golden veil

[Figure 298]

[Figure 299]

[Figure 300]

Robot horse

Fig. 13: Diffusion latent optimization vs. noise guidance. Latent optimization outperforms noise guidance in balancing fidelity and editability.

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Pink T-shirt Green T-shirt Blue T-shirt Yellow T-shirt Chocolate cake Fruit cake Jelo cake Birthday cake

Source Image

Source Image

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

Source Image Moss-covered street Icy street Cobblestoned street Sandy street

Kitten Tiger Raccoon Sheep

Source Image

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

Monet-style Winter Vintage Evening

Man Child Heavy makeup Angry

Source Image Source Image

(a) Editing results across various editing types

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

Red and wavy hair Source Image Yellow and plaid hoodie Source Image Leather and broken chair Source Image Wooden and carved table

Source Image

(b) Editing with multiple prompts

Fig. 14: More editing results of UnifyEdit. We highlight the target object with white dashed outlines in foreground editing. UnifyEdit can achieve balance across various editing types and can be applied to multiple target editing tokens.

Runtime and Memory Usage. We report the runtime and GPU memory usage for our proposed method with NTI [11] and the baseline methods on an Nvidia A100 (40GB) GPU in Table 3. The time consumption and memory usage of our method are mainly attributed to latent optimization in the denoising process, yet they remain moderate compared to the other baselines.

shown in Fig. 11(a), this results in weaker style rendering for the “Chinese painting” task. Conversely, Pbase(β1 − 3) lowers the GSAP curve, leading to weaker editing effects. Notably, β2 has a similar influence on the GCAA curve, but its effect leads to the opposite outcome in the final edited results. The editing effects for the “written” token are prominent in Pbase(β2 + 3), while they are negligible in Pbase(β2 − 3) (see Fig. 11(a)). The rate factors affect the rate at which the gradients change, reflected in the steepness of the curves, as shown in Fig. 11(c)(d). For instance, Pbase(k1 + 0.04) causes the GSAP curve to rise more quickly, increasing the influence of LSAP on the latent zt. In contrast, a smaller Pbase(k1 − 0.04) slows the increase of the GSAP curve, resulting a higher influence of LCAA. The editing effects for the “brown chair” are stronger under Pbase(k1 − 0.04) and weaker under Pbase(k1 + 0.04) (see Fig. 11(c)). Similarly, k2 has the same impact on the descent rate of GCAA and the overall editing results.

Additional Results. As shown in Fig. 14(a), our method effectively balances fidelity and editability across various editing tasks. Furthermore, as demonstrated in Fig. 14(b), our proposed method can also be applied to multiple target tokens (e.g., “red and wavy”).

##### 6 CONCLUSIONS AND FUTURE WORKS

In this work, we present one of the initial efforts to explicitly model the balance between fidelity and editability within a unified diffusion latent optimization framework. Our approach is novel in two ways: It incorporates attention-based constraints from the SA and CA that control fidelity and editability, and an adaptive timestep scheduler that balances these constraints. Quantitative and qualitative results demonstrate that UnifyEdit achieves a superior balance against existing methods across a broad spectrum of editing tasks. Overall, our method significantly advances the field of tuning-free diffusion-based TIE by offering a unified approach that explicitly controls the balance between fidelity and editability. This approach not only meets diverse editing requirements but can also be adjusted dynamically to align with users’ preferences.

###### 5.7 Discussions

Compatibility with Other Inversion Methods. As discussed in Section 4.2, our proposed method is compatible with other inversion techniques. We demonstrate an extreme case aimed at minimizing the gap between the SA maps from the target branch and those generated during the DDIM inversion [19] process defined in Eq. (3), similar to the inversion attention fusion proposed in [59]. Fig. 12 shows that our method integrates successfully with this approach, producing the desired editing results.

Comparisons with Noise Guidance. As discussed in Section 2.3, latent optimization uses the gradient g to directly optimize zt, resulting in zˆt = zt − g. Instead, noise guidance updates zt−1 by using the gradient to adjust the noise estimate, yielding ϵˆtθ = ϵtθ − g. Using noise guidance results in the ineffective “golden veil” and the loss of structural integrity in the “robot horse”, demonstrating its relative ineffectiveness in balancing editability and fidelity compared to our method (see Fig. 13).

However, since the SA map captures extensive layouts and semantic information, the proposed SA preservation constraint somewhat constrains the rigidity of target objects. Consequently, our method may face challenges with non-rigid transformations, such as changing a sitting dog into a jumping dog. We aim to address these challenges by developing a non-rigid self-attention constraint to enhance the method’s adaptability to dynamic transformations in future work.

##### REFERENCES

- [1] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, 2022. 1
- [2] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in CVPR, 2022. 1, 4
- [3] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” in NeurIPS, 2022. 1
- [4] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in CVPR, 2023. 1, 2, 3
- [5] K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su, “Magicbrush: A manually annotated dataset for instruction-guided image editing,” in NeurIPS,

2023. 1, 2, 3, 8

- [6] B. Kawar, S. Zada, O. Lang, O. Tov, H. Chang, T. Dekel, I. Mosseri, and M. Irani, “Imagic: Text-based real image editing with diffusion models,” in CVPR, 2023. 1, 2, 3, 8
- [7] Z. Zhang, L. Han, A. Ghosh, D. N. Metaxas, and J. Ren, “Sine: Single image editing with text-to-image diffusion models,” in CVPR, 2023. 1, 2, 3
- [8] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or, “Prompt-to-prompt image editing with cross attention control,” in ICLR, 2023. 1, 2, 3, 4, 5, 8, 10, 11, 12
- [9] N. Tumanyan, M. Geyer, S. Bagon, and T. Dekel, “Plug-and-play diffusion features for text-driven image-to-image translation,” in CVPR, 2023. 1, 2, 3, 4, 5, 10, 11, 12
- [10] G. Parmar, K. Kumar Singh, R. Zhang, Y. Li, J. Lu, and J.-Y. Zhu, “Zeroshot image-to-image translation,” in SIGGRAPH, 2023. 1, 2, 4
- [11] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. Cohen-Or, “Nulltext inversion for editing real images using guided diffusion models,” in CVPR, 2023. 1, 2, 3, 4, 5, 8, 11, 12, 14
- [12] M. Cao, X. Wang, Z. Qi, Y. Shan, X. Qie, and Y. Zheng, “Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing,” in ICCV, 2023. 1, 2, 3, 4, 5
- [13] G. Couairon, J. Verbeek, H. Schwenk, and M. Cord, “Diffedit: Diffusionbased semantic image editing with mask guidance,” in ICLR, 2023. 1, 2, 3, 5, 10, 11, 12
- [14] Q. Mao, L. Chen, Y. Gu, Z. Fang, and M. Z. Shou, “Mag-edit: Localized image editing in complex scenarios via mask-based attention-adjusted guidance,” in ACM MM, 2024. 1, 3, 4, 5, 10, 11, 12
- [15] O. Avrahami, O. Fried, and D. Lischinski, “Blended latent diffusion,” ACM TOG, 2023. 1, 3, 5
- [16] Y. Qiao, F. Wang, J. Su, Y. Zhang, Y. Yu, S. Wu, and G.-J. Qi, “Baret: Balanced attention based real image editing driven by target-text inversion,” in AAAI, 2024. 1, 2, 3, 4
- [17] R. Li, R. Li, S. Guo, and L. Zhang, “Source prompt disentangled inversion for boosting image editability with diffusion models,” in ECCV,

2024. 1, 2, 3, 4, 5, 10, 11, 12

- [18] V. Titov, M. Khalmatova, A. Ivanova, D. Vetrov, and A. Alanov, “Guideand-rescale: Self-guidance mechanism for effective tuning-free real image editing,” in ECCV, 2024. 1, 2, 3, 4, 10, 11, 12
- [19] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in ICLR, 2021. 1, 2, 3, 4, 6, 8, 10, 11, 14
- [20] X. Ju, A. Zeng, Y. Bian, S. Liu, and Q. Xu, “Pnp inversion: Boosting diffusion-based editing with 3 lines of code,” in ICLR, 2024. 1, 3, 5, 8
- [21] B. Liu, C. Wang, T. Cao, K. Jia, and J. Huang, “Towards understanding cross and self-attention in stable diffusion for text-guided image editing,” in CVPR, 2024. 1, 5
- [22] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020. 2
- [23] G. Kim, T. Kwon, and J. C. Ye, “Diffusionclip: Text-guided diffusion models for robust image manipulation,” in CVPR, 2022. 2, 3
- [24] M. Kwon, J. Jeong, and Y. Uh, “Diffusion models already have A semantic latent space,” in ICLR, 2023. 2, 3
- [25] D. Valevski, M. Kalman, E. Molad, E. Segalis, Y. Matias, and Y. Leviathan, “Unitune: Text-driven image editing by fine tuning a diffusion model on a single image,” ACM TOG, 2022. 2, 3
- [26] M. Brack, F. Friedrich, K. Kornmeier, L. Tsaban, P. Schramowski, K. Kersting, and A. Passos, “Ledits++: Limitless image editing using text-to-image models,” in CVPR, 2024. 2, 5
- [27] Q. Guo and T. Lin, “Focus on your instruction: Fine-grained and multiinstruction image editing by attention modulation,” in CVPR, 2024. 3, 5

- [28] S. Zhang, X. Yang, Y. Feng, C. Qin, C.-C. Chen, N. Yu, Z. Chen, H. Wang, S. Savarese, S. Ermon et al., “Hive: Harnessing human feedback for instructional visual editing,” in CVPR, 2024. 3
- [29] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” in NeurIPS, 2021. 3
- [30] O. Avrahami, D. Lischinski, and O. Fried, “Blended diffusion for textdriven editing of natural images,” in CVPR, 2022. 3, 5
- [31] W. Huang, S. Tu, and L. Xu, “Pfb-diff: Progressive feature blending diffusion for text-driven image editing,” Neural Networks, 2025. 3, 5, 10
- [32] Q. Wang, B. Zhang, M. Birsak, and P. Wonka, “Instructedit: Improving automatic masks for diffusion-based image editing with user instructions,” arXiv preprint arXiv:2305.18047, 2023. 3, 5
- [33] C. Tang, K. Wang, F. Yang, and J. van de Weijer, “Locinv: Localizationaware inversion for text-guided image editing,” CVPR 2024 AI4CC workshop, 2024. 3, 4, 5
- [34] K. Wang, X. Song, M. Liu, J. Yuan, and W. Guan, “Vision-guided and mask-enhanced adaptive denoising for prompt-based image editing,” arXiv preprint arXiv:2410.10496, 2024. 3, 4, 5
- [35] H. Chefer, Y. Alaluf, Y. Vinker, L. Wolf, and D. Cohen-Or, “Attend-andexcite: Attention-based semantic guidance for text-to-image diffusion models,” in SIGGRAPH, 2023. 4, 6
- [36] R. Rassin, E. Hirsch, D. Glickman, S. Ravfogel, Y. Goldberg, and G. Chechik, “Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment,” in NeurIPS, 2023. 4
- [37] J. Xie, Y. Li, Y. Huang, H. Liu, W. Zhang, Y. Zheng, and M. Z. Shou, “Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion,” in ICCV, 2023, pp. 7452–7461. 4
- [38] S. Ge, T. Park, J.-Y. Zhu, and J.-B. Huang, “Expressive text-to-image generation with rich text,” in ICCV, 2023. 4
- [39] O. Dahary, O. Patashnik, K. Aberman, and D. Cohen-Or, “Be yourself: Bounded attention for multi-subject text-to-image generation,” in ECCV,

2024. 4

- [40] J. Liu, T. Huang, and C. Xu, “Training-free composite scene generation for layout-to-image synthesis,” in ECCV, 2025. 4
- [41] C. Mou, X. Wang, J. Song, Y. Shan, and J. Zhang, “Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing,” in CVPR,

2024. 4

- [42] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in ICLR, 2021. 4
- [43] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” in NeruIPS workshop, 2021. 5
- [44] H. Cho, J. Lee, S. B. Kim, T.-H. Oh, and Y. Jeong, “Noise map guidance: Inversion with spatial context for real image editing,” in ICLR, 2023. 5
- [45] O. Patashnik, D. Garibi, I. Azuri, H. Averbuch-Elor, and D. Cohen-Or, “Localizing object-level shape variations with text-to-image diffusion models,” in ICCV, 2023. 5
- [46] S. Lu, Y. Liu, and A. W.-K. Kong, “Tf-icon: Diffusion-based training-free cross-domain image composition,” in ICCV, 2023. 5
- [47] OpenAI, “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774,

2023. 8

- [48] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Doll´ar, and R. Girshick, “Segmentanything model,” 2023. [Online]. Available: https://github.com/facebookresearch/segment-anything 8
- [49] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” 2022. [Online]. Available: https://huggingface.co/CompVis/stable-diffusion-v1-4 8
- [50] N. Tumanyan, O. Bar-Tal, S. Bagon, and T. Dekel, “Splicing vit features for semantic appearance transfer,” 2022. [Online]. Available: https://github.com/omerbt/Splice 10
- [51] J. Z. Wu, X. Li, D. Gao, Z. Dong, J. Bai, A. Singh, X. Xiang, Y. Li, Z. Huang, Y. Sun, R. He, F. Hu, J. Hu, H. Huang, H. Zhu, X. Cheng, J. Tang, M. Z. Shou, K. Keutzer, and F. Iandola, “Cvpr 2023 text guided video editing competition,” 2023. [Online]. Available: https://github.com/showlab/loveu-tgve-2023 10
- [52] “Amazon mechanical turk.” [Online]. Available: https://requester.mturk. com/create/projects/new 10
- [53] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or, “Prompt-to-prompt image editing with cross attention control,” 2022. [Online]. Available: https://github.com/google/ prompt-to-prompt 11
- [54] N. Tumanyan, M. Geyer, S. Bagon, and T. Dekel, “Plug-and-play diffusion features for text-driven image-to-image translation,” 2023. [Online]. Available: https://github.com/MichalGeyer/plug-and-play 11

- [55] R. Li, R. Li, S. Guo, and L. Zhang, “Source prompt disentangled inversion for boosting image editability with diffusion models,” 2024. [Online]. Available: https://github.com/leeruibin/SPDInv 11
- [56] V. Titov, M. Khalmatova, A. Ivanova, D. Vetrov, and A. Alanov, “Guide-and-rescale: Self-guidance mechanism for effective tuning-free real image editing,” 2024. [Online]. Available: https://github.com/ AIRI-Institute/Guide-and-Rescale 11
- [57] Q. Mao, L. Chen, Y. Gu, Z. Fang, and M. Z. Shou, “Magedit: Localized image editing in complex scenarios via maskbased attention-adjusted guidance,” 2024. [Online]. Available: https: //github.com/HelenMao/MAG-Edit 11
- [58] Q. Wang, B. Zhang, M. Birsak, and P. Wonka, “Instructedit: Improving automatic masks for diffusion-based image editing with user instructions.” 2023. [Online]. Available: https://github.com/QianWangX/ InstructEdit 11
- [59] C. Qi, X. Cun, Y. Zhang, C. Lei, X. Wang, Y. Shan, and Q. Chen, “Fatezero: Fusing attentions for zero-shot text-based video editing,” in ICCV, 2023. 14

