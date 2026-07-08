## DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization

Jisu Nam∗1, Heesu Kim2, DongJae Lee2, Siyoon Jin1, Seungryong Kim†1, Seunggyu Chang†2 1Korea University 2NAVER Cloud

https://ku-cvlab.github.io/DreamMatcher

# arXiv:2402.09812v2[cs.CV]23Apr2024

Textual Inversion Ours DreamBooth Ours CustomDiffusion Ours

Reference

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

v v

in the snow

inside a box with a blue house in the background

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

with a beautiful sunset inside a basket on top of green grass with sunflowers

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

jumping over a fence in an astronaut outfit with a landscape from the Moon

Figure 1. DreamMatcher enables semantically-consistent text-to-image (T2I) personalization. Our DreamMatcher is designed to be compatible with any existing T2I personalization models, without requiring additional training or fine-tuning. When integrated with them, DreamMatcher significantly enhances subject appearance, including colors, textures, and shapes, while accurately preserving the target structure as guided by the target prompt.

constrained to local editing since they disrupt the structure path of the pre-trained T2I model. To overcome this, we propose a novel plug-in method, called DreamMatcher, which reformulates T2I personalization as semantic matching. Specifically, DreamMatcher replaces the target values with reference values aligned by semantic matching, while leaving the structure path unchanged to preserve the versatile capability of pre-trained T2I models for generating diverse structures. We also introduce a semantic-consistent masking strategy to isolate the personalized concept from irrelevant regions introduced by the target prompts. Compatible with existing T2I models, DreamMatcher shows significant improvements in complex scenarios. Intensive analyses demonstrate the effectiveness of our approach.

### Abstract

The objective of text-to-image (T2I) personalization is to customize a diffusion model to a user-provided reference concept, generating diverse images of the concept aligned with the target prompts. Conventional methods representing the reference concepts using unique text embeddings often fail to accurately mimic the appearance of the reference. To address this, one solution may be explicitly conditioning the reference images into the target denoising process, known as key-value replacement. However, prior works are

†Co-corresponding author. ∗Work done during an internship at NAVER Cloud.

### 1. Introduction

The objective of text-to-image (T2I) personalization [17, 32, 44] is to customize T2I diffusion models based on the subject images provided by users. Given a few reference images, they can generate novel renditions of the subject across diverse scenes, poses, and viewpoints, guided by the target prompts.

Conventional approaches [14, 17, 21, 32, 44, 62] for T2I personalization often represent subjects using unique text embeddings [42], by optimizing either the text embedding itself or the parameters of the diffusion model. However, as shown in Figure 1, they often fail to accurately mimic the appearance of subjects, such as colors, textures, and shapes. This is because the text embeddings lack sufficient spatial expressivity to represent the visual appearance of the subject [22, 42]. To overcome this, recent works [8, 10, 18, 29, 34, 48, 53, 63, 65] enhance the expressivity by training T2I models with large-scale datasets, but they require extensive text-image pairs for training.

To address the aforementioned challenges, one solution may be explicitly conditioning the reference images into the target denoising process. Recent subject-driven image editing techniques [4, 9, 11, 28, 31, 37] propose conditioning the reference image through the self-attention module of a denoising U-Net, which is often called key-value replacement. In the self-attention module [25], image features from preceding layers are projected into queries, keys, and values. They are then self-aggregated by an attention operation [61]. Leveraging this mechanism, previous image editing methods [4, 37] replace the keys and values from the target with those from the reference to condition the reference image into the target synthesis process. As noted in [1, 24, 55, 60], we analyze the self-attention module into two distinct paths having different roles for T2I personalization: the query-key similarities form the structure path, determining the layout of the generated images, while the values form the appearance path, infusing spatial appearance into the image layout.

As demonstrated in Figure 2, our key observation is that the replacement of target keys with reference keys in the self-attention module disrupts the structure path of the pre-trained T2I model. Specifically, an optimal key point for a query point can be unavailable in the replaced reference keys, leading to a sub-optimal matching between target queries and reference keys on the structure path. Consequently, reference appearance is then applied based on this imperfect correspondence. For this reason, prior methods incorporating key and value replacement often fail at generating personalized images with large structural differences, thus being limited to local editing. To resolve this, ViCo [22] incorporates the tuning of a subset of model weights combined with key and value replacement. However, this approach necessitates a distinct tuning process

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

(a) (b) (c) (d) (e)

Figure 2. Intuition of DreamMatcher: (a) reference image, (b) disrupted target structure path by key-value replacement [4, 9, 11, 28, 31, 37], (c) generated image by (b), (d) target structure path in pre-trained T2I model [44], and (e) generated image by DreamMatcher. For visualization, principal component analysis (PCA) [41] is applied to the structure path. Key-value replacement disrupts the target structure, yielding sub-optimal personalized results, whereas DreamMatcher better preserves the target structure, producing high-fidelity subject images aligned with target prompts.

prior to its actual usage.

In this paper, we propose a plug-in method dubbed DreamMatcher that effectively transfers reference appearance while generating diverse structures. DreamMatcher concentrates on the appearance path within the self-attention module for personalization, while leaving the structure path unchanged. However, a simple replacement of values from the target with those from the reference can lead to structure-appearance misalignment. To resolve this, we propose a matching-aware value injection leveraging semantic correspondence to align the reference appearance toward the target structure. Moreover, it is essential to isolate only the matched reference appearance to preserve other structural elements of the target, such as occluding objects or background variations. To this end, we introduce a semantic-consistent masking strategy, ensuring selective incorporation of semantically consistent reference appearances into the target structure. Combined, only the correctly aligned reference appearance is integrated into the target structure through the self-attention module at each time step. However, the estimated reference appearance in early diffusion time steps may lack the fine-grained subject details. To overcome this, we introduce a sampling guidance technique, named semantic matching guidance, to provide rich reference appearance in the middle of the target denoising process.

DreamMatcher is compatible with any existing T2I personalized models without any training or fine-tuning. We show the effectiveness of our method on three different baselines [17, 32, 44]. DreamMatcher achieves state-of-theart performance compared with existing tuning-free plug-in methods [4, 49, 68] and even a learnable method [22]. As shown in Figure 1, DreamMatcher is effective even in extreme non-rigid personalization scenarios. We further validate the robustness of our method in challenging personalization scenarios. The ablation studies confirm our design choices and emphasize the effectiveness of each component.

×(𝑡 − 1)

×(𝑇 − 𝑡)

𝜖

[Figure 29]

[Figure 30]

|Inversion| |
|---|---|
| | |

Inversion

|SelfAttention|
|---|

|CrossAttention|
|---|

Encoder

Decoder

𝑧

𝑧

𝑧

𝑧

𝑧

𝐼

𝐼

|Semantic Matching & Consistency Modeling| |
|---|---|
| | |

|Semantic Matching Guidance|
|---|

𝐹  → ,

|𝑈| |
|---|---|
| | |

𝑀

𝑧 , ̂

𝜖

[Figure 31]

Guidance

|AMA|
|---|

|CrossAttention|
|---|

Decoder

𝑧

𝑧

𝑧

𝑧

×(𝑇 − 𝑡)

×(𝑡 − 1)

𝐼

P : A < 𝑆∗> 𝑟𝑢𝑛𝑛𝑖𝑛𝑔 𝑎𝑐𝑟𝑜𝑠𝑠 𝑎 𝑚𝑒𝑎𝑑𝑜𝑤

- Figure 3. Overall architecture: Given a reference image IX, appearance matching self-attention (AMA) aligns the reference appearance

into the fixed target structure in self-attention module of pre-trained personalized model ϵθ. This is achieved by explictly leveraging reliable semantic matching from reference to target. Furthermore, semantic matching guidance enhances the fine-grained details of the subject in the generated images.

- 2. Related Work Optimization-based T2I Personalization. Given a handful of images, T2I personalization aims to generate new image variations of the given concept that are consistent with the target prompt. Earlier diffusion-based techniques [14, 17, 21, 32, 44, 62] encapsulate the given concept within the textual domain, typically represented by a specific token. Textual Inversion [17] optimizes a textual embedding and synthesizes personalized images by integrating the token with the target prompt. DreamBooth [44] proposes optimizing all parameters of the denoising U-Net based on a specific token and the class category of the subject. Several works [7, 21, 22, 32, 35, 45, 55, 64] focus on optimizing weight subsets or an additional adapter for efficient optimization and better conditioning. For example, CustomDiffusion [32] fine-tunes only the cross-attention layers in the U-Net, while ViCo [22] optimizes an additional image encoder. Despite promising results, the aforementioned approaches often fail to accurately mimic the appearance of the subject.

a noise blending method between a pre-trained diffusion model and a T2I personalized model [44]. DreamMatcher is in alignment with these methods, designed to be compatible with any off-the-shelf T2I personalized models, thereby eliminating additional fine-tuning or training.

### 3. Preliminary

- 3.1. Latent Diffusion Models Diffusion models [25, 50] generate desired data samples from Gaussian noise through a gradual denoising process. Latent diffusion models [43] perform this process in the latent space projected by an autoencoder, instead of RGB space. Specifically, an encoder maps an RGB image x0 into a latent variable z0 and a decoder then reconstructs it back to x0. In forward diffusion process, Gaussian noise is gradually added to the latent zt at each time step t to produce the noisy latent zt+1. In reverse diffusion process, the neural network ϵθ(zt,t) denoises zt to produce zt−1 with the time step t. By iteratively sampling zt−1, Gaussian noise zT is transformed into latent z0. The denoised z0 is converted back to x0 using the decoder. When the condition, e.g., text prompt P, is added, ϵθ(zt,t,P) generates latents that are aligned with the text descriptions.
- 3.2. Self-Attention in Diffusion Models Diffusion model is often based on a U-Net architecture that includes residual blocks, cross-attention modules, and selfattention modules [25, 43, 50]. The residual block processes the features from preceding layers, the cross-attention module integrates these features with the condition, e.g., text prompt, and the self-attention module aggregates image features themselves through the attention operation.

Training-based T2I Personalization. Several studies [8, 10, 18, 29, 34, 48, 53, 63, 65] have shifted their focus toward training a T2I personalized model with large textimage pairs. For instance, Taming Encoder [29], InstantBooth [48], and FastComposer [65] train an image encoder, while SuTI [10] trains a separate network. While these approaches circumvent fine-tuning issues, they necessitate extensive pre-training with a large-scale dataset.

Plug-in Subject-driven T2I Synthesis. Recent studies [4, 20, 37, 47, 49, 68] aim to achieve subject-driven T2I personalization or non-rigid editing without the need for additional fine-tuning or training. Specifically, MasaCtrl [4] leverages dual-branch pre-trained diffusion models to incorporate image features from the reference branch into the target branch. FreeU [49] proposes reweighting intermediate feature maps from a pre-trained personalized model [44], based on frequency analysis. MagicFusion [68] introduces

Specifically, the self-attention module projects the image feature at time step t into queries Qt, keys Kt, and values Vt. The resulting output from this module is defined by:

QtKtT

Vt. (1)

√

SA(Qt,Kt,Vt) = Softmax

d

Reference

Reference

#### 4.1. Appearance Matching Self-Attention

|𝑄|
|---|

|𝑄|
|---|

|𝐾|
|---|

As illustrated in Figure 4, we propose an appearance matching self-attention (AMA) which manipulates only the appearance path while retaining the pre-trained target structure path, in order to enhance subject expressivity while preserving the target prompt-directed layout.

𝐾 𝑉

𝑉

𝐹  →  𝑀

Warping

Semantic Matching & Consistency Modeling

| |𝑉|
|---|---|
| | |

⊙

Target

Target

However, naively swapping the target appearance VtY with that from the reference VtX, which reformulates Equation 1, results in structure-appearance misalignment:

|𝑄|
|---|

|𝐾|
|---|

|𝑉|
|---|

|𝑄|
|---|

|𝐾|
|---|

|𝑉|
|---|

Structure Path

Appearance Path

Structure Path

Appearance Path

QYt (KtY )T

(a) Key-Value Replacement

(b) Appearance Matching Self-Attention

SA(QYt ,KtY ,VtX) = Softmax

VtX. (2)

√

- Figure 4. Comparison between (a) key-value replacement [4, 9, 11, 28, 31, 37] and (b) appearance matching self-attention (AMA): AMA aligns the reference appearance path toward the fixed target structure path through explicit semantic matching and consistency modeling.

d

To solve this, we propose a matching-aware value injection method that leverages semantic matching to accurately align the reference appearance VtX with the fixed target structure Softmax(QYt (KtY )T/

√

d). Specifically, AMA warps the reference values VtX by the estimated semantic correspondence FtX→Y from reference to target, which is a dense displacement field [12, 38, 56, 57, 59] between semantically identical locations in both images. The warped reference values VtX→Y are formulated by:

Here, Softmax(·) is applied over the keys for each query. Qt ∈ Rh×w×d, Kt ∈ Rh×w×d, and Vt ∈ Rh×w×d are the projected matrices, where h, w, and d refer to the height, width, and channel dimensions, respectively. As analyzed in [1, 24, 55, 60], we view the self-attention module as two distinct paths: the structure and appearance paths. More specifically, the structure path is defined by the similarities Softmax(QtKtT/

VtX→Y = W(VtX;FtX→Y ), (3) where W represents the warping operation [58].

√

d), which controls the spatial arrangement of image elements. The values Vt constitute the appearance path, injecting visual attributes such as colors, textures, and shapes, to each corresponding element within the image.

In addition, it is crucial to isolate only the matched reference appearance and filter out outliers. This is because typical personalization scenarios often involve occlusions, different viewpoints, or background changes that are not present in the reference images, as shown in Figure 1. To achieve this, previous methods [4, 22] use a foreground mask Mt to focus only on the subject foreground and handle background variations. Mt is obtained from the averaged cross-attention map for the subject text prompt (e.g., ⟨S∗⟩). With these considerations, Equation 3 can be reformulated as follows:

### 4. Method

Given a set of n reference images X = {InX}Nn1 , conventional methods [17, 32, 44] personalize the T2I mod-

els ϵθ(·) with a specific text prompt for the subject (e.g., ⟨S∗⟩). In inference, ϵθ(·) can generate novel scenes from random noises through iterative denoising processes with the subject aligned by the target prompt (e.g., A ⟨S∗⟩ in the jungle). However, they often fail to accurately mimic the subject appearance because text embeddings lack the spatial expressivity to represent the visual attributes of the subject [22, 42]. In this paper, with a set of reference images X and a target text prompt P, we aim to enhance the subject appearance in the personalized image IY , while preserving the detailed target structure directed by the prompt P. DreamMatcher comprises a reference-target dual-branch framework. IX is inverted to zTX via DDIM inversion [50] and then reconstructed to IˆX, while IY is generated from a random Gaussian noise zTY guided by P. At each time step, the self-attention module from the reference branch projects image features into queries QXt , KtX, and VtX, while the target branch produces QYt , KtY , and VtY . The reference appearance VtX is then transferred to the target denoising U-Net through its self-attention module. The overall architecture of DreamMatcher is illustrated in Figure 3.

VtW = VtX→Y ⊙ Mt + VtY ⊙ (1 − Mt), (4) where ⊙ represents Hadamard product [27].

AMA then implants VtW into the fixed target structure path through the self-attention module. Equation 2 is reformulated as:

QYt (KtY )T

AMA(QYt ,KtY ,VtW) = Softmax

VtW.

√

d

(5)

In our framework, we find semantic correspondence between reference and target, aligning with standard semantic matching workflows [12, 13, 26, 38, 56, 57]. Figure 5 provides a detailed schematic of the proposed matching process. In the following, we will explain the process in detail. Feature Extraction. Classical matching pipelines [12, 13, 26, 38, 56, 57] contain pre-trained feature extractors [6, 23,

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

ReferenceTargetReferenceTarget

|CycleConsistency| |
|---|---|
| | |

|PCA &Interpolation| |
|---|---|
| | |
| | |

|ChannelConcatenation| |
|---|---|
| | |
| | |

|CostComputation| |
|---|---|
| |(𝐻,|

|FlowComputation| |
|---|---|
| | |
| | |

𝑧

ChannelConcatenation

PCA &Interpolation

FlowComputation

CycleConsistency

CostComputation

[Figure 38]

[Figure 39]

𝜓

𝐹  → 

(𝐻,𝑊,𝐷)

(𝐻,𝑊,2)

[Figure 40]

𝐶 ,𝑊,𝐻,𝑊) 𝑈

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

(𝐻,𝑊)

𝜓

𝐹  → 

𝑧

(𝐻,𝑊,𝐷)

(𝐻,𝑊,2)

- Figure 5. Semantic matching and consistency modeling: We leverage internal diffusion features at each time step to find se-

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

mantic matching FtX→Y between reference and target. Additionally, we compute the confidence map of the predicted matches Ut through cycle-consistency.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

40] to obtain feature descriptors ψX and ψY from image pairs IX and IY . However, finding good features tailored for T2I personalization is not trivial due to the noisy nature of estimated target images in reverse diffusion process, requiring additional fine-tuning of the existing feature extractors. To address this, we focus on the diffusion feature space [54, 67] in the pre-trained T2I model itself to find a semantic matching tailored for T2I personalization.

Denoising

Figure 6. Diffusion feature visualization: Upper displays intermediate estimated images of reference and target, with the target generated by DreamBooth [44] using the prompt A ⟨S∗⟩ on the beach. Lower visualizes the three principal components of intermediate diffusion features. The similar semantics share similar colors.

Let ϵθ,l(·,t + 1) denote the output of the l-th decoder layer of the denoising U-Net [25] ϵθ at time step t+1. Given the latent zt+1 with time step t + 1 and text prompt P as inputs, we extract the feature descriptor ψt+1,l from the l-th layer of the U-Net decoder. The process is given by:

descriptors. This is formulated as:

ψtX+1(i) · ψtY+1(j) ∥ψtX+1(i)∥∥ψtY+1(j)∥

, (7)

Ct+1(i,j) =

###### ψt+1,l = ϵθ,l(zt+1, t + 1, P), (6)

where i,j ∈ [0,H) × [0,W), and ∥ · ∥ denotes l-2 normalization.

where we obtain ψtX+1,l and ψtY+1,l from ztX+1 and ztY+1, respectively. For brevity, we will omit l in the following dis-

Subsequently, we derive the dense displacement field from the reference to the target at time step t, denoted as FtX→Y ∈ RH×W×2, using the argmax operation [12] on the matching cost Ct+1. Figure 7(c) shows the warped reference image obtained using the predicted correspondence FtX→Y between ψtX+1 and ψtY+1 in the middle of the generation process. This demonstrates that the correspondence is established reliably in reverse diffusion process, even in intricate non-rigid target contexts that include large displacements, occlusions, and novel-view synthesis.

cussion.

To explore the semantic relationship within the diffusion feature space between reference and target, Figure 6 visualizes the relation between ψtX+1 and ψtY+1 at different time steps using principal component analysis (PCA) [41]. We observe that the foreground subjects share similar semantics, even they have different appearances, as the target image from the pre-trained personalized model often lacks subject expressivity. This observation inspires us to leverage the internal diffusion features to establish semantic matching between estimated reference and target at each time step of sampling phase.

#### 4.2. Consistency Modeling

As depicted in Figure 7(d), the forground mask Mt is insufficient to address occlusions and background clutters, (e.g., a chef outfit or a bouquet of flowers), as these are challenging to distinguish within the cross-attention module.

Based on this, we derive ψt+1 ∈ RH×W×D by combining PCA features from different layers using channel concatenation, where D is the concatenated channel dimension. Detailed analysis and implementation on feature extraction is provided in Appendix E.1.

To compensate for this, we introduce a confidence mask Ut to discard erroneous correspondences, thus preserving detailed target structure. Specifically, we enforce a cycle consistency constraint [30], simply rejecting any correspondence greater than the threshold we set. In other words, we only accept correspondences where a target location x remains consistent when a matched reference location, obtained by FtY →X, is re-warped using FtX→Y . We empirically set the threshold proportional to the target foreground

Flow Computation. Following conventional methods [12, 26, 56, 57, 59], we build the matching cost by calculating the pairwise cosine similarity between feature descriptors for both the reference and target images. For given ψtX+1 and ψtY+1 at time step t+1, the matching cost Ct+1 is computed by taking dot products between all positions in the feature

looking out from a window of a brick house

where λg is a hyperparameter that modulates the guidance strength, and σt represents the noise schedule parameter at time step t.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

OcclusionDisplacement Novel-view

Large

We design the guidance function g using z0X from DDIM inversion [50], which encapsulates detailed subject representation at the final reverse step t = 0. At each time step t, z0X is transformed to align with the target structure through FtX→Y , as follows:

in a chef outfit

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

in front of a bouquet of flowers

z0X,t→Y = W(z0X;FtX→Y ). (10)

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Synthesis

The guidance function gt at time step t is then defined as the pixel-wise difference between the aligned z0X,t→Y and the target latent zˆ0Y,t which is calculated by reparametrization trick [50], taking into account the semantic-consistent mask Mt′ :

(a) (b) (c) (d) (e)

- Figure 7. Correspondence visualization: (a) Reference image. (b) Estimated target image from DreamBooth [44] at 50% of the reverse diffusion process. (c) Warped reference image based on

predicted correspondence FtX→Y . (d) Warped reference image combined with foreground mask Mt. (e) Warped reference image combined with both Mt and confidence mask Ut.

1 |M′

z0X,t→Y (i) − zˆ0Y,t(i) , (11)

gt =

t| i∈M

′ t

where ∥ · ∥ denotes a l-2 norm.

area. This is formulated by:

Note that our approach differs from existing methods [2, 16, 37] that provide coarse appearance guidance by calculating the average feature difference between foregrounds. Instead, we leverage confidence-aware semantic correspondence to offer more precise and pixel-wise control.

1, if W FtY →X;FtX→Y (x) < γλc, 0, otherwise,

Ut(x) =

(8) where ∥·∥ denotes a l-2 norm, and W represents the warping operation [58]. FtY →X indicates the reverse flow field of its forward counterpart, FtX→Y . γ is a scaling factor designed to be proportional to foreground area, and λc is a hyperparameter. More details are available in Appendix E.2.

### 5. Experiments

#### 5.1. Experimental Settings

Dataset. ViCo [22] gathered an image-prompt dataset from previous works [17, 32, 44], comprising 16 concepts and 31 prompts. We adhered to the ViCo dataset and evaluation settings, testing 8 samples per concept and prompt, for a total of 3,969 images. To further evaluate the robustness of our method in complex non-rigid personalization scenarios, we created a prompt dataset divided into three categories: large displacements, occlusions, and novel-view synthesis. This dataset includes 10 prompts for large displacements and occlusions, and 4 for novel-view synthesis, all created using ChatGPT [39]. The detailed procedure and the prompt list are in the Appendix B.

Finally, we define a semantic-consistent mask Mt′ by combining Mt and Ut using the Hadamard product [27], so that Mt coarsely captures the foreground subject, while Ut finely filters out unreliable matches and preserves the finegrained target context. As shown in Figure 7(e), our network selectively incorporates only the confident matches, effectively addressing intricate non-rigid scenarios.

We now apply a confidence-aware modification to appearance matching self-attention in Equation 4, by replacing Mt with Mt′.

#### 4.3. Semantic Matching Guidance

Baseline and Comparison. DreamMatcher is designed to be compatible with any T2I personalized models. We implemented our method using three baselines: Textual Inversion [17], DreamBooth [44], and CustomDiffusion [32]. We benchmarked DreamMatcher against previous tuning-free plug-in models, FreeU [49] and MagicFusion [68], and also against the optimization-based model, ViCo [22]. Note that additional experiments, including DreamMatcher on Stable Diffusion or DreamMatcher for multiple subject personalization, are provided in Appendix E.

Our method uses intermediate reference values VtX at each time step. However, we observe that in early time steps, these noisy values may lack fine-grained subject details, resulting in suboptimal results. To overcome this, we further introduce a sampling guidance technique, named semantic matching guidance, to provide rich reference semantics in the middle of the target denoising process.

In terms of the score-based generative models [51, 52], the guidance function g steers the target images towards higher likelihoods. The updated direction ϵˆt at time step t is defined as follows [16]:

Evaluation Metric. Following previous studies [17, 22, 32, 44], we evaluated subject and prompt fidelity. For subject fidelity, we adopted the CLIP [42] and DINO [5] image

g(zt,t,P), (9)

ϵˆt = ϵθ(zt,t,P) − λgσt∇zt

Reference Textual Inv. DreamMatcher DreamBooth DreamMatcher CustomDiff. DreamMatcher

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

floating on top of water among the skyscrapers in New York city in the jungle

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

wearing a rainbow scarf in a luxurious interior living room in a chef outfit

- Figure 8. Qualitative comparison with baselines: We compare DreamMatcher with three different baselines, Textual Inversion [17], DreamBooth [44], and CustomDiffusion [32].

Method IDINO ↑ ICLIP ↑ TCLIP ↑ Textual Inversion [17] 0.529 0.762 0.220 DreamMatcher 0.588 (+11.2%) 0.778 (+2.1%) 0.217 (-1.4%) DreamBooth [44] 0.638 0.808 0.237 DreamMatcher 0.680 (+6.6%) 0.821 (+1.6%) 0.231 (-2.5%) CustomDiffusion [32] 0.667 0.810 0.218 DreamMatcher 0.700 (+4.9%) 0.821 (+1.4%) 0.223 (+2.3%)

Method IDINO ↑ ICLIP ↑ TCLIP ↑

MagicFusion [68] 0.632 0.811 0.233 FreeU [49] 0.632 0.806 0.236

DreamMatcher 0.680 0.821 0.231

- Table 2. Quantitative comparison with tuning-free methods. For this comparison, we used DreamBooth [44] as our baseline.

Method IDINO ↑ ICLIP ↑ TCLIP ↑

MagicFusion [68] 0.622 0.814 0.235 FreeU [49] 0.611 0.803 0.242

DreamMatcher 0.655 0.818 0.239

- Table 3. Quantitative comparison in challenging dataset. For this comparison, we used DreamBooth [44] as our baseline.

Method IDINO ↑ ICLIP ↑ TCLIP ↑ ViCo [22] 0.643 0.816 0.228 DreamMatcher 0.700 0.821 0.223

- Table 4. Comparison with optimization-based method. For this comparison, we used CustomDiffusion [32] as our baseline.

Table 1. Quantitative comparison with different baselines.

similarity, denoted as ICLIP and IDINO, respectively. For prompt fidelity, we adopted the CLIP image-text similarity TCLIP, comparing visual features of generated images to textual features of their prompts, excluding placeholders. Further details on evaluation metrics are in the Appendix D.1.

User Study. We conducted a user study comparing DreamMatcher to previous works [22, 49, 68]. Participants evaluated the generated images from different methods based on subject and prompt fidelity. 45 users responded to 32 comparative questions, totaling 1440 responses. Samples were chosen randomly from a large, unbiased pool. Additional details on the user study are in Appendix D.2.

Comparison with Plug-in Models. We compared DreamMatcher against previous tuning-free plug-in methods, FreeU [49] and MagicFusion [68]. Both methods demonstrated their effectiveness when plugged into DreamBooth [44]. For a fair comparison, we evaluated DreamMatcher using DreamBooth as a baseline. As shown in Table 2 and Figure 9, DreamMatcher notably outperforms these methods in subject fidelity, maintaining comparable prompt fidelity. The effectiveness of our method is also evident in Table 3, displaying quantitative results in challenging non-rigid personalization scenarios. This highlights the importance of semantic matching for robust performance in complex real-world personalization applications.

- 5.2. Results Comparison with Baselines. Table 1 and Figure 8 summarize the quantitative and qualitative comparisons with different baselines. The baselines [17, 32, 44] often lose key visual attributes of the subject such as colors, texture, or shape due to the limited expressivity of text embeddings. In contrast, DreamMatcher significantly outperforms these baselines by a large margin in subject fidelity IDINO and ICLIP, while effectively preserving prompt fidelity TCLIP. As noted in [22, 44], we want to highlight that IDINO better reflects subject expressivity, as it is trained in a self-supervised fashion, thus distinguishing the difference among objects in the same category. Additionally, we wish to note that better prompt fidelity does not always reflect

Comparison with Optimization-based Models. We further evaluated DreamMatcher against the optimizationbased model, ViCo [22], which fine-tunes an image adapter with 51.3M parameters. For a balanced comparison, we compared ViCo with DreamMatcher combined with CustomDiffusion [32], configured with a similar count of train-

in TCLIP. TCLIP is reported to imperfectly capture textimage alignment and has been replaced by the VQA-based evaluation [19, 66], implying its slight performance drop is negligible. More results are provided in Appendix F.1.

Reference ViCo DreamBooth MasaCtrl FreeU MagicFusion DreamMatcher

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

diving into a pool

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

floating in a pond

- Figure 9. Qualitative comparison with previous works [4, 22, 44, 49, 68]: For this comparison, DreamBooth [44] was used as the baseline of MasaCtrl, FreeU, MagicFusion, and DreamMatcher.

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Methods

Methods

0.17

ViCo

0.15

ViCo

0.24

MagicFu.

0.25

MagicFu.

(a) (b) (c) (d) (e)

0.24

FreeU

0.24

FreeU

Figure 11. Component analysis: (a) reference image, (b) generated image by DreamBooth [44], (c) with proposed semantic matching, (d) further combined with semantic-consistent mask, and (e) further combined with semantic matching guidance.

0.35

Ours

0.36

Ours

Ranking Scores (Subject Fidelity)

Ranking Scores (Prompt Fidelity)

Figure 10. User study.

Component IDINO ↑ ICLIP ↑ TCLIP ↑

able parameters (57.1M). Table 4 shows DreamMatcher notably surpasses ViCo in all subject fidelity metrics, without requiring extra fine-tuning. Figure 9 provides the qualitative comparison. More results are provided in Appendix F.2.

- (I) Baseline (DreamBooth [44]) 0.638 0.808 0.237

- (II) (I) + Key-Value Replacement (MasaCtrl [4]) 0.728 0.854 0.201

- (III) (I) + Semantic Matching 0.683 0.830 0.201

- (IV) (III) + Semantic-Consistent Mask (AMA) 0.676 0.818 0.232

- (V) (IV) + Semantic Matching Guid. (Ours) 0.680 0.821 0.231

User Study. We also present the user study results in Figure 10, where DreamMatcher significantly surpasses all other methods in both subject and prompt fidelity. Further details are provided in Appendix D.2.

Table 5. Component analysis. For this analysis, we used DreamBooth [44] for the baseline.

### 6. Conclusion

We present DreamMatcher, a tuning-free plug-in for text-toimage (T2I) personalization. DreamMatcher enhances appearance resemblance in personalized images by providing semantically aligned visual conditions, leveraging the generative capabilities of the self-attention module within pretrained T2I personalized models. DreamMatcher pioneers the significance of semantically aligned visual conditioning in personalization, offering an effective solution within the attention framework. Experiments show that DreamMatcher enhances the personalization capabilities of existing T2I models, outperforming previous tuning-free plugins, even in complex scenarios.

#### 5.3. Ablation Study

In Figure 11 and Table 5, we demonstrate the effectiveness of each component in our framework. (b) and (I) present the results of the baseline, while (II) shows the results of keyvalue replacement, which fails to preserve the target structure and generates a static subject image. (c) and (III) display AMA using predicted correspondence, enhancing subject fidelity compared to (b) and (I), but drastically reducing prompt fidelity, as it could not filter out unreliable matches. This is addressed in (d) and (IV), which highlight the effectiveness of the semantic-consistent mask in significantly improving prompt fidelity, up to the baseline (I). Finally, the comparison between (d) and (e) demonstrate that semanticmatching guidance improves subject expressivity with minimal sacrifice in target structure, which is further evidenced by (V). More analyses, including a user study comparing DreamMatcher and MasaCtrl, are in Appendix E.

### 7. Acknowledgements

This research was supported by the MSIT, Korea (IITP2024-2020-0-01819, ICT Creative Consilience Program, No.2021-0-02068, Artificial Intelligence Innovation Hub).

### References

- [1] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2, 4
- [2] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 843–852,

2023. 6

- [3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. A.6
- [4] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465, 2023. 2, 3, 4, 8, A.4, A.5, A.6, A.10, A.15, A.16
- [5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 6, A.2
- [6] Ken Chatfield, Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Return of the devil in the details: Delving deep into convolutional nets. arXiv preprint arXiv:1405.3531, 2014. 4
- [7] Hong Chen, Yipeng Zhang, Xin Wang, Xuguang Duan, Yuwei Zhou, and Wenwu Zhu. Disenbooth: Disentangled parameter-efficient tuning for subject-driven text-to-image generation. arXiv preprint arXiv:2305.03374, 2023. 3
- [8] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. Photoverse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023. 2, 3, A.5
- [9] Songyan Chen and Jiancheng Huang. Fec: Three finetuningfree methods to enhance consistency for real image editing. arXiv preprint arXiv:2309.14934, 2023. 2, 4, A.4
- [10] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Rui, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. arXiv preprint arXiv:2304.00186, 2023. 2, 3, A.5
- [11] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023. 2, 4, A.4
- [12] Seokju Cho, Sunghwan Hong, Sangryul Jeon, Yunsung Lee, Kwanghoon Sohn, and Seungryong Kim. Cats: Cost aggregation transformers for visual correspondence. Advances in Neural Information Processing Systems, 34:9011–9023,

2021. 4, 5

- [13] Seokju Cho, Sunghwan Hong, and Seungryong Kim. Cats++: Boosting cost aggregation with convolutions and

- transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. 4
- [14] Ziyi Dong, Pengxu Wei, and Liang Lin. Dreamartist: Towards controllable one-shot text-to-image generation via contrastive prompt-tuning. arXiv preprint arXiv:2211.11337, 2022. 2, 3
- [15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. A.3
- [16] Dave Epstein, Allan Jabri, Ben Poole, Alexei A Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. arXiv preprint arXiv:2306.00986,

2023. 6

- [17] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2, 3, 4, 6, 7, A.2, A.3, A.5, A.6, A.8, A.13, A.14
- [18] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Designing an encoder for fast personalization of text-to-image models. arXiv preprint arXiv:2302.12228, 2023. 2, 3, A.5
- [19] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. arXiv preprint arXiv:2310.11513, 2023. 7
- [20] Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, HyunJoon Jung, et al. Photoswap: Personalized subject swapping in images. arXiv preprint arXiv:2305.18286, 2023. 3
- [21] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023. 2, 3
- [22] Shaozhe Hao, Kai Han, Shihao Zhao, and Kwan-Yee K Wong. Vico: Detail-preserving visual condition for personalized text-to-image generation. arXiv preprint arXiv:2306.00971, 2023. 2, 3, 4, 6, 7, 8, A.2, A.3, A.6, A.9, A.15, A.16
- [23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 4
- [24] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2, 4, A.6
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3, 5
- [26] Sunghwan Hong, Jisu Nam, Seokju Cho, Susung Hong, Sangryul Jeon, Dongbo Min, and Seungryong Kim. Neural matching fields: Implicit representation of matching fields

- for visual correspondence. Advances in Neural Information Processing Systems, 35:13512–13526, 2022. 4, 5
- [27] Roger A Horn. The hadamard product. In Proc. Symp. Appl. Math, pages 87–169, 1990. 4, 6
- [28] Jiancheng Huang, Yifan Liu, Jin Qin, and Shifeng Chen. Kv inversion: Kv embeddings learning for text-conditioned real image action editing. arXiv preprint arXiv:2309.16608,

2023. 2, 4, A.4

- [29] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 2, 3, A.5
- [30] Wei Jiang, Eduard Trulls, Jan Hosang, Andrea Tagliasacchi, and Kwang Moo Yi. Cotr: Correspondence transformer for matching across images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6207– 6217, 2021. 5
- [31] Anant Khandelwal. Infusion: Inject and attention fusion for multi concept zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3017–3026, 2023. 2, 4, A.4
- [32] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 2, 3, 4, 6, 7, A.2, A.3, A.5, A.6, A.8, A.13, A.14
- [33] Jason Lee, Kyunghyun Cho, and Douwe Kiela. Countering language drift via visual grounding. arXiv preprint arXiv:1909.04499, 2019. A.2
- [34] Dongxu Li, Junnan Li, and Steven CH Hoi. Blipdiffusion: Pre-trained subject representation for controllable text-to-image generation and editing. arXiv preprint arXiv:2305.14720, 2023. 2, 3, A.5
- [35] Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable image synthesis with multiple subjects. arXiv preprint arXiv:2305.19327, 2023. 3
- [36] Yuchen Lu, Soumye Singhal, Florian Strub, Aaron Courville, and Olivier Pietquin. Countering language drift with seeded iterated learning. In International Conference on Machine Learning, pages 6437–6447. PMLR, 2020. A.2
- [37] Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421,

2023. 2, 3, 4, 6, A.3, A.4

- [38] Jisu Nam, Gyuseong Lee, Sunwoo Kim, Hyeonsu Kim, Hyoungwon Cho, Seyeon Kim, and Seungryong Kim. Diffmatch: Diffusion model for dense matching. arXiv preprint arXiv:2305.19094, 2023. 4
- [39] OpenAI. Gpt-4 technical report, 2023. 6, A.2
- [40] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 5

- [41] Karl Pearson. Liii. on lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin philosophical magazine and journal of science, 2(11):559– 572, 1901. 2, 5, A.3, A.5
- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 4, 6, A.2
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 2, 3, 4, 5, 6, 7, 8, A.2, A.3, A.4, A.5, A.6, A.8, A.13, A.14, A.15, A.16
- [45] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 3
- [46] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. A.2
- [47] Junyoung Seo, Gyuseong Lee, Seokju Cho, Jiyoung Lee, and Seungryong Kim. Midms: Matching interleaved diffusion models for exemplar-based image translation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2191–2199, 2023. 3
- [48] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023. 2, 3, A.5
- [49] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. arXiv preprint arXiv:2309.11497, 2023. 2, 3, 6, 7, 8, A.2, A.3, A.6, A.9, A.15, A.16
- [50] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3, 4, 6, A.2
- [51] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 6
- [52] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 6
- [53] Yu-Chuan Su, Kelvin CK Chan, Yandong Li, Yang Zhao, Han Zhang, Boqing Gong, Huisheng Wang, and Xuhui Jia.

- Identity encoder for personalized diffusion. arXiv preprint arXiv:2304.07429, 2023. 2, 3, A.5
- [54] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. arXiv preprint arXiv:2306.03881,

2023. 5, A.3

- [55] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 2, 3, 4
- [56] Prune Truong, Martin Danelljan, Luc V Gool, and Radu Timofte. Gocor: Bringing globally optimized correspondence volumes into your neural network. Advances in Neural Information Processing Systems, 33:14278–14290, 2020. 4, 5
- [57] Prune Truong, Martin Danelljan, and Radu Timofte. Glunet: Global-local universal network for dense flow and correspondences. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6258–6268,

2020. 4, 5

- [58] Prune Truong, Martin Danelljan, Fisher Yu, and Luc Van Gool. Warp consistency for unsupervised learning of dense correspondences. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10346– 10356, 2021. 4, 6
- [59] Prune Truong, Martin Danelljan, Radu Timofte, and Luc Van Gool. Pdc-net+: Enhanced probabilistic dense correspondence network. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 4, 5
- [60] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023. 2, 4, A.6
- [61] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2
- [62] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023. 2, 3
- [63] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 2, 3, A.5
- [64] Chendong Xiang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. A closer look at parameter-efficient tuning in diffusion models. arXiv preprint arXiv:2303.18181, 2023. 3
- [65] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023. 2, 3, A.5
- [66] Michal Yarom, Yonatan Bitton, Soravit Changpinyo, Roee Aharoni, Jonathan Herzig, Oran Lang, Eran Ofek, and Idan Szpektor. What you see is what you read? im-

- proving text-image alignment evaluation. arXiv preprint arXiv:2305.10400, 2023. 7
- [67] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. arXiv preprint arXiv:2305.15347, 2023. 5, A.3
- [68] Jing Zhao, Heliang Zheng, Chaoyue Wang, Long Lan, and Wenjing Yang. Magicfusion: Boosting text-to-image generation performance by fusing diffusion models. arXiv preprint arXiv:2303.13126, 2023. 2, 3, 6, 7, 8, A.2, A.3, A.6, A.9, A.15, A.16

### A. Implementation Details

For all experiments, we used an NVIDIA GeForce RTX 3090 GPU and a DDIM sampler [50], setting the total sampling time step to T = 50. We empirically set the time steps to t ∈ [4,50) for performing both our appearance matching self-attention and semantic matching guidance. We converted all self-attention modules in every decoder layer l ∈ [1,4) to the proposed appearance matching selfattention. We chose λc = 0.4 and λg = 75 for evaluation on the ViCo [22] dataset, and λc = 0.4 and λg = 50 for evaluation on the proposed challenging prompt list.

### B. Dataset

Prior works [17, 32, 44] in Text-to-Image (T2I) personalization have used different datasets for evaluation. To ensure a fair and unbiased evaluation, ViCo [22] collected an image dataset from these works [17, 32, 44], comprising 16 unique concepts, which include 6 toys, 5 live animals, 2 types of accessories, 2 types of containers, and 1 building. For the prompts, ViCo gathered 31 prompts for 11 non-live objects and another 31 prompts for 5 live objects. These were modified from the original DreamBooth [44] prompts to evaluate the expressiveness of the objects in more complex textual contexts. For a fair comparison, in this paper, we followed the ViCo dataset and its evaluation settings, producing 8 samples for each object and prompt, totaling 3,969 images.

Our goal is to achieve semantically-consistent T2I personalization in complex non-rigid scenarios. To assess the robustness of our method in intricate settings, we created a prompt dataset using ChatGPT [39], which is categorized into three parts: large displacements, occlusions, and novelview synthesis. The dataset comprises 10 prompts each for large displacements and occlusions, and 4 for novel-view synthesis, separately for live and non-live objects. Specifically, we define the text-to-image diffusion personalization task, provide an example prompt list from ViCo, and highlight the necessity of a challenging prompt list aligned with the objectives of our research. We then asked ChatGPT to create distinct prompt lists for each category. The resulting prompt list, tailored for complex non-rigid personalization scenarios, is detailed in Figure A.12.

### C. Baseline and Comparison C.1. Baseline

DreamMatcher is designed to be compatible with any T2I personalized model. We implemented our method using three baselines: Textual Inversion [17], DreamBooth [44], and CustomDiffusion [32].

Textual Inversion [17] encapsulates a given subject into 768-dimensional textual embeddings derived from the special token ⟨S∗⟩. Using a few reference images, this is

achieved by training the textual embeddings while keeping the T2I diffusion model frozen. During inference, the model can generate novel renditions of the subject by manipulating the target prompt with ⟨S∗⟩. DreamBooth [44] extends this approach by further fine-tuning a T2I diffusion model with a unique identifier and the class name of the subject (e.g., A [V] cat). However, fine-tuning all parameters can lead to a language shift problem [33, 36]. To address this, DreamBooth proposes a class-specific prior preservation loss, which trains the model with diverse samples generated by pre-trained T2I models using the category name as a prompt (e.g., A cat). Lastly, CustomDiffusion [32] demonstrates that fine-tuning only a subset of parameters, specifically the cross-attention projection layers, is efficient for learning new concepts. Similar to DreamBooth, this is implemented by using a text prompt that combines a unique instance with a general category, and it also includes a regularization dataset from the large-scale open image-text dataset [46]. Despite promising results, the aforementioned approaches frequently struggle to accurately mimic the appearance of the subject, including colors, textures, and shapes. To address this, we propose a tuning-free plug-in method that significantly enhances the reference appearance while preserving the diverse structure from target prompts.

#### C.2. Comparision

We benchmarked DreamMatcher against previous tuningfree plug-in models, FreeU [49] and MagicFusion [68], and also against the optimization-based model, ViCo [22].

The key insight of FreeU [49] is that the main backbone of the denoising U-Net contributes to low-frequency semantics, while its skip connections focus on high-frequency details. Leveraging this observation, FreeU proposes a frequency-aware reweighting technique for these two distinct features, and demonstrates improved generation quality when integrated into DreamBooth [44]. MagicFusion [68] introduces a saliency-aware noise blending method, which involves combining the predicted noises from two distinct pre-trained diffusion models. MagicFusion demonstrates its effectiveness in T2I personalization when integrating a personalized model, DreamBooth [44], with a general T2I diffusion model. ViCo [22] optimizes an additional image adapter designed with the concept of key-value replacement.

### D. Evaluation

#### D.1. Evaluation Metrics

For evaluation, we focused on two primary aspects: subject fidelity and prompt fidelity. For subject fidelity, following prior studies [17, 22, 32, 44], we adopted the CLIP [42] and DINO [5] image similarity metrics, denoted as ICLIP and IDINO, respectively. Note that IDINO is our preferred

𝑡 ∈ [4,50) 𝑡 ∈ [8,50) 𝑡 ∈ [12,50)

𝑙 = 1 𝑙 = 2 𝑙 = 3 𝑙 = 4

Reference

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

ReferenceTarget

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

- (a) AMA on different time steps, with fixed layers 𝑙 ∈ [1,4)
- (b) AMA on different layers, with fixed time steps 𝑡 ∈ [4,50)

Baseline

𝑙 ∈ [1,4) 𝑙 ∈ [2,4) 𝑙 ∈ [3,4)

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Figure A.1. Diffusion feature visualization at different decoder layers: The left side displays intermediate estimated reference and target images at 50% of the reverse diffusion process. The target is generated by DreamBooth [44] using the prompt A ⟨S∗⟩ on the beach. The right side visualizes the top three principal components of diffusion feature descriptors from different decoder layers l. Semantically similar regions share similar colors.

Figure A.2. Ablating AMA on different time steps and layers: The left section shows a reference image and a target image generated by the baseline [44]. The right section displays the improved target image generated by appearance matching self-attention on (a) different time steps and (b) different decoder layers. For this ablation study, we do not use semantic matching guidance.

metric for evaluating subject expressivity. As mentioned in [22, 44], DINO is trained in a self-supervised manner to distinguish objects within the same category, so that it is more suitable for evaluating different methods that aim to mimic the visual attributes of the same subject. For prompt fidelity, following [17, 22, 32, 44], we adopted the imagetext similarity metric TCLIP, comparing CLIP visual features of the generated images to CLIP textual features of the corresponding text prompts, excluding placeholders. Following previous works [17, 22, 32, 44], we used ViT-B/32 [15] and ViT-S/16 [15] for CLIP and DINO, respectively.

diate feature descriptors of the estimated reference image and target image from DreamBooth [44], at 50% of the reverse diffusion process. Our primary insight is that earlier layers capture high-level semantics, while later layers focus on finer details of the generated images. Specifically, l = 1 captures overly high-level and low-resolution semantics, failing to provide sufficient semantics for finding correspondence. Conversely, l = 4 focuses on too fine-grained details, making it difficult to find semantically-consistent regions between features. In contrast, l = 2 and l = 3 strike a balance, focusing on sufficient semantical and structural information to facilitate semantic matching. Based on this analysis, we use concatenated feature descriptors from decoder layers l ∈ [2,3], resulting in ψt ∈ RH×W×1920. We then apply PCA to these feature descriptors, which results in ψt ∈ RH×W×256 to enhance matching accuracy and reduce memory consumption. The diffusion feature visualization across different time steps is presented in Figure 6 in our main paper.

- D.2. User study

An example question of the user study is provided in Figure A.13. We conducted a paired human preference study about subject and prompt fidelity, comparing DreamMatcher to previous works [22, 49, 68]. The results are summarized in Figure 10 in the main paper. For subject fidelity, participants were presented with a reference image and generated images from different methods, and were asked which better represents the subject in the reference. For prompt fidelity, they were shown the generated images from different works alongside the corresponding text prompt, and were asked which aligns more with the given prompt. 45 users responded to 32 comparative questions, resulting in a total of 1440 responses. We distributed two different questionnaires, with 23 users responding to one and 22 users to the other. Note that samples were chosen randomly from a large, unbiased pool.

- E. Analysis

Note that our approach differs from prior works [37, 54, 67], which select a specific time step and inject the corresponding noise into clean RGB images before passing them through the pre-trained diffusion model. In contrast, we utilize diffusion features from each time step of the reverse diffusion process to find semantic matching during each step of the personalization procedure.

AMA on different time steps and layers. We ablate starting time steps and decoder layers in relation to the proposed appearance matching self-attention (AMA) module. Figure A.2 summarizes the results. Interestingly, we observe that applying AMA at earlier time steps and decoder layers effectively corrects the overall appearance of the subject, including shapes, textures, and colors. In contrast, AMA applied at later time steps and layers tends to more closely preserve the appearance of the subject as in the baseline. Note that injecting AMA at every time step yields sub-optimal re-

- E.1. Appearance Matching Self-Attention

Feature extraction. Figure A.1 visualizes PCA [41] results on feature descriptors extracted from different decoder layers. Note that, for this analysis, we do not apply any of our proposed techniques. PCA is applied to the interme-

Reference diving into a pool

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

DreamBooth MasaCtrl DreamMatcher

- Figure A.3. Relation between λc and personalization fidelity. In this ablation study, we evaluate our method using the proposed challenging prompt list.

[Figure 129]

- Figure A.4. Relation between λg and personalization fidelity. In this ablation study, we evaluate our method using the proposed challenging prompt list.

𝐼 0.655 0.751 0.726 𝐼 0.819 0.896 0.852 𝑇 0.239 0.205 0.238

- Figure A.5. DreamBooth vs. MasaCtrl vs. DreamMatcher.

Subject Fidelity

8%

| |
|---|

MasaCtrl

| |
|---|

DreamMatcher

33% 100% Win Rate

Prompt Fidelity

- Figure A.6. User study: In this study, DreamBooth [44] is used as the baseline for both MasaCtrl and DreamMatcher.

sults, as the baselines prior to time step 4 have not yet constructed the target image layout. Based on this analysis, we converted the self-attention module in the pre-trained U-Net into the appearance matching self-attention for t ∈ [4,50) and l ∈ [1,4) in all our evaluations.

#### E.4. Key-Value Replacement vs. DreamMatcher

MasaCtrl [4] introduced a key-value replacement technique for local editing tasks. Several subsequent works [4, 9, 11, 28, 31, 37] have adopted and further developed this framework. As shown in Figure A.5, which provides a qualitative comparison of DreamMatcher with MasaCtrl, the key-value replacement is prone to producing subject-centric images, often having poses similar to those of the subject in the reference image. This tendency arises because key-value replacement disrupts the target structure from the pre-trained self-attention module and relies on sub-optimal matching between target keys and reference queries. Furthermore, this technique does not consider the uncertainty of predicted matches, which leads to the injection of irrelevant elements from the reference image into the changed background or into newly emergent objects that are produced by the target prompts.

#### E.2. Consistency Modeling

- In Figure A.3, we show the quantitative relationship between the cycle-consistency hyperparameter λc and personalization fidelity. As we first introduce λc, prompt fidelity TCLIP drastically improves, demonstrating that the confidence mask effectively filters out erroneous matches, allowing the model to preserve the detailed target structure. Subsequently, higher λc values inject more reference appearance into the target structure, increasing IDINO and ICLIP, but slightly sacrificing prompt fidelity TCLIP. This indicates that users can control the extent of reference appearance and

target structure preservation by adjusting λc. The pseudo code for overall AMA is available in Algorithm 1.

E.3. Semantic Matching Guidance

- In Figure A.4, we display the quantitative relationship between semantic matching guidance λg and personalization fidelity. Increasing λg enhances subject fidelity IDINO and ICLIP, by directing the generated target zˆ0Y,t closer to the

In contrast, DreamMatcher preserves the fixed target structure and accurately aligns the reference appearance by explicitly leveraging semantic matching. Our method also takes into account the uncertainty of predicted matches, thereby filtering out erroneous matches and maintaining newly introduced image elements by the target prompts. Note that the image similarity metrics IDINO and ICLIP do not simultaneously consider both the preservation of the target structure and the reflection of the reference appearance. They only calculate the similarities between the overall pixels of the reference and generated images. As a result, the key-value replacement, which generates subject-centric images and injects irrelevant elements from reference images into the target context, achieves better image similarities than DreamMatcher, as seen in Table 5 in the main paper. However, as shown in Figure A.5, DreamMatcher more ac-

clean reference latent z0X. However, excessively high λg can reduce subject fidelity due to discrepancies between the reference and target latents in early time steps. We carefully ablated the parameter λg and chose λg = 75 for the ViCo dataset and λg = 50 for the proposed challenging dataset. The pseudo code for overall semantic matching guidance is available in Algorithm 2.

Method IDINO ↑ ICLIP ↑ TCLIP ↑

Warping only reference values (DreamMatcher) 0.680 0.821 0.231 Warping both reference keys and values 0.654 0.809 0.235

Table A.1. Ablation study on key retention.

[Figure 130]

Figure A.7. Statistical results from 5 sets of randomly selected reference images.

curately aligns the reference appearance into the target context, even with large structural displacements. More qualitative comparisons are provided in Figures A.17 and A.18.

This is further demonstrated in a user study comparing MasaCtrl [4] and DreamMatcher, summarized in Figure A.6. A total of 39 users responded to 32 comparative questions, resulting in 1248 responses. These responses were divided between two different questionnaires, with 20 users responding to one and 19 to the other. Samples were chosen randomly from a large, unbiased pool. An example of this user study is shown in Figure A.14. DreamMatcher significantly surpasses MasaCtrl for both fidelity by a large margin, demonstrating the effectiveness of our proposed matching-aware value injection method.

#### E.5. Justification of Key Retention

DreamMatcher brings warped reference values to the target structure through semantic matching. This design choice is rational because we leverage the pre-trained U-Net, which has been trained with pairs of queries and keys sharing the same structure. This allows us to preserve the pre-trained target structure path by keeping target keys and queries unchanged. To validate this, Table A.1 shows a quantitative comparison between warping only reference values and warping both reference keys and values, indicating that anchoring the target structure path while only warping the reference appearance is crucial for overall performance. Concerns may arise regarding the misalignment between target keys and warped reference values. However, we emphasize that our appearance matching self-attention accurately aligns reference values with the target structure, ensuring that target keys and warped reference values are geometrically aligned as they were pre-trained.

#### E.6. Reference Selection

We evaluate the stability of DreamMatcher against variations in reference image selection by measuring the variance of all metrics across five sets of randomly selected reference images. Figure A.7 indicates that all metrics are closely distributed around the average. Specifically, the average

Method IDINO ↑ ICLIP ↑ TCLIP ↑

Textual Inversion [17] 0.529 0.762 0.220 Stable Diffusion (SD) 0.516 0.770 0.215 DreamMatcher 0.571 (+10.74%) 0.785 (+1.95%) 0.214 (-0.50%)

###### Table A.2. Quantitative results of DreamMatcher on Stable Diffusion.

Reference Stable Diffusion DreamMatcher

[Figure 131]

[Figure 132]

[Figure 133]

< 𝑆∗ > 𝐴 < 𝑆∗ > 𝑤𝑖𝑡ℎ 𝐽𝑎𝑝𝑎𝑛𝑒𝑠𝑒 𝑚𝑜𝑑𝑒𝑟𝑛 𝑐𝑖𝑡𝑦 𝑠𝑡𝑟𝑒𝑒𝑡

Figure A.8. Qualitative results of DreamMatcher on Stable Diffusion.

IDINO is 0.683 with a variance of 6e−6, and the average TCLIP is 0.225 with a variance of 3e−5. This highlights that our method is robust to reference selection and consistently generates reliable results. We further discuss the qualitative comparisions- with different reference images in Section G.

#### E.7. DreamMatcher on Stable Diffusion

DreamMatcher is a plug-in method dependent on the baseline, so we evaluated DreamMatcher on pre-trained personalized models [17, 32, 44] in the main paper. In this section, we also evaluated DreamMatcher using Stable Diffusion as a baseline. Table A.2 and Figure A.8 show that DreamMatcher enhances subject fidelity without any off-the-shelf pre-trained models, even surpassing IDINO and ICLIP of Textual Inversion which optimizes the 769-dimensional text embeddings.

#### E.8. Multiple Subjects Personalization

As shown in Figure A.9, we extend DreamMatcher for multiple subjects. For this experiments, we used CustomDiffusion [32] as a baseline. Note that a simple modification, which involves batching two different subjects as input, enables this functionality.

#### E.9. Computational Complexity

We investigate time and memory consumption on different configurations of our framework, as summarized in Table A.3. As seen, DreamMatcher significantly improves subject appearance with a reasonable increase in time and memory, compared to the baseline DreamBooth [44]. Additionally, we observe that reducing the PCA [41] dimension of feature descriptors before building the cost volume does not affect the overall performance, while dramatically reducing time consumption. Note that our method, unlike previous training-based [8, 10, 18, 29, 34, 48, 53, 63, 65] or

Reference CustomDiffusion DreamMatcher

Reference DreamBooth DreamMatcher

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

< 𝑆∗ > 𝐴 𝑟𝑒𝑑 < 𝑆∗ > 𝑜𝑛 𝑡ℎ𝑒 𝑏𝑒𝑎𝑐ℎ

[Figure 141]

[Figure 142]

[Figure 143]

< 𝑆 ∗>,< 𝑆 ∗> < 𝑆 ∗> 𝑛𝑒𝑥𝑡 𝑡𝑜 < 𝑆 ∗> 𝑖𝑛 𝑡ℎ𝑒 𝑠𝑛𝑜𝑤

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

< 𝑆∗ > 𝐴 𝑟𝑒𝑑 < 𝑆∗ > 𝑜𝑛 𝑡𝑜𝑝 𝑜𝑓 𝑔𝑟𝑒𝑒𝑛 𝑔𝑟𝑎𝑠𝑠 𝑤𝑖𝑡ℎ 𝑠𝑢𝑛𝑓𝑙𝑜𝑤𝑒𝑟𝑠

- Figure A.10. Integrating an image editing technique: From left to right: the edited reference image by instructPix2Pix [3], the target image generated by the baseline, and the target image generated by DreamMatcher with the edited reference image. DreamMatcher can generate novel subject images by aligning the modified appearance with diverse target layouts.

Reference DreamBooth DreamMatcher

A <S*> in front of the Eiffel Tower

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

HardEasy

- Figure A.11. Impact of reference selection on personalization: The top row presents results using a reference image that is difficult to match, while the bottom row shows results using a reference image that is relatively easier to match. The latter, containing sufficient visual attributes of the subject, leads to improved personalized results. This indicates that appropriate reference selection can enhance personalization fidelity.

< 𝑆 ∗>,< 𝑆 ∗> < 𝑆 ∗> 𝑛𝑒𝑥𝑡 𝑡𝑜 < 𝑆 ∗> 𝑖𝑛 𝑡ℎ𝑒 𝑠𝑛𝑜𝑤

###### Figure A.9. Qualitative results of DreamMatcher for multiple subject personalization.

Component IDINO ICLIP TCLIP Time [s] Mem. [GB]

- (I) Baseline (DreamBooth [44]) 0.638 0.808 0.237 0.27 4.41

- (II) (I) + Appearance Matching Self-Att. (PCA 64) 0.675 0.818 0.233 0.38 4.49

- (III) (I) + Appearance Matching Self-Att. (PCA 256) 0.676 0.818 0.232 0.46 4.56

- (IV) (II) + Semantic Matching Guid. 0.680 0.820 0.231 0.57 4.85

Table A.3. Computational complexity: We used DreamBooth [44] as the baseline. For this analysis, we examine the time consumption for a single sampling time step.

optimization-based approaches [22], does not require any training or fine-tuning.

### F. More Results

- F.1. Comparison with Baselines

We present more qualitative results comparing with baselines, Textual Inversion [17], DreamBooth [44], and CustomDiffusion [32] in Figure A.15 and A.16.

- F.2. Comparison with Previous Works

We provide more qualitative results in Figure A.17 and A.18 by comparing DreamMatcher with the optimization-based method ViCo [22] and tuning-free methods MasaCtrl [4], FreeU [49], and MagicFusion [68].

we initially edit the reference image with existing editing methods to reflect the stylization prompt red ⟨S∗⟩, and then DreamMatcher generates novel scenes using this edited image. Our future work will focus on incorporating stylization techniques [3, 24, 60] into our framework directly, enabling the model to manipulate the reference appearance when the target prompt includes stylization.

### G. Limitation

Stylization. DreamMatcher may ignore stylization prompts such as A red ⟨S∗⟩ or A shiny ⟨S∗⟩, which do not appear in the reference images, as the model is designed to accurately inject the appearance from the reference. However, as shown in Figure A.10, combining off-the-shelf editing techniques [3, 24, 60] with DreamMatcher is highly effective in scenarios requiring both stylization and place alteration, such as A red ⟨S∗⟩ on the beach. Specifically,

Extreme Matching Case. In Appendix E.6, we demonstrate that our proposed method exhibits robust performance with randomly selected reference images. However, as depicted in Figure A.11, using a reference image that is relatively challenging to match may not significantly improve

the target image due to a lack of confidently matched appearances. This indicates even if our method is robust in reference selection, a better reference image which contains rich visual attributes of the subject will be beneficial for performance. Our future studies will focus on automating the selection of reference images or integrating multiple reference images jointly.

[Figure 154]

###### Figure A.12. Challenging text prompt list: Evaluation prompts in complex, non-rigid scenarios for both non-live and live subjects. ‘{}’ represents ⟨S∗⟩ in Textual Inversion [17] and ‘[V] class’ in DreamBooth [44] and CustomDiffusion [32].

[Figure 155]

- Figure A.13. An example of a user study comparing DreamMatcher with previous methods: For subject fidelity, we provide the reference image and generated images from different methods, ViCo [22], FreeU [49], MagicFusion [68] and DreamMatcher. For prompt fidelity, we provide the target prompt and the generated images from those methods. For a fair comparison, we randomly choose the image samples from a large, unbiased pool.

[Figure 156]

###### Figure A.14. An example of a user study comparing DreamMatcher with MasaCtrl [4]: For subject fidelity, we provide the reference image and images generated from two different methods, MasaCtrl and DreamMatcher. For prompt fidelity, the target prompt and generated images from these two methods are provided. For a fair comparison, image samples are randomly chosen from a large, unbiased pool.

- Algorithm 1 Pseudo-Code for Appearance Matching Self-Attention, PyTorch-like

def AMA(self, pca_feats, q_tgt, q_ref, k_tgt, k_ref, v_tgt, v_ref, mask_tgt, cc_thres, num_heads, **kwargs):

# Initialize dimensions and rearrange inputs. B, H, W = init_dimensions(q_tgt, num_heads) q_tgt, q_ref, k_tgt, k_ref, v_tgt, v_ref = rearrange_inputs(q_tgt, q_ref, k_tgt, k_ref, v_tgt, v_ref,

num_heads, H, W)

# Perform feature interpolation and rearrangement. src_feat, trg_feat = interpolate_and_rearrange(pca_feats, H) src_feat, trg_feat = l2_norm(src_feat, trg_feat)

# Compute similarity. sim = compute_similarity(trg_feat, src_feat)

# Calculate forward and backward similarities and flows. sim_backward = rearrange(sim, ‘‘b (Ht Wt) (Hs Ws) -> b (Hs Ws) Ht Wt’’) sim_forward = rearrange(sim, ‘‘b (Ht Wt) (Hs Ws) -> b (Ht Wt) Hs Ws’’)

flow_tgt_to_ref, flow_ref_to_tgt = compute_flows_with_argmax(sim_backward, sim_forward) # Compute cycle consistency error and confidence. cc_error = compute_cycle_consistency(flow_tgt_to_ref, flow_ref_to_tgt) fg_ratio = mask_tgt.sum() / (H * W) confidence = (cc_error < cc_thres * H * fg_ratio) # Warp value and apply semantic-consistent mask. warped_v = warp(v_ref, flow_tgt_to_ref) warped_v = warped_v * confidence + v_tgt * (1 - confidence) warped_v = warped_v * mask_tgt + v_tgt * (1 - mask_tgt) # Perform self-attention. aff = compute_affinity(q_tgt, k_tgt) attn = aff.softmax(-1) out = compute_output(attn, warped_v) return out

- Algorithm 2 Pseudo-Code for Semantic Matching Guidance, PyTorch-like

for i, t in enumerate(tqdm(self.scheduler.timesteps)):

# Define model inputs. latents = combine_latents(latents_ref, latents_tgt)

# Enable gradient computation for matching guidance. enable_gradients(latents)

# Sampling and feature extraction from the U-Net. noise_pred, feats = self.unet(latents, t, text_embeddings)

# Interpolation and concatenating features from different layers. src_feat_uncond, tgt_feat_uncond, src_feat_cond, tgt_feat_cond = interpolate_and_concat(feats)

# Perform PCA and normalize the features. pca_feats = perform_pca_and_normalize(src_feat_uncond, tgt_feat_uncond, src_feat_cond, tgt_feat_cond)

# Apply semantic matching guidance if required. if matching_guidance and (i in self.mg_step_idx):

_, pred_z0_tgt = self.step(noise_pred, t, latents) pred_z0_src = image_to_latent(src_img) uncond_grad, cond_grad = compute_gradients(pred_z0_tgt, pred_z0_src, t, pca_feats)

alpha_prod_t = self.scheduler.alphas_cumprod[t] beta_prod_t = 1 - alpha_prod_t noise_pred[1] -= grad_weight * beta_prod_t**0.5 * uncond_grad noise_pred[3] -= grad_weight * beta_prod_t**0.5 * cond_grad

# Apply classifier-free guidance for enhanced generation. if guidance_scale > 1.0:

noise_pred = classifier_free_guidance(noise_pred, guidance_scale)

# Step from z_t to z_t-1. latents = self.step(noise_pred, t, latents)

##### Reference Textual Inv. DreamMatcher DreamBooth DreamMatcher CustomDiff. DreamMatcher

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

in a magician outfit jumping over a fence in an astronaut outfit

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

in a basketball jersey leaping across rooftops in a city running across a busy city bridge

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

in firefighter uniform swimming upstream in a river diving into a deep blue ocean

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

wearing a bowtie wearing a backpack climbing a tree

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

wearing a rainbow scarf wearing a bowtie jumping into a pool

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

wearing a santa hat inside a cave entrance climbing a tree

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

in a luxurious interior living room climbing a tree submerged halfway in a crystal-clear lake

- Figure A.15. Qualitative comparision with baselines for live objects: We compare DreamMatcher with three different baselines,

###### Reference Textual Inv. DreamMatcher DreamBooth DreamMatcher CustomDiff. DreamMatcher

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

in soccer player kit at the edge of a swimming pool with Japanese modern city street

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

on top of green grass with sunflowers partially covered by sand in the desert seen from the side

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

on top of a white rug inside a basket nestled among rocks

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

with a beautiful sunset wearing a scarf in a grassy park with a bench

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

in the jungle on a sandy beach near the dunes with Japanese modern city street

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

in a movie theater wearing a top hat at the edge of a swimming pool

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

with a beautiful sunset among the skyscrapers in New York city

from the top

- Figure A.16. Qualitative comparision with baselines for non-live objects: We compare DreamMatcher with three different baselines,

Reference ViCo DreamBooth MasaCtrl FreeU MagicFusion DreamMatcher

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

in an astronaut outfit

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

looking out from a tent

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

submerged halfway in a crystal-clear lake

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

climbing a tree

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

jumping over a fence

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

wearing a backpack

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

swimming upstream in a river

- Figure A.17. Qualitative comparison with previous works [4, 22, 44, 49, 68] for live objects: For this comparison, DreamBooth [44] is used as the baseline for MasaCtrl [4], FreeU [49], MagicFusion [68], and DreamMatcher.

Reference ViCo DreamBooth MasaCtrl FreeU MagicFusion DreamMatcher

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

inside a basket

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

at the edge of a swimming pool

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

nestled among rocks

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

inside a box

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

at a picnic spot with a checkered blanket

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

inside a closet

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

in a schoolyard playground

- Figure A.18. Qualitative comparison with previous works [4, 22, 44, 49, 68] for non-live objects: For this comparison, DreamBooth [44] is used as the baseline for MasaCtrl [4], FreeU [49], MagicFusion [68], and DreamMatcher.

