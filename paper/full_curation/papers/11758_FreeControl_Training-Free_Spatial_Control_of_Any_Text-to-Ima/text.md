# arXiv:2312.07536v1[cs.CV]12Dec2023

## FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition

https://genforce.github.io/freecontrol/ Sicheng Mo†*, Fangzhou Mu§*, Kuan Heng Lin†, Yanli Liu‡, Bochen Guan‡, Yin Li§, Bolei Zhou†

†University of California, Los Angeles, §University of Wisconsin-Madison, ‡Innopeak Technology, Inc

[Figure 1]

Figure 1. Training-free conditional control of Stable Diffusion. (a) FreeControl enables zero-shot control of pretrained text-to-image diffusion models given an input condition image in any modality. (b) Compared to ControlNet [59], FreeControl achieves a good balance between spatial and image-text alignment when facing a conflict between the guidance image and text description. Further, it supports condition types (e.g., 2D projections of point clouds and meshes in the borrow row) for which constructing training pairs is difficult.

### Abstract

Recent approaches such as ControlNet [59] offer users fine-grained spatial control over text-to-image (T2I) diffusion models. However, auxiliary modules have to be trained for each type of spatial condition, model architecture, and checkpoint, putting them at odds with the diverse intents and preferences a human designer would like to convey to the AI models during the content creation process. In this work, we present FreeControl, a training-free approach for controllable T2I generation that supports multiple conditions,

* indicates equal contribution

architectures, and checkpoints simultaneously. FreeControl designs structure guidance to facilitate the structure alignment with a guidance image, and appearance guidance to enable the appearance sharing between images generated using the same seed. Extensive qualitative and quantitative experiments demonstrate the superior performance of FreeControl across a variety of pre-trained T2I models. In particular, FreeControl facilitates convenient training-free control over many different architectures and checkpoints, allows the challenging input conditions on which most of the existing training-free methods fail, and achieves competitive synthesis quality with training-based approaches.

### 1. Introduction

Text-to-image (T2I) diffusion models [4, 42] have achieved tremendous success in high-quality image synthesis, yet a text description alone is far from enough for humans to convey their preferences and intents for content creation. Recent advances such as ControlNet [59] enable spatial control of pretrained T2I diffusion models, allowing users to specify the desired image composition by providing a guidance image (e.g., depth map, human pose) alongside the text description. Despite their superior generation results, these methods [6, 30, 33, 55, 59, 62] require training an additional module specific to each type of spatial condition. Considering the large space of control signals, constantly evolving model architectures, and a growing number of customized model checkpoints (e.g., Stable Diffusion [44] fine-tuned for Disney characters or user-specified objects [24, 46]), this repetitive training on every new model and condition type is costly and wasteful, if not infeasible.

Beyond high training cost and poor scalability, controllable T2I diffusion methods face drawbacks that stem from their training scheme: they are trained to output a target image given a spatially-aligned control condition derived automatically from the same image using an off-the-shelf vision model (e.g., MiDaS [43] for depth maps, OpenPose [10] for human poses). This limits the use of many desired control signals that are difficult to infer from images (e.g., mesh, point cloud). Further, the trained models tend to prioritize spatial conditions over text descriptions, likely because the close spatial alignment of input-output image pairs exposes a shortcut. This is illustrated in Figure 1(b), where there is a conflict between the guidance image and text prompt (e.g., an edge map of a sofa chair vs. “an avocado chair”).

To address these limitations, we present FreeControl, a versatile training-free method for controllable T2I diffusion. Our key motivation is that feature maps in T2I models during the generation process already capture the spatial structure and local appearance described in the input text. By modeling the subspace of these features, we can effectively steer the generation process towards a similar structure expressed in the guidance image, while preserving the appearance of the concept in the input text. To this end, FreeControl combines an analysis stage and a synthesis stage. In the analysis stage, FreeControl queries a T2I model to generate as few as one seed image and then constructs a linear feature subspace from the generated images. In the synthesis stage, FreeControl employs guidance in the subspace to facilitate structure alignment with a guidance image, as well as appearance alignment between images generated with and without control.

FreeControl offers a significant advantage over trainingbased methods by eliminating the need for additional training on a pretrained T2I model, while adeptly adhering to concepts outlined in the text description. It supports a

wide array of control conditions, model architectures and customized checkpoints, achieves high-quality image generation with robust controllability in comparison to prior training-free methods [20, 31, 37, 53], and can be readily adapted for text-guided image-to-image translation. We conduct extensive qualitative and quantitative experiments and demonstrate the superior performance of our method. Notably, FreeControl excels at challenging control conditions on which prior training-free methods fail. In the meantime, it attains competitive image synthesis quality with training-based methods while providing stronger image-text alignment and supporting a broader set of control signals.

Contributions. (1) We present FreeControl, a novel method for training-free controllable T2I generation via modeling the linear subspace of intermediate diffusion features and employing guidance in this subspace during the generation process. (2) Our method presents the first universal training-free solution that supports multiple control conditions (sketch, normal map, depth map, edge map, human pose, segmentation mask, natural image and beyond), model architectures (e.g., SD 1.5, 2.1, and SD-XL 1.0), and customized checkpoints (e.g., using DreamBooth [46] and LoRA [24]). (3) Our method demonstrates superior results in comparison to previous training-free methods (e.g., Plugand-Play [53]) and achieves competitive performance with prior training-based approaches (e.g., ControlNet [59]).

### 2. Related Work

Text-to-image diffusion. Diffusion models [22, 49, 51] bring a recent breakthrough in text-to-image (T2I) generation. T2I diffusion models formulate image generation as an iterative denoising task guided by a text prompt. Denoising is conditioned on textual embeddings produced by language encoders [40, 41] and is performed either in pixel space [7, 34, 42, 48] or latent space [19, 39, 44], followed by cascaded super-resolution [23] or latent-to-image decoding [16] for high-resolution image synthesis. Several recent works show that the internal representations of T2I diffusion models capture mid/high-level semantic concepts, and thus can be repurposed for image recognition tasks [28, 58]. Our work builds on this intuition and explores the feature space of T2I models to guide the generation process.

Controllable T2I diffusion. It is challenging to convey human preferences and intents through text description alone. Several methods thus instrument pre-trained T2I models to take an additional input condition by learning auxiliary modules on paired data [6, 30, 33, 55, 59, 62]. One significant drawback of this training-based approach is the cost of repeated training for every control signal type, model architecture, and model checkpoint. On the other hand, training-free methods leverage attention weights and features inside a pre-trained T2I model for the control of ob-

ject size, shape, appearance and location [9, 15, 18, 38, 57]. However, these methods only take coarse conditions such as bounding boxes to achieve precise control over object pose and scene composition. Different from all the prior works, FreeControl is a training-free approach to controllable T2I diffusion that supports any spatial conditions, model architectures, and checkpoints within a unified framework.

Image-to-image translation with T2I diffusion. Controlling T2I diffusion becomes an image-to-image translation (I2I) task [25] when the control signal is an image. I2I methods map an image from its source domain to a target domain while preserving the underlying structure [25, 36, 47]. T2I diffusion enables I2I methods to specify target domains using text. Text-driven I2I is often posed as conditional generation [8, 26, 33, 59, 61, 62]. These methods finetune a pretrained model to condition it on an input image. Alternatively, recent training-free methods perform zeroshot image translation [20, 31, 37, 53] and is most relevant to our work. This is achieved by inverting the input image [32, 50, 56], followed by manipulating the attention weights and features throughout the diffusion process. A key limitation of these methods is they require the input to have rich textures, and hence they fall short when converting abstract layouts (e.g. depth) to realistic image. By contrast, our method attends to semantic image structure by decomposing features into principal components, thereby it supports a wide range of modalities as layout specifications. Customized T2I diffusion. Model customization is a key use case of T2I diffusion in visual content creation. By fine-tuning a pretrained model on images of custom objects or styles, several methods [5, 17, 27, 46] bind a dedicated token to each concept and insert them in text prompts for customized generation. Amid the growing number of customized models being built and shared by content creators [2, 3], FreeControl offers a scalable framework for zero-shot control of any model with any spatial condition.

### 3. Preliminary

Diffusion sampling. Image generation with a pre-trained T2I diffusion model amounts to iteratively removing noise from an initial Gaussian noise image xT [22]. This sampling process is governed by a learned denoising network ϵθ conditioned on a text prompt c. At a sampling step t, a cleaner image xt−1 is obtained by subtracting from xt a noise component ϵt = ϵθ(xt;t,c). Alternatively, ϵθ can be seen as approximating the score function for the marginal distributions pt scaled by a noise schedule σt [51]:

log pt(xt|c). (1)

ϵθ(xt;t,c) ≈ −σt∇xt

Guidance. The update rule in Equation 1 may be altered by a time-dependent energy function g(xt;t,y) through guidance (with strength s) [14, 15] so as to condition diffusion

[Figure 2]

Figure 2. Visualization of feature subspace given by PCA. Keys from the first self-attention in the U-Net decoder are obtained via DDIM inversion [50] for five images in different styles and modalities (top: person; bottom: bedroom), and subsequently undergo PCA. The top three principal components (pseudo-colored in RGB) provide a clear separation of semantic components.

sampling on auxiliary information y (e.g., class labels):

ϵˆθ(xt;t,c) = ϵθ(xt;t,c) − sg(xt;t,y). (2)

In practice, g may be realized as classifiers [14] or CLIP scores [34], or defined using bounding boxes [12, 57], attention maps [18, 37] or any measurable object properties [15].

Attentions in ϵθ. A standard choice for ϵθ is a U-Net [45] with self- and cross-attentions [54] at multiple resolutions. Conceptually, self-attentions model interactions among spatial locations within an image, whereas cross-attentions relate spatial locations to tokens in a text prompt. These two attention mechanisms complement one another and jointly control the layout of a generated image [9, 18, 38, 53].

### 4. Training-Free Control of T2I Models

FreeControl is a unified framework for zero-shot controllable T2I diffusion. Given a text prompt c and a guidance image Ig of any modality, FreeControl directs a pre-trained T2I diffusion model ϵθ to comply with c while also respecting the semantic structure provided by Ig throughout the sampling process of an output image I.

Our key finding is that the leading principal components of self-attention block features inside a pre-trained ϵθ provide a strong and surprisingly consistent representation of semantic structure across a broad spectrum of image modalities (see Figure 2 for examples). To this end, we introduce structure guidance to help draft the structural template of I under the guidance of Ig. To texture this template with

###### (a) Analysis Stage (b)

Synthesis Stage

###### Generation Branch

|[Figure 3]<br><br>[Figure 4]|
|---|

|[Figure 5]<br><br>[Figure 6]|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

“A photo of [man], with a background”

…

|[Figure 10]<br><br>[Figure 11]|
|---|

|[Figure 12]<br><br>[Figure 13]|
|---|

xT xt xt−1

Prompt

###### …

Seed Images

###### Output Sample with control

PCA ( Fts = [ … ] )

###### “A photo of a Lego man giving a lecture”

Bt = [ … ]

Structure Guidance

Appearance Guidance

Copy

Update

[Figure 14]

[Figure 15]

…

Input Condition

…

…

x¯T x¯t x¯t−1

[Figure 16]

pt(1) pt(2) pt(3) p(Nb)

t pt(c)

p(Nb+1)

[Figure 17]

t

DDIM Inversion

Output Sample without control

Dropped

Semantic bases

xtg

xtg−1

Bt = [ … ]

Semantic Bases for [man]

Guiding Branch

- Figure 3. Method overview. (a) In the analysis stage, FreeControl generates seed images for a target concept (e.g., man) using a pretrained diffusion model and performs PCA on their diffusion features to obtain a linear subspace as semantic basis. (b) In the synthesis stage, FreeControl employs structure guidance in this subspace to enforce structure alignment with the input condition. In the meantime, it applies appearance guidance to facilitate appearance transfer from a sibling image generated using the same seed without structure control.

the content and style described by c, we further devise appearance guidance to borrow appearance details from ¯I, a sibling of I generated without altering the diffusion process. Ultimately, I mimics the structure of Ig with its content and style similar to ¯I.

Method overview. FreeControl is a two-stage pipeline as illustrated in Figure 3. It begins with an analysis stage, where diffusion features of seed images undergo principal component analysis (PCA), with the leading PCs forming the time-dependent bases Bt as our semantic structure representation. Ig subsequently undergoes DDIM inversion [50] with its diffusion features projected onto Bt, yielding their semantic coordinates Sgt. In the synthesis stage, structure guidance encourages I to develop the same semantic structure as Ig by attracting St to Sgt. In the meantime, appearance guidance promotes appearance similarity between I and ¯I by penalizing the difference in their feature statistics.

#### 4.1. Semantic Structure Representation

Zero-shot spatial control of T2I diffusion demands a unified representation of semantic image structure that is invariant to image modalities. Recent work has discovered that selfattention features (i.e., keys and queries) of self-supervised Vision Transformers [52] and T2I diffusion models [9] are strong descriptors of image structure. Based on these findings, we hypothesize that manipulating self-attention features is key to controllable T2I diffusion.

A na¨ıve approach from PnP [53] is to directly inject the self-attention weights (equivalently the features) of Ig into the diffusion process of I. Unfortunately, this approach in-

troduces appearance leakage; that is, not only the structure of Ig is carried over but also traces of appearance details. As seen in Figure 6, appearance leakage is particularly problematic when Ig and I are different modalities (e.g., depth vs. natural images), common for controllable generation.

Towards disentangling image structure and appearance, we draw inspiration from Transformer feature visualization [35, 53] and perform PCA on self-attention features of semantically similar images. Our key observation is that the leading PCs form a semantic basis; It exhibits a strong correlation with object pose, shape, and scene composition across diverse image modalities. In the following, we leverage this basis as our semantic structure representation and explain how to obtain such bases in the analysis stage.

#### 4.2. Analysis Stage

Seed images. We begin by collecting Ns images that share the target concept with c. These seed images {Is} are generated with ϵθ using a text prompt ˜c modified from c. Specifically, ˜c inserts the concept tokens into a template that is intentionally kept generic (e.g., “A photo of [] with background.”). Importantly, this allows {Is} to cover diverse object shape, pose, and appearance as well as image composition and style, which is key to the expressiveness of semantic bases. We study the choice of Ns in Section 5.3.

Semantic basis. We apply DDIM inversion [50] on {Is} to obtain time-dependent diffusion features {Fst} of size Ns × C × H × W from ϵθ. This yields Ns × H × W distinct feature vectors, on which we perform PCA to obtain the time-dependent semantic bases Bt as the first Nb prin-

cipal components:

Bt = [p(1)t ,p(2)t ,...,p(tNb)] ∼ PCA({Fst}) (3)

Intuitively, Bt span semantic spaces St that connect different image modalities, allowing the propagation of image structure from Ig to I in the synthesis stage. We study the choice of Ft and Nb in Section 5.3.

Basis reuse. Once computed, Bt can be reused for the same text prompt or shared by prompts with related concepts. The cost of basis construction can thus be amortized over multiple runs of the synthesis stage.

#### 4.3. Synthesis Stage

The generation of I is conditioned on Ig through guidance. As a first step, we express the semantic structure of Ig with respect to the semantic bases Bt.

Inversion of Ig. We perform DDIM inversion [50] on Ig to obtain the diffusion features Fgt of size C × H × W and project them onto Bt to obtain their semantic coordinates Sgt of size Nb × H × W. For local control of foreground structure, we further derive a mask M (size H × W) from cross-attention maps of the concept tokens [18]. M is set to 1 (size H × W) for global control.

We are now ready to generate I with structure guidance to control its underlying semantic structure.

Structure guidance. At each denoising step t, we obtain the semantic coordinates St by projecting the diffusion features Ft from ϵθ onto Bt. Our energy function gs for structure guidance can then be expressed as

mij∥[st]ij − [sgt]ij∥22 i,j mij

gs(St; Sgt, M) = i,j

forward guidance

(1 − mij)∥ max([st]ij − τt, 0)∥22

+ w · i,j

,

i,j(1 − mij) backward guidance

where i and j are spatial indices for St, Sgt and M, and w is the balancing weight. The thresholds τt are defined as

[sgt]ij (4)

τt = max

i,j s.t. mij=0

with max taken per channel. Loosely speaking, [st]ij > τt indicates the presence of foreground structure. Intuitively, the forward term guides the structure of I to align with Ig in the foreground, whereas the backward term, effective when M ̸= 1, helps carve out the foreground by suppressing spurious structure in the background.

While structure guidance drives I to form the same semantic structure as Ig, we found that it also amplifies lowfrequency textures, producing cartoony images that lack appearance details. To fix this problem, we apply appearance

guidance to borrow texture from ¯I, a sibling image of I generated from the same noisy latent with the same seed yet without structure guidance.

Appearance representation. Inspired by DSG [15], we represent image appearance as {vt(k)}N

a≤Nb

k=1 , the weighted spatial means of diffusion features Ft:

σ([s(tk)]ij)[ft]ij i,j σ([s(tk)]ij)

vt(k) = i,j

, (5)

where i and j are spatial indices for St and Ft, k is channel index for [st]i,j, and σ is the sigmoid function. We repurpose St as weights so that different vt(k)’s encode appearance of distinct semantic components. We calculate {vt(k)} and {v¯t(k)} respectively for I and ¯I at each timestep t.

Appearance guidance. Our energy function ga for appearance guidance can then be expressed as

Na k=1 ∥vt(k) − v¯t(k)∥22

ga({vt(k)};{v¯t(k)}) =

. (6)

Na

It penalizes difference in the appearance representations and thus facilitates appearance transfer from ¯I to I.

Guiding the generation process. Finally, we arrive at our modified score estimate ϵˆt by including structure and appearance guidance alongside classifier-free guidance [21]:

ϵˆt = (1+s)ϵθ(xt;t,c)−sϵθ(xt;t,∅)+λs gs+λa ga (7) where s, λs and λa are the respective guidance strengths.

### 5. Experiments and Results

We report extensive qualitative and quantitative results to demonstrate the effectiveness and generality of our approach for zero-shot controllable T2I diffusion. We present additional results on text-guided image-to-image translation and provide ablation studies on key model components.

#### 5.1. Controllable T2I Diffusion

Baselines. ControlNet [59] and T2I-Adapter [33] learn an auxiliary module to condition a pretrained diffusion model on a guidance image. One such module is learned for each condition type. Uni-ControlNet [62] instead learns adapters shared by all condition types for all-in-one control. Different from these training-based methods, SDEdit [31] adds noise to a guidance image and subsequently denoises it with a pretrained diffusion model for guided image synthesis. Prompt-to-Prompt (P2P) [20] and Plug-and-Play (PnP) [53] manipulate attention weights and features inside pretrained diffusion models for zero-shot image editing. We compare our method with these strong baselines in our experiments.

[Figure 18]

- Figure 4. Qualitative comparison of controllable T2I diffusion. FreeControl supports a suite of control signals and three major versions of Stable Diffusion. The generated images closely follow the text prompts while exhibiting strong spatial alignment with the input images.

Experiment setup. Similar to ControlNet [59], we report qualitative results on eight condition types (sketch, normal, depth, Canny edge, M-LSD line, HED edge, segmentation mask, and human pose). We further employ several previously unseen control signals as input conditions (Figure 5), and combine our method with all major versions of Stable Diffusion (1.5, 2.1, and XL 1.0) to study its generalization on diffusion model architectures.

For a fair comparison with the baselines, we adapt the ImageNet-R-TI2I dataset from PnP [53] as our benchmark dataset. It contains 30 images from 10 object categories. Each image is associated with five text prompts originally for the evaluation of text-guided image-to-image translation. We convert the images into their respective Canny edge, HED edge, sketch, depth map, and normal map following ControlNet [59], and subsequently use them as input conditions for all methods in our experiments.

Evaluation metrics. We report three widely adopted metrics for quantitative evaluation; Self-similarity distance [52] measures the structural similarity of two images in the feature space of DINO-ViT [11]. A smaller distance suggests better structure preservation. Similar to [53], we report selfsimilarity between the generated image and the dataset image that produces the input condition. CLIP score [40] measures image-text alignment in the CLIP embedding space. A higher CLIP score indicates a stronger semantic match between the text prompt and the generated image. LPIPS

distance [60] measures the appearance deviation of the generated image from the input condition. Images with richer appearance details yield higher LPIPS score.

Implementation details. We adopt keys from the first selfattention in the U-Net decoder as the features Ft. We run DDIM inversion on Ns = 20 seed images for 200 steps to obtain bases of size Nb = 64. In the synthesis stage, we run DDIM inversion on Ig for 1000 steps, and sample I and ¯I by running 200 steps of DDIM sampling. Structure and appearance guidance are applied in the first 120 steps. λs ∈ [400,1000], λa = 0.2λs, and Na = 2 in all experiments.

Qualitative results. As shown in Figure 4, FreeControl is able to recognize diverse semantic structures from all condition modalities used by ControlNet [59]. It produces highquality images in close alignment with both the text prompts and spatial conditions. Importantly, it generalizes well on all major versions of Stable Diffusion, enabling effortless upgrade to future model architectures without retraining.

In Figure 5, we present additional results for condition types not possible with previous methods. FreeControl generalizes well across challenging condition types for which constructing training pairs is difficult. In particular, it enables superior conditional control with common graphics primitives (e.g., mesh and point cloud), domain-specific shape models (e.g., face and body meshes), graphics software viewports (e.g., Blender [13] and AutoCAD [1]), and simulated driving environments (e.g., MetaDrive [29]),

[Figure 19]

###### Figure 5. Qualitative results for more control conditions. FreeControl supports challenging control conditions not possible with trainingbased methods. These include 2D projections of common graphics primitives (row 1 and 2), domain-specific shape models (row 3 and 4), graphics software viewports (row 5), and simulated driving environments (row 6).

[Figure 20]

- Figure 6. Qualitative comparison on controllable T2I diffusion. FreeControl achieves competitive spatial control and superior imagetext alignment in comparison to training-based methods. It also escapes the appearance leakage problem manifested by the training-free baselines, producing high-quality images with rich content and appearance faithful to the text prompt.

Canny HED Sketch Depth Normal Self-Sim ↓ CLIP ↑ LPIPS ↑ Self-Sim ↓ CLIP ↑ LPIPS ↑ Self-Sim ↓ CLIP ↑ LPIPS ↑ Self-Sim ↓ CLIP ↑ LPIPS ↑ Self-Sim ↓ CLIP ↑ LPIPS ↑

Method

ControlNet [59] 0.042 0.300 0.665 0.040 0.291 0.609 0.070 0.314 0.668 0.058 0.306 0.645 0.079 0.304 0.637 T2I-Adapter 0.052 0.290 0.689 - - - 0.096 0.290 0.648 0.071 0.314 0.673 - - Uni-ControlNet 0.044 0.295 0.539 0.050 0.301 0.553 0.050 0.301 0.553 0.061 0.303 0.636 - - -

SDEdit-0.75 [31] 0.108 0.306 0.582 0.123 0.288 0.375 0.135 0.281 0.361 0.153 0.294 0.327 0.128 0.284 0.456 SDEdit-0.85 [31] 0.139 0.319 0.670 0.153 0.305 0.485 0.139 0.300 0.485 0.165 0.304 0.384 0.147 0.298 0.512 P2P [20] 0.078 0.253 0.298 0.112 0.253 0.194 0.194 0.251 0.096 0.142 0.248 0.167 0.100 0.249 0.198 PNP [53] 0.074 0.282 0.417 0.098 0.286 0.271 0.158 0.267 0.221 0.126 0.287 0.268 0.107 0.286 0.347 Ours 0.074 0.338 0.667 0.075 0.337 0.561 0.086 0.337 0.593 0.077 0.307 0.477 0.086 0.335 0.629

Table 1. Quantitative results on controllable T2I diffusion. FreeControl consistently outperforms all training-free baselines in structure preservation, image-text alignment and appearance diversity as measured by Self-similarity distance, CLIP score and LPIPS distance. It achieves competitive structure and appearance scores with the training-based baselines while demonstrate stronger image-text alignment.

thereby providing an appealing solution to visual design preview and sim2real.

Comparison with baselines. Figure 6 and Table 1 compare our methods to the baselines. Despite stronger structure preservation (i.e., small self-similarity distances), the training-based methods at times struggle to follow the text prompt (e.g. embroidery for ControlNet and origami for all baselines) and yield worse CLIP scores. The loss of text control is a common issue in training-based methods due to modifications made to the pretrained models. Our method is training-free, hence retaining strong text conditioning.

On the other hand, training-free baselines are prone to appearance leakage as a generated image shares latent states (SDEdit) or diffusion features (P2P and PnP) with the condition image. As a result, not only is the structure carried over but also undesired appearance, resulting in worse LPIPS scores. For example, all baselines inherit the textureless background in the embroidery example and the fore-

ground shading in the castle example. By contrast, our method decouples structure and appearance, thereby escaping appearance leakage.

Handling conflicting conditions. Finally, we study cases where spatial conditions have minor conflicts to input text prompts. We assume that a text prompt consists of a concept (e.g., batman) and a style (e.g., cartoon), and contrast a conflicting case with its aligned version. Specifically, a conflicting case includes (a) a text prompt with a feasible combination of concept and style; and (b) a spatial condition (i.e. an edge map) derived from real images without the text concept. The corresponding aligned case contains a similar text prompt, yet using a spatial condition from real images with the same concept. We input those cases into ControlNet, T2I-Adapter, and FreeControl, using a set of pre-trained and customized models.

Figure 7 shows the results. Our training-free FreeControl consistently generates high quality images that fit the mid-

[Figure 21]

- Figure 7. Controllable T2I generation of custom concepts. FreeControl is compatible with major customization techniques and readily supports controllable generation of custom concepts without requiring spatially-aligned condition images. By contrast, ControlNet fails to preserve custom concepts given conflicting conditions, whereas T2I-Adapter refuses to respect the condition image and text prompt.

dle ground of spatial conditions and text prompts, across all test cases and models. T2I-Adapter sometimes fails even with an aligned case (see Batman examples), not to mention the conflicting cases. Indeed, T2I-Adapter tends to disregard the condition image, leading to diminished controllability, as exemplified by Emma Watson example (conflicting). ControlNet can generate convincing images for aligned cases, yet often fall short in those conflicting cases. A common failure mode is to overwrite the input text con-

cept using the condition image, as shown by skeleton bike or house in a bubble examples (conflicting).

#### 5.2. Extension to Image-to-Image Translation

FreeControl can be readily extended to support image-toimage (I2I) translation by conditioning on a detailed / real image. A key challenge here is to allow FreeControl to preserve the background provided by the condition, i.e., the input content image. To this end, we propose two variants

Condition Ours Ours (w/o mask) Ours (Fixed Seed) Plug-and-Play P2P Pix2Pix-zero SDEdit-.75 SDEdit-.85

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

“A tattoo of a jeep”

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

“A sculpture of a husky”

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

“An embroidery of a bustard”

- Figure 8. Qualitative and quantitative comparison on text-guided image-to-image translation. FreeControl enables flexible control of image composition and style through guidance mask M and random seed (left). It strikes a good balance between structure preservation (self-similarity distance) and image-text alignment (CLIP score) in comparison to the baselines (right, better towards bottom right).

[Figure 50]

- Figure 9. Ablation on size of semantic bases Nb. Images are generated using the prompt “a Lego man giving a lecture”. They illustrate an inherent tradeoff between structure and appearance quality. A good balance can be achieved with Nb’s in the middle range.

of FreeControl. The first removes the mask M in structure guidance (i.e. w/o mask), and the second generates from the inverted latent xgT of the condition image (i.e. fixed seed). We find that removing the mask helps extract and maintain the background structure, and starting inference from xgT retains the appearance from the condition image.

Figure 8 evaluates FreeControl and its two variants for text-guided I2I, and compares to strong baselines for the I2I task including PnP [53], P2P [20], Pix2Pix-zero [37] and SDEdit [31]. The vanilla FreeControl, as we expect, often fails to preserve the background. However, our two variants with simple modification demonstrate impressive results as compared to the baselines, generating images that adhere to both foreground and background of the input image.

Further, we evaluate the self-similarity distance and CLIP score of FreeControl, its variants, and our baselines on the ImageNet-R-TI2I dataset. The results are summarized in Figure 8. Variants of FreeControl outperform all baselines with significantly improved structure preservation and visual fidelity, following the input text prompts.

#### 5.3. Ablation Study

Effect of guidance. As seen in Figure 10, structure guidance is responsible for structure alignment (−gs vs. Ours). Appearance guidance alone has no impact on generation in the absence of structure guidance (−ga vs. −gs,−ga). It only becomes active after image structure has shaped up, in which case it facilitates appearance transfer (−ga vs. Ours). Choice of diffusion features Ft. Figure 11 compares results using self-attention keys, queries, values, and their

preceding Conv features from up block.[1,2] in the U-Net decoder. It reveals that up block.1 in general carries more structural cues than up block.2, whereas keys better disentangle semantic components than the other features.

Number of seed images Ns. Figure 12 suggests that Ns has minor impact on image quality and controllability, allowing the use of as few as 1 seed image in the analysis stage. Large Ns diversifies image content and style, which helps perfect structural details (e.g. limbs) in the generated images.

Size of semantic bases Nb. Figure 9 presents generation results over the full spectrum of Nb. A larger Nb improves structure alignment yet triggers the unintended transfer of appearance from the input condition. Hence, a good balance is achieved with Nb’s in the middle range.

[Figure 51]

Figure 10. Ablation on guidance effect. Top: “leather shoes”; Bottom: “cat, in the desert”. gs and ga stand for structure and appearance guidance, respectively.

[Figure 52]

- Figure 11. Ablation on feature choice. Keys from self-attention of up block.1 in the U-Net decoder expose the strongest controllability. PCA visualization of the features are in the insets.

[Figure 53]

Condition Ns = 1 5 10 20

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Condition

[Figure 64]

[Figure 65]

[Figure 66]

Ours w/o Appearance

guidance Ours

[Figure 67]

- Figure 12. Ablation on number of seed images Ns. Top: “wooden sculpture of a man”; Bottom: “dog, in the snow”. Larger Ns brings minor improvement on structure alignment.

### 6. Conclusion

We present FreeControl, a training-free method for spatial control of any T2I diffusion models with many conditions. FreeControl exploits the feature space of pretrained T2I models, facilitates convenient control over many architectures and checkpoints, allows various challenging input conditions on which most of the existing training-free methods fail, and achieves competitive synthesis quality with training-based approaches. One limitation is that FreeContorl relies on the DDIM inversion process to extract intermediate features of the guidance image and compute additional gradient during the synthesis stage, resulting in increased inference time. We hope our findings and analysis can shed light on controllable visual content creation.

### References

- [1] Autocad. https : / / www . autodesk . com / products/autocad. 6
- [2] Civitai. https://civitai.com/. 3, 15
- [3] Hugging face. https://huggingface.co/. 3
- [4] Midjourney. https://www.midjourney.com. 2
- [5] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia, 2023. 3
- [6] Omri Avrahami, Thomas Hayes, Oran Gafni, Sonal Gupta, Yaniv Taigman, Devi Parikh, Dani Lischinski, Ohad Fried,

- and Xi Yin. Spatext: Spatio-textual representation for controllable image generation. In CVPR, 2023. 2
- [7] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [8] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 3
- [9] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, 2023. 3, 4
- [10] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In CVPR, 2017. 2
- [11] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 6
- [12] Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance. arXiv preprint arXiv:2304.03373, 2023. 3
- [13] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. 6
- [14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 3
- [15] Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. In NeurIPS, 2023. 3, 5
- [16] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 2

- [17] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In ICLR, 2023. 3
- [18] Songwei Ge, Taesung Park, Jun-Yan Zhu, and Jia-Bin Huang. Expressive text-to-image generation with rich text. In ICCV, 2023. 3, 5
- [19] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, 2022. 2
- [20] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. In ICLR, 2023. 2, 3, 5, 8, 10
- [21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2, 3
- [23] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. JMLR, 2022. 2

- [24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 2
- [25] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017. 3
- [26] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023. 3
- [27] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 3
- [28] Alexander C. Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion model is secretly a zero-shot classifier. In ICCV, 2023. 2
- [29] Quanyi Li, Zhenghao Peng, Lan Feng, Qihang Zhang, Zhenghai Xue, and Bolei Zhou. Metadrive: Composing diverse driving scenarios for generalizable reinforcement learning. TPAMI, 2022. 6
- [30] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023. 2
- [31] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2, 3, 5, 8, 10
- [32] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, 2023. 3
- [33] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 2, 3, 5
- [34] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML,

2022. 2, 3

- [35] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 4
- [36] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Semantic image synthesis with spatially-adaptive normalization. In CVPR, 2019. 3
- [37] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In SIGGRAPH, 2023. 2, 3, 10
- [38] Or Patashnik, Daniel Garibi, Idan Azuri, Hadar AverbuchElor, and Daniel Cohen-Or. Localizing object-level shape variations with text-to-image diffusion models. In ICCV,

2023. 3

- [39] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 6
- [41] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 2
- [42] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 2

- [43] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. TPAMI, 2020. 2
- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2
- [45] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 3
- [46] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2, 3
- [47] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In SIGGRAPH, 2022. 3
- [48] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2
- [49] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 2
- [50] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3, 4, 5, 14
- [51] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 2, 3
- [52] Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Splicing vit features for semantic appearance transfer. In CVPR, 2022. 4, 6
- [53] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, 2023. 2, 3, 4, 5, 6, 8, 10

- [54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 3
- [55] Andrey Voynov, Kfir Aberman, and Daniel Cohen-Or. Sketch-guided text-to-image diffusion models. In SIGGRAPH, 2023. 2
- [56] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In CVPR,

2023. 3

- [57] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In ICCV, 2023. 3
- [58] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In CVPR, 2023. 2
- [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 1, 2, 3, 5, 6, 8
- [60] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [61] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. Sine: Single image editing with textto-image diffusion models. In CVPR, 2023. 3
- [62] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. In NeurIPS, 2023. 2, 3, 5

### Supplementary Material

In the supplementary material, we present additional qualitative results (Section A) and ablation experiments (Section B), and discuss the limitations (Section C) and societal impact of our method (Section D). We hope this document complements the main paper.

### A. Additional Qualitative Results

Continuous control. Real-world content creation is a live experience, where an idea develops from a sketch into a more refined and finished piece of work. The intermediate states throughout this process may be interpreted as continuously evolving control signals. Figure 13 illustrates how FreeControl may assist an artist in his or her content creation experience. It produces spatially accurate and smoothly varying outputs guided by constantly changing conditions, thus serving as a source of inspiration over the course of painting.

Compositional control. By combining structure guidance from multiple condition images, FreeControl readily supports compositional control without altering the synthesis pipeline. Figure 14 presents our results using different combinations of condition types. The generated images are faithful to all input conditions while respect the text prompt.

### B. Additional Ablation Study

We now present additional ablations of our model.

Choice of threshold τt. Figure 15 demonstrates that no hard threshold within the range of [0,1] can fully eliminate spurious background signal while ensure a foreground structure consistent with the condition image. By contrast, our dynamic thresholding scheme, implemented as a per-channel max operation, allows FreeControl to accurately carve out the foreground without interference from the background.

Number of guidance steps. Figure 16 reveals that the first 40% sampling steps are key to structure and appearance formation. Applying guidance beyond that point has little to no impact on generation quality.

Choice of guidance weights λs and λa. Figure 17 confirms that FreeControl produces strong results within a wide range of guidance strengths. In particular, the output images yield accurate spatial structure when λs ≥ 400 and rich appearance details when λa ≥ 0.2λs. We empirically found that these ranges work for all examples in our experiments. Basis reuse across concepts. Once computed, the semantic bases St can be reused for the control of semantically related concepts. Figure 18 provides one such example, where St derived from seed images of man generalize well on other mammals including cat, dog and monkey, yet fail for the semantically distant concept of bedroom.

### C. Limitations

One limitation of FreeControl lies in its inference speed. Without careful code optimization, structure and appearance guidance result in 66% longer inference time (25 seconds) on average compared to vanilla DDIM sampling [50] (15 seconds) with the same number of sampling steps (200 in our experiments) on an NVIDIA A6000 GPU. This is on par with other training-free methods.

Another issue is that FreeControl relies on the pretrained VAE and U-Net of a Stable Diffusion model to encode the semantic structure of a condition image at a low spatial resolution (16 × 16). Therefore, it sometimes fails to recognize inputs with missing structure (e.g., incomplete sketch), and may not accurately locate fine structural details (e.g., limbs). Representative failure cases of FreeControl are illustrated in Figure 19.

### D. Societal Impact and Ethical Concerns

This paper presents a novel training-free method for spatially controlled text-to-image generation. Our method provides better control of the generation process with a broad spectrum of conditioning signals. We envision that our method provides a solid step towards enhancing AI-assisted visual content creation in creative industries and for media and communication. While we do not anticipate major ethical concerns about our work, our method shares common issues with other generative models in vision and graphics, including privacy and copyright concerns, misuse for creating misleading content, and potential bias in the generated content.

[Figure 68]

###### Figure 13. Controllable generation over the course of art creation. Images are generated from the same seed with the prompt ”a photo of a man and a woman, Pixar style” with a customized model from [2]. FreeControl yields accurate and consistent results despite evolving control conditions throughout the art creation timeline.

[Figure 69]

###### Figure 14. Qualitative results on compositional control. FreeControl allows compositional control of image structure using multiple condition images of potentially different modalities.

[Figure 70]

###### Figure 15. Ablation on threshold τt. Images are generated using the prompt ”leather shoe on the table”. Our dynamic threshold (max) encourages more faithful foreground structure and cleaner background in comparison to various hard thresholds (e.g., 0.1).

[Figure 71]

###### Figure 16. Ablation on number of guidance steps. Images are generated using the prompt ”a modern house, on the grass, side look”. Applying guidance beyond the first 40% diffusion steps (0.4) has little to no impact on the generation result.

[Figure 72]

- Figure 17. Ablation on guidance weights λs and λa. Images are generated with the prompt ”an iron man is giving a lecture”. FreeControl yields strong results across guidance weights.

[Figure 73]

- Figure 18. Ablation on basis reuse. The semantic bases computed for ”man” enable the controllable generation of semantically related concepts (cat, dog and monkey) while fall short for unrelated concepts (bedroom).

[Figure 74]

- Figure 19. Failure cases. FreeControl does not anticipate missing structure in the condition image (left) and may not accurately position fine structural details (limbs) in the output image (right).

