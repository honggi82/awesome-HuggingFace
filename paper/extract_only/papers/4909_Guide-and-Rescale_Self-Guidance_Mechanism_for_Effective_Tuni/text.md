# arXiv:2409.01322v3[cs.CV]25Sep2024

## Guide-and-Rescale: Self-Guidance Mechanism for Effective Tuning-Free Real Image Editing

Vadim Titov2∗, Madina Khalmatova4∗, Alexandra Ivanova1,2,3∗, Dmitry Vetrov5, and Aibek Alanov1,2

1 HSE University 2 AIRI titov.vn@phystech.edu, {a.ivanova,alanov}@airi.net 3 Skolkovo Institute of Science and Technology 4 UNSW Sydney m.khalmatova@student.unsw.edu.au 5 Constructor University, Bremen dvetrov@constructor.university

[Figure 1]

###### Original Image

###### Original Image

A photo of a lion A photo of a goat A photo of a wolf Lying down woman wearing yellow sweater

Lying down woman wearing yellow jacket

Lying down woman wearing red sweater

Lying down woman wearing green jacket

A photo of a tiger

###### Original Image

###### Original Image

A photo of a sad woman

A photo of a crying woman

A photo

Anime style face A pixar style face A mosaic depicting of a face

A photo of a woman

A photo of a happy woman

Fig. 1: Guide-and-Rescale for real image editing. Our method allows to manipulate images for a wide range of different editings. It achieves a good balance between quality of manipulation and preservation of the original image.

Abstract. Despite recent advances in large-scale text-to-image generative models, manipulating real images with these models remains a challenging problem. The main limitations of existing editing methods are that they either fail to perform with consistent quality on a wide range of image edits or require time-consuming hyperparameter tuning or fine-tuning of the diffusion model to preserve the image-specific appearance of the input image. We propose a novel approach that is built upon a modified diffusion sampling process via the guidance mechanism. In this work, we explore the self-guidance technique to preserve the overall structure of the input image and its local regions appearance that should not be edited. In particular, we explicitly introduce layout-preserving energy functions that are aimed to save local and global structures of the source image. Additionally, we propose a noise rescaling mechanism that allows to preserve noise distribution by balancing the

* Equal contribution.

norms of classifier-free guidance and our proposed guiders during generation. Such a guiding approach does not require fine-tuning the diffusion model and exact inversion process. As a result, the proposed method provides a fast and high-quality editing mechanism. In our experiments, we show through human evaluation and quantitative analysis that the proposed method allows to produce desired editing which is more preferable by humans and also achieves a better trade-off between editing quality and preservation of the original image. Our code is available at https://github.com/MACderRu/Guide-and-Rescale.

### 1 Introduction

In recent years, diffusion models [7,25] have been rapidly developed due to their high generation quality. Therefore, they have started to be actively used as a base model for text-to-image generative models [19–21], where an image has to be generated from a textual description. Although such models have already achieved impressive results, they are still difficult to apply to the task of editing real images. The main problem is to find a balance between making the edited image reflect the expected change and preserving the original structure and the parts that should not have been edited.

Existing approaches to image manipulation based on text-to-image models address this task in different ways and can be categorized into three main groups of methods. Approaches from the first category [10,26,29] use fine-tuning of the whole diffusion model on the input image in order to preserve its structure and all necessary details during editing. Although such a strategy gives good results, the optimization process makes them very long, which does not allow to apply them in practice. In the next group of methods [1,2,4–6,17,24], instead of the optimization step, they use internal representations of images in the diffusion model and replace them during generation to preserve parts of the original image that should not change during editing. These methods are significantly faster than optimization-based methods, but they are not universal and require careful tuning of hyperparameters. As a result, they can only work consistently for a narrow set of edits. Finally, the third group of methods [14–16, 27] focuses on building a high-quality reconstruction of the original image by minimizing the discrepancy between the forward and backward trajectories of the diffusion processes. The main problem of this group of methods is the additional time required to construct a good quality inversion.

In general, the currently available methods for manipulating real images have many limitations in terms of time, quality, and edit versatility, so it is still a challenging problem to find a model that is efficient, high quality, and covers most of the edit types.

In this paper, we investigate the guidance technique for the problem of real image editing. As most of the tunning-free approaches leverage UNet internal representations which are cached during inversion and used during editing generation, it causes misalignment between real features and inserted ones, and

therefore noise trajectory may leave the distribution of natural images. To overcome these issues we propose special energy functions named guiders that are designed to preserve overall source image structure and local regions that should not be edited as well. We explicitly introduce preserving term that smoothly manipulates overall noise trajectory distribution. Additionally, we extend our framework by the automatic mechanism of noise rescaling which prevents discrepancy of noise trajectory from initial diffusion sampling by balancing the norms of classifier-free guidance and our proposed guiders. Our approach does not require the exact reconstruction as well as the fine-tuning the diffusion model, so it is computationally efficient. Due to the flexibility of the proposed guiders, our method is versatile and supports many types of editing, from local changes of objects to global stylisation (see Fig. 1).

We apply our method to Stable Diffusion [20] model and conduct an extensive set of experiments for comparison with other methods. Our approach achieves a better balance between editing quality and preservation of the original image structure in terms of CLIP/LPIPS scores for a wide range of different editing types. For a standard image-to-image problem (Dog → Cat) we show that our method performs better than other baselines. Also, we provide user studies and show that our approach is most preferable in human evaluation.

### 2 Related Work

Diffusion models [7,25] show high-quality results in generative modeling. This is the reason why they could be used to solve the image editing problem. Text-toimage diffusion models [19–21] are particularly useful and common for solving this problem. One of the first text-guided image editing methods is SDEdit [13] whose main idea is to add the noise step by step and then denoise with different conditions, but the quality of the editing is not so high. We can distinguish three main groups of editing methods which we consider next.

Optimization-based methods. Methods in the first group [10,26,29] are based on fine-tuning the diffusion for the input image. In Imagic [10] and SINE [29] the diffusion model is fine-tuned on the original image, and the target prompt, which allows taking into account initial image characteristics but slows down the editing process. UniTune [26] fine-tunes the model only on the original image, but it still works slowly. However, UniTune, in contrast to Imagic and SINE, allows applying multiple edits to one image without any additional optimization.

Methods that utilize internal representations of the diffusion model. The following approaches [1,2,4–6,17,24] are based on using internal representations of the image in the text-to-image diffusion model, such as features from cross-attention and self-attention layers to preserve the structure and the details of the original image. Prompt-to-Prompt [6] controls the cross-attention by replacing maps for new tokens from the target prompt and preserving maps in the overlapping tokens. In [17] authors guide cross-attention maps to reference reconstruction with a source prompt during the denoising process. Plug-and-Play [24]

|DDIM Inversion|
|---|

| |[Figure 2]|
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

"A photo of a tiger"

Enc.

Guider Energy

Self-attention maps

Features

Guider signal rescaling

| | | |
|---|---|---|
| | | |

Function

| |
|---|

UNet

Denonising step

Self-attention maps

Features

|[Figure 3]|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

"A photo of a dog"

Dec.

|Sampling|
|---|

- Fig. 2: Overall scheme of the proposed method Guide-and-Rescale. First, our method uses a classic ddim inversion of the source real image. Then the method performs real image editing via the classical denoising process. For every denoising step the noise term is modified by guider that utilizes latents zt from the current generation process and time-aligned ddim latents zt∗.

outperforms Prompt-to-Prompt by using feature and self-attention outputs as conditions for the diffusion model to focus on spatial features and their selfaffinities. InstructPix2Pix [1] proposes an approach based on a combination of Prompt-to-Prompt and the language model to control editing through specific instructions, but it requires an additional dataset generation phase.

The limitation of previous methods is the difficulty of changing poses and object locations. MasaCtrl [2] uses mutual self-attention features based on reconstruction during editing to preserve the local content and textures of the original image. In ProxMasaCtrl [5], which is MasaCtrl extension, an additional proximal gradient step for the scaled noise difference is used to change the geometry and layout in edited images. Despite the ability to change object poses, the disadvantage of MasaCtrl is that the original image, the object background, is preserved inaccurately. Self-guidance [4] proposes an approach to simple editings such as changing object position, size, and color by using intermediate activations and attention interactions as guidance signals during generation. The method we propose is based on this self-guidance mechanism, but we develop it for any type of editing.

Methods that improve the inversion part. The next group of methods [14–16, 27] focuses on improving the quality of the image reconstruction, and it allows to preserve parts of the image that should not be edited and the overall structure. This is achieved by narrowing the gap between the forward and backward trajectories of the diffusion model. Null-text Inversion [15] is proposed as an adaptation of Prompt-to-Prompt [6] for real images and is based on the optimization of the null-text embedding which is used in classifier-free guidance. To avoid learning the null-text embedding for each image, a new method for

its approximation by the source prompt embedding is suggested in Negativeprompt Inversion [14], but it can lead to artifacts. EDICT [27] overcomes the restrictions of DDIM [23] in the image reconstruction by mathematically precise image inversion, but requires some additional computation time. Finally, AIDI [16] is proposed to increase the precision of the reconstruction by applying fixed-point iteration, Anderson acceleration in inversion, and blended guidance for sampling.

The main problem of the methods in this group is the extra time needed to achieve a high-quality reconstruction of the initial image. Our method in contrast does not require an optimization and additional phase for image reconstruction, and therefore it is more computationally efficient.

### 3 Method

#### 3.1 Preliminaries

Diffusion Model. The method uses a pre-trained text-to-image diffusion model. In our experiments, we use a publicly available Stable Diffusion (SD) [20] model, specifically its checkpoint stable-diffusion-v1-4. Our method does not finetune or modify it.

SD is a latent diffusion model (LDM) [20], meaning that the model operates in a latent space. Using a pre-trained VAE [11] encoder, Enc., one can encode an image into a sample from the latent space. And, likewise, given a latent z and a VAE decoder, Dec., one can obtain an image space sample x. However, our method does not rely on the existence of the latent space, so it may be applied to diffusion models, that perform sampling directly in the image space [21].

SD is a text-to-image model, meaning that it can be conditioned on textual captions. Our method heavily relies on this property, as it requires two prompts to be specified: a source prompt, describing the initial image, and a target prompt, outlining the desired result of editing.

We use DDIM [23] sampling with T = 50 intermediate steps, where a single sampling step is defined as:

zt−1 = atzt + btεθ(zt,t,y), (1)

where at,bt are specified coefficients, zt is a current latent, t is a current timestep, y is conditioning data and εθ(zt,t,y) is the noise, predicted by the diffusion model.

To allow the diffusion model to operate with real images, we need a noised latent zT, corresponding to this image. For this purpose, we use DDIM inversion, defined similarly to Equation 1:

zt+1 = a∗tzt + b∗tεθ(zt,t,y). (2)

Even though DDIM inversion is not stable, as noted by the authors of Prompt-to-Prompt [6], our method overcomes this issue without any enhancements to the inversion process or optimization and is capable of preserving features of the initial image, as we show in experiments.

Guidance. Guidance mechanism takes the prediction of the diffusion model and enhances it with information from additional data. Classifier guidance relies on a separate model, classifier p(y|zt), trained on noised latents. As long as this classifier is differentiable with respect to the current latent, we can subtract score function ∇zt

log p(y|zt) from the diffusion model prediction to sample from the conditional distribution p(zt|y) instead of the data distribution p(zt).

Training a separate classifier on specific data may sometimes be a resourceful task. Given a conditional diffusion model, one can rely on this model’s knowledge about additional data, that it inquires with conditioning capabilities. This guidance technique is called classifier-free guidance (CFG) [8]. To determine the sampling direction, which leads to correspondence with conditioning data y, CFG compares a conditional prediction of the model with an unconditional one. In case of text-to-image models, the latter can be easily obtained by conditioning the model on the empty text ∅ = “”. With CFG incorporated, the diffusion model prediction, used in both sampling, Equation 1, and inversion, Equation 2, takes the form of εˆθ(zt,t,y):

εˆθ(zt,t,y) = CFG(zt,t,y,w) = εθ(zt,t,∅) + w εθ(zt,t,y)−εθ(zt,t,∅) , (3)

where w is a guidance scale, that controls to which extent additional data y influences the generation process. For SD model, the guidance scale is typically chosen as w = 7.5. It is important to note, that with w = 1 CFG sampling step equals regular conditional sampling.

Self-guidance. As proposed in [4], the choice in guidance sources is not limited to either the classifier or the diffusion model itself. One can use any energy function g to guide the sampling process, as long as there exists a gradient with respect to zt. When the energy function uses outputs of internal layers of the diffusion model, the guidance process is called self-guidance. The authors of the method suggest defining energy function g on top of cross-attention maps Across := cross attn.[εθ(zt,t,y)] and the output of penultimate layer of the diffusion model decoder, features Ψ := features[εθ(zt,t,y)]. To add self-guidance to CFG, one has to modify εˆθ(zt,t,y) from Equation 3 as:

g(zt,t,y,Across,Ψ), (4) where v is a self-guidance scale.

εˆθ(zt,t,y) = CFG(zt,t,y,w) + v · ∇zt

#### 3.2 Guide and Rescale

Consider an initial image xinit, a source prompt ysrc and a target prompt ytrg. For example, let xinit be a photo of a woman with blue hair, wearing a shirt with a drawing. A source prompt, describing the initial image, can be defined as ysrc = “A photo of a woman wearing a shirt with a drawing”. A target prompt should describe the desired editing result. For instance, if we want to change the color of the shirt to red, the target prompt can be defined as ytrg = “A photo of a woman wearing a red shirt with a drawing”. This example is illustrated in

- Fig. 3.

[Figure 4]

- Fig. 3: Editing example. From left to right, first: initial image; second: naive editing, described in Equation 6; third: editing with simple energy function g from Equation 7; fourth: editing with the proposed method.

Our goal is to generate an image xedit, that would correspond to the editing, specified in ytrg. At the same time, xedit should contain all the features of xinit, not influenced by editing. For example, when changing the color of the shirt, xedit and xinit should be the same in all other aspects. Meaning, that we should preserve the background of the photo and the visual appearance of the woman, including the color of her hair, her facial expression, and her position.

A naive approach to obtain correspondence to ytrg can be described as follows. First, we obtain a DDIM inversion trajectory {zt∗}Tt=0 for xinit, conditioning on ysrc. We use CFG with guidance scale w = 1, i.e. regular conditional sampling: z0∗ = Enc.(xinit) zt∗+1 = a∗tzt∗ + b∗tεθ(zt∗,t,ysrc) 0≤t≤T−1.

(5)

A noised latent zT∗ has information about xinit encoded, so an obvious further step would be to synthesize an image, starting from obtained zT∗ and conditioning on ytrg, using CFG with standard sampling guidance scale w = 7.5:

zˆT = zT∗ zˆt−1 = atzˆt + btCFG(ˆzt,t,ytrg,7.5) 1≤t≤T xedit = Dec.(ˆz0).

(6)

This method achieves near-perfect coherence with ytrg. However, due to a mismatch of guidance scales and conditioning data in inversion and sampling processes, trajectory {zˆt}Tt=0 ends up being too far from {zt∗}Tt=0. This results in xinit being modified significantly, as shown by the second image in Fig. 3.

In previous works, there are different workarounds suggested. For example, substituting inner representations of the diffusion model with corresponding representations from {zt∗}Tt=0 [2,6,24], finetuning the model [10] or optimizing nulltext embeddings [15] to move {zˆt}Tt=0 closer to the inversion trajectory.

In this work, we suggest addressing this problem with modified self-guidance. As inversion trajectory is an almost perfect reconstruction trajectory for xinit, we suggest guiding the sampling process with respect to {zt∗}Tt=0. This way we will be able to preserve features of the initial image. For example, we can simply

define the function g as an L2 norm of the difference of the current latent, obtained with CFG sampling in Equation 6, and the corresponding inversion latent:

g(ˆzt,zt∗) = ∥zˆt − zt∗∥22. (7)

However, for proper editing, it is important to control, which regions of the image we want to preserve with the function g, and which areas we still want to vividly edit. Based on editing results (Fig. 3), the simple function in Equation 7 does not hold this property.

We found that applying guidance over inner representations of the model offers the desired control over image regions. To obtain this inner representation, we need to conduct two separate forward passes over the diffusion model.

First, for the inversion trajectory, we make a single sampling step with the same parameters, as in Equation 5:

z¯t∗−1 = atzt∗ + btεθ(zt∗,t,ysrc). (8)

Together with a new latent z¯t∗−1, we obtain inner representations of the diffusion model εθ, corresponding to the inversion trajectory. For now, we denote them as I∗, i.e. I∗ is a set of inner representations (for example, outputs of crossattention or self-attention layers, etc.) that are calculated during the forward pass of εθ(zt∗,t,ysrc). The I∗ will be defined more specifically below.

As I∗ is produced with conditioning on ysrc, a proper way to obtain inner representations for the current trajectory, denoted as I, would be to make a single sampling step, starting at a current latent zt and conditioning on ysrc:

z¯t−1 = atzt + btεθ(zt,t,ysrc). (9)

Conditioning on the same prompt ysrc is an important detail towards more stable editing process. This way, the resulting inner representations for the inversion and the current trajectories are aligned better, rather than when conditioning on different prompts. Besides, when preserving features of the initial image, we want to detach information about editing from the current state of these features. Our logic is, that with the sampling step in Equation 9 we want to see, how much has reconstruction of the initial image suffered, and compare it with the ideal reconstruction in Equation 8.

The current latent zt in Equation 9 is the result of applying CFG together with our self-guidance at the previous sampling step, initially being defined as zT = zT∗ .

We define the energy function as g(zt,zt∗,t,ysrc,I∗,I). A sampling step in Equation 4 can be rewritten as:

g(zt,zt∗,t,ysrc,I∗,I). (10)

εˆθ(zt,t,y) = CFG(zt,t,ytrg,7.5) + v · ∇zt

See the overall pipeline of the method in Fig. 2 and more details can be found in Appendix A. We suggest ways to specify Equation 10 by defining the inner representations sources in the following section, where we denote function g as a guider.

Guiders. We found that jointly including a Self-attention Guider and a Feature Guider into the generation process is sufficient for improving editing and preserving the initial image features. At the same time, they do not interfere with the editing, done by CFG. The effect of these guiders is illustrated in Fig 4a. More details about guiders and its properties can be found in Appendix B.

Self-attention Guider. As noted by the authors of [24], self-attention maps contain information about the layout of the image, i.e. relative positioning of objects. While preserving layout is a primary challenge for stylisation tasks, it is also useful for other editing types.

To utilize this finding, we suggest guiding through matching of self-attention maps from the current trajectory A¯selfi := self attn.[εθ(zt,t,ysrc)] and an ideal reconstruction trajectory A∗iself := self attn.[εθ(zt∗,t,ysrc)], where i corresponds to the index of the UNet layer. So, in this case I∗ = {A∗iself},I = {A¯selfi } and the guider is defined as follows:

g(zt,zt∗,t,ysrc,{A∗iself},{A¯selfi }) = Li=1 mean∥A∗iself − A¯selfi ∥22, (11) where L is a number of UNet layers.

Feature Guider. While preserving layout can significantly improve editing quality, it is not sufficient. For example, when applying local editing to a photo of a person, it is crucial to preserve visual features as well, such as face expressions.

To achieve this, for non-stylisation editing task we adopt an idea from [4]. The authors of this work propose an appearance function for controlling visual appearance of an object, which utilizes masked features. In this work we define features Φ as an output of the last up-block in UNet. We also found, that masking is not crucial and does not influence the quality of editing in our setting, so we eliminated it for simplicity. Similar to [4], we apply L1 norm to the difference of Φ¯ = features[εθ(zt,t,ysrc)] and Φ∗ = features[εθ(zt∗,t,ysrc)]:

g(zt,zt∗,t,ysrc,Φ∗,Φ¯) = mean∥Φ∗ − Φ¯∥1. (12)

For stylisation tasks, similar to [24], we define Φ as an output of the second ResNet block in the second UNet up-block. Similar to Equation 12, the guider is defined as follows:

g(zt,zt∗,t,ysrc,Φ∗,Φ¯) = mean∥Φ∗ − Φ¯∥22. (13)

Noise rescaling. Even though the proposed guiders significantly improve editing quality, the resulting method experiences inconsistency in terms of optimal guidance scales, when applied to a wide range of images.

We noticed that when our method’s behavior is unstable, norms of CFG and the sum of gradients of all the guiders show instability over sampling steps as well.

We suggest scaling the sum of guiders’ gradients by a factor, depending on the CFG norm. This way, the proportion of both editing and preservation will be consistent throughout the whole synthesis process and easily controllable.

More formally, a single noise sampling step will now be defined as:

[Figure 5]

[Figure 6]

(a) (b)

- Fig. 4: (a) Effect of the proposed guiders (Equation 11, Equation 12). Jointly applying both guiders preserves both layout and visual characteristics of unedited regions of the image. (b) Illustration of applying noise rescaling (Equation 14). This technique aligns the sum of guiders with CFG according to the coefficient, defined in Equation 15, therefore stabilizes editing and improves its quality.

ϵt = εθ(zt,t,∅) + w εθ(zt,t,y)−εθ(zt,t,∅) +

(14)

gi(zt,zt∗,t,ysrc,I∗,I),

+γ i vi · ∇zt

where γ is a scaling factor. With r being a numerical parameter, controlling the proportion between editing and preservation, γ can be computed as:

rcur(t) = ∥w εθ(zt,t,y) − εθ(zt,t,∅) ∥22 ∥ i vi · ∇zt

,

gi(zt,z∗

t ,t,ysrc)∥22

(15)

γ = r · rcur(t).

We found that keeping r in an interval, rather than defining it as a fixed numerical hyperparameter, works best in experiments. An interval can be defined so that the fraction of norms in Equation 15 does not exceed some pre-defined upper and lower boundaries. This way, together with obtaining the desired consistency, we still let the diffusion model decide, whether the current sampling step needs more editing or preservation. More formally:

 

rlower, r 1

cur(t) ≤ rlower 1

rcur(t), rlower < r 1

, (16)

cur(t) < rupper rupper, r 1

r(t) =



cur(t) ≥ rupper

An example of applying noise rescaling to the proposed method is provided in

- Fig. 4b. More details about rescaling and its analysis can be found in Appendix C.

### 4 Experiments

Experiment Setup. In order to comprehensively compare image manipulation methods, we decided to consider 4 different types of editing: 1) local manipulation of a person’s appearance and clothing, 2) changing person’s emotions,

- 3) replacing one animal with another, and 4) global image stylisation. For each type, we collected 20 different example edits on which we tested all methods. A more detailed description of these types of edits and all examples can be found in the Appendix D.

To compare the similarity of edited and real images we collected more extensive datasets. A particular case of animal-to-animal type of editing is the task of transformation dogs into cats. To compare methods on this task we collected 500 examples of dogs from the AFHQ dataset [3] and then compared the results of editing with cats from the same dataset. For other types of editing datasets and details are described in the Appendix D.

We compared our method with existing approaches that have publicly available source code: NPI [14], NPI Prox [5] and NTI [15] based on P2P [6], MasaCtrl [2], ProxMasaCtrl [5], PnP [24], and EDICT [27]. We used the authors’ original code with the default parameters recommended in the description of each method. For our method, we also fixed hyperparameters during evaluation. More details about our method setup can be found in Appendix F.

#### 4.1 Qualitative Comparison

In Fig. 5 we see the result of our method and previous approaches on examples of edits belonging to one of the 4 types of edits mentioned above. It can be seen that our method shows stable quality across all types of edits. In cases where the whole image does not need to be stylized, our method, unlike baselines, is good at preserving the structure and background that should not be changed. For example, when editing a person’s emotion, our method only affects the face area and does not change the rest of the person’s attributes and background, unlike other methods that corrupt the overall structure of the original image. This can be seen particularly well in local edits of different attributes of a woman. The MasaCtrl and ProxMasaCtrl methods strongly alter the geometry of the person, and the target edit is not always obtained. The P2P+NTI, P2P+NPI, and P2P+NPI Prox methods either transform the target attribute while corrupting others or edit nothing at all. The PnP method almost always achieves the desired edit but at the cost of severe loss of structure of the original image. The EDICT method performs best compared to the other methods but is still inferior to our method in most of these examples.

#### 4.2 Quantitative Comparison

To quantitatively evaluate the performance of our method, we decided to use an evaluation protocol similar to that used in EDICT [27]. This protocol measures two key properties of the method. The first is the extent to which the result of

###### Original Image EDICT MasaCtrl ProxMasaCtrl P2P + NTI P2P + NPI P2P + NPI Prox PnP Ours

[Figure 7]

A photo of a tiger −→ A photo of a cat

A photo of a deer −→ A photo of a zebra

A photo of a man −→ A photo of a laughing man

A photo of a woman −→ A photo of a frightened woman

A photo of a woman wearin ... −→ A photo of a woman wearing a red shirt with a ...

A photo of a woman wearin ... −→ A photo of a woman wearing a long sleeve with ...

A photo −→ Anime style face

A couple dancing −→ Watercolor drawing of a couple dancing

A person −→ A sculpture of a person

A photo −→ A pixar style face

- Fig. 5: Visual comparison of our method with baselines over different types of editing. Our approach shows more consistent results than existing methods and achieves a better trade-off between editing quality and preservation of the structure of the original image.

the method contains the desired edit. To measure this characteristic, we use the CLIP score [18], which is calculated between the edited image and the target prompt. The second property is the extent to which the method preserves the structure and details of the original image that should not change during editing. For this, we use the LPIPS metric [28] between the original image and its edited version.

The results of these metrics for our method and others are shown in Table 1. They show that our method achieves the best balance between preserving the original image structure and editing the target attribute. In particular, on the LPIPS metric, we perform best apart from the EDICT method, which we significantly outperform on the CLIP score. On the CLIP score, we also perform best apart from the PnP method, which is strongly inferior in its ability to preserve the structure of the original image in terms of the LPIPS metric. The other approaches show uniformly worse results with respect to these two metrics. It is also worth noting that compared to the most advanced EDICT and PnP methods, our method is more computationally efficient, as can be seen from the running time for editing an image. More analysis of these results can be found in Appendix G.

To quantitatively evaluate the distance between the distributions of edited and real images we calculate FID Score [22] on the dog-to-cat transformation task. As shown in Table 1 our method outperforms the others by FID Score. MasaCtrl and ProxMasaCtrl show the highest FID that is correlated with artifacts on edited images and its nonrealism in Fig. 21 in Appendix D.3. The closest FID Score to our method has PnP but it is shown to be several times slower than our method.

- Table 1: Comparison with baselines on image editing task by using 80 examples of 4 different edit types to compute the LPIPS and CLIP per edit. 500 samples of the dogto-cat type of editing are used to compute FID. Our method achieves the best balance between editing quality (CLIP) and preservation of the original image (LPIPS) and reaches the best similarity to the real image’s distribution (FID). The inference time (including the inversion part) is computed on a single GPU A100 in seconds.

Method LPIPS ↓ CLIP ↑ FID ↓ Time (s) ↓ ProxMasaCtrl [5] 0.267 0.215 94.53 12.94 MasaCtrl [2] 0.306 0.223 100.62 13.73 EDICT [27] 0.221 0.229 47.13 68.13 P2P [6] + NTI [15] 0.279 0.233 42.46 66.77 P2P [6] + NPI [14] Prox [5] 0.170 0.233 43.16 8.59 P2P [6] + NPI [14] 0.251 0.234 44.05 8.54 PnP [24] 0.366 0.256 39.55 197.0 Guide-and-Rescale (ours) 0.228 0.243 39.07 24.26

#### 4.3 User Study

Next, we conducted a user study to compare our method with baselines. We used the same example edits of the 4 types we described above. We evaluate how much our method is preferred by others in the form of a pairwise comparison question. We surveyed 62 users who answered a total of 960 questions. Each question showed the original image, the targeting prompt, and two results. Users had to answer two questions. The first question (Q1) is "Which image matches the desired editing description best?". The second question (Q2) is "Which image preserves the overall structure of the "Original Image" best?".

The results of this user study are shown in Table 2. Each value represents the percentage of the users that preferred our method over the corresponding baseline in the first column. We can see that our method is preferred over almost all baselines in terms of "Editing" quality and is subjectively similar to P2P + NPI Prox. However, our method is better than P2P + NPI Prox in terms of preservation quality. To summarize, our method is consistent over a wide range of different edits and subjectively gives better results on editing quality while preserving source image regions that should not be edited. Details of the user study can be found in the Appendix H.

- Table 2: User study that evaluates the subjective preference of our method that is preferred over all baselines.

Method Editing (Q1) Preservation (Q2)

MasaCtrl [2] 85 % 70 % ProxMasaCtrl [5] 82 % 63 % P2P [6] + NTI [15] 60 % 49 % P2P [6] + NPI [14] 59 % 58 % P2P [6] + NPI [14] Prox [5] 50 % 55 % PnP [24] 60 % 61 % EDICT [27] 56 % 59 %

### 5 Conclusion

In this paper, we propose a novel image editing method Guide-and-Rescale based on the self-guidance mechanism and show its effectiveness for a variety of editing types. It significantly improves the trade-off between editing quality and original image preservation. We show that it achieves more consistent and high-quality results than existing approaches both qualitatively and quantitatively.

### Acknowledgements

The analysis of related work in sections 1 and 2 were obtained by Aibek Alanov with the support of the grant for research centres in the field of AI provided by the Analytical Center for the Government of the Russian Federation (ACRF) in accordance with the agreement on the provision of subsidies (identifier of the agreement 000000D730321P5Q0002) and the agreement with HSE University No. 70-2021-00139. This research was supported in part through computational resources of HPC facilities at HSE University.

### References

- 1. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: CVPR (2023)
- 2. Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., Zheng, Y.: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 22560–22570 (October 2023)
- 3. Choi, Y., Uh, Y., Yoo, J., Ha, J.W.: Stargan v2: Diverse image synthesis for multiple domains. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2020)
- 4. Epstein, D., Jabri, A., Poole, B., Efros, A.A., Holynski, A.: Diffusion self-guidance for controllable image generation. arXiv preprint arXiv:2306.00986 (2023)
- 5. Han, L., Wen, S., Chen, Q., Zhang, Z., Song, K., Ren, M., Gao, R., Stathopoulos, A., He, X., Chen, Y., Liu, D., Zhangli, Q., Jiang, J., Xia, Z., Srivastava, A., Metaxas, D.: Improving tuning-free real image editing with proximal guidance

(2023)

- 6. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022)
- 7. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. arXiv preprint arxiv:2006.11239 (2020)
- 8. Ho, J., Salimans, T.: Classifier-free diffusion guidance (2022)
- 9. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 4396–4405 (2019)
- 10. Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., Irani, M.: Imagic: Text-based real image editing with diffusion models. In: Conference on Computer Vision and Pattern Recognition 2023 (2023)
- 11. Kingma, D.P., Welling, M.: Auto-Encoding Variational Bayes. In: 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 1416, 2014, Conference Track Proceedings (2014)
- 12. Lin, T.Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C.L., Dollár, P.: Microsoft coco: Common objects in context

(2015)

- 13. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: SDEdit: Guided image synthesis and editing with stochastic differential equations. In: International Conference on Learning Representations (2022)

- 14. Miyake, D., Iohara, A., Saito, Y., Tanaka, T.: Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models (2023)
- 15. Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794 (2022)
- 16. Pan, Z., Gherardi, R., Xie, X., Huang, S.: Effective real image editing with accelerated iterative diffusion inversion (2023)
- 17. Parmar, G., Kumar Singh, K., Zhang, R., Li, Y., Lu, J., Zhu, J.Y.: Zero-shot image-to-image translation. In: ACM SIGGRAPH 2023 Conference Proceedings. SIGGRAPH ’23, Association for Computing Machinery, New York, NY, USA

(2023). https://doi.org/10.1145/3588432.3591513, https://doi.org/10. 1145/3588432.3591513

- 18. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)
- 19. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1(2), 3 (2022)
- 20. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022)
- 21. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022)
- 22. Seitzer, M.: pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/ pytorch-fid (August 2020), version 0.3.0
- 23. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. In: International Conference on Learning Representations (2021), https://openreview.net/forum? id=St1giarCHLP
- 24. Tumanyan, N., Geyer, M., Bagon, S., Dekel, T.: Plug-and-play diffusion features for text-driven image-to-image translation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1921–1930 (June 2023)
- 25. Vahdat, A., Kreis, K., Kautz, J.: Score-based generative modeling in latent space. In: Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems. vol. 34, pp. 11287–11302. Curran Associates, Inc. (2021), https://proceedings.neurips.cc/paper_files/ paper/2021/file/5dca4c6b9e244d24a30b4c45601d9720-Paper.pdf
- 26. Valevski, D., Kalman, M., Matias, Y., Leviathan, Y.: Unitune: Text-driven image editing by fine tuning an image generation model on a single image. ArXiv abs/2210.09477 (2022), https://api.semanticscholar.org/CorpusID: 252968124
- 27. Wallace, B., Gokul, A., Naik, N.: Edict: Exact diffusion inversion via coupled transformations. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 22532–22541. IEEE Computer Society, Los Alamitos, CA, USA (jun 2023). https://doi.org/10.1109/CVPR52729.2023.02158, https://doi.ieeecomputersociety.org/10.1109/CVPR52729.2023.02158

- 28. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)
- 29. Zhang, Z., Han, L., Ghosh, A., Metaxas, D., Ren, J.: Sine: Single image editing with text-to-image diffusion models. arXiv preprint arXiv:2212.04489 (2022)

## Appendix

### A Overall Pipeline

The proposed method is summarized in Alg. 1.

We suggest setting the CFG scale equal standard sampling guidance scale for Stable Diffusion w = 7.5. For non-stylisation tasks, we set Self-attention Guider scale vself = 300000 and Feature Guider scale vfeat = 500, while for stylisation editing optimal guidance scales equal vself = 100000,vfeat = 2.5. Besides, we suggest disabling guiders for several last steps of synthesis for more stable editing. For this purpose, we define a threshold τ, denoting the number of synthesis steps when guiders are used. For non-stylisation tasks τ = 35, and for stylisation: τ = 25.

Noise rescaling boundaries for non-stylisation are suggested as rlower = 0.33, rupper = 3, while for stylisation task we recommend rfixed := rupper = rlower = 1.5.

Regarding notations in Alg. 1, DDIM sampling formula takes form of: DDIM Sample(zt,ϵt) = atzt + btϵt. (17)

DDIM inversion formula stays unchanged, as defined in Equation 2. It is also important to note, that coefficients used in Equation 17 and Equation 2 are defined as follows:

at = αt−1/αt, bt = √αt−1 1/αt−1 − 1 − 1/αt − 1 ,

(18)

a∗t = αt+1/αt, b∗t = √αt+1 1/αt+1 − 1 − 1/αt − 1 .

We use DDIM with T = 50 inner steps. For simplicity, we eliminate dependency on zt,zt∗,t,ysrc from guiders in Equa-

- tion 11 and Equations 12, 13 and obtain gself and gfeat correspondingly. Besides, we omit dependency on current timestep t from noise rescaling def-

inition in Equation 16 and reformulate it as:

 

rl · rcur, 1/rcur ≤ rl 1, rl < 1/rcur < ru ru · rcur, 1/rcur ≥ ru

. (19)

fγ(rl,ru,rcur) =



### B Guiders

In Sec. 3.2 we propose Self-attention Guider and Feature Guider.

Self-attention Guider. In Fig. 6 we show examples of applying this guider. We use this guider alone with CFG, and visualize the results of editing with different guidance scales v:

Algorithm 1 Guide-and-Rescale for Real Image Editing

Input: Real image xinit, source prompt ysrc, target prompt ytrg; DDIM steps T; guidance scales w, vself, vfeat; threshold τ; noise rescaling boundaries rlower, rupper. Function: VAE encoder Enc., VAE decoder Dec., DDIM Inversion (Equation 2), DDIM Sample (Equation 17), Self-attention Guider gself (Equation 11), Feature Guider gfeat (Equations 12, 13), noise rescaling fγ (Equation 19).

Output: Edited image xedit.

- 1: z0∗ = Enc.(xinit)
- 2: for t = 0, 1, . . . , T − 1 do
- 3: zt∗+1 = DDIM Inversion(zt∗, ysrc)
- 4: end for
- 5: zT = zT∗
- 6: for t = T, T − 1, . . . , 1 do
- 7: ∆cfg = w(εθ(zt, t, ytrg) − εθ(zt, t, ∅))
- 8: ϵcfg = εθ(zt, t, ∅) + ∆cfg
- 9: {A∗iself}Li=1, Φ∗ = εθ(zt∗, t, ysrc)
- 10: {A¯selfi }Li=1, Φ¯ = εθ(zt, t, ysrc)
- 11: ϵself = vself · gself({A∗iself}Li=1, {A¯selfi }Li=1)
- 12: ϵfeat = vfeat · gfeat(Φ∗, Φ¯)
- 13: rcur = ∥∆cfg∥22/∥∇zt(ϵself + ϵfeat)∥22
- 14: γ = fγ(rlower, rupper, rcur)
- 15: if T − t < τ then
- 16: ϵfinal = ϵcfg + γ · ∇zt(ϵself + ϵfeat)
- 17: else
- 18: ϵfinal = ϵcfg
- 19: end if
- 20: zt−1 = DDIM Sample(zt, ϵfinal)
- 21: end for
- 22: xedit = Dec.(z0) Return: xedit

ϵfinal = CFG(zt,t,ytrg,7.5) + v · ∇gself({A∗iself}Li=1,{A¯selfi }Li=1). (20)

This guider is formally defined in Equation 11. We calculate L2 norm of the difference of self-attention maps from the current sampling trajectory, conditioning on ysrc, and the reference inversion trajectory, where by L2 norm we mean the following:

∥X − Y ∥22 = (X − Y )2, (21)

where arithmetic operations are applied element-wise to the matrices of the same shape.

We take the mean of the resulting matrix and then sum these results over all UNet layers.

This guider preserves image layout, i.e. relative positioning of objects. In Fig. 8 and Fig. 9 we prove this statement by visualizing three leading principal

- (a) Non-stylisation editing example.

[Figure 9]

- (b) Stylisation editing example.

- Fig. 6: Illustration of applying Self-attention Guider. Self-attention Guider is defined in Equation 11.

components of self-attention maps. The visualization shows, that the colors are aligned with objects in the picture.

Feature Guider. This guider is defined differently for stylisation and nonstylisation editing tasks. It preserves the visual appearance of the whole picture. We provide examples of incorporating this guider into the editing pipeline in Fig. 7. As in Fig. 6, we show results of editing with CFG and Feature Guider, using different guidance scales:

ϵfinal = CFG(zt,t,ytrg,7.5) + v · ∇gfeat(Φ∗,Φ¯). (22) Feature Guider for non-stylisation. Formal definition is provided in Equa-

- tion 12. L1 norm in this guider is calculated similarly to Equation 21: ∥X − Y ∥1 = abs(X − Y ). (23)

For non-stylisation editing, we consider the output of the last decoder layer in UNet as features. We match features from the current sampling trajectory, conditioning on ysrc: εθ(zt,t,ysrc), and the reference inversion trajectory εθ(zt∗,t,ysrc). We provide visualization of these features in Fig. 10 and Fig. 11. For visualization purposes, we take the mean over feature dimension of these features, obtaining a matrix with spatial dimensions only, and visualize these matrices in Fig. 10. In Fig. 11, we show three leading principal components of features. This figure shows, that the same colors correspond to specific objects in the picture. However, the features are too noisy during the first synthesis steps, so this pattern is hardly distinguishable.

Feature Guider for stylisation. Formal definition is provided in Equation 13. This formula uses L2 norm defined in Equation 21.

For stylisation editing, we consider the output of the second ResNet block in the second decoder layer in UNet as features. In contrast to non-stylisation

(a) Non-stylisation editing example. Feature Guider is defined in Equation 12.

[Figure 11]

(b) Stylisation editing example. Feature Guider is defined in Equation 13.

Fig. 7: Illustration of applying Feature Guider.

features, for stylisation tasks, we define the current sampling trajectory as the one, conditioning on ytrg: εθ(zt,t,ytrg).

These features are visualized in Fig. 12 and Fig. 13.

[Figure 12]

(a) Self-attention maps {A¯selfi } from the current trajectory εθ(zt, t, ysrc)

[Figure 13]

(b) Self-attention maps {Ai∗self} from the reference inversion trajectory εθ(zt∗, t, ysrc)

- Fig. 8: Visualization of self-attention maps from Self-attention Guider (Equation 11) for non-stylisation editing example.

[Figure 14]

(a) Self-attention maps {A¯selfi } from the current trajectory εθ(zt, t, ysrc)

[Figure 15]

(b) Self-attention maps {Ai∗self} from the reference inversion trajectory εθ(zt∗, t, ysrc)

- Fig. 9: Visualization of self-attention maps from Self-attention Guider (Equation 11) for stylisation editing example.

[Figure 16]

(a) Features Φ¯ from the current trajectory εθ(zt, t, ysrc)

[Figure 17]

(b) Features Φ∗ from the reference inversion trajectory εθ(zt∗, t, ysrc)

- Fig. 10: Visualization of features from Feature Guider (Equation 12) for non-stylisation editing example. We take mean over feature dimension of features.

[Figure 18]

(a) Features Φ¯ from the current trajectory εθ(zt, t, ysrc)

[Figure 19]

(b) Features Φ∗ from the reference inversion trajectory εθ(zt∗, t, ysrc)

- Fig. 11: Visualization of features from Feature Guider (Equation 12) for non-stylisation

[Figure 20]

(a) Features Φ¯ from the current trajectory εθ(zt, t, ytrg)

[Figure 21]

(b) Features Φ∗ from the reference inversion trajectory εθ(zt∗, t, ysrc)

- Fig. 12: Visualization of features from Feature Guider (Equation 13) for stylisation editing example. We take mean over feature dimension of features.

[Figure 22]

(a) Features Φ¯ from the current trajectory εθ(zt, t, ytrg)

[Figure 23]

(b) Features Φ∗ from the reference inversion trajectory εθ(zt∗, t, ysrc)

- Fig. 13: Visualization of features from Feature Guider (Equation 13) for stylisation

### C Noise Rescaling

In Section 3.2 we define a noise rescaling technique that allows to control the balance between editing and preserving the content. We add Feature and Selfattention guiders to preserve the structure and the details of the image and use classifier-free guidance for editing. The greater the contribution of the guider in the sum of gradient norms, the greater the effect of that guider. To control the distribution of gradient norms, we add conditions on the gradient norm of guiders based on the gradient norm of CFG.

At each iteration, we compute the relation rcur(t) between the gradient norms and then we evaluate the scaling factor γ = r · rcur(t) as in Equation 15. We propose two approaches to choosing r.

Fixed noise rescaling. The first approach is to fix r as some constant, for example take rfixed = 1.5. This means that we rescale the gradient of the guiders in such a way that its norm is the scaled gradient norm of CFG at each iteration. For stylisation this is sufficient and it is shown in Fig. 15, but for other types of editing, we propose using a more complicated noise rescaler.

Noise rescaling in range. Another way to rescale the noise is the fix r in some range. The idea is to keep the relation between the sum of the gradient norms of the guiders and the CFG gradient norm, which is equal to 1/rcur in the range [rl,ru] and it means that rcur should be in the range [1/ru,1/rl]. Hence, r is

 

rl, 1/rcur ≤ rl 1/rcur, rl < 1/rcur < ru ru, 1/rcur ≥ ru

. (24)

r =



The scaling factor is then calculated similarly to Equation 19. In Fig. 14 we show the difference between editing without rescaling, with fixed rescaling, and with rescaling in range. We show in Fig. 15 that we do not need to use rescaling in range for stylisation because the gradient norm of guiders is always less than the range and is clipped at the bottom.

It is worth mentioning, that by simply setting rl = ru in Equation 24 we can transform noise rescaling in range into fixed noise rescaling.

### D Testing Datasets Description

We compared our pipeline for both local and global editing with existing approaches such as NTI [15], NPI [14] and NPI Prox [5] based on P2P [6], MasaCtrl [2], ProxMasaCtrl [5], PnP [24], and EDICT [27]. We tested methods on several datasets. Section D.1 describes our custom dataset consisting of four types of edits that we consider in our experiments. For each type, we manually assembled a dataset of 20 images. We also compared the methods on large datasets such as CoCo (Sec. D.2), AFHQ (Sec. D.3) and Wild-FFHQ (Sec. D.4).

- All results of these experiments are attached to the supplementary materials as separate pdf files.

[Figure 24]

- Fig. 14: Illustration of applying different noise rescalings. Without rescaling there are some artifacts, with fixed rescaling the editing is not sufficiently visible, and with rescaling in a range the editing has a high quality.

#### D.1 Custom Dataset Edit Types

Animal to animal. This is a local type of editing. The goal of this editing task is to change the depicted animal into a different one, preserving the background and the silhouette of the animal. Our test set contains pictures of various animals (cat, dog, tiger, horse, deer), some of which have very specific features (for example, a deer has antlers). We consider changing an animal into the one, having a naturally similar shape (for example, changing a tiger into a panther), making complicated swaps (for example, changing a deer into a cat), and changing the color of the animal with preserving the majority of the features (changing a white horse into a black one). Visual comparison for some of these examples is reported in Fig. 16.

Face in the wild. This is a local type of editing. In our setting, this task aims at changing the emotion of the person in the picture and preserving the background and the appearance of the person (meaning that the person in the edited picture should still resemble the person in the initial image). We conduct experiments on photographs of different people, taken from different perspec-

Source prompt: "A photo of a cat sitting next to a mirror" Target prompt: "A Picasso painting of a cat sitting next to a mirror"

Edited image (w/o rescaling)

Edited image (w/ fixed rescaling)

Guiders gradient norm (w/ rescaling in range)

Initial image

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

cfg

70

other range

60

| |
|---|

50

40

30

20

10

0

50 40 30 20 10 0

Source prompt: "a couple dancing" Target prompt: "A cartoon of a couple dancing"

Edited image (w/o rescaling)

Edited image (w/ fixed rescaling)

Guiders gradient norm (w/ rescaling in range)

Initial image

80

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

cfg

other range

70

| |
|---|

60

50

40

30

20

10

0

50 40 30 20 10 0

- Fig. 15: Illustration of applying different noise rescalings for stylisation. In the right plot, we show obtained gradient norms after rescaling in range and possible range for our guiders summary gradient norm.

tives. Our initial images depict people with different emotions as well (neutral and smiling). However, we still obtain good results in this task by not specifying the emotion in the source prompt at all. Every source prompt has a form of “A photo of a {man/woman}". The target prompt can be easily obtained from the source prompt by adding the desired emotion (for example, happy, smiling, crying) prior to the word “{man/woman}". Visual comparison for some of these examples is reported in Fig. 17.

Stylisation. This is a global type of editing. We tested the methods on different drawing styles (sketch, oil, watercolor), artist stylisations (Van Gogh, Picasso, Fernando Botero), changing the texture of objects (wooden statues and sculptures), changing face styles (anime, pixar, pop art). Visual comparison for some of these examples is reported in Fig. 19.

[Figure 31]

A photo of a deer −→ A photo of a antelope

A photo of a horse −→ A photo of a donkey

A photo of a dog −→ A photo of a horse

A photo of a cat sitting next to a mirror −→ A photo of a dog sitting next to a mirror

A photo of a deer −→ A photo of a cat

##### Fig. 16: Visual comparison of our method vs baselines. The examples are from Custom Dataset for ‘animal-2-animal‘ type of editing.

Original Image EDICT MasaCtrl ProxMasaCtrl P2P + NTI P2P + NPI P2P + NPI Prox PnP Ours

[Figure 32]

A photo of a man −→ A photo of an angry man

A photo of a man −→ A photo of a surprised man

A photo of a man −→ A photo of a frightened man

A photo of a woman −→ A photo of a smiling woman

A photo of a woman −→ A photo of a frightened woman

##### Fig. 17: Visual comparison of our method vs baselines. The examples are from Custom Dataset for ‘face-in-the-wild‘ type of editing.

[Figure 33]

A photo of a person in a black tuxedo −→ A photo of a person in a green tuxedo

A photo of a man wearing a grey hoodie −→ A photo of a man wearing a blue hoodie

A photo of a baby wearing ... −→ A photo of a baby wearing a long sleeve lying ...

A photo of a man in a checkered shirt −→ A photo of a blonde man in a checkered shirt

A photo of a woman with b ... −→ A photo of a woman with green hair wearing a g ...

##### Fig. 18: Visual comparison of our method vs baselines. The examples are from Custom Dataset for ‘person-in-the-wild‘ type of editing.

Original Image EDICT MasaCtrl ProxMasaCtrl P2P + NTI P2P + NPI P2P + NPI Prox PnP Ours

[Figure 34]

a person −→ sculpture of a person

a couple dancing −→ Watercolor drawing of a couple dancing

A photo of a cat sitting next to a mirror −→ A Picasso painting of a cat sitting next to a mirror

a photo −→ anime style face

a horse −→ A wooden sculpture of a horse

##### Fig. 19: Visual comparison of our method vs baselines. The examples are from Custom Dataset for ‘stylisation‘ type of editing..

#### D.2 Stylisation - CoCo dataset

The MS COCO (Microsoft Common Objects in Context) dataset [12] is a largescale dataset with 164K high-resolution images and annotations to them. We use a subset of 500 images from this dataset for the stylisation type of editing. Each testing object is constructed with a randomly picked image. The source prompt is the corresponding annotation for the picked image. The target prompt is constructed by concatenating the random style from the list of styles with the source prompt.

We randomly choose a style for each image from the list: ’Anime Style’, ’A pop art style’, ’A pixar style’, ’A Van Gogh painting of’, ’A Fernando Botero painting of’, ’A Ukiyo-e painting of’, ’A Picasso painting of’, ’A charocal painting of’, ’An oil painting of’, ’A sketch of’, ’A cubism painting of’, ’An impressionist painting of’, ’A watercolor drawing of’, ’A minecraft painting of’, ’Banksy art of’, ’da Vinci sketch of’, ’A mosaic depicting of’, ’A gothic painting of’, ’A geometric abstract painting of’, ’A white wool of’, ’A futurism painting of’, ’A Pixel art style’, ’Comic book style’, ’Cyberpunk style’, ’Flat style’.

We show visual comparison in the Fig. 20.

###### Original Image EDICT Ours ProxMasaCtrl P2P + NTI P2P + NPI P2P + NPI Prox PnP Ours

[Figure 35]

A dirty old red toilet next to many wires. −→ Anime Style A dirty old red toilet next to many wires.

A man riding a skateboard on a ramp. −→ A Van Gogh painting of A man riding a skateboard on a ramp.

A table with pies being m ... −→ Comic book style A table with pies being made ...

Man with yellow helmet riding a yellow motorcycle −→ A geometric abstract painting of Man with yellow helmet riding a yellow motorcycle

an old fashion double ove ... −→ A cubism painting of an old fashion double ove ...

- Fig. 20: Visual comparison of our method vs baselines. The examples are from MS COCO dataset for ‘stylisation‘ type of editing.

#### D.3 Dog2Cat - AFHQ

Animal FacesHQ (AFHQ) [3] is a dataset of 15k high-quality images at 512x512 resolution of animal faces. It contains three types of animals: cats, dogs, and wild animals. We use a subset of 500 images of dogs from this dataset for the dog-to-cat type of editing. Each testing object is constructed with a randomly picked image. We use ’a dog’ as a source prompt and ’a cat’ as a target prompt for the transformation of dogs into cats. We report FID in Table 1. To evaluate one we use a subset of 5k images of cats to compare the generated distribution and the original one.

We show visual comparison in the Fig. 21.

Original Image EDICT MasaCtrl ProxMasaCtrl P2P + NTI P2P + NPI P2P + NPI Prox PnP Ours

[Figure 36]

a dog −→ a cat

a dog −→ a cat

a dog −→ a cat

a dog −→ a cat

a dog −→ a cat

Fig. 21: Visual comparison of our method vs baselines. The examples are from AFHQ dataset for ‘dog-to-cat‘ type of editing.

#### D.4 Change of emotions - Wild FFHQ

We use a subset of 500 images from FFHQ dataset [9]. Each testing object is constructed with a wild face example from FFHQ. The source prompt is "A photo of a person". To construct the target prompt we used the following procedure: we randomly picked one of the emotion descriptions and inserted it into the prompt.

The set of emotion descriptions: ’frightened’, ’laughing’, ’shy’, ’surprised’, ’smiling’, ’crying’, ’angry’, ’sad’, ’happy’, ’emotionless’, ’disgusted’, ’painful’, ’thoughtful’, ’worried’, ’curious’

For that dataset, we show visual comparison in Fig. 22.

###### Original Image EDICT Ours ProxMasaCtrl P2P + NTI P2P + NPI P2P + NPI Prox PnP Ours

[Figure 37]

A photo of a person −→ A photo of a laughing person

A photo of a person −→ A photo of a happy person

A photo of a person −→ A photo of a surprised person

A photo of a person −→ A photo of a worried person

A photo of a person −→ A photo of a smiling person

- Fig. 22: Visual comparison of our method vs baselines. The examples are from Wild FFHQ dataset for ‘face-in-the-wild‘ type of editing.

### E Method Setup for Experiments

Method Name Used Implementation Special Parameters

self_attn_block_indices = [4, 5, 6, 7, 8, 9, 10, 11] out_layers_output_block_indices = [4] feature_injection_threshold = 40 scale = 10.

Plug-and-Play PnP github-repo

mix_weight = 0.93 init_image_strength = 0.8 guidance_scale = 3 MasaCTRL MasaCTRL github-repo

Edict Edict github-repo

STEP = 4 LAYPER = 10

npi = True npi_interp = 1 prox = ’l0’ quantile = 0.6 masa_step = 4 masa_layer = 10

ProxMasaCTRL Prox github-repo

cross_replace_steps = 0.8 self_replace_steps = 0.6 blend_word = None

NPI + p2p P2P github-repo

cross_replace_steps = 0.8 self_replace_steps = 0.6 blend_word = None prox = ’l0’ quantile = 0.6

NPI prox + p2p Prox github-repo

cross_replace_steps = 0.8 self_replace_steps = 0.6 blend_word = None opt_early_stop_epsilon (NullInv) = 1e-5 opt_num_inner_steps (NullInv) = 10

Null-Text + p2p P2P github-repo

- Table 3: Special arguments that were used while inference of baseline methods when comparing. Also, source repositories are provided.

### F Method Setup for Experiments

All experiments on methods comparison were conducted using authors’ official repositories except NPI [14] - for this method modified p2p code has been used. For each editing type of test example, default parameters were used. All the corresponding code implementation and special parameters of the method’s inference can be found in Table 3.

The setup for our method can be found in Sec. A.

### G Quantitative Evaluation

For quantitative comparison, paired-metric comparison is provided. That comparison is previously used in EDICT method [27].

We quantitatively benchmark all methods on 4 test datasets: Custom Dataset, CoCo, Wild-FFHQ, AFHQ. An exact explanation of the way we were using each dataset can be found in Sec. D.

LPIPS [28] metric is used to analyze original image preservation properties, CLIP score [18] is used to analyze semantic similarity to desired editing description. The motivation for using LPIPS metric is in the perception of one to visual changes. In common editing problems (except stylisation) there is a requirement to save non-editing areas unchanged so low values of LPIPS indicate that the method preserves the original structure of the image better. The motivation for using CLIP score is in its perception of the similarity of editing results to target editing description. As it follows by the motivation of metrics, the effective method will have lower LPIPS and higher CLIP score. Results for the custom dataset are shown in Table 1. For CoCo results are shown in Table 5. For FFHQ results are shown in Table 6.

Our method greatly outperforms [2,5] in both metrics, and confirmation of it is in visual comparisons. Compared to p2p method [6,14,15] our method is also superior in both metrics but with less gap, such behavior is aligned with visual results, as p2p works better than MasaCTRL. Compared to Edict [27] our method has comparable LPIPS value but vastly superior CLIP score value. In contrast, compared to PnP [24] our method has a comparable CLIP score value but a significantly better LPIPS value. To prove the superiority of our method over others we propose to average rank for each metric and analyze its average rank. We have introduced a metric called AverageRank to show comparison based on a single value, which is a final indicator of the quality of the method. That average ranking shows the superiority of our method over other baselines. Results are shown in Table 4.

For the CoCo-‘stylisation‘ problem there our method shows poor LPIPS because our result shows better stylisation and loses identical spatial edges that make LPIPS less. For the FFHQ problem, our method shows an optimal LPIPS score while having quite high CLIP score.

Also, we are examining our method on classic dog-to-cat problem. To estimate the quality of our method and compare it with other baselines we used data described in Sec. D.3. The main results on FID score are reported in Table 1. Our method outperforms all other baseline methods.

Method Name LPIPS Rank ↓ CLIPscore Rank ↓ FID Rank ↓ Average Rank ↓ Guide-and-Rescale (ours) 3 (0.228) 2 (0.243) 1 (39.07) 2.0

P2P [6] + NPI [14] Prox [5] 1 (0.170) 4 (0.233) 4 (43.16) 3.0

PnP [24] 8 (0.366) 1 (0.256) 2 (39.55) 3.7 P2P [6] + NPI [14] 4 (0.251) 3 (0.234) 5 (44.05) 4.0

EDICT [27] 2 (0.221) 6 (0.229) 6 (47.13) 4.7 P2P [6] + NTI [15] 6 (0.279) 5 (0.233) 3 (42.46) 4.7

ProxMasaCtrl [5] 5 (0.267) 8 (0.215) 7 (94.53) 6.7 MasaCtrl [2] 7 (0.306) 7 (0.223) 8 (100.62) 7.3

- Table 4: Comparison overall methods based on introduced AverageRank metric. Each rank shows the method’s position in a sorted metrics array. In brackets, there are the exact values of the corresponding metric.

Table 5: Comparison with baselines on CoCo for stylisation editing task.

Method LPIPS ↓ CLIP ↑

MasaCtrl [2] 0.300 0.240 ProxMasaCtrl [5] 0.227 0.231 P2P [6] + NTI [15] 0.338 0.272 P2P [6] + NPI [14] 0.394 0.265 P2P [6] + NPI [14] Prox [5] 0.208 0.263 EDICT [27] 0.273 0.253 PnP [24] 0.378 0.286

Guide-and-Rescale (ours) 0.449 0.290

- Table 6: Comparison with baselines on FFHQ for ‘change of emotions‘ editing task.

Method LPIPS ↓ CLIP ↑

MasaCtrl [2] 0.346 0.190 ProxMasaCtrl [5] 0.281 0.184 P2P [6] + NTI [15] 0.120 0.171 P2P [6] + NPI [14] 0.249 0.175 P2P [6] + NPI [14] Prox [5] 0.134 0.171 EDICT [27] 0.158 0.174 PnP [24] 0.251 0.190

Guide-and-Rescale (ours) 0.156 0.183

### H User Study

We have provided user studies experiment with the dataset described in Sec. D.1. For each method mentioned in Table 2 you can find inference hyperparameters described in Sec. F.

#### H.1 Experimental setup

For each user, there were shown the original image, the description of the desired editing, the result of our method, and the result of the baseline method.

Users had to answer two questions. The first question (Q1 - Edit property) is "Which image matches the desired editing description best?". The second question (Q2 - Preservation property) is "Which image preserves the overall structure of the ‘Original Image‘ best?". The user task consisted of three comparisons.

For each object from the dataset, there were seven pairs of comparison: Ours vs each of the baseline. To decrease user’s variance and improve the stability of user studies we set overlap for each unique testing object to 3. This means that every unique comparison was shown to 3 unique users. Also to decrease bias of user study results number of attempts for tasks for each user was limited to 6.

#### H.2 Metrics per editing type of Custom Dataset

Results for all types of editing are shown in Table 2. Results over each editing type are shown in Table [7, 8, 9, 10].

- Table 7: User studies that show the percentage of users that preferred our method. The current table shows a comparison on the subset of results that belong to face-inthe-wild editing type.

Method Preservation Editing

P2P [6] + NTI [15] 40 % 73 % P2P [6] + NPI [14] Prox [5] 32 % 47 % P2P [6] + NPI [14] 67 % 83 % PnP [24] 72 % 89 % EDICT [27] 68 % 68 % ProxMasaCtrl [5] 67 % 83 % MasaCtrl [2] 75 % 25 %

- Table 8: User studies that show the percentage of users that preferred our method. The current table shows the comparison on the subset of results that belong to animal2-animal editing type.

Method Preservation Editing

P2P [6] + NTI [15] 67 % 59 % P2P [6] + NPI [14] Prox [5] 92 % 67 % P2P [6] + NPI [14] 73 % 73 % PnP [24] 50 % 58 % EDICT [27] 80 % 67 % ProxMasaCtrl [5] 67 % 81 % MasaCtrl [2] 29 % 71 %

- Table 9: User studies that show the percentage of users that preferred our method. The current table shows the comparison on the subset of results that belong to stylisation editing type.

Method Preservation Editing

P2P [6] + NTI [15] 33 % 52 % P2P [6] + NPI [14] Prox [5] 64 % 36 % P2P [6] + NPI [14] 46 % 62 % PnP [24] 69 % 62 % EDICT [27] 50 % 33 % ProxMasaCtrl [5] 50 % 83 % MasaCtrl [2] 76 % 92 %

- Table 10: User studies that show the percentage of users that preferred our method. The current table shows the comparison on the subset of results that belong to personin-the-wild editing type.

Method Preservation Editing

P2P [6] + NTI [15] 50 % 58 % P2P [6] + NPI [14] Prox [5] 50 % 50 % P2P [6] + NPI [14] 52 % 33 % PnP [24] 61 % 33 % EDICT [27] 42 % 58 % ProxMasaCtrl [5] 73 % 82 % MasaCtrl [2] 100 % 100 %

