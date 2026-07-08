##### HD-Painter: High-Resolution and Prompt-Faithful Text-Guided Image Inpainting with Diffusion Models

# arXiv:2312.14091v3[cs.CV]18Mar2024

Hayk Manukyan1∗ Andranik Sargsyan1∗ Barsegh Atanyan1 Zhangyang Wang1,2 Shant Navasardyan1 Humphrey Shi1,3

1Picsart AI Research (PAIR) 2UT Austin 3Georgia Tech https://github.com/Picsart-AI-Research/HD-Painter

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

flowers

shirt

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

castle

lion

Figure 1. High-resolution (the large side is 2048 in all these examples) text-guided image inpainting results with our approach. The method is able to faithfully fill the masked region according to the prompt even if the combination of the prompt and the known region is highly unlikely. Zoom in to view high-resolution details.

###### Abstract

Recent progress in text-guided image inpainting, based on the unprecedented success of text-to-image diffusion models, has led to exceptionally realistic and visually plausible results. However, there is still significant potential for improvement in current text-to-image inpainting models, particularly in better aligning the inpainted area with user prompts and performing high-resolution inpainting. Therefore, we introduce HD-Painter, a training-free approach

*Equal contribution.

that accurately follows prompts and coherently scales to high resolution image inpainting. To this end, we design the Prompt-Aware Introverted Attention (PAIntA) layer enhancing self-attention scores by prompt information resulting in better text aligned generations. To further improve the prompt coherence we introduce the Reweighting Attention Score Guidance (RASG) mechanism seamlessly integrating a post-hoc sampling strategy into the general form of DDIM to prevent out-of-distribution latent shifts. Moreover, HD-Painter allows extension to larger scales by introducing a specialized super-resolution technique customized for inpainting, enabling the completion of missing regions

in images of up to 2K resolution. Our experiments demonstrate that HD-Painter surpasses existing state-of-the-art approaches quantitatively and qualitatively across multiple metrics and a user study. Code is publicly available at: https://github.com/Picsart-AI-Research/ HD-Painter .

###### 1. Introduction

The recent wave of diffusion models [11, 34] has taken the world by storm, becoming an increasingly integral part of our everyday lives. After the unprecedented success of textto-image models [26, 27, 30, 37] diffusion-based image manipulations such as prompt-conditioned editing [3, 10], controllable generation [19, 43], personalized and specialized image synthesis [9, 17, 29] became hot topics in computer vision leading to a huge amount of applications. Particularly, text-guided image completion or inpainting [1, 36, 37] allows users to generate new content in user-specified regions of given images based on textual prompts (see Fig. 1), leading to use cases like retouching specific areas of an image, replacing or adding objects, and modifying subject attributes such as clothes, colors, or emotion.

Pretrained text-to-image generation models such as Stable Diffusion [27], Imagen [30], and Dall-E 2 [26] can be adapted for image completion by blending diffused known regions with generated (denoised) unknown regions during the backward diffusion process. Although such approaches [1, 2] produce visually plausible completions, they are not well harmonized and lack global scene understanding, especially when denoising in high diffusion timesteps.

To address this, existing methods [21, 23, 27, 30], modify pretrained text-to-image models to take additional context information and fine-tune specifically for text-guided image completion. GLIDE [21] and Stable Inpainting [27] concatenate the mask and the masked image as additional channels to the input of the diffusion UNet, initializing the new convolutional weights with zeros, then fine tune the modified model using random masks together with the initial prompt.

However, SmartBrush [38] and Imagen Editor [36] mention the weak image-text alignment of such models, attributing it to the random masking strategies, and the misalignment of the global prompts used during training with the local context of the masked region. In this paper, we will address this issue as prompt neglect. To alleviate this problem, both papers introduce novel, object-aware masking strategies. Additionally SmartBrush proposes BLIP captioning approach, to ensure a better alignment of the inpainting prompt with the masked region. Nonetheless, we find that while this approach reduces the amount of prompt neglect, it also decreases the generation quality Tab. 1.

We noticed that prompt neglect is commonly expressed

in two ways: either the model fills in the masked region with background (background dominance, Fig. 4, columns 1, 3, 5), or the model completes a nearby object partially occluded by the mask (nearby object dominance, Fig. 4, columns 2, 4, 6). In both cases the issue seems to be caused by the model preferring the local context of the known region to the textual information provided by the prompt.

To address the mentioned problems we introduce Prompt-Aware Introverted Attention (PAIntA) block without any training or fine-tuning requirements. PAIntA enhances the self-attention scores according to the given textual condition aiming to decrease the impact of non-prompt-relevant information from the image known region while increasing the contribution of the prompt-aligned known pixels.

To improve the text-alignment of the generation results even further we apply a post-hoc guidance mechanism by leveraging the cross-attention scores. However the vanilla post-hoc guidance mechanism used by seminal works such as [6, 7], etc. may lead to generation quality degradation due to out-of-distribution shifts caused by the additional gradient term in the backward diffusion equation (see Eq. (4)). To this end we propose Reweighting Attention Score Guidance (RASG), a post-hoc mechanism seamlessly integrating the gradient component in the general form of DDIM process. This allows to simultaneously guide the sampling towards more prompt-aligned latents and keep them in their trained domain leading to visually plausible inpainting results.

With the combination of PAIntA and RASG our method gains a significant advantage over the current state-of-theart approaches by solving the issue of prompt neglect. In addition, by leveraging high-resolution diffusion models and time-iterative blending technology we design a simple yet effective pipeline for up to 2048 × 2048 resolution inpainting.

To summarize, our main contributions are as follows:

- • We introduce the Prompt-Aware Introverted Attention (PAIntA) layer to alleviate the prompt neglect issues of background and nearby object dominance in text-guided image inpainting.
- • To further improve the text-alignment of generation we present the Reweighting Attention Score Guidance (RASG) strategy which enables to prevent out-ofdistribution shifts while performing post-hoc guided sampling.
- • Our designed pipeline for text-guided image completion is training-free and demonstrates a significant advantage over current state-of-the-art approaches quantitatively and qualitatively. Moreover, with the additional help of our simple yet effective inpainting-specialized super-resolution framework we make high-resolution (up to 2048 × 2048) image completion possible.

###### 2. Related Work

###### 2.1. Image Inpainting

Image inpainting is the task of filling missing regions of the image in a visually plausible manner. Early deep learning approaches such as [20, 40, 41] introduce mechanisms to propagate deep features from known regions. Later [31, 39, 44, 45] utilize StyleGAN-v2-like [13] decoder and discriminative training for better image detail generation.

Image inpainting also benefited from diffusion models, particularly with the emergence of text-guided inpainting. Given a pre-trained text-to-image diffusion model [1, 2] replace the unmasked region of the latent by the noised version of the known region during sampling. However, as noted by [21], this leads to poor generation quality, as the denoising network only sees the noised version of the known region. [21, 23, 36, 38] fine-tune pretrained textto-image models for text-guided image inpainting by conditioning the denoising model on the inpainting mask and the known region, concatenating them with the input latents. [36, 38], in particular, use object-aware masking strategies, to improve image-text alignment of training samples. Alternatively, [43] obtains an inpainting model by attaching trainable modules to the UNet, while keeping the base model unchanged. We propose a training-free approach leveraging plug-and-play components PAIntA and RASG, improving text-prompt alignment. Moreover, our approach allows inpainting on high-resolution images (up to 2048 × 2048).

###### 2.2. Inpainting-Specific Architectural Blocks

Early deep learning approaches were designing special layers for better/more efficient inpainting. Particularly, [16, 20, 42] introduce special convolutional layers dealing with the known region of the image to effectively extract the information useful for visually plausible image completion. [40] introduces the contextual attention layer reducing the unnecessarily heavy computations of all-to-all selfattention for high-quality inpainting. In this work we propose Prompt-Aware Introverted Attention (PAIntA) layer, specifically designed for text-guided image inpainting. It aims to decrease (increase) the prompt-irrelevant (-relevant) information from the known region for better text aligned inpainting generation.

###### 2.3. Post-Hoc Guidance in Diffusion Process

Post-hoc guidance methods are backward diffusion sampling techniques which guide the next step latent prediction towards a specific objective function minimization. Such approaches appear to be extremely helpful when generating visual content especially with an additional constraint. Particularly [6] introduced classifier-guidance aiming to generate images of a specific class. Later CLIP-

guidance was introduced by [21] leveraging CLIP [25] as an open-vocabulary classification method. LDM [27] further extends the concept to guide the diffusion sampling process by any image-to-image translation method, particularly guiding a low-resolution trained model to generate ×2 larger images. [4] guides image generation by maximizing the maximal cross-attention score relying on multiiterative optimization process resulting in more text aligned results. [7] goes even further by utilizing the cross-attention scores for object position, size, shape, and appearance guidances. All the mentioned post-hoc guidance methods shift the latent generation process by a gradient term (see Eq. (6)) sometimes leading to image quality degradations.

To this end we propose the Reweighting Attention Score Guidance (RASG) mechanism allowing to perform posthoc guidance with any objective function while preserving the diffusion latent domain. Specifically for inpainting task, to alleviate the issue of prompt neglect, we benefit from a guidance objective function based on the openvocabulary segmentation properties of cross-attentions.

###### 3. Method

We first formulate the text-guided image completion problem followed by an introduction to diffusion models, particularly Stable Diffusion ([27]) and Stable Inpainting. We then discuss the overview of our method and its components. Afterwards we present our Prompt-Aware Introverted Attention (PAIntA) block and Reweighting Attention Score Guidance (RASG) mechanism in detail. Lastly our inpainting-specific super-resolution technique is introduced.

Let I ∈ RH×W×3 be an RGB image, M ∈ {0,1}H×W be a binary mask indicating the region in I one wants to inpaint with a textual prompt τ. The goal of text-guided image inpainting is to output an image Ic ∈ RH×W×3 such that Ic contains the objects described by the prompt τ in the region M while outside M it coincides with I, i.e. Ic⊙(1− M) = I ⊙ (1 − M).

###### 3.1. Stable Diffusion and Stable Inpainting

Stable Diffusion (SD) is a diffusion model that functions within the latent space of an autoencoder D(E(·)) (VQGAN [8] or VQ-VAE [35]) where E denotes the encoder and D the corresponding decoder. Specifically, let I ∈ RH×W×3 be an image and x0 = E(I), consider the following forward diffusion process with hyperparameters {βt}Tt=1 ⊂ [0,1]:

q(xt|xt−1) = N(xt; 1 − βtxt−1,βtI), t = 1,..,T (1)

where q(xt|xt−1) is the conditional density of xt given xt−1, and {xt}Tt=0 is a Markov chain. Here T is large enough to allow an assumption xT ∼ N(0,1). Then SD

learns a backward process (below similarly, {xt}0t=T is a Markov chain)

pθ(xt−1|xt) = N(xt−1;µθ(xt,t),σt1), t = T,..,1, (2)

and hyperparameters {σt}Tt=1, allowing the generation of a signal x0 from the standard Gaussian noise xT. Here µθ(xt,t) is defined by the predicted noise ϵtθ(xt) modeled as a neural network (see [11]): µθ(xt,t) =

√1−tαt ϵtθ(xt) . Then Iˆ = D(x0) is returned.

√1βt xt − β

The following claim can be derived from the main DDIM principle, Theorem 1 in [34].

CLAIM 1 After training the diffusion backward process (Eq. 2) the following {σt}Tt=1-parametrized family of DDIM sampling processes can be applied to generate highquality images:

√1 − αtϵtθ(xt) √αt

xt −

xt−1 = √αt−1

+

(3)

1 − αt−1 − σt2ϵtθ(xt) + σtϵt,

where√1 − αϵtt−∼1 canN(0be,1arbitrary), αt = parameters. ti=1(1 − βi), and 0 ≤ σt ≤

Usually (e.g. in SD or Stable Inpainting described below) σt = 0 is taken to get a deterministic process:

√1 − αtϵtθ(xt) √αt

xt −

xt−1 = √αt−1

+

(4)

1 − αt−1ϵtθ(xt), t = T,...,1.

For text-to-image synthesis, SD guides the processes with a textual prompt τ. Hence the function ϵtθ(xt) = ϵtθ(xt,τ), modeled by a UNet-like ([28]) architecture, is also conditioned on τ by its cross-attention layers. For simplicity sometimes we skip τ in writing ϵtθ(xt,τ).

As mentioned earlier, Stable DIffusion can be modified and fine-tuned for text-guided image inpainting. To do so [27] concatenate the features of the masked image IM = I ⊙ (1 − M) obtained by the encoder E, and the (downscaled) binary mask M to the latents xt and feed the resulting tensor to the UNet to get the estimated noise ϵtθ([xt,E(IM),down(M)],τ), where down is the downscaling operation to match the shape of the latent xt. Newly added convolutional filters are initialized with zeros while the rest of the UNet from a pretrained checkpoint of Stable Diffusion. Training is done by randomly masking images and optimizing the model to reconstruct them based on image captions from the LAION-5B ([32]) dataset. The resulting model shows visually plausible image completion and we refer to it as Stable Inpainting.

###### 3.2. HD-Painter: Overview

The overview of our method is presented in Fig. 2. The proposed pipeline is composed of two stages: text-guided image inpainting on the resolution H/4 × W/4 is applied followed by the inpainting-specific ×4 super-resolution of the generated content.

To complete the missing region M according to the given prompt τ we take a pre-trained inpainting diffusion model like Stable Inpainting, replace the self-attention layers by PAIntA layers, and perform a diffusion backward process by applying our RASG mechanism. After getting the final estimated latent x0, it is decoded resulting in an inpainted image Ilowc = D(x0) ∈ RH4 ×W4 .

To inpaint the original size image I ∈ RH×W we utilize the super-resolution stable diffusion from [27]. We apply the diffusion backward process of SD starting from XT ∼ N(0,1) and conditioned on the low resolution inpainted image Ilowc . After each step we blend the denoised X0pred with the original image’s encoding E(I) in the known region indicated by the mask (1−M) ∈ {0,1}H×W and derive the next latent Xt−1 by Eq. 4. After the final step we decode the latent by D(X0) and use Poisson blending ([22]) with I to avoid edge artifacts.

###### 3.3. Prompt-Aware Introverted Attention (PAIntA)

Throughout our experiments we noticed that existing approaches, such as Stable Inpainting, tend to ignore the userprovided prompt relying more on the visual context around the inpainting area. In the introdution we categorized this issue into two classes based on user experience: background dominance and nearby object dominance. Indeed, for example in Fig. 4, rows 1, 3, 4, the existing solutions (besides BLD) fill the region with background, and in rows 5, 6, they prefer to continue the animal and the car instead of generating a boat and flames respectively. We hypothesize that the visual context dominance over the prompt is attributed to the prompt-free, only-spatial nature of self-attention layers. To support this we visualize the self-attention scores (see Appendix) and observe a high similarity between the inpainted tokens and such known tokens of the image which have low similarity with the prompt (for more details see Appendix). Therefore, to alleviate the issue, we introduce a plug-in replacement for self-attention, Prompt-Aware Introverted Attention (PAIntA, see Fig. 3 (a)) which utilizes the inpainting mask M and cross-attention matrices to control the self-attention output in the unknown region. Below we discuss PAIntA in detail.

Let X ∈ R(h×w)×d be the input tensor of PAIntA. Similar to self-attention, PAIntA first applies projection layers to get the queries, keys, and values we denote by Qs,Ks,Vs ∈ R(h×w)×d respectively, and the similarity matrix Aself = QsK

T √ s

d ∈ Rhw×hw. Then we mitigate

[Figure 13]

- Figure 2. Our method has two stages: image completiton, and inpainting-specialized super-resolution (×4). For image completion in each diffusion step we denoise the latent xt by conditioning on the inpainting mask M and the masked downscaled image IM = down(I) ⊙ (1 − M) ∈ RH4 ×W4 ×3 (encoded with the VAE encoder E). To make better alignement with the given prompt our PAIntA

block is applied instead of self-attention layers. After predicting the denoised xpred0 in each step t, we provide it to our RASG guidance mechanism to estimate the next latent xt−1. For inpainting-specific super resolution we condition the high-resolution latent Xt denoising process by the lower resolution inpainted result Ilowc , followed by blending X0pred ⊙ M + E(I) ⊙ (1 − M). Finally we get Ic by Poisson blending the decoded output with the original image I.

the too strong influence of the known region over the unknown by adjusting the attention scores of known pixels contributing to the inpainted region. Specifically, leveraging the prompt τ, PAIntA defines a new similarity matrix:

(A˜self)ij =

A˜self ∈ Rhw×hw,

cj · (Aself)ij Mi = 1 and Mj = 0, (Aself)ij otherwise,

(5)

where cj shows the alignment of the jth feature token (pixel) with the given textual prompt τ.

We define {cj}hwj=1 using the cross-attention spatiotextual similarity matrix Scross = SoftMax(QcKcT/

√

d), where Qc ∈ R(h×w)×d, Kc ∈ Rl×d are query and key tensors of corresponding cross-attention layers, and l is the number of tokens of the prompt τ. Specifically, we consider CLIP text embeddings of the prompt τ and separate the ones which correspond to the words of τ and End of Text (EOT) token (in essence we just disregard the SOT token and the null-token embeddings), and denote the set of chosen indices by ind(τ) ⊂ {1,2,...,l}. We include EOT since (in contrast with SOT) it contains information about the prompt τ according to the architecture of CLIP text encoder. For each jth pixel we define its similarity with the prompt τ by summing up it’s similarity scores with the embeddings indexed from ind(τ), i.e. cj =

k∈ind(τ)(Scross)jk. Also, we found beneficial to normalize the scores cj = clip cj−maxmedian(c (ck; k=1,...,hw)

k; k=1,...,hw) ,0,1 , where clip is the clipping operation between [0,1].

Note that in vanilla SD cross-attention layers come after self-attention layers, hence in PAIntA to get query and key tensors Qc,Kc we borrow the projection layer weights from

the next cross-attention module (see Fig. 2). Finally we get the output of the PAIntA layer with the residual connection with the input: Out = X + SoftMax(A˜self) · Vs.

###### 3.4. Reweighting Attention Score Guidance (RASG)

To further enhance the generation alignment with the prompt τ we adopt a post-hoc sampling guidance mechanism [6] with an objective function S(x) leveraging the open-vocabulary segmentation properties of cross-attention layers. Specifically1 at each step the following update rule

is√1used− αaftert ·s∇predictingxt the noise ϵtθ(xt): ϵˆtθ(xt) ← ϵtθ(xt)+

S(xt), where s is a hyperparameter controlling the amount of the guidance. However, as also noted by [4], vanilla post-hoc guidance may shift the domain of diffusion latents xt−1 resulting in image quality degradations. Indeed, according to the (deterministic) DDIM process (Eq. 4) after substituting ϵtθ(xt) with ϵˆtθ(xt) we get

√1 − αtϵtθ(xt) √αt

xt −

xt−1 = √αt−1

+

1 − αt−1ϵtθ(xt) − ξt∇xt

(6)

S(xt), ξt = √1 − αt · s

√1 − αt√αt−1 √αt − 1 − αt−1 ,

hence in Eq. 4 we get the additional term −ξt∇xt

S(xt) which may shift the original distribution of xt−1.

To this end we introduce the Reweighting Attention Score Guidance (RASG) strategy which benefits from the general DDIM backward process (Eq. 3) and introduces a gradient reweighting mechanism resulting in latent domain preserva-

1for brevity: ϵtθ(xt) = ϵtθ([xt, E(IM), down(M)], τ).

[Figure 14]

- Figure 3. (a) PAIntA block takes an input tensor X ∈ Rh×w×3 and the CLIP embeddings of τ. After computing the self- and crossattention scores Aself and Across, we update the former (Eq. 5) by scaling with the normalized values {cj}hwj=1 obtained from Scross = SoftMax(Across). Finally the the updated attention scores A˜self are used for the convex combination of the values Vs to get the residual

√1 − αtϵθ(xt) and guides the xt−1 estimation process towards minimization of S(xt) defined by Eq. 9. Gradient reweighting makes the gradient term close to being sampled from N(0, 1) (green area) by so ensuring the domain preservation (blue area).

√αt−1

of PAIntA’s output. (b) RASG mechanism takes the predicted scaled denoised latent √αt−1xpred0 =

√αt xt −

tion. Specifically, according to Claim 1, xt−1 obtained either by Eq. 4 or by Eq. 3 will be in the required domain (see Fig. 3). Hence if in Eq. 3 we replace the stochastic component ϵt by the rescaled version of the gradient ∇xt

S(xt) (to make it closer to a sampling from N(0,1)), we will keep xt−1 in the required domain and at the same time will guide its sampling towards minimization of S(xt). Rescaling of the gradient ∇xt

S(xt) is done by dividing it on its standard deviation (we do not change the mean to keep the direction of the S(xt) minimization, for more discussion see Appendix). Thus, RASG sampling is done by the formula

√1 − αtϵθ(xt) √αt

xt −

xt−1 = √αt−1

+

(7)

1 − αt−1 − σt2ϵθ(xt) + σt ∇xt

S(xt) std(∇xt

.

S(xt))

Now let us define the function S(xt) (for more discussion on its choice see Appendix). First we consider all cross-

attention maps Across with the output resolution of 32H × W

32: A1cross,...,Amcross ∈ R(H/32·W/32)×l, where m is the number of such cross-attention layers, and l is the number of

token embeddings. Then for each k ∈ ind(τ) ⊂ {1,...,l} we average the attention maps and reshape to 32H × W32:

m

1 m

Akcross(xt) =

32×W32 . (8)

H

Aicross[:,k] ∈ R

i=1

Using post-hoc guidance with S(xt) we aim to maximize the attention scores in the unknown region determined by

the binary mask M ∈ {0,1}H×W, hence we take the average binary cross entropy between Ak(xt) and M (M is downscaled with NN interpolation, σ here is sigmoid):

H 32 · W32

[Mi log σ(Akcross(xt)i)+

S(xt) = −

(9)

i=1

k∈ind(τ)

(1 − Mi)log(1 − σ(Akcross(xt)i))].

###### 3.5. Inpainting-Specialized Conditional SuperResolution

Here we discuss our method for high-resolution inpainting utilizing a pre-trained diffusion-based super-resolution model. We leverage the fine-grained information from the known region to upscale the inpainted region (see Fig. 2.). Recall that I ∈ RH×W×3 is the original high-resolution image we want to inpaint, and E is the encoder of VQGAN [8]. We consider X0 = E(I) and take a standard Gaussian noise XT ∈ RH4 ×W4 ×4. Then we apply a backward diffusion process (Eq. 4) on XT by using the upscalespecialized SD model and conditioning it on the low resolution inpainted image Ilowc . After each diffusion step

we√1 −blendαtϵtθthe(Xtestimated))/√αt withdenoisedX0 bylatentusing MX0pred: = (Xt − X0pred ← M ⊙ X0pred + (1 − M) ⊙ X0, (10)

and use the new X0pred to determine the latent Xt−1 (by Eq. 4). After the last diffusion step X0pred is decoded and

blended (Poisson blending) with the original image I.

It’s worth noting that our blending approach is inspired by seminal works [1, 33] blending Xt with the noisy latents of the forward diffusion. In contrast, we blend high-

frequencies from X0 with the denoised prediction X0pred allowing noise-free image details propagate from the known region to the missing one during all diffusion steps.

###### 4. Experiments

###### 4.1. User Study

We also performed a user study for a qualitative comparison with the competitor state-of-the art methods. The 12 participants were shown 20 (image, mask, prompt) triplets and the inpainting results of all methods in random order. For each sample image we asked to select the best results based on (i) prompt alignment and (ii) overall quality, allowing the choice of no methods when all methods were bad, or multiple methods when the quality was similar. We calculate the total votes for all methods for each question. The results are presented in Fig. 5 demonstrating a clear advantage of our method in both aspects over all competitor methods.

###### 4.2. Implementation Details

We use DreamShaper 8 [18] version of Stable Inpainting and Stable Super-Resolution 2.0 as image completion and inpainting-specialized super-resolution baselines respectively. PAIntA is used to replace the self attention layers on the H/32×W/32 and H/16×W/16 resolutions for the first half of generation steps. For RASG we select only cross-attention similarity matrices of the H/32×W/32 resolution since utilizing higher resolutions did not offer significant improvements.

For hyperparameters {σt}Tt=1 we chose

σt = η (1 − αt−1)/(1 − αt) 1 − αt/αt−1, η = 0.1

###### 4.3. Experimental Setup

Here we compare with existing state-of-the-art methods such as GLIDE [21], Stable 2.0 Inpainting2 [27], DreamShaper Inpainting [18], Blended Latent Diffusion (BLD) [2], ControlNet-Inpainting3 [43] (with DreamShaper

###### 4 base), SDXL-Inpainting5 [23] and SmartBrush [38]. As authors of the SmartBrush paper don’t provide code and model, we reproduce it according to paper and refer to it as SmartBrush reprod.. Specifically, we build SmartBrush reprod. based on DreamShaper text-to-image model

- 2https : / / huggingface . co / stabilityai / stable -

diffusion-2-inpainting

- 3https : / / huggingface . co / lllyasviel / control _

v11p_sd15_inpaint

- 4https://huggingface.co/Lykon/dreamshaper-8
- 5https : / / huggingface . co / spaces / diffusers /

stable-diffusion-xl-inpainting

in order to have fair comparison with our method, which uses DreamShaper Inpainting as a baseline. We evaluate the methods on a random sample of 10000 (image, mask, prompt) triplets from the validation set of MSCOCO 2017 [15], where the prompt is chosen as the label of the selected instance mask. We noticed that when a precise mask of a recognizable shape is given to Stable Inpainting, it tends to ignore the prompt and inpaint based on the shape. To prevent this, we use the convex hulls of the object segmentation masks and compute the metrics accordingly.

We evaluate the CLIP score on a cropped region of the image using the bounding box of the input mask. As CLIP score can still assign high scores to adversarial examples, we additionally compute the generation class accuracy. So, we utilize a pre-trained instance detection model for MSCOCO: MMDetection [5]. We run it on the cropped area of the generated image, and, as there might be more than one objects included in the crop, we treat the example as positive if the prompt label is in the detected object list.

To measure the visual fidelity of the results we employ the LAION aesthetic score.6 The aesthetic score is computed by an MLP trained on 5000 image-rating pairs from the Simulacra Aesthetic Captions dataset [24], and can be used to assign a value from the [0,10] range to images based on their aesthetic appeal.

Finally, we employ PickScore [14] as a combined metric of text-alignment and visual fidelity. Being trained on real user feedback PickScore is able to not only assess the prompt-faithfulness of inpainting methods but also the generation quality, while reflecting the complex requirements of users. In our setting we apply PickScore between our vs other methods results and compute the percentage when it gives the advantage to our.

###### 4.4. Quantitative and Qualitative Analysis

Table 1 shows that our method outperforms the competitors in all metrics. It can be noticed that while SmartBrush trained over DreamShaper Inpainting improves the accuracy over the baseline, the CLIP score improvement is marginal and the overall quality is significantly dropped according to aesthetic score. On the other hand, our method significantly improves the prompt-alignment as measured by both CLIP score and accuracy while also maintaining the quality.

The examples in Fig. 4 demonstrate qualitative comparison between our method and the other state-of-the-art approaches. In many cases the baseline DreamShaper Inp. generates a background (Fig. 4, columns 1, 3, 5) or reconstructs the missing regions as continuation of the known region objects disregarding the prompt (Fig. 4, columns 4, 6, 7), while our method, thanks to the combination of PAIntA

6https : / / github . com / christophschuhmann / improved-aesthetic-predictor

antique greek vase

sunglasses

[Figure 15]

[Figure 16]

Masked

[Figure 17]

[Figure 18]

GLIDE

[Figure 19]

[Figure 20]

BLD

[Figure 21]

[Figure 22]

Stable 2.0 Inpainting

[Figure 23]

[Figure 24]

SDXL Inpainting

[Figure 25]

[Figure 26]

DreamshaperControlNet Inpainting

[Figure 27]

[Figure 28]

SmartBrush reprod.

[Figure 29]

[Figure 30]

DreamShaper Inpainting

[Figure 31]

[Figure 32]

Ours

white duck swimming in water

flowers shirt

wooden boat flames

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

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

[Figure 56]

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

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

- Figure 4. Comparison with state-of-the-art text-guided inpainting methods. Zoom in for details. For more comparison see Appendix.

PickScore

Model Name CLIP score ↑ Accuracy ↑ Aesthetic score ↑

↓

(Ours vs Baselines)

GLIDE [21] 25.14 43.39 % 4.48 57.81 % BLD [2] 24.23 49.12 % 4.81 55.35 % SDXL Inpainting [23] 24.79 53.94 % 4.69 58.69 % Stable 2.0 Inpainting [27] 24.86 51.38 % 4.88 55.64 % DreamShaper-ControlNet Inp. [43] 25.73 58.92 % 4.95 54.69 % SmartBrush reprod. [38] 25.79 66.36 % 4.85 54.23 % DreamShaper Inpainting [18] 25.62 59.02 % 4.96 51.98 % Ours 26.25 67.59 % 5.00 50.0 %

Table 1. Quantitative comparison.

| |193<br><br>153<br><br>146<br><br>120 115<br><br>58<br><br>16<br><br>9<br><br>146<br><br>110<br><br>34<br><br>25<br><br>59<br><br>35<br><br>4 5<br><br>prompt alignment<br><br>overall quality| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

200

150

TotalVotes

100

50

0

Ours DreamShaper Inpainting

SmartBrush reprod.

BLD DreamShaper

Stable 2.0 Inpainting

GLIDE SDXLInpainting

ControlNet Inpainting

- Figure 5. Total votes of each method based on our user study for prompt alignment and overall quality. Our method HD-Painter has a clear advantage over all competitors.

and RASG, successfully generates the target objects. Notice that even though DreamShaper-ControlNet-Inpainting and SmartBrush reprod. may also generate the required object, the quality of the generation is poor compared to ours.

Additionally, Fig. 1 demonstrates how effective our inpainting-specialized super-resolution is in seamlessly leveraging known region details for upscaling the generated region. We show more results in our Appendix, as well as comparison with vanilla Stable Super-Resolution approach [27] used as an upscaling method after inpainting.

###### 4.5. Ablation Study

In Tab. 2 we show that PAIntA and RASG separately on their own provide substantial improvements to the model quantitatively. We also provide more discussion on each of them in our supplementary material, including thorough analyses on their impact, demonstrated by visuals.

###### 5. Conclusion

In this paper, we introduced a training-free approach to text-guided high-resollution image inpainting, addressing the prevalent challenges of prompt neglect: background and nearby object dominance. Our contributions, the Prompt-Aware Introverted Attention (PAIntA) layer and the

CLIP score ↑ Accuracy ↑

Aesthetic score ↑

Model Name

base (DreamShaper Inp.) 25.62 59.02 % 4.96 only RASG 25.82 62.67 % 4.97 only PAIntA 26.06 64.17 % 4.99 RASG & PAIntA 26.25 67.59 % 5.00

Table 2. Ablation study for PAIntA and RASG.

Reweighting Attention Score Guidance (RASG) mechanism, effectively mitigate the mentioned issues leading our method to surpass the existing state-of-the-art approaches qualitatively and quantitatively. Additionally, our unique inpainting-specific super-resolution technique offers seamless completion in high-resolution images, distinguishing our method from existing solutions.

###### References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022.
- [2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM Transactions on Graphics (TOG), 42

(4):1–11, 2023.

- [3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023.
- [4] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023.
- [5] K Chen, J Wang, J Pang, Y Cao, Y Xiong, X Li, S Sun, W Feng, Z Liu, J Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arxiv 2019. arXiv preprint arXiv:1906.07155, 2019.
- [6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [7] Dave Epstein, Allan Jabri, Ben Poole, Alexei A Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. arXiv preprint arXiv:2306.00986, 2023.
- [8] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.
- [9] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.
- [10] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [12] Itseez. Open source computer vision library. https:// github.com/itseez/opencv, 2015.
- [13] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020.
- [14] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. arXiv preprint arXiv:2305.01569, 2023.
- [15] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.
- [16] Guilin Liu, Fitsum A Reda, Kevin J Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro. Image inpainting for irregular holes using partial convolutions. In Proceedings of the European conference on computer vision (ECCV), pages 85–100, 2018.

- [17] Haoming Lu, Hazarapet Tunanyan, Kai Wang, Shant Navasardyan, Zhangyang Wang, and Humphrey Shi. Specialist diffusion: Plug-and-play sample-efficient fine-tuning of text-to-image diffusion models to learn any unseen style. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14267–14276, 2023.
- [18] Lykon. Dreamshaper-8 inpainting. https://civitai. com / models / 4384 ? modelVersionId = 131004, 2023.
- [19] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023.
- [20] Shant Navasardyan and Marianna Ohanyan. Image inpainting with onion convolutions. In proceedings of the asian conference on computer vision, 2020.
- [21] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [22] Patrick P´erez, Michel Gangnet, and Andrew Blake. Poisson Image Editing. Association for Computing Machinery, New York, NY, USA, 1 edition, 2023.
- [23] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [24] John David Pressman, Katherine Crowson, and Simulacra Captions Contributors. Simulacra aesthetic captions. Technical Report Version 1.0, Stability AI, 2022. url https://github.com/JD-P/simulacra-aesthetic-captions .
- [25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [26] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022.

- [27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [28] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.
- [29] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven

- generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023.
- [30] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.
- [31] Andranik Sargsyan, Shant Navasardyan, Xingqian Xu, and Humphrey Shi. Mi-gan: A simple baseline for image inpainting on mobile devices. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7335–7345, 2023.
- [32] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.
- [33] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [34] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.
- [35] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [36] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi PontTuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18359– 18369, 2023.
- [37] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N¨uwa: Visual synthesis pretraining for neural visual world creation. In European conference on computer vision, pages 720–736. Springer, 2022.
- [38] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22428–22437, 2023.
- [39] Xingqian Xu, Shant Navasardyan, Vahram Tadevosyan, Andranik Sargsyan, Yadong Mu, and Humphrey Shi. Image completion with heterogeneously filtered spectral hints. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4591–4601, 2023.
- [40] Zili Yi, Qiang Tang, Shekoofeh Azizi, Daesik Jang, and Zhan Xu. Contextual residual aggregation for ultra high-resolution image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7508–7517, 2020.
- [41] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Generative image inpainting with con-

- textual attention. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5505–5514, 2018.
- [42] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Free-form image inpainting with gated convolution. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4471–4480, 2019.
- [43] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023.
- [44] Shengyu Zhao, Jonathan Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I Chang, and Yan Xu. Large scale image completion via co-modulated generative adversarial networks. arXiv preprint arXiv:2103.10428, 2021.
- [45] Haitian Zheng, Zhe Lin, Jingwan Lu, Scott Cohen, Eli Shechtman, Connelly Barnes, Jianming Zhang, Ning Xu, Sohrab Amirghodsi, and Jiebo Luo. Cm-gan: Image inpainting with cascaded modulation gan and object-aware training. arXiv preprint arXiv:2203.11947, 2022.

###### Appendix A. Extended Qualitative Comparison

Here in Fig. 15 we show more visual comparison with the other state-of-the-art methods. Fig. 16 includes more comparison on the validation set of MSCOCO 2017 [15]. The results show the advantage of our method over the baselines.

In Fig. 6 we compare our inpainting-specialized superresolution method with vanilla approaches of Bicubic or Stable Super-Resolution-based upscaling of the inpainting results followed by Poisson blending in the unknown region. We can clearly see that our method, leveraging the known region fine-grained information, can seamlessly fill in with high quality. In Figures 17 and 18 we show more visual comparison between our method and the approach of Stable Super-Resolution.

###### Appendix B. Discussion on PAIntA

In this section we discuss the effectiveness of the proposed PAIntA module as a plug-in replacement for self-attention (SA) layers. To that end, first we visualize SA similarity maps averaged across masked locations from resolutions H/16 × W/16 and H/32 × W/32 where PAIntA is applied (see Fig. 8). Then, we see that PAIntA successfully scales down the similarities of masked locations with prompt-unrelated locations from the known region, and, as a result, a prompt-specified object is generated inside the mask.

For a given resolution (H/16×W/16 or H/32×W/32), in order to visualize the average SA similarity map across masked pixels, first we resize the input mask to match the dimensions of the corresponding resolution (we use nearest interpolation in resize operation). Then, for each SA layer in the given resolution, we form a 2D similarity map by reshaping and averaging the similarity matrix rows corresponding to the masked region. Further, we average obtained 2D similarity maps across all SA layers (of the given resolution) and diffusion timesteps. More specifically, if A1self,...,ALself ∈ Rhw×hw (h×w is either H/16×W/16 or H/32 × W/32) are the self-attention matrices of Stable Inpainting layers of the given resolution, and, respectively, are being updated by PAIntA to the matrices A˜iself (see Eq. 5), then we consider the following similarity maps:

1 |M| · L i,M

A =

i=1

L

(Alself)i ∈ Rhw,

l=1

L

1 |M| · L i,M

A˜ =

(A˜lself)i ∈ Rhw,

i=1

l=1

and reshape them to 2D matrices of size h×w. So, Aij and A˜ij show the average amount in which masked pixels attend

to to other locations in the cases of the vanilla self-attention and PAIntA respectively. Finally, in order to visualize the similarity maps, we use bicubic resize operation to match it with the image dimensions and plot the similarity heatmap using JET colormap from OpenCV [12].

Next, we compare the generation results and corresponding similarity maps obtained from above procedure when PAIntA’s SA scaling is (the case of A˜) or is not (the case of A) used. Because PAIntA’s scaling is only applied on H/32 × W/32 and H/16 × W/16 resolutions, we are interested in those similarity maps. Rows 1-3 in Fig. 8 demonstrate visualizations on nearby object dominance issue (when known objects are continued to the inpainted region while ignoring the prompt) of the vanilla diffusion inpainting, while rows 4-6 demonstrate those of with background dominance issue (when nothing is generated, just the background is coherently filled in).

For example, on row 1, Fig. 8 in case of DreamShaper Inpainting without PAIntA generation, the average similarity of the masked region is dominated by the known regions of the car on both 16 and 32 resolutions. Whereas, as a result of PAIntA scaling application, the average similarity of the masked region with the car is effectively reduced, and the masked region is generated in accordance to the input prompt.

Row 4, Fig. 8 demonstrates an example where the result without PAIntA continues the background based on visual context instead of following the user prompt. In this case, visualization shows that usage of PAIntA successfully reduces the similarity of the masked region with the unrelated background. As a result, by reducing the similarity of masked region with the unrelated known regions PAIntA enables prompt-faithful generation. You can find additional examples of PAIntA’s effect on the final generation in Fig. 7.

###### Appendix C. Discussion on RASG

In this section we discuss the choice of RASG objective guidance function S(x), then demonstrate the effect of RASG and motivate the part of gradient reweighting by its standard deviation. Finally, we present additional examples of RASG’s effect on the final generation in Fig. 9.

###### Appendix C.1. The Objective Function S(x)

As we already mentioned in the main paper, Stable Inpainting may fail to generate certain objects in the prompt, completely neglecting them in the process. We categorized these cases into two types, namely background and nearby object dominance issues. [4] also mentions these issues but for text-to-image generation task, and refers them as catastrophic neglect problem. To alleviate this problem [4] propose a mechanism called generative semantic nursing, allowing the users to “boost” certain tokens in the prompt,

benchsailorhat

Input Bicubic Stable Super-Resolution Ours

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

- Figure 6. Comparison of our inpainting-specialized super-resolution approach with vanilla upscaling methods for inpainting. Best viewed when zoomed in.

white duck swimming

moon wine botttle in water clock boots

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

input

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

DreamShaper Inpainting

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

with PAIntA

Figure 7. Visual ablation of PAIntA. Generated images use the same seed. In row 3 only PAIntA is used.

ensuring their generation. In essence the mechanism is a post-hoc guidance with a chosen objective function maximizing the maximal cross-attention score of the image with the token which should be “boosted”. This approach can be easily adapted to the inpainting task by just restricting the maximum to be taken in an unknown region so that the ob-

ject is generated there, and averaging the objectives across all tokens, since we don’t have specific tokens to “boost”, but rather care about all of them. In other words, by our notations from the main paper, the following guidance ob-

without PAIntA (vanilla DreamShaper Inpainting) input

with PAIntA

SA map SA map SA map SA map

result result

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

flames

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

shirtsailorhatantiquegreekvasewhiterobot

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

flowers

Figure 8. Comparison of self-attention similarity maps averaged across masked pixels for generations without/with PAIntA’s scaling of the original self-attention scores. Images are generated from the same seed.

jective funciton can be used:

1 |ind(τ)|

{Ak(xt)i}. (11)

S(xt) = −

max

i: Mi=1

k∈ind(τ)

However we noticed that with this approach the shapes/sizes of generated objects might not be sufficiently aligned with the shape/size of the input mask, which is often desirable for text-guided inpainting (see Fig. 11). Therefore, we utilize the segmentation property of

cross-attention similarity maps, by so using Binary Cross Entropy as the energy function for guidance (see Eq. 9 in the main paper). As can be noticed from Fig. 11 the results with the binary cross-entropy better fit the shape of the inpaining mask.

###### Appendix C.2. Effect of RASG Strategy

Although the objective function S(x) defined by Eq. 9 (main paper) results in better mask shape/size aligned in-

input

DreamShaper Inpainting

with RASG

antique greek vase

clock wooden boat white fox space helmet

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Figure 9. Visual ablation of RASG. Generated images use the same seed. In row 3 only RASG is used.

painting, the vanilla post-hoc guidance may lead the latents to become out of their trained domain as also noted by [4]: “many updates of xt may lead to the latent becoming outof-distribution, resulting in incoherent images”. Due to this the post-hoc guidance mechanism (semantic nursing) by [4] is done using multiple iterations of very small, iterative perturbations of xt, which makes the process considerably slow. In addition, the generation can still fail if the iterative process exceeds the maximum iteration limit without reaching the necessary thresholds.

Thanks to RASG’s seamless integration of the ∇xt

S(xt) gradient component into the general form of DDIM diffusion sampling, our RASG mechanism keeps the modified latents xt within the expected distribution, while introducing large enough perturbations to xt with only one iteration of guidance per time-step. This allows to generate the objects described in the prompts coherently with the known region without extra-cost of time.

Fig. 10 demonstrates the advantage of RASG’s strategy over the vanilla guidance mechanism. Indeed, in the vanilla post-hoc guidance there is a hyperparameter s controlling the amount of guidance. When s is too small (e.g. close to 0 or for some cases s = 100) the vanilla guidance mechanism does not show much effect due to too small guidance from s∇xt

S(xt). Then with increasing the hyperparameter (s = 1000,10000) one can notice more and more text/shape alignment with prompt/inpainting mask, how-

ever the generated results are unnatural and incoherent with the known region. This is made particularly challenging by the fact, that different images, or even different starting seeds with the same input image might require different values of the perturbation strength to achieve the best result. In contrast, RASG approach is hyperparameter-free allowing both: prompt/mask-aligned and naturally looking results.

###### Appendix C.3. Rescaling with Standard Deviation

The core idea of RASG is to automatically scale perturbation using certain heuristics, such that the guidance process has a consistent effect on the output, without harming the quality of the image. Our main heuristic relies on the fact that [34] have defined a parametric family of stochastic denoising processes, which can all be trained using the same training objective as DDPM [11]. Recall the general form of parametric family of DDIM sampling processes:

√1 − αtϵtθ(xt) √αt

xt −

xt−1 = √αt−1

+

(12)

1 − αt−1 − σt2ϵtθ(xt) + σtϵt,

where ϵt ∼ N(0,1). Particularly ϵt can be taken to be collinear with the gradient ∇xt

S(xt) which will result in xt−1 distribution preservation by at the same time guiding the generation process towards minimization of S(xt).

[Figure 164]

[Figure 165]

[Figure 166]

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

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Dreamshaper Inpainting (w/o Guidance)

Guidance: s = 100

Guidance: s = 1000

Guidance: s = 10000

Input

RASG

- Figure 10. Comparison of RASG strategy with default Stable Inpainting and vanilla guidance mechanism with different guidance scales. In contrast to vanilla guidance, where the generation highly depends on the guidance scale, RASG consistently produces naturally looking and prompt-aligned results.

Therefore we propose to scale the gradient ∇xt

S(xt) with a value λ and use instead of ϵt in the general form of DDIM. To determine λ we analyse the distribution of ∇xt

S(xt) and found out that the values of the gradients have a distribution very close to a gaussian distribution, with 0 mean and some arbitary σ, which changes over timestep/image (Fig. 12). Therefore, computing the standard deviation of the values of ∇xt

S(xt), and normalizing it by λ = std(∇ 1

xtS(xt)) results in the standard normal distribution (see Fig. 13). So the final form of RASG guidance

strategy is

√1 − αtϵtθ(xt) √αt

xt −

xt−1 = √αt−1

+

1 − αt−1 − σt2ϵtθ(xt) + σt ∇xt

S(xt) std(∇xt

.

S(xt))

###### Appendix D. Limitations

(13)

Although our method improves the prompt-alignment of existing text-guided inpainting approaches, it still has a dependency on the backbone model, hence inherits some quality

dog elephant clock zebra

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

input

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Attend&Excite Objective

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Binary Cross Entropy

- Figure 11. Comparison of the Binary Cross Entropy engery function to modifed version of Attend & Excite. Images generated from the same seed.
- Figure 12. Histogram of ∇xtS(xt) values (i.e. before gradient standardization)

[Figure 200]

- Figure 13. Histogram of std∇(∇xtS(xt)

[Figure 201]

xtS(xt)) values (i.e. after gradient standardization)

limitations. Particularly it may generate extra limbs (the elephant in Fig. 14 has 5 legs) or illogical appearances (the sheep appears to have two bodies in Fig. 14 after the inpainting).

[Figure 202]

[Figure 203]

epephant

[Figure 204]

[Figure 205]

sheep

Figure 14. Failure examples produced by our approach.

###### Appendix E. Potential negative impacts

Our research strives to enhance the accuracy of object generation within the scope of text-guided image inpainting. However, it is crucial to acknowledge the potential negative impacts. The technology could be exploited to create deceptive imagery or disseminate misinformation, raising ethical concerns. While our method is training-free and does not introduce new biases, it is imperative to consider the potential propagation of biases from the base models we build upon. These biases could lead to the generation of content that inadvertently reflects societal or historical prejudices.

To counter these issues, it is essential for the broader research community to establish ethical standards and develop robust methods to detect AI-generated content. Furthermore, efforts should be made to diversify training datasets to reduce inherent biases. While these challenges are significant, the positive implications of our work in areas such as creative arts, design and content creation, when used responsibly, have the potential to surpass the negative repercussions.

###### Appendix F. More Examples of Our Method

We present more results of our method both for lowresolution (512 for the long side) images (Fig. 19), as well as high-resoltuion (2048 for the long side) (Figures 20, 21, 22).

Input

GLIDE

BLD

Stable 2.0 Inpainting

SDXL Inpainting

DreamShaperControlNet Inpainting

SmartBrush reprod.

DreamShaper Inpainting

Ours

moon wine bottle boots hamster

clock big horns

[Figure 206]

[Figure 207]

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

[Figure 228]

[Figure 229]

[Figure 230]

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

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Figure 15. More qualitative comparison results. Zoom in to view high-resolution details.

Input

GLIDE

BLD

Stable 2.0 Inpainting

SDXL Inpainting

DreamShaperControlNet Inpainting

SmartBrush reprod.

DreamShaper Inpainting

Ours

truck bear remote cup chair laptop

[Figure 260]

[Figure 261]

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

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Figure 16. More qualitative comparison results on MSCOCO 2017.

Input Stable Super-Resolution Ours

[Figure 314]

[Figure 315]

[Figure 316]

yellowheadphonesteddybearlake

|[Figure 317]|
|---|

|[Figure 318]|
|---|

[Figure 319]

[Figure 320]

[Figure 321]

|[Figure 322]|
|---|

|[Figure 323]|
|---|

[Figure 324]

[Figure 325]

[Figure 326]

|[Figure 327]|
|---|

|[Figure 328]|
|---|

lion

Input Stable Super-Resolution Ours

[Figure 329]

[Figure 330]

[Figure 331]

|[Figure 332]|
|---|

|[Figure 333]|
|---|

icecream

[Figure 334]

[Figure 335]

[Figure 336]

|[Figure 337]|
|---|

|[Figure 338]|
|---|

leathercouch

[Figure 339]

[Figure 340]

|[Figure 341]|
|---|

[Figure 342]

|[Figure 343]|
|---|

[Figure 344]

hoodielaptopbowtiespacehelmet

[Figure 345]

[Figure 346]

[Figure 347]

leathercouch

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

tractorfedora

[Figure 356]

[Figure 357]

spacesuitantiquegreekvase

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

Figure 19. More results of our method.

[Figure 362]

### parrotboots

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

## raccoonpottedroses

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

#### saladcastle

[Figure 371]

[Figure 372]

[Figure 373]

Figure 22. More high-resolution results of our method. Zoom in to view high-resolution details.

