## HiFi Tuner: High-Fidelity Subject-Driven Fine-Tuning for Diffusion Models

Zhonghao Wang1,2, Wei Wei4, Yang Zhao1, Zhisheng Xiao1, Mark Hasegawa-Johnson2, Humphrey Shi2,3, Tingbo Hou1

1Google, 2UIUC, 3Georgia Tech, 4Accenture

# arXiv:2312.00079v1[cs.CV]30Nov2023

### Abstract

[Figure 1]

[Figure 2]

[Figure 3]

“A can with a mountain in the background.”

This paper explores advancements in high-fidelity personalized image generation through the utilization of pretrained text-to-image diffusion models. While previous approaches have made significant strides in generating versatile scenes based on text descriptions and a few input images, challenges persist in maintaining the subject fidelity within the generated images. In this work, we introduce an innovative algorithm named HiFi Tuner to enhance the appearance preservation of objects during personalized image generation. Our proposed method employs a parameter-efficient fine-tuning framework, comprising a denoising process and a pivotal inversion process. Key enhancements include the utilization of mask guidance, a novel parameter regularization technique, and the incorporation of step-wise subject representations to elevate the sample fidelity. Additionally, we propose a reference-guided generation approach that leverages the pivotal inversion of a reference image to mitigate unwanted subject variations and artifacts. We further extend our method to a novel image editing task: substituting the subject in an image through textual manipulations. Experimental evaluations conducted on the DreamBooth dataset using the Stable Diffusion model showcase promising results. Fine-tuning solely on textual embeddings improves CLIP-T score by 3.6 points and improves DINO score by 9.6 points over Textual Inversion. When fine-tuning all parameters, HiFi Tuner improves CLIP-T score by 1.2 points and improves DINO score by 1.2 points over DreamBooth, establishing a new state of the art.

…

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

……

subject representation learning

referenceguided generation

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

DDIM inversion

step-wise subject source images & masks representations

[Figure 14]

[Figure 15]

generated image reference image

Figure 1. Illustration of HiFi Tuner. We first learn the step-wise subject representations with subject source images and masks. Then we select and transform the reference image, and use DDIM inversion to obtain its noise latent trajectory. Finally, we generate an image controlled by the prompt, the step-wise subject representations and the reference subject guidance.

images that align closely with textual guidance. Despite this achievement, the training data remains inherently limited in its coverage of all possible subjects. Consequently, it becomes infeasible for diffusion models to accurately generate images of specific, unseen subjects based solely on textual descriptions. As a result, personalized generation has emerged as a pivotal research problem. This approach seeks to fine-tune the model with minimal additional costs, aiming to generate images of user-specified subjects that seamlessly align with the provided text descriptions.

We identify three drawbacks of existing popular methods for subject-driven fine-tuning [9, 15, 31, 32]. Firstly, a notable imbalance exists between sample quality and parameter efficiency in the fine-tuning process. For example, Textual Inversion optimizes only a few parameters in the text embedding space, resulting in poor sample fidelity. Conversely, DreamBooth achieves commendable sample fidelity but at the cost of optimizing a substantial number of parameters. Ideally, there should be a parameter-efficient method that facilitates the generation of images with satisfactory sample fidelity while remaining lightweight for im-

### 1. Introduction

Diffusion models [14, 37] have demonstrated a remarkable success in producing realistic and diverse images. The advent of large-scale text-to-image diffusion models [29, 30, 33], leveraging expansive web-scale training datasets [7, 34], has enabled the generation of high-quality

proved portability. Secondly, achieving a equilibrium between sample fidelity and the flexibility to render objects in diverse scenes poses a significant challenge. Typically, as fine-tuning iterations increase, the sample fidelity improves, but the flexibility of the scene coverage diminishes. Thirdly, current methods struggle to accurately preserve the appearance of the input object. Due to the extraction of subject representations from limited data, these representations offer weak constraints to the diffusion model. Consequently, unwanted variations and artifacts may appear in the generated subject.

In this study, we introduce a novel framework named HiFi Tuner for subject fine-tuning that prioritizes the parameter efficiency, thereby enhancing sample fidelity, preserving the scene coverage, and mitigating undesired subject variations and artifacts. Our denoising process incorporates a mask guidance to reduce the influence of the image background on subject representations. Additionally, we introduce a novel parameter regularization method to sustain the model’s scene coverage capability and design a step-wise subject representation mechanism that adapts to parameter functions at different denoising steps. We further propose a reference-guided generation method that leverages pivotal inversion of a reference image. By integrating guiding information into the step-wise denoising process, we effectively address issues related to unwanted variations and artifacts in the generated subjects. Notably, our framework demonstrates versatility by extending its application to a novel image editing task: substituting the subject in an image with a user-specified subject through textual manipulations.

We summarize the contributions of our work as follows. Firstly, we identify and leverage three effective techniques to enhance the subject representation capability of textual embeddings. This improvement significantly aids the diffusion model in generating samples with heightened fidelity. Secondly, we introduce a novel reference-guided generation process that successfully addresses unwanted subject variations and artifacts in the generated images. Thirdly, we extend the application of our methodology to a new subjectdriven image editing task, showcasing its versatility and applicability in diverse scenarios. Finally, we demonstrate the generic nature of HiFi Tuner by showcasing its effectiveness in enhancing the performance of both the Textual Inversion and the DreamBooth.

### 2. Related Works

Subject-driven text-to-image generation. This task requires the generative models generate the subject provided by users in accordance with the textual prompt description. Pioneer works [4, 26] utilize Generative Adversarial Networks (GAN) [10] to synthesize images of a particular instance. Later works benefit from the success of diffusion

models [30, 33] to achieve a superior faithfulness in the personalized generation. Some works [6, 35] rely on retrievalaugmented architecture to generate rare subjects. However, they use weakly-supervised data which results in an unsatisfying faithfullness for the generated images. There are encoder-based methods [5, 16, 36] that encode the reference subjects as a guidance for the diffusion process. However, these methods consume a huge amount of time and resources to train the encoder and does not perform well for out-of-domain subjects. Other works [9, 31] fine-tune the components of diffusion models with the provided subject images. Our method follows this line of works as our models are faithful and generic in generating rare and unseen subjects.

Text-guided image editing. This task requires the model to edit an input image according to the modifications described by the text. Early works [9, 27] based on diffusion models [30, 33] prove the effectiveness of manipulating textual inputs for editing an image. Further works [1, 24] propose to blend noise with the input image for the generation process to maintain the layout of the input image. Prompt-to-Prompt [12, 25] manipulates the cross attention maps from the image latent to the textual embedding to edit an image and maintain its layout. InstructPix2Pix [2] distills the diffusion model with image editing pairs synthesized by Prompt-to-Prompt to implement the image editing based on instructions.

### 3. Methods

In this section, we elaborate HiFi Tuner in details. We use the denoising process to generate subjects with appearance variations and the inversion process to preserve the details of subjects. In section 3.1, we present some necessary backgrounds for our work. In section 3.2, we introduce the three proposed techniques that help preserving the subject identity. In section 3.3, we introduce the reference-guided generation technique, which merits the image inversion process to further preserve subject details. In section 3.4, we introduce an extension of our work on a novel image editing application – personalized subject replacement with only textual prompt edition.

##### 3.1. Backgrounds

Stable diffusion [30] is a widely adopted framework in the realm of text-to-image diffusion models. Unlike other methods [29, 33], Stable diffusion is a latent diffusion model, where the diffusion model is trained within the latent space of a Variational Autoencoder (VAE). To accomplish text-to-image generation, a text prompt undergoes encoding into textual embeddings c using a CLIP text encoder[28]. Subsequently, a random Gaussian noise latent xT is initialized. The process then recursively denoises noisy latent xt through a noise predictor network ϵθ with the conditioning

subject representation learning process

###### Subject Representation Learning (SRL)

×𝑁 ×𝐼

×𝐼

×𝐼

[Figure 16]

[Figure 17]

[Figure 18]

[ , ]

SRL

SRL

SRL

SRL

ℒ

... ... 𝑐

𝜙

𝑐 𝑐

𝑐

𝑐 𝑐 𝑐 𝑐

𝑡̅ = 𝑟𝑎𝑛𝑑(1,𝑇)

𝑡̅ = 𝑇

𝑡̅ = 𝑡

𝑡̅ = 1

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

reference image selection and transformation

DM

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

###### SAM

[Figure 30]

𝑡̅

𝐼

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Transform

cosine similarity

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

... DDIM

DDIM

Reference-Guided Generation (RGG)

SAM

𝑡̅ = 1

𝑡̅ = 𝑇

[Figure 41]

ℒ

[c,𝑐 ] [c,𝑐 ]

𝑥 𝐼

𝜙

DM

⨂

reference-guided generation process

[Figure 42]

[Figure 43]

[Figure 44]

1 − 𝑤

[Figure 45]

[Figure 46]

⨁

𝑡̅

DDIM-1

DDIM-1

DDIM-1

... ...

[Figure 47]

𝑡̅ = 𝑡

𝑡̅ = 1

𝑤

𝑡̅ = 𝑡

𝑥 ∗

⨂

###### DM

𝑐 𝑐 𝑐

𝑥 ∗ 𝑥 ∗

[Figure 48]

[Figure 49]

[Figure 50]

𝑡̅

[c,𝑐 ]

𝑥

[Figure 51]

[Figure 52]

[Figure 53]

⨁

DDIM

... RGG

... RGG

... DDIM

1 − 𝑤

𝑡̅ = 𝑇

𝑡̅ = 𝑡

𝑡̅ = 𝑡

𝑡̅ =1

𝜙 DM ⨂

𝑥

[c,𝑐 ] [c,𝑐 ] [c,𝑐 ] [c,𝑐 ]

𝐼

Figure 2. The framework of HiFi Tuner. The grey arrows stand for the data flow direction. The red arrows stand for the gradient back propagation direction. SAM stands for the Segment Anything [18] model. DM stands for the Stable Diffusion [30] model. DDIM and DDIM−1 stands for the DDIM denoising step and inversion step respectively.

Null-text inversion [25] method introduces an inversion-based approach to image editing, entailing the initial inversion of an image input to the latent space, followed by denoising with a user-provided prompt. This method comprises two crucial processes: a pivotal inversion process and a null-text optimization process. The pivotal inversion involves the reversal of the latent representation of an input image, denoted as x0, back to a noise latent representation, xT, achieved through the application of reverse DDIM. This process can be formulated as reparameterizing Eqn. (1) with w = 1:

of c. Finally, the VAE decoder is employed to project the denoised latent x0 onto an image. During the sampling process, a commonly applied mechanism involves classifierfree guidance [13] to enhance sample quality. Additionally, deterministic samplers, such as DDIM [38], are employed to improve sampling efficiency. The denoising process can be expressed as

xt−1 = F(t)(xt,c,ϕ)

(1)

= βtxt − γt(wϵθ(xt,c) + (1 − w)ϵθ(xt,ϕ)).

where βt and γt are time-dependent constants; w is the classifier-free guidance weight; ϕ is the CLIP embedding for a null string.

###### xt+1 = F−1(t)(xt,c) = βtxt + γtϵθ(xt,c) (3)

Textual inversion [9]. As a pioneer work in personalized generation, Textual Inversion introduced the novel concept that a singular learnable textual token is adequate to represent a subject for the personalization. Specifically, the method keeps all the parameters of the diffusion model frozen, exclusively training a word embedding vector cs using the diffusion objective:

We denote the latent trajectory attained from the pivotal inversion as [x∗0,...,x∗T]. However, naively applying Eqn. (1) for x∗T will not restore x∗0, because ϵθ(xt,c) ̸= ϵθ(x∗t−1,c). To recover the original image, Null-text inversion trains a null-text embedding ϕt for each timestep t force the the denoising trajectory to stay close to the forward trajectory [x∗0,...,x∗T]. The learning objective is

∥ϵθ(xt,[c,cs]) − ϵ∥22, (2)

Ls(cs) = min

cs

L(ht)(ϕt) = min

∥x∗t−1 − F(t)(xt,c,ϕt)∥22. (4)

where [c,cs] represents replacing the object-related word embedding in the embedding sequence of the training caption (e.g. “a photo of A”) with the learnable embedding cs. After cs is optimized, this work applies F(t)(xt,[c,cs],ϕ) for generating personalized images from prompts.

ϕt

After training, image editing techniques such as the promptto-prompt [12] can be applied with the learned null-text embeddings {ϕ∗t} to allow manipulations of the input image.

##### 3.2. Learning subject representations

We introduce three techniques for improved learning of the representations that better capture the given object.

Mask guidance One evident issue we observed in Textual Inversion is the susceptibility of the learned textual embedding, cs, to significant influence from the backgrounds of training images. This influence often imposes constraints on the style and scene of generated samples and makes identity preservation more challenging due to the limited capacity of the textual embedding, which is spent on unwanted background details. We present a failure analysis of Textual Inversion in the Appendix A. To address this issue, we propose a solution involving the use of subject masks to confine the loss during the learning process of cs. This approach ensures that the training of cs predominantly focuses on subject regions within the source images. Specifically, binary masks of the subjects in the source images are obtained using Segment Anything (SAM) [18], an off-theshelf instance segmentation model. The Eqn. (2) is updated to a masked loss:

∥M ⊙ (ϵθ(xt,[c,cs]) − ϵ)∥22, (5)

Ls(cs) = min

cs

where ⊙ stands for element-wise product, and M stands for a binary mask of the subject. This simple technique mitigates the adverse impact of background influences and enhancing the specificity of the learned textual embeddings.

Parameter regularization We aim for the learned embedding, cs, to obtain equilibrium between identity preservation and the ability to generate diverse scenes. To achieve this balance, we suggest initializing cs with a portion of the null-text embedding, ϕs, and introducing an L2 regularization term. This regularization term is designed to incentivize the optimized cs to closely align with ϕs:

∥M ⊙(ϵθ(xt, [c, cs])−ϵ)∥22+ws∥cs−ϕs∥22. (6)

Ls(cs) = min

cs

Here, cs ∈ Rn×d where n is the number of tokens and d is the embedding dimension, and ws is a regularization hyperparameter. We define ϕs as the last n embeddings of ϕ and substitute the last n embeddings in c with cs, forming [c,cs]. It is noteworthy that [c,cs] = c if cs is not optimized, given that ϕ constitutes the padding part of the embedding. This regularization serves two primary purposes. Firstly, the stable diffusion model is trained with a 10% caption drop, simplifying the conditioning to ϕ and facilitating classifier-free guidance [13]. Consequently, ϕ is adept at guiding the diffusion model to generate a diverse array of scenes, making it an ideal anchor point for the learned embedding. Secondly, due to the limited data used for training the embedding, unconstrained parameters may lead to overfitting with erratic scales. This overfitting poses a risk of generating severely out-of-distribution textual embeddings.

[Figure 54]

DDIM step

[Figure 55]

50

[Figure 56]

25

[Figure 57]

1

Figure 3. Step-wise function analysis of cs. We generate an image from a noise latent with DDIM and an optimized cs representing a subject dog. The text prompt is ”A sitting dog”. The top image is the result generated image. We follow [12] to obtain the attention maps with respect to the 5 token embeddings of cs as shown in the below images. The numbers to the left refer to the corresponding DDIM denoising steps. In time step 50, the 5 token embeddings of cs are attended homogeneously across the latent vectors. In time step 1, these token embeddings are attended mostly by the subject detailed regions such as the forehead, the eyes, the ears, etc.

Step-wise subject representations We observe that the learned textual embedding, cs, plays distinct roles across various denoising time steps. It is widely acknowledged that during the sampling process. In early time steps where t is large, the primary focus is on generating high-level image structures, while at smaller values of t, the denoising process shifts its emphasis toward refining finer details. Analogous functional distinctions exist for the role of cs. Our analysis of cs across time steps, presented in Fig. 3, underscores these variations. Motivated by this observation, we propose introducing time-dependent embeddings, cts, at each time step instead of a single cs to represent the subject. This leads to a set of embeddings, [c1s,...,cTs ], working collectively to generate images. To ensure smooth transitions between time-dependent embeddings, we initially train a single cs across all time steps. Subsequently, we recursively optimize cts following DDIM time steps, as illustrated in Algorithm 1. This approach ensures that cts is proximate to cts+1 by initializing it with cts+1 and optimizing it for a few steps. After training, we apply

xt−1 = F(t)(xt,[c,cts],ϕ) (7) with the optimized [c1s,...,cTs ] to generate images.

##### 3.3. Reference-guided generation

Shown in Figure 2, we perform our reference-guided generation in three steps. First, we determine the initial latent

Algorithm 1: Optimization algorithm for cts. T is DDIM time steps. I is the optimization steps per DDIM time step. X0 is the set of encoded latents of the source images. Ns(·) is the DDIM noise scheduler. Ls(·) refers to the loss function in Eqn. (6).

Result: Cs Cs = {}, cTs +1 = cs for t = [T, ..., 1] do

cts = cts+1 for i = [1, ..., I] do

ϵ ∼ N(0, 1), x0 ∈ X0, xt = Ns(x0, ϵ, t) cts = cts − η∇ct

Ls(cts) Cs = Cs ∪ {cts}

s

xT and follow the DDIM denoising process to generate an image. Thus, we can determine the subject regions of {xt} requiring guiding information and the corresponding reference image. Second, we transform the reference image and inverse the latent of the transformed image to obtain a reference latent trajectory, [x∗0,...,x∗T]. Third, we start a new denoising process from xT and apply the guiding information from [x∗0,...,x∗T] to the guided regions of {xt}. Thereby, we get a reference-guided generated image.

Guided regions and reference image. First, we determine the subject regions of xt that need the guiding information. Notice that xt ∈ RH×W×C, where H, W and C are the height, width and channels of the latent xt respectively. Following the instance segmentation methods [11, 22], we aim to find a subject binary mask Mg to determine the subset xst ∈ Rm×C corresponding to the subject regions. Because DDIM [38] is a deterministic denoising process as shown in Eqn. (1), once xT, c and ϕ are determined, the image to be generated is already determined. Therefore, we random initialize xT with Gaussian noise; then, we follow Eqn. (7) and apply the decoder of the stable diffusion model to obtain a generated image, Ig1; by applying Grounding SAM [18, 21] with the subject name to Ig1 and resizing the result to H × W, we obtain the subject binary mask Mg. Second, we determine the reference image by choosing the source image with the closest subject appearance to the subject in Ig1, since the referenceguided generation should modify {xt} as small as possible to preserve the image structure. As pointed out by DreamBooth [31], DINO [3] score is a better metric than CLIP-I [28] score in measuring the subject similarity between two images. Hence, we use ViT-S/16 DINO model [3] to extract the embedding of Ig1 and all source images. We choose the source image whose DINO embedding have the highest cosine similarity to the DINO embedding of Ig1 as the reference image, Ir. We use Grounding SAM [18, 21] to obtain the subject binary mask Mr of Ir.

Reference image transformation and inversion. First,

we discuss the transformation of Ir. Because the subject in Ig1 and the subject in Ir are spatially correlated with each other, we need to transform Ir to let the subject better align with the subject in Ig1. As the generated subject is prone to have large appearance variations, it is noneffective to use image registration algorithms, e.g. RANSAC [8], based on local feature alignment. We propose to optimize a transformation matrix

 

 

 

  (8)

 

 

cos(θ2) − sin θ2 0 sin θ2 cos(θ2) 0 0 0 1

θ1 0 0 0 θ1 0 0 0 1

1 0 θ3 0 1 θ4 0 0 1

Tθ =

composed of scaling, rotation and translation such that Tθ(Mr) best aligns with Mg. Here, {θi} are learnable parameters, and Tθ(·) is the function of applying the transformation to an image. Tθ can be optimized with

∥Tθ(Mr) − Mg∥11. (9)

Lt = min

θ

Please refer to the Appendix B for a specific algorithm optimizing Tθ. We denote the optimized Tθ as Tθ∗ and the result of Tθ∗(Mr) as Mr∗. Thereafter, we can transform Ir with Tθ∗(Ir) to align the subject with the subject in Ig1. Notice that the subject in Tθ∗(Ir) usually does not perfectly align with the subject in Ig1. A rough spatial location for placing the reference subject should suffice for the reference guiding purpose in our case. Second, we discuss the inversion of Tθ∗(Ir). We use BLIP-2 model [19] to caption Ir and use a CLIP text encoder to encode the caption to cr. Then, we encode Tθ∗(Ir) into x∗0 with a Stable Diffusion image encoder. Finally, we recursively apply Eqn. (3) to obtain the reference latent trajectory, [x∗0,...,x∗T].

Generation process. There are two problems with the reference-guided generation: 1) the image structure needs to be preserved; 2) the subject generated needs to conform with the context of the image. We reuse xT in step 1 as the initial latent. If we follow Eqn. (7) for the denoising process, we will obtain Ig1. We aim to add guiding information to the denoising process and obtain a new image Ig2 such that the subject in Ig2 has better fidelity and the image structure is similar to Ig1. Please refer to Algorithm 2 for the specific reference-guided generation process. As discussed in Section 3.2, the stable diffusion model focuses on the image structure formation at early denoising steps and the detail polishing at later steps. If we incur the guiding information in early steps, Ig2 is subject to have structural change such that Mr∗ cannot accurately indicate the subject regions. It is harmful to enforce the guiding information at later steps either, because the denoising at this stage gathers useful information mostly from the current latent. Therefore, we start and end the guiding process at middle time steps ts and te respectively. At time step ts, we substitute the latent variables corresponding to the subject region in xt with those in x∗t. We do this for three reasons: 1) the substitution enables the denoising process to assimilate the

Algorithm 2: Reference-guided generation algorithm. J is the number of optimization steps for ϕh per denoising step. L(ht)(·) refers to the loss function in Eqn. (10).

Result: x0 Inputs: ts, te, xT, Mr∗, c, ϕ, [c1s, ..., cTs ], [x∗0, ..., x∗T] for t = [T, ..., 1] do

if t == ts then ϕh = ϕ xt[Mr∗] = x∗t[Mr∗]

xt−1 = F(t)(xt, [c, cts], ϕ) if t ⩽ ts and t ⩾ te then

for j = [1, ..., J] do

ϕh = ϕh − η∇ϕhL(ht)(ϕh) xt−1[Mr∗] = F(t)(xt, [c, cts], ϕh)[Mr∗]

subject to be generated to the reference subject; 2) the latent variables at time step ts are close to the noise space so that they are largely influenced by the textual guidance as well; 3) the substitution does not drastically change the image structure because latent variables have small global effect at middle denoising steps. We modify Eqn. (4) to Eqn. (10) for guiding the subject generation.

L(ht)(ϕh) = min

∥x∗t−1[Mr∗] − F(t)(xt,[c,cts],ϕh)[Mr∗]∥22

ϕh

(10)

Here, xt[M] refers to latent variables in xt indicated by the mask M. Because ϕh is optimized with a few steps per denoising time step, the latent variables corresponding to the subject regions change mildly within the denoising time step. Therefore, at the next denoising time step, the stable diffusion model can adapt the latent variables corresponding to non-subject regions to conform with the change of the latent variables corresponding to the subject regions. Furthermore, we can adjust the optimization steps for ϕh to determine the weight of the reference guidance. More reference guidance will lead to a higher resemblance to the reference subject while less reference guidance will result in more variations for the generated subject.

##### 3.4. Personalized subject replacement

We aim to use the learned subject textual representations to replace the subject in an image with the user-specified subject. Although there are methods [20, 23, 39, 40] inpainting the image area with a user-specified subject, our method has two advantages over them. First, we do not specify the inpainting area of the image; instead, our method utilize the correlation between the textual embeddings and the latent variables to identify the subject area. Second, our method can generate a subject with various pose and appearance such that the added subject better conforms to the image context.

Algorithm 3: Personalized subject replacement algorithm. F−1(t) refers to Eqn. (3). K is the optimization steps for null-text optimization. L(ht)(·) refers to Eqn. (4)

Result: xg0 Inputs: xr0, cr, cg, [c1s, ..., cTs ] xr0∗ = xr0 for t = [0, ..., T − 1] do

xrt+1∗ = F−1(t)(xrt∗, cr)

xrT = xrT∗, ϕT = ϕ for t = [T, ..., 1] do

for k = [1, ..., K] do

ϕt = ϕt − η∇ϕtL(ht)(ϕt) xrt−1, art∗ = A(t)(xrt, cr, ϕt) ϕt−1 = ϕ∗t = ϕt

xgT = xrT∗ for t = [T, ..., 1] do

xgt−1 = F˜[(ctt)

s,wg](xgt, [cg, cts], ϕ∗t, art∗)

We first follow the fine-tuning method in Section 3.2 to obtain the step-wise subject representations [c1s,...,cTs ]. We encode the original image Ir to xr0 with the Stable Diffusion image encoder; then we use BLIP-2 model [19] to caption Ir and encode the caption into cr with the Stable Diffusion language encoder. We identify the original subject word embedding in cr and substitute that with the new subject word embedding wg to attain a cg (e.g. ‘cat’ → ‘dog’ in the sentence ‘a photo of siting cat’). Then we follow Algorithm 3 to generate the image with the subject replaced. Referring to the prompt-to-prompt paper [12], we store the step-wise cross attention weights with regard to the word embeddings in cr to art∗. A(t)(·,·,·) performs the same operations as F(t)(·,·,·) in Eqn. (1) but returns xt−1 and art∗. We also modify F(t)(·,·,·) to F˜[(ctt)

s,wg](·,·,·,art∗) such that all token embeddings use fixed cross attention weights art∗ except that [cts,wg] use the cross attention weights of the new denoising process.

### 4. Experiments

Dataset. We use the DreamBooth [31] dataset for evaluation. It contains 30 subjects: 21 of them are rigid objects and 9 of them are live animals subject to large appearance variations. The dataset provides 25 prompt templates for generating images. Following DreamBooth, we fine-tune our framework for each subject and generate 4 images for each prompt template, totaling 3,000 images.

Settings. We adopt the pretrained Stable Diffusion [30] version 1.4 as the text-to-image framework. We use DDIM with 50 steps for the generation process. For HiFi Tuner based on Textual Inversion, we implement both the learning of subject textual embeddings described in Section 3.2 and the reference-guided generation described in Section 3.3.

source image textual inversion ours (TI) DreamBooth ours (DB)

[Figure 58]

[Figure 59]

[Figure 60]

“A backpack on top of green grass with sunflowers around it.”

- (a)
- (b)
- (c)
- (d)
- (e)
- (f)

[Figure 61]

“A backpack on top of the sidewalk in a crowded street.”

[Figure 62]

[Figure 63]

[Figure 64]

“A can with a tree and autumn leaves in the background.”

[Figure 65]

“A candle in the jungle.”

[Figure 66]

“A race car toy in the snow.”

[Figure 67]

“A robot toy sits on top of a wooden table.”

Figure 4. Qualitative comparison. We implement our fine-tuning method based on both Textual Inversion (TI) and DreamBooth (DB). A visible improvement is made by comparing the images in the third column with those in the second column and comparing the images in the fifth column and those in the forth column.

We use 5 tokens for cs and adopts an ADAM [17] optimizer with a learning rate 5e−3 to optimize it. We first optimize

cs for 1000 steps and then recursively optimize cts for 10 steps per denoising step. We set ts = 40 and te = 10 and use an ADAM [17] optimizer with a learning rate 1e−2 to optimize ϕh. We optimize ϕh for 10 steps per DDIM denoising step. For HiFi Tuner based on DreamBooth, we follow the original subject representation learning process and implement the reference-guided generation described

in Section 3.3. We use the same optimization schedule to optimize ϕh as mentioned above. For the reference-guided generation, we only apply HiFi Tuner to the 21 rigid objects, because their appearances vary little and have strong need for the detail preservation.

Evaluation metrics. Following DreamBooth [31], we use DINO score and CLIP-I score to measure the subject fidelity and use CLIP-T score the measure the prompt fidelity. CLIP-I score is the average pairwise cosine similar-

Table 1. Quantitative comparison. Method DINO ↑ CLIP-I ↑ CLIP-T ↑ Real images 0.774 0.885 N/A Stable Diffusion 0.393 0.706 0.337 Textual Inversion [9] 0.569 0.780 0.255 Ours (Textual Inversion) 0.665 0.807 0.291 DreamBooth [31] 0.668 0.803 0.305 Ours (DreamBooth) 0.680 0.809 0.317

A cat sits on a table next to a vase of tulips

𝑐

A dog sits on a table next to a vase of tulips

| | | | | |
|---|---|---|---|---|

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

source images input image result image

Figure 5. Results for personalized subject replacement.

Table 2. Ablation study. Method DINO ↑ CLIP-I ↑ CLIP-T ↑

ity between CLIP [28] embeddings of generated images and real images, while DINO score calculates the same cosine similarity but uses DINO [3] embeddings instead of CLIP embeddings. As pointed out in the DreamBooth paper [31], DINO score is a better means than CLIP-I score in measuring the subject detail preservation. CLIP-T score is the average cosine similarity between CLIP [28] embeddings of the pairwise prompts and generated images.

Baseline (Textual Inversion) 0.567 0.786 0.293 + mask 0.606 0.788 0.292 + regularization 0.612 0.789 0.294 + step-wise representations 0.626 0.790 0.292 + reference guidance 0.665 0.807 0.291

Baseline (DreamBooth) 0.662 0.803 0.315 + reference guidance 0.680 0.809 0.317

Qualitative comparison. Fig. 4 shows the qualitative comparison between HiFi Tuner and other fine-tuning frameworks. HiFi Tuner possesses three advantages compared to other methods. First, HiFi Tuner is able to diminish the unwanted style change for the generated subjects. As shown in Fig. 4 (a) & (b), DreamBooth blends sun flowers with the backpack, and both DreamBooth and Textual Inversion generate backpacks with incorrect colors; HiFi Tuner maintains the styles of the two backpacks. Second, HiFi Tuner can better preserve details of the subjects. In Fig. 4 (c), Textual Inversion cannot generate the whale on the can while DreamBooth generate the yellow part above the whale differently compared to the original image; In Fig. 4 (d), DreamBooth generates a candle with a white candle wick but the candle wick is brown in the original image. Our method outperforms Textual Inversion and DreamBooth in preserving these details. Third, HiFi Tuner can better preserve the structure of the subjects. In Fig. 4 (e) & (f), the toy car and the toy robot both have complex structures to preserve, and Textual Inversion and DreamBooth generate subjects with apparent structural differences. HiFi Tuner makes improvements on the model’s structural preservation capability.

the original implementations, which results in higher CLIPT scores but lower DINO scores for the baselines. Thereafter, we can use our techniques to improve the subject fidelity so that both DINO scores and CLIP-T scores can surpass the original implementations. For HiFi Tuner based on Textual Inversion, we fine-tune the textual embeddings with 1000 steps. The four proposed techniques make steady improvements over the baseline in DINO score while maintain CLIP-T score. The method utilizing all of our proposed techniques makes a remarkable 9.8-point improvement in DINO score over the baseline. For HiFi Tuner based on DreamBooth, we fine-tune all the diffusion model weights with 400 steps. By utilizing the reference-guided generation, HiFi Tuner achieves a 1.8-point improvement over the baseline in DINO score.

Results for personalized subject replacement. We show the qualitative results in Figure 5. More results can be found in the Appendix C.

### 5. Conclusions

Quantitative comparison. We show the quantitative improvements HiFi Tuner makes in Table 1. HiFi Tuner improves Textual Inversion for 9.6 points in DINO score and 3.6 points in CLIP-T score, and improves DreamBooth for 1.2 points in DINO score and 1.2 points in CLIP-T score.

In this work, we introduce a parameter-efficient fine-tuning method that can boost the sample fidelity and the prompt fidelity based on either Textual Inversion or DreamBooth. We propose to use a mask guidance, a novel parameter regularization technique and step-wise subject representations to improve the sample fidelity. We invents a reference-guided generation technique to mitigate the unwanted variations and artifacts for the generated subjects. We also exemplify that our method can be extended to substitute a subject in an image with personalized item by textual manipulations.

Ablation studies. We present the quantitative improvements of adding our proposed techniques in Table 2. We observe that fine-tuning either DreamBooth or Textual Inversion with more steps leads to a worse prompt fidelity. Therefore, we fine-tune the networks with fewer steps than

### References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18208–18218, 2022. 2
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 2
- [3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision (ICCV), pages 9650–9660, 2021. 5, 8
- [4] Arantxa Casanova, Marlene Careil, Jakob Verbeek, Michal Drozdzal, and Adriana Romero Soriano. Instanceconditioned gan. Advances in Neural Information Processing Systems, 34:27517–27529, 2021. 2
- [5] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Rui, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. arXiv preprint arXiv:2304.00186, 2023. 2
- [6] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W. Cohen. Re-imagen: Retrieval-augmented text-to-image generator. In The Eleventh International Conference on Learning Representations (ICLR), 2023. 2
- [7] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. In The Eleventh International Conference on Learning Representations (ICLR), 2022. 1
- [8] Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381–395, 1981. 5
- [9] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In International Conference on Learning Representations (ICLR), 2023. 1, 2, 3, 8, 11
- [10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2
- [11] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE/CVF international conference on computer vision (ICCV), pages 2961– 2969, 2017. 5
- [12] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In International Conference on Learning Representations (ICLR), 2023. 2, 3, 4, 6

- [13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3, 4
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems (NeurIPS), 33:6840–6851, 2020. 1
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR),

2022. 1

- [16] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 2
- [17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 7

- [18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 3, 4, 5
- [19] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning (ICML), 2023. 5, 6
- [20] Tianle Li, Max Ku, Cong Wei, and Wenhu Chen. Dreamedit: Subject-driven image editing. arXiv preprint arXiv:2306.12624, 2023. 6
- [21] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 5
- [22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision (ICCV), pages 10012–10022, 2021. 5
- [23] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11461–11471, 2022. 6
- [24] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations (ICLR), 2022. 2
- [25] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 6038–6047, 2023. 2, 3
- [26] Yotam Nitzan, Kfir Aberman, Qiurui He, Orly Liba, Michal Yarom, Yossi Gandelsman, Inbar Mosseri, Yael Pritch, and

- Daniel Cohen-Or. Mystyle: A personalized generative prior. ACM Transactions on Graphics (TOG), 41(6):1–10, 2022. 2
- [27] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2085–2094, 2021. 2
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning (ICML), pages 8748–8763. PMLR, 2021. 2, 5, 8
- [29] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 1, 2

- [30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 10684–10695, 2022. 1, 2, 3, 6
- [31] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22500–22510, 2023. 1, 2, 5, 6, 7, 8
- [32] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models, 2023. 1
- [33] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems (NeurIPS), 35:36479–36494, 2022. 1, 2
- [34] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems (NeurIPS), 35:25278–25294,

2022. 1

- [35] Shelly Sheynin, Oron Ashual, Adam Polyak, Uriel Singer, Oran Gafni, Eliya Nachmani, and Yaniv Taigman. Knndiffusion: Image generation via large-scale retrieval. arXiv preprint arXiv:2204.02849, 2022. 2
- [36] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023. 2
- [37] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 1

- [38] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations (ICLR), 2020. 3, 5
- [39] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22428–22437, 2023. 6
- [40] Xin Zhang, Jiaxian Guo, Paul Yoo, Yutaka Matsuo, and Yusuke Iwasawa. Paste, inpaint and harmonize via denoising: Subject-driven image editing with pre-trained diffusion model. arXiv preprint arXiv:2306.07596, 2023. 6

### A. Failure analysis of Textual Inversion

[Figure 74]

A robot toy sits on the ground with trees in the background

𝑐

A fancy boot sits on the ground with trees in the background

| | | | | |
|---|---|---|---|---|

Figure 6. A failure analysis of Textual Inversion [9] method.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

We present a failure analysis of Textual Inversion as shown in Figure 6. The source image is a dog sitting on a meadow. For a prompt “a sitting dog”, the generated images mostly contain a dog sitting on a meadow and the dog’s appearance is not well preserved.

[Figure 79]

[Figure 80]

source images input image result image

#### B. Algorithm of optimizing Tθ Please refer to Algorithm 4 for optimizing Tθ.

A toy duck sits on a rock with ocean in the background

𝑐

A candle sits on a rock with ocean in the background

| | | | | |
|---|---|---|---|---|

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Algorithm 4: Algorithm of optimizing Tθ. P(M) ∈ RN×3 returns the coordinates where M == 1 and appends 1’s after the coordinates.

[Figure 85]

[Figure 86]

Result: Tθ∗ Inputs: Mr, Mg Pr = P(Mr), Pg = P(Mg) for l = [1, ..., L] do

source images input image result image

A cat sits in a jungle with grass around it

s = 0 Pt = Tθ(Pr) for pt ∈ Pt do

𝑐

A dog sits in a jungle with grass around it

| | | | | |
|---|---|---|---|---|

m = MAX FLOAT for pg ∈ Pg do

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

x = ∥pt − pg∥22 if x < m then

[Figure 91]

[Figure 92]

m = x s = s + m

θ = θ − η∇θs Tθ∗ = Tθ

source images input image result image

Figure 7. Results for personalized subject replacement.

### C. Results for personalized subject replacement

We show more results for the personalized subject replacement in Figure 7.

