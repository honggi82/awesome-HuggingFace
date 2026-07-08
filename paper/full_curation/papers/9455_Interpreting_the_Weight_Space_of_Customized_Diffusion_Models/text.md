### Interpreting the Weight Space of Customized Diffusion Models

Amil Dravid∗1,2 Yossi Gandelsman∗1 Kuan-Chieh Wang2

Rameen Abdal3 Gordon Wetzstein3 Alexei A. Efros1 Kfir Aberman2

# arXiv:2406.09413v3[cs.CV]22Nov2024

1UC Berkeley 2Snap Inc. 3Stanford University

weights2weights (w2w) Space

Diffusion Unet

weights

beard

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Single Image Inversion Identity Editing Sampling

Figure 1: weights2weights (w2w) space enables controllable creation of new customized diffusion models. We model a manifold of customized diffusion models as a subspace of weights that encodes different instances of a broad visual concept (e.g., human identities, dog breeds, etc.). This forms a space that supports inverting the subject (e.g., identity) from a single image into a model, editing the subject encoded in the model, and sampling new models that encode new instances of the visual concept. Each of these operations results in a new model that can consistently generate the subject.

#### Abstract

We investigate the space of weights spanned by a large collection of customized diffusion models. We populate this space by creating a dataset of over 60,000 models, each of which is a base model fine-tuned to insert a different person’s visual identity. We model the underlying manifold of these weights as a subspace, which we term weights2weights. We demonstrate three immediate applications of this space that result in new diffusion models – sampling, editing, and inversion. First, sampling a set of weights from this space results in a new model encoding a novel identity. Next, we find linear directions in this space corresponding to semantic edits of the identity (e.g., adding a beard), resulting in a new model with the original identity edited. Finally, we show that inverting a single image into this space encodes a realistic identity into a model, even if the input image is out of distribution (e.g., a painting). We further find that these linear properties of the diffusion model weight space extend to other visual concepts. Our results indicate that the weight space of fine-tuned diffusion models can behave as an interpretable meta-latent space producing new models.1

∗Equal contribution 1Project page: https://snap-research.github.io/weights2weights

Code: https://github.com/snap-research/weights2weights

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

[Figure 14]

| |Sampling<br><br>|Editing|Inversion|
|---|---|---|---|
|GANLatent space<br><br>|latent z → image|z + Δzchubby →image editing|[Figure 15]<br><br>[Figure 16]<br><br>→z→<br><br>image reconstructed image|
|[Figure 17]<br><br>w2w space<br><br>|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>model θ ∈ w2w→ identity<br><br>A [v] person…|[Figure 21]<br><br>θ + Δθchubby→ identity editing<br><br>A [v] person…|[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>→θ→<br><br>image reconstructed identity|

###### w2wGANspaceLatent space

[Figure 26]

[Figure 27]

- Figure 2: The weights2weights space operates as a meta-latent space. Unlike a traditional generative latent space, w2w space controls the model itself rather than single image instances. New identity-encoding models can be sampled from the space and edited by linearly traversing along semantic directions in weight space. Additionally, a single image can be inverted into the space to produce a model that consistently generates that identity.

#### 1 Introduction

Generative models have emerged as a powerful tool to model our rich visual world. In particular, the latent space of single-step generative models, such as generative adversarial networks (GANs) [18, 28], has been shown to linearly encode meaningful concepts in the output images. For instance, datasets of latent vectors were used to discover linear directions in the GAN latent space encoding different attributes (e.g., gender or age of faces) [20, 60]. Even earlier, datasets of images and keypoints were leveraged to discover subspaces of facial shape and appearance [6, 53].

We aim to extend this even further, using datasets of model weights instead datasets of images or latents. Can we discover such interpretable subspaces in the model weights themselves? Recently introduced personalization approaches, such as Dreambooth [54] or Custom Diffusion [34], may hint that this is the case. These methods aim to learn an instance of a subject, such as a person’s visual identity. Rather than searching for a latent code that represents an identity in the input noise space, these approaches customize diffusion models by fine-tuning on subject-specific images, which results in identity-specific model weights. We therefore hypothesize that a latent space can exist in the weights themselves.

To test our hypothesis, we fine-tune over 60,000 personalized models on individual identities to obtain points that lie on a manifold of customized diffusion model weights. To reduce the dimensionality of each data point, we use low-rank approximation (LoRA) [23] during fine-tuning and further apply Principal Components Analysis (PCA) to the set of data points. This forms our final space: weights2weights (w2w). Unlike traditional generative models like GANs, which model the pixel space of images, we model the weight space of these personalized models. Thus, each sample in our space corresponds to an identity-specific model which can consistently generate that subject. We provide a schematic in Fig. 2 that contrasts a typical latent space with our proposed w2w space, demonstrating the differences and analogies between these two representations. w2w space can be thought of as a meta-latent space, enabling controllable creation of new models instead of just images like a traditional latent space.

Creating this space unlocks a variety of applications that involve traversal in w2w (Fig. 1). First, we demonstrate that sampling model weights from w2w space corresponds to a new model encoding a novel subject. Second, we find linear directions in this space corresponding to semantic edits of the identity. Finally, we show that enforcing weights to live in this space enables a diffusion model to learn a subject given a single image, even if it is out of distribution.

We find that w2w space is highly expressive through quantitative evaluation on editing customized models and encoding new identities given a single image. Qualitatively, we observe this space supports sampling models that encode diverse and realistic identities, while also capturing the key characteristics of out-of-distribution identities. We finally demonstrate that similar weight subspaces exist for other visual concepts such as dog breeds and car types.

[Figure 28]

- Figure 3: Building weights2weights (w2w) space. We create a dataset of model weights where each model is personalized to a specific identity using low-rank updates (LoRA). These model weights lie on a weights manifold that we further project into a lower-dimensional subspace spanned by its principal components. We train linear classifiers to find disentangled edit directions in this space.

#### 2 Related Work

Image-based generative models. Various models have been proposed for image generation, including variational autoencoders (VAEs) [31], flow-based models [12, 32, 49], generative adversarial networks (GANs) [18], and diffusion models [22, 43, 62]. Within the realm of high-quality photorealistic image generation, GANs [25, 28, 29] and diffusion models [22, 43, 52, 63] have garnered significant attention due to their controllability and ability to produce high-quality images. Leveraging the compositionality of these models, methods for personalization and customization have been developed which aim to insert user-defined concepts via fine-tuning [16, 34, 40, 54]. Various works try to reduce the dimensionality of the optimized parameters for personalization either by operating in specific model layers [34] or in text-embedding space [16], by training hypernetworks [55], and by constructing a linear basis in text embedding space [73].

Latent space of generative models. Linear latent space models of facial shape and appearance were studied extensively in the 1990s, using PCA-based representations (e.g. Active Appearance Models [9], 3D Morphable Models [6]) as well as operating directly in pixel and keypoint space [53]. However, these techniques were restricted to aligned and cropped frontal faces. More recently, generative adversarial networks (GANs), particularly the StyleGAN series [26, 27, 28, 29], have showcased editing capabilities facilitated by their interpretable latent space. Furthermore, linear directions can be found in their latent space to conduct semantic edits by training linear classifiers or applying PCA [20, 60], among other methods for discovering semantic directions [8, 66]. Several methods aim to project real images into the GAN latent space in order to conduct this editing [1, 3, 51, 64, 77]. Beyond the latent space, works such as [4] found that directions could be discovered in the neuron activation space, suggesting the interpretability of weights.

Although diffusion models architecturally lack a GAN-like latent space, some works aim to discover similar spaces in these models. This has been explored in the UNet bottleneck layer [35, 41], noise space [10, 78], and text-embedding space [5]. Concept Sliders [17] explores the weight space for semantic image editing by conducting low-rank training with contrasting image or text pairs.

Weights as data. Past works have exploited the structure within weight space of deep networks for various applications. In particular, some have found linear properties of weights, enabling simple model ensembling and editing via arithmetic operations [24, 56, 59, 69]. Other works create datasets of neural network parameters for training hypernetworks [14, 19, 42, 55, 68], predicting properties of networks [58], and creating design spaces for models [46, 47].

#### 3 Method

We start by demonstrating how we create a manifold of model weights as illustrated in Fig. 3. We explain how we obtain low-dimensional data points for this space, each of which represents an individual subject from a broad class (i.e., identity). We then use these points to model a weights manifold. Next, we find linear directions in this manifold that correspond to semantic attributes and use them for editing the identities. Finally, we demonstrate how this manifold can be utilized for constraining an ill-posed inversion task with a single image to reconstruct its identity.

###### 3.1 Preliminaries

In this section, we first introduce latent diffusion models (LDM) [52], which we will use to create a dataset of weights. Then, we explain the approach for obtaining identity-specific models from LDM via Dreambooth [54] fine-tuning. We finally present a version of fine-tuning that uses lowdimensional weight updates (LoRA [23]). We will use the fine-tuned low-dimensional per-identity weights as data points to construct the weights manifold in Sec. 3.2.

Latent diffusion models [52]. We will extract weights from latent diffusion models to create w2w space. These models follow the standard diffusion objective [22] while operating on latents extracted from a pre-trained Variational Autoencoder [15, 31, 50]. With text, the conditioning signal is encoded by a text encoder (such as CLIP [44]), and the resulting embeddings are provided to the denoising UNet model. The loss of latent diffusion models is:

Ex,c,ϵ,t[wt||ϵ − ϵθ(xt,c,t)||22], (1)

where ϵθ is the denoising UNet, xt is the noised version of the latent for an image, c is the conditioning signal, t is the diffusion timestep, and wt is a time-dependent weight on the loss.

To sample from the model , a random Gaussian latent xT is deterministically denoised conditioned on a prompt for a fixed set of timesteps with a DDIM sampler [63]. The denoised latent is then fed through the VAE decoder to generate the final image.

Dreambooth [54]. To obtain an identity-specific model, we use the Dreambooth personalization method. Dreambooth fine-tuning introduces a novel subject into a pre-trained diffusion model given only a few images of it. During training, Dreambooth follows a two-part objective:

Ex,c,ϵ,t[wt||ϵ − ϵθ(xt,c,t)||22 + λwt′||ϵ′ − ϵθ(x′t,c′,t′)||22], (2)

where the first term corresponds to the standard diffusion denoising objective using the subjectspecific data x conditioned on the text prompt “[identifier] [class noun]” (e.g., “[v] person”), denoted c. The second term, weighted by λ, corresponds to a prior preservation loss, which involves the standard denoising objective using the model’s own generated samples x′ for the broader class c′ (e.g., “person”). This prevents the model from associating the class name with the specific instance, while also leveraging the semantic prior on the class.

Low Rank Adaptation (LoRA) [23]. Dreambooth requires fine-tuning all the weights of a model, which is a high–dimensional space. We turn to a more efficient fine-tuning scheme, LoRA, that modifies only a low-rank version of the weights. LoRA uses weight updates ∆W with a low intrinsic rank. For a base model layer W ∈ Rm×n, the LoRA update for that layer ∆W can be decomposed into ∆W = BA, where B ∈ Rm×r and A ∈ Rr×n are low-rank matrices with r ≪ min(m,n). During training, for each model layer, only the A and B are updated. This significantly reduces the number of trainable parameters. During inference, the low-rank weights are added residually to the weights of each layer in the base model and scaled by a coefficient α ∈ R: W + α∆W.

###### 3.2 Constructing the weights manifold

Creating a dataset of model weights. To construct the weights2weights (w2w) space, we begin by creating a dataset of model weights θi. We conduct Dreambooth fine-tuning on latent diffusion models in order to insert new subjects with the ability to control image instances using text prompts. This training is done with LoRA in order to reduce the space of model parameters. Each model is fine-tuned on a set of images corresponding to one human subject. After training, we flatten and concatenate all of the LoRA matrices, resulting in a data point θi ∈ Rd which represents one identity. After training over N different instances, we have our final dataset of model weights D = {θ1,θ2,...,θN}, representing a diverse array of subjects.

Modeling the weights manifold. We posit that our data D ⊆ Rd lies on a lower-dimensional manifold of weights that encode identities. A randomly sampled set of weights in Rd, would not be guaranteed to produce a valid model encoding identity as the d degrees of freedom can be fine-tuned for any purpose. Therefore, we hypothesize that this manifold is a subset of the weight space. Inspired by findings that high-level concepts can be encoded as linear subspaces of representations [13, 37, 45, 48], we model this subset as a linear subspace Rm where m < d, and call it weights2weights (w2w) space. We represent points in this subspace as a linear combination

of basis vectors w = {w1,...,wm}, wi ∈ Rd. In practice, we apply Principal Component Analysis (PCA) on the N models and keep the first m principal components for dimensional reduction and forming our basis of m vectors.

Sampling from the weights manifold. After modeling this weights manifold, we can sample a new model that lies on it, resulting in a new model that generates a novel identity. We sample a model represented with basis coefficients {β1,...,βm}, where each coefficient βk is sampled from a normal distribution with mean µk and standard deviation σk. The mean and standard deviation are calculated for each principal component k from the coefficients among all the training models.

###### 3.3 Finding Interpretable Weight Space Directions

We seek a direction n ∈ Rd defining a hyperplane that separates between binary identity properties embedded in the model weights (e.g., male/female), similarly to hyperplanes observed in the latent space of GANs [60]. We assume binary labels are given for attributes present in the identities encoded by the models. We then train linear classifiers using weights of the models as data based on these labels, imposing separating hyperplanes in weight space. Given an identity parameterized by weights θ, we can manipulate a single attribute by traversing in a direction n, orthogonal to the separating hyperplane: θedit = θ + αn. An edit operation in w2w spaee produces a new model with the original subject edited, allowing the model to generate infinitely many new images of the edited subject.

###### 3.4 Inversion into w2w Space

Traditionally, inversion of a generative model involves finding an input such as a latent code that best reconstructs a given image [38, 70]. This corresponds to finding a projection of the input onto the learned data manifold [77]. With w2w space, we model a manifold of model weights rather than images. Inspired by latent optimization methods [1, 77], we propose a gradient-based method of inverting a single identity from an image into our discovered space.

Given a single image x, we follow a constrained denoising objective:

Ex,c,ϵ,t[wt||ϵ − ϵθ(xt,c,t)||22] s.t. θ ∈ w2w (3)

max

θ

Specifically, we constrain the model weights to lie in w2w space by optimizing a set of basis coefficients {β1,...,βm} rather than the original parameters. Unlike Dreambooth, we do not employ a prior preservation loss, since the optimized model lies in the subspace defined by our dataset of weights, and inherits their priors.

#### 4 Experiments

We demonstrate w2w space on the visual concept of human identities for a variety of applications. We begin by detailing implementation details. Next, we use w2w space for 1) sampling new models encoding novel identities, 2) editing identity attributes in a consistent manner via linear traversal in w2w space, 3) embedding a new identity given a single image, and 4) projecting out-of-distribution identities into w2w space. Finally, we analyze how scaling the number of models in our dataset of model weights affects the disentanglement of attribute directions and preservation of identity.

###### 4.1 Implementation Details

Creating an identity dataset. We generate a synthetic dataset of ∼65,000 identities using [67], where each identity is associated with multiple images of that person. Each identity is based on an image with labeled binary attributes (e.g., male/female) from CelebA [36]. Each set of images corresponding to an identity is then used as data to fine-tune a latent diffusion model with Dreambooth. Further details on this dataset and train/test splits are provided in Appendix E.

Encoding identities into model weights. We conduct Dreambooth fine-tuning using LoRA with rank 1 on the identities. Following [56], we only fine-tune the query and value projection matrices in the cross-attention layers. We utilize the RealisticVision-v51 checkpoint2 based on Stable Diffusion

2https://huggingface.co/stablediffusionapi/realistic-vision-v51

1.5. Conducting Dreambooth fine-tuning on each identity training set results in a dataset of ∼65,000 weights θ where θ ∈ R100,000.

Finding semantic attribute directions. We utilize binary attribute labels from CelebA to train linear classifiers on the dataset of model weights we curated. We run Principal Component Analysis (PCA) on the ∼65,000 training models and project to the first 1000 principal components in order to reduce the dimensionality. The orthogonal edit directions are calculated via the analytic least squares solution on the matrix of projected training models D ∈ R65,000×1000, and then unprojected to the original dimensionality of the model weights: θ ∈ R100,000.

###### 4.2 Sampling from w2w Space

|Sampled Identity|Nearest Neighbor|
|---|---|
|[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]|[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]|
|[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]|[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]|
|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]|[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]|

We present images generated from models that were sampled from the weights manifold (i.e., w2w Space) in Fig. 4. We follow the sampling procedure from Sec. 3.2, and generate images from the sampled model. As shown, each new model encodes a novel, realistic, and consistent identity. Additionally, we present the nearest neighbor model among the training dataset of model weights. We use cosine similarity on the models’ principal component representations. Comparing with the nearest neighbors shows that these samples are not just copies from the dataset, but rather encode diverse identities with different attributes. Yet, the samples still demonstrate some similar features to the nearest neighbors. These include jawline and eye shape (top row), facial hair (middle row), and nose and eye shape (bottom row). Appendix A includes more such examples.

Figure 4: Identity samples from w2w space. We show the samples from w2w space do not overfit to nearest-neighbor identities, although they incorporate facial attributes from them. The identities are diverse and consistent across generations.

###### 4.3 Editing Subjects

Bald

Age

Chubby

We demonstrate how directions found by the linear classifiers can be used to edit subjects encoded in the models. It is desired that these edits are 1) disentangled (i.e., do not interfere with other attributes of the embedded subject and preserve all other concepts such as context) 2) identity preserving (i.e., the person is still recognizable) 3) and semantically aligned with the intended edit.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

BlackHair

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Beard

Baselines. We compare against a naïve baseline of prompting with the desired attribute (e.g., “[v] person with small eyes”), and then Concept Sliders [17], an instance-specific editing method which we adapt to subject editing. In particular, we train their most accessible method, the text-based slider, which trains LoRAs to modulate attributes in a pretrained diffusion model based on contrasting text prompts. We then apply these sliders to the personalized identity models.

Pale

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Woman Chubby Narrow Eyes

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Promptingw2wSliders

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

- Figure 5: Qualitative comparison. w2w edits preserve identity while being disentangled and semantically aligned. Concept Sliders [17] tends to exaggerate effects which induces artifacts and degrades identity, while prompting the subject with the desired edit has unexpected effects.

###### Table 1: Edits in w2w space preserve identity, are disentangled, and semantically aligned.

###### ID Score ↑ LPIPS ↓ CLIP Score ↑

Prompting Sliders w2w Prompting Sliders w2w Prompting Sliders w2w

Gender 0.39±0.08 0.33±0.09 0.45±0.09 0.30±0.05 0.39±0.09 0.31±0.03 1.98±0.78 3.50±0.68 4.13±0.59 Chubby 0.29±0.14 0.33±0.09 0.45±0.09 0.41±0.05 0.38±0.04 0.36±0.04 1.12±0.61 2.21±0.61 2.16±0.51

Eyes 0.52±0.06 0.53±0.04 0.72±0.05 0.32±0.03 0.30±0.02 0.19±0.02 0.17±0.17 0.01±0.22 0.59±0.19

Original + Flat Brows + Bangs + Straight Hair Original + Jawline + Eye Bags + Narrow Eyes

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

- Figure 6: Composing edits in w2w space. Each column represents fixed seed samples from an edited model. Multiple edits in w2w space minimally degrade the original identity or interfere with other concepts, while maintaining edit appearance across different samples.

Evaluation protocol. We evaluate these three methods for identity preservation, disentanglement, and edit coherence. To measure identity preservation, we first detect faces in the original generated images and the result of the edits using MTCNN [74]. We then calculate the similarity of the FaceNet [57] embeddings. We also use LPIPS [76] computed between the images before and after the edit to measure the degree of disentanglement with other visual elements, and CLIP score [21], to measure if the desired edit matches the text caption for the edit.

To generate samples, we fix a set of prompts and random seeds which are used as input to the held-out identity models. Then, we choose a set of identity-specific manipulations. For prompt-based editing, we augment the attribute description to the set of fixed prompts (e.g., “chubby [v] person"). For Concept Sliders and w2w, we apply the weight space edit directions to the personalized model with a fixed norm which determines the edit strength. The norm is calculated using the maximum projection component onto the edit direction among the training set of model weights.

w2w edits are identity preserving and disentangled. We evaluate over a range of identity-specific attributes and present three (gender, chubby, narrow eyes) in Tab. 1. Edits in w2w preserve the identity of the original subject as measured by the ID score. These edits are semantically aligned with the desired effect as indicated by the CLIP score while minimally interfering with other visual concepts, as measured by LPIPS. We note that the CLIP score can be noisy in this setting as text captions can be too coarse to describe attributes as nuanced as those related to the human face. We supplement this with a user study presented in Appendix B.

Qualitatively, w2w edits make the minimal amount of changes to achieve semantic and identitypreserving edits (Fig. 5). For instance, changing the gender of the man does not significantly change the facial structure or hair, unlike Concept Sliders or prompting with text descriptions. Prompting has inconsistent results, either creating no effect or making drastic changes. Concept Sliders tends to make caricaturized effects, such as making the man cartoonishly chubby and baby-like.

Composing edits. Edit directions in w2w space can be composed linearly as shown in Fig. 6. The first column represents samples from the original model, and each subsequent column represents samples from the edited models. Each row shares the same fixed random generation seed. The composed edits persist in appearance across different generations, binding to the identity. Furthermore, the edited weights result in a new model, where the subject has different attributes while still maintaining as much of the prior identity. This is in contrast to editing in a traditional latent space, where an edit only corresponds to a single image. Additionally, as we operate on an personalized identity-specific weight manifold, minimal changes are made to other concepts, such as scene layout or other people. For instance, in Fig. 6, adding edits to the woman does not interfere with the person standing by her.

[Figure 127]

Input +Chubby Input Inversion +Pointy Nose Input Inversion +Hair

Inversion

[Figure 128]

|[Figure 129]|
|---|

|[Figure 130]|
|---|

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

posinginadressfacecloseup

withTaylorSwiftasastatue

inabluedresssmiling

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

- Figure 7: Single image inversion reconstructs identity and enables editing in w2w space. We present generated samples from the inverted models. These inverted identities can be composed in novel contexts and edited using our discovered semantic directions in weight space. These edits persist in appearance across generation seeds and prompts.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

|[Figure 146]|
|---|

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Input Projection Input Projection Input Projection

- Figure 8: Projecting out-of-distribution identities. We show that our inversion method can convert unrealistic identities into realistic renderings with in-domain facial features. Each image represents a generated sample from the inverted model. The resulting identities can be composed in novel scenes, such as playing tennis or rendered into other artistic domains.

###### 4.4 Inverting Subjects

Evaluation protocol. We measure w2w space’s ability to represent novel identities by inverting a set of 100 random FFHQ [28] face images. We follow our inversion objective from eq. 3. We then provide a set of diverse prompts to generate multiple images and follow the identity preservation metric from Sec. 4.3 to measure subject fidelity. Implementation details are provided in Appendix C.

We compare our results to two approaches that use Dreambooth with rank-1 LoRA. The first is trained on a single image. The second is trained on multiple images of each identity. We generate such images by following our identity dataset construction from Sec. 4.1. This approach can be viewed as a pseudo-upper bound on modeling identity as it uses multiple images.

w2w space provides a strong identity prior. Inverting a single image into w2w space improves on the single image Dreambooth baseline and closes the gap with the Dreambooth baseline that uses multiple identity images (Tab. 2). Conducting Dreambooth fine-tuning with a single image in the original weight space leads to image overfitting and poor subject reconstruction as indicated by a lower ID score. In contrast, by constraining the optimized weights to lie on a manifold of identity weights, w2w inversion inherits the rich priors of the models used to discover the space. As such, it can extract a high-fidelity identity that is consistent and compositional across generations. We present qualitative comparisons against Dreambooth and single-image Dreambooth in Appendix C. We additionally compare against other personalization methods in that section.

Inverted models are editable. Fig. 7 demonstrates that a diverse set of identities can be faithfully represented in w2w space. After inversion, the encoded identity can be composed in novel contexts and poses. For instance, the inverted man (rightmost example) can be seen posing with a celebrity or rendered as a statue. Moreover, semantic edits can be applied to the inverted models while maintaining appearance across generations.

Table 2: w2w Inversion closes the gap with Dreambooth.

Method Single-Image ID Score ↑

DB-LoRA × 0.69 ± 0.01 DB-LoRA ✓ 0.43 ± 0.03 w2w ✓ 0.64 ± 0.01

###### 4.5 Out-of-Distribution Projection

w2w space captures out-of-distribution identities. We follow the w2w inversion method from Sec. 4.4 to project images of unrealistic identities (e.g., paintings, cartoons, etc.) onto the weights manifold, and present these qualitative results in Fig. 8. By constraining the optimized model to live in w2w space, the inverted identities are converted into realistic renditions of the stylized identities, capturing prominent facial features. In Fig. 8, notice how the inverted identities generate a similar

1.0

##### Entanglement

0.8

0.6

0.4

Mean

0.2

Black Hair - Pale Skin

Young - Bald Male - Beard

0.0

101 102 103 104 105

# of Models

0.7

0.6

0.5

IDScore

0.4

0.3

0.2

w2w inversion

0.1

DB-LoRA

0.0

102 103 104 105

# of Models

- Figure 9: Scaling dataset of models further disentangles classifier directions. We highlight the trend in disentanglement of three examples where attributes may be strongly correlated among identities. As the number of models is increased, the features are less entangled.

Figure 10: Scaling the number of models improves identity preservation. As the span of w2w space increases, inversion can reconstruct single-image identities more faithfully, approaching the pseudo-upper bound of multi-image Dreambooth (DB-LoRA).

blonde hairstyle and nose structure in the first example, defined jawline and lip shape in the second example, and head shape and big nose in the last example. As also shown in the figure, the inverted identities can also be translated to other artistic domains using text prompts. We present a variety of domains projected into w2w space in Appendix D.

###### 4.6 Effect of Number of Models Spanning w2w Space

We ablate the number of models used to create w2w space and investigate the expressiveness of the resulting space. In particular, we measure the degree of entanglement among the edit direction and how well this space can capture identity.

Disentanglement vs. the number of models. We find that scaling the number of models in our dataset of weights leads to less entangled edit directions in w2w space (Fig. 9). We vary the number of models in our dataset of weights and reapply PCA to establish a basis. We then measure the absolute value of cosine similarity (lower is better) between all pairs of linear classifier directions found for CelebA labels. We repeat this as we scale the number of model weights used to train the classifiers. We report the mean and standard deviation for these scores, along with three notable semantic direction pairs. We observe a trend in decreasing cosine similarity. Notably, pairs such as “Black Hair - Pale Skin,” “Young - Bald,” and “Male - Beard” which may correlate in the distribution of identities, become less correlated as we scale our dataset of model weights.

Identity preservation vs. the number of models. We observe that as we scale the number of models in our dataset of weights, identities are more faithfully represented in w2w space (Fig. 10). We follow the same procedure as the disentanglement ablation, reapplying PCA to establish a basis based on the dataset of model weights. Next, following Sec. 4.4, we optimize coefficients for this basis and measure the average ID score over the 100 inverted FFHQ evaluation identities. As each model in our dataset encodes a different instance of an identity, growing this dataset increases the span of w2w space and its ability to capture more diverse identities. We plot the average multi-image Dreambooth LoRA (DB-LoRA) ID score from Sec. 4.4, which is agnostic to our dataset of models. This establishes a pseudo-upper bound on identity preservation. Scaling enables w2w to represent identities given a single image with performance approaching that of traditional Dreambooth with LoRA, which uses multiple images and trains in a higher dimensional space.

#### 5 Extending to Other Domains

We extend our hypothesis of interpretable linear weight subspaces in diffusion models to other visual concepts beyond human identities. We apply the weights2weights framework to form a subspace for models encoding different dog breeds. To create a dataset for fine-tuning, we generate images with Stable Diffusion based on each of the 120 dog classes from ImageNet [11]. We then conduct Dreambooth fine-tuning on each set of dog breed images to create a dataset of 120 dog-encoding models, subsequently applying PCA. To find edit directions, we use GPT-4 [2] to create labels for

Small Large Van Sports Car

[Figure 157]

Figure 11: weights2weights linear subspaces can be created for other visual concepts. We follow the same procedure of applying PCA and finding edit directions with linear classifiers on datasets of models encoding dog breeds and models encoding car types.

each dog breed (e.g., wavy hair or not) and then train linear classifiers on the model weight principal component projections like Sec. 3.3. We apply the same framework to different car categories using models fine-tuned on images from a dataset of 197 different car types [33]. We present results for traversing edit directions in these two subspaces in Fig. 11. Each column represents samples from an edited model. Each row shares the same fixed random generation seed.

Our results provide further evidence that diffusion models can encode visual concepts linearly. This enables the creation of new models in a controlled manner via simple interpolation. For instance, in Fig. 11, we rewrite the model’s learned concept of a small golden retriever to make it bigger, or the model’s encoding of a red van to make it a sports car. Additionally, unlike older PCA-based methods [6, 53, 61, 65] which rely on aligned pixels or keypoints of human faces, weights2weights can extend to other domains beyond human identities. We refer the reader to Appendix G for more results of applying w2w space to other visual concepts.

- 6 Limitations Input Inversion

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Figure 12: weights2weights fails to capture identities with undersampled attributes.

As with any data-driven method, w2w space inherits the biases of the data used to discover it. For instance, co-occurring attributes in the identity-encoding models would cause linear classifier directions to entangle them (e.g. gender and facial hair). However, as we scale the number of models, spurious correlations will drop as evidenced by Fig. 9. These semantic directions are also limited by the labels present in CelebA. Additionally, the span of the w2w space is dictated by the models used to create it. Thus, w2w space can struggle to represent more complex identities as seen in Fig. 12. Inversion in these cases amounts to projecting onto the closest identity on the weights manifold. Despite this, our analysis on the size of the model dataset reveals that forming a space using a larger and more diverse set of identity-encoding models can mitigate this limitation.

- 7 Discussion and Broader Impact

We presented a paradigm for representing diffusion model weights as a point in a subspace defined by other customized models – weights2weights (w2w) space. This enabled applications analogous to those of a generative latent space – inversion, editing, and sampling – but producing model weights rather than images, resulting in what we term a meta-latent space. We demonstrated these applications on model weights encoding human identities and extended this space to other visual concepts. Although these applications could enable malicious manipulation of real human identities and model weights, we hope the community uses the framework to explore visual creativity as well as utilize this interpretable space for controlling models for safety.

#### Acknowledgements

The authors would like to thank Grace Luo, Lisa Dunlap, Konpat Preechakul, Sheng-Yu Wang, Stephanie Fu, Or Patashnik, Daniel Cohen-Or, and Sergey Tulyakov for helpful discussions. AD is supported by the US Department of Energy Computational Science Graduate Fellowship. Part of the work was completed by AD as an intern with Snap Inc. YG is funded by the Google Fellowship. Additional funding came from ONR MURI.

#### References

- [1] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In Proceedings of the IEEE/CVF international conference on computer vision, pages 4432–4441, 2019.
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [3] David Bau, Hendrik Strobelt, William Peebles, Jonas Wulff, Bolei Zhou, Jun-Yan Zhu, and Antonio Torralba. Semantic photo manipulation with a generative image prior. ACM Transactions on Graphics (TOG), 38(4):1–11, 2019.
- [4] David Bau, Jun-Yan Zhu, Hendrik Strobelt, Bolei Zhou, Joshua B Tenenbaum, William T Freeman, and Antonio Torralba. Gan dissection: Visualizing and understanding generative adversarial networks. In International Conference on Learning Representations, 2019.
- [5] Stefan Andreas Baumann, Felix Krause, Michael Neumayr, Nick Stracke, Vincent Tao Hu, and Björn Ommer. Continuous, subject-specific attribute control in t2i models by identifying semantic directions. arXiv preprint arXiv:2403.17064, 2024.
- [6] Volker Blanz and Thomas Vetter. A morphable model for the synthesis of 3d faces. In Proceedings of the 26th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH ’99, page 187–194, USA, 1999. ACM Press/Addison-Wesley Publishing Co.
- [7] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22560– 22570, 2023.
- [8] Anton Cherepkov, Andrey Voynov, and Artem Babenko. Navigating the gan parameter space for semantic image editing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3671–3680, 2021.
- [9] Timothy F Cootes, Gareth J Edwards, and Christopher J Taylor. Active appearance models. In Computer Vision—ECCV’98: 5th European Conference on Computer Vision Freiburg, Germany, June 2–6, 1998 Proceedings, Volume II 5, pages 484–498. Springer, 1998.
- [10] Yusuf Dalva and Pinar Yanardag. Noiseclr: A contrastive learning approach for unsupervised discovery of interpretable directions in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24209–24218, 2024.
- [11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [12] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. In International Conference on Learning Representations, 2016.
- [13] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. arXiv preprint arXiv:2209.10652, 2022.

- [14] Ziya Erkoç, Fangchang Ma, Qi Shan, Matthias Nießner, and Angela Dai. Hyperdiffusion: Generating implicit neural fields with weight-space diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14300–14310, 2023.
- [15] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.
- [16] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2022.
- [17] Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. arXiv preprint arXiv:2311.12092, 2023.
- [18] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.
- [19] David Ha, Andrew M Dai, and Quoc V Le. Hypernetworks. In International Conference on Learning Representations, 2016.
- [20] Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. Ganspace: Discovering interpretable gan controls. Advances in neural information processing systems, 33:9841–9850, 2020.
- [21] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528, 2021.
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [23] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2021.
- [24] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations, 2022.
- [25] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10124–10134, 2023.
- [26] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. Advances in neural information

- processing systems, 33:12104–12114, 2020.

[27] Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. Advances in neural information

- processing systems, 34:852–863, 2021.

- [28] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.
- [29] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8110–8119, 2020.
- [30] Diederik P Kingma and Jimmy Ba. Adam: a method for stochastic optimization. In International Conference on Learning Representations, 2014.

- [31] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. International Conference on Learning Representations, 2014.
- [32] Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems, 31, 2018.
- [33] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for finegrained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pages 554–561, 2013.
- [34] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multiconcept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023.
- [35] Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. Diffusion models already have a semantic latent space. In The Eleventh International Conference on Learning Representations, 2022.
- [36] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730–3738, 2015.
- [37] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013.
- [38] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023.
- [39] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4296–4304, 2024.
- [40] Yotam Nitzan, Kfir Aberman, Qiurui He, Orly Liba, Michal Yarom, Yossi Gandelsman, Inbar Mosseri, Yael Pritch, and Daniel Cohen-Or. Mystyle: A personalized generative prior. ACM Transactions on Graphics (TOG), 41(6):1–10, 2022.
- [41] Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, and Youngjung Uh. Understanding the latent space of diffusion models through the lens of riemannian geometry. Advances in Neural Information Processing Systems, 36:24129–24142, 2023.
- [42] William Peebles, Ilija Radosavovic, Tim Brooks, Alexei A Efros, and Jitendra Malik. Learning to learn with generative models of neural network checkpoints. arXiv preprint arXiv:2209.12892, 2022.
- [43] Ryan Po, Wang Yifan, Vladislav Golyanik, Kfir Aberman, Jonathan T Barron, Amit Bermano, Eric Chan, Tali Dekel, Aleksander Holynski, Angjoo Kanazawa, et al. State of the art on diffusion models for visual computing. In Computer Graphics Forum, volume 43, page e15063. Wiley Online Library, 2024.
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [45] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.
- [46] Ilija Radosavovic, Justin Johnson, Saining Xie, Wan-Yen Lo, and Piotr Dollár. On network design spaces for visual recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1882–1890, 2019.

- [47] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network design spaces. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10428–10436, 2020.
- [48] Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7237–7256, 2020.
- [49] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning, pages 1530–1538. PMLR, 2015.
- [50] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International conference on machine learning, pages 1278–1286. PMLR, 2014.
- [51] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Transactions on graphics (TOG), 42(1):1–13, 2022.
- [52] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [53] Duncan A Rowland and David I Perrett. Manipulating facial appearance through shape and color. IEEE computer graphics and applications, 15(5):70–76, 1995.
- [54] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.
- [55] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6527–6536, 2024.
- [56] Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning, 2023.
- [57] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815–823, 2015.
- [58] Konstantin Schürholt, Dimche Kostadinov, and Damian Borth. Self-supervised representation learning on neural network weights for model characteristic prediction. Advances in Neural Information Processing Systems, 34:16481–16493, 2021.
- [59] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. arXiv preprint arXiv:2311.13600, 2023.
- [60] Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Interpreting the latent space of gans for semantic face editing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9243–9252, 2020.
- [61] Lawrence Sirovich and Michael Kirby. Low-dimensional procedure for the characterization of human faces. Josa a, 4(3):519–524, 1987.
- [62] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [63] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2020.

- [64] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for stylegan image manipulation. ACM Transactions on Graphics (TOG), 40(4):1–14, 2021.
- [65] Matthew Turk and Alex Pentland. Eigenfaces for recognition. Journal of cognitive neuroscience, 3(1):71–86, 1991.
- [66] Andrey Voynov and Artem Babenko. Unsupervised discovery of interpretable directions in the gan latent space. In International conference on machine learning, pages 9786–9796. PMLR, 2020.
- [67] Kuan-Chieh Wang, Daniil Ostashev, Yuwei Fang, Sergey Tulyakov, and Kfir Aberman. Moa: Mixture-of-attention for subject-context disentanglement in personalized image generation. arXiv e-prints, pages arXiv–2404, 2024.
- [68] Kuan-Chieh Wang, Paul Vicol, James Lucas, Li Gu, Roger Grosse, and Richard Zemel. Adversarial distillation of bayesian neural network posteriors. In International conference on machine learning, pages 5190–5199. PMLR, 2018.
- [69] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR, 2022.
- [70] Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, and Ming-Hsuan Yang. Gan inversion: A survey. IEEE transactions on pattern analysis and machine intelligence, 45(3):3121–3138, 2022.
- [71] Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023.
- [72] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [73] Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, and Huicheng Zheng. Inserting anybody in diffusion models via celeb basis. Advances in Neural Information Processing Systems, 36, 2024.
- [74] Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, and Yu Qiao. Joint face detection and alignment using multitask cascaded convolutional networks. IEEE signal processing letters, 23(10):1499– 1503, 2016.
- [75] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.
- [76] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [77] Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, and Alexei A Efros. Generative visual manipulation on the natural image manifold. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14, pages 597–613. Springer, 2016.
- [78] Ye Zhu, Yu Wu, Zhiwei Deng, Olga Russakovsky, and Yan Yan. Boundary guided learning-free semantic control with diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

#### A Sampling

We present additional examples of models sampled from w2w space in Fig. 13. The sampled models encode a diverse array of identities which are not copied from the dataset of model weights, as seen by comparing them to the nearest neighbor models from the training set. However, there are attributes borrowed from the nearest neighbors which visually appear in the sampled identity. For instance, the sampled man in the first row shares a similar jawline to the nearest neighbor identity. The sampled identities also demonstrate the same ability as the original training identities to be composed into novel contexts. A variety of prompts are used in Fig. 13 and the identities are still consistent.

|Sampled Identity|Nearest Neighbor|
|---|---|
|[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]|[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]|
|[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]|[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]|
|[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]|[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]|
|[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]|[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]|
|[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>|[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]|
|[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]|[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]|

Figure 13: Sampled identity-encoding models from w2w space and their nearest neighbor models. The sampled identities share some characteristics with the nearest neighbors, but are still distinct. These identities can also be composed into novel contexts like a standard customized diffusion model.

#### B Model Editing in w2w Space

Qualitative Results. We display additional examples of applying edits in w2w space based on the directions discovered using linear classifiers and CelebA labels. In Fig. 14, we demonstrate how the strength of these edits can be modulated and combined with minimal interference. These edits are apparent even in more complex scenes beyond face images. Also, the edits do not degrade other present concepts, such as the dog near the man (top left example).

In Figs. 15 and 16, we demonstrate how multiple edits can be progressively added in a disentangled fashion with minimal degradation to the identity. Additionally, since we operate in a subspace of weight space, these edits persist with a consistent appearance across different generations. For instance, even the man exhibits the edits as a painting in Fig. 15.

Bald

Chubby

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

BlackHair

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Beard

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Pointy Nose

Age

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

Curls

Pale

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

###### Figure 14: Multiple edits can be controlled in a continuous manner.

[Figure 242]

Original +Chubby +Mustache +Big Eyes +Balding Original +Flat Brows +Bangs +Straight Hair +Old

[Figure 244]

Original +Chubby +Mustache +Big Eyes +Balding Original +Flat Brows +Bangs +Straight Hair +Old

[Figure 245]

- Figure 15: Composing four different edits with minimal identity degradation. These edits bind to the identity and persist in appearance across multiple generation seeds and prompts.

Original +Pale +Small Lips +Male

[Figure 246]

[Figure 247]

[Figure 248]

Original +Pointy Nose +Eye Bags +Big Lips

[Figure 249]

[Figure 250]

[Figure 251]

Original +Cheekbones +Curls +Thick Brows

[Figure 252]

[Figure 253]

[Figure 254]

- Figure 16: Additional examples of composing multiple edits. We provide more examples of semantic edits based on labels available from CelebA.

User Study. We present a two-alternative forced choice (2AFC) user study to evaluate the quality of identity edits in Tab. 3. Twenty-five users were given ten sets of images. Each set contained a randomly sampled original image of an identity, and then an image of that identity edited for an attribute using Concept Sliders [17], w2w, and text prompting. Users were then asked to choose between alternate pairs based on three criteria: identity preservation, alignment with the desired edit, and disentanglement. Our results in Tab. 3 show that users have a strong preference towards w2w edits. User instructions and an example question from the study are provided in Figs. 17, 18.

###### Table 3: User study on identity editing.

Method Win Rate (%) ↑ Sliders 28.4

w2w 71.6 Prompting 12.8

w2w 87.2

[Figure 255]

Figure 17: Instructions provided to users for the identity editing user study.

[Figure 256]

Figure 18: Example question from the identity editing user study.

#### C Inversion

We present additional details on w2w inversion and comparisons against training Dreambooth LoRA on a single image vs. multiple images.

Implementation Details: To conduct w2w inversion, we train on a single image following the objective from eq. 3. We qualitatively find that optimizing 10,000 principal component coefficients balances identity preservation with editability. This is discussed in Appendix F. We optimize for 400 epochs, using Adam [30] with learning rate 0.1, β1 = 0.9, β2 = 0.999 and with weight decay factor 1e-10. For conducting Dreambooth fine-tuning, we follow the implementation from Hugging Face 3 using LoRA with rank 1. To create a dataset of multiple images for an identity, we follow the procedure from Sec. 4.4.

w2w inversion is more efficient than previous methods. Inversion into w2w space results in a significant speedup in optimization as seen in Tab. 4, where we measure the training time on a single NVIDIA A100 GPU. Standard Dreambooth fine-tuning operates on the full weight space and incorporates an additional prior preservation loss which typically requires hundreds of prior images. In contrast, we only optimize a standard denoising objective on a single image within a low-dimensional weight subspace. Despite operating with lower dimensionality, w2w inversion performs closely to standard Dreambooth fine-tuning on multiple images with LoRA.

###### Table 4: Inversion into w2w space balances identity preservation and efficiency.

Single Image w2w Input DB-LoRA

Single Image DB-LoRA

w2w Inversion

Input DB-LoRA DB-LoRA Inversion

Method Single-Image # Param Opt. Time (s) Identity Fidelity ↑

[Figure 257]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

DB-LoRA × 99,648 220 0.69 ± 0.01 DB-LoRA ✓ 99,648 200 0.43 ± 0.03 w2w Inversion ✓ 10,000 55 0.64 ± 0.01

Qualitative Inversion Comparison. In Figs. 19 and 20, we present qualitative comparisons of w2w inversion against Dreambooth trained with multiple images and a single image. Although mult-image Dreambooth slightly outperforms w2w inversion in identity preservations, its samples tend to lack realism compared to w2w inversion. We hypothesize that this may be due to using generated images for prior preservation and training on synthetic identity images. Dreambooth trained on a single image either generates an artifacted version of the original image or random identities. Notice how inversion into w2w space is even able to capture key characteristics of the child although babies are nearly to completely absent in the identites based on CelebA used to fine-tune our dataset of models.

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

Single Image DB-LoRA

Single Image DB-LoRA

w2w Inversion

w2w Inversion Input DB-LoRA

Input DB-LoRA

[Figure 277]

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

Figure 19: Inversion into w2w space preserves identity and realism. We compare against Dreambooth fine-tuning with LoRA on multiple images and a single image.

Single Image w2w Input DB-LoRA

Single Image DB-LoRA

w2w Inversion

3https://github.com/huggingface/peft

Input DB-LoRA DB-LoRA Inversion

[Figure 297]

[Figure 298]

[Figure 300]

[Figure 303]

[Figure 304]

21

[Figure 305]

[Figure 309]

[Figure 310]

Single Image DB-LoRA

Single Image DB-LoRA

w2w Inversion

w2w Inversion Input DB-LoRA

Single Image w2w Input DB-LoRA

Single Image DB-LoRA

w2w Inversion

Single Image w2w Input DB-LoRA

Single Image DB-LoRA

w2w Inversion

Input DB-LoRA

Input DB-LoRA DB-LoRA Inversion

Input DB-LoRA DB-LoRA Inversion

Single Image DB-LoRA

Single Image DB-LoRA

w2w Inversion

w2w Inversion Input DB-LoRA

Single Image DB-LoRA

Single Image DB-LoRA

w2w Inversion

w2w Inversion Input DB-LoRA

Input DB-LoRA

Input DB-LoRA

Single Image w2w Input DB-LoRA

Single Image DB-LoRA

w2w Inversion

Input DB-LoRA DB-LoRA Inversion

Single Image DB-LoRA

Single Image DB-LoRA

w2w Inversion

w2w Inversion Input DB-LoRA

Input DB-LoRA

[Figure 437]

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

Single Image w2w Input DB-LoRA

Single Image DB-LoRA

w2w Inversion

Input DB-LoRA DB-LoRA Inversion

[Figure 457]

[Figure 458]

[Figure 460]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 469]

[Figure 470]

[Figure 474]

[Figure 475]

[Figure 476]

Single Image DB-LoRA

Single Image DB-LoRA

w2w Inversion

w2w Inversion Input DB-LoRA

Single Image w2w Input DB-LoRA

Single Image DB-LoRA

w2w Inversion

Single Image w2w Input DB-LoRA

Single Image DB-LoRA

w2w Inversion

Input DB-LoRA

Input DB-LoRA DB-LoRA Inversion

Input DB-LoRA DB-LoRA Inversion

[Figure 477]

[Figure 478]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 485]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 496]

[Figure 499]

[Figure 500]

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

[Figure 517]

[Figure 518]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 534]

[Figure 535]

[Figure 536]

Single Image DB-LoRA

Single Image DB-LoRA

w2w Inversion

w2w Inversion Input DB-LoRA

Single Image DB-LoRA

Single Image DB-LoRA

w2w Inversion

w2w Inversion Input DB-LoRA

Input DB-LoRA

Input DB-LoRA

[Figure 537]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 574]

[Figure 575]

[Figure 576]

###### Figure 20: Inversion into w2w space preserves identity and realism (cont.).

Comparison against Single Image Personalization Methods. We compare w2w inversion to single shot personalization methods Celeb-Basis [73] and IP-Adapter FaceID4 [72], following the same evaluation protocol from Sec. 4.4. These quantitative results are presented in Tab. 5. While Celeb-Basis optimizes in the input text embedding space for a given face image, IP-Adapter trains light-weight adapters [39, 75] to condition generation on any input face image.

We further conducted a two-alternative forced choice (2AFC) user study on the perceptual quality of w2w identity preservation. Twenty users were given ten sets of images. Each set contained a randomly sampled original image of the identity, and then three random images generated using IP-Adapter FaceID, Celeb-Basis, and w2w with the same random prompt. Users were then asked to choose between alternate pairs based on three criteria: identity preservation, prompt alignment, and diversity of generated images. Our results in Tab. 6 show that users found generations from w2w models capture identity better while also generating more diverse images that better align with the prompts.

Across both these metrics, w2w performs stronger than Celeb-Basis and IP-Adapter FaceID. Our results indicate that operating in our weight subspace is highly expressive and flexible as it is able to faithfully capture nuanced identity without overfitting to the input image. For instance, in Fig. 21, w2w inversion enables diverse generation with various poses, facial expressions, and clothing while maintaining identity.

Table 5: Single-shot personalization comparison.

Method ID Score ↑

Celeb-Basis 0.60 ± 0.02 IP-Adapter FaceID 0.62 ± 0.02

w2w 0.64 ± 0.01

Table 6: User study on identity inversion.

Method Win Rate (%) ↑ Celeb-Basis 13.4

w2w 86.6 IP-Adapter FaceID 22.8

###### w2w 77.2

## Original “A photo of V1 person”

IP-Adapterw2wCeleb-Basis

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

###### Figure 21: Qualitative comparison of single-shot personalization methods.

4https://huggingface.co/h94/IP-Adapter-FaceID

Below in Figs. 22 and 23, we provide the instructions provided to users for the identity inversion user study in addition to an example question.

[Figure 589]

Figure 22: Instructions provided to users for the identity inversion user study.

[Figure 590]

Figure 23: Example question from the identity inversion user study.

#### D Out of Distribution Projection

Additional examples of out-of-distribution projections are displayed in Fig. 24. A diverse array of styles and subjects (e.g. paintings, sketches, non-humans) can be distilled into a model in w2w space. After embedding an identity into this space, the model still retains the compositionality and rich priors of a standard personalized model. For instance, we can generate images using prompts such

- as “[v] person writing at a desk” (top example), “[v] person with a dog” (middle example), or “a painting of [v] person painting on a canvas” (bottom example).

Input Projection

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

Input Projection

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

Input Projection

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

- Figure 24: Projection into w2w space generalizes to a variety of inputs. A range of styles and entities can be inverted into a realistic identity in this space. Once a model is obtained, it can be prompted to generate the identity in a variety of contexts.

#### E Identity Datasets

In Fig. 25, we present examples of synthetic identity datasets used to conduct our Dreambooth fine-tuning as discussed in Sec 4. Each dataset is a set of ten images generated with [67] conditioned on single CelebA [36] images associated with binary attribute labels. Note that we only display a subset of images per identity in the figure. The same identity can occur multiple times in different images in CelebA but have different appearances. So, creating these synthetic datasets reduces intra-dataset diversity and creates a more consistent appearance for each subject. For instance, the first two rows in the figure are the same identity, but look drastically different. So we instead treat them as different identities associated with a different set of images.

For evaluating identity edits from Sec. 4.3, we hold out 100 identities, which results in leaving out ∼1000 models since multiple models may encode different instances of the same identity. For instance, if we left out the model encoding the identity in the first row of Fig. 25 for evaluation, the model encoding the second row identity would also be left out since it encodes the same identity but a different instance.

|CelebA Prior|Generated Dataset Examples|
|---|---|
|[Figure 606]|[Figure 607]<br><br>[Figure 608]<br><br>[Figure 609]<br><br>[Figure 610]|
|[Figure 611]|[Figure 612]<br><br>[Figure 613]<br><br>[Figure 614]<br><br>[Figure 615]|
|[Figure 616]|[Figure 617]<br><br>[Figure 618]<br><br>[Figure 619]<br><br>[Figure 620]|
|[Figure 621]|[Figure 622]<br><br>[Figure 623]<br><br>[Figure 624]<br><br>[Figure 625]|
|[Figure 626]|[Figure 627]<br><br>[Figure 628]<br><br>[Figure 629]<br><br>[Figure 630]|
|[Figure 631]|[Figure 632]<br><br>[Figure 633]<br><br>[Figure 634]<br><br>[Figure 635]|
|[Figure 636]|[Figure 637]<br><br>[Figure 638]<br><br>[Figure 639]<br><br>[Figure 640]|
|[Figure 641]|[Figure 642]<br><br>[Figure 643]<br><br>[Figure 644]<br><br>[Figure 645]<br><br>[Figure 646]|

- Figure 25: Fine-tuning on synthetic examples allows Dreambooth fine-tuning to distill a consistent identity. The left column shows a CelebA image used to condition generation of a set of identity-consistent images in the right column associated with that identity using [67]. The consistent appearance of the identity enables a more consistent identity encoding.

#### F Principal Component Basis

In this section, we analyze various properties of the Principal Component (PC) basis used to define w2w Space. We investigate the distribution of PC coefficients and the effect of the number of PCs on identity editing and inversion.

Distribution of PC Coefficients. We plot the histogram of the coefficient values for the first three Principal Components in Fig. 26. They appear roughly Gaussian. Next, we rescale the coefficients for these three components to unit variance for visualization purposes. We then plot the pairwise joint distributions for them in Fig. 27. The circular shapes indicates roughly diagonal covariances. Although the joint over other combinations of Principal Components may exhibit different properties, these results motivate us to model the PCs as independent Gaussians, leading to the w2w sampling strategy from Sec. 3.2.

Number of Principal Components for Identity Editing We empirically observe that training classifiers based on the 1000 dimensional PC representations (first 1000 PCs) of the model weights results in the most semantically aligned and disentangled edits directions. We visualize a comparison for the “goatee" direction in Fig. 28. After finding the direction, we calculate the maximum projection component onto the edit direction among the training set of model weights. This determines the edit strength. As seen in the figure, restricting to the first 100 Principal Components may be too coarse to achieve the fine-grained edit, instead relying on spurious cues such as skin color. Training with the first 10,000 Principal Components suffers from the curse of dimensionality and the discovered direction may edit other concepts such as eye color or clothes. Finding the direction using the first 1000 Principal Components achieves the desired edit with minimal entanglement with other concepts.

Number of Principal Components for Identity Inversion We qualitatively observe that inversion using the first 10,000 Principal Components balances identity preservation while not overfitting to the source image. We visualize a comparison in Fig. 29, where each column has a fixed seed and prompt. Optimizing with the first 1000 PCs underfits the identity and does not generate a consistent identity. Inversion with the first 20,000 Principal Components overfits to the source image of a face shot, which results in artifacted face images despite different generation seeds and prompts. Optimizing with the first 10,000 Principal Components enjoys the benefits of a lower dimensional representation than the original LoRA parameter space (∼100,000 trainable parameters), while still preserving identity and compositionality. This is supported quantitatively by Fig. 30, which shows the average ID score for 100 inverted FFHQ identities optimized over a varying number of principal components.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

- Figure 26: Histogram of principal component coefficients. The first three principal component coefficients appear approximately Gaussian.

- Figure 27: Pairwise joint histogram of principal component coefficients. We rescale the first three principal component coefficients and plot the pairwise joint distributions for visualization purposes. Given that the marginals are roughly Gaussian, the circular appearance of the joint suggests pairwise independence for the first three components.

+Chubby+Goatee Original 100 PCs 1000 PCs 10000 PCs

Original T=1000 T=800 T =600

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

- Figure 28: Edit results with varying number of Principal Components. Training classifiers to find semantic weight space directions with the first 1000 Principal Components achieves the most semantically aligned and disentangled results.

[Figure 659]

[Figure 660]

[Figure 661]

Inversion with 20000 PCs

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

Inversion with 1000 PCs

Inversion with 10000 PCs

Input

- Figure 29: Identity inversion results with varying number of principal components. We optimize the coefficients for the first 1000, 10,000, and 20,000 Principal Component. Each column indicates a fixed generation seed and prompt. Inversion with the first 10,000 components balances parameter efficiency, realism, and identity preservation without overfitting to the single image.

102 103 104

# of Principal Components

0.0

0.1

0.2

0.3

0.4

0.5

0.6

IDScore

- Figure 30: Identity preservation vs. number of principal components used for w2w inversion. We optimize the coefficients for the first N principal components up to 20,000 and measure the average ID score for 100 inverted FFHQ identities.

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

Visualizing Principal Components. We provide a visualization of traversals along a set of principal components in Fig. 31. The principal components change attributes of the identity, although various semantic attributes are entangled. For instance, the first PC appears to change age, hair color, and hair style. The second PC appears to change gender and skin complexion. The third PC seems to change age, skin complexion, and facial hair. This motivates our use of linear classifiers to find separating hyperplanes in weight space and disentangle these attributes.

[Figure 673]

- Figure 31: Traversal along the first three principal components in w2w space. The directions encode various entangled identity attributes such as age, gender, and facial hair.

#### G weights2weights for Other Visual Concepts

We find that similar subspaces can be created for other visual concepts beyond human identites. For instance, we apply the weights2weights framework to create two subspaces for models encoding different dog breeds and models encoding car types. We present examples of editing these models in Fig. 32. This suggest the generality of weights2weights and linear subspaces within diffusion model weights.

[Figure 674]

[Figure 675]

[Figure 676]

- Figure 32: Applying weights2weights edits on dog-encoding models and car-encoding models. We create two datasets of model weights and apply PCA to define two separate weight subspaces. We then train linear classifiers to find semantic edit directions.

#### H Multi-Concept Merging

Multiple models living in w2w space cannot be merged since they live in the same weight subspace. So, merging will lead to interpolation of the identities. However, w2w models can be merged with models lying approximately orthogonal to the subspace. This merging can be done adding the weights. We present an example of merging a model living in w2w space with a model fine-tuned to encode “Pixar" style in Fig. 33.

[Figure 677]

- Figure 33: Merging w2w models with non-identity models. Here, another model is fine-tuned to map V2 to “Pixar” style. The two models are merged with simple addition.

I Timestep Analysis

Edits in w2w space correspond to identity edits with minimal interference with other visual concepts. Although not a focus, image editing is achieved as a byproduct. For further context preservation, edits in w2w Space can be integrated with delayed injection [7, 17, 35, 71], where after T timesteps, the edited weights are used instead of the original ones. We visualize this in Fig. 34. Larger T in the range [700,1000] are helpful for more global attribute changes, while smaller [400,700] can be used for more fine-grained edits. However, by decreasing the timestep T, the strength of the edit is lost in favor of better context preservation. For instance, the dog’s face is better preserved in the second row at T = 600, although the man is not as chubby compared to other T.

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

[Figure 689]

Original T=1000 T=800 T =600

+Chubby

- Figure 34: Injecting edited weights at varying timesteps. Using the edited weights at a smaller timestep T better preserves context at the expense of edit strength and fidelity.

