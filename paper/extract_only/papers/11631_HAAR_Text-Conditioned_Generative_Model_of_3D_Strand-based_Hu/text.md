### Text-Conditioned Generative Model of 3D Strand-based Human Hairstyles

# arXiv:2312.11666v1[cs.CV]18Dec2023

Vanessa Sklyarova1,2 Egor Zakharov2 Otmar Hilliges2 Michael J. Black1 Justus Thies1,3

1Max Planck Institute for Intelligent Systems 2ETH Z¨urich 3Technical University of Darmstadt

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

...afro hairstyle... ...voluminous straight hair... ...man haircut... ...wavy short hairstyle...

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

...bob haircut... ...long wavy hairstyle... ...long straight hair... ...short curly hairstyle...

Figure 1. Given a text description, our method produces realistic human hairstyles. The usage of a 3D strand-based geometry representation allows it to be easily incorporated into existing computer graphics pipelines for simulation and rendering [3, 7, 9].

###### Abstract

We present HAAR, a new strand-based generative model for 3D human hairstyles. Specifically, based on textual inputs, HAAR produces 3D hairstyles that could be used as production-level assets in modern computer graphics engines. Current AI-based generative models take advantage of powerful 2D priors to reconstruct 3D content in the form of point clouds, meshes, or volumetric functions. However, by using the 2D priors, they are intrinsically limited to only recovering the visual parts. Highly occluded hair structures can not be reconstructed with those methods, and they only model the “outer shell”, which is not ready to be used in physics-based rendering or simulation pipelines. In contrast, we propose a first text-guided generative method that uses 3D hair strands as an underlying representation. Leveraging 2D visual question-answering (VQA) systems, we automatically annotate synthetic hair models that are generated from a small set of artist-created hairstyles. This allows us to train a latent diffusion model that operates in a common hairstyle UV space. In qualitative and quantita-

tive studies, we demonstrate the capabilities of the proposed model and compare it to existing hairstyle generation approaches. For results, please refer to our project page†.

###### 1. Introduction

There has been rapid progress in creating realistic, animatable 3D face and head avatars from images, video, and text. What is still missing is hair. Existing methods typically represent hair with a coarse mesh geometry, implicit surfaces, or neural radiance fields. None of these representations are compatible with the strand-based models used by existing rendering systems and do not enable animation of the resulting avatars with natural hair dynamics. Modeling and generating realistic 3D hair remains a key bottleneck to creating realistic, personalized avatars. We address this problem with HAAR (Hair: Automatic Animatable Reconstruction), which enables the generation of realistic and diverse hairstyles based solely on text descriptions. HAAR is

† https://haar.is.tue.mpg.de/

the first text-driven generative model that produces a classical strand-based hair representation that can be immediately imported into rendering systems and animated realistically. This approach replaces the complex and time-consuming process of manually creating 3D hairstyles with a chat-like text interface that can be used by a novice to create highquality 3D hair assets.

Previous work exploits generative models as learned priors to create 3D strand-based hair from images, videos, or random noise. In particular, Neural Haircut [47] reconstructs high-fidelity hairstyles from smartphone video captures without any specialized equipment by leveraging a pre-trained generative diffusion model. However, their strand-based generative model does not provide control over the geometry of the resulting hairstyles, substantially limiting the range of applications. Recently, GroomGen [57] introduced an unconditional generative model of hair. In contrast, we propose the first text-conditioned generative model for strand-based hairstyles that can be used for automated and fast hair asset generation.

Text-conditioned generative models like Stable Diffusion [42] are widely used for image and video generation and can be used to generate 3D shape from text [5, 6, 14, 25, 28, 29, 32, 38, 39, 48, 49, 54] by exploiting Score Distillation Sampling (SDS) [38]. These methods convert textual descriptions into 3D assets that, when rendered into multiple views, align with generated 2D images via differentiable rendering. These methods represent 3D shapes either as meshes [5, 25, 39], point clouds [6, 48] or volumes [28, 29, 32, 38, 49]. In particular, TECA [54] demonstrates how hair can be generated from text using a neural radiance field [34], combined with a traditional mesh-based head model [23]. However, the inherent problem with these SDS-based solutions is that they only capture the outer visible surface of the 3D shape. Even volumetric representations do not have a meaningful internal hair structure [54]. Thus, they can not be used for downstream applications like animation in graphics engines [3, 7].

Instead, what we seek is a solution with the following properties: (1) the hair is represented using classical 3D strands so that the hairstyle is compatible with existing rendering tools, (2) hair is generated from easy-to-use text prompts, (3) the generated hair covers a wide range of diverse and realistic hairstyles, (4) the results are more realistic than current generative models based SDS. To this end, we develop a text-guided generation method that produces strand-based hairstyles via a latent diffusion model. Specifically, we devise a latent diffusion model following the unconditional model used in Neural Haircut [47]. A hairstyle is represented on the scalp of a 3D head model as a texture map where the values of the texture map correspond to the latent representation of 3D hair strands. The individual strands are defined in a latent space of a VAE that captures

the geometric variation in the hair strand shape. To generate novel hair texture maps, we infer a diffusion network that takes a noise input and text conditioning. From the generated hair texture map, we can sample individual latent strands and reconstruct the corresponding 3D hair strands.

There are three remaining, interrelated, problems to address: (1) We need a dataset of 3D hairstyles to train the VAE and diffusion model. (2) We need training data of hairstyles with text descriptions to relate hairstyles to our representation. (3) We need a method to condition generated hair on text. We address each of these problems. First, we combine three different 3D hair datasets and augment the data to construct a training set of about 10K 3D hairstyles. Second, one of our key novelties is in how we obtain hairstyle descriptions. Here, we leverage a large vision-language model (VLM) [27] to generate hairstyle descriptions from images rendered from the 3D dataset. Unfortunately, existing visual question-answering (VQA) systems [22, 26, 27] are inaccurate and do not produce coherent hairstyle descriptions. To address these problems, we design a custom data-annotation pipeline that uses a pregenerated set of prompts that we feed into a VQA system [26] and produce final annotations by combining their responses in a single textual description. Finally, we train a diffusion model to produce the hair texture encoding conditioned on the encoding of textual hairstyle descriptions.

As Figure 1 illustrates, our strand-based representation can be used in classical computer graphics pipelines to realistically densify and render the hair [3, 7, 9]. We also show how the latent representation of hair can be leveraged to perform various semantic manipulations, such as up-sampling the number of strands in the generated hairstyle (resulting in better quality than the classical graphics methods) or editing hairstyles with text prompts. We perform quantitative comparisons with Neural Haircut as well as an ablation study to understand which design choices are critical. In contrast to SDS-based methods like TECA, HAAR is significantly more efficient, requiring seconds instead of hours to generate the hairstyle.

Our contributions can be summarized as follows:

- • We propose a first text-conditioned diffusion model for realistic 3D strand-based hairstyle generation,
- • We showcase how the learned latent hairstyle representations can be used for semantic editing,
- • We developed a method for accurate and automated annotation of synthetic hairstyle assets using off-the-shelf VQA systems.

The model will be available for research purposes.

###### 2. Related work

Recently, multiple text-to-3D approaches [5, 6, 14, 25, 28, 29, 32, 38, 39, 48, 49, 54] have emerged that were in-

spired by the success of text-guided image generation [40– 42, 44]. A body of work of particular interest to us is the one that uses image-space guidance to generate 3D shapes in a learning-by-synthesis paradigm. Initially, these methods used CLIP [41] embeddings shared between images and text to ensure that the results generated by the model adhere to the textual description [2, 12, 33]. However, the Score Distillation Sampling procedure (SDS) [38] has recently gained more popularity since it could leverage textto-image generative diffusion models, such as Stable Diffusion [42], to guide the creation of 3D assets from text, achieving higher quality. Multiple concurrent methods employ this SDS approach to map textual description into a human avatar [4, 14, 19, 24, 54]. In particular, the TECA [54] system focuses on generating volumetric hairstyles in the form of neural radiance fields (NeRFs) [34]. However, these approaches can only generate the outer visible surface of the hair without internal structure, which prevents it from being used out-of-the-box in downstream applications, such as simulation and physics-based rendering. Moreover, the SDS procedure used to produce the reconstructions is notoriously slow and may require hours of optimization to achieve convergence for a given textual prompt. Our approach is significantly more efficient, and is capable of generating and realistically rendering the hairstyles given textual prompts in less than a minute.

In contrast to the methods mentioned above, we also generate the hairstyles in the form of strands. Strand-accurate hair modeling has manifold applications in computer vision and graphics as it allows subsequent physics-based rendering and simulation using off-the-shelf tools [3, 7, 9]. One of the primary use cases for the strand-based generative modeling has historically been the 3D hair reconstruction systems [13, 20, 35, 43, 45–47, 52, 53, 55, 56]. Among the settings where it is most often used is the so-called one-shot case, where a hairstyle must be predicted using only a single image [13, 52, 55]. Approaches that tackle it leverage synthetic datasets of strand-based assets to train the models and then employ detailed cues extracted from the images, such as orientation maps [37], to guide the generation process. However, these systems are unsuitable for semanticsbased or even unconditional generation of hairstyles, as they rely heavily on these cues for guidance. A group of methods that is more closely related to ours is Neural Haircut [47] and GroomGen [57], in which a synthetic dataset of hairstyle assets is leveraged to train an unconditional generative model [16, 18, 42]. While useful for regularizing multi-view hair reconstruction [47], the degree of control over the synthesized output in such methods is missing. Our work addresses the issue of controllability in generative models for hair and is the first one to provide strand-based hairstyle generation capabilities given textual descriptions.

###### 3. Method

Given a textual description that contains information about hair curliness, length, and style, our method generates realistic strand-based hair assets. The resulting hairstyles can be immediately used in computer graphics tools that can render and animate the hair in a physically plausible fashion. Our pipeline is depicted in Figure 2. At its core is a latent diffusion model, which is conditioned on a hairstyle text embedding. It operates on a latent space that is constructed via a Variational Autoencoder (VAE) [18]. Following [43], this VAE is trained to embed the geometry of individual strands into a lower-dimensional latent space. During inference, the diffusion model generates this representation from Gaussian noise and the input text prompt, which is then upsampled to increase the number of strands and decoded using a VAE decoder to retrieve the 3D hair strands.

###### 3.1. Hairstyle parametrization.

We represent a 3D hairstyle as a set of 3D hair strands that are uniformly distributed over the scalp. Specifically, we define a hair map H with resolution 256 × 256 that corresponds to a scalp region of the 3D head model. Within this map, each pixel stores a single hair strand S as a polyline. As mentioned previously, our diffusion model is not directly operating on these 3D polylines, but on their compressed latent embeddings z. To produce z that encodes the strand S, we first convert the latter into the local basis defined by the Frenet frame of the face where the strand root is located. On this normalized data, we train a variational auto-encoder, which gives us access to an encoder E(S) and a decoder G(z).Using the encoder E(S), we encode the individual hair strands in the hair map H, resulting in a latent map Z that has the same spatial resolution. The decoded strand-based hair map is then denoted as Hˆ. In summary, with a slight abuse of notation, the maps are related to each other as follows: Z = E(H), and Hˆ = G(Z).

###### 3.2. Conditional Hair Diffusion Model

We use a pre-trained text encoder τ [22], that encodes the hairstyle description P into the embedding τ(P). This embedding is used as conditioning to the denoising network via a cross-attention mechanism:

Attention(Q,K,V ) = softmax

QKT √

d · V, (1)

where Q = WQ(i) · ϕi(Zt), K = WK(i) · τ(P), V = WV(i) · τ(P) with learnable projection matrices WQ(i),WK(i),WV(i). The denoising network is a 2D U-Net [15], where ϕi(Zt) denotes i-th intermediate representations of the U-Net produced for the latent hair map Zt at the denoising step t. For our training, we employ the EDM [16] formulation, following [47]. We denote the latent hair map with noise as

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

##### Inference

##### Training

[Figure 13]

[Figure 14]

Generated hair map

Latent hair map

Diffusion model

D

Decoder

Encoder Subsample & add noise

G

E

Upsample

Sec. 3.3 Sec. 3.1

Sec. 3.1

Zˆ

###### Z

τθ

τθ

H

Generated hairstyle

Hairstyle

Cross-attn.

Hˆ

... woman with afro hairstyle …

Sec. 3.2

P Text embedding

Caption

Sec. 3.4

- Figure 2. Overview. We present our new method for text-guided and strand-based hair generation. For each hairstyle H in the training set, we produce latent hair maps Z and annotate them with textual captions P using off-the-shelf VQA systems [26] and our custom annotation pipeline. Then, we train a conditional diffusion model D [16] to generate the guiding strands in this latent space and use a latent upsampling procedure to reconstruct dense hairstyles that contain up to a hundred thousand strands given textual descriptions. The generated hairstyles are then rendered using off-the-shelf computer graphics techniques [9].

In this way, we aim to preserve the local structure of strands near a partition and apply smoothing in regions with similar strand directions. To calculate the blending weights, we first compute the cosine similarity between neighboring 3D hair strands on the mesh grid and apply the non-linear function f(·) to control the influence of the particular interpolation type, which we empirically derived to be as follows:

Zt = Z + ϵ · σt, where ϵ ∼ N(0,I), and σt is the noise strength. We then use a denoiser D to predict the output:

Dθ(Zt,σt,P) = cst · Zt + cot · Fθ cit · Zt,cnt ,τ(P) , (2)

where the cst, cot, cit and cnt are the preconditioning factors for the noise level σt that follow [16], and Fθ denotes a U-Net network. The optimization problem is defined as:

1 − 1.63 · x5 where x ≤ 0.9 0.4 − 0.4 · x x > 0.9,

t,ϵ,Z,P λt · ∥Dθ(Zt,σt,P) − Z∥22 , (3) where λt denotes a weighting factor for a given noise level.

Eσ

min

(4)

f(x) =

θ

where x is the cosine similarity. Our final interpolation for each point on the mesh grid is defined as a blending between the nearest neighbor and bilinear interpolations with the weight f(x) and (1 − f(x)) correspondingly. The defined upsampling method ensures that in the vicinity of a partition, the weight of the nearest neighbor decreases linearly, and then diminishes at a polynomial rate. As a result of this scheme, we obtain realistic geometry in the regions with low similarity among strands. On top of that, we add Gaussian noise to the interpolated latents to increase the hair strands diversity, resulting in a more natural look.

###### 3.3. Upsampling

Due to the limited amount of available 3D hairstyles, the diffusion model is trained on a downsampled latent hair map Z′ with resolution 32 × 32 and, thus, only generates socalled ’guiding hair strands’. To increase the number of strands in the generated results, we upsample the latent hair map to the resolution of 512 × 512. A common way of upsampling a strand-based hairstyle to increase the number of strands is via interpolation between individual polylines.

In modern computer graphics engines [3, 7] multiple approaches, such as Nearest Neighbour (NN) and bilinear interpolation are used. Applying these interpolation schemes leads to over-smoothing or clumping results. In some more advanced pipelines, these schemes are combined with distance measures based on the proximity of strand origins or the similarity of the curves. Additionally, Blender and Maya [3, 7] introduce an option of adding noise into the interpolation results to further prevent clumping of the hair strands and increase realism. However, the described interpolation procedure requires a lot of manual effort and needs to be done for each hairstyle separately to obtain optimal parameters and resolve undesired penetrations.

###### 3.4. Data generation

3D hairstyle data. For training and evaluating the diffusion model, we use a small artist-created hairstyle dataset, that consists of 40 high-quality hairstyles with around 100,000 strands. To increase the diversity, we combine it with two publicly available datasets: CT2Hair [46] and USCHairSalon [13] that consist of 10 and 343 hairstyles, respectively. We align the three datasets to the same parametric head model and additionally augment each hairstyle using realistic squeezing, stretching, cutting, and curliness augmentations. In total, we train the model on 9825 hairstyles.

In this work, we propose an automatic approach with interpolation of the hairstyle in latent space by blending between nearest neighbor and bilinear interpolation schemes.

Hairstyle description. As these 3D hairstyles do not come with textual annotations, we use the VQA model

proves the results. Finally, we calculate the embeddings of the resulting hairstyle descriptions P using a BLIP encoder τ for both frontal and back views and average them to produce the conditioning used during training.

Q: What is the type of current hairstyle?

|[Figure 15]<br><br>[Figure 16]|
|---|

What is the relative length of the hairstyle? What is the texture of current hairstyle?

Describe actors with similar hairstyle type.

###### 3.5. Training details

Visual Question Answering (VQA)

To train the diffusion model, we sample a batch of hairstyles at each iteration, align them on a mesh grid of 256 × 256 resolution, and, then, subsample it into a size of 32×32. By training the diffusion model on these subsampled hairstyles we improve convergence and avoid overfitting. To accelerate the training, we use the soft Min-SNR [10] weighting strategy. It tackles the conflicting directions in optimization by using an adaptive loss weighting strategy. For more details, please refer to the original Min-SNR paper [10]. To evaluate the performance, we utilize an Exponential Moving Average (EMA) model and Euler Ancestral Sampling with 50 steps. The whole method is trained for about 5 days on a single NVIDIA A100, which corresponds to 160,000 iterations. Additional details are in the suppl. material.

∑

|Text Encoding|
|---|

A: The type of the hairstyle is bob …

N N

| |
|---|

… the length is short ... … the hairstyle is straight …

| |
|---|

| |
|---|

Hairstyle Embedding

| |
|---|

… This hairstyle is very popular ...

- Figure 3. Dataset collection. Rendered from frontal and back view hairstyles along with a predefined set of questions Q are sent through VQA [26, 27] to obtain hairstyle description, which is further encoded using frozen text encoder network [22].

LLaVA [26, 27] to automatically produce hairstyle descriptions from a set of predefined questions (see Figure 3). To do that, we first render all collected hairstyles using Blender [7] from frontal and back camera views. We use the standard head model and neutral shading for hairstyles to prevent bias to any particular type of hairstyle because of color or gender information. With the help of ChatGPT [36], we design a set of prompts, that include specific questions about length, texture, hairstyle type, bang, etc., as well as a set of general questions about historical meaning, professional look, occasions for such hairstyle, celebrities with similar type to increase generalization and variability of our conditioning model. We then use a random subset of these prompts for each hairstyle in the dataset to increase the diversity of annotations. For a full list of prompts that were used, please refer to the suppl. material.

###### 4. Experiments 4.1. Evaluation

We compare our method against competing approaches for generative hair modeling: TECA [54] and Neural Haircut [47]. TECA creates a compositional avatar that includes separate geometries for hair, body, and cloth using only a text description. This method represents hair using neural radiance fields (NeRF) [34] and focuses on the visual quality of generated avatars, not geometry reconstruction. Moreover, it takes multiple hours to generate a single sample using TECA because they rely on Score Distillation Sampling [38]. In our case, we concentrate on physically plausible geometry for the hair and require around 4.3 seconds to generate a hairstyle. Neural Haircut focuses on the reconstruction of realistic 3D hairstyles with a strand-based representation using monocular video or multi-view images captured under unconstrained lighting conditions. In this work, authors exploit a diffusion model to obtain some prior knowledge for better reconstruction quality. In contrast to our approach, the quality of the diffusion model is limited by the amount of data, the size of the model architecture, and the chosen training strategy. This model is unconditional, and thus cannot control the generated hairstyles.

The quality of visual systems is highly restricted by the diversity of data used during training. We have observed in our experiments that the accuracy of the produced hair captions is relatively low, or they contain very broad descriptions. In particular, we have noticed that the existing VQA systems have problems accurately reasoning about the hair length or the side of the parting. To improve the quality of VQA answers, similarly to [58], we add an additional system prompt “If you are not sure say it honestly. Do not imagine any contents that are not in the image”, which decreases the likelihood of the model hallucinating its responses. Further, we have observed that the VQA system works better when it does not use information from the previous answers. That allows us to not accumulate erroneous descriptions during the annotation session. We have also observed that the LLAVA model is biased toward confirming the provided descriptions instead of reasoning, so introducing a set of choices to the prompts substantially im-

Quality of unconditional diffusion. To compare the quality of the unconditional diffusion model, we re-train Neural Haircut [47] on the same training data and with the same scalp parametrization as our method. We evaluate the distance of the generated hairstyles to the training distribution using Minimum Matching Distance (MMD) [1] as well as coverage (Cov) [1] metrics. We use the 1-Nearest Neighbor accuracy (1-NNA) [30] metric, which is a leave-one-out ac-

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

[Figure 32]

Prompt: “A woman with afro hairstyle”

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

Prompt: “A woman with bob hairstyle”

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

Prompt: “A woman with long wavy hair”

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

[Figure 78]

[Figure 79]

[Figure 80]

Prompt: “A woman with straight long hair”

###### TECA Ours

- Figure 4. Comparison. Qualitative comparison of conditional generative models. We show several generations of TECA [54] and our model. For our results, we visualize the geometry obtained before (shown in pseudocolor) and after upsampling. Our model generates more diverse samples with higher-quality hairstyles. It is also worth noting that TECA, in some cases, does not follow the input descriptions well, producing short hair instead of long hair (bottom row). Digital zoom-in is recommended.

|Method|MMD↓ COV↑ 1-NNA → 0.5<br><br>|
|---|---|
|Neural Haircut [47] Our<br><br>|31507.7 0.18 0.34 21104.9 0.2 0.55|

- Table 1. Comparison of unconditional diffusion models. Our method generates samples with better quality and diversity.

curacy of the 1-NN classifier that assesses if two provided distributions are identical. The best quality is achieved for values closer to 0.5. Suppose, we have two datasets of generated and reference hairstyles denoted as Sg and Sr, where |Sg| = |Sr|. Then, the described metrics are defined as:

1 |Sr| y∈S

D(x, y) (5)

MMD(Sg, Sr) =

min

x∈Sg

r

1 |Sr|

COV(Sg, Sr) =

D(x, y)|x ∈ Sg}| (6)

|{arg min

y∈Sr

I[Nx ∈ Sg] + y∈Sr I[Ny ∈ Sr] | Sg | + | Sr |

1−NNA(Sg, Sr) = x∈Sg

, (7)

where I(·) is an indicator function, NF is the nearest neighbor in set Sr∪Sg\F and D is the squared distance between distributions, computed in the latent space of the VAE. In Table 1, we show the comparison based on these metrics.

Our method generates samples closer to the ground-truth distribution with higher diversity.

Finally, we conducted a user study. Participants were presented 40 randomly sampled hairstyle pairs obtained using Neural Haircut [47] and our method. We collected more than 1,200 responses on the question “Which hairstyle from the presented pair is better?”, and ours was preferred in 87.5 % of cases.

Quality of conditional diffusion. We compare the quality of our conditional generation with TECA [54]. We launch both of the methods for various prompts with several random seeds to obtain the hair volume that follows the desired text input. The qualitative comparison can be seen in Figure 4. While TECA produces great conditioning results most of the time, some severe artifacts are noticeable in the hair region. Furthermore, the diversity of generations is limited, and we see some failure cases even for simple prompts like “A woman with straight long hair”. With our method HAAR, we provide a way to obtain detailed physically plausible geometry with large variations.

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

[Figure 92]

###### Blender (nearest) Blender (bilinear) Ours (nearest) Ours (bilinear) Ours Ours w/ noise

- Figure 5. Upsampling. Comparison of different upsampling schemes used to interpolate between guiding strands (shown in dark color). For visualization purposes here we show around 15,000 strands. Blender interpolation is obtained in 3D space, while Ours is computed in latent space. Using the Nearest Neighbour in both variants produces better accuracy according to the guiding strand geometry (shown in dark color), but it results in an unrealistic global appearance. The bilinear schemes lead to the penetration of averaged hair strands and the loss of structure of the original guiding strands. Blending both these methods resolves proposed issues and results in realistic renders. Adding additional noise in latent space further increases realism and helps to get rid of the grid structure.

Text encoder CLIP BLIP Transf. Reference CSIM 0.174 0.189 0.172 0.206

- Table 2. Conditioning. Ablation on different conditioning schemes. With BLIP text encoder, we obtain better conditioning compared to CLIP and trainable Transformer network.

###### 4.2. Ablation study

Conditioning. The quality of the conditional diffusion model for hairstyle generation is highly dependent on the quality of the text encoder network τ(·). We ablate the performance of the conditional generation using pre-trained and frozen encoders, such as CLIP [41], BLIP [22] as well as a trained transformer network [50] implemented on top of a pre-trained BertTokenizer [8]. For more details on the architecture, please refer to the supplemental material. The intuition behind training additional networks for text encoding is that the quality of pre-trained encoders may be limited for a particular task (for example some specific hairstyle types), which results in wrong correlations between words and deteriorates the quality of the diffusion model.

We evaluate the performance using semantic matching between text and generated 3D hairstyles. Specifically, we use CLIP [41] and compute the cosine distance between images and their respective text prompts. To do that, we generate 100 hairstyles for 10 different prompts and then render from a frontal view using Blender [7]. Table 2 shows that the BLIP text encoder is providing the most effective conditioning. To show the upper-bound quality of this metric (’reference’), we calculate the CSIM on our ground-truth dataset with prompts obtained via VQA.

Upsampling scheme. We ablate the performance of different upsampling schemes needed to obtain a full hairstyle from a set of guiding strands, which can be seen in Figure 5. There is no one-to-one correspondence and during interpolation, a lot of artifacts can occur. The most common artifact is a visible grid structure which appears when using a Nearest Neighbour (NN) strategy. Bilinear interpolation leads to scalp penetrations due to averaging the nearest strands on top of the head, and it deteriorates the local shape of curls. The computer graphics engines, such as Blender [7] and Maya [3], either do not provide enough control or require a lot of manual effort in setting up the optimal parameters for each hairstyle separately. We find that the combination of NN and Bilinear using our proposed scheme leads to the best-looking results of renders. Furthermore, adding noise in the latent space results in more realistic hairstyles. Note, for visualization we show an example with a reduced density of around 15,000 strands; increasing it leads to less bald regions, especially, in the region of a partition.

###### 4.3. Hairstyle editing

Similar to Imagic [17], we do text-based hairstyle editing, see Figure 6. Given an input hairstyle and a target text that corresponds to the desired prompt, we edit the hairstyle in a way that it corresponds to the prompt while preserving the details of the input hairstyle. To do that we first do textual inversion of the input hairstyle. We obtain etgt that corresponds to the target prompt P. After optimizing it with a fixed diffusion model Dθ using a reconstruction loss, we acquire eopt. Conditioning on the obtained text embedding eopt does not lead to the same target hairstyle. So, to pro-

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

###### Input Image eopt 0.25 0.5 0.75 “Short straight”

- Figure 6. Hairstyle editing. Similar to Imagic [17], we edit the input image using a text prompt. We provide editing results without additionally tuning the diffusion model (first two rows) and with it (second two rows). Finetuning the diffusion model results in smoother editing and better preservation of input hairstyle.

[Figure 117]

[Figure 118]

- Figure 7. Limitations. Our failure cases include penetration into the scalp region (left), which in principle can be resolved in a postprocessing step. Additionally, for the afro hairstyles (right), the degree of strands’ curliness needs to be increased.

tations. Especially, when used in a physics simulation, the interpenetrations can be resolved in a postprocessing step. Another limitation of our method is that we only consider geometry, we do not generate the hair color and texture which would be an interesting direction for future work.

###### 5. Conclusion

We have presented HAAR, the first method that is able to conditionally generate realistic strand-based hairstyles using textual hairstyle descriptions as input. Not only can such a system accelerate hairstyle creation in computer graphics engines, but it also bridges the gap between computer graphics and computer vision. For computer graphics, generated hairstyles could be easily incorporated into tools like Blender for hair editing and physics-based animation. For computer vision, our system can be used as a strong prior for the generation of avatars or to create synthetic training data of realistic hairstyles. While being limited by data, we think that this method is a first step in the direction of controllable and automatic hairstyle generation.

vide a smooth transition, we freeze eopt and fine-tune Dθ. Finally, we linearly interpolate between etgt and eopt. For more information, please refer to the supplemental material.

###### 4.4. Limitations

The quality of generated hairstyles is limited by the variety and quality of our dataset, in terms of both the diversity of geometry assets and the accuracy of textual annotations. The main failure cases include the generation of hairstyles with scalp interpenetrations and lack of curliness for some extreme hairstyles, see Figure 7. In theory, these limitations can be addressed with a dataset that contains more diverse samples of curly hairstyles, as well as human-made anno-

###### Acknowledgements

Vanessa Sklyarova was supported by the Max Planck ETH Center for Learning Systems. Egor Zakharov’s work was funded by the “AI-PERCEIVE” ERC Consolidator Grant, 2021. We sincerely thank Giorgio Becherini for rendering

hairstyles and Joachim Tesch for realistic hair simulations. Also, we thank Yao Feng and Balamurugan Thambiraja for their help during the project and Hao Zhang for aiding us with the TECA comparison.

Disclosure. MJB has received research gift funds from Adobe, Intel, Nvidia, Meta/Facebook, and Amazon. MJB has financial interests in Amazon, Datagen Technologies, and Meshcapade GmbH. While MJB is a consultant for Meshcapade, his research in this project was performed solely at, and funded solely by, the Max Planck Society.

###### References

- [1] Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. Learning representations and generative models for 3D point clouds. In Proceedings of the 35th International Conference on Machine Learning, pages 40–49. PMLR, 2018. 5
- [2] Shivangi Aneja, Justus Thies, Angela Dai, and Matthias Nießner. ClipFace: Text-guided Editing of Textured 3D Morphable Models. In ArXiv preprint arXiv:2212.01406,

2022. 3

- [3] Autodesk, INC. Maya. 1, 2, 3, 4, 7
- [4] Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, and KwanYee K Wong. Dreamavatar: Text-and-shape guided 3d human avatar generation via diffusion models. arXiv preprint arXiv:2304.00916, 2023. 3
- [5] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2
- [6] Zilong Chen, Feng Wang, and Huaping Liu. Text-to-3d using gaussian splatting, 2023. 2
- [7] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2023. 1, 2, 3, 4, 5, 7, 12
- [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In North American Chapter of the Association for Computational Linguistics,

2019. 7, 11

- [9] Epic Games. Unreal engine. 1, 2, 3, 4, 13
- [10] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7441–7451, 2023. 5
- [11] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 11
- [12] Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. Avatarclip: Zero-shot textdriven generation and animation of 3d avatars. ACM Transactions on Graphics (TOG), 41(4):1–19, 2022. 3

- [13] Liwen Hu, Chongyang Ma, Linjie Luo, and Hao Li. Singleview hair modeling using a hairstyle database. ACM Transactions on Graphics (TOG), 34:1 – 9, 2015. 3, 4
- [14] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, and Justus Thies. TeCH: Text-guided Reconstruction of Lifelike Clothed Humans. In International Conference on 3D Vision (3DV), 2024. 2, 3
- [15] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. CVPR, 2017. 3
- [16] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 3, 4, 11
- [17] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Conference on Computer Vision and Pattern Recognition 2023,

2023. 7, 8

- [18] Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. 3
- [19] Nikos Kolotouros, Thiemo Alldieck, Andrei Zanfir, Eduard Gabriel Bazavan, Mihai Fieraru, and Cristian Sminchisescu. Dreamhuman: Animatable 3d avatars from text. 2023. 3
- [20] Zhiyi Kuang, Yiyang Chen, Hongbo Fu, Kun Zhou, and Youyi Zheng. Deepmvshair: Deep hair modeling from sparse views. SIGGRAPH Asia 2022 Conference Papers,

2022. 3

- [21] Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio Savarese, and Steven C.H. Hoi. LAVIS: A one-stop library for language-vision intelligence. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 31–41, Toronto, Canada, 2023. Association for Computational Linguistics. 11
- [22] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,

2022. 2, 3, 5, 7, 11

- [23] Tianye Li, Timo Bolkart, Michael. J. Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4D scans. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 36(6):194:1–194:17, 2017. 2
- [24] Tingting Liao, Hongwei Yi, Yuliang Xiu, Jiaxiang Tang, Yangyi Huang, Justus Thies, and Michael J Black. Tada! text to animatable digital avatars. ArXiv, 2023. 3
- [25] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [26] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 2, 4, 5, 11
- [27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2, 5, 11

- [28] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion, 2023. 2
- [29] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object, 2023. 2
- [30] David Lopez-Paz and Maxime Oquab. Revisiting classifier two-sample tests. In International Conference on Learning Representations, 2017. 5
- [31] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2017. 11, 12
- [32] Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, and Andrea Vedaldi. Realfusion: 360 reconstruction of any object from a single image. In CVPR, 2023. 2
- [33] Oscar Michel, Roi Bar-On, Richard Liu, Sagie Benaim, and Rana Hanocka. Text2mesh: Text-driven neural stylization for meshes. arXiv preprint arXiv:2112.03221, 2021. 3
- [34] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2, 3, 5
- [35] Giljoo Nam, Chenglei Wu, Min H. Kim, and Yaser Sheikh. Strand-accurate multi-view hair capture. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 155–164, 2019. 3
- [36] ChatGPT. OpenAI, 2023. 5, 13, 16
- [37] Sylvain Paris, H´ector M. Brice˜no, and Fran¸cois X. Sillion. Capture of hair geometry from multiple images. ACM SIGGRAPH 2004 Papers, 2004. 3
- [38] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv,

2022. 2, 3, 5

- [39] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, and Bernard Ghanem. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023. 2
- [40] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 3
- [41] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. ArXiv, abs/2204.06125, 2022. 3, 7, 11
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 11
- [43] Radu Alexandru Rosu, Shunsuke Saito, Ziyan Wang, Chenglei Wu, Sven Behnke, and Giljoo Nam. Neural strands:

- Learning hair geometry and appearance from multi-view images. European Conference on Computer Vision (ECCV), 2022. 3
- [44] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. 2022. 3
- [45] Shunsuke Saito, Liwen Hu, Chongyang Ma, Hikaru Ibayashi, Linjie Luo, and Hao Li. 3d hair synthesis using volumetric variational autoencoders. ACM Transactions on Graphics (TOG), 37:1 – 12, 2018. 3
- [46] Yuefan Shen, Shunsuke Saito, Ziyan Wang, Olivier Maury, Chenglei Wu, Jessica Hodgins, Youyi Zheng, and Giljoo Nam. Ct2hair: High-fidelity 3d hair modeling using computed tomography. ACM Transactions on Graphics, 42(4): 1–13, 2023. 4
- [47] Vanessa Sklyarova, Jenya Chelishev, Andreea Dogaru, Igor Medvedev, Victor Lempitsky, and Egor Zakharov. Neural haircut: Prior-guided strand-based hair reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 19762–19773, 2023. 2, 3, 5, 6
- [48] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653,

2023. 2

- [49] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184, 2023. 2
- [50] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems. Curran Associates, Inc.,

2017. 7, 11

- [51] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, 2020. Association for Computational Linguistics. 11
- [52] Keyu Wu, Yifan Ye, Lingchen Yang, Hongbo Fu, Kun Zhou, and Youyi Zheng. Neuralhdhair: Automatic high-fidelity hair modeling from a single image using implicit neural representations. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1516–1525,

2022. 3

- [53] Lingchen Yang, Zefeng Shi, Youyi Zheng, and Kun Zhou. Dynamic hair modeling from monocular videos using deep neural networks. ACM Transactions on Graphics (TOG), 38: 1 – 12, 2019. 3

- [54] H. Zhang, Y. Feng, P. Kulits, Y. Wen, J. Thies, and M. J. Black. Teca: Text-guided generation and editing of compositional 3d avatars. arXiv, 2023. 2, 3, 5, 6, 11
- [55] Yujian Zheng, Zirong Jin, Moran Li, Haibin Huang, Chongyang Ma, Shuguang Cui, and Xiaoguang Han. Hairstep: Transfer synthetic to real using strand and depth maps for single-view 3d hair modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12726–12735, 2023. 3
- [56] Yi Zhou, Liwen Hu, Jun Xing, Weikai Chen, Han-Wei Kung, Xin Tong, and Hao Li. Hairnet: Single-view hair reconstruction using convolutional neural networks. In European Conference on Computer Vision, 2018. 3
- [57] Yuxiao Zhou, Menglei Chai, Alessandro Pepe, Markus Gross, and Thabo Beeler. GroomGen: A high-quality generative hair model using hierarchical latent representations.

2023. 2, 3

- [58] Deyao Zhu, Jun Chen, Kilichbek Haydarov, Xiaoqian Shen, Wenxuan Zhang, and Mohamed Elhoseiny. Chatgpt asks, blip-2 answers: Automatic questioning towards enriched visual descriptions. ArXiv, abs/2303.06594, 2023. 5

## Supplemental Material

###### A. Implementation and training details

Hairstyle diffusion model. For conditional diffusion model, we use the U-Net architecture from [42] with the following parameters: image size = 32 × 32, input channels = 64, num res blocks = 2, num heads = 8, attention resolutions = (4,2,1), channel mult = (1,2,4,4), model channels = 320, use spatial transformer = True, context dim = 768, legacy = False.

Our training pipeline uses the EDM [16] library and we optimize the loss function using AdamW [31] with learning rate = 10−4, β = [0.95,0.999], ϵ = 10−6, batch size = 8, and weight decay = 10−3.

List of prompts. Below we include the list of prompts used during data annotation using a VQA model. After each of the prompts, we add ‘If you are not sure say it honestly. Do not imagine any contents that are not in the image. After the answer please clear your history.’ to the input.

- • ‘Describe in detail the bang/fringe of depicted hairstyle including its directionality, texture, and coverage of face?’
- • ‘What is the overall hairstyle depicted in the image?’
- • ‘Does the depicted hairstyle longer than the shoulders or shorter than the shoulder?’
- • ‘Does the depicted hairstyle have a short bang or long bang or no bang from frontal view?’
- • ‘Does the hairstyle have a straight bang or Baby Bangs or Arched Bangs or Asymmetrical Bangs or Pin-Up Bangs or Choppy Bangs or curtain bang or side swept bang or no bang?’
- • ‘Are there any afro features in the hairstyle or no afro features?’

- • ‘Is the length of the hairstyle shorter than the middle of the neck or longer than the middle of the neck?’
- • ‘What are the main geometry features of the depicted hairstyle?’
- • ‘What is the overall shape of the depicted hairstyle?’
- • ‘Is the hair short, medium, or long in terms of length?’
- • ‘What is the type of depicted hairstyle?’
- • ‘What is the length of hairstyle relative to the human body?’
- • ‘Describe the texture and pattern of hair in the image.’
- • ‘What is the texture of depicted hairstyle?’
- • ‘Does the depicted hairstyle is straight or wavy or curly or kinky?’
- • ‘Can you describe the overall flow and directionality of strands?’
- • ‘Could you describe the bang of depicted hairstyle including its directionality and texture?’
- • ‘Describe the main geometric features of the hairstyle depicted in the image.’
- • ‘Is the length of a hairstyle buzz cut, pixie, ear length, chin length, neck length, shoulder length, armpit length or mid-back length?’
- • ‘Describe actors with similar hairstyle type.’
- • ‘Does the hairstyle cover any parts of the face? Write which exact parts.’
- • ‘In what ways is this hairstyle a blend or combination of other popular hairstyles?’
- • ‘Could you provide the closest types of hairstyles from which this one could be blended?’
- • ‘How adaptable is this hairstyle for various occasions (casual, formal, athletic)?’
- • ‘How is this hairstyle perceived in different social or professional settings?’
- • ‘Are there historical figures who were iconic for wearing this hairstyle?’
- • ‘Could you describe the partition of this hairstyle if it is visible?’

Text-based models. For the VQA, we found that “LLaVAv1.5” model [26, 27] produces the best text descriptions with a relatively low hallucination rate. As shown in Table 2, we experimented with different text encoder models. We used “ViT-L/14” configuration for CLIP [41] and “blip feature extractor” from [21] library for BLIP [22]. In the ablation experiment, we compare its result with an optimizable Transformer [50] build on top of the pre-trained BERTTokenizer [8] from the transformers [51] library with configuration “bert-base-uncased”. For the Transformer network, we use BERTEmbedder from [42] with n layer = 6, max seq len = 256, n embed = 640.

###### B. Additional Ablations and Results

Qualitative comparison. We show an extended comparison with TECA [54] with more complex prompts that show the compositional abilities of the models (see Figure 8).

Importance of classifier-free-guidance. To improve the sample quality of the conditional model, we use classifierfree-guidance [11]. During training, we optimize conditional and unconditional models at the same time, by using text embedding with zeros in 10% of cases. During inference, we fix the random seed and show changes in sample quality, sweeping over the guidance strength w. As we can see in Figure 9, higher weights improve the strength of conditional prompts, but increasing it too much leads to out-ofdistribution samples with a high degree of inter-head penetrations. In our experiments, we fix the guidance weight to w = 1.5.

[Figure 119]

[Figure 120]

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

Prompt: “bob hairstyle with long bang”

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

[Figure 149]

[Figure 150]

Prompt: “A woman with curly short hairstyle”

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

[Figure 164]

[Figure 165]

[Figure 166]

Prompt: “A woman with shoulder-length wavy hair”

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

Prompt: “Bob hairstyle with afro features”

###### TECA Ours

- Figure 8. Comparison. Extended comparison with TECA. Our method produces higher quality samples with greater diversity than ones generated in TECA, and our representation allows the animation of the hair in a physics simulator.

Hairstyle interpolation. We linearly interpolate between two text prompts P1 and P2 by conditioning the diffusion model Dθ on a linear combination of text embeddings (1 − α)τ(P1) + ατ(P2), where α ∈ [0,1], and τ is the text encoder. For interpolation results obtained for different prompt pairs that differ in length and texture please see Figure 10. One can notice that the interpolation between two types of textures, e.g. “wavy“ and “straight“ usually starts appearing for α close to 0.5, while length reduction takes many fewer interpolation steps.

Hairstyle editing. For optimization eopt, we do 1500 steps with the optimizer Adam with a learning rate of 10−3. For diffusion fine-tuning, we do 600 steps with optimizer AdamW [31] with a learning rate of 10−4, β = [0.95,0.999], ϵ = 10−6, and weight decay 10−3. Both stages are optimized using the same reconstruction loss used during the training of the main model. The entire editing pipeline takes around six minutes on a single NVIDIA A100. See Figure 11 for more editing results with and without fine-tuning.

Upsampling scheme. We provide more results on the different upsampling schemes for “long straight“ and “long wavy“ hairstyles (see Figure 12). While Blender [7] interpolation in 3D space produces either results with a high

level of penetration (bilinear upsampling) or very structured (Nearest Neighbour) hairstyles, we are able to easily blend between two types in latent space, combining the best from the two schemes. Adding noise helps eliminate the grid structure inherited from the nearest neighbor sampling and, thus, improves realism. For noising the latent space, we calculate a standard deviation Zσ ∈ R1×1×M of latent map after interpolation Z ∈ RN×N×M, where N is a grid resolution and M = 64 is the dimension of latent vector that encodes the entire hair strand. The final noised latent map is Z = Z + Zσ ⊙ X ⊙ Y , where X ∈ RN×N×1 with elements xijk ∼ N(0.15,0.05), Y ∈ RN×N×1 with elements yijk = 2qijk −1, where qijk ∼ Bernoulli(0.5). In such a way, we independently add some small random noise to each latent vector on the mesh grid.

Generalization capabilities. Our conditional diffusion model can distinguish between different texture types, lengths, bangs, and some popular hairstyles, like the bob, and afro. It models the correlation between gender and hairstyle length, but at the same time, the capacity of the model is limited by the accuracy of the VQA and text encoder system. Asking more general questions improves the generalization quality, but the answers may be less accurate and lead to additional noise during training. To test the generalization capabilities of our model, we evaluate it on out-

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

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

[Figure 203]

[Figure 204]

[Figure 205]

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

#### w = 0 0.5 0.8 1. 1.2 1.5 2. w = 2.5

- Figure 9. Classifier-free guidance. Quality of samples during changing the guidance weight w from 0 to 2.5. Weight w = 0 corresponds to unconditional generation, while w = 1 - to conditional. For w > 1 we obtain over-conditioned results. In our experiments, we fix w = 1.5, as higher weights lead to more penetrations and reduced realism. The first four rows correspond to generation samples for the prompt “voluminous straight hair“ with two different random seeds, while the last four - for “wavy long hair“.

of-distribution prompts and attempt to generate hairstyles of particular celebrities. We use ChatGPT [36] to describe the hairstyle type of a particular celebrity and use the resulting prompt for conditioning. To our surprise, we find that even given the limited diversity of the hairstyles seen during training, our model can reproduce the general shape of the hairstyle. We show results illustrating the generalization capabilities of our model by reconstructing celebrity hairstyles for “Cameron Diaz“ and “Tom Cruise“ (see Figure 13). Between different random seeds hairstyles preserve the main features, like waviness and length, but could change the bang style.

Finally, we show the results of our conditional model on different hairstyle types, by conditioning the model on hairstyle descriptions from [36] (see Figure 14).

Simulations. The hairstyles generated by our diffusion

model are further interpolated to resolution 512 × 512 and then imported into the Unreal Engine [9] as a hair card. We tested simulations in two scenarios: integration into a realistic game environment with manual character control as well as simple rotation movements for different types of hairstyles. The realism of simulations highly depends on the physical characteristics of hair, e.g. friction, stiffness, damping, mass, elasticity, resistance, and collision detection inside the computer graphics engine. An interesting research direction for future work may include the prediction of individual physical properties for each hairstyle that could further simplify the artists’ work. For simulation results, please refer to the supplemental video.

###### Input Interpolated hairstyles Input

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

[Figure 262]

“short straight hair“ “short curly hair“

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

“long wavy hair“ “short straight hair“

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

“long wavy“ “long straight“

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

“wavy bob “ “man haircut“

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

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

“casual woman“ “short haircut“

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

“bob hairstyle“ “buzz cut“

Figure 10. Hairstyle interpolation. Linear interpolation between two given textual prompts.

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

Input Image eopt 0.125 0.25 0.375 0.5 0.625 0.75 0.875 “Short hair“

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

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

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

Input Image eopt 0.125 0.25 0.375 0.5 0.625 0.75 0.875 “Straight long“

- Figure 11. Hairstyle editing. Extended editing results of our model. In each section of four images, we provide editing results without additionally tuning the diffusion model (first two rows) and with it (second two rows). Finetuning the diffusion model results in smoother editing and better preservation of input hairstyle.

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

###### Blender (nearest) Blender (bilinear) Ours (nearest) Ours (bilinear) Ours Ours w/ noise

- Figure 12. Upsampling. Extended results on hairstyle interpolation between guiding strands obtained using different schemes. For better visual comparison, we interpolate hairstyles to around 15,000 strands and additionally visualize guiding strands (shown in dark color) for Ours methods with interpolation in latent space. Our final method with additional noise improves the realism of hairstyles by removing the grid-like artifacts.

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

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

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

- Figure 13. Generalization. Hairstyles generated for celebrities “Cameron Diaz“ (first two rows) and “Tom Cruise“ (last two rows) using descriptions from [36]. Several variations of hairstyles with corresponding guiding strands are generated for each celebrity.

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

“Long Layered Waves“ “Long Shag“ “Hawk Fade“ “Layered Hair“ “Long Layers“

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

“Cropped Curls“ “Voluminous Curls“ “Blowout“ “Pixie Cut“ “Layered Bob“

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

“Pinup Style“ “Shaggy Bob“ “Straight Lob“ “Afro“ “Side Part“

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

“French Crop“ “Curtain Bangs“ “Bowl Cut“ “Spiky Hair“ “Wavy Hair“

Figure 14. Conditional generation. Random samples generated for input prompts with classifier-guidance weight w = 1.5.

