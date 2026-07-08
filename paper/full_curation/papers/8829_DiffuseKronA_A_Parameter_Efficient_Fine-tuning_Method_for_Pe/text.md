## DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Models

#### Shyam Marjit1* Harshit Singh1* Nityanand Mathur1* Sayak Paul2 Chia-Mu Yu3 Pin-Yu Chen4

Project Page: https://diffusekrona.github.io/

Input Images Generated Images by DiﬀuseKronA

# arXiv:2402.17412v2[cs.CV]28Feb2024

|[Figure 1]|
|---|

|[Figure 2]|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

|[Figure 9]|
|---|

|[Figure 10]|
|---|

“A [V] Dog” in theofVersaillesMirror Hall in the Versaillesgardens of in Mount Fuji wearingand a blacka black bowtop-tiehat in a police outﬁt with abackgroundblue house in the

|[Figure 11]|[Figure 12]|
|---|---|
|[Figure 13]|[Figure 14]|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

wearing Victorian-era clothing, reading a book in a classic British library

wearing a Dhoti-Kurta Ensemble

watercolour painting, mountains in background

“A [V] person” blossomsstanding underof a cherrythe pinktree

selﬁe in front of Eiﬀel Tower

taking a shot in basketball

|[Figure 21]|[Figure 22]|
|---|---|
|[Figure 23]|[Figure 24]|

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

upset and sitting under an

is sharing a romantic moment

as an artist, sitting in front of

“A [V] anime” in witha serenea focusedgarden,expression,meditating around ﬂowers and butterﬂies

on a sunny day, wearing a cute

is blushing, holding a

umbrella on a rainy day,

on the dance ﬂoor with a

an easel, painting a beautiful and expressing her art

summer dress, laughing and

notebook and a pen, trying to initiate a conversation

holding a love letter in her hand

partner in a weekend party

playing with friends at a park

Figure 1. DiffuseKronA achieves superior image quality and text alignment across diverse input images and prompts, all the while upholding exceptional parameter efficiency. Here, [V] denotes a unique token used for fine-tuning a specific subject in the text-to-image diffusion model. We showcase human face editing in Fig. 10, and car modifications in Fig. 11, allowing for a wider range of applications.

### Abstract

ter efficiency and the quality of T2I personalized image synthesis. Addressing these constraints, we introduce DiffuseKronA, a novel Kronecker product-based adaptation module that not only significantly reduces the parameter count by 35% and 99.947% compared to LoRA-DreamBooth and the original DreamBooth, respectively, but also enhances the quality of image synthesis. Crucially, DiffuseKronA mitigates the issue of hyperparameter sensitivity, delivering consistent highquality generations across a wide range of hyperparameters, thereby diminishing the necessity for extensive fine-tuning. Furthermore, a more controllable decomposition makes DiffuseKronA more interpretable and even can achieve up to a 50% reduction with results comparable to LoRADreambooth. Evaluated against diverse and complex input images and text prompts, DiffuseKronA

In the realm of subject-driven text-to-image (T2I) generative models, recent developments like DreamBooth and BLIP-Diffusion have led to impressive results yet encounter limitations due to their intensive fine-tuning demands and substantial parameter requirements. While the low-rank adaptation (LoRA) module within DreamBooth offers a reduction in trainable parameters, it introduces a pronounced sensitivity to hyperparameters, leading to a compromise between parame-

*Equal contribution. 1Indian Institute of Information Technology Guwahati, India 2Hugging Face 3National Yang Ming Chiao Tung University, Hsinchu, Taiwan 4IBM Research, New York, USA. Correspondence to: Shyam Marjit <shyam.marjit@iiitg.ac.in>, Pin-Yu Chen <pinyu.chen@ibm.com>.

Preprint Under Review. Copyright 2024 by the author(s).

consistently outperforms existing models, producing diverse images of higher quality with improved fidelity and a more accurate color distribution of objects, all the while upholding exceptional parameter efficiency, thus presenting a substantial advancement in the field of T2I generative modeling.

- 1. Introduction In recent years, text-to-image (T2I) generation models (Gu

- et al., 2022; Chang et al., 2023; Rombach et al., 2022; Podell
- et al., 2023; Yu et al., 2022) have rapidly evolved, generating intricate and highly detailed images that often defy discernment from real-world photographs. The current stateof-the-art has marked significant progress and demonstrated substantial improvement, which hints at a future where the boundary between human imagination and computational representation becomes increasingly blurred. In this context, subject-driven T2I generative models (Ruiz et al., 2023a; Li et al., 2023a) unlock creative potential such as image editing, subject-specific property modifications, art renditions, etc. Works like DreamBooth (Ruiz et al., 2023a), BLIP-Diffusion (Li et al., 2023a) seamlessly introduce new subjects into the pre-trained models while preserving the priors learned by the original model without impacting its generation capabilities. These approaches excel at retaining the essence and subject-specific details across various styles when fine-tuned with few-shot examples, leveraging foundational pre-trained latent diffusion models (LDMs) (Rombach et al., 2022). However, DreamBooth with Stable Diffusion (Rombach

- et al., 2022) suffers from some primary issues, such as incorrect prompt context synthesis, context appearance entanglement, and hyperparameter sensitivity. Additionally, DreamBooth finetunes all parameters of latent diffusion model’s (Rombach et al., 2022) UNet and text encoder (Radford et al., 2021), which significantly increases the trainable parameter count, making the finetuning process expensive. Here, the widely used low-rank adaptation module (Hu et al.,

- 2021) (LoRA) within DreamBooth attempts to significantly trim the parameter counts but it magnifies the aforementioned DreamBooth-reported issues, which makes a complete tradeoff between parameter efficiency and satisfactory subject-driven image synthesis. Moreover, it suffers from high sensitivity to hyperparameters, necessitating extensive fine-tuning to achieve desired outputs. This motivates us to design a more robust and effective parameter-efficient finetuning (PEFT) method for adapting T2I generative models to subject-driven personalized generation.

In this paper, we introduce DiffuseKronA, a novel parameter-efficient module that leverages the Kronecker product-based adaptation module for fine-tuning T2I diffu-

Advantages of DiffuseKronA

LoRA

✓ Easy to plug-and-play: Preserves the original network topology & generation ability

✓Simple and small: ~3.8M parameters and ~14.95MB storage

+

✓Enhanced Flexibility and interpretability

✓ Generalizable

✓~35% smaller than LoRADreambooth

Pretrained Weights Updated Weights

Kronecker Adapter

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Constrained on

Figure 2. Schematic illustration: LoRA is limited to one controllable parameter, the rank r; while the Kronecker product showcases enhanced interpretability by introducing two controllable parameters a1 and a2 (or equivalently b1 and b2).

sion models, focusing on few-shot adaptations. LoRA adheres to a vanilla encoder-decoder type architecture, which learns similar representations within decomposed matrices due to constrained flexibility and similar-sized matrix decomposition (Tahaei et al., 2022b). In contrast, Kronecker’s decomposition exploits patch-specific redundancies, offering a much higher-rank approximation of the original weight matrix with less parameter count and greater flexibility in representation by allowing different-sized decomposed matrices. This fundamental difference is attributed to several improvements including parameter reduction, enhanced stability, and greater flexibility. Moreover, it effectively captures crucial subject-specific spatial features while producing images that closely adhere to the provided prompts. This results in higher quality, improved fidelity, and more accurate color distribution in objects during personalized image generation, achieving comparable results to state-of-the-art techniques.

Our key contributions are as follows:

❶ Parameter Efficiency: DiffuseKronA significantly reduces trainable parameters by 35% and 99.947% as compared to LoRA-DreamBooth and vanilla DreamBooth using SDXL (Podell et al., 2023) as detailed in Table 2. By changing Kronecker factors, we can even achieve up to a 50% reduction with results comparable to state-of-the-art as demonstrated in Figure 26 in the Appendix.

❷ Enhanced Stability: DiffuseKronA offers a much more stable image-generation process formed within a fixed spectrum of hyperparameters when fine-tuning, even when working with complicated input images and diverse prompts. In Figure 4, we demonstrate the trends associated with hyperparameter changes in both methods and highlight our superior stability over LoRA-DreamBooth.

❸ Text Alignment and Fidelity: On average, DiffusekronA captures better subject semantics and large contextual prompts. We refer the readers to Figure 7 and Figure 8 for qualitative and quantitative comparisons, respectively.

❹ Interpretablilty: Notably, we conduct extensive analysis to explore the advantages of the Kronecker product-based

adaptation module within personalized diffusion models. More controllable decomposition makes DiffusekronA more interpretable as demonstrated in Figure 2. Extensive experiments on 42 datasets under the few-shot setting demonstrate the aforementioned effectiveness of DiffuseKronA, achieving the best trade-off between parameter efficiency and satisfactory image synthesis.

### 2. Related Works

Text-to-Image Diffusion Models. Recent advancements in T2I diffusion models such as Stable Diffusion (SD) (Rombach et al., 2022; Podell et al., 2023), Imagen (Saharia et al.,

- 2022), DALL-E2 (Ramesh et al., 2022) & E3 (Betker et al.,
- 2023), PixArt-α (Chen et al., 2023), Kandinsky (Lui et al., 2019), and eDiff-I (Balaji et al., 2022) have showcased remarkable efficacy in modeling data distributions, yielding impressive results in image synthesis and opening the door for various creative applications across domains. Compared to the previous iterations of the SD model, Stable Diffusion XL (SDXL) (Podell et al., 2023) represents a significant advancement in T2I synthesis owing to a larger backbone and an improved training procedure. In this work, we mainly incorporate SDXL due to its impressive capability to generate high-resolution images, prompt adherence, as well as better composition and semantics.

Subject-driven T2I Personalization. Given only a few images (typically 3 to 5) of a specific subject, T2I personalization techniques aim to synthesize diverse contextual images of the subject based on textual input. In particular, Textual Inversion (Gal et al., 2022) and DreamBooth (Ruiz et al., 2023a) were the first lines of work. Textual Inversion finetunes text embedding, while DreamBooth fine-tunes the entire network using an additional preservation loss as regularization, resulting in visual quality improvements that show promising outcomes. More recently, BLIP-Diffusion (Li

- et al., 2023a) enables zero-shot subject-driven generation capabilities by performing a two-stage pre-training process leveraging the multimodal BLIP-2 (Li et al., 2023b) model. These studies focus on single-subject generation, with later works (Kumari et al., 2023; Han et al., 2023; Ma et al., 2023; Tewel et al., 2023) delving into multi-subject generation.

PEFT Methods within T2I Personalization. In contrast to foundational models (Ruiz et al., 2023a; Li et al., 2023a) that fine-tune large pre-trained models at full scale, several seminal works (Kumari et al., 2023; Han et al., 2023; Ruiz et al., 2023b; Ye et al., 2023) in parameter-efficient finetuning (PEFT) have emerged as a transformative approach. Within the realm of PEFT techniques, low-rank adaptation methods (Hu et al., 2021; von Platen et al., 2023) has become a de-facto way of reducing the parameter count by introducing learnable truncated Singular Value Decomposition (SVD) modules into the original model weights on essential layers. For instance, Custom Diffusion (Kumari

et al., 2023) focuses on fine-tuning the K and V matrices of the cross-attention, introducing multiple concept generation for the first time, and employing LoRA for efficient parameter compression. SVDiff (Han et al., 2023) achieves parameter efficiency by fine-tuning the singular values of the weight matrices with a Cut-Mix-Unmix data augmentation technique to enhance the quality of multi-subject image generation. Hyper-Dreambooth (Ruiz et al., 2023b) proposed a hypernetwork to make DreamBooth rapid and memoryefficient for personalized fidelity-controlled face generation. T2I-Adapters (Mou et al., 2023), a conceptually similar approach to ControlNets (Zhang et al., 2023), makes use of an auxiliary network to compute the representations of the additional inputs and mixes that with the activations of the UNet. Mix-of-Show (Gu et al., 2023), on the other hand, involves training distinct LoRA models for each subject and subsequently performing fusion.

In context, the LoRA-Dreambooth (Ryu, 2023) technique has encountered difficulties due to its poor representational capacity and low interpretability, and to address these constraints we introduce DiffuseKronA. Our method is inspired by the KronA technique initially proposed by (Edalati et al., 2022b). However, there are key distinctions: (1) The original paper was centered around language models, whereas our work extends this exploration to LDMs, particularly in the context of T2I generation. (2) Our focus lies on the efficient fine-tuning of various modules within LDMs. (3) More importantly, we investigate the impact of altering Kronecker factors on subject-specific generation, considering interpretability, parameter efficiency, and subject fidelity. It is also noteworthy to mention that LoKr (Yeh et al., 2023) is a concurrent work, and we discuss the key differences in Appendix F.

### 3. Methodology

Problem Formulation. Given a pre-trained T2I latent diffusion model Dϕ with size |Dϕ| and weights denoted by ϕ, we aim to develop a parameter-efficient adaptation technique with trainable parameters θ of size m such that m ≪ |Dϕ| holds (i.e. efficiency) while attaining satisfactory and comparable performance with a full fine-tuned model. At inference, newly trained parameters will be integrated with their corresponding original weight matrix, and diverse images can be synthesized from the new personalized model, Dϕ+θ. Method Overview. Figure 3 shows an overview of our proposed DiffuseKronA for PEFT of T2I diffusion models in subject-driven generation. DiffuseKronA only updates parameters in the attention layers of the UNet model while keeping text encoder weights frozen within the SDXL backbone. Here, we first outline a preliminary section in Section 3.1 followed by a detailed explanation of DiffuseKronA in Section 3.2. Particularly, in Section 3.2, we provide insights and mathematical explanations of “Why Dif-

Reconstruction Loss

Latent Embedding

Subject Images

|[Figure 31]|[Figure 32]|
|---|---|
|[Figure 33]|[Figure 34]|

|[Figure 35]|[Figure 36]| |
|---|---|---|
|[Figure 37]|[Figure 38]| |

Add & Norm

|+noise| |
|---|---|
| | |

|predict| |
|---|---|
| | |

+noise

predict

Transformer Encoder

Q K V

Q K V

Q K V

Q K V

Feed Forward

[Figure 39]

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Add & Norm

Add & Norm Multi-Head Attention

|Pretrained Weights<br><br>(Query/Key/Value/Out)<br><br>[Figure 40]|
|---|

Attention Self-

Feed Forward

Cross-

"A [V] Dog"

CLIP Text Encoder

- a) Fine-tuning

|predict<br><br>| |
|---|---|
| | |

predict

- b) Inference

| | | | |
|---|---|---|---|
| | | | |
| | | | |

Add & Norm Multi-Head Attention

Add & Norm Multi-Head Attention

Attention

Iterative Denoising Pipeline

Updated Weights,

Noise Map,

Output

|[Figure 41]|
|---|

|[Figure 42]| |
|---|---|
| | |

Positional Encoding Output

Q K V

Q K V

Q K V

Q K V

| | |
|---|---|
|Input Embedding| |

Embedding

| | |
|---|---|
|Latent Embedding| |

| | |
|---|---|
|Conditioning, e.g. text prompt| |

|"A [V] dog inside a doghouse with a matching home in the background."| |
|---|---|
| | |

Constrained on

CLIP Text Encoder

a.2) Kronecker Adapter

a.1) Transformer Layer

- Figure 3. Overview of DiffuseKronA: (a) Fine-tuning process involves optimizing the multi-head attention parameters (a.1) using Kronecker Adapter, elaborated in the subsequent block, a.2, (b) During inference, newly trained parameters, denoted as θ, are integrated with the original weights Dϕ and images are synthesized using the updated personalized model Dϕ+θ.

fuseKronA is a more parameter-efficient and interpretable way of fine-tuning Diffusion models compared to vanilla LoRA?”

#### 3.1. Preliminaries

T2I Diffusion Models. LDMs (Rombach et al., 2022), a prominent variant of probabilistic generative Diffusion models denoted as Dϕ, aim to produce an image xgen = Dϕ (ϵ,c) by incorporating a noise map ϵ ∼ N(0,I) and a conditioning embedding c = T (P) derived from a text prompt P using a text encoder, T . LDMs transform the input image x ∈ RH×W×3 into a latent representation z ∈ Rh×w×v through an encoder E, where z = E (x) and v is the latent feature dimension. In this context, the denoising diffusion process occurs in the latent space, Z, utilizing a conditional UNet (Ronneberger et al., 2015) denoiser Dϕ to predict noise ϵ at the current timestep t given the noisy latent zt and generation condition c. In brief, the denoising training objective of an LDM Dϕ can be simplified to:

EE(x),c,ϵ∼N(0,I),t∼U(0,1) wt ∥Dϕ (zt|c,t) − ϵ∥22 , (1)

where U denotes uniform distribution and wt is a timedependent weight on the loss.

Low Rank Adaptation (LoRA). Pre-trained large models exhibit a low “intrinsic dimension” for task adaptation (Hu et al., 2021; Han et al., 2023), implying efficient learning after subspace projection. Based on this, LoRA (Hu et al., 2021) hypothesizes that weight updates also possess low “intrinsic rank” during adaptation and inject trainable rank decomposition matrices into essential layers of the model for task adaptations, significantly reducing the number of trainable parameters.

In the context of a pre-trained weight matrix W0 ∈ Rd×k, the update of W0 is subject to constraints imposed through the representation of the matrix as a low-rank decomposition W0 + ∆W := W0 + AB, where A ∈ Rd×r, B ∈ Rr×h, and the rank r ≪ min(d,h). As a result, the sizes of

- A and B are significantly smaller than W0, reducing the number of trainable parameters. Throughout the training

process, W0 remains fixed, impervious to gradient updates, while the trainable parameters are contained within A and

- B. For h = W0x, the modified forward pass is formulated as follows:

##### f(x) = W0x + ∆Wx + b0 := WLoRA x + b0, (2)

where b0 is the bias term of the pre-trained model.

LoRA-DreamBooth. LoRA (Hu et al., 2021) is strategically employed to fine-tune DreamBooth with the primary purpose of reducing the number of trainable parameters. LoRA injects trainable modules with low-rank decomposed matrices in WQ, WK, WV , and WO weight matrices of attention modules within the UNet and text encoder. During training, the weights of the pre-trained UNet and text encoder are frozen and only LoRA modules are tuned. However, during inference, the weights of fine-tuned LoRA modules are annexed to the corresponding pre-trained weights. Moreover, this task does not increase the inference time.

#### 3.2. DiffuseKronA

LoRA demonstrates effectiveness in the realm of diffusion models but is hindered by its limited representation power. In contrast, the Kronecker product offers a more nuanced representation by explicitly capturing pairwise interactions between elements of two matrices. This ability to capture intricate relationships enables the model to learn and represent complex patterns in the data with greater detail.

Kronecker Product (⊗) is a matrix multiplication method that allows multiplication between matrices of different shapes. For two matrices A ∈ Ra

1×b2, each block of their Kronecker product A⊗B ∈ Ra

1×a2 and B ∈ Rb

1b1×a2b2 is defined by multiplying the entry Ai,j with B such that

 . (3)

 

a1,1B · · · a1,a2B

Factorization Constraint Kronecker down factor A ∈ Ra

Decomposed Matrix Factor Name

Module Parameters

Notation

... .

A ⊗ B =

.

1×a2

a1b1 = d Kronecker up factor B ∈ Rb

aa1,1B · · · aa1,a2B

a1a2 + b1b2

1×b2 a2b2 = h LoRA down projection A ∈ Rd×r

The Kronecker product can be used to create matrices that represent the relationships between different sets of model parameters. These matrices encode how changes in one set of parameters affect or interact with another set. In Figure 9, we showcase how Kronecker product works. Interestingly, it does not suffer from rank deficiency as low-rank downprojection does, as in the case of techniques such as LoRA and Adapter. The Kronecker product has several advantageous properties that make it a good option for handling complex data (Greenewald & Hero, 2015).

r(d + h) r ≪ min(d,h) LoRA up projection B ∈ Rr×h

Table 1. Comparing Kronecker factors and LoRA projections.

on different datasets, denoted as δl = ∥θl′ − θl∥/∥θl∥, where θl′ and θl represent the updated and pre-trained model parameters of layer l. Their findings indicated that the crossattention module exhibited a relatively higher δ, signifying its pivotal role in the fine-tuning process. In light of these studies, we conducted fine-tuning on the attention layers and observed their high effectiveness. Additional details on this topic are available in Appendix D.

Kronecker Adapter (KronA). Firstly introduced in studying PEFT of language models (Edalati et al., 2022b), The Kronecker product takes advantage of the structured relationships encoded in the matrices. Instead of explicitly performing all the multiplications required to compute the product A⊗B, the following equivalent matrix-vector multiplication can be applied, reducing the overall computational cost. This is particularly beneficial when working with large matrices or when computational resources are constrained:

A closer look at LoRA v.s. DiffuseKronA. Higher-rank matrices are decomposable to a higher number of singular vectors, capturing better expressibility and allowing for a richer capacity for PEFT. In LoRA, the rank of the resultant update matrix ∆Wlora is bounded by the minimum rank between matrices A and B, i.e. rank(∆Wlora) = min(rank(A),rank(B)). Conversely, in DiffuseKronA, the matrix rank ∆WKronA = A ⊗ B is the product of the ranks of matrices A and B, i.e. rank(∆WKronA) = rank(A) · rank(B), which can be properly configured to produce a higher-rank matrix than LoRA while maintaining lower-rank decomposed matrices than LoRA. Hence, for personalized T2I diffusion models, DiffuseKronA is expected to carry more subject-specific information in lesser parameters, as compared in Table 2 and Table 3. More details are provided in Appendix E.

2×a2(x)A⊤ (4) where A⊤ is transposed to A. The rationale is that a vector y ∈ Rm·n can be reshaped into a matrix Y of size m × n using the mathematical operation ηm×n(y). Similarly, Y ∈ Rm×n can also be transformed back into a vector by stacking its columns using the γ(Y ) operation. This approach achieves O(blog b) computational complexity and O(log b) space complexity for a b-dimensional vector, a drastic improvement over the standard unstructured Kronecker multiplication (Zhang et al., 2015).

(A ⊗ B)x = γ Bηb

### 4. Experiments

In this section, we assess the various components of personalization using DiffuseKronA through a comprehensive ablation study to confirm their effectiveness, using SDXL (von Platen et al., 2023) and SD (CompVis, 2021) models as backbones. Furthermore, we have conducted an insightful comparison between DiffuseKronA and LoRA-DreamBooth in six aspects in Section 4.3 and also compare DiffuseKronA with other related prior works in Section 4.4, highlighting our superiority.

Fine-Tuning Diffusion Models with KronA. In essence, KronA can be applied to any subset of weight matrices in a neural network for parameter-efficient adaptation as specified in the equation below, where U denotes different modules in diffusion models, including Key (K), Query (Q), Value (V ), and Linear (O) layers. During fine-tuning, KronA modules are applied in parallel to the pre-trained weight matrices. The Kronecker factors are multiplied, scaled, and merged into the original weight matrix after they have been adjusted. Hence, like LoRA, KronA maintains the same inference time.

#### 4.1. Datasets and Evaluation

Datasets. We have performed extensive experimentation on four types of subject-specific datasets: (i) 12 datasets (9 are from (Ruiz et al., 2023a) and 3 are from (Kumari et al., 2023)) of living subjects/pets such as stuffed animals, dogs, and cats; (ii) dataset of 21 unique objects including sunglasses, backpacks, etc.; (iii) our 5 collected datasets on cartoon characters including Super-Saiyan, Akimi, Kiriko, Shoko Komi, and Hatake Kakashi; (iv) our 4 collected datasets on facial images. More details are given in Ap-

∆WU = AU ⊗ BU,U ∈ {K,Q,V,O}; Wfine-tuned = Wpre-trained + ∆W.

(5)

Previous studies (Kumari et al., 2023; von Platen et al., 2023; Tewel et al., 2023) have conducted extensive experiments to identify the most influential modules in the fine-tuning process. In (Kumari et al., 2023; Li et al., 2020), authors explored the rate of changes in each module during fine-tuning

[Figure 43]

- Figure 4. Comparison between DiffuseKronA and LoRA-DreamBooth across varying learning rates on SDXL. In our approach, we set the value of a2 to 64. DiffuseKronA produces favorable results across a wider range of learning rates, specifically from 1 × 10−4 to 1 × 10−3. In contrast, no discernible patterns are observed in LoRA. The right part of the figure shows plots of Text & Image Alignment for LoRA-DreamBooth and DiffuseKronA, where points belonging to DiffuseKronA seem to be dense and those of LoRA-DreamBooth seems to be sparse, signifying that DiffuseKronA tends to be more stable than LoRA-DreamBooth while changing learning rates.

pendix B.

Implementation Details. We observe that ∼ 1000 iterations, employing a learning rate of 5×10−4, and utilizing an average of 3 training images prove sufficient for generating desirable results. The training process takes ∼ 5 minutes for SD (CompVis, 2021) and ∼ 40 minutes for SDXL (von Platen et al., 2023) on a 24GB NVIDIA RTX-3090 GPU.

Evaluation metrics. We evaluate DiffuseKronA on (1) Image-alignment: we compute the CLIP (Radford et al., 2021) visual similarity (CLIP-I) and DINO (Caron et al., 2021) similarity scores of generated images with the reference concept images, and (2) Text-alignment: we quantify the CLIP text-image similarity (CLIP-T) between the generated images and the provided textual prompts. A detailed mathematical explanations are available in Appendix C.

- 4.2. Unlocking the Optimal Configurations of DiffuseKronA

Throughout our experimentation, we observed the following trends and found the optimal configuration of hyperparameters for better image synthesis using DiffuseKronA. How to perform Kronecker decomposition? Unlike LoRA, DiffuseKronA features two controllable Kronecker factors, as illustrated in Table 1, providing greater flexibility in decomposition. Our findings reveal that the dimensions of the downward Kronecker matrix A must be smaller than those of the upward Kronecker matrix B. Specifically, we determined the optimal value of a2 to be precisely 64, while a1 falls within the set {2,4,8}. Remarkably, among all pairs of (a1,a2) values, (4,64) yields images with the highest

fidelity. Additionally, it has been observed that images exhibit minimal variation with learning rates when a2 = 64, as depicted in Figure 4 and Figure 15. Detailed ablation about Kronecker factors, their initializations, and their impact on fine-tuning is provided in Appendix D.2.

Effect of learning rate. DiffuseKronA produces consistent results across a wide range of learning rates. Here, we observed that the images generated for a learning rate closer to the optimal learning rate value 5 × 10−4 generate similar images. However, learning rates exceeding 1 × 10−3 contribute to model overfitting, resulting in high-fidelity images but with diminished emphasis on input text prompts. Conversely, learning rates below 1 × 10−4 lead to lower fidelity in generated images, prioritizing input text prompts to a greater extent. This pattern is evident in Figure 4, where our approach produces exceptional images that faithfully capture both the input image and the input text prompt. Additional results are provided in Appendix D.3 to justify the same.

Additionally, we conducted investigations into model ablations, examining (a) choice of modules to fine-tune the model in Appendix D.1 (b) effects of no training images in Appendix D.5 and steps in Appendix D.4, (c) one-shot model performance in Appendix D.5.1, and (d) effect of inference hyperparameters such as the number of inference steps and the guidance score in Appendix D.6.

4.3. Exploring Model Performance: LoRA-Dreambooth vs DiffuseKronA

We use SDXL and employ our DiffuseKronA to generate

DiffuseKronA LoRA-DreamBooth

Input images

[Figure 44]

|[Figure 45]|[Figure 46]|
|---|---|
| |[Figure 47]|

[Figure 48]

with Eiffel Tower in background

"A [V] clock"

[Figure 49]

[Figure 50]

|[Figure 51]|[Figure 52]|
|---|---|
| |[Figure 53]|

in a jungle, placed on a rock

"A [V] toy"

Figure 5. DiffuseKronA preserving superior fidelity.

Input images DiffuseKronA LoRA-DreamBooth

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

"A [V] anime"

relaxing at the ramen shop, enjoying a bowl of ramen with chopsticks and soup

[Figure 59]

[Figure 60]

|[Figure 61]|[Figure 62]|
|---|---|
| |[Figure 63]|

"A [V] anime"

is standing at a podium and conﬁdently giving a public speech with the crowd at a bustling city

Figure 6. DiffuseKronA illustrating enhanced text alignment.

MODEL TRAIN. TIME (↓) # PARAM (↓) MODEL SIZE (↓)

LoRA-DreamBooth ∼ 38 min. 5.8 M 22.32 MB DiffuseKronA ∼ 40 min. 3.8 M 14.95 MB

SDXL

LoRA-Dreambooth ∼ 5.3 min. 1.09 M 4.3 MB DiffuseKronA ∼ 5.52 min. 0.52 M 2.1MB

SD

Table 2. Exploring model efficiency metrics (DiffuseKronA variant used (a1 = 4 and a2 = 64).

images from various subjects and text prompts and show its effectiveness in generating images with high fidelity, more accurate color distribution of objects, text alignment, and stability as compared to LoRA-DreamBooth.

Fidelity & Color Distribution. Our approach consistently produces images of superior fidelity compared to LoRADreamBooth, as illustrated in Figure 5. Notably, the clock generated by DiffuseKronA faithfully reproduces the intricate details, such as the exact depiction of the numeral 3, mirroring the original image. In contrast, the output from LoRA-DreamBooth exhibits difficulties in achieving such high fidelity. Additionally, DiffuseKronA demonstrates improved color distribution in the generated images, a feature clearly evident in the RC Car images in Figure 5. Moreover, it struggles to maintain fidelity to the numeral numeral 1 on the chest of the sitting toy. Additional examples are shown in Figure 23 in the Appendix.

Text Alignment. DiffuseKronA comprehends the intricacies and complexities of text prompts provided as input, producing images that align with the given text prompts, as depicted in Figure 6. The generated image of the anime character in response to the prompt exemplifies the meticulous attention DiffuseKronA pays to detail. It elegantly captures the presence of a shop in the background and accompanying soup bowls. In contrast, LoRA-DreamBooth struggles to generate an image that aligns seamlessly with the complex input prompt. DiffuseKronA not only generates images that align with text but is also proficient in producing a diverse range of images for a given input. More supportive examples are shown in Figure 24 in the Appendix.

Superior Stability. DiffuseKronA produces images that

closely align with the input images across a wide range of learning rates, which are specifically optimized for our approach. In contrast, LoRA-DreamBooth neglects the significance of input images even within its optimal range1 which is evident in Figure 4. The generated dog images by DiffuseKronA maintain a high degree of similarity to the input images throughout its optimal range, while LoRADreamBooth struggles to perform at a comparable level. Additional examples are shown in Figure 16 in Appendix.

MODEL CLIP-I (↑) CLIP-T (↑) DINO (↑) LoRA-DreamBooth

0.785 0.301 0.661 ± 0.062 ± 0.027 ± 0.127

0.809 0.322 0.677 DiffuseKronA

± 0.052 ± 0.021 ±0.100

Table 3. Quantitative comparison of CLIP-I, CLIP-T, and DINO scores between DiffuseKronA and LoRA-Dreambooth. The obtained values are average across 42 datasets, with a learning rate of 5 × 10−4 for DiffuseKronA and 1 × 10−4 for LoRA-DreamBooth.

Complex Input images and Prompts. DiffuseKronA consistently performs well, demonstrating robust performance even when presented with intricate inputs. This success is attributed to the enhanced representational power of Kronecker Adapters. As depicted in Figure 1, DiffuseKronA adeptly captures the features of the human face and anime characters, yielding high-quality images. Additionally, from the last row of Figure 1, it is evident that DiffuseKronA elegantly captures the semantic nuances of the text. For instance, considering the context of, “without blazer” and “upset sitting under the umbrella”, DiffuseKronA generates exceptional images which demonstrate that even when the input text prompt is huge, DiffuseKronA adeptly captures various concepts mentioned as nouns in the text. It generates images that encompass all the specified concepts while maintaining a coherent and meaningful overall relationship. Furthermore, we refer the readers to Figure 10 and Figure 11 in the Appendix.

1Optimal learning rates are determined through extensive experimentation. Additionally, we have considered observations from (von Platen et al., 2023; Ruiz et al., 2023a) while fine-tuning LoRA-DreamBooth.

- (a)

[Figure 64]

[Figure 65]

(d)

“A [V] dog”

[Figure 66]

[Figure 67]

[Figure 68]

- (b) “A [V] plushy”

Input Images

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

|DiffuseKronA|
|---|

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

LoRA-DreamBooth

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

“A [V] sculpture”

“A [V] sculpture”

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

DreamBooth LoRA-SVDiff SVDiff Custom Diffusion

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

[Figure 127]

[Figure 128]

[Figure 129]

- (c)

“A [V] sculpture holding flowers”

[Figure 130]

[Figure 131]

[Figure 132]

“A [V] plushy on a skateboard in times square”

[Figure 133]

“A [V] dog in a swimming pool”

[Figure 134]

[Figure 135]

[Figure 136]

“A [V] sculpture made of glass”

- Figure 7. Qualitative comparison between SVDiff, Custom Diffusion, DreamBooth, LoRA-DreamBooth, and our DiffuseKronA. Baseline visual images are extracted from Figure 5 of SVDiff (Han et al., 2023). Notably, our methods’ results are generated considering a2 = 8. We maintain the original settings of all these methods and used the SD CompVis-1.4 (CompVis, 2021) variant to ensure a fair comparison.

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

- Figure 8. Quantitative comparison in terms of a) parameter reduction (↑ better), and b) text & image alignment using CLIP-I and DINO with CLIP-T scores, independently computed for each prompt on the same set of input images shown in Figure 7.

Quantitative Results. The distinction in the performance of DiffuseKronA and LoRA-DreamBooth is visually evident and is further supported by quantitative measures presented in Table 3, where our model constantly generates images with better DINO and CLIP-I scores and maintains good CLIP-T. The scores for individual datasets are present in Appendix B. Furthermore, a detailed comparison of our method with other low-rank decomposition methods including LoKr and LoHA (Yeh et al., 2023) are being compared qualitatively and quantitatively in Figure 26 and Table 7, respectively.

#### 4.4. Comparison with State-of-the-arts

We compare DiffuseKronA with four related methods, including DreamBooth (Ruiz et al., 2023a), LoRADreamBooth (von Platen et al., 2023), Custom Diffusion (Kumari et al., 2023), SVDiff (Han et al., 2023), and

MODEL # PARAMETERS (↓) CLIP-I (↑) CLIP-T (↑) DINO (↑) Custom Diffusion 57.1 M

0.769 0.241 0.603 ± 0.043 ± 0.029 ± 0.055 DreamBooth 982.5 M

0.796 0.268 0.701 ± 0.051 ± 0.013 ± 0.062 LoRA-DreamBooth 1.09 M

0.808 0.260 0.710 ± 0.042 ± 0.017 ± 0.0517 SVDiff 0.44 M

0.806 0.265 0.705 ± 0.045 ± 0.019 ± 0.053

0.822 0.269 0.732 DiffuseKronA 0.32 M

± 0.0259 ± 0.011 ± 0.039

Table 4. Quantitative comparison of DiffuseKronA (used variant, a1 = 8 and a2 = 64) with SOTA in terms of the number of trainable parameters, text-alignment, and image-alignment scores. The scores are derived from the same set of images and prompts as depicted in Figure 7 and Figure 31.

LoRA-SVDiff (Han et al., 2023). As shown in Figure 7, our DiffuseKronA generates high-fidelity images that adhere to input text prompts due to the structure-preserving ability and multiplicative rank property of Kronecker product-based

adaption. The images generated by LoRA-DreamBooth often require extensive fine-tuning to achieve the desired results. Methods like custom diffusion take more parameters to fine-tune the model. As compared to SVDiff our proposed approach excels in both (a) achieving superior image-text alignment, as depicted in Figure 8, and (b) maintaining parameter efficiency. For each method, we showcase text and image alignment scores in Figure 8 and DiffuseKronA obtains the best alignment qualitatively and quantitatively. Additional results across a variety of datasets and prompts are presented in Figure 31 and Figure 32. Moreover, we present the average scores of all baseline models across 12 datasets, each evaluated with 10 prompts in Table 4.

### 5. Conclusion

We proposed a new parameter-efficient adaption module, DiffuseKronA, to enhance text-to-image personalized diffusion models, aiming to achieve high-quality image generation with improved parameter efficiency. Leveraging the Kronecker product’s capacity to capture structured relationships in weight matrices, DiffuseKronA produces images closely aligned with input text prompts and training images, outperforming LoRA-DreamBooth in visual quality, text alignment, fidelity, parameter efficiency, and stability. DiffuseKronA thus provides a new and efficient tool for advancing text-to-image personalized image generation tasks.

### References

Balaji, Y., Nah, S., Huang, X., Vahdat, A., Song, J., Kreis, K., Aittala, M., Aila, T., Laine, S., Catanzaro, B., et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2:3, 2023.

Caron, M., Touvron, H., Misra, I., J´egou, H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers, 2021.

Chang, H., Zhang, H., Barber, J., Maschinot, A., Lezama, J., Jiang, L., Yang, M.-H., Murphy, K., Freeman, W. T., Rubinstein, M., et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.

Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., and Li, Z. Pixart-α: Fast training of diffusion transformer for photorealistic textto-image synthesis, 2023.

CompVis. stable-diffusion, 2021. URL https:// github.com/CompVis/stable-diffusion.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Edalati, A., Tahaei, M. S., Rashid, A., Nia, V. P., Clark, J. J., and Rezagholizadeh, M. Compacter: Efficient low-rank hypercomplex adapter layers. arXiv preprint arXiv:2106.04647, 2021.

Edalati, A., Tahaei, M., Kobyzev, I., Nia, V. P., Clark, J. J., and Rezagholizadeh, M. Krona: Parameter efficient tuning with kronecker adapter. arXiv preprint

- arXiv:2212.10650, 2022a.

Edalati, A., Tahaei, M., Kobyzev, I., Nia, V. P., Clark, J. J., and Rezagholizadeh, M. Krona: Parameter efficient tuning with kronecker adapter. arXiv preprint

- arXiv:2212.10650, 2022b.

Edalati, A., Tahaei, M., Rashid, A., Nia, V., Clark, J., and Rezagholizadeh, M. Kronecker decomposition for gpt compression. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 219–226. Association for Computational Linguistics, 2022c.

Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A. H., Chechik, G., and Cohen-Or, D. An image is worth one word: Personalizing text-to-image generation using textual inversion, 2022. URL https://arxiv.org/ abs/2208.01618.

Greenewald, K. and Hero, A. O. Robust kronecker product pca for spatio-temporal covariance estimation. IEEE Transactions on Signal Processing, 63(23):6368–6378, 2015.

Gu, S., Chen, D., Bao, J., Wen, F., Zhang, B., Chen, D., Yuan, L., and Guo, B. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10696–10706, 2022.

Gu, Y., Wang, X., Wu, J. Z., Shi, Y., Chen, Y., Fan, Z., Xiao, W., Zhao, R., Chang, S., Wu, W., et al. Mixof-show: Decentralized low-rank adaptation for multiconcept customization of diffusion models. arXiv preprint arXiv:2305.18292, 2023.

Hameed, M. G. A., Tahaei, M. S., Mosleh, A., Nia, V. P., Chen, H., Deng, L., Yan, T., and Li, G. Convolutional neural network compression through generalized kronecker product decomposition. IEEE Transactions on Neural Networks and Learning Systems, 34(5):2205–2219, 2023.

Han, L., Li, Y., Zhang, H., Milanfar, P., Metaxas, D., and Yang, F. Svdiff: Compact parameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023.

He, K., Zhang, X., Ren, S., and Sun, J. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification, 2015.

He, X., Li, C., Zhang, P., Yang, J., and Wang, X. E. Parameter-efficient model adaptation for vision transformers. arXiv preprint arXiv:2203.16329, 2022.

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., and Gelly, S. Parameter-efficient transfer learning for nlp. In Int. Conf. Mach. Learn., pp. 2790–2799. PMLR, 2019.

Hu, E. J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. In ICLR, 2021.

Kumari, N., Zhang, B., Zhang, R., Shechtman, E., and Zhu, J.-Y. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1931– 1941, 2023.

Li, C., Farkhoor, H., Liu, R., and Yosinski, J. Measuring the intrinsic dimension of objective landscapes. In

International Conference on Learning Representations, 2018.

Li, D., Li, J., and Hoi, S. C. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. arXiv preprint arXiv:2305.14720, 2023a.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023b.

Li, Y., Zhang, R., Lu, J., and Shechtman, E. Few-shot image generation with elastic weight consolidation. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Lui, C., Bhowmick, S. S., and Jatowt, A. Kandinsky: Abstract art-inspired visualization of social discussions. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR’19, pp. 1345–1348, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450361729. doi: 10.1145/ 3331184.3331411. URL https://doi.org/10.

1145/3331184.3331411.

Ma, J., Liang, J., Chen, C., and Lu, H. Subjectdiffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023.

Mou, C., Wang, X., Xie, L., Zhang, J., Qi, Z., Shan, Y., and Qie, X. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023.

Nagy, J. G. and Perrone, L. Kronecker products in image restoration. In Advanced Signal Processing Algorithms, Architectures, and Implementations XIII, volume 5205, pp. 155–163. International Society for Optics and Photonics, 2003.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234–241. Springer, 2015.

Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., and Aberman, K. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pp. 22500–22510, June 2023a.

Ruiz, N., Li, Y., Jampani, V., Wei, W., Hou, T., Pritch, Y., Wadhwa, N., Rubinstein, M., and Aberman, K. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models, 2023b.

Ryu, S. Low-rank adaptation for fast text-to-image diffusion fine-tuning, 2023. URL https://github.com/ cloneofsimo/lora.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35: 36479–36494, 2022.

Tahaei, M., Charlaix, E., Nia, V., Ghodsi, A., and Rezagholizadeh, M. KroneckerBERT: Significant compression of pre-trained language models through kronecker decomposition and knowledge distillation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2116–2127, Seattle, United States, 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main. 154. URL https://aclanthology.org/2022.

naacl-main.154.

Tahaei, M., Charlaix, E., Nia, V., Ghodsi, A., and Rezagholizadeh, M. Kroneckerbert: Significant compression of pre-trained language models through kronecker decomposition and knowledge distillation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2116–2127, 2022b.

Tewel, Y., Gal, R., Chechik, G., and Atzmon, Y. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH 2023 Conference Proceedings, pp. 1– 11, 2023.

Thakker, U., Beu, J., Gope, D., Zhou, C., Fedorov, I., Dasika, G., and Mattina, M. Compressing rnns for iot devices by 15-38x using kronecker products. arXiv preprint arXiv:1906.02876, 2019.

von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., and Wolf, T. Diffusers: State-of-the-art diffusion models, 2023. URL https: //github.com/huggingface/diffusers.

Wang, D., Wu, B., Zhao, G., Yao, M., Chen, H., Deng, L., Yan, T., and Li, G. Kronecker cp decomposition with fast multiplication for compressing rnns. IEEE Transactions on Neural Networks and Learning Systems, 34(5): 2205–2219, 2023.

Ye, H., Zhang, J., Liu, S., Han, X., and Yang, W. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Yeh, S.-Y., Hsieh, Y.-G., Gao, Z., Yang, B. B., Oh, G., and Gong, Y. Navigating text-to-image customization: From lycoris fine-tuning to model evaluation. arXiv preprint arXiv:2309.14859, 2023.

Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B. K., et al. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022.

Zhang, A., Tay, Y., Zhang, S., Chan, A., Luu, A. T., Hui, S., and Fu, J. Beyond fully-connected layers with quaternions: Parameterization of hypercomplex multiplications with 1/n parameters. In International Conference on Learning Representations, 2020.

Zhang, L., Rao, A., and Agrawala, M. Adding conditional control to text-to-image diffusion models, 2023.

Zhang, X., Yu, F. X., Guo, R., Kumar, S., Wang, S., and Chang, S.-F. Fast orthogonal projection based on kronecker product. In 2015 IEEE International Conference on Computer Vision (ICCV), pp. 2929–2937, 2015. doi: 10.1109/ICCV.2015.335.

## Table of Contents

- A Background . . . . . . . . . . . . . . . . . . 12
- B Datasets Descriptions . . . . . . . . . . . . . 15
- C Evaluation Metrics . . . . . . . . . . . . . . 15
- D DiffuseKronA Ablations Study . . . . . . . . 16

- D.1 Choice of modules to fine-tune the model16
- D.2 Effect of Kronecker Factors . . . . . . . 17
- D.3 Effect of learning rate . . . . . . . . . . 21
- D.4 Effect of training steps . . . . . . . . . 21
- D.5 Effect of the number of training images 24 D.5.1 One shot image generation . . . . . 24
- D.6 Effect of Inference Hyperparameters . . 24

- E Detailed study on LoRA-DreamBooth vs DiffuseKronA . . . . . . . . . . . . . . . . . 24

- E.1 Multiplicative Rank Property and Gradient Updates . . . . . . . . . . . . . . 24
- E.2 Fidelity & Color Distribution . . . . . . 25
- E.3 Text Alignment . . . . . . . . . . . . . 28
- E.4 Complex Input images and Prompts . . 28
- E.5 Qualitative and Quantitative comparison 28

- F Comparison with other Low-Rank Decomposition methods . . . . . . . . . . . . . . . 29
- G Comparison with state-of-the-arts . . . . . . 40
- H Practical Implications . . . . . . . . . . . . 40

### A. Background

Primarily in 1998, the practical implications of the Kronecker product were introduced in (Nagy & Perrone, 2003) for the task of image restoration. This study presented a flexible preconditioning approach based on Kronecker product and singular value decomposition (SVD) approximations. The approach can be used with a variety of boundary conditions, depending on what is most appropriate for the specific deblurring application.

In the realm of parameter-efficient fine-tuning (PEFT) of large-scale models in deep learning, several literature studies (Tahaei et al., 2022a; Edalati et al., 2022a; He et al., 2022; Thakker et al., 2019) have explored the efficacy of Kronecker products, illustrating their applications across diverse domains.

Most of the shown images in this study are generated using the SDXL (Podell et al., 2023) backbone. However, for comparison figures, we have used the SD CompVis-1.4 (CompVis, 2021) variant and we have explicitly mentioned in the captions of these figures.

In context, COMPACTER (Edalati et al., 2021) was the first line of work that proposes a method for fine-tuning large-scale language models with a better trade-off between task performance and the number of trainable parameters than prior work. It builds on top of ideas from adapters (Houlsby et al., 2019), low-rank optimization (Li et al., 2018) (by leveraging Kronecker products), and parameterized hypercomplex multiplication layers (Zhang et al., 2020). KroneckerBERT (Tahaei et al., 2022a) significantly compressed Pre-trained Language Models (PLMs) through Kronecker decomposition and knowledge distillation. It leveraged Kronecker decomposition to compress the embedding layer and the linear mappings in the multi-head attention, and the feed-forward network modules in the Transformer layers within BERT (Devlin et al., 2018) model. The model outperforms state-of-the-art compression methods on the GLUE and SQuAD benchmarks. In a similar line of work, KronA (Edalati et al., 2022a) proposed a Kronecker product-based adapter module for efficient fine-tuning of Transformer-based PLMs (T5 (Raffel et al., 2020)) methods on the GLUE benchmark.

|*|*|*|*|
|---|---|---|---|
|*|*|*|*|
|*|*|*|*|
|*|*|*|*|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Figure 9. Demonstrating the functioning of the Kronecker product.

Apart from the efficient fine-tuning of PLMs, studies also shed some light on applying Kronecher products in the compression of convolution neural networks (CNNs) and vision transformers (ViTs). For instance, in (Hameed et al., 2023), the authors compressed CNNs through generalized Kronecker product decomposition (GKPD) with a fundamental objective to reduce both memory usage and the required floating-point operations for convolutional layers in CNNs. This approach offers a plug-and-play module that can be effortlessly incorporated as a substitute for any convolutional layer, offering a convenient and adaptable solution. Recently proposed, KAdaptation (He et al., 2022) studies parameter-efficient model adaptation strategies for ViTs on the image classification task. It formulates efficient model

Input Images Generated Images by DiﬀuseKronA

|[Figure 212]|[Figure 213]|
|---|---|
|[Figure 214]|[Figure 215]|

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

“A [V] person” dressedSilk Sherwaniin a Banarasi takingbasketballa shot in in inan aastronautspaceship suit mountainswatercolourin backgroundpainting, cookingin a chef'sin a kitchenoutﬁt, blossomsstanding underof a cherrythe pinktree

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

|[Figure 228]|[Figure 229]|
|---|---|
|[Figure 230]|[Figure 231]|

showcasing a Classic Suit from Ralph Lauren

dressed as a knight, standing in a medieval castle

in a Classic DoubleBreasted Suit with checkered pattern

standing under the pink blossoms of a cherry tree

in a Nike outﬁt with a jacket, t-shirt, and a sporty cap

selﬁe in front of Eiﬀel Tower

“A [V] person”

Input Images Generated Images by DiﬀuseKronA

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

|[Figure 238]|[Figure 239]|
|---|---|
|[Figure 240]|[Figure 241]|

as a chef, preparing dishes in a busy kitchen

a space explorer, wearing a high-tech spacesuit

sitting on a swing under a cherry blossom tree

at a summer ﬁreworks festival

riding a bicycle through a quaint town

playing with a puppy in a grassy ﬁeld

“A [V] anime”

[Figure 242]

|[Figure 243]|[Figure 244]|
|---|---|
|[Figure 245]|[Figure 246]|

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

as a guardian of underwater connecting dimensions

in traditional Japanese attire like goddess of water

in a cyberpunk rebellion, leading a group ﬁghting

integrated into her robotic parts.”

working as an engineer looking like a captain

as ice sculptures inside a glass

“A [V] anime”

Figure 10. The results for the human face and anime characters generation highlight our method’s endless application in creating portraits, animes, and avatars.

Input Images Generated Images by DiﬀuseKronA

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

|[Figure 256]|
|---|

[Figure 257]

[Figure 258]

|[Figure 259]|
|---|

A [V] car on crowded street

A [V] car on beach A [V] car on rain-

A stunt [V] car ﬂying through the air

A [V] car in wheat ﬁeld

A [V] car of pink colour

soaked streets

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

|[Figure 266]|
|---|

A [V] car in pillars of heaven

A [V] silver car in front of Eiﬀel Tower

A [V] car with open sunroof in mountains

A [V] car in vibrant graﬃti art theme.

A [V] shiny car of golden colour

A [V] car submerged in water”

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

|[Figure 273]|
|---|

A [V] car with mountains in background

A [V] dusty car showcasing its charm on streets of Italy

A [V] car painting blending from swarm of butterﬂies.

A [V] car integrated in neon-lit cyberpunk cityscape

A [V] car in high speed, like cinematic eﬀect

A [V] car with ghostly eﬀects with ethereal eﬀects.

“A [V] c ar”

###### Figure 11. Results for car modifications and showcasing our method’s potential application in the Automobile industry.

|[Figure 274]|
|---|

|[Figure 275]|
|---|

|[Figure 276]|
|---|

|[Figure 277]|
|---|

|[Figure 278]|
|---|

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

|[Figure 284]|
|---|

|[Figure 285]|
|---|

|[Figure 286]|
|---|

|[Figure 287]|
|---|

|[Figure 288]|
|---|

|[Figure 289]|
|---|

|[Figure 290]|
|---|

|[Figure 291]|
|---|

|[Figure 292]|
|---|

|[Figure 293]|
|---|

|[Figure 294]|
|---|

|[Figure 295]|
|---|

|[Figure 296]|
|---|

|[Figure 297]|
|---|

|[Figure 298]|
|---|

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

|[Figure 302]|
|---|

|[Figure 303]|
|---|

|[Figure 304]|
|---|

|[Figure 305]|
|---|

|[Figure 306]|
|---|

|[Figure 307]|
|---|

|[Figure 308]|
|---|

|[Figure 309]|
|---|

|[Figure 310]|
|---|

|[Figure 311]|
|---|

|[Figure 312]|
|---|

|[Figure 313]|
|---|

|[Figure 314]|
|---|

|[Figure 315]|
|---|

- Figure 12. A collection of sample images representing all individual subjects involved in this study. Our collected subjects are highlighted in green.

adaptation as a subspace training problem via Kronecker Adaptation (KAdaptation) and performs a comprehensive benchmarking over different efficient adaptation methods.

being applied for GPT compression (Edalati et al., 2022c) which attempts to compress the linear mappings within the GPT-2 model. The proposed model, Kronecker GPT-2 (KnGPT2) is initialized based on the Kronecker decomposed version of the GPT-2 model. Subsequently, it undergoes a very light pre-training on only a small portion of the training data with intermediate layer knowledge distillation (ILKD). From the aforementioned literature study, we have witnessed the efficacy of Kronecker products for the task of model compression within various domains including NLP, RNN, CNN, ViT, and GPT space. Consequently, it has sparked considerable interest in exploring its impact on Generative models.

On the other hand, authors of (Thakker et al., 2019) compressed RNNs for resource-constrained environments (e.g. IOT devices) using Kronecker product (KP) by 15-38x with minimal accuracy loss and by quantizing the resulting models to 8 bits, the compression factor is further pushed to 50x. In (Wang et al., 2023), RNNs are compressed based on a novel Kronecker CANDECOMP/PARAFAC (KCP) decomposition, derived from Kronecker tensor (KT) decomposition, by proposing two fast algorithms of multiplication between the input and the tensor-decomposed weight. Besides all of the above, Kronecker decomposition is also

### B. Datasets Descriptions

We have incorporated a total of 25 datasets from DreamBooth (Ruiz et al., 2023a), encompassing images of backpacks, dogs, cats, and stuffed animals. Additionally, we integrated 7 datasets from custom diffusion (Kumari et al., 2023) to introduce variety in our experimentation. To assess our model’s ability to capture spatial features on faces, we curated a dataset consisting of 4 to 7 images each of 4 humans, captured from different angles. To further challenge our model against complex input images and text prompts, we compiled a dataset featuring 6 anime images from various sources. All datasets are categorized into four groups: living animals, non-living objects, anime, and human faces. Furthermore, the keywords utilized for finetuning the model remain consistent with those specified in the original papers. In Figure 12, we present a sample image for all the considered subjects used in this study.

Image Attribution. Our collected datasets are taken from the following resources:

- – https://wallpaperforu.com/tag/ komi-shouko-wallpaper/page/2/
- – https://www.tiktok.com/@anime_ geek00/video/7304798157894995243
- – https://wall.alphacoders.com/big. php?i=1305702
- – http://m.gettywallpapers.com/ komi-can-t-communicate-wallpapers/

#### • Kakashi Hatake:

- – https://www.ranker.com/list/ best-kakashi-hatake-quotes/ ranker-anime?page=2;
- – https://www.wallpaperflare.com/ search?wallpaper=Hatake+Kakashi
- – https://www.peakpx.com/en/ hd-wallpaper-desktop-kiptm
- – https://in.pinterest.com/pin/ 584060645404620659/

- • Rolls Royce:

- – https://www.peakpx.com/en/ hd-wallpaper-desktop-pxxec
- – https://4kwallpapers.com/ cars/rolls-royce-ghost-2020/ white-background-5k-8k-2554.html
- – https://www.cardekho.com/ Rolls-Royce/Rolls-Royce_Ghost/ pictures#leadForm
- – https://www.rolls-roycemotorcars. com/en_US/showroom/ ghost-digital-brochure.html

- • Hugging Face: https://huggingface.co/brand
- • Nami:

- – http://m.gettywallpapers.com/ nami-pfps-2/
- – https://tensor.art/models/ 616615209278282245
- – https://www.facebook.com/ NamiHotandCute/?locale=bs_BA
- – https://k.sina.cn/article_ 1655152542_p62a79f9e02700nhhe.html

- • Kiriko:

- – https://in.pinterest.com/pin/ 306948530865002366/
- – https://encrypted-tbn2.gstatic.com/ images?q=tbn:ANd9GcSDSk98Uw3O2XW_ RFC1jD_Kmw70JWU459euVYtU9nn1CpzPDcwS
- – https://comisc.theothertentacle.com/ overwatch+kiriko+fanart
- – https://www.1999.co.jp/eng/11030018

- • Shoko Komi:

### C. Evaluation Metrics

We utilize metrics introduced in DreamBooth (Ruiz et al., 2023a) for evaluation: DINO and CLIP-I scores measure subject fidelity, while CLIP-T assesses image-text alignment. The DINO score is the normalized pairwise cosine similarity between the ViT-S/16 DINO embeddings of the generated and input (real) images. Similarly, the CLIP-I score is the normalized pairwise CLIP ViT-B/32 image embeddings of the generated and input images. Meanwhile, the CLIP-T score computes the normalized cosine similarity between the given text prompt and generated image CLIP embeddings.

Let’s denote the pre-trained CLIP Image encoder as I, the CLIP text encoder as T , and the DINO model as V. We measure cosine similarity between two embeddings x and y as sim(x, y) = ||x||·||x.yy||. Given two sets of images, we represent the input image set as R = {Ri}ni=1 and generated image set as G = {Gi}mi=1 corresponding to the prompt set P = {Pi}mi=1, where m and n represents the number of generated and input images, respectively and R,G ∈ R3×H×W (H and W is the height and width of the image). Then, CLIP-I image-to-image and CLIP-T text-toimage similarity scores would be computed as SCLIPI and SCLIPT , respectively.

1 mn

SCLIPI =

n

m

sim(I (Ri), I (Gj)) (6)

i=1

j=1

1 m

SCLIPT =

m

sim(I (Gi), T (Pi)) (7)

i=1

Subject Cat Cat2 Dog2 Dog Dog3 Dog6

CLIP-I 0.858 ± 0.017 0.826 ± 0.030 0.833 ± 0.023 0.854 ± 0.015 0.789 ± 0.027 0.845 ± 0.031 CLIP-T 0.348 ± 0.033 0.343 ± 0.030 0.331 ± 0.028 0.349 ± 0.029 0.338 ± 0.025 0.323 ± 0.032

DINO 0.814 ± 0.025 0.752 ± 0.021 0.750 ± 0.049 0.856 ± 0.008 0.549 ± 0.060 0.788 ± 0.017 Subject Dog5 Dog7 Dog8 Doggy Cat3 Cat4

CLIP-I 0.824 ± 0.024 0.853 ± 0.015 0.829 ± 0.021 0.734 ± 0.031 0.834 ± 0.034 0.861 ± 0.016 CLIP-T 0.337 ± 0.026 0.334 ± 0.025 0.343 ± 0.026 0.329 ± 0.030 0.348 ± 0.029 0.349 ± 0.032

DINO 0.761 ± 0.001 0.730 ± 0.049 0.717 ± 0.050 0.686 ± 0.039 0.744 ± 0.031 0.863 ± 0.030 Subject Nami (Anime) Kiriko (Anime) Kakshi (Anime) Shoko Komi (Anime) Harshit (Human) Nityanand (Human)

CLIP-I 0.781 ± 0.035 0.738 ± 0.039 0.834 ± 0.028 0.761 ± 0.029 0.724 ± 0.018 0.665 ± 0.031 CLIP-T 0.337 ± 0.029 0.320 ± 0.032 0.318 ± 0.031 0.356 ± 0.028 0.297 ± 0.036 0.307 ± 0.030

DINO 0.655 ± 0.023 0.483 ± 0.041 0.617 ± 0.061 0.596 ± 0.024 0.555 ± 0.025 0.447 ± 0.068 Subject Shyam (Human) Teapot Robot Toy Backpack Dog Backpack Rc Car

CLIP-I 0.731 ± 0.015 0.836 ± 0.051 0.828 ± 0.026 0.907 ± 0.026 0.774 ± 0.037 0.797 ± 0.020 CLIP-T 0.297 ± 0.026 0.347 ± 0.025 0.285 ± 0.032 0.347 ± 0.021 0.333 ± 0.027 0.321 ± 0.027

DINO 0.531 ± 0.030 0.528 ± 0.132 0.642 ± 0.023 0.660 ± 0.088 0.649 ± 0.037 0.651 ± 0.065 Subject Shiny Shoes Duck Clock Vase Plushie1 Monster Toy

CLIP-I 0.806 ± 0.025 0.845 ± 0.023 0.825 ± 0.062 0.827 ± 0.013 0.897 ± 0.014 0.782 ± 0.041 CLIP-T 0.308 ± 0.023 0.303 ± 0.016 0.308 ± 0.035 0.332 ± 0.026 0.308 ± 0.030 0.308 ± 0.029

DINO 0.735 ± 0.090 0.682 ± 0.049 0.590 ± 0.158 0.705 ± 0.025 0.813 ± 0.027 0.573 ± 0.060 Subject Plushie2 Plushie3 Building Book Car HuggingFace

CLIP-I 0.803 ± 0.022 0.792 ± 0.015 0.852 ± 0.013 0.695 ± 0.023 0.830 ± 0.024 0.810 ± 0.002 CLIP-T 0.324 ± 0.024 0.337 ± 0.031 0.268 ± 0.023 0.301 ± 0.022 0.299 ± 0.032 0.288 ± 0.042

DINO 0.728 ± 0.020 0.766 ± 0.033 0.742 ± 0.019 0.579 ± 0.040 0.684 ± 0.036 0.692 ± 0.001

Table 5. Average metrics (CLIP-I, CLIP-T, and DINO scores) from various prompt runs for each subject using our proposed method.

Similarly, the DINO image-to-image similarity score would be computed as

1 mn

SDINO =

m

n

sim(V (Ri), V (Gj)). (8)

j=1

i=1

Notably, the DINO score is preferred to assess subject fidelity owing to its sensitivity to differentiate between subjects within a given class. In personalized T2I generations, all three metrics should be considered jointly for evaluation to avoid a biased conclusion. For instance, models that copy training set images will have high DINO and CLIP-I scores but low CLIP-T scores, while a vanilla T2I generative model like SD and SDXL without subject knowledge will produce high CLIP-T scores with poor subject alignment. As a result, both models are not considered desirable for the subject-driven T2I generation. In Table-5, we showcase mean subject-specific CLIP-I, CLIP-T, and DINO scores along with standard deviations computed across 36 datasets, with a total of around 1600 generated images and prompts.

### D. DiffuseKronA Ablations Study

As outlined in 4.2 of the main paper, we explore various trends and observations derived from extensive experimentation on the datasets specified in Figure 12.

#### D.1. Choice of modules to fine-tune the model

Within the UNet network’s transformer block, the linear layers consist of two components: a) attention matrices and b) a feed-forward network (FFN). Our investigation focuses on discerning the weight matrices with the highest importance for fine-tuning, aiming for efficiency in parameter utilization.

Our findings reveal that fine-tuning only the attention weight matrices, namely (WK,WQ,WV ,WO), proves to be the most impactful and parameter-efficient strategy. Conversely, fine-tuning the FFN layers does not significantly enhance image synthesis quality but substantially increases the parameter count, approximately doubling the computational load. Refer to Figure 13 for a visual representation comparing synthesis image quality with and without fine-tuning FFN layers on top of attention matrices. This graph unequivocally demonstrates that incorporating MLP layers does not enhance fidelity in the results. On the contrary, it diminishes the quality of generated images in certain instances, such as “A [V] backpack in sunflower field”, while concurrently escalating the number of trainable parameters substantially, approximately 2x times.

This approach of exclusively fine-tuning attention layers not only maximizes efficiency but also helps maintain a lower overall parameter count. This is particularly advanta-

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

Input Images

|[Figure 320]|[Figure 321]|
|---|---|
|[Figure 322]|[Figure 323]|

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

- a) “A [V] teapot”

withMLPwithoutMLP

“in front of Eiﬀel Tower” “of red colour” “on top of dirt road ”

Input Images

“on top of water ”

- b)“A [V] backpack”

“on purple rug”

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

withMLP

|[Figure 332]|[Figure 333]|
|---|---|
|[Figure 334]|[Figure 335]|

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

withoutMLP

“in sunﬂower ﬁeld” “in front of Eiﬀel “on a cobblestone Tower” street”

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

- Figure 13. Qualitative and Quantitative comparison between fine-tuning with MLP and w/o MLP. Fine-tuning MLP layers introduces more parameters and doesn’t enhance image generation compared with fine-tuning solely attention-weight matrices. So, the best outcomes and efficient use of parameters occur when only attention weight (without MLP) matrices are fine-tuned.

geous when computational resources are limited, ensuring computational efficiency in the fine-tuning process.

Kronecker factors Ak and Bk. We observed that initializing both factors with the same type of initialization failed to preserve fidelity. Surprisingly, initializing Bk with zero yielded the best results in the fine-tuning process.

#### D.2. Effect of Kronecker Factors

As illustrated in Figure 14, images initialized with (Ak = Normals, Bk = 0) and (Ak = KU, Bk = 0) produce the most favorable results, while images initialized with (Ak = Normals, Bk = Normals) and (Ak = XU, Bk = XU) result in the least satisfactory generations. Here, s ∈ 1,2 denotes two different normal distributions - N (0,1/a2) and N 0, min(d,h) respectively, where d and h represents in features and out features dimension.

How to initialize the Kronecker factors? Initialization plays a crucial role in the fine-tuning process. Networks that are poorly initialized can prove challenging to train. Therefore, having a well-crafted initialization strategy is crucial for achieving effective fine-tuning. In our experiments, we explored three initialization methods: Normal initialization, Kaiming Uniform initialization (He et al., 2015), and Xavier initialization. These methods were applied to initialize the

Ak ~ Normal1 Bk=0

Ak=0 Bk ~ Normal1

Ak ~ Normal1 Bk ~ Normal1

Ak ~ Normal2 Bk=0

Ak=0 Bk ~ Normal2

Ak ~ Normal2 Bk ~ Normal2

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

5001000

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

Ak ~ XU Bk=0

Ak=0 Bk ~ XU

Ak=0 Bk=0

Ak ~ XU Bk=XU

AK ~ KU Bk=0

Ak=0 Bk ~ KU

AK ~ KU Bk ~ KU

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

5001000

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

Input Images

|[Figure 397]<br><br>[Figure 398]<br><br>[Figure 399]|[Figure 400]<br><br>[Figure 401]<br><br>[Figure 402]|
|---|---|
|[Figure 403]<br><br>[Figure 404]<br><br>[Figure 405]|[Figure 406]<br><br>[Figure 407]<br><br>[Figure 408]|

“with a blue house in the background”

Ak ~ Normal1 Bk=0

Ak=0 Bk ~ Normal2

Ak ~ Normal2 Bk ~ Normal2

Ak ~ Normal2 Bk=0

Ak=0 Bk ~ Normal1

Ak ~ Normal1 Bk ~ Normal1

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

[Figure 423]

[Figure 424]

[Figure 425]

5001000

“A [V] boot"

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

Ak ~ XU Bk=0

Ak=0 Bk ~ XU

Ak ~ XU Bk=XU

AK ~ KU Bk=0

Ak=0 Bk ~ KU

AK ~ KU Bk ~ KU

Ak=0 Bk=0

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

5001000

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

“on top of mirror”

- Figure 14. Impact of different initialization strategies: optimal outcomes are achieved when initializing Bk to zero while initializing Ak with either a Normal or Kaiming uniform distribution.

Effect of size of Kronecker Factors. The size of the Kronecker factors significantly influences the images generated by DiffuseKronA. Larger Kronecker factors tend to produce images with higher resolution and more detailing, while smaller Kronecker factors result in lower-resolution images with less detailing. Images generated with larger Kronecker factors tend to look more realistic, while those generated with smaller Kronecker factors appear more abstract. Varying the Kronecker factors can result in a wide range of images, from highly detailed and realistic to more abstract and lower resolution.

very high fidelity and detail. The features of the dog and the house in the background are more defined and realistic with the house having a blue colour as mentioned in the prompt. When a1 is halved (4) while maintaining the same (64) results in images where the dog and the house are still quite detailed due to the high value of a2, but perhaps less so than in the previous case due to the smaller value of a1. However, when the factors are small ≤ 8, not only the generated images do not adhere to the prompt, but the number of trainable parameters increases drastically.

In Table 6, we present the count of trainable parameters corresponding to different Kronecker factors.

- In Figure 15 when both a1 and a2 are set to relatively high values (8 and 64 respectively), the generated images are of

a2=2 a2=4 a2=8 a2=16 a2=32 a2=64 a2=128

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

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

a1=2

[Figure 488]

[Figure 489]

a1=4

Input Images

[Figure 490]

[Figure 491]

|[Figure 492]|[Figure 493]|
|---|---|
|[Figure 494]|[Figure 495]|

a1=8

a1=16

a1=32

“A [V] dog”

[Figure 496]

a1=64

a1=128

“A [V] dog with city in background”

a2=2 a2=4 a2=8 a2=16 a2=32 a2=64 a2=128

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

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

a1=2

[Figure 514]

[Figure 515]

a1=4

Input Images

[Figure 516]

[Figure 517]

|[Figure 518]|[Figure 519]|
|---|---|
|[Figure 520]<br><br>[Figure 521]|[Figure 522]|

a1=8

a1=16

a1=32

“A [V] dog”

[Figure 523]

a1=64

a1=128

“A [V] dog with blue house in the background”

a2=2 a2=4 a2=8 a2=16 a2=32 a2=64 a2=128

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

a1=2

a1=4

Input Images

|[Figure 531]<br><br>[Figure 532]<br><br>[Figure 533]|[Figure 534]<br><br>[Figure 535]<br><br>[Figure 536]|
|---|---|
|[Figure 537]<br><br>[Figure 538]<br><br>[Figure 539]|[Figure 540]<br><br>[Figure 541]<br><br>[Figure 542]|

a1=8

a1=16

a1=32

“A [V] teapot”

a1=64

a1=128

“A [V] toy on top of a mirror”

a2=2 a2=4 a2=8 a2=16 a2=32 a2=64 a2=128

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

a1=2

a1=4

Input Images

|[Figure 550]<br><br>[Figure 551]|[Figure 552]<br><br>[Figure 553]|
|---|---|
|[Figure 554]<br><br>[Figure 555]|[Figure 556]<br><br>[Figure 557]|

a1=8

a1=16

a1=32

“A [V] teapot”

a1=64

a1=128

“A [V] teapot in a beach”

- Figure 15. Effect of Kronecker factors i.e., a1 and a2 in image generations. Optimal selection of a1 and a2 considers image fidelity and parameter count. Following this, we choose a1 and a2 as 4 and 64, respectively, interpreting that the lower Kronecker factor (A) should have a lower dimension compared to the upper Kronecker factor (B).

###### a1 a2 # Parameters

###### a1 a2 # Parameters

###### a1 a2 # Parameters

###### a1 a2 # Parameters

2 119399520

2 238799040 4 119402880 8 59708160

2 119402880

2 59708160 4 29867520 8 14960640

4 59701440 8 29854080

4 59708160 8 29867520

1

16 14933760

2

16 29867520 32 14960640 64 7534080

4

16 14960640

8

16 7534080 32 3874560 64 2152320

32 7480320 64 3767040

32 7534080 64 3874560

128 1937280

128 3874560

128 2152320

128 1506240

###### a1 a2 # Parameters

###### a1 a2 # Parameters

###### a1 a2 # Parameters

###### a1 a2 # Parameters

2 29867520 4 14960640 8 7534080

2 14960640

2 7534080 4 3874560 8 2152320

2 3874560 4 2152320 8 1506240

4 7534080 8 3874560

16

16 3874560 32 2152320 64 1506240

32

16 2152320 32 1506240 64 1613280

64

16 1506240 32 1613280 64 2526960

16 1613280 32 2526960 64 4704120

128

128 9233340

128 1613280

128 2526960

128 4704120

Table 6. Effect of the size of Kronecker factors (i.e. a1 & a2) in terms of trainable parameter count.

1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4 1e-3 2e-3 3e-3 4e-3 5e-3

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

Input Images

|[Figure 573]|
|---|

|[Figure 574]|
|---|

“A [V] teddy on sand with stones nearby”

|[Figure 575]|
|---|

|[Figure 576]|
|---|

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

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

“A [V] pl shie”

“A [V] teddy on top of the sidewalk of the road”

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

|[Figure 609]|
|---|

|[Figure 610]|
|---|

“A [V] dog image in the form of Vincent Van Gogh painting”

|[Figure 611]|
|---|

|[Figure 612]|
|---|

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

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

“A [V] dog”

“A [V] dog lying on top of wooden ﬂoor”

- Figure 16. Effect of learning rate on subject fidelity and text adherence. The most favorable results are obtained using learning rate 5 × 10−4.

#### D.3. Effect of learning rate

The learning rate factor influences the alignment of generated images towards both text prompts and input images. Our approach yields better results when using learning rates near 5×10−4. Higher learning rates, typically around 10−3, compel the model to overfit, resulting in images closely mirroring the input images and largely ignoring the input text prompts. Conversely, lower learning rates, below 10−4, cause the model to overlook the input images, concentrating solely on the provided input text.

- In Figure 16, for “A [V] teddy on sand with stones nearby” when the learning rate is ≥ 1 × 10−3, the generated teddy bears closely resemble the input images. Additionally, the sand dunes in the images vanish, along with the removal of stones. Conversely, for learning rates in the intermediate ranges, the sand dunes and pebbles remain distinctly visible. In the context of “A [V] dog image in the form of a Vincent Van Gogh painting” in Figure 16, images close

to the rightmost edge lack a discernible painting style, appearing too similar to the input images. Conversely, images near the leftmost side exhibit a complete sense of Van Gogh’s style but lack the features present in the input images. Notably, in the images positioned in the middle, there is an excellent fusion of the painting style and the features of the input images.

#### D.4. Effect of training steps

In T2I personalization, the timely attainment of satisfactory results within a specific number of iterations is crucial. This not only reduces the overall training time but also helps prevent overfitting to the training images, ensuring efficiency and higher fidelity in image generation. With SDXL, we successfully generate desired-fidelity images within 500 iterations, if the input images and prompt complexity are not very high. However, in cases where the input image complexity or the prompt complexity requires additional refinement, it is better to extend the training up to 1000

[Figure 628]

[Figure 629]

[Figure 630]

“A [V] cat floating on water in a swimming pool”

[Figure 631]

[Figure 632]

“A [V] cat in a doctor outfit”

[Figure 633]

[Figure 634]

[Figure 635]

“A [V] backpack on top of a white rug”

[Figure 636]

[Figure 637]

“A [V] backpack with a city in the background”

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

|[Figure 648]|
|---|

|[Figure 649]|
|---|

“A [V] dog on a cobblestone street”

|[Figure 650]|
|---|

|[Figure 651]|
|---|

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

“A [V] dog”

“A [V] dog wearing a black top hat and a monocle”

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

|[Figure 672]|
|---|

|[Figure 673]|
|---|

“A [V] sunglasses worn by a bear”

|[Figure 674]|
|---|

|[Figure 675]|
|---|

[Figure 676]

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

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

“A [V] sunglasses”

“A [V] sunglasses with the Eiffel Tower in the background”

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

|[Figure 708]|
|---|

|[Figure 709]|
|---|

“A [V] cat wearing a pink sunglasses”

|[Figure 710]|
|---|

|[Figure 711]|
|---|

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

“A [V] cat”

“A [V] cat in a police outfit”

- Figure 17. Effect of training steps in image generation on SDXL. In the case of simple prompts (row 1), DiffuseKronA consistently delivers favorable results between steps 500 and 1000. Conversely, for more complex prompts (row 2), reaching the desired outcome might necessitate waiting until after step 1000.

iterations as depicted in Figure 17 and Figure 18. The images generated by DiffuseKronA show a clear progression in quality with respect to different steps. As the steps increase, the model seems to refine the details and improve the quality of the images. This iterative process allows the model to gradually improve the image, adding more details and making it more accurate to the prompt.

erates a basic image of a cat floating on water. As the iterations progress and reach 500, the model refines the image, adding more details such as the color and texture of the cat, the ripples in the water, and the details of the swimming pool. At 1000 steps the image is a detailed and realistic representation of the prompt.

In Figure 17, “A backpack on top of a white rug”, the early iterations produce a simple image of a backpack on a white surface. However, as the iterations increase, the model adds

- In Figure 17 for instance, “A cat floating on water in a swimming pool”, in the initial iterations, the model gen-

CLIP-I CLIP-T DINO

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

- Figure 18. Plots depicting image alignment, text alignment, and DINO scores against training iterations. The scores are computed from the same set of images and prompts as depicted in Figure 17.

|[Figure 743]|
|---|

Input Image

“A [V] logo”

“of soft pastel colour palette” “with a futuristic neon glow” “of 8-bit video game graphics”

[Figure 744]

[Figure 745]

[Figure 746]

“made with water colours” “sticker on a notebook” “holding a guitar in hand”

[Figure 747]

[Figure 748]

[Figure 749]

“made as a coin ”

[Figure 750]

“in an art deco style”

[Figure 751]

- Figure 19. One-shot image generation results showcase the remarkable effectiveness of DiffuseKronA while preserving high fidelity and better text alignment.

more details to the backpack, such as the zippers, pockets, and straps. It also starts to add texture to the white rug, making it look more realistic. By the final iteration, the white rug gets smoother in texture producing a fine image.

#### D.5. Effect of the number of training images

- D.5.1. ONE SHOT IMAGE GENERATION

The images are high-quality and accurately represent the text prompts. They are clear and well-drawn, and the content of each image matches the corresponding text prompt perfectly. For instance, in Figure 19, the image of the “A [V] logo” is a yellow smiley face with hands. The “made as a coin” prompt resulted in a grey ghost with a white border, demonstrating the model’s ability to incorporate abstract concepts. The “futuristic neon glow” and “made with watercolours” prompts resulted in a pink and a yellow octopus respectively, showcasing the model’s versatility in applying different artistic styles. The model’s ability to generate an image of a guitar-playing octopus on a grey notebook from the prompt “sticker on a notebook” is a testament to its advanced capabilities.

The images are diverse in terms of style and content which is impressive, especially considering that these images were generated in a one-shot setting which makes it suitable for image editing tasks. While our model demonstrates remarkable proficiency in generating compelling results with a single input image, it encounters challenges when attempting to generate diverse poses or angles. However, when supplied with multiple images (2, 3, or 4), our model adeptly captures additional spatial features from the input images, facilitating the generation of images with a broader range of poses and angles. Our model can effectively use the information from multiple input images to generate more accurate and detailed output images as depicted in Figure 20.

- D.6. Effect of Inference Hyperparameters

Guidance Score (α). The guidance score, denoted as α, regulates the variation and distribution of colors in the generated images. A lower guidance score produces a more subdued version of colors in the images, aligning with the description provided in the input text prompt. In contrast, a higher guidance score results in images with more vibrant and pronounced colors. Guidance scores ranging from 7 to 10 generally yield images with an appropriate and welldistributed color palette.

In the example of “A [V] toy” in Figure 21, when the prompt is “made of purple color”, it is evident that a reddish lavender hue is generated for a guidance score of 1 or 3. Conversely, with a guidance score exceeding 15, a mulberry shade is produced. For guidance scores close to 8, images with a pure purple color are formed.

Number of inference Steps. The number of steps plays a crucial role in defining the granularity of the generated images. As illustrated in Figure 22, during the initial steps, the model creates a subject that aligns with the text prompt and begins incorporating features from the input image. With the progression of generation, finer details emerge in the images. Optimal results, depending on the complexity of prompts, are observed within the range of 30 to 70 steps, with an average of 50 steps proving to be the most effective. However, exceeding 100 steps results in the introduction of noise and a decline in the quality of the generated images.

The quality of the generated images appears to improve with an increase in the number of inference steps. For instance, the images for the prompt “a toy” and “wearing sunglasses” appear to be of higher quality at 50 and 75 inference steps respectively, compared to at 10 inference steps.

### E. Detailed study on LoRA-DreamBooth vs DiffuseKronA

In this section, we expand our analysis of model performance (from Section 4.3), comparing LoRA-DreamBooth and DiffuseKronA across various aspects, including fidelity, color distribution, text alignment, stability, and complexity.

E.1. Multiplicative Rank Property and Gradient Updates

Let A and B be m × n and p × q matrices respectively. Suppose that A has rank r and B has rank s.

- Theorem E.1. Ranks for dot product are bound by the rank of multiplicand and multiplier, i.e. rank(A · B) = min(rank(A), rank(B)) = min(r, s).
- Theorem E.2. Ranks for Kronecker products are multiplicative i.e. rank(A ⊗ B) = rank(A) × rank(B) = r × s.

Since the Kronecker Product has the advantage of multiplicative rank, it has a better representation of the underlying distribution of images as compared to the dot product.

Another notable difference between the Low-rank decomposition (LoRA) and the Kronecker product is when computing the derivatives, denoted by d(·). In the case of LoRA, d(A · B) = d(A)·B+A·d(B). But in the case of the Kronecker product, d(A ⊗ B) = d(A) ⊗ d(B). The gradient updates in LoRA are direct without a structured relationship, whereas the Kronecker product preserves the structure during an update. While a dot product is simpler and LoRA updates each parameter independently, a Kronecker product introduces structured updates that can be beneficial when preserving relationships between parameters stored in A and B.

Input Images

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

|[Figure 757]|
|---|

- 1 Image
- 2 Images
- 3 Images
- 4 Images

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

|[Figure 763]|[Figure 764]|
|---|---|

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

|[Figure 770]|[Figure 771]|
|---|---|
| |[Figure 772]|

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

|[Figure 778]|[Figure 779]|
|---|---|
|[Figure 780]|[Figure 781]|

“on top of mirror” “ﬂoating in milk” “on pink cloth” “in beach” “in front of Eiﬀel Tower”

“A [V] toy”

Input Images

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

|[Figure 787]<br><br>[Figure 788]|
|---|

- 1 Image
- 2 Images
- 3 Images
- 4 Images

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

|[Figure 794]<br><br>[Figure 795]|[Figure 796]<br><br>[Figure 797]|
|---|---|

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

|[Figure 803]<br><br>[Figure 804]|[Figure 805]<br><br>[Figure 806]|
|---|---|
| |[Figure 807]<br><br>[Figure 808]|

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

|[Figure 814]<br><br>[Figure 815]|[Figure 816]<br><br>[Figure 817]|
|---|---|
|[Figure 818]<br><br>[Figure 819]|[Figure 820]<br><br>[Figure 821]|

“on top of “A [V] dog”

“on a cobblestone “Wearing sunglasses” “from side view” “in snow” street” Wooden floor”

- Figure 20. The influence of training images on fine-tuning. Even though DiffuseKronA produces impressive results with a single image, the generation of images with a broader range of perspectives is enhanced when more training images are provided with variations.

#### E.2. Fidelity & Color Distribution

DiffuseKronA generates images of superior fidelity as compared to LoRA-DreamBooth in lieu of the higher representational power of Kronecker Products along with its ability to capture spatial features.

In the example of “A [V] backpack” in Figure 23, the following observations can be made:

(1) “with the Eiffel Tower in the background”: The backpack generated by DiffuseKronA is pictured with the Eiffel Tower in the background, creating a striking contrast be-

tween the red of the backpack and the muted colors of the cityscape, which LoRA-DreamBooth fails to do.

- (2) “city in background”: The backpack generated by DiffuseKronA is set against a city backdrop, where the red color stands out against the neutral tones of the buildings, whereas, LoRA-DreamBooth does not generate high contrast between images.
- (3) “on the beach”: The image generated by DiffuseKronA shows the backpack on a beach, where the red contrasts with the blue of the water and the beige of the sand.

1 3 5 7 9 10 15 25 50

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

Training Images

|[Figure 834]|
|---|

|[Figure 835]|
|---|

“wearing headphones”

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

|[Figure 864]|
|---|

|[Figure 865]|
|---|

“in Times Square”

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

“A [V] Dog”

“wearing sunglasses”

1 3 5 7 9 10 15 25 50

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

|[Figure 905]<br><br>[Figure 906]|
|---|

|[Figure 907]<br><br>[Figure 908]|
|---|

“ﬂoating on water”

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

|[Figure 928]<br><br>[Figure 929]|
|---|

|[Figure 930]<br><br>[Figure 931]|
|---|

“made of purple colour”

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

“A [V] toy”

“on the street in a city ”

- Figure 21. Images produced by adjusting the guidance score (α) reveal that a score of 7 produces the most realistic results. Increasing the score beyond 7 significantly amplifies the contrast of the images.

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

|[Figure 989]|
|---|

|[Figure 990]|
|---|

|[Figure 991]|
|---|

|[Figure 992]|
|---|

Input Images

“A [V] Dog”

10 20 30 40 50 75 100 150 200

“wearing sunglasses”

“wearing headphones”

[Figure 993]

“in Times Square”

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

|[Figure 1039]<br><br>[Figure 1040]|
|---|

|[Figure 1041]<br><br>[Figure 1042]|
|---|

|[Figure 1043]<br><br>[Figure 1044]|
|---|

“A [V] toy”

10 20 30 40 50 75 100 150 200

|[Figure 1045]<br><br>[Figure 1046]|
|---|

“on the street in a city”

“ﬂoating on water”

[Figure 1047]

“made of purple colour”

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

- Figure 22. The influence of inference steps on image generation. Optimal results are achieved in the range of 50-70 steps, striking a balance between textual input and subject fidelity. Here, we opted for 50 inference steps to minimize inference time.

DiﬀuseKrona LoRA-DreamBooth

DiﬀuseKrona LoRA-DreamBooth

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

Input Images

Input Images

|[Figure 1081]|
|---|
|[Figure 1082]|
|[Figure 1083]|
|[Figure 1084]|

|[Figure 1085]|
|---|
|[Figure 1086]|
|[Figure 1087]|
|[Figure 1088]|

“which is transparent with milk inside”

“with Eiﬀel Tower in the background”

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

“with a city in the background”

“ﬂoating on top of water”

[Figure 1095]

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

“on beach”

“in front of mirror”

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

“A [V] teapot”

“A [V] backpack”

“ﬂoating on top of water”

“made of purple colour”

DiﬀuseKrona LoRA-DreamBooth

DiﬀuseKrona LoRA-DreamBooth

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

Input Images

Input Images

|[Figure 1122]<br><br>[Figure 1123]|
|---|
|[Figure 1124]<br><br>[Figure 1125]|
|[Figure 1126]<br><br>[Figure 1127]|
|[Figure 1128]<br><br>[Figure 1129]|

|[Figure 1130]<br><br>[Figure 1131]|
|---|
|[Figure 1132]<br><br>[Figure 1133]|
|[Figure 1134]<br><br>[Figure 1135]|
|[Figure 1136]<br><br>[Figure 1137]|

“with Eiﬀel Tower in the background”

“in sunﬂower ﬁeld”

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

“in sunﬂower ﬁeld”

“in front of wheat ﬁeld”

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

“ﬂoating on top of water”

“in front of Eiﬀel Tower”

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

“A [V] toy”

“A [V] plushie”

“ﬂoating on water”

“of purple colour, in the jungle ”

Figure 23. Comparison of fidelity and color preservation in DiffuseKronA and LoRA-DreamBooth.

DiﬀuseKrona LoRA-DreamBooth

DiﬀuseKrona LoRA-DreamBooth

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

[Figure 1173]

Input Images

Input Images

|[Figure 1174]|[Figure 1175]|
|---|---|
|[Figure 1176]|[Figure 1177]|

|[Figure 1178]|[Figure 1179]|
|---|---|
|[Figure 1180]|[Figure 1181]|

“attempting to cook a meal, with his culinary skills”

“which is transparent with milk inside”

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

“A [V] vase” b) “A [V] anime”

a)

“standing under a full moon with his red eyes activated and bleeding”

“with sunﬂowers inside”

DiﬀuseKrona LoRA-DreamBooth

DiﬀuseKrona LoRA-DreamBooth

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

Input Images

Input Images

|[Figure 1190]|[Figure 1191]|
|---|---|
|[Figure 1192]|[Figure 1193]|

|[Figure 1194]|[Figure 1195]|
|---|---|
|[Figure 1196]|[Figure 1197]|

“at a karaoke bar, holding a microphone,

“in a beach”

with her friends cheering”

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

c) “A [V] anime”

d)“A [V] shoes”

“sipping a cup of tea and writing notes with the other hand”

“on top of mirror”

Figure 24. Comparison of text alignment in generated images by our proposed DiffuseKronA and LoRA-DreamBooth.

#### E.3. Text Alignment

white blazer and a smiling face, while LoRA-DreamBooth struggles to maintain both the color and the smile on the face.

DiffuseKronA is more accurate in aligning text with images compared to the Lora-DreamBooth. For instance, in the first row, DiffuseKronA correctly aligns the text with “sunflowers inside” with the image of a vase with sunflowers, whereas LoRA-DreamBooth fails to align the sunflower in the vase of the same color as of input images.

Similarly, for the prompt “A [V] clock” in Figure 25, DiffuseKronA accurately reproduces detailed numbers, particularly 3, from the input images. Although it encounters challenges in preserving the structure of numbers while creating a clock of cubical shape, it still maintains a strong focus on text details— a characteristic lacking in LoRADreamBooth.

In more complex input examples like in Figure 24, such as the one involving anime in “A [V] character”, the generated images by LoRA-DreamBooth lack the sense of cooking a meal and a karaoke bar, whereas DiffuseKronA consistently produces images that closely align with the provided text prompts.

#### E.5. Qualitative and Quantitative comparison

We have assessed the image generation capabilities of DiffuseKronA and LoRA-DreamBooth on SDXL (Podell et al., 2023). Our findings reveal that DiffuseKronA excels in generating images with high fidelity, more accurate color distribution, and greater stability compared to LoRADreamBooth.

#### E.4. Complex Input images and Prompts

DiffuseKronA demonstrates a notable emphasis on capturing nuances within text prompts and excels in preserving intricate details from input images to the highest degree. In contrast, LoRA-DreamBooth lacks these properties. This distinction is evident in Figure 25, where, for the prompt “A [V] face”, DiffuseKronA successfully generates an ivory-

DiﬀuseKrona LoRA-DreamBooth

DiﬀuseKrona LoRA-DreamBooth

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

“wearing a White Dinner Jacket by Giorgio Armani

“made of blue colour ”

of ivory-white colour with a smile on face”

Input Images

Input Images

|[Figure 1210]<br><br>[Figure 1211]<br><br>[Figure 1212]|[Figure 1213]<br><br>[Figure 1214]<br><br>[Figure 1215]|
|---|---|
|[Figure 1216]<br><br>[Figure 1217]<br><br>[Figure 1218]|[Figure 1219]<br><br>[Figure 1220]<br><br>[Figure 1221]|

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

|[Figure 1231]<br><br>[Figure 1232]|[Figure 1233]<br><br>[Figure 1234]|
|---|---|
|[Figure 1235]<br><br>[Figure 1236]|[Figure 1237]<br><br>[Figure 1238]|

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

“wearing a classic suit from Ralph Lauren of a navy blue

“on top of white rug”

b) “A [V] clock”

suit with peak lapels and a subtle pinstripe pattern”

a) “A [V] face”

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

“dressed in a Banarasi Silk Sherwani, adorned

“of cube shape”

with intricate zari work”

- Figure 25. Comparison of image generation on complex prompts and input images by DiffuseKronA and LoRA-DreamBooth.

### F. Comparison with other Low-Rank Decomposition methods

pletely dependent upon dimension, and it would not be a square matrix always.

- 1 def factorization(dim: int, factor: int =

-1) -> tuple[int, int]:

- 2

- 3 if factor > 0 and (dim % factor) == 0:

- 4 m = factor

- 5 n = dim // factor

- 6 if m > n:

- 7 n, m = m, n

- 8 return m, n

- 9 if factor < 0:

- 10 factor = dim

- 11 m, n = 1, dim

- 12 length = m + n

- 13 while m < n:

- 14 new_m = m + 1

- 15 while dim % new_m != 0:

- 16 new_m += 1

- 17 new_n = dim // new_m

- 18 if new_m + new_n > length or new_m > factor:

- 19 break

- 20 else:

- 21 m, n = new_m, new_n

- 22 if m > n:

- 23 n, m = m, n

- 24 return m, n

In this section, we compare our DiffuseKronA with lowrank methods other than LoRA, specifically with LoKr (Yeh et al., 2023) and LoHA (Yeh et al., 2023). We also note that our implementation is independent of the LyCORIS project (Yeh et al., 2023), and we did not use LoKr nor LoHA in DiffuseKronA1. We summarize the key differences between DiffuseKronA and these methods as follows:

❶ DiffuseKronA has 2 controllable parameters (a1 and a2), which are chosen manually through extensive experiments (refer to Figure 15 and Table 6), whereas LoKr (Yeh et al., 2023) follows the procedure mentioned in the FACTORIZATION function (see right) which depends on input dimension and another hyper-parameter called factor. Following the descriptions on the implementation of Figure 2 in (Yeh et al., 2023), and we quote “we set the factor to 8 and do not perform further decomposition of the second block”, the default implementation makes A a square matrix of dimension (factor × factor). Notably, for any factor, f > 0, A would always be a square matrix of shape (f × f) which is a special case (a subset) of DiffusKronA (diagonal entry in Figure 15) but for f = −1, A matrix size would be com-

Listing 1. This code snippet is extracted from the official LyCORIS codebase (Link).

1To ensure a fair comparison, we have incorporated LoKr and LoHA into the SDXL backbone.

These attributes make our way of performing Kronecker decomposition a superset of LoKr, offering greater control and flexibility compared to LoKr. On the other hand, LoHA has only one controllable parameter, i.e., rank, similar to LoRA.

❷ LoKr takes the generic form of ∆W = A ⊗ (B · C), and LoHA adopts ∆W = (A · B) ⊙ (C · D), where ⊙ denotes the Hadamard product. For more details, we refer the readers to Figure 1 in (Yeh et al., 2023). Based on the definition, LoHA does not explore the benefits of using Kronecker decomposition.

❸ Yeh et al. (2023) provided the first use of Kronecker decomposition in Diffusion model fine-tuning but limited analysis in the few-shot T2I personalization setting. In our study, we conducted detailed analysis and exploration to demonstrate the benefits of using Kronecker decomposition. Our new insights include large-scale analysis of parameter efficiency, enhanced stability to hyperparameters, and improved text alignment and fidelity, among others.

❹ We further compare our DiffuseKronA with LoKr and LoHA using the default implementations from (Yeh et al., 2023) in Figure 26 and Figure 27, respectively. However, the default settings were used in the SD variant, and it is also evident that personalized T2I generations are very sensitive to model settings and hyper-parameter choices. Bearing these facts, we also explored the hyperparameters in both adapters. In Figure 28, we have presented the ablation study examining the factors and ranks for LoKr utilizing SDXL, while in Figure 29, we showcase an ablation study on the learning rate. Moreover, Figure 30 features an ablation study on the learning rate and rank for LoHA using SDXL. These analyses reveal that for LoKr, the optimal factor is -1 and the optimal rank is 8, with a learning rate of 1 × 10−3; while for LoHA, the optimal rank is 4, with a learning rate of 1 × 10−4.

Additionally, quantitative comparisons are conducted, encompassing parameter count alongside image-to-image and image-to-text alignment scores, as detailed in Table 7 and Table 8. The results in Table 7 indicate that although LoKr marginally possesses fewer parameters still DiffuseKronA with a1 = 16 achieves superior CLIP-I, CLIP-T, and DINO scores. This contrast is readily noticeable in the visual examples depicted in Figure 26. For the prompt “A [V] toy with the Eiffel Tower in the background”, LoKr fails to construct the Eiffel Tower in the background, unlike DiffuseKronA (a1 = 16). Similarly, in the case of “A [V] teapot floating on top of water” LoKr distorts the teapot’s spout, whereas DiffuseKronA maintains fidelity. In the case of “A [V] toy” (last row), the results of DiffuseKronA are much more aligned as compared to LoKr for both prompts. Conversely, for dog and cat examples, all the methods demonstrate similar visual appearance in

terms of fidelity as well as textual alignment. Consequently, it’s evident that while LoKr reduces parameter count, it struggles with complex input images or text prompts with multiple contexts. Hence, DiffusekronA achieves efficiency in parameters while upholding average scores across CLIP-I, CLIP-T, and DINO metrics. Hence, achieving a better tradeoff between parameter efficiency and personalized image generation.

MODEL # PARAMETERS (↓) CLIP-I (↑) CLIP-T (↑) DINO (↑) DiffuseKronA

0.799 0.267 0.648

3.8 M

a1 = 2 ±0.073 ±0.048 ±0.122 DiffuseKronA

0.809 0.268 0.651

7.5 M

a1 = 4 ±0.086 ±0.055 ±0.142 DiffuseKronA

0.815 0.313 0.649

2.1 M

a1 = 8 ±0.074 ±0.024 ±0.139 DiffuseKronA 0.817 0.301 0.654

1.5 M

±0.078 ±0.038 ±0.127 LoRA-DreamBooth

a1 = 16

0.807 0.288 0.635

5.8 M

rank = 4 ±0.077 ±0.033 ±0.136

LoKr

0.801 0.287 0.646

1.36 M

f = −1, rank = 8 ±0.065 ±0.049 ±0.147

LoKr

0.812 0.277 0.639 f = 8 ± 0.069 ±0.042 ±0.111 LoHA

14.9 M

0.818 0.299 0.641

20.9 M

rank = 4 ±0.064 ±0.041 ±0.120

- Table 7. Quantitative comparison of DiffuseKronA with low-rank decomposition methods namely LoRA, LoKr, and LoHA in terms of the number of trainable parameters, text-alignment, and imagealignment scores. The scores are computed from the same set of images and prompts as depicted in Figure 26.

MODEL # PARAMETERS (↓) CLIP-I (↑) CLIP-T (↑) DINO (↑)

LoKr

238.7 M

0.825 0.244 0.727 f = 2 ±0.037 ±0.024 ±0.036 LoKr

59.7 M

0.784 0.246 0.683 f = 4 ± 0.063 ±0.030 ±0.051 LoKr

14.9 M

0.749 0.292 0.568 f = 8 ±0.067 ±0.064 ±0.075 LoKr

3.8 M

0.707 0.231 0.472

f = 16 ± 0.121 ±0.025 ±0.160 DiffuseKronA 0.806 0.281 0.653

a1 = 8

2.1 M

± 0.028 ± 0.070 ± 0.045

- Table 8. Quantitative comparison of DiffuseKronA with varying factors (i.e. 2, 4, 8, 16) of LoKr in terms of the number of trainable parameters, text-alignment, and image-alignment scores. The scores are computed from the same set of images and prompts as depicted in Figure 27.

|LoRA DreamBooth|
|---|

|DiffuseKronA<br><br>(a1 =8, a2 =64)|
|---|

|DiffuseKronA<br><br>(a1 =16, a2 =64)|
|---|

|LoHA<br><br>(Rank = 4)|
|---|

|LoKr<br><br>(Factor = 8)|
|---|

|LoKr<br><br>(Factor = -1, r = 8)|
|---|

|DiffuseKronA<br><br>(a1=2, a2 =64)|
|---|

|DiffuseKronA<br><br>(a1 =4, a2 =64)|
|---|

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

|[Figure 1259]|[Figure 1260]|
|---|---|
|[Figure 1261]|[Figure 1262]|

“A [V] cat on top of a purple rug in a forest”

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

“A [V] cat”

“A [V] cat on a cobblestone street”

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

|[Figure 1279]|[Figure 1280]|
|---|---|
|[Figure 1281]|[Figure 1282]|

“A [V] dog in a dog house”

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

“A [V] dog”

“A [V] dog sleeping”

[Figure 1291]

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

|[Figure 1299]|[Figure 1300]|
|---|---|
|[Figure 1301]|[Figure 1302]|

“A [V] toy with the Eiffel Tower in the background”

[Figure 1303]

[Figure 1304]

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

“A [V] toy”

“A [V] toy floating on top of water”

[Figure 1311]

[Figure 1312]

[Figure 1313]

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

[Figure 1322]

[Figure 1323]

[Figure 1324]

|[Figure 1325]|[Figure 1326]|
|---|---|
|[Figure 1327]|[Figure 1328]|

“A [V] teapot on top of a dirt road”

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1333]

[Figure 1334]

[Figure 1335]

[Figure 1336]

“A [V] teapot”

“A [V] teapot floating on top of water”

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

[Figure 1344]

[Figure 1345]

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

|[Figure 1350]|[Figure 1351]|
|---|---|
|[Figure 1352]|[Figure 1353]|

“A [V] toy on top of a wooden floor”

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

[Figure 1361]

“A [V] toy”

“A [V] toy in the snow”

- Figure 26. Qualitative comparison of four variants of DiffusekronA with other low-rank methods including LoRA, LoKr, and LoHA. Learning rates: DiffusekronA (5 × 10−4), LoRA (1 × 10−4), LoKr (1 × 10−3) & LoHA (1 × 10−4).

factor = 2

factor = 4 factor = 8 factor = 16

DiﬀuseKronA

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

[Figure 1366]

Input Images

|[Figure 1367]|[Figure 1368]|
|---|---|
|[Figure 1369]|[Figure 1370]|

“A [V] person with Eiﬀel Tower in the background”

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

[Figure 1375]

“A [V] person”

“A Monet-inspired painting of a [V] face standing near a blooming lily pond”

factor = 2 factor = 4 factor = 8 factor = 16

DiﬀuseKronA

[Figure 1376]

[Figure 1377]

[Figure 1378]

[Figure 1379]

[Figure 1380]

Input Images

|[Figure 1381]<br><br>[Figure 1382]|[Figure 1383]<br><br>[Figure 1384]|
|---|---|
|[Figure 1385]<br><br>[Figure 1386]|[Figure 1387]<br><br>[Figure 1388]|

“A [V] toy in snow”

[Figure 1389]

[Figure 1390]

[Figure 1391]

[Figure 1392]

[Figure 1393]

“A [V] toy”

“A [V] toy in forest”

factor = 2 factor = 4 factor = 8 factor = 16

DiﬀuseKronA

[Figure 1394]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

Input Images

|[Figure 1399]<br><br>[Figure 1400]|[Figure 1401]<br><br>[Figure 1402]|
|---|---|
|[Figure 1403]<br><br>[Figure 1404]|[Figure 1405]<br><br>[Figure 1406]|

“A [V] anime as a chef, showing her culinary skills”

[Figure 1407]

[Figure 1408]

[Figure 1409]

[Figure 1410]

[Figure 1411]

“A [V] anime”

“A [V] anime reading a book in a library wearing a headphone”

- Figure 27. Qualitative comparison. Results are shown for the default factors given by the LoKr implementation, with the varying factors being 2, 4, 8, and 16.

db = True

db = False

f=-1 f=2 f=4 f=8

f=-1 f=2 f=4 f=8

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

[Figure 1417]

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

[Figure 1422]

[Figure 1423]

[Figure 1424]

[Figure 1425]

[Figure 1426]

[Figure 1427]

- r=1
- r=2

- r=1
- r=2

[Figure 1428]

[Figure 1429]

[Figure 1430]

[Figure 1431]

[Figure 1432]

[Figure 1433]

[Figure 1434]

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

[Figure 1441]

[Figure 1442]

[Figure 1443]

Input Images

[Figure 1444]

[Figure 1445]

[Figure 1446]

[Figure 1447]

[Figure 1448]

[Figure 1449]

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

r=4

r=4

[Figure 1461]

[Figure 1462]

[Figure 1463]

[Figure 1464]

[Figure 1465]

[Figure 1466]

[Figure 1467]

[Figure 1468]

[Figure 1469]

[Figure 1470]

[Figure 1471]

[Figure 1472]

[Figure 1473]

[Figure 1474]

[Figure 1475]

[Figure 1476]

[Figure 1477]

r=8

r=8

“A Monet-inspired painting of a [V] face standing near a blooming lily pond”

db = True db = False

[Figure 1478]

f=-1 f=2 f=4 f=8

f=-1 f=2 f=4 f=8

[Figure 1479]

[Figure 1480]

[Figure 1481]

[Figure 1482]

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

[Figure 1489]

[Figure 1490]

[Figure 1491]

[Figure 1492]

[Figure 1493]

[Figure 1494]

- r=1
- r=2

- r=1
- r=2

[Figure 1495]

[Figure 1496]

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

[Figure 1508]

[Figure 1509]

[Figure 1510]

[Figure 1511]

[Figure 1512]

[Figure 1513]

[Figure 1514]

[Figure 1515]

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

“A [V] person”

r=4

r=4

[Figure 1528]

[Figure 1529]

[Figure 1530]

[Figure 1531]

[Figure 1532]

[Figure 1533]

[Figure 1534]

[Figure 1535]

[Figure 1536]

[Figure 1537]

[Figure 1538]

[Figure 1539]

[Figure 1540]

[Figure 1541]

[Figure 1542]

[Figure 1543]

r=8

r=8

“A [V] person with Eiﬀel Tower in the background”

- Figure 28. Ablation study on factor and rank for LoKr using SDXL, with a learning rate of 1 × 10−3. We found that the optimal factor and rank are -1 and 8, respectively. We also experimented with db=True, which indicates further low-rank decomposition of both matrices A and B, whereas db=False means only matrix B is decomposed further. (continued..)

db = True

db = False

lr=1e-4

lr=5e-4 lr=1e-3 lr=5e-3 lr=1e-4 lr=5e-4 lr=1e-3 lr=5e-3

[Figure 1544]

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

[Figure 1550]

[Figure 1551]

- r=1
- r=2

- r=1
- r=2

[Figure 1552]

[Figure 1553]

[Figure 1554]

[Figure 1555]

[Figure 1556]

[Figure 1557]

[Figure 1558]

[Figure 1559]

Input Images

[Figure 1560]

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

[Figure 1567]

[Figure 1568]

r=4

r=4

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

[Figure 1576]

[Figure 1577]

r=8

r=8

“A [V] anime as a chef, showing her culinary skills”

db = True db = False

[Figure 1578]

lr=1e-4 lr=5e-4 lr=1e-3 lr=5e-3 lr=1e-4 lr=5e-4 lr=1e-3 lr=5e-3

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

[Figure 1584]

[Figure 1585]

[Figure 1586]

- r=1
- r=2

- r=1
- r=2

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

[Figure 1592]

[Figure 1593]

[Figure 1594]

[Figure 1595]

[Figure 1596]

[Figure 1597]

[Figure 1598]

[Figure 1599]

[Figure 1600]

[Figure 1601]

[Figure 1602]

[Figure 1603]

“A [V] anime”

r=4

r=4

[Figure 1604]

[Figure 1605]

[Figure 1606]

[Figure 1607]

[Figure 1608]

[Figure 1609]

[Figure 1610]

[Figure 1611]

r=8

r=8

“A [V] anime reading a book in a library wearing a headphone”

- Figure 28. Ablation study on factor and rank for LoKr using SDXL, with a learning rate of 1 × 10−3. We found that the optimal factor and rank are -1 and 8, respectively. We also experimented with db=True, which indicates further low-rank decomposition of both matrices A and B, whereas db=False means only matrix B is decomposed further. (continued..)

db = True

db = False

f=-1 f=2 f=4 f=8

f=-1 f=2 f=4 f=8

[Figure 1612]

[Figure 1613]

[Figure 1614]

[Figure 1615]

[Figure 1616]

[Figure 1617]

[Figure 1618]

[Figure 1619]

[Figure 1620]

[Figure 1621]

[Figure 1622]

[Figure 1623]

[Figure 1624]

[Figure 1625]

[Figure 1626]

[Figure 1627]

- r=1
- r=2

- r=1
- r=2

[Figure 1628]

[Figure 1629]

[Figure 1630]

[Figure 1631]

[Figure 1632]

[Figure 1633]

[Figure 1634]

[Figure 1635]

[Figure 1636]

[Figure 1637]

[Figure 1638]

[Figure 1639]

[Figure 1640]

[Figure 1641]

[Figure 1642]

[Figure 1643]

Input Images

[Figure 1644]

[Figure 1645]

[Figure 1646]

[Figure 1647]

[Figure 1648]

[Figure 1649]

[Figure 1650]

[Figure 1651]

[Figure 1652]

[Figure 1653]

[Figure 1654]

[Figure 1655]

[Figure 1656]

[Figure 1657]

[Figure 1658]

[Figure 1659]

[Figure 1660]

r=4

r=4

[Figure 1661]

[Figure 1662]

[Figure 1663]

[Figure 1664]

[Figure 1665]

[Figure 1666]

[Figure 1667]

[Figure 1668]

[Figure 1669]

[Figure 1670]

[Figure 1671]

[Figure 1672]

[Figure 1673]

[Figure 1674]

[Figure 1675]

[Figure 1676]

[Figure 1677]

r=8

r=8

“A [V] toy in forest” “A [V] toy in forest”

db = True db = False

[Figure 1678]

f=-1 f=2 f=4 f=8

f=-1 f=2 f=4 f=8

[Figure 1679]

[Figure 1680]

[Figure 1681]

[Figure 1682]

[Figure 1683]

[Figure 1684]

[Figure 1685]

[Figure 1686]

[Figure 1687]

[Figure 1688]

[Figure 1689]

[Figure 1690]

[Figure 1691]

[Figure 1692]

[Figure 1693]

[Figure 1694]

- r=1
- r=2

- r=1
- r=2

[Figure 1695]

[Figure 1696]

[Figure 1697]

[Figure 1698]

[Figure 1699]

[Figure 1700]

[Figure 1701]

[Figure 1702]

[Figure 1703]

[Figure 1704]

[Figure 1705]

[Figure 1706]

[Figure 1707]

[Figure 1708]

[Figure 1709]

[Figure 1710]

[Figure 1711]

[Figure 1712]

[Figure 1713]

[Figure 1714]

[Figure 1715]

[Figure 1716]

[Figure 1717]

[Figure 1718]

[Figure 1719]

[Figure 1720]

[Figure 1721]

[Figure 1722]

[Figure 1723]

[Figure 1724]

[Figure 1725]

[Figure 1726]

[Figure 1727]

“A [V] toy”

r=4

r=4

[Figure 1728]

[Figure 1729]

[Figure 1730]

[Figure 1731]

[Figure 1732]

[Figure 1733]

[Figure 1734]

[Figure 1735]

[Figure 1736]

[Figure 1737]

[Figure 1738]

[Figure 1739]

[Figure 1740]

[Figure 1741]

[Figure 1742]

[Figure 1743]

r=8

r=8

“A [V] toy in snow” “A [V] toy in snow”

- Figure 28. Ablation study on factor and rank for LoKr using SDXL, with a learning rate of 1 × 10−3. We found that the optimal factor and rank are -1 and 8, respectively. We also experimented with db=True, which indicates further low-rank decomposition of both matrices A and B, whereas db=False means only matrix B is decomposed further. (end)

db = True

db = False

lr=1e-4

lr=5e-4 lr=1e-3 lr=5e-3 lr=1e-4 lr=5e-4 lr=1e-3 lr=5e-3

[Figure 1744]

[Figure 1745]

[Figure 1746]

[Figure 1747]

[Figure 1748]

[Figure 1749]

[Figure 1750]

[Figure 1751]

- r=1
- r=2

- r=1
- r=2

[Figure 1752]

[Figure 1753]

[Figure 1754]

[Figure 1755]

[Figure 1756]

[Figure 1757]

[Figure 1758]

[Figure 1759]

Input Images

[Figure 1760]

[Figure 1761]

[Figure 1762]

[Figure 1763]

[Figure 1764]

[Figure 1765]

[Figure 1766]

[Figure 1767]

[Figure 1768]

r=4

r=4

[Figure 1769]

[Figure 1770]

[Figure 1771]

[Figure 1772]

[Figure 1773]

[Figure 1774]

[Figure 1775]

[Figure 1776]

[Figure 1777]

r=8

r=8

“A [V] person with Eiﬀel Tower in the background”

db = True db = False

[Figure 1778]

lr=1e-4 lr=5e-4 lr=1e-3 lr=5e-3 lr=1e-4 lr=5e-4 lr=1e-3 lr=5e-3

[Figure 1779]

[Figure 1780]

[Figure 1781]

[Figure 1782]

[Figure 1783]

[Figure 1784]

[Figure 1785]

[Figure 1786]

- r=1
- r=2

- r=1
- r=2

[Figure 1787]

[Figure 1788]

[Figure 1789]

[Figure 1790]

[Figure 1791]

[Figure 1792]

[Figure 1793]

[Figure 1794]

[Figure 1795]

[Figure 1796]

[Figure 1797]

[Figure 1798]

[Figure 1799]

[Figure 1800]

[Figure 1801]

[Figure 1802]

[Figure 1803]

“A [V] person”

r=4

r=4

[Figure 1804]

[Figure 1805]

[Figure 1806]

[Figure 1807]

[Figure 1808]

[Figure 1809]

[Figure 1810]

[Figure 1811]

r=8

r=8

“A Monet-inspired painting of a [V] face standing near a blooming lily pond”

- Figure 29. Ablation study on factor and learning rate for LoKr using SDXL, with a fixed factor of -1. We found that the optimal learning rate and rank are 1 × 10−3 and 8, respectively. We also experimented with db=True, which indicates further low-rank decomposition of both matrices A and B, whereas db=False means only matrix B is decomposed further. (continued..)

db = True

db = False

lr=1e-4

lr=5e-4 lr=1e-3 lr=5e-3 lr=1e-4 lr=5e-4 lr=1e-3 lr=5e-3

[Figure 1812]

[Figure 1813]

[Figure 1814]

[Figure 1815]

[Figure 1816]

[Figure 1817]

[Figure 1818]

[Figure 1819]

- r=1
- r=2

- r=1
- r=2

[Figure 1820]

[Figure 1821]

[Figure 1822]

[Figure 1823]

[Figure 1824]

[Figure 1825]

[Figure 1826]

[Figure 1827]

Input Images

[Figure 1828]

[Figure 1829]

[Figure 1830]

[Figure 1831]

[Figure 1832]

[Figure 1833]

[Figure 1834]

[Figure 1835]

[Figure 1836]

r=4

r=4

[Figure 1837]

[Figure 1838]

[Figure 1839]

[Figure 1840]

[Figure 1841]

[Figure 1842]

[Figure 1843]

[Figure 1844]

[Figure 1845]

r=8

r=8

“A [V] toy in forest” “A [V] toy in forest”

db = True db = False

[Figure 1846]

lr=1e-4 lr=5e-4 lr=1e-3 lr=5e-3 lr=1e-4 lr=5e-4 lr=1e-3 lr=5e-3

[Figure 1847]

[Figure 1848]

[Figure 1849]

[Figure 1850]

[Figure 1851]

[Figure 1852]

[Figure 1853]

[Figure 1854]

- r=1
- r=2

- r=1
- r=2

[Figure 1855]

[Figure 1856]

[Figure 1857]

[Figure 1858]

[Figure 1859]

[Figure 1860]

[Figure 1861]

[Figure 1862]

[Figure 1863]

[Figure 1864]

[Figure 1865]

[Figure 1866]

[Figure 1867]

[Figure 1868]

[Figure 1869]

[Figure 1870]

[Figure 1871]

“A [V] toy”

r=4

r=4

[Figure 1872]

[Figure 1873]

[Figure 1874]

[Figure 1875]

[Figure 1876]

[Figure 1877]

[Figure 1878]

[Figure 1879]

r=8

r=8

“A [V] toy in snow” “A [V] toy in snow”

- Figure 29. Ablation study on factor and learning rate for LoKr using SDXL, with a fixed factor of -1. We found that the optimal learning rate and rank are 1 × 10−3 and 8, respectively. We also experimented with db=True, which indicates further low-rank decomposition of both matrices A and B, whereas db=False means only matrix B is decomposed further. (continued..)

db = True

db = False

f=-1 f=2 f=4 f=8

f=-1 f=2 f=4 f=8

[Figure 1880]

[Figure 1881]

[Figure 1882]

[Figure 1883]

[Figure 1884]

[Figure 1885]

[Figure 1886]

[Figure 1887]

- r=1
- r=2

- r=1
- r=2

[Figure 1888]

[Figure 1889]

[Figure 1890]

[Figure 1891]

[Figure 1892]

[Figure 1893]

[Figure 1894]

[Figure 1895]

Input Images

[Figure 1896]

[Figure 1897]

[Figure 1898]

[Figure 1899]

[Figure 1900]

[Figure 1901]

[Figure 1902]

[Figure 1903]

[Figure 1904]

r=4

r=4

[Figure 1905]

[Figure 1906]

[Figure 1907]

[Figure 1908]

[Figure 1909]

[Figure 1910]

[Figure 1911]

[Figure 1912]

[Figure 1913]

r=8

r=8

“A [V] anime reading a book in a library wearing a headphone”

db = True db = False

[Figure 1914]

f=-1 f=2 f=4 f=8

f=-1 f=2 f=4 f=8

[Figure 1915]

[Figure 1916]

[Figure 1917]

[Figure 1918]

[Figure 1919]

[Figure 1920]

[Figure 1921]

[Figure 1922]

- r=1
- r=2

- r=1
- r=2

[Figure 1923]

[Figure 1924]

[Figure 1925]

[Figure 1926]

[Figure 1927]

[Figure 1928]

[Figure 1929]

[Figure 1930]

[Figure 1931]

[Figure 1932]

[Figure 1933]

[Figure 1934]

[Figure 1935]

[Figure 1936]

[Figure 1937]

[Figure 1938]

[Figure 1939]

“A [V] anime”

r=4

r=4

[Figure 1940]

[Figure 1941]

[Figure 1942]

[Figure 1943]

[Figure 1944]

[Figure 1945]

[Figure 1946]

[Figure 1947]

r=8

r=8

“A [V] anime as a chef, showing her culinary skills”

- Figure 29. Ablation study on factor and learning rate for LoKr using SDXL, with a fixed factor of -1. We found that the optimal learning rate and rank are 1 × 10−3 and 8, respectively. We also experimented with db=True, which indicates further low-rank decomposition of both matrices A and B, whereas db=False means only matrix B is decomposed further. (end)

lr = 5e-5 lr = 1e-4 lr = 5e-4 lr = 1e-3 lr = 5e-3

lr = 1e-5

[Figure 1948]

[Figure 1949]

[Figure 1950]

[Figure 1951]

[Figure 1952]

[Figure 1953]

Rank=4Rank=8Rank=4Rank=8Rank=4Rank=8Rank=4Rank=8Rank=4Rank=8Rank=4Rank=8

[Figure 1954]

[Figure 1955]

[Figure 1956]

[Figure 1957]

[Figure 1958]

[Figure 1959]

|[Figure 1960]|[Figure 1961]|
|---|---|
|[Figure 1962]|[Figure 1963]|

“A [V] person with Eiffel Tower in the background”

[Figure 1964]

[Figure 1965]

[Figure 1966]

[Figure 1967]

[Figure 1968]

[Figure 1969]

[Figure 1970]

[Figure 1971]

[Figure 1972]

[Figure 1973]

[Figure 1974]

[Figure 1975]

“A Monet-inspired painting of a [V] face standing near a blooming lily pond”

lr = 1e-5 lr = 5e-5 lr = 1e-4 lr = 5e-4 lr = 1e-3 lr = 5e-3

[Figure 1976]

[Figure 1977]

[Figure 1978]

[Figure 1979]

[Figure 1980]

[Figure 1981]

[Figure 1982]

[Figure 1983]

[Figure 1984]

[Figure 1985]

[Figure 1986]

[Figure 1987]

|[Figure 1988]|[Figure 1989]|
|---|---|
|[Figure 1990]|[Figure 1991]|

“A [V] toy in snow”

[Figure 1992]

[Figure 1993]

[Figure 1994]

[Figure 1995]

[Figure 1996]

[Figure 1997]

[Figure 1998]

[Figure 1999]

[Figure 2000]

[Figure 2001]

[Figure 2002]

[Figure 2003]

“A [V] toy in forest”

lr = 1e-5 lr = 5e-5 lr = 1e-4 lr = 5e-4 lr = 1e-3 lr = 5e-3

[Figure 2004]

[Figure 2005]

[Figure 2006]

[Figure 2007]

[Figure 2008]

[Figure 2009]

[Figure 2010]

[Figure 2011]

[Figure 2012]

[Figure 2013]

[Figure 2014]

[Figure 2015]

|[Figure 2016]|[Figure 2017]|
|---|---|
|[Figure 2018]|[Figure 2019]|

“A [V] anime reading a book in a library wearing a headphone”

[Figure 2020]

[Figure 2021]

[Figure 2022]

[Figure 2023]

[Figure 2024]

[Figure 2025]

[Figure 2026]

[Figure 2027]

[Figure 2028]

[Figure 2029]

[Figure 2030]

[Figure 2031]

- Figure 30. Ablation study on learning rate and rank for LoHA using SDXL. We found the optimal learning rate and rank to be 1 × 10−4 and 4, respectively.

### G. Comparison with state-of-the-arts

Qualitative Comparison. In this section, we extend upon 4.4 of the main paper, comparing DiffuseKronA with Stateof-the-art text-to-image personalization models including DreamBooth, LoRA-DreamBooth, SVDiff, Textual Invention, and Custom Diffusion.

- (1) Textual Inversion (Gal et al., 2022) is a fine-tuning method that optimizes a placeholder embedding to reconstruct the training set of subject images. Learning a new concept requires 3,000 steps, which takes around 30 minutes on an A100 GPU (Li et al., 2023a).
- (2) DreamBooth (Ruiz et al., 2023a) refines the entire network through additional preservation loss as a form of regularization, leading to enhancements in visual quality that exhibit promising results. Updating DreamBooth for a new concept typically requires about 6 minutes on an A100 GPU (Li et al., 2023a).
- (3) LoRA-DreamBooth (Ryu, 2023) explores low-rank adaptation for parameter-efficient fine-tuning attention-weight matrices of the text-to-image diffusion model. Fine-tuning LoRA-DreamBooth for a new concept typically takes about

5 minutes on a single 24GB NVIDIA RTX-3090 GPU.

- (4) SVDiff (Han et al., 2023) involves fine-tuning the singular values of the weight matrices, leading to a compact and efficient parameter space that reduces the risk of overfitting and language drifting. It took around 15 minutes on a single 24GB NVIDIA RTX-3090 GPU1.
- (5) Custom diffusion (Kumari et al., 2023) involves selective fine-tuning of weight matrices through a conditioning mechanism, enabling parameter-efficient refinement of diffusion models. This approach is further extended to encompass multi-concept fine-tuning. The fine-tuning time of Custom diffusion is around 6 minutes on 2 A100 GPUs.

Qualitative Comparison. DiffuseKronA consistently produces images closely aligned with the input images and consistently integrates features specified in the input text prompt. The enhanced fidelity and comprehensive comprehension of the input text prompts can be attributed to the structure-preserving capability and improved expressiveness facilitated by Kronecker product-based adaptation. The images generated by LoRA-DreamBooth are not of high quality and demand extensive experimentation for improvement, as depicted in Figure 31. As depicted in the figure, DiffuseKronA not only generates well-defined images but also has a better color distribution as compared to Custom Diffusion.

1SVDiff did not release official codebase, we used open-source code for SVDiff results in Figure 31.

### H. Practical Implications

- • Content Creation: It can be used to generate photorealistic content from text prompts.
- • Image Editing and In-painting: The model can be used to edit images or fill in missing parts of an image.
- • Super-Resolution: It can be used to enhance the resolution of images.
- • Video Synthesis: The model can be used to generate videos from text prompts.
- • 3D Assets Production: It can be used to create 3D assets from text prompts.
- • Personalized Generation: The model can be used in personalized generation with DreamBooth fine-tuning.
- • Resource Efficiency: The model is resource-efficient and can be trained with limited resources.
- • Model Compression: The model allows for architectural compression, reducing the number of parameters, MACs per sampling step, and latency.

|SVDiff|
|---|

|DiffuseKronA|
|---|

LoRA-DreamBooth DreamBooth Textual Inversion

Custom Diffusion

[Figure 2032]

[Figure 2033]

[Figure 2034]

[Figure 2035]

[Figure 2036]

[Figure 2037]

[Figure 2038]

[Figure 2039]

[Figure 2040]

[Figure 2041]

[Figure 2042]

[Figure 2043]

[Figure 2044]

[Figure 2045]

[Figure 2046]

[Figure 2047]

[Figure 2048]

[Figure 2049]

“A [V] dog in front of Times Square”

[Figure 2050]

[Figure 2051]

[Figure 2052]

[Figure 2053]

[Figure 2054]

[Figure 2055]

[Figure 2056]

[Figure 2057]

[Figure 2058]

[Figure 2059]

[Figure 2060]

[Figure 2061]

[Figure 2062]

[Figure 2063]

[Figure 2064]

[Figure 2065]

[Figure 2066]

[Figure 2067]

[Figure 2068]

[Figure 2069]

[Figure 2070]

“Painting of [V] dog at a beach by artist Claude Monet”

- (a)

Input Images

“A [V] dog”

- (b) “A [V] chair”

[Figure 2071]

[Figure 2072]

[Figure 2073]

- (c) “A [V] cat”

[Figure 2074]

[Figure 2075]

[Figure 2076]

- (d) “A [V] plushie”

[Figure 2077]

[Figure 2078]

[Figure 2079]

[Figure 2080]

[Figure 2081]

[Figure 2082]

[Figure 2083]

[Figure 2084]

[Figure 2085]

[Figure 2086]

[Figure 2087]

[Figure 2088]

[Figure 2089]

[Figure 2090]

[Figure 2091]

[Figure 2092]

[Figure 2093]

[Figure 2094]

“A [V] dog in construction outfit”

[Figure 2095]

[Figure 2096]

[Figure 2097]

[Figure 2098]

[Figure 2099]

[Figure 2100]

[Figure 2101]

[Figure 2102]

[Figure 2103]

[Figure 2104]

[Figure 2105]

[Figure 2106]

[Figure 2107]

[Figure 2108]

[Figure 2109]

[Figure 2110]

[Figure 2111]

[Figure 2112]

“A photo of Floor lamp on the side of [V] chair”

[Figure 2113]

[Figure 2114]

[Figure 2115]

[Figure 2116]

[Figure 2117]

[Figure 2118]

[Figure 2119]

[Figure 2120]

[Figure 2121]

[Figure 2122]

[Figure 2123]

[Figure 2124]

[Figure 2125]

[Figure 2126]

[Figure 2127]

[Figure 2128]

[Figure 2129]

[Figure 2130]

“A [𝑉 ] chair near a pool”

[Figure 2131]

[Figure 2132]

[Figure 2133]

[Figure 2134]

[Figure 2135]

[Figure 2136]

[Figure 2137]

[Figure 2138]

[Figure 2139]

[Figure 2140]

[Figure 2141]

[Figure 2142]

[Figure 2143]

[Figure 2144]

[Figure 2145]

[Figure 2146]

[Figure 2147]

[Figure 2148]

“A [V] cat at a beach with a view of seashore”

[Figure 2149]

[Figure 2150]

[Figure 2151]

[Figure 2152]

[Figure 2153]

[Figure 2154]

[Figure 2155]

[Figure 2156]

[Figure 2157]

[Figure 2158]

[Figure 2159]

[Figure 2160]

[Figure 2161]

[Figure 2162]

[Figure 2163]

[Figure 2164]

[Figure 2165]

[Figure 2166]

“A photo of [𝑉 ] cat”

[Figure 2167]

[Figure 2168]

[Figure 2169]

[Figure 2170]

[Figure 2171]

[Figure 2172]

[Figure 2173]

[Figure 2174]

[Figure 2175]

[Figure 2176]

[Figure 2177]

[Figure 2178]

[Figure 2179]

[Figure 2180]

[Figure 2181]

[Figure 2182]

[Figure 2183]

[Figure 2184]

[Figure 2185]

[Figure 2186]

“A backpack in the style of [V] tortoise plushy”

[Figure 2187]

[Figure 2188]

[Figure 2189]

[Figure 2190]

[Figure 2191]

[Figure 2192]

[Figure 2193]

[Figure 2194]

[Figure 2195]

[Figure 2196]

[Figure 2197]

[Figure 2198]

[Figure 2199]

[Figure 2200]

[Figure 2201]

[Figure 2202]

[Figure 2203]

[Figure 2204]

[Figure 2205]

“A photo of [V] tortoise plushy swimming in pool”

- Figure 31. Qualitative comparison between generated images by DiffuseKronA, LoRA-DreamBooth, Textual Inversion, DreamBooth,

and Custom Diffusion. Notably, our methods’ results are generated considering a2 = 8. We maintained the original settings of all these methods and used the SD CompVis-1.4 (CompVis, 2021) variant to ensure a fair comparison.

[Figure 2206]

[Figure 2207]

[Figure 2208]

| |[Figure 2209]<br><br>[Figure 2210]| |
|---|---|---|
| | | |
| |[Figure 2211]| |

(a) “A [V] dog”

[Figure 2212]

[Figure 2213]

| |[Figure 2214]<br><br>[Figure 2215]| |
|---|---|---|
| |[Figure 2216]| |

(b) “A [V] chair”

[Figure 2217]

[Figure 2218]

| |[Figure 2219]<br><br>[Figure 2220]| |
|---|---|---|
| | |[Figure 2221]|

(c) “A [V] cat”

[Figure 2222]

[Figure 2223]

[Figure 2224]

| |[Figure 2225]|[Figure 2226]|
|---|---|---|
| |[Figure 2227]| |

(d) “A [V] plushie”

- Figure 32. Quantitative comparison of DiffuseKronA with SOTA on Text-Image Alignment. The scores are computed from the same set of images and prompts as depicted in Figure 31.

