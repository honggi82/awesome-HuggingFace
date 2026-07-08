# arXiv:2404.15449v1[cs.CV]23Apr2024

## ID-Aligner: Enhancing Identity-Preserving Text-to-Image Generation with Reward Feedback Learning

Weifeng Chen∗

chenwf35@mail2.sysu.edu.cn Sun Yat-sen University

Jiachang Zhang∗

zhangjch58@mail2.sysu.edu.cn Sun Yat-sen University

Jie Wu†

wujie10558@gmail.com ByteDance Inc.

Hefeng Wu†

wuhefeng@gmail.com Sun Yat-sen University

Xuefeng Xiao

xiaoxuefeng.ailab@bytedance.com ByteDance Inc.

Liang Lin

linliang@ieee.org Sun Yat-sen University

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

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Figure 1: We present ID-Aligner, a general framework to boost the performance of identity-preserving text-to-image generation from the feedback learning perspective. We introduce an identity consistency reward and an identity aesthetic reward to enhance the identity preservation and the visual appeal of the generated characters. Our approach can apply to both the LoRA-based and Adapter-based methods, and exhibit superior performance compared with existing methods.

### ABSTRACT

key challenges remain: (1) It is hard to maintain the identity characteristics of reference portraits accurately, (2) The generated images lack aesthetic appeal especially while enforcing identity retention, and (3) There is a limitation that cannot be compatible with LoRAbased and Adapter-based methods simultaneously. To address these issues, we present ID-Aligner, a general feedback learning framework to enhance ID-T2I performance. To resolve identity features lost, we introduce identity consistency reward fine-tuning to utilize the feedback from face detection and recognition models to

The rapid development of diffusion models has triggered diverse applications. Identity-preserving text-to-image generation (ID-T2I) particularly has received significant attention due to its wide range of application scenarios like AI portrait and advertising. While existing ID-T2I methods have demonstrated impressive results, several

∗Both authors contributed equally. Work done during internship at ByteDance †Corresponding author.

improve generated identity preservation. Furthermore, we propose identity aesthetic reward fine-tuning leveraging rewards from human-annotated preference data and automatically constructed feedback on character structure generation to provide aesthetic tuning signals. Thanks to its universal feedback fine-tuning framework, our method can be readily applied to both LoRA and Adapter models, achieving consistent performance gains. Extensive experiments on SD1.5 and SDXL diffusion models validate the effectiveness of our approach. Project Page: https://idaligner.github.io/

### KEYWORDS

Text-to-Image Generation, Diffusion Model, Feedback Learning, Identity-Preserving Generation

### 1 INTRODUCTION

In recent years, the field of image synthesis has experienced a remarkable revolution with the emergence of diffusion models. These powerful generative diffusion models, exemplified by significant milestones such as DALLE-2 [23] and Imagen [26], have completely reshaped the landscape of text-to-image (T2I) generation. Moreover, the development of these models has also given rise to many related application tasks, such as image editing [13], controllable image generation [21, 42], and so on. Among these, the identitypreserving text-to-image generation received widespread attention due to its broad application scenarios like E-commerce advertising, AI portraits, image animation, and virtual try-it-on. It aims to generate new images about the identity of a specific person in the reference images with the guidance of the textual prompt. There are numerous advanced research works on this task. Early works resort to the low-rank (LoRA) [7] to adapt the pre-trained text-to-image diffusion model to the given a few reference portrait images and achieve recontextualization of the particular identity. Recently, IP-Adapter [39] achieved impressive personalized portrait generation by inserting an adapter model into the attention module of the diffusion model and fine-tuning using a high-quality large-scale dataset of facial images. However, despite these achievements, these methods still fall short in several aspects: (i) They cannot achieve accurate identity preservation. Existing methods typically employ a mean squared error (MSE) loss during training, which is unable to explicitly learn image generation that faithfully captures the characteristics of the reference portrait as shown in Fig.6. (ii) The generated image tends to lack appeal, especially when enforcing identity consistency. For example, the state-of-the-art method InstantID [32] introduces an extra IdentityNet to retain the information of reference portrait. While high fidelity, such a strict constraint is also prone to generating rigid images or characters with distorted/unnatural limbs and poses as depicted in Fig.4. (iii) Existing methods either rely on LoRA [7] or Adapter [39] to achieve ID-T2I generation and lack a general method that is compatible with these two paradigms.

In this work, drawing inspiration from the recent advancements in feedback learning within the diffusion model [1, 35, 37], we present ID-Aligner, a framework to boost the identity image generation performance with specially designed reward models via feedback learning. Specifically, we introduce an identity consistency reward tuning to boost identity preservation. It employs the face

detection model along with the face recognition model as the reward model to measure identity consistency and provide specialized feedback on identity consistency, which enables superior identity consistency during the recontextualization of the portrait in the reference images. In addition, to enhance the aesthetic quality of the identity generation, we further introduce an identity aesthetic reward tuning, which exploits a tailored reward model trained with human-annotated preference feedback data and automatically constructs character structure feedback data to steer the model toward the aesthetic appealing generation. Our method is very flexible and can be applied to not only the adapter-based model but also the LoRA-based model and achieve a consistent performance boost in both identity preservation and aesthetic quality. We also observe the significant acceleration effect with the LoRA-based model, facilitating its wide application. Extensive experiments demonstrate the superiority of our method upon the existing method, such as IP-Adapter [39], PhotoMaker [11], and InstantID [32]. Our contributions are summarized as follows:

- • We present ID-Aligner, a general feedback learning framework to improve the performance of identity-preserving text-to-image generation in both identity consistency and aesthetic quality. To the best of our knowledge, this is the first work to address this task through feedback learning.
- • We introduce a universal method that can be applied to both the LoRA-based model and the Adapter-based model. Theoretically, our approach can boost all the existing trainingbased identity-preserving text-to-image generation methods.
- • Extensive experiments have been conducted with various existing methods such as IP-Adapter, PhotoMaker, and InstanceID, validating the effectiveness of our method in improving identity consistency and aesthetic quality.

### 2 RELATED WORKS

Text-to-Image Diffusion Models. Recently, diffusion models [6, 30] have showcased remarkable capabilities in the realm of textto-image (T2I) generation. Groundbreaking works like Imagen [26], GLIDE [15], and DALL-E2 [23] have emerged, revolutionizing textdriven image synthesis and reshaping the landscape of the T2I task. Notably, the LDM [24] model, also known as the stable diffusion model, has transformed the diffusion process from pixel space to latent space, significantly improving the efficiency of training and inference. Building upon this innovation, the Stable Diffusion XL (SDXL) [20] model has further enhanced training strategies and achieved unprecedented image generation quality through parameter scaling. The development of these models has also triggered various applications, including image editing [2, 5, 13, 31], controllable image generation [9, 14, 42], etc.

Identity-Preserving Image Generation. ID-preserving image Generation [3, 11, 29, 32, 38, 39] has emerged as a significant application of text-to-image generation, capturing widespread attention due to its diverse range of application scenarios. The primary objective of this application is to generate novel images about a particular identity of one or several reference images guided by textual prompts. Unlike the conventional text-generated image task, it is crucial not only to ensure the performance of prompt

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Face Encoder

Face Encoder

𝓛𝒊𝒅_𝒔𝒊𝒎 + 𝓛𝒊𝒅_𝒂𝒆𝒔

Reference Face

[Figure 29]

[Figure 30]

[Figure 31]

❄

[Figure 32]

[Figure 33]

[Figure 34]

Aesthetic Reward

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

🔥 🔥 🔥 🔥

|Adapter|
|---|

|Adapter|
|---|

|Adapter|
|---|

|Adapter|
|---|

Adapter

Adapter

Adapter

Adapter

LDM

𝑥

[Figure 39]

[Figure 40]

Prompt: a man hold a cake

Face Detection

(b) ID-Aligner For Adapter Model

Reference Face

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Face Encoder

Face Detection

𝓛𝒊𝒅_𝒍𝒐𝒓𝒂 + 𝓛𝒊𝒅_𝒂𝒆𝒔 + 𝓛𝒊𝒅_𝒔𝒊𝒎

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

🔥 LoRA

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Face Encoder

[Figure 57]

[Figure 58]

Denoise Latents

Aesthetic

+ Reward

[Figure 59]

❄

Reference Images LDM Prompt: a woman read a book

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Face Detection

(a) ID-Aligner For LoRA Model

Figure 2: The overview of the proposed ID-Aligner. Our method exploits face detection and face encoder to achieve identity preservation via feedback learning. We further incorporated the aesthetic reward model to improve the visual appeal of the generation results. Our method is a general framework that can be applied to both LoRA and Adapter methods.

comprehension and generation fidelity but also to maintain consistency of the ID information between the newly generated images and the reference images. Few-shot methods [7, 12, 25, 29, 33] attempted to to finetune the diffusion model given several reference images to learn the ID features. However, this approach requires specific fine-tuning for each character, which limits its flexibility. PhotoMaker [11] achieves ID preservation by fine-tuning part of the Transformer layers in the image encoder and merging the class and image embeddings. IP-Adapter-FaceID [39] uses face ID embedding in the face recognition model instead of CLIP [22] image embedding to maintain ID consistency. Similarly, InstantID [32] uses a FaceEncoder to extract semantic Face Embedding and inject the ID information via Decoupled Cross-Attention. In addition, an IdentityNet is designed to introduce additional spatial control. In contrast to these approaches, our method relies on feedback learning, eliminating the need for intricate network structures. It offers exceptional versatility and effectiveness, seamlessly adapting to various existing methods while significantly enhancing ID Preservation.

Human Feedback for Diffusion Models. Inspired by the success of reinforcement learning with human feedback (RLHF) in the field of Large Language Models (LLMs) [16–18], researchers [1, 34, 35] have tried to introduce feedback-based learning into the field of text-to-image generation. Among these, DDPO [1] employs reinforcement learning to align the diffusion models with the supervision provided by the additional reward models. Different from DDPO, HPS [34, 35] exploits the reward model trained on the collected preference data to filter the preferred data and then achieve feedback learning via a supervised fine-tuning manner. Recently, ImageReward [37] proposes a ReFL framework to achieve preference fine-tuning, which performs reward scoring on denoised images within a predetermined diffusion model denoising step range through a pre-trained reward model, backpropagates and updates the diffusion model parameters. Recently, UniFL [40] proposes a unified framework to enhance diffusion models via feedback learning. Inspire by these, in this paper, we propose a reward feedback learning algorithm that focuses on optimizing ID-T2I models.

### 3 METHOD

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

We introduce ID-Aligner, a pioneering approach that utilizes the feedback learning method to enhance the performance of identity (ID) preserving generation. The outline of our method can be seen in Fig. 2. We resolve the ID-preserving generation via a reward feedback learning paradigm to enhance the consistency with a reference face image and aesthetic of generated images.

### 3.1 Text-to-Image Diffusion Model

Text-to-image diffusion models leverage diffusion modeling to generate high-quality images based on textual prompts via the diffusion model, which generates desired data samples from Gaussian noise through a gradual denoising process. During pre-training, a sampled image 𝑥 is first processed by a pre-trained VAE [4, 10] encoder to derive its latent representation 𝑧. Subsequently, random noise is injected into the latent representation through a forward diffusion process, following a predefined schedule {𝛽𝑡}𝑇 . This process can be formulated as 𝑧𝑡 = √𝛼𝑡𝑧 +

###### Worse Better/Preferred Unpreferred

Feedback on Structure ( Automate Constructed )

Feedback on Appeal ( Human Annotation )

Figure 3: The illustration of the aesthetic feedback data construction. We take an “AI + Expert” way to generate the feedback data. Left: The automatic data construction for the feedback data on the character structure generation. We resort to ControlNet [42] to manually generate the structure-distorted negative samples. Right: Human annoatated preference data over images.

√1 − 𝛼𝑡𝜖, where 𝜖 ∈ N(0, 1) is the random noise with identical dimension to 𝑧, 𝛼𝑡 = 𝑠 𝑡=1 𝛼𝑠 and 𝛼𝑡 = 1 − 𝛽𝑡. To achieve the denoising process, a UNet 𝜖𝜃 is trained to predict the added noise in the forward diffusion process, conditioned on the noised latent and the text prompt 𝑐. Formally, the optimization objective of the UNet is:

This loss function is based on comparison pairs between images, where each comparison pair contains two images (𝑥𝑖 and 𝑥𝑗) and prompt𝑐. 𝑓𝜃 (𝑥,𝑐) represents the reward score given an image 𝑥 and a prompt 𝑐. We therefore term 𝑓𝜃 as ℜ𝑎𝑝𝑝𝑒𝑎𝑙 for appealing reward.

L(𝜃) = E𝑧,𝜖,𝑐,𝑡 [||𝜖 − 𝜖𝜃 (√︁𝛼𝑡𝑧 + √︁1 − 𝛼𝑡𝜖,𝑐,𝑡)||22]. (1)

### 3.2 Identity Reward

In addition, we design a structure reward model that can distinguish distorted limbs/body from natural one. To train a model that can access the whether the structure of image is reasonable or not, we collect a set of text-image pairs containing positive and negative samples. Specifically, we use the images from LAION [28] filtered with human detector. We then use a pose estimation model to generate a pose, which can be treat as undistored human structure. We then randomly twiste the pose and utilize ControlNet [42] to generate distored body as negative samples, as is shown in left of Fig.3. Once the positive and negative pairs are available, similarly, we train the structure reward model with the same loss of Eq. 5 as well and term structure reward model as ℜ𝑠𝑡𝑟𝑢𝑐𝑡.

#### Identity Consistency Reward: Given the reference image 𝑥0ref

and the generated image 𝑥0′. Our objective is to assess the ID similarity of the particular portrait. To achieve this, we first employ the

face detection model FaceDet to locate the faces in both images. Based on the outputs of the face detection model, we crop the corresponding face regions and feed them into the encoder of a face recognition model FaceEnc. This allows us to obtain the encoded face embeddings for the reference face Eref and the generated face Egen, i.e.,

Eref = FaceEnc(FaceDet(𝑥0ref)), (2) Egen = FaceEnc(FaceDet(𝑥0′)). (3)

Then, the identity aesthetic reward model is defined as

Subsequently, we calculate the cosine similarity between these two face embeddings, which serves as the measure of ID retention during the generation process. We then consider this similarity as the reward signal for the feedback tuning process as follows:

ℜ𝑖𝑑_𝑎𝑒𝑠(𝑥,𝑐) = ℜ𝑎𝑝𝑝𝑒𝑎𝑙 (𝑥,𝑐) + ℜ𝑠𝑡𝑟𝑢𝑐𝑡 (𝑥,𝑐). (6)

### 3.3 ID-Preserving Feedback Learning

In the feedback learning phase, we begin with an input prompt 𝑐, initializing a latent variable𝑥𝑇 at random. The latent variable is then progressively denoised until reaching a randomly selected timestep 𝑡. At this point, the denoised image 𝑥0′ is directly predicted from 𝑥𝑡. The reward model obtained from the previous phase is applied to this denoised image, generating the expected preference score. This preference score enables the fine-tuning of the diffusion model to align more closely with our ID-Reward that reflects identity consistency and aesthetic preferences:

ℜ𝑖𝑑_𝑠𝑖𝑚(𝑥0′,𝑥0ref) = cose_sim(Egen, Eref). (4)

Identity Aesthetic Reward: In addition to the identity consistency reward, we introduce an identity aesthetic reward model focusing on appealing and quality. It consists of human preference of appeal and a reasonable structure.

First, we train a reward model with self-collected human annotation preference dataset that can score the image and reflect human preference over the appeal, as is shown in right of Fig.3. We employ the pretrained model provided by ImageReward [37] and finetune it with the following loss:

L𝑖𝑑_𝑠𝑖𝑚 = E𝑐∼𝑝(𝑐)E𝑥0′∼𝑝(𝑥0′|𝑐) [1 − ℜ𝑖𝑑_𝑠𝑖𝑚(𝑥0′,𝑥0𝑟𝑒𝑓 )], (7) L𝑖𝑑_𝑎𝑒𝑠 = E𝑐∼𝑝(𝑐)E𝑥0′∼𝑝(𝑥0′|𝑐) [−ℜ𝑖𝑑_𝑎𝑒𝑠(𝑥0′,𝑐)]. (8)

L𝜃 = −𝐸(𝑐,𝑥𝑖,𝑥𝑗)∼D[𝑙𝑜𝑔(𝜎(𝑓𝜃 (𝑥𝑖,𝑐) − 𝑓𝜃 (𝑥𝑗,𝑐)))]. (5)

Algorithm 1 ID-Preserving Reward Feedback Learning for Adapter model

- 1: Dataset: Identity preservation generation text-image dataset D = {(txt1, ref_face1), ...(txt𝑛, ref_face𝑛)}
- 2: Input: LDM with pre-trained adapter parameters 𝑤0, face detection model FaceDet, encoder of a face recognition model FaceEnc.
- 3: Initialization: The number of noise scheduler time steps 𝑇, add noise timestep𝑇𝑎, denoising time step 𝑡.
- 4: for data point (txt𝑖, ref_face𝑖) ∈ D do
- 5: 𝑥𝑇 ← RandNoise // Sample a Guassion noise.
- 6: 𝑡 ← Rand(𝑇1, 𝑇2) // Pick a random denoise time step 𝑡 ∈ [𝑇1,𝑇2]
- 7: for 𝑗 = 𝑇, ..., 𝑡 + 1 do
- 8: no grad: 𝑥𝑗−1 ← LDM𝑤𝑖 {𝑥𝑗 |(txt𝑖, ref_face𝑖)}
- 9: end for
- 10: with grad: 𝑥𝑡−1 ← LDM𝑤𝑖 {𝑥𝑡 |(txt𝑖, ref_face𝑖)}
- 11: 𝑥0′ ← 𝑥𝑡−1 // Predict the original latent by noise scheduler
- 12: img𝑖′ ← VaeDec(𝑥0′ ) // From latent to image
- 13: a𝑖′ ← FaceDet(img0′ ) // Detect the face area in the denoised image
- 14: emb𝑖′, emb𝑖 ← FaceEnc(a𝑖′), FaceEnc(ref_face𝑖) // Extract the embedding of generated face and reference face
- 15: L𝑖𝑑_𝑟𝑒𝑤𝑎𝑟𝑑 ← L𝑖𝑑_𝑠𝑖𝑚(emb𝑖′, emb𝑖) + L𝑖𝑑_𝑎𝑒𝑠(img𝑖′) // ID reward loss
- 16: 𝑤𝑖+1 ← 𝑤𝑖 // Update Adapter𝑤𝑖
- 17: end for

Finally, we use the weighted sum of these two reward objectives to fine-tune the diffusion for ID-preserving image generation:

L𝑖𝑑_𝑟𝑒𝑤𝑎𝑟𝑑 = 𝛼1L𝑖𝑑_𝑠𝑖𝑚 + 𝛼2L𝑖𝑑_𝑎𝑒𝑠, (9) where 𝛼1 and 𝛼2 are the balancing coefficients.

Our ID-Aligner is a universal method that can be applied to both the LoRA-based model and the Adapter-based model for IDpreserving generation, as described below in detail.

ID-Aligner For Adapter Model. IP-Adapter is pluggable model for diffusion model, which enable a face image as identity control. We optimize this model with reward feedback learning, as shown in Fig.2(a). We follow the same spirits of ReFL [37] to utilize a tailor reward model to provide a special feedback signal on identity consistency. Specifically, given a reference image of a particular portrait

and a textual control prompt, (𝑥0ref,𝑝), we first iteratively denoise a randomly initialized latent without gradient until a random time

step𝑇𝑑 ∈ [𝐷𝑇1,𝐷𝑇2], yielding 𝑥𝑇𝑑. Then, a further denoise step is executed with a gradient to obtain 𝑥𝑇𝑑−1 and directly get the predicted denoised image 𝑥0′ from 𝑥𝑇𝑑−1. Afterward, a reward model is utilized to score on 𝑥0′ and steer the model toward to the particular direction according to the reward model guidance. Here, we use weighted sum of similarity reward 𝐿𝑖𝑑_𝑠𝑖𝑚 and aesthetic reward 𝐿𝑖𝑑_𝑎𝑒𝑠 to fetch loss 𝐿𝑖𝑑_𝑟𝑒𝑤𝑎𝑟𝑑 in Eq. 9 to optimize the model. The complete process is summarized in Algorithm 1.

ID-Aligner For LoRA Model. LoRA is an efficient way to achieve identity-preserving generation. Given single or several reference images of the particular portrait, it quickly adapts the pre-trained LDM to the specified identity by solely fine-tuning some pluggable

Algorithm 2 ID-Preserving Reward Feedback Learning for LoRA model

- 1: Dataset: Several personalized text-image pairs dataset D = {(txt1, img1), ...(txt𝑛, img𝑛)}
- 2: Input: LDM with LoRA parameters 𝑤0, face detection model FaceDet, encoder of a face recognition model FaceEnc.
- 3: Initialization: The number of noise scheduler time steps 𝑇, add noise timestep𝑇𝑎, denoising time step 𝑡.
- 4: emb𝑟𝑒𝑓 ← Average(FaceEnc(FaceDet(img𝑖))),𝑖 ∈ D // extract ID embeddings of personalized images.
- 5: for data point (txt𝑖, img𝑖) ∈ D do
- 6: 𝑥𝑇 ← RandNoise // Sample a Guassion noise.
- 7: 𝑥𝑙 // Add noise into the latent 𝑥0 according to Eq.1 // Denoising
- 8: with grad: 𝑥𝑙−1 ← LDM𝑤𝑖 {𝑥𝑙 |(txt𝑖)} // ID-Reward Loop
- 9: 𝑡 ← Rand(𝑇1, 𝑇2) // Pick a random denoise time step 𝑡 ∈ [𝑇1,𝑇2]
- 10: for 𝑗 = 𝑇, ..., 𝑡 + 1 do
- 11: no grad: 𝑥𝑗−1 ← LDM𝑤𝑖 {𝑥𝑗 |(txt𝑖)}
- 12: end for
- 13: with grad: 𝑥𝑡−1 ← LDM𝑤𝑖 {𝑥𝑡 |(txt𝑖)}
- 14: 𝑥0′ ← 𝑥𝑡−1 // Predict the original latent by noise scheduler
- 15: img𝑖′ ← VaeDec(𝑥0′ ) // From latent to image
- 16: a𝑖′ ← FaceDet(img0′ ) // Detect the face area in the denoised image
- 17: emb𝑖′ ← FaceEnc(a𝑖′) // Extract the embedding of generated face
- 18: L𝑖𝑑_𝑟𝑒𝑤𝑎𝑟𝑑 ← L𝑖𝑑_𝑠𝑖𝑚(emb𝑖′, emb𝑟𝑒𝑓 ) + L𝑖𝑑_𝑎𝑒𝑠(img𝑖′) + L𝑚𝑠𝑒 ( 𝑥𝑙−1 , 𝑥0 ) // ID reward loss + denoising MSE loss
- 19: 𝑤𝑖+1 ← 𝑤𝑖 // Update LoRA𝑤𝑖
- 20: end for

extra low-rank parameters matrix of the network. However, fewshot learning in diffusion model to learn to generate a new person is highly depends on the provided dataset, which may require faces from different aspect or environment to avoid over-fitting. In this paper, we propose a more efficient way for ID LoRA training by applying the mentioned ID reward. As is shown in Fig.2(b), we train the LoRA with weighted sum of a denoising loss 𝐿𝑖𝑑_𝑙𝑜𝑟𝑎 in Eq.1 and ID-reward loss 𝐿𝑖𝑑_𝑟𝑒𝑤𝑎𝑟𝑑 in Eq.9. The 𝐿𝑖𝑑_𝑙𝑜𝑟𝑎 enables the model to learn the face structure while the 𝐿𝑖𝑑_𝑠𝑖𝑚 guide the model to learn identity information. The extra 𝐿𝑖𝑑_𝑎𝑒𝑠 is applied for improving the overall aesthetic of images. The complete process is summarized in Algorithm 2, which is slightly difference from the adapter one in terms of loss design.

### 4 EXPERIMENTS

Traning Dataset: We carefully curated a portrait dataset specifically for ID-preserving generation training. Specifically, we employed the MTCNN face detector [41] to filter the image from the LAION dataset [28]. This process finally resulted in over 200,000 images that contained faces. These images were used for both LoRA and Adapter training. For adapter-fashion training, we cropped the face from each image and used it as the reference identity.

FastComposer IP-Adapter Ours

IP-Adapter PhotoMaker InstantID Ours

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

"a * with a city in the background"

"a * reading a book"

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

"a * wearing a santa hat"

"a * walking the dog"

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

"a * in the jungle"

"a * holding a glass of wine" (a) SD15 (b) SDXL

#### Figure 4: Visual comparison of different Adapter-based identity conditional generation methods based on SD15 and SDXL.

To enhance the model’s generalization capability, we further collect a high-quality prompt set from JourneyDB [19] for identityconditioned generation. To make sure the prompt is compatible with the ID-preserving generation task, we manually filter the prompt containing the referring word about the identity summarized by chatGPT[16], such as ’the girl’, ’the man’, which finally yielded a final set of prompts describing human.

Training & Inference Details: For Adapter model, we take stable-diffusion-v1-5 [24] and SDXL[20] as the base text-toimagegenerationmodels, andtakethe widely recognized IP-Adapter [8] as the baseline model. We initialize the adapter weights with the pre-trained weight of IP-Adapter-faceid_plusv2 [8]. During training, only the adapter parameters are updated, ensuring compatibility with other models of the same structure. The model is trained using the 512x512 (1024x1024 for SDXL) resolution image with a batch size of 32. The learning rate is 10−6, and totally trained for 10,000 iterations. Following the practice of [37], the guidance scale is set to 1.0 during feedback learning. The 𝛼1 is set as 0.2 and 𝛼2 is set as 0.001. As for LoRA model, we collect 5 images for each identity. We use bucket adaptive resolution during LoRA training with a batch size of 1. The learning rate is 5 ∗ 10−5 for LoRA layers of UNet and 1 ∗ 10−4 for LoRA layers of Text encoder. The LoRA training is based on stable-diffusion-v1-5 [24] and SDXL [20] and totally trained for 2,000 iterations. For both LoRA and Adapter training, we exploit FaceNet [27] as the face detection model and MTCNN [41] face recognition model for the face embedding extraction. During inference, the DDIM scheduler [30] is employed, sampling 20 steps for generation. The guidance scale is set to 7.0, and the fusion strength of the adapter module is fixed at 1.0.

Evaluation settings: We evaluate the identity-preserving generation ability of our method with the prompt from the validation

set of FastComposer[36]. These prompts encompass four distinct types, namely action behavior, style words, clothing accessories, and environment. These diverse prompts facilitate a comprehensive evaluation of the model’s capacity to retain the identity throughout the generation process. For the LoRA model, we collect 5 images for 6 characters ranging from black, white, and yellow skin. Separate LoRA is trained for each character, and the performance is evaluated individually. In the case of the adapter model, we carefully gather an image collection of about 20 portraits from various sources including existing face datasets and the internet. This collection represents a rich spectrum of identities, spanning various genders such as men and women, ages encompassing both adults and children, and diverse skin tones including black, white, and yellow. These images serve as conditional reference images during the evaluation process. Following [11], we report the face similarity score (similarity between the generated face and the reference face), DINO score (similarity between the perceptual representation of the generated image and the reference image), CLIP-I (semantic similarity of generated image and reference images), and CLIP-T (semantic similarity of text prompt and the generated images) to evaluate the performance of our method.

### 4.1 Experimental results

4.1.1 Qualitative Comparison. We conduct qualitative experiment of ID-Aligner for IP-Adapter and Lora model.

Adapter Model: We compare our model’s performance with baseline methods and other state-of-the-art adapter-based models. As illustrated in Figure 4, we conduct experiments on both the SD15 and SDXL models. In Figure 4(a), which showcases the results for the SD15-based model, our method demonstrates superior identity

Table 1: Quantitative comparison between the state-of-the-art methods. The best results are highlighted in bold, while the second-best results results are underlined.

Architecture Model Face Sim.↑ DINO↑ CLIP-I↑ LAION-Aes↑ CLIP-T↑

FastComposer 0.486 0.498 0.616 5.44 24.0 IP-Adapter 0.739 0.586 0.684 5.54 22.0 Ours 0.800 0.606 0.727 5.59 20.6

SD1.5

IP-Adapter 0.512 0.460 0.541 5.85 24.5

InstantID 0.783 0.445 0.606 5.58 22.8 PhotoMaker 0.520 0.497 0.641 5.54 23.6

SDXL

Ours 0.619 0.499 0.602 5.88 23.7

IP-Adapter + ID-Cons

Lora

IP-Adapter + ID-Cons + ID-Aes

Lora + ID-Reward Lora Lora + ID-Reward

IP-Adapter

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

"a * baking cookies"

"a * wearing pink glasses"

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

"a * with a blue house in the background“

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

"a * in police outfit"

"a * in the snow"

#### Figure 5: Visual results of LoRA ID-Aligner methods based on SDXL.

"a * giving a lecture"

preservation and aesthetic quality. For instance, in the second example of "a * walking the dog," both FastComposer and IP-Adapter fail to generate a reasonable image of a baby, resulting in lower image quality. The third example highlights our method’s ability to better preserve the identity, aligning closely with the reference face. Regarding the SDXL-based model in 4(b), InstantID exhibits the best capability for identity-preserving generation. However, it has lower flexibility due to the face ControlNet, where the generated images heavily rely on the input control map. For example, in the case of "a * holding a glass of wine," InstantID only generates an avatar photo, while other methods can produce half-body images without the constraint of the face structure control map. We show competitive face similarity with it. Meanwhile, our method have better aesthetic that any other methods, the clarity and aesthetic appeal is better than other, for example, the color of the second case and the concrete structure of the third case.

Figure 6: The effectiveness ablation of the proposed identity consistency reward (ID-Cons) and the identity aesthetic reward (ID-Aes).

preservation consistency. Specifically, our approach achieves a Face Sim. score of 0.800, surpassing IP-Adapter’s 0.739 and FastComposer’s 0.486, suggesting better face identity preservation. Additionally, our higher DINO (0.606) and CLIP-I (0.727) scores demonstrate improved overall subject consistency. Our method also yields the highest LAION-Aesthetics (LAION-Aes) score of 5.59, indicating enhanced aesthetic quality compared to the baselines. Regarding the SDXL model, InstantID exhibits the highest Face Sim. score of 0.783, outperforming our method (0.619) and the other baselines in terms of face identity preservation. However, our approach achieves competitive performance on the DINO (0.499) and CLIP-I (0.602) metrics, suggesting comparable overall identity consistency. Notably, our method obtains the highest LAION-Aes score of 5.88 among all SDXL-based techniques, demonstrating its ability to generate aesthetically pleasing images while maintaining identity consistency. We also note that there is a slight performance drop in the semantic alignment between the prompt and the generated image after the optimization. This is because the model is forced to focus on identity adaptation, and will inevitably overlook the textual prompt to some extent. This phenomenon is also observed in lots of existing identity preservation generation works [11, 25, 32].

LoRA Model: Fig.5 showcase the results of incorporating our method into the LoRA model. It is evident that our method significantly boosts identity consistency (the male character case) and visual appeal (the female character case) compared with the naive LoRA method.

4.1.2 Quantitative Comparison. Tab.1 presents a quantitative comparison of our proposed method with several state-of-the-art techniques for identity-preserving image generation, evaluated across various metrics. The methods are categorized based on the underlying architecture, with results reported separately for the SD1.5 and SDXL models. For the SD1.5 model, our method outperforms FastComposer and IP-Adapter in terms of Face Similarity (Face Sim.), DINO, and CLIP-I scores, indicating superior identity

4.1.3 Ablation Study. We conduct ablation study to analyze the effectiveness of each component in our method.

#### Table 2: Generalization study of our method on different base T2I models: Dreamshaper (SD1.5) and RealVisXL (SDXL).

Model Face Sim.↑ DINO↑ CLIP-I↑ IP-Adapter-Dreamshaper 0.598 0.583 0.591 IP-Adapter-Dreamshaper + ID-Reward 0.662 (+10.7%) 0.588 (+0.8%) 0.616 (+4.2%) IP-Adapter-RealVisXL 0.519 0.488 0.575 IP-Adapter-RealVisXL + ID-Reward 0.635 (+22.3%) 0.509 (+4.3%) 0.623 (+8.3%)

[Figure 125]

#### Figure 8: User preferences on Text fidelity, Image quality, Face Similarity for different methods. We visualize the proportion of total votes that each method has received.

#### Figure 7: The illustration of the accelerated identity adaptation for the LoRA model. Left: LoRA trained based on SD1.5. Right: LoRA trained based on SDXL.

Generalization Study: To demonstrate the generalization ability of our approach, We utilized the widely recognized Dreamshaper1 and RealVisXL2 for the open-sourced alternatives of SD15 and SDXL, and validate our method with these text-to-image models. According to the results of Tab.2, our method delivers a consistent performance boost on these alternative base models. Specifically, our method brings 10.7% and 4.2% performance boosts with Dreamshaper in terms of face similarity and image similarity measured by CLIP, which means better identity preservation. Moreover, our method obtained more significant performance improvement with the more powerful model in SDXL architecture. For instance, our method surpasses the original RealVisXL model with 22.3% in identity preservation, and 8.3% improvements in CLIP-I. This demonstrates the superior generalization ability of our method on different text-to-image models.

Identity Reward: We conduct an ablation study to evaluate the impact of the proposed identity consistency reward and aesthetic reward. As illustrated in Fig.6, applying the identity consistency reward boosts the identity similarity significantly. For example, both the two cases generated by the baseline model encounters severe identity loss with a notably different appearance from the reference portrait. However, after optimizing the model using the identity consistency reward, the generated character exhibits a more similar outlook. This improvement can be attributed to the specialized identity reward provided by the face detection and face embedding extraction model, which guides the model to preserve the desired identity features as much as possible. Furthermore, the incorporation of the identity aesthetic reward further enhances the visual appeal of the generated image, particularly in improving the structural aspects of the characters. For example, in the first row, despite the preservation of identity features achieved through the identity consistency reward, the generated hands of the character still exhibit distortion. However, this issue is effectively resolved by the identity aesthetic reward, which benefits from tailor-curated feedback data. These ablation results underscore the crucial role played by our proposed identity consistency and aesthetic rewards in achieving high-quality, identity-preserving image generation.

4.1.4 User Study. To gain a comprehensive understanding, we conducted a user study to compare our method with IP-adapterplusv2 [39], PhotoMaker [11], and InstantID [32]. We presented 50 generated text-image pairs and a reference face image to each user. For each set, users were asked to vote for the best one or two choices among the four methods, based on three criteria: (1) textfidelity - which image best matches the given prompt, (2) Image Quality - which image looks the most visually appealing, and (3) Face similarity - which image most closely resembles the reference face. Users could choose two options if it was difficult to select a clear winner. We collected a total of 500 votes from 10 users.

Fast Identity Adaptation: The LoRA approach is a test-time finetuning method and needs to train a separate LoRA model for each portrait. This poses a significant challenge for the application as it requires enough training time to ensure adequate identity adaptation. Thanks to the targeted feedback on identity consistency, we found our method can accelerate the identity adaptation of LoRA training significantly as demonstrated in Fig.7. This effect is particularly prominent when adapting to the SDXL, as conventional LoRA adaptation for SDXL is inherently slow due to the larger number of parameters required to update. In contrast, the id-aligner considerably reduces the fine-tuning time to achieve the same level of face similarity.

As shown in Fig. 8, the results align with the quantitative study in Fig. 4. InstantID achieved the highest face similarity score, while our method secured the second-best face similarity score. Our method obtained the highest aesthetic score and the second-highest textimage consistency score. Overall, our method performed well across all indicators and exhibited a relatively balanced performance compared to other methods.

- 1https://huggingface.co/Lykon/DreamShaper
- 2https://huggingface.co/SG161222/RealVisXL_V3.0

### 5 CONCLUSION

In this paper, we introduced ID-Aligner, an algorithm crafted to optimize image generation models for identity fidelity and aesthetics through reward feedback learning. We introduces two key rewards: identity consistency reward and identity aesthetic reward, which can be seamlessly integrated with adapter-based and LoRA-based text-to-image models, consistently improving identity consistency and producing aesthetically pleasing results. Experimental results validate the effectiveness of ID-Aligner, demonstrating its superior performance.

### REFERENCES

- [1] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. 2023. Training Diffusion Models with Reinforcement Learning. arXiv:2305.13301 [cs.LG]
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2022. Instructpix2pix: Learning to follow image editing instructions. arXiv preprint arXiv:2211.09800

(2022).

- [3] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, and Min Zheng. [n.d.]. PhotoVerse: Tuning-Free Image Customization with Text-to-Image Diffusion Models. ([n.d.]).
- [4] Patrick Esser, Robin Rombach, and Bjorn Ommer. 2021. Taming Transformers for High-Resolution Image Synthesis. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr46437.2021. 01268
- [5] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022).
- [6] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. arXiv:2006.11239 [cs.LG]
- [7] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685 [cs.CL]
- [8] Ye Hu, Jun Zhang, Sibo Liu, Xianpei Han, and Wei Yang. 2023. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. (Aug 2023).
- [9] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou.

2023. Composer: Creative and Controllable Image Synthesis with Composable Conditions. (Feb 2023).

- [10] Diederik P Kingma and Max Welling. 2022. Auto-Encoding Variational Bayes. arXiv:1312.6114 [stat.ML]
- [11] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. 2023. PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding. (Dec 2023).
- [12] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. 2023. Subject-Diffusion:Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning. (Jul 2023).
- [13] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2022. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. arXiv:2108.01073 [cs.CV]
- [14] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. 2023. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453 (2023).
- [15] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2022. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. arXiv:2112.10741 [cs.CV]
- [16] OpenAI. 2023. Introducing chatgpt. arXiv:2303.08774 [cs.CL]
- [17] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35 (2022), 27730–27744.
- [18] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. arXiv:2203.02155 [cs.CL]
- [19] Junting Pan, Keqiang Sun, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, and Hongsheng Li. 2023. JourneyDB: A Benchmark for Generative Image Understanding. arXiv:2307.00716 [cs.CV]
- [20] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. [n.d.]. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. ([n.d.]).
- [21] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, Stefano Ermon, Yun Fu, and Ran Xu. 2023. UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild. arXiv:2305.11147 [cs.CV]
- [22] Alec Radford, JongWook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Askell Amanda, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. Cornell University - arXiv,Cornell University arXiv (Feb 2021).
- [23] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen.

2022. Hierarchical Text-Conditional Image Generation with CLIP Latents. arXiv:2204.06125 [cs.CV]

- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Diffusion Models.

- arXiv:2112.10752 [cs.CV]
- [25] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2022. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.
- [26] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. arXiv:2205.11487 [cs.CV]
- [27] Florian Schroff, Dmitry Kalenichenko, and James Philbin. 2015. FaceNet: A unified embedding for face recognition and clustering. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr. 2015.7298682
- [28] Christoph Schuhmann, §§°°romain Beaumont, Vencu Vencu, Ade Gordon, Wightman Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, °°jenia Jitsev, UC Berkeley, and Gentec Data. [n. d.]. LAION5B: An open large-scale dataset for training next generation image-text models. ([n.d.]).
- [29] Jing Shi, Wei Xiong, Zhe Lin, and HyunJoon Jung. 2023. InstantBooth: Personalized Text-to-Image Generation without Test-Time Finetuning. (Apr 2023).
- [30] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2022. Denoising Diffusion Implicit Models. arXiv:2010.02502 [cs.LG]
- [31] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. 2022. Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation. arXiv preprint arXiv:2211.12572 (2022).
- [32] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. 2024. InstantID: Zero-shot Identity-Preserving Generation in Seconds. arXiv:2401.07519 [cs.CV]
- [33] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo.

2023. ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation. (Feb 2023).

- [34] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv:2306.09341 [cs.CV]
- [35] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023. Human Preference Score: Better Aligning Text-to-Image Models with Human Preference. arXiv:2303.14420 [cs.CV]
- [36] Guangxuan Xiao, Tianwei Yin, WilliamT Freeman, Frédo Durand, and Song Han. [n.d.]. FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention. ([n.d.]).
- [37] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2023. ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. arXiv:2304.05977 [cs.CV]
- [38] Yuxuan Yan, Chi Zhang, Rui Wang, Pei Cheng, Gang Yu, and Bin Fu. 2023. FaceStudio: Put Your Face Everywhere in Seconds. (Dec 2023).
- [39] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. arXiv:2308.06721 [cs.CV]
- [40] Jiacheng Zhang, Jie Wu, Yuxi Ren, Xin Xia, Huafeng Kuang, Pan Xie, Jiashi Li, Xuefeng Xiao, Weilin Huang, Min Zheng, Lean Fu, and Guanbin Li. [n.d.]. UniFL: Improve Stable Diffusion via Unified Feedback Learning. ([n.d.]).
- [41] Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, and Yu Qiao. 2016. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks. IEEE Signal Processing Letters (Oct 2016), 1499–1503. https://doi.org/10.1109/lsp. 2016.2603342
- [42] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding Conditional Control to Text-to-Image Diffusion Models. arXiv:2302.05543 [cs.CV]

### A MORE RESULTS

- A.1 Generalization

Our evaluation strategy focuses on the crucial aspect of generalization, ensuring that our model’s capabilities extend beyond specific scenarios. To achieve this, we leverage a diverse array of prompts categorized into four distinct groups. These prompts provide a comprehensive assessment of the model’s proficiency in preserving identity across varied contexts. The specifics of our evaluation prompts are meticulously documented in Table 4, offering transparency and reproducibility in our methodology. In Figure 9, we visually showcase the outputs generated by our model across these varied prompt categories. From depicting individuals in different environmental contexts ("context" prompts) to showcasing diverse attire choices ("accessory" prompts), engaging in various activities ("action" prompts), and adopting different stylistic elements ("style" prompts), our model consistently demonstrates its versatility and adaptability. This comprehensive exploration underscores the robustness of our approach in handling a wide spectrum of identity-preserving generation tasks. By effectively navigating through diverse prompts, our model exemplifies its ability to capture the essence of individual identity across a multitude of scenarios, thereby showcasing its potential for real-world applications and further advancing the field of generative models.

- A.2 Identity Mixing

We further explore the application of our method in identity mixing. Specifically, given 2 different identity reference images, we aim to generate the synthesized image that contains the characteristics of both identities. This poses a higher demand for the model’s identitypreserving ability. We show some results of our method in identity mixing in Fig. 10. We provide a white lady mixing with another black lady, a white lady mixing with another yellow man, black man mixing with another yellow man to show the generalization of our model.

- A.3 More Comparison with Other Methods

We present additional visualization results of both other methods and our own in Figure 11. It is evident that our method outperforms in terms of identity-preserving generation, exhibiting superior performance in both identity consistency and visual appeal. It is worth noting that the InstantID method [32] achieves better identity preservation by incorporating an additional IdentityNet. This approach is similar to ControlNet [42] and imposes a strict condition on the original reference. However, this constraint often leads to the generation of rigid images that closely follow the exact pose of the reference image. As a result, some unreasonable results are produced, as exemplified by the case of "a {class_token} cooking a meal". In contrast, our method consistently generates more visually pleasing results, which have a better trade-off in terms of text consistency and identity consistency.

### B LIMITATIONS

(1) Our algorithm is designed to enhance existing models through reward feedback learning. However, if the existing model is already robust, the improvement may be marginal.

- (2) Enhancing face similarity may occasionally compromise prompt consistency, as the emphasis on facial features might lead to undesirable outcomes. This issue can be mitigated by reducing the intensity of identity control.
- (3) Biases inherent in T2I models and their training data can impact results. Certain identities may yield more favorable outcomes, while others may produce dissimilar results.

### C BROADER IMPACT

The impact of our research extends across multiple dimensions. Academically, our method serves as a foundational framework for integrating diffusion models with other expert models, such as face recognition models. This integration contributes significantly to the advancement of generative models. Practically, our technique holds immense transformative potential across a wide spectrum of industries, including entertainment, portraiture, advertising, and beyond. By providing a means to generate high-quality human images with fidelity, our approach offers unprecedented opportunities for creativity and innovation. Nevertheless, it is essential to recognize and address the ethical considerations inherent in the widespread adoption of such technology. The capacity to produce lifelike human images raises legitimate concerns regarding privacy, potential misuse, and the dissemination of false information. Thus, we underscore the critical importance of developing and adhering to stringent ethical guidelines to ensure the responsible and ethical utilization of this groundbreaking technology.

Table 3: The selected open-sourced text-to-image models in the Civitai community.

Model URL

Realistic Vision v60-b1 https://civitai.com/models/4201/realistic-vision-v60-b1

DreamShaper https://civitai.com/models/4384/dreamshaper GhostMix https://civitai.com/models/36520/ghostmix ToonYou https://civitai.com/models/30240/toonyou

Disney Pixar Cartoon TypeB https://civitai.com/models/75650/disney-pixar-cartoon-typeb Disney Stylev1 https://civitai.com/models/114413/disney-stylev1

Category Prompt

Category Prompt

|Accessory<br><br>|"a {class_token} wearing a red hat" "a {class_token} wearing a Santa hat" "a {class_token} wearing a rainbow scarf" "a {class_token} wearing a black top hat and a monocle" "a {class_token} in a chef outfit" "a {class_token} in a firefighter outfit" "a {class_token} in a police outfit" "a {class_token} wearing pink glasses" "a {class_token} wearing a yellow shirt" "a {class_token} in a purple wizard outfit"|
|---|---|
|Style|"a painting of a {class_token} in the style of Banksy" "a painting of a {class_token} in the style of Vincent Van Gogh" "a colorful graffiti painting of a {class_token}" "a watercolor painting of a {class_token}" "a Greek marble sculpture of a {class_token}" "a street art mural of a {class_token}" "a black and white photograph of a {class_token}" "a pointillism painting of a {class_token}" "a Japanese woodblock print of a {class_token}" "a street art stencil of a {class_token}"<br><br>|

|Cotenxt<br><br>|"a {class_token} in the jungle" "a {class_token} in the snow" "a {class_token} on the beach" "a {class_token} on a cobblestone street" "a {class_token} on top of pink fabric" "a {class_token} on top of a wooden floor" "a {class_token} with a city in the background" "a {class_token} with a mountain in the background" "a {class_token} with a blue house in the background" "a {class_token} on top of a purple rug in a forest"|
|---|---|
|Action|"a {class_token} riding a horse" "a {class_token} holding a glass of wine" "a {class_token} holding a piece of cake" "a {class_token} giving a lecture" "a {class_token} reading a book" "a {class_token} gardening in the backyard" "a {class_token} cooking a meal" "a {class_token} working out at the gym" "a {class_token} walking the dog" "a {class_token} baking cookies"<br><br>|

#### Table 4: The prompt used in the evaluation procedure of the adapter model. Following [36], we exploit diverse prompts of four distinct categories to evaluate the performance of our method in identity-preserving generation comprehensively.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Context

[Figure 132]

[Figure 133]

Style

[Figure 134]

[Figure 135]

Action

[Figure 136]

[Figure 137]

Accessory

#### Figure 9: Visualizations of generation results obtained by using prompt in Tab.4. It shows prompts of different types including "context", "style", "action" and "accesory".

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

50%

#### Figure 10: Visualizations of generation results obtained by incorporating our optimized IP-Adapter by mixing two people as image condition. The prompt is "a person in snowy day, closeup". The mixing rate range from 0 to 1 with 0.1 as interval.

InstantID PhotoMaker IP-Adapter Ours

[Figure 149]

a {class_token*} cooking a meal"

[Figure 150]

a {class_token*} baking cookies

[Figure 151]

a {class_token*} in the jungle

[Figure 152]

a {class_token*} wearing pink glasses

[Figure 153]

a watercolor painting of a {class_token*}

[Figure 154]

a Japanese woodblock print of a {class_token*}

#### Figure 11: Visualizations of generation results obtained by incorporating our optimized IP-Adapter into more open-source text-to-image base models in Civitai.

