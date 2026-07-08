## DreamO: A Unified Framework for Image Customization

# arXiv:2504.16915v5[cs.CV]27Nov2025

CHONG MOU, Intelligent Creation Team, ByteDance; Peking University, China YANZE WU∗†, Intelligent Creation Team, ByteDance, China WENXU WU, Intelligent Creation Team, ByteDance, China ZINAN GUO, Intelligent Creation Team, ByteDance, China PENGZE ZHANG, Intelligent Creation Team, ByteDance, China YUFENG CHENG, Intelligent Creation Team, ByteDance, China YIMING LUO, Intelligent Creation Team, ByteDance, China FEI DING†, Intelligent Creation Team, ByteDance, China SHIWEN ZHANG, Intelligent Creation Team, ByteDance, China XINGHUI LI, Intelligent Creation Team, ByteDance, China MENGTIAN LI, Intelligent Creation Team, ByteDance, China MINGCONG LIU, Intelligent Creation Team, ByteDance, China YUNSHENG JIANG, Intelligent Creation Team, ByteDance, China SHAOJIN WU, Intelligent Creation Team, ByteDance, China SONGTAO ZHAO, Intelligent Creation Team, ByteDance, China JIAN ZHANG, Peking University, China QIAN HE, Intelligent Creation Team, ByteDance, China XINGLONG WU, Intelligent Creation Team, ByteDance, China

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

“A boy wearing shorts and a shirt in the shallow stream” “a robot is fighting with the monster , in the city , cinematic feel”

“A toy holding a sign saying "DreamO", on the mountain” “The dog is eating the berry bowl”

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

“Two women make delicious food in the kitchen. The woman on the left is from [ref#2]. The woman on the right is from [ref#1].”

“Generate a same style image. A castle” “Generate a same style image. A dog play in the park” “The woman held the toy above her head in the park”

[Figure 24]

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

“Portrait, Chibi”

“A girl rides a giant dog, walking in the noisy modern city”

“A man hold a box in the snowing”

“a man wear a sunglasses and wear a yellow hat in the forest”

Fig. 1. The image customization capability of our proposed DreamO.

∗Correspondence: Yanze Wu <wuyanze123@gmail.com> †Project leads

author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the

SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2137-3/2025/12 https://doi.org/10.1145/3757377.3763956

Recently, extensive research on image customization (e.g., identity, subject, style, background, etc.) demonstrates strong customization capabilities in large-scale generative models. However, most approaches are designed for specific tasks, restricting their generalizability to combine different types of condition. Developing a unified framework for image customization remains an open challenge. In this paper, we present DreamO, an image customization framework designed to support a wide range of tasks while facilitating seamless integration of multiple conditions. Specifically, DreamO utilizes a diffusion transformer (DiT) framework to uniformly process input of different types. During training, we introduce a feature routing constraint to facilitate the precise querying of relevant information from reference images. Additionally, we design a placeholder strategy that associates specific placeholders with conditions at particular positions, enabling control over the placement of conditions in the generated results. Moreover, we employ a progressive training strategy to ensure smooth model convergence and correct the generation quality of the final output. Extensive experiments demonstrate that the proposed DreamO can effectively perform various image customization tasks with high quality and flexibly integrate different types of control conditions. Project page: https://mc-e.github.io/project/DreamO

CCS Concepts: • Computing methodologies → Computer vision. Additional Key Words and Phrases: Diffusion-models, Image customization ACM Reference Format:

Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, Mengtian Li, Mingcong Liu, Yunsheng Jiang, Shaojin Wu, Songtao Zhao, Jian Zhang, Qian He, and Xinglong Wu. 2025. DreamO: A Unified Framework for Image Customization. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25), December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3757377.3763956

1 INTRODUCTION

Due to the high-quality image generation and stable performance of diffusion models [Ho et al. 2020], substantial research efforts focus on controllable generation by leveraging their generative priors [Mou et al. 2024; Zhang et al. 2023]. Among these, image customization aims to ensure that generated outputs remain consistent with a reference image in specific attributes, such as identity [Guo et al. 2024; Wang et al. 2024a; Xiao et al. 2024b], object appearance [He et al. 2025; Huang et al. 2024a; Ruiz et al. 2023], virtual try-on [Choi et al. 2024; Luan et al. 2025; Wan et al. 2025], and style [Qi et al. 2024; Wu et al. 2021; Xing et al. 2024]. Despite the abundance of task-specific approaches, developing a unified framework for image customization remains a challenge.

Early research Composer [Huang et al. 2023] jointly trains a diffusion model with multi-condition input, e.g., depth, color, sketch. Some methods [Qin et al. 2023; Zhao et al. 2023] train additional control blocks [Mou et al. 2024; Zhang et al. 2023] to support general spatial control on the generation result, which greatly saves training costs. However, their control ability is restricted to some simple spatial conditions, and the interactions between conditions are rigid and have control redundancy. Recently, the DiT [Peebles and Xie 2023] framework greatly scales up the performance of diffusion models. Based on DiT, OminiControl [Tan et al. 2024] proposes to integrate image conditions by unified sequence with diffusion latent. It can perform various customization tasks, e.g., identity, color, and layout. Despite its advantages, OminiControl is trained separately on different tasks, struggling to process multiple

conditions. Recently, OmniGen [Xiao et al. 2024a] trains a general generation control based on a pre-trained large language model [Abdin et al. 2024] (LLMs). UniReal [Chen et al. 2024b] achieves this through video generation pretraining followed by full-model posttraining. However, we argue that high-quality, multi-concept image customization cannot be achieved by merely leveraging the general capabilities of large language models like OmniGen or video models such as UniReal. Instead, it requires specialized architectural designs. Currently, the research community lacks an efficient and effective method specifically tailored for image customization under multi-concept and multi-conditional scenarios.

In this paper, we design a unified image customization approach based on a pre-trained DiT. With a low training cost and a single model, our method can support various conditions (e.g., identity, subject, try-on, and style) and enables interactions among different kinds of condition, as shown in 1. Specifically, we follow the unified sequence conditioning format introduced in OminiControl [Tan et al. 2024], and introduce a routing constraint on the internal representations of DiT during training. This ensures content fidelity and promotes the disentanglement of different control conditions. A placeholder strategy is also designed to enable control over the placement of conditions in the generated results. In addition, we construct large-scale training data covering multiple tasks and design a progressive training strategy. This enables the model to progressively acquire robust and generalized image customization capabilities.

In summery, this paper has the following contributions:

- • We propose DreamO, a unified framework for image customization. It achieves various complex and multi-condition customization tasks by training a small set of additional parameters on a pre-trained DiT model.
- • Based on representation correspondences within the diffusion model, we design a feature routing constraint to enhance consistency fidelity and enable effective decoupling in multicondition scenarios.
- • We introduce a progressive training strategy to facilitate convergence in multi-task and complex task settings. Moreover, we design a placeholder strategy to establish correspondence between textual descriptions and condition images.
- • Extensive experiments demonstrate that our method not only produces high-quality results in various image customization tasks, but also exhibits strong flexibility in multi-condition scenarios.

2 RELATED WORKS 2.1 Diffusion Models

As a powerful generation paradigm, the diffusion model [Dhariwal and Nichol 2021; Ho et al. 2020] rapidly dominates the image generation community. Its high generation quality and stable performance have been successfully applied to various tasks , e.g., text-to-image generation [Nichol et al. 2022; Ramesh et al. 2022; Rombach et al. 2022; Saharia et al. 2022a] and image editing [Hertz et al. 2022; Meng et al. 2021; Mou et al. 2023; Yang et al. 2023]. Some strategies, such as latent diffusion [Rombach et al. 2022] and flow matching sampling [Lipman et al. 2022], are proposed to enhance the performance. Most early works [Nichol et al. 2022; Rombach et al. 2022]

[Figure 36]

[Figure 37]

“In the forest, a man wears a hat and a sunglasses”

[Figure 38]

: Position Embedding

PE

Text Encoder

: Condition Embedding

CE

VAE

... ...

... ... ...

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

PE PE CE PE Text Input Layer Condition Input Layer Image Input Layer

[Figure 48]

[Figure 49]

[Figure 50]

DiT

LoRA

Fig. 2. Overview of our proposed DreamO, which can uniformly handle commonly used consistency-aware generation control.

utilize the UNet architecture as the diffusion model. Recently, the diffusion transformer [Peebles and Xie 2023] (DiT) architecture has emerged as a superior choice, offering improved performance through straightforward scalability.

influence region of condition features for identity-specific generation in multi-face scenarios, and AnyStory [He et al. 2025] further extends this approach to subject-driven generation. Some recent works [Yamaguchi and Yanai 2024] show that the cross-attention map in the DiT framework also exhibits spatial properties. In this paper, we explore routing constraints in the DiT framework.

- 2.2 Controllable Image Generation

Advancements in diffusion models drive the rapid development of controllable image generation. In this community, text is the most fundamental conditioning input, e.g., Stable Diffusion [Rombach et al. 2022], Imagen [Saharia et al. 2022b], and DALL-E2 [Ramesh et al. 2022]. To achieve accurate spatial control, some methods, e.g., ControlNet [Zhang et al. 2023] and T2I-Adapter [Mou et al. 2024], propose adding control modules on pre-trained diffusion models. UniControl [Qin et al. 2023; Zhao et al. 2023] propose to unify different spatial conditions with a joint condition input. In addition to spatial control, some unspatial conditions are also studied. The IP-Adapter [Ye et al. 2023] utilizes cross-attention to inject image prompt in the diffusion model to control some unspatial properties, e.g., identity and style. In addition to these representative works, other related attempts [Gal et al. 2022; He et al. 2025; Hua et al. 2023; Kumari et al. 2023; Li et al. 2023; Ma et al. 2024a,b] also help to broaden the scope of controllable image generation.

Recent advancements in DiT-based diffusion models further promote the development of controllable image generation. For instance, in-context LoRA [Huang et al. 2024b] and OminiControl [Tan et al. 2024] introduce a novel approach by concatenating all input tokens (i.e., text, image, and conditions) and training LoRA with task-specific datasets for various applications. Subsequently, OmniGen [Xiao et al. 2024a] and UniReal [Chen et al. 2024b] optimize the entire diffusion model in multiple stages on larger-scale training data, achieving improved understanding of input conditions.

- 2.3 Cross-attention Routing in Diffusion Models

3 METHOD

- 3.1 Preliminaries

The Diffusion Transformer (DiT) model [Peebles and Xie 2023] employs a transformer as the denoising network to refine diffusion latent. Specifically, in the input, the 2D image latent z𝑡 ∈ R𝑐×𝑤×ℎ is patchified into a sequence of 1D tokens z𝑡 ∈ R𝑐×(

𝑤 𝑝 ×ℎ𝑝 ), where (𝑤,ℎ) is spatial size, 𝑐 is the number of channels, and 𝑝 is the patch size. The image and text tokens are concatenated and processed by the DiT model in a unified manner. Apart from model architectures, more efficient sampling strategies (e.g., Flow Matching [Lipman et al. 2022]) are also proposed. Unlike DDPM [Ho et al. 2020], Flow Matching conducts the forward process by linearly interpolating between noise and data in a straight line. At the time step 𝑡, latent z𝑡 is defined as: z𝑡 = (1 −𝑡)z0 +𝑡𝜖, where z0 is the clean image, and 𝜖 ∈ N(0, 1) is the Gaussian noise. The model is trained to directly regress the target velocity given the noised latent z𝑡, timestep t, and condition 𝑦. The objective is to minimize the mean squared error:

𝐿𝑑𝑖𝑓 𝑓 = 𝐸[||(z0 − 𝜖) − V𝜃 (z𝑡,𝑡,𝑦)||2], (1)

where V𝜃 refers to the diffusion model. The DiT framework and Flow Matching are widely used in some recent diffusion models, such as Stable Diffusion3 [Esser et al. 2024] and Flux [flu 2023a].

- 3.2 Overview

An overview of our method is presented in Fig. 2. Specifically, we utilize the Flux-1.0-dev [flu 2023a] as the base model to build a unified framework for image customization, e.g., style, identity, subject appearance, and try-on. Given 𝑛 condition images C = {C1, ..., C𝑛}, we first reuse the VAE [Kingma et al. 2013] of Flux to encode the condition image to the same latent space as noisy latent. Note that the size of the condition image is flexible. Higher resolutions are

Existing studies (e.g., Prompt-to-Prompt [Hertz et al. 2022]) demonstrate that text-visual cross-attention maps inherently establish spatial-semantic correspondence between linguistic tokens and visual generation, i.e., the similarity response aligns with the spatial region of the corresponding subject in the generated result. Building upon this observation, UniPortrait [He et al. 2024] constrains the

[Figure 51]

[Figure 52]

Train Stages Capability Examples

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Subject-driven Data (Initial consistency training)

Result w/o Routing Constrain Layer 2 Layer 8 Layer 14

Condition Image

“On a rainy day a toy car on a windowsill of a quiet study, where droplets streak the glass and cast rippling patterns.”

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

“In the room, a boy hold a box, and he is laughing. In his side, another Thermos Cup on the table.”

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Subject-driven + Identity-driven + Try-on + Style Data (Train on all data for multi-task ability)

Prompt

Result w Routing Constrain Layer 2 Layer 8 Layer 14

Fig. 3. Visualization of cross-attention maps in subject-driven image generation. The first row shows results from a model trained without routing constraints, while the second row presents results from a model trained with routing constraints.

[Figure 68]

“In the beach, a woman hold a perfume and wear a robe.”

[Figure 69]

[Figure 70]

Flux-generated Data (Final fine-tune to improve quality )

“a woman wearing white dress is playing guitar, the background is the concert hall.”

recommended for detail-rich images to preserve clarity, while lower resolutions are sufficient for images with fewer details, thereby reducing compression costs. Then, all tokens (i.e., image, text, condition) are concatenated along the sequence dimension and fed into Flux. To enable the model to incorporate the condition input, we introduce a condition mapping layer at the input of Flux. The position embedding (PE) of condition tokens is aligned with that of the noisy latent using Rotary Position Embedding (RoPE) [Su et al. 2024]. Inspired by non-overlapping position embedding in OminiControl [Tan et al. 2024], we extend these embeddings along the diagonal in a similar fashion. In addition, we introduce a trainable and index-wise condition embedding (CE) R10×𝑐, which is directly added to condition tokens. Following OminiControl [Tan et al. 2024], we integrate Low-Rank Adaptation (LoRA) [Ryu 2023] modules into Flux as trainable parameters.

Fig. 4. The progressive training pipeline of our method. Left column shows the three training stages of our method. Right column shows the generation capability after the training of each stage.

the second row of Fig. 3, after training with routing constraint, the attention of the condition image clearly focuses on the target subject, and the result shows improved consistency with the reference image in terms of details. In addition to improved consistency, this strategy also helps decoupling in multi-reference cases.

In addition to image-to-image routing constraint, we also design placeholder-to-image routing constraint to establish correspondences between textual descriptions and condition inputs. Specifically, for the 𝑖-𝑡ℎ condition, we append a placeholder [𝑟𝑒𝑓 #𝑖] after the corresponding instance name, e.g., “A women from [ref#1] and a woman from [ref#2] is walking in the park.”. During the training for multi-condition tasks, we calculate the similarity between the conditional image tokens and the placeholder tokens. The routing constraint ensures that the similarity between C𝑖 and [𝑟𝑒𝑓 #𝑖] is 1, while it is 0 for all other pairs:

- 3.3 Routing Constraint

Inspired by UniPortrait [He et al. 2024] and AnyStory [He et al. 2025], in this paper, we design routing constraint in the DiT framework for general image customization tasks. As illustrated in Fig. 2, within the condition-guided framework, cross-attention exists between condition images and the generation result:

𝑛∑︁𝑐−1

1 𝑛𝑐

||𝑆𝑜𝑓 𝑡𝑚𝑎𝑥(Q𝑐𝑜𝑛𝑑,𝑖 × K𝑇𝑡𝑒𝑥𝑡,𝑖) − B𝑖||22, (4)

𝐿ℎ𝑜𝑙𝑑𝑒𝑟 =

𝑖=0

Q𝑐𝑜𝑛𝑑,𝑖K𝑇𝑖𝑚𝑔

, (2)

M =

√

where K𝑡𝑒𝑥𝑡,𝑖 refers to the text feature of [𝑟𝑒𝑓 #𝑖]. B𝑖 is a binary matrix, where the value is 1 when the placeholder matches the condition image, and 0 otherwise.

𝑑

where Q𝑐𝑜𝑛𝑑,𝑖 ∈ R𝑙𝑐𝑜𝑛𝑑,𝑖×𝑐 refers to the condition tokens of the 𝑖-𝑡ℎ condition image. K𝑖𝑚𝑔 ∈ R𝑙×𝑐 is the tokens of noisy image latent. The cross-attention map M ∈ R𝑙𝑐𝑜𝑛𝑑,𝑖×𝑙 is a dense similarity between the 𝑖-𝑡ℎ condition image and the generation result. To obtain the global response of the condition image in different locations of the generated output, we average the dense similarity matrix along the 𝑙𝑐𝑜𝑛𝑑,𝑖 dimension, resulting in a response map M ∈ R𝑙, representing the global similarity of the condition image on the generated result. To constrain the image-to-image attention focus on the specific subject, MSE loss is employed to optimize the attention within DiT across condition images and the generation result:

The final loss function of our method is defined as: 𝐿 = 𝜆𝑑𝑖𝑓 𝑓 · 𝐿𝑑𝑖𝑓 𝑓 + 𝜆𝑟𝑜𝑢𝑡𝑒 · 𝐿𝑟𝑜𝑢𝑡𝑒 + 𝜆ℎ𝑜𝑙𝑑𝑒𝑟 · 𝐿ℎ𝑜𝑙𝑑𝑒𝑟, (5)

where 𝜆𝑑𝑖𝑓 𝑓 , 𝜆𝑟𝑜𝑢𝑡𝑒 and 𝜆ℎ𝑜𝑙𝑑𝑒𝑟 are loss weights. To allow the model to handle regular text input, we introduce normal text without

placeholders with a probability of 50% and discard 𝐿ℎ𝑜𝑙𝑑𝑒𝑟 accordingly. Note that 𝐿𝑟𝑜𝑢𝑡𝑒 and 𝐿ℎ𝑜𝑙𝑑𝑒𝑟 do not incur significant additional computational overhead during training (2.5s/iter vs. 3s/iter).

3.4 Training Data Construction

𝑛∑︁𝑙−1

### 𝑛∑︁𝑐−1

To achieve generalized image customization, we collect training data, covering a wide range of tasks.

1 𝑛𝑐 × 𝑛𝑙

||M𝑖𝑗 − M𝑡𝑎𝑟𝑔𝑒𝑡,𝑖||22, (3)

𝐿𝑟𝑜𝑢𝑡𝑒 =

𝑗=0

𝑖=0

Identity paired data. Since high-quality identity paired data [Li et al. 2024] is difficult to collect from the Internet, we adopt the open-source ID customization method PuLID [Guo et al. 2024] for dataset construction. Specifically, we generate two images of the

where 𝑖 and 𝑗 refer to the condition index and layer index. 𝑛𝑐 and 𝑛𝑙 is the number of conditions and number of layers, respectively. M𝑡𝑎𝑟𝑔𝑒𝑡 refers to the subject mask for the target image. As shown in

Identity Driven

PhotoMaker InstantID PuLID DreamO PhotoMaker InstantID PuLID DreamO

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

“A sculpture of portrait”

“A woman is fishing by the lake”

Subject / IP Driven

MS-Diffusion OmniGen DreamO

###### MS-Diffusion OmniGen DreamO

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

“The character is riding a motorcycle down the street” “vase and dog inside a giant hourglass”

MS-Diffusion OmniGen OminiControl DreamO

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

“a backpack floating in an ocean of milk”

“A woman and another woman walk on the street”

Try-on

MagicClothing IMAGDressing OmniGen OminiControl DreamO

###### OmniGen DreamO

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

“A girl wear a shirt and shorts on the beach”

“A child playing in the park”

Style Driven

###### Style Image StyleShot StyleAlign InstantStyle DeaDiff CSGO DreamO

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

“A boy”

Fig. 5. Visual comparison between our DreamO and other methods.

same identity using PuLID-FLUX, which then serve as mutual references. We also provide PuLID-SDXL with a reference face image and a text prompt describing the desired style to produce stylized training pairs. Finally, we collect 150K photorealistic data and 60K stylized identity data.

training data. To rectify the absence of character-related conditions, we collect 100K paired character-related data through retrieval. For multi-subject-driven image customization, we construct some twocolumn images through concatenation on the Subject200K dataset. In addition, we employ X2I-subject [Xiao et al. 2024a] dataset in multi-subject-driven training. To improve human-driven generation, we develop a pipeline similar to MovieGen [Polyak et al. 2025].

Subject-driven data. For single-subject-driven image customization, we utilize the Subject200K [Tan et al. 2024] dataset as part of

Starting with a long-video dataset, we apply content-aware scene detection to extract short clips. Mask2Former [Cheng et al. 2022] is used to generate human masks for key frames and perform object tracking. For cross-clip instance matching, we use SigLip [Zhai et al. 2023] embeddings and clustering.

Try-on data. We create a paired try-on dataset from two sources. One part consists of paired model and clothing images collected directly from the Web. For the other, we first crawl high-quality model images as ground truth, then use image segmentation [Jin 2023; Jin et al. 2024] to extract clothing items and generate the corresponding pairs. All images are manually filtered to remove low-quality samples, resulting in a dataset of 500K try-on pairs.

Style-driven data. This paper tackles two style transfer tasks: (1) style reference control with text descriptions of content, and (2) style reference with content reference images. For the first task, we use an internal style customization model based on SDXL to generate images with the same style but different content from two distinct prompts. For the second task, training requires style reference images, content reference images, and target images, where the target image must match both the style and content structure of the reference images. Based on the type 1 dataset, we produce the content reference for each style image using Canny-guided Flux [flu

- 2023c]. More details are provided in the Appendix. Routing mask extraction. To obtain the labels for the routing constraint (i.e., Eq.3), we use LISA[Lai et al. 2024] to extract mask of object conditioned on the text descriptions. In certain complex datasets, we employ InternVL [Chen et al. 2024a] to generate description of the target object. More details are provided in the Appendix.

Although the training data is constructed separately for different tasks, we observe the emergence of some cross-task capabilities. For instance, the model can customize the combination of ID and Try-on (as shown in Fig. 12), which does not exist in the training data.

- 3.5 Progressive Training Process

In experiments, we find that training directly on all data makes convergence difficult. This is mainly due to the limited capacity of LoRA [Ryu 2023] optimization, making it difficult for the model to capture task-specific capabilities under complex data distributions. In addition to convergence, the quality of the output after training deviates from the original prior of Flux [flu 2023a]. This divergence is caused by the impact of some low-quality training samples.

To address these issues, we design a progressive training strategy that allows the model to smoothly converge across different tasks while mitigating the influence of training data on the generation prior of Flux. The training pipeline is shown in Fig. 4. Specifically, we first optimize the model on subject-driven training data to initiate the model with a consistency-preserving capability. Note that the Subject200K [Tan et al. 2024] training data is generated by the base model (i.e., Flux), thus shares a similar distribution with the model generation space, which facilitates fast convergence. Since the X2I-subject [Xiao et al. 2024a] dataset is synthetically generated by MS-Diffusion [Wang et al. 2024b], a lot of training samples contain undesired artifacts and distortions. Therefore, during this warm-up stage, the two-column Subject200K images described in

Sec. 3.4 are also utilized as part of the training data to facilitate rapid convergence of the multi-reference control. The right part of Fig. 4 illustrates that after the first training stage, the model acquires an initial subject-driven generation capability and presents strong textfollowing performance. In the second training stage, we incorporate all the training data and perform full-data tuning. This allows the model to further converge on all subtasks defined in this work.

After the second stage of full-data training, we observe that the generation quality is heavily influenced by the training data, particularly by low-quality training samples. To realign the generation quality with the generative prior of Flux, we design an image quality refinement training stage. Specifically, we utilize Flux to generate around 40𝐾 training samples. During training, we use the original images as references to guide the model in reconstructing itself. To prevent copy-paste effects, we drop 95% tokens of reference images. After a shot-time optimization, the generation quality improved significantly, achieving alignment with the generation prior of Flux.

4 EXPERIMENT

- 4.1 Implementation Details

In this paper, we adopt Flux-1.0-dev as the base model. The rank of LoRA [Ryu 2023] is set as 128, resulting in a total parameter increase of 478M. During training, we employ the Adam [Loshchilov and Hutter 2017] optimizer with a learning rate of 4e-5 and train on 8 NVIDIA A100 80G GPUs. The batch size is set as 8. The first training stage consists of 20K iterations, followed by 90K iterations in the second stage, and finally 3K iterations in the last training stage. In inference, we use Flux-Turbo [flu 2023b] for acceleration, enabling the generation of 1024 × 1024 results within 10s. Unless specified, all results in this paper are based on the Turbo model. Some of the example inputs are processed using BEN2 [Meyer and Spruyt 2025] to remove background.

- 4.2 Qualitative Comparsion

To validate the performance of DreamO, we conduct comparisons with recent state-of-the-art methods across multiple subtasks. The visual comparison is shown in Fig. 5. The first part presents a comparison between DreamO and SOTA identity-customization methods i.e., PhotoMaker [Li et al. 2024], InstantID [Wang et al. 2024a], PuLID [Guo et al. 2024]. The results demonstrate that DreamO can inject identity information with high fidelity across various scenes, while offering impressive flexibility for customization.

The secondpartcomparesDreamO with recent subject-customization

methods,includingsingle-task frameworks (i.e.,MS-Diffusion[Wang et al.2024b])andunifiedgeneration frameworks (i.e.,OmniGen[Xiao et al. 2024a], OminiControl [Tan et al. 2024]). The results demonstrate that DreamO achieves higher subject fidelity and better text consistency in both single-subject and multi-subbject scenarios.

The third part shows comparison in virtual try-on, indicating that DreamO can effectively place clothing in scenes that align with the provided text with high fidelity to the reference images. Unlike methods like IMAGDressing, which produce high-fidelity clothing but lose text alignment, DreamO maintains both.

The last part presents the comparison between DreamO and recent style-customization methods, i.e., StyleShot [Gao et al. 2024],

Table 1. Quantitative evaluation of subject-driven customization.

Single-subject Customization Multi-subject Customization

MS-Diffusion OmniGen OminiControl DreamO MS-Diffusion OmniGen DreamO

CLIP-sim ↑ 0.8989 0.8824 0.8220 0.9150 0.7686 0.7605 0.7775 DINO-sim ↑ 0.7746 0.7582 0.6089 0.8056 0.6113 0.5646 0.6253 Text-sim ↑ 31.78 31.74 31.12 31.92 31.34 29.55 31.46

Table 2. Quantitative evaluation of text-driven style transfer. The Text-sim↑ is computed by the Cosine similarity between CLIP [Radford et al. 2021] image embedding and CLIP text embedding.

StyleAlign StyleShot InstantStyle DEADiff CSGO DreamO

Style-sim ↑ 0.7122 0.6922 0.6988 0.7269 0.7296 0.7340 Text-sim ↑ 0.2566 0.2693 0.2721 0.2656 0.2701 0.2750

Table 3. Quantitative evaluation of identity-driven customization.

PhotoMaker InstantID PuLID DreamO Face-sim ↑ 0.212 0.590 0.5829 0.607 Text-sim ↑ 0.2520 0.2294 0.2534 0.2570

Table 4. Quantitative evaluation of try-on.

MagCloth IMAGDressing OmniGen OminiControl DreamO

CLIP-sim ↑ 0.5977 0.8405 0.7265 0.7065 0.7613 Text-sim ↑ 30.17 17.74 27.83 28.79 30.47

StyleAlign [Wu et al. 2021], InstantStyle [Wang et al. 2024c], DeaDiff [Qi et al. 2024], and CSGO [Xing et al. 2024]. It can be observed that DreamO has weaker content intrusion, better text alignment, and higher style fidelity in the generated results.

- 4.3 Quantitative Comparison

In addition to qualitative comparison, we conduct quantitative comparisons for each task. We present the comparison of identity customization in Tab. 3, which is evaluated on Unsplash-50 [Gal et al. 2024]. Here, we provide 9 prompts for each face. Following PuLID [Guo et al. 2024], the Face-Sim represents the ID cosine similarity, with ID embeddings extracted by CurricularFace [Huang et al. 2020]. We also compute the CLIP cosine similarity between the generated result and the prompt to measure the text-following ability of different methods. As can be seen, our DreamO shows better face similarity and text-based customization ability.

Tab. 1 presents the quantitative comparison in single- and multisubject customization. We use DreamBench [Ruiz et al. 2023] as the testset of single-subject customization. For multi-subject customization, we randomly select 20 pairs from DreamBench and provide 25 prompts for each. During testing, we generate four images with different seeds for each test sample. Here, we calculate the CLIP cosine similarity and Dino [Caron et al. 2021] cosine similarity between the generated result and the reference images as a measure of subject consistency. To improve accuracy, we remove the background of the generated results and then calculate similarity. Additionally, we

Table 5. Ablation study of different model settings in multi-subject-driven customization.

w/o CE w/o RC w/o PT DreamO

CLIP-sim ↑ 0.7697 0.7448 0.7349 0.7775 DINO-sim ↑ 0.6097 0.5540 0.5381 0.6253 Text-sim ↑ 31.26 28.42 28.31 31.46

employ CLIP cosine similarity between the text description and the generated result as a measure of content alignment. Tab. 1 shows that our method outperforms others in subject consistency while demonstrating strong text-following ability.

The quantitative comparison of the try-on is presented in Tab. 4. We select 300 reference garments from VITON-HD [Choi et al. 2021] encompassing various styles and colors as the test data. During testing, we provide 10 prompts for each cloth. The CLIP consine similarity between the generated result and reference cloth is employed to measure the try-on accuracy. Here, we only crop the cloth from the result to compute the CLIP-sim. The CLIP cosine similarity between the generated results and the prompt is employed to measure the text-following ability of different methods. The result in Tab. 4 shows the attractive performance of our DreamO. Although IMAGDressing [Shen et al. 2025] has higher CLIP-sim, it can only generate images with a white background (i.e., Fig. 5 ) with little text-following ability.

For style customization, we construct an evaluation dataset, containing 249 style images and 24 prompt. Each method generates 249× 24stylecustomizationresults. Weuse the pre-trained CSD [Somepalli et al. 2024] to extract the style feature of generated results and reference images, and calculate the cosine similarity between them as a measure of style consistency. Furthermore, we compute the CLIP [Radford et al. 2021] similarity between stylized results and text descriptions as the metric of content consistency. The results are shown in Tab. 2, which demonstrate that our method has better performance in style consistency and content consistency.

4.4 User Study

In addition to the automatic evaluation metrics, we also conduct a user study for manual evaluation of different methods. Specifically, for each task (i.e., style, object, identity, and try-on customization), we assigned 6 test samples and invited 20 volunteers to rate on three aspects: text alignment, reference alignment, and image quality. The scoring range was set from 0 to 5, where a higher score indicates greater satisfaction. Fig. 9 shows that our DreamO achieves better performance in these three evaluation aspects.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

- IMG 1

- IMG 2 w/o Placeholder Loss w Placeholder Loss w/o Placeholder Loss w Placeholder Loss

[Figure 123]

“Two women walk in the park. The woman on the left is from [ref#2]. The woman on the right is from [ref#1].”

“Two women walk in the park. The woman on the left is from [ref#1]. The woman on the right is from [ref#2].”

#### Fig. 6. The ablation study of the placeholder-to-image routing constraint.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

###### RES RES RES RES

- IMG 1

- IMG 2

IMG 1

[Figure 132]

w/o routing constrain w routing constrain w/o routing constrain w routing constrain

“A woman walks in the park” “In the room, two toys on the table”

[Figure 133]

[Figure 134]

- Fig. 7. The ablation study of routing constraint in our proposed DreamO.

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

“A boy hold a box, and he is laughing. In his side, another Thermos Cup on the table”

“In front of the grocery store, the girl is wearing a green dress and a polka dot headscarf”

w/o progressive training Full Implementation w/o quality tuning Full Implementation

[Figure 140]

- IMG 1

- IMG 2

IMG 1

[Figure 141]

- Fig. 8. The ablation study of progressive training in our proposed DreamO.

Fig. 9. The user study of different methods.

Placeholder-to-image routing constraint. As demonstrated in Eq. 4, this paper designs a placeholder-to-image routing constraint to build the routing relationship between placeholders and specific images. Fig. 6 shows the effect of this loss term. It can be seen that without this loss, placeholders struggle to precisely control their corresponding images. After applying this loss, placeholders can bind to specific reference objects, enabling individual control over particular objects during multi-subject customization.

- 4.5 Ablation Study

Routing constraint. In this paper, we introduce a routing constraint into DiT training to enhance generation fidelity and facilitate the decoupling of multi-condition control. To evaluate its effectiveness, we ablate the routing constraint during training, with results shown in Fig. 7. In single-condition generation, its removal leads to degraded reference fidelity, e.g., the clothing color becomes inconsistent with the reference. In multi-condition settings, it causes condition coupling, e.g., features of the two toys are crossed. These results confirm that the routing constraint improves the fidelity and disentanglement of different conditions.

Quantitative results. In addition to the visual comparison, we show the quantitative results of the ablation study, as shown in Tab. 5. The experiment is conducted on multi-subject-driven customization. It can be seen that not using the routing constraint (RC) and progressive training strategy (PT) significantly impacts performance, leading to a decrease in the reference consistency and text following. We also study the role of condition embedding (CE), and its absence results in a decline in the reference consistency.

Progressive training. To enable the model to better converge on all sub-tasks under complex data distributions and to rectify the impact of training data distribution on generation quality, we design a progressive training strategy. The effectiveness of this strategy is demonstrated in Fig. 8. One can see that directly training the model on all datasets leads to suboptimal convergence, particularly in complex tasks such as multi-subject consistency. Warming up on a smaller and easier-to-learn dataset (e.g., Subject200K [Tan et al.

5 CONCLUSION

In this study, we introduce DreamO, a unified framework designed for generalized image customization across diverse condition types (e.g., identity, style, subject, and try-on) within a single pre-trained DiT architecture. To facilitate this, we construct a large-scale training dataset. By embedding all condition types into the DiT input sequence and incorporating a feature routing constraint, DreamO achieves high-fidelity consistency while effectively disentangling heterogeneous control signals. In addition, we design a progressive training strategy that enables the model to incrementally acquire diverse control capabilities under complex data distributions, while

- 2024]) before joint training improves convergence, but the generation quality is easily influenced by the training data distribution, deviating from the generation priors of Flux. By introducing an image quality tuning stage, the model can produce higher-quality generation results.

maintaining the image quality inherent to the base model. Comprehensive experiments demonstrate that DreamO excels in performing a wide range of image customization tasks with high-quality results.

REFERENCES

- 2023a. "https://github.com/black-forest-labs/flux?tab=readme-ov-file". "https://github. com/black-forest-labs/flux?tab=readme-ov-file"
- 2023b. https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha. https:// huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha
- 2023c. https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev. https://huggingface. co/black-forest-labs/FLUX.1-Canny-dev

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. 2024. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219 (2024).

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 2021. Emerging Properties in Self-Supervised Vision Transformers. In Proceedings of the International Conference on Computer Vision (ICCV).

Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. 2024b. UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics. arXiv preprint arXiv:2412.07774 (2024).

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. 2024a. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 (2024).

Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. 2022. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1290–1299.

Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. 2021. Viton-hd: Highresolution virtual try-on via misalignment-aware normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 14131–14140.

Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. 2024. Improving diffusion models for authentic virtual try-on in the wild. In European Conference on Computer Vision. Springer, 206–235.

Prafulla Dhariwal and Alexander Nichol. 2021. Diffusion models beat gans on image synthesis. Advances in neural information processing systems 34 (2021), 8780–8794.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022).

Rinon Gal, Or Lichter, Elad Richardson, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2024. Lcm-lookahead for encoder-based text-to-image personalization. In European Conference on Computer Vision. Springer, 322–340.

Junyao Gao, Yanchen Liu, Yanan Sun, Yinhao Tang, Yanhong Zeng, Kai Chen, and Cairong Zhao. 2024. Styleshot: A snapshot on any style. arXiv preprint arXiv:2407.01414 (2024).

Zinan Guo, Yanze Wu, Chen Zhuowei, Peng Zhang, Qian He, et al. 2024. Pulid: Pure and lightning id customization via contrastive alignment. Advances in Neural Information Processing Systems 37 (2024), 36777–36804.

Junjie He, Yifeng Geng, and Liefeng Bo. 2024. UniPortrait: A Unified Framework for Identity-Preserving Single-and Multi-Human Image Personalization. arXiv preprint arXiv:2408.05939 (2024).

Junjie He, Yuxiang Tuo, Binghui Chen, Chongyang Zhong, Yifeng Geng, and Liefeng Bo. 2025. AnyStory: Towards Unified Single and Multiple Subject Personalization in Text-to-Image Generation. arXiv preprint arXiv:2501.09503 (2025).

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33 (2020), 6840–6851.

Miao Hua, Jiawei Liu, Fei Ding, Wei Liu, Jie Wu, and Qian He. 2023. Dreamtuner: Single image is enough for subject-driven generation. arXiv preprint arXiv:2312.13691

(2023).

Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. 2023. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778 (2023).

Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. 2024b. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775 (2024).

Mengqi Huang, Zhendong Mao, Mingcong Liu, Qian He, and Yongdong Zhang. 2024a. RealCustom: narrowing real text word for real-time open-domain text-to-image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7476–7485.

Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. 2020. Curricularface: adaptive curriculum learning loss for deep face recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 5901–5910.

Zhenchao Jin. 2023. Sssegmenation: An open source supervised semantic segmentation toolbox based on pytorch. arXiv preprint arXiv:2305.17091 (2023).

Zhenchao Jin, Xiaowei Hu, Lingting Zhu, Luchuan Song, Li Yuan, and Lequan Yu. 2024. IDRNet: Intervention-driven relation network for semantic segmentation. Advances in Neural Information Processing Systems 36 (2024).

Diederik P Kingma, Max Welling, et al. 2013. Auto-encoding variational bayes. Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu.

2023. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1931–1941.

Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia.

2024. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9579–9589.

Dongxu Li, Junnan Li, and Steven Hoi. 2023. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems 36 (2023), 30146–30166.

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. 2024. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8640–8650.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022). Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv

preprint arXiv:1711.05101 (2017).

Junsheng Luan, Guangyuan Li, Lei Zhao, and Wei Xing. 2025. MC-VTON: Minimal Control Virtual Try-On Diffusion Transformer. arXiv preprint arXiv:2501.03630

(2025).

Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. 2024a. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In ACM SIGGRAPH 2024 Conference Papers. 1–12.

Yuhang Ma, Wenting Xu, Jiji Tang, Qinfeng Jin, Rongsheng Zhang, Zeng Zhao, Changjie Fan, and Zhipeng Hu. 2024b. Character-Adapter: Prompt-Guided Region Control for High-Fidelity Character Customization. arXiv preprint arXiv:2406.16537 (2024).

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2021. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 (2021).

Maxwell Meyer and Jack Spruyt. 2025. BEN: Using Confidence-Guided Matting for Dichotomous Image Segmentation. arXiv preprint arXiv:2501.06230 (2025).

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. 2023. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421 (2023).

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. 2024. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence. 4296–4304.

Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. 2022. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. In International Conference on Machine Learning. PMLR, 16784–16804.

William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. 2025. Movie Gen: A Cast of Media Foundation Models. arXiv:2410.13720 [cs.CV] https://arxiv.org/abs/2410. 13720

Tianhao Qi, Shancheng Fang, Yanze Wu, Hongtao Xie, Jiawei Liu, Lang Chen, Qian He, and Yongdong Zhang. 2024. Deadiff: An efficient stylization diffusion model with disentangled representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8693–8702.

Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. 2023. Unicontrol: A unified diffusion model for controllable visual generation in the wild. arXiv preprint arXiv:2305.11147 (2023).

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. 8748–8763.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 (2022).

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10684–10695.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 22500–22510.

Simo Ryu. 2023. Low-rank adaptation for fast text-to-image diffusion fine-tuning. Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. 2022a. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487 (2022).

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022b. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems 35 (2022), 36479– 36494.

Fei Shen, Xin Jiang, Xin He, Hu Ye, Cong Wang, Xiaoyu Du, Zechao Li, and Jinhui Tang.

2025. Imagdressing-v1: Customizable virtual dressing. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 6795–6804.

Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. 2024. Measuring Style Similarity in Diffusion Models. arXiv preprint arXiv:2404.01292 (2024).

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568 (2024), 127063.

Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. 2024. OminiControl: Minimal and Universal Control for Diffusion Transformer. arXiv preprint arXiv:2411.15098 (2024).

Zhenchen Wan, Dongting Hu, Weilun Cheng, Tianxi Chen, Zhaoqing Wang, Feng Liu, Tongliang Liu, Mingming Gong, et al. 2025. MF-VITON: High-Fidelity Mask-Free Virtual Try-On with Minimal Input. arXiv preprint arXiv:2503.08650 (2025).

Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. 2024c. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733 (2024).

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. 2024a. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519 (2024).

Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. 2024b. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209 (2024).

Zongze Wu, Yotam Nitzan, Eli Shechtman, and Dani Lischinski. 2021. Stylealign: Analysis and applications of aligned stylegan models. arXiv preprint arXiv:2110.11323

(2021).

Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. 2024b. Fastcomposer: Tuning-free multi-subject image generation with localized attention. International Journal of Computer Vision (2024), 1–20.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. 2024a. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340 (2024).

Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li. 2024. Csgo: Content-style composition in text-to-image generation. arXiv preprint arXiv:2408.16766 (2024).

Rento Yamaguchi and Keiji Yanai. 2024. Exploring Cross-Attention Maps in Multi-modal Diffusion Transformers for Training-Free Semantic Segmentation. In Proceedings of the Asian Conference on Computer Vision. 260–274.

Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. 2023. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18381–18391.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023).

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision. 11975–11986.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. 2023. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems 36 (2023), 11127–11150.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

###### RES RES RES

- IMG 1

- IMG 2

[Figure 148]

###### IMG 1 IMG 1

“A man wearing a white shirt and black robe, walks on an old street”

“She spins lightly, her hands dance, and she dances among flowers”

“A man is dancing with a woman in the room”

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

###### RES RES RES

###### IMG 1 IMG 1 IMG 1

“She skillfully stir-fried the dishes in the pot” “The woman is playing guitar in the park”

“The woman is watching TV on the couch”

#### Fig. 10. The capability of our proposed DreamO in identity-driven image customization.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

RES RES IMG 1 RES

###### IMG 1

IMG 2

[Figure 161]

###### IMG 1

“An Akita dog, wearing a judo suit, in Taekwondo Hall. The dog holding a sign saying DreamO”

“In a garden, a woman holds a perfume towards the camera”

“A toy and candles by the lake.”

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

- IMG 1

- IMG 2

###### IMG 1

[Figure 168]

###### IMG 2 IMG 1

[Figure 169]

“An evil Husky, raising its front paws, fights against the curtains, with a bitten sofa and coffee table in the background, revealing the cotton inside the sofa”

“In the street, a teddy bear carrying a school bag”

“A dog from [ref#1] and another dog from [ref#2] in the jungle”

Fig. 11. The capability of our proposed DreamO in subject-driven image customization.

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

RES RES RES

- IMG 1

- IMG 2

- IMG 1

- IMG 2

[Figure 174]

[Figure 175]

- IMG 1

- IMG 2

[Figure 176]

[Figure 177]

[Figure 178]

“In the forest, a man in a robe” “A woman wearing a green bikini is scattered on the beach”

“In the room, A blue toy dog wear a sunglasses”

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

RES RES RES

- IMG 1

- IMG 2

- IMG 1

- IMG 2

IMG 1 IMG 2

[Figure 186]

[Figure 187]

[Figure 188]

IMG 3

“the woman is wearing the red over-ear headphones , on a stage”

“A beautiful woman in cheongsam by the lake”

“In the forest, a woman wearing denim shorts and a T-shirt”

Fig. 12. The capability of our proposed DreamO in try-on image customization.

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

IMG 1 RES

RES RES

[Figure 195]

IMG 1 IMG 1 IMG 2

“Generate a same style image. Notre Dame de Paris” “Generate a same style image. A shark”

“Generate a same style image. A girl”

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

- IMG 1

- IMG 2

###### RES

RES

RES

[Figure 202]

IMG 2

“Generate a same style image. A dog”

“Generate a same style image. Eiffel Tower” “Generate a same style image. A girl”

Fig. 13. The capability of our proposed DreamO in style-driven image customization.

