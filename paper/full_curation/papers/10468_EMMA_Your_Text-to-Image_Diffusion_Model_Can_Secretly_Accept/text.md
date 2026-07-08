# arXiv:2406.09162v1[cs.CV]13Jun2024

## EMMA: Your Text-to-Image Diffusion Model Can Secretly Accept Multi-Modal Prompts

Yucheng Han1,2∗ Rui Wang2∗† Chi Zhang2∗ ‡ Juntao Hu2 Pei Cheng2 Bin Fu2 Hanwang Zhang1 1Nanyang Technological University 2 Tencent 1 {yucheng002, hanwangzhang}@ntu.edu.sg 2 {raywwang, johnczhang, jetthu, brianfu}@tencent.com https://tencentqqgylab.github.io/EMMA

||[Figure 1]|
|---|
<br><br>|A woman is walking<br><br>on a bustling street.|[Figure 2]|
|---|---|
<br><br>[Figure 3]<br><br>[Figure 4]<br><br>Conditions<br><br>Generated image<br><br>Text Face Portrait|
|---|

|[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>…<br><br>…<br><br>…<br><br>…<br><br>Portrait + Video<br><br>Portrait+ Portrait Cartoon<br><br>Awoman wearing a purple dress with white flowers is walking in a bustling ci ty street.<br><br>A woman wearing a dress is walking in a bustling...<br><br>Awoman wearing a green<br><br>dress with white<br><br>flowers is walking in a bustling ci ty street.<br><br>Awoman wearing a green dress with white flowers is walking in a bustling ci ty street.<br><br>[Figure 25]<br><br>+ Prompt<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>Portrait+ Clay|
|---|

|[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>A woman… is<br><br>enjoying her breakfast, … while reading<br><br>a newspaper...<br><br>A woman…<br><br>The painting, a vivid portra yal of the…<br><br>In the heart of a sunlit park, a woman is playing the guitar…<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>A woman is standing in a crowded subway.<br><br>A woman... She is waving her hands.<br><br>A woman… She is showing off her arm muscles.<br><br>…, a woman is engaged in reading, …<br><br>A woman is sitting on a crowded subway.<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>A cat sprints down a street…, a towering skyscraper can be…<br><br>A cat sits on a warm sandy… a<br><br>line of palm<br><br>tree…<br><br>On a windowsill<br><br>… Beside<br><br>the cat, a plant…<br><br>A cat is ascending a tall oak tree... filled with flowers … …<br><br>Cond. Image<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>|
|---|

Figure 1: EMMA could compose multiple multi-modal conditions (on the top left branch) without further finetuning, while still maintaining strong text control over the generated results (bottom branch). Furthermore, EMMA could combine various existing diffusion models in communities without training.

∗Equal contributions. Work was done when Yucheng Han was a Research Intern at Tencent. †Project Leader. ‡Corresponding Author.

#### Abstract

Recent advancements in image generation have enabled the creation of high-quality images from text conditions. However, when facing multi-modal conditions, such as text combined with reference appearances, existing methods struggle to balance multiple conditions effectively, typically showing a preference for one modality over others. To address this challenge, we introduce EMMA, a novel image generation model accepting multi-modal prompts built upon the state-of-the-art text-to-image (T2I) diffusion model, ELLA. EMMA seamlessly incorporates additional modalities alongside text to guide image generation through an innovative Multi-modal Feature Connector design, which effectively integrates textual and supplementary modal information using a special attention mechanism. By freezing all parameters in the original T2I diffusion model and only adjusting some additional layers, we reveal an interesting finding that the pre-trained T2I diffusion model can secretly accept multi-modal prompts. This interesting property facilitates easy adaptation to different existing frameworks, making EMMA a flexible and effective tool for producing personalized and context-aware images and even videos. Additionally, we introduce a strategy to assemble learned EMMA modules to produce images conditioned on multiple modalities simultaneously, eliminating the need for additional training with mixed multi-modal prompts. Extensive experiments demonstrate the effectiveness of EMMA in maintaining high fidelity and detail in generated images, showcasing its potential as a robust solution for advanced multi-modal conditional image generation tasks.

#### 1 Introduction

The field of image generation has recently experienced significant growth, driven by advancements from both academic and industrial researchers. Recent models, such as DALLE-3 and Stable Diffusion 3 Esser et al. [2024], have elevated text-conditioned image generation to unprecedented levels. These models, requiring only simple textual instructions, demonstrate remarkable capability in generating high-quality images with intricate details. These approaches typically involve a classifierfree mechanism during the diffusion process to integrate conditions. For example, in the widely adopted Stable Diffusion, text prompts work as conditions of the diffusion network via cross-attention mechanisms to enable text-to-image translation.

Recent studies have also explored image generation conditioned on multi-modal prompts, which require simultaneous guidance from multiple modalities. For example, IP-Adapter Ye et al. [2023] guides image generation by referring to both image prompts and textual instructions, through developed cross-attention modules. Similarly, FaceStudio Yan et al. [2023] adopted a hybrid guidance framework and could utilize stylized images, facial images, and textual prompts as conditions for personalized portrait generation. Based on these techniques, a variety of interesting applications have emerged, such as subject-driven image generation Pan et al. [2023], Li et al. [2024], Purushwalkam et al. [2024], personalized image generation Wang et al. [2024], Yan et al. [2023], and artistic portrait creation Ye et al. [2023]. However, previous works employ distinct strategies for multi-modal prompts, and the genetic architecture of general multi-modal guided image generation remains unknown.

One of the main challenges facing current design paradigms is how to balance various conditions. During the image generation process, when multiple conditions are used, current methods may tend to favor certain conditions over others. For instance, it is observed that IP-Adapter Ye et al. [2023], which relies on text prompts and image features as conditions, may predominantly be influenced by the image features. This can be attributed to the inherent limitations within the model architectures of existing methods, which do not effectively manage the varying complexities associated with different conditions. When training a model on multi-modal prompts, it often learns to control just one condition effectively, neglecting more challenging ones. This results in a bias towards easier conditions. For example, if a network is trained with both an object image and its description as conditions, it might overly rely on the image to generate object appearance, failing to adequately learn from the description. This issue highlights the need for strategies in training that ensure balanced learning across all conditions to maintain the model’s versatility and fairness. Furthermore, the scarcity of multi-modal training datasets in specialized domains exacerbates the issue. Taking

subject-driven image generation as an example, a series of models (Kosmos-g Pan et al. [2023], BootPig Purushwalkam et al. [2024], SSR-Encoder Zhang et al. [2023]) uses cropped object images to serve both as conditions and as the ground truth, which is a common practice in this area. However, models trained on such datasets are limited to a simple copy-paste functionality and may ignore the textual conditions. The absence of suitable training datasets becomes increasingly problematic with an increasing number of conditions. The limitations of model architecture and the lack of appropriate training datasets make it difficult to achieve a balanced approach for image generation models with multiple conditions.

To address the challenge above, we aim to design a more flexible paradigm for multi-modal guidance, that could well balance multiple conditions. In this paper, we introduce EMMA. Our proposed EMMA is built upon the state-of-the-art text-conditioned diffusion model ELLA Hu et al. [2024], which trains a transformer-like module, named Perceiver Resampler, to connect text embeddings from pre-trained text encoders and pre-trained diffusion models for better text-guided image generation. ELLA can effectively utilize pre-trained text and diffusion knowledge to achieve SOTA results in dense prompt-based image generation without the need to adjust their raw parameters. ELLA has strong text-to-image generation ability, and our proposed EMMA could merge information from other modalities into text features for guidance. This is inspired by Flamingo Alayrac et al. [2022], a multi-modal large language model aiming at multi-modal understanding. Flamingo employs a strategy where it encodes images and text separately and integrates image features into text features using cross-attention within various transformer layers in the large language model. In this way, Flamingo adopts text as the primary carrier of information and integrates information from other modalities into LLM precisely for multi-modal understanding. Similarly, leveraging the transformer structure used by ELLA, which extracts features from the LLM to inject into SD, we introduce information from other modalities in the intermediate layers of these transformers to facilitate multimodal guidance.

In detail, to control the image generation process by modalities beyond text, EMMA incorporates our proposed Assemblable Gated Perceiver Resampler (AGPR), which leverages cross-attention to inject information from additional modalities beyond texts. In our design, the AGPR blocks are strategically interleaved with the blocks of the Perceiver Resampler of ELLA. This arrangement ensures an effective integration of multi-modal information. During training, we freeze the raw modules of ELLA to maintain the control ability of text conditions. Finally, we get a series of models based on different conditions, such as text features combined with facial features, and text features combined with object-level image features.

Notably, EMMA is inherently designed to handle multi-modal prompts as conditions, allowing for the straightforward combination of different multi-modal configurations. This is achieved by the gate mechanism in our AGPR, which could control the way of injecting information from other modalities into the textual features. This advantage enables diverse and complex inputs to be synthesized into a unified generation framework without the need for additional training. For example, image features can be utilized to depict the main subject, while finer-grained facial features provide identity information.

As EMMA does not necessitate modifications to the underlying diffusion model, i.e. the U-net model or DiT Chen et al. [2023], Peebles and Xie [2023] model, it is readily compatible with a multitude of existing works based on the Stable Diffusion framework. By directly replacing the condition modules with EMMA, a series of interesting applications could be produced with no need for further training, such as Portrait generation, Cartoon generation, and subject-driven video generation shown in Figure 1.

Our key contributions are as follows:

- 1. Novel Integration Mechanism for Multi-modal prompts: We introduce the EMMA, a pioneering approach that merges features of multi-modal prompts into the image generation process without compromising textual control. Our approach significantly enhances the flexibility and applicability of image generation by enabling the synergistic interaction of multiple modalities. This innovation allows for the creation of high-quality images that are responsive to a variety of input conditions.
- 2. Modular and Efficient Model Training: Our framework facilitates the modular assembly of models conditioned on different modalities, streamlining the process and eliminating the need for retraining when new conditions are introduced. This efficient training procedure conserves resources and accelerates the model’s adaptability to novel tasks.

|A woman is walking on a bustling street.<br><br>Text encoder<br><br>U-Net<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>| |…<br><br>x N|
|---|---|
|PR Block<br><br>AGPR Block<br><br>[Figure 64]<br><br>❄ 🔥<br><br>[Figure 65]| |
<br><br>……<br><br>Face encoder<br><br>Image encoder<br><br>Face feature<br><br>Image feature|
|---|

……

(a) The Frame work of EMMA. PR is short for Perceiver Resampler; AGPR is short for Assemblable Gated Perceiver Resampler.

|𝑔 𝑔<br><br>TimeAware CrossAttention<br><br>TimeAwareFFN<br><br>TimeAware CrossA en on<br><br>TimeAwareFFN<br><br>𝑔 𝑔<br><br>Linear<br><br>Linear|
|---|

|TimeAware CrossAttention<br><br>TimeAwareFFN<br><br>Text features Perceiver Resampler Block<br><br>Learnable tokens|
|---|

(b) Perceiver Resampler (PR) Block in ELLA.

|TimeAware CrossA en on<br><br>TimeAwareFFN<br><br>𝑔<br><br>Assemblable Gated Perceiver Resampler Block<br><br>Linear<br><br>𝑔<br><br>Image features<br><br>Learnable tokens|
|---|

(d) Illustra on of the modular assembly of models condi oned on diﬀerent modali es.

(c) Our proposed Assemblable Gated Perceiver Resampler (AGPR) Block.

- Figure 2: The model architecture of our proposed EMMA. (a) The framework of our EMMA. (b) The architecture of the Perceiver Resampler block proposed in ELLA Hu et al. [2024] (c) The architecture of our Assemblable Gated Perceiver Resampler block. The orange part is the novel part introduced in our AGPR block compared with the Perceiver Resampler block. (d) The pipeline of the composite process.

- 3. Universal Compatibility and Adaptability: EMMA works as a plug-and-play module without fine-tuning for a spectrum of existing and emerging models, including various image and video generation applications. Its compatibility with the Stable Diffusion framework and other models enhances its utility across diverse domains.
- 4. Robust Performance and Detail Preservation: Through our experiments, we have confirmed the robustness of the EMMA model against various control signals, ensuring that it preserves both textual and visual details in the generated images. The model’s architecture is designed to be scalable and flexible, accommodating a wide range of conditions and applications while maintaining high fidelity and quality.

#### 2 Related Work

Text-to-Image Diffusion Models. Text-to-image diffusion models have made significant strides in producing high-quality and diverse images. These models depend on robust text encoders to interpret intricate image descriptions. Several models, such as GLIDENichol et al. [2021], LDMRombach

- et al. [2022], DALL-E 2Ramesh et al. [2022], and Stable DiffusionRombach et al. [2022], Podell
- et al. [2023], leverage the pre-trained CLIPRadford et al. [2021] model to generate text embeddings. Other models like ImagenSaharia et al. [2022], Pixart-αChen et al. [2023], ELLAHu et al. [2024], and DALL-E 3Betker et al. [2023] employ large pre-trained language models, such as T5Raffel et al. [2020], to enhance their understanding of text. Some models, including eDiff-IBalaji et al. [2022] and EMUDai et al. [2023], use a combination of both CLIP and T5 embeddings to improve their capabilities. ParaDiffusionWu et al. [2023] proposes fine-tuning the LLaMA-2Touvron et al. [2023] model during diffusion model training and utilizing the fine-tuned language model text features as a condition. To further enhance the prompt following ability, we integrate large language models (LLMRaffel et al. [2020], Touvron et al. [2023], Zhang et al. [2024]) with pre-trained CLIP-based models, using techniques such as TSC (Textual Style Control).

Subject-driven Image Generation. This category includes studies focused on enhancing personalization and subject specificity in image generation through innovative techniques and architectures. Subject-Diffusion Ma et al. [2023a] integrates text and image semantics for personalized generation

without test-time fine-tuning. ELITE Wei et al. [2023] and FastComposer Xiao et al. [2023] reduce the need for fine-tuning by employing efficient encoding and attention mechanisms for personalized image generation. BLIP-Diffusion Li et al. [2024] and Kosmos-G Pan et al. [2023] utilize pre-trained models for quick and effective personalized image generation. Unified Multi-Modal Latent Diffusion Ma et al. [2023b] and IP-Adapter Ye et al. [2023] enhance image quality by integrating multimodal inputs to align images with textual descriptions. FaceStudio Yan et al. [2023], InstantID Wang et al. [2024], and PhotoMaker Li et al. [2023] address the high resource demands of previous models and include features for identity preservation, critical for high-fidelity tasks like artistic portrait generation. The MoA (Mixture-of-Attention) Ostashev et al. [2024] uses a novel mechanism to separate subject and context for better image quality. BootPIG Purushwalkam et al. [2024] uses the reference net to introduce low-level information and achieves pixel-level control over generated images. The most recent and related work is SSR-Encoder Zhang et al. [2023], which uses cross-attention to inject image information into text features and supports selective feature extraction.

Optimization-based subject-driven image generation. The paper Gal et al. [2022] introduces a method to personalize text-to-image generation through unique embeddings derived from userprovided images, enhancing the creation of unique concepts. Dreambooth Ruiz et al. [2023] describes a technique for fine-tuning text-to-image models to produce novel, contextualized images of a specific subject using a unique identifier. The paper Liu et al. [2023] explores the concept of neurons in diffusion models that facilitate customized generation and efficient storage. A subsequent study Liu et al. [2023] addresses synthesizing images with multiple subjects using text embeddings and spatial layouts to improve the quality and control of the synthesis.

#### 3 Methodology

##### 3.1 Model Architecture

The overall pipeline of EMMA is depicted in Figure 2 (a). Our model’s conditions encompass two aspects. One is the textual feature, and the other is the customized image features, such as visual clip features or facial embeddings. In EMMA, we inject text features through Perceiver Resampler blocks proposed by ELLA Hu et al. [2024] as shown in Figure 2 (b). The image features are perceived by our newly proposed module named Assemblable Gated Perceiver Resampler as shown in Figure 2 (c).

To be more specific, we categorize EMMA into three main components and describe them in detail. Text Encoder: T5 Chung et al. [2024] is equipped to understand rich textual content. Prior research has shown that T5 is adept at extracting textual features, which makes it well-suited for supplying textual features to downstream tasks.

Image Generator: In the realm of image generation, numerous researchers and practitioners have fine-tuned various models on a clip-specific basis, aligning with their specific goals and data types. We strive for our final network to ensure the generalization of features, thereby maximizing the use of the high-quality models prevalent in the community.

Multi-modal Feature Connector: The network architecture is depicted in Figure 2. Drawing inspiration from Flamingo Alayrac et al. [2022] and ELLA, the connector consists of two alternating stacked network modules: the Perceiver Resampler and the Assemblable Gated Perceiver Resampler. The Perceiver Resampler is primarily tasked with integrating textual information, while the Assemblable Gated Perceiver Resampler is designed to incorporate additional information. These network modules use an attention mechanism to assimilate multimodal information into the learnable token embeddings, which are then supplied to the U-net as conditions. We give the definitions of these blocks as follows. The connector contains K learnable tokens, denoted by Latent. Time embeddings, textual features, and additional conditions are represented by t, T, and C, respectively.

The Perceiver Resampler block can be divided into two parts:

###### L = L + TimeAwareAttn(L,T,t), (1) L = L + TimeAwareFFN(L,t). (2)

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

A woman is about to deliver a letter.

She is tripped over a branch.

The dog just wants to play with her.

The woman goes back home.

She is chased by a dog.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

A man is having breakfast in his home.

He hears there is a treasure nearby.

The man finds an old wooden house.

He runs into a forest.

He finds the treasure.

- Figure 3: Images generated by our EMMA with portrait conditions. Two sets of images are generated for two separate stories. The first set of images is about a mailing woman chased by a dog. The second set of images is about a man finding treasures.

Here, TimeAwareAttn and TimeAwareFFN are custom attention and feedforward neural network (FFN) modules that utilize AdaLN to integrate time embeddings into the inputs. The advantages of this approach have been demonstrated by ELLA.

The Assemblable Gated Perceiver Resampler is formulated similarly:

L = L + AttnGate · TimeAwareAttn(L,C,t), (3) L = L + FFNGate · TimeAwareFFN(L,t). (4)

In these equations, AttnGate and FFNGate are two sets of gates that regulate the feature integration. Their definitions are as follows:

AttnGate = λ · Linear(L) · A (5) FFNGate = λ · Linear(L) · F (6)

Here, λ is the gate scale, a fixed hyperparameter, and A and F are global gates. Linear(L) are separable gates.

##### 3.2 Image Generation with Multiple Conditions

Developing Text-to-Image Capability. Through ELLA’s training paradigm, we have developed a text-to-image model endowed with robust text-to-image capabilities. As illustrated in the first row of Figure 4, ELLA can generate images that strictly adhere to instructions, which forms the foundation for EMMA’s multi-modal guidance.

Selective Modular Feature Training. To bolster the stability and enhance the final performance of the training process, we have integrated several innovative design elements into the network architecture. For example, the alternating structure between the Perceiver Resampler and the Assemblable Gated Perceiver Resampler is designed to limit the feature space of the network’s intermediate layers. This prevents image information from imparting excessive prior knowledge that might compromise the text’s control and disrupt the final generation outcomes. The Assemblable Gated Perceiver Resampler includes separated gates that enable the incorporation of additional features into a few trainable embeddings.

Assembling Modules for Multi-Condition Image Generation. After establishing strong models for each individual condition, we have devised an innovative approach that enables the model to

- Table 1: Quantitative comparison for style conditioning of our proposed with other methods on the COCO validation set with four samples for every image. The best results are in bold (adapted from Ye et al. [2023]).

Style Method

Reusable to custom models

Supports native control

Multimodal prompts

Composition ability

CLIP-T ↑ CLIP-I ↑ Training from scratch

Open unCLIP 0.608 0.858 Kandinsky-2-1 0.599 0.855 Versatile Diffusion 0.587 0.830

Fine-tuning from text-to-image model

SD Image Variations 0.548 0.760 SD unCLIP 0.584 0.810 Adapters

Uni-ControlNet (Global Control) 0.506 0.736 T2I-Adapter (Style) 0.485 0.648 ControlNet Shuffle 0.421 0.616 IP-Adapter 0.588 0.828 EMMA w/o separated gates 0.572 0.834 EMMA 0.594 0.860

- Table 2: Quantitative comparison for portrait conditioned image generation. The best results are in bold.

Method IP-Adapter BLIP-Diffusion SSR-Encoder EMMA CLIP-T ↑ 49.54 56.27 58.75 64.00

DINO ↑ 27.23 26.84 25.47 29.86

amalgamate existing modules and produce images conditioned by multiple factors. As depicted in the figure, we integrate the Assemblable Gated Perceiver Resampler. Without additional training, the model can synthesize all input conditions and generate novel outputs. This demonstrates the potential for image generation without relying on a pre-existing training dataset.

The process can be mathematically expressed as: L = L +

λi · AttnGatei · TimeAwareAttn(L,Ci,ti), (7)

i

L = L +

i

λi · FFNGatei · TimeAwareFFN(L,ti). (8)

In this manner, various conditions can be applied to the image generation process without the need for further training.

#### 4 Experiments

##### 4.1 Dataset settings

Common object dataset. We also collect datasets for common objects. Following ELLA Hu et al. [2024], we filter images collected from LAION Schuhmann et al. [2022] and COYO Byeon et al. [2022] with an aesthetic score over 6 and a minimum short edge resolution of 512 pixels. We generate several random masks to provide guidance for the central object. In this way, we can train the model on a large-scale dataset.

Portrait dataset. We collect an internal dataset containing 400K images for 100K human IDs. Our EMMA targeted at portrait generation is fine-tuned on the internal dataset for 200K iterations. The test dataset uses 32 portraits and 20 prompts for each portrait, which are crawled from the Unsplash website and available under a use license.

Text-to-image generation

|[Figure 76]<br><br>Text Face Portrait<br><br>[Figure 77]|
|---|

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Image generation with text + X conditions

Condition

Generated Image

[Figure 85]

[Figure 86]

[Figure 87]

+

/

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

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Image generation with composite conditions

Conditions

Generated Image

[Figure 109]

[Figure 110]

[Figure 111]

+ +

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

- Figure 4: Visualization for our EMMA’s generalization ability under different conditions. Each column shares the same text prompts. We show three kinds of conditions. The first row shows the results when there is only the text condition. The second row shows the results under multi-modal conditions, such as text plus face conditions and text plus portrait conditions. The bottom row shows the results under composite conditions.

##### 4.2 Training Details

We train our model based on the principles established by the Stable Diffusion 1.5, with modifications to suit our experimental requirements. The model employs a half-precision floating-point (fp16) data type for efficiency. We only change the conditioner and keep all the other key components unchanged, including the pre-trained Variational Autoencoder (VAE), the noise scheduler, and the UNet.

All the experiments are done on 8 A100 GPUs. We manage a total training batch size of 256, with micro batches of 16 per GPU. We implement gradient clipping at a value of 1.0. The optimizer of choice is AdamW, which is configured with a learning rate of 0.0001. This setup includes betas of 0.9 and 0.999, an epsilon value of 1e-8, and a weight decay of 0.01. The learning rate is adjusted linearly from 10% to 100% over the course of 1000 iterations. For different conditions, we employ different feature extractors and datasets, which are detailed in the Appendix.

Person Condition Face Condition Style Condition

[Figure 126]

[Figure 127]

[Figure 128]

- Figure 5: Visualization for gate values under different conditions. The horizontal axis is the token index, while the vertical axis is the depth of the Layer. We found that the gate values show sparsity features in different layers. We also found that models trained under different conditions pay attention to different tokens, which is the basis of module composition.

##### 4.3 Personalized Story Diffusion

Given specific character information, our proposed EMMA could generate different images according to the text instruction, which makes it possible to generate results telling a story while maintaining character consistency. As shown in Figure 3, we can generate a series of images based on a given portrait following text instructions. The persons could do various actions, which benefit from the strong instruction-following abilities of EMMA.

##### 4.4 Quantitative Evaluation.

Style Conditioned Generation. Following the evaluation settings of IP-Adapter Ye et al. [2023], we evaluate the CLIP-T and CLIP-I scores of all methods on the COCO validation set. There are 5000 prompts in the validation set. We generate four images for each prompt as described in IP-Adapter Ye et al. [2023].

Portrait Generation. We collect a dataset of portraits and construct 20 human action prompts based on the ActivityNet validation set. Building on this, we tested the generation capabilities of various subject-driven image generation methods and assessed the scores using the CLIP-T score and the DINO score metrics. Results are shown in Table 2, and our proposed EMMA achieves the highest score against previous methods.

Seperable Gate mechanism. As shown in Table 1, we compare EMMA models trained under style conditions with and without separated gates. The EMMA with separated gates shows better performance, which is because such a design introduces finer control over different token embeddings. As observed in Figure 5, different tokens play different roles given specific conditions. Without the separated gates, the generated results will easily be influenced by unrelated token embeddings.

##### 4.5 Visualization

Different Conditions for Portrait Creation. We have presented a variety of portrait generation outcomes. As seen in Figure 4, our approach excels in maintaining key image elements like clothing and adheres closely to textual instructions. The top row illustrates the output of text-to-image generation, depicting a woman engaged in various activities across different settings. The middle row displays results from multi-modal image generation, where additional conditions such as facial or portrait traits yield images of a character that align with given instructions. The bottom row presents composite condition image generation, where we can produce images that follow instructions while retaining facial features from one image and portrait elements from another.

Gate value visualization. In our proposed EMMA, the gate design is a crucial module that enables free combination within our model. This design introduces an increased number of model parameters, enhancing the model’s expressive capabilities. Furthermore, we observe a distinctive distribution of token indices of the significant gated values across various models. This unique pattern of token index distribution is crucial for the adaptability of our method, enabling flexible and unrestricted model integration. The visualization result is shown in Figure 5.

#### 5 Conclusion

In this paper, we propose EMMA, a multi-modal image generation model that has the potential to revolutionize the way images are created from diverse conditions. By integrating text and additional modalities through a unique Multi-modal Feature Connector, EMMA achieves a level of fidelity and detail in image generation that is unmatched by existing methods. Its modular allows for easy adaptation to various frameworks. Additionally, EMMA could composite existing modules to produce images conditioned on multiple modalities at the same time, eliminating the need for additional training. EMMA provides a highly efficient and adaptable solution for personalized image production. In conclusion, EMMA’s innovative approach to image generation sets a new benchmark for balancing multiple input modalities. As the field of generative models continues to evolve, EMMA is poised to become a cornerstone in the development of more sophisticated and user-friendly technologies, driving the next wave of innovation in AI-driven content creation.

Limitations. The current version of EMMA is only capable of processing English prompts. In the future, we will try to implement the same algorithm in diffusion models supporting multilingual prompts.

#### References

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Yuxuan Yan, Chi Zhang, Rui Wang, Yichao Zhou, Gege Zhang, Pei Cheng, Gang Yu, and Bin Fu. Facestudio: Put your face everywhere in seconds. arXiv preprint arXiv:2312.02663, 2023.

Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992, 2023.

Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024.

Senthil Purushwalkam, Akash Gokul, Shafiq Joty, and Nikhil Naik. Bootpig: Bootstrapping zero-shot personalized image generation capabilities in pretrained diffusion models. arXiv preprint arXiv:2401.13974, 2024.

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.

Yuxuan Zhang, Jiaming Liu, Yiren Song, Rui Wang, Hao Tang, Jinpeng Yu, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. arXiv preprint arXiv:2312.16272, 2023.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479– 36494, 2022.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.

Weijia Wu, Zhuang Li, Yefei He, Mike Zheng Shou, Chunhua Shen, Lele Cheng, Yan Li, Tingting Gao, Di Zhang, and Zhongyuan Wang. Paragraph-to-image generation with information-enriched diffusion model, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model, 2024.

Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023a.

Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023.

Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023.

Yiyang Ma, Huan Yang, Wenjing Wang, Jianlong Fu, and Jiaying Liu. Unified multi-modal latent diffusion for joint subject and text conditional image generation. arXiv preprint arXiv:2303.09319, 2023b.

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. arXiv preprint arXiv:2312.04461, 2023.

Daniil Ostashev, Yuwei Fang, Sergey Tulyakov, Kfir Aberman, et al. Moa: Mixture-of-attention for subjectcontext disentanglement in personalized image generation. arXiv preprint arXiv:2404.11565, 2024.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.

Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in diffusion models for customized generation. arXiv preprint arXiv:2303.05125, 2023.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35: 25278–25294, 2022.

Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.

Minchul Kim, Anil K Jain, and Xiaoming Liu. Adaface: Quality adaptive margin for face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18750–18759, 2022.

#### A Appendix / supplemental material

##### A.1 Broader Impacts

The broader impacts of our novel multi-modal image generation model extend across various domains and societal aspects. Here, we provide a comprehensive reflection on the potential implications and ethical considerations associated with our advancements in conditional image generation.

Impact on Creative Industries: The ability to generate images from text and additional modalities can revolutionize various creative industries, from graphic design to film and gaming. While this may lead to concerns about job displacement, we anticipate that our model will primarily serve as a tool to augment the creative process, allowing professionals to achieve greater efficiency and explore new artistic frontiers.

Accessibility and Empowerment: By enabling the generation of high-fidelity images based on textual descriptions, our model can democratize the creation of visual content. This empowers individuals, including those without specialized artistic skills, to bring their ideas to life. We aim to make our technology accessible to a wide range of users, fostering creativity and innovation.

Education and Research: Our model can be a powerful educational tool, providing students and researchers with a means to visualize complex concepts and data. It can also facilitate scientific discovery by generating images that aid in the understanding of abstract or theoretical concepts, thereby enhancing learning and research outcomes.

Ethical Use of Technology: The potential for misuse of image generation technology, such as creating deepfakes or manipulating visual content for deceptive purposes, is a significant concern. We are dedicated to promoting the ethical use of our technology and are actively developing safeguards against such misuse. This includes:

- • Watermarking and Traceability: Implementing features that allow the traceability of generated images, preventing unauthorized use and ensuring accountability.
- • Ethical Guidelines: Establishing clear guidelines for the ethical use of our model, emphasizing the importance of transparency and honesty in the generation and dissemination of images.
- • Collaboration with Stakeholders: Engaging with artists, content creators, and legal experts to develop a robust framework that protects intellectual property and ensures fair use.
- • Public Awareness: Educating the public about the capabilities and limitations of our technology, promoting responsible use and critical thinking regarding the authenticity of visual content.

Environmental Considerations: We are cognizant of the environmental impact associated with the computational requirements of AI models. Our approach to feature integration and the use of time embeddings aim to reduce the computational footprint, aligning with our commitment to sustainable AI development.

In conclusion, while our multi-modal image generation model presents exciting opportunities for innovation and creativity, it also comes with a set of ethical and societal responsibilities. We are dedicated to addressing these challenges proactively, ensuring that our technology is developed and used in a manner that is beneficial, responsible, and respectful of diverse societal values.

##### A.2 Safeguards

- 1. During training, our model utilized Stable Diffusion 1.5 which is capable of detecting NFSW content. This could prevent our model from generating and learning NFSW images.
- 2. The source of our internal dataset could guarantee that there is not any NFSW content.

##### A.3 License

- 1. Stable Diffusion 1.5: The CreativeML OpenRAIL M license.
- 2. LAION: MIT License.

- 3. COYO: CC-BY-4.0 License.
- 4. ELLA: Apache-2.0 license.

- A.4 More visualization

More visualization using portrait condition is shown in Figure 6. We show portrait generation results for both males and females. They all share the same generation prompts as those in Figure 1.

The prompts are listed below:

- 1. A person sits on a checked picnic blanket in the lush, green park, surrounded by blooming wildflowers and tall trees. She is enjoying her breakfast, which consists of a toasted bagel with cream cheese and a steaming cup of coffee while reading a newspaper held delicately in her hand. The sun peeks through the branches, casting dappled shadows across the scene.
- 2. a person is deeply engrossed in her artistic endeavor within a serene park surrounded by blossoming wildflowers and towering trees. The painting, a vivid portrayal of the park’s essence, captures the interplay of light and shadow as the sun’s rays dance through the foliage above. The tranquil setting enhances her focus, as the natural beauty of the park becomes an integral part of her creation.
- 3. In the heart of a sunlit park, a person is playing guitar. Around her, vibrant pink cherry blossoms bloom profusely from their branches, creating a canopy of soft, delicate petals overhead. The lush green grass below is sprinkled with a tapestry of multi-colored wildflowers swaying gently in the breeze. A few nearby benches invite passersby to pause and enjoy the harmonious blend of nature and music.

- A.5 Adaptation to existing extensions in community.

Since our proposed EMMA does not require training the diffusion models, we can utilize commonly used community-based diffusion models trained on CLIP text features, such as the picXreal and ToonYou models, which are representative of portrait and anime styles, respectively. Furthermore, our model can even be transferred to results from models like animatediff, which are secondary developments based on diffusion models. The results on these open-source communities are illustrated in Figure 7 and Figure 8.

- A.6 More Training Details

- A.6.1 Training Settings for Different Conditions

Text features plus common object features. We train the model on our collected common object dataset for 200K iterations. The image feature extractor is CLIP-H/14, and we send both the global features and local features as the key and value features for cross-attention. The weights of this model also work as the initialization for the models conditioned on text features plus portrait features.

Text features plus style features. The model is trained on the common object dataset. The image features are also collected by CLIP-H/14 but only use the global features. The image features are then projected to 4 tokens by an extra linear layer. All the data processing procedures follow the IP-Adapter Ye et al. [2023].

Text features plus face features. The model is trained on our own collected facial dataset for 200K iterations. We first detect and use only the face area for feature processing. Then we use AdaFace Kim et al. [2022] for feature extraction and use them as the key and value features.

- A.6.2 More Ablations

Freeze Perceiver Resamplers. Freezing the Perceiver Resamplers is an essential method for constructing effective multi-modal guidance. During training, we freeze the parameters of Perceiver Resamplers to keep the text following ability. Not freezing these layers will make it impossible for the composite of different EMMA models.

Different assemble methods. Our EMMA architecture enables the fusion of models from different conditions to form new models. Since these models do not require training, how to merge them becomes a question worth designing and contemplating. In addition to the combination methods outlined in our paper, such as those in formulas 3 and 4, we have designed several groups of results. Experimental results demonstrate that our method can significantly better integrate model characteristics. The way we merge models is also significantly related to the distinct patterns in the distribution of gate values.

Object-centric mask. During training and inference, we add an object-centric mask to avoid the influence of background image information.

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

###### Figure 6: Illustration of more portrait generation examples. The first row is the condition image, and the following rows show the results of EMMA. The prompts for those examples are listed in our Appendix.

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

###### Figure 7: Generated results using EMMA and ToonYou’s U-net. The first column is the condition image, and the following columns show the corresponding generated results. The prompts are also the same as Figure 6.

### Time

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

- Figure 8: Generated results using EMMA and AnimateDiff’s U-net. The conditional image and prompts are shown in Figure 1.

