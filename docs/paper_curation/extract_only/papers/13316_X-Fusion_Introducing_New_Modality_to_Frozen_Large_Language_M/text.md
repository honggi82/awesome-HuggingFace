# arXiv:2504.20996v1[cs.CV]29Apr2025

## X-Fusion: Introducing New Modality to Frozen Large Language Models

Sicheng Mo1 Thao Nguyen2 Xun Huang3 Siddharth Srinivasan Iyer3 Yijun Li3 Yuchen Liu3 Abhishek Tandon3 Eli Shechtman3 Krishna Kumar Singh3 Yong Jae Lee2 Bolei Zhou1 Yuheng Li3 1University of California, Los Angeles 2University of Wisconsin–Madison 3Adobe Research

https://sichengmo.github.io/XFusion/

### Abstract

We propose X-Fusion, a framework that extends pretrained Large Language Models (LLMs) for multimodal tasks while preserving their language capabilities. XFusion employs a dual-tower design with modality-specific weights, keeping the LLM’s parameters frozen while integrating vision-specific information for both understanding and generation. Our experiments demonstrate that XFusion consistently outperforms alternative architectures on both image-to-text and text-to-image tasks. We find that incorporating understanding-focused data improves generation quality, reducing image data noise enhances overall performance, and feature alignment accelerates convergence for smaller models but has minimal impact on larger ones. Our findings provide valuable insights into building efficient unified multimodal models.

### 1. Introduction

Large Language Models (LLMs) [1–13] have not only achieved unprecedented capabilities for language processing tasks (e.g., conversational AI [14–20]), but also emerged as foundation tools to solve multiple languagerelated challenges (e.g., coding [21–23]). However, as humans, we do not communicate solely through text, but also extend to other modalities, such as vision. For example, instead of simply saying “This dog is cute”, we might show a photo of the dog to enhance the message: “[image] is cute” (Fig. 1). Thus, a truly versatile AI model should not only understand, reason, and generate textual output, but also must have abilities to understand, reason, and generate visual information. Moreover, these models should be unified to process and generate language and vision simultaneously, creating a more comprehensive interactive experience.

To achieve a unified model, some approaches focus on training unified vision-language models entirely from scratch using next-token prediction loss [24–27]. A recent alternative, Transfusion [28], adopts a domain-specific strategy by combining next-token prediction loss for lan-

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

is cute .

Diffusion Loss

CE Loss

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

dog is cute .

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

X-Fusion

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

#### LLM

[Figure 27]

[Figure 28]

Vision Language

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

This dog is cute

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

[BOI] [EOI] is cute

Figure 1. We introduce X-Fusion - a novel framework that adapts pretrained LLMs (e.g., LLaMA) to new modalities (e.g., vision) while retaining their language capabilities and world knowledge.

guage with diffusion loss for images. This hybrid architecture has significantly advanced performance, demonstrating greater promise than purely autoregressive approaches like Chameleon [26]. However, training such models from scratch demands immense computational resources (e.g., [28] trained on 2T tokens) and necessitates full retraining for each new modality. Given these shortcomings, another prominent research direction explores how to reuse powerful pretrained LLMs and introduce vision abilities to them [29–31], offering a more practical and efficient way for unified multimodal model training.

Research on adapting LLMs [14–18] with image understanding has shown promising results through “visual instruction tuning” [32–35]. These models typically fine-tune the LLM to align the text feature space with pretrained vision encoders (e.g., CLIP [36]), thus, might degrade original language capabilities [37, 38]. Unlike image understanding, image generation poses greater difficulties, as it demands output capabilities in a new feature space. A large body of work [29, 30, 39–43] tackled this issue by leveraging pretrained image generation models (e.g., Stable Diffusion [44]). However, as these frameworks are not

[Figure 45]

[Figure 46]

▷ Input Image:

▷ Input Image:

The image features a classic bicycle with a

A French Bulldog stands on a bicycle, donning a black leather jacket, white-

[Figure 47]

|[Figure 48]<br><br>[Figure 49]|
|---|

[Figure 50]

large round clock on its front wheel, its face

is white with black numbers and hands, indicating a time of 10:10. The bicycle

rimmed goggles, and a colorful bandana

on its head. The canine is positioned as if ready to ride, with the bicycle's handlebars

appears to be stationary and the background

is plain, allowing the bicycle and clock to be the focal point. The clock and bicycle are set

and part of the frame visible beneath it.

The backdrop features a blue surface, suggesting the photo was taken outdoors,

against a white backdrop, emphasizing their

details and shapes.

possibly at a casual gathering or event.

[Figure 51]

[Figure 52]

▷ Input Image:

▷ Input Image:

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Two women stand close together, smiling for the camera, sharing a vibrant pink umbrella to shield

Curious feline with a striped coat is peeking from behind a beige cabinet door, its eyes wide and alert. The setting is indoors, with a muted, warm-colored wall and a textured rug underfoot. The cat's fur is well-groomed, and its expression suggests a sense of playfulness and interest in the surrounding environment.

them from the rain. The woman on the left is clad in a

black coat and a pink scarf, while the woman on the right wears a white puffy coat. They appear cheerful

and dressed for a day out in the city, with a backdrop

featuring a historic building and other pedestrians navigating the wet streets.

Figure 2. Captions generated by X-Fusion demonstrate high details and strong visual alignment with the image inputs.

unified, this approach creates several limitations: limited cross-modal reasoning, restricted in-context learning, and increased error accumulation [45]. Most critically, these approaches typically require fine-tuning the LLM backbone, degrading inherited text generation ability [37, 38]. This raises fundamental research question: Is there a better way to introduce new modalities to pretrained LLMs?

Motivated by these observations, we propose X-Fusion, a new approach that addresses two challenges: (i) retaining the language abilities of the pre-trained LLM while (ii) adapting it with image generation capabilities. First, XFusion freezes all language weights, denoted as the text tower, thus preserving the inherent language abilities. Second, instead of fine-tuning the LLM, we introduce a vision tower with separate vision weights in each layer to help process visual information for the LLM (Fig 1). This approach aligns text and vision features not only at input or output level, but also at the intermediate processing level. It is worth noting that this architecture is flexible in terms of design — the vision and text towers can have asymmetric architectures. Moreover, the framework naturally extends to additional modalities (e.g., audio) by introducing dedicated modality-specific towers, ensuring efficient and scalable multimodal integration while keeping each modality independent.

While architectural innovations are crucial for integrating visual capabilities into LLMs, nowadays, understanding the impact of training data is equally important. Thus, we conduct a comprehensive set of ablation studies from a data-centric perspective to build a scalable training strategy. First, we note that creating image understanding samples without introducing noise to the images is crucial for training diffusion-based unified models. This approach enhances both image generation and understanding abilities due to more semantic visual representation learned from

clean images. Based on this key component, we further observed cross task synergy—including more image understanding data that enhances generation performance. We also investigate the effectiveness of aligning our vision features with additional pre-trained representations. Our findings suggest that this extra alignment loss term may help smaller models converge faster, but the benefit diminishes as the model size increases.

In short, our contributions are: (i) X-Fusion - a novel framework that adapts pretrained LLMs to new modalities while retaining their language capabilities. (ii) A systematic study on training strategy, offering insights for optimizing multi-modal learning. (iii) Experimental results on both image-to-text and text-to-image tasks, validating the effectiveness of the proposed architecture.

### 2. Related Work

Large Multimodal Models. The development of artificial intelligence has historically followed separate, modalityspecific paths. Large Language Models (LLMs) [1–13] exclusively processed text input and generated textual responses, while computer vision models specialized in visual content understanding (e.g., object detection [46]) or visual generation (e.g., StyleGAN [47]). The emergence of vision encoder models like CLIP [36] bridged text and image modalities, enabling two key advances: (1) Vision-language models that allow LLMs to “see” (e.g., LLaVA [32]); and (2) Conditional visual generation models that can process textual input (e.g., Stable Diffusion [44]). Current research frontiers are focusing on integrating vision and language capabilities into unified models that can both process and generate multimodal content. There are three main approaches: (1) Merging LLMs with pretrained image generation models (e.g., DreamLLM [29], GILL [30]); (2) Training LMMs via next-token prediction (e.g., Chameleon [26]); or (3)

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

An astronaut riding a camel on Mars during a sunset.

A medieval castle floating upside down over a still lake.

A sentient AI robot holding a flower, contemplating human emotions.

A ship sailing across a desert made of stars, its sails catching cosmic winds.

A futuristic warrior queen commanding an army of robotic.

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

A photo of a whimsical garden where oversized

A photo of a cunning fox in a suit and tie reading a

A photo of a red book and a yellow vase on a polished

A photo of a playful puppy chasing vibrant holographic

A crystalline rose blooming on a frozen lake.

flowers sing in the breeze.

newspaper in a city park.

table by a sunny window.

bubbles in a park.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

A graceful swan gliding effortlessly over a lake that reflects liquid silver under a shimmering moon.

A photo of a small blue hardcover book on top of a larger red hardcover book in a study

A photo of a delectable apple pie with vanilla ice cream on

A photo of a cybernetic parrot reciting futuristic tales

A photo of a panda in an

apron holding a latte art kit in a cozy kitchen.

a light blue plate.

on a digital perch.

Figure 3. Images generated by X-Fusion demonstrate high visual quality and strong text alignment with the input prompts.

Training LMMs using both diffusion and next-token prediction losses (e.g., Transfusion [28]). Following the third approach, which has achieved state-of-the-art results across modalities, we instead propose initializing from frozen LLMs rather than training from scratch, significantly reducing computational costs and retaining the LLMs knowledge.

Leveraging Pretrained LLMs. Since the success of LLMs [1–13], researchers have discovered that pretrained LLMs can serve as effective baselines for various purposes (e.g., coding [21–23]). Beyond being adapted for domain-specific tasks, prior works have shown that these pretrained models can also be fine-tuned to acquire new abilities, paving the way for a cost-effective approach to include more modalities like image understanding (e.g., LLaVA [32], Mini-GPT-4 [34]), image generation (e.g., GILL [30], DreamLLM [29], SEED-X [40]), or both image understanding and generation (e.g., Show-o [48], Emu3 [27], MetaMorph [31]). However, this method is

not without limitations—often, when fine-tuning the LLMs’ backbone, it risks compromising the original knowledge. To alleviate this shortcoming, in this work, we propose a novel method to extend vision-related abilities for LLMs while keeping all the language layers frozen, thus maintaining its original language abilities untouched. A concurrent work is LMFusion [49], with a similar high-level approach. However, they use joint attention across text and vision tokens, which is less flexible than our proposed model design.

Task-specific Weights. People have been exploring the use of specialized models tailored to different tasks in both the language [50–53] and vision domains [54–56]. The use of unshared parameters has also been explored in multi-modal settings. For instance, CLIP [36] and ImageBind [57] focus on representation learning, while other works [58– 60] emphasize vision-conditioned language model pretraining. However, these multi-modal approaches predominantly focus on visual understanding, neglecting generation

tasks. Importantly, they are typically trained from scratch, which is computationally expensive. A concurrent work, Playground-v3 [61], takes a different approach by building upon LLMs, but freezes them to perform generation tasks only, without addressing visual understanding.

### 3. Preliminaries

In this section, we provide a brief preliminary overview of the state-of-the-art recipe (proposed by [28]) for training unified models in a hybrid manner, incorporating next-token prediction for language and diffusion models for image.

##### 3.1. Language Modeling via Autoregression

LLMs are typically trained using an autoregressive modeling objective, where the joint probability of a sequence of language tokens xtxt = {xtxt1 ,xtxt2 ,...,xtxtN} is factorized as a product of conditional probabilities:

N

Pθ(xtxti | xtxt<i,c).

Pθ(xtxt | c) =

i=1

Here, c represents optional conditioning information, which could include extracted features from other modalities (e.g., image representations) or task-specific context. However, in the standard setting, LLMs are usually trained without any additional conditioning (c is absent), and the predictions depend solely on the preceding tokens xtxt<i.

The training objective for this autoregressive model is to minimize the negative log-likelihood of the observed data:

N

log Pθ(xtxti | xtxt<i,c) .

LAR = Extxt,c −

i=1

If conditioning information c is introduced (e.g., in multimodal or task-specific setups), it allows the model to extend its capabilities to tasks such as conditional text generation and cross-modal understanding.

##### 3.2. Image Modeling with Diffusion

Diffusion models have emerged as one of the most successful approaches for image generation thanks to their ability to generate high-quality images. Diffusion models are trained to gradually reverse the process of adding noise to an image, starting from a noise vector ximgT and progressively generating less noisy samples ximgT−1,ximgT−2,...,ximg0 . The goal is to produce high-quality images by learning a denoising function fθ parameterized by θ that can reverse this process.

The diffusion model training objective aims to minimize the difference between the predicted and true noise. Specifically, for each time step t, the objective is to solve the following denoising problem on the image data ximg:

LDM = Ex,ϵ∼N(0,I),t ∥ϵ − fθ(ximgt ,t,c)∥22 ,

where ximgt is the noisy image at time step t, uniformly sampled from {1,...,T}, and fθ(ximgt ,t,c) is the denoising function that predicts the noise added to ximgt conditioned on the time step t and context c (often is text prompt).

### 4. X-Fusion

In this section, we describe X-Fusion, a unified framework that adapts pretrained LLMs for vision tasks while preserving their inherent language capabilities. As illustrated in Fig. 1, X-Fusion processes both image and text inputs within a single model. To maintain the pretrained LLM’s language knowledge, we freeze its weights and introduce new trainable parameters to handle vision inputs.

Tokenizer. Text inputs are tokenized using the original LLM’s tokenizer to produce text tokens. Images are processed through a pretrained visual encoder to obtain latent representations. The encoders can be either a lowlevel feature compression model, like latent VAE [62], or CLIP [36]/DINO [63]-like semantic encoders. These latent representations can optionally be further encoded into vision tokens using an additional trainable vision layer. In most of our experiments, we use VAE from SD1.5 [62] as the pretrained visual encoder. After that, we use a trainable linear patchify layer with a patch size of 2 × 2 to further compress image features into image tokens. The combined interleaved image-text tokens are then passed into XFusion, where the trainable parameters facilitate joint optimization for both image understanding and generation.

Dual Tower. Let denote the tokenized input embeddings as: Ein = {e1,e2,...,eM}.

where token ei can be either text or vision token.

To effectively process a mixture of textual and visual modalities while retaining the original language information, each layer in X-Fusion should incorporate new trainable weights alongside a frozen language layer. Specifically, we introduce two components for each layer: A frozen text transformer block Ftxt, and a trainable vision transformer block Fimg.

Both components operate on the embeddings Ein, and their respective outputs are given by:

Htxt = Ftxt(Ein), Himg = Fimg(Ein).

Intuitively, the vision layer needs to process visual tokens conditioned on texture features for generation tasks. Additionally, it also need to extract features suitable for visual understanding tasks, ensuring that the visual features can be effectively interpreted by the frozen language layer. The outputs of the text and vision blocks are selectively combined to form the output sequence of the block:

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

[Figure 103]

Language Token Vision Token

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

|𝛾| |
|---|---|
| | |

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

(init as 0)

Language MLP

Vision MLP

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

###### Joint Attention

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Language Layer Vision Layer

Mixed Layer

Language Layer Gated Layer

Language QKV

Vision QKV

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

(a) Single Tower

(b) Gated Tower

(c) Dual Projection

(d) Dual Tower (Ours)

- Figure 4. Conceptual comparison of four model architecture baselines. Here, we illustrate how each layer processes the sequential multi-modal feature. (a) Single Tower: Directly fine-tuning pre-trained LLM. (b) Gated Layer: Duplicate Each LLM layer as the gated vision layer, (c) Dual Projection: Duplicate QKV matric and MLP layer for vision modality, (d) Dual Tower: Duplicated transformer block for vision modality.

Hout = {h1,h2,...,hM}, where

htxti , if xi ∈ xtxt, himgi , if xi ∈ ximg.

hi =

Here, htxti ∈ Htxt and himgi ∈ Himg are the outputs of the text and vision blocks, respectively, for the i-th token. In our design, we initialize each vision block by copying the parameters in the corresponding language transformed layer, which consists of one self-attention layer, one MLP layer, and two normalization layers.

Finally, each text embedding is decoded into discrete tokens through a linear classification head. The output image feature modeling can be flexible, and could involve methods such as diffusion (L2-loss), continuous autoregressive modeling (cosine regression), or discretized autoregressive modeling (cross-entropy), depending on the choice of pretrained visual encoder and design decisions. In this paper, we primarily focus on diffusion modeling. It is important to note that the best modeling approach for new modalities is case-specific and may be explored as future work.

X-Fuse (Optional) In the explanation above, we select vision tokens from the vision layers and text tokens from the text layers. Thus, in practice, we do not need to calculate the vision query tokens in the text layer, nor the text query tokens in the vision layers. This is a main design choice, as it results in the same attention FLOPs as other baseline variants shown in Fig. 4, which we will discuss in the experimental section.

Optionally, we introduce an operation called X-Fuse, which sacrifices FLOPs to improve performance. Taking the text feature as an example, we also calculate the text query features in the vision layer. To fuse the text feature from both towers, we compute:

α ∗ htxt-txti + β ∗ htxt-imgi

Here, the superscript indicates the source tower of the text features: ”txt-txt” refers to text features from the text tower, and ”txt-img” refers to text features from the vision tower. The scalars α and β are learnable parameters. A similar operation can be used to fuse image features as well.

Training. The final training objective combines the autoregressive loss (LAR) and the image denoising loss (LDM), with their respective weighting coefficients λAR and λDM:

L = λAR · LAR + λDM · LDM We choose LLaMA-3 family [17] as our as pretrained LLMs for X-Fusion and use the flow matching scheduler following Stable Diffusion 3 [64]. We set λAR = 0.2 and λDM = 1, then train with AdamW optimizer (β1 = 0.9, β2 = 0.95), a linear warm-up scheduler, a maximum learning rate of lr = 1 × e−4, and DeepSpeed Stage 2 [65] distributed training on H100 GPUs. We train with batches of 0.8M tokens for 100k steps, processing a total of 0.08T tokens for most experiments. We extend the training to 200K steps on our X-Fusion model initialized from LLaMA3.1-8B and report the evaluation results in Supplementary Section A. Task. We evaluate X-Fusion’s performance on: (i) image understanding and (ii) image generation, as these tasks reflect the connection between visual and textual modalities.

- • Image Generation: To assess image generation, we evaluate text-to-image performance. Specifically, we generate 30K images using randomly sampled prompts from the MS-COCO [66] and report image quality using FID [67].
- • Image Understanding: To assess image understanding, we evaluate image captioning (or image-to-text) performance. Specifically, we generate captions for 30K images from the MS-COCO dataset [66] and report caption quality using BLIP2-ITM [68]. We also considered additional metrics, such as CIDEr [69] and BertScore [70], however, these were either inappropriate for long captions or failed to capture meaningful changes during training. Further

text only text2img img2text Model MMLU (↑) FID (↓) CLIP (↑) BLIP(↑) LLaMA3.2-1B [17] 32.2 — — —

Single Tower 25.0 19.10 22.63 30.2 Gated Tower 32.2 24.51 21.91 14.5 Dual Projection 32.2 20.22 22.46 30.9 Dual Tower (Ours) 32.2 14.20 22.81 31.3

Table 1. Architecture design comparison. Our dual-tower approach surpasses other baselines in image generation tasks and delivers competitive performance in image understanding, while maintaining the original language capability.

analysis can be found in the Supplementary.

Data. Otherwise stated, we sample 0.08T tokens/patches from an in-house licensed dataset. We create image-caption pairs by center-cropping and resizing images to 256 × 256, and pairing them with detailed captions generated by the InternVL-2.0 26B model [35]. These pairs are then formatted for both image-to-text tasks (serving as understanding data), and text-to-image tasks (serving as generation data).

### 5. Architecture Design Choices for Adding Vision Abilities

To evaluate the effectiveness of our Dual Tower design, we compare it against three alternative transformer block variants (Fig. 4) that are designed for multimodal integration. Apart from the transformer blocks, all other components—including tokenizers, encoder, and decoder modules—are kept identical across configurations.

- • Single Tower: This simple baseline uses the original language model transformer block to process both inputs directly. Note that this is equivalent to Transfusion [28] but with pre-trained LLM as initialization (Fig. 4a).
- • Gated Tower: Inspired by [71], we duplicate the language transformer block into a trainable “gated block” (Fig. 4b), with both blocks taking the same input sequence:

Htxt = Ltxt(Ein), Hgate = Lgate(Ein).

After that, the gated block output will be added to the language transformer block output, multiplied by a learnable value γ which is initialized as 0:

Hout = Htxt + γ ∗ Hgate.

- • Dual Projection: We copy the language weights consisting of one self-attention and one MLP (Fig. 4c). However, the data flow is different from dual-tower. First of

all, based on the modality of ei, we apply separate projection matrices for query (Q), key (K), and value (V), and then a joint attention is operated on all tokens:

Hattn = attn(QKVmodality(ei)),

After that, the output feature from self-attention is passed through a modality-specific MLP:

Hout = MLPmodality(Hattn).

This variant is similar to concurrent work [49]. However, dual-tower offers more flexibility, as the vision and language layers can be designed differently.

FLOPs. Let N denote the number of text tokens and M denote the number of image tokens. For all three alternative designs, the attention complexity is O((N + M)2). In our dual-tower design, for a fair comparison, we choose not to use the X-Fuse operation. As a result, the complexity in the vision tower is O(M · (N + M)) (since we do not need query tokens for text), and the complexity in the text tower is O(N · (N + M)) (since we do not need query tokens for image). In total, this results in the same complexity of O((N + M)2). The effectiveness of X-Fuse will be presented in a later section.

Quantitative Results. Tab. 1 presents a comparison of different architectural designs, all are using pretrained LLaMA-3.2-1B [17]. Among them, the Dual Tower architecture achieves the best performance in both image generation and understanding tasks. In contrast, the Gated Layer architecture performs the weakest in both tasks, likely due to the limitations of simple addition operations. Among baselines, the Single-Tower model delivers decent performance across both tasks; however, our Dual Tower model achieves a 23% lower FID while maintaining the same number of training parameters. More importantly, the SingleTower model compromises the inherent knowledge of the original language model due to training on T2I and I2T tasks. To assess the model’s general knowledge, we evaluate it on the MMLU benchmark [72], a multiple-choice test with four answer options, using a 5-shot setting. Results show that the Single Tower model’s performance drops to 25.0, which is equivalent to chance-level performance.

Dual Tower and Dual Projection share a common insight: modality-specific operations. While their text generation capabilities are nearly same, Dual Tower outperforms in image generation. This superiority can be attributed to its flexibility: The vision layers in Dual Tower can generate new QKV representations for input text features, followed by attention operations between the image and the new text features. In contrast, Dual Projection is limited to using the original language model’s text QKV matrices for attention computation. Also, it is worth noting that conceptually, Dual Tower offers greater flexibility: while our current implementation replicates the language layer as the vision layer, the vision layer does not have to be identical to the language layer, as long as it produces outputs with the same feature dimensions. For the rest of the paper, all experiments will use Dual Tower design.

Image Quality - FID

Image-Text Alignment - CLIP

Caption Quality - BLIP

20

40

- 19

- 20

- 21

- 22

- 23

40

18

35

16

2.4 times acceleration

30

14

30

Data Ratio - T2I/I2T

20 40 60 80

100/0 66/33 50/50 33/66 0/100

25

20

20

10

15

0 20 40 60 80

0 20 40 60 80

0 20 40 60 80

Training Text-to-Image Tokens: Billon

Training Text-to-Image Tokens: Billon

Training Image-to-Text Tokens: Billon

- Figure 5. Performance of image generation and understanding at various data ratios. Increasing visual understanding data improves visual generation performance.

20 40 60 80 100

15

20

25

30

35

40

45

Image Quality - FID

20 40 60 80 100

- 19

- 20

- 21

- 22

- 23

Image-Text Alignment - CLIP

Max Noise Amount (%)

100%

75% 50% 25% 0%

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

20 40 60 80 100

0

5

10

15

20

25

30

Caption Quality - BLIP

Steps (K)

- Figure 6. Performance of image generation and understanding at various noise limits in the image-to-text samples. Providing clear images for image-to-text samples enhances visual generation and understanding simultaneously.

Text processing flexibility affects image generation but not image understanding performance.

### 6. Effect of Data Ratios and Noise on Generation and Understanding Tasks

We now turn to examine the effects of training data on XFusion’s performance. In the following section, we address two key questions: (i) How does noise level affect performance in a joint training setting? (ii) Does multitask training provide mutual benefits between tasks?

##### 6.1. Effect of Noise Amount

As a unified model, it should be able to perform both image denoising and text autoregressive tasks simultaneously on input data. However, the level of noise applied to images in image-to-text (I2T) samples remains a question. While diffusion-based image modeling benefits from noisy input for generation tasks, excessive noise can degrade visual quality and hinder image understanding. This issue was also noted by [28], where they proposed limiting the diffusion noise on I2T samples to a maximum of t = 50%T to reduce distortion for visual understanding while still can treat those noisy images as generation training data.

We argue that while this approach helps, it may not be optimal. We hypothesize that training with clean im-

ages (i.e., without adding noise to I2T samples) can lead to a stronger vision tower for image understanding, ultimately improving generation quality, although this reduces the amount of denoising data available for generation.

To validate our hypothesis,we conducted comprehensive experiments where we systematically varied the max noise level for image understanding (I2T) tasks: 0% (clean image), 25%, 50%, 75%, and 100% (normal generation setting). Throughout these experiments, we maintained a 1:2 data ratio between understanding and generation tasks. Results are provided in Fig. 6. As can be seen, generally the more noisy the images are, the more image understanding performance is degraded. Our proposed strategy (providing clean image for understanding) consistently achieves the best results for image understanding (2nd and 3rd col.). Interestingly, using clean image for understanding also helps to boost performance of image generation! (1st col.).

Following [73], we analyze the model’s behavior by conducting layer-wise feature representation experiments through linear probing. We report the Top-1 accuracy on the ImageNet dataset [74]. As shown in Fig. 8, using clean images results in better feature representation for understanding tasks compared to noisy images (Transfusion setting). Similarly, the generative features are also superior, leading to improved generation results. We provide details of linear probing experiments in Supplementary Section A.

Image Quality - FID

Image-Text Alignment - CLIP

Caption Quality - BLIP

- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20

24.0

80

| | | | | |
|---|---|---|---|---|
| |18.5| | | |
| | | | | |
| |14.7| | | |
| | | | | |
| |14.3<br><br>13.9| | | |
| |13.4 13.6| | | |
| | | | | |
| | | | | |

| |23.7 23.7| | | |
|---|---|---|---|---|
| |23.1 23.0<br><br>23.2| | | |
| |22.8| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| |75.0<br><br>77.0| | | |
|---|---|---|---|---|
| |68.5<br><br>72.5| | | |
| | | | | |
| |61.3<br><br>55.8| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

75

23.5

70

23.0

65

22.5

60

55

22.0

50

21.5

45

21.0

40

1B 3B 8B

1B 3B 8B

1B 3B 8B

X-Fusion X-Fusion With REPA

- Figure 7. Performance comparison of models of different sizes (1B, 3B, and 8B) with and without additional feature alignment loss. The effectiveness of alignment diminishes as model size increases.

4 6 8 10 12 14 16

Layer Depth

30

40

50

60

ValAccuracy(%)

Image Understanding

4 6 8 10 12 14 16

Layer Depth

50

60

70

80

Image Generation

0% - Ours

50% - Transfusion

- Figure 8. Linear Probe Results. We use the trained model as a feature extractor and train an additional linear layer for image classification on ImageNet [74]. Models trained with our training strategy constantly obtain higher feature quality.

how do their performance diverge?, and vice versa.

We observe that incorporating image understanding data (I2T) improves generation quality (T2I). As the proportion of I2T data increases while keeping total T2I data fixed, generation performance consistently improves (1st and 2nd panels). In contrast, visual generation data (T2I) does not positively impact the understanding task (I2T) (third panel). Overall, there’s an asymmetric relationship: understanding data benefits generation, but generation data does not enhance understanding. Based on our findings, we recommend a 66/33 (or 2:1) training ratio, which strikes a strong balance and optimizes performance across both tasks.

Incorporating image understanding data enhances image generation performance, while adding generation data does not impact image understanding tasks.

Using clean images for visual understanding improve performance for both tasks.

##### 6.2. Effect of Data Ratio

Along with the noise addition strategy, task data ratio is also a major factor for training unified multimodal models. To investigate the synergy between visual understanding and generation from data’s perspective, we kept the architecture unchanged and trained for 100k steps with a batch size of 0.8M tokens on a total of 0.08T tokens, varying the composition of text-to-image (T2I) and image-to-text (I2T) tasks. Specifically, we begin with the training data composed entirely of T2I tasks (100% T2I, 0% I2T, or denoted as 100/0 for short), then progressively decrease the proportion of T2I data while increasing I2T data (i.e., 66/33, 50/50, 33/66) until the dataset consists solely of I2T tasks (0/100). Results are reported at every 10k iterations.

### 7. Pretrained Representations for Vision Feature Regularization

Inspired by [73], we explore whether aligning vision features in our X-Fusion with a pretrained encoder (e.g., CLIP [36]) can improve X-Fusion’s generation and understanding. Following [73], we align the vision features from the 8th layer of X-Fusion with the penultimate features extracted from a pretrained CLIP model. Specifically, we take the vision feature Himg from the dual tower and project it using a trainable linear projection W to match the feature dimension of CLIP. The alignment loss is then computed as the cosine distance between the projected feature and the CLIP feature, encouraging our intermediate vision features to closely resemble those of CLIP. The regularization loss is defined as Lalign = 1 − cos WHimg,HCLIP , where HCLIP is the vision feature extracted from the pretrained CLIP encoder, and cos(·,·) represents the cosine similarity.

Fig. 5 illustrates the results. To investigate how training data from one task influences performance on other tasks, we plot performance metrics against the number of tokens observed for each specific task (e.g., first column, x-axis indicates how many image generation tokens are trained). In another words, we want to ask: When two models are trained on an identical number of image generation tokens, but differ in their exposure to image understanding tokens,

To evaluate effectiveness and scalability, we tested this approach on our dual-tower model with three different base model sizes: 1B, 3B, and 8B. We use λAR = 0.5 in this abla-

Pretrained DiT

###### X-Fusion

###### X-Fusion(Pretrained DiT) X-Fusion

[Figure 187]

[Figure 188]

###### ▷ Input Image:

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

a

Image depicts a home office setup with a laptop open on a desk, displaying a webpage.

a

To the right of the laptop, there's a desktop computer with a monitor showing a webpage.

The desk is cluttered with various items, including a printer, a lamp, and a few books. A window with curtains is visible in the background, and items are scattered around the room.

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

X-Fusion(Pretrained DiT)

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Image showcases a home office setup with a laptop open on a desk, displaying the screen with a photo.

a

To the right, there's a desktop computer with a prominent tower, a

monitor, a keyboard, and a mouse. Behind the desk, a curtain with

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

floral patterns is visible, suggesting a personal space.

a

On the desk, there's a mouse and a mousepad, indicating a wellequipped workspace.

- Figure 9. Qualitative comparison between pretrained DiT model, X-Fusion(DiT) and vanilla X-Fusion on image generation and understanding task. By initializing the vision tower from pretrained text-to-image diffusion model, X-Fusion(DiT) achieves stronger image generation capability and competitive performance on image understanding compared to vanilla X-Fusion.

###### ▷ Conversation 1: ▷ Conversation 2:

[Figure 239]

tion study. As Fig. 7 shows, alignment accelerates training and improves performance but with two key findings: (1) Its impact diminishes with model size, even slightly degrading 8B model performance at 100k iterations. (2) Regardless of size, alignment loss imposes a performance ceiling, likely set by the external encoder’s representational power. These findings highlight that aligning with external representations is particularly beneficial for smaller models, but the benefit may be less for larger models. Future work could explore whether using a more powerful external encoder (e.g., a larger or improved vision backbone) might further push this performance boundary.

[Figure 240]

[Figure 241]

[Figure 242]

Generate 4 books, from left to right colors are: blue, yellow, yellow, green

Generate a lakeside scene featuring a wooden bench

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

|[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]|
|---|

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

What color is the second book from the right?

How to make this park scary?

[Figure 258]

[Figure 259]

[Figure 260]

You can apply a filter to make it dark night and horrifying

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Green

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Can you change it to yellow?

Apply the edit

[Figure 271]

[Figure 272]

|[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]|
|---|

[Figure 276]

Aligning with pre-trained vision features (i.e., CLIP) improves performance for smaller models but has significantly less impact on larger ones.

[Figure 277]

[Figure 278]

Figure 10. Interactive Generation. Our X-Fusion model can follow user instructions to understand, generate, or edit images.

### 8. Extension of X-Fusion

In this section, we introduce three extensions of X-Fusion, including X-Fuse layer, transferring from pretraind diffusion models, and finetune for downstream tasks.

##### 8.2. Transfer from Pretrained Diffusion Model

While X-Fusion successfully kept its language generation capability, its image generation capability still needs to be trained from scratch. One solution is to transfer the knowledge from existing image generation models. In our dualtower design, each block in the language and vision tower processes the entire feature sequence independently, therefore allowing non-identical block designs in both towers. With this advantage, we could transfer the image generation capability from a large-scale pretrained diffusion model that uses diffusion transformers [64, 75].

##### 8.1. Effectiveness of X-Fuse Layer

So far, our study is conducted on a dual-tower architecture without the X-Fuse operation, maintaining the same FLOPs as other baseline designs. In Sec 5, we further propose the X-Fuse operation, which merges features from both towers to trade increased FLOPs for improved performance. We conduct an ablation study to evaluate this design, and the results are shown in Fig 12. As illustrated, applying the XFuse operation leads to improvements in both image generation and understanding capabilities.

We train a variation of X-Fusion using Llama3.1-8B as the language tower and an in-house pretrained text-

###### VQA Object Localization Image In/Out-painting

###### Image Editing

|[Figure 279]<br><br>[Figure 280]|
|---|

|[Figure 281]<br><br>[Figure 282]|
|---|

|[Figure 283]<br><br>[Figure 284]|
|---|

|[Figure 285]<br><br>[Figure 286]|
|---|

|[Figure 287]<br><br>[Figure 288]|
|---|

|[Figure 289]<br><br>[Figure 290]|
|---|

|[Figure 291]<br><br>[Figure 292]|
|---|

|[Figure 293]<br><br>[Figure 294]|
|---|

Q: What is flying

toward the boy?

Q: Could you

Q: Inpaint: tall

Q: Outpaint: an

Q: Change the robin

Q: Change the food

Q: Change the

X-Fusion:

Q: Could you highlight the animal that meow

[Figure 295]

[Figure 296]

[Figure 297]

highlight the snowman?

buildings that fit the environment.

animal that’s the best friend of human.

to a silver robin sculpture.

containing the most vitamins to an apple.

smaller zebra to a rhino.

Frisbee

meow?

|[Figure 298]<br><br>[Figure 299]|
|---|

X-Fusion: X-Fusion: X-Fusion: X-Fusion: X-Fusion: X-Fusion: X-Fusion:

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

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

|[Figure 321]<br><br>[Figure 322]|
|---|

|[Figure 323]<br><br>[Figure 324]|
|---|

|[Figure 325]<br><br>[Figure 326]|
|---|

|[Figure 327]<br><br>[Figure 328]|
|---|

|[Figure 329]<br><br>[Figure 330]|
|---|

|[Figure 331]<br><br>[Figure 332]|
|---|

|[Figure 333]<br><br>[Figure 334]|
|---|

Q: What

holiday is this?

###### X-Fusion:

[Figure 335]

[Figure 336]

[Figure 337]

Christmas

- Figure 11. Qualitative results of fine-tuned X-Fusion model on downstream tasks including: visual question answering (VQA), image editing, localization, and in/out-painting tasks.

3 6 9 12 15 18

Total Tokens: Billion

20

30

40

50

60

Image Quality FID

3 6 9 12 15 18

Total Tokens: Billion

10

20

30

40

50

60

70

Caption Quality BLIP

W/t X-Fuse

W X-Fuse

W X-Fuse-proj

- Figure 12. Ablation: X-Fuse layer. Our X-Fusion model with the X-Fuse layer outperforms the baseline X-Fusion model on image generation and understanding tasks.

(VQA)—using internal datasets for 20k training steps. In Figure 11, we demonstrate that our unified X-Fusion model can handle multiple tasks without creating taskspecific models or weights. We evaluated image-editing performance on publicly available datasets, including PIEBench [76] and SmartEdit [77]. Notably, X-Fusion showcased strong instruction-based editing capability when challenged to select between multiple objects—such as “smaller zebra” and “food containing the most vitamins.” Furthermore, driven by X-Fusion’s generalized capability across vision-and-language tasks, we unlock new interactive vision–language applications by enabling a single model to both generate, understand, and edit images with naturallanguage instructions from users, as illustrated in Figure 10.

to-image DiT model as its vision tower, notated as XFusion(Pretrained DiT), using the same training recipe in the previous section for 50K iterations. To align the feature dimension in both towers, we add linear projection layers within the X-Fuse layer. Figure 12 shows that this operation further enhances the model’s capability. We also empirically find that the introduction of the X-Fuse layer could accelerate the convergence when initializing from a pretrained diffusion model. Figure 9 qualitatively compares the image generation and image understanding performance between the pre-trained DiT model, X-Fusion(Pretrained DiT), and vanilla X-Fusion-8B models. X-Fusion(Pretrained DiT) obtained stronger image generation capability and on-par image understanding performance compared to the vanilla XFusion.

### 9. Conclusion

This paper introduces X-Fusion, a novel framework for adapting pretrained Large Language Models to new modalities (e.g., vision) while retaining their original language capabilities. We propose a Dual Tower architecture in which language weights remain frozen, while visual features are processed via a trainable vision tower with separate weights. Our experimental results demonstrate that this Dual Tower approach outperforms other architectural variants in both image understanding and image generation tasks. Alongside this novel architecture, we provide a systematically comprehensive set of ablation studies that offer valuable insights from a data perspective. Our findings reveal that: (i) incorporating understanding-focused data improves generation performance, (ii) reducing noise in image data enhances overall results, and (iii) feature alignment benefits primarily smaller models. We hope our paper will step forward building an unified Large Multimodal Models in a more effiecient way.

##### 8.3. Fine-tune X-Fusion

Our X-Fusion model has been pre-trained on T2I and I2T tasks and has achieved strong cross-modal performance. Can X-Fusion extend its capabilities to other downstream vision-and-language tasks? We fine-tuned our model on four tasks simultaneously—including image editing, localization, outpainting, and Visual Question Answering

### References

- [1] Alec Radford and Karthik Narasimhan. Improving language understanding by generative pre-training. 2018. 1, 2, 3
- [2] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.
- [3] Tom B. Brown and et al. Language models are few-shot learners, 2020.
- [4] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 2020.
- [5] Jack W. Rae and et al. Scaling language models: Methods, analysis & insights from training gopher, 2022.
- [6] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners, 2022.
- [7] Jordan Hoffmann and et al. Training compute-optimal large language models, 2022.
- [8] PaLM team. Palm: Scaling language modeling with pathways, 2022.
- [9] PaLM 2 team. Palm 2 technical report, 2023.
- [10] Albert Q. Jiang and et al. Mixtral of experts, 2024.
- [11] Albert Q. Jiang and et al. Mistral 7b, 2023.
- [12] DeepSeek-AI. Deepseek llm: Scaling open-source language models with longtermism, 2024.
- [13] Gemma Team. Gemma: Open models based on gemini research and technology, 2024. 1, 2, 3
- [14] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. 1
- [15] Llama 2 team. Llama 2: Open foundation and fine-tuned chat models, 2023.
- [16] Phi-3 team. Phi-3 technical report: A highly capable language model locally on your phone, 2024.
- [17] Abhimanyu Dubey and et al. The llama 3 herd of models. ArXiv, abs/2407.21783, 2024. 5, 6
- [18] Qwen team. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1
- [19] InternLM2 team. Internlm2 technical report, 2024.
- [20] DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024. 1
- [21] Mark Chen and et al. Evaluating large language models trained on code, 2021. 1, 3
- [22] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis, 2023.
- [23] Raymond Li and et al. Starcoder: may the source be with you!, 2023. 1, 3
- [24] Lili Yu and et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning, 2023. 1

- [25] Gemini Team. Gemini: A family of highly capable multimodal models, 2024.
- [26] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 1, 2
- [27] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 3
- [28] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. 2024. 1, 3, 4, 6, 7, 16
- [29] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Dreamllm: Synergistic multimodal comprehension and creation, 2024. 1, 2, 3
- [30] Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Generating images with multimodal language models. NeurIPS,

2023. 1, 2, 3

- [31] Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024. 1, 3
- [32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 1, 2, 3
- [33] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023.
- [34] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, 2024. 3
- [35] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 1, 6, 14
- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021. 1, 2, 3, 4, 8, 14
- [37] Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. Investigating the catastrophic forgetting in multimodal large language model finetuning. In Conference on Parsimony and Learning (Proceedings Track), 2023. 1, 2
- [38] Didi Zhu, Zhongyi Sun, Zexi Li, Tao Shen, Ke Yan, Shouhong Ding, Chao Wu, and Kun Kuang. Model tailor: mitigating catastrophic forgetting in multi-modal large language models. In Proceedings of the 41st International Con-

- ference on Machine Learning, ICML’24. JMLR.org, 2024. 1, 2
- [39] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. 2023. 1
- [40] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024. 3
- [41] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. arXiv preprint arXiv:2310.01218, 2023.
- [42] Zineng Tang, Ziyi Yang, Mahmoud Khademi, Yang Liu, Chenguang Zhu, and Mohit Bansal. Codi-2: In-context, interleaved, and interactive any-to-any generation. 2023.
- [43] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 1

- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 1, 2, 14
- [45] Tadas Baltruˇsaitis, Chaitanya Ahuja, and Louis-Philippe Morency. Multimodal machine learning: A survey and taxonomy, 2017. 2
- [46] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection, 2016. 2
- [47] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks,

2019. 2

- [48] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3
- [49] Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Llamafusion: Adapting pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188, 2024. 3, 6
- [50] Noam M. Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. ArXiv, abs/1701.06538, 2017. 3
- [51] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam M. Shazeer, and Z. Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. ArXiv, abs/2006.16668, 2020.
- [52] William Fedus, Barret Zoph, and Noam M. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. ArXiv, abs/2101.03961, 2021.
- [53] Nan Du and et al. Glam: Efficient scaling of language models with mixture-of-experts. 2021. 3

- [54] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andr´e Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. In Neural Information Processing Systems,

2021. 3

- [55] Yuheng Li, Yijun Li, Jingwan Lu, Eli Shechtman, Yong Jae Lee, and Krishna Kumar Singh. Collaging class-specific gans for semantic image synthesis. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 14398–14407, 2021.
- [56] Jiapeng Zhu, Ceyuan Yang, Kecheng Zheng, Yinghao Xu, Zifan Shi, and Yujun Shen. Exploring sparse moe in gans for text-conditioned image synthesis. ArXiv, abs/2309.03904,

2023. 3

- [57] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind one embedding space to bind them all. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15180–15190, 2023. 3
- [58] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, and Furu Wei. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. ArXiv, abs/2208.10442, 2022. 3
- [59] Wenhui Wang, Hangbo Bao, Li Dong, and Furu Wei. Vlmo: Unified vision-language pre-training with mixture-ofmodality-experts. ArXiv, abs/2111.02358, 2021.
- [60] Sheng Shen, Zhewei Yao, Chunyuan Li, Trevor Darrell, Kurt Keutzer, and Yuxiong He. Scaling vision-language models with sparse mixture of experts. ArXiv, abs/2303.07226,

2023. 3

- [61] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-toimage alignment with deep-fusion large language models. ArXiv, abs/2409.10695, 2024. 4
- [62] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. CVPR, 2022. 4, 16
- [63] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Q. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russ Howes, Po-Yao (Bernie) Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Huijiao Xu, Herv´e J´egou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. ArXiv, abs/2304.07193,

- 2023. 4

[64] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

- 2024. 5, 9

- [65] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters.

In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’20, page 3505–3506, New York, NY, USA, 2020. Association for Computing Machinery. 5

- [66] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 5
- [67] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 5, 14
- [68] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 5

- [69] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation, 2015. 5, 14
- [70] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert, 2020. 5, 14
- [71] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 6

- [72] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021. 6
- [73] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. ArXiv, abs/2410.06940, 2024. 7, 8, 14
- [74] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition,

2009. CVPR 2009. IEEE Conference on, pages 248–255. IEEE, 2009. 7, 8

- [75] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

- 2023. 9

[76] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Direct inversion: Boosting diffusion-based editing with 3 lines of code. arXiv preprint arXiv:2310.01506,

- 2023. 10

- [77] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition, pages 8362– 8371, 2024. 10

[78] Xinlei Chen, Zhuang Liu, Saining Xie, and Kaiming He. Deconstructing denoising diffusion models for self-supervised learning. ArXiv, abs/2401.14404, 2024. 14

## X-Fusion: Introducing New Modality to Frozen Large Language Models (Supplementary Material)

In the supplementary material, we present implementation details (Section A) and additional experiment results and analysis(Section B). We also discuss the societal impact of our method (Section C) and limitations (Section D). For sections and figures, we use numbers (e.g., Sec. 1) to refer to the main paper and capital letters (e.g., Sec. A) to refer to this supplement. We hope this document complements the main paper.

### A. Implementation Details

In this section, we present the implementation details, including the model, experiments, and the choice of evaluation metrics.

##### A.1. Model Details

Our X-Fusion uses the pre-trained VAE model with the compression ratio of 4 from Stable Diffusion [44] and follows the flow matching training from Stable Diffusion 3. During inference, we use the classifier-free guidance scale of 5.5 as we empirically find that it provides optimal visual results.

##### A.2. Caption Quality Evaluation Metric

In our main paper, we use pairs of images and long captions (generated from InternVL [35]) as training data for both text-to-image (T2I) and image-to-text (I2T) tasks. The use of long captions encourages the model to generate more detailed images, while also forcing the model captures finegrained semantic details for understanding tasks. For generation evaluation, we use standard metrics like FID [67] and CLIP scores [36]. However, we discovered that the CIDEr score [69], commonly used to evaluate caption quality in recent papers such as Chameleon and Transfusion, is not suitable for long captions. Therefore, it is not an ideal metric for monitoring our ablation results.

The CIDEr score, which relies on n-gram overlap, tends to give lower scores for long captions. This is because COCO ground truth captions are short, and even accurate long captions may have minimal n-gram overlap, leading to a lower score. Our evaluation shows that the CIDEr score between human-written COCO short captions and InternVL long captions is only 1e-5, an insignificantly low value. We also evaluate other metrics, such as BertScore [70] and CLIP score [36]. However, neither is well-suited for assessing long captions. Figure A presents qualitative examples where the reference caption is a human-written COCO ground truth caption. Among the generated captions, Sample 1 is a long and informative caption, while Sample 2

is short and of low quality. As shown in the figure, both BertScore and CLIP score assign relatively high scores to these two vastly different captions, making it difficult to gauge the quality of training progress. In contrast, the BLIP score effectively differentiates between the two captions, making it a more suitable evaluation metric.

##### A.3. Linear Probing Experiments

We followed previous literature [73, 78] for the details of the linear probing experiment settings. We use the trained Dual Tower model as the feature extractor and train an additional classification head along with a parameter-free normalization layer. Specifically, we extract the image representation from the vision tower instead of the language tower. We used a batch size of 16,384 with the Adam optimizer without weight decay and set the learning rate to 6.4 × 10−3. We report the top-1 accuracy on the validation dataset.

As the objective of the linear probing experiments is to examine the quality of the learned visual features, we consider two settings for extracting the visual representation when the model is used for visual understanding tasks and visual generation tasks. For the understanding mode, we provide clear images without noise and set the corresponding timestep to t = 0. For the generation mode, since X-Fusion is partially trained with text-to-image generation tasks, we add a text prompt preceding the noisy image and set the corresponding timestep to t = 20.

### B. Additional Experiments Results

In the main paper, we systematically explored the architectural design choices, data types, and other training factors to identify an effective approach for training a unified model with a frozen language model. Our experiments primarily focused on 1B parameter models. In this supplementary material, we extend our study by training the model with the LLaMA-3.1-8B architecture using our final training recipe. Specifically, we employ a Dual Tower design, with a 2:1 data ratio for generation and understanding tasks. Additionally, we ensure that image-to-text samples are created with cleaned images and without feature regularization from an external encoder. The model is trained with 0.8M tokens per batch for 200k iterations. Table A compares our model’s performance with other state-of-the-art unified models. Our model achieves comparable captioning and image generation quality while maintaining its original language capabilities. We also provide additional qualitative results for image generation and image understanding in Figure C and D,

Image Reference Caption

BertScore CLIP BLIP

Sample 1:

This image showcases a modern kitchen with a neutral color palette. The space features light wood cabinetry, a stainless steel sink, and an array of hanging pendant lights. A vase with flowers adds a touch of nature, and a patterned rug anchors the flooring. The room is well-lit, with natural light streaming through windows and the ceiling fans providing circulation.

- 1. This is the view of a kitchen stove, kitchen sink, counter, and cabinets.
- 2. A wooden kitchen center island with a rug in front of it.
- 3. The sink is on the island of a large kitchen.
- 4. A kitchen with a stove a sink and a counter
- 5 . A kitchen with a sink, stove, flower vase and

[Figure 338]

0.86 30.9

91.2

Sample 2: This image is a photo of kitchen.

0.89 30.9

10.4

wine rack.

- Figure A. Comparison of Different Evaluation Metrics. The BLIP score effectively differentiates between the two captions with varying levels of detail, whereas the other metrics do not.

###### Language Image Understanding Image Generation Method Base LLM MMLU ↑ COCO BLIP ↑ COCO FID ↓ Language Generation Only

Llama2 7B 45.3 - LLaMA-3.1 8B 66.7 - LLaMA-3.2-Vision 11B 66.7 82.3 InternVL2.0-26B - 81.1 Visual Generation Only

Stable Diffusion - - 9.6 Unified Models Emu-3 35.3 79.6 12.8 Janus DeepSeek1.3B - 70.8 8.5 Chameleon-7B 52.1 54.1 26.7 Show-O Phi-1.5B - - 9.2 Transfusion - - 6.7 LLaVAFusion LLaVA-Next8B - - 8.2 MetaMorph LLaMA-3.1 8B - - 11.8 X-Fusion (Ours) LLaMA-3.1 8B 66.7 80.0 11.5

Table A. Quantitative Comparison with State-of-the-Art Models. X-Fusion achieves competitive performance compared to other leading unified models. For the image understanding task, we prompt open-sourced models to generate detailed captions.

respectively. As shown in Figure C, our X-Fusion achieved great visual quality and image-text alignment. Also, Figure D illustrates the strong performance of X-Fusion for visual understanding.

### C. Social Impact

Our work presents an efficient approach to integrating new modalities into frozen LLMs, enabling both image understanding and generation. By systematically studying architectural and data-centric design choices, we provide insights that enhance the scalability and efficiency of multimodal learning, reducing computational costs and making

such models more accessible. This has potential applications in assistive AI, creative content generation, and visionlanguage understanding, benefiting areas like accessibility and education.

However, multimodal AI also raises ethical concerns, including bias in training data, misinterpretation of images, and potential misuse of generated content. To ensure responsible deployment, future work should focus on dataset fairness, robustness evaluations, and ethical safeguards. Our research contributes to the development of scalable and responsible multimodal AI, promoting more adaptable and efficient vision-language models.

▷ Input Image:

[Figure 339]

[Figure 340]

A gray tabby cat with striking yellow eyes is the focal point, lying down on a white surface with a beige bed in the background. To the left of the cat, there's a small, brown, and white Chihuahua sleeping in a pet bed. Both animals exude a calm demeanor, with the tabby cat's gaze directed towards the viewer. The scene is set

in a room with a workbench in the background, creating

a cozy and comfortable atmosphere for the pets.

[the dog is a Shiba Inu]

▷ Input Image:

[Figure 341]

[Figure 342]

An adult giraffe stands in the center of the image, its long legs firmly planted amidst a bustling urban setting. The giraffe is positioned on a road flanked by parked cars, with a red SUV visible on the far side. The sky above is overcast, casting a muted light over the scene. The surrounding environment suggests a juxtaposition of wildlife and city life, with the giraffe taking center

stage amidst the chaos of cars and buildings.

[there is no visible “red SUV”]

- Figure B. Limitations. Similar to other large multimodal models, our model is prone to hallucinating in its generated concepts.

### D. Limitation

X-Fusion is not without limitations. First, like other visionlanguage models, it is prone to hallucinations, occasionally generating inaccurate or misleading outputs (Fig. B). This may be due to the model operating in the latent space of LDM [62], which might not accurately capture fine details. A potential solution is to use a more robust latent VAE encoder. Second, although our model has shown promising results, there is still room for improvement in image quality, such as by training with higher-resolution images. Third, while X-Fusion’s vision tower can be designed to be different from its language tower to control the number of new learnable parameters, vanilla X-Fusion doubles the number of model parameters, thereby reducing training efficiency compared to Transfusion [28]. Nevertheless, this paper serves as a study that explores and analyzes various factors, including architecture and training strategies for unified MLLM training, offering valuable insights for future research.

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

A photo of royal raccoon

A photo of a robot artist

A photo of a tranquil night scene with a yellow crescent

A photo of a large gray elephant underwater on the ocean floor

A croissant-shaped spaceship traveling through a galaxy of

portraits in a grand castle.

painting in a modern art studio.

moon and serene village.

with vibrant marine life.

bakery planets.

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

A pumpkin patch with glowing pumpkin lanterns.

A photo of a black and white dog observing Picasso-style art

A mystical chameleon that shifts its colors to reflect ancient

A snowy owl perched on a branch under falling

A colossal ancient tree with a

door carved into its trunk.

in a museum.

runes.

snowflakes.

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

A photo of an astronaut on a

A smiling cat with a monocle

A polar bear wearing

A mystical garden where

A museum that only exists in

camel in an otherworldly desert.

and top hat sipping tea.

sunglasses, sipping a cold drink

enchanted fireflies illuminate

dreams, vanishing upon waking.

on an iceberg.

ancient stone pathways.

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

A giant mushroom city inhabited

A robotic hummingbird drinking

A photo of a red dog lying in a

A photo of a clear glass fish

A lone telephone booth in the

by tiny fairies and gnomes.

nectar from a mechanical flower.

cozy living room, gazing at a

tank in a tranquil underwater

middle of a desert, ringing

vibrant potted plant.

scene with vibrant fishes.

persistently.

- Figure C. Additional qualitative results of X-Fusion model. Our generation samples demonstrated good visual quality and achieved great image-text alignment.

[Figure 363]

[Figure 364]

▷ Input Image: ▷ Input Image:

[Figure 365]

[Figure 366]

Row of brown elephants is seen walking in

Two women are seen from behind, each

single file along a road, possibly in a national park, with each elephant having a

carrying a surfboard, as they walk along a

sandy beach. The ocean waves are crashing in the background, and other surfers are visible,

rider. The setting is natural, with greenery

in the background, and the elephants

some actively riding the waves. The sky is

appear calm.

overcast, suggesting a cool or humid day.

[Figure 367]

[Figure 368]

▷ Input Image: ▷ Input Image:

In the image, a black and white dog stands

[Figure 369]

In the image, a man is leading a buffalo in a

[Figure 370]

in the foreground, facing away from the

grassy field, holding a colorful umbrella to

camera, with its tail slightly raised. It is positioned in a grassy field that stretches

shield them both. The buffalo seems calm and is close to the man, with other buffaloes

towards a calm body of water, likely the sea,

in the background, indicating a herd. In the

visible in the distance. A flock of white sheep is scattered across the grass, grazing

distance, there are people, suggesting a public event or fair. The scene is lively, with

peacefully. The sky is clear, suggesting a

other individuals and activities in the

calm, sunny day.

vicinity.

[Figure 371]

[Figure 372]

▷ Input Image: ▷ Input Image:

Three elderly men are seated on park

[Figure 373]

[Figure 374]

Image displays a neatly arranged bathroom featuring a bathtub with a wooden cabinet next

benches in an outdoor setting. The man on

the left wears a light-colored shirt and dark pants, the middle man is dressed in a

to it. Above the bathtub, there is a shower head.

To the right, there is a sink with a wooden cabinet below it, and the sink appears to be

white shirt and dark pants, and the man

on the right is in a black shirt and dark pants. They all have white hair, and the

positioned next to a toilet. The tiles are light

beige, and there is a window providing natural light to the room.

setting includes a building with arched

windows in the background.

[Figure 375]

[Figure 376]

▷ Input Image: ▷ Input Image:

[Figure 377]

[Figure 378]

Two motorbikes are parked on a dark asphalt

A grey and white cat stands on a grassy area

road, captured from a high-angle perspective. The top bike is green with black seats, while the

near a curb, gazing intently to the side. Behind

the cat, there are two black cars parked next to each other. The scene suggests a quiet street

bottom bike is primarily white with black

accents. Both bikes feature luggage racks at the back, and the one on the top appears to be

setting, with the cat possibly alerted to

something out of frame.

equipped with additional protective gear.

[Figure 379]

[Figure 380]

▷ Input Image: ▷ Input Image:

In the image, a group of individuals is

[Figure 381]

[Figure 382]

In a serene park setting, a wooden bench with metal supports casts long, artistic shadows on

engaged in winter sports atop a snowy slope.

A prominent figure in a blue jumpsuit and yellow helmet is skiing downhill, while other

the pavement. The bench is situated on a

grassy area, surrounded by lush greenery. In the background, cars are parked near a curb,

snowboarders and skiers stand or move along

the slope in the background. The scene is one of outdoor activity and camaraderie, with the

and the scene is bathed in the soft glow of

sunlight, suggesting a peaceful morning.

individuals dressed in colorful winter gear,

typical for skiing and snowboarding.

▷ Input Image: ▷ Input Image:

[Figure 383]

[Figure 384]

Woman with a cheerful expression stands in

[Figure 385]

[Figure 386]

front of the grand, Gothic-style Duomo di Milano in Italy. She is holding a yellow

Woman with red hair is holding a vibrant red umbrella with a white pattern, which is open

umbrella, providing a pop of color against

the grey, overcast sky. The Duomo's intricate details and tall spires are visible in the

above her head. She appears to be walking

outdoors, possibly on a rainy day, with a blurred background that suggests an urban environment.

background, creating a striking contrast

between the historical architecture and the woman's modern attire.

- Figure D. Additional qualitative results of X-Fusion model. Our generation captions demonstrated good text quality and achieved great image-text alignment.

