## Personalized Text-to-Image Generation with Auto-Regressive Models

Kaiyue Sun1 Xian Liu2 Yao Teng1 Xihui Liu1 1The University of Hong Kong 2The Chinese University of Hong Kong

Code: https://github.com/KaiyueSun98/T2I-Personalization-with-AR

# arXiv:2504.13162v1[cs.CV]17Apr2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

|[Figure 5]|[Figure 6]|
|---|---|
|[Figure 7]|[Figure 8]|

swimming sleeping

[Figure 9]

[Figure 10]

in Acropolis in the jungle

Input images: dog in a batman suit

in front of city

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

|[Figure 15]|[Figure 16]|
|---|---|
|[Figure 17]|[Figure 18]|

mountain in the back on pink fabric

[Figure 19]

[Figure 20]

autumn leaves in the background in the jungle Eiffel Tower in the background

Input images: sneaker purple

Figure 1. Overview. Using only a few reference images (typically 3-5) of a subject (left), we fine-tune an auto-regressive model to generate personalized images of the subject in diverse contexts (right), guided by text prompts.

animals

###### Re-contextualization

Accessorization

### Abstract

to the leading diffusion-based personalization methods. The results highlight the effectiveness of auto-regressive models in personalized image generation, offering a new direction for future research in this area.

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

Personalized image synthesis has emerged as a pivotal application in text-to-image generation, enabling the creation of images featuring specific subjects in diverse contexts. While diffusion models have dominated this domain, autoregressive models, with their unified architecture for text and image modeling, remain underexplored for personalized image generation. This paper investigates the potential of optimizing auto-regressive models for personalized image synthesis, leveraging their inherent multimodal capabilities to perform this task. We propose a two-stage training strategy that combines optimization of text embeddings and fine-tuning of transformer layers. Our experiments on the auto-regressive model demonstrate that this method achieves comparable subject fidelity and prompt following

Training Images

Training Images

dog on the beach

wearing pink glasses in a chef outfit

dog

wearing a santa hat

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

### 1. Introduction

The rapid advancement of text-to-image generation models has revolutionized the field of computer vision, enabling the creation of highly realistic and diverse images from textual descriptions. Among the various applications of these models, personalized image synthesis—generating images of specific subjects in new contexts—has garnered significant attention. This capability is particularly valuable for applications in digital art, advertising, and virtual reality, where the ability to seamlessly integrate personalized con-

in the snow on top of pink fabric

on top of a wooden floor

in a purple wizard outfit

in a police outfit wearing a yellow shirt

in a firefighter outfit

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Training Images

wearing a yellow shirt in a chef outfit

dog on the beach

in the snow on top of pink fabric

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

in a firefighter outfit

with a city in the background

with a mountain in the background

wearing a rainbow scarf

wearing a black top hat and a monocle

with a blue house in the background

on a cobblestone street

tent into diverse scenes is crucial.

While diffusion models have been at the forefront of personalized image generation, auto-regressive models, which employ a unified architecture for text and image modeling, have not been extensively explored for this task. Autoregressive models [18, 22, 30, 37, 41] have demonstrated remarkable success in text-to-image generation by predicting image tokens sequentially. However, their potential for personalized image synthesis remains largely untapped. This paper aims to investigate the adaptation of auto-regressive models for personalized image generation.

We propose a novel two-stage training strategy that firstly optimizes text embeddings and then fine-tunes transformer layers together. Our experiments on the LuminamGPT 7B model [18] show that this approach outperforms existing optimization-based techniques like Textual inversion [6] and shows comparable performance with Dreambooth [26] in terms of subject fidelity and prompt following. The results underscore the potential of auto-regressive models in personalized image generation and pave the way for future research in this domain.

This work explores the potential of auto-regressive models for personalized image synthesis, adapting them to meet the specific demands of text-to-image generation. Our findings suggest that auto-regressive models, when properly optimized, can achieve competitive performance in personalized image generation, offering a promising alternative to diffusion-based approaches.

### 2. Related Work

#### 2.1. Personalized Image Synthesis.

Personalized image synthesis focuses on generating images of specific subjects in novel contexts while seamlessly integrating them into the new scene. The advent of diffusion models [11, 28, 29] has significantly advanced text-guided image generation, offering remarkable flexibility to personalize text-to-image diffusion models [1, 4, 6, 7, 9, 10, 13– 15, 17, 20, 21, 23, 24, 26, 27, 33, 38, 40]. These models can customize single or multiple subjects using a few reference images or even a single image. Current diffusion-based personalization models can be broadly divided into two categories: optimization-based models and tuning-free models. Optimization-based models [1, 6, 9, 10, 14, 15, 26, 33], such as Textual Inversion [6] and DreamBooth [26], optimize embeddings of text tokens or fine-tune the backbone to implant a new subject into the model’s output domain. In contrast, tuning-free models [4, 7, 13, 17, 20, 21, 24, 27, 38, 40] pre-train a separate encoder to extract the subject features, enabling zero-shot personalized image generation without per-subject optimization.

Recent advancements in unified multimodal models [8, 31, 39] have expanded personalized image synthesis into

broader multimodal tasks. These unified generative models typically rely on carefully curated datasets for instruction tuning or large-scale training. However, when personalized image generation is included as one of many tasks for these models, its performance often lags behind models specially designed for this task.

- 2.2. Auto-regressive Image Generation.

Auto-regressive models generate data through next token prediction. Early auto-regressive image generation models like PixelCNN [34] and PixelRNN [35] operated at the pixel level, but their computational demands became prohibitive due to the high dimensionality of images. The introduction of VQVAE [36] addressed this by compressing images into discrete tokens and enabled a two-stage approach: image tokenization followed by auto-regressive modeling of token distributions. This framework revitalized auto-regressive image generation.

Building on this, models like DALL-E [22] and CogView [5] extended auto-regressive methods to text-toimage generation. They compress images into tokens with an autoencoder. By concatenating the text tokens and image tokens, they train a decoder-only transformer to predict the next image token. Parti [41] and LlamaGen [30] employ separate text encoders to inject the text features into the auto-regressive decoder.

Chameleon [32] advances the framework by enabling mixed-modal generation of interleaved text and image sequences using a unified transformer architecture, eliminating the need for separate text encoders. LuminamGPT [18] inherits Chameleon’s tokenizers and trains an auto-regressive transformer from scratch, enabling image generation and other vision-centric tasks. These developments highlight the versatility and scalability of autoregressive models in multimodal generation.

Diffusion models and auto-regressive models differ fundamentally in their paradigms. Diffusion models excel in iterative refinement, while auto-regressive models employ a unified architecture for text and image modeling, making them inherently compatible with multimodal tasks. Despite this, auto-regressive models specifically tailored for personalized image generation remain unexplored. This work aims to investigate the potential of optimizing auto-regressive models for personalized image generation.

- 3. Preliminaries

3.1. Personalizing Text-to-Image Models via Optimization

Textual Inversion. Textual Inversion [6] proposes a personalization method by creating a new “pseudo-word” (e.g., S∗) within the text embedding space of a text-to-image diffusion model. Using just 3-5 images of a specific subject

provided by the user, this method optimizes the embedding vector corresponding to the pseudo-word to represent that subject. This word can then be used to compose natural language prompts, such as “a S∗ on the beach”, to generate personalized images in novel contexts.

DreamBooth. Instead of a “pseudo-word”, DreamBooth [26] opts to optimize a unique identifier “[V]” that precedes the subject class (for example, “a [V] cat/dog/toy on the beach”). This approach helps to link the prior knowledge of the class with the subject, thereby reducing training time. However, using the class name can lead to a gradual loss of the model’s broader semantic knowledge during fine-tuning, a phenomenon known as language drift. To address this issue, a class-specific prior preservation loss is introduced to retain the model’s ability to generate diverse instances of the class.

These optimization-based approaches are implemented and proved to be effective on text-to-image diffusion models. They can effectively perform various personalization tasks, including subject recontextualization, text-guided view synthesis, and artistic rendering.

In this paper, we explore the adaptation of these optimization-based personalization techniques to autoregressive models and offer insights into the finetuning of auto-regressive models.

#### 3.2. Text-to-Image Generation via Next-Token Prediction

Auto-regressive text-to-image models generate images in three steps. First, a tokenizer converts the input text into a sequence of discrete tokens, which are transformed into vector embeddings. These text embeddings, denoted as c, are then fed into an auto-regressive transformer that outputs logits lt. The logits are converted into probabilities where the next image token xt is sampled. The newly sampled token is concatenated with the preceding tokens to predict the subsequent token. Finally, an image decoder translates the complete sequence of tokens x = (x1,x2,...,xT) into image pixels.

Training objective. During training, the autoregressive transformer models the conditional probability p(xt | x1,x2,...,xt−1,c) of the sequential tokens using the standard next-token prediction objective. We denote x1∼t−1 = {x1,x2,...,xt−1} , the model predicts the next token xt ∈ V , where V denotes the vocabulary. The loss function f for a single prediction can be written as follows:

L(θ) = f (yt,pθ (xt | x1∼t−1,c)), (1) pθ (xt | x1∼t−1,c) = Softmax(lt), (2)

where L(θ) is the loss, parameterized by the model parameters θ and loss function f. In image generation, we predict the tokens from the image split of the total vocabulary. yt

represents the target label of the next token, which is derived by tokenizing the ground-truth image associated with the input text. f is cross-entropy loss.

Classifier-free guidance. Most auto-regressive text-toimage models adopt classifier-free guidance (CFG) to enhance the quality of generated images. When generating an image token, the logits processed by CFG, denoted as lt cfg are formulated as follows:

′

′

t, (3)

lt cfg = s(lt − l

t) + l

where lt represents the original logits that are conditioned on the complete input text; l

′

t refers to the contextindependent logits, which are independent of any prior token. The variable s denotes the guidance scale used in classifier-free guidance.

In this work, we aim to personalize the auto-regressive model by finding a unique identifier of a subject and finetuning the transformer layers using a set of reference images of that subject.

### 4. Method

Personalizing a text-to-image diffusion model generally involves two strategies. The first strategy is to associate a unique text embedding with the subject. This text embedding can either represent the subject as a whole or serve as an adjective describing the subject class. However, because the number of parameters for a text embedding is limited, personalized images often struggle to capture all the essential features of the subject. To effectively embed the subject’s appearance in the model, fine-tuning of the model parameters is usually required. Figure 2 shows the overview of our fine-tuning strategy.

In this section, we present our method for personalizing an auto-regressive model and explain the rationale behind our choices.

#### 4.1. Optimizing Text Embeddings

We generally follow the DreamBooth [26] approach to optimize a text embedding for a specific subject. We introduce a placeholder word [V], to represent the unique identifier of the new subject we wish to learn. The input text that includes the identifier [V] and the subject class name is then converted to tokens. We replace the embedding associated with the token for [V] with a new randomly initialized embedding, denoted as v∗. With a small set of reference images (e.g. 3-5) of the subject in various backgrounds or poses, we optimize v∗ based on the cross-entropy loss defined in Equation 1. For the input text, we use the templates provided by Textual Inversion [6], which contain neural context such as “A photo of [V] [class name]”, “A rendition of [V] [class name]”. Our optimization goal can thus

Input Prompt

Target Images

||||||
|---|---|---|---|---|

“A photo of [V] dog”

Tokenizer

Ground-Truth Image Tokens

|Text Tokens| | |
|---|---|---|
| |48, 291, 63, ∗, 87| |

301 352 565 789 473 862 981 642 319

301 406 565 789 355 790 981 800 319

Token Embeddings

Generated Image Tokens

v48 v63 v87 v291 v*

|[Figure 59]|
|---|

…

…

Transformer Layers

[Figure 60]

[Figure 61]

#### 4.2. Fine-tuning Transformer Layers

Input Prompt

Target Images

|[Figure 62]|[Figure 63]|[Figure 64]|[Figure 65]|[Figure 66]|
|---|---|---|---|---|

We conduct experiments using the Lumina-mGPT 7B model [18]. We have observed that the generated images fail to accurately replicate the reference subject if we optimize the text embeddings only. Additionally, when optimizing the text embeddings on a single data point, the model does not overfit; instead, after a slight decrease in the loss, it stabilizes around a specific level. Given the limited capacity of text embeddings, fine-tuning the auto-regressive transformer becomes necessary to effectively implant the subject into the model’s output domain.

“A photo of [V] dog”

Tokenizer

[Figure 67]

Ground-Truth Image Tokens

|Text Tokens| | |
|---|---|---|
| |48, 291, 63, ∗, 87| |

301 352 565 789 473 862 981 642 319

301 406 565 789 355 790 981 800 319

Token Embeddings

Generated Image Tokens

v48 v63 v87 v291 v*

|[Figure 68]|
|---|

…

…

Transformer Layers

[Figure 69]

[Figure 70]

Two-stage training. DreamBooth [26] fine-tunes the layers conditioned on the text embeddings and the diffusion UNet simultaneously. In our experiments, we find that when finetuning the text embeddings and transformer layers together, the text embeddings cannot get fully trained. If we revert to the original transformer layers during inference, the text embeddings alone fail to convey any meaningful content. To address this issue, we devise a two-stage training strategy. In the first stage, we fully optimize the text embeddings, and in the second stage, we fine-tune the transformer layers to maximize the subject fidelity. This two-stage approach is mutually beneficial: the first stage stabilizes the training and reduces the effort needed in the second stage, while the second stage compensates for any defects from the first stage due to its inherent limitations.

- Figure 2. Overview of Fine-tuning. We fine-tune a text-to-image auto-regressive model using 3-5 input images, each paired with a text prompt that includes a unique identifier and the subject class name (e.g., “A photo of [V] dog”). The process involves two stages: first, we fine-tune the text embedding for the identifier [V], and second, we additionally fine-tune the transformer layers to enhance the model’s performance.

be defined as follows:

f (yt,pθ (xt | x1∼t−1,c)),∀t (4)

v∗ = arg min

v

It is expected to encourage embedding v∗ to learn the common features in the reference images while discard elements that are unique to each image, such as the background.

#### 4.3. Implementation details

Per-image tokens Textual Inversion [6] introduces a scheme that inserts a unique token for each reference image in the text template. Given n reference images, the method associates each image with a unique placeholder Si, where i = 1,...n. The embeddings corresponding to these placeholders are optimized to capture the distinct features of each individual image, while minimizing the influence of non-common elements on the universal placeholder [V]. Textual Inversion [6] composes input prompts using the format “A photo of [V] [class name] with Si”. We have observed that auto-regressive models are sensitive to specific wording. When prompts use different prepositions or structures during inference, the quality of the generated images tends to decrease. To avoid this issue, we use the format “A photo of [V] [class name] Si”, where Si can represent any form of phrase.

Unless otherwise specified, we maintain the original hyperparameter settings of Lumina-mGPT [18]. The text embeddings are initialized randomly. Our experiments use a batch size of 1, which corresponds to approximately 8 textimage pairs. In the first stage, we set the learning rate for optimizing the text embeddings to 0.01, training for ∼1500 steps. In the second stage, ∼ 70 steps with a learning rate of 5 × 10−6 is enough for full fine-tuning. We find that these training parameters work well for most cases. For some common subjects (e.g., dog, cat), satisfactory results can be achieved with fewer training steps in both stages. However, for some challenging subjects, more training steps in the second stage are required to obtain better results. The first stage takes about 15 minutes, and the second stage takes only 2 minutes on a single H100 GPU.

Additionally, we find that introducing the per-image tokens in all training prompts affects the quality of the generated images when we attempt to reconstruct the subject using the prompt “A photo of [V] [class name]”. However, removing all per-image tokens makes it more challenging to eliminate content that is irrelevant to the subject. To strike a balance, we opt for a compromise strategy that incorporates both formats. This strategy improves the robustness of the subject learning process. In our implementation, we use a 1:1 ratio for the two formats.

### 5. Experiments

#### 5.1. Dataset and Evaluation

We evaluate our model’s personalization capability on Dreambench [26], which provides a dataset consisting of 30 subjects, each with 4-6 images. These subjects are divided into two groups: 21 objects and 9 live subjects/pets. Each subject is tested on 25 prompts, which include scenarios such as re-contextualization, accessorization, and prop-

erty modification. Their purpose is to assess whether the key features of the subject can be preserved under different semantic modifications while the generated image adheres to the prompt. Following Dreambench [26], we employ DINO [2] and CLIP-I [25] to assess subject fidelity, and CLIP-T [25] to measure the prompt following. For evaluation, we generate images using a fixed CFG value of 4.0 and an image top-k value of 2000.

Subject fidelity is computed as follows: for each generated image of a subject, we calculate the average similarity between that image and all reference images. This process is repeated for all 25 images generated from the 25 prompts, and the results are averaged to obtain the subject fidelity score for that subject. The overall subject fidelity score for the model is then derived by averaging the scores across all 30 subjects in the dataset. For prompt following, the score for each generated image is calculated by computing the cosine similarity between CLIP [25] embeddings of the image and the corresponding prompt.

#### 5.2. Quantitative Results

Table 1 presents the evaluation results of various models on Dreambench [26]. By fine-tuning the auto-regressive model of Lumina-mGPT [18] using our method, it outperforms Textual Inversion [6], Re-Imagen [3], and zeroshot BLIP-Diffusion [17] in both subject fidelity (Dino and CLIP-I) and prompt following (CLIP-T). Additionally, it achieves comparable results to stable diffusion-based DreamBooth [26] and fine-tuned BLIP-Diffusion [17] in DINO. Notably, our method achieves the highest prompt following score among all the models listed. These findings demonstrate that auto-regressive models can be finetuned to incorporate new concepts without compromising their original generation capabilities.

Method DINO ↑ CLIP-I ↑ CLIP-T ↑ Real Images 0.774 0.885 N/A

Textual Inversion [6] 0.569 0.780 0.255 Re-Imagen [3] 0.600 0.740 0.270 DreamBooth (Stable Diffusion) [26] 0.668 0.803 0.305 DreamBooth (Imagen) [26] 0.696 0.812 0.306 BLIP-Diffusion (zero-shot) [17] 0.594 0.779 0.300 BLIP-Diffusion (fine-tune) [17] 0.670 0.805 0.302

Ours (Lumina-mGPT [18]) 0.671 0.785 0.314

Table 1. Quantitative results comparison on Dreambench [26]. We show subject fidelity (DINO, CLIP-I) and prompt following (CLIP-T) scores across different models. For all three metrics, the scores range from 0 to 1, where a higher score indicates better performance. The bold values highlight the highest score achieved.

#### 5.3. Qualitative Results

In Figure 3 and Figure 4, we present qualitative generation results of our model. The re-contextualization examples

demonstrate the model’s ability to accurately reproduce the subject’s appearance while merging it into the new backgrounds. Furthermore, the model can accurately modify the color and shape properties of the subject, even in challenging cases such as “cube-shaped”. This indicates that the model not only learns new concepts, but also effectively decomposes and recomposes them with its prior knowledge. In accessorization examples, the model can seamlessly integrate subjects with outfits, demonstrating its ability to understand the structure and meaning of the subject rather than merely replicating its appearance. These results validate the model’s strong ability to follow prompts and maintain high subject fidelity, as reflected in the quantitative evaluation.

#### 5.4. Ablation Studies

Class-Prior ablation. As noted in DreamBooth [26], finetuning all layers of a diffusion model can lead to issues of language drift [16, 19] and reduced output diversity. This occurs because fine-tuning a pre-trained diffusion model on a small set of images causes it gradually forget how to generate subjects of the same class as the target subject, while also reducing the diversity of its outputs. However, when fine-tuning all layers of the transformer within an autoregressive model, we do not observe these issues. As illustrated in Figure 5, we present four examples, each in a row. After fine-tuning the model with the images in the first column using the prompt “A photo of [V] [class name]”, the model can still generate diverse and distinct images when prompted with “A photo of a [class name]” or “A photo of a [class name] in the jungle”, showing no significant resemblance to the training images. We investigate the impact of removing the class name in the training prompt by fine-tuning with prompts such as “A photo of [V]”. The quantitative results, shown in Table 2, indicate that both methods yield comparable results, with fine-tuning using the class name performing slightly better. This suggests that auto-regressive models are more robust in preserving their pre-trained knowledge, eliminating the need for including the Prior Preservation Loss during personalization.

Method DINO ↑ CLIP-I ↑ CLIP-T ↑

Ours (Lumina-mGPT) w/o class name 0.668 0.785 0.312 Ours (Lumina-mGPT) 0.671 0.785 0.314

Table 2. Comparison of subject fidelity and prompt following scores. Training with and without subject class names in prompts.

Fine-tuning transformer with LoRA. We fine-tune the transformer layers using LoRA [12] with different LoRA ranks, ranging from 16 to 256. We turn on LoRA for one layer every N layers from the total 32 self-attention layers, where N can be set to 1, 2, or 4. The LoRA layers are attached to the projection matrices of the query, key, and

objects

Re-contextualization Property Modification

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

Training Images

Training Images

backpack on the beach

can purple

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

on top of a wooden floor

with a wheat field in the background

in the jungle

wet red cube shaped

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Training Images

vase on the beach

shiny purple

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

on top of pink fabric

with a tree and autumn leaves in the background

in the jungle

wet red cube shaped

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

Training Images

Training Images

glasses in the jungle

in the snow

on top of pink fabric sneaker purple

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

with a blue house in the background

on top of green grass with sunflowers

on a cobblestone street

with a city in the background

wet red cube shaped

- Figure 3. Qualitative results. We generate images of personalized objects to showcase the generative capabilities of re-contextualization and property modification.

value features. After optimizing the unique identifier for each subject in the first stage, we finetune the transformer layers with LoRA for about 100 ∼ 170 steps in the second stage. During inference, we maintain a fixed CFG value of

- 4.0 and set the image top-k to 2000.

animals

###### Re-contextualization

Accessorization

When the number of trainable layers is fixed, increasing the LoRA rank leads to better subject fidelity. Similarly, for a given rank, training more layers enhances subject fidelity. Generally, more trainable parameters correlate with improved subject fidelity, but there are exceptions. For instance, increasing the rank or trainable parameters but training fewer layers can degrade subject fidelity. This suggests that training more layers has a greater impact on performance than purely increasing the LoRA rank or trainable parameters. It is also indicated in Table 3 that fine-tuning

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

Training Images

Training Images

dog on the beach

wearing pink glasses in a chef outfit

dog

wearing a santa hat

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

As shown in Table 3, subject fidelity improves with higher LoRA ranks (r↑) and more trainable layers (N↓). Although there is a trade-off between subject fidelity and prompt following, prompt following remains strong across different training configurations.

in the snow on top of pink fabric

on top of a wooden floor

in a purple wizard outfit

in a police outfit wearing a yellow shirt

in a firefighter outfit

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

6

[Figure 141]

Training Images

wearing a yellow shirt in a chef outfit

dog on the beach

in the snow on top of pink fabric

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

in a firefighter outfit

with a city in the background

with a mountain in the background

wearing a rainbow scarf

wearing a black top hat and a monocle

with a blue house in the background

on a cobblestone street

objects

Re-contextualization Property Modification

Training Images

Training Images

backpack on the beach

can purple

on top of a wooden floor

with a wheat field in the background

in the jungle

wet red cube shaped

Training Images

vase on the beach

shiny purple

on top of pink fabric

with a tree and autumn leaves in the background

in the jungle

wet red cube shaped

[Figure 182]

Training Images

Training Images

glasses in the jungle

in the snow

on top of pink fabric sneaker purple

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

with a blue house in the background

on top of green grass with sunflowers

on a cobblestone street

with a city in the background

wet red cube shaped

animals

###### Re-contextualization

Accessorization

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

Training Images

Training Images

dog on the beach

wearing pink glasses in a chef outfit

dog

wearing a santa hat

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

in the snow on top of pink fabric

on top of a wooden floor

in a purple wizard outfit

in a police outfit wearing a yellow shirt

in a firefighter outfit

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Training Images

wearing a yellow shirt in a chef outfit

dog on the beach

in the snow on top of pink fabric

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

in a firefighter outfit

with a city in the background

with a mountain in the background

wearing a rainbow scarf

wearing a black top hat and a monocle

with a blue house in the background

on a cobblestone street

- Figure 4. Qualitative results. We generate images of personalized animals to showcase the generative capabilities of re-contextualization and accessorization.

performance by optimizing only the text embeddings, without fine-tuning the transformer layers. The quantitative results in Table 4 show that subject fidelity is significantly lower when training only the embeddings than training both the embeddings and the transformer layers. Since the dimension of a single token embedding is limited to 4096, training only the embeddings is insufficient to capture the complex appearance of the subject. Therefore, fine-tuning the full model is necessary for optimal performance.

Every N # Trainable Layer Parameters DINO ↑ CLIP-I ↑ CLIP-T ↑

Rank

- N = 1 12.6M 0.657 0.781 0.316

- N = 2 6.3M 0.640 0.773 0.316

r = 16

N = 4 3.2M 0.630 0.769 0.319

- N = 1 50.4M 0.657 0.778 0.312

- N = 2 25.2M 0.656 0.777 0.315

r = 64

N = 4 12.6M 0.638 0.769 0.317

- N = 1 201.4M 0.668 0.785 0.312

- N = 2 100.7M 0.654 0.775 0.315

r = 256

N = 4 50.4M 0.640 0.772 0.317 Full fine-tune 1610.6M 0.671 0.785 0.314

The qualitative comparison is shown in Figure 6. The second and third columns display images generated by models fine-tuned only on text embeddings. Although these models are able to capture basic elements of the subject, like color, shape, and texture, they struggle to reproduce those fine details that define the subject’s identity, often resulting in unrealistic or distorted patterns. For example, in the third and fourth rows, the boot has two tips, and the dog appears with two bodies. Additionally, due to the limited capacity of text embeddings, the subject sometimes fails to appear as the main focus in the image, as seen in the first row where the backpack is nearly invisible.

- Table 3. Quantitative results comparison on Dreambench [26] under different training configurations of Lumina-mGPT [18]. We compare subject fidelity (DINO, CLIP-I) and prompt following (CLIP-T) scores across different LoRA ranks r and varying number of training layers. N denotes the interval at which trainable layers are applied, with one trainable layer every N layers.

parameters within all the transformer layers results in better performance than fine-tuning with LoRA. Optimizing only text embeddings. We assess the model’s

###### Training images Free-form prompts with the class name

###### Training images Fine-tune Embeddings only + transformer layers

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

“A backpack floating on top of water”

“A [v] backpack on top of a wooden floor”

“A [v] backpack in the jungle”

“A [v] backpack in the jungle”

Training Images: backpack “A backpack on the beach” “A backpack in the jungle”

Training Images: backpack

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

“A sneaker floating on top of water”

“A [v] toy on a cobblestone street”

Training Images: sneaker “A sneaker” “A sneaker in the jungle”

“A [v] toy on the beach ” “A [v] toy on the beach ”

Training Images: toy

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

“A [v] boot on a cobblestone street”

Training Images: cat “A cat” “A cat in the jungle” “A cat wearing a santa hat

“A [v] boot on the beach” “A [v] boot on the beach”

Training Images: cat

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

“A [v] dog on top of a wooden floor”

“A [v] dog with a mountain in the background”

“A [v] dog with a mountain in the background”

Training Images: dog “A dog” “A dog in the jungle” “A dog wearing a santa hat”

Training Images: dog

- Figure 5. Preservation of class semantic priors. Fine-tuning auto-regressive models with a set of reference images does not result in language drift or reduced output diversity. The first column displays the training images, the next three columns show images generated using free-form prompts that include the specific subject class name.

- Figure 6. Qualitative comparison of fine-tuning strategies: text embeddings only vs. text embeddings and transformer layers. The first column shows the input images. The second and third columns display images generated by models fine-tuned only on text embeddings, while the fourth column shows results from models fine-tuned on both text embeddings and transformer layers.

[Figure 273]

[Figure 274]

[Figure 275]

“A [v] dog from top view”

“A [v] dog from bottom view”

“A [v] dog from back view”

[Figure 276]

[Figure 277]

[Figure 278]

Training Images: candle

[Figure 279]

[Figure 280]

[Figure 281]

“A [v] candle in the style of Van Gogh”

“A pencil sketch of a [v] candle”

“A chinese ink painting of a [v] candle”

[Figure 282]

[Figure 283]

[Figure 284]

“A [v] dog panda” “A [v] dog lion” “A [v] dog hippo”

Training images Novel view synthesis

Art renditions

Property modification

[Figure 285]

[Figure 286]

[Figure 287]

Training Images: dog

[Figure 288]

[Figure 289]

[Figure 290]

Training Images: dog

[Figure 291]

[Figure 292]

[Figure 293]

|[Figure 294]|
|---|

|[Figure 295]|
|---|

[Figure 296]

[Figure 297]

[Figure 298]

|[Figure 299]|
|---|

[Figure 300]

[Figure 301]

[Figure 302]

|[Figure 303]|
|---|

|[Figure 304]|
|---|

|[Figure 305]|
|---|

- Figure 7. Failure cases of various applications. This figure presents applications of novel view synthesis, artistic renditions, and property modifications. The first column displays the input images. Symbols in the bottom-left corner indicate whether the generated images accurately reflect the prompts. For the failure cases, we include comparison images generated with the same prompt but without the “[v]” identifier, allowing us to assess the model’s inherent capabilities alongside the effects of fine-tuning.

Method DINO ↑ CLIP-I ↑ CLIP-T ↑

Ours (Lumina-mGPT) w/o transformer layers 0.601 0.754 0.320 Ours (Lumina-mGPT) w/ transformer layers 0.671 0.785 0.314

- Table 4. Comparison of subject fidelity and prompt following score. Training with and without transformer layers.
- 5.5. Limitations

Our model demonstrates capabilities in recontextualization, accessorization, and simple property modifications such as color and shape. Beyond these, we aim to explore additional applications. However, results indicate that auto-regressive models struggle with complex scenarios requiring extensive prior knowledge or deep integration of multiple concepts. Failure cases are illustrated in Figure 7. A symbol in the bottom-left corner of each image indicates whether the generated image aligns with the prompt. For comparison, we also generate an image using the same prompt, but without the “[v]” identifier, which is positioned in the top-left corner.

Novel view synthesis. The top row of Figure 7 showcases attempts to generate images of the dog from novel viewpoints. The model is tasked with generating perspectives of the specific dog it has never encountered (e.g., top, bottom, or back views). While the model extrapolates class knowl-

edge to somewhat successfully generate top and back views, it fails to produce a correct bottom view due to overfitting to the input images. Although the back view is generated, flaws in the dog’s rear make the result appear unnatural.

Art renditions. The middle row of Figure 7 displays the model’s attempts to produce artistic renditions of the object. While the model successfully transfers a candle into Van Gogh’s style, it fails with the other two styles. In the pencil sketch and Chinese ink painting examples, the model recognizes the styles but misinterprets them as objects rather than applying them to the subject. This issue may stem from the token-based nature of the auto-regressive model, which associates the identifier “[v]” with image tokens of the input subject. Artistic renditions, however, requires replacing the subject with entirely different tokens. As a result, the model circumvents the challenge by reflecting the keywords as objects rather than applying the intended styles.

Property modification. Figure 3 illustrates some cases of property modification. The model performs well with simple tasks, such as altering color or shape. However, it struggles with more complex feature combinations. For example, in the top-left corners of the bottom row of Figure 7, the model fails to merge features of two animals (e.g., “a dog panda” or “a dog lion”). Instead, it generates the second animal mentioned in the prompt only, completely omitting the “dog”. When the prompt aims to combine features of the specific Chow Chow dog with another species, the model partially incorporates elements of the dog but fails to create a cohesive fusion. This limitation mirrors the challenges seen in artistic renditions, highlighting the model’s lack of flexibility in handling advanced feature integration.

### 6. Conclusion

In this paper, we demonstrate the potential of autoregressive models for personalized image synthesis through a two-stage training strategy, first optimizing text embedding and then fine-tuning transformer. Our approach achieves comparable subject fidelity and prompt following to the state-of-the-art stable diffusion-based methods such as DreamBooth [26]. However, auto-regressive models are slow, taking minutes to generate images, and the fine-tuning also requires 15-20 minutes, limiting real-time applicability. Additionally, the ability to create personalized images raises ethical concerns, such as misuse for misleading content, a challenge common to all generative models. Future work should focus on improving efficiency, addressing ethical risks, and ensuring responsible advancements in personalized generative technologies.

### References

[1] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia, 2023. 2

- [2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 5
- [3] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-augmented text-to-image generator. arXiv preprint arXiv:2209.14491, 2022. 5
- [4] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. In NIPS, 2023. 2
- [5] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. In NIPS, 2021. 2
- [6] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2, 3, 4, 5
- [7] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG), 2023. 2
- [8] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024. 2
- [9] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. In NIPS, 2023. 2
- [10] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In ICCV, 2023. 2
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NIPS, 2020. 2
- [12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 5
- [13] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 2
- [14] Jimyeong Kim, Jungwon Park, and Wonjong Rhee. Selectively informative description can reduce undesired embedding entanglements in text-to-image personalization. In CVPR, 2024. 2
- [15] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 2
- [16] Jason Lee, Kyunghyun Cho, and Douwe Kiela. Countering language drift via visual grounding. In EMNLP, 2019. 5

- [17] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. 2023. 2, 5
- [18] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 2, 4, 5, 7
- [19] Yuchen Lu, Soumye Singhal, Florian Strub, Aaron Courville, and Olivier Pietquin. Countering language drift with seeded iterated learning. In International Conference on Machine Learning, pages 6437–6447. PMLR, 2020. 5
- [20] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subjectdiffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In ACM SIGGRAPH,

2024. 2

- [21] Zhendong Mao, Mengqi Huang, Fei Ding, Mingcong Liu, Qian He, and Yongdong Zhang. Realcustom++: Representing images as real-word for real-time customization. arXiv preprint arXiv:2408.09744, 2024. 2
- [22] OpenAI. Dalle-2, 2023. 2
- [23] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992, 2023. 2
- [24] Maitreya Patel, Sangmin Jung, Chitta Baral, and Yezhou Yang. λ-eclipse: Multi-concept personalized text-to-image diffusion models by leveraging clip latent space. arXiv preprint arXiv:2402.05195, 2024. 2
- [25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 5
- [26] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2, 3, 4, 5, 7, 9
- [27] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. In CVPR, 2024. 2
- [28] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NIPS, 2019. 2
- [29] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 2
- [30] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 2
- [31] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In CVPR, 2024. 2

- [32] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2
- [33] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH, 2023. 2
- [34] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. In NIPS, 2016. 2
- [35] A¨aron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In ICML,

2016. 2

- [36] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. In NIPS, 2017. 2
- [37] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 2
- [38] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In ICCV, 2023. 2
- [39] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 2
- [40] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arxiv:2308.06721,

2023. 2

- [41] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022. 2

