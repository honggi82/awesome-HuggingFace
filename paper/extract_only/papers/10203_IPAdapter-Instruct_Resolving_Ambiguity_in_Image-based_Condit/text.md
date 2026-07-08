# arXiv:2408.03209v2[cs.CV]27Aug2024

## IPAdapter-Instruct: Resolving Ambiguity in Image-based Conditioning using Instruct Prompts

Ciara Rowles, Shimon Vainer, Dante De Nigris, Slava Elizarov, Konstantin Kutsy, Simon Donné

Unity Technologies Corresponding author: ciara.rowles@unity3d.com

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

A girl with hat and scarf

A girl with hat and scarf

VariationImage

Composition Person

<no prompt>

a phoenix

Image Variation

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

transfer

transfer

Style

Style

empty city streets

a bird house

ExtractBird

Person

[Figure 10]

[Figure 11]

Composition Style

[Figure 12]

[Figure 13]

-headphones

a fountain

A man standing in a studio

A man standing in a studio

Fig. 1: By adding a conditioning instruction prompt, IPAdapter-Instruct combines multiple image conditioning methods for easier handling and more efficient training.

Abstract. Diffusion models continuously push the boundary of stateof-the-art image generation, but the process is hard to control with any nuance: practice proves that textual prompts are inadequate for accurately describing image style or fine structural details (such as faces). ControlNet [43] and IPAdapter [39] address this shortcoming by conditioning the generative process on imagery instead, but each individual instance is limited to modeling a single conditional posterior: for practical use-cases, where multiple different posteriors are desired within the same workflow, training and using multiple adapters is cumbersome. We propose IPAdapter-Instruct, which combines natural-image conditioning with “Instruct” prompts to swap between interpretations for the same conditioning image: style transfer, object extraction, both, or something else still? IPAdapterInstruct efficiently learns multiple tasks with minimal loss in quality compared to dedicated per-task models.

Keywords: Image Generation, Diffusion Models, Image Conditioning

### 1 Introduction

The field of image generation has received fresh impetus from diffusion model theory. Modeling image generation as an iterative process that reverses the diffusion of images into pure noise, diffusion models have achieved state-of-the-art performance in image generation tasks. They have proven to be more robust to train than the previous GAN-based state-of-the-art models, and to be more widely applicable in practice: a key aspect of which is the ease of class-free conditioning of the generation on text prompts using classifier-free guidance, offering unprecedented control over the output image.

However, the common proverb “An image is worth a thousand words” is as relevant as ever: crafting text prompts that result in exactly the desired image is a non-trivial task, referred to as prompt engineering. Even when the desired image is clear in the mind of the user, it is difficult to express it in such a way that the model output replicates it accurately; expressing intent through images is often far more intuitive than through text. This concept has spurred the development of image-based conditioning such as ControlNet [43] and IPAdapter [39] approaches. From scribbles and sketches or stylistic examples to many other modalities: these methods allow users to express their intent in the image domain, where spatial information and stylistic cues can be more easily conveyed. Yet in practice every instance of these methods only offers a single way of guiding the generation based on the image prompt: when a user wishes to alternate between different ways of conditioning or combine them, each of the models needs to be trained separately. Managing the various models to correctly combine them and switch between them is complex and cumbersome.

We aim to tackle the ambiguity of conditioning with images. In the case of ControlNets, which condition the image generation on e.g. canny edge maps, or depth images, or normal images, the intention of the user is unambiguous: “generate an image such that <modality> extracted from it matches my condition image”. However, when conditioning with natural RGB images, the intention is unclear: is the user asking for generated images that have similar style? That contain the same object? That are set in the same background? That have the same composition? We propose IPAdapter-Instruct as a solution: a single model that can be textually instructed how to interpret the condition image, a task for which text instructions are well suited. Training such a model to handle multiple conditioning aspects is not only feasible, but even results in a flexible model that is on-par with task-specific models, as we show in our ablation experiments.

We discuss how to condition on this text instruction, as well as train a specific instance with five ways of interpreting the condition image: total replication (as the original IPAdapter [39]), transferring the style, replicating the composition, placing an object in a new image, and placing a person in a new image.

### 2 Related Work

##### 2.1 Diffusion Models

Diffusion models generate images conditioned on text by learning to reverse a gradual diffusion process [11, 31, 32], typically in a latent-pixel domain for its efficiency and low-level prior [6, 25]. Unfortunately, these diffusion models are costly to train: because of memory and computational requirements, and because of the need for immense volumes of training data [30]. While much work is focused on inference speed or distillation into smaller models, we consider those efforts orthogonal to our work and omit the relevant literature for brevity’s sake.

However, the text prompts that condition these models are finicky and inaccurate for conveying user intent [37], with some authors instead “translating” user prompts into model-aligned “system language” [8, 42]. Although negative prompts provide additional control, they can interfere with the original prompt or even be ignored [2], while still being restricted in their expressiveness. These difficulties imply a need for more expressive control, which we believe to be in images. Therefore, we now discuss both image-based conditioning for diffusion models and diffusion models for image-to-image translation tasks.

##### 2.2 Image-based Control for Diffusion Models

IPAdapter [39] comprises a small neural translator to project from the input image’s embedding, such as from ViT-H/14 [3] CLIP [23], onto the embedding space used by the text encoder; the network cross-attends to these novel embeddings in additional cross-attention layers similar to those of the text prompt, effectively enabling it to use an image as prompt input. IPAdapter is trained to reproduce the input image exactly — only “by coincidence” is the emergent behavior of the model to flexibly transfer the style and content of the condition image to the output images. Our IPAdapter-Instruct approach both makes the desired behavior explicit (through an “Instruct” prompt) and supervises the various behaviors directly during training (through task-specific datasets).

UnCLIP-based Approaches [24] retrain the base model to reproduce an image based on its CLIP embedding, similar to IPAdapter, but replace the text prompt with the image condition completely, attaining a single mode of control over the output. Since the entire model is retrained for this purpose, it risks catastrophic forgetting of the original model’s capabilities, and is incompatible with other residual changes to the base model such as LoRA’s [14]: we instead prefer to build off IPAdapter for keeping the base model intact.

Composer Models [15] retrain the diffusion model by conditioning it on a large set of control signals automatically extracted from the target image that intend to exhaustively describe it, with the individual conditioning modalities assumed to be unambiguous in their intention. While this method offers the highest level of control, the entire model must be retrained to achieve this (similar to unCLIP), resulting in no compatibility with other models (such as LoRAs) and no straightforward way to retrain an existing model.

InstantStyle [36] conditions the output on the style of an input image without training. To do so, the CLIP space embedding of its textual description is subtracted from that of the image to obtain a “style direction vector” in CLIP space: the text prompt cross attention layers in some blocks are then extended to also attends to this style vector. While this model is neither explicitly trained to model the conditioning process, its success is undeniable. We integrate this task (by generating a task dataset using InstantStyle) into the training process for only a small overhead, as part of the general training procedure.

ControlNet [43] functions by cloning the base model’s encoder. The clone’s inputs are replaced with the conditioning image, and its outputs are added as residuals to the original model’s hidden states [41,43]. Compared to IPAdapter, ControlNet input modalities tend to vary more widely: canny edge responses, normal maps, densely encoded poses, etc. While ControlNet models are mostly task-specific, but ControlNet++ [17] flexibly handles multiple input modalities: but as there is little ambiguity between the various conditions, they found no need for an “Instruct” prompt. We find that these models implement much more spatially localized control over the output compared to IPAdapter, while being more flexible and interoperable than the image-to-image models discussed below: ControlNets are compatible with IPAdapters, as they both keep the base model intact and only perform residual edits on the base model’s hidden states.

##### 2.3 Image-to-Image Diffusion Models and Instruct-based Control

Image-to-image models address pixel-aligned problems that largely preserve the input, such as inpainting [20,27], image restoration [5,27,38], superresolution [16], or depth estimation [29]. They shine on tasks where the output is pixel-aligned to the input and the task is clearly expressed in text form. We instead focus on tasks where the output can and should vary drastically from the input, while the task cannot be clearly expressed in text form alone (e.g. identity or style).

Palette [27] and InstructIR [5] show that a generalist model outperforms task-specific models. While the former doesn’t require instruction as its inputs are characteristic to each task, InstructIR [5] conditions on the noise model with text-based soft task routing — similar to the cross attention to our proposed instruction, although our instructions are separated well enough in CLIP space (Fig. 4) that we do not require the dedicated projection layer like InstructIR.

For more high-level image translation tasks, InstructPix2Pix [4] finetunes a base model to reinterpret its text prompt to perform e.g. object replacement or style transfer, losing compatibility with pre-trained LoRAs and ControlNets in the process. InstructPix2Pix is trained on Prompt2Prompt [9] outputs, maximally preserving the input images in a pixel-aligned way. InstructPix2Pix found multiple guidance necessary to handle the dual image and instruction conditions: IPAdapter and our proposed IPAdapterInstruct again benefit from a frozen base model, which allows a scale factor on the residual connection to control the influence of conditions.

### 3 Method

##### 3.1 Preliminaries

Diffusion models [11,31,32] iteratively generate images by gradually denoising pure noise — they are trained to reverse a diffusion process that gradually transforms an image into noise (typically white Gaussian noise). We write the forward noise process as starting from the data distribution z0 ∼ p(z) and ending with pure noise samples zT ∼ N(0,I), over the course of T time steps. The immediate forward process is formally specified as

√αt+1zt,(1 − αt+1)I , (1)

zt+1 ∼ p(zt+1|zt) = N

where αt denotes the so-called noise schedule. Given this forward process, the diffusion model is trained to model the immediate denoising distributions, noted as pˆ(zt|zt+1). During training, time steps are randomly sampled, and we directly supervise pˆ(zt|zt+1) by first sampling p(zt|z0),z0 ∼ p(z) followed by zt+1 ∼ p(zt+1|zt). Luckily, the exponential nature of additive white Gaussian noise implies a closed-form direct conditional zt ∼ p(zt|z0) which means that sampling p(zt|z0) is constant-cost in terms of t. By iteratively running the diffusion model for subsequent time steps, we can sample from the full generative model, written as

1

pˆ(zt − 1|zt), zT ∼ N(0,I). (2)

pˆ(z0|zT) =

T

As completely unconditional sampling is not very useful, diffusion models are trained to condition the generative distribution on an auxiliary input text prompt T , modeling instead p(z0|zT,T ), a technique known as Classifier-Free Guidance [12]. An IPAdapter model [39] provides further controllability by additionally conditioning the generative model on a condition image C to model p(z0|zT,T ,C). In practice, it leverages a pre-trained text-conditioned model and adds a cross-attention layer to the (projected) image condition after every prompt cross-attention layer. The base model is kept frozen to preserve its generative performance and expressivity. The condition image is encoded into the low-dimensional CLIP space [23]. This low-dimensional embedding does not contain pixel-accurate information, but rather a high-level semantic representation of the image with concepts of composition, style, subject, and object identities.

IPAdapters are trained by first sampling a data element z0 and then setting C = z0. I.e. the model is only supervised to (attempt to) reproduce the condition image exactly — even though this is not the (only) intended use-case. Although the model is never trained to produce images with a different caption than the condition image, it shows emergent capabilities to do exactly that: it tends to generate images with the same style, composition, and identities as the condition. However, it lacks any controllability of these aspects, and sometimes fails any or all of these aspects, depending on the text prompt.

##### 3.2 Instruction-guided Image Conditioning

As discussed in Sec. 2 and Sec. 3.1, existing techniques that condition on images do not provide a clear way to control the condition’s influence on the output: especially for natural condition images, there is no one way of incorporating their information and content. Instead, we let the user clarify intent through an additional “Instruct” prompt I, which we will refer to as the instruction or instruct prompt to distinguish it from the text prompt T . This means we now model the probability distribution p(z0|zT,T ,C,I). We consider the original IPAdapter [39] an instantiation of this model where I =“Reproduce everything from this image” — our approach models a wider posterior of which IPAdapter is a marginal. In this paper we discuss five distinct generation tasks for the joint IPAdapter-Instruct model:

- – Replication: variations of the condition (as IPAdapter),
- – Style: an image in the style of the condition,
- – Composition: an image with the same structure as the condition,
- – Object: an image containing the object in the condition, and
- – Face: an image containing the face of the person in the condition.

Given proper datasets and training procedures, five separate IPAdapter instances could well handle each of these tasks: but that workflow is cumbersome, both at training time and at inference time. Instead IPAdapter-Instruct is trained for all tasks simultaneously, making training the entire task set more efficient and making inference more practical. Furthermore, multi-task learning has proven to be beneficial in many contexts [40]. Sec. 3.3 outlines the model architecture in detail before Sec. 3.4 exhaustively outlines each of the above tasks and their training supervisiong.

##### 3.3 Model architecture

Our model architecture is based on the transformer projection model from IPAdapter+ [39]. We first discuss the original IPAdapter+ architecture, followed by our modifications.

IPAdapter+ adds a cross-attention layer to a projected encoding of the condition image after every cross-attention to the text prompt T , as shown in Fig. 2. The condition image is first encoded to the CLIP domain and then projected to an IPAdapter+-specific space with single linear layer before passing through a small transformer model as shown in Fig. 3.

Introducing the instruction. For IPAdapter-Instruct, we’ve modified the projection transformer model to introduce an additional attention layer at every iteration that also attends to the CLIP embedding of the instruction, as shown in Fig. 3. In this way, the model is granted the capacity to extract the relevant information (per the instruction) from the condition embedding.

|Text Prompt Features| |
|---|---|
| | |

Text prompt Cross-Attention

IPAdapter Cross-Attention

| |Text Condition Features|
|---|---|
| | |

#### z − t + 1 z − t

- Fig. 2: IPAdapter(+) and IPAdapter-Instruct both inject the conditioning in the same way, using an additional cross-attention layer after every text prompt cross attention layer. Only the way the condition is condensed into conditioning features differs, as

- discussed in Sec. 3.3 and shown in Fig. 3. The original network weights (including the text prompt cross-attention) are kept frozen and only the new cross-attention is being learned.

Encoding the instruction. We have chosen to encode the instruction using a text embedding model, embedding it into the same space the condition image is being embedded into by the original IPAdapter(+) model. Given the discrete nature of our task set, we could also learn task-specific embeddings for each of the tasks, and use those instead. However, by leveraging the powerful pretrained ViT-H/14 [3] model, we benefit from the semantic richness of the CLIP embedding space and from representing both the instruction and the condition in the same space. Although we do not investigate it in detail in this work, our intuition is that this leads to more flexible and robust instruction understanding and provides a better starting point for additional future tasks.

##### 3.4 Tasks and Dataset Generation

We build a dedicated dataset for each of the distinct tasks, which we discuss in detail below. For the instruction prompts, we generate example instruct prompts for each task using a large language model (LLM) [1]: these prompts are randomly sampled during the training procedure. To ensure that each task is well identifiable in CLIP space, we assign a keyword to each task and remove any instructions that contain a keyword from another task. Figure 4 shows a t-SNE visualization of the instruction’s embeddings for each of the tasks (as well as the “average” instruction used in the ablation studies). Example entries from all of these datasets are shown in Fig. 5.

Learned Latent Vector

Condition Image (CLIP space)

Linear

Cross-Attention

Feed-forward

Input Instruction (CLIP space)

|Cross-Attention| |
|---|---|
| | |

Linear

Feed-forward

×N

Linear

LayerNorm

IPAdapter-Instruct Condition Features

[Figure 14]

- Fig. 3: The architecture of IPAdapter+’s condition image projection (orange), and our architecture with the additional attention layers to the instruct prompt (green). All layers within the transformer are residual layers.

Fig. 4: t-SNE visualization of the CLIP embeddings of 1000 instruction prompts per task in our dataset. Note how well the tasks are delineated, despite their spread. Hard-coded prompts for the ablation study are shown as crosses.

Image replication As in IPAdapter’s training procedure, the goal is to create slight but subtle variations of the input image. As this mode replicates IPAdapter’s original behavior, we expect the same emergent capability to reuse this mode of instruction for other tasks. To create the training dataset, we use JourneyDB [22] dataset, collecting 42,000 random examples with their original text prompts. The instruction prompts are generated by querying ChatGPT4 [1] to “Generate ways to describe taking everything from an image of varying lengths, do not use the words composition, style, face or object”.

Style Preservation In style preservation, the user wants to extract only the style of the condition image and apply it to a new image — although this is not well-defined, it is taken to encompass the color scheme and general art style. Identities, layout and composition are not intended to leak into the generation. In order to create the style training dataset, we start from the ehristoforu/midjourneyimages [13] style dataset and a large-scale art dataset [28]. The condition images are sampled from the style dataset, whereas the target images are created using InstantStyle [36] with prompts from the art dataset and the condition as the style source. I.e. this mode is supervised to generate images as InstantStyle would. The instruction prompts are generated by querying ChatGPT4 to “Generate ways to describe taking the style from an image of varying lengths, do not use the words composition, object, face or everything”. This dataset consists only of 20000 examples, as InstantStyle is computationally expensive and slow.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Transfer the distinctive style [...]

Manifest the facial details [...]

Unravel pizza from the image [...]

Absorb all elements of the picture.

Retain visual [...] composition.

Native American shaman [...]

A photograph [...] human subject

a pile of chocolate chip cookies

dog

pizza

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

(e) Face extraction

(d) Composition

(c) Object extraction

(b) Style transfer

(a) Image replication

- Fig. 5: Example dataset entries for each of the five tasks. Instructions and prompts are shortened for print.

Object Extraction Here, the goal is to place the object from the condition in a new scene, preserving its identity as much as possible, similar to MagicInsert [26]. We generate this dataset based on COCO [18]: for 35000, as condition, we crop out the relevant object and (if necessary) pad with a random color. The target image is the original dataset image, for which a text prompt is provided by GPT4o [21]. The instruction prompts are generated by querying ChatGPT4 to “Generate ways to describe taking an object from an image of varying lengths, do not use the words composition, style, face or everything”. To provide more semantic information (which the typical user will have access to) to the image projection layer from Fig. 3, we replace the ‘object’ keyword in the instruction with the name of the object.

Structural Preservation Finally, we also create a dataset for structural preservation. This is intended to replicate the behavior of a scribble ControlNet [43] model, which generates images with similar canny edge profiles as the condition, but without having to explicitly generate those edge images first. For the scribble dataset, we use the CommonCanvas dataset [7] and generate new target images using the lllyasviel/sd-controlnet-scribble ControlNet from their canny edge maps and original prompts— the original images themselves are used as the condition. The instruction prompts are generated by querying ChatGPT4 to “Generate ways to describe taking the composition from an image of varying lengths, do not use the words style, object, face or everything”.

Identity Preservation As humans are extremely sensitive to facial features, we also create a dedicated dataset for face preservation using CelebA [19]. We sampled 40000 matching image pairs, and GPT4o [21] provided the text prompts. Half of the condition images were zoomed in on the face in order to focus even more on facial features. The instruction prompts are generated by querying ChatGPT4 to “Generate ways to describe taking the face or identity from an image of varying lengths, do not use the words composition, style, everything or object”.

- 3.5 Training Process Our training process follows IPAdapter [39] training procedure, using the datasets

- discussed in Sec. 3.4. The base model of choice is StableDiffusion 1.5 [25] for its excellent balance of output diversity, controllability and accessibility — it remains a staple model in the community for these reasons. The base model remains fully frozen, and we initialize the original IPAdapter elements from the IPAdapter+ weights for SD1.5 [33]. Most of the new residual layers are initialized with white noise (σ = 10−4). The final activation are zero-initialized to initialize replicating the base IPAdapter.

For the main model, we use a batch size of 512 and a learning rate of 10−6 for a total of 100000 steps. For the ablation studies, to stymie the cost, we use a batch size of 64 and a learning rate of 10−7 for a total of 100000 steps visually, the results have already converged and further training is unlikely to impact our ablation conclusions. Similar to IPAdapter [39], we find that for user inference the impact of the residual connection from the IPAdapter needs to be scaled back 20%-40%, to avoid overpowering the text cross attention.

- 4 Experiments and results

- 4.1 Datasets and metrics

Section 3 already discussed the dataset creation and training procedures in detail. For each of the task datasets, we hold out a validation set of 1000 images for quantitative and qualitative evaluations. We use the following metrics for evaluating each of the tasks:

- – CLIP-I [39]: cosine similarity between the generated image and the condition, to indicate how much information was passed from the condition to the generation.
- – CLIP-T [39]: ClipScore [10] between the generated image and the original caption of the condition, to indicate the success of the replication task.
- – CLIP-P: ClipScore [10] between the generated image and the user text prompt, to indicate how well the text prompt was followed for the nonreplication tasks.
- – CLIP Style Score (CLIP-S): cosine similarity between (a) the generated image’s CLIP embedding minus its text prompt’s CLIP embedding and (b) the condition image’s CLIP embedding minus its known text prompt’s embedding, motivated by InstantStyle [36]’s performance, to indicate style transfer success.

Style

[Figure 25]

[Figure 26]

IPAdapterInstruct

Single-task

[Figure 27]

Profileofacat Red-green

[Figure 28]

[Figure 29]

storefront

Composition

[Figure 30]

[Figure 31]

Agirlanddog

inthecouch Apink

unicorncake

IPAdapterInstruct

Single-task

[Figure 32]

[Figure 33]

Object

[Figure 34]

[Figure 35]

IPAdapterInstruct

Single-task

[Figure 36]

Awoman’sPlayingsoccer

[Figure 37]

[Figure 38]

handbag

Face

[Figure 39]

[Figure 40]

carryingadog Aclownwith

awoman

paintedface

IPAdapterInstruct

Single-task

[Figure 41]

[Figure 42]

- Fig. 6: Qualitative comparison of our proposed IPAdapter-Instruct model to taskspecific models. We omit the Replication task as we compare against the base IPAdapter+ model in Fig. 9. Performance is similar, but occasionally the joint model leaks unintended aspects — Sec. 4.4 shows that further training resolves this.

##### 4.2 Compared to task-specific models

We compare the performance of our model to that of models that were trained on each of the tasks specifically. Please refer Fig. 6 for qualitative examples of the different models, and to Tab. 1 for a quantitative overview.

We see that our proposed model is on-par with or slightly better than the single-task models, while condensing them into a single model. As Fig. 7 shows, the model trains just as fast as the single-task models, but for all tasks simultaneously. This drastically decreases total training time and cost for the entire task set — aside from simplifying inference code handling.

##### 4.3 Compared to a fixed instruction set

We ablate the choice to generate random prompts for the individual tasks rather than use a single, hard-coded, instruction prompt for each task. In the latter case, Tab. 1 shows that the random prompts are more effective than the fixed prompts — we intuit that this is because the model is forced to leverage similarities between the various tasks to some extent, when various wordings of different tasks are close-by in the prompt embedding space. Furthermore, we show in

- Fig. 8 that the effect of varying the instruction for a given task has only a minimal, if noticeable, effect, allowing for minor output exploration.

Table 1: Quantitative comparison of our IPAdapter-Instruct model to task-specific models, and to a model trained and evaluated with hardcoded queries. Very little quantitative difference is found so that we prefer the single joint model with flexible instruction queries. Higher is better for all of these metrics.

Replication Style Object Composition Faces

CLIP-I CLIP-T CLIP-S CLIP-P CLIP-I CLIP-P CLIP-I CLIP-P CLIP-I CLIP-P Single-task Model 0.9152 0.8523 1.0988 0.7615 0.7670 0.6351 0.7337 0.6424 0.5154 0.6012

IPAdapter-Instruct 0.9098 0.8542 1.1171 0.7776 0.8040 0.6402 0.7082 0.6517 0.5264 0.5976 Fixed instruct prompts 0.9086 0.8528 1.1141 0.7810 0.8029 0.6408 0.7151 0.6517 0.5159 0.6009

[Figure 43]

[Figure 44]

1.30

StyleCLIP-P

###### StyleCLIP-S

1.25

0.7

Single-task Model

1.20

IPAdapter-Instruct

0.6

1.15

0.5

1.10

0 20000 40000 60000 80000 100000 Training Step

###### (a)

(a) (b)

[Figure 45]

[Figure 46]

0.8

FacesCLIP-P

###### FacesCLIP-I

0.60

0.7

Single-task Model

IPAdapter-Instruct

0.6

0.58

0.5

0 20000 40000 60000 80000 100000 Training Step

(b)

(c) (d)

- Fig. 7: We show the evolution of validation scores in function of training step for the (a) style and (b) face tasks, and note that the joint model’s training is on par with the single-task training speed.

Fig. 8: Rewording the instruction prompt for style copy (a, b) and face transfer (c, d), drawn randomly from validation instruct prompts, shows a much more subtle impact than varying initial noise.

##### 4.4 Qualitative results for the main model

Finally, Fig. 9 provides qualitative results for our proposed approach against IPAdapter+ and InstructPix2Pix. We see that it is hard to guide IPAdapter+ towards specific use-cases, whereas our model is easily controllable through the instruction. It succeeds only when the user intent is clear: when the input image and prompt match perfectly the user requires a slight variation of the condition, and when they do not match at all the user requires style transfer. It is also suboptimal at face identity, although our method shows that the CLIP space can be pushed much further. InstructPix2Pix1 struggles when the output should differ significantly from the input (as expected) and neither can it retain facial identity, but it shines when only minor additions or edits to the input are required. We find that its general output quality is slightly lower, which we attribute to fine-tuning versus freezing the base model.

1 We evaluate only the available pretrained network [4]. Retraining InstructPix2Pix with our datasets is an option, but we consider it out of scope for this manuscript.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Recreate the image Transfer the style Extract the bird Copy composition Extract the face

A flying phoenix A bird house A bird with a rock A tabby cat A girl with hat and scarf

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

IPAdapter-InstructInstructPix2Pix[4]IPAdapter+[33]

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

Recreate the image Transfer the style Extract the zebra Copy composition Extract the face

A round house in winter A chest full of rich clothing A palm tree like a zebra A dog’s face A woman holding a sword

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

IPAdapter-InstructIPAdapter+[33]InstructPix2Pix[4]

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

###### Fig. 9: Qualitative examples of the proposed approach compared to IPAdapter+ [33,39] and InstructPix2Pix [4]. IPAdapter fails whenever the user intent is not clear, while InstructPix2Pix fails whenever the output should differ significantly from the input. Please refer to Sec. 4.4 for a detailed discussion.

Style Scribble

[Figure 87]

[Figure 88]

Face Pose

[Figure 89]

[Figure 90]

Oyster with a pearl Mountain explosion

[Figure 91]

[Figure 92]

(a) Combining IPAdapter-Instruct with a style instruction and a scribble ControlNet [34,43] to only control style without bleeding other concepts from the condition works extremely well.

Female Chef Female Firefighter

[Figure 93]

[Figure 94]

(b) Similarly, we can use IPAdapter-Instruct with a face extraction instruction in combination with a pose ControlNet [35, 43] for finegrained control over the final pose.

- Fig. 10: Examples of the interoperability of IPAdapter-Instruct with ControlNets for the underlying image diffusion model.

Finally, similar to IPAdapter but contrary to InstructPix2Pix, we retain compatibility with ControlNet and LoRA models, as seen in Fig. 10. Our model succesfully conditions the generation process as before, while the ControlNets offers pixel-precise guidance.

### 5 Conclusion, Limitations, and Future Work

In this work, we have introduced IPAdapter-Instruct to disambiguate user intent when conditioning image diffusion models on input images: by introducing an instruction prompt that specifies user intent, this joint model can be trained efficiently without losing performance. This compacts multiple adapters into a single prompt and image combination, while retaining the benefits from keeping the base diffusion model intact, such as remaining compatible with its LoRAs.

We found the main limitation to be the creation of the training datasets: it is time-consuming and strongly restricted by source data availability, but has a significant impact on the task performances. They also clearly impose biases on the conditioning model: style transfer is biased towards MidJourney [22], face extraction works best on real photos, and — most strikingly — object extraction benefits significantly from colored padding of the inputs.

We find that our model struggles most with the composition task: even when structure is preserved, it tends to bleed style. This is not unexpected: the composition task (as pixel-precise guidance) is arguably better suited for ControlNet or InstructPix2Pix. We envision future work to combine both InstructPix2Pix and IPAdapterInstruct into a single instruction-conditioned model.

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 2. Ban, Y., Wang, R., Zhou, T., Cheng, M., Gong, B., Hsieh, C.J.: Understanding the impact of negative prompts: When and how do they take effect? arXiv preprint arXiv:2406.02965 (2024)
- 3. Beaumont, R.: Vit-h/14 clip model (2023), https://huggingface.co/laion/ CLIP-ViT-H-14-laion2B-s32B-b79K [Accessed: July 15th, 2024]
- 4. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023)
- 5. Conde, M.V., Geigle, G., Timofte, R.: High-quality image restoration following human instructions. In: Proceedings of the European Conference on Computer Vision (ECCV). Springer (2024)
- 6. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for highresolution image synthesis. In: Forty-first International Conference on Machine Learning (2024)
- 7. Gokaslan, A., Cooper, A.F., Collins, J., Seguin, L., Jacobson, A., Patel, M., Frankle, J., Stephenson, C., Kuleshov, V.: Commoncanvas: Open diffusion models trained on creative-commons images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8250–8260 (2024)
- 8. Hao, Y., Chi, Z., Dong, L., Wei, F.: Optimizing prompts for text-to-image generation. Advances in Neural Information Processing Systems 36 (2024)
- 9. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. In: Proceedings of the International Conference on Learning Representations (ICLR) (2023)
- 10. Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. pp. 7514–7528 (2021)
- 11. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)
- 12. Ho, J., Salimans, T.: Classifier-free diffusion guidance. In: NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications (2021)
- 13. Hristoforu, E.: Midjourneyimages dataset (2024), https://huggingface.co/ datasets/ehristoforu/midjourney-images [Accessed: July 15th, 2024]
- 14. Hu, E.J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022)
- 15. Huang, L., Chen, D., Liu, Y., Shen, Y., Zhao, D., Zhou, J.: Composer: Creative and controllable image synthesis with composable conditions. In: International Conference on Machine Learning. pp. 13753–13773. PMLR (2023)
- 16. Li, H., Yang, Y., Chang, M., Chen, S., Feng, H., Xu, Z., Li, Q., Chen, Y.: Srdiff: Single image super-resolution with diffusion probabilistic models. Neurocomputing 479, 47–59 (2022)
- 17. Li, M., Yang, T., Kuang, H., Wu, J., Wang, Z., Xiao, X., Chen, C.: Controlnet++: Improving conditional controls with efficient consistency feedback (2024), https: //arxiv.org/abs/2404.07987

- 18. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. pp. 740–755. Springer (2014)
- 19. Liu, Z., Luo, P., Wang, X., Tang, X.: Deep learning face attributes in the wild. In: Proceedings of International Conference on Computer Vision (ICCV) (December 2015)
- 20. Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L.: Repaint: Inpainting using denoising diffusion probabilistic models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11461– 11471 (2022)
- 21. OpenAI: Gpt-4o (2024), https://openai.com/index/hello-gpt-4o/ [Accessed: July 15th, 2024]
- 22. Pan, J., Sun, K., Ge, Y., Li, H., Duan, H., Wu, X., Zhang, R., Zhou, A., Qin, Z., Wang, Y., Dai, J., Qiao, Y., Li, H.: Journeydb: A benchmark for generative image understanding (2023)
- 23. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML (2021)
- 24. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1(2), 3 (2022)
- 25. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 26. Ruiz, N., Li, Y., Wadhwa, N., Pritch, Y., Rubinstein, M., Jacobs, D.E., Fruchter, S.: Magic insert: Style-aware drag-and-drop (2024)
- 27. Saharia, C., Chan, W., Chang, H., Lee, C., Ho, J., Salimans, T., Fleet, D., Norouzi, M.: Palette: Image-to-image diffusion models. In: ACM SIGGRAPH 2022 conference proceedings. pp. 1–10 (2022)
- 28. Saleh, B., Elgammal, A.: Large-scale classification of fine-art paintings: Learning the right metric on the right feature. International Journal for Digital Art History

(2) (2016)

- 29. Saxena, S., Kar, A., Norouzi, M., Fleet, D.J.: Monocular depth estimation using diffusion models. arXiv preprint arXiv:2302.14816 (2023)
- 30. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., Schramowski, P., Kundurthy, S., Crowson, K., Schmidt, L., Kaczmarczyk, R., Jitsev, J.: Laion-5b: An open largescale dataset for training next generation image-text models (Oct 2022), http: //arxiv.org/abs/2210.08402v1
- 31. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: International conference on machine learning. pp. 2256–2265. PMLR (2015)
- 32. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. In: International Conference on Learning Representations (2020)
- 33. https://huggingface.co/h94: Pre-trained ipadapter+ weights for stable diffusion 1.5 (2024), https://huggingface.co/h94/IP-Adapter/blob/main/models/ipadapter-plus_sd15.bin [Accessed: July 15th, 2024]
- 34. https://huggingface.co/h94: Pre-trained scribble controlnet weights for stable diffusion 1.5 (2024), https://huggingface.co/lllyasviel/sd-controlnetscribble [Accessed: July 15th, 2024]

- 35. https://huggingface.co/lllyasviel: Pre-trained openpose controlnet weights for stable diffusion 1.5 (2024), https://huggingface.co/lllyasviel/control_ v11p_sd15_openpose [Accessed: July 15th, 2024]
- 36. Wang, H., Wang, Q., Bai, X., Qin, Z., Chen, A.: Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733

(2024)

- 37. Witteveen, S., Andrews, M.: Investigating prompt engineering in diffusion models. arXiv preprint arXiv:2211.15462 (2022)
- 38. Xia, B., Zhang, Y., Wang, S., Wang, Y., Wu, X., Tian, Y., Yang, W., Van Gool, L.: Diffir: Efficient diffusion model for image restoration. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13095–13105 (2023)
- 39. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arxiv:2308.06721

(2023)

- 40. Zamir, A.R., Sax, A., Shen, W., Guibas, L.J., Malik, J., Savarese, S.: Taskonomy: Disentangling task transfer learning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3712–3722 (2018)
- 41. Zavadski, D., Feiden, J.F., Rother, C.: Controlnet-xs: Designing an efficient and effective architecture for controlling text-to-image diffusion models. arXiv preprint arXiv:2312.06573 (2023)
- 42. Zhan, J., Ai, Q., Liu, Y., Pan, Y., Yao, T., Mao, J., Ma, S., Mei, T.: Prompt refinement with image pivot for text-to-image generation. arXiv preprint arXiv:2407.00247 (2024)
- 43. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3836–3847 (2023)

