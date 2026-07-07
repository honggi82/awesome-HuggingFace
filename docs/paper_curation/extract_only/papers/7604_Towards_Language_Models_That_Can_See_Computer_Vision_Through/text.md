# arXiv:2306.16410v1[cs.CL]28Jun2023

## Towards Language Models That Can See: Computer Vision Through the LENS of Natural Language

[Figure 1]

#### William Berrios† Gautam Mittal†§ Tristan Thrush†§

Douwe Kiela†§ Amanpreet Singh† †Contextual AI; §Stanford University

### Abstract

[Figure 2]

We propose LENS , a modular approach for tackling computer vision problems by leveraging the power of large language models (LLMs). Our system uses a language model to reason over outputs from a set of independent and highly descriptive vision modules that provide exhaustive information about an image. We evaluate the approach on pure computer vision settings such as zero- and few-shot object recognition, as well as on vision and language problems. LENS can be applied to any off-the-shelf LLM and we find that the LLMs with LENS perform highly competitively with much bigger and much more sophisticated systems, without any multimodal training whatsoever. We open-source our code at https://github.com/ContextualAI/lens and provide an interactive demo1.

[Figure 3]

[Figure 4]

Trained from scratch Pretrained and frozen

Architecture Data

Source

Output Text

Surfing

Cross - Modality Encoder

[Figure 5]

[Figure 6]

[Figure 7]

Image Encoder

Text Encoder

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Q: What is the dog doing?

COCO Visual Genome VQA GQA Visual7W ...

[Figure 17]

Millions of paired image/text samples

Output Text

Surfing

XATTN Layer

[Figure 18]

FrozenLLM

[Figure 19]

LMBlock

XATTN Layer

[Figure 20]

[Figure 21]

LMBlock

[Figure 22]

Perceiver

[Figure 23]

Image Encoder

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Q: What is the dog doing?

image-video text pairs

[Figure 33]

###### M3W

43M webpages ~2B samples

Old-Style Pretraining

Flamingo

[Figure 34]

Output Text

Surfing

[Figure 35]

FrozenLLM

[Figure 36]

Q- Former FC Layer

[Figure 37]

[Figure 38]

Image Encoder

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Q: What is the dog doing?

###### COCO Visual Genome CC12M SBU LAION-400M

[Figure 48]

115M images + synthetic captions

BLIP-2

Output Text

Surfing

Frozen LLM

[Figure 49]

Objects Captions

Attributes

[Figure 50]

Visual Descriptors

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Q: What is the dog doing?

[Figure 60]

No additional pre-training data

[Figure 61]

LENS (Ours)

[Figure 62]

(a) Multimodal Pretraining (b) No Multimodal Pretraining

Figure 1: Comparison of approaches for aligning visual and language modalities: (a) Multimodal pretraining using a paired or web dataset, and (b) LENS , a pretraining-free method that can be applied to any off-the-shelf LLM without the need for additional multimodal datasets. Unlike LENS, prior methods are computationally intensive and require joint alignment pretraining on large multimodal datasets to perform visual tasks.

[Figure 63]

1https://lens.contextual.ai/

Correspondence to lens@contextual.ai.

### 1 Introduction

In recent years, Large Language Models (LLMs) have revolutionized natural language understanding, showcasing remarkable capabilities in semantic understanding, question answering and text generation [42, 10, 6], especially in zero-shot and few-shot settings. Several approaches have been proposed for employing LLMs on vision-related tasks as shown in Fig. 1(a). One technique involves training a vision encoder to represent each image as a sequence of continuous embeddings, enabling comprehension by the LLM [55]. Another employs a frozen vision encoder that has been trained contrastively while introducing new layers into the frozen LLM that are subsequently trained from scratch [47, 52, 2]. Furthermore, an alternative approach suggests using both a frozen vision encoder (pre-trained contrastively) and a frozen LLM aligning them by training a lightweight transformer [35, 27, 63].

While we have seen progress in the above research directions, the computational expense associated with the additional pretraining stage(s) still remains a challenge. Besides, large corpora of datasets containing images/videos and text are needed for aligning visual and language modalities on top of an existing LLM. An example of this is Flamingo [2] which introduces new cross-attention layers into an LLM to incorporate visual features, which are then pre-trained from scratch. Despite using a pretrained image encoder [5] and a pretrained frozen LLM [19], the multimodal pre-training stage still demands a staggering 2 billion image-text pairs along with 43 million webpages [64, 32], an undertaking that can last for approximately 15 days. Instead, as shown in Fig. 1(b), we can extract information from visual inputs and generate detailed textual representations (e.g. tags, attributes, actions, relationships, among others) using a diverse set of “vision modules” and then feed this information directly to the LLM avoiding the additional multimodal pretraining.

[Figure 64]

We introduce LENS (Large Language Models ENnhanced to See) a modular approach that leverages a LLM as the “reasoning module” and operates over independent “vision modules”. In the LENS approach, we first extract rich textual information using pretrained vision modules such as contrastive models [47, 50, 13, 5] and image-captioning models[34, 35]. Subsequently, the text is fed into the LLM allowing it to perform object recognition and vision and language (V&L) tasks. LENS eliminates the need for extra multimodal pretraining stages or data, bridging the gap between the modalities at zero cost. By integrating LENS, we get a model which works across domains out of the box without any additional cross-domain pretraining [24, 20, 2, 35]. Furthermore, this integration enables us to leverage the latest advancements in both computer vision and natural language processing out of the box, maximizing the benefits derived from these fields.

In summary, our contributions are as follows:

- • We propose LENS, a modular approach that addresses computer vision tasks by harnessing the few-shot, in-context learning abilities of language models through natural language descriptions of visual inputs.
- • LENS enables any off-the-shelf LLM to have visual capabilities without requiring auxiliary training or data. We utilize frozen LLMs to handle object recognition and visual reasoning tasks without the need for additional vision-and-language alignment or multimodal data.
- • Experimental results demonstrate that our approach achieves zero-shot performance that is competitive with or superior to end-to-end jointly pre-trained models like Kosmos and Flamingo.

### 2 Related Work

#### 2.1 Large Language Models capabilities

LLMs have demonstrated remarkable abilities for natural language understanding and reasoning. GPT-3 [6] is a notable example of such models, which can accurately solve complex tasks including translation, natural language inference, and common sense reasoning in a zero-shot or few-shot setting. Recently, more powerful versions such as GPT-3.5 and GPT-4 [45] were designed to understand, interact and generate human-like responses [43]. These models are also known for their ability to perform a wide variety of tasks by showing a few examples in the prompt [6]. Recent efforts have also been made to develop open-source LLMs that can compete with GPT-3, such as BLOOM [56],

OPT [62], LLaMA [54], FlanT5 [7] among others. However, all these models cannot directly solve tasks that require reasoning from a visual input stimulus. Our work leverages these LLMs as frozen language models and provides them with textual information obtained from the “vision modules” allowing them to perform object recognition and V&L tasks.

- 2.2 Contrastive Models for Solving Vision and Language tasks

Foundation models such as [47, 50, 23, 13, 61] have demonstrated the ability to specify any visual concept based on an external vocabulary without the restriction of classes or labels presented in supervised models. However, previous work [49], [26] has shown that these contrastive models are unable to directly solve tasks in zero or few shot settings. To address this, [51] proposed a method using CLIP in VQA tasks by converting questions to a mask template that CLIP can answer, but their approach required fine-tuning for extending the model’s capabilities to other tasks such as visual entailment [58]. In our work, we propose to leverage the capabilities of contrastive models and combine them with a crowdsourced open-source vocabulary to assign tags and attributes present in the image, which combined with frozen LLM can solve diverse V&L tasks.

- 2.3 Large Language Models for Vision Applications

- 2.3.1 Image Captioning

The field of image captioning has seen a significant surge in recent years, with the objective of generating natural language descriptions for images. To this end, various deep-learning models have been proposed. Notably, the recent models include BLIP[34] and BLIP-2[35], which achieve great performance on NoCaps[1] and COCO[36]. Concurrently, ChatGPT has been leveraged to generate richer visual descriptions, along with BLIP-2 [44]. In another work, Socratic Models [60] and Visual Clues [59] also use textual data to bridge the domain gap between vision-language models and language models. In particular, Visual Clues constructs a semantic representation of an image using structured textual prompts that include image tags, object attributes/locations, and captions. This approach leverages the GPT-3 large language model to generate image captions. Our work is inspired by Visual Clues, but instead of generating captions, we aim to utilize the raw compelling vision information with a frozen LLM in order to solve vision tasks.

- 2.3.2 Vision and Language tasks

LLMs can be leveraged in multiple ways in order to perform V&L task, these are mainly divided in two sections.

Multimodal pretraining. These approaches align vision and language modalities in different ways. For example, Tsimpoukelli et al. [55], opts to finetune only the visual encoder and generate embeddings that are fed into a frozen LLM. Others, such as Flamingo [2], train additional crossattention layers for alignment. Works like BLIP2[34] and Mini-GPT4[63] reduce the size of extra layers and pretrained lightweight modules while freezing the vision encoder. However, in all cases, joint alignment of vision and language requires significant computing resources and training data, making it challenging to leverage state-of-the-art LLMs. Additionally, these approaches can hinder the reasoning abilities for which LLMs are renowned.

Language-driven Modular Alignment: These approaches couple LLMs with different modules in order to align the visual and language modalities. Concurrent work Guo et al. [17] uses off-the-shelf LLMs for solving pure Visual Question Answering tasks like VQA 2.0 [16] and OK-VQA [39]. In contrast, LENS extends the capacity of LLM to also solve object recognition tasks and also it does not involve any question-guided information extraction. Another work, PromptCap [21], trains a question-aware captioning model using synthesized examples with GPT-3 for solving VQA tasks. In contrast, LENS leverages “vision modules” without requiring any additional pre-training stage. Likewise, ViperGPT [53] also leverages black box LLMs such as Instruct GPT and Codex to achieve great results on different VQA benchmarks but heavily relies on BLIP2 which needs extra training rounds of multimodal pre-training. Additionally, all the above methods rely on a “top-down” approach in which attention mechanisms are driven by nonvisual or task-specific contexts. However, our proposed approach differs from these methods as we utilize a “bottom-up” [3] approach. Our method does not involve any question-guided information extraction, which is a more challenging

task. Despite this, LENS achieves notable results that are comparable to these question-aware models.

Tag Module

[Figure 65]

pug dog, shirt, surf table

Domain Vocabulary

....

Contrastive Model

[Figure 66]

[Figure 67]

###### Input Image

Attribute Module

[Figure 68]

blue shirt, gray surf table, brown dog,

Attribute Vocabulary

....

Contrastive Model

[Figure 69]

Intensive Captioning Module

- Caption 1
- Caption 2 Caption n

[Figure 70]

Image Captioning Model

[Figure 71]

###### Text Prompt

Tags: surfing, surfer, pug, water dog, pug dog Attributes:

- - pug dog which has double-coated fur.
- - pug dog which has small to medium size.
- - pug dog which has four-legged mammal.
- - surfboard which has leash attached to the tail.
- - pug dog which has short, smooth coat. Captions:
- - a dog is sitting on a surf board in the water.
- - a dog wearing a blue shift on a surfboard in the water.
- - a small dog riding a surfboard in the water.
- - a dog wearing a blue shirt is on a surfboard in the ocean
- - a dog in a wet suit surfing on a surfboard.

LargeLanguageModel

###### Output text

The dog is a pug and it is wearing a blue shirt.

Question: What is the breed's dog? and What is the

[Figure 72]

dog wearing? Answer:

- Figure 2: The LENS framework. LENS executes computer vision and visual reasoning tasks through a frozen LLM and a set of “vision modules”. LENS leverages these vision modules to retrieve a textual description for an image which is used by the “reasoning module” (LLM) to generate a response for a given query.

### 3 Method

[Figure 73]

We present a novel framework called LENS (Fig. 2), which aims to enhance the capabilities of frozen LLMs by enabling them to handle vision as well as vision-and-language tasks on top of their existing natural language understanding capabilities. In contrast to existing approaches, LENS provides a unified framework that facilitates the operation of a LLM’s “reasoning module" on textual data extracted from a set of independent and highly descriptive “vision modules”. More importantly, it eliminates the computational overhead of aligning the visual domain with the text domain through additional joint pretraining on multimodal data, a requirement in prior work for solving V&L tasks. [2, 35, 63, 15, 27].

To summarize, given an image I, we leverage the vision modules to extract all conceivable textual information T that can describe the image, encompassing objects, attributes and captions, without limiting it to specific task instructions. Subsequently, a frozen LLM can process the generic prompts T concatenated with task-specific prompts, and perform object recognition or visual reasoning tasks. In this section, we introduce the essentials of the “vision modules", outline the main components of LENS, and then discuss the prompt design.

#### 3.1 Visual Vocabularies

For LENS, visual vocabularies act as a bridge to convert an image into textual information which can then be handled by an existing LLM. We develop vocabularies for common objects and attributes.

Tags: To create a diverse and comprehensive tag vocabulary for a contrastive model’s image tagging, we collect tags from various sources. These include multiple image classification datasets such as [48, 33, 8, 46, 41, 4, 57, 28], object detection and semantic segmentation datasets [18, 36, 31] along with the visual genome dataset [29].

Attributes: Following the methodology presented in Menon & Vondrick [40], we employ a large language model, GPT-3, to generate descriptions of the visual characteristics that differentiate each object category within our object vocabulary.

LENS CLIP

Datasets L14- FlanT5XL L14- FlanT5XXL H14- FlanT5XL H14- FlanT5XXL L14 H14 Pets [46] 90.1 92.0 92.6 92.4 87.8 90.1 DTD [8] 47.6 49.0 57.8 58.5 50.7 53.7 Aircraft [38] 31.1 30.1 38.5 38.5 29.5 38.0 Caltech101 [33] 71.3 71.9 75.4 75.5 70.4 75.6 Flowers102 [41] 73.0 76.4 76.6 76.7 75.5 74.9 Food101 [4] 90.9 90.9 90.8 92.1 89.8 92.6 Cars [28] 75.9 76.3 92.9 93.6 75.9 93.4 Cifar10 [30] 95.0 94.9 95.7 95.5 95.0 95.6 ImageNet-1k [9] 69.6 69.2 73.0 73.1 70.7 75.6 Vision Avg. 71.6 (-0.1) 72.3 (+0.6) 77.0 (+0.4) 77.3 (+0.7) 71.7 76.6

- Table 1: Zero-shot results for LENS in object recognition tasks: We present the performance of various LENS variations and compare them with the out-of-the-box performance of CLIP (Radford et al., 2021). In the majority of benchmarks, LENS demonstrates competitive or superior performance compared to CLIP.

[Figure 74]

#### 3.2 LENS Components

LENS consists of 3 distinct vision modules and 1 reasoning module, each serving a specific purpose based on the task at hand. These components are as follows:

Tag Module. Given an image, this module identifies and assigns tags to the image. To accomplish this, we employ a vision encoder (CLIP) that selects the most suitable tags for each image. In our work, we adopt a common prompt: "A photo of {classname}" for object tagging in order to make our framework flexible across domains without the need for manual/ensemble prompt tuning [47]. We use the object vocabulary built in Section 3.1 as our class options.

Attributes Module. We utilize this module to identify and assign relevant attributes to the objects present in the image. For this purpose, we employ a contrastively pretrained vision encoder called CLIP, while incorporating the task-specific prompts outlined in [40]. The vision encoder classifies the objects based on the attributes vocabulary generated in Section 3.1.

Intensive Captioner. We utilize an image captioning model called BLIP and apply stochastic top-k sampling [12] to generate N captions per image. This approach allows us to capture and encompass diverse aspects of the visual content within an image. These diverse captions are then directly passed to the "reasoning module" without any modifications.

Reasoning Module. We adopt a frozen LLM as our reasoning module, which is capable of generating answers based on the textual descriptions fed by the vision modules, along with the task-specific instructions. LENS seamlessly integrates with any black box LLM, streamlining the process of adding vision capabilities to them and expediting the overall speed.

#### 3.3 Prompt Design

With the textual information obtained from the vision modules, we construct complete prompts for the LLM by combining them. We formatted the tags module as Tags: {Top-k tags}, the attributes modules as Attributes: {Top-K attributes}, the intensive captioning module as Captions:

{Top-N Captions}. In particular, for the hateful-memes task, we incorporate an OCR prompt as OCR: this is an image with written "{meme text}" on it. Finally, we append the specific question prompt: Question: {task-specific prompt} \n Short Answer: at the end. You can see this prompt in action in our demo1.

### 4 Experiments

In this section, we conduct extensive experiments and analyses to show the efficacy of LENS. First, we compare LENS with other state-of-the-art models [47] in object recognition. Then, we also evaluate LENS on vision-and-language reasoning tasks and compare it to multimodal foundation models

[2, 22]. We also perform ablation studies on important design choices such as prompt components and prompt patterns for each task.

#### 4.1 Datasets

For object recognition, we conduct experiments using 9 benchmark datasets presented in [47]. We examine the performance of our method in zero-shot, 1-shot, and 3-shot settings, aiming to showcase the capabilities of the frozen LLM in incorporating contextual learning [6].For vision and language reasoning, we focus on zero-shot benchmarks since we didn’t see an improvement while evaluating LENS in few-shot settings. We evaluate our approach on the test-dev split of the VQA 2.0 dataset [16] and the OK-VQA dataset [39] test set. We also explore the performance of LENS on the dev and test-seen sets of the Hateful Memes dataset [25] and the test set of Rendered SST2 [47]. For a detailed overview of each task, including dataset sizes, the number of classes, and the specific evaluation metrics employed, please refer to Table 6 in the supplementary material.

#### 4.2 Implementation Details

We use OpenCLIP-H/142 and CLIP-L/143 as our default vision encoders in both tags and attributes modules. We adopt BLIP-large4 captioning checkpoint finetuned on COCO [36] in intensive captioning module. In this module, we perform a top-k sampling [12], where k represents the desired number of captions and generates a maximum of k = 50 captions per image. Finally, we adopt Flan-T5 models as our default family of frozen LLMs [37]. To generate answers in line with the evaluation tasks, we employ beam search with number of beams equal to 5. Additionally, we apply a length penalty equal to -1, encouraging the generation of concise answers as in [35]. These experiments were conducted on 8 NVIDIA A100 (40GB) GPUs.

We perform task-specific optimizations on LENS to achieve the best performance. For object recognition, we utilize the tag module, which operates on the classes corresponding to each dataset. Additionally, we employ the attribute module, which utilizes the attribute vocabulary. Based on our preliminary experiments, we skip the intensive captioning modules. In VQA tasks, we solely use the intensive captioning module, as our experiments showed that tags and captions did not provide significant improvement. For the Hateful Memes [25] and Rendered-SST2 datasets, we incorporate the tag, attributes, and captioning modules. We generate only one caption using beam search with a width of 5.

#### 4.3 Results

We evaluate LENS across vision and vision & language tasks. For vision, we evaluated 8 benchmarks and compared them with state-of-the-art models in object recognition [47] in zero- and few-shot settings. For vision & language, we evaluate four representative tasks for visual question answering and compare them with state-of-the-art models that employ a frozen LLM and that require additional pre-training stage(s) and big corpora of paired datasets for aligning the vision and language modalities. In these tasks, we only report zero-shot results since we do not observe an improvement while employing in-context learning.

Object Recognition: In Table 1, we observe that on zero-shot, LENS composed by ViT-H/14 [11] as the visual backbone and with Flan-T5xxl as the frozen LLM outperforms in +0.7% on average to equal-size CLIP which employs a common prompt. Interestingly, our experiments reveal that for object recognition tasks, there appears to be no direct relationship between the size of the frozen LLM and classification performance. However, we do observe a correspondence between the size of the tagger architecture (ViT backbone) and performance.

In Fig. 3, we plot the average vision performance on all datasets except ImageNet (due to its large size), and observe that more shots help to increase performance under any combination of visual backbone and frozen LLM. Also, we again observe that there is no direct relationship between a better frozen LLM with respect to performance. However, we do see that a better visual backbone helps to increase the average vision performance.

- 2https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
- 3https://huggingface.co/openai/clip-vit-large-patch14
- 4https://huggingface.co/Salesforce/blip-image-captioning-large

[Figure 75]

76.8

CLIP H/14 CLIP L/14

71.7

- Figure 3: Average few-shot performance of LENS on vision tasks. We conducted tests using various flavors of CLIP vision encoders and a frozen Flan-T5 LLM. Larger LLMs provide better performance in zero-shot and three-shot settings, while larger vision encoders enhance performance across all settings.

# Trainable Params

VQAv2 OK-VQA Rendered - SST2 Hateful Memes test-dev test test dev test-seen

Models

Kosmos-1 1.6B 51.0 - 67.1 63.9 -

Flamingo3B 1.4B 49.2 41.2 - - 53.7 Flamingo9B 1.8B 51.8 44.7 - - 57.0 Flamingo80B 10.2B 56.3 50.6 - - 46.4 BLIP-2ViT-L FlanT5

103M 62.3 39.4 - - BLIP-2ViT-g FlanT5

XL

108M 65.0 45.9 - - -

XXL

LENS Flan-T5XL 0 57.9 32.8 83.3 58.0 59.3 LENS Flan-T5XXL 0 62.6 43.3 82.0 59.4 62.5

- Table 2: Comparison with the state-of-the-art methods on zero-shot settings on VQAv2 [16], OK-VQA [39], Rendered-SST [47], and Hateful Memes [25]. Trainable parameters represent the number of parameters needed for aligning the vision modality with frozen LLM. LENS consistently outperforms or reasonably competes with extensively pretrained methods that rely on large amounts of data for multimodal alignment.

Vision and Language: The comparative performance analysis of LENS in relation to other systems is presented in Table 2. The results obtained from our experiments clearly demonstrate the highly competitive nature of LENS, even when compared to significantly larger and more sophisticated systems such as Flamingo [2], BLIP-2 [35], and Kosmos [22].

Specifically, our findings reveal that on the VQA 2.0 [16], LENS Flan-T5XXL achieves superior performance over Flamingo9B and Kosmos-1 by 11% and 15%, respectively. Furthermore, LENS outperforms the most powerful variant of Flamingo by 6 points. Moreover, our Intensive Captioning module, which utilizes a ViT-L vision encoder, is on par with the largest BLIP-2 architecture that employs ViT-G as its vision encoder. In addition, our best LENS model surpasses multiple versions of Flamingo in on Hateful Memes [25] and exhibits better performance compared to Kosmos on the Rendered-SST2 benchmark. It is perhaps not particularly surprising that our approach does so well on Rendered SST2, where a good language model is needed to understand the sentiment of text which is extracted from an image. In this way, Rendered SST2 is not just about linking image features directly to text; it is also about interpreting what that text actually means. On OK-VQA, our model’s performance does not match that of Flamingo, potentially due to the fact that the 70B Chinchilla language model using in Flamingo80B possesses a larger knowledge base than our best reasoning module, as also suggested in [35].

[Figure 76]

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

Write down the facts that you know about this flower.

Tell me something about the history of this place.

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Is this photo unsual?

[Figure 111]

[Figure 112]

The Great Wall of China is a fortification built by the ancient Chinese to keep out invaders.

Phalaenopsis sanderiana is a species of orchid. It is also known as the moth orchid, butterfly orchid, or moth plant. It is a perennial plant with a waxy or velvety texture. It may have a single large bloom or several smaller blooms.

[Figure 113]

Yes

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Why?

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

The house is upside down.

something else?

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Where does it come from?

[Figure 136]

How could you get out of the house?

The Great Wall of China is a UNESCO World Heritage Site.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Phalaenopsis sanderiana is native to the Philippines.

[Figure 142]

[Figure 143]

The house has a tunnel coming out of it.

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

What is the color of this text?

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

The text color is purple.

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

What does it say?

[Figure 188]

[Figure 189]

[Figure 190]

The text says "red".

[Figure 191]

[Figure 192]

What is this?

What is this?

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

An incandescent light bulb surronded by green leaves

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

A light fixture that has a plant inside of it.

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Are the leaves inside of the light bulb?

Are the plants inside of the lightbulb?

What about this one?

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

The text color is red.

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

No, the leaves are outside the lightbulb

Yes.

[Figure 227]

What does it say?

[Figure 228]

[Figure 229]

[Figure 230]

Is it glowing?

Is it glowing?

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

The text says "green"

[Figure 240]

[Figure 241]

Yes, the lightbulb is glowing

No.

[Figure 242]

[Figure 243]

[Figure 244]

- Figure 4: Selected examples of LENS using Tag and Attributes Modules with OpenCLIP-H/14 as the vision encoder, Intensive Captioning Module and Flan-T5xxl as the LLM.

- 4.4 Ablations on LENS components Prompt Template Acc. (Avg.) Objects 76.6 Attributes 74.7 Objects + Attributes 77.0

Object recognition: In Table 3, we conducted an ablation study of LENS’s components on object recognition using Flan-T5XL and CLIPH/14, as discussed in Section 4.3. We present the average accuracy across the benchmarks. By utilizing only the tag module, we exclusively rely on CLIP and observe similar performance as CLIP-H/14 in Table 1. However, we noticed a drop in performance when using only the attributes module. When combined with tags, attributes significantly contribute to enhancing the performance of LENS by +0.4%. This demonstrates that LENS serves as a robust baseline compared to using only a single vision module such as CLIP. For a detailed overview on each dataset, please refer to Table 7 in supplementary material.

Table 3: Ablations on vision datasets. We report average accuracy on the vision datasets discussed in Section 4.1. The object information helps more than the attributes but together they are complimentary and lead to overall better performance.

Visual Reasoning: For the VQA 2.0 dataset (Goyal et al., 2017), we conducted ablations using our model name, which is equipped with Flan-T5XXL, on the minival split. As shown in Table 5, we noticed that increasing the number of captions generated by our Intensive Captioning module led to a gradual improvement in performance. However, it eventually reaches a saturation point, indicating that the module provides valuable information about the image only up to a certain threshold.

We also conducted ablation studies on the LENS components using the dev set of the Hateful-Memes benchmark [25]. Table 4 demonstrates that a combination of the global captioning, tags, and attributes modules is essential for achieving high performance on this task. Specifically, we observed that both tags and attributes contribute more to the performance improvement compared to the global captioning module when combined with OCR. However, it is important to note that all of these

Prompt Template ROC-AUC OCR 57.2 Objects + OCR 58.4 Attributes + OCR 59.3 Caption + OCR 57.2 All 59.4

Table 4: Hateful Memes [25] ablations. Adding more visual information on top of OCR improves the performance consistently though attributes help the most.

Prompt Template VQA-ACC Question 37.2 Intensive Captioning (1) + Question 52.5 Intensive Captioning (5) + Question 56.6 Intensive Captioning (20) + Question 59.1 Intensive Captioning (50) + Question 60.4

Table 5: Ablation results on VQA 2.0 [16]. Increasing the number of intensive captions improves the performance of LENS gradually on VQA but starts saturating eventually.

components are necessary and their combined usage results in the best performance. We also present several qualitative examples from LENS in Fig. 4, illustrating its reasoning capabilities by answering questions about complex scenes and scenarios.

- 5 Conclusion We introduce LENS , a generic and computationally efficient method that enables a frozen LLM to effectively coordinate vision modules, resulting in competitive performance even when compared to larger multimodally pretrained systems. LENS offers adaptability to various open-source or black-box language models, regardless of their pretraining or multimodal data, thereby providing flexibility and scalability for future improvements in performance within the community.

[Figure 245]

By leveraging the strengths of LLMs and our modular approach, LENS represents a significant advancement in task-solving without the need for additional pretraining. Its seamless integration with diverse vision tasks showcases its versatility and potential for widespread application.

In future work, an intriguing direction to explore would be expanding the applicability of LENS by incorporating it into tasks involving different modalities. For instance, integrating LENS into audio classification or video action reasoning tasks could yield valuable insights. This expansion would involve orchestrating the roles of the LLM and integrating it with complementary modules.

- 6 Limitations

As with any research work, LENS has its own limitations. We aim to address a few of them in this section. Firstly, the vision capability of LENS heavily relies on its underlying vision components, namely CLIP and BLIP. Although these models have shown notable performance improvements, there is still room for further enhancement by leveraging their strengths and combining them with LLMs. We demonstrate a few failure cases of LENS in Fig. 5 in the supplementary material. Future research should explore methods to effectively integrate these models and harness the synergies between vision and language components to achieve even better performance across diverse tasks.

Secondly, it is important to acknowledge that conducting evaluation experiments with LENS models requires substantial computational resources. For example, our experiments were conducted using 8*A100, which may pose challenges for smaller or medium-sized labs, as well as underrepresented communities with limited access to such resources. However, it is worth noting that the computational costs associated with evaluating LENS models are comparatively lower than the extensive training requirements of larger visual-language models, such as Flamingo, which can demand upwards of 500k TPU hours. Nonetheless, efforts should be made to make computational resources more accessible and explore methods for reducing the computational burden while maintaining the effectiveness of LENS.

### Acknowledgments

We would like to express our gratitude to the Fatima Fellowship and Hugging Face for granting computational resources for the preliminary experiments of this project.

### References

- [1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. nocaps: novel object captioning at scale. In Proceedings of International Conference on Computer Vision. IEEE, oct 2019.
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Proceedings of Advances in Neural Information Processing Systems, 2022.
- [3] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6077–6086, 2018.
- [4] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In Proceedings of European Conference on Computer Vision, 2014.
- [5] Andy Brock, Soham De, Samuel L Smith, and Karen Simonyan. High-performance largescale image recognition without normalization. In Proceedings of International Conference on Machine Learning, pp. 1059–1071. PMLR, 2021.
- [6] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.
- [7] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.
- [8] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, , and A. Vedaldi. Describing textures in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2014.
- [9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 248–255. Ieee, 2009.
- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2019.
- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2021.
- [12] Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pp. 889–898, Melbourne, Australia, July 2018. Association for Computational Linguistics.

- [13] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. arXiv preprint arXiv:2211.07636, 2022.
- [14] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. CVPR Workshop, 2004.
- [15] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790, 2023.
- [16] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6904–6913, 2017.
- [17] Jiaxian Guo, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Boyang Li, Dacheng Tao, and Steven C. H. Hoi. From images to textual prompts: Zero-shot vqa with frozen large language models. arXiv preprint arXiv:2212.10846, 2023.
- [18] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.
- [19] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.
- [20] Ronghang Hu and Amanpreet Singh. Unit: Multimodal multitask learning with a unified transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1439–1449, 2021.
- [21] Yushi Hu, Hang Hua, Zhengyuan Yang, Weijia Shi, Noah A. Smith, and Jiebo Luo. Promptcap: Prompt-guided image captioning for vqa with gpt-3. arXiv preprint arXiv:2211.09699, 2023.
- [22] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, and Furu Wei. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045, 2023.
- [23] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. arXiv preprint arXiv:2102.05918, 2021.
- [24] Woojeong Jin, Yu Cheng, Yelong Shen, Weizhu Chen, and Xiang Ren. A good prompt is worth millions of parameters: Low-resource prompt-based learning for vision-language models. arXiv preprint arXiv:2110.08484, 2022.
- [25] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. Proceedings of Advances in Neural Information Processing Systems, 33: 2611–2624, 2020.
- [26] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region supervision. arXiv preprint arXiv:2102.03334, 2021.
- [27] Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. Grounding language models to images for multimodal inputs and outputs. Proceedings of International Conference on Machine Learning, 2023.

- [28] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for finegrained categorization. In Proceedings of International IEEE Workshop on 3D Representation and Recognition, Sydney, Australia, 2013.
- [29] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalanditis, Li-Jia Li, David A Shamma, Michael Bernstein, and Li Fei-Fei. Visual genome: Connecting language and vision using crowdsourced dense image annotations. 2016.
- [30] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.
- [31] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Tom Duerig, and Vittorio Ferrari. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. arXiv preprint arXiv:1811.00982, 2018.
- [32] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. Obelisc: An open web-scale filtered dataset of interleaved image-text documents. 2023.
- [33] Fei-Fei Li, Marco Andreeto, Marc’Aurelio Ranzato, and Pietro Perona. Caltech 101, Apr 2022.
- [34] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of International Conference on Machine Learning, pp. 12888–12900. PMLR, 2022.
- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [36] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll’a r, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. arXiv preprint arXiv:1405.0312, 2014.
- [37] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023.
- [38] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Finegrained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.
- [39] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3195–3204, 2019.
- [40] Sachit Menon and Carl Vondrick. Visual classification via description from large language models. Proceedings of International Conference on Learning Representations, 2023.
- [41] M-E. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing, Dec 2008.
- [42] OpenAI. Language models are unsupervised multitask learners. OpenAI Blog, 2019. URL https://openai.com/blog/better-language-models/.
- [43] OpenAI. Chatgpt. https://openai.com/, 2021. Accessed: Month Day, Year.
- [44] OpenAI. Gpt. https://openai.com/blog/dall-e-3/, 2021. Accessed on March 28, 2023.
- [45] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [46] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar. Cats and dogs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2012.

- [47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of International Conference on Machine Learning, pp. 8748–8763. PMLR, 2021.
- [48] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. Proceedings of International Journal on Computer Vision, 2015.
- [49] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit vision-and-language tasks? arXiv preprint arXiv:2107.06383, 2021.
- [50] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. FLAVA: A foundational language and vision alignment model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
- [51] Haoyu Song, Li Dong, Wei-Nan Zhang, Ting Liu, and Furu Wei. Clip models are few-shot learners: Empirical studies on vqa and visual entailment. arXiv preprint arXiv:2203.07190, 2022.
- [52] Yi-Lin Sung, Jaemin Cho, and Mohit Bansal. Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks. arXiv preprint arXiv:2112.06825, 2022.
- [53] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. arXiv preprint arXiv:2303.08128, 2023.
- [54] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [55] Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. arXiv preprint arXiv:2106.13884, 2021.
- [56] BigScience Workshop. Bloom: A 176b-parameter open-access multilingual language model, 2023.
- [57] J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, and A. Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3485–3492, June 2010. doi: 10.1109/CVPR.2010.5539970.
- [58] Ning Xie, Farley Lai, Derek Doran, and Asim Kadav. Visual entailment: A novel task for fine-grained image understanding. arXiv preprint arXiv:1901.06706, 2019.
- [59] Yujia Xie, Luowei Zhou, Xiyang Dai, Lu Yuan, Nguyen Bach, Ce Liu, and Michael Zeng. Visual clues: Bridging vision and language foundations for image paragraph captioning. arXiv preprint arXiv:2206.01843, 2022.
- [60] Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, and Pete Florence. Socratic models: Composing zero-shot multimodal reasoning with language. arXiv preprint arXiv:2204.00598, 2022.
- [61] Yan Zeng, Xinsong Zhang, and Hang Li. Multi-grained vision language pre-training: Aligning texts with visual concepts. arXiv preprint arXiv:2111.08276, 2022.
- [62] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

- [63] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [64] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: an open, billion-scale corpus of images interleaved with text. arXiv preprint arXiv:2304.06939, 2023.

## Supplementary Material

### A LENS datasets

- In Table 6, we present the statistics for each of the datasets utilized in the evaluation of LENS. We provide details regarding the evaluation metric employed, as well as whether the evaluation was conducted in an open-ended manner or a closed-ended manner with a fixed class vocabulary. .

Dataset Split(s) Size Evaluation Method Metric Image Classification Oxford-IIIT Pets [46] test 3,669 close-ended mean per class Describable Textures [8] test 1,880 close-ended accuracy Caltech-101 [14] test 6,085 close-ended accuracy Oxford Flowers 102 [41] test 6,149 close-ended mean per class FGVC Aircraft [38] test 3,333 close-ended mean per class Food101 [4] test 25,250 close-ended accuracy Cifar10 [4] test 10,000 close-ended accuracy ImageNet-1k [4] validation 50,000 close-ended accuracy Vision & Language Hateful Memes [25] dev 500 open-ended ROC AUC Hateful Memes [25] test-seen 1,000 open-ended ROC AUC VQA 2.0 [16] tesdev 107,394 open-ended VQA accuracy OK-VQA [39] validation 5,046 open-ended VQA accuracy Rendered SST2 [39] validation 1,821 open-ended VQA accuracy

- Table 6: Datasets examined for evaluation of LENS. We describe the size statistics, the split, the method of evaluation, and metrics used for each of these datasets.

B Detailed Ablations in Object Recognition

In Table 7, we present a detailed ablation result in the object recognition task presented in Section 4.3

Dataset Objects Attributes Objects + Attributes

Pets [46] 90.1 91.0 92.6 DTD [8] 53.7 51.5 57.8 Aircraft [38] 38.0 36.5 38.5 Caltech101 [33] 75.6 71.6 75.4 Flowers102 [41] 74.9 75.6 76.6 Food101 [4] 92.6 89.1 90.8 Cars [28] 93.4 92.1 92.9 Cifar10 [30] 95.6 93.4 95.7 ImageNet-1k [9] 75.6 71.5 73.0

Avg. Vision 76.8 75.1 77.5

- Table 7: Ablations results using OpenCLIP-H/14 as vision encoder and Flan-T5XL as the LLM

### C Failure examples

In Fig. 5, we showcase some of the failure cases for our LENS. These involve (i) failures from vision modules to correctly identify objects or attributes (ii) inconsistency between the responses (iii) presuppositions and in-built biases and (iv) forgetting and limitations of context windows of the LLMs.

Components Prompt Tag: Top-1 CLIP Tag Attributes: Top-K Attributes Question: Task specific

prompt

Short Answer: {answer}

- Table 8: Object recognition prompt used in LENS

Components Prompt Captions: Top-N captions Question: e.g Who is doing

"x" action? Short Answer: {answer}

Table 9: VQA prompt used in LENS

Components Prompt Attributes: Top-K Attributes

tags Tags: Top-K Object tags Caption: BLIP-Caption OCR: OCR Meme text Question: Is the image

hateful or not hateful?

Short Answer: {answer}

Table 10: Hateful-memes prompt used in LENS

[Figure 246]

[Figure 247]

[Figure 248]

What is the color of this text?

[Figure 249]

The text is purple.

[Figure 250]

What does it say?

An image with the following features is provided. Tags: Spider man, Duel, Shootfighting, Fist bump. Attributes:

[Figure 251]

The text says "Rebec receptor".

- - cartoon which has cartoon-like art style.
- - teammate which has body language.
- - enemy which is aggressive body language.
- - hero which is a signature weapon or tool Captions:
- - a cartoon picture of spider- man attacking another.
- - a spider - man that is next to a truck.
- - a person is playing with something and another man is pointing at the other one.
- - a man points a gun at something in front of a van.
- - a man in a blue and red suit and a man in blue outfit.

[Figure 252]

lol what?

[Figure 253]

The text is purple

[Figure 254]

What is this?

[Figure 255]

A cartoon picture of Spider-Man attacking another.

(a) (b)

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

What color are the people's shoes?

[Figure 260]

The people's shoes are black.

[Figure 261]

[Figure 262]

Are you sure?

[Figure 263]

Yes, I am.

[Figure 264]

Does the first image have a cat?

[Figure 265]

No.

(c) (d)

- Figure 5: Incorrect outputs of LENS using Tag and Attributes Modules with OpenCLIP-H/14 as

the vision encoder, Intensive Captioning Module and Flan-T5xxl as the LLM. (a) Incorrect Visual Information (b) Inconsistency (c) Presuppositions (d) Forgetting.

