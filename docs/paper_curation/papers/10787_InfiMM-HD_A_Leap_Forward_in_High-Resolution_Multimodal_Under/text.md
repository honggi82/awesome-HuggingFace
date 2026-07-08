## InfiMM-HD: A Leap Forward in High-Resolution Multimodal Understanding

# arXiv:2403.01487v1[cs.CV]3Mar2024

Haogeng Liu*12 Quanzeng You*3 Xiaotian Han3 Yiqi Wang3 Bohan Zhai3 Yongfei Liu3 Yunzhe Tao3 Huaibo Huang12 Ran He*12 Hongxia Yang*3

### Abstract

Multimodal Large Language Models (MLLMs) have experienced significant advancements recently. Nevertheless, challenges persist in the accurate recognition and comprehension of intricate details within high-resolution images. Despite being indispensable for the development of robust MLLMs, this area remains underinvestigated. To tackle this challenge, our work introduces InfiMM-HD, a novel architecture specifically designed for processing images of different resolutions with low computational overhead. This innovation facilitates the enlargement of MLLMs to higher-resolution capabilities. InfiMM-HD incorporates a cross-attention module and visual windows to reduce computation costs. By integrating this architectural design with a four-stage training pipeline, our model attains improved visual perception efficiently and cost-effectively. Empirical study underscores the robustness and effectiveness of InfiMM-HD, opening new avenues for exploration in related areas. Codes and models can be found https://huggingface.co/Infi-MM/infimmhd

[Figure 1]

GQA

63.5 63.3 63.1

IconQA

OK-VQA

51.3

65.5

50.5

61.3 62.6

STVQA

SQA-IMG

67.7 67.0

83.6

58.9

70.6 69.4

71.6

61.3

68.1 67.6

37.1

InfiMM

70.7

TextVQA

37.4

32.9

65.9

36.4

80.7

82.0

37.6

67.7

85.9

83.4

MMMU

VQAv2

86.2

87.2

87.9

71.6

MMbench

POPE

Figure 1. InfiMM-HD showcases superior performance across various tasks, thanks to its enhanced training pipeline and highresolution inputs, consistently outperforming recent methods in downstream tasks.

form web browsing, by leveraging combined single-modal capabilities from pretrained vision encoders and LLMs.

### 1. Introduction

The landscape of Multimodal Large Language Models (MLLMs) has been revolutionized by integrating pretrained vision encoders with Large Language Models (LLMs) (Han et al., 2023; Wang et al., 2024; Han et al., 2024), a trend exemplified by developments in Flamingo (Alayrac et al., 2022), BLIP-2 (Li et al., 2023b), LLaVA (Liu et al., 2023b), MiniGPT-4 (Zhu et al., 2023a) and etc. MLLMs can exhibit emergent vision-language capabilities (Yang et al., 2023). For example, they can write codes according to images, convert plots in images into Markdown format tables, and per-

*Equal contribution 1MAIS & CRIPAC, Institute of Automation, Chinese Academy of Sciences, China. 2School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China. 3ByteDance, Inc.. Correspondence to: Haogeng Liu <liuhaogeng22@mails.ucas.ac.cn>.

Effective integration of pretrained vision encoders with Large Language Models in MLLMs relies on carefully designed vision-language bridging modules. These modules address two critical aspects: the transformation and alignment of visual tokens to LLM-compatible formats, and the subsequent utilization of these transformed tokens.

For the transformation and alignment of visual tokens, models such as Flamingo (Alayrac et al., 2022) and BLIP-2 (Li et al., 2023b) employ Perceiver-Resampler/Q-Former techniques for transforming visual tokens, offering flexibility and nuanced alignment with language counterparts but at the cost of higher computational demands and potential information loss (Cha et al., 2023). Conversely, models such as LLaVA and MiniGPT-v2 take a different route, using simpler Multi-Layer Perceptron (MLP) approaches. While this reduces computational complexity and the number of learnable parameters, it might not capture the full intricacies of visual data.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

###### Detailed description

[Figure 6]

[Figure 7]

[Figure 8]

InfiMM-HD: The picture shows a small action figure of a female character holding a gun, standing in front of an advertisement for a video game called "I-KOHL Pheromonic Make-Up." The advertisment features a woman in a red and white outfit, holding a weapon, and the text "In War and Love." The background is a blue gradient.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

###### Ref-Region Description

###### Detailed info extraction

[Figure 15]

[Figure 16]

[Figure 17]

User: What is the Entspricht in Euro in the recipe?

[Figure 18]

[Figure 19]

InfiMM-HD: The Entsprich in Euro is 36.33.

[Figure 20]

[Figure 21]

[Figure 22]

User: What is the day in the recipe? Answer in format of xx.xx.xxxx.

[Figure 23]

[Figure 24]

User: Please provide a short description for this region: [0.06,0.7,0.27,0.79].

[Figure 25]

[Figure 26]

InfiMM-HD: 30.07.2007 Infimm-HD: Yellow flowers.

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

User: What is the price of Gloki?

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

User: Please provide a short description for this region: [0.17,0.07,0.29,0.11].

InfiMM-HD: The price of the drink Gloki is 5.00.

[Figure 35]

[Figure 36]

Infimm-HD: A green street sign.

Figure 2. Example outputs by InfiMM-HD, highlighting the model’s adeptness in fine-grained visual perception.

The integration phase of visual and language tokens is equally vital. Flamingo-style architectures employ crossattention mechanisms, facilitating intricate interactions between token types without expanding the token sequence length. This approach efficiently manages the computational load. LLaVA-style models, however, use a direct concatenation method, which, while straightforward, increases token sequence length and computational complexity.

Though different, both architecture utilized low image resolution due to pretrained Vision Transformer (ViT) encoders (Jiang et al., 2023; Radford et al., 2021; He et al.,

- 2021). Low resolution suffices for basic image-level semantic understanding but falls short for detailed, region-level analysis. Recent efforts (Wang et al., 2023; Li et al., 2023c; Lin et al., 2023) aim to enable MLLMs to handle higherresolution images. However, significant challenges remain, primarily because the computational demands tend to increase quadratically in relation to the sequence length for larger images. For instance, increasing the image resolution from 224 × 224 (Dosovitskiy et al., 2021) to 448 × 448 multiplies the self-attention computation by 16 times. To address these challenges, we introduce InfiMM-HD, an

innovative MLLM architecture designed for processing highresolution image. InfiMM-HD innovatively merges methodologies from both Flamingo and LLaVA styles in MLLMs. For the transformation and alignment phase, it adopts an MLP-based approach, akin to LLaVA, effectively transforming and aligning visual tokens into formats compatible with LLMs. This strategy balances computational efficiency with precise processing. In the integration phase, InfiMMHD utilizes a cross-attention mechanism, reminiscent of Flamingo-style MLLMs, to seamlessly incorporate visual token features with language tokens. This approach mitigates the computational challenges associated with longer token sequences as previously mentioned. Notably, while the exploration of high-resolution image input capabilities in Flamingo-style MLLMs is still an emerging area of research, InfiMM-HD marks a significant, pioneering advancement in this domain, blending the best of both worlds to enhance MLLM performance with high-resolution visual inputs.

To overcome the resolution constraints of pretrained vision encoders, InfiMM-HD is strategically trained in four stages, enhancing resolution handling while maintaining visionlanguage alignment. Initially, the model is pretrained with

224 × 224 resolution images for efficient visual-language alignment. Subsequently, it continues pretraining with interpolated positional embeddings for 448 × 448 images from multiple datasets, keeping the LLMs frozen. This is followed by training with full-resolution images, resized to the nearest multiple of 448×448, added with 2D positional embeddings and crop to multiple subimages. In the final stage, the model undergoes visual instruction fine-tuning, freezing the vision encoder and making LLM trainable to enhance instruction-following capability. This structured training approach is crucial for the model’s adaptability and performance across various input resolutions. The contributions of our work can be summarized as follows:

- • We present InfiMM-HD, a pioneering MLLM that employs an MLP-based approach for visual token transformation and alignment, coupled with a Flamingo-style cross-attention mechanism for enhanced and efficient integration of transformed visual and language tokens. It is uniquely designed to seamlessly process highresolution image inputs.
- • We present a four-stage training pipeline that effectively achieves a high-resolution Multimodal Large Language Model with reduced training cost, from initial low-resolution pretraining stage, to continue pretraining stage for knowledge injection and alignment, to dynamic resolution adaption stage for high resolution adoption and finally go through visual instruction fine-tuning stage.
- • Experiments conducted across diverse benchmarks showcase the remarkable proficiency of our model in the realm of visual perception. Additionally, comprehensive ablation studies underscore the distinctive superiority of our design within the context of crossattention-style Multimodal Language Model architectures.

### 2. Related Work

The advent of Large Language Models (LLMs) has catalyzed the development of MLLMs. Flamingo (Alayrac

- et al., 2022) integrates pretrained language models into the MLLM paradigm, employing a gated-cross attention mechanism to fuse visual information into textual sequences. In contrast, BLIP-2 (Li et al., 2023b), MiniGPT4 (Zhu et al., 2023a), and LLaVA (Liu et al., 2023b) propose a paradigm shift, transforming visual signals into soft tokens and directly incorporating them into language models. Shikra (Chen et al., 2023) concentrates on referential dialogue. OtterHD (Li et al., 2023a) fine-tunes Fuyu-8B (Bavishi
- et al., 2023) with instructional guidance, enabling ViT-free MLLMs.

Despite the progress we have seen, some problems still exist. (Zhai et al., 2023) points out that misalignment between visual representation and language causes hallucination. (Zhang et al., 2023) reveals that enhancing the input resolution will significantly increase MLLM’s Optical Character Recognition (OCR) ability. More and more experiments suggest the presence of an information bottleneck in contemporary vision encoders (Tong et al., 2024; Zhai et al., 2023). The resolution of the image stands out as a critical factor that constrains the capacity for visual processing. The study by (Tong et al., 2024) highlights that contemporary MLLMs still face systematic challenges, particularly in aligning visual and textual modalities effectively.

There are some works trying to solve the problem. SPHINX (Lin et al., 2023), introduced by Lin et al., employs a multilayer perception (MLP) to establish connections between visual signals and language. This model leverages multiple vision encoders, amalgamating their output features to construct a robust visual representation. To deal with high resolution image input, SPHINX breaks down input high-resolution images into smaller sub-images and then concatenate visual tokens directly with text tokens. It introduces limitations related to increasing input sequence lengths for the Large Language Model (LLM).

The Monkey model (Li et al., 2023c) addresses this challenge by incorporating a shared resampler. This approach involves compressing each input subimage using the resampling technique from Flamingo (Alayrac et al., 2022) and directly concatenate the visual tokens with text sequence, effectively upscaling the input image resolution to 1344×896. However, the reliance on learnable queries within the perceiver architecture for extracting and compressing information from the raw output of the vision encoder raises concerns about the model’s adaptability across diverse application scenarios.

We assert that the information compression process should intricately intertwine with contextual instructions, allowing for discernment of pertinent details essential for instruction completion. We introduce InfiMM-HD, which establishes connections between vision and language through cross attention mechanism. This departure from the reliance on learnable queries aims to enhance the adaptability and applicability of the model across a broader spectrum of scenariosmore detailed vision perception. Besides, it enables the model to consume high-resolution images at lower cost than previously proposed methods.

### 3. Methods

In this section, we introduce InfiMM architecture and propose a training pipeline for elevating MLLM’s input image resolution with reduced cost. To the best of our knowl-

[Figure 37]

Distribution of Image Sizes

lion), and the Gated Cross Attention Module (approximately 0.7 billion).

261k 325k

During this investigation, we depart from the conventional paradigm of the LLaVA-style structure. This deviation is essential due to its compromised compatibility with highresolution images, as demonstrated in previous studies (Li et al., 2023c; Lin et al., 2023). Notably, the processing of an image with dimensions 1344 × 1344 yields an extensive token sequence comprising 9217 tokens when employing a patch size of 14. Despite the capability of Large Language Models (LLMs) to accommodate sequence lengths up to 32k, the utilization of 9k tokens per image inherently imposes constraints on the performance of Multimodal Language Models (MLLMs), particularly in scenarios involving multiple images. This limitation is consequential, underscoring the necessity for alternative architectural considerations. These considerations aim to tackle the challenges posed by high-resolution image processing within the context of contemporary language models. We adopt a cross-attention module for the integration of visual information at a reduced dimensionality of 768. This method, in contrast to the LLaVA-style architecture, incurs significantly lower computational costs while accommodating extended sequences. Meanwhile, our experiments demonstrate its effective of extracting visual information.

28k

Frequency(logscale)

8130

5644

1830

8411004

310

247

146

24

17

0 200 400 600 800 1000 1200

Image Size

- Figure 3. Visualization of the distribution of image sizes from the LLaVA 665k dataset indicates a predominant clustering of resolutions between 500-700, mixed with some high-resolution examples. Dynamic resolution utilization during training is key for efficient resource management.

edge, we are the pioneers in achieving HD MLLM using the Flamingo-style architecture.

#### 3.1. Model Architecture

The proposed model consists of three components: a Vision Transformer Encoder, a Gated Cross Attention Module, and a Large Language Model. The comprehensive architecture is elucidated in Figure 4. While the illustration showcases a single image, it is imperative to note that follow flamingo’s design, our module can also deal with multiple image as input. Following prior work (Li et al., 2023b; Wang et al., 2023), for the Vision Transformer, we employ EVA2-CLIP2E (Sun et al., 2023), utilizing the output from the penultimate layer as the extracted vision features. The Gated Cross Attention Module leverages text hidden states as queries, and vision features as keys and values. Different from the gating methodology introduced in Flamingo (Alayrac et al.,

- 2022), we incorporate an element-wise tanh gating mechanism for activation. The language model in this study is instantiated using Vicuna (Chiang et al., 2023).

To ensure an effective assimilation of visual information, the Gated Cross Attention Module is strategically inserted every four layers between the decoder layers of Large Language Model. This decision stems from our empirical observation that inserting the module every two layers results in approximately 50% of the gates weights near 0, rendering the cross-attention module ineffective. The model showcases an impressive aggregate of 18 billion parameters, intricately allocated among three key components: the Vision Transformer (4.4 billion), the Large Language Model (12.9 bil-

Table 1. Details on the training data of CPT and DRA.

Task Dataset Samples

COCO Caption (Chen et al., 2015) 205k

Image Caption

TextCaps (Sidorov et al., 2020) 55k VizWiz Caption (Gurari et al., 2020) 55k

VQAV2 (Antol et al., 2015) 443k

OKVQA (Marino et al., 2019) 9k

General VQA

VizWiz VQA (Gurari et al., 2018) 20k

GQA (Hudson & Manning, 2019) 471k

A-OKQA (Schwenk et al., 2022) 17k

TextVQA (Singh et al., 2019) 34k OCRVQA (Mishra et al., 2019) 166k

Text-oriented VQA

STVQA (Biten et al., 2019) 26k DocVQA (Mathew et al., 2021) 63k LLaVAR (Zhang et al., 2023) 16k

Region Description VG (Krishna et al., 2017) 429k Total - 2.00m

#### 3.2. Training Details

We have established a four-stage training procedures for improving MLLMs’ capability of processing high-resolution images, as shown in Figure 5. These stages are denoted as the Pretraining (PT), Continue Pretraining (CPT), Dynamic Resolution Adaption (DRA), and Instruction Finetuning (IFT).

Pretraining Stage (PT): This stage is mainly for initially

POS(0, 0) POS(0, 1) Mike Ehrmann.

###### POS(1, 0) POS(1, 1)

Gated Cross-Attn n

| | |
|---|---|
| | |

LM Block n

###### Vision Encoder

Vision Encoder

Vision Encoder

Vision Encoder

Vision Encoder

...

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Gated Cross-Attn 1

LM Block 1

Global Image Local Images

What is the text in the image?

448×448

- Figure 4. The architectural framework of InfiMM-HD is outlined, where POS(i, j) represents the positional embedding of local patches, with (i, j) indicating their position within the overall image. The model progresses through various training stages, each characterized by selectively training different modules. A detailed explanation of this strategic approach will be elaborated in the following sections.

aligning vision features and language features. During this stage, both the Vision Transformer (ViT) and Large Language Model (LLM) are frozen, only the Gated Cross Attention module is trainable. In this stage, all of the images are resized to 224 × 224 to keep the low training cost.

Continue Pretraining Stage (CPT): In this stage, we employ bilinear interpolation of positional embedding to extend the ViT’s capability to process image of resolution 448×448. The ViT and Gated Cross Attention modules are trainable. Training datasets mainly focus on image captioning and visual question-answering tasks. Detailed information about training datasets is listed in Table 1.

Dynamic Resolution Adaption (DRA): In Figure 3, the outcomes illustrating the sizes of images in the LLaVA-665k dataset (Liu et al., 2023a). Upon meticulous examination of the dataset, it becomes apparent that not all images exhibit resolutions reaching up to 1344. In contrast to conventional practices of uniformly resizing images to a fixed resolution, such an approach incurs unnecessary computational costs, while dynamic image resolution may be cost friendly. To facilitate dynamic resolution inputs, we incorporate the 2D position embedding method proposed in (Wang & Liu, 2021) for individual sub-images. We adopts dynamic input image resolution, ranging from 448 × 448 to 1344 × 1344, during training. Subsequently, the resized image is divided into sub-images of 448 × 448. We also keep an original image thumbnail of 448×448. Finally we use ViT to extract features from each sub-image and original image thumbnail, concatenated directly to form the final vision feature. We use the same training datasets as CPT stage, and keep both the ViT and Gated Cross Attention modules trainable.

Instruction Finetuning Stage (IFT): In this final stage, our goal is to make the model better follow user instructions without losing high-resolution visual perception capability. Thus, we keep the ViT frozen, but let the Gated Cross Attention modules and LLM trainable.

The proposed four-stage training pipeline is key to stabilize training while elevate input image resolution gradually.

### 4. Experiments

In this section, we first discuss about experiment setup. Then we show main results of InfiMM-HD and list series of ablation studies to prove the importance of our proposed modules in InfiMM-HD.

#### 4.1. Setup

Training Dataset. For Pretrainin (PT) stage, training data includes both image-text pairs and interleaved image-text. Image-text pair data includes 140M samples filtered from LAION-2B (Li et al., 2023b), COYO (Byeon et al., 2022), and Laion-coco (Schuhmann et al.). Interleaved imagetext data is sampled from MMC4 (Zhu et al., 2023b) and OBELISIC (Lauren¸con et al., 2023) randomly with 50% for training.

The datasets utilized in the Continue Pretraining (CPT) and Dynamic Resolution Adaption (DRA) stages are enumerated in Table 1. During the Instruction Finetuning (IFT) stage, we amalgamate datasets from LLaVA-665k (Liu et al., 2023a), LLaVAR (Zhang et al., 2023), TextVQA (Singh et al., 2019), and ScienceQA (Lu et al., 2022). This fusion is motivated by the fact that the original LLaVA-665k dataset

Stage1: Pretraining

Stage2: Continue Pretraining

Stage3: Dynamic Resolution Adaption

Stage4: Instruction Finetuning

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Gated Cross Attnn

Gated Cross Attnn

Gated Cross Attnn

Gated Cross Attnn

[Figure 47]

Decoder Layern

Decoder Layern

Decoder Layern

Decoder Layern

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

...

...

...

...

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Gated Cross Attn1

Gated Cross Attn1

Gated Cross Attn1

Gated Cross Attn1

[Figure 52]

Decoder Layer1

Decoder Layer1

Decoder Layer1

Decoder Layer1

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 53]

[Figure 54]

ViT

ViT

ViT

ViT

Resized Image (224)

Resized Image and subimages (448)

Resized Image and subimages (448)

Resized Image (448)

Text

Text

Text

Text

- Figure 5. The four stages of InfiMM-HD training pipeline. Each stage is characterized by distinct trainable modules, datasets, and the resolution of images input to ViT. Our experimental findings confirm the efficacy of our approach, demonstrating the benefits of progressively transitioning from 224x224 to high-resolution images.

[Figure 55]

Q: What is the characters on the left top of the image? A: ijd2sa5d Q: What is the color of the characters on the left top? A: Black

- Figure 6. Illustration of data augmentation: Introducing randomly generated characters with diverse colors into arbitrary regions of the image. Corresponding questions are then generated to complement the original query.

data augmentation methodology. Concretely, this process involves the incorporation of scene-irrelevant characters into images, followed by the generation of corresponding questions. In this context, we randomly introduce two sets of characters, each positioned at the left-top and right-bottom corners of the image, respectively, with distinct colors. The questions generated include inquiries such as “What character is situated at the left-top of the image?”, “What is the color of the characters located at the right-bottom of the image?”, and “How many characters are present in the lefttop region of the image?”. It is imperative to note that this data augmentation technique is exclusively applied to the GQA dataset, resulting in the generation of approximately 100,000 question-answer pairs. Remarkably, we observe the effectiveness of this augmentation approach in enhancing our model’s proficiency in text recognition. Examples can be found in Figure 6.

Training Details. The training process was facilitated through the utilization of deepspeed (Aminabadi et al., 2022), and the FusedAdam optimizer was selected to orchestrate optimization tasks. Additional intricacies related to the experimental configurations are thoroughly delineated in the accompanying Appendix 8.

Evaluation. We evaluate InfiMM-HD across a diverse array of VQA tasks. For general VQA tasks, we leverage benchmarks such as OKVQA (Marino et al., 2019), VQAV2 (Antol et al., 2015), GQA (Hudson & Manning, 2019), and ScienceQA (Lu et al., 2022). These datasets, while not demanding advanced detail visual perception capabilities from the model, effectively gauge models’ ability to understand general scenes and follow user instructions. Moreover, to scrutinize our model’s fine-grained detail perception capability, we incorporate text-oriented VQA datasets, including TextVQA (Singh et al., 2019), STVQA (Biten et al., 2019),

lacks text-oriented samples. Consequently, we supplement this deficiency by incorporating additional data from these diverse sources.

In the IFT stage, we primarily utilize the LLaVA-665k dataset (Liu et al., 2023a) by default. Alongside, we incorporate additional datasets such as TextVQA, LLAVAR, and ScienceQA to enrich our instruction tuning dataset.

Text-oriented Data Augmentation. Due to a scarcity of text-oriented data, we employ a straightforward but effective

- Table 2. Results on general VQA task. The table exclusively presents the performance of our generalist model, showcasing its superiority compared with various models.

Model LLM In-house data OKVQA IconVQA GQA VQAv2 ScienceQAimg Flamingo-80B (Alayrac et al., 2022) - ✓ 50.6 - - 56.3 Palm-E-12B (Driess et al., 2023) - ✓ 60.1 - - 77.7 Qwen-VL (Bai et al., 2023) Qwen-7B ✓ 58.6 - 59.3 79.5 67.1 Qwen-VL-Chat (Bai et al., 2023) Qwen-7B ✓ 56.6 - 57.5 78.2 68.2 CogVLM (Wang et al., 2023) Vicuna-7B ✓ 58.9 - - 83.4 Monkey (Li et al., 2023c) Qwen-7B ✓ 61.3 - 60.7 80.3 69.4

BLIP-2 (Li et al., 2023b) Vicuna-13B × 45.9 40.6 41.0 - Shikra (Chen et al., 2023) Vicuna-13B × 47.2 - - 77.4 mPLUG-Owl2(Ye et al., 2023) LLaMA2-7B × 57.7 - 56.1 79.4 68.7 LLaVA 1.5 (Liu et al., 2023a) Vicuna-13B × - - 63.3 80.0 71.6 Sphinx-2K (Lin et al., 2023) LLaMA2-13B × 62.6 50.5 63.1 80.7 70.6

InfiMM-HD Vicuna-13B × 65.5 51.3 63.5 82.0 83.6

- Table 3. Evaluation results for text-oriented Visual Question Answering (VQA) task. For STVQA, Monkey randomly samples data from the train set for evaluation.

Model Res In-house data TextVQA OCRVQA STVQA Qwen-VL-Chat(Bai et al., 2023) 448 × 448 ✓ 61.5 70.5 Monkey (Li et al., 2023c) 1344 × 768 ✓ 67.6 - 67.7 UniDoc (Gu et al., 2022) 224 × 224 × 40.7 34.5 30.8 DocPedia (Feng et al., 2023) 2560 × 2560 × 60.2 57.2 45.5 BLIP-2 (Li et al., 2023b) 224 × 224 × - 40.6 LLaVA1.5 (Liu et al., 2023a) 336 × 336 × 48.5 - Sphinx-2K (Lin et al., 2023) 768 × 768 × 61.2 67.8 InfiMM-HD (all are piexl-only) dynamic × 70.7 66.0 67.0

and OCRVQA (Mishra et al., 2019). We assess the logical reasoning capabilities of our model by employing newly introduced benchmarks, including MM-VET (Yu et al., 2023), MME (Fu et al., 2023), MMbench (Liu et al., 2023c), InfiMM-Eval (Han et al., 2023), and MMMU (Yue et al., 2023). Notably, the MMMU (Yue et al., 2023) presents challenging tasks that demand advanced subject knowledge and deliberate reasoning at a collegiate level. These tasks span diverse fields such as physics, chemistry, and biology. The MM-VET benchmark assesses the integrated capabilities of models.

- 4.2. Main Results

timodal multiple-choice questions covering a wide range of scientific subjects, represents a significant expansion of the benchmarking landscape. In these varied tasks, our model has shown remarkable effectiveness, indicating significant performance improvements. By outperforming its closest competitor by an average margin of 3.88%, our model not only showcases its superior capacity for integrating multimodal information but also its proficiency in leveraging extensive prior knowledge to navigate through a diverse array of questions successfully.

In addition to the general VQA assessment, we further explore our model’s detailed visual perception capability by evaluating on text-oriented datasets, including TextVQA, OCRVQA and STVQA, as demonstrated in Figure A. Quantitative results are outlined in Table 2. These results underscore the effectiveness of our proposed high-resolution model in comprehending intricate textual details within images.

We also evaluate InfiMM-HD on recently proposed MLLMs evaluation benchmarks, including MMMU, MMVet, InfiMM-Eval, MMB, MME, and POPE. Compared with previous VQA datasets, these datasets include more comprehensive evaluation aspects of MLLMs, requiring more complex reasoning capabilities. Evaluation results are outlined in Table 4. It is noteworthy that no single model excels across all benchmarks, with each model exhibiting its unique strengths and limitations. Our proposed model demonstrates commendable overall performance, highlighting its adaptability and competence across diverse disciplines.

We present evaluation results of general VQA and textoriented VQA tasks in this section.

Table 3 presents results for general VQA tasks. It is essential to underscore that the scope of evaluations in OKVQA, GQA, and VQAv2 extends beyond the mere assessment of models’ visual perception capabilities, they also critically examine the models’ ability to utilize prior knowledge effectively, thereby providing a more comprehensive evaluation of models’ overall functionality. Additionally, ScienceQA (Lu et al., 2022), which comprises 21,000 mul-

#### 4.3. Ablation Study

To elucidate the direct impact of input image resolution on model performance, we conducted an ablation study. In this investigation, different resolution inputs underwent an

Table 4. Results obtained from benchmarks intricately designed for MLLMs with heightened complexity.

Model POPE MMEP MMEC MMB MM-VET InfiMM-Eval MMMU (val)

BLIP-2 (Li et al., 2023b) 85.3 1293.8 - - 22.4 - Shikra (Chen et al., 2023) - - - 58.8 - - LLaVA 1.5 (Liu et al., 2023a) 85.9 1531.3 295.4 67.7 35.4 32.62 36.4 Qwen-VL-Chat (Bai et al., 2023) - 1487.5 360.7 60.6 - 37.39 35.9 Sphinx-2K (Lin et al., 2023) 87.2 1470.6 326.8 65.9 40.2 - 32.9

Ours 87.9 1472.3 329.4 71.6 38.9 37.42 37.6

- Table 5. Evaluation Results for models trained with different input resolutions. Here dynamic means the model supports resolution ranging from 448 × 448 to 1344 × 1344. During inference, we don’t limit resolution to 1344.

Resolution GQA VQAv2 OCRVQA DocVQA TextVQA

224 × 224 60.7 78.7 57.6 25.6 50.0 448 × 448 61.3 80.5 58.7 44.9 64.1

dynamic 63.5 82.0 66.0 55.1 70.7

- Table 6. Ablation study Results for position embedding. w/o PE means removing the positional embedding.

Resolution GQA VQAv2 OCRVQA DocVQA TextVQA

dynamic (w/o PE) 63.3 81.6 65.4 53.0 70.3 dynamic 63.5 82.0 66.0 55.1 70.7

identical set of four training stages. To ensure experimental equitability, we conduct the same model training without DRA by two epochs on the multi-task datasets. The results are presented in Table 5. An observable pattern suggests that elevating the input images’ resolution boosts the efficacy of the model, especially in tasks necessitating an understanding of textual nuances, such as TextVQA (Singh et al., 2019) and DocVQA (Mathew et al., 2021). In contrast, for general VQA tasks, a coarse scene overview is often sufficient for accurate question-answering, underscoring the context-dependent influence of resolution on model efficacy.

Intuitively, as we cropped the input image into subimages, to maintain the spatial information, it is important adding a position embedding for each subimage. To figure out its impact, we carried out ablation study on 2D position embedding, with the results listed in Table 6. The findings suggest that removing the position embedding slightly influences model performance. But on DocVQA, it faces apparently degradation. This phenomenon may be attributed to the fact that DocVQA predominantly comprises documents, where the correspondence between various components holds significance, directly reflected through spatial information.

In our model, the perceiver resampler is removed compared

Table 7. Evaluation Results for models trained with different resolution. PC here means the model has perceriver resampler.

Configuration GQA VQAv2 DocVQA TextVQA

224 × 224 60.7 78.7 25.6 50.0 224 × 224 (PC) 57.7 79.0 25.2 48.9

448 × 448 61.3 80.5 44.9 64.1 448 × 448 (PC) 56.9 79.5 30.7 56.0

with the origin Flamingo. To figure out its impact, we investigated the significance of the perceiver resampler with ablation study. A comparison with models incorporating the perceiver resampler is presented in Table 7. As the table indicates, the perceiver resampler would become an information bottleneck, constraining the model’s performance improvement with increased resolution.

#### 4.4. Limitations

This study introduces an enhancement to MLLMs to effectively process high-resolution images. Results marks significant advancements in various dimensions. Despite these achievements, certain limitations persist. In practice, the model exhibits deficiencies in tasks oriented towards text comprehension. Our ongoing efforts involve exploring more effective modal alignment strategies while augmenting the dataset to enhance overall model performance.

### 5. Conclusions

In this work, we present InfiMM-HD, an improvement over Flamingo-style MLLM designed for processing highresolution input images. Our approach leverages a crossattention mechanism to seamlessly integrate visual information with language model in a low-dimensional space. To address the formidable computational demands associated with high-resolution images, we partition the input highresolution image into smaller sub-images, each subjected to individual processing using a shared Vision Transformer (ViT) specifically tailored for relatively lower resolutions. Additionally, we establish a four-stage training pipeline to

construct the proposed model, ensuring low computational costs are incurred. The proposed model is thus characterized by its comprehensive design and an emphasis on minimizing computational resources.

### 6. Broader Impact

Our model, despite its capabilities, may encounter challenges, including the generation of inaccurate information and susceptibility to perceptual illusions. Furthermore, akin to many machine learning models, it may manifest biases influenced by underlying value systems. Recognizing these potential issues is crucial for ensuring the responsible and ethical deployment of such technologies.

### References

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

Aminabadi, R. Y., Rajbhandari, S., Zhang, M., Awan, A. A., Li, C., Li, D., Zheng, E., Rasley, J., Smith, S., Ruwase, O., and He, Y. Deepspeed inference: Enabling efficient inference of transformer models at unprecedented scale, 2022.

Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., and Parikh, D. Vqa: Visual question answering. In International Conference on Computer Vision (ICCV), 2015.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023.

Bavishi, R., Elsen, E., Hawthorne, C., Nye, M., Odena, A., Somani, A., and Ta¸sırlar, S. Introducing our multimodal models, 2023. URL https://www.adept. ai/blog/fuyu-8b.

Biten, A. F., Tito, R., Mafla, A., Gomez, L., Rusinol, M., Valveny, E., Jawahar, C., and Karatzas, D. Scene text visual question answering. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4291– 4301, 2019.

Byeon, M., Park, B., Kim, H., Lee, S., Baek, W., and Kim, S. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/ coyo-dataset, 2022.

Cha, J., Kang, W., Mun, J., and Roh, B. Honeybee: Localityenhanced projector for multimodal llm, 2023.

Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., and Zhao, R. Shikra: Unleashing multimodal llm’s referential dialogue magic, 2023.

Chen, X., Fang, H., Lin, T.-Y., Vedantam, R., Gupta, S., Dollar, P., and Zitnick, C. L. Microsoft coco captions: Data collection and evaluation server, 2015.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https://lmsys.org/blog/ 2023-03-30-vicuna/.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale, 2021.

Driess, D., Xia, F., Sajjadi, M. S. M., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., Huang, W., Chebotar, Y., Sermanet, P., Duckworth, D., Levine, S., Vanhoucke, V., Hausman, K., Toussaint, M., Greff, K., Zeng, A., Mordatch, I., and Florence, P. Palm-e: An embodied multimodal language model, 2023.

Feng, H., Liu, Q., Liu, H., Zhou, W., Li, H., and Huang, C. Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding, 2023.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., and Ji, R. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2023.

Gu, J., Kuen, J., Morariu, V. I., Zhao, H., Barmpalios, N., Jain, R., Nenkova, A., and Sun, T. Unified pretraining framework for document understanding, 2022.

Gurari, D., Li, Q., Stangl, A. J., Guo, A., Lin, C., Grauman, K., Luo, J., and Bigham, J. P. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3608–3617, 2018.

Gurari, D., Zhao, Y., Zhang, M., and Bhattacharya, N. Captioning images taken by people who are blind. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16, pp. 417–434. Springer, 2020.

Han, X., You, Q., Liu, Y., Chen, W., Zheng, H., Mrini, K., Lin, X., Wang, Y., Zhai, B., Yuan, J., Wang, H., and Yang, H. InfiMM-Eval: Complex Open-Ended Reasoning Evaluation For Multi-Modal Large Language Models.

arXiv e-prints, art. arXiv:2311.11567, November 2023. doi: 10.48550/arXiv.2311.11567.

Han, X., You, Q., Liu, Y., Chen, W., Zheng, H., Mrini, K., Lin, X., Wang, Y., Zhai, B., Yuan, J., Wang, H., and Yang, H. Infimm-eval: Complex open-ended reasoning evaluation for multi-modal large language models, 2023.

Han, X., Wang, Y., Zhai, B., You, Q., and Yang, H. Coco is ”all” you need for visual instruction fine-tuning, 2024.

He, K., Chen, X., Xie, S., Li, Y., Doll´ar, P., and Girshick, R. Masked autoencoders are scalable vision learners, 2021.

Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700– 6709, 2019.

Jiang, D., Liu, Y., Liu, S., Zhang, X., Li, J., Xiong, H., and Tian, Q. From clip to dino: Visual encoders shout in multi-modal large language models, 2023.

Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D. A., et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017.

Lauren¸con, H., Saulnier, L., Tronchon, L., Bekman, S., Singh, A., Lozhkov, A., Wang, T., Karamcheti, S., Rush, A. M., Kiela, D., Cord, M., and Sanh, V. Obelics: An open web-scale filtered dataset of interleaved image-text documents, 2023.

Li, B., Zhang, P., Yang, J., Zhang, Y., Pu, F., and Liu, Z. Otterhd: A high-resolution multi-modality model, 2023a.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023b.

Li, Z., Yang, B., Liu, Q., Ma, Z., Zhang, S., Yang, J., Sun, Y., Liu, Y., and Bai, X. Monkey: Image resolution and text label are important things for large multi-modal models, 2023c.

Lin, Z., Liu, C., Zhang, R., Gao, P., Qiu, L., Xiao, H., Qiu, H., Lin, C., Shao, W., Chen, K., Han, J., Huang, S., Zhang, Y., He, X., Li, H., and Qiao, Y. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models, 2023.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning, 2023a.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning, 2023b.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., Chen, K., and Lin, D. Mmbench: Is your multi-modal model an all-around player?, 2023c.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.

Marino, K., Rastegari, M., Farhadi, A., and Mottaghi, R. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pp. 3195–3204, 2019.

Mathew, M., Karatzas, D., and Jawahar, C. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

Mishra, A., Shekhar, S., Singh, A. K., and Chakraborty, A. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pp. 947–952. IEEE, 2019.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision, 2021.

Schuhmann, C., K¨opf, A., Vencu, R., Coombes, T., and Beaumont, R. Laion coco: 600m synthetic captions from laion2b-en. https://laion.ai/blog/ laion-coco/.

Schwenk, D., Khandelwal, A., Clark, C., Marino, K., and Mottaghi, R. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pp. 146–162. Springer, 2022.

Sidorov, O., Hu, R., Rohrbach, M., and Singh, A. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pp. 742–758. Springer, 2020.

Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.

Sun, Q., Fang, Y., Wu, L., Wang, X., and Cao, Y. Eva-clip: Improved training techniques for clip at scale, 2023.

Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms, 2024.

Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., Xu, J., Xu, B., Li, J., Dong, Y., Ding, M., and Tang, J. Cogvlm: Visual expert for pretrained language models, 2023.

- Wang, Y., Chen, W., Han, X., Lin, X., Zhao, H., Liu, Y., Zhai, B., Yuan, J., You, Q., and Yang, H. Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning, 2024.
- Wang, Z. and Liu, J.-C. Translating math formula images to latex sequences using deep neural networks with sequence-level training. International Journal on Document Analysis and Recognition (IJDAR), 24(1-2):63–75, 2021.

Yang, Z., Li, L., Lin, K., Wang, J., Lin, C.-C., Liu, Z., and Wang, L. The dawn of lmms: Preliminary explorations with gpt-4v(ision), 2023.

Ye, Q., Xu, H., Ye, J., Yan, M., Hu, A., Liu, H., Qian, Q., Zhang, J., Huang, F., and Zhou, J. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration, 2023.

Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., and Wang, L. Mm-vet: Evaluating large multimodal models for integrated capabilities, 2023.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., Wei, C., Yu, B., Yuan, R., Sun, R., Yin, M., Zheng, B., Yang, Z., Liu, Y., Huang, W., Sun, H., Su, Y., and Chen, W. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi, 2023.

Zhai, B., Yang, S., Xu, C., Shen, S., Keutzer, K., and Li, M. Halle-switch: Controlling object hallucination in large vision language models, 2023.

Zhang, Y., Zhang, R., Gu, J., Zhou, Y., Lipka, N., Yang, D., and Sun, T. Llavar: Enhanced visual instruction tuning for text-rich image understanding, 2023.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023a.

Zhu, W., Hessel, J., Awadalla, A., Gadre, S. Y., Dodge, J., Fang, A., Yu, Y., Schmidt, L., Wang, W. Y., and Choi, Y. Multimodal C4: An open, billion-scale corpus of images interleaved with text. arXiv preprint arXiv:2304.06939, 2023b.

### A. Training Configuration

[Figure 56]

Figure 7. Example of a TextVQA sample, exemplifying its intricate scrutiny of nuanced visual perception.

We report the detailed training hyper-parameters and settings of InfiMM-HD in Table 8. Our model was trained with 64 NVIDIA A100-SXM-80GB. The pretraining stage cost about 72 hours. (If only use the image-text pair, it costs 30 hours.) The remaining three stages cost less than 21 hours.

Table 8. Details of the training Configuration. For the second stage, we utilize bilinear interpolation to extend the origin ViT to support resolution 448. IT means image text pair. And IIT means interleaved image text sequence.

Configuration Pretraining Continue Pretraining Dynamic Resolution Adaption Instruction Finetuning

ViT init. EVA2-CLIP2-E (res 224) EVA2-CLIP2-E (res 448) ViT from 2nd-stage ViT from 3rd-stage LLM init. Vicuna-13b Vicuna-13b Vicuna-13b Vicuna-13b

Gated cross-attention init. random InfiMM-HD 1st stage InfiMM-HD 2nd stage InfiMM-HD 3rd stage

Image resolution 224 448 dynamic(448-1344) dynamic(448-1344) ViT sequence length 257 1024 1025 1025

LLM sequence length 32 (IT);384 (IIT) 128 128 1024 Optimizer AdamW Optimizer hyperparameter β1 = 0.9,β2 = 0.95,eps = 1e−8 β1 = 0.9,β2 = 0.999,eps = 1e−5

Peak learning rate 1e−4 1e−5 1e−5 5e−6 Minimum learning rate 1e−4 1e−6 1e−6 5e−7 Learning rate schedule cosine decay Weight decay 0.1 Gradient clip 1.0

Training steps 120k 8k 8k 11k

warm steps 6k 400 400 550 Global batch size 10240 (IT);768 (IIT) 256 256 64

Gradient accumulation steps 2 1 1 2

Gradient ACC. 2 1 1 2 Numerical precision bfloat16

Gradient checkpointing × × ✓ × Deepspeed Zero Stage 2 Training resource 64 NVIDIA A100-SXM-80GB

### B. Summary of Evaluation Benchmarks

We provided a detailed summary of evaluation benchmarks we used and their corresponding metrics in Table 9. Note that ANLS means Average Normalized Levenshtein Similarity.

Table 9. Details on the test dataset.

Task Dataset Description Split Metric

VQAV2 VQA on natural images test-dev VQA Score(↑) OKVQA VQA on natural images but need world knowledge val VQA Score(↑) IconQA Abstract diagram understanding and visual language reasoning test EM(↑)

General VQA

GQA VQA on scene understandig and reasoning test-dev EM(↑) ScienceQA Multimodal multi choice VQA on science filed test Accuracy(↑)

TextVQA VQA about text in natural scene val VQA Score(↑)

OCRVQA VQA on images of book covers val EM(↑)

Text-oriented VQA

STVQA VQA covering reading and reasoning about text test ANLS(↑) DocVQA VQA on images from documents test ANLS(↑)

MME Evalutaion for MLLM on perception and cognition Perception and Cognition Accuracy(↑) MM-VET Dialog style VQA on integrated ability test GPT-4 score(↑)

MMbench Comprehensive evalutaion with multi choice VQA test Accuracy(↑) score(↑)

Other Benchmarks

POPE Object hallucination in MLLM adversial F1 score(↑)

InfiMM Complex Open-ended Reasoning test GPT-4 score(↑)

MMMU College-level multi choice VQA val Accuracy(↑)

