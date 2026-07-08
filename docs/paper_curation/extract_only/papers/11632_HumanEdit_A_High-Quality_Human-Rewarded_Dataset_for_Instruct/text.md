# arXiv:2412.04280v2[cs.CV]6May2025

## HumanEdit: A High-Quality Human-Rewarded Dataset for Instruction-based Image Editing

Jinbin Bai1,2∗ Wei Chow1∗ Ling Yang3 Xiangtai Li2 Juncheng Li1 Hanwang Zhang2,4 Shuicheng Yan1,2†

1National University of Singapore 2Skywork AI 3Peking University 4Nanyang Technological University

* Equal contributions, † Corresponding author Project Page: https://viiika.github.io/HumanEdit

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Add a red butterfly on the white ball. Add

Convert a suitcase into a dog. Replace

Remove the metal elephant ornament from the vase Remove

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Move the white golf ball on the right side of the club to the lower left.

Remove the middle deer, making the total number of deer go from 3 to 2.

The female wearing the white coat left hand was tucked into her white lab coat

Counting

Action

Relation

Figure 1: Data examples of instruction-guided image editing in HumanEdit. Our dataset encompasses six distinct editing categories. In the images, gray shapes represent masks, which are provided for every photograph. Moreover, approximately half of the dataset includes instructions that are sufficiently detailed to enable editing without masks. It is important to note that, for conciseness, masks are depicted directly on the original images within this paper; however, in the dataset, the original images and masks are stored separately.

### Abstract

We present HumanEdit, a high-quality, human-rewarded dataset specifically designed for instruction-guided image editing, enabling precise and diverse image manipulations through open-form language instructions. Previous large-scale editing datasets often incorporate minimal human feedback, leading to challenges in aligning datasets with human preferences. HumanEdit bridges this gap by employing human annotators to construct data pairs and administrators to provide feedback. With meticulously curation, HumanEdit comprises 5,751 images and requires more than 2,500 hours of human effort across four stages, ensuring both accuracy and reliability for a wide range of image editing tasks. The dataset includes six distinct types of editing instructions: Action, Add, Counting, Relation, Remove, and Replace, encompassing a broad spectrum of real-world scenarios. All images in the dataset are accompanied by masks, and for a subset of the data, we ensure that the instructions are sufficiently detailed to support mask-free editing. Furthermore, Hu-

CVPR 2025 AI for Content Creation (AI4CC) Workshop. : jinbin.bai@u.nus.edu

manEdit offers comprehensive diversity and high-resolution 1024 × 1024 content sourced from various domains, setting a new versatile benchmark for instructional image editing datasets. With the aim of advancing future research and establishing evaluation benchmarks in the field of image editing, we release HumanEdit at https://huggingface.co/datasets/BryanW/HumanEdit.

### 1 Introduction

In the fields of computer vision and graphics, image-to-image synthesis has been a foundational topic of research for many years. Pioneering works such as CycleGAN [Zhu et al., 2017], CartoonGAN [Chen et al., 2018, Wang and Yu, 2020], and StyleGAN [Karras et al., 2019] have achieved remarkable success in tasks ranging from unpaired image-to-image translation to high-quality image synthesis. Recent advancements in diffusion models [Rombach et al., 2022b, Podell et al., 2023, Sauer et al., 2024] have propelled text-to-image generation to unprecedented levels, largely due to the availability of massive datasets like LAION-5B [Schuhmann et al., 2022a], which provide the necessary scale and diversity for training state-of-the-art models. Building upon these exceptional text-to-image foundation models, numerous works have extended their applications to image-to-image editing [Brooks et al., 2023, Bai et al., 2023, Feng et al., 2024], video generation [Blattmann et al.,

- 2023, Tian et al., 2024, Yang et al., 2024b] , 3D generation [Yi et al., 2024b, Wu et al., 2024, Yi et al.,
- 2024a, 2023], and more.

A critical task within image-to-image synthesis is applying semantic edits to specific regions of images. Such operations, categorized as Local Editing [Yu et al., 2024a], are exemplified by works like InstructPix2Pix [Brooks et al., 2023], which enables image editing based on textual instructions. The increasing demand for precise image editing has driven the creation of specialized datasets [Yang et al., 2024a, Ge et al., 2024a, Zhang et al., 2024a], enabling fine-grained tasks such as style modifications [Zhang et al., 2017], object changes [Bai et al., 2023, Feng et al., 2024, Shi et al., 2024, Zhou et al., 2024], and background alterations [Zhang and Agrawala, 2024].

Recently, a number of instruction-based image editing datasets and models have been introduced to advance the performance of models in local editing tasks, such as EMU-Edit [Sheynin et al., 2024], HQ-Edit [Hui et al., 2024], SEED-Data-Edit [Ge et al., 2024a], EditWorld [Yang et al., 2024a], UltraEdit [Zhao et al., 2024], and AnyEdit [Yu et al., 2024a]. Despite their contributions, most of these datasets are constructed with image synthesis models and large language models, incorporating minimal human feedback. Consequently, these datasets often fall short of practical applicability. A key challenge lies in aligning datasets with human preference, as the distribution of training data tends to be noisy and misaligned with real-world user editing instructions. This discrepancy gives rise to several issues in image editing tasks. For instance, the phrasing of editing instructions and the mask regions often fail to reflect actual user needs, and the edited outputs frequently exhibit artifacts or inconsistencies with human performance (e.g., body distortions). These intrinsic dataset biases are difficult to address solely through improvements in model architectures and training schedule. Although datasets like MagicBrush [Zhang et al., 2024a] attempt to address this gap by employing human annotators, they suffer from limitations in image quality and resolution due to the constraints of their original image sources. These shortcomings hinder their ability to support high-quality and high-resolution editing scenarios effectively. We provide a detailed discussion of these limitations in Section 3.

Table 1: Distribution of 6 types of our human-rewarded editing instructions.

Add Rmove Replace Action Counting Relation Sum

HumanEdit-full 801 1,813 1,370 659 698 410 5,751 HumanEdit-core 30 188 97 37 20 28 400

Recognizing the importance of addressing these challenges in training dataset to advance instructional image-to-image translation, we introduce HumanEdit, a high-quality instructional image editing dataset featuring human-annotated instructions. HumanEdit includes 5,751 high-quality image pairs, each accompanied by editing instructions and detailed image descriptions, and spans six editing categories: Action, Add, Counting, Relation, Remove, and Replace (Tab. 1). HumanEdit offers several advantages:

- • Enhanced Data Quality: Through multi-round quality control, HumanEdit achieves higher data accuracy and consistency compared to existing datasets. The dataset underwent multiple rounds of validation and modification, totaling approximately 2,500 hours of effort, ensuring suitability for fine-tuning or evaluation benchmarks.
- • Diverse and High-Resolution Sources: Unlike MagicBrush [Zhang et al., 2024a], which is limited to the COCO dataset [Lin et al., 2014], HumanEdit is sourced from a broader range of origins and includes higher-resolution images, catering to high-fidelity, photo-realistic editing tasks.
- • Mask Differentiation: HumanEdit categorizes images into those requiring masks and those that do not, providing masks where necessary to support diverse fine-tuning and evaluation needs.
- • Increased Diversity: Analyses such as word cloud visualizations, Vendi Score calculations, sunburst charts, river charts and categorizations of image pair types underscore the dataset’s superior diversity.
- • Categorization Across Dimensions:By classifying editing tasks into six distinct dimensions, HumanEdit provides a clear framework for evaluation and development.

We detail the four-stage annotation pipeline in Section 2 and present dataset statistics in Section 3 and Appendix A, including sunburst charts, river charts, and categorizations of image pair types. A guidance book is provided in Appendix B for future research reference, along with failure cases excluded from HumanEdit in Appendix C.

To provide the performance benchmark on HumanEdit for future evaluation and development, we report several baselines in both mask-free and mask-provided settings in Section 4, including InstructPix2Pix [Brooks et al., 2023], MGIE [Fu et al., 2023], HIVE [Zhang et al., 2024b], MagicBrush [Zhang et al., 2024a], Blended Latent Diffusion [Avrahami et al., 2023], GLIDE [Nichol et al., 2021], aMUSEd [Patil et al., 2024] and Meissonic [Bai et al., 2024]. Default hyperparameters are used to ensure reproducibility and fairness. And we draw some conclusions, for example, most methods perform better on Add tasks than on Remove tasks. Mask-provided methods generally achieve superior performance in semantic-level evaluation metrics compared to pixel-level metrics. Furthermore, even for Add tasks, challenges persist in cases requiring domain-specific knowledge or handling unfamiliar instructions, such as “Add a petal in the middle of the white puppy’s forehead.”

This dataset establishes a benchmark for future research, fostering the development of advanced image-to-image translation and editing models.

### 2 Dataset Annotation Pipeline

The data collection process is outlined in Figure 2, which divides the workflow into four distinct stages:

In the first stage, we design a comprehensive tutorial and quiz to ensure high-quality annotations. The tutorial provides detailed guidance on effectively using the DALL-E 2 platform, along with essential annotation guidelines. More information about the tutorial can be found in Appendix B. We recruit over ten workers from an internal platform, train them using the tutorial, and conduct a quiz to evaluate their understanding. The top ten performers are selected as annotators.

In the second stage, we carefully curate high-resolution images from Unsplash [Ali et al., 2023] and assign them to the selected annotators. Each annotator assesses the assigned images for their suitability based on predefined quality criteria. Images that fail to meet these criteria are replaced with new candidates, while suitable images proceed to the next stage.

In the third stage, annotators create novel and diverse editing instructions for the curated images. They utilize the DALL-E 2 platform to define mask areas, generate edited images, and provide captions for the results. Each submission package includes the original image, the mask, the edited image, the editing instruction, and the caption. These submissions are then forwarded to administrators for quality review to make sure being aligned with human performance.

In the fourth stage, administrators perform a two-tier quality review and human feedback. If the edited image meets the required quality standards but the accompanying instructions or captions

- Stage 2

Image Selection and Quality Check

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

…

- Stage 3

Stage 1 Annotator Training and Selection

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Toturial Quiz

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

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

[Figure 49]

[Figure 50]

Stage 4 Review and Database Inclusion HI-Edit

Masking and Generation

Figure 2: Overview of data collection process.

are problematic, the submission is returned to stage three for re-annotation. Submissions with poor editing quality are discarded. We refer to this process as human-rewarded, as annotators with good performance receive higher rewards, while those with poor performance are removed from the annotator teams. Examples of failure cases excluded from HumanEditcan be found in Appendix C. Data pairs that pass the quality threshold are included in the final HumanEditdataset. Over the course of the annotation process, approximately 20,000 images were annotated, with 5,751 high-quality images retained in the final dataset.

Finally, we leverage Llama 3.2-Vision [Dubey et al., 2024] to generate refined captions for all original and edited images, ensuring consistency and clarity across the dataset.

- Table 2: Comparison of existing image editing datasets. “Real Image for Edit” denotes whether real images are used for editing instead of images generated by models. “Real-world Scenario” indicates whether images edited by users in the real world are included. “Human” denotes whether human annotators are involved. “Ability Classification” refers to evaluating the edit ability in different dimensions. “Mask” indicates whether rendering masks for editing is supported. “Non-Mask Editing” denotes the ability to edit without mask input.

Dataset Real Image for Edit Real-world Scenario Human Ability Classification Mask Non-Mask Editing InstructPix2Pix [Brooks et al., 2023] ✗ ✗ ✗ ✗ ✗ ✓

MagicBrush [Zhang et al., 2024a] ✓ ✗ ✓ ✗ ✓ ✗

GIER [Shi et al., 2020] ✓ ✓ ✓ ✗ ✗ ✗ MA5k-Req [Shi et al., 2021] ✓ ✓ ✓ ✗ ✗ ✗

TEdBench [Kawar et al., 2023] ✓ ✓ ✓ ✗ ✗ ✓

HQ-Edit [Hui et al., 2024] ✗ ✗ ✗ ✗ ✗ ✓ SEED-Data-Edit [Ge et al., 2024a] ✓ ✓ ✓ ✗ ✗ ✓

AnyEdit [Yu et al., 2024a] ✓ ✓ ✗ ✗ ✓ ✓ HumanEdit ✓ ✓ ✓ 6 ✓ ✓

### 3 Dataset Statistics

Related Datasets Comparison. InstructPix2Pix [Brooks et al., 2023] utilizes Prompt-toPrompt [Hertz et al., 2022] to generate source and target images based on input and edit prompts from the LAION-Aesthetics [Schuhmann et al., 2022b] dataset. However, all images are model-generated, thereby lacking real-world authenticity. MagicBrush [Zhang et al., 2024a] employs crowdworkers on Amazon Mechanical Turk (AMT) to manually annotate images from the MS COCO dataset, using the DALL-E 2 platform for multi-round editing annotations. Although it offers diversity, all images

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Remove the man and his child. Remove Tokyo Tower

Remove the Buck

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Remove the kitten from the bush. Remove all the cattle. Add a ship on the lake

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Remove the lime from the noodles. Remove a white macaron. Add a rose on the woman’s hands

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Replace the distinctive object in the picture with one that looks similar to the other objects

Change the woman's outfit in the picture to a brown suspender skirt with a blue blouse underneath

Replace the blue mask on a woman's face with a silver one.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Remove tall palm trees from the house Replace the cucumber slices on your eyes with a piece of chrysanthemum

Alter curly hair to normal short hair for women

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Removing the baby and the mother holding it

Let the person who doesn't have a necklace also have a necklace

Change your pants and socks to pants of the same color as your shirt

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

The word '猫' is replaced with the extension of the plum blossom above

Add a heart shape in the lower center of the image

Add a fly to the transparent ball of the prototype

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Turn the bag into a bouquet of flowers. Remove the phone on the desk What if the cake was never cut?

Figure 3: More examples of instruction-guided image editing in HumanEdit.

are sourced from MS COCO and only support masked editing. HQ-Edit leverages GPT-4 [OpenAI, 2023a] to generate image descriptions and editing instructions, creating paired images with GPT-

- 4V [OpenAI, 2023b] and DALL-E 3 [Betker et al., 2023]. These paired images are divided into source and target images, with instructions rewritten by GPT-4V. Nonetheless, this method often fails to preserve the fine-grained details of the source image in the target image, resulting in generated images that lack realism. GIER [Shi et al., 2020] and MA5k-Req [Shi et al., 2021] only support filter changes, offering very limited richness. SEED-Data-Edit [Ge et al., 2024a] boasts a larger dataset and supports unmasked editing, but it lacks capability classification and does not provide masks. A more detailed comparison can be found in Table 2. Additionally, although SEED-Data-Edit has a large scale, the annotation process involves using several VLMs to generate instructions and captions, which may introduce hallucinations [Yu et al., 2024b, Liu et al., 2024]. In contrast, HumanEdit uses high-resolution original images, selecting higher-quality images as sources through VLM scoring and human selection. It supports both masked and unmasked editing, providing a high-quality dataset and benchmark for image editing.

High Image Resolution. The distribution of input image resolutions is depicted in Figure 6(c). Most images in our dataset (62.3%) have a resolution greater than 1000, with 33.8% of them exceeding 1200. For image edited by DALL-E2 [Ramesh et al., 2022], lower resolution images are upsampled first, meaning that higher input image resolutions result in higher fidelity in the edited output images, as output images with a fixed size of 1024 × 1024. In contrast, MagicBrush has only 46.6 input images above 1000, 25.3% less than us, with the rest of input images has only 512 × 512 resolution.

Support Non-mask Editing. HumanEdit operates on real images and does not require additional inputs such as image masks or extra views of the object. As shown in Figure 6(a), HumanEdit provides masks for all data, with 46.5% of the data supporting editing without masks. In contrast, datasets like MagicBrush require masks for editing. We believe this feature makes HumanEdit more versatile and applicable, as real-world editing often does not involve using masks.

Diverse Data Sources. As illustrated in Figure 6(b), the majority of our data originates from Unsplash [Ali et al., 2023], a website dedicated to photography. The images on this platform are known for their exceptional aesthetic quality, characterized by professional composition, lighting, and subject matter [Li et al., 2023b]. Our dataset is a carefully curated subset, selected from a pool of 57,000 crawled images, ensuring high quality and rich diversity.

| | |
|---|---|
| | |

[Figure 99]

250

200

Frequency

150

100

50

0

flowerspeoplenumberhandgirlhatclothesflowerhandsmanwomanpersonbirdcolorboyleavesmoonbutterflyhair treeheaddogballdressskytreescornercarcupchild

- Figure 4: (a) The distribution chart of the first 30 objects in the editing instructions for HumanEdit. (b) The word cloud representation of the objects present in the editing instructions for HumanEdit.

Rich Editing Instruction. HumanEdit encompasses a diverse array of edit instructions, including object addition, replacement, and removal, action changes, color alterations, text or pattern modifications, and object quantity adjustments. Keywords associated with each edit type span a wide spectrum, encompassing various objects, actions, and attributes, as depicted in Figure 9 and Figure 10. This diversity underscores HumanEdit’s ability to capture a comprehensive range of editing scenarios, facilitating robust training and evaluation of instruction-guided image editing models.

### 4 HI-EDIT Benchmark

Baselines. To provide the performance benchmark on HumanEdit, we consider multiple baselines in both mask-free and mask-provided settings. For all baselines, we adopt the default hyperparameters available in the official code repositories to guarantee reproducibility and fairness.

straighten

spread

open

clench

pick

reach

hand

fold

shake

unroll arm

insert

number

take

hug

head

unfold

eye

raise

stretch butterfly

lower

hair

hide

top

hang

bee

point

pull

hat

twist

sky

bow

close

color

have

pant

coil

Action

hold

clothe

put

skirt

organize

aircraft

face

apple

lift

ship

stand

drop

girl

reduce

| |
|---|

Counting

flower

tilt

increase

strawberry

boy

show

| |
|---|

puppy

Add

introduce

ball

| |
|---|

boat

add

| |
|---|

moon Replace

bird

decrease

bag

| |
|---|

balloon

change

Relation

stone

dog

alter

bottle

turn

cup

embrace

phone

glass

replace

plant

child

place

car

transform

switch

man

| |
|---|

swap

mountain

move

leave

Remove

transfer

pattern

keep

make

person

pot

tree

| |
|---|

woman

people

remove

house

building

word

other

eliminate

- Figure 5: The river chart of HumanEdit-full. The first node of the river represents the type of edit, the second node corresponds to the verb extracted from the instruction, and the final node corresponds to the noun in the instruction. To maintain clarity, we only selected the top 50 most frequent nouns. The river chart of HumanEdit-core can be seen in Figure 11 in Appendix.

no need for mask 53.1%

46.9% need mask

(a) MASK distribution

MSCOCO

4.2%

Unsplash

95.8%

(b) Image Source Distribution

500-600

2.4% 600-800

31.4%

800-1000

3.5%

1000-1200

28.9%

>1200 33.8%

(c) Resolution Distribution

500 53.4%

46.6% 1024

(d) Resolution Distribution

- Figure 6: (a) The distribution of images for which HumanEdit requires masking, where no need for mask refers to editing instructions that are already clear and comprehensive enough, and we believe that no masking is necessary for the model to complete the editing. (b) The distribution of the sources of all input images for HumanEdit. (c) The distribution of resolutions for all input images in HumanEdit. (d) The distribution of resolutions for all input images in MagicBrush.

##### For mask-free baselines, we consider:

- • InstructPix2Pix [Brooks et al., 2023]. InstructPix2Pix Utilizes automatically generated instruction-based image editing data by large language models to fine-tune Stable Diffusion [Rombach et al., 2022a], enabling instruction-based image editing during inference without requiring any test-time tuning.
- • MGIE [Fu et al., 2023]. MLLM-Guided Image Editing (MGIE) explores how Multimodal Large Language Models [Chow et al., 2024, Pan et al., 2024, Li et al., 2023a] assist in generating edit instructions. MGIE learns to derive expressive instructions and provides explicit guidance for the editing process. The model integrates this visual imagination and performs image manipulation through end-to-end training.
- • HIVE SD1.5 [Zhang et al., 2024b]. HIVE stands for Human Feedback for Instructional Visual Editing. The reward model is trained on supplementary data annotated by humans who rank the variant outputs of the fine-tuned InstructPix2Pix model. HIVE undergoes further fine-tuning using this reward model derived from these human rankings.
- • MagicBrush [Zhang et al., 2024a]. MagicBrush curates a well-structured editing dataset with detailed human annotations and fine-tunes its model on this dataset using the InstructPix2Pix Brooks et al. [2023] framework.

##### For mask-provided baselines, we consider:

- Table 3: Quantitative study on mask-free baselines on HumanEdit. The best results are marked in bold.

Methods L1↓ L2↓ CLIP-I↑ DINO↑ CLIP-T↑

HumanEdit-full

InstructPix2Pix [Brooks et al., 2023] 0.1601 0.0551 0.7716 0.5335 0.2591 MGIE [Fu et al., 2023] 0.1240 0.0535 0.8697 0.7221 0.2661 HIVE SD1.5 [Zhang et al., 2024b] 0.1014 0.0278 0.8526 0.7726 0.2777 MagicBrush [Zhang et al., 2024a] 0.0807 0.0298 0.8915 0.7963 0.2676

HumanEdit-core

InstructPix2Pix [Brooks et al., 2023] 0.1625 0.0570 0.7627 0.5349 0.2533 MGIE [Fu et al., 2023] 0.1294 0.0610 0.8670 0.7359 0.2589 HIVE SD1.5 [Zhang et al., 2024b] 0.1162 0.0373 0.8441 0.7038 0.2563 MagicBrush [Zhang et al., 2024a] 0.0760 0.0283 0.8946 0.8121 0.2619

- • Blended Latent Diffusion SDXL [Avrahami et al., 2023]. Latent Diffusion [Rombach et al., 2022a] can generate an image from a given text (text-to-image LDM). However, it lacks the ability to edit an existing image in a local way. Blended Latent Diffusion incorporates Blended Diffusion [Avrahami et al., 2022] into text-to-image LDM by utilizing CLIP [Radford et al., 2021] guidance during the masked region denoising process and integrates it with the context from the noisy source image at each denoising timestep to enhance the region-context consistency of the generated target image.
- • GLIDE [Nichol et al., 2021]. GLIDE stands for Guided Language to Image Diffusion for Generation and Editing. To achieve better results on image editing tasks, OpenAI fine-tunes their model by modifying the model architecture to have four additional input channels: a second set of RGB channels and a mask channel. In addition, they initialize the corresponding input weights for these new channels to zero before fine-tuning. During fine-tuning, random regions of training examples are erased, and the remaining portions are fed into the model along with a mask channel as additional conditioning information.
- • aMUSEd [Patil et al., 2024]. aMUSEd is a lightweight text-to-image model based on the MUSE architecture, which supports zero-shot image editing. For editing tasks, the mask directly determines which tokens are initially masked.
- • Meissonic [Bai et al., 2024]. Meissonic is a non-autoregressive mask image modeling text-to-image synthesis model that can generate 1024 x 1024 high-resolution images. By incorporating a comprehensive suite of architectural innovations, advanced positional encoding strategies, and optimized sampling conditions, Meissonic substantially improves MIM’s performance and efficiency to a level comparable with state-of-the-art diffusion models like SDXL. Due to the architecture of masked generative transformer, Meissonic also supports zero-shot image editing by masking the corresponding tokens.

Evaluation Metrics. Follow the similar settings from previous works [Brooks et al., 2023, Zhang et al., 2024a], we utilize L1 and L2 to measure the average pixel-level absolute difference between the generated image and ground truth image, and CLIP-I and DINO to measure the image quality with the cosine similarity between the generated image and reference ground truth image using their CLIP [Radford et al., 2021] and DINO [Caron et al., 2021] embeddings, and CLIP-T [Ruiz et al., 2023, Chen et al., 2024] to measure the text-image alignment with the cosine similarity between local descriptions and generated images CLIP embeddings.

HumanEdit Benchmark. Tables 3 and 4 summarize the quantitative results for mask-free and mask-provided methods, respectively. Mask-free methods are given only textual instructions to edit images, while mask-provided methods receive both instructions and corresponding masks.

Additionally, Table 5 presents the quantitative results across six distinct types of editing instructions. We believe this categorization can facilitate fine-grained advancements in instruction-based image editing tasks. The table reveals several noteworthy observations: for instance, most methods perform better on Add tasks than on Remove tasks. Moreover, mask-provided methods generally achieve superior performance in semantic-level evaluation metrics compared to pixel-level ones.

###### Table 4: Quantitative study on mask-provided baselines on HumanEdit. The best results are marked in bold.

Methods L1↓ L2↓ CLIP-I↑ DINO↑ CLIP-T↑

HumanEdit-full

Blended Latent Diff. SDXL [Avrahami et al., 2023] 0.0481 0.0151 0.9178 0.8481 0.2681 GLIDE [Nichol et al., 2021] 0.0391 0.0120 0.9388 0.8800 0.2676 aMUSEd [Patil et al., 2024] 0.0673 0.0187 0.9149 0.8588 0.2771 Meissonic [Bai et al., 2024] 0.0627 0.0177 0.9324 0.8806 0.2710

HumanEdit-core

Blended Latent Diff. SDXL [Avrahami et al., 2023] 0.0496 0.0162 0.9116 0.8550 0.2640 GLIDE [Nichol et al., 2021] 0.0379 0.0113 0.9413 0.8961 0.2656 aMUSEd [Patil et al., 2024] 0.0665 0.0184 0.9138 0.8743 0.2747 Meissonic [Bai et al., 2024] 0.0608 0.0166 0.9348 0.8943 0.2694

HumanEdit-mask

Blended Latent Diff. SDXL [Avrahami et al., 2023] 0.0478 0.0154 0.9065 0.8223 0.2650 GLIDE [Nichol et al., 2021] 0.0377 0.0117 0.9343 0.8687 0.2665 aMUSEd [Patil et al., 2024] 0.0654 0.0179 0.9097 0.8497 0.2785 Meissonic [Bai et al., 2024] 0.0604 0.0166 0.9303 0.8783 0.2715

To provide further insights, Figures 7 and 8 showcase visual examples of results from mask-provided methods. These examples highlight that existing methods perform well on Add and Remove editing tasks but struggle with more complex tasks such as Relation and Action. Furthermore, even for Add tasks, challenges persist in cases requiring domain-specific knowledge or handling unfamiliar instructions, such as “Add a petal in the middle of the white puppy’s forehead.”

It is important to note that comparisons between methods might be unfair because of differences in implementation and fine-tuning. These tables are intended to establish a benchmark for HumanEdit to support future research and evaluation.

### 5 Conclusion

In this work, we introduce HumanEdit, a high-quality, human-rewarded dataset for instructional image editing. Previous large-scale editing datasets often incorporate minimal human feedback, leading to challenges in aligning datasets with human preferences. HumanEdit bridges this gap by employing human annotators to construct data pairs and administrators to provide feedback. Designed to address the growing demand for precise and versatile image editing capabilities, HumanEdit comprises six types of editing instructions: Action, Add, Counting, Relation, Remove, and Replace. The dataset stands out for its meticulous quality control, diverse sources, and inclusion of high-resolution images, offering unparalleled reliability and utility for model development. Furthermore, HumanEdit provides explicit differentiation between tasks requiring masks and those that do not, ensuring comprehensive support for a wide range of editing scenarios.

#### 6 Acknowledgements This work was supported in part by NUS Start-up Grant A-0010106-00-00.

[Relation] Place the woman on the right side of the man.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Replace] Change the orange bag to green.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Add] Add an eagle.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Action] Show the person jumping on skateboards.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Replace] Replace tulips with sunflowers.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Add] Change the lipstick to red.

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Add] Add a dog.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Remove] Remove the window frames.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

- Figure 7: Qualitative comparisons between mask-provided baselines. The first three rows show the original images, corresponding masks, and ground truth edited images from DALL-E 2. The subsequent four rows present results generated by Blended Latent Diffusion SDXL, GLIDE, aMUSEd, and Meissonic, respectively.

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

- Figure 8: Qualitative comparisons between mask-provided baselines. The first three rows show the original images, corresponding masks, and ground truth edited images from DALL-E 2. The subsequent four rows present results generated by Blended Latent Diffusion SDXL, GLIDE, aMUSEd, and Meissonic, respectively.

Table 5: Quantitative study on six different types of editing instructions on HumanEdit. The best results are marked in bold.

Methods L1↓ L2↓ CLIP-I↑ DINO↑ CLIP-T↑

HumanEdit-Add

InstructPix2Pix [Brooks et al., 2023] 0.1152 0.0329 0.8135 0.6230 0.2764 MGIE [Fu et al., 2023] 0.0934 0.0274 0.8770 0.7391 0.2806 HIVE SD1.5 [Zhang et al., 2024b] 0.0885 0.0234 0.8863 0.7811 0.2706 MagicBrush [Zhang et al., 2024a] 0.0580 0.0167 0.9102 0.8562 0.2745 Blended Latent Diff. SDXL [Avrahami et al., 2023] 0.0344 0.0073 0.9285 0.8856 0.2665 GLIDE [Nichol et al., 2021] 0.0315 0.0078 0.9410 0.8995 0.2600 aMUSEd [Patil et al., 2024] 0.0581 0.0130 0.9148 0.8672 0.2695 Meissonic [Bai et al., 2024] 0.0544 0.0129 0.9303 0.8787 0.2669

HumanEdit-Action

InstructPix2Pix [Brooks et al., 2023] 0.1324 0.0398 0.7514 0.5789 0.2617 MGIE [Fu et al., 2023] 0.0982 0.0383 0.8788 0.7909 0.2658 HIVE SD1.5 [Zhang et al., 2024b] 0.0972 0.0280 0.8592 0.7613 0.2640 MagicBrush [Zhang et al., 2024a] 0.0723 0.0245 0.9028 0.8357 0.2668 Blended Latent Diff. SDXL [Avrahami et al., 2023] 0.0416 0.0109 0.9391 0.9015 0.2712 GLIDE [Nichol et al., 2021] 0.0384 0.0114 0.9487 0.9018 0.2683 aMUSEd [Patil et al., 2024] 0.0629 0.0156 0.9230 0.8919 0.2732 Meissonic [Bai et al., 2024] 0.0577 0.0145 0.9430 0.9126 0.2677

HumanEdit-Counting

InstructPix2Pix [Brooks et al., 2023] 0.1628 0.0586 0.8124 0.5850 0.2716 MGIE [Fu et al., 2023] 0.1380 0.0641 0.8726 0.6971 0.2716 HIVE SD1.5 [Zhang et al., 2024b] 0.1211 0.0442 0.8826 0.7431 0.2705 MagicBrush [Zhang et al., 2024a] 0.1058 0.0434 0.8677 0.7103 0.2707 Blended Latent Diff. SDXL [Avrahami et al., 2023] 0.0527 0.0180 0.9334 0.8892 0.2766 GLIDE [Nichol et al., 2021] 0.0392 0.0127 0.9523 0.9104 0.2772 aMUSEd [Patil et al., 2024] 0.0699 0.0213 0.9270 0.8816 0.2814 Meissonic [Bai et al., 2024] 0.0674 0.0217 0.9394 0.8967 0.2750

HumanEdit-Remove

InstructPix2Pix [Brooks et al., 2023] 0.1624 0.0504 0.7240 0.4188 0.2325 MGIE [Fu et al., 2023] 0.1259 0.0572 0.8677 0.7235 0.2525 HIVE SD1.5 [Zhang et al., 2024b] 0.1179 0.0375 0.8362 0.6562 0.2474 MagicBrush [Zhang et al., 2024a] 0.0690 0.0232 0.8985 0.8249 0.2572 Blended Latent Diff. SDXL [Avrahami et al., 2023] 0.0451 0.0133 0.9055 0.8322 0.2608 GLIDE [Nichol et al., 2021] 0.0313 0.0072 0.9493 0.9119 0.2661 aMUSEd [Patil et al., 2024] 0.0621 0.0156 0.9148 0.8702 0.2715 Meissonic [Bai et al., 2024] 0.0557 0.0132 0.9367 0.9048 0.2673

HumanEdit-Relation

InstructPix2Pix [Brooks et al., 2023] 0.1741 0.0647 0.8069 0.5851 0.2828 MGIE [Fu et al., 2023] 0.1420 0.0656 0.8762 0.7061 0.2768 HIVE SD1.5 [Zhang et al., 2024b] 0.1298 0.0460 0.8689 0.7005 0.2793 MagicBrush [Zhang et al., 2024a] 0.0884 0.0334 0.8985 0.7865 0.2823 Blended Latent Diff. SDXL [Avrahami et al., 2023] 0.0628 0.0213 0.9190 0.8174 0.2832 GLIDE [Nichol et al., 2021] 0.0553 0.0192 0.9136 0.7983 0.2755 aMUSEd [Patil et al., 2024] 0.0809 0.0267 0.9076 0.8095 0.2862 Meissonic [Bai et al., 2024] 0.0825 0.0283 0.9171 0.8142 0.2768

HumanEdit-Replace

InstructPix2Pix [Brooks et al., 2023] 0.1910 0.0770 0.7887 0.5692 0.2697 MGIE [Fu et al., 2023] 0.1391 0.0620 0.8603 0.6946 0.2698 HIVE SD1.5 [Zhang et al., 2024b] 0.1265 0.0443 0.8582 0.7087 0.2726 MagicBrush [Zhang et al., 2024a] 0.0984 0.0409 0.8757 0.7513 0.2716 Blended Latent Diff. SDXL [Avrahami et al., 2023] 0.0567 0.0206 0.9095 0.8096 0.2683 GLIDE [Nichol et al., 2021] 0.0495 0.0188 0.9194 0.8247 0.2663 aMUSEd [Patil et al., 2024] 0.0761 0.0237 0.9072 0.8259 0.2861 Meissonic [Bai et al., 2024] 0.0710 0.0227 0.9239 0.8462 0.2762

### A More Figures

In addition to the statistical charts mentioned in Section 3, we also provide a sunburst chart analysis of the instructions, as shown in Figure 9 and Figure 10. Due to space constraints, we have selected only the top 50 most frequent nouns for visualization.

woman

balloon

word

logo

person

watch

man

bird

bottle

boy

building

people

car

remove

child

cyclist

swing

bat

separate

dog

turn

sheep

show

doughnut

flamingo

replace

bud

put

transform

flower

girl

make

mountain

swap

fruit

change

add

hat

gesture

girl

sunflower

tree

mountain

plant

passenger

penguin

fruit

petal

banana

pumpkin

belt

plant

pyramid

reflection

bee

clothe

pant

top

ripple

ring

hat

skirt

count

12

10

8

6

4

2

- Figure 9: An Overview of Keywords in HumanEdit-core Edit Instructions: The inner circle represents the verb in the edit instruction, while the outer circle illustrates the noun following the verb in each instruction.

remove

change

reduce

replace

add

turn

show

move

raise

put

place

hold

people

person

tree

man

pattern

flower

leave

word bird

building

woman

car

girl

logo

mountain

color clothe

flower

hat

skirt number pant

top dress

eye

sky

position

number

flower

flower

hat

hair

bee

boat

ball

balloon

bird

flower

people

girl

boy

puppy

ball

moon

bird

hand

arm

hand

ball

hand

10

20

30

40

50

60

70

80

count

- Figure 10: An Overview of Keywords in HumanEdit-full Edit Instructions: The inner circle represents the verb in the edit instruction, while the outer circle highlights the noun associated with the verb in each instruction.

###### The river chart of HumanEdit-core is shown in Figure 11. The full can be seen in Figure 5.

| |
|---|

people man

word balloon

watch

child flamingo

logo

| |
|---|

| |
|---|

bird

iphone

car

building

Remove

remove

cyclist fruit

woman

boy

person

dog

flower

pumpkin

elephant

number

avocado

bottle

tree

swap

move blueberry

Counting plant

reduce

boat

increase

glass bouquet

Relation

place

bud

keep girl

add shoe

fork

Add

bee

put

ball

cup

transform

moon Replace

hold

mountain

show

| |
|---|

wing

turn

apple

hand

replace

lemon

Action hat

change

phone

spread

clothe

raise

top

unfold

make

skirt

pant

gesture

- Figure 11: The river chart of HumanEdit-core. The first node of the river represents the type of edit, the second node corresponds to the verb extracted from the instruction, and the final node corresponds to the noun in the instruction. To maintain clarity, we only selected the top 50 most frequent nouns.

### B Guidance Book for Annotators

##### B.1 Edit Cases for Annotators

The following provides some annotation examples and the required submission content for annotators. We have conducted basic classification to help annotators develop a better understanding of the annotation task and to enrich the editing content as much as possible.

- (1) Object Related. Object-centered editing can be categorized into the following four types.

- (1.1) Object Removal. As shown in Figure 12, this task primarily involves removing certain objects from an image, typically those that are more prominent or easily distinguishable.

[Figure 212]

[Figure 213]

[Figure 214]

Editing instruction: Remove all the people. Edited image description: Two yellow umbrellas with black patches are placed on the grass. Three additional trees are scattered across the grass. In the distance, there is a road and a park.

Original Mask Output

Figure 12: Case of Object Removal.

- (1.2) Object Replacement. As shown in Figure 13 and Figure 14, we modify the type of an object, change a part of an object, or alter its shape.

Editing instruction: Let the table turns into a white puppy. Edited image description: A white puppy sits on a red carpet. In front of it is a black sofa. On the left is a white chair with a red pillow on it. There is a bookshelf between the sofa and the chair.

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Original Mask Output

Figure 13: Object Replacement Example I.

Editing instruction: Let windows become square. Edited image description: Above the bathtub is a square window.

[Figure 221]

[Figure 222]

[Figure 223]

Original Mask Output

Figure 14: Object Replacement Example II.

- (1.3) Object Addition. As shown in Figure 15, we add an object to the original image.

Editing instruction: Add a mobile phone. Edited image description: There is a mobile phone next to the two pizzas.

[Figure 224]

[Figure 225]

[Figure 226]

Original Mask Output

Figure 15: Case of Object Addition.

- (1.4) Object Counting Change. As shown in Figure 16, we can also alter the number of objects in the image. However, it is important to note that the number of objects cannot be reduced to zero (which would be equivalent to removal), nor can it be increased from none to any (which would be

considered addition).

Editing instruction: Let the gifts’ number changed from 1 to 4. Edited image description: There are 4 gifts wrapped in different sizes on the side of a blue box.

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Original Mask Output

Figure 16: Case of Object Counting Change.

- (2) Action Change. As shown in Figure 17, if the subject is a specific organism, its actions can also be altered.

Editing instruction: The elephant turned up its trunk. Edited image description: An elephant stands facing each other, its trunk turned up.

Original Mask Output

[Figure 231]

[Figure 232]

[Figure 233]

Figure 17: Case of Action Change.

- (3) Relation Change. As shown in Figure 18, another type of editing involves modifying the relationships between objects.

Editing instruction: Move the gloves from the right side of the cloth basket to the left side. Edited image description: From left to right are two gloves, a white box without straps and a green shoe.

[Figure 234]

[Figure 235]

[Figure 236]

Original Mask Output

Figure 18: Case of Relation Change.

##### B.2 Notes for Annotators

- (1) Selection of Prompt Words. When using DALL-E 2, if only an editing instruction (as shown in Figure 1) is provided, the model’s generated results are often poor. It is recommended to use a detailed description of the target image (as shown in Figure 2). For example: Editing instruction: "Let the boy turn into a girl."

[Figure 237]

- Figure 19: An Example of Prompt Word Selection

Target Image Caption: Four parrots are perched on a girl’s shoulders, arms, and head.

[Figure 238]

- Figure 20: An Example of Prompt Word Selection

- (2) Image Resolution. After uploading the image, click ’crop’ first, then click ’Edit image’ to proceed with editing.

[Figure 239]

- Figure 21: Performing a Crop Operation on the DALL-E 2 Platform.

[Figure 240]

- Figure 22: Performing an Editing Operation on the DALL-E 2 Platform.

- (3) Avoid Editing Irrelevant Areas. When masking areas, avoid using too large a mask, as this can lead to distortion or result in editing that does not cover the intended area. For example, in the Figure 23 below, the boat paddle disappears, which is unreasonable.

[Figure 241]

Figure 23: An Illustration of Avoiding Edits in Irrelevant Areas.

Additionally, if the editing task is to add a giraffe, the expected result should be the addition of two giraffes as shown in Figure 24. However, the output image shows excessive changes (likely due to an overly large mask area). A reminder: the mask area should not be too large; it should be appropriate. Also, the giraffe’s head in this example is generated unrealistically.

[Figure 242]

- Figure 24: An Illustration of Avoiding Edits in Irrelevant Areas.

When the instruction is to remove a person, it is best not to change the car for Figure 25.

[Figure 243]

- Figure 25: An Illustration of Avoiding Edits in Irrelevant Areas.

The following masking as shown in Figure 26 is done well: the instruction is to change the background, and everything except for the dog is masked.

[Figure 244]

Figure 26: An Illustration of Avoiding Edits in Irrelevant Areas.

- (4) Quality of Edits Ensure. DALL-E 2 sometimes struggles to interpret instructions accurately, so attention to detail in editing structures is important. For example, in the following case shown in Figure 27, the fingers are distorted and do not resemble a normally outstretched hand.

[Figure 245]

Figure 27: A Case for Ensuring Edit Quality.

As demonstrated in Figure 28 , the image description is "The back view of a large calico cat sitting next to two other cats," but the actual image shows four cats.

[Figure 246]

Figure 28: A Case for Ensuring Edit Quality.

The car door in Figure 29 has disappeared, which is also unreasonable (this issue was caused by an overly large masked area).

[Figure 247]

Figure 29: A Case for Ensuring Edit Quality.

- (5) Success Rate. DALL-E 2 has a relatively low success rate. If multiple regenerations or instruction modifications do not yield satisfactory results, it may be best to abandon the task. The exact number of attempts before abandonment is left to the discretion of the annotator. For simplicity, only one editing instruction should be tried for each image, and the best result should be selected.
- (6) Consistency in Style Before and After Editing. If the original image is black and white, the edited result should also be in black and white. Generally, DALL-E 2’s generated results tend to adhere to the original style, so there is no need to explicitly guide the editing in terms of style. However, attention should be paid when selecting the final result to ensure consistency.

[Figure 248]

Figure 30: An Illustration of Consistency in Style Before and After Editing.

- B.3 Initial Image Selection

As mentioned in Section 2, we implement a rigorous selection process to ensure the quality of the original images. It is important to note that, at the beginning of the annotation process, annotators are still given the opportunity to reselect the original image. Figure 31 below illustrates an example of the selection process. For instance, image (a) is acceptable, while (b) contains some unusual artifacts, (c) has poor image quality, and (d) has low resolution and lacks sufficient visual information.

[Figure 249]

Figure 31: Examples of valid and invalid images. The first image is valid, while the following three images are invalid.

- B.4 Image Editing Process and Annotation Platform

- (1) Log in to the DALL·E 2 platform and click "Try DALL-E" to upload an image.

[Figure 250]

Figure 32: Log in to the DALL·E 2 platform and click "Try DALL-E" to upload an image.

##### (2) After uploading the image, a cropping page will be displayed.

[Figure 251]

Figure 33: After uploading the image, a cropping page will be displayed.

##### (3) Click the "Edit" button to enter the editing window.

[Figure 252]

- Figure 34: Click the "Edit" button to enter the editing window.

##### (4) Drag the editing points to select the area to be edited.

[Figure 253]

- Figure 35: Drag the editing points to select the area to be edited.

Next, input the editing instructions in the text box. For example, if your task is to change an object, first select the person, and then define the editing instruction as “change the boy into a girl.” At this point, combine the previously selected description and imagine the expected edited image (the more detailed the description, the better), and enter it in the text box. For example, “Four parrots are perched on a cute girl’s arms and shoulders” (Note: the output box for editing instructions will only appear after selecting the editing contours). Then save the mask and choose to have the model generate the image.

[Figure 254]

Figure 36: Input the editing instructions in the text bo.

[Figure 255]

- Figure 37: Generate edited images.

If the generated result is of poor quality (e.g., none of the images meet the requirements), you can click the “regenerate” button to try again.

[Figure 256]

- Figure 38: Regenerate edited images.

However, please avoid generating the same instruction more than three times. Instead, try modifying the instruction to make it more precise. For example, change the expected image description to “A cute little girl with her arms outstretched, with four parrots perched on her head, shoulders, and arms.”

[Figure 257]

Figure 39: Regenerated images are still not satisfactory and may require revised instructions.

Once you find an image that seems appropriate, click on it to download and finish the editing process.

[Figure 258]

Figure 40: Download and finish the editing process.

- (5) Result selection. Please ensure that the final selected image is semantically accurate and as realistic as possible, without significant flaws. Below are some examples of poor results in Figure 41. Please try to avoid these mistakes, as we will consider instructions that result in such issues as non-compliant.

[Figure 259]

Figure 41: Defective Image Example.

- (6) Submission of results. Finally, you need to submit the following materials as a group to our platform.

[Figure 260]

Figure 42: Submission Example.

### C Failure Cases (not included in HumanEdit)

It is important to emphasize that our images underwent rigorous review and filtering. As mentioned in Section 2, the experts annotated approximately 20,000 images, but only 5,751 images were retained in the final HumanEdit. In this section, we present some common failure cases encountered during our data validation process. Additional examples can be found in Appendix B.

##### C.1 Inherent Limitations of DALL-E 2

The image generation rate of DALL-E 2 is relatively low, and we have identified several inherent limitations.

Mismatch between editing results and instructions. For example, in Figure 43, the instruction was "make the nose larger," but no modification was applied. In Figure 44, the instruction was "a lantern hanging in front of the window," but DALL-E 2 simply removed the original object without replacing it. In Figure 45, the instruction was "a plate of cucumbers and a bouquet of roses," but the roses did not appear.

[Figure 261]

- Figure 43: An Illustration of the Mismatch Between Editing Results and Instructions.

[Figure 262]

- Figure 44: An Illustration of the Mismatch Between Editing Results and Instructions.

[Figure 263]

- Figure 45: An Illustration of the Mismatch Between Editing Results and Instructions.

Limited Editing Capabilities for Specific Types. DALL-E 2 exhibits limited performance in editing certain types of content, such as counting and relational editing tasks. Similar limitations are observed in other models as well. For instance, in Figure 46, the editing instruction "the girl is standing on tiptoe" was attempted multiple times by the experimental team, but a satisfactory result could not be achieved despite dozens of trials. A similar issue is seen in Figure 47, where the editor intended to close the owl’s eyes, but DALL-E 2 continuously altered the state of the owl’s eyes without successfully achieving the desired effect.

In the example shown in Figure 48, the goal was to "add a red barbell," but DALL-E 2 appears to be insensitive to the number of objects, with the resulting images mostly [Podell et al., 2023, Ge et al., 2024b] showing a reduction in the number of objects rather than an addition. In Figure 49, the editor intended to move the blueberry from the top right corner of the spoon to the top left corner, but this attempt also failed. The issue of removing rather than adding objects seems to be a common challenge across most models and may represent a significant current limitation.

[Figure 264]

- Figure 46: An Illustration of the Limited Editing Capabilities for Specific Types.

[Figure 265]

- Figure 47: An Illustration of the Limited Editing Capabilities for Specific Types.

[Figure 266]

- Figure 48: An Illustration of the Limited Editing Capabilities for Specific Types.

[Figure 267]

- Figure 49: An Illustration of the Limited Editing Capabilities for Specific Types.

Below are additional examples where DALL-E 2 fails to perform effective editing. In such cases, annotators may need to try multiple attempts and adjust the masked regions, as generation is limited to those areas. The instruction for Figure 1 is "A young boy wearing a beret"; the instruction for Figure 2 is "A girl sitting far from the computer, pointing at it"; and the instruction for Figure 3 is "A man raising his left fist."

[Figure 268]

- Figure 50: An Illustration of the Limited Editing Capabilities for Specific Types.

[Figure 269]

- Figure 51: An Illustration of the Limited Editing Capabilities for Specific Types.

[Figure 270]

- Figure 52: An Illustration of the Limited Editing Capabilities for Specific Types.

##### C.2 Editing Errors

Some of the editing results exhibit defects, which we have excluded from our analysis. For example, in Figure 53, the flower appears somewhat distorted. In Figure 54, the instruction is "add printed patterns," but the generated image lacks any printed patterns. In Figure 55, the instruction is "The puppy’s ears stood up," yet the editing effect is not clearly visible. In Figure 56, the instruction is to raise the person’s head, but instead, the person’s eyes have been altered.

[Figure 271]

Figure 53: An example of object distortion.

[Figure 272]

Figure 54: The discrepancy between the instruction and the generated image.

[Figure 273]

- Figure 55: An example of subtle editing effects.

[Figure 274]

- Figure 56: An example of inconsistent editing.

##### C.3 Other Errors

Additionally, we carefully reviewed the sentences in our dataset to ensure that all instructions and captions are grammatically correct and accurate. We employed a large language model [Dubey et al., 2024] to assist in the review process, followed by manual verification. Common errors identified include:

1. The need to add "The," "A," or other determiners before nouns, such as changing "Dog raises paw" to "The dog raises its paw." 2. Incorrect pronoun references, as seen in "Move the football to the top of your feet," where "your" should be replaced with "the man’s" or another appropriate description. 3. Other minor errors, such as "Lilacs change from two to one," which should be "changed" or "changes."

### References

Zahid Ali, Chesser Luke, and Carbone Timothy. Unsplash, 2023. Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural

images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18208–18218, 2022.

Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM transactions on graphics (TOG), 42(4):1–11, 2023.

Jinbin Bai, Zhen Dong, Aosong Feng, Xiao Zhang, Tian Ye, Kaicheng Zhou, and Mike Zheng Shou. Integrating view conditions for image synthesis. arXiv preprint arXiv:2310.16002, 2023.

Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient highresolution text-to-image synthesis. arXiv preprint arXiv:2410.08261, 2024.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. Advances in Neural Information Processing Systems, 36, 2024.

Yang Chen, Yu-Kun Lai, and Yong-Jin Liu. Cartoongan: Generative adversarial networks for photo cartoonization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9465–9474, 2018.

Wei Chow, Juncheng Li, Qifan Yu, Kaihang Pan, Hao Fei, Zhiqi Ge, Shuai Yang, Siliang Tang, Hanwang Zhang, and Qianru Sun. Unified generative and discriminative training for multi-modal large language models. arXiv preprint arXiv:2411.00304, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Aosong Feng, Weikang Qiu, Jinbin Bai, Kaicheng Zhou, Zhen Dong, Xiao Zhang, Rex Ying, and Leandros Tassiulas. An item is worth a prompt: Versatile image editing with disentangled control. arXiv preprint arXiv:2403.04880, 2024.

Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. arXiv preprint arXiv:2309.17102, 2023.

Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007, 2024a.

Zhiqi Ge, Juncheng Li, Qifan Yu, Wei Zhou, Siliang Tang, and Yueting Zhuang. Demon24: Acm mm24 demonstrative instruction following challenge. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11426–11428, 2024b.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Promptto-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks, 2019. URL https://arxiv.org/abs/1812.04948.

Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Conference on Computer Vision and Pattern Recognition 2023, 2023.

Juncheng Li, Kaihang Pan, Zhiqi Ge, Minghe Gao, Wei Ji, Wenqiao Zhang, Tat-Seng Chua, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Fine-tuning multimodal llms to follow zero-shot demonstrative instructions. In The Twelfth International Conference on Learning Representations, 2023a.

Juncheng Li, Siliang Tang, Linchao Zhu, Wenqiao Zhang, Yi Yang, Tat-Seng Chua, Fei Wu, and Yueting Zhuang. Variational cross-graph reasoning and adaptive structured semantics learning for compositional temporal grounding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10):12601–12617, 2023b.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.

Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253, 2024.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

R OpenAI. Gpt-4 technical report. arxiv 2303.08774. View in Article, 2(5), 2023a. R OpenAI. Gpt-4v (ision) system card. Citekey: gptvision, 2023b. Kaihang Pan, Siliang Tang, Juncheng Li, Zhaoyu Fan, Wei Chow, Shuicheng Yan, Tat-Seng Chua,

Yueting Zhuang, and Hanwang Zhang. Auto-encoding morph-tokens for multimodal llm. arXiv preprint arXiv:2405.01926, 2024.

Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen. amused: An open muse reproduction. arXiv preprint arXiv:2401.01808, 2024.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022a.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022b.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015, 2024.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022a. URL https://arxiv.org/abs/2210.08402.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022b.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.

Jing Shi, Ning Xu, Trung Bui, Franck Dernoncourt, Zheng Wen, and Chenliang Xu. A benchmark and baseline for language-driven image editing. In Proceedings of the Asian Conference on Computer Vision, 2020.

Jing Shi, Ning Xu, Yihang Xu, Trung Bui, Franck Dernoncourt, and Chenliang Xu. Learning by planning: Language-guided global image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13590–13599, 2021.

Qingyu Shi, Lu Qi, Jianzong Wu, Jinbin Bai, Jingbo Wang, Yunhai Tong, Xiangtai Li, and MingHusang Yang. Relationbooth: Towards relation-aware customized object generation. arXiv preprint arXiv:2410.23280, 2024.

Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Jingmin Chen, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, Di Zhang, and Bin Cui. Videotetris: Towards compositional text-tovideo generation. Advances in Neural Information Processing Systems, 2024.

Xinrui Wang and Jinze Yu. Learning to cartoonize using white-box cartoon representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8090–8099, 2020.

Zike Wu, Pan Zhou, Xuanyu Yi, Xiaoding Yuan, and Hanwang Zhang. Consistent3d: Towards consistent high-fidelity text-to-3d generation with deterministic sampling prior. arXiv preprint arXiv:2401.09050, 2024.

Ling Yang, Bohan Zeng, Jiaming Liu, Hong Li, Minghao Xu, Wentao Zhang, and Shuicheng Yan. Editworld: Simulating world dynamics for instruction-following image editing. arXiv preprint arXiv:2405.14785, 2024a.

Ling Yang, Zhilong Zhang, Zhaochen Yu, Jingwei Liu, Minkai Xu, Stefano Ermon, and Bin CUI. Cross-modal contextualized diffusion models for text-guided visual generation and editing. In International Conference on Learning Representations, 2024b.

Xuanyu Yi, Jiajun Deng, Qianru Sun, Xian-Sheng Hua, Joo-Hwee Lim, and Hanwang Zhang. Invariant training 2d-3d joint hard samples for few-shot point cloud recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14463–14474, 2023.

Xuanyu Yi, Zike Wu, Qiuhong Shen, Qingshan Xu, Pan Zhou, Joo-Hwee Lim, Shuicheng Yan, Xinchao Wang, and Hanwang Zhang. Mvgamba: Unify 3d content generation as state space sequence modeling. arXiv preprint arXiv:2406.06367, 2024a.

Xuanyu Yi, Zike Wu, Qingshan Xu, Pan Zhou, Joo-Hwee Lim, and Hanwang Zhang. Diffusion time-step curriculum for one image to 3d generation. arXiv preprint arXiv:2404.04562, 2024b.

Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea, 2024a. URL https://arxiv.org/abs/2411.15738.

Qifan Yu, Juncheng Li, Longhui Wei, Liang Pang, Wentao Ye, Bosheng Qin, Siliang Tang, Qi Tian, and Yueting Zhuang. Hallucidoctor: Mitigating hallucinatory toxicity in visual instruction data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12944–12953, 2024b.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36, 2024a.

Lvmin Zhang and Maneesh Agrawala. Transparent image layer diffusion using latent transparency. arXiv preprint arXiv:2402.17113, 2024.

Lvmin Zhang, Yi Ji, Xin Lin, and Chunping Liu. Style transfer for anime sketches with enhanced residual u-net and auxiliary classifier gan. In 2017 4th IAPR Asian conference on pattern recognition (ACPR), pages 506–511. IEEE, 2017.

Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9026–9036, 2024b.

Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. arXiv preprint arXiv:2407.05282, 2024.

Donghao Zhou, Jiancheng Huang, Jinbin Bai, Jiaze Wang, Hao Chen, Guangyong Chen, Xiaowei Hu, and Pheng-Ann Heng. Magictailor: Component-controllable personalization in text-to-image diffusion models. arXiv preprint arXiv:2410.13370, 2024.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.

