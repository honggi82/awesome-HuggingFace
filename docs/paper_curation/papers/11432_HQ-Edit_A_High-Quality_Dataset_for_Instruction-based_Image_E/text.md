## HQ-Edit: A High-Quality Dataset for Instruction-based Image Editing

# arXiv:2404.09990v1[cs.CV]15Apr2024

Mude Hui†, Siwei Yang† , Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, Cihang Xie

†equal contribution University of California, Santa Cruz

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

AlignmentCoherenceResolution

(a) Zoom in on the fox and add snowflakes falling around it. (b) Alter her hair color to black.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

InstructPix2Pix HIVE MagicBrush HQ-Edit

(e) Comparison between InstrictPix2Pix, HIVE, MagicBrush and HQ-Edit.

(c) Replace the Amazon rainforest background with the underwater scenery of the Great Barrier Reef and adjust the parrot's position and wings to depict it flying.

(d) Replace the silver teapot with a ceramic blue and white patterned teapot with a similar wooden handle and a matching ceramic lid.

Fig. 1: (a) - (d): example images and edit instructions from HQ-Edit. (e): we compare the dataset quality between our HQ-Edit and existing ones. Note that “Alignment” and “Coherence” are our newly developed metrics (introduced in Sec. 3.4) for measuring image/text qualities.

Abstract. This study introduces HQ-Edit, a high-quality instructionbased image editing dataset with around 200,000 edits. Unlike prior approaches relying on attribute guidance or human feedback on building datasets, we devise a scalable data collection pipeline leveraging advanced foundation models, namely GPT-4V and DALL-E 3. To ensure its high quality, diverse examples are first collected online, expanded, and then used to create high-quality diptychs featuring input and output images with detailed text prompts, followed by precise alignment ensured through post-processing. In addition, we propose two evaluation metrics, Alignment and Coherence, to quantitatively assess the quality of image edit pairs using GPT-4V. HQ-Edit’s high-resolution images, rich in detail and accompanied by comprehensive editing prompts, substantially enhance the capabilities of existing image editing models. For example, an HQ-Edit finetuned InstructPix2Pix can attain state-of-theart image editing performance, even surpassing those models fine-tuned with human-annotated data. The project page is https://thefllood. github.io/HQEdit_web.

Keywords: Image Editing · Generative Models

### 1 Introduction

The recent advancements in text-to-image generative models [7,10,18,20,22] have catalyzed a new era in diverse real-world applications ranging from advertising and photography to digital art and movie production. Among these generative models, applications of domain-specific image conditioned generations [9,21,27, 30], and multi-modal non-specific generation methods [16,23,29] have gathered significant attention.

Our work concentrates on applications of highly accurate, general instructionbased single image editing without relying on external attribute guidance, as proposed in previous studies [2, 8, 12, 24, 26]. We identify that this particular challenge has not been adequately addressed in the literature yet. To the best of our knowledge, one of the major hurdles in training an instruct-based image editing model lies in the limited availability of high-quality datasets pairing editing instructions with corresponding images. This challenge was best tackled by the seminal work InstructPix2Pix [3]. Specifically, it first leverages GPT-3 [4] to generate both an instruction and an edited image caption based on a given image description; then, it applies Stable Diffusion (SD1.5) [20] and Prompt-toPrompt [8] to create the paired input and output images. However, their underlying models, namely SD1.5 and GPT-3, are outdated compared to current state-of-the-art counterparts such as DALL-E 3 and GPT-4. Consequently, these models produce images with lower resolution and suboptimal edit-image alignment. Subsequent studies also attempted to improve it via incorporating human feedback [32] or segmentation masks [5,31], yet the generated data continue to exhibit one or more of the aforementioned issues, as showcased in Figure 1.

In this work, we aim to leverage the ability from the best text-image models, i.e., DALL-E 3 [14], GPT4 & GPT4V [15], to build a high-quality dataset for improving the image editing datasets. Ideally, in case of accessing the model weights, it should provide high-resolution images that offer rich detail, both in their visual content and the accompanying instructions; Also, it should provide more precise alignment between textual instructions and image pairs, ensuring edits are applied as directed while maintaining fidelity in areas not subject to modification.

However, only with the access to their APIs, in this study, we discover a way of pair image generation with DALL-E 3 based on prompt-engineer, which enable a similar Prompt-to-Prompt process, yielding high-quality editing image pairs, which we name as HQ-Edit. HQ-Edit provides a significant leap forward, featuring high image resolutions of approximately 900×900 pixels—nearly double that of existing datasets, and comprises around 200,000 detailed edit instructions. Moreover, unlike prior approaches relying on attribute guidance or human feedback, HQ-Edit is synthetically generated through a scalable pipeline that harnesses the image text understanding capabilities of powerful foundation models of GPT-4V and DALL-E 3.

Our data curation process comprises three key steps: Expansion - Generation - Post-processing. Firstly, in the Expansion phase, we extract seed triplets with high diversity—consisting of input/output image descriptions along

with edit instructions—from online sources. Subsequently, we leverage GPT-4 to expand these initial triplets into around 100,000 instances, ensuring the comprehensive diversity of edit instructions. In the subsequent Generation phase, the seed triplets are processed by GPT-4 to merge and refine into detailed diptych prompts for DALL-E 3, creating diptychs with input and output image pairs displayed side-by-side. Note this diptych-based prompting design is motivated by the finding that, compared to generating input images and output images separately, generating diptychs generally exhibits superior quality, with better alignment and consistency in edit-irrelevant areas. Lastly, the generated diptychs and refined prompts undergo post-processing to ensure precise alignment between the paired images and their corresponding instructions. Specifically, 1) each diptych is decomposed into paired images, which undergo warping and filtering to ensure correspondence; 2) the instructions are refined using rewritten instructions from GPT-4V; and 3) the inverse-edit instructions are also generated, allowing for the transformation of output images back into their input counterparts.

On top of HQ-Edit, we introduce two metrics, Alignment and Coherence, to comprehensively and quantitatively evaluate the quality of image edit pairs. The first metric, Alignment, checks for semantic consistency with the edit prompt, ensuring accurate modification of mentioned objects while preserving image fidelity. The second metric, Coherence, evaluates the edited image’s aesthetic quality, including lighting and shadow consistency, style coherence, and edge smoothness. Extensive empirical results show that our synthetically created HQ-Edit can even surpass human-annotated data in enhancing instruction-based image editing models. For example, the HQ-Edit finetuned InstructPix2Pix model substantially outperforms its vanilla version, achieving a 12.3 increase at Alignment, and a 5.64 enhancement at Coherence.

### 2 Related works

Text Guided Image Editing Model Text guided image editing models have been extensively discussed recently. Prompt2Prompt [8] modifies words in the original prompts to perform both local editing and global editing by crossattention control. Imagic [11] optimizes a text embedding that aligns with the input image, then interpolates it with the target description, thus generating correspondingly different images for editing. DiffEdit [6] locate edit position based on text (generate mask), and limit diffusion model to generate the mask area. An important type of Text Guided is the instruction, which describes where, what and how an image should be edited. Instruction-based image editing model will follow the instruction without requiring elaborate descriptions or region masking, and enables users to modify images more easily and flexibly. InstructPix2Pix [3] is the first instruction-based image editing model, by fine-tuning the Stable Diffusion [20] on a dataset of image editing examples, which generated by GPT-3 [4] and Prompt2Prompt. Subsequent work, such as HIVE [32] and Magicbrush [31], have focused on improving the quality or quantity of the dataset.

Instruction-based Image Editing Datasets Since it can be challenging to collect high-quality open data for image editing, early approaches construct datasets by manually labeling image pairs [31]. While this ensured a degree of quality, it inherently restricted the scale and diversity of the dataset. For example, Magicbrush [31] contains about only 10,000 edits, and predominantly focuses on object-level transformations, largely overlooking global edits like style or weather changes. On the other hand, there have been endeavors to synthesize large-scale datasets. For example, InstructPix2Pix [3] leverages GPT-3 and Prompt2Prompt [8] to generate editing pairs, and HIVE [32] introduces reinforcement learning from human feedback to better align the data with human expectations. However, these synthetic data often have the drawback of low quality and inaccurate editing, resulting in such trained image editing models outputting low-quality images and deviating from the actual edit instructions. FaithfulEdits [5] attempts to mitigate these issues by using inpainting techniques, followed by a filtering process involving VQA models. Yet, this method tends to underperform, particularly in global edits requiring extensive image modification, like style transfer.

Different from these existing approaches, in our study, we leverage the latest foundation models like GPT-4 and DALL-E 3 to generate high-quality image editing pairs at scale. We also introduce additional enhancements, e.g., using GPT-4V to rewrite the edit instruction to align with the images more closely.

### 3 HQ-Edit Dataset

The process of collecting HQ-Edit, illustrated in Figure 2, comprises three phases. Initially, triples of input/output image descriptions and edit instructions are expanded into 100,000 instances during the Expansion phase (Section 3.1). Subsequently, these instances are refined into detailed prompts for DALL-E 3 to generate diptychs in the Generation phase (Section 3.2). Finally, alignment and refinement occur in the Post-processing phase (Section 3.3).

#### 3.1 Expansion

To initialize, we first collect a small yet representative dataset comprising 203 samples from online sources. To ensure alignment between the text descriptions and image pairs, we manually revise the descriptions based on the disparities in content. Additionally, we include 90 samples from the Emu Edit [23] test set. We refer to these 293 samples as seed triplets, with each triplet comprising input/output image descriptions along with corresponding edit instructions.

To increase its size, we follow the pipeline presented in Self-instruct [28], which applies large language models on a small set of seed samples to generate a large volume of expansions that are both high in quality and consistent with the seed structure. Specifically, we utilize GPT-4 to expand this initial set of 293 seed triplets into around 100,000 instances, ensuring a thorough representation of diverse image editing scenarios. The detailed prompt employed for generating

###### Step #1: Expansion Step #2: Generation Step #3: Post-Processing

Input Output Edit

Input Output Edit

| | | |
|---|---|---|
| | | |

Input Output Edit

ExpandedSeed

[Figure 10]

[Figure 11]

GPT-4

Input Output Edit

Diptych Prompt

GPT-4

###### GPT-4V

DALL·E 3

Rewritten Edit

Input Output Edit

[Figure 12]

Rewritten Inverse Edit

Diptych

Input Output Edit

Input Output

Input Output Edit

Image Processer

[Figure 13]

Rewritten Edit

Rewritten Inverse Edit

[Figure 14]

[Figure 15]

Input Image Description Output Image Description Edit Instruction

Input Output Edit

[Figure 16]

[Figure 17]

Input Image

Output Image

- Fig. 2: Our method consists of three steps: (1)Expansion: Massively generating image descriptions and edit instructions based on seed samples using GPT-4. (2)Generation: Generating diptychs using GPT-4V and DALL-E according to image descriptions and instructions. (3)Post-Processing: Post-process diptychs and edit instructions with GPT-4V and other various methods to produce image pairs and further enhance the quality of the dataset in different aspects.

these triplets with GPT-4 is provided in Appendix A. This strategy not only broadens the scope of edit instructions but also leverages GPT-4’s knowledge to enrich the diversity and detail of image descriptions and edit instructions.

#### 3.2 Generation

Upon acquiring the essential instructions and image descriptions from Expansion (Section 3.1), the next step is to generate paired images that align with the instruction data. We hereby employ DALL-E 3 [14], a state-of-the-art image generation model capable of producing high-resolution images based on textual descriptions. However, DALL-E 3 is not originally designed for instruction-based image editing, and therefore cannot directly produce paired images. Thus, we devised a workaround by creating diptychs consisting of input and output images side by side, followed by post-processing (Section 3.3) to reconstruct paired images. Interestingly, we note that generating input and output images together in diptych form, rather than separately, significantly enhances the relevance and correspondence between image pairs. As outlined in Figure 2, each triplet is fed to GPT-4 to form a diptych prompt for DALL-E 3 to generate a diptych. Moreover, to refine the diptych prompts and improve consistency between image pairs, GPT-4 is also utilized to elaborate further on the prompts. For instance, a basic description like “an elder Asian woman” can be enriched into “an elderly East Asian woman with wrinkle-lined skin and white hair pulled back neatly,

##### Table 1: An example of the diptych prompt.

Input/Output/Edit Diptych Prompt For DALL-E 3

Generate a diptych with two side-by-side images. On the left, depict a

Input: a graffiti-covered urban alley

vibrant, narrow urban alley teeming with colorful graffiti on its walls.

Details should include assorted tags and street art in various styles, with a depth indicating the alley stretches far back. Miscellaneous urban elements like a dumpster, a stray cat, and fire escape ladders should be present, and a subtle sunlight to cast soft shadows, indicating a daytime

Edit: present the photo with a high-

contrast black and white effect

setting. On the right, replicate this scene exactly but convert the image

into high-contrast black and white with stark lighting to enhance textures and shadows, and accentuate the details of the graffiti, giving an edgy, gritty aesthetic. Each element from the left image must be recognizable in monochrome, especially the contrasts between the shaded areas and

Output: a high-contrast black and white image of a graffiti-covered alley

the illuminated ones created by an overhead midday light.

wearing a traditional red and gold silk hanbok”. This enrichment adds complexity to the prompts and subsequently to the generated diptychs. An example of the enhanced diptych prompt is shown in Table 1. Overall, this process yields 98,675 data samples comprising input-output text pairs, edit instructions, and diptych images.

#### 3.3 Post-processing

After generating the diptych and its corresponding prompt, we implement a tailored post-processing stage aimed at decomposing the diptych back into paired images and further refining the quality of both image pairs and text instructions. This process involves two key steps: image post-processing and instruction refinement. First, for image post-processing, we decompose the diptych into paired images and employ warping and filtering techniques to ensure their alignment. Secondly, for instruction refinement, we enhance the instructions by incorporating rewritten instructions from GPT-4V and generating inverse-edit instructions, doubling the total edits to 197,350.

Image Post-processing The goal of image post-processing is to decompose the diptych into paired images as well as to improve their correspondence. We later use correspondence as a quality control to (optionally) filter our training set. It consists of three steps: Decomposing, Warping, and Filtering, which are detailed below.

- 1. Decomposing horizontally separates diptychs generated by DALL-E 3 into image pairs using a retrained object detection model. Specifically, we train a YOLOv8 [19] object detector on 3,000 diptych images, where human annotators manually mark bounding boxes for both left and right segments.
- 2. Warping aligns the decomposed paired images based on semantic correspondence between input and output images. We employ DIFT [25], an advanced diffusion-based model, to establish pixel-wise semantic correlations between paired images. By leveraging semantic correspondence, we determine the homography, which maps pixels from the input image to corresponding pixels

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

|[Figure 22]|
|---|

(a) Diptych (b) After Decomposing (c) After Warping

- Fig. 3: The effect of decomposing and warping in image post-processing. Filtering is not demonstrated in this figure. Without warping, there is a part of the desk edge in the output image on the right. This issue is addressed after warping.

(a) Input/Output Image (b) Before v. After Instruction Refinement

[Figure 23]

[Figure 24]

"Remove the pouf and patterned rug, add a glass coffee table with a white base, a large beige rug, and a potted monstera plant."

Edit (After):

Inverse Edit (After): "Remove the glass coffee table, beige rug, and potted monstera plant, add back the textured pouf and patterned rug."

Edit (Before): "Replace the chair with a rustic wooden bench."

- Fig. 4: The effect of rewriting and inversion in edit post-processing. After postprocessing, the edit instruction is of greater complexity and aligns better with the input/output image pair.

in the output image, facilitating the precise alignment between them. An example of warping in improving alignment between input and output images is illustrated in Figure 3.

- 3. Filtering assesses image distortion post-warping and retains those with minimal distortion for training purposes. When the dimensions of the image before warping are denoted as {w1, w2, h1, h2}, and those after warping as {w3, w4, h3, h4}, any image undergoing more than a 50% deformation on any single dimension before and after warping, such as w1 < 0.5 ∗ w3, is filtered out. Note that this step is applied exclusively to the InstructPix2Pix fine-tuning process for selecting high-quality training samples from our HQ-Edit dataset.

Instruction Refinement While image post-processing improves alignment between input and output images, further refinement is vital to ensure that editing instructions are well-aligned with image pairs. First, by leveraging GPT-4V, we rewrite edit instructions based on the differences between input and output image details, thereby enhancing the detail of the text descriptions. Rewriting not only helps fix discrepancies in existing descriptions but also includes visual differences between background objects, which are often omitted in the original text descriptions. Additionally, we use GPT-4V to directly generate inverse-edit instructions for transforming output images back to input images. This simple strategy can effectively double the instruction count but at a marginal cost.

Distribution of Edit Instruction Length in InstructPix2Pix

Distribution of Edit Instruction Length in HQEdit

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

3.5

25

3.0

20

2.5

Percentage(%)

Percentage(%)

2.0

15

1.5

10

1.0

5

0.5

0

0.0

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

Edit Length

Edit Instruction Length

- Fig. 5: The histograms illustrate the distribution of edit instruction lengths for HQEdit and InstructPix2Pix. HQEdit exhibits a more uniform and dispersed distribution, indicating a broader diversity in the length of its instructions. This suggests HQEdit’s instructions are presented with greater detail and flexibility, offering a richer information to carry out editing tasks more effectively.

Overall, as demonstrated in Figure 5, the application of rewriting and inversion techniques substantially increases both the length and diversity of edit instructions. This enrichment leads to a dataset enhanced with a wider range of composite operations, resulting in a broader distribution of instruction lengths. Our edit instructions not only have a larger average length but also display a more expansive distribution, underscoring the effectiveness of these augmentation strategies.

As depicted in Figure 4, while the original edit instruction consists of merely 7 words, GPT-4V improves its comprehensiveness by increasing both the length and the variety of edit operations.

#### 3.4 Data Quality Assessment

Diversity of Edit Instruction Unlike previous studies which either focus on global or object editing [3,31,32], our editing operations span a broad spectrum, encompassing both global operations—such as altering the weather, modifying the background, and transforming the style—and local operations, which include a variety of object-based editing. Figure 6 provides a comprehensive overview of the keywords in the edit instructions of HQ-Edit. This diversity of edit instructions indicates that our HQ-Edit incorporates a vast range of editing tasks, thereby demonstrating its extensive coverage of potential editing operations.

Alignment and Coherence To quantitatively evaluate the quality of editing, we introduce two formal metrics: Alignment and Coherence. The Alignment metric assesses the semantic consistency of edits with the given prompt, ensuring accurate modifications while preserving fidelity in the rest of the image. On the other hand, the Coherence metric evaluates the overall aesthetic quality of the edited image, considering factors such as lighting and shadow consistency, style coherence, and edge smoothness. These metrics, performed using GPT-4V, produce scores from 0 to 100, with higher scores indicating better alignment or coherence.

its

traditional

background

white

revert

scene

is

green

red

sky

more

black

setting

blue

butterﬂy

maintaining

while

transform

chair

lighting

modern

color

restore

colorful

add

by

monarch

vibrant

alter

wooden

hammer

tea

attire

has

adjust

remove

clear

are

large

image

her

back

vintage

that

including

brown

now

trees

original

metal

cat

same glass

futuristic

dark

ornate

change

table

cherry

elements

light

(56.47%)objectTransform

snowy

snow

landscape

small

position

restoring

their

intricate

tools

orange

car

been

replace

digital

forest

ﬂowers

yellow

glowing

ﬂoral

have

view

wall

leather

japanese replaced

golden

Change background (3.08%)

change

Change weather

Change globalstyle

back

sky

(4.19%)

(10.11%)

remove

background

colorChange (4.59%)

change

time

day

Removeobject(10.85%)

add

snow

(10.71%)objectAdd

scene

lighting

season

vibrant

clear

winter

eﬀect

revert

change

alter

colors

remove

revert

more

original

trees

color

night

image

lighting

blue

adjust

ground

moon

reﬂect

sunset

ground,

starry

trees, sky,

leaves sun

weather

setting warm

by

green

original

transition

snowy

maintaining

restore

daytime

remove

bright

full

falling

while

vibrant

remove

sky.

wings

autumn

green

orange

night,

removing replace

butterﬂy's

color

white

bright

original

orange

change

hair

clouds

vibrant

color

sunny

environment

black

red

stars

adding

back

blue

glow

covering

more

landscape

foliage

restore spring

snow,

sunlight

snowﬂakes

nighttime

atmosphere.

summer

scene.

daylight

atmosphere

ﬂowers

daytime,

snow-covered

original

background.

enhance

change

butterﬂy

add

replace

add

additional

setting

scene,

scene.

wooden

back

people

ﬂying

above

leaving

ﬂowers

vibrant

large

next

small

its

image

additional

only

right

background

lighting

remove

table

background

include

snow

revert

change

green

scene

back

all

sky

colorful

create

original

side

wooden

image

replace

blue

white

around

scene

red

lighting

place

more

sky

returning

more

white

blue

red

colorful scene.

time

return

clear

image.

green

restore

snow

setting

table

small

background.

ﬂowers

butterﬂy

around

cherry

without

reverting

people

- Fig. 6: Distribution of edit types and keywords in instructions. The inner ring depicts the types of edit instructions and the outer circle shows the frequencies of instruction keywords. This demonstrates the rich diversity contained within our instructions.

[Figure 25]

[Figure 26]

Coherence : 20 Coherence : 40 Coherence : 60 Coherence : 80 Coherence : 100

[Figure 27]

[Figure 28]

[Figure 29]

- Fig. 7: Examples of different Coherence. As the Coherence score increases, the image quality improves significantly.

We illustrate the Alignment metric in Table 2, presenting the instructions for scoring Alignment with GPT-4V and corresponding evaluation results with varying scores. Specifically, GPT-4V is provided with evaluation instructions to assess whether the changes between the two images align with the instruction of the EDIT TEXT, utilizing different criteria for various types of edits, such as global editing (e.g., stylization) and local editing (e.g., object removal). Similarly, for coherence assessment, separate instructions are provided to GPT-4V to determine coherence based on lighting, shadows, scene logic, element edges, and overall visual appearance (detailed prompts are presented in Appendix A.3). Example images showing different Coherence scores are provided in Figure 7, suggesting a potential (positive) correlation with human perception. We also provide two randomly sampled data points from HQ-Edit in Figure 8 for visual assessment (more such examples are presented in Appendix 14).

To further validate the effectiveness of our proposed metrics, as detailed in Section 4.2, we conducted a human evaluation on 1,651 image pairs generated by DALL-E 3. Notably, our metric exhibited a much higher correlation to human preference compared to the popular CLIP score.

##### Table 2: Evaluation metrics based on GPT-4V. The evaluation instructions and results are in a simplified version. Editing follows the edit instructions more accurately as Alignment increases.

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Input Image "Alignment": 40 "Alignment": 80 "Alignment": 100

Evaluation Instruction: EDIT TEXT: Add a comet in the sky From 0 to 100, rate EDIT TEXT's Correctness and Comprehensiveness. Correctness: whether the text mentions any change not made between two images. Comprehensiveness: whether the text misses any change made between two images.

- 1. If EDIT TEXT concerns stylization or lighting, no content should change.
- 2. If the EDIT TEXT is about a local change, no irrelevant area should be changed.
- 3. The first image should not have the attribute described inside the EDIT TEXT.
- 4. Check if the second image retains the left image's key attribute .

Evaluation Results from GPT-4V Explanation: The EDIT TEXT requested the addition of a comet in the sky, which has been done correctly in

the second image. The comet is visible in the sky with its tail,and no other changes appear to have

been made to the image. The rest of the scene, including the stylization, lighting, and content, remains unchanged, which is consistent with the EDIT TEXT instructions. Alignment: 100

Table 3: Comparison between different editing datasets. Dataset Alignment ↑ Coherence ↑

InstructPix2Pix [3] 68.29 83.35 HIVE [32] 9.85 84.65 MagicBrush [31] 80.61 65.42 HQ-Edit 92.80 91.87

Comparisons To demonstrate the superior data quality of our dataset compared to existing public editing datasets, we conducted evaluations on 500 randomly sampled data points from InstructPix2Pix, HIVE, MagicBrush, and HQEdit, assessing their Alignment and Coherence metrics (Table 3). HQ-Edit significantly outperforms all other datasets with Alignment and Coherence scores of 92.80 and 91.87, respectively, compared to InstructPix2Pix (68.29 and 83.35), HIVE (9.85 and 84.65), and MagicBrush (80.61 and 65.42), demonstrating its superior data quality.

"input": "A German Shepherd with a black and tan coat,

"input": "portrait of a majestic golden eagle in flight against a

pointed ears, and a dog tag is sitting on a grassy lawn with trees

warm sunset sky."

and sunlight in the background. "

"edit": "Replace the German Shepherd with a fluffy white cat with blue eyes and a long tail, keeping the same background. "

"edit": "adjust the overall image to a nighttime setting with a darker sky and cooler tones."

"inverse-edit": "Replace the fluffy white cat with the original German Shepherd with a black and tan coat, pointed ears, and a dog tag. "

"inverse-edit": "adjust the overall image to a daytime setting with a brighter sky and warmer tones."

"output": "A fluffy white cat with blue eyes and a long tail is sitting on the same grassy lawn with trees and sunlight in the background."

"output": "portrait of a majestic golden eagle in flight against a cooler, night-time sky with visible clouds and a darker ambiance."

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

“Alignment”: 100 “Coherence”:100 “Alignment”: 60 “Coherence”:100

- Fig. 8: Example data sampled from HQ-Edit. Our data contains two main parts, Instruction (input, edit, inverse-edit, output) and Image (input image, output image). The two samples highlight that, 1) the image is densely packed with details, 2) the input and ouput offers a comprehensive description of the input and output image, and 3) the edit and inverse-edit instructions precisely delineate the transformations occurring between the two images.

### 4 Experiments

#### 4.1 Experiment Setup

Baselines We conducted a comparative analysis with existing open-source textbased image editing methods, i.e., DiffEdit [6], Imagic [11], PromptInverse [13], HIVE [32], MagicBrush [31]. To ensure reproducibility and fairness, we utilized default hyperparameters from the official implementations. Our testing set comprised the 293 samples mentioned in Section 3.1, with all input images generated by DALL-E 3 based on the input image descriptions.

Implementation Details We choose InstructPix2Pix [3] as our default model, and use HQ-Edit to fine-tune it. During training, we set the image resolution to 512, total training steps to 15000, learning rate to 5e-5, and conditioning dropout prob to 0.05. During the editing, we set the image guidance scale to 1.5, the instruct guidance scale to 7.0, and the number of inference steps to 20.

Table 4: Comparison of Alignment, Clip Score, and Human Evaluation Score. Method AVG. Score ↑ Correlation ↑

Alignment 41.78 0.3592 Clip Directional Similarity 25.12 -0.1446 Human Evaluation Score 61.21 1.0

#### 4.2 Human Evaluation

To verify the consistency of our developed Alignment metric with human preference, we conduct a human evaluation of 1,651 image pairs generated by DALL-E

- 3. We utilize Gradio [1] to create the evaluation platform. For each assessment, edit instructions, the input/output image pairs, and their corresponding descriptions are provided for evaluation. We categorize whether the change between the input image and the output image matches the corresponding edit instruction into the following 5 levels:

- 1. Totally not related.
- 2. Not following edit, but there is some relation between the two images.
- 3. OK image pair, but not following the edit instruction.
- 4. Good image pair, but need to modify the edit instruction for better alignment.
- 5. Perfectly follows the edit instruction.

We report the results in Table 4. As different metrics have different ranges (i.e., Alignment from 0 to 100, Clip Directional Similarity from 0 to 1, and Human Evaluation Score from 1 to 5), a normalization procedure to a common scale of 0 to 100 is initially undertaken, followed by the computation of the average score. Furthermore, we use Pearson Correlations to analyze the correlation between Alignment and Clip Directional Similarity to Human Evaluation Score.

We can observe that the proposed Alignment metric significantly surpasses CLIP [17] Directional Similarity in accurately evaluating the fidelity of edit instructions to reflect the alterations between the input and output images. This notable discrepancy underscores a significant limitation of CLIP Directional Similarity, namely its inability to comprehensively grasp the nuances of the editing process and accurately retain fidelity to the intricate details of the images.

- 4.3 Quantitative Evaluation

The comparison between our model and existing text-based image editing models is shown in Table 5. Compared to other methods, our model performs best in all metrics. Specifically, our model outperforms the vanilla InstructPix2Pix, achieving a notable increase of 12.30 in Alignment (from 34.71 to 47.01) and

- 5.56 in Coherence (from 80.52 to 86.16). Furthermore, it is noteworthy that our model surpasses HIVE and MagicBrush, two methods fine-tuned on InstructPix2Pix, further validating its capability to enhance InstructPix2Pix’s image editing outcomes beyond their respective datasets.

Table 5: Comparison with existing text-based image editing models. Method Alignment ↑ Coherence ↑

|Imagic [11] DiffEdit [6] PromptInverse [13]|1.50 63.58<br><br>21.53 81.81<br><br>22.82 80.85<br><br><br>|
|---|---|
|InstructPix2Pix [3] /Base /XL|34.71 80.52<br><br>35.03 84.45<br><br><br>|
|HIVE [32] w/conditional w/weighted<br><br>|40.34 82.93 40.68 84.94|
|MagicBrush [31] HQ-Edit<br><br>|43.77 84.19 47.01 86.16|

Table 6: Ablation experiments on Post-processing.

RAW Rewrite Inverse Warp Filter Alignment ↑ Coherence ↑

34.71 80.52 ✓ 16.83 85.74 ✓ ✓ 28.62 86.68 ✓ ✓ ✓ 34.42 87.53 ✓ ✓ ✓ ✓ 43.41 87.56 ✓ ✓ ✓ ✓ ✓ 47.01 86.16

This distinction underscores the superior efficacy of HQ-Edit in augmenting InstructPix2Pix’s image editing capabilities in comparison to existing datasets. Furthermore, it emphasizes the comprehensive nature of our dataset, which comprises high-quality images and edit instructions, thereby establishing a robust foundation for more intuitive and effective image editing procedures.

#### 4.4 Qualitative Evaluation

As shown in Figure 9, a comparative analysis of various models’ performance is visually presented, with each column dedicated to showcasing the results from a distinct model. For example, in the second line, only the model trained with HQ-Edit understands the ground region in the edit instruction and correctly adds the flowers in it as required. It can also be seen in Figure 10 that the model trained with HQ-Edit can carry out various types of edit operations. This observation not only underscores HQ-Edit’s advanced understanding of spatial and contextual directives but also its capability to precisely manipulate image content in accordance with specific editing specifications.

#### 4.5 Ablation Study

We hereby ablate the effectiveness of different post-processing strategies, introduced in Sec. 3.3. Specifically, we use “RAW” to denote the simply decomposed DALL-E 3 images (i.e., image pairs that directly splitted from diptych), and use

Input InstructPix2Pix HIVE Magic Brush HQ-Edit

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Add snow to the scene of the image

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Make flowers on ground region in the image

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Make the subject surface to be iron material

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Make her hair color to purple

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Morph her into a robot

##### Fig. 9: Qualitative comparison of InstructPix2Pix, MagicBrush, HIVE and HQ-Edit. HQ-Edit demonstrates a more comprehensive diversity of editing instructions and possesses the capability to manipulate images with greater precision and detail.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Input Image Make the image anime style

Change the background of the subject to beach

Add some snow to the scene of the image

Add wings

##### Fig. 10: Qualitative results with the same input image but with different edit instructions. HQ-Edit enhances the editing capabilities of InstructPix2Pix by enabling it to modify the same image of a black cat in various ways.

“Rewrite”, “Filter”, “Warp”, and “Inverse” to mark whether the corresponding operations are applied for further processing. For example, applying all these four operations to process these will lead to our HQ-Edit dataset.

Table 6 reports the corresponding results. Interestingly, by comparing the first row and the second row, we note that directly fine-tuning the model on the raw DALL-E 3 images enhances its performance on Alignment but hurts Coherence. This potentially suggests that while the image quality of these DALL-E 3 generated images exceeds that of the InstructPix2Pix dataset, the alignment between the image and edit instruction is less satisfactory. This issue can be mitigated with our post-processing techniques. For example, our rewrite method, when compared to the second row’s results, delivers improvements of 11.79 in Alignment and 0.94 in Coherence. This boost, primarily enhancing the images’ alignment with the edit operation, indicates DALL-E 3’s challenges in producing accurate images from dypitch prompts—a gap our method effectively bridges. Additionally, employing the inverse technique, which acts as a form of data augmentation, further elevates Alignment by 5.2 and Coherence by 0.94. The warp technique serves to augment both pre- and post-edit image alignment, resulting in a notable 5.2 increase in alignment accuracy. Nonetheless, the application of warp may occasionally lead to undesirable levels of image distortion. Through the implementation of a filtering mechanism targeting such occurrences, we not only achieve a further enhancement in image alignment, registering a 3.6 increase, but also mitigate the associated data volume. Consequently, this filtering process incurs a marginal reduction in Coherence, specifically by 1.4 points, yet remains superior to other baselines.

These results indicate that HQ-Edit holds significant potential to enhance instruction-based edit models, especially when combined with effective postprocessing.

### 5 Conclusion

In this study, we present a automatic way to synthesize large-scale image editing dataset. Specifically, we leverage two foundation models, GPT-4V and DALL-E 3, to automatically generate, rewrite, and expand a set of seed image editing data with high-quality. Additionally, we develop two GPT-4V-based evaluation metrics to assess the alignment of the edited images to the editing instruction, and the coherence of the image content. Our extensive experiments demonstrate that models trained on HQ-Edit set a new state-of-the-art performance in the task of instruction image editing.

### References

- 1. Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., Zou, J.: Gradio: Hasslefree sharing and testing of ml models in the wild. arXiv preprint arXiv:1906.02569

(2019) 12

- 2. Avrahami, O., Lischinski, D., Fried, O.: Blended diffusion for text-driven editing of natural images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18208–18218 (2022) 2
- 3. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023) 2, 3, 4, 8, 10, 11, 13
- 4. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901 (2020) 2, 3
- 5. Chakrabarty, T., Singh, K., Saakyan, A., Muresan, S.: Learning to follow objectcentric image editing instructions faithfully. In: The 2023 Conference on Empirical Methods in Natural Language Processing (2023) 2, 4
- 6. Couairon, G., Verbeek, J., Schwenk, H., Cord, M.: Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427 (2022) 3, 11, 13
- 7. Gu, S., Chen, D., Bao, J., Wen, F., Zhang, B., Chen, D., Yuan, L., Guo, B.: Vector quantized diffusion model for text-to-image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10696– 10706 (2022) 2
- 8. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022) 2, 3, 4
- 9. Hu, L., Gao, X., Zhang, P., Sun, K., Zhang, B., Bo, L.: Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117 (2023) 2
- 10. Huang, Y., Huang, J., Liu, Y., Yan, M., Lv, J., Liu, J., Xiong, W., Zhang, H., Chen, S., Cao, L.: Diffusion model-based image editing: A survey. arXiv preprint arXiv:2402.17525 (2024) 2
- 11. Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., Irani, M.: Imagic: Text-based real image editing with diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6007–6017 (2023) 3, 11, 13
- 12. Ling, H., Kreis, K., Li, D., Kim, S.W., Torralba, A., Fidler, S.: Editgan: Highprecision semantic image editing. Advances in Neural Information Processing Systems 34, 16331–16345 (2021) 2
- 13. Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794 (2022) 11, 13
- 14. OpenAI: GPT-4v System Card. https://openai.com/research/dall-e-3-system-card

(2023) 2, 5

- 15. OpenAI: GPT-4v System Card. https://openai.com/research/gpt-4v-system-card

(2023) 2

- 16. Pan, X., Dong, L., Huang, S., Peng, Z., Chen, W., Wei, F.: Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992 (2023) 2
- 17. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021) 12

- 18. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1(2), 3 (2022) 2
- 19. Reis, D., Kupec, J., Hong, J., Daoudi, A.: Real-time flying object detection with yolov8. arXiv preprint arXiv:2305.09972 (2023) 6
- 20. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022) 2, 3
- 21. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22500–22510 (2023) 2
- 22. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022) 2
- 23. Sheynin, S., Polyak, A., Singer, U., Kirstain, Y., Zohar, A., Ashual, O., Parikh, D., Taigman, Y.: Emu edit: Precise image editing via recognition and generation tasks. arXiv preprint arXiv:2311.10089 (2023) 2, 4
- 24. Shi, Y., Yang, X., Wan, Y., Shen, X.: Semanticstylegan: Learning compositional generative priors for controllable image synthesis and editing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11254– 11264 (2022) 2
- 25. Tang, L., Jia, M., Wang, Q., Phoo, C.P., Hariharan, B.: Emergent correspondence from image diffusion. arXiv preprint arXiv:2306.03881 (2023) 6
- 26. Wallace, B., Gokul, A., Naik, N.: Edict: Exact diffusion inversion via coupled transformations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22532–22541 (2023) 2
- 27. Wang, P., Shi, Y.: Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201 (2023) 2
- 28. Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N.A., Khashabi, D., Hajishirzi, H.: Self-instruct: Aligning language model with self generated instructions (2022) 4
- 29. Wu, S., Fei, H., Qu, L., Ji, W., Chua, T.S.: Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519 (2023) 2
- 30. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models (2023) 2
- 31. Zhang, K., Mo, L., Chen, W., Sun, H., Su, Y.: Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems 36 (2024) 2, 3, 4, 8, 10, 11, 13
- 32. Zhang, S., Yang, X., Feng, Y., Qin, C., Chen, C.C., Yu, N., Chen, Z., Wang, H., Savarese, S., Ermon, S., et al.: Hive: Harnessing human feedback for instructional visual editing. arXiv preprint arXiv:2303.09618 (2023) 2, 3, 4, 8, 10, 11, 13

### A Prompts

We list all the prompts we used for data collection, including the EXPAND PROMPT used for the Expansion step; DIPTYCH PROMPT and REWRITE PROMPT used for the Generation step; and two metric prompt ALIGNMENT PROMPT and COHERENCE PROMPT for the evaluation.

- A.1 Step #1: Expansion

EXPAND PROMPT (GPT-4)

You are required to generate num examples considering the given examples. The examples should vary widely, including different human characteristics (such as race, age, and body type), various animals, insects, furniture, tools, or any object types, etc., and diverse backgrounds (like different countries, natural environments, landscapes, or skies). The editing attributes should also be diverse. Make sure the examples are clear, concise, comprehensive, and easier for DALL-E 3 to generate this diptych image following the prompt. Describe the first image in "INPUT_DESCRIPTION" like "input", the second image in "OUTPUT_DESCRIPTION" like "output", both "INPUT_DESCRIPTION" and "OUTPUT_DESCRIPTION" should be independent complete sentences, and the operation that edits the first image to the second image in "EDIT_OPERATION", and the operation that edits the second image to the first image in "INVERSE_EDIT_OPERATION", the output should be a list of JSON format as such: { "input": "INPUT_DESCRIPTION", "edit": "EDIT_OPERATION", "edit_inv": "INVERSE_EDIT_OPERATION", "output": "OUTPUT_DESCRIPTION". }. Do not output anything else, all examples should have complete keys "input", "edit", "edit_inv", and "output".

- A.2 Step #2: Generation

#### REWRITE PROMPT (GPT-4)

Please rewrite the following prompt to make it more clear and concise, and easier for DALL-E 3 to generate this diptych image follow the prompt. The original prompt is: {prompt}. The output prompt should start with "REVISED":

#### DIPTYCH PROMPT (DALL-E 3)

Create a diptych image that consists two images. The left image is {prompt}; The right image keep everything the same but {edit_action}.

#### A.3 Evaluation Metric ALIGNMENT PROMPT (GPT-4V)

From 0 to 100, how much do you rate for EDIT TEXT in terms of the correct and comprehensive description of the change from the first given image to the second given image? Correctness refers to whether the text mentions any change that are not made between two images. Comprehensiveness refers to whether the text misses any change that are made between two images. The second image should have minimum change to reflect the changes made with EDIT TEXT. Be strict about the changes made between two images:

- 1. If the EDIT TEXT is about stylization or lighting change, then no content should be changed and all the details should be preserved.
- 2. If the EDIT TEXT is about a local change, then no irrelevant area nor image style should be changed.
- 3. The first image should not have the attribute described inside the EDIT TEXT, rate low, (<80) if this happens.
- 4. Be aware to check whether the second image does maintain the important attribute in the left image that is not reflected in the EDIT TEXT. Rate low (<50) if two images are not related. Provide a few lines for explanation and give the final response in a json format as such: { "Explanation": "", "Score": "", }

#### COHERENCE PROMPT (GPT-4V)

Rate the Coherence of the provided image on a scale from 0 to 100, with

- 0 indicating extreme disharmony characterized by numerous conflicting or clashing elements, and 100 indicating perfect harmony with all components blending effortlessly. Your evaluation should rigorously consider the following criteria:
- 1. Consistency in lighting and shadows: Confirm that the light source and corresponding shadows are coherent across various elements, with no discrepancies in direction or intensity.
- 2. Element cohesion: Every item in the image should logically fit within

- the scene’s context, without any appearing misplaced or extraneous.
- 3. Integration and edge smoothness: Objects or subjects should integrate seamlessly into their surroundings, with edges that do not appear artificially inserted or poorly blended.
- 4. Aesthetic uniformity and visual flow: The image should not only be aesthetically pleasing but also facilitate a natural visual journey, without abrupt interruptions caused by disharmonious elements. Implement a stringent scoring guideline:

- - Award a high score (90-100) solely if the image could pass as a flawlessly captured scene, devoid of any discernible disharmony.
- - Assign a moderate to high score (70-89) if minor elements of disharmony are present but they do not significantly detract from the overall harmony.
- - Give a moderate score (50-69) if noticeable disharmonious elements are evident, affecting the image’s harmony to a moderate degree.
- - Allocate a low score (30-49) for images where disharmonious elements are prominent, greatly disturbing the visual harmony.
- - Reserve the lowest scores (0-29) for images with severe disharmony, where the elements are so discordant that it disrupts the intended aesthetic.

Your assessment must be detailed, highlighting the specific reasons for the assigned score based on the above criteria. Conclude with a response formatted in JSON as shown below: { "Explanation": "<Insert detailed explanation here>", "Score": <Insert precise score here> }

### B Additional Experiment B.1 More visualization results

We visualize the data of InstructPix2Pix in Fig. 11, of MagicBrush in Fig. 12, of HIVE in Fig. 13, and HQ-Edit in Fig. 14 with the Edit instruction, Aligment and Coherence. This shows that HQ-Edit possesses higher image quality and better image-text alignment.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Edit: Make her a farmer

Edit: swap the cyclist for a biker

Alignment：80 Coherence：65

Alignment：40 Coherence：90

##### Fig. 11: Data of InstructPix2Pix, the left side is the input image and the right side is the output image.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Edit: Add a dolphin jumping out of the water

Edit: Turn on the faucet

Alignment：100 Coherence：75

Alignment：0 Coherence：95

##### Fig. 12: Data of MagicBrush, the left side is the input image and the right side is the output image.

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Edit: Change retro to futuristic

Edit: make the man a woman

Alignment：85 Coherence：95

Alignment：50 Coherence：30

##### Fig. 13: Data of HIVE, the left side is the input image and the right side is the output image.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Edit: change her hair color to blonde and add waves to it Alignment：100 Coherence：95

Edit: Replace the heavy-duty power drill with a high-tech precision power tool. Alignment：100 Coherence：95

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Edit: Transform the elderly woman into a young woman, change her traditional dress to a modern black leather jacket, replace her sandals with white sneakers, and add a black handbag beside her on the bench. Alignment：100 Coherence：90

Edit: Change the weather to rainy. Alignment：100 Coherence：95

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Edit: Replace the metal hammer with a plastic toy hammer with a bright orange and blue handle. Alignment：80 Coherence：95

Edit: Change the chameleon's body to a vivid blue hue while keeping the green color on its head crest and tail. Alignment：100 Coherence：100

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Edit: Alter the bird's color to vibrant blue. Change the backdrop to include a framed floral tapestry. Alignment：100

Edit: Replace the Japanese tea set with a Victorian tea set, including porcelain teapots and cups with floral designs, add a lace tablecloth, silver cutlery, and a decorative golden tea strainer. Change the backdrop to include a framed floral tapestry. Alignment：100 Coherence：88

Coherence：95

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

##### Fig. 14: Data of HQ-Edit, the left side is the input image and the right side is the output image.

： ：

： ：

