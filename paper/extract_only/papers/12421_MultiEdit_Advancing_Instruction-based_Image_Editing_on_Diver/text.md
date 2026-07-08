# arXiv:2509.14638v1[cs.CV]18Sep2025

[Figure 1]

[Figure 2]

## MultiEdit: Advancing Instruction-based Image Editing on Diverse and Challenging Tasks

Mingsong Li1,2, Lin Liu1, Hongjun Wang1,3, Haoxing Chen1, Xijun Gu1,4, Shizhan Liu1, Dong Gong2, Junbo Zhao1,4, Zhenzhong Lan1,5, Jianguo Li1†

1Inclusion AI 2University of New South Wales 3The University of Hong Kong 4Zhejiang University 5Westlake University

### Abstract

Current instruction-based image editing (IBIE) methods struggle with challenging editing tasks, as both editing types and sample counts of existing datasets are limited. Moreover, traditional dataset construction often contains noisy image-caption pairs, which may introduce biases and limit model capabilities in complex editing scenarios. To address these limitations, we introduce MultiEdit, a comprehensive dataset featuring over 107K high-quality image editing samples. It encompasses 6 challenging editing tasks through a diverse collection of 18 non-style-transfer editing types and 38 style transfer operations, covering a spectrum from sophisticated style transfer to complex semantic operations like person reference editing and in-image text editing. We employ a novel dataset construction pipeline that utilizes two multi-modal large language models (MLLMs) to generate visual-adaptive editing instructions and produce high-fidelity edited images, respectively. Extensive experiments demonstrate that fine-tuning foundational open-source models with our MultiEdit-Train set substantially improves models’ performance on sophisticated editing tasks in our proposed MultiEdit-Test benchmark, while effectively preserving their capabilities on the standard editing benchmark. We believe MultiEdit provides a valuable resource for advancing research into more diverse and challenging IBIE capabilities. Our dataset is available at https://huggingface.co/datasets/inclusionAI/MultiEdit.

### 1 Introduction

Image editing (Zhu et al., 2017; Couairon et al., 2023; Zhang et al., 2023b; Huang et al., 2025) seeks to enable controllable image generation by modifying target components while preserving the integrity of other visual details. Among various editing paradigms, instruction-based image editing (IBIE) (Hui et al., 2025; Zhang et al., 2023a; Sheynin et al., 2024; Yu et al., 2025; Shi et al., 2024; Zhang et al., 2025) provides an intuitive and flexible approach, enabling users to modify images simply by providing natural language commands, without the need for complex descriptions or region-specific masks. While IBIE presents a promising avenue, current models still face challenges in consistently executing a wide array of instructions with high fidelity. This difficulty can be primarily attributed to a primary bottleneck: the limitations of available training data.

Specifically, existing datasets (Zhang et al., 2023a; Sheynin et al., 2024; Zhao et al., 2024; Yu et al., 2025) focus primarily on relatively simple edits within natural images, such as local or global modifications of single entities. They largely neglect more complex and fine-grained scenarios essential for real-world applications, such as edits requiring spatial or semantic reasoning (e.g., object reference editing and view editing) and manipulations within structured images (e.g., movie posters and GUI interfaces).

Addressing this critical data gap presents two fundamental challenges. First, conventional data construction pipelines (Zhang et al., 2023a; Sheynin et al., 2024; Zhao et al., 2024; Yu et al., 2025) are potentially flawed. They typically utilize large language models (LLMs) to rephrase an image’s original caption into an editing instruction and a target caption. This approach is problematic because captions, as textual abstractions of

†Corresponding Authors.

Pose Clothing Hairstyle Figure

Skin Color

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

Person Reference Editing

Change the pose of the man in the white shirt on the left to a crouchi ng position as i f preparing. to leap

Change the hairstyle of the man in the white Cubs uniform standing near home plate to a short buzz cut.

Change the skin color of the woman in the foreground wearing a white tennis outfit and serving the ball.

Change the uniform of the player wearing number 19 (white jersey) to a bright red color.

Replace the white shirt and green tie of the central man in the foreground with a black turtleneck sweater.

###### Expression Display Medium

###### Font Style Font Color

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Text Editing

[Figure 21]

Change the font color of \"Life as a House\" in the center to blue.

Change the text \"Anne of the Thousand Days\" (center) to \"Queen of a Thousand Days\" while keeping the font and style consistent.

Change font style of text \"LEONARDO DI CAPRIO THE BASKETBALL DIARIES\" to bold italic.

Change the display medium to a notebook with the same document content visible.

###### Mosaic Art Pop Art

###### The Simpsons

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Style Transfer

Change this image to /Mosaic Art/ style.

Change this image to /The Simpsons/ style.

Change this image to /Pop Art/ style.

Icon Attributes

Display Medium

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

GUI Editing

Change the GUI display medium from a smartphone screen to a laptop screen.

Change the color of the top right download icon to green.

###### Person Landmark Object

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

View Editing

Generate alternative views of the aircraft f r o m s l i g h t l y d i f f e r e n t a n g l e s , maintaining visible parts of the original background.

Generate alternative views of the covered boats from slightly lower and angled perspectives, retaining overlap with the original scene.

Generate alternative views of the person: side profile, back view, and 45-degree angle, maintaining overlap with original pose and attire.

Color Shape

###### Scale Position

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

###### Object Reference Editing

Scale down the small elephant located near the center of the image, walking between the larger elephants.

Change the color of the streetlamp positioned to the left of the train to red.

Change the shape of the second boat from the left to a circular raft.

Move the clock tower in the background slightly to the right.

- Figure 1: Overview of the MultiEdit dataset, illustrating its qualitative diversity and compositional structure. The left panel showcases the qualitative diversity of our 6 editing tasks, featuring representative examples from all 18 non-style-transfer editing types and several of the 38 style transfer operations. On the right, the sunburst chart outlines task distribution within MultiEdit, where the inner ring shows the 6 categories and the outer ring details the distribution of specific editing types. This dual perspective highlights the breadth and depth of challenging image editing scenarios covered in MultiEdit.

rich visual content, are susceptible to information loss, noise, and misalignment (Yu et al., 2025; Li et al., 2025; Wang et al., 2025). These fundamental flaws are not only preserved but often amplified during the rephrasing process, rendering the entire pipeline prone to generating low-quality data or prohibitively labor-intensive to correct. Second, a key hurdle lies in the practical execution of advanced edits, as existing editing models often struggle to achieve high-fidelity results for sophisticated tasks without specific training data. However, the recent introduction of the SOTA ImageGen model (i.e., GPT-Image-1 (OpenAI, 2025a;b)) represents a pivotal milestone in multi-modal generative modeling. Its exceptional capacity in complex compositional editing and fine-grained attribute manipulation reveals vast yet largely untapped potential for IBIE tasks.

To address these challenges, we introduce MultiEdit, a large-scale dataset featuring over 107K fine-grained triplets of original images, editing instructions, and edited images. The dataset is meticulously designed to address 6 challenging editing tasks through a comprehensive collection of 56 editing types, spanning 18 distinct non-style-transfer operations and 38 style variations. This composition covers a broad spectrum from creative style transfer to complex semantic operations like reference editing and in-image text editing, with distribution visualized in Fig. 1 and statistical details in Tab. 1. MultiEdit’s construction is guided by a novel multi-modal large language model (MLLM)-driven pipeline with two key innovations. First, to counter potentially flawed conventional pipelines (Zhang et al., 2023a; Sheynin et al., 2024; Zhao et al., 2024; Yu et al., 2025), our approach utilizes the SOTA MLLM (OpenAI, 2025a) to directly interpret source images and generate highly relevant, diverse, and adaptive editing instructions—termed visual-adaptive editing instructions, which ensures firm visual grounding while circumventing reliance on potentially inaccurate

- Table 1: Detailed composition of the MultiEdit dataset, which encompasses 107,634 samples covering diverse editing scenarios from object and person reference editing to creative style transfer tasks. For each of the 6 task categories, we show the source datasets, specific editing types with their corresponding counts, and the precise split of samples into MultiEdit-Train (106,534) and MultiEdit-Test (1,100) sets.

Tasks Detailed Source Datasets editing types Train Samples Test Samples Total Samples

color (2,778), shape (2,565),

Object Reference Editing MS-COCO-object

9,851 200 10,051 scale (2,370), and position (2,338)

pose (1,463), clothing (1,526), hairstyle (1,326),

Person Reference Editing MS-COCO-person

6,891 250 7,141 skin color (1,306), and figure (1,520)

font style (1,012), expression (1,002),

Text Editing Movie-poster

3,860 200 4,060 display medium (1,039), and font color (1,007)

GUI Editing GUI-world icon attributes (997) and display medium (1,883) 2,780 100 2,880

IGPair-person person 11,316 50 11,366 Google-landmark-DS-V2 landmark 11,469 50 11,519

View Editing

DIS5K object 5,270 50 5,320

IGPair-person

12,957 50 13,007 AM2K 4,913 50 4,963

38 styles across 3 groups: Animation Style, Modern and Digital Art, and Classical and Traditional Art†

Style Transfer

Google-landmark-DS-V2 18,379 50 18,429 MS-COCO-object 18,848 50 18,898

Total Count - - 106,534 1,100 107,634

†The 38 styles across 3 groups of Style Transfer task: Animation Style (e.g., ‘Studio Ghibli’, ‘The Simpsons’, and ‘3D Pixar’), Modern and Digital Art (e.g., ‘Pop Art’, ‘3D Illustration’, and ‘Isometric LEGO’), and Classical and Traditional Art (e.g., ‘Mosaic Art’, ‘Paper Cut’, and ‘Xu Beihong’).

captions. Second, to ensure high-fidelity realization of these advanced edits, we leverage the outstanding generative capabilities of the SOTA ImageGen model (i.e., GPT-Image-1 (OpenAI, 2025b)) to generate target edited images, producing faithful visual outcomes for our fine-grained instructions.

To facilitate comprehensive evaluations of IBIE model performance on these complex tasks, we construct MultiEdit-Test, a carefully curated benchmark comprising 1,100 high-quality samples from MultiEdit, covering all targeted tasks and editing types (50 samples per editing type). The remaining 106,534 samples form the large-scale MultiEdit-Train set for model training. Through experiments in various open source models (Esser et al., 2024; Zhao et al., 2024; Yu et al., 2025; Liu et al., 2025), we demonstrate that fine-tuning foundational models with our MultiEdit-Train set substantially elevates their sophisticated editing abilities while maintaining performance on the standard editing benchmark.

Our main contributions are summarized as follows:

- • We introduce MultiEdit, a comprehensive large-scale IBIE dataset comprising over 107K samples targeting 6 challenging image editing tasks covering 56 subcategory editing types (18 non-styletransfer and 38 style transfer). We also release MultiEdit-Test, a carefully curated benchmark of 1.1K samples to assess complex editing performance.
- • We design a novel MLLM-driven data construction pipeline that leverages the SOTA MLLM for visual-adaptive instruction generation directly from source images and the SOTA ImageGen for high-fidelity edited image generation, circumventing limitations of caption-based approaches.
- • We conduct extensive experiments demonstrating that fine-tuning open-source models with our MultiEdit-Train set substantially improves model performance on sophisticated editing tasks while maintaining their capabilities on the standard editing benchmark. We also explore various multi-task learning strategies to establish effective fine-tuning practices for our MultiEdit dataset.

### 2 MultiEdit

#### 2.1 Editing Type Definition

MultiEdit serves as a comprehensive supplement to existing foundational datasets by specifically targeting complex IBIE tasks. As illustrated in Fig. 1 and Tab. 1, the dataset encompasses 6 specially designed task categories, with each data instance forming a triplet of an original image, an editing instruction, and the corresponding edited image. Spanning these categories, the dataset includes 18 distinct non-style-transfer editing types and a comprehensive style transfer component comprising 38 styles across 3 groups.

###### Dataset Collection

###### Source Data Preparation

[Figure 46]

Source Images Edited Images + (Edit Instructions)

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Change the text \"BLUE CHIPS\" to \"GOLDEN SHOTS\" while maintaining the same font and style.

Real World Image Datasets

[Figure 51]

Random/Targeted Selection

[Figure 52]

[Figure 53]

Change the clothing of the person on the right wearing a white shirt and black shorts to a red shirt and blue jeans.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Source Images

The Simpsons 3D Pixar

Image-text Consistency

[Figure 59]

[Figure 60]

Post-quality Assessment

[Figure 61]

[Figure 62]

[Figure 63]

Expert-designed Meta-instructions

SOTA MLLM

Image-text Consistency

[Figure 64]

[Figure 65]

[Figure 66]

Pre-quality Assessment

[Figure 67]

[Figure 68]

Data Cleaning

[Figure 69]

Task-specific Rule-based Filtering

Adaptive Editing-instructions

SOTA ImageGen

Visual-adaptive Editing Instruction Generation

Diverse Image Editing Tasks

- Figure 2: Overview of our MLLM-driven data construction pipeline for MultiEdit. The pipeline consists of three main stages with integrated quality checks: (1) Source data preparation through strategic selection from diverse real-world datasets; (2) Visual-adaptive editing instruction generation using the SOTA MLLM, incorporating expert-designed meta-instructions followed by data cleaning and pre-quality assessment; (3) Diverse-task image editing with the SOTA ImageGen, culminating in comprehensive post-quality assessment to ensure dataset quality.

The 6 task categories are detailed as follows:

- (1) Object reference editing modifies specific attributes (color, shape, scale, and position) of referenced objects, using images from non-person classes of MS-COCO (Lin et al., 2014).
- (2) Person reference editing targets referenced individuals within images, altering their pose, clothing, hairstyle, skin color, and figure, using images from the person class of MS-COCO.
- (3) Text editing focuses on textual elements within movie posters, covering modifications in font style, expression, display medium, and font color, using images from the Movie-poster dataset (Dewidar, 2019).
- (4) GUI editing modifies icon attributes and the display medium of GUI elements, using images of diverse digital interfaces (iOS, Android, and websites) from the GUI-world dataset (Chen et al., 2024).
- (5) View editing generates alternative views of subjects within images, encompassing edits for persons (from IGPair (Shen et al., 2025)), landmarks (from Google-landmark-DS-V2 (Weyand et al., 2020)), and general objects (from DIS5K (Qin et al., 2022)).
- (6) Style transfer applies new artistic styles to images sourced from IGPair-person, AM2K (Li et al., 2022), Google-landmark-DS-V2, and MS-COCO-object. It comprises 38 distinct styles organized into 3 groups: Animation Styles (e.g., ‘Studio Ghibli’ and ‘The Simpsons’), Modern and Digital Art (e.g., ‘Pop Art’ and ‘Geometric Low Poly’), and Classical and Traditional Art (e.g., ‘Oil Painting’ and ‘Paper Cut’). Representative examples of these editing tasks are shown in Figs. 1 and 4.

#### 2.2 Dataset Collection

To overcome the complexities and misalignments of conventional caption-based methods (Zhang et al., 2023a; Sheynin et al., 2024; Zhao et al., 2024; Yu et al., 2025), we introduce a novel MLLM-driven pipeline for MultiEdit. This pipeline operates solely on source images, utilizing the SOTA MLLM for instruction generation and the SOTA ImageGen model for image editing, which significantly streamlines data construction.

Source Data Preparation. As shown in Fig. 2, the initial stage involves preparing source images from real-world datasets through two strategies: random and targeted selection. For random selection, applied to the majority of our primary tasks, we randomly sample source images from established public datasets such as Movie-poster and IGPair-person until the pre-defined quotas are met. For targeted selection, employed for specialized tasks, we perform more intricate curation. For instance, we construct MS-COCO-person and MS-COCO-object subsets by sampling exclusively from the ‘person’ and all other categories within the MS-COCO dataset, respectively. Similarly, for GUI editing, we derive source images by first extracting frames from videos in the GUI-world dataset and then performing random sampling. This dual-strategy approach yields a collection of over 68K source images covering our diverse range of editing tasks.

Visual-adaptive Editing Instruction Generation. Conventional data construction pipelines rely on abstracted captions that are prone to information loss, misalignment, and noise, building on a potentially flawed foundation. To overcome this limitation, we employ the SOTA MLLM (OpenAI, 2025a) to bypass captions and directly generate editing instructions from source images in our MLLM-driven pipeline, as shown in Fig. 2. By integrating expert-designed meta-instructions with input images for various editing tasks, the SOTA MLLM utilizes its advanced visual understanding to generate visual-adaptive editing instructions specific to each image. In Appendix B, we systematically detail the design of expert-designed meta-instructions for generating precise editing instructions across 6 task categories and 56 edit types. Our approach leverages direct visual grounding to significantly reduce information loss and misalignment issues, generating context-aware and accurate instructions for advanced IBIE tasks. This direct visual grounding approach applies to all non-style-transfer tasks. For style transfer, we adopt a template-based strategy, generating 5 instructions for each source image (i.e., Change this image to /XXXX/ style.) based on a prioritized list of styles. Finally, this stage yields a total of 136K source image and editing instruction pairs.

Diverse Image Editing Tasks. While our pipeline excels at generating fine-grained visual-adaptive editing instructions, a key hurdle lies in their practical execution. Due to the lack of specific training data for our targeted complex scenarios, existing open-source models often struggle to render such sophisticated edits with high fidelity, creating a critical gap between our fine-grained instructions and the resulting visual outcomes. To bridge this gap, our MLLM-driven framework employs the powerful multi-modal generative capabilities of the SOTA ImageGen (OpenAI, 2025b) for high-fidelity image editing. This approach ensures that the generated outcomes not only align with the complexity of diverse and challenging IBIE scenarios but also faithfully adhere to the fine-grained details of each visual-adaptive instruction.

Data Cleaning and Quality Assessment. While our MLLM-driven pipeline is highly automated, achieving superior data quality hinges on validating its two critical outputs, i.e., the generated editing instruction and the edited image. An ideal instruction should be explicit and logically sound, and the corresponding edited image should be faithful to the instruction and free of artifacts. To enforce these standards and ensure overall data quality, as shown in Fig. 2, we implement a two-step quality assessment process.

- (1) Data Cleaning and Pre-quality Assessment. This initial step aims to purify the instruction-image pairs through a meticulous process of data cleaning and pre-assessment. The process commences with data cleaning, where we first employ rule-based filters to automatically reject instructions generated from a misalignment between the source image content and the editing type. This step is followed by task-specific curation, such as removing data pairs with low-quality source images (e.g., blurry video frames in GUI editing data). Subsequently, the pre-quality assessment involves evaluating the semantic alignment, clarity, and contextual relevance between each source image and its corresponding instruction.
- (2) Post-quality Assessment. Following diverse-task image editing, the post-quality assessment critically evaluates the generated images based on two key criteria: fidelity, ensuring the consistency of edited regions with the corresponding editing instructions, and preservation, maintaining the integrity of non-edited regions. To efficiently execute this assessment, we employ a strategic two-tiered review process. Initially, a small set of samples from each generation task undergoes a manual accuracy check. Tasks falling below a 70% accuracy threshold are flagged for a comprehensive manual review of all their images. Conversely, for tasks exceeding the 70% threshold, we leverage Qwen2.5-VL (Bai et al., 2025) for assisted screening, with human experts providing oversight on ambiguous cases to guarantee final quality.

- 11k

- 12k 11,766

11,555 Style Groups

Animation Styles

Modern and Digital Art

Classical and Traditional Art

- 4k

- 5k

4,404

NumberofSamples

2k

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |1,812<br><br>1,623 1,582<br><br>1,441 1,436 1,436| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |1,326 1,294<br><br>1,004<br><br>944 898| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |701<br><br>547<br><br>494<br><br>835<br><br>739 727 716 697<br><br>640 624<br><br>550<br><br>495<br><br>436<br><br>837<br><br>700<br><br>631<br><br>542<br><br>442| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |300 299<br><br>41<br><br>384<br><br>200 199| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

2k

- 0k
- 1k
- 2k

0k

StudioGhibliTheSimpsonsKyotoAnimationPokemon3DPixarDragonBallDetectiveConanOnePieceDemonSlayerNarutoDoraemonAttackOnTitanCartoonArtAnimeArtSatoshiKonGlitchArtTypographyArt3DVoxelArtGeometricLowPolyPopArtOneLineDrawing3DIllustrationSmudgedOilPaintingLineDrawingIsometricLEGOTechnicalDrawingIllustrationArtWatercolorPixelArtCyberneticSurrealismMosaicArtPointillismOilPaintingChineseInkPaintingPaperCutClaudeMonetArtNouveauXuBeihongPaperCutCraft

- Figure 3: Distribution of the style transfer task in MultiEdit. The dataset comprises 55,297 samples across 38 distinct artistic styles, organized into 3 categories: Animation Styles, Modern and Digital Art, and Classical and Traditional Art. With specific sample counts detailed for each style, the distribution strategically emphasizes popular styles like ‘Studio Ghibli’ and ‘The Simpsons’ while maintaining balanced coverage across diverse artistic domains. The broken y-axis accommodates the scale differences.

#### 2.3 Characteristics and Statistics

Constructed through our streamlined pipeline, MultiEdit ultimately comprises over 107K high-quality, fine-grained triplets, each consisting of an original source image, a corresponding visual-adaptive editing instruction, and the resultant edited image. The scope of the involved 6 editing tasks within MultiEdit is notably broad and advanced, spanning from complex semantic operations like reference-based object or person editing and in-image text editing, to specialized applications such as GUI editing and generative view editing for diverse subjects, including persons, landmarks, and objects.

A visual overview of MultiEdit is provided in Fig. 1, which showcases representative examples with a sunburst chart illustrating the compositional structure. For a detailed breakdown, Tab. 1 presents the precise composition for each task, including source datasets, precise train-test splits, and the sample distribution across specific editing types. Beyond these core tasks, MultiEdit features a dedicated style transfer component, comprising 55,297 samples across 38 distinct styles organized into 3 groups. Fig. 3 quantifies the precise sample count for each of all 38 styles, revealing a distribution strategically weighted towards highly popular styles like ‘Studio Ghibli’ and ‘The Simpsons’ while maintaining broad coverage of others. Complementing this distributional view, Fig. 4 provides a qualitative showcase of all 38 styles, each illustrated with a representative source-edit pair.

#### 2.4 Multi-task Learning Framework

The heterogeneous nature of MultiEdit, encompassing 6 distinct task categories with varying complexity and data distributions, naturally forms as a multi-task learning (MTL) problem (Kendall et al., 2018; Liu et al., 2019; Yu et al., 2020; Liu et al., 2021). Specifically, our dataset exhibits a hierarchical complexity structure: style transfer tasks are data-abundant yet relatively straightforward, tasks such as object and person reference editing and view editing present moderate complexity with semantic reasoning requirements, while text and GUI editing tasks constitute the most challenging scenarios with related fewer training samples. This inherent task imbalance and complexity heterogeneity necessitate a systematic MTL approach to effectively leverage the diverse supervision signals across all editing tasks.

Within the MTL framework, existing optimization strategies can be systematically categorized across three complementary dimensions.

• Data-driven MTL (DMTL) solutions address task imbalance through sophisticated sampling strategies, including task-weighted data sampling, curriculum learning schedules (Bengio et al., 2009),

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

[Figure 81]

Studio Ghibli The Simpsons Kyoto Animation Pokemon

3D Pixar Dragon Ball

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

Detective Conan

Cartoon Art

One Piece Demon Slayer Naruto Doraemon Attack On Titan

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

Anime Art

Geometric Low Poly

Glitch Art Typography Art 3D Voxel Art

Satoshi Kon

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

3D Illustration Line Drawing Smudged Oil Painting Isometric LEGO

Pop Art

One Line Drawing

[Figure 120]

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

Illustration Art Technical Drawing Watercolor Pixel Art Cybernetic Surrealism Mosaic Art

Pointillism

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

Xu Beihong

Claude Monet

Paper Cut

Paper Cut Craft

Oil Painting

Chinese Ink Painting

Art Nouveau

- Figure 4: Representative examples of all 38 styles from the style transfer subset of MultiEdit, grouped by category. Background colors correspond to the 3 groups: Animation Styles (pink), Modern and Digital Art (blue), and Classical and Traditional Art (purple). Best viewed when zoomed in.

and adaptive balancing techniques that dynamically adjust training data distribution based on task difficulty and learning progress. In our work, we explore a data-driven curriculum by partitioning our MultiEdit-Train set Dtrain into K complexity-based task groups {Dk}. During each training stage s, a weighted random sampler determines the selection probability for any sample xi ∈ Dk based on its group’s pre-defined weight wk(s):

wk(s) ∑Kj=1 |Dj| · w(js)

P(xi | Dk, s) =

. (1)

By adjusting the weight vector w(s) between stages, we can dynamically shift the training focus among the pre-defined task groups as training progresses.

- • Architecture-driven MTL (AMTL) solutions tackle the challenge of parameter sharing by designing specialized network components, ranging from task-specific pathways (Misra et al., 2016; Chen et al., 2023) and cross-task attention mechanisms (Liu et al., 2019) to adaptive parameter allocation strategies that optimize capacity distribution across different learning objectives.
- • Loss-driven MTL (LMTL) solutions focus on gradient dynamics and objective balance, encompassing uncertainty-based loss weighting (Kendall et al., 2018), gradient conflict mitigation techniques (Yu et al., 2020), and adaptive loss scaling methods that ensure balanced learning across heterogeneous

tasks. In our work, we explore two sample-wise strategies for re-weighting the per-sample loss ℓi, computed as:

ℓi = Mean ωt · (ϵˆi − ϵi)2 , (2)

where ϵˆi is the model’s noise prediction, ϵi is the target noise, ωt is the timestep-dependent weighting factor, and Mean[·] denotes spatial averaging over latent dimensions. The final re-weighted loss Li

for sample i is then given by:

Li = ωi · ℓi, (3) where ωi is the sample-wise re-weighting coefficient determined by one of the following strategies.

- (1) A heuristic approach sets the weight ωi for a sample from task group k within a mini-batch B based on its inverse frequency with a smoothing factor α:

ωi ∝ (Nk(B) + α)−1, (4) where Nk(B) denotes the count of samples from group k in the current mini-batch B.

- (2) A dynamic approach derives the weight ωi from the L2 norm of the gradient of the per-sample loss

with respect to the model’s noise prediction, as Gi = ∥∇ϵˆiℓi∥2. We test two weighting schemes: setting ωi to be either directly proportional to Gi to amplify harder examples, or inversely proportional to Gi to stabilize training:

ωi ∝ Gi (proportional) or ωi ∝ G−1

i (inverse). (5)

### 3 Experiments

#### 3.1 Experimental Setup

Baselines. We conducted experiments with open-source IBIE methods, including Stable Diffusion 3 (SD3) (Esser et al., 2024), UltraEdit (Zhao et al., 2024), AnyEdit (Yu et al., 2025), and Step1X-Edit (Liu et al., 2025). For SD3 and UltraEdit, we adopt SD3-Medium-based models and use the stage-1 UltraEdit for free-form image editing. For Step1X-Edit, we employ version 1.0.

Benchmarks & Metrics. In addition to our MultiEdit-Test benchmark, we utilize EmuEdit-Test (Sheynin et al., 2024) to conduct more comprehensive experiments. Following prior work (Zhao et al., 2024; Yu et al., 2025), we adopt semantic similarity (i.e., CLIPimg and CLIPtxt (Radford et al., 2021)), visual similarity (i.e., DINO (Caron et al., 2021)), and pixel-level differences (i.e., L1 and L2 distances) as metrics to evaluate the effectiveness of different methods on IBIE tasks.

Implementation Details. To ensure consistent referencing across experiments, we adopt the following notation to denote all model variants trained on our MultiEdit-Train (ME) dataset: ME-{Base-Model}{Training-Method}-{Dataset-Info}, and the components are:

- • Base-Model: Base model (SD = SD3-based and UEdit = UltraEdit-based).
- • Training-Method: Training strategy (SFT = Supervised Fine-tuning, DMTL = Data-driven MTL, and LMTL = Loss-driven MTL).
- • Dataset-Info: Optional data composition (Mix = MultiEdit-Train + Mixed training data from UltraEdit).

For the training setup, we fine-tune models for 5 epochs using the AdamW optimizer (Loshchilov & Hutter, 2017). Training uses a constant-with-warm-up learning-rate schedule with final linear annealing, starting at 5e-5 with a 12-step warm-up and linearly decaying to 5e-7 in the final epoch. Across 8 GPUs, we maintain an effective batch size of 256, configured with a per-GPU batch size of 4 and 8 gradient accumulation steps. For mixed data training, warm-up steps are scaled proportionally to the mixed data size (e.g., 24 steps for 100K mixed samples). The detailed model inference guidance experiment and the guidance summary are available in Appendix C.

#### 3.2 Quantitative Evaluation

We first benchmark four baseline methods on MultiEdit-Test set, evaluating their performance across the diverse tasks proposed in MultiEdit. At the same time, to explore the efficacy of our proposed MultiEdit dataset, we fine-tune two foundational open-source models, SD3 and UltraEdit, on our MultiEdit-Train set. For SD3 and UltraEdit, we only activate MMDiT and keep the rest of the models frozen. To further refine

SD3: https://huggingface.co/stabilityai/stable-diffusion-3-medium UltraEdit: https://github.com/HaozheZhao/UltraEdit AnyEdit: https://dcd-anyedit.github.io/ Step1X-Edit: https://step1x-edit.github.io/

- Table 2: Quantitative comparison of different methods on MultiEdit-Test and EmuEdit-Test. Rows in gray indicate fine-tuned models with our MultiEdit-Train set. The results demonstrate that foundational models can be significantly improved for our complex editing scenarios by supplementing them with targeted data. Results of AnyEdit on EmuEdit-Test are from its original paper. Best and second-best results are shown in bold and underlined, respectively.

MultiEdit-Test EmuEdit-Test

Method

CLIPimg↑ DINO↑ L1↓ L2↓ CLIPimg↑ CLIPtxt↑ DINO↑ L1↓ L2↓ SD3 0.6817 0.5751 0.1983 0.0854 0.7019 0.2443 0.5094 0.1708 0.0625 UltraEdit 0.8017 0.7303 0.1976 0.0871 0.8685 0.2706 0.8055 0.1137 0.0406 AnyEdit 0.8067 0.7246 0.1926 0.0826 0.8720 0.2850 0.8210 0.0700 Step1X-Edit 0.8335 0.7466 0.1954 0.0852 - - - - -

ME-SD-SFT 0.7759 0.7359 0.1901 0.0858 0.7832 0.2662 0.6431 0.2055 0.0878 ME-SD-SFT-Mix100K 0.7863 0.7586 0.1926 0.0883 0.8319 0.2723 0.7551 0.1535 0.0619 ME-UEdit-SFT 0.8127 0.8019 0.1827 0.0813 0.8384 0.2690 0.7597 0.1903 0.0796 ME-UEdit-DMTL 0.8174 0.8071 0.1814 0.0800 0.8376 0.2681 0.7610 0.1896 0.0793

their distinct strengths, we enhance the foundational SD3 via mixed data training with 100K image editing data from UltraEdit. For the more advanced UltraEdit, we adopt the best explored MTL strategy, i.e., the task-weighted data sampling strategy of the data level to modulate the task sampling distribution within our MultiEdit-Train set. Further experiments of these strategies are detailed in Sec. 4.

As illustrated in Tab. 2, even advanced methods like AnyEdit and Step1X-Edit lack full proficiency in handling the diverse and complex tasks in MultiEdit-Test. This is particularly evident in DINO scores, where Step1X-Edit only reaches approximately 0.75. In contrast, both SD3 and UltraEdit exhibit evident performance improvements on our MultiEdit-Test set after direct SFT. For the SD3 model, this process yields substantial gains on MultiEdit-Test, with CLIPimg and DINO scores increasing by approximately 9.4% and 16.1%, respectively. Notably, this training also boosts its CLIPimg performance on EmuEdit-Test set by about 8.1%. Building upon this, our mixed data training strategy provides a further performance lift on both datasets, ultimately achieving peak CLIPimg and DINO scores of 0.8319 and 0.7551 on EmuEdit-Test. For the UltraEdit model, direct SFT yields a performance increase of approximately 1.1% in CLIPimg and 7.2% in DINO on MultiEdit-Test, while effectively preserving its capabilities on EmuEdit-Test. Adopting the task-weighted data sampling strategy in DMTL further improves its performance, achieving a CLIPimg score close to the SOTA model Step1X-Edit and surpassing it on DINO by over 5%. This demonstrates that by supplementing with targeted data, even foundational models can be significantly improved for complex editing scenarios. We also report detailed results of the selected methods across all 6 tasks of our MultiEdit-Test in Appendix D.

#### 3.3 Qualitative Evaluation

Figs. 5 and 6 present a qualitative comparison of four baseline methods against our two best-performing finetuned models in the 6 MultiEdit-Test task scenarios. From this comparison, we draw several key observations. (1) In line with our quantitative results, fine-tuning with targeted data significantly enhances foundational models like SD3 and UltraEdit for diverse and complex editing tasks, evident in their markedly improved instruction comprehension and precise editing. (2) Despite these improvements, there remains scope for further enhancement for our fine-tuned models regarding fine-grained details, particularly in facial fidelity. (3) The challenges posed by our diverse and complex data scenarios are evident even in the performance of state-of-the-art models, with Step1X-Edit showing limited instruction comprehension in tasks like object reference editing (Fig. 5, row 2) and person reference editing (Fig. 5, row 5). (4) AnyEdit struggles with style transfer, leaving ample room for improvement with the targeted data.

### 4 Discussion

In this section, we discuss several key strategies for optimizing model training on our MultiEdit-Train set. We primarily investigate multi-task learning (MTL) approaches at the data and loss levels. We also explore the synergy achieved by mixing our MultiEdit-Train set with external data for broader task compatibility.

Change the color of the yellow front panel of the train to red.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

###### Object Reference Editing

Change the color of the rock near the polar bear's front paw to bright blue.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

###### Object Reference Editing

Change the tall silver refrigerator on the right to a spherical shape.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

###### Object Reference Editing

Change the figure of the man in the yellow referee uniform to be wearing a blue tracksuit.

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

###### Person Reference Editing

Change the clothing of the woman on the left wearing a blue top and floral pants to a red dress.

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

###### Person Reference Editing

Change font color of "James Stewart Shelley Winters Dan Duryea Stephen McNally" to red.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Text Editing

Change the font style of the text "The Legend of Boggy Creek" at the bottom to italic.

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Text Editing

Change the display medium of the document to appear on a notebook.

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

###### Text Editing

Input SOTA ImageGen SD3 UltraEdit AnyEdit Step1X-Edit

ME-SD-SFT-Mix ME-UEdit-DMTL

- Figure 5: Qualitative comparison of different methods across various tasks on MultiEdit-Test, including object reference editing, person reference editing, and text editing. Best viewed when zoomed in.

#### 4.1 Multi-task Optimization Strategies

Task-weighted Data Sampling. To validate the data-driven MTL (DMTL) strategy formulated in Sec. 2.4, we implement a heuristic task-weighted data sampling scheme guided by pre-defined complexity groups. Specifically, we categorize our tasks into four groups: G1 (style transfer), G2 (object reference editing and view editing (object)), G3 (person reference editing, view editing (person), and view editing (landmark)), and G4 (text editing and GUI editing). Following the curriculum learning principle (Bengio et al., 2009), we apply a two-stage training regimen with distinct sampling weight vectors w(s) (initial 2 epochs and subsequent 3 epochs). The specific weighting configuration for Setting1 is as follows:

Change the lock icon in the center of the screen to red color.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

GUI Editing

Change the GUI display medium to a notebook screen.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

GUI Editing

Generate alternative perspectives of the person, such as side, back, and angled views, maintaining overlap with the original fur collar.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

View Editing

Generate alternative views of the covered boats from slightly lower and angled perspectives, retaining overlap with the original scene.

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

View Editing

Change this image to /The Simpsons/ style.

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Style Transfer

Change this image to /Geometric Low Poly/ style.

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Style Transfer

Change this image to /Pop Art/ style.

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Style Transfer

Change this image to /3D Pixar/ style.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

Style Transfer

Input SOTA ImageGen SD3 UltraEdit AnyEdit Step1X-Edit ME-SD-SFT-Mix ME-UEdit-DMTL

- Figure 6: Qualitative comparison of different methods across various tasks on MultiEdit-Test, including GUI editing, view editing, and style transfer. Best viewed when zoomed in.

#### Task-weighted data sampling configuration instance: Setting1

- Stage 1 (epochs 0-1): weight vector w(1) = (w(G11), w(G12), w(G13), w(G14)) = (1,1,1,5);
- Stage 2 (epochs 2-4): weight vector w(2) = (w(G21), w(G22), w(G23), w(G24)) = (1,1,1,10)

As quantified in the DMTL part of Tab. 3, while standard SFT improves upon the original UltraEdit, our task-weighted data sampling strategies yield further performance gains. Notably, Setting4 emerges as the optimal configuration, delivering the best overall results on both MultiEdit-Test and EmuEdit-Test. This validates the strategy of aggressively up-weighting the more complex G3 and G4 tasks. Thus, we adopt it as a standard setup of DMTL for our primary experiments.

Sample-wise Loss Re-weighting. To complement the DMTL strategy, we also investigate the loss-driven MTL (LMTL) paradigm by implementing the two sample-wise re-weighting strategies defined in Sec. 2.4.

- Table 3: Comparison of data-driven and loss-driven strategies for multi-task fine-tuning on our MultiEditTrain set. Best results per group are in bold.

UltraEdit Config

MultiEdit-Test EmuEdit-Test

CLIPimg↑ DINO↑ L1↓ L2↓ CLIPimg↑ CLIPtxt↑ DINO↑ L1↓ L2↓

UltraEdit 0.8017 0.7303 0.1976 0.0871 0.8685 0.2706 0.8055 0.1137 0.0406 ME-UEdit-SFT 0.8127 0.8019 0.1827 0.0813 0.8384 0.2690 0.7597 0.1903 0.0796

ME-UEdit-DMTL: task-weighted data sampling

- +Setting1† (1,1,1,5)-(1,1,1,10) 0.8154 0.8038 0.1850 0.0816 0.8326 0.2686 0.7494 0.1988 0.0849
- +Setting2 (1,1,1,3.5)-(1,1,1,7.5) 0.8169 0.8056 0.1872 0.0834 0.8337 0.2686 0.7488 0.2045 0.0892
- +Setting3 (1.5,1,2,10)-(1.5,1,2.5,15) 0.8053 0.7930 0.1828 0.0822 0.8356 0.2681 0.7561 0.1956 0.0833
- +Setting4 (1.5,1,2,10)-(1.5,1,3.5,20) 0.8174 0.8071 0.1814 0.0800 0.8376 0.2681 0.7610 0.1896 0.0793

ME-UEdit-LMTL: sample-wise loss re-weighting

+Task Group Balancing (α = 0) 0.8118 0.7988 0.1874 0.0841 0.8350 0.2686 0.7538 0.1930 0.0818 +Task Group Balancing (α = 1.0) 0.8100 0.7991 0.1865 0.0832 0.8347 0.2689 0.7551 0.1922 0.0809 Gradient Focusing, proportional 0.8080 0.7952 0.1856 0.0828 0.8341 0.2679 0.7516 0.1937 0.0819 Gradient Balancing, inverse 0.8133 0.8051 0.1834 0.0814 0.8357 0.2681 0.7579 0.1910 0.0807

† Please refer Sec. 4.1 for the detailed setting instance of task-weighted data sampling.

- Table 4: Comparison of fine-tuning SD3 with different mixed UltraEdit data. Best results are shown in bold.

MultiEdit-Test EmuEdit-Test

SD3 Config

CLIPimg↑ DINO↑ L1↓ L2↓ CLIPimg↑ CLIPtxt↑ DINO↑ L1↓ L2↓

SD3 0.6817 0.5751 0.1983 0.0854 0.7019 0.2443 0.5094 0.1708 0.0625 ME-SD-SFT 0.7759 0.7359 0.1901 0.0858 0.7832 0.2662 0.6431 0.2055 0.0878

+Mix50K 0.7839 0.7554 0.1885 0.0857 0.8292 0.2725 0.7435 0.1516 0.0600 +Mix100K 0.7863 0.7586 0.1926 0.0883 0.8319 0.2723 0.7551 0.1535 0.0619 +Mix150K 0.7856 0.7574 0.1876 0.0848 0.8409 0.2720 0.7668 0.1464 0.0578 +Mix200K 0.7876 0.7611 0.1972 0.0933 0.8376 0.2729 0.7573 0.1572 0.0645

First, corresponding to Eq. 4, we implement the heuristic intra-batch task group balancing using three predefined categories: (1) style transfer, (2) the most challenging text and GUI editing, and (3) all others. Here, we test a standard version (α = 0) with direct inverse frequency and a smoothed version applying additive smoothing with the smoothing factor α = 1.0 to the group counts, enhancing training stability. Second, we explore the more dynamic gradient-based sample re-weighting approach that re-weights each sample based on the L2 norm of the gradient of the per-sample loss with respect to the model’s noise prediction, as Gi = ∥∇ϵˆiℓi∥2. Corresponding to Eq. 5, we test two opposing strategies: gradient focusing (ωi ∝ Gi), which uses directly proportional weights to prioritize challenging examples, and gradient balancing (ωi ∝ G−1

i ), which assigns weights inversely proportional to the gradient norm to improve stability.

As illustrated in the LMTL part of Tab. 3, for the intra-batch task group balancing loss strategies, the unsmoothed version slightly outperforms its smoothed counterpart. In gradient-based re-weighting strategies, gradient balancing employing inverse proportionality demonstrates superior performance to gradient focusing, and emerges as the best-performing method within the LMTL group. Nevertheless, this improvement is surpassed by the more direct and effective DMTL strategy of task-weighted data sampling, which achieves higher performance gains on MultiEdit-Test.

#### 4.2 Mixed Data Training

To validate that mixing our challenging MultiEdit-Train data with broader external data enhances general task compatibility, we fine-tune SD3 on our MultiEdit-Train set while incrementally mixing UltraEdit data from 50K to 200K samples, with detailed results in Tab. 4. Compared to SFT solely on the MultiEdit-Train set, this

mixed data strategy yields marginal improvements on MultiEdit-Test but delivers more significant gains on EmuEdit-Test, achieving peak scores of 0.8409 on CLIPimg and 0.7668 on DINO. This result suggests a strong synergy between the complex tasks in MultiEdit and the more foundational editing tasks from UltraEdit. Based on these results, we adopt the 100K data mixing configuration for our main method comparisons, as presented in Tab. 2.

### 5 Related Work

Instruction-based Image Editing Models. Instruction-based image editing (IBIE) is a rapidly advancing field that enables image modification through natural language commands. InstructPix2Pix (Brooks et al., 2023) pioneers this paradigm by a prompt-to-prompt (Hertz et al., 2022) method and generating a large-scale synthetic dataset for IBIE. In contrast, MagicBrush (Zhang et al., 2023a) demonstrates the value of real-world data by manually curating a high-quality yet expensive 10K-pair dataset from human annotators. Building on these foundational methods, subsequent research introduces numerous advancements. Some models focus on boosting multi-modal interaction and instruction fidelity through novel architectural designs, including SmartEdit (Huang et al., 2024), UltraEdit (Zhao et al., 2024), and AnyEdit (Yu et al., 2025). In parallel, other works concentrate on training efficiency. For example, ICEdit (Zhang et al., 2025) employs LoRAMoE hybrid tuning to achieve strong performance while minimizing training costs. Concurrently, another prominent line of research unlocks new possibilities by integrating MLLM-based multi-modal reasoning with diffusion-based controllable generation, like Step1X-Edit (Liu et al., 2025) and Qwen-Image (Wu et al., 2025).

Instruction-based Image Editing Datasets. High-quality training data are the cornerstone for developing powerful IBIE models. While foundational datasets like InstructPix2Pix and MagicBrush pave the way, subsequent research largely focuses on increasing scale and generation fidelity. For instance, Emu Edit (Sheynin

- et al., 2024) scales its dataset to 10 million proprietary pairs but offers limited public access. HQ-Edit (Hui
- et al., 2025) uses DALL-E 3 to generate 197K high-fidelity pairs, though these pairs lack the pixel-to-pixel alignment critical for fine-grained details. In a different approach, UltraEdit (Zhao et al., 2024) contributes 4 million pairs using an inpainting-based strategy. Beyond scale and fidelity, the diversity and complexity of task design are equally critical. Datasets like AnyEdit (Yu et al., 2025) begin to address this by curating 2.5 million pairs across a wide array of tasks, including more complex types like viewpoint and implicit editing. However, existing datasets still lack the breadth to cover multifaceted real-world scenarios and the depth required for training models on complex compositional and detail-oriented editing tasks.

The SOTA ImageGen-related Synthetic Datasets The recent release of the SOTA ImageGen (i.e., GPTImage-1) marks a significant milestone in multi-modal generative modeling. Following its introduction, several empirical studies benchmark its capabilities, evaluating a heterogeneous set of tasks from generation quality and editing proficiency to advanced multi-modal reasoning (Chen et al., 2025b; Yan et al., 2025; Cao et al., 2025). Capitalizing on these advanced capabilities, a key research trend is to leverage the SOTA ImageGen to generate curated training corpora, in an effort to bridge the performance gap between open-source and proprietary models. For instance, OmniConsistency (Song et al., 2025) utilizes the SOTA ImageGen’s exceptional stylization consistency to generate 2.6K style transfer image pairs. Focusing on more complex generation, Echo-4o-Image (Ye et al., 2025) provides an approximately 180K-sample dataset generated by the SOTA ImageGen to boost model performance on demanding tasks like multi-reference synthesis and complex instruction execution. Similarly, ShareGPT-4o-Image (Chen et al., 2025a) presents a 91K dataset, divided into 45K T2I samples and 46K IBIE examples. At the most enormous scale, GPT-ImageEdit-1.5M (Wang et al., 2025) constructs a massive IBIE corpus of over 1.5 million samples by unifying and refining existing datasets.

Following this line of work, based on the SOTA ImageGen, we introduce MultiEdit, a new large-scale IBIE dataset specifically targeting a spectrum of diverse and challenging image editing tasks. Structurally, this dataset comprises 6 distinct categories, spanning 18 non-style-transfer editing types and a comprehensive style transfer component with 38 styles across 3 groups.

### 6 Conclusion

In this work, we introduce MultiEdit, a comprehensive dataset for advancing instruction-based image editing beyond simple scenarios. It features over 107K high-quality editing samples from 6 challenging task categories, spanning 18 non-style-transfer editing types and a comprehensive style transfer component with 38 styles. Our dataset pipeline leverages the SOTA MLLM for visual-adaptive instruction generation and the SOTA ImageGen for high-fidelity edited image generation. We formally partition the complete

collection into a large-scale MultiEdit-Train set (∼106K samples) for model training and a carefully curated MultiEdit-Test benchmark (1.1K samples) for rigorous evaluation. Our experiments demonstrate that finetuning foundational models (SD3-based) with our MultiEdit-Train set, along with explored multi-task learning strategies, greately improves their sophisticated editing capabilities while effectively preserving performance on the standard benchmark. Ultimately, we believe MultiEdit provides a crucial resource to facilitate exploration into more diverse and challenging capabilities.

### Limitations

While MultiEdit represents an advance in diverse and challenging IBIE tasks, its current limitations also delineate clear pathways for our future research. First, as a supplementary dataset targeting diverse and challenging image editing tasks, MultiEdit requires integration with broader foundational corpora to train a general-purpose IBIE model from scratch effectively. Second, in experiments, the full potential of our dataset is constrained by the current SD-based architecture. Thus, a critical next step involves leveraging next-generation models, such as FLUX and unified multi-modal architectures, to unlock higher performance levels on complex image editing tasks.

### References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In ICML, pp. 41–48, 2009.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, pp. 18392–18402, 2023.

Pu Cao, Feng Zhou, Junyi Ji, Qingye Kong, Zhixiang Lv, Mingjian Zhang, Xuekun Zhao, Siqi Wu, Yinghui Lin, and Qing Song. Preliminary explorations with gpt-4o (mni) native image generation. arXiv preprint arXiv:2505.05501, 2025.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, pp. 9650–9660, 2021.

Dongping Chen, Yue Huang, Siyuan Wu, Jingyu Tang, Liuyi Chen, Yilin Bai, Zhigang He, Chenlong Wang, Huichi Zhou, Yiqiang Li, et al. Gui-world: A dataset for gui-oriented multimodal llm-based agents. arXiv preprint arXiv:2406.10819, 2024.

Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025a.

Sixiang Chen, Jinbin Bai, Zhuoran Zhao, Tian Ye, Qingyu Shi, Donghao Zhou, Wenhao Chai, Xin Lin, Jianzong Wu, Chao Tang, et al. An empirical study of gpt-4o image generation capabilities. arXiv preprint arXiv:2504.05979, 2025b.

Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik G Learned-Miller, and Chuang Gan. Mod-squad: Designing mixtures of experts as modular multi-task learners. In CVPR, pp. 11828–11837, 2023.

Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic

image editing with mask guidance. In ICLR, 2023. Mostafa Dewidar. Inferring movie genres from their poster. Learning, 1, 2019. Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller,¨ Harry Saini, Yam Levi,

Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-toprompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Liangliang Cao, and Shifeng Chen. Diffusion model-based image editing: A survey. IEEE TPAMI, 2025.

Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In CVPR, pp. 8362–8371, 2024.

Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Cihang Xie, and Yuyin Zhou. Hq-edit: A high-quality dataset for instruction-based image editing. In ICLR, 2025.

Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In CVPR, pp. 7482–7491, 2018.

Jizhizi Li, Jing Zhang, Stephen J Maybank, and Dacheng Tao. Bridging composite and real: towards end-to-end deep image matting. IJCV, 130(2):246–266, 2022.

Ming Li, Xin Gu, Fan Chen, Xiaoying Xing, Longyin Wen, Chen Chen, and Sijie Zhu. Superedit: Rectifying and facilitating supervision for instruction-based image editing. arXiv preprint arXiv:2505.02370, 2025.

Jinpeng Lin, Min Zhou, Ye Ma, Yifan Gao, Chenxi Fei, Yangjian Chen, Zhang Yu, and Tiezheng Ge. Autoposter: A highly automatic and content-aware design system for advertising poster generation. In ACM MM, pp. 1250–1260, 2023.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.

Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. NeurIPS, 34:18878–18890, 2021.

Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In CVPR, pp. 1871–1880, 2019.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2017. Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task

learning. In CVPR, pp. 3994–4003, 2016. OpenAI. Addendum to gpt-4o system card: 4o image generation. 2025a. OpenAI. Introducing our latest image generation model in the api. 2025b. Xuebin Qin, Hang Dai, Xiaobin Hu, Deng-Ping Fan, Ling Shao, and Luc Van Gool. Highly accurate

dichotomous image segmentation. In ECCV, pp. 38–56. Springer, 2022.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pp. 8748–8763, 2021.

Fei Shen, Xin Jiang, Xin He, Hu Ye, Cong Wang, Xiaoyu Du, Zechao Li, and Jinhui Tang. Imagdressing-v1: Customizable virtual dressing. In AAAI, volume 39, pp. 6795–6804, 2025.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In CVPR, pp. 8871–8879, 2024.

Yichun Shi, Peng Wang, and Weilin Huang. Seededit: Align image re-generation to image editing. 2024. Yiren Song, Cheng Liu, and Mike Zheng Shou. Omniconsistency: Learning style-agnostic consistency from

paired stylization data. arXiv preprint arXiv:2505.18445, 2025. Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt-imageedit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025. Tobias Weyand, Andre Araujo, Bingyi Cao, and Jack Sim. Google landmarks dataset v2-a large-scale benchmark for instance-level recognition and retrieval. In CVPR, pp. 2575–2584, 2020.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782, 2025.

Junyan Ye, Dongzhi Jiang, Zihao Wang, Zhenghao Hu Leqi Zhu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, Conghui He, and Weijia Li. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025.

Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In CVPR, 2025.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. NeurIPS, 33:5824–5836, 2020.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. NeurIPS, 36:31428–31449, 2023a.

Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690, 2025.

Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. Sine: Single image editing with text-to-image diffusion models. In CVPR, pp. 6027–6037, 2023b.

Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. NeurIPS, 37: 3058–3093, 2024.

JunYan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In ICCV, pp. 2223–2232, 2017.

###### Movie-poster

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Translate the text \"He's too fast...too quick...and two fisted\" at the top into Chinese.

Translate the text \"TOM CRUISE MI:2 MAY 24\" at the center of the image into Chinese.

Translate the text \"OF HUMAN BONDAGE\" in the center into Chinese while preserving its position and style.

Auto-poster

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

Change \"薄皮亲妈 耐受救星\" (purple box text) to express excitement instead of reassurance.

Change text \“润白防晒 清透防水\” (line 1) to express morex authoritative. Retain the original style.

Change text \"抢! 买正装送正装\" (top center) to express urgency more dramatically.

- Figure 7: Editing triplets involving Chinese elements, sampled from the Movie-poster and Auto-poster datasets. Each row contrasts one successful edit (the first triplet) with two failure cases, revealing the limitations of the SOTA ImageGen’s Chinese text handling abilities.

### A Text Editing Related to Chinese Characters

Given the growing attention on rendering non-alphabetic languages (Wu et al., 2025), we evaluate the SOTA ImageGen’s Chinese text handling abilities during the construction of MultiEdit. As shown in Fig. 7, for the text editing task based on the Movie-poster dataset (Dewidar, 2019), the model demonstrates reasonable performance in translating short English phrases into Chinese. However, rendering longer Chinese texts leads to diminished stability and detail preservation, often failing to achieve flawless presentation. The challenges are even more pronounced with the Auto-poster dataset (Lin et al., 2023) for text editing, which contains substantial pre-existing Chinese advertising text. Corruptions and inaccuracies are commonly observed when using the SOTA ImageGen model to edit in-image text, underscoring the current challenges in this text editing domain. This observation of the SOTA ImageGen aligns with the findings in recent work (Yan et al., 2025; Song et al., 2025; Wu et al., 2025).

Consequently, due to these identified shortcomings, we exclude the Auto-poster subset for text editing and the Chinese Translation editing type of related tasks from the final MultiEdit dataset to ensure its overall quality and effectiveness. We leave the challenge of high-quality image editing involving non-alphabetic languages, particularly Chinese, for our future work.

### B Systematic Expert-designed Meta-instructions

As detailed in Sec. 2.2, our pipeline employs the SOTA MLLM to generate the visual-adaptive editing instruction from each source image. This generation process is structured by a set of expert-designed metainstructions. For style transfer, we utilize a template-based strategy to encompass diverse artistic styles. For other complex tasks, we systematically define a suite of meta-instructions, which are enumerated below.

#### Object Reference Editing

- (1) Color: Please randomly pick an object (other than a person) from the image, and give an instruction for the image editing model to change its color. Please describe the picked object’s position precisely with words (e.g., 2nd chair from the left), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.
- (2) Shape: Please randomly pick an object (other than a person) from the image, and give an instruction for the image editing model to change its shape. Please describe the picked object’s position precisely with words (e.g., 2nd chair from the left), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.
- (3) Scale: Please randomly pick an object (other than a person) from the image, and give an instruction for the image editing model to change its scale. Please describe the picked object’s position precisely with words (e.g., 2nd chair from the left), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.
- (4) Position: Please randomly pick an object (other than a person) from the image, and give an instruction for the image editing model to change its position. Please describe the picked object’s position precisely with words (e.g., 2nd chair from the left), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.

#### Person Reference Editing

- (1) Pose: Please randomly pick a person from the image, and give an instruction for the image editing model to change its pose. Please describe the picked person’s position precisely with words (e.g., 2nd man from the left), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.
- (2) Clothing: Please randomly pick a person from the image, and give an instruction for the image editing model to change its clothing. Please describe the picked person’s position precisely with words (e.g., 2nd man from the left), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.
- (3) Hairstyle: Please randomly pick a person from the image, and give an instruction for the image editing model to change its hairstyle. Please describe the picked person’s position precisely with words (e.g., 2nd man from the left), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.
- (4) Skin Color: Please randomly pick a person from the image, and give an instruction for the image editing model to change its skin color. Please describe the picked person’s position precisely with words (e.g., 2nd man from the left), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.
- (5) Figure: Please randomly pick a person from the image, and give an instruction for the image editing model to change its figure. Please describe the picked person’s position precisely with words (e.g., 2nd man from the left), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.

#### Text Editing

- (1) Font Style: Please randomly select some text or a paragraph from the image, and provide an instruction for the image editing model to change its font style. Precisely indicate the position of the selected content (e.g., line 2-3, 2nd paragraph, ”use this on-line proclamation”), and output the edit instruction directly without any additional words. Ensure the instruction is within 50 tokens, and avoid selecting too small pieces of content.
- (2) Expression: Please randomly select some text or a paragraph from the image, and provide an instruction for the image editing model to change its expression. Precisely indicate the position of the selected content (e.g., line 2-3, 2nd paragraph, ”use this on-line proclamation”), and output the edit instruction directly without any additional words. Ensure the instruction

- is within 50 tokens, and avoid selecting too small pieces of content.
- (3) Display Medium: Please randomly provide an instruction for the image editing model to change the given document’s display medium (e.g., computer screen and notebook). Please output the edit instruction directly without any additional words. Ensure the instruction is within 50 tokens.
- (4) Font Color: Please randomly select some text or a paragraph from the image, and provide an instruction for the image editing model to change its font color. Precisely indicate the position of the selected content (e.g., line 2-3, 2nd paragraph, ”use this on-line proclamation”), and output the edit instruction directly without any additional words. Ensure the instruction is within 50 tokens, and avoid selecting too small pieces of content.

#### GUI Editing

- (1) Icon Attributes: Please randomly pick an icon from the image, and give an instruction for the image editing model to change its attributes (e.g., color and shape). Please describe the picked icon position precisely with words (e.g., the top right home icon), and output the edit instruction directly without any additional words. Please make sure the edit instruction is within 50 tokens.
- (2) Display Medium: Please randomly provide an instruction for the image editing model to change the given GUI’s display medium (e.g., computer screen and notebook). Please output the edit instruction directly without any additional words. Ensure the instruction is within 50 tokens.

#### View Editing

- (1) Person: Based on this image, create an editing instruction for the image editing model to generate alternative views of the person shown. Please provide the instruction directly without additional commentary, and keep it within 50 tokens. Ensure the new perspective maintains some overlap with the original image.
- (2) Landmark: Based on this image, create an editing instruction for the image editing model to generate alternative views of the landmark shown. Please provide the instruction directly without additional commentary, and keep it within 50 tokens. Ensure the new perspective maintains some overlap with the original image.
- (3) Object: Based on this image, create an editing instruction for the image editing model to generate alternative views of the object shown. Please provide the instruction directly without additional commentary, and keep it within 50 tokens. Ensure the new perspective maintains some overlap with the original image.

### C Model Inference Guidance Experiments

For model inference, we employ a consistent setting of 50 inference steps for all models, with the exception of AnyEdit, for which we use its default of 100 steps. Regarding the guidance scale, we adhere to the default configurations for baseline models. AnyEdit uses image guidance scale of 3.0 and text guidance scale of 3.0, and Step1X-Edit operates with a cfg guidance of 6.0. The foundational SD3 directly utilizes the hyperparameters of StableDiffusion3Img2ImgPipeline with a strength of 0.6 and a guidance scale of 7.0.

We utilize the StableDiffusion3InstructPix2PixPipeline for the base UltraEdit and all our fine-tuned models. To determine the optimal guidance, we conduct detailed hyperparameter experiments for guidance scale based on UltraEdit and our enhanced ME-UEdit-DMTL, with results presented in Tab. 5. Based on this analysis and our qualitative observations, we adopt a universal text guidance scale of 4.0 for UltraEdit and our fine-tuned models. For the image guidance scale, UltraEdit sets it to 2.0. Our UltraEdit-based fine-tuned models use 3.5 on MultiEdit-Test and 5.0 on EmuEdit-Test, while our SD3-based variants maintain 3.5 on MultiEdit-Test and adopt 2.0 on EmuEdit-Test, aligning with the base UltraEdit’s setting. The final parameter configurations for all models are summarized in Tab. 6.

Table 5: Guidance scale tests on MultiEdit-Test and EmuEdit-Test. Best results per group are shown in bold.

MultiEdit-Test EmuEdit-Test

Method Text-G

Image-G CLIPimg↑ DINO↑ L1↓ L2↓ Image-G CLIPimg↑ CLIPtxt↑ DINO↑ L1↓ L2↓

- 1.5 0.7826 0.7072 0.1938 0.0846 1.5 0.8538 0.2741 0.7688 0.1197 0.0421
- 2.0 0.7932 0.7229 0.1994 0.0882 2.0 0.8649 0.2722 0.7936 0.1177 0.0419

5.0

- 2.5 0.7943 0.7198 0.2067 0.0927 2.5 0.8643 0.2703 0.7952 0.1219 0.0436
- 3.0 0.7907 0.7107 0.2137 0.0974 3.0 0.8566 0.2684 0.7850 0.1297 0.0466

- 1.5 0.7882 0.7163 0.1925 0.0839 1.5 0.8580 0.2740 0.7773 0.1173 0.0414
- 2.0 0.7968 0.7264 0.1988 0.0878 2.0 0.8662 0.2708 0.7969 0.1159 0.0412

UltraEdit

4.5

- 2.5 0.7983 0.7235 0.2056 0.0921 2.5 0.8647 0.2695 0.7986 0.1207 0.0432
- 3.0 0.7920 0.7133 0.2129 0.0968 3.0 0.8568 0.2679 0.7880 0.1285 0.0461

1.5 0.7931 0.7210 0.1919 0.0835 1.5 0.8611 0.2725 0.7862 0.1153 0.0406 2.0 0.8017 0.7303 0.1976 0.0871 2.0 0.8685 0.2706 0.8055 0.1137 0.0406

4.0

- 2.5 0.7997 0.7283 0.2044 0.0913 2.5 0.8651 0.2682 0.8018 0.1191 0.0426
- 3.0 0.7934 0.7156 0.2118 0.0961 3.0 0.8580 0.2663 0.7905 0.1273 0.0457

- 3.0 0.8096 0.7985 0.1847 0.0833 4.5 0.8366 0.2737 0.7555 0.1957 0.0834

- 3.5 0.8129 0.8024 0.1838 0.0818 5.0 0.8383 0.2716 0.7570 0.1944 0.0824
- 4.0 0.8136 0.7984 0.1852 0.0826 5.5 0.8374 0.2703 0.7585 0.1928 0.0813

- 4.5 0.8148 0.7986 0.1862 0.0825 6.0 0.8378 0.2693 0.7596 0.1920 0.0808

5.0

- 3.0 0.8122 0.8039 0.1828 0.0815 4.5 0.8376 0.2719 0.7578 0.1937 0.0817

- 3.5 0.8153 0.8028 0.1831 0.0813 5.0 0.8377 0.2704 0.7593 0.1922 0.0811
- 4.0 0.8161 0.8052 0.1832 0.0808 5.5 0.8378 0.2690 0.7604 0.1902 0.0796

- 4.5 0.8160 0.8000 0.1842 0.0807 6.0 0.8369 0.2673 0.7604 0.1899 0.0794

ME-UEdit-DMTL

4.5

- 3.0 0.8168 0.8069 0.1795 0.0791 4.5 0.8381 0.2696 0.7587 0.1910 0.0802

- 3.5 0.8174 0.8071 0.1814 0.0800 5.0 0.8376 0.2681 0.7610 0.1896 0.0793
- 4.0 0.8171 0.8060 0.1812 0.0791 5.5 0.8377 0.2672 0.7608 0.1898 0.0793

- 4.5 0.8162 0.7985 0.1857 0.0818 6.0 0.8382 0.2661 0.7591 0.1892 0.0789

4.0

Table 6: Inference hyperparameters summary for all methods.

Method Inference Step Guidance Config

SD3 50 strength = 0.6, guidance scale = 7.0 UltraEdit 50 image guidance scale = 2.0, text guidance scale = 4.0 AnyEdit 100 image guidance scale = 3.0, text guidance scale = 3.0 Step1x-Edit 50 cfg guidance = 6.0

SD3-based: image guidance scale = 3.5, text guidance scale = 4.0 UltraEdit-based: image guidance scale = 3.5, text guidance scale = 4.0 SFT models for EmuEdit-Test 50

SFT models for MultiEdit-Test 50

SD3-based: image guidance scale = 2.0, text guidance scale = 4.0 UltraEdit-based: image guidance scale = 5.0, text guidance scale = 4.0

### D Detailed Quantitative Results on MultiEdit-Test

For a detailed task-level analysis, we present quantitative results in Tabs. 7 and 8 comparing the bestperforming fine-tuned method based on our MultiEdit-Train set (ME-UEdit-DMTL) with four baseline methods, which are also supported by the qualitative examples in Figs. 5 and 6. Specifically, ME-UEditDMTL consistently excels across a diverse range of challenging tasks, including object reference editing, GUI editing, text editing, and style transfer. In contrast, baseline SOTA methods show apparent limitations in specific scenarios. For instance, Step1X-Edit struggles with challenging text editing tasks involving complex display mediums, while AnyEdit’s effectiveness is restricted in style transfer scenarios.

- Table 7: Task-level quantitative comparison of different methods on reference and GUI editing from our MultiEdit-Test benchmark. Best and second-best results are shown in bold and underlined, respectively.

Object Reference Editing Person Reference Editing GUI Editing color shape scale position pose clothing hairstyle skin figure icon display SD3

Metric

CLIPimg↑ 0.7609 0.7135 0.7473 0.7608 0.6994 0.6757 0.6616 0.6855 0.7217 0.7033 0.6705 DINO↑ 0.6204 0.5311 0.5833 0.6201 0.6285 0.5864 0.5878 0.6308 0.6308 0.7457 0.7383

L1↓ 0.1590 0.1572 0.1770 0.1675 0.1888 0.1723 0.1795 0.1708 0.1861 0.1337 0.2700 L2↓ 0.0542 0.0513 0.0636 0.0606 0.0710 0.0643 0.0659 0.0614 0.0724 0.0690 0.1724 UltraEdit

CLIPimg↑ 0.8823 0.8278 0.8940 0.8851 0.8423 0.8419 0.8453 0.8582 0.8442 0.7914 0.7394 DINO↑ 0.8286 0.7402 0.8289 0.8053 0.7610 0.7833 0.8138 0.8365 0.7721 0.8153 0.7875

L1↓ 0.1477 0.1502 0.1602 0.1625 0.1786 0.1688 0.1649 0.1630 0.1802 0.1324 0.2774 L2↓ 0.0493 0.0491 0.0565 0.0589 0.0680 0.0646 0.0608 0.0591 0.0708 0.0715 0.1811 AnyEdit

CLIPimg↑ 0.8935 0.8600 0.8971 0.8626 0.8506 0.8695 0.8515 0.8726 0.8484 0.7815 0.7076 DINO↑ 0.8469 0.7868 0.8247 0.7632 0.7598 0.8149 0.8238 0.8591 0.7727 0.7817 0.6854

L1↓ 0.1414 0.1392 0.1660 0.1559 0.1713 0.1561 0.1578 0.1523 0.1696 0.1582 0.3278 L2↓ 0.0475 0.0441 0.0595 0.0559 0.0628 0.0585 0.0558 0.0526 0.0649 0.0847 0.2009 Step1X-Edit

CLIPimg↑ 0.8867 0.8549 0.9010 0.8947 0.8626 0.8812 0.8682 0.8828 0.8619 0.8872 0.6037 DINO↑ 0.8075 0.7329 0.8250 0.8263 0.7748 0.8319 0.8464 0.8430 0.7953 0.8951 0.4499

L1↓ 0.1473 0.1523 0.1647 0.1532 0.1789 0.1572 0.1596 0.1569 0.1707 0.1271 0.3723 L2↓ 0.0529 0.0529 0.0601 0.0546 0.0675 0.0593 0.0568 0.0556 0.0650 0.0696 0.2335 ME-UEdit-DMTL

CLIPimg↑ 0.8809 0.8607 0.8912 0.8905 0.8334 0.8546 0.8401 0.8551 0.8467 0.7744 0.7220 DINO↑ 0.8473 0.8041 0.8419 0.8418 0.8050 0.8304 0.8510 0.8570 0.8147 0.8448 0.8047

L1↓ 0.1470 0.1435 0.1568 0.1593 0.1708 0.1545 0.1633 0.1545 0.1711 0.1311 0.2378 L2↓ 0.0528 0.0490 0.0564 0.0601 0.0677 0.0596 0.0622 0.0581 0.0690 0.0710 0.1485

- Table 8: Task-level quantitative comparison of different methods on text editing, view editing, and style transfer sub-tasks from our MultiEdit-Test benchmark. Best and second-best results are shown in bold and underlined, respectively.

Text Editing View Editing Style Transfer

Metric

fontStyle expression display fontColor person landmark object person animal landmark object SD3

CLIPimg↑ 0.6375 0.6202 0.4809 0.6089 0.6782 0.7093 0.8136 0.7087 0.6584 0.6313 0.6505 DINO↑ 0.5245 0.5183 0.4423 0.5300 0.7026 0.6124 0.6673 0.5619 0.3642 0.4319 0.3927

L1↓ 0.2188 0.2063 0.3374 0.2148 0.1327 0.1978 0.1880 0.1933 0.2406 0.2452 0.2266 L2↓ 0.1062 0.0975 0.1894 0.1093 0.0504 0.0773 0.0699 0.0836 0.0986 0.1031 0.0878 UltraEdit

CLIPimg↑ 0.7368 0.7324 0.7080 0.7706 0.8395 0.8215 0.8738 0.7107 0.7458 0.7052 0.7410 DINO↑ 0.6774 0.6917 0.6900 0.7204 0.8184 0.7199 0.7641 0.6280 0.4510 0.5424 0.5897

L1↓ 0.2175 0.2066 0.3464 0.2132 0.1458 0.2017 0.1867 0.2070 0.2566 0.2493 0.2315 L2↓ 0.1071 0.0996 0.1984 0.1108 0.0578 0.0818 0.0711 0.0935 0.1101 0.1055 0.0911 AnyEdit

CLIPimg↑ 0.7556 0.7392 0.6528 0.7816 0.8507 0.8435 0.8869 0.7023 0.7741 0.7117 0.7462 DINO↑ 0.6852 0.6931 0.5802 0.7235 0.7925 0.7460 0.8023 0.5857 0.4545 0.5526 0.6093

L1↓ 0.2113 0.2011 0.3256 0.2028 0.1912 0.1904 0.1712 0.1866 0.2291 0.2319 0.2140 L2↓ 0.1020 0.0958 0.1741 0.1003 0.0912 0.0733 0.0614 0.0793 0.0911 0.0947 0.0798 Step1X-Edit

CLIPimg↑ 0.7766 0.7811 0.5886 0.8033 0.9190 0.8393 0.8991 0.8335 0.8626 0.8065 0.8434 DINO↑ 0.7101 0.6892 0.4585 0.7555 0.8633 0.7156 0.7797 0.7351 0.6714 0.6881 0.7304

L1↓ 0.2090 0.2040 0.3283 0.2042 0.1242 0.2038 0.1807 0.1967 0.2372 0.2440 0.2272 L2↓ 0.0994 0.0962 0.1757 0.1021 0.0486 0.0842 0.0674 0.0838 0.0962 0.1025 0.0909 ME-UEdit-DMTL

CLIPimg↑ 0.7037 0.7016 0.6276 0.7187 0.8361 0.8170 0.8809 0.8617 0.8865 0.8263 0.8736 DINO↑ 0.7736 0.7758 0.7093 0.7881 0.8547 0.7686 0.7863 0.8226 0.7880 0.7531 0.7938

L1↓ 0.1981 0.1950 0.2810 0.1899 0.1581 0.1952 0.1864 0.1658 0.2096 0.2156 0.2069 L2↓ 0.0989 0.0980 0.1528 0.0993 0.0783 0.0793 0.0740 0.0682 0.0855 0.0880 0.0829

