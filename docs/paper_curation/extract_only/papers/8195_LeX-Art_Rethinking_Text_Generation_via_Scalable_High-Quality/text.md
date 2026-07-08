# arXiv:2503.21749v1[cs.CV]27Mar2025

## LeX-Art: Rethinking Text Generation via Scalable High-Quality Data Synthesis

Shitian Zhao1* Qilong Wu1∗ Xinyue Li1∗ Bo Zhang1 Ming Li1 Qi Qin1 Dongyang Liu1,2 Kaipeng Zhang1 Hongsheng Li2 Yu Qiao1 Peng Gao1 Bin Fu1† Zhen Li2† 1Shanghai AI Laboratory 2The Chinese University of Hong Kong https://zhaoshitian.github.io/lexart/

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>MultipleWords<br><br>"Creativity", "Art", "Inspiration", "Imagination",<br><br>"Design",<br><br>"Color", "Craft", "Vision"<br><br>"Learn", "Play", "Grow",<br><br>"Build",<br><br>"Create", "Explore", "Think", "Solve", "Imagine"<br><br>“Sweet”, “Delight”, "Joy",<br><br>"Celebrate",<br><br>"Tasty", "Treats", "Artisan", "Festive", "Beautiful", "Homemade"<br><br>| |
|---|---|
|[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>ComplexLayout<br><br>"Steel", "Panther", "Rockdown", "In", "The",<br><br>"Lockdown",<br><br>"Gift"<br><br>"SIMPLY", "Bullet", "Proof", "Nothing",<br><br>"SAUCER"<br><br>"Brain",<br><br>"Teasers",<br><br>"Challenge", "Accepted", "Mind", "Games", "Logic", "Quest"<br><br>| |
|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>TextAttributes<br><br>"Hot" in the<br><br>cursive font style; "Rod" in the cursive font style; "Clear",<br><br>which are<br><br>3D letters<br><br>"HAPPY" in green, "NEW" in purple,<br><br>"YEAR"<br><br>in yellow<br><br>"Birthday" in yellow, "Billy" in yellow<br><br>| |

Figure 1. Given the prompts for visual text generation, our proposed LeX-FLUX and LeX-Lumina can generate text images with multiple words, aesthetic complex layout, and good text attributes controllability.

### Abstract

We introduce LeX-Art, a comprehensive suite for highquality text-image synthesis that systematically bridges the gap between prompt expressiveness and text rendering fidelity. Our approach follows a data-centric paradigm, constructing a high-quality data synthesis pipeline based on Deepseek-R1 to curate LeX-10K, a dataset of 10K high-resolution, aesthetically refined 1024×1024 images. Beyond dataset construction, we develop LeX-Enhancer,

∗ Equal Contribution † Corresponding authors

a robust prompt enrichment model, and train two textto-image models, LeX-FLUX and LeX-Lumina, achieving state-of-the-art text rendering performance. To systematically evaluate visual text generation, we introduce LeXBench, a benchmark that assesses fidelity, aesthetics, and alignment, complemented by Pairwise Normalized Edit Distance (PNED), a novel metric for robust text accuracy evaluation. Experiments demonstrate significant improvements, with LeX-Lumina achieving a 79.81% PNED gain on CreateBench, and LeX-FLUX outperforming baselines in color (+3.18%), positional (+4.45%), and font accuracy

(+3.81%). Our codes, models, datasets, and demo are publicly available.

### 1. Introduction

Visual text generation is an important task for text-to-image (T2I) models. Producing text images that are clear, accurate, and aesthetically pleasing significantly enhances productivity in the design industry. Previous efforts, constrained by the limitations of foundational models such as SD-1.5 [38] and SD-XL [34] have resulted in poor performance in visual text generation. Consequently, researchers has focused predominantly on improving the accuracy of text, while overlooking an equally important aspect: the aesthetic quality of the generated text and its seamless integration and interaction with image content.

Due to the limitations of base T2I models, recent works have adopted control-based [54] approaches for accurate visual text rendering, incorporating glyph information into the generation pipeline through specialized glyph modules, e.g., AnyText [43, 44], GlyphControl [52], TextDiffuser [5, 6], and Glyph-ByT5 [31, 32]. Incorporating glyph information in the image generation process can enhance control over text accuracy and layout in generated images, which is particularly beneficial for paragraph text generation. However, these methods often compromise diversity, aesthetics, and seamless integration with surrounding visual content. While ensuring text accuracy—regardless of its fusion with the background—may suffice for structured text scenarios like poster design, it falls short in applications requiring minimal yet visually harmonious text, such as slogans, logos, and artistic typography. In these cases, aesthetic appeal, integration with the overall image, and dynamic layout become crucial.

To achieve high-quality multi-word visual text generation with diverse font styles, dynamic layouts, and strong aesthetic appeal, we take an alternative approach that avoids the constraints of control-based methods. Instead of introducing additional control signals, we aim to maximize the inherent text rendering capabilities of base T2I models through high-quality data curation and supervised fine-tuning.

Existing open-sourced and web-crawled text image datasets [12, 40, 43, 44] exhibit significant limitations in enhancing the text rendering capabilities of foundation models. This is primarily due to the scarcity of high-quality text image samples within these datasets, making it challenging to meet the demands of model training. Given these limitations, relying solely on curated datasets is insufficient for achieving high-quality text rendering. Instead, a more scalable and systematic approach is required to generate diverse, aesthetically rich, and well-integrated text samples. To address these challenges, we introduce LeX-Art, a com-

prehensive framework for high-quality text-image synthesis and enhancement, designed to systematically bridge the gap between prompt expressiveness and high-quality rendered text.

As shown in Fig. 2, LeX-Art consists of three key components: (1) dataset construction, (2) model fine-tuning, and (3) benchmark evaluation. Our approach begins by leveraging the strong reasoning capabilities of DeepSeekR1 [13] to refine seed prompts extracted from the AnyWord3M [43] dataset. These enriched prompts incorporate detailed textual attributes such as font styles, color schemes, and spatial layouts, significantly improving the quality of textual descriptions for image synthesis. With these enhanced prompts, we construct LeX-10K, a dataset of 10K high-quality text-image pairs. To ensure data reliability, we introduce a multi-stage filtering and recaptioning process: Q-Align [50] and Paddle-OCR-v3 [26] assess fidelity, aesthetic quality, and text bounding box coverage, while a knowledge-augmented recaptioning module refines textual descriptions to enhance alignment with generated images. Beyond dataset creation, we leverage 60,856 DeepSeek-R1enhanced prompts to fine-tune a locally deployable model, LeX-Enhancer, which specializes in robust prompt enrichment for text-image generation. We further fine-tune two visual text generation models, LeX-FLUX and LeX-Lumina, using LeX-10K. Notably, our experiments demonstrate that LeX-FLUX achieves superior text rendering performance, while even the lightweight LeX-Lumina (2B) benefits significantly, highlighting the impact of our data synthesis strategy. To systematically evaluate our approach, we propose LeX-Bench, a benchmark designed to assess visual text generation across multiple dimensions, including text fidelity, aesthetics, and alignment with input prompts. Furthermore, we introduce Pairwise Normalized Edit Distance (PNED), a new metric for evaluating text accuracy. PNED computes the discrepancy between ground-truth and OCRdetected words using NED-based [53] matching, making it a flexible metric for non-glyph-conditioned visual text generation. Experimental results demonstrate that LeX-Art provides a scalable and effective framework for improving text-image generation, offering a systematic solution to the challenges of high-quality text rendering.

### 2. Related Work

Text-to-image models. Image generation has long been a pivotal problem in the field of artificial intelligence. To efficiently and effectively generate high-quality images, researchers have developed a variety of algorithms. These include Generative Adversarial Networks (GANs) [4, 11, 19, 20, 48, 55]; Variational Autoencoders (VAEs) [1, 21, 23, 37, 45, 46]; and Diffusion Models [3, 7, 10, 16]. However, in recent years, Flow Matching [28, 30] has emerged as the de facto paradigm in image generation due to its training

stability and sampling efficiency.

###### LeX-Art Framework

As described above, with continuous advancements in modeling paradigms and architectural designs, image generation models have achieved remarkable progress in generating realistic portraits, high-resolution images, and text image synthesis. Among these, text rendering has become increasingly important as a challenging task with significant economic value, making it a key metric for evaluating the capabilities of text-to-image models. In SD3 [9] and Seedream [10], they use DPO [36, 47] to improve the model’s text rendering ability. In Playground-v3 [29], they use a fine-grained caption to improve the text rendering capability. In this work, we focus on improving the text rendering capabilities of T2I models via post-training with highquality synthetic data.

Datasets & Models

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

R1-Distilled Data LeX-10K

User: Simple caption + Instruction Assistant: Enhanced Caption

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

:

:

Visual text generation. A prevailing explanation for the difficulty of text rendering in text-to-image generation models is that the text information in the prompt, after being processed by the text encoder, often becomes distorted or inadequately preserved. This degradation prevents the model from effectively rendering textual content within the generated image. To address this issue, much work has focused on architectural modifications [5, 6, 31–33, 43, 44, 52], such as incorporating additional modules to explicitly inject glyph information into the image generation process.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

|LeXFLUX|
|---|

|LeX-Enhancer|
|---|

|LeXLumina|
|---|

However, models like FLUX [24, 25] have demonstrated that even without glyph modules, a model conditioned solely on prompts can achieve decent text rendering capabilities. Furthermore, most contemporary text-toimage models already possess a baseline ability to render text [9, 10, 24, 25, 27, 42], though they struggle with challenges such as multi-text scenarios, fine-grained text attribute control, and complex layouts.

Benchmark & Metric

|LeXFLUX/ Lumina|
|---|

LeX-Bench

[Figure 26]

[Figure 27]

Easy, Medium, Hard

[Figure 28]

[Figure 29]

Given this, instead of introducing architectural modifications, we explore an alternative perspective by enhancing the model’s text rendering capability through data-centric approaches. Specifically, we aim to improve the model’s ability to handle the aforementioned challenges by carefully designing and augmenting the training data. This approach leverages the inherent potential of existing model architectures while addressing their limitations in text rendering.

|PNED|
|---|

Evaluation Result

[Figure 30]

[Figure 31]

[Figure 32]

: Seed Prompts : R1-Distilled Data : Trainable

Figure 2. Illustration of LeX-Art Framework.

### 3. LeX-Art

tion. Together, as shown in Fig. 2, these modules form a scalable solution for improving visual text generation.

To tackle the challenges of accurate and aesthetic text rendering in image generation, we propose LeX-Art, a unified framework that enhances both data quality and model capability. LeX-Art consists of four components: (a) LeX-10K, a high-quality dataset built through prompt refinement and multi-stage filtering; (b) LeX-Enhancer, a prompt enrichment model, along with two fine-tuned generation models, LeX-FLUX and LeX-Lumina; (c) LeX-Bench, a benchmark for evaluating fidelity, aesthetics, and alignment; and (d) PNED, a new metric for flexible text accuracy evalua-

#### 3.1. LeX-10K dataset

Existing web-crawled text image datasets, such as AnyWord-3M [43, 44] and MARIO-10M [5, 6], suffer from several key limitations, including inconsistent and low resolution (typically below 1024 × 1024), frequent blurriness from direct online sourcing, lack of aesthetic curation resulting in unbalanced and unappealing layouts, and shallow captions that overlook crucial visual details like font,

|[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>Prompt Enhance<br><br>Knowledge-Augmented Recaption<br><br>Filtering<br><br>An image of London venues for every type of christmas party, with<br><br>the text on it: "party", "christmas",<br><br>"EVERY", "TYPE", "OF", "FOR", "VENUES", "LONDON".<br><br>The image features a vibrant, festive composition centered around bold, stylized text arranged dynamically against a warm, glowing background reminiscent of a winter evening in London. The primary text "LONDON" is positioned prominently at the top in large, uppercase serif letters with a metallic gold finish, casting subtle shadows and bordered by tiny twinkling lights. Below it, the word "VENUES" stretches across a curved, ribbon-like banner in deep crimson red with white trim, adorned with holly sprigs and pine cones. The phrase "FOR EVERY TYPE OF CHRISTMAS PARTY" is scattered playfully around the central elements: "FOR" and "OF" appear in smaller, embossed silver text on either side of "EVERY," which is highlighted in bright white with a soft halo effect inside a green wreath. "TYPE" is rendered in bold, sans-serif black letters on a golden gift box tilted slightly to the right, while "CHRISTMAS" glows in red script with a subtle glitter texture beside "PARTY," which bursts upward in bubbly, multicolored letters surrounded by floating confetti and snowflakes. The background blends soft gradients of twilight blue and purple, dotted with distant silhouettes of London landmarks like the Shard and Big Ben, illuminated by golden window lights. Warm ambient lighting creates gentle highlights and shadows, enhancing the depth of the layered elements without overwhelming the clarity of the text.<br><br>DeepSeek-R1<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>FLUX.1dev<br><br>[Figure 45]<br><br>Text to<br><br>Image<br><br>Q-Align<br><br>[Figure 46]<br><br>PP-OCR<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>The image presents a festive design featuring bold, illuminated text and decorative elements on a warm, glowing background. At the top, the word "LONDON" is rendered in large,<br><br>metallic gold serif letters with glowing<br><br>lights embedded in them, casting subtle shadows. Below, "VENUES" is displayed on a crimson red ribbon with white trim, slightly curved and prominently centered. A lush green wreath adorned with holly berries and pine cones frames the black text "FOR EVERY OF," though the placement of "OF" appears<br><br>slightly awkward. To the right,<br><br>"Christmas" is written in glowing red script, while "PARTY" is depicted in large, red block letters with a glittery texture. A small golden gift box with a red ribbon sits near the base of the wreath, adding a festive touch. The background blends warm hues of<br><br>orange and brown with soft, glowing<br><br>light orbs scattered throughout, creating a cozy holiday atmosphere. The overall composition is vibrant and celebratory, though some text alignment appears slightly inconsistent.<br><br>GPT-4o<br><br>[Figure 50]<br><br>PromptSeed LeX-10K Dataset<br><br>[Figure 51]<br><br>|
|---|

- Figure 3. The framework of data construction pipeline. The red words in the R1 enhanced prompt are not rendered in the generated image, and it is fixed after the knowledge-augmented recaption by gpt-4o.

|Original Caption | GPT-4o | Gemini-2.0-flash | Claude-3.5-Haiku | DeepSeek-R1|
|---|
|[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]|

- Figure 4. Images generated by FLUX.1 [dev] [25] based on different prompts. The origin caption from the first raw to the bottom raw:

(1) A poster with the words Good Music remixed and unreleased on it, with text on it: “UNRELEASED”, “REMIXED”, “GOOD.MUSIC”, “KANYEWEST”, “SPERIOD”. (2) A movie poster, with text on it: “AFACE”, “WITHOUT”, “EYES”, “DOL”, “JUL”. (3) A menu of a fast food restaurant that contains “Sandwich Combo”, “Grilled Chicken”, “Lettuce”, “Tomato”, “Mayo”, “Fries&Drink”, and “Pepsi”.

[Figure 67]

[Figure 68]

[Figure 69]

- Figure 5. Image quality score and image aesthetics score distribution of AnyText dataset [43] and LeX-10K. We randomly sampled 10K data entries from AnyWord-3M. Using QAlign [50], we calculated the quality scores and aesthetic scores for these 10K data entries along with the images in LeX-10K, and visualized the distributions of these two types of scores. We observed that LeX-10K generally has higher quality scores and aesthetic scores overall.

color, and text positioning. To address these issues, we introduce LeX-10K, a high-quality text image dataset constructed through a structured four-stage pipeline: (1) enhancing prompts using DeepSeek-R1 [13], (2) filtering and selecting high-resolution, visually pleasing images, (3) refining captions via knowledge-augmented recaptioning with GPT-4o [18]. The data construction pipeline is shown in Fig. 3.

DeepSeek-R1 as a prompt enhancer. In text-to-image generation, enriching prompts with fine-grained visual details has been shown to significantly improve output quality [41]. Motivated by this, we first curated a large-scale dataset of millions of simple captions extracted from text images. To ensure data quality, we filtered out OCR results that were empty, meaningless (e.g., “*”), or overly verbose (e.g., exceeding 50 words), and retained only captions with fewer than 15 words.

To enrich these simple captions with detailed visual attributes, we prompted various large language models (LLMs) using a well-designed instruction (see Appendix A). As shown in Fig. 4, we evaluated GPT-4o [18], Gemini-2.0-flash [8], Claude3.5-Haiku [2], and DeepSeek-

R1 [13]. Among them, DeepSeek-R1 stood out by generating more fine-grained descriptions, including color, spatial layout, and font style of each text element. In contrast, other models tended to provide only high-level summaries of the text content. These detailed prompts are crucial for highquality text image generation, underscoring the need for a rigorous selection process to retain the most effective outputs.

Synthetic image filtering. To mitigate the impact of random seeds on image quality, we generate five distinct images for each enhanced prompt using different seeds. Following the Best-of-N approach [49], we first select the image with the highest quality and aesthetic scores. We then further discard any images in which the rendered text is too small to read.

To address issues of blurriness and poor readability observed in some generated images, we adopt a two-stage filtering strategy. First, we use Q-Align [50] to score each image based on a weighted combination of quality and aesthetic scores, with quality assigned twice the weight to emphasize clarity. The five images are then ranked by their overall scores, and the highest-scoring image is retained. Second, since small rendered text correlates with lower visual clarity, we apply Paddle-OCR-v3 [26] to detect all text regions and calculate their bounding box areas. We discard images where the largest text region is smaller than a threshold of 4000 pixels. After applying both filtering steps, we curate a final dataset of 10,000 high-quality images.

Knowledge-augmented recaptioning. While filtering improves image quality, ensuring accurate alignment between images and captions remains crucial. However, due to the limitations of FLUX.1 [dev] [25], the generated images do not always perfectly match the enhanced prompts—common issues include missing text, incorrect font styles, or wrong colors. To further improve the quality of image-text pairs, we leverage GPT-4o to refine the captions based on the actual visual content. Specifically, each image and its original caption are fed into GPT-4o, which is instructed to revise the caption to better reflect the image while preserving as much of the original content as possible. The detailed prompt instructions are provided in Appendix B. We find that this knowledge-augmented recaptioning process better preserves key visual details—such as font style and text attributes.

Data statistics. Following the generation, filtering, and knowledge-augmented recaptioning steps, we construct LeX-10K, a dataset comprising 10K high-quality text images paired with well-aligned, knowledge-rich captions. As illustrated in Fig. 5, LeX-10K achieves significantly higher image quality and aesthetic scores compared to a randomly

sampled subset of AnyWord-3M [43], highlighting its advantage in text rendering tasks.

#### 3.2. LeX-Enhancer, LeX-FLUX and LeX-Lumina

To achieve high-quality text rendering in text-to-image (T2I) generation, both knowledge-rich prompts and strong T2I models are essential. To this end, we develop a prompt enhancer, LeX-Enhancer, and two optimized T2I models, LeX-FLUX and LeX-Lumina.

LeX-Enhancer is a lightweight prompt enhancement model distilled from DeepSeek-R1. Specifically, we collect 60,856 prompt pairs before and after R1 enhancement, termed as LeX-R1-60K, and fine-tune a Qwen2.5-14B model using LoRA [17] to replicate the detailed prompting capabilities of R1. This enables efficient, large-scale generation of high-quality, visually grounded prompts.

To fully leverage the enhanced prompts, we fine-tune two strong T2I models on our curated LeX-10K dataset. LeX-FLUX is based on FLUX.1 [dev] (12B) [25], finetuned with a small batch size and conservative learning rate to ensure stable, high-fidelity generation. LeXLumina, built on Lumina-Image 2.0 (2B) [35], is optimized for lightweight deployment using a larger batch size and dropout strategy. Together, LeX-FLUX and LeX-Lumina offer a flexible balance between quality and efficiency for diverse deployment scenarios. Training details are provided in Sec. 4.1.

#### 3.3. Pairwise Normalized Edit Distance (PNED)

While OCR Recall reflects whether the generated image contains text, it does not sufficiently capture the textual accuracy of Text-to-Image (T2I) models. To address this limitation, we introduce Pairwise Normalized Edit Distance (PNED)—a more flexible metric designed to assess how well the generated text matches the input prompt.

A commonly used metric, Normalized Edit Distance (NED) (Algorithm 2 in Appendix E), is effective for evaluating text accuracy in glyph-conditioned models, where the generated text typically exhibits a strict one-to-one correspondence with the prompt. However, this assumption does not hold in non-glyph-conditioned models, where the text may appear in different orders or spatial arrangements.

To accommodate such variations, PNED treats both the prompt text and the OCR-extracted text as unordered sets of words rather than fixed sequences. It computes pairwise edit distances between words using the Hungarian Algorithm [22], then aggregates the matched word scores and applies penalties for unmatched words. The full procedure is described in Algorithm 1. By leveraging PNED, we provide a more robust and generalizable metric for evaluating text accuracy across diverse T2I models.

Algorithm 1 Pairwise Normalized Edit Distance (PNED)

- 1: Input: Prompt word set X of size n, OCR result word set Y of size m
- 2: Initialize cost matrix C ∈ Rn×m, with unmatched cost penalty defined as 1
- 3: for i = 1 to n do
- 4: for j = 1 to m do
- 5: Ci,j ← NED(Xi,Yj)
- 6: end for
- 7: end for
- 8: (r,c) ← HungarianAlgorithm(C) ▷ Find optimal word assignment
- 9: M ← k Cr

k,ck ▷ Total cost of matched word pairs

- 10: U ← |n − m| × 1 ▷ Penalty for unmatched words
- 11: return (M + U) ▷ Final score

#### 3.4. LeX-Bench

Benchmark construction. To assess text rendering performance in text-to-image generation, we introduce LeXBench, a benchmark comprising 1,310 carefully designed prompts. Each prompt contains two parts: an Image Caption describing the image content, and a Text Caption specifying the text to be rendered. The combined format is: {Image Caption}, with the text on it: {Text Caption}., e.g., A picture of a blue and green abstract people logo on a purple background, with the text on it: “AREA”, “PEOPLE”.

The prompts are derived from the AnyWord-3M [43] dataset. The Image Caption originates from the original image captions annotated by Florence-2 [51], and the Text Caption is extracted using Paddle-OCR-v3 [26]. The samples used in LeX-Bench are non-overlapping with those used for LeX-10K. To ensure fluency, quality, and diversity, we filter out captions containing meaningless single letters or misspelled words, and then further refine the remaining prompts using GPT-4o for compatibility with T2I models. The exact instructions used in GPT-4o are provided in Appendix C. Finally, as shown in Fig. 6, LeX-Bench includes 1,310 prompts categorized by complexity: 630 EasyLevel (2–4 words), 480 Medium-Level (5–9 words), and 200 Hard-Level (10–14 words). The Easy-Level prompts also specify additional constraints such as text color, font, or position, detailed in Appendix D.

Evaluation formulation. We evaluate model performance through three complementary methods: OCRbased evaluation, GPT-4o-based Visual Question Answering (VQA), and human preference assessment. The OCRbased evaluation includes OCR Recall, which measures the match rate between generated and reference text, and our proposed Pairwise Normalized Edit Distance (PNED) (Section 3.3), designed to assess generation accuracy in non-

13 14

12

11

2

- 2 words
- 3 words
- 4 words

10

A picture of a group of people gathered in front of a world map, with the text on it: "Communicate" in purple, "Execute" in yellow.

[Figure 70]

[Figure 71]

[Figure 72]

Color

- 9

- 2 words
- 3 words
- 4 words

A picture of a group of people in formal attire posing together, with the text on it: "groove" in the sans-serif style; "FOUnDATiOn" in the upright font style.

[Figure 73]

- 7
- 8

Font

[Figure 74]

[Figure 75]

3

[Figure 76]

[Figure 77]

A picture of a vintage gas station sign hanging from a tree in a park, with the text on it: "CROWN" at the lower right corner of the image, "GAS" at the bottom of the image.

- 2 words
- 3 words
- 4 words

[Figure 78]

Position

4

6

5

- Figure 6. Overview of LeX-Bench. Prompts in LeX-Bench are split into three levels: 630 Easy-Level (2–4 words), 480 MediumLevel (5–9 words), and 200 Hard-Level (10–14 words). Prompts of the easy level also contain text attributes: color, font, position.

Methods LeX-Enhancer Acc. ↑ CLIPScore ↑

|ControlNet [52] TextDiffuser [5] GlyphControl [52] AnyText [43]|-<br><br>|0.5837 0.5921 0.3710 0.7239<br><br>|0.8448 0.8685 0.8847 0.8841|
|---|---|---|---|
|LeX-Lumina|✗ ✓<br><br>|0.3840 0.6220<br><br>|0.8100 0.8832|
|LeX-FLUX<br><br>|✗ ✓|0.5220 0.7110<br><br>|0.8754 0.8918<br><br>|

- Table 1. Comparison with glyph-conditioned models on AnyText-Benchmark [43]. We compare LeX-FLUX, LeXLumina and glyph-conditioned methods, i.e., ControlNet [52], TextDiffuser [5], GlyphControl [52], and AnyText [43] on AnyText-Benchmark. We observe that glyph-conditioned methods perform better in terms of text rendering accuracy, due to the incorporation of glyph information. But even without the glyph information, our methods have a compatible performance with glyph-controlled methods and have a better performance on prompt-image alignment.

glyph-conditioned T2I models. For VQA-based evaluation, we use GPT-4o to assess whether the generated image satisfies specified attributes. This yields the VQA Score, with sub-metrics including Color-Score, PositionScore, and Font-Score. Finally, we report the Preference Win Rate, which reflects human judgments on the quality and realism of the generated text images. Detailed procedures for all evaluations are provided in Appendix G.

### 4. Experiments

#### 4.1. Settings

Implementation details. We curate 10K high-resolution (1024 × 1024) text-image dataset to finetune the FLUX.1 [dev] (12B) [25] and Lumina-Image 2.0 (2B) [35]. All models are trained on 8 A100 GPUs using PyTorch’s Fully Sharded Data Parallel (FSDP) framework with bf16 precision. LeX-FLUX is finetuned with a global batch size of 8,

[Figure 79]

Figure 7. Human preference result on text accuracy, text recall rate and aesthetics for LeX-Lumina. For ease of illustration, we visualize the proportion of votes where LeX-Lumina wins, loses and ties with Lumina-Image 2.0.

a learning rate of 1e-6, classifier-free guidance (CFG) scale of 1, and 6,000 training steps, while LeX-Lumina is trained with a global batch size of 256, a learning rate of 1e-4, CFG drop rate of 0.1, and 10,000 training steps. The prompt enhancer adopts LoRA with rank = 64 and α = 32, trained for 500 steps with a global batch size of 16, a learning rate of 1e-4, and a maximum prompt length of 5,120 tokens. To support long sequences under limited resources, we enable sequence parallelism with a factor of 2.

Benchmarks and metrics. We evaluate our models, LeX-FLUX and LeX-Lumina, on several widely-used text rendering benchmarks, including SimpleBench [52], CreateBench [52], and AnyText-Benchmark [43]. These benchmarks provide diverse text prompts and assess models using OCR-based metrics such as sentence accuracy (Sen. Acc), normalized edit distance (NED) [53], and OCR-F1. In addition, they report CLIP Score [14] and Frechet Inception Distance (FID) [15] to evaluate visual-textual alignment and image quality.

#### 4.2. Comparisons

Comparison with state-of-the-art models. We compare LeX-FLUX and LeX-Lumina with glyph-conditioned models, including ControlNet [52], TextDiffuser [5], GlyphControl [52], and AnyText [43], on the AnyTextBench benchmark. As shown in Tab. 1, glyph-conditioned methods generally achieve higher text rendering accuracy by leveraging explicit glyph information. Nevertheless, even without such glyph guidance, our models achieve competitive results; for instance, LeX-FLUX reaches an accuracy score of 0.7110, which is close to AnyText’s 0.7239. Furthermore, our methods outperform glyph-conditioned baselines in prompt-image alignment, as indicated by CLIPScore [14]—LeX-FLUX achieves a score of 0.8918, surpassing AnyText’s 0.8841.

Fig. 8 presents a qualitative comparison between LeXLumina, LeX-FLUX, and other glyph-conditioned models.

Datasets Metric FLUX.1 [dev] LeX-FLUX Lumina-Image-2.0 LeX-Lumina LeX-Enhancer - ✗ ✓ ✗ ✓ ✗ ✓ ✗ ✓

Traditional Benchmark

|SimpleBench|PNED ↓ Recall ↑ Aesthetic ↑<br><br>|0.74 0.37 0.89 0.92 3.25 3.10<br><br>|1.30 0.35 0.83 0.94 3.35 3.19<br><br>| |0.45 0.47 0.70 0.84 2.46 2.89<br><br>|2.24 0.51 0.65 0.88<br><br>3.12 3.17<br><br><br>|
|---|---|---|---|---|---|---|
|CreateBench|PNED ↓ Recall ↑ Aesthetic ↑<br><br>|2.60 1.68 0.63 0.78<br><br>3.67 3.70<br><br><br>|2.36 1.45 0.66 0.80<br><br>3.74 3.72<br><br><br>| |5.40 2.76 0.50 0.70 3.21 3.62<br><br>|1.98 1.09 0.56 0.74 3.66 3.76<br><br>|
|AnyText-1Ken<br><br>|PNED ↓ Recall ↑ Aesthetic ↑|6.22 3.70 0.51 0.70 3.23 3.50<br><br>|6.25 3.87 0.52 0.71 3.30 3.50<br><br>| |5.43 3.86 0.35 0.60 2.92 3.36<br><br>|3.66 3.38 0.38 0.62 3.45 3.56<br><br>|

Our Proposed Benchmark

|LeX-Bencheasy<br><br>|Color ↑ Position ↑ Font ↑ PNED ↓ Recall ↑ Aesthetic|42.69 41.58 28.57 31.27 49.36 67.62 1.71 1.16 0.66 0.76 3.34 3.67<br><br>|42.54 45.87 28.25 33.02 52.06 71.43 1.73 1.06 0.64 0.77 3.30 3.67<br><br>| |46.67 55.08 32.06 40.48 50.79 65.56 1.83 1.25 0.47 0.64 3.13 3.52<br><br>|54.60 57.14 25.87 36.51 53.33 68.41 1.54 1.01 0.56 0.70 3.48 3.74<br><br>|
|---|---|---|---|---|---|---|
|LeX-Benchmedium<br><br>|PNED ↓ Recall ↑ Aesthetic|5.30 3.87 0.39 0.52 3.66 3.85<br><br>|5.53 4.03 0.35 0.52 3.67 3.88<br><br>| |5.52 5.10 0.19 0.33 3.37 3.76<br><br>|4.85 4.43 0.22 0.32 3.69 3.91<br><br>|
|LeX-Benchhard<br><br>|PNED ↓ Recall ↑ Aesthetic ↑|13.38 9.49 0.19 0.30 3.60 3.96<br><br>|13.12 9.42 0.15 0.30 3.64 3.95<br><br>| |11.49 10.66 0.07 0.12 3.45 3.87<br><br>|10.27 9.87 0.07 0.08 3.63 3.91<br><br>|

- Table 2. Performance comparison of FLUX.1 [dev] [25] and LeX-FLUX; Lumina-Image 2.0 [35] and LeX-Lumina. Note that recall is computed based on NED thresholding at 0.3 to mitigate the impact of minor character errors. The color intensity in each cell indicates the relative performance, with darker green representing better performance (lower PNED or higher Recall/Aesthetic scores) and lighter green representing worse performance, with colors scaled between the best and worst values in each row.

Overall, LeX-Lumina and LeX-FLUX consistently demonstrate superior performance in text clarity, controllability, and visual aesthetics. They produce sharper and more accurate text, while existing methods like TextDiffuser [5] and AnyText [43] often yield blurry or distorted results. In terms of text attributes, both LeX models exhibit strong control over color, font style, and positioning—for instance, accurately rendering colored text in the first three examples where other models fail to even produce legible outputs. Furthermore, LeX-Lumina and LeX-FLUX generate more coherent and visually appealing compositions by effectively integrating text with complex backgrounds. Notably, LeXFLUX blends text seamlessly with a 3D microphone in the first case and enhances the last three examples with refined lighting effects. These results highlight the advantages of our models in generating high-quality, well-aligned text in challenging visual contexts.

Improvements from fine-tuning on LeX-10K. We quantitatively evaluate LeX-FLUX and LeX-Lumina on LeX-Bench, with results summarized in Tab. 2. LeXFLUX, obtained by fine-tuning FLUX.1 [dev] on LeX-

- 10K, exhibits consistent improvements in text rendering accuracy, aesthetics, and controllability of text attributes.

Specifically, its PNED on LeX-Bench decreases by 0.02 compared to FLUX.1 [dev], and its average aesthetic score increases by 0.09 across three standard benchmarks. In addition, the model achieves noticeable gains in color consistency, text positioning, and font fidelity relative to the base model. LeX-Lumina achieves even more substantial gains. Its PNED scores improve by 0.41 and 0.39 on the traditional benchmark (e.g., SimpleBench, CreateBench, and AnyText) and LeX-Bench, respectively. In terms of recall, it outperforms the baseline by 0.04 and 0.02 on the same benchmarks. Moreover, LeX-Lumina exhibits significantly enhanced controllability over text attributes compared to Lumina-Image 2.0. Overall, while both models benefit from the proposed dataset and fine-tuning, LeX-Lumina exhibits broader and more significant improvements across all metrics. We further compare the text rendering quality of Lumina-Image 2.0 [35] with LeX-Lumina (see Fig. 9), and FLUX.1 [dev] [25] with LeX-FLUX (see Fig. 10), to assess the visual improvement. Both LeX-Lumina and LeX-FLUX produce sharper, more legible text with better alignment and fewer visual artifacts. These results clearly demonstrate the effectiveness of fine-tuning on LeX-10K in enhancing text rendering quality.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

###### TextDiffuserAnyTextLeX-LuminaLeX-FLUX

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

|A picture of a rock star-themed magnet featuring a microphone, with the text on it: "Little" in white, "ROCK" in<br><br>orange, "STAR" in<br><br>brown.<br><br>|A picture of a sign on<br><br>a white fur<br><br>background, with the text on it: "BLACK" in orange, "ES" in black, "ATTER" in white.|A picture of a black and white decorative sign on a dark background, with the text on it: "Home" in cyan, "Sweet" in black,<br><br>"Stateroom" in red.<br><br>|A picture of a for-<br><br>sale sign in front of a white house, with the text on it: "HOUSE", which are 3D letters; "SALE" in the upright font style.|A picture of a festive banner hanging on a wall, with the text on it: "Trick" at the right of the image, "Treat" at the top of the image.<br><br>|
|---|---|---|---|---|

- Figure 8. Qualitative comparison between LeX-Lumina, LeX-FLUX and glyph-conditioned models. We compare our models with AnyText [43] and TextDiffuser [5] for five different prompts. We observe that our models generally achieve high fidelity, better text attribute controllability and higher aesthetics.

Human preference study. In the experiments, we evaluated the performance of LeX-Lumina against LuminaImage 2.0 [35] using aesthetic metrics and OCR-related metrics, including PNED and Recall. To further investigate whether LeX-Lumina demonstrates a clear advantage over Lumina-Image 2.0 in terms of human preference, we conducted a user study.

In this study, human annotators were presented with 40 pairs of images, where each pair consisted of outputs gener-

ated by Lumina-Image 2.0 and LeX-Lumina given the same prompt. For each pair, the annotators were asked to answer three questions: (1) Which image has a higher aesthetic quality? (2) Which image renders the text more accurately? (3) Which image renders the text more completely with respect to the prompt? For each question, annotators could choose from four options: Image 1, Image 2, Both good, or Both bad. These questions were designed to evaluate the models’ capabilities in terms of aesthetics (Aesthet-

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

LeX-Enhancer LeX-Enhancer LeX-Enhancer LeX-Enhancer✅✅❌❌ Lumina-Image2.0LeX-Lumina

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

Lumina-Image2.0LeX-Lumina

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Prompt 4: The Boulet Brothers Dragula Season Three, with text on it: "THE", "RA", "SA4GONEARAz".

Prompt 2: A picture of a birthday card featuring a colorful mermaid tail design, with the text on it: "HAPPY", "BIRTHDAY", "OUR", "LITTLE", "MERMAID".

Prompt 3: A picture of a poster symbolizing the essence of music and poetry, with the text on it: "MUSIC", "POETRY", "IS", "WITH", "ROSS".

Prompt 1: A woman wearing a black hat and sunglasses standing in front of a fence, with text

Prompt 5: A picture of a festive Labor Day design featuring stars on a wooden background, with the text on it: "HAPPY" at the right of the image, "LABOR" at

TextPrompt

on it: "XXKPO", "Dmm", "STEPHENS", "GUILTY“.

the center of the image, "DAY" at the lower left corner of the image.

- Figure 9. Qualitative comparison between Lumina-Image 2.0 [35] and LeX-Lumina. The first column shows Lumina-Image-2.0 without LeX-Enhancer using Simple Caption; the second column shows the trained LeX-Lumina without LeX-Enhancer using Simple Caption; the third column shows Lumina-Image-2.0 with LeX-Enhancer enabled; and the fourth column shows LeX-Lumina with LeXEnhancer enabled. We observe that (1) LeX-Lumina exhibits a better text rendering capability in terms of text fidelity and aesthetics; (2) LeX-Enhancer exhibits a strong capability for enhancing simple prompts.

ics), text rendering accuracy (Accuracy), and text rendering completeness (Recall).

The results of the user study are summarized in Fig. 7. As shown, LeX-Lumina achieved win rates of 40%, 26%, and 36% for aesthetic quality, text rendering accuracy, and text rendering completeness, respectively. These results demonstrate that fine-tuning Lumina-Image 2.0 on the LeX10K dataset produces LeX-Lumina, which achieves supe-

rior performance in human preference metrics compared to the original model.

#### 4.3. Ablations

Effectiveness of Lex-Enhancer. In Section 3.1, we demonstrated that prompts enhanced using DeepSeek-R1 significantly improve the aesthetic quality and text accuracy of generated images. To further evaluate the prompt

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

LeX-Enhancer❌ LeX-EnhancerTextPrompt✅ LeX-Enhancer✅ LeX-Enhancer❌

FLUX.1[dev]LeX-FLUXFLUX.1[dev]LeX-FLUX

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

|Prompt 1: A picture of a justice scale placed on a table, with the text on it: "CRIMINAL" in orange, "JUSTICE" in red.|Prompt 2: A picture of a lighthouse surrounded by<br><br>a snowy landscape, with the text on it: "UNDER" in yellow, "MILK" in gray, "WOOD" in red.|Prompt 3: A picture of a French flag-themed stamp indicating its origin, with the text on it: "IN" in purple, "MADE" in blue, "NI" in cyan.|Prompt 4: A picture of a blue and red grand opening design with stars, with the text on it: "BLOWOUT" at the right of the image, "SALE" at the right of the image, "OPENING" at the center of the image, "GRAND" at the upper left corner of the image.|Prompt 5: A picture of a soccer ball on a grassy field, with the text on it: "FOOTBALL", which are 3D letters; "CHAIRMAN" in the upright font style; "PRO" in the cursive font style.|
|---|---|---|---|---|

- Figure 10. Qualitative comparison between FLUX.1 [dev] [25] and LeX-FLUX. The first column shows FLUX.1 [dev] without LeXEnhancer using Simple Caption; the second column shows the trained LeX-FLUX without LeX-Enhancer using Simple Caption; the third column shows FLUX.1 [dev] with LeX-Enhancer enabled; and the fourth column shows LeX-FLUX with LeX-Enhancer enabled. We observe that (1) LeX-FLUX exhibits a better text rendering capability in terms of text fidelity and text attributes controllability; (2) LeXEnhancer exhibits a strong capability for enhancing simple prompts.

enhancement capability of our trained LeX-Enhancer, we conducted a series of experiments across multiple benchmarks. Specifically, we evaluated prompts enhanced by LeX-Enhancer on three benchmarks—SimpleBench [52], AnyText-Benchmark [43], and LeX-Bench—using four models: FLUX.1 [dev], LeX-FLUX, Lumina-Image 2.0, and LeX-Lumina, both with and without LeX-Enhancer.

Detailed results are provided in Tab. 2. Across all four models and six benchmarks, prompts enhanced by LeXEnhancer consistently led to significant improvements in text rendering accuracy, aesthetic quality, and controllability of text attributes. Specifically, FLUX.1 [dev] achieved a 1.43-point gain in PNED on LeX-Bench-medium, LeXFLUX saw a 19.37% increase in text style scores on LeX-

Data Scale Metric LeX-Lumina LeX-Enhancer - ✗ ✓

|0|PNED ↓ Recall ↑ Aesthetic ↑<br><br>|6.13 5.54 0.23 0.33 3.29 3.70<br><br>|
|---|---|---|
|1K<br><br>|PNED ↓ Recall ↑ Aesthetic ↑|5.68 5.21 0.24 0.33 3.62 3.86<br><br>|
|5K<br><br>|PNED ↓ Recall ↑ Aesthetic ↑<br><br>|5.79 5.23 0.25 0.33 3.45 3.79|
|10K|PNED ↓ Recall ↑ Aesthetic ↑<br><br>|5.60 5.15 0.27 0.35 3.56 3.83<br><br>|

- Table 3. Performance comparison on LeX-Bench with different training data scales. When scaling the training data, we observe the improving performance in terms of text rendering fidelity and aesthetics.

Bench, Lumina-Image 2.0 obtained a 20% improvement in Recall on CreateBench, and LeX-Lumina achieved a

- 9.64% boost in text position scores on LeX-Bench. In addition, comparisons with glyph-conditioned models reveal that LeX-Enhancer significantly improves both the Acc. scores [43] and CLIPScore [14] of LeX-FLUX and LeXLumina, as shown in Tab. 1. Fig. 9 and Fig. 10 spresent qualitative comparisons of images generated by LuminaImage 2.0 and LeX-Lumina, as well as FLUX.1 [dev] and LeX-FLUX, with and without LeX-Enhancer, further illustrating the effectiveness of our method. These results underscore the critical role of prompt quality in visual text generation and highlight the value of prompt enhancement for training effective text-to-image models.

Study on training data scale As shown in Tab. 3, we conduct an ablation study on Lumina-Image 2 by training it with 1K, 5K, and 10K text-image pairs to evaluate the impact of data scale on text rendering quality. We observe clear improvements in text accuracy, layout alignment, and aesthetic quality as the size of the dataset increases. Smaller datasets tend to cause more distortions and inconsistencies, while larger datasets lead to better fidelity and spatial consistency. Based on these findings, we fine-tune our models on the larger 10K dataset. In addition to scaling data, we find that LeX-Enhancer consistently improves performance when integrated with LeX-Lumina. Furthermore, with the proposed scalable text-image data construction method, we expect to obtain more high-quality samples to further boost model performance.

### 5. Conclusion

We present LeX-Art, a data-centric framework that systematically improves the text rendering capability of textto-image models. By leveraging DeepSeek-R1, we construct LeX-10K, a high-quality dataset of 10K refined textimage pairs through enriched prompts and multi-stage filtering. Building on this, we develop LeX-Enhancer, a lightweight prompt enrichment model trained on 60K enhanced prompts, and fine-tune two models—LeX-FLUX and LeX-Lumina—to achieve strong performance in text fidelity, layout, and aesthetics. To evaluate visual text generation, we introduce LeX-Bench, a benchmark covering fidelity, aesthetics, and attribute control, along with PNED, a robust metric for OCR-based text accuracy. Extensive results show that LeX-Art offers a scalable and effective solution for high-quality visual text generation, bridging the gap between prompt expressiveness and rendering precision.

### References

- [1] Jinwon An and Sungzoon Cho. Variational autoencoder based anomaly detection using reconstruction probability. Special lecture on IE, 2(1):1–18, 2015. 2
- [2] Anthropic. Claude 3.5 haiku. https : / / www . anthropic.com/claude/haiku, 2024. 5
- [3] Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, Lluis Castrejon, Kelvin Chan, Yichang Chen, Sander Dieleman, Yuqing Du, et al. Imagen

3. arXiv preprint arXiv:2408.07009, 2024. 2

- [4] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018. 2
- [5] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser: Diffusion models as text painters. Advances in Neural Information Processing Systems, 36:9353–9387, 2023. 2, 3, 7, 8, 9, 16, 17
- [6] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser-2: Unleashing the power of language models for text rendering. In European Conference on Computer Vision, pages 386–402. Springer, 2024. 2, 3
- [7] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling, 2025. 2
- [8] Google DeepMind. Gemini-2.0-flash. https : //deepmind.google/technologies/gemini/ flash/, 2025. 5
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 3

- [10] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi,

- Shiqi Sun, Yu Tian, Zhi Tian, Peng Wang, Xun Wang, Ye Wang, Guofeng Wu, Jie Wu, Xin Xia, Xuefeng Xiao, Linjie Yang, Zhonghua Zhai, Xinyu Zhang, Qi Zhang, Yuwei Zhang, Shijia Zhao, Jianchao Yang, and Weilin Huang. Seedream 2.0: A native chinese-english bilingual image generation foundation model, 2025. 2, 3
- [11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2
- [12] Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, et al. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. Advances in Neural Information Processing Systems, 35:26418–26431,

2022. 2

- [13] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2, 5, 15
- [14] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718,

2021. 7, 12

- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 6
- [18] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 5, 15
- [19] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134,

2017. 2

- [20] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2
- [21] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013. 2
- [22] H. W. Kuhn. The hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1-2):83–97,

1955. 6

- [23] Matt J Kusner, Brooks Paige, and Jos´e Miguel Hern´andezLobato. Grammar variational autoencoder. In International

conference on machine learning, pages 1945–1954. PMLR,

2017. 2

- [24] Black Forest Labs. Flux.1.1. https : / / blackforestlabs.ai/1-1-pro/, 2024. 3
- [25] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 3, 4, 5, 6, 7, 8, 11, 15
- [26] Chenxia Li, Weiwei Liu, Ruoyu Guo, Xiaoting Yin, Kaitao Jiang, Yongkun Du, Yuning Du, Lingfeng Zhu, Baohua Lai, Xiaoguang Hu, et al. Pp-ocrv3: More attempts for the improvement of ultra lightweight ocr system. arXiv preprint arXiv:2206.03001, 2022. 2, 5, 6
- [27] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024. 3
- [28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2
- [29] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 3
- [30] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2
- [31] Zeyu Liu, Weicong Liang, Zhanhao Liang, Chong Luo, Ji Li, Gao Huang, and Yuhui Yuan. Glyph-byt5: A customized text encoder for accurate visual text rendering. In European Conference on Computer Vision, pages 361–377. Springer,

2024. 2, 3

- [32] Zeyu Liu, Weicong Liang, Yiming Zhao, Bohan Chen, Lin Liang, Lijuan Wang, Ji Li, and Yuhui Yuan. Glyph-byt5-v2: A strong aesthetic baseline for accurate multilingual visual text rendering. arXiv preprint arXiv:2406.10208, 2024. 2
- [33] Jian Ma, Mingjun Zhao, Chen Chen, Ruichen Wang, Di Niu, Haonan Lu, and Xiaodong Lin. Glyphdraw: Seamlessly rendering text with intricate spatial structures in text-to-image generation. arXiv preprint arXiv:2303.17870, 2023. 3
- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [35] Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Xinyue Li, Dongyang Liu, Xiangyang Zhu, Will Beddow, Erwann Millon, Wenhai Wang Victor Perez, Yu Qiao, Bo Zhang, Xiaohong Liu, Hongsheng Li, Chang Xu, and Peng Gao. Lumina-image 2.0: A unified and efficient image generative framework. https://github.com/ Alpha-VLLM/Lumina-Image-2.0, 2025. 6, 7, 8, 9, 10
- [36] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct

- preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023. 3
- [37] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019. 2
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [39] Runway.Inc. Gen3-alpha prompt guide. https : //help.runwayml.com/hc/en-us/articles/ 23996932381203 - How - to - prompt - within Runway, 2024. 15
- [40] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 2
- [41] Ideogram Team. Ideogram. https://ideogram.ai/ t/explore, 2024. 5
- [42] Kolors Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint,

2024. 3

- [43] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. arXiv preprint arXiv:2311.03054, 2023. 2, 3, 5, 6, 7, 8, 9, 11, 12, 16, 17
- [44] Yuxiang Tuo, Yifeng Geng, and Liefeng Bo. Anytext2: Visual text generation and editing with customizable attributes. arXiv preprint arXiv:2411.15245, 2024. 2, 3
- [45] Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder. Advances in neural information processing systems, 33:19667–19679, 2020. 2
- [46] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2
- [47] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024. 3
- [48] Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, and Chen Change Loy. Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European conference on computer vision (ECCV) workshops, pages 0–0, 2018. 2
- [49] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171,

2022. 5

- [50] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for

- visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023. 2, 5
- [51] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818– 4829, 2024. 6, 15
- [52] Yukang Yang, Dongnan Gui, Yuhui Yuan, Weicong Liang, Haisong Ding, Han Hu, and Kai Chen. Glyphcontrol: glyph conditional control for visual text generation. Advances in Neural Information Processing Systems, 36:44050–44066,

2023. 2, 3, 7, 11

- [53] Li Yujian and Liu Bo. A normalized levenshtein distance metric. IEEE transactions on pattern analysis and machine intelligence, 29(6):1091–1095, 2007. 2, 7
- [54] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 2
- [55] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223– 2232, 2017. 2

Appendix

- A. Instruction Used in Prompt Enhancer

To enhance short captions into those suitable for generating high-quality images, it is crucial to first understand the characteristics of an effective caption. Based on insights from the community [39], a good caption should possess the following attributes: it should be detailed, well-structured, rich in knowledge, and written in a descriptive tone. To achieve this, we devised the instruction shown below to transform concise captions into more comprehensive and informative ones.

Prompt Enhancer Template Simple Caption: <Simple Caption>

Above is the simple caption of an image with text. Please deduce the detailed description of the image based on this simple caption. Note:

- 1.The description should only include visual elements and should not contain any extended meanings.
- 2.The visual elements should be as rich as possible, such as the main objects in the image, their respective attributes, the spatial relationships between the objects, lighting and shadows, color style, any text in the image and its style, etc.
- 3.The output description should be a single paragraph and should not be structured.
- 4.The description should avoid certain situations, such as pure white or black backgrounds, blurry text, excessive rendering of text, or harsh visual styles.
- 5.The detailed caption should be human-readable and fluent.
- 6.Avoid using vague expressions such as “may be” or “might be”; the generated caption must be in a definitive, narrative tone.
- 7.Do not use negative sentence structures, such as “there is nothing in the image,” etc. The entire caption should directly describe the content of the image.
- 8.The entire output should be limited to 200 words.

- B. Instruction Used in Knowledge-Augmented Recaption

As mentioned in the paper, the captions enhanced by DeepSeek-R1 [13] contain rich knowledge, such as font styles, layout configurations, color schemes, and more. However, due to the inherent limitations of FLUX.1 [dev] [25], the images generated using these prompts do not always align perfectly with the prompts themselves. To address this issue, we designed a knowledge-augmented recaptioning step to further refine the prompts, ensuring they

are fully aligned with the image content while preserving the rich knowledge introduced by DeepSeek-R1 [13].

###### Knowledge-Augmented Recaption Template

Image: <Image> Original Caption: <Original Caption> Instruction: This is the caption of an image with text rendered on and the corresponding image. There might be some artifacts in the image. For example, some of the texts were not be rendered correctly in the generated image. I need you to refer to the provided caption and corresponding generated image, refine the caption based on the generated image. Note:

- 1.The refined caption should fully describe the generated image.
- 2.In the refinded caption, the misalignment of the original caption and the generated image should be fixed and the other visual details should be keeped.
- 3.Directly output the refined caption.
- 4.The output description should be a single paragraph and should not be structured.
- 5.The entire output should be limited to 200 words.

### C. Instruction Used in Prompt Refinement (LeX-Bench Curation)

Since Florence-2 [51] may include textual content from the image in its descriptions or occasionally produce outputs that lack fluency, we leveraged GPT-4o [18] to refine and adjust the image descriptions generated by Florence-2.

###### Prompt Refinement Template

Below is the caption of an image along with the text I provided. Please revise this caption, ensuring that the revised caption does not include the text I provided while maintaining the original meaning as much as possible. Note:

- 1.The refined caption should be kept brief and concise, and it should describe an image containing no text.
- 2.Directly give me the refined caption.
- 3.Maybe the refined caption could start with “An image of...” or “A picture of...”.
- 4.Remember the provided text must not be included in the refined caption.
- 5.The refined caption should be fluent.
- 6.Most importantly: the refined caption must not contain any text to be rendered on the image. Simple Caption: <Simple Caption > Text: <OCR Results >

### D. The Format of Text Conditions

We design three conditions for text presentation, focusing on color, font, and position. This section provides a detailed description of these three conditions. It is important to note that the evaluation of color and font is conducted using GPT-4o, and a series of tests are performed to ensure that GPT-4o can generate accurate and reliable results. Below, we elaborate on the specific design of each condition.

Color. For the color condition, we select a set of 12 colors as condition. The color condition and the format of caption are as follows:

###### Color Condition List

colors = [“red”, “blue”, “green”, “yellow”, “orange”, “purple”, “pink”, “brown”, “black”, “white”, “gray”, “cyan”].

Prompt Format with Color Condition {Image Caption}, with the text on it: {Text Caption} in {color}; {Text Caption}in {color} ....

This allows for a systematic exploration of how different colors interact with fonts and other design elements.

Font. In the font condition, we employ five pairs of contrasting font styles to ensure diversity and clarity. To maintain design consistency, opposing font styles from the same pair are not presented within the same image. For instance, “3D style” and “flat style” are never displayed together, as their will complicate the visual coherence of the image. These pairs and format are as follows:

###### Font Condition List

fonts = [ [“cursive style”, “block style”], [“3D style”, “flat style”], [“sans-serif”, “serif”], [“upright”, “slant”], [“rounded”, “angular”] ]

###### Prompt Format with Font Condition

{Image Caption}, with the text on it: {Text Caption}, {font description}; {Text Caption}, {font description}; ....

The detailed {font description} are as follows: “cursive style”: in the cursive font style “block style”: in the block font style “3D style”: which are 3D letters “flat style”: which sits flat

“sans-serif”: in the sans-serif style “serif”: in the serif style “upright”: in the upright font style “slant”: in the slant font style “rounded”: in the rounded font style “angular”: in the angular font style

Position. Finally, for the position condition, we select a variety of spatial placements for the text. These conditions and format are as follows:

###### Position Condition List

positions = [“top”, “bottom”, “left”, “right”, “upper left corner”, “lower left corner”, “upper right corner”, “lower right corner”, “center”].

###### Prompt Format with Position Condition

{Image Caption}, with the text on it: {Text Caption}, at the {position} of the image; {Text Caption}, at the {position} of the image....

This systematic approach ensures that the text placement is both diverse and well-documented for further analysis.

### E. Normalized Edit Distance Algorithm

Edit Distance (ED), also known as Levenshtein Distance, measures the minimum number of operations required to transform one string into another. The allowed operations include character insertions, deletions, and substitutions, each contributing a unit cost to the total edit distance. This metric is widely used for text similarity evaluation, particularly in tasks such as optical character recognition (OCR), spelling correction, and machine translation.

To improve interpretability and enable comparisons across different string lengths, Normalized Edit Distance (NED) is introduced. NED normalizes the computed edit distance by dividing it by the maximum length of the two strings, ensuring a value between 0 and 1, where 0 indicates identical strings and 1 represents completely different strings. The formal computation of NED is shown in Algorithm 2, where a two dimensional dynamic programming (DP) table is used to iteratively compute the minimum edit cost between two input strings.

### F. Data Samples from LeX-10K

In Fig. 11, we show the comparison of data samples from AnyWord-3M [43], MARIO-10M [5], and LeX-10K. It is obvious that images from LeX-10K are better, regarding aesthetics and diversity.

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

###### AnyText-3MMARIO-10MLeX-10K

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

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Figure 11. Comparison of data samples from AnyWord-3M [43], MAION-10M [5] and LeX-10K.

Algorithm 2 Normalized Edit Distance (NED)

- 1: Given two strings a and b of lengths n and m. Initialize DP table D ∈ R(n+1)×(m+1). Define edit cost c. Initialize Di,0 ← i, D0,j ← j for all i,j.
- 2: for i = 1 to n do
- 3: for j = 1 to m do
- 4: Di,j ← min(Di−1,j+1,Di,j−1+1,Di−1,j−1+ I(ai ̸= bj))
- 5: end for
- 6: end for
- 7: return Dn,m/max(n,m)

### G. The Evaluation of Text Attributes

The evaluation of color, font, and position conditions is conducted using OCR and VQA. Specifically, OCR is first employed to detect all textual elements within the image. The detected words are then split by spaces and subjected to fuzzy matching against the ground truth (GT) texts provided in the prompt. After matching, it is checked whether the GT condition for each word corresponds to how the text is actually presented in the image.

Color Score. Once the detected words are matched with the GT, the bounding box of the matched text is cropped from the image. To ensure accuracy, the cropped area is slightly larger than the detected bounding box. The cropped region is then passed to GPT-4o for verification using a structured query format:

###### VQA Format of Color Attribute

The text “{text}” is in the color of {color}? Answer me using “yes” or “no”.

Here, “text” represents the word detected by OCR, and “color” corresponds to the color condition specified in the prompt.

Font Score. The matched text is cropped and passed to GPT-4o similarly. However, unlike the color evaluation, the query format is designed to accommodate font pair comparisons:

###### VQA Format of Font Attribute

The text “{text}” is {font A} or {font B}? Answer me using either “{font A}” or “{font B}” only.

Here, “{text}” is the OCR-detected word, “{font A}” represents the font specified in the prompt as the GT condition, and “{font B}” is the corresponding paired font associated with “font A”.

Position Score. The coordinate ranges corresponding to the nine predefined position conditions are set in advance.

The bounding box of the OCR-detected text is compared against the coordinate range corresponding to the matched GT condition. If the bounding box falls within the specified range, the condition is considered satisfied.

### H. More Generated Samples of LeX-FLUX and LeX-Lumina

In Fig. 12, we show more generated images of LeX-FLUX and LeX-Lumina.

### I. Justification of Pairwise Normalized Edit Distance (PNED) as a Metric

To validate the effectiveness of PNED as a reliable metric for evaluating text rendering performance, we conduct a systematic empirical study through controlled perturbation experiments. Our objective is to demonstrate that PNED effectively captures the degree of deviation between detected text sequences and ground truth, while being robust to sequence order variations.

#### I.1. Experimental Design

We design a controlled experiment where we can precisely manipulate the degree of text perturbation and analyze how PNED responds to these changes. The experiment consists of the following key components:

- 1) Dataset Generation: We synthesize a dataset of N =

100 samples, where each sample contains a list of strings with lengths randomly distributed between 1 and 20. Each string comprises lowercase ASCII characters with lengths randomly distributed between 3 and 8 characters.

- 2) Perturbation Mechanism: We implement a stochas-

tic perturbation algorithm that accepts a parameter α ∈ [0,1] controlling the perturbation intensity. For each string, the algorithm randomly applies one of six operations with probability α: - Character insertion - Character deletion Character replacement - String splitting into two substrings - String addition (adding a new random string) - String deletion (removing an existing string)

- 3) Evaluation Protocol: We treat the original dataset as

ground truth and the perturbed dataset as OCR predictions. We compute PNED scores between corresponding pairs under two conditions: - Ordered: maintaining the original sequence order - Shuffled: randomly permuting the sequence order to simulate OCR outputs with uncertain ordering

#### I.2. Results and Analysis

Fig. 13 illustrates the relationship between the perturbation intensity α and the resulting PNED scores. Several key observations support the validity of PNED as a metric:

1) Monotonicity: PNED scores demonstrate a consistent monotonic increase with respect to α, confirming that the metric effectively captures the degree of text degradation.

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

##### LeX-LuminaLeX-FLUX

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

Figure 12. Showcase of text rendering results from LeX-Lumina (first two rows) and LeX-FLUX (last two rows) on text-to-image tasks. The examples demonstrate the models’ ability to generate clear, well-aligned, and aesthetically pleasing text within images.

- 2) Order Invariance: The minimal difference between

ordered and shuffled conditions (solid blue vs. dashed red lines) validates that PNED, through its optimal matching mechanism, successfully handles sequence order variations - a crucial property for OCR evaluation.

- 3) Unbounded Range: Unlike normalized metrics,

PNED values are not constrained to [0,1]. Instead, they represent the raw distance between sequences, making it suitable for scenarios where absolute deviation is more meaningful than relative similarity.

- 4) Sensitivity: The metric shows appropriate sensitiv-

ity to perturbations, with distinguishable changes in PNED scores across different α values, indicating its effectiveness in discriminating varying degrees of OCR errors.

These empirical results demonstrate that PNED satisfies key desirable properties of an evaluation metric: monotonicity, order invariance, unboundedness, and appropri-

ate sensitivity. Furthermore, its ability to handle variablelength sequences and account for unmatched elements makes it particularly suitable for real-world OCR evaluation scenarios where predictions may contain splits, merges, or missing characters.

[Figure 198]

Figure 13. Validation of PNED metric through controlled perturbation experiments. The solid blue line represents PNED scores with maintained sequence order, while the dashed red line shows scores with shuffled sequences.

### J. Self-Enhancement and Knowledge Distillation

We observe a self-enhancement effect in FLUX.1 [dev] when fine-tuned on high-quality self-distilled data. While the direct performance gain from self-training is subtle, the impact is significantly amplified when the data is used for knowledge distillation. Specifically, using FLUX.1 [dev] as a teacher model for Lumina-Image 2.0 leads to a substantial improvement in the student model’s text rendering accuracy, layout coherence, and aesthetic quality. Meanwhile, the teacher itself, FLUX.1 [dev] can be self-enhanced by the self-distilled data.

This observation highlights the scalability of our approach: as the teacher model continues to improve through iterative refinement on better-curated data, the performance gains cascade down to student models more effectively. This suggests that even small advancements in a strong base model can translate into major enhancements for smaller, more efficient models, making this distillation process a promising direction for scaling text-to-image models with improved text generation capabilities.

