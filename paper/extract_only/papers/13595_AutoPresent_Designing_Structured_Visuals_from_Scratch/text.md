# AUTOPRESENT: Designing Structured Visuals from Scratch

Jiaxin Ge1* Zora Zhiruo Wang2* Xuhui Zhou2 Yi-Hao Peng2 Sanjay Subramanian1 Qinyue Tan2 Maarten Sap2 Alane Suhr1† Daniel Fried2† Graham Neubig2† Trevor Darrell1†

1University of California, Berkeley 2Carnegie Mellon University

arXiv:2501.00912v2[cs.CV]19Jun2025

|[Figure 1]<br><br>[Figure 2]<br><br>|[Figure 3]|
|---|
|
|---|

search _image

[Figure 4]

Input: Present Airbnb's misión with a mission statement and a relevant image, for English and Spanish audience.

add_text

set_background_color generate_image add_title

import pptx presentation = pptx.Presentation() slide = presentation.slides.add_slide()

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

add_title

add_title(‘Misión’, font_size=42) path = search_image(‘Logo of Airbnb’) add_image(path, coords=(2.5, 2, 6, 4)) add_text(“Create a world where anyone..”) set_background_color(rgb=(266,165,0))

take_ snapshot

... ... ... ...

add_text

[Figure 9]

add_title

| | |
|---|---|
| | |

presentation.save("business.pptx")

add_shape (box)

search_image

add_shape (arrow)

- Figure 1. Automatically generating slides from natural language instructions. We propose AUTOPRESENT, a tool-augmented code generation method that follows natural language instructions to design slides from scratch, as shown in the examples. This allows for precise control over all elements, including textual content, images, visual layouts, coloring, and more.

## Abstract

AUTOPRESENT, an 8B LLAMA-based model trained on 7k pairs of instructions paired with code for slide generation, and achieve results comparable to the closed-source model GPT-4O. We further explore iterative design refinement where the model is tasked to self-refine its own output, and we found that this process improves the slide’s quality. We hope that our work will provide a basis for future work on generating structured visuals. Our code, data, demo, and video demonstrations are publicly available at https: //github.com/para-lost/AutoPresent

Designing structured visuals such as presentation slides is essential for communicative needs, necessitating both content creation and visual planning skills. In this work, we tackle the challenge of automated slide generation, where models produce slide presentations from natural language (NL) instructions. We first introduce the SLIDESBENCH benchmark, the first benchmark for slide generation with 7k training and 585 testing examples derived from 310 slide decks across 10 domains. SLIDESBENCH supports evaluations that are (i) reference-based to measure similarity to a target slide, and (ii) reference-free to measure the design quality of generated slides alone. We benchmark endto-end image generation and program generation methods with a variety of models, and find that programmatic methods produce higher-quality slides in user-interactable formats. Built on the success of program generation, we create

## 1. Introduction

Designing structured visuals such as presentation slides from scratch is an essential skill for effective communication and conveying complex ideas [30]. Among various forms of visual communication, creating a compelling set of slides is a challenging problem, requiring content creation (text, pictures, diagrams, and more) and visual planning skills, to ensure the slides are well designed [25] and convey insights with clarity [3, 36]. Even human experts

*Equal Contribution. †Equal Contribution.

may need to spend hours iterating and polishing their slide decks [10] to produce high-quality designs with clear insights. While digital agents have demonstrated impressive capabilities in tasks such as software engineering [49], web navigation [46, 54], and free-form image design generation [8, 32], their creative capabilities in generating semistructured communicative media like slide decks has not been extensively tested. Therefore, we ask: Can we employ powerful AI agents to create high-quality presentation slides that are well-structured and insight-revealing?

In this work, we formulate the natural language (NL) to slide generation task. At a high level, the user provides the system with a natural language instruction about the desired slide, and the system then generates an editable presentation, as shown in Figure 1. We consider three types of user instructions: (1) detailed instruction with images. (2) detailed instructions only. (3) high-level instructions, reflecting varying levels of design freedom.

Since there are no existing tools for quantifying agent performance in slide generation tasks, we propose the SLIDESBENCH benchmark (§2) as a training source and test bed for method comparisons. SLIDESBENCH contains 7k training examples and 585 testing examples of varied instruction difficulties, constructed from 310 publicly available slide decks from 10 different domains, including art, business, and technology. To evaluate generated slides, we introduce two sets of evaluation metrics: referencebased metrics to examine position, content, and color match against the reference slide; and reference-free metrics inspired by slide design principles [5, 9, 34, 39, 44] to measure the design quality of agent-created slides alone, given that many good designs for the same instructions may vary from the reference slide.

To enable controlled and structured slide generation, we propose to create slides using program generation, where a model first generates a program from the natural language instruction, and then the program is executed to get the slide. We apply this approach to large language models (LLMs; LLAMA [11], DeepseekCoder [16], CodeLlama [33], GPT-4O [2]) and vision-language models (VLMs; LLAVA [51]). As illustrated in Figure 1, given a natural language instruction, the model first generates a Python program and then executes it to obtain a PPTX slide. We find that small models such as LLAMA (8B) and LLAVA (7B) are often unable to produce executable code. While GPT-4o can produce reasonable slides, it still exhibits a substantial gap in design quality compared to human-generated slides (§5). By further conducting iterative refinement, we find that models can self-refine and further improve slide quality. We also find that code generation approaches substantially outperform end-to-end image generation methods (Stable Diffusion [32], Dall-E [8]).

To further enhance the current model’s ability to gener-

ate high-quality slides, we present our open-sourced AUTOPRESENT (8B) model (§4.2) which is fine-tuned from LLAMA 8B on the SLIDESBENCH training set. AUTOPRESENT achieves state-of-the-art performance among small open-sourced models and approaches the performance of the closed-sourced model GPT-4o. Since directly generating a long program is difficult for current models [14], we further create the SLIDESLIB library to simplify the program generation process. SLIDESLIB contains high-level functions that are basic such as add title, and imagerelated such as search image and generate image. We show that LLMs and VLMs generally perform better when given access to SLIDESLIB.

Our main contributions can be summarized as follows:

- • We formulate the NL-to-slide generation task and build SLIDESBENCH, the first benchmark for slide generation, which contains 7k training and 585 test examples and supports automatic evaluations.
- • We leverage NL-to-program generation methods with refinement to produce high-quality slides, and benchmark diffusion models, VLMs, and LLMs.
- • We train an 8B parameter open-source LLM, AUTOPRESENT, that approaches the performance of GPT-4o, and design a programmatic tool library SLIDESLIB that facilitates slide program generation across models.

## 2. SLIDESBENCH

In this section, we describe the creation of the SLIDESBENCH benchmark. Each instance consists of a natural language instruction to create a slide, and the slide itself (in PPTX format) as a reference. SLIDESBENCH includes three scenarios of varying difficulty levels designed to evaluate models with different user input. We describe the slide data collection (§2.1), three task setups (§2.2), and the annotation process (§2.3).

### 2.1. Slides Data Collection

We search the web and collect presentation slide decks from 10 domains, including art, marketing, environment, technology, etc. To select the highest-quality slide decks from each domain, we manually go through the relevant slide decks and conduct initial processing, by checking if all its slides (i) have visually structured layouts, and (ii) extractable media such as images (if any). For the slide decks with all slides satisfying (i) and (ii) in each domain, we incorporate one slide deck into the test set, and others into the training set. This results in a total of 10 and 300 slide decks (in PPTX format) for testing and training, each containing 20 slides on average. To respect the rights of the slide creators, we do not redistribute the slides. Instead, we provide a list of URLs for the slides that we used so that others can download the slides directly from the original website. We also provide an opt-out mechanism for any creator who does

[Figure 10]

Detailed Instructions w/ Images

Create a slide with the following elements:

- 1. Background Color: Use a solid coral color … …

Detailed Instructions Only

[Figure 11]

media/im age_0.jpg

[Figure 12]

media/image_1. jpg Agent

[Figure 13]

[Figure 14]

(baseline) viaimagegeneration

via code generation

import pptx prs = pptx.Presentation(...)

Create a slide with the following elements:

- 1. Background Color: Use a solid coral color for the entire slide.
- 2. Logo: Place the Airbnb icon on the entire left side of the slide. Place the text "airbnb" logo on the right side of the slide, centered vertically.
- 3. Title: Add the text "Business Case" in a prominent font, centered vertically below the text logo.
- 4. Below the title, list the names: Daniel Consuegra, Alejandra Del …

Input Instruction

Output Slide

Reference Slide

reference free

text: 5.0 image: 4.0

color:2.0 layout: 4.0

[Figure 15]

content: 95 color: 20 position: 63

element match: 45

Slide Generation

exec

Slide Evaluation

reference based

Figure 2. Illustration of SLIDESBENCH. Each example of SLIDESBENCH consists of three instructions: Detailed Instructions with Images, Detailed Instructions Only, and High-Level Instructions. The model is tasked to generate a slide based on the instruction, and the generated slide is evaluated on the metrics suite, which contains both the reference-free metrics and the reference-based metrics.

not want their slides in the dataset. We provide implementation details in §A.

- 2.2. Three Task Setups

High-Level Instructions

Create a title slide for the Airbnb business case, featuring the logo, title, and names of the presenters on a vibrant background.

### 2.3. Example Annotation

To annotate the dataset, we collect natural language instructions paired with each slide. For each slide, we create three versions corresponding to the three setups in §2.2.

Detailed Instructions with Images To produce detailed instruction with images, we use a scalable approach combining human-written examples and model-generated annotations. For each slide deck, we first write instructions for three example slides manually — including all necessary information (content, layout, formatting) to reproduce the slide, and providing paths to the images used in the slide (e.g., media/image 0.png), as shown in Figure 2 (top). We then use these (human-written instruction, reference slide) pairs as few-shot examples to prompt LLM (specifically, gpt-4o-mini) to generate natural language instructions for each slide in the current slide deck.* Further, for the test set, we manually examined and refined the instructions by correcting incorrect specifications, adding missing content, and removing unnecessary or untrue content.

We formulate the task as an NL-to-slide generation process. Given the reference slide, we curate three versions of natural language instructions, as shown in Figure 2, to represent slide generation tasks under varied difficulty levels. We introduce each setup below.

Detailed Instructions with Images The first and easiest setting is to provide the models with all the necessary information and assets to produce the reference slide, including text and image content, formatting and layout specifications. This setting evaluates models’ visual planning abilities, such as arranging spatial layouts, maintaining formatting consistency, balancing content proportions, and emphasizing key elements.

Detailed Instructions Only Since a user may not specify, or know exactly what images to put on a slide, we propose a detailed instruction only setting, where we provide the same natural language instruction provided in the detailed instruction with images setting, but replace the provided images with their natural language descriptions (e.g., “two people shaking hands”) generated by gpt-4o-mini. We then instruct the models to obtain the images using image searching or image generation tools. This setting further challenges models to interpret complex or compositional descriptions of images and obtain visuals that align with the slide context.

Detailed Instructions Only To produce detailed instruction only, we replace the image paths (e.g., media/image 0.png) with the natural language descriptions of the images(‘‘an artistic, colorful background’’). These descriptions are generated by gpt-4o-mini. For the test set, we manually refine the instructions to ensure that they do not refer to unavailable image paths (e.g., removing phrases like “use the provided images”), as shown in Figure 2 (middle).

High-Level Instructions To create high-level instructions, we start with a similar approach by manually annotating three examples and then prompting the model to generate for all slides. Human-written instructions only provide a topical description of the slide and intentionally leave out specific content or layout details. This process ensures that the generated instructions remain concise and general, as shown in Figure 2 (bottom).

High-Level Instructions In contrast to users who have a concrete target slide in mind and can spell out all detailed instructions, some users may only be able to express their needs on a high level. We thus devise a high-level instruction setting, where the natural language instructions are rather high-level and only provide a general topical idea of the slide, such as “create a title slide for Airbnb,” instead of detailing what logos and text to add and where, as exemplified in Figure 2. Models in this case need to both acquire or create content, and arrange the elements properly.

*Including the three slides with human-written instructions, to ensure instructions for all slides are consistent in style and specificity.

Overall, the instructions have an average of 115.6, 118.3, and 26.6 words under detailed instruction with images, detailed instruction only, and high-level instructions settings respectively, accompanied by an average of 1.1, 0.0, and 0.0 provided images.

## 3. Evaluation Metrics

In this section, we describe the evaluation metrics that we designed for SLIDESBENCH. We propose two sets of evaluation metrics: reference-based metrics for measuring models’ instruction-following abilities (§3.1), and reference-free metrics to examine the design quality of model-generated content (§3.2). We also use executability to examine the success rate of each model (§3.3).

### 3.1. Reference-Based Metrics

Inspired by Design2Code [38] metrics, we implement four dimensions to examine the similarity between the modelproduced slides and the reference slide.

Element matching For the slide layout, we measure the total sizes of matched elements (in generated and reference slides) divided by the total sizes of all element, where each textbox, image, or shape constitutes an element. More concretely, we accurately parse out each element in the generated and reference slides, and compute their maximum matching using the match library.

Content similarity For each pair of matched elements, we compute their content similarity. If the reference element is text, we calculate the textual similarity using the cosine similarity of the embeddings produced with sentencetransformer with the default all-MiniLM-L6-v2 model [29]. If the reference element is an image, we calculate the CLIP score [19] of the image in two elements. We report the average content similarity across all matched element pairs, if either element contains a non-empty text string or an image component.

Color similarity We also measure the coloring similarity using the CIEDE2000 color difference formula [26], to quantify the perceptual difference between the colors. For every matched element pair, we measure the text font color similarity and element background color (if any). We additionally measure the color similarity between the background color of generated and reference slides.

Position similarity In addition to content and formatting, we also calculate the positional similarity between each pair of matched elements. More concretely, we follow Si et al. [38] to normalize the element coordinates to [0,1] by the slide page length and width. We compute the Manhattan distance between the elements and formulate positional similarity as sim(r,g) = 1−max(abs(xr −xg,yr −yg)).

Note that a low text, color, or position similarity score could come from differences in text, color, and positions, or derivative errors caused by the inaccurate element-matching

process (e.g., it may match the title box in the generated slide to a content textbox in the reference slide, which has different content or coloring requirements).

### 3.2. Reference-Free Metrics

A well-designed slide generated by models may look very different from the reference slide. Therefore, we also propose four reference-free evaluation metrics, to independently assess the design quality of model-generated slides. To establish the metrics, we surveyed a wide range of literature on slide design principles [5, 9, 34, 39, 44], and summarized four major points as below and detailed in Table 1:

Metric Criteria

Text The title should be simple and clear to indicate the main point. For main content, avoid too many texts and keep words concise. Use a consistent and readable font size, style, and color.

Image Use high-quality images with a reasonable proportion. Layout Elements should be aligned, do not overlap, and have

sufficient margins to each other. All elements should not exceed the page.

Color Use high-contrast color especially between the text and the background. Avoid using high-glaring colors.

Table 1. Reference-free metrics, all evaluated in 0-5 scale.

Text Using concise texts is important for slides to engage with the audience. An ideal slide should have a clear title, concise main content, and readable formatting.

Image Using appropriate visuals can engage audiences. We hence measure if models can find high-quality images and properly use them to enhance the slide quality.

Layout Slide layout is crucial to create visual balance. We examine whether all elements are within the slide, have no overlap, and align properly with the relevant elements.

Color Vivid and consistent color use in slides can help deliver insights. We check if the slide uses high-contrast colors to facilitate visibility, and avoid high-glaring colors to discourage user engagement.

Validation of Reference-Free Evaluation For all the metrics, we provide the image version of the slide and ask the gpt-4o model to produce a score between 0–5. To examine the reliability of this model-based evaluation, we conduct a human study and compare the intraclass correlation coefficient (ICC) between two human annotators and model evaluation, on all ground-truth slides. Our examination gives high ICC scores across all four metrics: 73.8%– 85.3%, which are well within the range of what is typically considered “high agreement”. In experiments in later sections, we scale these 0-5 scale scores to the 0–100 range to

enable comparisons on this more standard scale. †

### 3.3. Executability

Particularly for methods based on code generation (§4.1), we additionally measure the execution success rate to account for invalid programs. Concretely, we count the percentage of successfully executing programs generated by models among all examples. We report reference-based and free scores for executable slides only, to fairly compare their design quality. But we report ‘Overall’ scores for all slides by assigning zeros to non-executing slides, to account for execution failures. We report all metrics for successfully executing and all slides in §E.

## 4. Method

We introduce our main method — slide generation via code generation, optionally using our SLIDESLIB toolkit (§4.1). Then, we present AUTOPRESENT, trained on 7k slides, that achieves performance on par with strong GPT model (§4.2).

### 4.1. Slides via NL-to-Code Generation

Generating Python Programs Given natural language instructions in §2, models are tasked with generating Python programs using publicly available libraries such as python-pptx. The model receives two (natural language instruction, Python program) pairs as in-context examples, followed by the test instruction, and generates a Python program which is then executed and will ideally yield a PPTX file containing the requested slide.

Generating Programs with SLIDESLIB Nonetheless, the programs above could be very long and complex (170 lines on average), which could be challenging for models to generate entirely correctly, as shown in previous work [14]. To address this, we design SLIDESLIB, a library that provides easier-to-use interfaces for several common actions such as setting a title or setting background color. Using SLIDESLIB, the average program length is reduced to 13 lines, significantly easing the generation task. As shown in Table 2, SLIDESLIB includes 4 functions for basic operations and 3 functions for image search and generation, these functions allow models to produce more concise and modular programs. To enable the model to generate programs using SLIDESLIB, we follow the visual programming method [43] by providing a prompt that includes the documentation of the functions and two in-context examples. See more SLIDESLIB details in §B.

†We still evaluate with 0-5 scale to maintain a robust, human-aligned evaluation process.

#### Function Description

add title Insert a title in the slide. add text Insert text at a specific location. add bullet points Insert a textbox with bullet points. add image Insert image at a specific location.

generate image Call an image generator (Dall-E 3)

given a query. search image Search for an image on a search en-

gine (Bing).

search screenshot Display a query on a web browser (Google Chrome) and take a snapshot of the search result.

Table 2. Basic (top) and image-specific (bottom) functions provided by SLIDESLIB.

## 4.2. AUTOPRESENT

Using the slides in the training set of SLIDESBENCH, we construct (natural language instruction, program) pairs to form training data to train an open-sourced 8B model, AUTOPRESENT. This model is based on the LLAMA-3.1-8BInstruct and trained using LoRA [21] with a rank of 128.

Training Data Construction To create (natural language instruction, program) training pairs, we generate two versions of canonical program solutions for each slide:

- (i) Basic Python Programs We derive canonical programs (that is, programs that can be executed to reproduce the slide) without SLIDESLIB. To do this, we manually design an extraction script that (i) extracts each element (e.g., text and image) on the given slide, and (ii) produces a rulebased program that adds each element to the slide. After extracting and adding each element to the slide, the resulting program accurately reproduces the original slide.
- (ii) SLIDESLIB Python Programs We also generate canonical programs using SLIDESLIB, by transforming snippets from the programs above into SLIDESLIB function calls. To reproduce images in detailed instruction only and high-level instructions settings, we generate a short caption for each image and provide it to GPT-4o to generate the program for producing that image using search image or generate image functions. More details of this automatic program generation process are in §B.2.

Training Set Composition After obtaining three instructions and two program versions for each example, we construct four versions of the training data, each with 7k examples:

- 1. (detailed instruction with images, python program)
- 2. (detailed instruction with images, SLIDESLIB program)
- 3. (detailed instruction only, SLIDESLIB program)

Reference-Based Reference-Free

Method Execution%

Overall

element content color position text image layout color

Reference 100.0 – 59.7 81.5 73.5 65.7 – End-to-end Image Generation

Stable-Diffusion* 100.0 74.5 33.4 9.0 75.0 19.6 45.1 36.9 40.5 48.0 DALLE 3* 100.0 75.5 39.9 9.2 76.1 32.7 87.3 56.7 53.4 50.2

Code Generation w/o SLIDESLIB

LLaVA (7B) 11.3 61.9 97.3 6.2 70.8 41.6 100.0 29.2 25.7 6.1 CodeLLaMA (7B) 5.1 63.6 94.0 11.2 74.0 52.0 43.0 48.0 40.0 3.1 LLaMA (8B) 2.1 74.0 94.6 12.5 81.2 50.0 8.3 50.0 50.0 1.3 GPT-4o 89.2 83.3 91.6 10.5 77.0 51.9 72.8 53.7 54.7 55.1

AUTOPRESENT (ours) 79.0 67.7 79.7 10.9 75.9 45.3 62.7 54.2 60.9 45.2

Code Generation w/ SLIDESLIB

LLaVA (7B) 20.0 80.5 80.5 3.5 64.0 37.5 48.0 29.5 43.5 9.7 CodeLLaMA (7B) 48.7 80.3 89.8 9.4 69.3 45.9 66.8 45.1 49.9 30.3 LLaMA (8B) 54.4 78.3 91.2 7.5 69.5 46.0 68.2 47.6 53.1 33.5 GPT-4o 86.7 86.2 92.5 12.7 76.3 54.6 83.7 70.5 59.4 58.0

AUTOPRESENT (ours) 84.1 84.2 92.2 18.1 67.2 47.8 73.2 58.6 64.7 55.0

- Table 3. Results with detailed instructions with images. We found that small models like LLAVA (7B) and LLAMA (8B) can barely generate any slides, while AUTOPRESENT (8B) generates slides on par with GPT-4o. All the models still underperform humans.

- 4. (high-level instructions, SLIDESLIB program)

These training sets allowed us to train four specialized models that address different challenges, which we report in Table 3 (1,2) and Table 4 (3,4).

- 4.3. Iterative Refinement

Slide generation is by nature an iterative process and often requires visual-based refinements after the first draft. To enable models to refine slides as humans do, we explore an iterative refinement procedure, where the model is tasked to self-refine the slide it generated. Specifically, in the setting using SLIDESLIB, we provide GPT-4o (capable of consuming slide images) with the original language instruction, the program it generated in the first pass, and a snapshot of the rendered slide; the model is then asked to generate a new program based on these information to refine the slide quality by tweaking colors, spacing, and other aspects of the slide. See the prompts of this process in §D.

- 5. Experiments and Results

We first introduce the experimental setup (§5.1), then present the results under various scenarios (§5.2).

- 5.1. Experimental Setup

Code Generation Approaches For code generation approaches, we sample n = 3 responses and iteratively go through them, using the first successfully executing program as the final output of the model. If none of the n

responses execute successfully, we count it as an execution failure. In addition to AUTOPRESENT (§4.2), we benchmark several LMs out-of-the-box, including open-weights LLAMA 3.1 (8B, Instruct), the code generation models DeepseekCoder-7B-v1.5 and CodeLlaMa-7B-Instruct, the vision-language LLAVA v1.5 model (with a Vicuna-7Bv1.5 LM backbone); and the proprietary GPT-4O model (the gpt-4o-2024-08-06 checkpoint).

End-to-End Image Generation We compare code generation with end-to-end neural image generation methods, which are a common way to produce visuals. These methods are good at creating scenic or artistic images, but may be imprecise in content (esp. text) and do not support easy further modification by users. We benchmark StableDiffusion 2 [32] and DALL-E 3 [8] by asking them to output slides given the natural language instructions. We adjust our reference-based evaluation procedure by first segmenting slide images into elements using Tesseract OCR[40] and further parse out the texts of the elements, then applying the default calculation process as in §3. For the detailed instruction with images setting, we also report the results of the end-to-end image generation methods, marked with a “*” to indicate that they do not actually use the image inputs.

### 5.2. Quantitative Results and Analysis

Table 3 shows the result of detailed instruction with images scenario and Table 4 shows the result of detailed instruction

w/o SlidesLib w/ SlidesLib

Reference

LLaMA-3.1 (8B) LLaVA-1.5 (7B) AutoPresent (8B) GPT-4o

Dall-E 3

GPT-4o

[Figure 16]

[Figure 17]

|[Figure 18]|
|---|

|[Figure 19]|
|---|

[Figure 20]

[Figure 21]

DetailedInstructions

[Figure 22]

+Images DetailedInstructions

[Figure 23]

|[Figure 24]|
|---|

[Figure 25]

[Figure 26]

[Figure 27]

|[Figure 28]|
|---|

|Execution Error|
|---|

[Figure 29]

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

[Figure 33]

[Figure 34]

[Figure 35]

Only High-Level

[Figure 36]

|[Figure 37]|
|---|

|Execution Error|
|---|

|Execution Error|
|---|

[Figure 38]

|[Figure 39]|
|---|

[Figure 40]

[Figure 41]

[Figure 42]

|[Figure 43]|
|---|

|[Figure 44]|
|---|

[Figure 45]

|Execution Error|
|---|

|Execution Error|
|---|

Instructions

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

[Figure 49]

|Execution Error|
|---|

|Execution Error|
|---|

[Figure 50]

Figure 3. Examples of slides generated by different methods in three scenarios. End-to-end image generation methods fail to generate structured and clear slides. Small open-sourced models like LLAMA and LLAVA can barely generate any usable slides, while AUTOPRESENT produces quality slides. Adding SLIDESLIB improves GPT-4o’s performance on detailed instruction only and high-level instruction tasks.

Detailed Instructions Only High-Level Instructions

Method

open-source models such as LLAMA 3.1 and LLAVA barely produce any working slides out of the box. Although the significant gaps of 49.9–55.0 points exist in the detailed instruction with images setting, this gap shrinks to 22.2–34.6 when no visuals are provided a priori, in detailed instruction only and high-level instructions scenarios (Table 4). This demonstrates significant challenges in obtaining images in slides. In contrast to the low performance of open-weight models out-of-the-box, AUTOPRESENT’s performance approaches that of GPT-4O.

exec ref-based ref-free overall exec ref-based ref-free overall End-to-End Image Generation

SD2 100.0 48.0 35.5 48.0 100.0 47.7 31.5 47.7 DALLE 3 100.0 50.2 57.5 50.2 100.0 50.7 53.6 52.2

Code Generation w/o SLIDESLIB

LLaVA 17.9 56.9 47.4 9.3 19.5 50.2 47.3 9.5 DeepseekCoder 2.6 59.6 37.5 1.3 22.6 57.6 43.0 11.4 CodeLLaMA - - - - 21.0 57.9 54.4 12.2 LLaMA 4.6 61.4 35.1 2.8 8.7 55.6 50.1 4.8 GPT-4o 50.3 66.8 50.0 28.7 70.8 60.3 57.0 39.7

Code Generation w/ SLIDESLIB

LLaVA 17.4 58.2 33.8 8.0 25.1 50.1 36.7 10.9 DeepseekCoder 24.1 57.1 43.4 12.1 31.8 53.0 48.7 16.2 CodeLLaMA - - - - 35.9 56.6 53.4 20.3 LLaMA 60.5 61.7 56.6 37.4 76.9 56.8 58.3 43.7 GPT-4o 87.7 64.2 65.8 56.3 97.4 60.1 71.2 58.5

End-to-End Image Generation When no visuals are provided, end-to-end image-generation methods perform worse than the best code-generation approaches in both the reference-based and reference-free metrics, especially in generating accurate content. These methods also often produce creative figures without aligning with the design principles of slides, indicating its poorer controllability.

AUTOPRESENT 89.2 61.9 58.7 55.2 86.6 55.2 61.5 47.8

- Table 4. Results under detailed instruction only and high-level instructions settings. We assign 100% execution success rates for all end-to-end image generation methods because they do not generate programs and would not suffer from execution errors.

Effect of SLIDESLIB SLIDESLIB brings observable gains in LLAMA and LLAVA in all three scenarios by at most 34.0 points; and similarly increases the strong GPT4O performance across scenarios, especially when no images are provided. This suggests the benefits of generating more modular and concise programs for structured visual

only and high-level instructions scenarios.

In the top row of Table 3, we first measure the scores of the reference slides, which shows that the quality of the human-created slides is among the highest.

Compared to the scores achieved by GPT-4O, smaller

design.

VLM vs. LLM When no helper functions are presented, the one VLM that we tested (LLAVA) outperforms its LLM counterpart LLAMA in all scenarios by 5.1–7.5 points. However, LLAVA shows limited ability in using functions presented in context, as demonstrated by the large margin the library-augmented LLAMA has over LLAVA (12.1– 26.2). All LLMs (LLAMA, GPT-4O) perform worse when the instructions become less specified (detailed instruction with images → detailed instruction only → high-level instructions). Nonetheless, SLIDESLIB can greatly mitigate this degradation due to the loss of input specificity, and help models produce better outcomes across all three scenarios.

### 5.3. Qualitative Case Study

We illustrate several models-produced slides in Figure 3. For end-to-end image generation methods, the design is more creative and often more attractive, but the text does not constitute meaningful words, or even the characters themselves are not valid.

On the other hand, code generation methods, especially weaker LLAMA and LLAVA models, suffer more from visual layout — elements often overlap with each other or exceed the canvas, making it challenging for the audience to obtain all information clearly.

In contrast, AUTOPRESENT generates slides with appropriate layouts without undesirable element overlaps. In addition, they better follow the user instructions and are not overly creative like the image generation methods.

### 5.4. Perceptual Evaluation

Detailed+Images Detailed Only t-stat p-val t-stat p-val

Model Pairs

(GPT-4o, LLAMA) 13.206 0.000 8.630 0.000 (AUTOPRESENT, LLAMA) 13.180 0.000 2.955 0.004 (GPT-4o, AUTOPRESENT) -0.445 0.657 8.203 0.000

Table 5. Paired t-test results comparing model performance across detailed instruction only setting and detailed instruction with images setting. AUTOPRESENT and GPT-4o outperforms LLAMA with a statistically significant difference in both settings.

We performed a qualitative evaluation on 10 randomly selected slides from each domain generated by GPT-4o, Llama-8B, and AUTOPRESENT under the detailed instruction with images and detailed instruction only settings. We also add the ground-truth reference slide to evaluate the performance gap between current models and human slide creators. We shuffle these slides and ask the annotators to rank each slide from 1-5 based on how likely they would be to use the slide. For the detailed instruction with images setting, we collect 25 responses in total, and for detailed in-

###### 5.0 Detailed Instructions with Images

5.0 Detailed Instructions Only

| |2.20<br><br>1.32<br><br>2.23<br><br>3.70| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |2.41<br><br>1.48<br><br>2.12<br><br>3.94| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

4.5

4.5

4.0

4.0

3.5

3.5

3.0

3.0

2.5

2.5

2.0

2.0

1.5

1.5

1.0

1.0

GPT-4o Llama3 AutoPresent Reference

GPT-4o Llama3 AutoPresent Reference

Model

Model

Figure 4. Perceptual evaluation results on detailed instruction (1) with images and (2) only settings. We ask the users to score the quality of each slide from 1-5 and report the average score of each model. The user reported preference on GPT-4o and AUTOPRESENT compared with LLAMA, while still having a gap with human-designed slides.

struction only, we collect 16 responses in total. We provide more details of the evaluation process in §F.

The result is shown in Figure 4. By performing the paired t-test, we found differences between the models pairs in terms of user preferences, as shown in Table 5: (1) In both settings, AUTOPRESENT and GPT-4o perform statistically significantly better than LLAMA. (2) In detailed instruction with images setting, GPT-4o and AUTOPRESENT has no significant differences (3) In the detailed instruction only setting, AUTOPRESENT is slightly worse than gpt-4o, aligning with our quantitative evaluations in Table 4. All three models still have an overall performance gap compared with human-designed slides, indicating room for improvement on the slide generation task.

### 5.5. Result after Iterative Refinement

Iteration Detailed + Images Detailed Only High-Level

- 0 58.0 56.3 58.5

- 1 59.5 59.5 59.8

- 2 59.3 60.1 61.3

- 3 60.1 59.4 61.4

Table 6. Overall scores after applying multi-rounds of refinement in the three scenarios, demonstrating that refinement boosts performance in all three scenarios.

Finally, as shown by Table 6, we find that refinement improves model performance on all three challenges. By doing an ablation on the round of iterations, we find that while continued refinement often increases the scores, the first iteration usually gives the biggest performance improvement. We present representative cases after doing one round of refinement in Figure 5, which indicates that refinement can improve content layout and detailed controls on coloring and sizing.

###### Instruction

###### Before Reﬁnement After Reﬁnement

existing slide.

[Figure 51]

[Figure 52]

Create a slide with a collage of images depicting various market activities, arranged in circular frames, set against a neutral background.

## 7. Conclusion and Limitations

In this work, we address the challenge of creating structured visuals from scratch. Specifically, we introduced SLIDESBENCH, the first benchmark for automatic slide generation with evaluation metrics based on and free of reference slides. We benchmark multiple end-to-end image and program generation approaches, and demonstrate that AUTOPRESENT with SLIDESLIB achieves comparable performance with the top GPT-4O model. Our further exploration in iterative refinement also reveals certain effectiveness in self-refinement. This work is an initial step towards automated generation of structured visuals. Specifically, it focuses on single-slide generation and produces full slide code in a single pass, without leveraging iterative design workflows. Future research could address these limitations by expanding to full slide decks, adopting gradual and interactive slide generation, and incorporating slide-specific features like animations. Further, integrating more design principles, such as optimizing for attention capture and information clarity, would be crucial for making generated slides more impactful and effective.

|[Figure 53]|
|---|

[Figure 54]

Create a slide with the following elements:

- 1. Background: Use a gradient background transitioning from dark green to lighter green.
- 2. Text: add “as an …

[Figure 55]

|[Figure 56]|
|---|

Background: Set the slide background to an image of the Brooklyn Bridge.

2. Title: Add the text "NYC" in large, bold, white font. Center it horizontally on the slide for emphasis …

Figure 5. Auto-refinement results with GPT-4o, where the model further addresses some previously neglected instructions (marked in green), such as shape, background color, and text.

## 6. Related Work

Language and Vision Model-Based Agents Agents based on large language models (LLMs) [2, 11] and visionlanguage models (VLMs) [4, 51] have been widely adopted in various tasks such as web navigation [24, 52, 54], software engineering [49, 50], and web development [27, 38]. Creation of presentation materials is another common task [10] that has both similarities and differences from these more widely examined tasks.

## Acknowledgment

We would like to thank Yutong Bai for helping us draft Figure 1 and providing feedback on the paper, David Chan for providing detailed suggestions on the introduction, and Frank Xu and Sean Welleck for discussions at the initial stage of this project. Junyi Zhang and Haven Feng for feedback on the project.

Generating Programs for Vision Tasks End-to-end image generation models such as diffusion [20, 32, 48] and GAN [15, 35] are widely used at producing scenic images, yet falling short on more structured visuals such as websites and slides [38]. Generating programs (i.e., imageediting actions) is a useful means to get structured visuals [14, 18, 41, 43, 45], including Tikz figures [6, 7], SVG [31, 37], posters [51], and user interfaces [28, 38]. However, they often require detailed inputs and are limited to specific, simple figure types, so they are still far from creating complex, editable presentation slides from scratch. Our work extends this line of research by formulating and benchmarking the natural-language-to-slide generation task.

## References

- [1] Gamma - ai-powered document creation and storytelling platform. Accessed: 2024-11-14. 9
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 2, 9

- [3] Shaikh Mostafa Al Masum, Mitsuru Ishizuka, and Md Tawhidul Islam. ’auto-presentation’: a multi-agent system for building automatic multi-modal presentation of a topic from world wide web information. In IEEE/WIC/ACM International Conference on Intelligent Agent Technology, pages 246–249. IEEE, 2005. 1, 9
- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

Automatic Slide Generation Previous works on slide creation mostly focus on basic extraction from provided documents [12, 22, 23, 36, 42] or having models generate content given a topic [1, 3, 47] without addressing how to organize content visually. More recently, some benchmarks [17, 53] and methods [13] have emerged that follow detailed instructions for slide editing (e.g., adjust the font size of the title from 20 to 24) of an existing slide. In contrast, we synthesize more complex and structured programs that can generate slides from scratch, including content creation, visual arrangement, and fine-grained editing, instead of refining an

2022. 9

- [5] Angie Arriesgado. 45 tips to speed up your powerpoint design workflow, 2019. 2, 4
- [6] Jonas Belouadi, Anne Lauscher, and Steffen Eger. Automatikz: Text-guided synthesis of scientific vector graphics with tikz. ArXiv, abs/2310.00367, 2023. 9
- [7] Jonas Belouadi, Simone Paolo Ponzetto, and Steffen Eger. Detikzify: Synthesizing graphics programs for scientific figures and sketches with tikz. arXiv preprint arXiv:2405.15306, 2024. 9
- [8] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 2, 6
- [9] The LinkedIn community. How do you use design principles and best practices to evaluate and improve your slide layout and formatting?, 2024. 2, 4
- [10] decktopus. Top presentation statistics for 2021, 2021. 2, 9
- [11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 2, 9

- [12] Tsu-Jui Fu, William Yang Wang, Daniel McDuff, and Yale Song. Doc2ppt: Automatic presentation slides generation from scientific documents. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 634–642, 2022. 9
- [13] Apurva Gandhi, Thong Q Nguyen, Huitian Jiao, Robert Steen, and Ameya Bhatawdekar. Natural language commanding via program synthesis. arXiv preprint arXiv:2306.03460, 2023. 9
- [14] Jiaxin Ge, Sanjay Subramanian, Baifeng Shi, Roei Herzig, and Trevor Darrell. Recursive visual programming. In European Conference on Computer Vision, pages 1–18. Springer,

2025. 2, 5, 9

- [15] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 9
- [16] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024. 2
- [17] Yiduo Guo, Zekai Zhang, Yaobo Liang, Dongyan Zhao, and Nan Duan. Pptc benchmark: Evaluating large language models for powerpoint task completion. arXiv preprint arXiv:2311.01767, 2023. 9
- [18] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14953–14962, 2023. 9
- [19] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2021. 4

- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 9
- [21] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Lowrank adaptation of large language models. In International Conference on Learning Representations, 2021. 5
- [22] Yue Hu and Xiaojun Wan. Ppsgen: Learning-based presentation slides generation for academic papers. IEEE transactions on knowledge and data engineering, 27(4):1085–1097,

2014. 9

- [23] Min-Yen Kan. Slideseer: A digital library of aligned document and presentation pairs. In Proceedings of the 7th ACM/IEEE-CS joint conference on Digital libraries, pages 81–90, 2007. 9
- [24] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. VisualWebArena: Evaluating multimodal agents on realistic visual web tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2024. 9
- [25] Wenyuan Kong, Zhaoyun Jiang, Shizhao Sun, Zhuoning Guo, Weiwei Cui, Ting Liu, Jianguang Lou, and Dongmei Zhang. Aesthetics++: Refining graphic designs by exploring design principles and human preference. IEEE Transactions on Visualization and Computer Graphics, 29(6):3093–3104,

2022. 1

- [26] M Ronnier Luo, Guihua Cui, and Bryan Rigg. The development of the cie 2000 colour-difference formula: Ciede2000. Color Research & Application: Endorsed by Inter-Society Color Council, The Colour Group (Great Britain), Canadian Society for Color, Color Science Association of Japan, Dutch Society for the Study of Color, The Swedish Colour Centre Foundation, Colour Society of Australia, Centre Franc¸ais de la Couleur, 26(5):340–350, 2001. 4
- [27] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021. 9
- [28] Yi-Hao Peng, Faria Huq, Yue Jiang, Jason Wu, Xin Yue Li, Jeffrey P Bigham, and Amy Pavel. Dreamstruct: Understanding slides and user interfaces via synthetic data generation. In European Conference on Computer Vision, pages 466–485. Springer, 2025. 9
- [29] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2019. 4
- [30] Garr Reynolds. Presentation Zen: Simple ideas on presentation design and delivery. New Riders, 2011. 1
- [31] Juan A Rodriguez, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, David Vazquez, Christopher Pal, and Marco Pedersoli. Starvector: Generating scalable vector graphics code from images. arXiv preprint arXiv:2312.11556, 2023. 9

- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 6, 9
- [33] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950,

- 2023. 2

[34] Hitesh Sahni. Presentation design: A step-by-step guide,

- 2024. 2, 4

- [35] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. In International conference on machine learning, pages 30105–30118. PMLR,

2023. 9

- [36] Athar Sefid and Jian Wu. Automatic slide generation for scientific papers. In Third International Workshop on Capturing Scientific Knowledge co-located with the 10th International Conference on Knowledge Capture (K-CAP 2019), SciKnow@ K-CAP 2019, 2019. 1, 9
- [37] Pratyusha Sharma, Tamar Rott Shaham, Manel Baradad, Stephanie Fu, Adrian Rodriguez-Munoz, Shivam Duggal, Phillip Isola, and Antonio Torralba. A vision check-up for language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14410–14419, 2024. 9
- [38] Chenglei Si, Yanzhe Zhang, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: How far are we from automating front-end engineering? arXiv preprint arXiv:2403.03163, 2024. 4, 9
- [39] Think Outside The Slide. Latest annoying powerpoint survey results, 2019. 2, 4
- [40] Ray Smith. An overview of the tesseract ocr engine. In Ninth international conference on document analysis and recognition (ICDAR 2007), pages 629–633. IEEE, 2007. 6
- [41] Sanjay Subramanian, Medhini Narasimhan, Kushal Khangaonkar, Kevin Yang, Arsha Nagrani, Cordelia Schmid, Andy Zeng, Trevor Darrell, and Dan Klein. Modular visual question answering via code generation. arXiv preprint arXiv:2306.05392, 2023. 9
- [42] Edward Sun, Yufang Hou, Dakuo Wang, Yunfeng Zhang, and Nancy XR Wang. D2s: Document-to-slide generation via query-based text summarization. arXiv preprint arXiv:2105.03664, 2021. 9
- [43] D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11888–11898, 2023. 5, 9
- [44] Utrecht University. Four principles for making a good powerpoint presentation, 2024. 2, 4
- [45] Zhiruo Wang, Graham Neubig, and Daniel Fried. TroVE: Inducing verifiable and efficient toolboxes for solving programmatic tasks. In Forty-first International Conference on Machine Learning, 2024. 9

- [46] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024. 2
- [47] Thomas Winters and Kory W Mathewson. Automatically generating engaging presentation slide decks. In International Conference on Computational Intelligence in Music, Sound, Art and Design (Part of EvoStar), pages 127–141. Springer, 2019. 9
- [48] Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. Svgdreamer: Text guided svg generation with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4546–4555, 2024. 9
- [49] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. arXiv preprint arXiv:2405.15793,

2024. 2, 9

- [50] John Yang, Carlos E Jimenez, Alex L Zhang, Kilian Lieret, Joyce Yang, Xindi Wu, Ori Press, Niklas Muennighoff, Gabriel Synnaeve, Karthik R Narasimhan, et al. Swe-bench multimodal: Do ai systems generalize to visual software domains? arXiv preprint arXiv:2410.03859, 2024. 9
- [51] Tao Yang, Yingmin Luo, Zhongang Qi, Yang Wu, Ying Shan, and Chang Wen Chen. Posterllava: Constructing a unified multi-modal layout generator with llm. arXiv preprint arXiv:2406.02884, 2024. 2, 9
- [52] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. In Advances in Neural Information Processing Systems, pages 20744–

20757. Curran Associates, Inc., 2022. 9

- [53] Zekai Zhang, Yiduo Guo, Yaobo Liang, Dongyan Zhao, and Nan Duan. Pptc-r benchmark: Towards evaluating the robustness of large language models for powerpoint task completion. arXiv preprint arXiv:2403.03788, 2024. 9
- [54] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations, 2024. 2, 9

## A. SLIDESBENCH Details

### A.1. Slide Deck Domains

The 10 domains we cover in SLIDESBENCH include:

- 1. Art Photos
- 2. Business
- 3. Career
- 4. Design
- 5. Entrepreneur
- 6. Environment
- 7. Food
- 8. Marketing
- 9. Social Media
- 10. Technology

### A.2. Slide Deck Source

There existis large amount of slide decks on the internet including Google Search, Bing Search etc. For convenience, we collect a list of slides from the slideshare.com website.

### A.3. Slide Deck Statistics Per Domain

The average images per domain and average text blocks per domain are shown in Figure 6.

[Figure 57]

Figure 6. SlidesBench statistics on different domains.

## B. SLIDESLIB Details

In this section, we provide the detailed documentation and examples for all functions in our SLIDESLIB.

### B.1. SLIDESLIB Implementation

Figure 7 shows the basic functions and Figure 8 shows the image-oriented functions.

### B.2. SLIDESLIB Usage Example

Figure 9 shows two example programs using multiple SLIDESLIB functions to produce slides.

### B.3. SLIDESLIB Usage Percentage

The percentage of each action taken by GPT-4o, AUTOPRESENT, and Llama-3.1 in all 3 scenarios are reported in Figure 10. On average, the most common actions are add text (36.3%), add image (20.3%), and add title (13.5%).

## C. Training Details for AUTOPRESENT

The training parameters for AUTOPRESENT are summarized in Table 7.

Parameter Value LoRA Parameters

|LoRA rank|128<br><br>|
|---|---|
|LoRA alpha|32|
|LoRA dropout|0|
|Random state<br><br>|3407|
|RS-LoRA<br><br>|Disabled|
|LoFT-Q config|None|

Trainer Parameters

|Batch size (per device)|1<br><br>|
|---|---|
|Gradient accumulation steps|2|
|Warmup steps|20<br><br>|
|Epochs<br><br>|1|
|Learning rate<br><br>|3e-4|
|Mixed precision<br><br>|FP16|
|Weight decay<br><br>|0.01|
|Scheduler<br><br>|Linear|
|Seed|3407|

Table 7. Training details for AUTOPRESENT. LoRA and Trainer parameters are described in detail.

## D. Refinement Details

We provide the prompts that we used for auto-refinement in Figure 11. We input the instruction and the in-context examples, the previous code generated by the model, and the snapshot of the slide generated by executing this code to the model and let it correct the code.

## E. Detailed Results

We report two sets of evaluation metrics (reference-based and reference-free) in both their average value on all slides

add_title(slide, text, font_size, font_color, background_color) """Add a title text to the slide with custom font size and font color (RGB tuple). Args:

slide: Slide object as in pptx library text: str, Title text to be added font_size: int, Font size in int (point size), e.g., 44 font_color: tuple(int,int,int), RGB color, e.g., (0, 0, 0) background_color: Optional, tuple(int,int,int), RGB color, e.g., (255, 255, 255)

Rets: slide: Slide object with the title added

""" add_text(slide, text, coords, font_size, bold, color, background_color, auto_size) """Add a text box at a specified location with custom text and color settings. Args:

slide: Slide object as in pptx library text: str, Text to be added coords: list(float), [left, top, width, height] in inches font_size: int, Font size in int (point size), e.g., 20 bold: bool, True if bold-type the text, False otherwise color: tuple(int,int,int), RGB color, e.g., (0, 0, 0) background_color: Optional, tuple(int,int,int), RGB color, e.g., (255, 255, 255) auto_size: bool, True if auto-size the text box, False otherwise

Rets: slide: Slide object with the text box added

""" add_bullet_points(slide, bullet_points, coords, font_size, color, background_color) """Add a text box with bullet points. Args:

slide: Slide object as in pptx library bullet_points: list(str), List of texts to be added as bullet points coords: list(float), [left, top, width, height] in inches font_size: int, Font size in int (point size), e.g., 18 color: tuple(int,int,int), RGB color, e.g., (0, 0, 0) background_color: Optional, tuple(int,int,int), RGB color, e.g., (255, 255, 255)

Rets: slide: Slide object with the bullet points added

""" add_image(slide, image_path, coords) """Add an image in the provided path to the specified coords and sizes. Args:

slide: Slide object as in pptx library image_path: str, Path to the image file coords: list(float), [left, top, width, height] in inches

Rets: slide: Slide object with the image added

""" set_background_color(slide, color) """Set background color for the current slide. Args:

slide: Slide object as in pptx library color: tuple(int,int,int), RGB color, e.g., (255, 255, 255)

Rets:

modified slide object """

Figure 7. Documentation for the basic functions in our SLIDESLIB.

(i.e., un-weighted by execution success) and on successfully rendered slides (i.e., weighted by execution success).

### E.1. Detailed Instructions with Images

Table 8 shows all metrics down-weighted by the execution success rate; Table 9 shows reference-based and reference-

google_search_screenshot(question, save_path) """Search a question on Google, and take a screenshot of the search result. Save the screenshot to save_path, and return the path. Args:

question: str, The question to search on Google. save_path: str, The path to save the screenshot.

Returns: The path of the saved screenshot.

""" search_image(query, save_path) """Search for an image on Google and download the result to save_path. Args:

query: str, The query to search for. save_path: str, The path to save the downloaded image.

Rets: the save_path.

""" generate_image(query, save_path) """Generate an image using diffusion model based on a text query, and save the image to the path. Args:

query: str, The text query to generate the image. save_path: str, The path to save the generated image.

Rets:

The path of the saved image """

Figure 8. Documentation for the image-oriented functions in our SLIDESLIB.

free metrics without down-weighting by execution success.

### E.2. Detailed Instructions Only

Table 10 shows all metrics down-weighted by the execution success rate; Table 11 shows reference-based and referencefree metrics without down-weighting by execution success.

### E.3. High-Level Instructions Challenge

Table 12 shows all metrics down-weighted by the execution success rate; Table 13 shows reference-based and referencefree metrics without down-weighting by execution success.

## F. Perceptual Analysis

In this section, we provide perceptual analysis details. We build a google doc and ask the user to score each slide from 1-5 (1 is the worst and 5 is the best), as shown in Figure 12.

An example of the question is shown in Figure 13.

# Create slide with the title ’NLP Can Answer Questions’ in large, bolded font in the top center of the

page. Below it, put a screenshot of the google search result of the question ’Where was the first movie theater in the U.S?’ in the middle of the page.

from pptx import Presentation from pptx.util import Inches, Pt from library import add_text, google_search_screenshot, add_image

presentation = Presentation() presentation.slide_width = Inches(16) presentation.slide_height = Inches(9)

slide_layout = presentation.slide_layouts[0] # choose a layout template slide = presentation.slides.add_slide(slide_layout) add_text(slide, "NLP Can Answer Questions", coords=(1, 0.5, 8, 1), font_size=36) img_path = google_search_screenshot("Where was the first movie theater in the U.S?", save_path="

screenshot.png") add_image(slide, "screenshot.png", coords=(2.5, 2, 6, 4)) presentation.save("target_path.pptx")

# Create a slide titled ’Interior Design’ in bold, dark-green color in the center of the page. For the background, consider using a picture with a color, artistic vibe, ensure enough contrast between the colors of text and background.

from pptx import Presentation from pptx.util import Inches, Pt from library import generate_image, add_image, add_text

presentation = Presentation() presentation.slide_width = Inches(16) presentation.slide_height = Inches(9) slide_layout = presentation.slide_layouts[5] # choose a layout template slide = presentation.slides.add_slide(slide_layout)

background_img = generate_image("An colorful, artistic background", "colorful.png") add_image(slide, "colorful.png", coords=(0.0, 0.0, 16, 9)) add_text(slide, ’Interior Design’, coords=(0.0, 2.4, 13.3, 1.3), font_size=80, bold=True, color=(0, 0,

0), background_color=(255, 255, 255), auto_size=True) presentation.save("path.pptx") ‘‘‘

Figure 9. Example programs to produce slides using SLIDESLIB.

Reference-Based Reference-Free

Method Execution%

Average

block text color position text image layout color

Human 100.0 - 59.7 81.5 73.5 65.7 Code Generation w/o Library

LLaVA (7B) 11.3 7.0 11.0 0.7 8.0 4.7 11.3 3.3 2.9 6.1 LLaMA (8B) 2.1 1.5 1.9 0.3 1.7 1.0 0.2 1.0 1.0 1.3 GPT-4o 89.2 74.3 80.7 9.4 68.7 46.3 64.9 47.9 48.8 55.1

AUTOPRESENT (ours) 79.0 53.5 63.0 8.6 60.0 35.8 49.5 42.8 48.1 46.3

Code Generation w/ Expert-Designed Library

LLaVA (7B) 20.0 16.1 16.1 0.7 12.8 7.5 9.6 5.9 8.7 9.7 LLaMA (8B) 54.4 42.6 49.6 4.1 37.8 25.0 37.1 25.9 28.9 33.5 GPT-4o 86.7 74.7 80.2 11.0 66.1 47.3 72.5 61.1 51.4 58.0

AUTOPRESENT (ours) 84.1 70.8 77.5 15.2 56.5 40.2 61.6 49.3 54.4 55.0

- Table 8. Slide generation results (weighted by execution success) under the detailed instructions with images scenario.

Reference-Based Reference-Free

Method Execution%

Avg

block text color pos text img layout color

Human 100.0 - 59.7 81.5 73.5 65.7 Code Generation w/o Library

LLaVA (7B) 11.3 61.9 97.3 6.2 70.8 41.6 100.0 29.2 25.7 6.1 LLaMA (8B) 2.1 74.0 94.6 12.5 81.2 50.0 8.3 50.0 50.0 1.3 GPT-4o 89.2 83.3 91.6 10.5 77.0 51.9 72.8 53.7 54.7 55.1

AUTOPRESENT 79.0 67.7 79.7 10.9 75.9 45.3 62.7 54.2 60.9 46.3 Code Generation w/ Expert-Designed Library

LLaVA (7B) 20.0 80.5 80.5 3.5 64.0 37.5 48.0 29.5 43.5 9.7 LLaMA (8B) 54.4 78.3 91.2 7.5 69.5 46.0 68.2 47.6 53.1 33.5 GPT-4o 86.7 86.2 92.5 12.7 76.3 54.6 83.7 70.5 59.4 58.0

AUTOPRESENT (ours) 84.1 84.2 92.2 18.1 67.2 47.8 73.2 58.6 64.7 55.0

- Table 9. Slide generation results (un-weighted by execution success) under the detailed instructions with images scenario.

Reference-Based Reference-Free

Method Execution%

Average

block text color position text image layout color

End-to-End Image Generation

Stable-Diffusion 100.0 74.5 33.4 9.0 75.0 19.6 45.1 36.9 40.5 48.0 DALLE 3 100.0 75.5 39.9 9.2 76.1 32.7 87.3 56.7 53.4 50.2

Code Generation w/o Library

LLaVA (7B) 17.9 12.2 16.3 1.4 12.4 7.9 15.3 5.7 5.0 9.5 LLaMA (8B) 4.6 63.0 87.0 17.4 80.4 30.4 19.6 41.3 47.8 2.8 GPT-4o 50.3 42.2 50.0 6.0 39.8 27.1 15.3 29.0 29.2 32.2

Code Generation w/ Expert-Designed Library

LLaVA (7B) 17.4 15.6 15.5 0.9 10.5 5.7 6.2 4.1 7.5 8.3 LLaMA (8B) 60.5 45.1 55.5 5.2 43.6 29.5 44.3 29.6 33.4 37.4 GPT-4o 87.7 72.3 80.8 6.0 65.9 46.6 73.0 58.5 52.9 56.3

AUTOPRESENT (ours) 89.2 70.2 82.7 9.3 58.5 43.0 47.7 55.3 63.2 55.2

Table 10. Results (weighted by execution success) under detailed instructions only scenario.

Reference-Based Reference-Free

Method Execution%

Overall

block text color position text image layout color

End-to-End Image Generation

Stable-Diffusion 100.0 74.5 33.4 9.0 75.0 19.6 45.1 36.9 40.5 48.0 DALLE 3 100.0 75.5 39.9 9.2 76.1 32.7 87.3 56.7 53.4 50.2

Code Generation w/o Library

LLaVA (7B) 17.9 68.2 91.1 7.8 69.3 44.1 85.8 31.8 27.9 9.5 LLaMA (8B) 4.6 2.9 4.0 0.8 3.7 1.4 0.9 1.9 2.2 2.8 GPT-4o 50.3 83.9 92.4 11.9 79.1 53.9 30.4 57.7 58.1 32.2

Code Generation w/ Expert-Designed Library

LLaVA (7B) 17.4 89.7 89.1 5.2 60.3 32.8 35.6 23.6 43.1 8.3 LLaMA (8B) 60.5 74.5 91.7 8.6 72.1 48.8 73.2 29.6 48.9 37.4 GPT-4o 87.7 82.4 92.2 6.9 75.2 53.1 83.3 66.7 60.3 56.3

AUTOPRESENT (ours) 89.2 78.7 92.7 10.4 65.6 48.2 53.5 62.0 70.9 55.2

Table 11. Results (un-weighted by execution success) under detailed instructions only scenario.

Reference-Based Reference-Free

Method Execution%

Average

block text color position text image layout color

End-to-End Image Generation

Stable-Diffusion 100.0 72.0 33.2 8.3 77.2 3.3 49.3 35.6 37.8 47.7 DALLE 3 100.0 73.5 48.2 7.6 77.3 14.9 89.7 57.2 52.4 51.7

CodeGen-based Methods w/o Library

LLaVA (7B) 19.5 14.9 13.2 1.7 13.6 8.0 16.8 5.9 6.2 10.0 LLaMA (8B) 8.7 7.6 6.3 0.7 4.7 4.6 2.4 5.0 5.4 4.8 GPT-4o 70.8 54.6 54.2 7.5 54.4 42.4 19.2 51.9 48.0 39.0

CodeGen-based Methods w/ Library

LLaVA (7B) 25.1 20.4 17.8 1.6 15.4 9.2 9.7 6.9 11.0 11.5 LLaMA (8B) 76.9 55.4 58.3 5.6 55.7 39.5 56.5 40.3 43.0 43.7 GPT-4o 97.4 77.0 75.8 7.7 73.7 59.7 73.8 78.7 65.4 58.5

AUTOPRESENT (ours) 86.6 63.5 66.4 10.2 51.1 41.4 34.2 64.0 73.3 47.8

Table 12. Results (weighted by execution success) under high-level instructions scenario.

Reference-Based Reference-Free

Method Execution%

Average

block text color position text image layout color

End-to-End Image Generation

Stable-Diffusion 100.0 72.0 33.2 8.3 77.2 3.3 49.3 35.6 37.8 47.7 DALLE 3 100.0 73.5 48.2 7.6 77.3 14.9 89.7 57.2 52.4 51.7

CodeGen-based Methods w/o Library

LLaVA (7B) 19.5 76.4 67.7 8.7 69.7 41.0 86.2 30.3 31.8 10.0 LLaMA (8B) 8.7 87.4 72.4 8.0 54.0 52.9 27.6 57.5 62.1 4.8 GPT-4o 70.8 77.1 76.8 10.6 76.8 59.9 27.1 73.3 67.8 39.0

CodeGen-based Methods w/ Library

LLaVA (7B) 25.1 81.3 70.9 6.4 61.4 36.7 38.6 27.5 43.8 11.5 LLaMA (8B) 76.9 72.0 75.7 7.3 72.4 51.3 73.4 52.4 55.9 43.7 GPT-4o 97.4 79.0 77.8 7.9 75.6 61.3 75.8 80.7 67.1 58.5

AUTOPRESENT (ours) 86.6 73.3 76.7 11.8 59.0 47.8 39.5 73.9 84.6 47.8

Table 13. Results (un-weighted by execution success) under high-level instructions scenario.

API Usage by Condition

add_bullet_points add_image add_text add_title generate_image search_image set_background_color

high_leveldetailed_wimgdetailed_only

Llama

| |
|---|

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

Presenter

| |
|---|

GPT-4o

| |
|---|

| |
|---|

| |
|---|

Llama

| |
|---|

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |

Presenter

| |
|---|

GPT-4o

Llama

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

Presenter

GPT-4o

0 20 40 60 80 100

Usage Percentage

- Figure 10. The percentage of each action taken by different models.

""" You are an expert presentation slides designer who creates modern, fashionable, and stylish slides

using Python code. Your job is to generate the required PPTX slide by writing and executing a Python script. Make sure to follow the guidelines below and do not skip any of them:

- 1. Ensure your code can successfully execute. If needed, you can also write tests to verify your code.
- 2. Maintain proper spacing and arrangements of elements in the slide: make sure to keep sufficient spacing between different elements; do not make elements overlap or overflow to the slide page.
- 3. Carefully select the colors of text, shapes, and backgrounds, to ensure all contents are readable.
- 4. The slides should not look empty or incomplete. When filling the content in the slides, maintain good design and layout.

Follow the instruction below to create the slide. If the instruction is long and specific, follow the instruction carefully and add all elements as

required; if it is short and concise, you will need to create some content (text, image, layout) and implement it

into the slide. If you need to use the provided images, refer to the image file names in the instructions. Finally, your code should save the pptx file to path "output.pptx" API Libraries: # INSERT_API_DESCRIPTIONS_HERE ## Examples # INSERT_IN_CONTEXT_EXAMPLES_HERE Modification Task: Instruction: INSERT_INSTRUCTION_HERE Previous Code: INSERT_PREV_CODE_HERE Slide Snapshot : See image. Task: Based on the observed drawbacks in the provided slide image, modify the existing code accordingly

to improve the slide’s design and functionality.

Your modification: def generate_presentation(): """

- Figure 11. Prompt we used for Auto-Refinement. The model receives the APIs and instruction, the previous generated slide and code, and is tasked to re-write the code to do slide refinement.

""" Please score each slide from 1-5 based on your preference to use this slide in a real presentation. 5

is the best, 1 is the worst.

Carefully reading each slide’s content before ranking. """

#### Figure 12. Instruction we used for the perceptual evaluation.

[Figure 58]

##### Figure 13. An example of the perceptual analysis question. We ask the human to score the quality of the slide from 1-5.

