# arXiv:2310.08541v2[cs.CV]14Aug2024

## Idea2Img: Iterative Self-Refinement with GPT-4V for Automatic Image Design and Generation

Zhengyuan Yang , Jianfeng Wang , Linjie Li, Kevin Lin, Chung-Ching Lin , Zicheng Liu , and Lijuan Wang

Microsoft {zhengyang,jianfw,lindsey.li,keli,chungching.lin,zliu,lijuanw}@microsoft.com https://idea2img.github.io/

Abstract. We introduce “Idea to Image,”1 an agent system that enables multimodal iterative self-refinement with GPT-4V(ision) for automatic image design and generation. Humans can quickly identify the characteristics of different text-to-image (T2I) models via iterative explorations. This enables them to efficiently convert their high-level generation ideas into effective T2I prompts that can produce good images. We investigate if systems based on large multimodal models (LMMs) can develop analogous multimodal self-refinement abilities that enable exploring unknown models or environments via self-refining tries. Idea2Img cyclically generates revised T2I prompts to synthesize draft images, and provides directional feedback for prompt revision, both conditioned on its memory of the probed T2I model’s characteristics. The iterative self-refinement brings Idea2Img various advantages over vanilla T2I models. Notably, Idea2Img can process input ideas with interleaved image-text sequences, follow ideas with design instructions, and generate images of better semantic and visual qualities. The user preference study validates the efficacy of Idea2Img on automatic image design and generation via multimodal iterative self-refinement.

Keywords: Multimodal Agents · Self-Refinement · Large Multimodal Models · Image Design and Generation

### 1 Introduction

“Image design and generation” aims to create an image from a high-level user idea. This input IDEA can contain interleaved reference images, such as “the dog looks like the one in the image,” or with instructional texts specifying the intended design usage, such as “a logo for the Idea2Img system.” To convert IDEA into an image, humans may first draft detailed descriptions of the imagined image, and then use text-to-image (T2I) models [36,39,40,42,63] to generate the image. This manual process for users to search for an ideal detailed description (i.e., T2I prompt) that fits the T2I model typically involves iterative exploration [51, 67]. As shown in Figure 1, humans may first design and draft an

[Figure 1] for Automatic Image Design and Gener_images/imageFile1.png>)

1 Short for “Idea2Img.” System logo design assisted by Idea2Img.

[Figure 2] for Automatic Image Design and Gener_images/imageFile2.png>)

Idea2Img Framework

Input: Multimodal user-imagined IDEA to generate

###### IDEA 2:

###### IDEA 1:

[Figure 3] for Automatic Image Design and Gener_images/imageFile3.png>)

[Figure 4] for Automatic Image Design and Gener_images/imageFile4.png>)

[Figure 5] for Automatic Image Design and Gener_images/imageFile5.png>)

[Figure 6] for Automatic Image Design and Gener_images/imageFile6.png>)

photo of Bill Gates with the same suit as in the given image on the street

, with a dog looks like this one in the image

photo of Bill Gates with the same hand gesture as in the given image

, with a dog looks like this one in the image

[Figure 7] for Automatic Image Design and Gener_images/imageFile7.png>)

IDEA

Idea2Img Framework Multimodal iterative self-refinement

draft image * N

[Figure 8] for Automatic Image Design and Gener_images/imageFile8.png>)

|[Figure 9] for Automatic Image Design and Gener_images/imageFile9.png>)<br><br>Idea2Img Framework| |
|---|---|
| | |

text feedback

Draft Image Selection

- ②

- ③

select best draft image

[Figure 10] for Automatic Image Design and Gener_images/imageFile10.png>)

LMM with Memory

###### Prompt Generation

①

draft image * 1

generate initial/revised prompt

###### IDEA 3:

###### Feedback Reflection

[Figure 11] for Automatic Image Design and Gener_images/imageFile11.png>)

[Figure 12] for Automatic Image Design and Gener_images/imageFile12.png>)

cartoon drawing of the person as in the given image playing with a dog on the beach

, with a dog looks like this one in the image

① ② ③

give text feedback

text prompt * N

text feedback

draft image * N

Unknown model or environment to explore

[Figure 13] for Automatic Image Design and Gener_images/imageFile13.png>)

|[Figure 14] for Automatic Image Design and Gener_images/imageFile14.png>)<br><br>Output: AportraitofBillGatesona bustling city street, with his right hand raised in a friendly wave with his palm facing forward, standing next to a Siberian Husky with striking blue eyes and a playful tongue sticking out, in a lively and welcoming setting with natural daylight.<br><br>[Figure 15] for Automatic Image Design and Gener_images/imageFile15.png>)|
|---|

[Figure 16] for Automatic Image Design and Gener_images/imageFile16.png>)

|[Figure 17] for Automatic Image Design and Gener_images/imageFile17.png>)<br><br>Idea2Img Framework| |
|---|---|
| | |

T2I Model

- Fig. 1: Idea2Img framework enables LMMs to mimic human-like exploration to use a T2I model, enabling the design and generation of an imagined image specified as a multimodal input IDEA. The iterative process involves LMMs functioning in different roles to refine the image creation. Specifically, LMMs will (1) generate and revise text prompts for the T2I model, (2) select the best draft images, and (3) provide feedback on the errors and revision directions. This multimodal iterative self-refinement process requires LMMs to memorize the T2I model’s characteristics observed in previous iterations as humans and adjust T2I prompts accordingly.

initial T2I prompt based on their imagined IDEA to generate. Then, they can obtain multiple draft images with a T2I model, select the most promising draft, write text feedback, and further revise the T2I prompt. As this iteration progresses, we humans can swiftly grasp the characteristics of a specific T2I model, such as words that the model can not understand, finally producing a good image generated by a suitable T2I prompt. Given the remarkable capabilities of large multimodal models (LMMs) [14, 31, 57], we explore if we can build systems based on LMMs to develop similar iterative self-refinement ability, thereby relieving humans from the tedious process of converting ideas to images.

Iterative self-refinement is one intrinsic ability humans possess when exploring unknown environments and solving complicated problems. Large language models (LLMs) agent systems [9,27,46] have demonstrated the effectiveness of self-refinement in better addressing natural language processing tasks, such as acronym generation, sentiment retrieval, text-based environment exploration, etc. Transitioning from text-only tasks to multimodal environments poses new challenges of improving, assessing, and verifying multimodal contents, such as multiple interleaved image-text sequences. For example, when learning to use T2I models, LMMs need to improve the generation with revised T2I prompts, assess multiple images in detail to select the best draft, and verify the draft image with the multimodal IDEA to provide text feedback. These steps, each requiring different multimodal understanding capabilities, jointly enable the intriguing multimodal iterative self-refinement ability. Such an LMM framework can au-

.

IDEA T2I Idea2Img IDEA T2I Idea2Img IDEA T2I Idea2Img

Object count

Knowledge

Visual design

[Figure 18] for Automatic Image Design and Gener_images/imageFile18.png>)

[Figure 19] for Automatic Image Design and Gener_images/imageFile19.png>)

[Figure 20] for Automatic Image Design and Gener_images/imageFile20.png>)

[Figure 21] for Automatic Image Design and Gener_images/imageFile21.png>)

[Figure 22] for Automatic Image Design and Gener_images/imageFile22.png>)

[Figure 23] for Automatic Image Design and Gener_images/imageFile23.png>)

an image of a hand holding an iphone. The image is used for illustrating how to take a screen shot on iphone

5 people sitting around a table drinking beer and eating buffalo wings

a logo suitable for a stylish hotel.

Scene text

Prompt following

Intended usage

[Figure 24] for Automatic Image Design and Gener_images/imageFile24.png>)

[Figure 25] for Automatic Image Design and Gener_images/imageFile25.png>)

[Figure 26] for Automatic Image Design and Gener_images/imageFile26.png>)

[Figure 27] for Automatic Image Design and Gener_images/imageFile27.png>)

[Figure 28] for Automatic Image Design and Gener_images/imageFile28.png>)

[Figure 29] for Automatic Image Design and Gener_images/imageFile29.png>)

a whole cake on the table with words Azure Research written on the cake

a plate that has no bananas on it. there is a glass without orange juice next to it

an image of a car perfect for a children's painting competition

Concept customization & visual pointing

Visual attribute referring

[Figure 30] for Automatic Image Design and Gener_images/imageFile30.png>)

[Figure 31] for Automatic Image Design and Gener_images/imageFile31.png>)

[Figure 32] for Automatic Image Design and Gener_images/imageFile32.png>)

[Figure 33] for Automatic Image Design and Gener_images/imageFile33.png>)

[Figure 34] for Automatic Image Design and Gener_images/imageFile34.png>)

[Figure 35] for Automatic Image Design and Gener_images/imageFile35.png>)

Cartoon drawing of Mr

###### photo of the object pointed by the blue arrow, and a brown corgi dog

Bean playing tennis, with the same cloth and pose as in the given image

Opposite style transfer

Style transfer

[Figure 36] for Automatic Image Design and Gener_images/imageFile36.png>)

[Figure 37] for Automatic Image Design and Gener_images/imageFile37.png>)

[Figure 38] for Automatic Image Design and Gener_images/imageFile38.png>)

[Figure 39] for Automatic Image Design and Gener_images/imageFile39.png>)

[Figure 40] for Automatic Image Design and Gener_images/imageFile40.png>)

[Figure 41] for Automatic Image Design and Gener_images/imageFile41.png>)

painting of a corgi dog with style different from this one in the image

painting of a corgi dog with style similar to this one in the image

Visual design & style

Image manipulation

[Figure 42] for Automatic Image Design and Gener_images/imageFile42.png>)

[Figure 43] for Automatic Image Design and Gener_images/imageFile43.png>)

[Figure 44] for Automatic Image Design and Gener_images/imageFile44.png>)

[Figure 45] for Automatic Image Design and Gener_images/imageFile45.png>)

[Figure 46] for Automatic Image Design and Gener_images/imageFile46.png>)

[Figure 47] for Automatic Image Design and Gener_images/imageFile47.png>)

A hand drawing of a room where people can sleep and study. Hand drawing shows the sketch and looks like the given image.

a drawing with the background changed to a beach

In-context entity and style transfer

Multiple concepts customization

[Figure 48] for Automatic Image Design and Gener_images/imageFile48.png>)

[Figure 49] for Automatic Image Design and Gener_images/imageFile49.png>)

[Figure 50] for Automatic Image Design and Gener_images/imageFile50.png>)

[Figure 51] for Automatic Image Design and Gener_images/imageFile51.png>)

Find the the image style pattern in the left two dog images and apply it on the top right people in tree pose image. Provide a textual description that keeps the content in the people in tree pose image, with the correct style pattern.

[Figure 52] for Automatic Image Design and Gener_images/imageFile52.png>)

[Figure 53] for Automatic Image Design and Gener_images/imageFile53.png>)

with a dog looks like this one in the image

Photo of Bill Gates with the same cloth as in the given image

[Figure 54] for Automatic Image Design and Gener_images/imageFile54.png>)

[Figure 55] for Automatic Image Design and Gener_images/imageFile55.png>)

###### Visual attribute referring & style transfer

Blending images for new visual design

[Figure 56] for Automatic Image Design and Gener_images/imageFile56.png>)

[Figure 57] for Automatic Image Design and Gener_images/imageFile57.png>)

[Figure 58] for Automatic Image Design and Gener_images/imageFile58.png>)

[Figure 59] for Automatic Image Design and Gener_images/imageFile59.png>)

A logo with a design that naturally blends the two given images as a new logo.

[Figure 60] for Automatic Image Design and Gener_images/imageFile60.png>)

[Figure 61] for Automatic Image Design and Gener_images/imageFile61.png>)

[Figure 62] for Automatic Image Design and Gener_images/imageFile62.png>)

[Figure 63] for Automatic Image Design and Gener_images/imageFile63.png>)

, with the image style similar to this one in the image

and the second image is

A painting of a tennis game

The first image is

IDEA T2I Idea2Img IDEA T2I Idea2Img

- Fig. 2: Overview of the image design and generation scenarios enabled by Idea2Img. In each sub-figure, the image and text in the left green box are the user input IDEA. The center image is the baseline results directly generated by the same T2I model with a human-written T2I prompt, and the image on the right is generated with the T2I prompt discovered by Idea2Img’s iterative self-refinement exploration.

tomatically learn to tackle various real-world problems [57] via self-exploration, such as navigating GUI to use electronic devices, exploring unknown physical environments via an embodied agent, engaging in electronic games, and so on. In this study, we focus on “image design and generation” as the task to study the multimodal iterative self-refinement ability.

To this end, we introduce Idea2Img, a multimodal iterative self-refinement framework for automatic image design and generation. As illustrated in Figure 1, Idea2Img involves an LMM, GPT-4V(ision) [1,31–33], interacting with a

T2I model to probe its usage and find an effective T2I prompt. The LMM will act in different roles to analyze the return signal from the T2I model (i.e., draft images) and design the next round’s queries (i.e., text T2I prompts). The three roles of generating T2I prompts, selecting draft images, and reflecting feedback together enable the multimodal iterative self-refinement ability. Specifically, (1) Prompt generation: GPT-4V generates N text prompts that correspond to the input multimodal user IDEA, conditioned on the previous text feedback and refinement history; (2) Draft image selection: GPT-4V carefully compares N draft images for the same IDEA and select the most promising one; (3) Feedback reflection: GPT-4V examines the discrepancy between the draft image and the IDEA. GPT-4V then provides feedback on what is incorrect, the plausible causes, and how T2I prompts may be revised to obtain a better image. Furthermore, Idea2Img is enhanced with a memory module that stores all prompt exploration histories, including previous draft images, text prompts, and feedback. The Idea2Img framework iterates among these three steps with GPT-4V for automatic image design and generation.

To users, Idea2Img functions as an enhanced image design and generation assistant. Compared with T2I models, Idea2Img can handle design instructions instead of requiring detailed image description, support the multimodal IDEA input, and generate images of better semantic and visual qualities. We overview representative image design and generation scenarios in Figure 2. For example, Idea2Img can incorporate the visual design and intended usage description in IDEA, extract arbitrary visual information from the input image, and process IDEA with arbitrarily interleaved image-text sequences. Built upon these new functionalities and scenarios of interest, we develop an evaluation IDEA set with 104 samples, containing complicated queries that humans may fail in their first trials. We perform user preference studies on Idea2Img with different T2I models. The consistent user preference score improvements on different image generation models, e.g., +26.9% with SDXL [36], indicate the effectiveness of Idea2Img in image design and generation.

Our contributions are summarized as follows.

- – We study “automatic image design and generation,” which aims to create an image from an input IDEA. This new multimodal IDEA input enables visual creation with reference image inputs and instructions on desired designs.
- – We explore the multimodal iterative self-refinement ability in GPT-4V-based agent systems, showcasing its effectiveness in improving, assessing, and verifying multimodal contents.
- – We propose Idea2Img, a multimodal iterative self-refinement framework that enhances any image generation model for visual design, enabling various new image creation functionalities, and achieving better generation qualities.
- – We present an evaluation set with 104 challenging multimodal IDEA. The consistent user preference score improvements, when experimented on different image generation models, indicate Idea2Img’s effectiveness in automatic image design and generation.

### 2 Related Work

LLM-based self-refinement. Idea2Img is inspired by the effectiveness of iterative self-refinement in LLM-based agent systems [27,34,46] in exploring unknown environments and tasks, built upon the successful LLM agents [15,35,37,43,56, 61,66]. Self-refine [27] takes the same LLM to iteratively critique its outputs and leverage this feedback to enhance its predictions, showing effectiveness across various NLP tasks. Reflexion [46] explores a self-reflective LLM system on the text-based environment exploration task [47] and multi-hop QA [60]. Despite the success, LLM-based self-refinement naturally can not understand multimodal inputs. Consequently, the explored tasks and environments are limited to the natural language description, such as AlfWorld [47]. Idea2Img explores the potential of an LMM-based iterative self-refinement system for multimodal environment exploration, from a simple T2I model to other more complicated environments.

Multimodal agents. Our Idea2Img is related to multimodal agents [16,22,26, 44,49,52,58,64] that chain external tools such as T2I or vision-language models with LLMs for multimodal tasks. For instance, MM-ReAct [58] integrates ChatGPT with multiple vision tools for multimodal reasoning and action, enabling it to solve various complicated visual understanding tasks. Visual ChatGPT [52] empowers ChatGPT to allocate various image generation models, such as Stable Diffusion [40], img2img model [28], ControlNet [65], enabling multi-step visual editing and generation. The primary difference between Idea2Img and existing multimodal agent studies [52, 58] lies in the approach to understand the tool usage. Existing studies assume the knowledge of how to best use each tool and provide such information to LLMs via text instructions or in-context examples. In contrast, the optimal usage of the tool remains unknown in Idea2Img and requires iterative exploration. Another minor distinction is that Idea2Img utilizes LMMs instead of LLMs, thereby does not require general visual understanding tools such as a caption model [50,53].

Extensions of base T2I models. Idea2Img provides a more natural way for users to design and produce their desired visual content. This framework, which extends T2I models for new functionalities, is related to various works in improving base T2I models [36,39,40,42,63]. These studies include extending the base T2I model to better follow user prompts [5,7,10,12], finding magic words in T2I prompts for better visual quality [51,67], supporting extra image input for image manipulation [6,17,18,28], style transfer [13], visual concept customization [2,8,19,41,45], and so on. While specialized T2I extensions can address a single specific functionality, Idea2Img offers a more unified and widely applicable framework. That is, a single Idea2Img framework can handle various generation scenarios, ranging from style transfer to attribute customization, without requiring separate models or task-specific model design and finetune. More importantly, Idea2Img effectively collaborates with those enhanced generative models, consistently improving them by exploring suitable text prompts.

③ ① ② ⑤ ⑥ ④

[Figure 64] for Automatic Image Design and Gener_images/imageFile64.png>)

① ③

[Figure 65] for Automatic Image Design and Gener_images/imageFile65.png>)

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

②

- Fig. 3: The framework overview of Idea2Img, which takes an LMM [31,32] to explore a T2I model via multimodal iterative self-refinement, leading to an effective T2I prompt for the input user IDEA. The rounded rectangle shape indicates a GPT-4V call.

### 3 Idea2Img Framework

Figure 3 illustrates the Idea2Img framework. Idea2Img framework involves two core pre-trained models, i.e., the GPT-4V(ision) as the LMM M and a textconditioned image generation model2 to explore G. Idea2Img also contains a memory m that stores insights on G discovered by M during previous iterations. Execution flow. We begin with an overview of the key steps in M iteratively exploring the use of G. Starting from the top-left of Figure 3, “initial prompt generation” converts the input multimodal user IDEA into T2I text prompts, later producing multiple draft images with T2I model G. “Draft image selection” then selects the best draft image among them for the current iteration. The selected image is either output as the final prediction or continues for further refinement, depending on the stop condition. For the latter, “feedback reflection” compares the current best draft image with the multimodal IDEA, and summarizes the major discrepancy as text feedback. With the iteration history and text feedback, “revised prompt generation” then drafts revised T2I prompts and continues the iterative self-refinement with the new set of draft images.

1 Initial prompt generation. This step generates N initial T2I prompts y00,...,y0N−1 following the input user IDEA x, by prompting M with LMM

prompt pgen:

y00,...,y0N−1 = M(x,pgen) (1)

The “initial prompt generation” requires M to understand multimodal user IDEA x and convert design IDEA into descriptive T2I prompts. LMM prompt pgen is a zero-shot prompt without in-context examples.

2 We will show image generation models other than T2I later in experiments. For clarity, we use T2I as a representative generation model to introduce Idea2Img.

With the “initial prompt generation” step, Idea2Img can understand user IDEA with interleaved image-text sequences, instead of the text-only T2I prompts

containing the image description. Specifically, (1) IDEA can be a high-level design or usage instead of the detailed image description, such as “a car image for a children’s painting competition”; and (2) IDEA can take multiple images and use interleaved text instruction to extract arbitrary visual information of interest, including image style, visual entity, object attributes, etc. Then, in iteration t = 0 as well as future iterations t = t, each T2I prompt ytn is separately sent to the T2I model G, resulting in N draft images int = G(ytn),n = 0,...,N − 1.

2 Draft image selection. With the N draft images in iteration t, “draft

image selection” selects the best draft image i∗t by prompting M with LMM prompt pselect:

i∗t = M(i0t,...,iNt −1,x,pselect). (2)

The design of a “draft image selection” step is motivated by the observation that T2I models could generate bad images with good prompts. This step is designed to filter out low-quality images, and avoid the quality perturbation to dominate the iterative refinement.

The task of selecting the best image requires M to compare and grade both the semantics and visual quality of N similar draft images. We find such a “spot the difference” task challenging for LMMs, and only the very recent models [31, 57] are capable of performing the selection reliably.

- 3 Feedback reflection. After obtaining the selected image i∗t, the framework

checks the stop condition, such as if the current iteration t exceeds the maximum T. Idea2Img then outputs i∗t as the output image or proceeds the refinement process to the “feedback reflection” step accordingly.

“Feedback reflection” aims to provide text feedback ft that describes the direction to improve for draft image i∗t. The steps prompts M with LMM prompt pfb, conditioned on the draft image i∗t and memory m:

ft = M(i∗t,m,x,pfb). (3)

“Feedback reflection” takes M to compare an image i∗t with the multimodal user IDEA x, and summarize the gap as text feedback ft. The step not only requires M to identify the discrepancy between image i∗t and IDEA x, but also benefits from writing the major errors to make the iteration effective. In practice, we find it helpful to explicitly specify the aspects to check, such as style, entity, attributes, appearance, etc., via text instructions or in-context examples in LMM prompt pfb. Furthermore, we add text instructions to pfb to have M “focus on one thing to improve in each feedback,” and “provide a high-level explanation of how to modify prompts to address the given feedback.”

4 / 1 Revised prompt generation. Finally, “prompt generation” takes text

feedback ft and memory m to draft N revised prompt yt0+1,...,ytN+1−1 , by prompting M with LMM prompt previse:

yt0+1,...,ytN+1−1 = M(ft,m,x,previse). (4)

Generating revised prompts requires M to understand the property of G stored in memory m, thereby drafting new T2I prompts that could most likely address the issue identified in ft. We empirically demonstrate that Idea2Img can generate better prompts for G via iterative self-refinement.

Memory module. Memory m is one important design in Idea2Img. m has the format of interleaved image-text sequences that store all previous iterations’ feedback, selected draft image, and the corresponding text prompts:

mt = y0∗,i∗0,f0,...,yt∗−1,i∗t−1,ft−1 . (5)

It allows LMM M to understand the properties and capabilities of the T2I model G in use, such as a keyword that G may not understand or a complicated scene that G fail to generate, and incorporate such knowledge in generating the revised T2I prompts y. For example, it may describe the appearance of a yoga pose in detail, instead of only mentioning its name in y. Examples are shown in Appendix Figures A-D, when comparing initial and refined prompts y0 and yT.

### 4 Experiments

#### 4.1 Experiment Settings

Compared model variants. We mainly compare the following three models in image generation.

- – “Initial-round manual prompt” is the baseline T2I prompt written by humans with minor prompt engineering. It serves as the baseline of a T2I prompt that merely contains key information in IDEA.
- – “Initial-round Idea2Img prompt” is the LMM-generated T2I prompt in the initial round. Specifically, the max iteration T = 1, and LMM M is only used for initial prompt generation and draft image selection, but not feedback reflection nor revised prompt generation. This Idea2Img variant is used to ablate Idea2Img’s gain from prompt generation and selection, vs. the further iterative refinement.
- – “Iterative self-refined Idea2Img prompt” is complete Idea2Img pipeline with the max iteration T = 3.

Evaluation samples and metrics. For the quantitative evaluation, we collect a dataset of 104 user IDEA as input queries. Among them, 33 queries contain text only, 43 queries contain an image-text sequence with a single image, and the remaining 28 contains a sequence with two or more images. The text in most IDEA contains not only descriptive content text that describes the scene to generate, but also instructional text such as “a logo for commercial advertising” or “generate the pointed dog in blue.” All test queries are manually composed.

We then perform the user preference study as the main quantitative metric. Users are presented with the IDEA and multiple images to select the best one for each IDEA. The evaluation script automatically shuffles the order during evaluation to prevent the influence of image orders.

Table 1: User preference scores when applying Idea2Img onto different image generation models (compare the three scores in the middle section within each row individually). We observe that “Iterative self-refined Idea2Img prompt” is consistently favored across all experimented image generation models. ∆iteration reports the preference gain from the iterative Idea2Img over the initial-round Idea2Img.

|User preference score (%)|Initial-round Initial-round Iterative self-refined<br><br>manual prompt Idea2Img prompt Idea2Img prompt<br><br>|∆iteration|
|---|---|---|
|SDXL v1.0 DeepFloyd IF SD v2.1 SD v1.5 SDXL-img2img IF-img2img|13.5 29.8 56.7<br><br>14.4 34.6 51.0<br><br><br>13.5 40.4 46.2 8.6 43.3 48.1 8.6 34.6 56.7 8.6 38.5 52.9<br><br>|+26.9 +16.3 +5.8 +4.8 +16.3 +14.4<br><br>|

[Figure 66] for Automatic Image Design and Gener_images/imageFile66.png>)

[Figure 67] for Automatic Image Design and Gener_images/imageFile67.png>)

Idea2Img

SDXL v1.0 DeepFloyd IF SD v2.1 SD v1.5

- Fig. 4: User preference scores among T2I models before and after iterative selfrefinement. We observe that the initially favored T2I model, SDXL, benefits more from the Idea2Img iteration.

Experimented T2I models. We experiment Idea2Img on a wide variety of T2I model G with diverse model capacities and functionalities. Specifically, we study Stable Diffusion (SD) v1.5 [40], SD v2.1, SDXL v1.0 with refiner [36], and DeepFloyd IF (IF-I-XL and IF-II-L) [20]. Other than T2I models, we also consider the img2img pipeline (i.e., SDEdit [28]) for SDXL and DeepFloyd IF, as a demonstration of using Idea2Img for the text-conditioned image-to-image generation. The default strength t0 in the img2img pipeline is 1.00. SDXL-img2img and IF-img2img are the same as SDXL and IF (i.e., T2I) when IDEA contains text only, and condition on the first image with IDEA contains multiple images. LMM prompts pgen,pselect,pfb,previse are kept the same for all experimented T2I models. Appendix Section B shows the complete LMM prompts.

#### 4.2 Image Generation Results

User preference evaluation. Table 1 compares the user preference when selecting from the three images generated by “initial-round manual prompt,” “initial-round Idea2Img prompt,” and “iterative self-refined Idea2Img prompt,”

for each user IDEA with the same T2I model. Among T2I models with different model sizes and functionalities, Idea2Img leads to consistent improvements in user preference. The initial-round Idea2Img prompt already improves the initialround manual prompt, by effectively understanding the multimodal user IDEA and selecting the best draft images. The full Idea2Img framework further improves from the initial-round Idea2Img results with the multimodal iterative self-refinement. For example, when using SDXL v1.0, users prefer the images generated by Idea2Img 59/104 = 56.7% times, compared with the baseline of 14/104 = 13.5%. Similar improvements are observed on all experimented T2I models, as shown in the bolded column “iterative self-refined Idea2Img prompt.”

Furthermore, we examine which T2I model benefits the most from the LMM iterative self-refinement. By comparing the ∆iteration in Table 1 that represents the difference between first-round and iterative Idea2Img user preference, we observe that stronger T2I models tend to benefit more from LMM refinements. For example, SDXL and IF become more favored 26.9% and 16.3% times after iteration, compared with the 5.8% and 4.8% for SD v2.1 and SD v1.5. The trend that stronger T2I models benefit more from Idea2Img is also observed in Figure 4’s analysis, where users pick their preferred image generated by different T2I models. After Idea2Img’s iterative refinement, the initially favored model SDXL benefits more from the iteration, resulting in an even higher user preference rate, from 46.2% to 65.4%. We conjecture that the better language understanding ability in stronger T2I models enables them to better follow revised T2I prompts. They also have a better image generation capability that makes it possible to generate challenging scenes, when given a good T2I prompt optimized by Idea2Img. Nonetheless, Idea2Img is effective across T2I models of varying capacities, consistently leading to a higher user preference score.

Qualitative comparisons. Idea2Img could help users generate images that better follow IDEA, such as the correct object counts in Figure 5(a). Idea2Img enables visual content design, in contrast to conventional T2I that requires a detailed visual content description. For example in Figure 5(b), Idea2Img designs visual logo based on the instruction of “a logo for a 2024 conference in Seattle.” The power of LMMs allows Idea2Img to extract arbitrary information from the input image for visual generation. This could be any object in the image like “the circled dog” in Figure 5(c) or the image style like in Figure 5(d). Such general visual conditioning ability can be seamlessly extended to compose multiple visual and text conditions, such as composing the camera angle and image style in Figure 5(e) and two objects in Figure 5(f).

Other than SDXL, Idea2Img is effective in finding text prompts for other image generation models. This includes arbitrary T2I models (e.g., SD v2.1 [40], DeepFloyd IF [20], DALL·E 3 [30], etc.), text-conditioned image-to-image models (e.g., SDXL-img2img and IF-img2img with SDEdit [28]), and other specialist generation models (e.g., reward-tuned T2I [11,21], region-controlled generators [23,59,65], and other specialist models [3,6,41]). Figure 6 overviews Idea2Img working with different image generation models. We show additional qualitative results and discussions in Appendix Section A.1.

Initial-round Idea2Img prompt

Iterative self-refined Idea2Img prompt

Input user IDEA Initial-round manual prompt

[Figure 68] for Automatic Image Design and Gener_images/imageFile68.png>)

[Figure 69] for Automatic Image Design and Gener_images/imageFile69.png>)

[Figure 70] for Automatic Image Design and Gener_images/imageFile70.png>)

|[Figure 71] for Automatic Image Design and Gener_images/imageFile71.png>)<br><br>5 people sitting around a table drinking beer and eating buffalo wings|
|---|

|[Figure 72] for Automatic Image Design and Gener_images/imageFile72.png>)<br><br>A group of 5 friends sitting around a wooden table, drinking beer and eating buffalo wings in a casual setting.|
|---|

|[Figure 73] for Automatic Image Design and Gener_images/imageFile73.png>)<br><br>A group of 5 friends sitting around a wooden table, with one person at the head of the table and two people on each side, drinking beer and eating buffalo wings in a casual setting, with a window in the background and a warm, inviting atmosphere.|
|---|

###### (a)

5 people sitting around a table drinking beer and eating buffalo wings

[Figure 74] for Automatic Image Design and Gener_images/imageFile74.png>)

[Figure 75] for Automatic Image Design and Gener_images/imageFile75.png>)

[Figure 76] for Automatic Image Design and Gener_images/imageFile76.png>)

|[Figure 77] for Automatic Image Design and Gener_images/imageFile77.png>)<br><br>a logo for a 2024 conference held in Seattle.|
|---|

|[Figure 78] for Automatic Image Design and Gener_images/imageFile78.png>)<br><br>A logo design for a 2024 conference held in Seattle, featuring the Seattle skyline and the conference title in a modern font, with a blue and green color scheme.|
|---|

|[Figure 79] for Automatic Image Design and Gener_images/imageFile79.png>)<br><br>A logo design for a 2024 conference held in Seattle, with a modern and minimalist design, featuring the Seattle skyline and the conference title in a geometric sans serif font, with a blue and green color scheme.|
|---|

(b)

a logo for a 2024 conference held in Seattle. Please provide a textual description of the design before generating

[Figure 80] for Automatic Image Design and Gener_images/imageFile80.png>)

[Figure 81] for Automatic Image Design and Gener_images/imageFile81.png>)

[Figure 82] for Automatic Image Design and Gener_images/imageFile82.png>)

|[Figure 83] for Automatic Image Design and Gener_images/imageFile83.png>)<br><br>photo of a dog looks like the circled one in the image running on the floor that is a black and white dog|
|---|

|[Figure 84] for Automatic Image Design and Gener_images/imageFile84.png>)<br><br>A Cavalier King Charles Spaniel running on a tiled floor, happy mood, bright lighting|
|---|

|[Figure 85] for Automatic Image Design and Gener_images/imageFile85.png>)<br><br>A small Cavalier King Charles Spaniel with black and white fur, running on a tiled floor, tongue out, happy mood, bright lighting|
|---|

(c)

[Figure 86] for Automatic Image Design and Gener_images/imageFile86.png>)

photo of a dog looks like the circled one in the image running on the floor

[Figure 87] for Automatic Image Design and Gener_images/imageFile87.png>)

[Figure 88] for Automatic Image Design and Gener_images/imageFile88.png>)

[Figure 89] for Automatic Image Design and Gener_images/imageFile89.png>)

|[Figure 90] for Automatic Image Design and Gener_images/imageFile90.png>)<br><br>Painting of Mt rainier mountain with tree and lake in the foreground, with style similar to this one in the image that has colorful blocks|
|---|

|[Figure 91] for Automatic Image Design and Gener_images/imageFile91.png>)<br><br>A painting of Mt Rainier mountain with a tree and lake in the foreground, in a colorful and abstract style similar to a Picasso painting.|
|---|

|[Figure 92] for Automatic Image Design and Gener_images/imageFile92.png>)<br><br>A painting of Mt Rainier mountain with a tree and lake in the foreground, with bold lines and bright colors, with a cubist and surrealistic style, with a focus on geometric shapes and patterns.|
|---|

(d)

[Figure 93] for Automatic Image Design and Gener_images/imageFile93.png>)

Painting of Mt rainier mountain with tree and lake in the foreground, with style similar to this one in the image

[Figure 94] for Automatic Image Design and Gener_images/imageFile94.png>)

[Figure 95] for Automatic Image Design and Gener_images/imageFile95.png>)

[Figure 96] for Automatic Image Design and Gener_images/imageFile96.png>)

|[Figure 97] for Automatic Image Design and Gener_images/imageFile97.png>)<br><br>A painting of a tennis game from camera angle, with the image style similar to this one in the image that is impressionist|
|---|

|[Figure 98] for Automatic Image Design and Gener_images/imageFile98.png>)<br><br>A painting of a tennis game from a top-down camera angle, with the image style similar to Claude Monet's impressionist paintings, with a bright and sunny atmosphere.|
|---|

|[Figure 99] for Automatic Image Design and Gener_images/imageFile99.png>)<br><br>A painting of a tennis game from a top-down camera angle, with the image style similar to Claude Monet's impressionist paintings, with a bright and sunny atmosphere, with a blue sky and green trees in the background, with the players wearing white clothes, with a large crowd in the stands.|
|---|

(e)

[Figure 100] for Automatic Image Design and Gener_images/imageFile100.png>)

[Figure 101] for Automatic Image Design and Gener_images/imageFile101.png>)

, with the image style similar to this one in the image

A painting of a tennis game from camera angle

[Figure 102] for Automatic Image Design and Gener_images/imageFile102.png>)

[Figure 103] for Automatic Image Design and Gener_images/imageFile103.png>)

[Figure 104] for Automatic Image Design and Gener_images/imageFile104.png>)

[Figure 105] for Automatic Image Design and Gener_images/imageFile105.png>)

[Figure 106] for Automatic Image Design and Gener_images/imageFile106.png>)

|[Figure 107] for Automatic Image Design and Gener_images/imageFile107.png>)<br><br>Photo of Bill Gates with the same hand gesture as in the given image that is waving hand, with a dog looks like this one in the image that is a pug dog|
|---|

|[Figure 108] for Automatic Image Design and Gener_images/imageFile108.png>)<br><br>Bill Gates in a formal suit on a bustling city street, raising his hand in a friendly wave like a charming businessman, with a cute pug dog with a wrinkled face and large eyes by his side, under natural daylight, with a sense of approachability.|
|---|

|[Figure 109] for Automatic Image Design and Gener_images/imageFile109.png>)<br><br>Bill Gates wearing a suit and tie, standing on a busy street with tall buildings, waving with his right hand raised and palm facing forward, with a small pug dog with an adorably wrinkled face and big round eyes next to him, in daylight, exuding friendliness and warmth.|
|---|

(f)

[Figure 110] for Automatic Image Design and Gener_images/imageFile110.png>)

[Figure 111] for Automatic Image Design and Gener_images/imageFile111.png>)

Photo of Bill Gates with the same hand gesture as in the given image

with a dog looks like this one in the image

- Fig. 5: The comparisons among initial-round manual prompt, initial-round Idea2Img prompt, and iterative self-refined Idea2Img prompt, with SDXL [36] as the T2I model.

How Idea2Img may assist humans? We use selected qualitative results to highlight the scenarios where humans might find Idea2Img most helpful in image design and generation, compared with conventional T2I generation.

- 1. New functionalities with multimodal IDEA inputs. Idea2Img provides a more natural way for human interaction, where users do not have to describe their desired image solely through texts and painstakingly search for the right prompt word. Instead, the multimodal IDEA allows Idea2Img to precisely extract specific elements from one or multiple input images, such as the dog breed and color, pointed objects, artist style, camera view, and more, as shown in Figure 5. Finding proper words that the T2I model can understand for such visual concepts could be tedious for humans, e.g., the art style “with bold lines and bright colors, with a cubist and surrealistic style, with a focus on geometric shapes and patterns.” in Figure 5(d). Idea2Img automates this process via Idea2Img iterative self-refinement.
- 2. New functionalities with instructional inputs. Vanilla T2I models struggle to understand T2I prompts that describe the intended visual design or purpose of the generated image, such as “a logo for a 2024 conference held in Seattle” in Figure 5(b). Instead, the prompt needs to be a comprehensive description of the image to generate, demanding extra drafting effort from users, such as “...the Seattle skyline in the center and the conference title

###### Input user IDEA Initial-round manual prompt Initial-round Idea2Img prompt Iterative self-refine Idea2Img prompt

SD v2.1

[Figure 112] for Automatic Image Design and Gener_images/imageFile112.png>)

[Figure 113] for Automatic Image Design and Gener_images/imageFile113.png>)

[Figure 114] for Automatic Image Design and Gener_images/imageFile114.png>)

|[Figure 115] for Automatic Image Design and Gener_images/imageFile115.png>)<br><br>6 cakes placed in a box|
|---|

|[Figure 116] for Automatic Image Design and Gener_images/imageFile116.png>)<br><br>A cardboard box with 6 homemade cakes, each with a unique design and flavor, ready to be delivered.|
|---|

|[Figure 117] for Automatic Image Design and Gener_images/imageFile117.png>)<br><br>A brown cardboard box with 6 homemade cakes, each with a different flavor and design, arranged in a symmetrical pattern.|
|---|

6 cakes placed in a box

DeepFloyd IF

|[Figure 118] for Automatic Image Design and Gener_images/imageFile118.png>)<br><br>[Figure 119] for Automatic Image Design and Gener_images/imageFile119.png>)|
|---|

[Figure 120] for Automatic Image Design and Gener_images/imageFile120.png>)

[Figure 121] for Automatic Image Design and Gener_images/imageFile121.png>)

|[Figure 122] for Automatic Image Design and Gener_images/imageFile122.png>)<br><br>a logo for a 2023 conference held in Paris.|
|---|

|[Figure 123] for Automatic Image Design and Gener_images/imageFile123.png>)<br><br>A contemporary logo for a 2023 conference in Paris, showcasing a stylized Eiffel Tower, elegant fonts, and a color palette inspired by the French flag, with a touch of sophistication.|
|---|

|[Figure 124] for Automatic Image Design and Gener_images/imageFile124.png>)<br><br>A sleek and modern logo for a 2023 conference in Paris, incorporating a stylized Eiffel Tower, the year '2023' in a classy font, and a tricolor background inspired by the French flag. The design includes the word 'Paris' at the bottom in a refined font, symbolizing sophistication and culture.|
|---|

a logo for a 2023 conference held in Paris. Please provide a textual description of the design before generating

DALL·E 3

|[Figure 125] for Automatic Image Design and Gener_images/imageFile125.png>)<br><br>[Figure 126] for Automatic Image Design and Gener_images/imageFile126.png>)|
|---|

[Figure 127] for Automatic Image Design and Gener_images/imageFile127.png>)

[Figure 128] for Automatic Image Design and Gener_images/imageFile128.png>)

|[Figure 129] for Automatic Image Design and Gener_images/imageFile129.png>)<br><br>Find the the image style pattern in the left two dog images and apply it on the top right people waving hand image|
|---|

|[Figure 130] for Automatic Image Design and Gener_images/imageFile130.png>)<br><br>The image of a man waving is stylized with a low-poly design, where the scene is broken down into geometric shapes filled with bold and vibrant colors, creating a visually striking and contemporary aesthetic, under the bright sky.|
|---|

|[Figure 131] for Automatic Image Design and Gener_images/imageFile131.png>)<br><br>A man in a formal suit is captured mid-motion, his hand raised in a wave, each finger distinctly visible. The entire scene is reimagined in a low-poly art style, where each facet of the man and his attire is defined by geometric shapes filled with a palette of bold, contrasting colors. The background remains a solid, untextured expanse, free of additional elements like sun rays or clouds, ensuring the focus remains solely on the colorful, stylized figure of the man waving. The lighting is neutral, casting no shadows or highlights, preserving the simplicity of the scene.|
|---|

[Figure 132] for Automatic Image Design and Gener_images/imageFile132.png>)

[Figure 133] for Automatic Image Design and Gener_images/imageFile133.png>)

Find the the image style pattern in the left two dog images and apply it on the top right people waving hand image. Provide a textual description that keeps the content in the people waving hand image, with the correct style pattern.

DALL·E 3

[Figure 134] for Automatic Image Design and Gener_images/imageFile134.png>)

[Figure 135] for Automatic Image Design and Gener_images/imageFile135.png>)

[Figure 136] for Automatic Image Design and Gener_images/imageFile136.png>)

|[Figure 137] for Automatic Image Design and Gener_images/imageFile137.png>)<br><br>Two dogs running on the beach, the left one looks like a fluffy golden dog and the right one looks like the left dog in the image that is black and white|
|---|

|[Figure 138] for Automatic Image Design and Gener_images/imageFile138.png>)<br><br>Two dogs, one with the appearance of a Cavalier King Charles Spaniel with black and white fur and the other resembling a brown and white Yorkshire Terrier, are energetically running side by side on a sandy beach, their tails wagging in excitement. The backdrop features a serene ocean view, clear skies, gentle waves, bright sunlight casting soft shadows, and distant seagulls soaring in the sky, adding a lively, playful atmosphere to the scene.|
|---|

|[Figure 139] for Automatic Image Design and Gener_images/imageFile139.png>)<br><br>Two dogs, a Yorkshire Terrier with detailed brown and white fur that glistens in the sunlight, on the left of a black and white Cavalier King Charles Spaniel, are immersed in a game of chase along a scenic beach. The ocean waves, with their rhythmic motion, complement the dogs’ lively energy. The sky, a masterpiece of soft, blended colors, serves as a canvas for the seagulls that glide gracefully, casting intricate shadows that dance on the sandy surface, amplifying the scene’s vibrancy.|
|---|

[Figure 140] for Automatic Image Design and Gener_images/imageFile140.png>)

[Figure 141] for Automatic Image Design and Gener_images/imageFile141.png>)

and the right one looks like the left dog in the image

Two dogs running on the beach, the left one looks like

SDXL img2img

[Figure 142] for Automatic Image Design and Gener_images/imageFile142.png>)

[Figure 143] for Automatic Image Design and Gener_images/imageFile143.png>)

[Figure 144] for Automatic Image Design and Gener_images/imageFile144.png>)

|[Figure 145] for Automatic Image Design and Gener_images/imageFile145.png>)<br><br>image that depicts what will happen next based on the story in the two given frames. Please provide a textual description of the image before generating. The first frame in the story is rice on sushi roll, and the second frame in the story is ingredients on rice|
|---|

|[Figure 146] for Automatic Image Design and Gener_images/imageFile146.png>)<br><br>A close-up of a sushi roll in the making, with a pile of rice on a seaweed sheet and salmon and cucumber being added, next step is rolling and cutting the sushi.|
|---|

|[Figure 147] for Automatic Image Design and Gener_images/imageFile147.png>)<br><br>A close-up of a sushi roll in the making, with a pile of rice on a seaweed sheet and salmon and cucumber being added, next step is rolling and cutting the sushi, with a bamboo mat and knife nearby, in a kitchen setting.|
|---|

image that depicts what will happen next based on the story in the two given frames. Please provide a textual description of the image before generating. The first frame in the story is

[Figure 148] for Automatic Image Design and Gener_images/imageFile148.png>)

[Figure 149] for Automatic Image Design and Gener_images/imageFile149.png>)

, and the second frame in the story is

DeepFloyd IF img2img

|[Figure 150] for Automatic Image Design and Gener_images/imageFile150.png>)<br><br>photo with a design that composites the two given objects into the new photo. The placement of the objects does not adhere to the laws of physics. Please provide a textual description of the objects and design before generating. The first object is a marble table, and the second object is a coke|
|---|

|[Figure 151] for Automatic Image Design and Gener_images/imageFile151.png>)<br><br>A marble table with a Coca-Cola bottle floating above it, defying gravity, in a modern living room, with soft ambient lighting, minimalistic decor, and a hint of surrealism.|
|---|

|[Figure 152] for Automatic Image Design and Gener_images/imageFile152.png>)<br><br>In a contemporary living room, a Coca-Cola bottle is suspended in mid-air above a marble table, with a mystical aura, warm lighting, and sleek decor, evoking a sense of magic and defying gravity.|
|---|

[Figure 153] for Automatic Image Design and Gener_images/imageFile153.png>)

[Figure 154] for Automatic Image Design and Gener_images/imageFile154.png>)

[Figure 155] for Automatic Image Design and Gener_images/imageFile155.png>)

photo with a design that composites the two given objects into the new photo. The placement of the objects does not adhere to the laws of physics. Please provide a textual description of the objects and design before generating. The first object is

[Figure 156] for Automatic Image Design and Gener_images/imageFile156.png>)

[Figure 157] for Automatic Image Design and Gener_images/imageFile157.png>)

, and the second object is

- Fig. 6: The comparisons among initial-round manual prompt, initial-round Idea2Img prompt, and iterative self-refined Idea2Img prompt, with different image generation models. Additional qualitative results and discussions are in Appendix A.1.

below it ...”. In contrast, Idea2Img effectively understands the instructional texts in IDEA and creates images accordingly.

- 3. Better semantic and visual quality. Finally, the iterative refinement allows Idea2Img to generate images with better semantic and visual qualities, leading to an effective automatic image creation assistant.
- 4.3 LMM Feedback, Revision, and Selection

We show representative LMM outputs for “feedback reflection,” “revised prompt generation,” and “draft image selection.” Additional results are in Appendix A.2. Feedback reflection. Figure 7(a) shows the text feedback generated by GPT-

- 4V for the user IDEA and the draft image and T2I prompt. Idea2Img can effectively check if the generated image is correct, and verify if the draft image corresponds to the visual descriptions in IDEA. This includes the breed of the dog in (a.1), as well as art styles, objects, visual attributes, etc. In addition to identifying the discrepancy, Idea2Img also points to the plausible directions that may improve the T2I prompt in the text feedback. For example, in (a.2), Idea2Img provides guidance to have generated images better follow the user intention of “an image for a children’s painting competition,” by “specifically mentioning that the car should be simple and cartoon-like.”

[Figure 158] for Automatic Image Design and Gener_images/imageFile158.png>)

[Figure 159] for Automatic Image Design and Gener_images/imageFile159.png>)

[Figure 160] for Automatic Image Design and Gener_images/imageFile160.png>)

|[Figure 161] for Automatic Image Design and Gener_images/imageFile161.png>)|
|---|

|[Figure 162] for Automatic Image Design and Gener_images/imageFile162.png>)|
|---|

[Figure 163] for Automatic Image Design and Gener_images/imageFile163.png>)

[Figure 164] for Automatic Image Design and Gener_images/imageFile164.png>)

[Figure 165] for Automatic Image Design and Gener_images/imageFile165.png>)

|[Figure 166] for Automatic Image Design and Gener_images/imageFile166.png>)|
|---|

|[Figure 167] for Automatic Image Design and Gener_images/imageFile167.png>)|
|---|

- (a) Feedback reflection: The right column shows the examples of generated text feedback. Revised prompt and image

cartoon drawing of Mr Bean with the same cloth as in the given image

playing with a dog looks like this one on the beach

|[Figure 168] for Automatic Image Design and Gener_images/imageFile168.png>)<br><br>The dog in the generated image is not the same breed as the one in the given image. The dog in the given image is a Yorkshire Terrier, while the dog in the generated image is a Boston Terrier. To address this issue, the sentence prompt should be modified to specifically mention the breed of the dog as a Yorkshire Terrier.|
|---|

|[Figure 169] for Automatic Image Design and Gener_images/imageFile169.png>)<br><br>A humorous scene of Mr Bean and a small dog playing fetch on a sandy beach with blue skies in the background.|
|---|

|[Figure 170] for Automatic Image Design and Gener_images/imageFile170.png>)<br><br>A humorous scene of Mr Bean wearing a yellow tennis outfit and playing with a Yorkshire Terrier on a beach with palm trees in the background.|
|---|

Input user IDEA Draft prompt and image to revise Text feedback

[Figure 171] for Automatic Image Design and Gener_images/imageFile171.png>)

[Figure 172] for Automatic Image Design and Gener_images/imageFile172.png>)

[Figure 173] for Automatic Image Design and Gener_images/imageFile173.png>)

[Figure 174] for Automatic Image Design and Gener_images/imageFile174.png>)

- (b.1)

|[Figure 175] for Automatic Image Design and Gener_images/imageFile175.png>)<br><br>The image generated by the AI art generation model does not follow the user imagined IDEA of the scene as there are bananas on the plate and orange juice in the glass. To address this, the sentence prompt can be modified to specify that there should be no bananas on the plate and no orange juice in the glass.|
|---|

a plate that has no bananas on it. there is a glass without orange juice next to it

|[Figure 176] for Automatic Image Design and Gener_images/imageFile176.png>)<br><br>A breakfast scene with a plate that has no bananas and a glass without orange juice, with a blue background and a bright lighting.|
|---|

|[Figure 177] for Automatic Image Design and Gener_images/imageFile177.png>)<br><br>A close-up view of an empty plate with no other objects on it, and an empty glass without orange juice on a wooden table, with a white background.|
|---|

[Figure 178] for Automatic Image Design and Gener_images/imageFile178.png>)

[Figure 179] for Automatic Image Design and Gener_images/imageFile179.png>)

[Figure 180] for Automatic Image Design and Gener_images/imageFile180.png>)

[Figure 181] for Automatic Image Design and Gener_images/imageFile181.png>)

- (b.2)

[Figure 182] for Automatic Image Design and Gener_images/imageFile182.png>)

[Figure 183] for Automatic Image Design and Gener_images/imageFile183.png>)

- (b) Revised prompt generation: The right column shows the examples of revised prompts.

|[Figure 184] for Automatic Image Design and Gener_images/imageFile184.png>)|
|---|

|[Figure 185] for Automatic Image Design and Gener_images/imageFile185.png>)|
|---|

[Figure 186] for Automatic Image Design and Gener_images/imageFile186.png>)

[Figure 187] for Automatic Image Design and Gener_images/imageFile187.png>)

|[Figure 188] for Automatic Image Design and Gener_images/imageFile188.png>)<br><br>[Figure 189] for Automatic Image Design and Gener_images/imageFile189.png>)|
|---|

[Figure 190] for Automatic Image Design and Gener_images/imageFile190.png>)

[Figure 191] for Automatic Image Design and Gener_images/imageFile191.png>)

[Figure 192] for Automatic Image Design and Gener_images/imageFile192.png>)

|[Figure 193] for Automatic Image Design and Gener_images/imageFile193.png>)<br><br>[Figure 194] for Automatic Image Design and Gener_images/imageFile194.png>)|
|---|

[Figure 195] for Automatic Image Design and Gener_images/imageFile195.png>)

[Figure 196] for Automatic Image Design and Gener_images/imageFile196.png>)

[Figure 197] for Automatic Image Design and Gener_images/imageFile197.png>)

[Figure 198] for Automatic Image Design and Gener_images/imageFile198.png>)

- (c) Draft image selection: The right column shows the examples of the draft image selection index and justification.

Fig. 7: GPT-4V’s outputs in Idea2Img for image feedback, revision, and selection.

Revised prompt generation. Figure 7(b) compares the T2I prompts before and after the prompt revision, showcasing how text feedback may help the refinement. For example, in (b.1), the revised T2I prompt specifies “an empty plate with no other objects” to preclude the T2I model from generating bananas, which occurred with the previous prompt “no bananas.” In (b.2), the revised T2I prompt includes a detailed description of “a yellow tennis outfit” and “a Yorkshire Terrier” to generate the queried clothing and dog.

Draft image selection. Performing draft image selection requires LMMs to compare multiple similar draft images and pick the one that best matches the multimodal input IDEA. Figure 7(c) shows the selection results generated by Idea2Img. GPT-4V is prompted to give justifications and scores for each draft image, in addition to the final selection. We observe that Idea2Img could comprehensively judges different aspects in IDEA, and gives reasonable scores and selection indexes. E.g., finding the image with both sleep and study area in (c.1), verifying content and style in (c.2), and other examples in Appendix Figure G.

### 5 Limitation and Discussion

Tasks beyond image generation. Idea2Img explores the emergent ability of multimodal self-refinement in LMM-based systems, through the image design and generation task. Specifically, Idea2Img views the T2I model to use as an unknown multimodal environment to explore, and iteratively refines T2I prompts to find its optimal usage. This concept mirrors the intrinsic human approach of iterative problem-solving when faced with unknown environments or challenges. We leave its extension to other intriguing tasks, e.g., GUI navigation [55], embodied agents [29], and complicated visual reasoning [38,54], for future exploration.

From a single image generation model to multiple tools. Idea2Img explores using a single image generation model, such as a text-to-image model [40] or a text-conditioned image-to-image model [28]. When needed, other specialized generative models like ControlNet [65], inpainting [3], region-controlled T2I generation [23,59], customized generation [8,41], and video generation [48,62] can be seamlessly switched and supported. That is, Idea2Img could broadly boost different visual generation models of diverse specialties by exploring their optimal text description or instruction prompts. Beyond a single generation model, Idea2Img can also be used to allocate multiple tools as in multimodal agent studies [52,58]. In this case, Idea2Img isn’t limited to optimizing the use of individual tools but also investigates their effective collaboration when used together, such as generator selection and multi-step visual generation.

Consolidating explored knowledge. We have shown the effectiveness of LMM iterative self-refinement in automatic image design and generation. Idea2Img can also help to consolidate or distill the explored knowledge into T2I model parameters, such that no inference-time iterative refinement is needed when encountering seen generation scenarios. One could collect a dataset using Idea2Img for a scenario of interest, and fine-tune a T2I model with the explored selfrefinement trajectory. Storing the probed knowledge as sample-agnostic prompt for each image generation model is another promising direction [15,37,66]. Finally, with minimal extra computation, we find it helpful to use the explored T2I prompt history as in-context examples for prompt re-writing and expansion, improving from the zero-shot expansion like the one in ChatGPT-Dalle-3 [1,4].

### 6 Conclusion

We have presented Idea2Img, a multimodal iterative self-refinement framework that leverages GPT-4V(ision) for image design and generation. Idea2Img explores the emergent capabilities of iterative self-refinement in LMM-based agent systems, showcasing its effectiveness in improving, assessing, and verifying the generated multimodal content. The user preference study demonstrates Idea2Img’s capability in assisting humans to find the optimal usage of generation models for automatic image design and generation.

#### Acknowledgment

We are deeply grateful to OpenAI for providing access to their exceptional tool [1, 31–33]. We also extend heartfelt thanks to our Microsoft colleagues for their insights, with special acknowledgment to Faisal Ahmed, Ehsan Azarnasab, and Lin Liang for their constructive feedback.

In this supplementary material, we begin with showing additional qualitative results in Section A.1, in supporting Idea2Img’s effectiveness on different image generation models, including Dalle-3 [4,30], SDXL [36], SDXL-img2img [28,36], DeepFloyd IF [20], among others. In Section A.2, we show GPT-4V’s outputs to probe how Idea2Img helps image creation during the iterative self-refinement, and the possibility of replacing GPT-4V with other LMMs. Section B introduce remaining implementation details.

### A Qualitative Results

#### A.1 Qualitative Comparisons

Figures A-D show additional qualitative results of the comparison in Table 1. Figure A presents examples of Idea2Img explores the use of SDXL, a representative T2I model. Figure B examines SDXL-img2img, a simple text-conditioned image-to-image model that adds noise to the input image and then performs text-conditioned denoising [28]. Figures C, D contain the results of Idea2Img working with Dalle-3 and other image generation models.

SDXL. Idea2Img could help users generate images that better follow IDEA, such as the one with correct object counts and rendered scene texts in Figures A(a,b). Idea2Img enables the visual content design that can create images from a text instruction of its desired usage, in contrast to the detailed image description required in the conventional T2I generation. For example in Figure A(c), Idea2Img designs a logo based on the user IDEA of “having a logo for a 2024 conference in Seattle.” Idea2Img can also understand user IDEA to search for images with high aesthetic scores and great visual details, or its opposite direction with “minimal face details” in (d). The LMM allows Idea2Img to extract arbitrary information from the input image for visual generation. This could be any specific object in the image, such as “the dog on the left” or “the dog pointed to via a red circle” in (e). Figure A(f) shows an example of extracting the painting style, which requires art knowledge for humans to describe accurately. The image input can even be an in-context example that defines the desired image transformation, such as the visual style transfer shown in (g). The ability to extract arbitrary information from the input image can be seamlessly extended to compose multiple visual and text conditions, such as composing the camera angle and image style in (h) and the two entities in (I).

SDXL-img2img. Idea2Img is also effective in finding T2I prompts for the textconditioned image-to-image model SDXL-img2img, as shown in Figure B. Figures B(c) and (d) illustrate generating images that follow and differ from the

- (b)
- (c)
- (d)
- (e)
- (f)
- (g)

Find the the image style pattern in the left two dog images and apply it on the top right people waving hand image. Provide a textual description that keeps the content in the people waving hand image, with the correct style pattern.

- (h)
- (i)

Initial-round Idea2Img prompt

Iterative self-refined Idea2Img prompt

Input user IDEA Initial-round manual prompt

[Figure 199] for Automatic Image Design and Gener_images/imageFile199.png>)

[Figure 200] for Automatic Image Design and Gener_images/imageFile200.png>)

[Figure 201] for Automatic Image Design and Gener_images/imageFile201.png>)

|[Figure 202] for Automatic Image Design and Gener_images/imageFile202.png>)<br><br>5 people sitting around a table drinking beer and eating buffalo wings|
|---|

|[Figure 203] for Automatic Image Design and Gener_images/imageFile203.png>)<br><br>A group of 5 friends sitting around a wooden table, drinking beer and eating buffalo wings in a casual setting.|
|---|

|[Figure 204] for Automatic Image Design and Gener_images/imageFile204.png>)<br><br>A group of 5 friends sitting around a wooden table, with one person at the head of the table and two people on each side, drinking beer and eating buffalo wings in a casual setting, with a window in the background and a warm, inviting atmosphere.|
|---|

(a) (a)

5 people sitting around a table drinking beer and eating buffalo wings

(a)

[Figure 205] for Automatic Image Design and Gener_images/imageFile205.png>)

[Figure 206] for Automatic Image Design and Gener_images/imageFile206.png>)

[Figure 207] for Automatic Image Design and Gener_images/imageFile207.png>)

[Figure 208] for Automatic Image Design and Gener_images/imageFile208.png>)

|[Figure 209] for Automatic Image Design and Gener_images/imageFile209.png>)<br><br>a whole cake on the table with words Azure Research written on the cake|
|---|

|[Figure 210] for Automatic Image Design and Gener_images/imageFile210.png>)<br><br>A whole cake on a wooden table with the words Azure Research written on it in blue icing, with a white tablecloth and a vase of flowers in the background.|
|---|

|[Figure 211] for Automatic Image Design and Gener_images/imageFile211.png>)<br><br>A top-down perspective of a cake on a table, with the words "Azure Research" meticulously written in blue icing on the top, surrounded by a modern kitchen with sunlight filtering through.|
|---|

(b)

a whole cake on the table with words Azure Research written on the cake

[Figure 212] for Automatic Image Design and Gener_images/imageFile212.png>)

[Figure 213] for Automatic Image Design and Gener_images/imageFile213.png>)

[Figure 214] for Automatic Image Design and Gener_images/imageFile214.png>)

|[Figure 215] for Automatic Image Design and Gener_images/imageFile215.png>)<br><br>a logo for a 2024 conference held in Seattle.|
|---|

|[Figure 216] for Automatic Image Design and Gener_images/imageFile216.png>)<br><br>A logo design for a 2024 conference held in Seattle, featuring the Seattle skyline and the conference title in a modern font, with a blue and green color scheme.|
|---|

|[Figure 217] for Automatic Image Design and Gener_images/imageFile217.png>)<br><br>A logo design for a 2024 conference held in Seattle, with a modern and minimalist design, featuring the Seattle skyline and the conference title in a geometric sans serif font, with a blue and green color scheme.|
|---|

(c)

a logo for a 2024 conference held in Seattle. Please provide a textual description of the design before generating

[Figure 218] for Automatic Image Design and Gener_images/imageFile218.png>)

[Figure 219] for Automatic Image Design and Gener_images/imageFile219.png>)

[Figure 220] for Automatic Image Design and Gener_images/imageFile220.png>)

|[Figure 221] for Automatic Image Design and Gener_images/imageFile221.png>)<br><br>A vague portrait of Bill Gates with minimal visual and face details|
|---|

|[Figure 222] for Automatic Image Design and Gener_images/imageFile222.png>)<br><br>A portrait of Bill Gates with minimalistic style and vague facial features, in a monochrome color scheme.|
|---|

|[Figure 223] for Automatic Image Design and Gener_images/imageFile223.png>)<br><br>A portrait of Bill Gates with an extremely abstract and vague style, with almost no emphasis on the facial features, in a monochrome color scheme, with a completely abstract background.|
|---|

(d)

A vague portrait of Bill Gates with minimal visual and face details

[Figure 224] for Automatic Image Design and Gener_images/imageFile224.png>)

[Figure 225] for Automatic Image Design and Gener_images/imageFile225.png>)

[Figure 226] for Automatic Image Design and Gener_images/imageFile226.png>)

|[Figure 227] for Automatic Image Design and Gener_images/imageFile227.png>)<br><br>photo of a dog looks like the circled one in the image running on the floor that is a black and white dog|
|---|

|[Figure 228] for Automatic Image Design and Gener_images/imageFile228.png>)<br><br>A Cavalier King Charles Spaniel running on a tiled floor, happy mood, bright lighting|
|---|

|[Figure 229] for Automatic Image Design and Gener_images/imageFile229.png>)<br><br>A small Cavalier King Charles Spaniel with black and white fur, running on a tiled floor, tongue out, happy mood, bright lighting|
|---|

(e)

[Figure 230] for Automatic Image Design and Gener_images/imageFile230.png>)

photo of a dog looks like the circled one in the image running on the floor

[Figure 231] for Automatic Image Design and Gener_images/imageFile231.png>)

[Figure 232] for Automatic Image Design and Gener_images/imageFile232.png>)

[Figure 233] for Automatic Image Design and Gener_images/imageFile233.png>)

|[Figure 234] for Automatic Image Design and Gener_images/imageFile234.png>)<br><br>Painting of Mt rainier mountain with tree and lake in the foreground, with style similar to this one in the image that has colorful blocks|
|---|

|[Figure 235] for Automatic Image Design and Gener_images/imageFile235.png>)<br><br>A painting of Mt Rainier mountain with a tree and lake in the foreground, in a colorful and abstract style similar to a Picasso painting.|
|---|

|[Figure 236] for Automatic Image Design and Gener_images/imageFile236.png>)<br><br>A painting of Mt Rainier mountain with a tree and lake in the foreground, with bold lines and bright colors, with a cubist and surrealistic style, with a focus on geometric shapes and patterns.|
|---|

(f)

[Figure 237] for Automatic Image Design and Gener_images/imageFile237.png>)

Painting of Mt rainier mountain with tree and lake in the foreground, with style similar to this one in the image

[Figure 238] for Automatic Image Design and Gener_images/imageFile238.png>)

[Figure 239] for Automatic Image Design and Gener_images/imageFile239.png>)

[Figure 240] for Automatic Image Design and Gener_images/imageFile240.png>)

[Figure 241] for Automatic Image Design and Gener_images/imageFile241.png>)

|[Figure 242] for Automatic Image Design and Gener_images/imageFile242.png>)<br><br>Find the the image style pattern in the left two dog images and apply it on the top right people waving hand image.|
|---|

|[Figure 243] for Automatic Image Design and Gener_images/imageFile243.png>)<br><br>A scene of a man in a suit waving his hand, with the same style as the geometric dog image on the left.|
|---|

|[Figure 244] for Automatic Image Design and Gener_images/imageFile244.png>)<br><br>A person in a business suit waving with his right hand, depicted in a polygonal art style reminiscent of the dog image, with a mosaic of colorful geometric shapes, against a backdrop of a blue sky.|
|---|

(g)

[Figure 245] for Automatic Image Design and Gener_images/imageFile245.png>)

[Figure 246] for Automatic Image Design and Gener_images/imageFile246.png>)

[Figure 247] for Automatic Image Design and Gener_images/imageFile247.png>)

[Figure 248] for Automatic Image Design and Gener_images/imageFile248.png>)

[Figure 249] for Automatic Image Design and Gener_images/imageFile249.png>)

|[Figure 250] for Automatic Image Design and Gener_images/imageFile250.png>)<br><br>A painting of a tennis game from camera angle, with the image style similar to this one in the image that is impressionist|
|---|

|[Figure 251] for Automatic Image Design and Gener_images/imageFile251.png>)<br><br>A painting of a tennis game from a top-down camera angle, with the image style similar to Claude Monet's impressionist paintings, with a bright and sunny atmosphere.|
|---|

|[Figure 252] for Automatic Image Design and Gener_images/imageFile252.png>)<br><br>A painting of a tennis game from a top-down camera angle, with the image style similar to Claude Monet's impressionist paintings, with a bright and sunny atmosphere, with a blue sky and green trees in the background, with the players wearing white clothes, with a large crowd in the stands.|
|---|

(h)

[Figure 253] for Automatic Image Design and Gener_images/imageFile253.png>)

[Figure 254] for Automatic Image Design and Gener_images/imageFile254.png>)

, with the image style similar to this one in the image

A painting of a tennis game from camera angle

[Figure 255] for Automatic Image Design and Gener_images/imageFile255.png>)

[Figure 256] for Automatic Image Design and Gener_images/imageFile256.png>)

[Figure 257] for Automatic Image Design and Gener_images/imageFile257.png>)

[Figure 258] for Automatic Image Design and Gener_images/imageFile258.png>)

[Figure 259] for Automatic Image Design and Gener_images/imageFile259.png>)

|[Figure 260] for Automatic Image Design and Gener_images/imageFile260.png>)<br><br>Photo of Bill Gates with the same hand gesture as in the given image that is waving hand, with a dog looks like this one in the image that is a pug dog|
|---|

|[Figure 261] for Automatic Image Design and Gener_images/imageFile261.png>)<br><br>Bill Gates in a formal suit on a bustling city street, raising his hand in a friendly wave like a charming businessman, with a cute pug dog with a wrinkled face and large eyes by his side, under natural daylight, with a sense of approachability.|
|---|

|[Figure 262] for Automatic Image Design and Gener_images/imageFile262.png>)<br><br>Bill Gates wearing a suit and tie, standing on a busy street with tall buildings, waving with his right hand raised and palm facing forward, with a small pug dog with an adorably wrinkled face and big round eyes next to him, in daylight, exuding friendliness and warmth.|
|---|

(i)

[Figure 263] for Automatic Image Design and Gener_images/imageFile263.png>)

[Figure 264] for Automatic Image Design and Gener_images/imageFile264.png>)

Photo of Bill Gates with the same hand gesture as in the given image

with a dog looks like this one in the image

###### Fig. A: The comparisons among the initial-round manual prompts, initial-round Idea2Img prompts, and the iterative self-refined Idea2Img prompts, with the SDXL v1.0 [36] used as the T2I model.

[Figure 265] for Automatic Image Design and Gener_images/imageFile265.png>)

A scene of a man in a suit with his right hand raised in a friendly wave with his palm facing forward, with the same style as the geometric dog image on the left, with a blue sky background, with a slight smile on his face, with a modern and abstract style.

[Figure 266] for Automatic Image Design and Gener_images/imageFile266.png>)

Bill Gates in a suit, standing in front of a building, with a pug dog sitting next to him, waving his hand in a friendly manner

[Figure 267] for Automatic Image Design and Gener_images/imageFile267.png>)

[Figure 268] for Automatic Image Design and Gener_images/imageFile268.png>)

Bill Gates in a suit, standing in front of a building, with a pug dog sitting on the ground next to his legs, waving his hand in a friendly manner

Initial-round Idea2Img prompt

Iterative self-refined Idea2Img prompt

Input user IDEA Initial-round manual prompt

|[Figure 269] for Automatic Image Design and Gener_images/imageFile269.png>)<br><br>[Figure 270] for Automatic Image Design and Gener_images/imageFile270.png>)|
|---|

[Figure 271] for Automatic Image Design and Gener_images/imageFile271.png>)

[Figure 272] for Automatic Image Design and Gener_images/imageFile272.png>)

|[Figure 273] for Automatic Image Design and Gener_images/imageFile273.png>)<br><br>photo of a dog looks like the one in the given image running on the beach that is a fluffy golden dog|
|---|

|[Figure 274] for Automatic Image Design and Gener_images/imageFile274.png>)<br><br>A small dog with a fluffy coat running on the beach, with the ocean in the background.|
|---|

|[Figure 275] for Automatic Image Design and Gener_images/imageFile275.png>)<br><br>A playful Yorkshire Terrier with a collar running on the beach, with the sun setting over the ocean in the background.|
|---|

- (a)

[Figure 276] for Automatic Image Design and Gener_images/imageFile276.png>)

photo of a dog looks like the one in the given image running on the beach

[Figure 277] for Automatic Image Design and Gener_images/imageFile277.png>)

[Figure 278] for Automatic Image Design and Gener_images/imageFile278.png>)

[Figure 279] for Automatic Image Design and Gener_images/imageFile279.png>)

|[Figure 280] for Automatic Image Design and Gener_images/imageFile280.png>)<br><br>A watercolor cartoon logo based on the design in the given image that is a llama|
|---|

|[Figure 281] for Automatic Image Design and Gener_images/imageFile281.png>)<br><br>A logo design of a llama with a red scarf and beanie, in a watercolor cartoon style, with a soft and pastel color palette.|
|---|

|[Figure 282] for Automatic Image Design and Gener_images/imageFile282.png>)<br><br>A logo design of a llama standing on all fours, with a red scarf and beanie, in a watercolor cartoon style, with a soft and pastel color palette, similar to the design in the given image.|
|---|

- (b)

[Figure 283] for Automatic Image Design and Gener_images/imageFile283.png>)

A watercolor cartoon logo based on the design in the given image

[Figure 284] for Automatic Image Design and Gener_images/imageFile284.png>)

[Figure 285] for Automatic Image Design and Gener_images/imageFile285.png>)

[Figure 286] for Automatic Image Design and Gener_images/imageFile286.png>)

|[Figure 287] for Automatic Image Design and Gener_images/imageFile287.png>)<br><br>a watercolor painting of the same house with the same number of windows and the frontal view as the one in the sketch that is a two-level house with four large windows and three small windows in the center|
|---|

|[Figure 288] for Automatic Image Design and Gener_images/imageFile288.png>)<br><br>A watercolor rendition of a house with a frontal view and six windows, with a focus on the architectural details|
|---|

|[Figure 289] for Automatic Image Design and Gener_images/imageFile289.png>)<br><br>A watercolor illustration of a house with a porch and six windows, with an emphasis on the symmetry and style of the house in the sketch.|
|---|

(c)

[Figure 290] for Automatic Image Design and Gener_images/imageFile290.png>)

a watercolor painting of the same house with the same number of windows and the frontal view as the one in the sketch

[Figure 291] for Automatic Image Design and Gener_images/imageFile291.png>)

[Figure 292] for Automatic Image Design and Gener_images/imageFile292.png>)

[Figure 293] for Automatic Image Design and Gener_images/imageFile293.png>)

|[Figure 294] for Automatic Image Design and Gener_images/imageFile294.png>)<br><br>portrait of Bill Gates with style different from this one in the image that is impressionist|
|---|

|[Figure 295] for Automatic Image Design and Gener_images/imageFile295.png>)<br><br>A portrait of Bill Gates in a cartoon style, with bright colors and a cheerful mood.|
|---|

|[Figure 296] for Automatic Image Design and Gener_images/imageFile296.png>)<br><br>A portrait of Bill Gates in a cartoon style, with bright colors and a cheerful mood, with a light-colored background to make the portrait stand out.|
|---|

- (d)

[Figure 297] for Automatic Image Design and Gener_images/imageFile297.png>)

portrait of Bill Gates with style different from this one in the image

|[Figure 298] for Automatic Image Design and Gener_images/imageFile298.png>)<br><br>[Figure 299] for Automatic Image Design and Gener_images/imageFile299.png>)|
|---|

[Figure 300] for Automatic Image Design and Gener_images/imageFile300.png>)

[Figure 301] for Automatic Image Design and Gener_images/imageFile301.png>)

|[Figure 302] for Automatic Image Design and Gener_images/imageFile302.png>)<br><br>a dog looks like this one in the image running on the grass, but in blue color . The dog is pug dog, with the image style similar to this one in the image that contains blocks|
|---|

|[Figure 303] for Automatic Image Design and Gener_images/imageFile303.png>)<br><br>A pug dog with a blue coat running on a field of green grass, with a style reminiscent of cubism.|
|---|

|[Figure 304] for Automatic Image Design and Gener_images/imageFile304.png>)<br><br>A pug dog with a blue coat running on a field of green grass, with its legs stretched out and its tail wagging as it runs, in a style reminiscent of cubism.|
|---|

- (e)

, with the image style similar to this one in the image

[Figure 305] for Automatic Image Design and Gener_images/imageFile305.png>)

[Figure 306] for Automatic Image Design and Gener_images/imageFile306.png>)

a dog looks like this one in the image running on the grass, but in blue color

[Figure 307] for Automatic Image Design and Gener_images/imageFile307.png>)

[Figure 308] for Automatic Image Design and Gener_images/imageFile308.png>)

[Figure 309] for Automatic Image Design and Gener_images/imageFile309.png>)

[Figure 310] for Automatic Image Design and Gener_images/imageFile310.png>)

|[Figure 311] for Automatic Image Design and Gener_images/imageFile311.png>)<br><br>Cartoon drawing with a design that naturally blends the two given images as a new minimalist drawing. Please provide a textual description of the design before generating. The first image is a llama logo, and the second image is a person in suit|
|---|

|[Figure 312] for Automatic Image Design and Gener_images/imageFile312.png>)<br><br>A cartoon drawing of a llama wearing a suit and waving, blending the minimalist style of the first image with the professional attire of the second image.|
|---|

|[Figure 313] for Automatic Image Design and Gener_images/imageFile313.png>)<br><br>A whimsical cartoon of a llama with a human body dressed in a stylish suit, waving its hand, combining the charm of a cute llama drawing and the elegance of a man in a suit, in a minimalist style, with a muted color palette.|
|---|

(f)

Cartoon drawing with a design that naturally blends the two given images as a new minimalist drawing. Please provide a textual description of the design before generating. The first image is

[Figure 314] for Automatic Image Design and Gener_images/imageFile314.png>)

[Figure 315] for Automatic Image Design and Gener_images/imageFile315.png>)

, and the second image is

- (g)

[Figure 316] for Automatic Image Design and Gener_images/imageFile316.png>)

[Figure 317] for Automatic Image Design and Gener_images/imageFile317.png>)

[Figure 318] for Automatic Image Design and Gener_images/imageFile318.png>)

|[Figure 319] for Automatic Image Design and Gener_images/imageFile319.png>)<br><br>photo with a design that composites the two given objects into the new photo. The placement of the objects does not adhere to the laws of physics. Please provide a textual description of the objects and design before generating. The first object is a marble table, and the second object is a coke|
|---|

|[Figure 320] for Automatic Image Design and Gener_images/imageFile320.png>)<br><br>A surreal scene of a Coca-Cola bottle and a marble table merging into one, in an abstract space with distorted dimensions, bright colors, and dynamic lighting.|
|---|

|[Figure 321] for Automatic Image Design and Gener_images/imageFile321.png>)<br><br>An imaginative scene where a Coca-Cola bottle is floating and twisting above a marble table, defying gravity, with the bottle and table merging into each other in places, surrounded by a kaleidoscope of colors and ethereal lighting that gives a dreamlike atmosphere.|
|---|

photo with a design that composites the two given objects into the new photo. The placement of the objects does not adhere to the laws of physics. Please provide a textual description of the objects and design before generating. The first object is

[Figure 322] for Automatic Image Design and Gener_images/imageFile322.png>)

[Figure 323] for Automatic Image Design and Gener_images/imageFile323.png>)

, and the second object is

[Figure 324] for Automatic Image Design and Gener_images/imageFile324.png>)

[Figure 325] for Automatic Image Design and Gener_images/imageFile325.png>)

[Figure 326] for Automatic Image Design and Gener_images/imageFile326.png>)

|[Figure 327] for Automatic Image Design and Gener_images/imageFile327.png>)<br><br>image that depicts what will happen next based on the story in the two given frames. Please provide a textual description of the image before generating. The first frame in the story is rice on sushi roll, and the second frame in the story is ingredients on rice|
|---|

|[Figure 328] for Automatic Image Design and Gener_images/imageFile328.png>)<br><br>A close-up of a sushi roll in the making, with a pile of rice on a seaweed sheet and salmon and cucumber being added, next step is rolling and cutting the sushi.|
|---|

|[Figure 329] for Automatic Image Design and Gener_images/imageFile329.png>)<br><br>A close-up of a sushi roll in the making, with a pile of rice on a seaweed sheet and salmon and cucumber being added, next step is rolling and cutting the sushi, with a bamboo mat and knife nearby, in a kitchen setting.|
|---|

- (h)

image that depicts what will happen next based on the story in the two given frames. Please provide a textual description of the image before generating. The first frame in the story is

[Figure 330] for Automatic Image Design and Gener_images/imageFile330.png>)

[Figure 331] for Automatic Image Design and Gener_images/imageFile331.png>)

, and the second frame in the story is

|[Figure 332] for Automatic Image Design and Gener_images/imageFile332.png>)<br><br>[Figure 333] for Automatic Image Design and Gener_images/imageFile333.png>)|
|---|

|[Figure 334] for Automatic Image Design and Gener_images/imageFile334.png>)<br><br>[Figure 335] for Automatic Image Design and Gener_images/imageFile335.png>)|
|---|

[Figure 336] for Automatic Image Design and Gener_images/imageFile336.png>)

|[Figure 337] for Automatic Image Design and Gener_images/imageFile337.png>)<br><br>Two cats running on the beach, the right one looks like a gray cat and the left one looks like a orange cat|
|---|

|[Figure 338] for Automatic Image Design and Gener_images/imageFile338.png>)<br><br>A gray cat and an orange cat running on the beach, with the gray cat on the right and the orange cat on the left.|
|---|

|[Figure 339] for Automatic Image Design and Gener_images/imageFile339.png>)<br><br>A gray cat with its head down and eyes looking forward and an orange cat with its eyes closed, both running on a beach with the gray cat on the right and the orange cat on the left, with the ocean in the background.|
|---|

(i)

[Figure 340] for Automatic Image Design and Gener_images/imageFile340.png>)

[Figure 341] for Automatic Image Design and Gener_images/imageFile341.png>)

and the left one looks like

Two cats running on the beach, the right one looks like

Fig. B: The comparisons among the initial-round manual prompts, initial-round Idea2Img prompts, and the iterative self-refined Idea2Img prompts, with the SDXLimg2img [28, 36] used as the image generation model. Instead of random noise, the image generation starts from the input image with added noise [28], showing the effectiveness of Idea2Img on text-conditioned image-to-image pipelines.

[Figure 342] for Automatic Image Design and Gener_images/imageFile342.png>)

A cartoon drawing of a llama wearing a gray suit with a black tie, with his right hand raised in a friendly wave with his palm facing forward, blending the minimalist style of the first image with the professional attire of the second image.

photo with a design that naturally blends the two given objects into the new photo. The placement of the objects perfectly adhere to the laws of physics and is common in real life. Please provide a textual description of the objects and design before generating. The first object is

[Figure 343] for Automatic Image Design and Gener_images/imageFile343.png>)

, and the second object is

A painting of a tennis game from a top-down camera angle, with the image style similar to Claude Monet's impressionist paintings, with a bright and sunny atmosphere, with a blue sky and green trees in the background, with the players wearing white clothes, with a large crowd in the stands.

A group of 5 friends sitting around a wooden table, with one person at the head of the table and two people on each side, drinking beer and eating buffalo wings in a casual setting, with a window in background and a warm, inviting atmosphere.

photo with a design that naturally blends the two given objects into the new photo. The placement of the objects perfectly adhere to the laws of physics and is common in real life. The first object is a marble table, and the second object is a coke

[Figure 350] for Automatic Image Design and Gener_images/imageFile350.png>)

- (f)

- (e)

(d)

- (c)

- (b)

|[Figure 351] for Automatic Image Design and Gener_images/imageFile351.png>)<br><br>On a peaceful beach, a silhouette of a person is captured in the midst of performing a tree pose yoga, with the backdrop of a setting sun and a large tree. The scene is reimagined with a geometric, polygonal art style, echoing the aesthetic of a stylized, faceted dog, where each shape captures and reflects the light differently, creating a mosaic of sunset hues.|
|---|

Find the the image style pattern in the left two dog images and apply it on the top right people in tree pose image. Provide a textual description that keeps the content in the people in tree pose image, with the correct style pattern.

|[Figure 352] for Automatic Image Design and Gener_images/imageFile352.png>)<br><br>Find the the image style pattern in the left two dog images and apply it on the top right people in tree pose image|
|---|

|[Figure 353] for Automatic Image Design and Gener_images/imageFile353.png>)<br><br>Amidst the gentle waves lapping at the shore and the sun bidding adieu, a person is encapsulated in the serene act of performing the tree pose yoga. A tree, grand and robust, shares the spotlight, its leaves whispering tales of nature’s splendor. The scene is reimagined, with every element, from the yogi to the distant horizon, adorned in larger, more distinct geometric shapes, echoing the style of the polygonal dog images. Each shape, bold and less complex, is a canvas capturing the sunset’s myriad hues, crafting a cohesive, visual narrative.|
|---|

|[Figure 354] for Automatic Image Design and Gener_images/imageFile354.png>)<br><br>A dynamic logo showcasing the Space Needle piercing through the number "2024", symbolizing innovation and progress, set against a starry night sky over Seattle, with the city lights shimmering, and the conference details illuminated by a moonlit glow.|
|---|

a logo for a 2024 conference held in Seattle. Please provide a textual description of the design before generating

|[Figure 355] for Automatic Image Design and Gener_images/imageFile355.png>)<br><br>a logo for a 2024 conference held in Seattle. Please provide a textual description of the design before generating|
|---|

|[Figure 356] for Automatic Image Design and Gener_images/imageFile356.png>)<br><br>An elegant logo for the 2024 conference, where the Space Needle forms part of the "2024", adorned with subtle imagery of a speaker's podium and audience, set against a simplified Seattle skyline, with a clean, monochromatic palette, precise detailing, and a soft lighting to accentuate the conference theme and professionalism.|
|---|

|[Figure 357] for Automatic Image Design and Gener_images/imageFile357.png>)<br><br>The image of a man waving is stylized with a low-poly design, where the scene is broken down into geometric shapes filled with bold and vibrant colors, creating a visually striking and contemporary aesthetic, under the bright sky.|
|---|

Find the the image style pattern in the left two dog images and apply it on the top right people waving hand image. Provide a textual description that keeps the content in the people waving hand image, with the correct style pattern.

|[Figure 358] for Automatic Image Design and Gener_images/imageFile358.png>)<br><br>Find the the image style pattern in the left two dog images and apply it on the top right people waving hand image|
|---|

|[Figure 359] for Automatic Image Design and Gener_images/imageFile359.png>)<br><br>A man in a formal suit is captured mid-motion, his hand raised in a wave, each finger distinctly visible. The entire scene is reimagined in a low-poly art style, where each facet of the man and his attire is defined by geometric shapes filled with a palette of bold, contrasting colors. The background remains a solid, untextured expanse, free of additional elements like sun rays or clouds, ensuring the focus remains solely on the colorful, stylized figure of the man waving. The lighting is neutral, casting no shadows or highlights, preserving the simplicity of the scene.|
|---|

|[Figure 360] for Automatic Image Design and Gener_images/imageFile360.png>)<br><br>A minimalist black and white logo of a llama, inspired by the clean lines and simplicity of a stethoscope that forms a paw print, flat design, monochrome.|
|---|

A logo of llama with the exact same style as in the given image

|[Figure 361] for Automatic Image Design and Gener_images/imageFile361.png>)<br><br>A logo of llama with the exact same style as in the given image that has a minimalist design|
|---|

|[Figure 362] for Automatic Image Design and Gener_images/imageFile362.png>)<br><br>A black and white logo where a stethoscope seamlessly forms a llama, the tubing outlines the body and legs, and the earpieces and diaphragm shape the head, akin to the minimalist style of a paw print, no facial features, embodying pure simplicity, elegance, and clean lines.|
|---|

Input user IDEA Initial-round manual prompt

Initial-round Idea2Img prompt

Iterative self-refined Idea2Img prompt

A hand drawing of a person with a clear background. Hand drawing shows the sketch of the scene and looks like the given image

. The person has a pose of

|[Figure 363] for Automatic Image Design and Gener_images/imageFile363.png>)<br><br>A sketch of a person, their hand raised in a greeting, is detailed with intricate lines and shading, showcasing the folds of their suit and their poised stance, set against a clear, unblemished background, in grayscale, the simplicity of the backdrop amplifying the complexity of the hand-drawn subject, with a focus on realism infused with artistic flair.|
|---|

|[Figure 364] for Automatic Image Design and Gener_images/imageFile364.png>)<br><br>A hand drawing of a person with a clear background. Hand drawing shows the sketch of the scene and looks like the given image . The person has a pose of waving hands|
|---|

|[Figure 365] for Automatic Image Design and Gener_images/imageFile365.png>)<br><br>In monochrome, a person is depicted with their hand raised, palm out, in a waving gesture, encapsulated by the spontaneous and rough lines of a hand drawing. The background is immaculate and void of detail, spotlighting the figure. The artwork exudes the energy of a quick sketch, where the focus is on capturing the gesture and posture with immediacy, echoing the unrestrained strokes found in an artist’s sketchbook.|
|---|

|[Figure 366] for Automatic Image Design and Gener_images/imageFile366.png>)<br><br>A portrait of a corgi dog, not utilizing the thick, dynamic brushstrokes and intense color contrasts seen in a scene of a golden wheat field beneath a dramatic, dark sky, soft lighting, hyper-realistic style, serene setting.|
|---|

painting of a corgi dog with style different from this one in the image

|[Figure 367] for Automatic Image Design and Gener_images/imageFile367.png>)<br><br>painting of a corgi dog with style different from this one in the image that is oil painting|
|---|

|[Figure 368] for Automatic Image Design and Gener_images/imageFile368.png>)<br><br>A corgi dog depicted in a pop art style, featuring bold, flat colors and comic book-inspired aesthetics, no wheat field or dramatic sky, set against a backdrop of a bustling cityscape with skyscrapers, neon lights, and street art, illuminated by the glow of streetlights, emphasizing an urban, contemporary atmosphere.|
|---|

|[Figure 369] for Automatic Image Design and Gener_images/imageFile369.png>)<br><br>Two dogs, one with the appearance of a Cavalier King Charles Spaniel with black and white fur and the other resembling a brown and white Yorkshire Terrier, are energetically running side by side on a sandy beach, their tails wagging in excitement. The backdrop features a serene ocean view, clear skies, gentle waves, bright sunlight casting soft shadows, and distant seagulls soaring in the sky, adding a lively, playful atmosphere to the scene.|
|---|

|[Figure 370] for Automatic Image Design and Gener_images/imageFile370.png>)<br><br>Two dogs running on the beach, the left one looks like a fluffy golden dog and the right one looks like the left dog in the image that is black and white|
|---|

|[Figure 371] for Automatic Image Design and Gener_images/imageFile371.png>)<br><br>Two dogs, a Yorkshire Terrier with detailed brown and white fur that glistens in the sunlight, on the left of a black and white Cavalier King Charles Spaniel, are immersed in a game of chase along a scenic beach. The ocean waves, with their rhythmic motion, complement the dogs’ lively energy. The sky, a masterpiece of soft, blended colors, serves as a canvas for the seagulls that glide gracefully, casting intricate shadows that dance on the sandy surface, amplifying the scene’s vibrancy.|
|---|

, with the image style similar to this one in the image

|[Figure 372] for Automatic Image Design and Gener_images/imageFile372.png>)<br><br>An artwork showcasing a tennis game viewed from a high angle, players are frozen in intense play, surrounded by a packed audience, the painting style is akin to impressionism with blurred lines and a play of light and shadow, evoking a sense of movement and energy, in the midst of a clear day.|
|---|

|[Figure 373] for Automatic Image Design and Gener_images/imageFile373.png>)<br><br>A painting of a tennis game from camera angle, with the image style similar to this one in the image that is impressionist|
|---|

|[Figure 374] for Automatic Image Design and Gener_images/imageFile374.png>)<br><br>A vivid impressionist painting captures a dynamic tennis match from an elevated perspective, where athletes are engaged in a fierce competition on a distinct blue court. The stadium is brimming with an enthusiastic crowd, their faces a blend of colors reflecting the excitement of the moment. The artwork is characterized by soft, blurred lines and a harmonious play of light and shadow, reminiscent of a clear, sunny day, enhancing the visual appeal and bringing the scene to life.|
|---|

|[Figure 375] for Automatic Image Design and Gener_images/imageFile375.png>)<br><br>A sleek, modern car parked against a vibrant cityscape, illuminated by the golden hour sunlight, glossy paint reflecting the surrounding lights, clean and polished look.|
|---|

an image of a car that can be used for commercial advertising

|[Figure 376] for Automatic Image Design and Gener_images/imageFile376.png>)<br><br>an image of a car that can be used for commercial advertising|
|---|

|[Figure 377] for Automatic Image Design and Gener_images/imageFile377.png>)<br><br>The image captures a state-of-the-art car positioned elegantly against a modern city backdrop. The lighting is balanced and clear, ensuring the car's sleek design and features are prominently displayed. The surrounding environment, though vibrant, doesn’t overshadow the car, making it the undeniable centerpiece, ideal for commercial advertising.|
|---|

[Figure 378] for Automatic Image Design and Gener_images/imageFile378.png>)

(a)

A painting of a tennis game from camera angle

The lively scene captures a golden and grey Yorkshire Terrier mid-leap, its joyful expression accentuated by the open mouth and bright eyes. The backdrop is a busy street, pedestrians in mid-stride, and colorful storefronts offering a visual feast. The overcast lighting lends a soft glow, illuminating the dog's fur and creating a dynamic interplay of light and shadow on the street.

A portrait of a corgi dog, rendered in a style that mirrors the precision and clarity of a high-resolution photograph, showcasing intricate details from the texture of its fur to the reflections in its eyes, set in a calm outdoor setting, vibrant yet natural colors, under the gentle glow of the afternoon sun.

Five friends, each clutching a distinct beer bottle, are immersed in a spirited discussion around a polished wooden table. A generous serving of buffalo wings, glistening with sauce, commands the center. The room, adorned with vintage decor, basks in the golden hue of hanging lights, casting intricate shadows and highlighting the bonds of friendship.

[Figure 379] for Automatic Image Design and Gener_images/imageFile379.png>)

[Figure 380] for Automatic Image Design and Gener_images/imageFile380.png>)

[Figure 384] for Automatic Image Design and Gener_images/imageFile384.png>)

[Figure 385] for Automatic Image Design and Gener_images/imageFile385.png>)

[Figure 386] for Automatic Image Design and Gener_images/imageFile386.png>)

[Figure 387] for Automatic Image Design and Gener_images/imageFile387.png>)

[Figure 388] for Automatic Image Design and Gener_images/imageFile388.png>)

[Figure 389] for Automatic Image Design and Gener_images/imageFile389.png>)

[Figure 390] for Automatic Image Design and Gener_images/imageFile390.png>)

A simplistic car painted in bold, unshaded primary colors is parked on a uniform green landscape, no gradients, under a plain blue sky with basic cloud shapes, next to a playground with elementary structures painted in solid colors, ensuring an uncomplicated, child-friendly painting experience.

[Figure 392] for Automatic Image Design and Gener_images/imageFile392.png>)

[Figure 393] for Automatic Image Design and Gener_images/imageFile393.png>)

[Figure 394] for Automatic Image Design and Gener_images/imageFile394.png>)

[Figure 395] for Automatic Image Design and Gener_images/imageFile395.png>)

[Figure 396] for Automatic Image Design and Gener_images/imageFile396.png>)

|[Figure 397] for Automatic Image Design and Gener_images/imageFile397.png>)<br><br>[Figure 398] for Automatic Image Design and Gener_images/imageFile398.png>)|
|---|

[Figure 399] for Automatic Image Design and Gener_images/imageFile399.png>)

|[Figure 400] for Automatic Image Design and Gener_images/imageFile400.png>)<br><br>[Figure 401] for Automatic Image Design and Gener_images/imageFile401.png>)|
|---|

[Figure 402] for Automatic Image Design and Gener_images/imageFile402.png>)

[Figure 403] for Automatic Image Design and Gener_images/imageFile403.png>)

|[Figure 404] for Automatic Image Design and Gener_images/imageFile404.png>)<br><br>[Figure 405] for Automatic Image Design and Gener_images/imageFile405.png>)|
|---|

|[Figure 406] for Automatic Image Design and Gener_images/imageFile406.png>)<br><br>[Figure 407] for Automatic Image Design and Gener_images/imageFile407.png>)|
|---|

[Figure 408] for Automatic Image Design and Gener_images/imageFile408.png>)

[Figure 409] for Automatic Image Design and Gener_images/imageFile409.png>)

[Figure 410] for Automatic Image Design and Gener_images/imageFile410.png>)

[Figure 411] for Automatic Image Design and Gener_images/imageFile411.png>)

[Figure 412] for Automatic Image Design and Gener_images/imageFile412.png>)

[Figure 413] for Automatic Image Design and Gener_images/imageFile413.png>)

[Figure 414] for Automatic Image Design and Gener_images/imageFile414.png>)

[Figure 415] for Automatic Image Design and Gener_images/imageFile415.png>)

- (g)

[Figure 416] for Automatic Image Design and Gener_images/imageFile416.png>)

[Figure 417] for Automatic Image Design and Gener_images/imageFile417.png>)

and the right one looks like the left dog in the image

Two dogs running on the beach, the left one looks like

[Figure 418] for Automatic Image Design and Gener_images/imageFile418.png>)

[Figure 419] for Automatic Image Design and Gener_images/imageFile419.png>)

- (h)

[Figure 420] for Automatic Image Design and Gener_images/imageFile420.png>)

[Figure 421] for Automatic Image Design and Gener_images/imageFile421.png>)

[Figure 422] for Automatic Image Design and Gener_images/imageFile422.png>)

[Figure 423] for Automatic Image Design and Gener_images/imageFile423.png>)

- (i)

[Figure 424] for Automatic Image Design and Gener_images/imageFile424.png>)

[Figure 425] for Automatic Image Design and Gener_images/imageFile425.png>)

###### Fig. C: The comparisons among the initial-round manual prompts, initial-round Idea2Img prompts, and the iterative self-refined Idea2Img prompts, with the Dalle3 [30] used as the T2I model.

- (a)

|[Figure 426] for Automatic Image Design and Gener_images/imageFile426.png>)<br><br>A marble table with a Coca-Cola bottle floating above it, defying gravity, in a modern living room, with soft ambient lighting, minimalistic decor, and a hint of surrealism.|
|---|

|[Figure 427] for Automatic Image Design and Gener_images/imageFile427.png>)<br><br>photo with a design that composites the two given objects into the new photo. The placement of the objects does not adhere to the laws of physics. Please provide a textual description of the objects and design before generating. The first object is a marble table, and the second object is a coke|
|---|

|[Figure 428] for Automatic Image Design and Gener_images/imageFile428.png>)<br><br>In a contemporary living room, a Coca-Cola bottle is suspended in mid-air above a marble table, with a mystical aura, warm lighting, and sleek decor, evoking a sense of magic and defying gravity.|
|---|

- SD v1.5

- SD v2.1

DeepFloyd IF

DeepFloyd IF img2img

|[Figure 429] for Automatic Image Design and Gener_images/imageFile429.png>)<br><br>A happy dog with pink fur running on a street with buildings in the background, daytime, sunny.|
|---|

photo of the circled dog but with background changed to a street

|[Figure 430] for Automatic Image Design and Gener_images/imageFile430.png>)<br><br>photo of the circled dog but with background changed to a street that is a pink fluffy dog|
|---|

|[Figure 431] for Automatic Image Design and Gener_images/imageFile431.png>)<br><br>A Yorkshire Terrier with pink fur running on a bustling street with skyscrapers in the background, daytime, bright, traffic.|
|---|

[Figure 432] for Automatic Image Design and Gener_images/imageFile432.png>)

[Figure 433] for Automatic Image Design and Gener_images/imageFile433.png>)

[Figure 434] for Automatic Image Design and Gener_images/imageFile434.png>)

[Figure 435] for Automatic Image Design and Gener_images/imageFile435.png>)

|[Figure 436] for Automatic Image Design and Gener_images/imageFile436.png>)<br><br>A cardboard box with 6 homemade cakes, each with a unique design and flavor, ready to be delivered.|
|---|

|[Figure 437] for Automatic Image Design and Gener_images/imageFile437.png>)<br><br>6 cakes placed in a box|
|---|

|[Figure 438] for Automatic Image Design and Gener_images/imageFile438.png>)<br><br>A brown cardboard box with 6 homemade cakes, each with a different flavor and design, arranged in a symmetrical pattern.|
|---|

[Figure 439] for Automatic Image Design and Gener_images/imageFile439.png>)

[Figure 440] for Automatic Image Design and Gener_images/imageFile440.png>)

[Figure 441] for Automatic Image Design and Gener_images/imageFile441.png>)

|[Figure 442] for Automatic Image Design and Gener_images/imageFile442.png>)<br><br>A casual portrait of Elon Musk and Mark Zuckerberg, both wearing grey sweaters and black shirts, with a white background.|
|---|

generating an image with the same cloth as in the given image, but change the left persons to Elon Musk and the right person to Mark Zuckerberg

|[Figure 443] for Automatic Image Design and Gener_images/imageFile443.png>)<br><br>generating an image with the same cloth as in the given image, but change the left persons to Elon Musk and the right person to Mark Zuckerberg|
|---|

|[Figure 444] for Automatic Image Design and Gener_images/imageFile444.png>)<br><br>Elon Musk and Mark Zuckerberg standing in a relaxed pose, with their shoulders touching, in a grey background, with Elon Musk wearing a grey sweater and Mark Zuckerberg wearing a black shirt.|
|---|

|[Figure 445] for Automatic Image Design and Gener_images/imageFile445.png>)<br><br>A kitchen scene with an empty plate and a glass without orange juice on a wooden table, with a white background.|
|---|

a plate that has no bananas on it. there is a glass without orange juice next to it

|[Figure 446] for Automatic Image Design and Gener_images/imageFile446.png>)<br><br>a plate that has no bananas on it. there is a glass without orange juice next to it|
|---|

|[Figure 447] for Automatic Image Design and Gener_images/imageFile447.png>)<br><br>A close-up view of a wooden table with an empty glass and a white plate without bananas, with a white background.|
|---|

[Figure 448] for Automatic Image Design and Gener_images/imageFile448.png>)

[Figure 449] for Automatic Image Design and Gener_images/imageFile449.png>)

[Figure 450] for Automatic Image Design and Gener_images/imageFile450.png>)

[Figure 451] for Automatic Image Design and Gener_images/imageFile451.png>)

[Figure 452] for Automatic Image Design and Gener_images/imageFile452.png>)

[Figure 453] for Automatic Image Design and Gener_images/imageFile453.png>)

[Figure 454] for Automatic Image Design and Gener_images/imageFile454.png>)

|[Figure 455] for Automatic Image Design and Gener_images/imageFile455.png>)<br><br>A contemporary logo for a 2023 conference in Paris, showcasing a stylized Eiffel Tower, elegant fonts, and a color palette inspired by the French flag, with a touch of sophistication.|
|---|

a logo for a 2023 conference held in Paris. Please provide a textual description of the design before generating

|[Figure 456] for Automatic Image Design and Gener_images/imageFile456.png>)<br><br>a logo for a 2023 conference held in Paris.|
|---|

|[Figure 457] for Automatic Image Design and Gener_images/imageFile457.png>)<br><br>A sleek and modern logo for a 2023 conference in Paris, incorporating a stylized Eiffel Tower, the year '2023' in a classy font, and a tricolor background inspired by the French flag. The design includes the word 'Paris' at the bottom in a refined font, symbolizing sophistication and culture.|
|---|

A logo of llama with the exact same style as in the given image

|[Figure 458] for Automatic Image Design and Gener_images/imageFile458.png>)<br><br>A monochromatic logo that combines a llama's face with a stethoscope, similar to the given image, where the stethoscope forms the face and the footprints are inside, in a sleek and minimalist style, with bold black lines.|
|---|

|[Figure 459] for Automatic Image Design and Gener_images/imageFile459.png>)<br><br>A logo of llama with the exact same style as in the given image that has a minimalist design|
|---|

|[Figure 460] for Automatic Image Design and Gener_images/imageFile460.png>)<br><br>A logo in which a stethoscope is artistically shaped with its tubing outlining a llama's face and its ear tips as the llama's ears, with paw prints inside the face, akin to the given image, in a simple, modern design with bold black contours.|
|---|

|[Figure 461] for Automatic Image Design and Gener_images/imageFile461.png>)<br><br>A playful dog with a unique blue coat and a bright yellow collar, enjoying a run on the beach with the waves crashing behind it.|
|---|

photo of a dog looks like the one in the given image running on the beach, but change the dog color to blue and the collar color to yellow

|[Figure 462] for Automatic Image Design and Gener_images/imageFile462.png>)<br><br>photo of a dog looks like the one in the given image running on the beach, but change the dog color to blue and the collar color to yellow that is a pug dog|
|---|

|[Figure 463] for Automatic Image Design and Gener_images/imageFile463.png>)<br><br>A pug-like dog with a unique blue fur, adorned with a plain yellow collar on its neck, racing across the beach with sand flying under its paws, and a tranquil sea with gentle waves as the backdrop.|
|---|

[Figure 464] for Automatic Image Design and Gener_images/imageFile464.png>)

[Figure 465] for Automatic Image Design and Gener_images/imageFile465.png>)

[Figure 466] for Automatic Image Design and Gener_images/imageFile466.png>)

[Figure 467] for Automatic Image Design and Gener_images/imageFile467.png>)

[Figure 468] for Automatic Image Design and Gener_images/imageFile468.png>)

[Figure 469] for Automatic Image Design and Gener_images/imageFile469.png>)

photo with a design that composites the two given objects into the new photo. The placement of the objects does not adhere to the laws of physics. Please provide a textual description of the objects and design before generating. The first object is

, and the second object is

Input user IDEA Initial-round manual prompt

Initial-round Idea2Img prompt

Iterative self-refined Idea2Img prompt

- (b)

- (c)

- (d)

- (e)

- (f)

- (g)

- (h)

6 cakes placed in a box

|[Figure 470] for Automatic Image Design and Gener_images/imageFile470.png>)<br><br>[Figure 471] for Automatic Image Design and Gener_images/imageFile471.png>)|
|---|

[Figure 472] for Automatic Image Design and Gener_images/imageFile472.png>)

[Figure 473] for Automatic Image Design and Gener_images/imageFile473.png>)

[Figure 477] for Automatic Image Design and Gener_images/imageFile477.png>)

A hand holding an iPhone 12 with the camera app open and the shutter button visible, taking a photo with a blurred background and natural lighting, with a focus on the hand and phone.

A hand holding an iPhone 12 to take a photo, with a blurred background and natural lighting.

an image of a hand holding an iphone 12 to take a photo

an image of a hand holding an iphone 12 to take a photo

[Figure 480] for Automatic Image Design and Gener_images/imageFile480.png>)

[Figure 481] for Automatic Image Design and Gener_images/imageFile481.png>)

[Figure 482] for Automatic Image Design and Gener_images/imageFile482.png>)

[Figure 483] for Automatic Image Design and Gener_images/imageFile483.png>)

[Figure 484] for Automatic Image Design and Gener_images/imageFile484.png>)

[Figure 485] for Automatic Image Design and Gener_images/imageFile485.png>)

[Figure 486] for Automatic Image Design and Gener_images/imageFile486.png>)

[Figure 488] for Automatic Image Design and Gener_images/imageFile488.png>)

[Figure 491] for Automatic Image Design and Gener_images/imageFile491.png>)

[Figure 492] for Automatic Image Design and Gener_images/imageFile492.png>)

[Figure 493] for Automatic Image Design and Gener_images/imageFile493.png>)

photo with a design that composites the two given objects into the new photo. The placement of the objects does not adhere to the laws of physics. Please provide a textual description of the objects and design before generating. The first object is a marble table, and the second object is a coke

A gray cat with its head down and eyes looking forward and an orange cat with its eyes closed, both running on a beach with the gray cat on the right and the orange cat on the left, with the ocean in the background.

A gray cat and an orange cat running on the beach, with the gray cat on the right and the orange cat on the left.

- Fig. D: The comparisons among the initial-round manual prompts, initial-round Idea2Img prompts, and the iterative self-refined Idea2Img prompts, with other image generation models, including SD v1.5, SD v2.1 [40], DeepFloyd IF, and IF-img2img [20].

##### reference image style of “watercolor and impressionist,” respectively. Idea2Img can design visual contents with the inspiration of input images, e.g., a cartoon drawing that blends the llama and the person in suits in (f), and composing the

coke with the table in an imaginative way in (g). (h) illustrates a novel scenario of generating an image to represent the anticipated action of rolling sushi.

Dalle-3 and other generation models. Idea2Img demonstrates its effectiveness across different image generation models. Figure C shows the results generated by Idea2Img with Dalle-3. We access Dalle-3 via Bing Image Creator3, which excludes the ChatGPT prompt rewrite. Idea2Img could better release Dalle-3’s strong prompt-following capability and show impressive results, especially for challenging queries. This includes polishing the logo design in Figure C(a), drafting car advertisements in (b), creating unique image styles in (c), and enhancing the design with reference images in (d). When confronted with more challenging tasks, Idea2Img with Dalle-3 excels. For the visual in-context generation problem in (e) and (f), Idea2Img finds the pattern in the input grid image and explores T2I prompts for the desired image design. The framework also proves effectiveness when handling multiple reference images, such as the two dogs in (g), the hand drawing of a person’s pose in (h), and the tennis game with a queried style in (i).

Furthermore, Figure D shows the Idea2Img results on other T2I models, including SD v1.5, v2.1, DeepFloyd IF, and IF-img2img. Despite the variance in the base T2I models’ capacity, Idea2Img consistently helps design and generate better images.

#### A.2 LMM Feedback, Revision, and Selection

One may wonder how GPT-4V behaves and performs in each role throughout Idea2Img’s iterative self-refinement pipeline, i.e., “feedback reflection,” “revised prompt generation,” and “draft image selection.” We show corresponding qualitative results as follows.

Feedback reflection. Figure E shows text feedback generated by GPT-4V for the user IDEA, draft image, and T2I prompt. Idea2Img can effectively check if the generated image is correct, such as the number of oranges in (a) and the misspelled scene text "ALURE RESEACHE" in (b). In addition to the text descriptions in IDEA, Idea2Img can verify if the draft image corresponds to the visual descriptions in IDEA. This includes the color and breed of the dog in (e), the exact art style in (f), and the same cloth and pose in (g). Furthermore, Idea2Img can understand and verify the interleaved image-text pairs in IDEA, as shown in Figures E(h,i).

In addition to identifying the discrepancy, Idea2Img can also point to the plausible directions for improving the T2I prompt in the text feedback. For example, in Figure E(c), GPT-4V mentions that “the person is not in yoga mountain pose, but the T2I prompt has already mentioned mountain pose,” “the AI model might not understand what mountain pose is, and prompt should be modified to specifically mention what mountain pose is.” Similarly, in Figure E(d), Idea2Img provides guidance to have generated images better follow the user intention of

3 https://www.bing.com/create

[Figure 494] for Automatic Image Design and Gener_images/imageFile494.png>)

|[Figure 495] for Automatic Image Design and Gener_images/imageFile495.png>)|
|---|

|[Figure 496] for Automatic Image Design and Gener_images/imageFile496.png>)|
|---|

[Figure 497] for Automatic Image Design and Gener_images/imageFile497.png>)

|[Figure 498] for Automatic Image Design and Gener_images/imageFile498.png>)|
|---|

|[Figure 499] for Automatic Image Design and Gener_images/imageFile499.png>)|
|---|

|[Figure 500] for Automatic Image Design and Gener_images/imageFile500.png>)|
|---|

[Figure 501] for Automatic Image Design and Gener_images/imageFile501.png>)

|[Figure 502] for Automatic Image Design and Gener_images/imageFile502.png>)|
|---|

|[Figure 503] for Automatic Image Design and Gener_images/imageFile503.png>)|
|---|

[Figure 504] for Automatic Image Design and Gener_images/imageFile504.png>)

|[Figure 505] for Automatic Image Design and Gener_images/imageFile505.png>)|
|---|

|[Figure 506] for Automatic Image Design and Gener_images/imageFile506.png>)|
|---|

[Figure 507] for Automatic Image Design and Gener_images/imageFile507.png>)

|[Figure 508] for Automatic Image Design and Gener_images/imageFile508.png>)|
|---|

|[Figure 509] for Automatic Image Design and Gener_images/imageFile509.png>)|
|---|

[Figure 510] for Automatic Image Design and Gener_images/imageFile510.png>)

[Figure 511] for Automatic Image Design and Gener_images/imageFile511.png>)

|[Figure 512] for Automatic Image Design and Gener_images/imageFile512.png>)|
|---|

|[Figure 513] for Automatic Image Design and Gener_images/imageFile513.png>)|
|---|

[Figure 514] for Automatic Image Design and Gener_images/imageFile514.png>)

[Figure 515] for Automatic Image Design and Gener_images/imageFile515.png>)

|[Figure 516] for Automatic Image Design and Gener_images/imageFile516.png>)|
|---|

|[Figure 517] for Automatic Image Design and Gener_images/imageFile517.png>)|
|---|

[Figure 518] for Automatic Image Design and Gener_images/imageFile518.png>)

[Figure 519] for Automatic Image Design and Gener_images/imageFile519.png>)

|[Figure 520] for Automatic Image Design and Gener_images/imageFile520.png>)|
|---|

|[Figure 521] for Automatic Image Design and Gener_images/imageFile521.png>)|
|---|

[Figure 522] for Automatic Image Design and Gener_images/imageFile522.png>)

[Figure 523] for Automatic Image Design and Gener_images/imageFile523.png>)

[Figure 524] for Automatic Image Design and Gener_images/imageFile524.png>)

|[Figure 525] for Automatic Image Design and Gener_images/imageFile525.png>)|
|---|

|[Figure 526] for Automatic Image Design and Gener_images/imageFile526.png>)|
|---|

[Figure 527] for Automatic Image Design and Gener_images/imageFile527.png>)

[Figure 528] for Automatic Image Design and Gener_images/imageFile528.png>)

###### Fig. E: Examples of the generated text feedback. The left column shows the multimodal input user IDEA, and the center column shows the draft image to process as well as its corresponding text prompts. The right column shows the text feedback generated by GPT-4V. The dark blue color highlights the identified discrepancies.

###### Input user IDEA Draft prompt and image to revise Text feedback

###### Revised prompt and image

[Figure 529] for Automatic Image Design and Gener_images/imageFile529.png>)

[Figure 530] for Automatic Image Design and Gener_images/imageFile530.png>)

[Figure 531] for Automatic Image Design and Gener_images/imageFile531.png>)

|[Figure 532] for Automatic Image Design and Gener_images/imageFile532.png>)<br><br>A person practicing yoga dolphin pose at a beach with clear blue water and white sand, during sunrise.|
|---|

|[Figure 533] for Automatic Image Design and Gener_images/imageFile533.png>)<br><br>The person in the image is not practicing yoga dolphin pose, but in a different yoga pose. Because the prompt has already mention dolphin pose, the AI art model might not understand what is dolphin pose. To address this, the sentence prompt should be modified to specifically mention what is a dolphin pose: posture where one balances on the forearms with the legs lifted, forming an inverted V shape with the body.|
|---|

|[Figure 534] for Automatic Image Design and Gener_images/imageFile534.png>)<br><br>A person on a sandy beach practicing yoga in the dolphin pose, which involves being on both forearms and toes with hips raised towards the ceiling and facing the ground, as the waves gently crash, the sun sets in the background casting a warm glow, and a serene atmosphere surrounds the scene.|
|---|

(a)

a person practicing yoga dolphin pose at beach

[Figure 535] for Automatic Image Design and Gener_images/imageFile535.png>)

[Figure 536] for Automatic Image Design and Gener_images/imageFile536.png>)

[Figure 537] for Automatic Image Design and Gener_images/imageFile537.png>)

[Figure 538] for Automatic Image Design and Gener_images/imageFile538.png>)

|[Figure 539] for Automatic Image Design and Gener_images/imageFile539.png>)<br><br>A breakfast scene with a plate that has no bananas and a glass without orange juice, with a blue background and a bright lighting.|
|---|

|[Figure 540] for Automatic Image Design and Gener_images/imageFile540.png>)<br><br>The image generated by the AI art generation model does not follow the user imagined IDEA of the scene as there are bananas on the plate and orange juice in the glass. To address this, the sentence prompt can be modified to specify that there should be no bananas on the plate and no orange juice in the glass.|
|---|

|[Figure 541] for Automatic Image Design and Gener_images/imageFile541.png>)<br><br>A close-up view of an empty plate with no other objects on it, and an empty glass without orange juice on a wooden table, with a white background.|
|---|

(b)

a plate that has no bananas on it. there is a glass without orange juice next to it

[Figure 542] for Automatic Image Design and Gener_images/imageFile542.png>)

[Figure 543] for Automatic Image Design and Gener_images/imageFile543.png>)

|[Figure 544] for Automatic Image Design and Gener_images/imageFile544.png>)<br><br>A detailed image of an iPhone 12 being held in a hand, with the screen on and the fingers ready to take a screenshot.|
|---|

|[Figure 545] for Automatic Image Design and Gener_images/imageFile545.png>)<br><br>The image generated by the AI art generation model does not follow the user imagined IDEA of the scene as the fingers are not ready to take a screenshot. To address this, the sentence prompt can be modified to specify that the fingers should be pressing the side button and the volume up button at the same time.|
|---|

|[Figure 546] for Automatic Image Design and Gener_images/imageFile546.png>)<br><br>A detailed image of a hand holding an iPhone 12, with the fingers ready to take a screenshot by pressing the side button and the volume up button at the same time, with a clear view of the screen.|
|---|

(c)

an image of a hand holding an iphone 12. The image is used for illustrating how to take a screen shot on iphone

[Figure 547] for Automatic Image Design and Gener_images/imageFile547.png>)

[Figure 548] for Automatic Image Design and Gener_images/imageFile548.png>)

|[Figure 549] for Automatic Image Design and Gener_images/imageFile549.png>)<br><br>A Cavalier King Charles Spaniel running on a tiled floor, with a happy mood, in a modern kitchen.|
|---|

|[Figure 550] for Automatic Image Design and Gener_images/imageFile550.png>)<br><br>A black and white Cavalier King Charles Spaniel running on a wooden floor, with a playful mood, in a cozy kitchen.|
|---|

|[Figure 551] for Automatic Image Design and Gener_images/imageFile551.png>)<br><br>The dog in the generated image is not in the same color as the one in the given image. The dog in the given image is a black and white Cavalier King Charles Spaniel, while the dog in the generated image is a brown and white Cavalier King Charles Spaniel. To address this issue, the sentence prompt should be modified to specifically mention the breed of the dog as a black and white Cavalier King Charles Spaniel.|
|---|

(d)

[Figure 552] for Automatic Image Design and Gener_images/imageFile552.png>)

photo of a dog looks like the left one in the given image running on the floor

[Figure 553] for Automatic Image Design and Gener_images/imageFile553.png>)

[Figure 554] for Automatic Image Design and Gener_images/imageFile554.png>)

|[Figure 555] for Automatic Image Design and Gener_images/imageFile555.png>)<br><br>A logo design of a llama with a red scarf and beanie, in a watercolor cartoon style, with a soft and pastel color palette.|
|---|

|[Figure 556] for Automatic Image Design and Gener_images/imageFile556.png>)<br><br>A logo design of a llama in a watercolor cartoon style, with a red beanie and scarf, standing on all fours, with a soft and pastel color palette, similar to the design in the given image.|
|---|

|[Figure 557] for Automatic Image Design and Gener_images/imageFile557.png>)<br><br>The llama in the generated image is not in the same design as the one in the given image. The llama in the given image is standing on all fours, while the llama in the generated image is only showing its head. To address this issue, the sentence prompt should be modified to specifically mention that the llama should be standing on all fours, and the design should be similar to the one in the given image.|
|---|

(e)

[Figure 558] for Automatic Image Design and Gener_images/imageFile558.png>)

- A watercolor cartoon logo based on the design in the given image

[Figure 559] for Automatic Image Design and Gener_images/imageFile559.png>)

[Figure 560] for Automatic Image Design and Gener_images/imageFile560.png>)

|[Figure 561] for Automatic Image Design and Gener_images/imageFile561.png>)<br><br>A room with a cozy atmosphere, hand-drawn in a style that resembles a pen and ink sketch.|
|---|

|[Figure 562] for Automatic Image Design and Gener_images/imageFile562.png>)<br><br>A room with a comfortable bed and a study area with a desk and a bookshelf, hand-drawn in a style that looks like a pen and ink sketch.|
|---|

|[Figure 563] for Automatic Image Design and Gener_images/imageFile563.png>)<br><br>The image generated by the AI art generation model does not follow the user imagined IDEA of the scene as there is no study area in the image. To address this, the sentence prompt can be modified to specify that there should be a study area in the room, such as a desk with a lamp and a bookshelf.|
|---|

(f)

[Figure 564] for Automatic Image Design and Gener_images/imageFile564.png>)

A hand drawing of a room where people can sleep and study. Hand drawing is a style that looks like the given image.

[Figure 565] for Automatic Image Design and Gener_images/imageFile565.png>)

[Figure 566] for Automatic Image Design and Gener_images/imageFile566.png>)

[Figure 567] for Automatic Image Design and Gener_images/imageFile567.png>)

|[Figure 568] for Automatic Image Design and Gener_images/imageFile568.png>)<br><br>A scene of a man in a suit waving his hand, with the same style as the geometric dog image on the left.|
|---|

|[Figure 569] for Automatic Image Design and Gener_images/imageFile569.png>)<br><br>A scene of a man in a suit with his right hand raised in a friendly wave with his palm facing forward, with the same style as the geometric dog image on the left, with a blue sky background, with a slight smile on his face, with a modern and abstract style.|
|---|

|[Figure 570] for Automatic Image Design and Gener_images/imageFile570.png>)<br><br>The man in the generated image is not the same as the one in the given image. This is because the prompt mentions 'a man' but the AI art model can not understand image inputs. To address this, the sentence prompt should be modified to specifically mention the man in the given image is 'a man in a suit with his right hand raised in a friendly wave with his palm facing forward'.|
|---|

(g)

[Figure 571] for Automatic Image Design and Gener_images/imageFile571.png>)

[Figure 572] for Automatic Image Design and Gener_images/imageFile572.png>)

Find the the image style pattern in the left two dog images and apply it on the top right people waving hand image. Provide a textual description that keeps the content in the people waving hand image, with the correct style pattern.

[Figure 573] for Automatic Image Design and Gener_images/imageFile573.png>)

[Figure 574] for Automatic Image Design and Gener_images/imageFile574.png>)

|[Figure 575] for Automatic Image Design and Gener_images/imageFile575.png>)<br><br>A humorous scene of Mr Bean and a small dog playing fetch on a sandy beach with blue skies in the background.|
|---|

|[Figure 576] for Automatic Image Design and Gener_images/imageFile576.png>)<br><br>The dog in the generated image is not the same breed as the one in the given image. The dog in the given image is a Yorkshire Terrier, while the dog in the generated image is a Boston Terrier. To address this issue, the sentence prompt should be modified to specifically mention the breed of the dog as a Yorkshire Terrier.|
|---|

|[Figure 577] for Automatic Image Design and Gener_images/imageFile577.png>)<br><br>A humorous scene of Mr Bean wearing a yellow tennis outfit and playing<br><br>with a Yorkshire Terrier on a beach with palm trees in the background.|
|---|

(h)

[Figure 578] for Automatic Image Design and Gener_images/imageFile578.png>)

[Figure 579] for Automatic Image Design and Gener_images/imageFile579.png>)

Cartoon drawing of Mr Bean with the same cloth as in the given image

playing with a dog looks like this one on the beach

[Figure 580] for Automatic Image Design and Gener_images/imageFile580.png>)

[Figure 581] for Automatic Image Design and Gener_images/imageFile581.png>)

|[Figure 582] for Automatic Image Design and Gener_images/imageFile582.png>)<br><br>The cats in the generated image are both Russian Blue cats, while the user imagined IDEA of the scene specifies that one of the cats should be a domestic shorthair. To address this issue, the sentence prompt should be modified to specifically mention that the domestic shorthair cat is orange in color.|
|---|

|[Figure 583] for Automatic Image Design and Gener_images/imageFile583.png>)<br><br>A playful scene of a Russian Blue cat and a domestic shorthair cat chasing each other on the beach, the sand is white and the sky is a deep blue, the sun is shining brightly.|
|---|

|[Figure 584] for Automatic Image Design and Gener_images/imageFile584.png>)<br><br>A playful scene of a Russian Blue cat and an orange domestic shorthair cat running on the white sandy beach, the sky is a deep blue and the sun is shining brightly.|
|---|

(i)

[Figure 585] for Automatic Image Design and Gener_images/imageFile585.png>)

[Figure 586] for Automatic Image Design and Gener_images/imageFile586.png>)

Two cats running on the beach, the right one looks like

and the left one looks like

[Figure 587] for Automatic Image Design and Gener_images/imageFile587.png>)

A person in a yoga dolphin pose, posture where one balances on the forearms with the legs lifted, forming an inverted V shape with the body, on a rocky beach with crashing waves and a cloudy sky.

[Figure 588] for Automatic Image Design and Gener_images/imageFile588.png>)

[Figure 589] for Automatic Image Design and Gener_images/imageFile589.png>)

###### A minimalist kitchen with a table showcasing an empty plate and an empty glass,

A breakfast scene with an empty plate and a glass without orange juice, with a blue background and a bright lighting.

devoid of bananas and orange juice respectively, under the soft glow of morning light, evoking a sense of scarcity.

[Figure 590] for Automatic Image Design and Gener_images/imageFile590.png>)

A scene of a man in a suit with his right hand raised in a friendly wave with his palm facing forward, with the same style as the geometric dog image on the left, with a blue sky background, with a slight smile on his face, with a modern and abstract style.

A scene of a man in a suit with his right hand raised in a friendly wave with his palm facing forward, with the same style as the geometric dog image on the left, with a blue sky background.

- Fig. F: Examples of the revised prompts. The four columns, from left to right, show the input user IDEA, the draft image to be revised, generated text feedback, and the revised T2I prompt and image. The dark blue color highlights the identified discrepancies in text feedback, and how they are addressed in the revised T2I prompt. We note that the example only shows a single round of self-refinement. Therefore, the revised T2I prompt may have remaining issues to be further addressed.

|[Figure 592] for Automatic Image Design and Gener_images/imageFile592.png>)|[Figure 593] for Automatic Image Design and Gener_images/imageFile593.png>)<br><br>[Figure 594] for Automatic Image Design and Gener_images/imageFile594.png>)|
|---|---|
|[Figure 595] for Automatic Image Design and Gener_images/imageFile595.png>)<br><br>[Figure 596] for Automatic Image Design and Gener_images/imageFile596.png>)|[Figure 597] for Automatic Image Design and Gener_images/imageFile597.png>)|
|[Figure 598] for Automatic Image Design and Gener_images/imageFile598.png>)|[Figure 599] for Automatic Image Design and Gener_images/imageFile599.png>)<br><br>[Figure 600] for Automatic Image Design and Gener_images/imageFile600.png>)|

[Figure 601] for Automatic Image Design and Gener_images/imageFile601.png>)

|[Figure 602] for Automatic Image Design and Gener_images/imageFile602.png>)|
|---|

[Figure 603] for Automatic Image Design and Gener_images/imageFile603.png>)

|[Figure 604] for Automatic Image Design and Gener_images/imageFile604.png>)|
|---|

[Figure 605] for Automatic Image Design and Gener_images/imageFile605.png>)

|[Figure 606] for Automatic Image Design and Gener_images/imageFile606.png>)|
|---|

[Figure 607] for Automatic Image Design and Gener_images/imageFile607.png>)

[Figure 608] for Automatic Image Design and Gener_images/imageFile608.png>)

[Figure 609] for Automatic Image Design and Gener_images/imageFile609.png>)

[Figure 610] for Automatic Image Design and Gener_images/imageFile610.png>)

|[Figure 611] for Automatic Image Design and Gener_images/imageFile611.png>)|
|---|

[Figure 612] for Automatic Image Design and Gener_images/imageFile612.png>)

[Figure 613] for Automatic Image Design and Gener_images/imageFile613.png>)

[Figure 614] for Automatic Image Design and Gener_images/imageFile614.png>)

[Figure 615] for Automatic Image Design and Gener_images/imageFile615.png>)

[Figure 616] for Automatic Image Design and Gener_images/imageFile616.png>)

|[Figure 617] for Automatic Image Design and Gener_images/imageFile617.png>)|
|---|

[Figure 618] for Automatic Image Design and Gener_images/imageFile618.png>)

[Figure 619] for Automatic Image Design and Gener_images/imageFile619.png>)

|[Figure 620] for Automatic Image Design and Gener_images/imageFile620.png>)|
|---|

[Figure 621] for Automatic Image Design and Gener_images/imageFile621.png>)

[Figure 622] for Automatic Image Design and Gener_images/imageFile622.png>)

[Figure 623] for Automatic Image Design and Gener_images/imageFile623.png>)

[Figure 624] for Automatic Image Design and Gener_images/imageFile624.png>)

[Figure 625] for Automatic Image Design and Gener_images/imageFile625.png>)

[Figure 626] for Automatic Image Design and Gener_images/imageFile626.png>)

[Figure 627] for Automatic Image Design and Gener_images/imageFile627.png>)

[Figure 628] for Automatic Image Design and Gener_images/imageFile628.png>)

|[Figure 629] for Automatic Image Design and Gener_images/imageFile629.png>)|
|---|

[Figure 630] for Automatic Image Design and Gener_images/imageFile630.png>)

[Figure 631] for Automatic Image Design and Gener_images/imageFile631.png>)

[Figure 632] for Automatic Image Design and Gener_images/imageFile632.png>)

[Figure 633] for Automatic Image Design and Gener_images/imageFile633.png>)

[Figure 634] for Automatic Image Design and Gener_images/imageFile634.png>)

[Figure 635] for Automatic Image Design and Gener_images/imageFile635.png>)

|[Figure 636] for Automatic Image Design and Gener_images/imageFile636.png>)|
|---|

[Figure 637] for Automatic Image Design and Gener_images/imageFile637.png>)

[Figure 638] for Automatic Image Design and Gener_images/imageFile638.png>)

[Figure 639] for Automatic Image Design and Gener_images/imageFile639.png>)

[Figure 640] for Automatic Image Design and Gener_images/imageFile640.png>)

[Figure 641] for Automatic Image Design and Gener_images/imageFile641.png>)

[Figure 642] for Automatic Image Design and Gener_images/imageFile642.png>)

|[Figure 643] for Automatic Image Design and Gener_images/imageFile643.png>)|
|---|

[Figure 644] for Automatic Image Design and Gener_images/imageFile644.png>)

[Figure 645] for Automatic Image Design and Gener_images/imageFile645.png>)

###### Fig. G: Examples of the draft image selection. The right column shows justification ratings and the final selection. The dark blue color highlights the identified problems and justifications for the draft image selection. The selected draft image in the round is visualized with the blue box shown in the middle column.

2 reason not correct

[Figure 646] for Automatic Image Design and Gener_images/imageFile646.png>)

[Figure 647] for Automatic Image Design and Gener_images/imageFile647.png>)

|[Figure 648] for Automatic Image Design and Gener_images/imageFile648.png>)|
|---|

[Figure 649] for Automatic Image Design and Gener_images/imageFile649.png>)

|[Figure 650] for Automatic Image Design and Gener_images/imageFile650.png>)|
|---|

[Figure 651] for Automatic Image Design and Gener_images/imageFile651.png>)

[Figure 652] for Automatic Image Design and Gener_images/imageFile652.png>)

|[Figure 653] for Automatic Image Design and Gener_images/imageFile653.png>)|
|---|

[Figure 654] for Automatic Image Design and Gener_images/imageFile654.png>)

|[Figure 655] for Automatic Image Design and Gener_images/imageFile655.png>)|
|---|

- (a) Feedback reflection: The right column shows the examples of the generated text feedback.

|[Figure 656] for Automatic Image Design and Gener_images/imageFile656.png>)|
|---|

|[Figure 657] for Automatic Image Design and Gener_images/imageFile657.png>)|
|---|

|[Figure 658] for Automatic Image Design and Gener_images/imageFile658.png>)|
|---|

[Figure 659] for Automatic Image Design and Gener_images/imageFile659.png>)

[Figure 660] for Automatic Image Design and Gener_images/imageFile660.png>)

[Figure 661] for Automatic Image Design and Gener_images/imageFile661.png>)

|[Figure 662] for Automatic Image Design and Gener_images/imageFile662.png>)|
|---|

|[Figure 663] for Automatic Image Design and Gener_images/imageFile663.png>)|
|---|

|[Figure 664] for Automatic Image Design and Gener_images/imageFile664.png>)|
|---|

[Figure 665] for Automatic Image Design and Gener_images/imageFile665.png>)

[Figure 666] for Automatic Image Design and Gener_images/imageFile666.png>)

[Figure 667] for Automatic Image Design and Gener_images/imageFile667.png>)

[Figure 668] for Automatic Image Design and Gener_images/imageFile668.png>)

[Figure 669] for Automatic Image Design and Gener_images/imageFile669.png>)

[Figure 670] for Automatic Image Design and Gener_images/imageFile670.png>)

[Figure 671] for Automatic Image Design and Gener_images/imageFile671.png>)

- (b) Revised prompt generation: The right column shows the examples of the revised prompts.

- Fig. H: LLaVA-1.5-13B’s [24] outputs in Idea2Img for image feedback and revision.

“an image for a children’s painting competition,” by “specifically mentioning that the car should be simple and cartoon-like.”

Revised prompt generation. Figure F compares the T2I prompts before and after the revision, for visualizing how text feedback helps the revision. For example, (a) the revised T2I prompt includes a detailed description of the “yoga dolphin pose” to generate the correct body pose; (b) the revised T2I prompt mentions “an empty plate with no other objects” to avoid the T2I model misunderstand the prompt “no bananas;” (c) T2I model generates the correct hand gesture with Idea2Img providing text description on how to take a screenshot. Idea2Img also effectively addresses the identified errors in text feedback and improves the prompts for multimodal input IDEA, including the dog color in Figure F(d), the llama design in Figure F(e), the study area in Figure F(f), the human gesture in Figure F(g), the dog breed and human clothing in Figure F(h), and the color of the two cats in Figure F(i).

Draft image selection. T2I models may generate low-quality images even with good T2I prompts. To ensure consistent improvements in each iteration, it is critical to reduce such generation noise by selecting from multiple draft images in each round. Performing such selection requires GPT-4V to compare multiple similar draft images and pick the one with the best overall quality. Figure G shows the selection results generated by GPT-4V. The LMM prompt is designed such that GPT-4V gives justifications and scores for each draft image, in addition to the final selection index. Such intermediate thoughts not only help humans interpret the selection process, but also serve as the chain of thought

to improve the selection performance. We observe that GPT-4V can compare different aspects mentioned in the IDEA and give reasonable scores and selection index. For example, checking the scene text spelling in Figure G(a); verifying the phone screen and model in Figure G(b); counting the number of apples and bananas in Figure G(c); verifying the ball color and dog action in Figure G(d); finding the image with both sleep and study area in Figure G(e); selecting the image that best fits the given image style in Figure G(f); verifying the image content and style in Figure G(g); locating the best-blended image in Figure G(h); and finding the image with correct dog color and image style in Figure G(I).

LMMs alternative to GPT-4V. After observing the effectiveness of Idea2Img with GPT-4V, a natural question is whether we can replace GPT-4V with more accessible and lightweight alternatives. Figure H examines LLaVA-1.513B [24, 25], a leading open-source LMM, using the same test cases as those in the main paper’s Figure 6. Despite the promising results, LMMs alternative to GPT-4V may not be ready yet for the Idea2Img-like iterative self-refinement framework, with two major bottlenecks. First, most current LMMs lack the ability to process complex interleaved image-text sequences, therefore limiting Idea2Img in understanding multimodal IDEA, processing memory, and referencing in-context examples. This limitation also prevents us from conducting image selection experiments in Figure H, as we did in Figure 6(c) with GPT-4V. Second, the weaker multimodal reasoning capability [64] will significantly increase the noise in Idea2Img’s iteration and make the framework ineffective. For example, in Figure H(a.2), LLaVA fails to capture the correct direction to improve the image, and in (b.1), it repeats the same T2I prompt without effective revision.

### B Idea2Img Code, Data, and Gallery

We will release the Idea2Img code, evaluation queries, and generated samples.

We show the used LMM prompts pgen,pselect,pfb,previse as follows. The colored texts indicate the corresponding multimodal contents, such as IDEA or the history memory. LMM prompts are kept the same for different image generation models and input IDEA.

#### Initial prompt generation pgen:

You are a helpful assistant.

Instruction: Given a user imagined IDEA of the scene, converting the IDEA into a self-contained sentence prompt that will be used to generate an image. Here are some rules to write good prompts:

- - Each prompt should consist of a description of the scene followed by modifiers divided by commas.
- - The modifiers should alter the mood, style, lighting, and other aspects of the scene.
- - Multiple modifiers can be used to provide more specific details.

- - When generating prompts, reduce abstract psychological and emotional descriptions.
- - When generating prompts, explain images and unusual entities in IDEA with detailed descriptions of the scene.
- - Do not mention ’given image’ in output, use detailed texts to describe the image in IDEA instead.
- - Generate diverse prompts.
- - Each prompt should have no more than 50 words. IDEA: IDEA input. End of IDEA. Based on the above information, you will write N detailed prompts exactly about the IDEA follow the rules. Each prompt is wrapped with <START> and <END>.

#### Draft image selection pselect:

You are a helpful assistant. You are a judge to rank provided images. Below are N images generated by an AI art generation model, indexed from 0 to N-1. From scale 1 to 10, decide how similar each image is to the user imagined IDEA of the scene. IDEA: IDEA input. End of IDEA. List of draft images.

Let’s think step by step. Check all aspects to see how well these images strictly follow the content in IDEA, including having correct object counts, attributes, entities, relationships, sizes, appearance, and all other descriptions in the IDEA. Then give a score for each input images. Finally, consider the scores and select the image with the best overall quality with image index 0 to N-1 wrapped with <START> and <END>. Only wrap single image index digits between <START> and <END>.

#### Feedback reflection pfb:

You are a helpful assistant.

You are iteratively refining the sentence prompt by analyzing the images produced by an AI art generation model, seeking to find out the differences between the user imagined IDEA of the scene and the actual output. If the generated image is not perfect, provide key REASON on ways to improve the image and sentence prompt to better follow the user imagined IDEA of the scene. Here are some rules to write good key REASON:

- - Carefully compare the current image with the IDEA to strictly follow the details described in the IDEA, including object counts, attributes, entities, relationships, sizes, and appearance. Write down what is different in detail.
- - Avoid hallucinating information or asks that is not mentioned in IDEA.
- - Explain images and unusual entities in IDEA with detailed text descriptions of the scene.
- - Explain how to modify prompts to address the given reflection reason.
- - Focus on one thing to improve in each REASON.
- - Avoid generating REASON identical with the REASON in previous rounds. IDEA: IDEA input. End of IDEA. This is the round t of the iteration. The iteration history are: Memory module history. Based on the above information, you will write REASON that is wrapped with <START> and <END>. REASON:

#### Feedback reflection previse:

You are a helpful assistant. Instruction: Given a user imagined IDEA of the scene, converting the IDEA into a sentence prompt that will be used to generate an image. Here are some rules to write good prompts:

- - Each prompt should consist of a description of the scene followed by modifiers divided by commas.
- - The modifiers should alter the mood, style, lighting, spatial details, and other aspects of the scene.
- - Multiple modifiers can be used to provide more specific details.
- - When generating prompts, reduce abstract psychological and emotional descriptions.
- - When generating prompts, explain images and unusual entities in IDEA with detailed descriptions of the scene.
- - Do not mention ’given image’ in output, use detailed texts to describe the image in IDEA.
- - Generate diverse prompts.
- - Output prompt should have less than 50 words. IDEA: IDEA input. End of IDEA. You are iteratively improving the sentence prompt by looking at the images generated by an AI art generation model and find out what is different from the given IDEA. This is the round t of the iteration. The iteration history are: Memory module history. Generated sentence prompt for current round t is: prompt

Corresponding image generated by the AI art generation model: image However, reflection Based on the above information, to improve the image, you will write N detailed prompts exactly about the IDEA follow the rules. Make description of the scene more detailed and add modifiers to address the given key reasons to improve the image. Avoid generating prompts identical with the ones in previous rounds. Each prompt is wrapped with <START> and <END>.

### References

- 1. Chatgpt can now see, hear, and speak. https://openai.com/blog/chatgpt-cannow-see-hear-and-speak (2023)
- 2. Avrahami, O., Aberman, K., Fried, O., Cohen-Or, D., Lischinski, D.: Break-a-scene: Extracting multiple concepts from a single image. arXiv preprint arXiv:2305.16311

(2023)

- 3. Avrahami, O., Lischinski, D., Fried, O.: Blended diffusion for text-driven editing of natural images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18208–18218 (2022)
- 4. Betker, J., Goh, G., Li, J., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., Manassra, W., Dhariwal, P., Chu, C., Jiao, Y., Ramesh, A.: Improving image generation with better captions (2023)
- 5. Black, K., Janner, M., Du, Y., Kostrikov, I., Levine, S.: Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301 (2023)
- 6. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023)
- 7. Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., Cohen-Or, D.: Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. arXiv preprint arXiv:2301.13826 (2023)
- 8. Chen, W., Hu, H., Li, Y., Rui, N., Jia, X., Chang, M.W., Cohen, W.W.: Subjectdriven text-to-image generation via apprenticeship learning. arXiv preprint arXiv:2304.00186 (2023)
- 9. Chen, X., Lin, M., Schärli, N., Zhou, D.: Teaching large language models to selfdebug. arXiv preprint arXiv:2304.05128 (2023)
- 10. Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., Lee, K.: Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. arXiv preprint arXiv:2305.16381 (2023)
- 11. Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., Lee, K.: Reinforcement learning for fine-tuning textto-image diffusion models. Advances in Neural Information Processing Systems 36

(2024)

- 12. Feng, W., He, X., Fu, T.J., Jampani, V., Akula, A.R., Narayana, P., Basu, S., Wang, X.E., Wang, W.Y.: Training-free structured diffusion guidance for compositional text-to-image synthesis. In: The Eleventh International Conference on Learning Representations (2022)
- 13. Gatys, L.A., Ecker, A.S., Bethge, M.: A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576 (2015)
- 14. Google: Bard. https://bard.google.com (2023), accessed: 2023-07-17

- 15. Guo, Y., Liang, Y., Wu, C., Wu, W., Zhao, D., Duan, N.: Learning to program with natural language. arXiv preprint arXiv:2304.10464 (2023)
- 16. Gupta, T., Kembhavi, A.: Visual programming: Compositional visual reasoning without training. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14953–14962 (2023)
- 17. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-or, D.: Prompt-to-prompt image editing with cross-attention control. In: The Eleventh International Conference on Learning Representations (2022)
- 18. Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., Irani, M.: Imagic: Text-based real image editing with diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6007–6017 (2023)
- 19. Kumari, N., Zhang, B., Zhang, R., Shechtman, E., Zhu, J.Y.: Multi-concept customization of text-to-image diffusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1931–1941 (2023)
- 20. Lab, D.: Deepfloyd if. https://github.com/deep-floyd/IF (2023)
- 21. Lee, K., Liu, H., Ryu, M., Watkins, O., Du, Y., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Gu, S.S.: Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192 (2023)
- 22. Li, C., Gan, Z., Yang, Z., Yang, J., Li, L., Wang, L., Gao, J.: Multimodal foundation models: From specialists to general-purpose assistants. arXiv preprint arXiv:2309.10020 (2023)
- 23. Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: Gligen: Open-set grounded text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22511–22521 (2023)
- 24. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 (2023)
- 25. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. arXiv preprint arXiv:2304.08485 (2023)
- 26. Lu, P., Peng, B., Cheng, H., Galley, M., Chang, K.W., Wu, Y.N., Zhu, S.C., Gao, J.: Chameleon: Plug-and-play compositional reasoning with large language models. arXiv preprint arXiv:2304.09842 (2023)
- 27. Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al.: Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651 (2023)
- 28. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 (2021)
- 29. Nasiriany, S., Xia, F., Yu, W., Xiao, T., Liang, J., Dasgupta, I., Xie, A., Driess, D., Wahid, A., Xu, Z., et al.: Pivot: Iterative visual prompting elicits actionable knowledge for vlms. arXiv preprint arXiv:2402.07872 (2024)
- 30. OpenAI: Dall·e 3 system card. https://cdn.openai.com/papers/DALL_E_3_ System_Card.pdf (2023)
- 31. OpenAI: Gpt-4 technical report (2023)
- 32. OpenAI: Gpt-4v(ision) system card (2023), https://cdn.openai.com/papers/ GPTV_System_Card.pdf
- 33. OpenAI: Gpt-4v(ision) technical work and authors. https://cdn.openai.com/ contributions/gpt-4v.pdf (2023)
- 34. Pan, L., Saxon, M., Xu, W., Nathani, D., Wang, X., Wang, W.Y.: Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies. arXiv preprint arXiv:2308.03188 (2023)

- 35. Paranjape, B., Lundberg, S., Singh, S., Hajishirzi, H., Zettlemoyer, L., Ribeiro, M.T.: Art: Automatic multi-step reasoning and tool-use for large language models. arXiv preprint arXiv:2303.09014 (2023)
- 36. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- 37. Pryzant, R., Iter, D., Li, J., Lee, Y.T., Zhu, C., Zeng, M.: Automatic prompt optimization with" gradient descent" and beam search. arXiv preprint arXiv:2305.03495 (2023)
- 38. Qi, J., Ding, M., Wang, W., Bai, Y., Lv, Q., Hong, W., Xu, B., Hou, L., Li, J., Dong, Y., et al.: Cogcom: Train large vision-language models diving into details through chain of manipulations. arXiv preprint arXiv:2402.04236 (2024)
- 39. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125

(2022)

- 40. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022)
- 41. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22500–22510 (2023)
- 42. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., Ghasemipour, S.K.S., Ayan, B.K., Mahdavi, S.S., Lopes, R.G., et al.: Photorealistic textto-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487 (2022)
- 43. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., Scialom, T.: Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761 (2023)
- 44. Shen, Y., Song, K., Tan, X., Li, D., Lu, W., Zhuang, Y.: Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580

(2023)

- 45. Shi, J., Xiong, W., Lin, Z., Jung, H.J.: Instantbooth: Personalized text-to-image generation without test-time finetuning. arXiv preprint arXiv:2304.03411 (2023)
- 46. Shinn, N., Cassano, F., Labash, B., Gopinath, A., Narasimhan, K., Yao, S.: Reflexion: Language agents with verbal reinforcement learning (2023)
- 47. Shridhar, M., Yuan, X., Côté, M.A., Bisk, Y., Trischler, A., Hausknecht, M.: Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768 (2020)
- 48. Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al.: Make-a-video: Text-to-video generation without textvideo data. arXiv preprint arXiv:2209.14792 (2022)
- 49. Surís, D., Menon, S., Vondrick, C.: Vipergpt: Visual inference via python execution for reasoning. arXiv preprint arXiv:2303.08128 (2023)
- 50. Wang, J., Yang, Z., Hu, X., Li, L., Lin, K., Gan, Z., Liu, Z., Liu, C., Wang, L.: Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100 (2022)
- 51. Wang, Z.J., Montoya, E., Munechika, D., Yang, H., Hoover, B., Chau, D.H.: Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models. arXiv preprint arXiv:2210.14896 (2022)

- 52. Wu, C., Yin, S., Qi, W., Wang, X., Tang, Z., Duan, N.: Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671 (2023)
- 53. Wu, J., Wang, J., Yang, Z., Gan, Z., Liu, Z., Yuan, J., Wang, L.: Grit: A generative region-to-text transformer for object understanding. arXiv preprint arXiv:2212.00280 (2022)
- 54. Wu, P., Xie, S.: V*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135 17 (2023)
- 55. Yan, A., Yang, Z., Zhu, W., Lin, K., Li, L., Wang, J., Yang, J., Zhong, Y., McAuley, J., Gao, J., et al.: Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation. arXiv preprint arXiv:2311.07562 (2023)
- 56. Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q.V., Zhou, D., Chen, X.: Large language models as optimizers. arXiv preprint arXiv:2309.03409 (2023)
- 57. Yang, Z., Li, L., Lin, K., Wang, J., Lin, C.C., Liu, Z., Wang, L.: The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421

(2023)

- 58. Yang*, Z., Li*, L., Wang*, J., Lin*, K., Azarnasab*, E., Ahmed*, F., Liu, Z., Liu, C., Zeng, M., Wang, L.: Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381 (2023)
- 59. Yang, Z., Wang, J., Gan, Z., Li, L., Lin, K., Wu, C., Duan, N., Liu, Z., Liu, C., Zeng, M., et al.: Reco: Region-controlled text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14246–14255 (2023)
- 60. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdinov, R., Manning, C.D.: Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600 (2018)
- 61. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y.: React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629

(2022)

- 62. Yin, S., Wu, C., Yang, H., Wang, J., Wang, X., Ni, M., Yang, Z., Li, L., Liu, S., Yang, F., et al.: Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346 (2023)
- 63. Yu, J., Xu, Y., Koh, J.Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B.K., et al.: Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research (2022)
- 64. Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: Mmvet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490 (2023)
- 65. Zhang, L., Agrawala, M.: Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543 (2023)
- 66. Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.J., Huang, G.: Expel: Llm agents are experiential learners. arXiv preprint arXiv:2308.10144 (2023)
- 67. Zhu, W., Wang, X., Lu, Y., Fu, T.J., Wang, X.E., Eckstein, M., Wang, W.Y.: Collaborative generative ai: Integrating gpt-k for efficient editing in text-to-image generation. arXiv preprint arXiv:2305.11317 (2023)

