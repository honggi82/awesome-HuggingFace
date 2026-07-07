# arXiv:2510.06679v1[cs.CV]8Oct2025

## DREAMOMNI2: MULTIMODAL INSTRUCTION-BASED EDITING AND GENERATION

Bin Xia1,4, Bohao Peng1, Yuechen Zhang1, Junjia Huang4, Jiyang Liu4, Jingyao Li1, Haoru Tan3, Sitong Wu1, Chengyao Wang1, Yitong Wang4, Xinglong Wu4, Bei Yu1, and Jiaya Jia2 1CUHK 2HKUST 3HKU 4ByteDance Inc

https://github.com/dvlab-research/DreamOmni2

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

IMG 1 IMG 2 RES IMG 1 IMG 2

RES IMG 1

RES

[Figure 9]

[Figure 10]

[Figure 11]

IMG 3

IMG 2 IMG 3

[Figure 12]

[Figure 13]

IMG 3 IMG 4

The cat from Image 1 and the dog from Image 2 are sitting side by side,

The woman from image 1 and the man from image 2 are standing in

Picture 1 is hung on the wall of a bedroom. The cup in Picture 3, made

with the background inside a car. The style of the image is the same as

front of a mountain. The dog from image 3 is standing between them.

of the same material as the plate in Picture 2, is placed on the table.

in Image 3.

The style of the image is the same as in image 4.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

RES

[Figure 21]

[Figure 22]

IMG 1 IMG 2

[Figure 23]

IMG 3

IMG 1

IMG 1 IMG 2

[Figure 24]

[Figure 25]

IMG 3

RES IMG 2

IMG 3

RES

On a fighting stage, two people are engaged in combat. Their

The girl from Image 1 is holding the shampoo from Image 2 on the beach,

The girl in Figure 1 has the same hairstyle as the girl in Figure 2. She is standing on a grassland, lifting her arm, with the bird from Figure 3 perched

movements are shown in Figure 3: the person in Figure 1 is the one on

with the sea visible in the background. The light condition of the image is the same as in Image 3.

the left in Figure 3, and the person in Figure 2 is the one on the right.

on it.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

RES SRC

RES

SRC

[Figure 30]

RES

[Figure 31]

[Figure 32]

SRC REF1

[Figure 33]

REF1

[Figure 34]

REF1

Replace the right person in the first image with the person in

Make the person from the first image has the same pose as person

Make the first image have the same image style as the second image.

the second image.

from the second image.

[Figure 35]

[Figure 36]

RES

RES

[Figure 37]

[Figure 38]

SRC

RES

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

SRC REF1

SRC REF1

[Figure 43]

REF1

Make the girl's hat from the first image have the same color scheme

Make the bag in the first image have the same pattern as the machine

Make the bag in the first image have the same material as the box in the

as the sweather in the second the image.

in the second image.

second image.

Figure 1: The gallery of overview: Enabling multimodal instruction-based editing and generation, extending beyond concrete objects to abstract attributions.

ABSTRACT

Recent advancements in instruction-based image editing and subject-driven generation have garnered significant attention, yet both tasks still face limitations in meeting practical user needs. Instruction-based editing relies solely on language instructions, which often fail to capture specific editing details, making reference images necessary. Meanwhile, subject-driven generation is limited to combining concrete objects or people, overlooking broader, abstract concepts. To address these challenges, we propose two novel tasks: multimodal instruction-based editing and generation. These tasks support both text and image instructions and extend the scope to include both concrete and abstract concepts, greatly enhancing their practical applications. We introduce DreamOmni2, tackling two primary challenges: data creation and model framework design. Our data synthesis

pipeline consists of three steps: (1) using a feature mixing method to create extraction data for both abstract and concrete concepts, (2) generating multimodal instruction-based editing training data using the editing and extraction models, and (3) further applying the extraction model to create training data for multimodal instruction-based editing. For the framework, to handle multi-image input, we propose an index encoding and position encoding shift scheme, which helps the model distinguish images and avoid pixel confusion. Additionally, we introduce joint training with the VLM and our generation/editing model to better process complex instructions. In addition, we have proposed comprehensive benchmarks for these two new tasks to drive their development. Experiments show that DreamOmni2 has achieved impressive results. Models and codes will be released.

- 1 INTRODUCTION

Recent advancements in unified generation and editing models (OpenAI, 2025; Google, 2025b) have gained significant attention and praise in the market. The success of these models can be attributed to several factors: (1) They greatly improve user experience by simplifying the process, allowing users to perform various design tasks within a single model without the need to switch between different ones. (2) Unified models reduce deployment costs for service providers. (3) Academically, they contribute to the exploration of AGI and world models, enabling the accurate understanding of user instructions and the creation or modification of real-world visual content.

Current released works (Batifol et al., 2025; Wu et al., 2025a; Deng et al., 2025) mainly focus on instruction-based editing and subject-driven generation with a text prompt and a single source image input, but both have limitations in application and advancing intelligence. (1) For instruction-based editing (Brooks et al., 2023; Liu et al., 2025; Xia et al., 2025b), instructions alone often fail to fully capture the user’s intent. For example, when a user says, “make the bag in the image have the same pattern as the dress in the given image,” it’s difficult to describe the complex pattern of “dress” with words. Thus, accurate editing requires multimodal instructions, including reference images and text. Notably, this challenge involves not only modifying objects but also any abstract attributes, such as texture, material, posture, hairstyle, and design style, which are difficult to describe with words. (2) Subject-driven generation models (Xiao et al., 2025; Wu et al., 2025c) and even commercial unified models (Google, 2025b) mainly focus on generating content from specific concrete objects or people, with limited research on referencing more general abstract attributions from input images.

To create a more intelligent and all-encompassing unified creation tool, we propose DreamOmni2. The biggest challenge lies in the training data, so we introduce a comprehensive data pipeline for multimodal instruction-based editing and generation, consisting of the following steps (Fig. 2): (1) We propose a feature mixing scheme to exchange attention features between two batches, allowing the model to generate pairs of images with the same abstract attribute or concrete object. Compared to the previous diptych method (Wu et al., 2025c) for generating image pairs, our scheme achieves a higher success rate, produces images with greater resolution, and completely eliminates any content blending at the edges when the pair of images is split. (2) Using the pairs generated in Step 1, we train a generation-editing model as an extraction model. This model extracts concrete objects or abstract attributions from the given image and generates another based on instructions. Compared to previous methods (Wu et al., 2025c; Chen et al., 2025) relying on segmentation and detection, our extraction model offers three key advantages: it can handle abstract concepts, occluded objects, and generate more diverse reference images. We then generate multimodal instruction-based editing training data, which includes a target image, a source image, an editing instruction, and multiple reference images. We use a text-to-image (T2I) model to generate a target image based on multiple keywords or select one from a real image database. The extraction model then generates reference images for one of the keywords. Additionally, we use an instruction-based editing model (Batifol et al., 2025) to transform the content defined by the selected keyword into something different, obtaining the source image. (3) We create multimodal instruction-based generation data by applying the extraction model to generate several reference images based on keywords from the source images created in Step 2. Thus, we build data for generating images from multiple reference images.

Furthermore, the current SOTA unified generation and editing models (Batifol et al., 2025) still cannot handle multiple image inputs. To this end, we propose the Dreamomni2 framework. First, we

propose an index encoding and position encoding shift scheme. Index encoding helps the model identify the input image’s index, improving its understanding of the referenced image in the instructions. Position encoding is shifted based on previous inputs, preventing pixel confusion and the copy-and-paste effect in the generated results. In addition, we propose a joint training scheme for the generation/editing model and VLM. While instructions in generation and editing models are typically simple, real-world user instructions are often irregular and logically complex. A VLM pretrained on large-scale corpora can better understand these complex intentions, translating instructions into a form the model can comprehend, significantly improving performance in real-world scenarios. Furthermore, for these two new tasks, we have built the DreamOmni2 benchmark using real image data. This allows for a more accurate assessment of the model’s generalization and performance in real-world scenarios. Our main contributions are fourfold:

- • We propose two highly practical tasks: multimodal instruction-based editing and generation guided by any concrete or abstract concept. Introducing these two tasks makes current unified generation and editing models smarter and more versatile creative tools.
- • We propose a three-stage data creation pipeline. Leveraging this pipeline, we have built a high-quality, comprehensive multimodal instruction-based editing and generation dataset.
- • We present the DreamOmni2 framework, which introduces the index encoding and position encoding shift scheme, enabling the model to handle multi-reference image inputs. Additionally, we propose a joint training scheme for the generation/editing model and VLM, enhancing the model’s ability to understand complex user instructions.
- • For these two new tasks, we propose a DreamOmni2 benchmark built from real image data. Experiments demonstrate the effectiveness of DreamOmni2 in real-world scenarios.

- 2 RELATED WORK

Instruction-based Editing refers to modifying an image based on a user’s language instruction (Deng et al., 2025; Sheynin et al., 2023; Xia et al., 2024). The main challenge of this task lies in the creation of high-quality and accurate editing datasets. As a pioneering work, InstructP2P (Brooks et al., 2023) introduced an instruction-based image editing dataset by fine-tuning GPT-3 and Promptto-Prompt (Hertz et al., 2022) with SD 1.5 (Rombach et al., 2022). Since then, many other approaches (Zhang et al., 2024; Xia et al., 2025b; Wei et al., 2024; Ge et al., 2024) for creating datasets have emerged, such as employing people to create data, using inpainting methods, collage-based methods, and using different expert models. Recently, DreamVE (Xia et al., 2025a) has unified instruction-based image and video editing. However, language-based editing is limited, as many details in real-world scenarios can’t be captured with words, requiring reference images for better description. To this end, we propose multimodal instruction-based editing, enabling guidance from concrete objects or abstract attributes in reference images. This makes unified image generation and editing models (Batifol et al., 2025; Wu et al., 2025a) more comprehensive and practical.

Subject-driven Generation has been extensively studied. Methods like Dreambooth (Ruiz et al.,

- 2023) and textual inversion (Gal et al., 2022) fine-tune models on multiple images of the same subject, enabling subject-driven generation. However, this requires users to prepare several images and perform fine-tuning for each new subject, which is not user-friendly. Later approaches like IP-adapter (Ye et al., 2023) and BLIP-diffusion (Li et al., 2023) used visual encoders to compress the subject of a reference image into a vector and inject it into a diffusion model, enabling subjectdriven generation without fine-tuning. IC LoRA (Huang et al., 2024) and Ominicontrol (Tan et al.,
- 2024) further explored the inherent image reference capabilities of DIT models (Peebles & Xie, 2023). Recently, unified generation and editing models (Xia et al., 2025b; Batifol et al., 2025; Wu et al., 2025a; Xiao et al., 2025) have adopted the simple approach of encoding reference images as visual tokens, concatenating them with text and noise tokens, and feeding them into the DIT model. However, prior methods focus mainly on concrete objects, limiting their ability to capture broader abstract concepts. In this paper, we propose multimodal instruction-based generation, a task that enables referencing any concrete objects or abstract attributes in reference images to generate new ones. This extends the scope of subject-driven generation and enhances its practicality.

[Figure 44]

[Figure 45]

Softmax

Source Query Source Key Source Value

FeedForward

A cartoon male sea - captain with grizzled mutton - chops and gray hair strolls on a weathered dock.

Noisy Emb

Source Image

Softmax

[Figure 46]

[Figure 47]

Target Query Target Key Target Value

Feature Mixing in Attention

DIT Block × N

Text Emb

A cartoon old white - wise man with curly white hair strolls through a grand hall. He's wearing a magnificent robe with intricate gold embroidery.

Stage 1. Creation of Training Data for Extraction Model. Target Image

[Figure 48]

[Figure 49]

[Figure 50]

A snake made of the same material as the dog in the given image is coiled on the ground.

Turn the material of the dog into plush material.

Ins Editng Model

Extraction Model

Target Image Source Image Reference Image

Stage 2. Creation of Multimodal Instruction-based Editing Data.

[Figure 51]

A trout from the given image is swimming in a clear mountain stream with smooth pebbles and green moss on the riverbed.

[Figure 52]

[Figure 53]

A wolf from the given image is howling at the full moon in a snowy forest with tall pine trees all around.

Extraction Model

Source Image Reference Image 1 Reference Image 2

Stage 3. Creation of Multimodal Instruction-based Generation Data.

- Figure 2: The Overview of DreamOmni2’s training data construction. (1) In stage 1, we use a feature mixing scheme to leverage the base model’s T2I capabilities, creating high-quality data pairs with concrete objects and abstract attributes. (2) In stage 2, we generate multimodal instructionbased editing data. Using stage 1 data, we train an extraction model to simulate objects or attributes in the target image and generate a reference image based on instructions. Additionally, we use an instruction-based editing model to modify the extracted objects or attributes in the target image to be different, creating the source image. This generates training pairs from reference and source images to the target image. (3) In stage 3, we extract objects from stage 2’s source images to create new reference images, forming training data for generating target images from reference images.

- 3 METHODOLOGY

- 3.1 SYNTHETIC DATA

Multimodal instruction-based editing and generation are new tasks, with the main challenge being the lack of training data. For multimodal instruction-based editing, the previous data creation pipeline (Brooks et al., 2023; Wei et al., 2024) involves generating triplets of instructions, source images, and target images. However, this approach does not allow for creating data that incorporates reference images as a condition for editing. For multimodal instruction-based generation, the previous subject-generation data pipeline (Wu et al., 2025c; Chen et al., 2025) relies on segmentation detection models to create reference images. This approach makes it difficult to synthesize data for generating reference abstract attributions or occluded concrete objects.

To address the training data problem for these two tasks, we propose a comprehensive synthetic data pipeline. Specifically, as illustrated in Fig. 2, our approach consists of three stages. In the first stage,

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Concrete Object

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Global Attribution

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Local Attribution

(a) Multimodal Instructionbased Editing Data Distribution

(b) Multimodal Instruction-based Generation Data Distribution

(c) Example Images

- Figure 3: Data distribution and samples for multimodal instruction-based editing and generation training data. Our dataset is comprehensive and diverse, including the generation and editing of concrete objects as well as abstract attributions, such as local and global attributions.

we introduce a feature mixing scheme, where a dual-branch structure is employed to simultaneously generate both the source image and the target image as follows:

### QK⊤

V , (1)

√

Attntar(Q,K,V ) = softmax

d

where Q = [Qntar;Qttar], K = [Ktarn ;Ktart ;Ksrcn ], and V = [Vtarn ;Vtart ;Vsrcn ]. Qttar, Ktart , and Vtart are the text features from the target branch, while Qntar, Ktarn , and Vtarn are the noise features from the target branch. Ksrcn and Vsrcn are the noise features from the source branch at the same layer as the Ktarn and Vtarn . [;] indicates token (or called length) dimension concatenation.

Our feature mixing scheme leverages the model’s inherent T2I capability to generate paired training data. Compared to the previous UNO (Wu et al., 2025c) diptych generation method, our feature mixing scheme has several clear advantages: (1) The diptych method halves the image resolution by forcing two images into one, while Feature Mixing generates in two branches without reducing resolution. (2) The diptych approach often misplaces the dividing line, leading to content blending. Our method avoids this issue. (3) Data generated by the feature mixing scheme is of higher quality and accuracy than that from the diptych approach. Then, we use the data to train extraction models. Our training data not only enhances the base model (Batifol et al., 2025)’s ability to extract concrete objects but also enables it to capture abstract concepts, a capability it previously lacked.

Afterward, as shown in Fig. 2 stage 2, we create multimodal instruction-based editing data. Specifically, we first create target images, using both T2I model-generated data and real images. For T2I-generated images, we randomly select diverse element keywords (e.g., objects or attributes) and use an LLM to compose a prompt, which the T2I model then uses to generate the target image. For real images, we directly use a VLM to extract keywords. T2I data is more flexible, allowing any concept combination, while real images reflect natural distributions. Thus, we combine both types of data. Next, using the extraction model trained in stage 1, we extract an object or attribution from the target image based on a selected keyword to create a reference image. We then apply instructionbased editing model (Batifol et al., 2025) to alter the selected keyword in the target image, obtaining the source image. Finally, we use an LLM to generate the editing instructions, forming a training tuple consisting of the source image, instruction, reference image, and target image.

After that, as shown in Fig. 2 stage 3, we create multimodal instruction-based generation data. We use the extraction model to extract keywords from the source image in stage 2, generating reference images. By combining these with the reference images from stage 2, we can obtain training tuples consisting of multiple reference images, an instruction, and a target image.

Our created dataset is shown in Fig. 3. Our dataset includes both real and synthetic target data, covering a wide range of object categories for generation and editing, including various abstract attributions and concrete objects. Additionally, we provide a comprehensive set of reference images, with cases ranging from one to five references, enabling the model to handle a wide variety of tasks.

- 3.2 FRAMEWORK AND TRAINING

The unified generation and editing base model (Batifol et al., 2025) can only process a single input image. To this end, we propose the DreamOmni2 framework. In multimodal instruction-based

- Table 1: Comparison between our DreamOmni2 benchmark and existing related benchmarks.

Editing Target Concrete Object Abstract Attribution

Benchmarks Task Type Num Reference

DreamBooth (Ruiz et al., 2023) Generation Single ✓ ✗ OmniContext (Wu et al., 2025b) Generation Multiple ✓ ✗

DreamOmni2 (Ours) Generation & Editing Multiple ✓ ✓

tasks, users typically reference images as “image 1”, “image 2” for convenience. However, in DIT, positional encoding alone cannot accurately distinguish the index of reference images. Therefore, we solve this by adding an index encoding to positional channels. Although index encoding helps distinguish reference images, we found that the position encoding still requires an offset based on the size of the previously input reference images. By adding this offset to the position encoding, we observed a reduction in copy-and-paste artifacts and pixel confusion between reference images.

Currently, training instructions for generation and editing models are usually well-structured with a fixed format. However, real-world user instructions are often irregular or logically inconsistent, creating a gap that can hinder the model’s understanding and reduce performance. To address this, we propose joint training of the VLM and generation models, enabling the VLM to interpret complex user instructions and output them in the structured format used in training, helping the editing and generation model better understand user intent. For multimodal instruction-based editing, the predefined output format combines user instructions with refined image descriptions, while for multimodal instruction-based generation, the VLM directly outputs a refined image description.

During training, we fine-tune Qwen2.5-VL (Wang et al., 2024) 7B to learn the predefined standard output format, with a learning rate of 1 × 10−5, using approximately 10 A100 hours. We then train the editing and generation models using LoRA on Flux Kontext (Batifol et al., 2025) to perform multimodal instruction-based editing and generation with the predefined standard instruction format. Notably, by using LoRA for training, we can retain the original instruction-editing capabilities of Kontext. As soon as a reference image is detected, our LoRA is activated, seamlessly integrating multimodal instruction-based editing and generation into the unified model. Additionally, we train LoRA for generation and editing separately, as the distinction between generation and editing lies in whether the consistency of the source image is preserved. Since instructions often do not clarify whether the user intends to edit or generate, separate training allows users to make their own choice. Both DreamOmni2 editing and generation LoRA are trained on a batch size of 16 and a learning rate of 5 × 10−6, consuming about 384 A100 hours.

- 3.3 BENCHMARK

Currently, no benchmark exists for multimodal instruction-based editing and generation. As shown in Tab. 1, DreamBooth (Ruiz et al., 2023) only supports single-image generation. Although OmniContext (Wu et al., 2025b) includes some multi-reference testing cases, it focuses solely on concrete object combinations and does not evaluate multimodal instruction-based editing or the inclusion of abstract attributes. To address this, we propose the DreamOmni2 benchmark to drive progress in these areas. Our benchmark is comprehensive, consisting of real images to accurately assess the model’s performance in real-world scenarios. The test cases cover a variety of categories, including the reference generation and editing of abstract attributions (global and local) and concrete objects. More details about our DreamOmni2 benchmark can be found in the Appendix.

- 4 EXPERIMENTS

Evaluation on Multimodal Instruction-based Image Editing. As shown in Tab. 2, we compare several competitive models that natively support multiple image inputs, such as DreamO (Mou et al.,

- 2025), Omnigen2 (Wu et al., 2025b), and Qwen-Image-Edit-2509 (Wu et al., 2025a). Although Kontext (Batifol et al., 2025) and Qwen-Image-Edit (Wu et al., 2025a) do not natively support multiple image inputs, we applied the method from Diffusers (von Platen et al., 2022), which combines multiple images into one input. We also compared closed-source commercial models, such as Nano Banana (Google, 2025b) and GPT-4o (OpenAI, 2025). We tested editing examples with concrete objects and abstract attributions on DreamOmni2 benchmark. The models were evaluated for success rates by Gemini 2.5 (Google, 2025a) and Doubao 1.6 (ByteDance, 2025), and several professional engineers manually assessed the results. As shown in Tab. 2, our DreamOmni2 achieved the best

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Replace the TV in the first image with the bag in the second image.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Replace the facial cleansing brush in the first image with the bag in the second image.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Make the first image has the same light

condition as the second image.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Replace the first image have the same image style as the second image.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Make the person from the first image has the

same pose as person from the second image.

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Make the bag from the first image have the same color scheme as the printer in the

second the image.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Make the words in the first image have the same font as the words in the second image.

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Make the person in the first image have the

same hairstyle as the person in the second image.

DreamO OmniGen2 Kontext Qwen-Edit-2509 GPT-4o Nano Banana DreamOmni2

Inputs

(Ours)

- Figure 4: Visual comparison of multimodal instruction-based editing. Compared to other competitive methods and even closed-source commercial models (GPT-4o and Nano Banana), DreamOmni2 shows more accurate editing results and better consistency.

- Table 2: Quantitative comparison of multimodal instruction-based editing. We use Gemini (Google, 2025a) and Doubao (ByteDance, 2025) to evaluate the success editing ratio of different models on concrete objects and abstract attributions, respectively. In addition, “Human” refers to professional engineers assessing the editing success rates of all models.

Concrete Object Abstract Attribution

Method

Gemini↑ Doubao↑ Human↑ Gemini↑ Doubao↑ Human↑ GPT-4o (OpenAI, 2025) 0.6829 0.7805 0.5610 0.7195 0.7439 0.5793

Nano Banana (Google, 2025b) 0.6829 0.7073 0.5366 0.6463 0.5488 0.3293

UNO (Wu et al., 2025c) 0.0000 0.0244 0.0000 0.0061 0.0183 0.0000 DreamO (Mou et al., 2025) 0.0244 0.0732 0.0000 0.0183 0.0183 0.0000

Omnigen2 (Wu et al., 2025b) 0.2195 0.2927 0.2927 0.0427 0.0793 0.0305 Qwen-Image-Edit (Wu et al., 2025a) 0.0976 0.1463 0.0244 0.0244 0.0183 0.0000

Kontext (Batifol et al., 2025) 0.0488 0.1220 0.0976 0.0183 0.0122 0.0122 Qwen-Image-Edit-2509 (Wu et al., 2025a) 0.2683 0.2927 0.2195 0.0488 0.1159 0.0427

DreamOmni2 (Ours) 0.5854 0.6585 0.6098 0.5854 0.6280 0.6829

performance in human evaluations. In VLM tests, DreamOmni2 significantly outperformed opensource models and achieved results close to those of commercial models. In fact, GPT-4o and Nano Banana often introduced unintended changes or inconsistencies in the edited attribution, which were not aligned with the reference images. These issues are difficult for VLMs to detect accurately. Additionally, GPT-4o caused the edited images to appear yellowed.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

A spaceship with

flowing lines is soaring

over a planet. The design style of the spaceship matches the outfit of the woman in

the given image.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

On the cup, "Story" is displayed in the same

font style as the

reference image.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

The man from

image 2 is adopting the same posture as the woman in image 1.

The background is

inside a spaceship.

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

The dog in image 1 is

frolicking on

the beach, and the style of the image is the same as in

image 2.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

The character from the second image

is holding the

item from the first image.

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

The logo from

the first image is printed on the object from the second image

and placed in a

bedroom.

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

The telephone in image 1 is placed

[Figure 208]

[Figure 209]

on a desk, which

is piled with books. The color tone of the image matches that of

image 2.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

The woman from image 2 has the same hairstyle as

the man from

image 1. The woman is sitting in an office.

DreamO OmniGen2 Kontext Qwen-Edit-2509 GPT-4o Nano Banana DreamOmni2

Inputs

(Ours)

- Figure 5: Visual comparison of multimodal instruction-based generation. Our DreamOmni2 significantly outperforms current open-source models and achieves generation results comparable to closed-source commercial models (GPT-4 and Nano Banana).

Qualitative results are shown in Fig. 4, where we present visualizations of editing cases involving various concrete objects and abstract attributes. It is clear that DreamOmni2 produces more accurate edits with better consistency. This further demonstrates the impressive performance of our approach in multimodal instruction-based editing.

Evaluation on Multimodal Instruction-based Image Generation. As shown in Tab. 3, our method outperforms the commercial model Nano Banana in both human evaluations and assessments by Doubao 1.6 and Gemini 2.5, achieving results comparable to GPT-4o. Compared to open-source models like DreamO, Omnigen2, and Qwen-Edit-2509, which focus primarily on generating images with multiple concrete objects, DreamOmni2 still significantly outperforms them in both generation accuracy and object consistency, even within their specialized domains. This further underscores the effectiveness of DreamOmni2 in multimodal instruction-based generation.

Quantitative results, as shown in Fig. 5, indicate that open-source models struggle with generating abstract attributes. Even in generating concrete objects, which these models are specifically optimized for, DreamOmni2 outperforms them in both instruction adherence and object consistency. Furthermore, DreamOmni2 even outperforms the commercial model Nano Banana.

Joint Training. As shown in Tab. 4, we validate the impact of joint training of generation or editing and VLM. Scheme 1 represents the base model, Kontext. In Scheme 2, we train the generation and editing models with basic instructions without introducing VLM. In Scheme 3, we train the VLM with standard descriptive instructions and input the VLM-generated descriptions into Kontext. In

- Table 3: Quantitative comparison of multimodal instruction-based generation. We use Gemini (Google, 2025a) and Doubao (ByteDance, 2025) to evaluate the success editing ratio on concrete objects and abstract attributions, respectively. In addition, “Human” refers to professional engineers assessing the editing success rates of all models.

Concrete Object Abstract Attribution

Method

Gemini↑ Doubao↑ Human↑ Gemini↑ Doubao↑ Human↑ GPT-4o (OpenAI, 2025) 0.6250 0.6250 0.5610 0.6889 0.6333 0.5793

Nano Banana (Google, 2025b) 0.5000 0.5417 0.5366 0.5556 0.5111 0.3293

UNO (Wu et al., 2025c) 0.0000 0.0000 0.0000 0.0333 0.0556 0.0000 DreamO (Mou et al., 2025) 0.0417 0.0833 0.0000 0.0667 0.0222 0.0000

Omnigen2 (Wu et al., 2025b) 0.2083 0.2500 0.2927 0.1000 0.0778 0.0305 Qwen-Image-Edit (Wu et al., 2025a) 0.0417 0.1250 0.0244 0.0889 0.1000 0.0000

Kontext (Batifol et al., 2025) 0.2500 0.3750 0.0976 0.0556 0.1222 0.0122 Qwen-Image-Edit-2509 (Wu et al., 2025a) 0.1250 0.2917 0.2195 0.1111 0.1556 0.0427

DreamOmni2 (Ours) 0.5833 0.6667 0.6098 0.5778 0.6333 0.6829

Table 4: The validation of joint training for generation or editing models and VLM.

Editing Generation

Generation or Editing Model Training

VLM Training

Method

Concrete Object Abstract Attribution Concrete Object Abstract Attribution

- Scheme 1 ✗ ✗ 0.1220 0.0122 0.3750 0.1222

- Scheme 2 ✓ ✗ 0.3659 0.3171 0.4583 0.3444

- Scheme 3 ✗ ✓ 0.2439 0.3415 0.5417 0.4778

Scehme 4 (Ours) ✓ ✓ 0.6585 0.6280 0.6667 0.6333

Scheme 4, we perform joint training of the VLM and our generation or editing model on our data. Comparing Scheme 2 with Scheme 1, we see that our data significantly enhances the model’s ability to handle multimodal instruction-based editing and generation. Comparing Scheme 3 with Scheme

- 4, we observe that introducing VLM helps the generation and editing models better understand realworld user complex instructions, improving performance. Moreover, our joint training scheme in Scheme 4 outperforms Scheme 2 and Scheme 3, demonstrating its effectiveness.

Index and Position Encoding. As shown in Tab. 5, we compare different encoding schemes to help the model adapt to multiple image inputs. Comparing Scheme 3 and Scheme 1, we find that adding index encoding enables the model to understand which image corresponds to references like ”Image 1,” ”Image 2,” and ”Image 3” in user instructions, resulting in more accurate generation and editing. Additionally, when comparing Scheme 3 and Scheme 4, we observe that with the inclusion of index encoding, multiple images require position encoding shifts instead of using the same position encoding. This adjustment prevents the copy-and-paste effect and improves the model’s editing and generation performance. Therefore, in DreamOmni2, we incorporate index encoding along with position encoding shifts for multiple reference images.

Table 5: The validation of different encoding schemes for multiple image inputs.

Method

Index Encoding

Position Encoding Shift

Editing Generation

Concrete Object Abstract Attribution Concrete Object Abstract Attribution

- Scehme 1 ✗ ✗ 0.2439 0.2805 0.2917 0.2222

- Scehme 2 ✗ ✓ 0.4634 0.5427 0.5417 0.5111

- Scehme 3 ✓ ✗ 0.3415 0.3902 0.4167 0.4556

Scheme 4 (Ours) ✓ ✓ 0.6585 0.6280 0.6667 0.6333

- 5 CONCLUSION

Current instruction-based editing relies on language, but it often struggles to clearly describe desired edits. Therefore, reference images are needed to guide the process. Additionally, subject-driven generation models typically focus on concrete objects and cannot generate images based on abstract concepts. To this end, we propose two new tasks: multimodal instruction-based editing and generation, where references include both concrete objects and abstract attributions. These tasks face two main challenges: training data and the framework supporting multi-image input. For training data, we introduce a three-stage data synthesis pipeline. In stage 1, we use a feature mixing approach to create data for an extraction model, which can generate images with the same elements (objects or attributes) as the given image. In stage 2, we use the extraction and instruction-based editing models to create multimodal instruction-based editing data. In stage 3, we apply the extraction model

to stage 2 data to generate multimodal instruction-based generation data. For the framework, we design an index encoding and position encoding shift scheme to help the model distinguish multiple images and avoid the copy-and-paste effect. We also propose a joint training scheme for the generation/editing model and the VLM, improving the model’s ability to understand complex real-world instructions. Extensive experiments show the impressive performance of DreamOmni2.

REFERENCES

Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pp. arXiv–2506, 2025. 2, 3, 5, 6, 7, 9

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image

editing instructions. In CVPR, 2023. 2, 3, 4 ByteDance. Doubao. https://www.doubao.com/, 2025. 6, 7, 9 Bowen Chen, Mengyi Zhao, Haomiao Sun, Li Chen, Xu Wang, Kang Du, and Xinglong Wu. Xverse:

Consistent multi-subject control of identity and semantic attributes via dit modulation. arXiv preprint arXiv:2506.21416, 2025. 2, 4

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2, 3

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3

Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A

hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007, 2024. 3 Google. Gemini. https://deepmind.google/models/gemini/, 2025a. 6, 7, 9 Google. Nano banana. https://aistudio.google.com/models/

gemini-2-5-flash-image, 2025b. 2, 6, 7, 9

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626,

2022. 3

Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 3

Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. NeurIPS, 2023. 3

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 2

Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915, 2025. 6, 7, 9

OpenAI. Gpt-4o image generation. https://openai.com/index/

introducing-4o-image-generation/, 2025. 2, 6, 7, 9 William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3 Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-

resolution image synthesis with latent diffusion models. In CVPR, 2022. 3

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR,

2023. 3, 6

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. arXiv preprint arXiv:2311.10089, 2023. 3

Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024. 3

Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/ huggingface/diffusers, 2022. 6

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6

Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. In ICLR, 2024. 3, 4

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a. 2, 3, 6, 7, 9

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b. 6, 7, 9

Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. ICCV, 2025c. 2, 4, 5, 7, 9

Bin Xia, Shiyin Wang, Yingfan Tao, Yitong Wang, and Jiaya Jia. Llmga: Multimodal large language model based generation assistant. In ECCV, 2024. 3

Bin Xia, Jiyang Liu, Yuechen Zhang, Bohao Peng, Ruihang Chu, Yitong Wang, Xinglong Wu, Bei Yu, and Jiaya Jia. Dreamve: Unified instruction-based image and video editing. arXiv preprint arXiv:2508.06080, 2025a. 3

Bin Xia, Yuechen Zhang, Jingyao Li, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, and Jiaya Jia. Dreamomni: Unified image generation and editing. In CVPR, 2025b. 2, 3

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In CVPR,

2025. 2, 3 Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023. 3 Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. NeurIPS, 2024. 3

A APPENDIX

- A.1 DREAMOMNI2 BENCHMARK

Our DreamOmni2 benchmark includes 205 multimodal instruction-based editing test cases and 114 instruction-based generation test cases. Visualizations of the editing and generation test cases are shown in Fig. 7 and Fig. 6, respectively. The benchmark covers a wide range of test cases, with input reference images ranging from one to five, and encompasses diverse local and global attributes, as well as concrete objects. The DreamOmni2 Benchmark will be released.

|[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>The woman in image 1 is holding the camera in image 2, and the<br><br>woman in image 3 is holding the cactus in image 4. The two women are standing side by side on the beach by the sea.|
|---|

|[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>The girl in Image 1 is on the left, the girl in Image 2 is in the center, and the boy<br><br>in Image 3 is on the right. They are sitting around a card table, playing cards.|
|---|

|[Figure 226]<br><br>[Figure 227]<br><br>In an office, the character from the first image and the<br><br>character from the second image are moving a desk.|
|---|

|[Figure 228]<br><br>[Figure 229]<br><br>The man in image 1 is lying on the bed, and the lighting<br><br>condition of the image is the same as in image 2.|
|---|

|[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>The car in Figure 1 is parked in a park, with the book from Figure<br><br>2 placed on its hood. The car has the same color scheme as the<br><br>machine in Figure 3.|
|---|

|[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>The woman in image 1 is holding the cup from image 3. The camera in<br><br>image 2 has the same color scheme as the engine in image 4. The<br><br>camera is placed on a table.|
|---|

|[Figure 237]<br><br>[Figure 238]<br><br>The dog in image 1 is frolicking on the beach, and the<br><br>style of the image is the same as in image 2.|
|---|

|[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>The girl from Image 1 is holding the shampoo from Image<br><br>2 on the beach, with the sea visible in the background.<br><br>The light condition of the image is the same as in Image 3.|
|---|

|[Figure 242]<br><br>[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]<br><br>The men in image 1 and image 2 are standing on<br><br>the beach, with a sign from image 3 beside them.<br><br>The tone of the image is the same as in image 4.|
|---|

Concrete Objects Local Attribution Global Attribution

- Figure 6: Examples of multimodal instruction-based generation in DreamOmni2 benchmark.

|Add the clock from the second image on the table in the first image.<br><br>[Figure 246]<br><br>[Figure 247]|
|---|

|[Figure 248]<br><br>[Figure 249]<br><br>Replace the TV in the first image with the bag in the second image.|
|---|

|[Figure 250]<br><br>[Figure 251]<br><br>Replace the person in the first image with the person in the second image.|
|---|

|[Figure 252]<br><br>[Figure 253]<br><br>Make the person from the first image has the same<br><br>pose as person from the second image.|
|---|

|[Figure 254]<br><br>[Figure 255]<br><br>Make the girl's dress from the first image have the same color<br><br>scheme as the shoe in the second the image.|
|---|

|[Figure 256]<br><br>[Figure 257]<br><br>Make the person in the first image have the same<br><br>hairstyle as the person in the second image.|
|---|

|[Figure 258]<br><br>[Figure 259]<br><br>Make the first image has the same light condition as the second image.|
|---|

|[Figure 260]<br><br>[Figure 261]<br><br>Make the first image have the same image style as the second image.|
|---|

|[Figure 262]<br><br>[Figure 263]<br><br>Make the first image has the same color tone as the second image.|
|---|

Concrete Objects Local Attribution Global Attribution

- Figure 7: Examples of multimodal instruction-based editing in DreamOmni2 benchmark.

- A.2 MORE MULTIMODAL INSTRUCTION-BASED EDITING CASES

As shown in Fig. 8, Fig. 9, Fig. 10, Fig. 11, Fig. 12, Fig. 13, Fig. 14, Fig. 15, Fig. 16, Fig. 17, Fig. 18, Fig. 19, Fig. 20, and Fig. 21, we present additional visual cases of DreamOmni2 on the multimodal instruction-based editing task.

- A.3 MORE MULTIMODAL INSTRUCTION-BASED GENERATION CASES

As shown in Fig. 22, Fig. 23, Fig. 24, Fig. 25, Fig. 26, Fig. 27, Fig. 28, Fig. 29, and Fig. 30, we present additional visual cases of DreamOmni2 on the multimodal instruction-based generation task.

|Add the clock from the second image on the table in the<br><br>first image.<br><br>[Figure 264]<br><br>[Figure 265]|[Figure 266]|
|---|---|
|[Figure 267]<br><br>[Figure 268]<br><br>Make the woman from the second image stand on the road<br><br>in the first image.|[Figure 269]|
|[Figure 270]<br><br>[Figure 271]<br><br>Make the woman from the second image sit on the sofa in the first image.|[Figure 272]|
|[Figure 273]<br><br>[Figure 274]<br><br>Make the man from the second image sit on the sofa in the<br><br>first image.|[Figure 275]|

|[Figure 276]<br><br>[Figure 277]<br><br>Add the person from the second image to the first image.|[Figure 278]|
|---|---|
|[Figure 279]<br><br>[Figure 280]<br><br>Replace the person in the first image with the person in the second image.|[Figure 281]|
|[Figure 282]<br><br>[Figure 283]<br><br>Replace the dog in the first image with the dog in the second image.|[Figure 284]|
|[Figure 285]<br><br>[Figure 286]<br><br>Replace the person in the first image with the person in the<br><br>second image.|[Figure 287]|

|[Figure 288]<br><br>[Figure 289]<br><br>Replace the logo in the first image with the logo in the second image.|[Figure 290]|
|---|---|
|[Figure 291]<br><br>[Figure 292]<br><br>Replace the right person in the first image with the person in<br><br>the second image.|[Figure 293]|
|[Figure 294]<br><br>[Figure 295]<br><br>Replace the person in the first image with the person in the<br><br>second image.|[Figure 296]|
|[Figure 297]<br><br>[Figure 298]<br><br>Replace the person in the first image with the person in the<br><br>second image.|[Figure 299]|

|[Figure 300]<br><br>[Figure 301]<br><br>Replace the cat in the first image with the husky dog in the<br><br>second image.|[Figure 302]|
|---|---|
|[Figure 303]<br><br>[Figure 304]<br><br>Replace the whale in the first image with the car in the<br><br>second image.|[Figure 305]|
|[Figure 306]<br><br>[Figure 307]<br><br>Replace the stapler in Image 1 with the glasses from Image 2.|[Figure 308]|
|[Figure 309]<br><br>[Figure 310]<br><br>Replace the facial cleansing brush in the first image with the<br><br>bag in the second image.|[Figure 311]|

|[Figure 312]<br><br>[Figure 313]<br><br>Replace the lantern in the first image with the dog in the second image.|[Figure 314]|
|---|---|
|[Figure 315]<br><br>[Figure 316]<br><br>Replace the man in the first image with the woman in the<br><br>second image.|[Figure 317]|
|[Figure 318]<br><br>[Figure 319]<br><br>Make the first image has the same light condition as the second image.|[Figure 320]|
|[Figure 321]<br><br>[Figure 322]<br><br>Make the first image has the same light condition as the second image.|[Figure 323]|

|[Figure 324]<br><br>[Figure 325]<br><br>Make the first image has the same light condition as the second image.|[Figure 326]|
|---|---|
|[Figure 327]<br><br>[Figure 328]<br><br>Make the first image has the same light condition as the<br><br>second image.|[Figure 329]|
|[Figure 330]<br><br>[Figure 331]<br><br>Make the first image has the same light condition as the<br><br>second image.|[Figure 332]|
|[Figure 333]<br><br>[Figure 334]<br><br>Make the first image has the same light condition as the second image.|[Figure 335]|

|[Figure 336]<br><br>[Figure 337]<br><br>Make the first image have the same image style as the<br><br>second image.|[Figure 338]|
|---|---|
|[Figure 339]<br><br>[Figure 340]<br><br>Make the first image have the same image style as the second image.|[Figure 341]|
|[Figure 342]<br><br>[Figure 343]<br><br>Make the first image have the same image style as the second image.|[Figure 344]|
|[Figure 345]<br><br>[Figure 346]<br><br>Make the first image have the same image style as the second image.|[Figure 347]|

|[Figure 348]<br><br>[Figure 349]<br><br>Make the first image has the same color tone as the second<br><br>image.|[Figure 350]|
|---|---|
|[Figure 351]<br><br>[Figure 352]<br><br>Make the first image has the same color tone as the second image.|[Figure 353]|
|[Figure 354]<br><br>[Figure 355]<br><br>Make the first image has the same color tone as the second image.|[Figure 356]|
|[Figure 357]<br><br>[Figure 358]<br><br>Make the first image has the same color tone as the second image.|[Figure 359]|

|[Figure 360]<br><br>[Figure 361]<br><br>Make the person from the first image has the same pose as<br><br>person from the second image.|[Figure 362]|
|---|---|
|[Figure 363]<br><br>[Figure 364]<br><br>Make the person from the first image has the same pose as person from the second image.|[Figure 365]|
|[Figure 366]<br><br>[Figure 367]<br><br>Make the girl's dress from the first image have the same<br><br>color scheme as the shoe in the second the image.|[Figure 368]|
|[Figure 369]<br><br>[Figure 370]<br><br>Make the mop from the first image have the same design style as the hammer in the second the image.|[Figure 371]|

|[Figure 372]<br><br>[Figure 373]<br><br>Make the shoe from the first image have the same design style as the dress in the second the image.|[Figure 374]|
|---|---|
|[Figure 375]<br><br>[Figure 376]<br><br>Make the book from the first image have the same design<br><br>style as the watch in the second the image.|[Figure 377]|
|[Figure 378]<br><br>[Figure 379]<br><br>Make the box from the first image have the same design<br><br>style as the machine in the second the image.|[Figure 380]|
|[Figure 381]<br><br>[Figure 382]<br><br>Make the bird from the first image have the same design<br><br>style as the telephone booth in the second the image.|[Figure 383]|

|[Figure 384]<br><br>[Figure 385]<br><br>Make the person in the first image have the same expression as the person in the second image.|[Figure 386]|
|---|---|
|[Figure 387]<br><br>[Figure 388]<br><br>Make the person in the first image have the same<br><br>expression as the person in the second image.|[Figure 389]|
|[Figure 390]<br><br>[Figure 391]<br><br>Make the person in the first image have the same hairstyle<br><br>as the person in the second image.|[Figure 392]|
|[Figure 393]<br><br>[Figure 394]<br><br>Make the person in the first image have the same hairstyle<br><br>as the person in the second image.|[Figure 395]|

|[Figure 396]<br><br>[Figure 397]<br><br>Make the person in the first image have the same makeup as the person in the second image.|[Figure 398]|
|---|---|
|[Figure 399]<br><br>[Figure 400]<br><br>Make the bottle in the first image have the same material as the microwave in the second image.|[Figure 401]|
|[Figure 402]<br><br>[Figure 403]<br><br>Make the words in the first image have the same font as the words in the second image.|[Figure 404]|
|[Figure 405]<br><br>[Figure 406]<br><br>Make the words in the first image have the same font as the words in the second image.|[Figure 407]|

|[Figure 408]<br><br>[Figure 409]<br><br>Make the car in the first image have the same pattern as the<br><br>mouse in the second image.|[Figure 410]|
|---|---|
|[Figure 411]<br><br>[Figure 412]<br><br>Make the dress in the first image have the same pattern in<br><br>the second image.|[Figure 413]|
|[Figure 414]<br><br>[Figure 415]<br><br>Make the T-shirt in the first image have the same pattern in<br><br>the second image.|[Figure 416]|
|[Figure 417]<br><br>[Figure 418]<br><br>Make the paper in the first image have the same pattern in the second image.|[Figure 419]|

|[Figure 420]<br><br>[Figure 421]<br><br>Replace the background of first image with the second image|[Figure 422]|
|---|---|
|[Figure 423]<br><br>[Figure 424]<br><br>Replace the background of first image with the second image|[Figure 425]|
|[Figure 426]<br><br>[Figure 427]<br><br>Replace the background of first image with the second image|[Figure 428]|
|[Figure 429]<br><br>[Figure 430]<br><br>Replace the background of first image with the second image|[Figure 431]|

|[Figure 432]<br><br>[Figure 433]<br><br>In the scene, the character from the first image stands on the left, and the character from the second image stands on the right. They are shaking hands against the backdrop of a spaceship interior.|[Figure 434]|
|---|---|
|[Figure 435]<br><br>[Figure 436]<br><br>The character from the first image is holding the item from the second picture.|[Figure 437]|
|[Figure 438]<br><br>[Figure 439]<br><br>The character from the second image is holding the<br><br>item from the first image.|[Figure 440]|
|[Figure 441]<br><br>[Figure 442]<br><br>On a lawn, the woman from the first image is sitting on the grass, and the backpack from the second image is placed beside her.|[Figure 443]|

|[Figure 444]<br><br>[Figure 445]<br><br>The man from the first image is wearing the clothes from the second image and is sitting on a sofa.|[Figure 446]|
|---|---|
|[Figure 447]<br><br>[Figure 448]<br><br>[Figure 449]<br><br>The girl from Image 1 is holding the bag from Image 2 and<br><br>standing in a bedroom. The color tone of the image is the same as in Image 3.|[Figure 450]|
|[Figure 451]<br><br>[Figure 452]<br><br>[Figure 453]<br><br>The parrot from Image 1 is wearing the hat from Image 2 and standing on the ground, with a forest in the background. The color tone of the image is the same as in Image 3.|[Figure 454]|
|[Figure 455]<br><br>[Figure 456]<br><br>[Figure 457]<br><br>The car from Image 1 is parked on the street, and the bird from Image 2 is standing on the roof of the car. The style of the image is the same as in Image 3.|[Figure 458]|

|[Figure 459]<br><br>[Figure 460]<br><br>[Figure 461]<br><br>The man from Image 1 and the woman from Image 2 are standing in a laboratory. The light condition of the image is<br><br>the same as in Image 3.|[Figure 462]|
|---|---|
|[Figure 463]<br><br>[Figure 464]<br><br>[Figure 465]<br><br>The man in Figure 1 is sitting on a sofa in a room, next to a<br><br>refrigerator from Figure 2. The refrigerator’s design style matches the object shown in Figure 3.|[Figure 466]|
|[Figure 467]<br><br>[Figure 468]<br><br>[Figure 469]<br><br>The girl from Image 1 and the girl from Image 2 are standing in front of the car from Image 3.|[Figure 470]|
|[Figure 471]<br><br>[Figure 472]<br><br>[Figure 473]<br><br>The woman from Image 1 is sitting on the chair, the woman from Image 2 is standing on the left, and the woman from Image 3 is<br><br>standing on the right. The background is in a living room.|[Figure 474]|

|[Figure 475]<br><br>[Figure 476]<br><br>[Figure 477]<br><br>[Figure 478]<br><br>The man from Image 1 stands next to the woman from Image<br><br>2. The woman is wearing the hat from Image 4, which has the logo from Image 3 on it. The background is by the lake.|[Figure 479]|
|---|---|
|[Figure 480]<br><br>[Figure 481]<br><br>[Figure 482]<br><br>[Figure 483]<br><br>The man in image 1 is standing and holding the toy from<br><br>image 3. The woman from image 2 is standing next to him.<br><br>The background is from image 4.|[Figure 484]|
|[Figure 485]<br><br>[Figure 486]<br><br>[Figure 487]<br><br>[Figure 488]<br><br>The man from Image 2 is holding the ship model from Image 1, and the man from Image 3 is wearing the headphones from Image 4. They are in a coffee shop.|[Figure 489]|
|[Figure 490]<br><br>[Figure 491]<br><br>The car in image 1 is driving through the city, surrounded by tall<br><br>buildings. The style of the image is the same as in image 2.|[Figure 492]|

|[Figure 493]<br><br>[Figure 494]<br><br>The woman in image 1 is standing on the runway, and the lighting condition of the image is the same as in image 2.|[Figure 495]|
|---|---|
|[Figure 496]<br><br>[Figure 497]<br><br>The table from image 1 is made of the same material as the<br><br>jar in image 2. The table is placed on the beach.|[Figure 498]|
|[Figure 499]<br><br>[Figure 500]<br><br>The book cover in image 1 features the same pattern as in<br><br>image 2. The book is placed on a desk in the study.|[Figure 501]|
|[Figure 502]<br><br>[Figure 503]<br><br>The microphone from image 1 is placed in image 2.|[Figure 504]|

|[Figure 505]<br><br>A majestic lion resting on a rocky outcrop. The color tone of the image is the same as in Image 1.|[Figure 506]|
|---|---|
|A warrior stands on the battlefield. The lighting conditions of<br><br>the image are the same as in the reference image.<br><br>[Figure 507]|[Figure 508]|
|A blonde girl with a high ponytail, wearing a black long dress,<br><br>stands at the front of the classroom. The style of the image is the same as the given image.<br><br>[Figure 509]|[Figure 510]|
|A spaceship is flying in the sky, with the sun visible in the<br><br>background. The style of the image is the same as in Image 1.<br><br>[Figure 511]|[Figure 512]|

|A girl wearing a pink skirt and a white long-sleeve shirt, with<br><br>long golden hair. She strikes the same pose as the man in the<br><br>given image. The background is a field of flowers.<br><br>[Figure 513]|[Figure 514]|
|---|---|
|Generate a helicopter soaring above a city skyline at dusk. The color scheme of the helicopter is the same as that of the motorcycle.<br><br>[Figure 515]|[Figure 516]|
|A sleek smartphone resting on a table, with its design featuring smooth curves and a modern look. The color scheme of the phone matches the outfit of the man in the reference image.<br><br>[Figure 517]|[Figure 518]|
|A stylish hairdryer placed on a vanity table. The design style of the hairdryer is inspired by the comb in the given image.<br><br>[Figure 519]|[Figure 520]|

|Generate a woman in a floral summer dress, walking barefoot<br><br>along a beach at sunset. Her hair flows in the breeze, and she<br><br>smiles softly as she watches the waves. Her makeup is the same as the woman in the given image. The background features a golden sun dipping below the horizon, casting warm hues over the calm ocean and sandy shore.<br><br>[Figure 521]|[Figure 522]|
|---|---|
|A woman in a cozy knitted sweater and denim jeans, sitting by<br><br>a fireplace, sipping tea while reading a book. Her hair is styled in loose waves, and she has a calm, content expression. Her makeup is the same as the woman in the given image.<br><br>[Figure 523]|[Figure 524]|
|[Figure 525]<br><br>A vintage motorcycle is parked on a cobblestone street. The material of the motorcycle is the same as the vase.|[Figure 526]|
|On the cup, "Story" is displayed in the same font style as the<br><br>reference image.<br><br>[Figure 527]|[Figure 528]|

|A table with the texture matching the one in the given image, placed in the center of a cozy living room. The table is surrounded by comfortable chairs, with soft lighting coming from a nearby lamp.<br><br>[Figure 529]|[Figure 530]|
|---|---|
|Generate a bottle placed on a wooden kitchen counter. Its texture matches the given image. The bottle is surrounded by fresh fruit, a small bowl of herbs, and a cup of steaming tea.<br><br>[Figure 531]|[Figure 532]|

#### Figure 30: Multimodal instruction-based generation cases of DreamOmni2.

