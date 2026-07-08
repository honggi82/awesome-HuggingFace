# arXiv:2512.22525v1[cs.CV]27Dec2025

## DreamOmni3: Scribble-based Editing and Generation

Bin Xia1,2, Bohao Peng1, JiyangLiu2, Sitong Wu1, Jingyao Li1, Junjia Huang2, Xu Zhao2, Yitong Wang2, Ruihang Chu1, Bei Yu1, Jiaya Jia 3 1 CUHK, 2 ByteDance Inc, 3 HKUST

https://github.com/dvlab-research/DreamOmni3

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

SRC RES SRC RES

###### SRC RES

Add a boy inside the black circle.

Anime Boy

Add a boy inside the black circle.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

###### SRC RES

SRC RES

###### SRC RES

Change the color of the bag in the black circle

Change the hairstyle of the man in the red circle

Delete the person inside the green circle.

to purple.

to a bald head.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

SRC RES

SRC RES

RES

[Figure 18]

SRC

[Figure 19]

REF

[Figure 20]

REF

Insert the toy in the red circle from the second image into the red

Make the woman's clothing in the red circle of the first image

circle in the first image.

have the same color scheme as the sweater in the second image.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

###### RES

[Figure 25]

[Figure 26]

SRC RES

[Figure 27]

SRC

SRC

[Figure 28]

IMG1 IMG2

RES

A man is in the red circle, and a woman with long hair is

In the living room, the sofa from image 1 is placed in the black circle, and the painting from image 2 is hung in the red square.

A dragon is holding a bag in a garden under

in the green circle. They are shaking hands, with a

the moonlight.

volcano in the background.

Figure 1. The gallery of DreamOmni3, which has scribble-based editing and generation capabilities.

### Abstract

Recently unified generation and editing models have achieved remarkable success with their impressive performance. These models rely mainly on text prompts for instruction-based editing and generation, but language of-

ten fails to capture users’ intended edit locations and finegrained visual details. To this end, we propose two tasks: scribble-based editing and generation, that enables more flexible creation on graphical user interface (GUI) combining user textual, images, and freehand sketches. We introduce DreamOmni3, tackling two challenges: data cre-

ation and framework design. Our data synthesis pipeline includes two parts: scribble-based editing and generation. For scribble-based editing, we define four tasks: scribble and instruction-based editing, scribble and multimodal instruction-based editing, image fusion, and doodle editing. Based on DreamOmni2’s dataset, we extract editable regions and overlay hand-drawn boxes, circles, doodles or cropped image to construct training data. For scribble-based generation, we define three tasks: scribble and instruction-based generation, scribble and multimodal instruction-based generation, and doodle generation, following similar data creation pipelines. For the framework, instead of using binary masks, which struggle with complex edits involving multiple scribbles, images, and instructions, we propose a joint input scheme that feeds both the original and scribbled source images into the model, using different colors to distinguish regions and simplify processing. By applying the same index and position encodings to both images, the model can precisely localize scribbled regions while maintaining accurate editing. Finally, we establish comprehensive benchmarks for these tasks to promote further research. Experimental results demonstrate that DreamOmni3 achieves outstanding performance, and models and code will be publicly released.

### 1. Introduction

The recent success of unified generation and editing models [3, 5, 10] can be attributed to three key factors: (1) Unified training for generation and editing not only improves performance on existing tasks through mutual enhancement but also gives rise to a wide range of new editing and generation capabilities. (2) It significantly lowers the user barrier, enabling all functions within a single model without the need to choose specialized ones. (3) They exhibit strong multimodal understanding, responding to real-world visuals, marking progress toward world models and AGI.

The current unified generation and editing models focus on generation and editing based on images and text as instructions. However, this approach often falls short of meeting users’ interactive needs. For example, certain editing or generation positions in the image are difficult to describe with language, some objects may be hard for users to identify by name, or there may be multiple identical objects that are hard to distinguish. In these cases, manual annotation is necessary. Furthermore, users may wish to make more flexible and creative additions, deletions, or modifications to the content in the image, such as through drawing. In these scenarios, it’s essential for the model to not only understand images and language but also understand the editing and generation intentions behind the user’s manual scribbles.

To create a more intelligent and comprehensive unified creation tool, we present DreamOmni3, a model that inte-

grates our proposed scribble-based editing and generation with the existing unified editing and generation framework. First, we define the tasks of scribble-based editing and generation to guide our data creation. Specifically, as shown in Fig. 2 (a), we categorize scribble-based editing into four types: scribble and instruction-based editing, scribble and multimodal instruction-based editing, image fusion, and doodle editing. Additionally, as shown in Fig. 2 (b), we categorize scribble-based generation into three types: scribble and instruction-based generation, scribble and multimodal instruction-based generation, and doodle generation.

To train DreamOmni3, the main challenge is the lack of data. To address this, we introduce a comprehensive data pipeline for scribble-based editing and generation. Scribble-based editing data creation: (1) For multimodal instruction-based editing, we use the Refseg service on the DreamOmni2 dataset to obtain the coordinates and size of the edited objects in both the source and reference images, marking the edited positions with hand-drawn circles or boxes. (2) For instruction-based editing, we use the same data as in step (1) but without the reference image. (3) For image fusion, we extract the edited objects from the reference image and paste them onto the corresponding position on the source image. (4) For doodle editing, we crop the edited objects from the target image, generate sketches, and place them back into the source image. Scribble-based generation data creation: (1) For multimodal instructionbased generation, we use Refseg to locate the edited objects in the image and mark the circles or boxes on the blank canvas. (2) For instruction-based generation, we remove the reference image from the data created in step (1). (3) For doodle generation, we extract the edited objects from the target image, generate sketches, and place them on a blank canvas at the same position.

For the framework, we chose to input the source image with scribbles rather than using a binary mask for two reasons: (1) It avoids the complexity and computational cost of binary masks when editing multiple areas. Scribbles, distinguished by color, are easier to describe with instructions. (2) It is compatible with existing unified editing and generation models, which require RGB inputs, not binary masks.

Based on the above analysis, we propose the DreamOmni3 framework. Specifically, we propose a joint input scheme for editing by feeding both the original source image and the scribbled source image. This allows us to retain both the scribbled information and the pixels covered by the scribbles. To better align the source image with its scribbled version, both share the same index and position encodings. For the reference image, to distinguish it from the source images and avoid pixel confusion, we adopt the same position shift and index encoding scheme as in DreamOmni2 (Fig. 2 (c)). Furthermore, for these tasks, we construct a DreamOmni3 benchmark using real-world im-

age data, enabling a more accurate evaluation of the model’s generalization and real-world performance. Our main contributions are fourfold:

- • We introduce two highly useful tasks for unified generation and editing models: scribble-based editing and scribble-based generation. These tasks can be combined with language and image instructions, enhancing the creative usability of unified models and providing clear definitions for targeted optimization and future research.
- • We propose a pipeline to create a high-quality, comprehensive dataset for scribble-based editing and generation.
- • We present DreamOmni3, a framework that supports text, image, and scribble inputs with complex logic. It takes both the source image and scribble as input, maintaining editing consistency while accurately interpreting the scribble’s intent. We also design position and index encoding schemes to differentiate between the scribble and source image, ensuring compatibility with existing unified generation and editing architectures.
- • For these tasks, we build the DreamOmni3 benchmark using real-world image data. Experiments demonstrate the strong effectiveness of DreamOmni3 in real scenarios.

### 2. Related Work

Mask-based Editing refers to editing operations performed on selected regions or items of an image. It can generally be divided into three categories: (1) Image inpainting [22, 23, 25–27] is a common technique where users paint over the area they wish to modify. The model then regenerates that region based on the instruction while preserving the rest of the image. (2) Since traditional inpainting completely regenerates the masked region and cannot preserve its original structure or color, MagicQuill [8] introduces auxiliary inputs such as edge maps or low-resolution images to help maintain contour and color consistency. (3) Some approaches [9, 13, 14] inject compressed object IDs from a reference image into the masked region to insert specific objects. Compared to these methods, DreamOmni3 offers several advantages: (a) Traditional inpainting datasets are created by simply masking parts of an image, which prevents the model from reasoning about environmental effects such as lighting or shadow changes. Our dataset, built through instruction-based editing, allows the model to respond more coherently to contextual changes. (b) Inpainting often suffers from inaccurate masks — regions may be over- or under-edited. Even MagicQuill, though guided by edge maps, still requires precise manual masking. In contrast, our instruction-based approach is far more robust: users can roughly circle a region, and the model accurately understands both the intent and the editing scope, providing a much more user-friendly experience. (c) Previous methods relied on inconsistent input formats, making them hard to unify under a single framework. DreamOmni3 estab-

lishes a unified architecture that simplifies use and integration. (d) Benefiting from this unified design, DreamOmni3 not only extends existing editing capabilities but also introduces new ones. For example, in image fusion, prior methods [9] could only merge a single reference image into a masked area, whereas our model supports multimodal inputs (multiple images, text, scribbles, and more) for richer and more precise edits. Moreover, DreamOmni3 bridges the gap between image generation and editing, enabling interactive experiences like direct sketching on tablets.

Mask-based Generation incorporates masks into the image generation process to control content within specific regions. It can be broadly categorized into three types. For mask-guided text-to-image generation, Multi-diffusion [19] first explored a training-free approach for multi-region generation on text-to-image models, but the results were unstable. GLIGEN [6] improved this by introducing an attention module and encoding bounding boxes through Fourier embeddings. Later works [17, 29, 30] further optimized attention mechanisms and refined the granularity of input information. For mask-guided subject-driven generation, MS-diffusion [18] employs a grounding resampler to associate visual information with specific entities and spatial constraints. Nevertheless, existing mask-based generation frameworks are often overly complex in both input format and inference process. In contrast, DreamOmni3 allows users to paint a mask and combine it with image and language instructions to generate more complex outputs. Its unified design aligns with existing frameworks, bridging mask-based generation and image editing within a single system, greatly enhancing usability.

### 3. Methodology

#### 3.1. Synthetic Data

The biggest challenge in scribble-based editing and generation is the lack of data. We need to construct a dataset that incorporates language, images, and scribbles as instructions, and develop the ability to combine these three instruction types for complex editing tasks, enabling smarter editing tools. We found that DreamOmni2 effectively unified language and image instructions and introduced the multimodal instruction editing and generation task, along with corresponding training data. Therefore, we directly use DreamOmni2’s multimodal instruction editing and generation data as the base, and further extend it to create a dataset that includes scribbles as instructions.

As shown in Fig. 2 (a), we divide scribble-based editing into four tasks: (1) Scribble and multimodal instructionbased editing: We use Referseg service to locate the editing objects in both the reference and target images. Since users often draw imperfect shapes, we manually created 30 different squares and circles for a scribble symbol template

A cartoon old white - wise man with curly white hair strolls through a grand hall. He's

wearing a magnificent robe with intricate

gold embroidery.

[Figure 29]

[Figure 30]

||[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>Target Image<br><br>Task 1&2: Scribble-based instruction and Multimodal instruction-based editing data<br><br>Reference Image1 Reference Image2 Reference Image3<br><br>|Referseg<br><br>Service| |
|---|---|
| | |
<br><br>| |[Figure 35]<br><br>[Figure 36]<br><br>……<br><br>[Figure 37]|
|---|---|
| | |
<br><br>Scribble Symbol Template Library<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>Source Image S-Reference Image1 S-Reference Image2 S-Reference Image3<br><br>|Ins-Editing<br><br>Model| |
|---|---|
| | |
<br><br>Remove<br><br>extracted objects<br><br>[Figure 41]|
|---|
<br><br>|Task 3: Doodle Editing<br><br>[Figure 42]<br><br>Target Image<br><br>|Referseg<br><br>Service|
|---|
<br><br>|Ins-Editing<br><br>Model| |
|---|---|
| | |
<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>Extract a doodle<br><br>similar to a human<br><br>drawing<br><br>|Ins-Editing<br><br>Model| |
|---|---|
| | |
<br><br>[Figure 47]<br><br>Remove<br><br>extracted objects<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>Source Image|
|---|
<br><br>|Task 4: Image Fusion<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>Target Image<br><br>|Ins-Editing<br><br>Model|
|---|
<br><br>Remove<br><br>extracted objects<br><br>Reference Image1 Reference Image2<br><br>|Referseg<br><br>Service|
|---|
<br><br>Source Image<br><br>[Figure 56]<br><br>[Figure 57]<br><br>|
|---|
<br><br>(a) Creation of Scribble-based Editing Data|
|---|

|(b) Creation of Scribble-based Generation Data<br><br>|[Figure 58]<br><br>Task 1&2: Scribble-based instruction and<br><br>Multimodal instruction-based generation data<br><br>[Figure 59]<br><br>Source Image Source Image<br><br>|
|---|
<br><br>|[Figure 60]<br><br>[Figure 61]<br><br>Source Image Source Image<br><br>Task 3: Doodle Generation|
|---|
|
|---|

|(c) Framework<br><br>|MM-DIT Models|
|---|
<br><br>Noisy Token<br><br>|VLM|
|---|
<br><br>Editing<br><br>Instruction<br><br>Translated<br><br>Editing<br><br>Instruction<br><br>|Text Encoder|
|---|
<br><br>Token Concat<br><br>|TokenText|
|---|
<br><br>TokenText<br><br>|ImageToken| |
|---|---|
| | |
<br><br>Eocoding (x,y,𝟎)<br><br>Output<br><br>|IMG1|
|---|
<br><br>|VAE|
|---|
<br><br>IMG1 Token<br><br>Eocoding (x,y,𝟏)<br><br>|Srcibble<br><br>IMG1 (opt)|
|---|
<br><br>|VAE|
|---|
<br><br>IMG1 Token<br><br>Eocoding (x,y,𝟏)<br><br>|IMG2| |
|---|---|
| | |
<br><br>|VAE|
|---|
<br><br>IMG2 Token<br><br><br>Eocoding (x+𝒘𝟏,y,𝟐)<br><br>…… …… ……<br><br>|IMGn| |
|---|---|
| | |
<br><br>Eocoding (x+𝐰𝟏+…+ 𝒘𝐧−𝟏,y,𝒏)<br><br>VAE<br><br>|IMGn Token|
|---|
<br><br>ImageToken<br><br>Joint input Scheme<br><br>Same Encoding Scheme<br><br>|
|---|

TokenText

- Figure 2. The overview of DreamOmni3’s training data construction and framework. The overview of DreamOmni3’s training data construction and framework: (a) We create scribble-based editing training data. For scribble and multimodal instruction-based editing, we use Referseg to locate edit objects and paste corresponding scribbles onto the source and reference images to create training pairs. For scribble and instruction-based editing, we omit the reference image. For doodle editing, we use a dedicated model to convert the edit objects into abstract sketches and paste them back into the source image. For image fusion, we crop objects from the reference image and paste them into the corresponding position on the source image to build training pairs. (b) Scribble-based generation training data is created similarly to editing, except the source image is replaced with a blank white canvas. (c) DreamOmni3 builds on the framework of DreamOmni2 [24], introducing a joint input scheme for scribble inputs. We also apply the same encoding scheme to both the source and scribbled images, ensuring better pixel alignment and perfect compatibility with previous image and language instruction editing.

|Eocoding (x+𝐰𝟎+…+ 𝒘𝐧−𝟏,y,𝒏 − 𝟏)|
|---|

library. After determining the object’s position in the reference image, we randomly select and resize a symbol from the library, then paste it to create a scribble-based reference image (S-Reference Image). For the source image, we modify the target image by either removing or altering the object, then add a corresponding scribble symbol, resulting in the scribble-based source image (S-source image). This process generates target images based on either the reference image (or S-reference image) and source image. The reference image can include or exclude scribbles depending on the instruction. Our editing types encompass a wide range of both concrete objects and abstract attributes, as demonstrated in DreamOmni2 [24]. (2) Scribble-based instruction editing: This task requires no special data preparation. We simply remove the reference image from (1) and add an object description to the instruction. (3) Doodle editing: We use Referseg to locate the editing object and the instruction-based editing model to turn it into a simple abstract doodle. We avoid using Canny edge detection due to users’ imperfect drawings, which require aesthetic cor-

rections from the model. Instead, we use the GPT-Image-1 model, which is better suited for generating doodles with non-strict pixel consistency. (4) Image fusion: In this task, users can extract objects from one image and insert them into the target image. We first remove the target object’s corresponding section using the instruction-based editing model. Next, we use Referseg to locate and crop the object from the reference image, then paste it into the target image after resizing it to fit. This results in the source image.

Position Encoding Shift Index Encoding

As shown in Fig. 2 (b), we divide scribble-based generation into three tasks: (1) For scribble-based multimodal instruction generation, the method is similar to that for scribble-based multimodal instruction editing, with the key difference being the placement of scribble symbols on a white canvas to create the source image. This enables the model to refer attributions or objects of the reference image and generate them at corresponding symbol positions. (2) For scribble-based instruction generation, we remove the reference image from step (1) and add descriptions of the removed objects in the instructions. (3) For doodle genera-

tion, the method is similar to doodle editing, with the final sketch placed on a white canvas to allow the model to generate the corresponding objects and background based on the sketch and instructions.

Our data is created based on DreamOmni2’s multireference image generation and editing training datasets. In our created dataset, the scribble-based editing dataset contains several types of data. The scribble-based multimodal instruction editing includes 32K training samples, while scribble-based instruction editing has about 14K training samples. The image fusion dataset consists of 16K training samples, and the doodle editing dataset contains 8K training samples. Notably, both scribble-based multimodal instruction editing and scribble-based instruction editing cover a wide range of editing categories. These include edits related to abstract properties, such as design style, color scheme, and hairstyle, as well as edits of concrete objects, such as various objects, people, and animals, with the ability to add, remove, or modify them. In contrast, image fusion and doodle editing primarily focus on the task of adding concrete objects into the image.

In the scribble-based generation dataset, scribble-based multimodal instruction generation has 29K training samples, scribble-based instruction generation has 10K training samples, and doodle generation has 8K training samples. Both scribble-based multimodal instruction generation and scribble-based instruction generation involve generating concrete objects as well as referencing abstract attributions. However, doodle generation is mainly focused on generating concrete objects.

#### 3.2. Framework and Training

The current unified generation and editing models primarily focus on instruction-based editing and subject-driven generation. Recently, DreamOmni2 extended this model to multi-reference generation and editing. However, the input format for doodle instructions still requires exploration. In DreamOmni3, we considered two input schemes: one using binary masks, similar to inpainting, and another using a joint input of the source image and the source image with doodles. Since doodles inevitably alter parts of the source image, but editing requires consistency in the nonedited areas, we need to input doodle information while preserving the source image details. There are two possible approaches: the traditional inpainting method with binary masks and our proposed joint input of the source image and the doodle-modified source image. Our approach offers two key advantages over binary masks in inpainting: (1) Simplicity and Efficiency: Our joint input is simpler and more efficient. Binary masks become problematic when there are multiple doodles in the reference or source image, as each doodle needs a separate mask, significantly increasing computational load. Moreover, using language to link doo-

dles in both images is challenging with binary masks. In contrast, our joint input allows for easy differentiation using colors during drawing, and simple language instructions can establish correspondences using image indices and doodle colors. (2) Better Model Integration: Existing unified generation and editing models are trained on RGB images. Our joint input scheme also uses masks in the original RGB space of the source image, allowing for better utilization of the model’s existing image-text understanding capabilities, and seamlessly integrating with the model’s original capabilities to create a more unified and intelligent creative tool.

Our framework design is shown in Fig. 2 (c), building upon the DreamOmni2 framework with additional adaptation for scribble instruction input. Our joint input scheme is optional. Specifically, when the source image in the editing task contains scribbles, we input both the source image and the scribbled source image into the MM-DIT models. However, if the reference images contain scribbles, we do not use the joint input scheme, as there is no need to maintain pixel-level consistency in the non-edited areas of the reference image, and adding an extra input would unnecessarily increase computational cost. For scribble-based generation tasks, we also avoid using the joint input scheme since there is no need for pixel-level preservation. Using the joint input scheme does introduce two challenges: (1) it adds an extra image, which affects the indexing of subsequent input images and could potentially confuse users, and (2) the model must correctly map pixel relationships between the source image and the doodle-modified source image. To address these, we use the same index encoding and position encoding scheme for both the source image and the scribbled source image. Experiments show that this encoding effectively resolves these issues, seamlessly integrating doodle editing capabilities into the existing unified generation and editing model framework.

During training, we use DreamOmni2’s VLM (Qwen2.5-VL 7B [16]) and FLUX Kontext’s joint training scheme. For more details, please refer to the supplementary. We train our model using LoRA with a rank of 256. Notably, by leveraging LoRA for training, we retain the original instruction-editing capabilities of Kontext. When a user inputs an image with scribbles into the model, our LoRA is activated, seamlessly integrating multimodal instruction-based editing and generation into the unified model. Additionally, since the multi-reference generation and editing capabilities in the previous DreamOmni2 model were trained separately with two LoRAs, our generation and editing models are also trained with separate LoRAs to ensure compatibility. The training process took approximately 400 A100 hours.

#### 3.3. Benchmark and Evaluation

We propose scribble-based editing and generation that integrates language, images, and scribble instructions. Currently, there is no dedicated benchmark to evaluate these tasks. To foster the development of this valuable direction, we introduce the DreamOmni3 benchmark. Our benchmark is comprehensive, consisting of real images to accurately assess model performance in real-world scenarios. The test cases cover a wide range of editing and generation tasks, including the four editing tasks and three generation tasks proposed in our paper. The editing categories are also diverse, encompassing both abstract property edits and concrete object edits. More details about the DreamOmni3 benchmark can be found in the supplementary materials.

Traditional metrics, such as DINO [28] and CLIP [12], are not sufficient to accurately evaluate the complex and diverse instruction-based editing and generation tasks we propose [11]. Recent works like Step1x-Edit [7] and Kontext [1] rely on VLMs and human evaluations for instruction editing. Given the increased complexity of our Scribblebased Editing and Generation tasks, evaluation must be based on VLMs. We propose a comprehensive set of evaluation standards for these new tasks, focusing on three key aspects: (1) accuracy in following the instructions in the generated edits, (2) consistency in human appearance, objects, and abstract attributes, (3) avoidance of severe visual artifacts, and (4) alignment of the generated or edited content with the specified doodle regions. Only when all these criteria are met do we consider the editing or generation task successful, ensuring relevance to real-world use cases. We apply the same standards for human evaluations, and our results show that VLM-based assessments align closely with human evaluations. The system prompt used for VLM evaluation can be found in the supplementary materials.

### 4. Experiments

Evaluation on Scribble-based Editing. As shown in Tab. 1, we compared several competitive unified generation and editing models, such as Omnigen2 [21], Qwenimage-Edit-2509 [20], and DreamOmni2 [24]. In addition, although Kontext does not natively support multi-image input, we applied a method from Diffusers [15] that combines multiple images into one for input. We also compared closed-source commercial models, Nano Banana [5] and GPT-4o [10]. We tested the performance of all models on the DreamOmni3 benchmark, which we constructed using real images for doodle editing, and evaluated the model outputs’ pass rates using Gemini 2.5 [4] and Doubao 1.6 [2]. Additionally, we had several people manually evaluate the results from different models and derive the pass rates. The evaluation standards are outlined in Sec. 3.3. For manual assessment, each case is evaluated by 5 reviewers, and a

Table 1. Quantitative comparison of scribble-based editing. We use Gemini [4] and Doubao [2] to evaluate the success editing ratio of different models on concrete objects and abstract attributions, respectively. In addition, “Human” refers to professional engineers assessing the editing success rates of all models.

Method Gemini↑ Doubao↑ Human↑ GPT-4o [10] 0.6125 0.5375 0.5875

Nano Banana [5] 0.5125 0.4250 0.4125 Omnigen2 [21] 0.1000 0.0875 0.0500

Kontext [1] 0.0875 0.1125 0.0250 Qwen-image-Edit-2509 [20] 0.2250 0.2000 0.1625

DreamOmni2 [24] 0.2000 0.2375 0.1750 DreamOmni3 (Ours) 0.5250 0.4500 0.5750

case is deemed successful if it receives approval from more than 3 evaluators. As can be seen, DreamOmni3 achieved the best results in human evaluations. For large language model evaluations, DreamOmni3 also outperformed opensource models and showed results comparable to commercial models. Notably, GPT-4o frequently exhibits a yellowing issue in the images, and the pixels in the non-edited areas of the output often do not match those in the input image. Meanwhile, Nano Banana showed several issues, such as copy-and-paste effects and incorrect object proportions. These issues are difficult for Vision Language Models (VLMs) to detect accurately.

Qualitative results are shown in Fig. 3. We present a variety of editing results, including scribble-based multimodal instruction edits in the first and second rows, scribble-based instruction edits in the third and fourth rows, image fusion results in the fifth row, and a comparison of doodle editing results in the sixth row. As seen, DreamOmni3 produces more accurate edits with better consistency.

Evaluation on Scribble-based Generation. As shown in Tab. 2, our method outperforms Nano Banana in both human evaluations and assessments by Doubao and Gemini, with performance similar to GPT-4o. In fact, we found that GPT-4o and Nano Banana are not specifically optimized for scribble scenarios, and even when explicitly instructed not to generate scribble marks, these models still often output scribbles along with the generated results. The evaluation standards are outlined in Sec. 3.3, and further details can be found in the supplementary. Compared to opensource models, the current SOTA models, DreamOmni2 and Qwen-image-edit-2509, also do not focus on or optimize for these new tasks, resulting in lower success rates on scribble-related tasks. This further demonstrates the effectiveness and necessity of our DreamOmni3 for scribblebased generation.

Quantitative results are shown in Fig. 4. We found that open-source models often retain scribbles in their outputs, which is related to the lack of task-specific optimization. Commercial models, Nano Banana and GPT-4o, perform relatively better, but Nano Banana exhibits some unnatural

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

[Figure 72]

Insert the toy in the red circle from the second

image into the red circle in the first image.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Make the bag in the red circle of the first

image have the same color scheme as the printer in the red circle of the second image.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Add a black car inside the blue circle.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Replace the woman in the red circle with a man.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

The man walk on rail. Fuse the image

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

OmniGen2 Kontext Qwen-Edit-2509 GPT-4o Nano Banana DreamOmni3 (Ours)

Inputs

DreamOmni2

- Figure 3. Visual comparison of scribble-based editing. Compared to other competitive methods and even closed-source commercial models (GPT-4o and Nano Banana), DreamOmni3 shows more accurate editing results and better consistency.

- Table 2. Quantitative comparison of scribble-based generation. We use Gemini [4] and Doubao [2] to evaluate the success editing ratio on concrete objects and abstract attributions, respectively. In addition, “Human” refers to professional engineers assessing the editing success rates of all models.

Method Gemini↑ Doubao↑ Human↑ GPT-4o [10] 0.5814 0.4884 0.3953

Nano Banana [5] 0.4651 0.4186 0.2326 Omnigen2 [21] 0.1163 0.0930 0.0465

Kontext [1] 0.0930 0.0465 0.1395 Qwen-image-Edit-2509 [20] 0.1628 0.0698 0.1163

DreamOmni2 [24] 0.1628 0.1163 0.0465 DreamOmni3 (Ours) 0.5116 0.4651 0.5349

collage-like artifacts.

Joint Input. As shown in Tab. 3, we compare the impact of training with our custom dataset versus using joint inputs of source images and scribbled source images on model performance. In Scheme 1, we use the base model, Kontext, which only takes the scribbled source image as input. In Scheme 2, we combine the source image and the scribbled source image as joint input but without training on our

dataset. In Scheme 3, we train the model on our dataset, but input only the scribbled source image. In Scheme 4, we use joint input and train the model on our dataset. Comparing Scheme 3 with Scheme 1, we observe that our dataset significantly improves the model’s scribble-based editing and generation capabilities. Comparing Scheme 3 with Scheme 4, joint input significantly boosts editing performance, while the improvement in generation tasks is less pronounced. This is because scribbles can obscure edited regions of the source image, so joint input ensures the model can see the original image information, improving editing consistency. In contrast, generation tasks don’t require high pixel-level consistency, making the improvement less significant. Based on these results, we only use joint input for editing tasks when the source image contains scribbles. For other reference images or generation tasks with scribbles, we input the scribbled image directly.

Index and Position Encoding. As shown in Tab. 4, we compare different encoding schemes for the source image and the scribbled source image as joint inputs. The results show that using the same index encoding and position en-

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

In the living

[Figure 120]

[Figure 121]

r o om , t h e

[Figure 122]

woman from

image 1 is

standing in

the red circle,

and the cat

from image 2

[Figure 123]

is sleeping in

t h e b l a c k

square, resting

on the table.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

In front of a house,

the man from image

[Figure 130]

[Figure 131]

[Figure 132]

1 is standing in the

green circle, the car

from image 3 is

[Figure 133]

parked in the black

[Figure 134]

circle, and the

backpack fr om

image 2 is placed in

the red square, on

the car's hood.

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

A man is in the

red circle, and a

woman with long

hair is in the green

circle. They are

shaking hands,

with a volcano in

the background.

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

A girl is standing

in the red circle,

and a boy is sitting

in the green circle.

They are watching

the sunset.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

A person with

rabbit ears is

watering a tree.

OmniGen2 Kontext Qwen-Edit-2509 GPT-4o Nano Banana DreamOmni3 (Ours)

DreamOmni2

Inputs

- Figure 4. Visual comparison of scribble-based generation. Our DreamOmni3 significantly outperforms current open-source models and achieves generation results comparable to closed-source commercial models (GPT-4o and Nano Banana).

- Table 3. Joint input scheme for scribble-based generation and editing.

Generation or Editing Model Training Joint Input Editing Generation

Method

- Scheme 1 ✗ ✗ 0.1125 0.0465

- Scheme 2 ✗ ✓ 0.1375 0.0465

- Scheme 3 ✓ ✗ 0.3500 0.4419

Scheme 4 (Ours) ✓ ✓ 0.4500 0.4651

Table 4. Different encoding schemes for scribble inputs.

Same Index Encoding

Same Position Encoding Editing Generation

Method

- Scheme 1 ✗ ✗ 0.3750 0.2791

- Scheme 2 ✗ ✓ 0.4000 0.4186

- Scheme 3 ✓ ✗ 0.4250 0.3488

Scheme 4 (Ours) ✓ ✓ 0.4500 0.4651

coding for both inputs yields the best performance. We believe this is due to two main reasons: (1) Matching the position and index encodings allows for better pixel-level alignment between the two images, enabling the model to more accurately locate the scribbles and preserve pixel-level information, leading to more consistent and precise editing. (2) Using the same encodings means that subsequent reference images do not need to modify their encodings, maintaining consistency with the original training format and allowing the model to better leverage its pre-trained editing capabilities for more accurate results.

### 5. Conclusion

The current unified generation and editing model performs image edits based on text instructions. However, language struggles to accurately describe edit locations and capture all the user’s intended details. To enhance this, we propose two tasks: scribble-based editing and generation, allowing users to simply use a brush in the GUI to make edits. This method can combine language, image, and scribble instructions, offering more flexibility. Building on this, we introduce DreamOmni3, addressing the challenge of limited training data. Using DreamOmni2 data, we developed a data creation scheme based on Referseg to generate highquality, accurate datasets that integrate scribbles, text, and image instructions. We also tackled the model framework issue, as binary masks are inadequate for complex realworld needs. When multiple masks are present, they are hard to distinguish and describe with language. To solve this, we propose a scribble-based approach, where different masks are easily differentiated by brush color, allowing for the handling of any number of masks. Since scribbles may obscure some image details, we introduce a joint input scheme, feeding both the original and scribbled images into the model. We further optimized this by using the same index and position encoding to preserve the details while

maintaining accurate correspondence to reference images. Lastly, we developed a benchmark and evaluation framework that combines scribbles, text, and image instructions, fostering the advancement of this field.

### References

- [1] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506,

2025. 6, 7

- [2] ByteDance. Doubao. https://www.doubao.com/,

2025. 6, 7

- [3] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2
- [4] Google. Gemini. https://deepmind.google/ models/gemini/, 2025. 6, 7
- [5] Google. Nano banana. https://aistudio.google. com/models/gemini-2-5-flash-image, 2025. 2, 6, 7
- [6] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023. 3
- [7] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 6
- [8] Zichen Liu, Yue Yu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Wen Wang, Zhiheng Liu, Qifeng Chen, and Yujun Shen. Magicquill: An intelligent interactive image editing system. In CVPR, 2025. 3
- [9] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instructionbased image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487, 2025. 3
- [10] OpenAI. Gpt-4o image generation. https : / / openai.com/index/introducing-4o-imagegeneration/, 2025. 2, 6, 7
- [11] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation. arXiv preprint arXiv:2406.16855, 2024. 6
- [12] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 6
- [13] Jaskirat Singh, Jianming Zhang, Qing Liu, Cameron Smith, Zhe Lin, and Liang Zheng. Smartmask: Context aware highfidelity mask generation for fine-grained object insertion and layout control. In CVPR, 2024. 3

- [14] Wensong Song, Hong Jiang, Zongxing Yang, Ruijie Quan, and Yi Yang. Insert anything: Image insertion via in-context editing in dit. arXiv preprint arXiv:2504.15009, 2025. 3
- [15] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 6
- [16] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 5
- [17] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. Instancediffusion: Instance-level control for image generation. In CVPR, 2024. 3
- [18] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024. 3
- [19] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024. 3
- [20] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 6, 7
- [21] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 6, 7
- [22] Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xinglong Wu, Yapeng Tian, Wenming Yang, and Luc Van Gool. Diffir: Efficient diffusion model for image restoration. In ICCV, 2023. 3
- [23] Bin Xia, Shiyin Wang, Yingfan Tao, Yitong Wang, and Jiaya Jia. Llmga: Multimodal large language model based generation assistant. In ECCV, 2024. 3
- [24] Bin Xia, Bohao Peng, Yuechen Zhang, Junjia Huang, Jiyang Liu, Jingyao Li, Haoru Tan, Sitong Wu, Chengyao Wang, Yitong Wang, et al. Dreamomni2: Multimodal instruction-based editing and generation. arXiv preprint arXiv:2510.06679, 2025. 4, 6, 7
- [25] Bin Xia, Yuechen Zhang, Jingyao Li, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, and Jiaya Jia. Dreamomni: Unified image generation and editing. In CVPR, 2025. 3
- [26] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In CVPR, 2023.
- [27] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In CVPR, 2023. 3
- [28] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr

- with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022. 6
- [29] Hong Zhang, Zhongjie Duan, Xingjun Wang, Yingda Chen, and Yu Zhang. Eligen: Entity-level controlled image generation with regional attention. arXiv preprint arXiv:2501.01097, 2025. 3
- [30] Dewei Zhou, You Li, Fan Ma, Xiaoting Zhang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. In CVPR, 2024. 3

