##### VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning

Zhong-Yu Li1,4* Ruoyi Du2,4* Juncheng Yan3,4 Le Zhuo4 Qilong Wu4 Zhen Li5† Peng Gao4 Zhanyu Ma2 Ming-Ming Cheng1† 1VCIP, CS, Nankai University 2Beijing University of Posts and Telecommunications 3Tsinghua University 4Shanghai AI Laboratory 5The Chinese University of Hong Kong

Project page: https://visualcloze.github.io

### arXiv:2504.07960v4[cs.CV]7Jan2026

||In-context examples<br><br>[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>+<br><br>+<br><br>= ?|
|---|
<br><br>Understand the task<br><br>Fill blank grid by reasoning<br><br>|[Figure 5]<br><br>| |
|---|
<br><br>[Figure 6]<br><br>| |?|
|---|---|
<br><br>|[Figure 7]|
|---|
<br><br>Visual Prompt Target|
|---|
<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>Layout Style<br><br>Subject<br><br>[Figure 12]<br><br>Extend to diverse tasks||[Figure 13]<br><br>Image Restoration|[Figure 14]|
|---|---|
<br><br>|[Figure 15]<br><br>Condition genera|[Figure 16]<br><br>tion|
|---|---|
<br><br>|[Figure 17]<br><br>Dense Prediction|[Figure 18]|
|---|---|
<br><br>|[Figure 19]<br><br>Pose Estimation|[Figure 20]|
|---|---|
|
|---|---|
||[Figure 21]<br><br>Multi-View|[Figure 22]|
|---|---|
<br><br>|[Figure 23]<br><br>Subject-driven|[Figure 24]<br><br>Result1|[Figure 25]<br><br>Result2|
|---|---|---|
<br><br>|[Figure 26]<br><br>Editing (Add)|[Figure 27]|
|---|---|
<br><br>|[Figure 28]<br><br>Relighting|[Figure 29]|
|---|---|
<br><br>|[Figure 30]<br><br>Doodle|[Figure 31]|
|---|---|
<br><br>|[Figure 32]<br><br>Style|
|---|
<br><br>|[Figure 33]|
|---|
<br><br>|[Figure 34]<br><br>Transfer|
|---|
<br><br>Tra<br><br>|[Figure 35]<br><br>Virtual Try-On|[Figure 36]|[Figure 37]|
|---|---|---|
<br><br>|[Figure 38]<br><br>Subject + Layout|
|---|
<br><br>|[Figure 39]<br><br>+ Style|
|---|
<br><br>|[Figure 40]|[Figure 41]|
|---|---|
<br><br>ut<br><br>|[Figure 42]<br><br>Editing (Remove)|[Figure 43]|
|---|---|
<br><br>|[Figure 44]<br><br>Subject-driv|[Figure 45]<br><br>en Editing|[Figure 46]|
|---|---|---|
<br><br>|[Figure 47]<br><br>Drawing|[Figure 48]|[Figure 49]|[Figure 50]|
|---|---|---|---|
<br><br>|[Figure 51]<br><br>Multi-View|[Figure 52]|
|---|---|
| |

Figure 1. The top left illustrates our universal image generation framework based on visual in-context learning. Given one query of a specific task, the generative model learns the task by observing a few in-context examples presented as demonstrations. For each task, the generation result is indicated by a red box.

|Without In-context Example<br><br>|[Figure 53]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 54]<br><br>Target|
|---|
|
|---|

|+ Two In-context Example<br><br>|[Figure 55]<br><br>Target|
|---|
<br><br>[Figure 56]<br><br>[Figure 57]<br><br>|[Figure 58]<br><br>Visual Prompt|
|---|
<br><br>[Figure 59]<br><br>[Figure 60]|
|---|

||[Figure 61]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 62]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 63]<br><br>Target|
|---|
<br><br>Without In-context Example|
|---|

|[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>|[Figure 70]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 71]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 72]<br><br>Target|
|---|
<br><br>+ Two In-context Examples|
|---|

|[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>|[Figure 76]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 77]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 78]<br><br>Target|
|---|
<br><br>+ One In-context Example|
|---|

|+ One In-context Example<br><br>|[Figure 79]<br><br>Target|
|---|
<br><br>[Figure 80]<br><br>[Figure 81]<br><br>|[Figure 82]<br><br>Visual Prompt|
|---|
|
|---|

Each row shows a process to edit the image with the given editing instruction. The editing instruction in the last row is: making [IMAGE1] the standing woman [IMAGE2] sit down and give the thumbs up.

Each row presents multi-view of a face, given a frontal face reconstruction task that leverages [IMAGE1] a left side of the face and [IMAGE2] a right side of the face, to generate [IMAGE3] that faces the center of the lens. The the last image in the final row is: the woman's frontal face that faces the center of the lens.

- Figure 2. Unseen Tasks : Generalizing to tasks unseen during training via in-context learning. More in-context examples lead to more accurate results.

###### Abstract

Recent progress in diffusion models significantly advances various image generation tasks. However, the current mainstream approach remains focused on building task-specific models, which have limited efficiency when supporting a wide range of different needs. While universal models attempt to address this limitation, they face critical challenges, including generalizable task instruction, appropriate task distributions, and unified architectural design. To tackle these challenges, we propose VisualCloze, a universal image generation framework, which supports a wide range of in-domain tasks, generalization to unseen ones, unseen unification of multiple tasks, and reverse generation. Unlike existing methods that rely on language-based task instruction, leading to task ambiguity and weak generalization, we integrate visual in-context learning, allowing models to identify tasks from visual demonstrations. Meanwhile, the inherent sparsity of visual task distributions hampers the learning of transferable knowledge across tasks. To this end, we introduce Graph200K, a graph-structured dataset that establishes various interrelated tasks, enhancing task density and transferable knowledge. Furthermore, we uncover that our unified image generation formulation shared a consistent objective with image infilling, enabling us to leverage the strong generative priors of pre-trained infilling models without modifying the architectures.

###### 1. Introduction

Recent advancements in image generation, propelled by the progress of diffusion models [15, 33, 88], have led to a

∗ Equal contribution † Corresponding author

wide range of applications, including image editing [69], style transfer [64, 81], virtual try-on [11, 12], and personalized generation [38, 54], among others. However, these tasks typically require task-specific models, which limit efficiency and scalability for real-world applications. In recent years, there has been growing interest in universal generative models [27, 39, 44], aiming to handle diverse image generation tasks, even unseen ones, within a single unified framework. Despite significant progress, some critical issues remain to be addressed, such as (1) distinguishable and generalizable task instruction, (2) comprehensive task coverage during training, and (3) a unified model architecture.

An ideal task instruction is crucial for guiding the model to process the desired task effectively. Existing methods primarily rely on language instructions [27, 44] or task-specific tokens [39] to distinguish the task to be performed. However, the complexity of visual tasks and the inherent gap between vision and language modalities make it hard for the model to understand language-only task descriptions, which leads to task confusion [39] and hinders generalization on unseen tasks [35, 71]. Moreover, pre-learned task-specific tokens constrain the model only to handle seen tasks. In contrast, large language models (LLMs) have successfully achieved unified multi-task modeling, partially due to the rise of in-context learning [5], which allows models to adapt various tasks using only a few demonstrations. We aim to replicate the concept of in-context learning in the pure visual modality, where the model learns the desired task directly from a few visual examples as task demonstrations, as shown in Fig. 1 (Left Top). In this setting, in-context learning shows strong potential for universal image generation. We summarize four key findings: (1) it supports various in-domain tasks with reduced task ambiguity (Fig. 1);

In-ContextExample

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

|[Figure 90]<br><br>Visual Prompt|
|---|

|[Figure 91]<br><br>Target|
|---|

|[Figure 92]<br><br>Visual Prompt|
|---|

|[Figure 93]<br><br>Target|
|---|

|[Figure 94]<br><br>Target|
|---|

|[Figure 95]<br><br>Target|
|---|

|[Figure 96]<br><br>Target|
|---|

Depth conditioned generation and relighting in one step.

Estimate 1) depth, 2) surface normal, and 3) edges in one step.

- Figure 3. Unseen Tasks : Leveraging in-context learning to unify multiple seen tasks into a single-step unseen task. Left: Unifying the [Depth to Image] and [Relighting] task into a single [Depth to Images with Various Lighting] task. Right: Unifying multiple dense prediction tasks into a joint prediction task. Results without visual context can be found in the appendix.

(2) it generalizes to unseen tasks (Fig. 2, Fig. 8); (3) as an unseen strategy for task unification, it can integrate multiple sub-tasks into a single step and generate intermediate results (Fig. 3); (4) it enables reverse generation, i.e., inferring a set of conditions from a given target (Fig. 9). While prior works [1, 3, 4, 43, 66, 71, 82] have also explored in-context learning in vision, they are largely constrained to specific domains (such as dense prediction or style transfer [67, 87]), or simplified generation settings involving only one condition and one target image [43, 60].

From the perspective of task distribution, visual tasks are inherently sparse compared to those in natural language processing because task-specific datasets [71, 85] for different tasks have minimal overlap [19, 32, 79]. Such sparse task learning isolates the knowledge of each task and limits the model from learning shared features across tasks. Moreover, the weak correlations between tasks hinder knowledge transfer and adaptability to new tasks. However, existing works in multi-task learning [10, 16, 31, 53] have verified the benefits of overlapping knowledge across related tasks. To alleviate the sparsity of visual tasks, we introduce a graph-structured dataset, Graph200K, where each image is associated with annotations spanning five metatasks, i.e., conditional generation [80], IP preservation [76], style transfer [81], image editing [69], and restoration [77]. By combining different conditions, we train the model with a variety of tasks that overlap with each other. Given this highly overlapping and compact task space, our dataset significantly increases task density, allowing the model to learn shared and transferable knowledge more effectively.

For the architecture design, it is essential to 1) accommodate flexible task formats [27, 35, 71], ensuring seamless in-context learning, and 2) remain compatible with state-ofthe-art models [33, 88] to fully leverage their strong generative priors. In this work, we find that the state-of-the-art image infilling model [33] has a consistent objective with our

in-context learning based universal generative formulation. Specifically, we concatenate all input and output images together, where the objective of a task is to fill the output area. This alignment enables us to build our model upon advanced general-purpose infilling models without additional modifications, achieving powerful universal generation capabilities with minimal data and training costs.

In this work, we propose a universal image generation framework, VisualCloze, which fine-tunes FLUX.1-Filldev [33] with interrelated tasks sampled from Graph200K to learn transferable knowledge and support visual in-context learning. As the number of in-context examples increases, we observe enhanced performances and reduced task confusion, enabling the model to support a broad spectrum of in-domain tasks, including conditional generation, image restoration, editing, style transfer, IP-preservation, and their combinations. On unseen tasks, the model also shows a certain degree of generalization ability, as shown in Fig. 2. In summary, our main contributions are as follows:

- • We propose an in-context learning based universal image generation framework that supports a wide range of indomain tasks and exhibits generalization to unseen ones.
- • We design a graph-structured dataset, Graph200K, which constructs a compact task space, enabling flexible online task sampling and promoting the models to learn shared and transferable knowledge across tasks.
- • Our unified image generation formulation shares a consistent objective with the state-of-the-art infilling model, enabling exceptional performance through minimal tuning without modifying the structure.

###### 2. Related Work

###### 2.1. Image Generation

Recent advances in text-to-image generation have achieved remarkable performance, largely driven by the development

of autoregressive models [41, 58, 78] and diffusion models [2, 13, 15, 18, 24, 40, 42, 48, 51]. Among these, rectified flow transformers [15, 17, 33, 88] have shown great training efficiency and overall performance. Building on these foundational models, diverse applications have emerged, such as conditional generation [80], style transfer [64], and personalized generation [38]. More recently, universal models that address various tasks [35, 44, 83] have been explored. For example, unified models like OmniGen [71] leverage large vision language models to consolidate multiple tasks into a single framework. Similarly, UniReal [9] unifies image generation tasks as discontinuous video generation. However, they still face issues such as over-reliance on language instructions, isolation and sparsity of visual tasks, and architecture design accommodating flexible task formats. To address these issues, we propose a universal image generation framework that unifies generation tasks as image infilling. Through visual in-context learning and our Graph200K dataset that constructs a denser task space to learn transferable knowledge, our method alleviates ambiguity to support a diverse set of in-domain tasks and generalizes to tasks unseen during training.

|Condition<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]|
|---|

|Image<br><br>[Figure 101]|
|---|

|Restoration<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]|
|---|

|Editing<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]|
|---|

|Style Transfer<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]|
|---|

|Reference<br><br>[Figure 117]|
|---|

subject + layout + style = stylization

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

Combining diverse tasks

Figure 4. Illustration of the proposed Graph200K dataset. Each image is annotated for five meta-tasks, i.e., conditional generation, image restoration, image editing, IP preservation, and style transfer. Using these tasks, we can combine a wide range of complex tasks, such as the bottom of the figure.

###### 2.2. Visual In-context Learning

Along with the emergence of large language models, such as GPT-3 [5], in-context learning [14] has been an effective approach to allow the language model to understand and perform complex tasks given a few demonstrations. Early works [21, 22] in vision modality propose image analogies to create an image filter from examples automatically. In recent years, leveraging inpainting model [3, 4, 82], masked image modeling [43, 66, 67], or vision-language model [1, 86], visual in-context learning is proposed to handle more tasks. However, they mainly focus on dense prediction [55, 59, 87] or visual understanding [63]. OmniGen [71] also leverages in-context learning to generalize to unseen domains, e.g., segmenting unseen concepts when the model has learned the segmentation task during training. However, it mainly focuses on simple tasks of dense prediction, and the gap between the unseen and training domains is still limited. Some recent works [34, 43, 60, 68] extend visual in-context learning to image generation, but they are still limited by simple tasks such as conditional generation and dense prediction. Moreover, the sparsity of visual tasks makes it difficult for models to learn transferable and overlapping knowledge across tasks, limiting the generation ability of in-context learning. In contrast, we introduce a graph-structured dataset that supports interrelated tasks and thus constructs a more dense task space, promoting the model to learn shared and transferable knowledge and enhance its adaptability.

###### 3. Dataset

Recent works [26, 44, 71] have made great progress in unified image generation. However, their generalization to unseen tasks remains highly limited. We partially attribute this issue to the sparsity and isolation of visual tasks, hindering the model from learning shared features across tasks and handling unseen ones. Moreover, weak correlations between tasks further hinder knowledge transfer, restricting the adaptability of models. Therefore, increasing task density or strengthening task inter-relations helps improve the generalization ability of models via a compact task distribution. In this paper, we take the Subject200K [61] dataset as a starting point and construct our Graph200K dataset by augmenting each image with 49 types of annotations spanning five meta-tasks. This enriched annotation space enables flexible construction of a wide range of related tasks by sampling and combining arbitrary subsets of annotations across different meta-tasks, as illustrated in Fig. 4.

###### 3.1. Graph-Structured Multi-Task Dataset

In natural language processing, tasks overlap significantly, facilitating strong cross-task learning ability. In contrast, visual tasks are inherently distinct, posing challenges for vision models to achieve similar generalization ability via

instruction tuning. To ease this issue, we introduce a GraphStructured Multi-Task Dataset. As illustrated in Fig. 4 (a), given a text-to-image dataset, each image is treated as the central node of a graph, around which diverse task annotations are constructed, including those for various spatial conditions, degradations, image editing results, reference image for IP-preservation, and style transfer with various reference styles. The construction process for each task pair is detailed in the next section.

As shown in Fig. 4, each task annotation forms a bidirectional edge with the image. Thus, the graph is strongly connected, which means that for any two nodes, bidirectional paths exist between them. In other words, a generation task can be formulated as a path within the graph. The nodes along a path (except the end node) serve as condition images, which is analogous to the question in instruction fine-tuning, while the target image (the end node) plays the role of the answer. Specifically, there are 49 types of nodes in our Graph200K, and we sample up to 134 highly overlapping tasks, making the model learn more compact and shared representations across tasks. Moreover, it enriches the diversity and flexibility of our instruction fine-tuning data. For example, the path reference → editing → image corresponds to the task of image editing with reference, as shown in Fig. 4 bottom.

###### 3.2. Dataset Construction

For convenience, we inherit subject-driven data from the Subjects200K [61]. Additionally, 32 different degradations are applied online to the images to acquire restoration data. We summarize the data construction methods in this section for the remaining three tasks.

Conditional generation. Each image is paired with 12 distinct conditions generated by specialized models, including canny edges [6], HED edges [72], Hough lines [20], semantic segmentation maps [37], depth maps [74], shape normal maps [73], and human keypoints [7], following ControlNet [80]. This work extends the conditions by incorporating SAM2 [50] masks, foreground segmentation, and open-world boxes and masks. The foreground segmentation, derived from the RMBG [84], supports diverse tasks such as inpainting and foreground extraction. Open-world bounding boxes are generated through the grounding caption capability of Qwen2-VL [65], which are processed using SAM2 [50] to produce corresponding masks.

Style transfer. We transfer the style of images according to reference in both semantic-variant and semanticinvariant settings. Specifically, the semantic-invariant transfer adopts InstantStyle [64] to preserve the semantic content, while the semantic-variant transfer relies on FLUX.1Redux-dev [33], using the style embeddings and depth as

conditions. For each image, we randomly generate five stylized versions. Mixing the two tasks pushes the model to follow the in-context examples better to avoid ambiguity.

Image editing. We design two types of editing tasks, including background-variant and background-invariant editing. The background-invariant editing begins with localizing the subjects. Then, we leverage a large vision-language model, Qwen2-VL [65], to modify the image caption with a new object that replaces the original subject. The image, with the subject masked, is subsequently processed by the FLUX.1-Fill-dev [33] inpainting model to integrate the alternative object into the masked region. The above operation is repeated five times to enrich the dataset. For background-variant editing, the difference lies in the last step, which utilizes FLUX.1-Redux-dev [33] with depth as the condition and the modified caption as the text prompt.

###### 3.3. Other Data

To further expand the range of tasks and enhance the generalization ability of models, we incorporate several opensource datasets during training, including VITON-HD [11] for virtual try-on and PhotoDoodle [28] for artistic image editing. For image editing tasks, we also extend the dataset with OmniEdit [69]. Specifically, two sub-tasks, i.e., object addition and removal, are used for training. The other editing tasks, such as attribute modification and environment change, are treated as unseen tasks to assess the generalization ability of the trained model. Furthermore, we leverage a portion of high-quality internal data, covering tasks of the drawing process [62] and multi-view generation [29].

###### 4. Method

This paper identifies the core challenges in building a universal image generation model, including the need for a clearly defined and generalizable task formulation, visual task sparsity, and the lack of a unified framework for multitask learning. In the previous section, we addressed the issue of task sparsity by constructing the compact Graph200K dataset. Sec. 4.1 introduces visual in-context learning as the ideal paradigm for universal task formulation. Afterward, Sec. 4.2 considers the image infilling model a unified multi-task framework, achieving strong generalization capabilities with minimal cost.

###### 4.1. Visual In-context Learning

Language instructions are usually used to specify the generation definition to handle multiple visual generation tasks with a single generative model. However, due to the gap between vision and language, the text comprehension ability of image generation models remains limited. This issue leads to task confusion [39] in existing universal generative models and weak generalization to unseen tasks. Inspired

with a size of (L × W,(C + 1) × H). Then, the model can complete a task by infilling the target grids based on the surrounding context, akin to solving visual cloze puzzles. Therefore, we build our unified framework, VisualCloze, based on the general image infilling architecture capable of handling multiple resolutions.

concatenation

query

examples+

[Figure 122]

[Figure 123]

[Figure 124]

temporal

context

[Figure 125]

[Figure 126]

[Figure 127]

in-

𝐶

|[Figure 128]<br><br>Visual Prompt|
|---|

|?<br><br>Target|
|---|

|[Figure 129]<br><br>Visual Prompt|
|---|

Consistent with common diffusion-based infilling model designs, our model can be formulated as follows:

Xˆ = f(X | T,M), (1)

spatial concatenation (𝐿 images)

where X is the concatenated image, with the last grid left blank, T is the language instruction, M is the mask condition, and Xˆ represents the infilled result. The mask M is a binary matrix with the size of (H × (C + 1),W × L):

Figure 5. Concatenating images when applying position embeddings. The L images within C in-context examples and the query are first concatenated horizontally. Then, these concatenated rows are concatenated temporally to handle mismatched aspect ratios.

 

1 if i ∈ [H × (C − 1),H × C)

(2)

and j ∈ [W × (L − 1),W × L), 0 otherwise,

M(i,j) =



by the success of few-shot learning on large language models [5], we recognize that visual context may serve as a more friendly task instruction for visual generative models, given their superior visual understanding capabilities.

where M(i,j) = 1 indicates that the pixel will be masked and generated by the infilling model. Equ. (2) masks the region in the last row and column, i.e., the target image. During training, we also randomly mask one of the first L − 1 grids with a probability of 0.5, promoting reverse generation shown in Sec. 5.1. For the inference stage, we can crop Xˆ to obtain the target image easily.

Therefore, in this paper, we re-propose visual in-context learning to build a universal and generalizable image generation system. For the sake of description, here we assume the image input-output of arbitrary conditional generation task as a query consisting of L − 1 condition images and a blank target ∅ to be completed by the model, i.e., X = concat({x1,...,xL−1,∅}). In Sec. 5.1, we demonstrate that our method can be extended to more general scenarios, where it can generate images at arbitrary positions and in any quantity rather than just the single image at the end of the query. During training, we randomly provide up to C in-context examples, each containing L images as the query. This strategy ensures the generalization ability of models across different numbers of in-context examples. In our experiments, we show that providing in-context examples as task demonstrations not only helps alleviate task confusion and boost model performance across in-domain tasks [39], but also enhances the generalization ability on unseen tasks.

Aligned optimization objective. A key benefit of this design is that our VisualCloze formulation shares a highly consistent objective with general image infilling models without architectural modifications or explicit input conditions. This consistency allows us to directly fine-tune advanced image infilling models using the newly constructed dataset while maximizing the utilization of the prior knowledge of foundation models. In contrast, existing taskspecific models often require introducing additional learnable modules [38, 69] or adapting to extra condition inputs [61], which may compromise the native capabilities of the model.

Language instructions. Note that the design of language instruction is also necessary for VisualCloze because it is responsible for defining the grid image layout, describing the caption of the image to be generated, and specifying the task intent when in-context examples are unavailable. In our unified framework, the instruction consists of three parts: (1) layout instruction, which describes the (C +1) ×W layout of the grid image; (2) task instruction, which specifies the task type; and (3) content instruction, which describes the content of the target image. The details about the instructions are available in Appendix A. By restructuring the three components X, T, and M in Equ. (1), we achieve a unified multi-task framework for image generation with the general image infilling paradigm and support in-context learning.

###### 4.2. Unified Multi-task Framework

Unlike previous visual in-context learning methods that primarily focus on scenarios with a single image condition and a single context [43, 60], in this work, we aim to construct a unified framework capable of handling varying numbers of conditions and contexts, allowing for flexible adaptation to diverse tasks. For ease of description, we first assume all images processed by the model share the same size, W ×H, and we extend to the scenario with mismatched aspect ratios at the end of this section. In this way, given C in-context examples and the query, each containing L images, all images can be concatenated into a complete grid-layout image

Positional embedding. In the preceding section, all images are concatenated into a grid-layout image and we can apply positional embedding (i.e., RoPE [57]) on this large image. However, a potential limitation lies in composing a grid image from in-context examples with varying aspect ratios. To overcome this issue, we leverage the 3D-RoPE in Flux.1-Fill-dev to concatenate the query and in-context examples along the temporal dimension, as shown in Fig. 5, effectively overcoming this issue without introducing any noticeable performance degradation.

- 4.3. Implementation Details

We use FLUX.1-Fill-dev [33] as our foundation model, considering its outstanding performance among open-source image infilling models. In this work, LoRA [25] is chosen to fine-tune the model instead of fully fine-tuning it to reduce training costs and preserve the capabilities of the foundation model. The resulting LoRA can also be fused with other LoRAs in the community, enabling more widespread applications. Specifically, we set the rank of LoRA as 256. The model is tuned for 20,000 iterations with an accumulated batch size of 64 on 8 × A100 GPUs. We employ the AdamW optimizer with a learning rate of 1e−4. Following FLUX.1-Fill-dev, we incorporate the lognorm noise strategy with dynamic time shifting. During training, the number of in-context examples is set up to 2 (i.e., C as defined in Sec. 4.2), while L, the number of images involved in a task, varies between 2 and 4 in the Graph200K dataset. During inference, the number of in-context examples can be generalized to a larger number. To balance computational efficiency, each image is resized to the area of 384 × 384 or 512 × 512 before concatenating them into a grid layout. High-resolution outputs can be obtained in practical applications through simple post-up-scaling techniques [45].

- 5. Experiments

- 5.1. Qualitative Analysis of In-context Learning

This section presents a series of experiments demonstrating the effectiveness of in-context learning across different tasks, especially those unseen during training. Based on our extensive experiments, we summarize four key findings that highlight the role of in-context learning.

In-Context Learning Findings 1

In-context learning can mitigate task confusion for seen tasks.

Task ambiguity on seen tasks. The model occasionally experiences task confusion, failing to interpret the intended objective accurately, especially on dense prediction tasks. In-context learning effectively alleviates this issue

||[Figure 130]|
|---|
<br><br>|[Figure 131]|
|---|
<br><br>|[Figure 132]|
|---|
<br><br>Without In-context Learning|
|---|

||[Figure 133]|
|---|
<br><br>+ In-context Learning<br><br>|[Figure 134]|
|---|
<br><br>|[Figure 135]|
|---|
|
|---|

|[Figure 136]|
|---|

- (a) Image to Pose
- (b) Image to Depth
- (c) Image to Edge
- (d) Normal to Image

||[Figure 137]|
|---|
<br><br>|[Figure 138]|
|---|
<br><br>|[Figure 139]|
|---|
<br><br>Without In-context Learning|
|---|

||[Figure 140]|
|---|
<br><br>|[Figure 141]|
|---|
<br><br>|[Figure 142]|
|---|
<br><br>+ In-context Learning|
|---|

|[Figure 143]|
|---|

||[Figure 144]|
|---|
<br><br>|[Figure 145]|
|---|
<br><br>|[Figure 146]|
|---|
<br><br>Without In-context Learning|
|---|

||[Figure 147]|
|---|
<br><br>|[Figure 148]|
|---|
<br><br>[Figure 149]<br><br>+ In-context Learning|
|---|

|[Figure 150]|
|---|

||[Figure 151]|
|---|
<br><br>|[Figure 152]|
|---|
<br><br>|[Figure 153]|
|---|
<br><br>Without In-context Learning|
|---|

||[Figure 154]|
|---|
<br><br>|[Figure 155]|
|---|
<br><br>[Figure 156]<br><br>+ In-context Learning|
|---|

|[Figure 157]|
|---|

Figure 6. In-context learning mitigates the task ambiguity in seen tasks. We show three results using different initial noises.

by providing task-specific demonstrations. For example, in Fig. 6 (a) and (c), the model may produce noisy results without in-context examples in pose estimation and edge detection, while increasing the number of in-context examples enhances the performance and stability. In depth estimation shown in Fig. 6 (b), in-context examples also improve the accuracy when the model originally makes inaccurate estimates, especially in distant areas. Additionally, in some tasks like conditional generation, we note that the model can generate satisfactory results stably even without in-context examples, as shown in Fig. 6 (d). However, the quantitative comparison in Tab. 1 still shows that using in-context learning can further improve the accuracy of task completion.

In-Context Learning Findings 2

In-context learning supports generalization to unseen tasks, where providing more in-context examples could lead to more accurate generation.

Generalization on unseen tasks. Beyond mitigating task confusion, in-context learning also enables the model to generalize to tasks unseen during training. Fig. 2 has shown the model can successfully generate frontal faces from sideview images and transfer editing instructions [8] through in-context learning, even though they are not encountered during training. Here, we present additional examples of unseen tasks. For instance, although the model is trained exclusively on image editing tasks involving object addi-

|[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>|[Figure 161]<br><br>Visual Prompt|
|---|
<br><br>[Figure 162]<br><br>|[Figure 163]<br><br>Target|
|---|
<br><br>+ Two In-context Examples|
|---|

|Without In-context Example<br><br>|[Figure 164]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 165]<br><br>Target|
|---|
|
|---|

|[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>+ Two In-context Examples<br><br>|[Figure 170]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 171]<br><br>Target<br><br>|
|---|
|
|---|

||[Figure 172]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 173]<br><br>Target|
|---|
<br><br>Without In-context Example|
|---|

|[Figure 174]<br><br>[Figure 175]<br><br>|[Figure 176]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 177]<br><br>Target<br><br>|
|---|
<br><br>+ One In-context Example|
|---|

|+ One In-context Example<br><br>[Figure 178]<br><br>[Figure 179]<br><br>|[Figure 180]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 181]<br><br>Target|
|---|
|
|---|

[Figure 182]

[Figure 183]

Task Prompt: In each row, a logical task is demonstrated to achieve [IMAGE2] a high-aesthetic image based on [IMAGE1] an aesthetically pleasing photograph. Each row shows a process to edit the image with the given editing instruction. The editing instruction in the last row is: <editing instruction> change the setting to a winter scene. <\editing instruction>

Task Prompt: In each row, a logical task is demonstrated to achieve [IMAGE2] a high-aesthetic image based on [IMAGE1] an aesthetically pleasing photograph. Each row shows a process to edit the image with the given editing instruction. The editing instruction in the last row is: <editing instruction> turn the color of sunglasses to green. <\editing instruction>

- Figure 7. Unseen Tasks : Although the image editing tasks seen by the model are only about object addition and object removal, it can still generalize to other types of editing tasks, such as environment modification (Left) and attribute transformation (Right), through in-context learning. More unseen tasks are shown in Fig. 2.

|[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>+ Two In-context Examples|
|---|

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>+ Two In-context Examples|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

a purple sneaker and a toy on the table.

a animal toy and a blue bag on the beach

- Figure 8. Unseen Tasks : VisualCloze is capable of performing multi-subject driven generation [70], even though the model was only exposed to single subject-driven generation tasks during training. Best viewed by zooming in.

Multi-task consolidation. Meanwhile, we also find that through in-context learning, we can consolidate multiple tasks into a single execution step, which can be viewed as another form of unseen task. Fig. 3 has shown two examples, where we 1) merge conditional generation and relighting shown on the left and 2) perform depth estimation, surface normal estimation, and edge detection simultaneously shown on the right. Similarly, Fig. 11 illustrates how we can combine multiple conditions for conditional generation to achieve finer control. For instance, generating a portrait based on keypoints provides only rough information about the location and body pose. In such cases, contour conditions can be used to control the attributes of other visual elements.

In-Context Learning Findings 4

Different in-context learning examples lead to varying effects, where examples that can better convey mission intent can achieve better and more stable generation.

tion and removal, it still generalizes to other types of editing tasks, such as environment changes and attribute modifications, as shown in Fig. 7. Furthermore, as demonstrated in Fig. 8, the model, trained solely on single-subject generation, can generate images preserving identities of multiple subjects. These results highlight that in-context learning is an effective guidance mechanism, enabling adaptation to novel tasks without retraining.

Varying effects of different in-context examples. Following prior works [46, 52] on the prompt selection, we also find that different in-context examples could impact the generation quality. Specifically, it is crucial that in-context examples provide correct and strong guidance about the task intention. For example, as shown in Fig. 10 (left), when the side faces are more towards the front than in Fig. 10 (right), the success rate of correctly generating frontal faces has dropped dramatically.

In-Context Learning Findings 3

In-context learning enables task unification, an unseen strategy that consolidating sub-tasks into a single step and generating intermediate results.

||[Figure 202]<br><br>Visual Prompt|
|---|
<br><br>|[Figure 203]<br><br>Target|
|---|
<br><br>|[Figure 204]<br><br>Target|
|---|
<br><br>Task Prompt: In each row, a method uses[IMAGE1] gray-shaded depth map with distinct edges, [IMAGE2] Artistically rendered content for generating [IMAGE3] High-definition picture in a unique art style.<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>TwoIn-ContextExamples|
|---|

|[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>|[Figure 217]<br><br>Visual Prompt|
|---|
<br><br>[Figure 218]<br><br>[Figure 219]<br><br>Task Prompt: Every row demonstrates how to transform [IMAGE1] an image with vivid details into [IMAGE2] gray-scale depth map with clear object boundaries, [IMAGE3] rgb normal map for bump mapping effects, [IMAGE4] soft-edged map from hed detection through a logical approach.<br><br>|[Figure 220]<br><br>Target|
|---|
<br><br>|[Figure 221]<br><br>Target|
|---|
<br><br>|[Figure 222]|
|---|
<br><br>TwoIn-ContextExamples|
|---|

TwoIn-ContextExamples

TwoIn-ContextExamples

- Figure 9. Unseen Tasks : Through in-context learning, we can perform reverse generation from targets to conditions. For example, (a) decomposing the layout and style from a stylized image and (b) inferring the image, depth, and surface normal simultaneously from an edge map, which is the reverse task of Fig. 3 (Left).

|[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>|[Figure 229]|
|---|
<br><br>|[Figure 230]|
|---|
<br><br>TwoIn-contextExamples<br><br>|?|
|---|
|
|---|

TwoIn-contextExamples

|[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>|[Figure 237]|
|---|
<br><br>|[Figure 238]|
|---|
<br><br>|?|
|---|
<br><br>TwoIn-contextExamples|
|---|

||[Figure 239]<br><br>[Figure 240]|
|---|
<br><br>|[Figure 241]<br><br>[Figure 242]|
|---|
<br><br>|[Figure 243]<br><br>[Figure 244]|
|---|
<br><br>100% success among three trials|
|---|

||[Figure 245]<br><br>[Figure 246]<br><br>❌|
|---|
<br><br>|[Figure 247]<br><br>[Figure 248]<br><br>❌|
|---|
<br><br>|[Figure 249]<br><br>[Figure 250]|
|---|
<br><br>33% success among three trials|
|---|

TwoIn-contextExamples

- Figure 10. Illustration of the impact of different in-context examples on in-context learning. In the second example on the left, the left and right faces are too biased towards the front, so they do not show the core goal of the task intention.

[Figure 251]

[Figure 252]

[Figure 253]

|[Figure 254]<br><br>Visual Prompt|
|---|

|[Figure 255]<br><br>Visual Prompt|
|---|

|[Figure 256]<br><br>Target|
|---|

Task Prompt: Every row demonstrates how to transform [IMAGE1] human pose with colored lines for bone structure and [IMAGE2] canny map with sharp white edges and dark into [IMAGE3] a visually striking and clear picture through a logical approach.

Figure 11. Unseen Tasks : Unseen combinations of multiple tasks. For conditional generation, we integrate multiple conditions achieve more precise control. More examples are shown in Fig. 3.

In-Context Learning Findings 5

In-context learning can guide bilateral generation, even for the reverse process that is unseen during training.

Bilateral generation. In addition to generating the target from a set of given conditions, our model also shows the capability of reverse generation, i.e., inferring the underlying conditions from the target. Although our model has randomly treated one condition image as the target when

training as described in Sec. 4.2, it can generalize to a more challenging and unseen setting during inference, i.e., inferring all conditional images from only the target image. For instance, as illustrated in Fig. 9 (left), the model can reverse-engineer both the original and the style reference images given a stylized image, demonstrating the ability to disentangle the content and style representations. Similarly, as shown in Fig. 9 (right), the model can generate the corresponding real image, depth estimation, and surface normal estimation from an edge image, representing the inverse task of Fig. 3 (left). The ability to perform such

Controllability Quality Text Consistency F1 ↑ RMSE ↓ FID [23] ↓ SSIM ↑ MAN-IQA [75] ↑ MUSIQ [30] ↑ CLIP-Score [49] ↑

Condition Method Context

ControlNet [80] 0.13 - 46.06 0.34 0.31 45.45 34.10 OminiControl [61] 0.47 - 29.58 0.61 0.44 61.40 34.40 OneDiffusion [35] 0.39 - 32.76 0.55 0.46 59.99 34.99 OmniGen [71] 0.43 - 51.58 0.47 0.47 62.66 33.66 Oursdev 0 0.39 - 30.36 0.61 0.48 61.13 35.03

Canny

- Oursfill 0 0.35 - 30.60 0.55 0.49 64.39 34.98

- Oursfill 1 0.36 - 31.34 0.55 0.49 64.12 34.96

- Oursfill 2 0.36 - 31.15 0.56 0.49 64.08 34.85

ControlNet [80] - 23.70 36.83 0.41 0.44 60.17 34.49 OminiControl [61] - 21.44 36.23 0.52 0.44 60.18 34.08 OneDiffusion [35] - 10.35 39.03 0.49 0.49 60.49 34.71 OmniGen [71] - 15.07 86.08 0.26 0.49 64.90 29.72 Oursdev 0 - 25.06 42.14 0.53 0.46 58.95 34.80

Depth

- Oursfill 0 - 10.31 33.88 0.54 0.48 64.85 35.10

- Oursfill 1 - 9.91 34.44 0.54 0.49 64.32 34.95

- Oursfill 2 - 9.68 34.88 0.54 0.48 64.29 34.89

ControlNet [80] - 37.82 53.28 0.49 0.45 61.92 33.80 OminiControl [61] - 19.70 26.17 0.85 0.45 60.70 34.53

OneDiffusion [35] - - - - - - OmniGen [71] - - - - - - Oursdev 0 - 25.03 56.76 0.74 0.38 46.68 33.52

Deblur

- Oursfill 0 - 26.53 40.59 0.74 0.46 59.62 34.56

- Oursfill 1 - 25.87 36.93 0.76 0.48 61.58 34.82

- Oursfill 2 - 25.57 36.28 0.76 0.48 61.77 34.82

- Table 1. Quantitative comparison on conditioning generation and image restoration. The methods that train a specialist for each task are marked as gray color. Except for these methods, the best method is bolded, and the second best method is underlined.

Method Context DINOv2 CLIP-I CLIP-T

OminiControl [61] 73.17 87.70 33.53 OneDiffusion [35] 73.88 86.91 34.85 OmniGen [71] 67.73 83.43 34.53

Oursdev 0 78.05 87.68 35.06

- Oursfill 0 80.41 89.63 35.16
- Oursfill 1 79.33 89.22 35.02
- Oursfill 2 80.32 89.36 35.01

- Table 2. Quantitative comparison for subject-driven image generation. We report clip scores on text alignment and style consistency. Specialists are shaded in gray. Among the remaining methods, the best is emphasized in bold, while the second best is underlined.

text↑ image↑

InstantStyle [64] 0.27 0.60 OmniGen [71] 0.27 0.52 Oursdev 0.30 0.53 Oursfill 0.29 0.55

Table 3. Quantitative comparison for style transfer. We report CLIP scores on text alignment and style consistency. The specialists are indicated in gray. Among the others, the top-performing one is highlighted in bold, and the second best is underlined.

Control [61]. The details of the evaluation metrics are provided in Appendix C. Additionally, we fine-tune FLUX.1dev [33] using the same settings as FLUX.1-Fill-dev for comparison and refer to the tuned models as Oursdev and Oursfill. The details of Oursdev are shown in Appendix B.

reverse tasks highlights the flexibility and robustness in understanding complex relationships between different types of image representations.

For conditional generation and image restoration, we evaluate the models based on three criteria, i.e., controllability, visual quality, and text consistency, following the evaluation approach of OminiControl [61]. As shown in Tab. 1, our framework demonstrates comparable controllability to existing universal methods while achieving superior visual quality and text consistency. Compared to spe-

###### 5.2. Main Results

We compare our method with universal generative models, including OmniGen [71] and OneDiffusion [35], as well as specialized models, such as ControlNet [80] and Omini-

[Figure 257]

[Figure 258]

[Figure 259]

DepthtoImageDebluring

[Figure 260]

[Figure 261]

[Figure 262]

Condition Oursdev Oursfill

Figure 12. Comparison between Flux.1-dev (Oursdev) and Flux.1Fill-dev (Oursfill).

cialized methods, our model performs on par with the best results and even outperforms them on the depth-to-image.

In the style transfer task, we measure text consistency and style alignment using the CLIP [49] model. As reported in Tab. 3, our method outperforms OmniGen [71] by 2% and 3% in text alignment and style consistency, respectively. Even when compared with InstantStyle-Plus [81], a specialized model, we achieve a 2% improvement in text consistency, with only a slight decrease in style alignment.

Furthermore, we evaluate the models on subject-driven image generation and report semantic alignment using the DINOv2 [47], CLIP-I [49], and CLIP-T [49] scores. Across all these metrics, our method consistently delivers improvements, as shown in Tab. 2. For example, compared to the specialized model OminiControl [61], we achieve improvements of 7.15%, 1.66%, and 1.48% in these three scores.

Advantages of the infilling model. Our method (Oursfill) is built on FLUX.1-Fill-dev [33], which shares the same objective as our unified image generation framework. To verify its effectiveness, we also fine-tune Fill.1-dev [33] (Oursdev) using identical settings. Unlike Oursfill, which requires no modifications, Oursdev necessitates model adaptations for universal image generation, as shown in Appendix B. Despite its simplicity, Oursfill achieves superior performance across multiple tasks.

As shown in Tab. 1, Oursdev achieves a higher F1 score than Oursfill in the canny-to-image generation. However, in other tasks, Oursfill demonstrates a significant advantage. For instance, in the depth-to-image generation, Oursfill reduces RMSE from 25.06 to 10.31. In the deblurring task, Oursfill achieves superior quality by lowering RMSE while maintaining a higher SSIM. In subject-driven image generation, Tab. 2 shows that Oursfill consistently outperforms Oursdev. Additionally, in semantic-invariant style transfer, Oursfill delivers comparable performance to Oursdev, as shown in Tab. 3.

Fig. 12 presents a visual comparison, where Oursfill demonstrates clear advantages over Oursdev. Notably, in the depth-to-image generation, images produced by Oursdev frequently exhibit diagonal streak artifacts, which significantly degrade visual fidelity. Considering the advantages in performance, visual quality, and architectural efficiency, Oursfill stands out as the superior model.

Quantitative comparison on in-context learning. Here, we further analyze the impact of in-context learning on seen tasks. Tab. 1 demonstrates the impact of in-context learning on different image generation tasks. Under the canny condition, our method without in-context examples achieves an FID of 30.60, which improves to 31.15 with two incontext examples. When conditioned on depth, the RMSE decreases from 10.31 to 9.68 as the number of in-context examples increases, indicating enhanced structural consistency. Similarly, in the deblurring task, RMSE decreases from 26.53 to 25.57, reflecting improved fidelity to the original content. These results highlight in-context learning as an effective guidance mechanism, enabling the model to better align with the task intent.

###### 6. Limitations

While our model demonstrates strong stability across most in-domain tasks, it still exhibits some instability in specific tasks, such as object removal. This limitation suggests that the performance is sensitive to certain task characteristics. Additionally, the stability of the model on unseen tasks is still insufficient. Apart from the difficulty of the task and the difference with seen tasks, ambiguous in-context examples may also lead to less stable results, as discussed in Sec. 5.1.

###### 7. Conclusion

In this work, we propose VisualCloze, a universal image generation framework that addresses key challenges in existing methods, including generalizable instruction design, appropriate task distributions, and unified architectural design. Rather than relying solely on language-based instructions to convey task intent, we re-propose visual incontext learning, enabling the model to learn tasks from a few demonstrations. This approach improves generalization to unseen tasks and reduces task ambiguity. To overcome the sparsity of visual task distributions, which limits the learning of transferable knowledge, we construct Graph200K, a graph-structured dataset that establishes interrelated tasks. In this compact task space, the model is promoted to learn transferable representations and improve adaptability. Meanwhile, we identify the consistent objective between image infilling and our universal generation formulation , allowing us to seamlessly adapt generalpurpose infilling models for universal generation without

architectural modifications. Experimental results show that our approach supports a diverse set of in-domain tasks using in-context learning while demonstrating strong generalization to unseen tasks.

###### 8. Acknowledgments

This work is supported by the National Natural Science Foundation of China (Grant No. 62225604 and 62206272), and the Shenzhen Science and Technology Program (JCYJ20240813114237048).

###### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022. 3, 4
- [2] Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In ICLR,

2023. 4

- [3] Ivana Balazevic, David Steiner, Nikhil Parthasarathy, Relja Arandjelovic, and Olivier J Henaff. Towards in-context scene understanding. In NeurIPS, 2023. 3, 4
- [4] Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei A Efros. Visual prompting via image inpainting. In NeurIPS, 2022. 3, 4
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020. 2, 4, 6
- [6] John Canny. A computational approach to edge detection. IEEE TPAMI, 1986. 5
- [7] Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, and Y. A. Sheikh. Openpose: Realtime multi-person 2d pose estimation using part affinity fields. IEEE TPAMI, 2019. 5
- [8] Lan Chen, Qi Mao, Yuchao Gu, and Mike Zheng Shou. Edit transfer: Learning image editing via vision in-context relations. arXiv preprint arXiv:2503.13327, 2025. 7
- [9] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, Hui Ding, Zhe Lin, and Hengshuang. Unireal: Universal image generation and editing via learning realworld dynamics. arXiv preprint arXiv:2412.07774, 2024. 4
- [10] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In ICML,

2018. 3

- [11] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In CVPR, 2021. 2, 5
- [12] Zheng Chong, Xiao Dong, Haoxiang Li, shiyue Zhang, Wenqing Zhang, Hanqing Zhao, xujie zhang, Dongmei Jiang, and Xiaodan Liang. CatVTON: Concatenation is all you need for virtual try-on with diffusion models. In ICLR, 2025. 2
- [13] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In NeurIPS, 2021. 4
- [14] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2024. 4
- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. 2, 4
- [16] Christopher Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. In NeurIPS, 2021. 3
- [17] Peng Gao, Le Zhuo, Dongyang Liu, Ruoyi Du, Xu Luo, Longtian Qiu, Yuhang Zhang, Chen Lin, Rongjie Huang, Shijie Geng, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024. 4
- [18] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Mdtv2: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389,

2024. 4

- [19] Golnaz Ghiasi, Barret Zoph, Ekin D. Cubuk, Quoc V. Le, and Tsung-Yi Lin. Multi-task self-training for learning general representations. In ICCV, 2021. 3
- [20] Geonmo Gu, Byungsoo Ko, SeoungHyun Go, Sung-Hyun Lee, Jingeun Lee, and Minchul Shin. Towards light-weight and real-time line segment detection. In AAAI, 2022. 5
- [21] Aaron Hertzmann. Algorithms for rendering in artistic styles. PhD thesis, New York University, Graduate School of Arts and Science, 2001. 4
- [22] Aaron Hertzmann, Charles E. Jacobs, Nuria Oliver, Brian Curless, and David H. Salesin. Image analogies. In Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques, 2001. 4
- [23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 10, 16
- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 4
- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 7

- [26] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Huanzhang Dou, Yupeng Shi, Yutong Feng, Chen Liang, Yu Liu, and Jingren Zhou. Group diffusion transformers are unsupervised multitask learners. arXiv preprint arxiv:2410.15027, 2024. 4
- [27] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arxiv:2410.23775, 2024. 2, 3
- [28] Shijie Huang, Yiren Song, Yuxuan Zhang, Hailong Guo, Xueyin Wang, Mike Zheng Shou, and Jiaming Liu. Photodoodle: Learning artistic image editing from few-shot pairwise data. arXiv preprint arXiv:2502.14397, 2025. 5
- [29] Zehuan Huang, Yuanchen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. Mv-adapter: Multi-view consistent image generation made easy. arXiv preprint arXiv:2412.03632, 2024. 5
- [30] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 10, 16
- [31] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In CVPR, 2018. 3
- [32] Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In CVPR, 2017. 3
- [33] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 3, 4, 5, 7, 10, 11, 16
- [34] Bolin Lai, Felix Juefei-Xu, Miao Liu, Xiaoliang Dai, Nikhil Mehta, Chenguang Zhu, Zeyi Huang, James M Rehg, Sangmin Lee, Ning Zhang, et al. Unleashing in-context learning of autoregressive models for few-shot image manipulation. arXiv preprint arXiv:2412.01027, 2024. 4
- [35] Duong H. Le, Tuan Pham, Sangho Lee, Christopher Clark, Aniruddha Kembhavi, Stephan Mandt, Ranjay Krishna, and Jiasen Lu. One diffusion to generate them all, 2024. 2, 3, 4, 10
- [36] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. In NeurIPS, 2023. 16
- [37] Kunchang Li, Yali Wang, Junhao Zhang, Peng Gao, Guanglu Song, Yu Liu, Hongsheng Li, and Yu Qiao. Uniformer: Unifying convolution and self-attention for visual recognition. IEEE TPAMI, 2023. 5
- [38] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In CVPR,

2024. 2, 4, 6

- [39] Weifeng Lin, Xinyu Wei, Renrui Zhang, Le Zhuo, Shitian Zhao, Siyuan Huang, Junlin Xie, Yu Qiao, Peng Gao, and Hongsheng Li. Pixwizard: Versatile image-to-image visual assistant with open-language instructions. arXiv preprint arXiv:2409.15278, 2024. 2, 5, 6
- [40] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023. 4

- [41] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 4
- [42] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4
- [43] Yihao Liu, Xiangyu Chen, Xianzheng Ma, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong. Unifying image processing as visual prompting question answering. arXiv preprint arXiv:2310.10513, 2023. 3, 4, 6
- [44] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instructionbased image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487, 2025. 2, 4
- [45] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 7
- [46] Noor Nashid, Mifta Sintaha, and Ali Mesbah. Retrievalbased prompt selection for code-related few-shot learning. In 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pages 2450–2462. IEEE, 2023. 8
- [47] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 11, 16
- [48] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 4, 16
- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 10, 11, 16
- [50] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 5
- [51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 4
- [52] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633, 2021. 8
- [53] Sebastian Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017. 3
- [54] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2, 16

- [55] Dianmo Sheng, Dongdong Chen, Zhentao Tan, Qiankun Liu, Qi Chu, Jianmin Bao, Tao Gong, Bin Liu, Shengwei Xu, and Nenghai Yu. Towards more unified in-context visual understanding. In CVPR, 2024. 4
- [56] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983,

2023. 16

- [57] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021. 7
- [58] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 4
- [59] Yanpeng Sun, Qiang Chen, Jian Wang, Jingdong Wang, and Zechao Li. Exploring effective factors for improving visual in-context learning. arXiv preprint arXiv:2304.04748, 2023. 4
- [60] Yasheng SUN, Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, and Hideki Koike. Imagebrush: Learning visual in-context instructions for exemplarbased image manipulation. In NeurIPS, 2023. 3, 4, 6
- [61] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 3, 2024. 4, 5, 6, 10, 11
- [62] Paints-Undo Team. Paints-undo github page, 2024. 5
- [63] Alex Jinpeng Wang, Linjie Li, Yiqi Lin, Min Li, Lijuan Wang, and Mike Zheng Shou. Leveraging visual tokens for extended text contexts in multi-modal learning. NeurIPS,

2024. 4

- [64] Haofan Wang, Peng Xing, Renyuan Huang, Hao Ai, Qixun Wang, and Xu Bai. Instantstyle-plus: Style transfer with content-preserving in text-to-image generation. arXiv preprint arXiv:2407.00788, 2024. 2, 4, 5, 10
- [65] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 5
- [66] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In CVPR, 2023. 3, 4
- [67] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. Seggpt: Towards segmenting everything in context. In ICCV, 2023. 3, 4
- [68] Zhendong Wang, Yifan Jiang, Yadong Lu, yelong shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, and Mingyuan Zhou. In-context learning unlocked for diffusion models. In NeurIPS, 2023. 4
- [69] Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. arXiv preprint arXiv:2411.07199, 2024. 2, 3, 5, 6

- [70] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025. 8
- [71] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 2, 3, 4, 10, 11
- [72] Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In CVPR, 2015. 5
- [73] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE TPAMI, 2023. 5
- [74] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024. 5
- [75] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1191–1200, 2022. 10, 16
- [76] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [77] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild. arXiv preprint arXiv:2401.13627, 2024. 3
- [78] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022. 4
- [79] Hayoung Yun and Hanjoo Cho. Achievement-based training progress balancing for multi-task learning. In ICCV, 2023. 3
- [80] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 3, 4, 5, 10
- [81] Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. Inversion-based style transfer with diffusion models. In CVPR, 2023. 2, 3, 11
- [82] Yuanhan Zhang, Kaiyang Zhou, and Ziwei Liu. What makes good examples for visual in-context learning? In NeurIPS,

2023. 3, 4

- [83] Canyu Zhao, Mingyu Liu, Huanyi Zheng, Muzhi Zhu, Zhiyue Zhao, Hao Chen, Tong He, and Chunhua Shen. Diception: A generalist diffusion model for visual perceptual tasks. arXiv preprint arXiv:2502.17157, 2025. 4
- [84] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. CAAI Artificial Intelligence Research, 2024. 5

- [85] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017. 3
- [86] Yucheng Zhou, Xiang Li, Qianning Wang, and Jianbing Shen. Visual in-context learning for large vision-language models. arXiv preprint arXiv:2402.11574, 2024. 4
- [87] Muzhi Zhu, Yang Liu, Zekai Luo, Chenchen Jing, Hao Chen, Guangkai Xu, Xinlong Wang, and Chunhua Shen. Unleashing the potential of the diffusion model in few-shot semantic segmentation. In NeurIPS, 2024. 3, 4
- [88] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Xiangyang Zhu, Fu-Yun Wang, Zhanyu Ma, Xu Luo, Zehan Wang, Kaipeng Zhang, Lirui Zhao, Si Liu, Xiangyu Yue, Wanli Ouyang, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-next : Making lumina-t2x stronger and faster with next-dit. In NeurIPS,

2024. 2, 3, 4

###### Appendix A. Instruction Format

In our unified framework, the instruction consists of three parts: (1) layout instruction, which describes the layout of the grid image; (2) task instruction, which specifies the task type; and (3) content instruction, which describes the content of the target image. Fig. 13 illustrates the instructions for concept fusion of style, subject, and layout (Fig. 13 upper) and image editing with reference (Fig. 13 bottom). The content instruction is omitted for some tasks that provide strong visual cues in conditions, like style transfer.

###### Appendix B. Fine-tuning FLUX.1-dev Model

Apart from FLUX.1-Fill-dev, we also adapt our method to FLUX.1-dev [33], a common text-to-image generative model. Unlike the infilling model that shares a consistent objective with universal image generation, FLUX.1-dev requires customized modifications to process clean condition images and noise target images. Specifically, after concatenating images in a grid layout like the infilling model, we always keep the region corresponding to the conditions as clean latent embeddings throughout the sampling process. This strategy requires modifications in image sampling because FLUX.1-Fill-dev takes noise latent embeddings as input. Moreover, for the adaLN-Zero block [48], it is critical to calculate the separate mean and shift parameters for the regions of clean conditions and noise target by feeding T = 0 and T = t into the adaLN-Zero, respectively. t indicates the timestep in each sampling step and gradually increases from 0 to 1 along the sampling process. This strategy aligns with the pre-training domain of FLUX.1-dev, where different noise levels correspond to different mean and shift. As shown in Fig. 14, this strategy ensures the visual fidelity.

###### Appendix C. Evaluation Metrics C.1. Conditioning Generation

We assess the models from controllability, quality, and text consistency to evaluate image generation quality in conditioning generation and image restoration tasks.

Controllability. For conditional image generation, we measure the difference between the input conditions and those extracted from generated images. Specifically, we calculate the F1 Score for the cany-to-image task and RMSE for the depth-to-image task. Additionally, for debluring, we measure the RMSE between original and restored images.

Generation quality. We measure the Generation quality using FID [23], SSIM, MAN-IQA [75], and MANIQA [75]. FID [23] measures the similarity between generated and real image feature distributions. SSIM evalu-

ates perceptual quality by comparing luminance, contrast, and structural patterns between images. It calculates local patch statistics and combines them into a composite score ranging from −1 to 1, with higher values indicating better structural preservation. MANIQA [75] and MUSIQ [30] leverage neural networks to predict image quality scores.

Text consistency. Leveraging the powerful multi-modal capability of CLIP [49], we also measure the semantic alignment between generated images and text prompts, which reflects how the model follows instructions.

###### C.2. Subject Driven Generation

Following DreamBooth [54] and BLIP-Diffusion [36], we measure DINOv2 [47], CLIP-I [49], and CLIP-T scores for the comparison of subject-driven image generation. DINOv2 [47] and CLIP-I scores measure the alignment between the reference subject and generated images through cosine similarity and CLIP score, respectively. CLIP-T measures the alignment between the generated image and the corresponding text prompt.

###### C.3. Style Transfer

Following StyleDrop [56], we assess the performance of style transfer according to text consistency and style alignment. For text alignment, we measure the cosine similarity between embeddings of generated images and text prompts, where the embeddings are extracted by CLIP [49]. Regarding style consistency, we measure the cosine similarity between embeddings of generated images and style reference. Note that these two metrics should be considered together because the style consistency will reach 1.0 if the model collapses, where the model completely copies style reference as a composite image and ignores text instructions.

|[Figure 263]<br><br>In-context examples| |
|---|---|
|Conditions|Target|

Layout instruction:

12 images are organized into a grid of 3 rows and 4 columns, evenly spaced.

Task instruction:

Each row describes a process that begins with [IMAGE1] white edge lines on black from canny detection, [IMAGE2] Photo with a strong artistic theme, [IMAGE3] a reference image showcasing the dominant object and results in [IMAGE4] High-quality visual with distinct artistic touch.

Content instruction:

#### ∅

|[Figure 264]<br><br>In-context examples| |
|---|---|
|Conditions|Target|

Layout instruction:

A 3x3 grid containing 9 images, aligned in a clean and structured layout

Task instruction:

Every row provides a step-by-step guide to evolve [IMAGE1] a reference image with the main subject included, [IMAGE2] an image with flawless clarity into [IMAGE3] a high-quality image.

Content instruction:

The bottom-right corner image presents: A glossy gel nail polish bottle. At the edge of a bustling city park, this item rests on vibrant green grass, captured with a subtle bokeh effect as joggers and pets move in the background.

(a) Concatenated images (b) Language instructions

Figure 13. Examples of language instructions that contain prompts about the layout of the concatenated image, task intent, and content of the target image.

[Figure 265]

Condition Target Condition Target

(a) separate mean and shift (b) unified mean and shift

Figure 14. Effects of separate mean and shift in fine-tuning FLUX.1-dev.

