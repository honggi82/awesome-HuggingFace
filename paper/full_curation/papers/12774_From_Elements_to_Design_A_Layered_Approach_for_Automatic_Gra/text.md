### From Elements to Design: A Layered Approach for Automatic Graphic Design Composition

Jiawei Lin1, Shizhao Sun2, Danqing Huang2, Ting Liu1, Ji Li2, Jiang Bian2 1Xi’an Jiaotong University, 2Microsoft Research

kylelin@stu.xjtu.edu.cn, tingliu@mail.xjtu.edu.cn, {shizsu, dahua, jili5, jiabia}@microsoft.com

# arXiv:2412.19712v1[cs.CV]27Dec2024

[Figure 1]

[Figure 2]

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

1799 Via Garda Dr 844

PIZZA\nPARTY\nDAY MAY 17

Background + Underlay + Logo/Image + Text + Embellishment

(a) Design Composition (b) Design Layers

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

(c) Generated Graphic Designs

Figure 1. (a) Given a set of multimodal elements as input, our approach automatically composes them into a cohesive, balanced, and aesthetically pleasing graphic design. (b) Since a holistic design can be divided into different layers according to element semantics, we achieve the design composition task in a layer-by-layer manner. (c) Our approach is able to craft high-quality design pieces.

###### Abstract

tion process smoother and clearer. The experimental results demonstrate the effectiveness of LaDeCo in design composition. Furthermore, we show that LaDeCo enables some interesting applications in graphic design, such as resolution adjustment, element filling, design variation, etc. In addition, it even outperforms the specialized models in some design subtasks without any task-specific training.

In this work, we investigate automatic design composition from multimodal graphic elements. Although recent studies have developed various generative models for graphic design, they usually face the following limitations: they only focus on certain subtasks and are far from achieving the design composition task; they do not consider the hierarchical information of graphic designs during the generation process. To tackle these issues, we introduce the layered design principle into Large Multimodal Models (LMMs) and propose a novel approach, called LaDeCo, to accomplish this challenging task. Specifically, LaDeCo first performs layer planning for a given element set, dividing the input elements into different semantic layers according to their contents. Based on the planning results, it subsequently predicts element attributes that control the design composition in a layer-wise manner, and includes the rendered image of previously generated layers into the context. With this insightful design, LaDeCo decomposes the difficult task into smaller manageable steps, making the genera-

###### 1. Introduction

Graphic design is an artistic discipline dedicated to creating visual content that attracts attention and communicates messages effectively. Creating visually appealing designs today relies on human designers with both artistic creativity and technical expertise to skillfully integrate multimodal graphic elements like images, headlines, and decorative embellishments, etc. This is a complex and time-consuming process that requires careful consideration of many aspects. For example, as shown in Figure 1a, it is important to ensure that the main object (i.e., pizza) is not obscured by other elements. For readability, there should be sufficient contrast

between the text and the underlay. Additionally, designers also need to adjust the element sizes to make the design balanced. In this work, we refer to the challenging process of composing a set of elements into a holistic design as design composition (see Figure 1a).

To ease the burden on human designers, recently, there has been a growing interest in developing generative models to streamline this process. Most existing work focuses on certain typical subtasks of design composition. For example, some previous approaches investigate content-aware layout generation [7, 8, 27, 32, 37], which aims to automatically arrange graphic elements on a given canvas while ensuring that the main object remains unobstructed. Although these methods are capable of creating high-quality layouts, they typically only consider the background image content while overlooking the content of other elements. In addition, they do not predict text-related attributes during the layout generation process, limiting their ability to produce fully integrated designs. Another popular subtask is called typography generation [5, 11, 13, 26, 28]. Its goal is to generate font, color, size, and other attributes for text elements, enhancing both aesthetics and readability. However, it ignores the visual elements in graphic designs. Together, all these studies fall short of holistic design creation. Consequently, users have to manually integrate models of different functions to achieve design composition, which brings high costs and unnecessary obstacles.

To be best of our knowledge, FlexDM [11] is the only attempt towards automatic design composition. By representing graphic designs as a flat combination of element attributes, FlexDM formulates various design tasks as masked field prediction problems, including the design composition task. While the flat representation provides practicality and versatility, it has a notable limitation: it overlooks the inherent hierarchical structure within graphic designs. This hierarchical structure arises because human designers usually follow the layered design principle. Specifically, it involves arranging elements in separate semantic layers, starting with a background and then gradually adding underlays, images, texts and small embellishments (see Figure 1b). Such layered principle brings two main benefits. First, each preceding layer provides a strong foundation for designing subsequent layers, aiding cohesion. Second, grouping similar elements by layer clarifies the design process and enhances workflow efficiency.

Based on the insight, we propose LaDeCo (see Figure 2), a layered design composition method built upon Large Multimodal Models (LMMs) [2, 3, 29, 35, 38]. Considering that input elements are inherently multimodal and that design composition can be formulated by sequentially predicting attributes for each element, we choose LMMs as the backbone for this task. To support the layered mechanism, we develop a layer planning module. Specifically, we carefully

design the task prompts and leverage GPT-4o [1] to predict the semantic label for each input element by considering its content. Elements sharing the same label are placed on the same layer, thus reaching layer planning. Subsequently, LaDeCo divides the generation process into several steps according to the layer planning results. At each step, LMMs are asked to predict element attributes within a single layer. After each step, the previously generated layers will be rendered as an intermediate design image and fed back into the LMMs, providing contextual information for following layer predictions.

Thanks to the novel design, LaDeCo decomposes the challenging task into smaller manageable steps. At each step, the model only focuses on design composition of the current layer, making the process more accessible. Additionally, by rendering intermediate designs and adding them into the context, the model can better generate subsequent layers based on previous layers. The layerwise generation also offers flexibility to allow LaDeCo supporting certain subtasks of design composition without any task-specific training, as shown in Section 4.5.

To validate the effectiveness of our approach, we conduct experiments on a publicly available dataset Crello [31] to compare it with the state-of-the-art baselines. Specifically, the comparisons cover the full design composition task and its subtasks: content-aware layout generation and typography generation. Both quantitative and qualitative experimental results show that LaDeCo significantly outperforms the baseline models in design composition. Through ablation studies, we demonstrate the effectiveness of the layer planning module and layered design composition introduced in LaDeCo. Besides, we show that LaDeCo enables some interesting applications such as resolution adjustment, element filling, and design variation, further showcasing the practicality and versatility of LaDeCo. In addition, LaDeCo even surpasses the specialized models in the two subtasks.

###### 2. Related Work

Graphic Design Composition. There are three aspects of relevant research on graphic design composition: contentaware layout generation, typography generation, and the holistic design composition. We’ll introduce them below.

Content-Aware Layout Generation [4, 7, 8, 22, 27, 32]. It is an emerging research topic that studies layout generation conditioned on a given canvas. LayoutPrompter [22] proposes a RAG-based approach to retrieve training samples with the most similar saliency bounding boxes as prompts, and achieves content-aware layout generation via in-context learning. PosterLlama [27] introduces a LLM-based twostage training method for this task. It first keeps the backbone parameters fixed and trains the adapter, and then finetunes the backbone to generate layouts in the HTML format. PosterLLaVA [32] further improves practicality by

Image Embedding

Please predict the label of the given element

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Text Embedding

[Figure 25]

from “background”, “undelay”,

[Figure 26]

[{"left": 297, "top": 534,

[{

“logo/image”, “text”, and “embellishment”.

Rendered Layers Visual Element Text Element Canvas Size

| |
|---|

|[Figure 27]|
|---|

Render Render

|[Figure 28]|
|---|

"left": 276, "top": 200, "width": 388, "height": 386

... "font": "Playfair Display", "color": [195, 154, 98], "text_align": "center",

| |
|---|

(Only applicable to training samples)

[Figure 29]

The holistic design is:

| |
|---|

}]

...}, ...]

| |
|---|

LLM

Pretrained LMM

“embellishment”

…

…

Projector Vision Encoder

Projector Vision Encoder + Pooling

Projector

Projector Vision Encoder + Pooling

Predicted layers:

|canvas height: 960px<br><br>canvas width: 1080 px|
|---|

|“PIZZA\nPART Y\nDAY”<br><br>“MAY 17”<br><br>“1799 Via Garda Dr 844”|
|---|

[Figure 30]

Vision Encoder

[Figure 31]

[Figure 32]

[Figure 33]

+ Pooling

[Figure 34]

+ Pooling

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

background underlay logo/image text embellishment

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

(a) Layer Planning (b) Layered Design Composition

Figure 2. Illustration of our proposed LaDeCo. First, it utilizes GPT-4o [1] to annotate the semantic labels for input elements. The layer structure is obtained from the predictions. Then LaDeCo fine-tunes LMMs to achieve layered design composition. After generating each layer, the intermediate designs will be rendered as images and fed back into LMMs to guide subsequent layer generation.

supporting more user requirements (e.g., element relationships). Compared to our work, these existing methods do not consider element content, nor predict text attributes, and thus cannot create a complete design.

Typography Generation [5, 11–13, 26]. This task studies visually compelling, harmonious text rendering on a graphic design. Some relevant work [5, 26] regards text styles as images and formulates typography generation as text image generation in the pixel space. For editability, some other work directly predicts editable text attributes (e.g., font type, color, size) and utilizes the renderer to visualize stylized text elements. FlexDM [11] achieves this by masked field prediction, generating all attributes in a single pass. COLEs [12, 13] propose Typography-LMM to autoregressively predict text attributes based on the input canvas. However, these methods still can not craft holistic graphic designs since they do not consider the visual elements.

Design Composition. FlexDM [11] can accomplish design composition by simultaneously masking element positions and text attributes. However, it treats all input elements equally and does not incorporate the critical layered information during the generation process. In summary, we are the first to introduce layered design composition and develop a novel approach based on it.

Other Topics in Graphic Design. Earlier, some studies investigate content-agnostic layout generation. To meet diverse user needs, previous methods have considered various input conditions, such as element attributes [10, 14, 16, 17, 20, 22, 34], element relationships [16, 18], textual descriptions [21, 22], and more. However, the contentagnostic nature makes them unable to adapt to the input elements, in which element deformations often occur. Recently, there is an interesting topic studying design content

generation [12, 13, 15]. We emphasize that we focus on design composition from user-provided elements rather than generate the element content as presented in their work.

Large Multimodal Models (LMMs). LMMs have shown strong capabilities in understanding multimodal contexts and generating plausible responses across diverse domains [1, 2, 24, 33, 35, 38]. In this work, we leverage them to effectively achieve design composition from multimodal input elements. We introduce the layered design principle in our approach, enabling LMMs to tackle different semantic layers iteratively instead of generating all layers at once. To the best of our knowledge, we are the first work to adopt LMMs for layered design generation. Furthermore, our layerwise generation method also reflects the idea of chain-of thought (CoT) reasoning [30], which has been proven to be effective in enhancing reasoning performance [25, 36].

###### 3. LaDeCo

###### 3.1. Problem Formulation

The input and output of the design composition task are a set of multimodal elements and the element attributes, respectively. When obtaining the predicted attributes, we can adopt an off-the-shelf renderer to render the input elements into a holistic graphic design, thereby achieving design composition. In this work, input elements are categorized into two modalities: image modality and text modality. For image modality elements, the output attributes are four bounding box parameters, i.e., left and top coordinates, element width and height. For text modality elements, we consider eight more attributes, namely angle, font, font size, color, text alignment, capitalization, letter spacing, and line height. We empirically find that these attributes are suffi-

cient to describe a high-quality design.

###### 3.2. Method Overview

In this work, we take inspiration from the layered design principle to decompose a holistic graphic design into different layers and progressively create these layers to reach the complete design, making the design composition process smoother and clearer. Here, a layer is a collection of graphic elements with same semantic labels. To be more specific, our method includes two key techniques, namely a layer planning module and a layered design composition process. The layer planning module is responsible for categorizing input elements into pre-defined layers. Then, in the layered design composition process, our approach predicts element attributes in a layerwise manner and gathers them together to get the complete attributes.

###### 3.3. Layer Planning

The very first step here is to determine a reasonable layer structure. By examining numerous completed design pieces and consulting experienced designers, in this work, we con-

- sider 5 design layers, namely background, underlay, logo/image, text, and embellishment layers in the placement order (see Figure 1b). By sequentially rendering these layers on the empty canvas G0, we obtain G1 through G5 (see Figure 4), where G1 represents only the background layer, G2 includes the background layer plus the underlay layer, and so forth, with G5 representing the complete, finalized design. Notably, the layer structure is not limited to our proposed one. There is flexibility to add or remove some as long as it is reasonable.

Although the publicly available dataset does not contain any layer information for the elements, we find it is feasible to infer from element content. For example, a solid colored rectangular box might be an underlay, and a element with a star is an embellishment. Therefore, we formulate layer planning as an element content understanding problem and leverage pre-trained LMMs to resolve it. In the implementation, we employ GPT-4o [1] to automatically generate input element labels (see Figure 2a), thereby achieving layer planning. To guide GPT-4o effectively, we carefully craft prompts, clearly defining the problem, describing the characteristics of each element label, and requesting the model to output the most appropriate one for input elements. For training samples where ground truth designs are available, we also include the design images and some metadata (e.g., the canvas size, element size) into the model input to enhance the prediction accuracy. For full details of the prompt text, please refer to the supplementary materials.

###### 3.4. Layered Design Composition

Our main idea here is to generate element attributes in a layerwise manner from background to embellishment lay-

ers, according to the layer planning results in Section 3.3. The rendered images of previous layers are incorporated in the context for the prediction of the subsequent layers. It is noteworthy that this process requires strong understanding capabilities of element content. For example, as shown in the row 3, column 1 of Figure 3, the text element with an email address should be placed at the bottom of the canvas, and the barbershop logo should be placed at the top. Meanwhile, the understanding of intermediate results is also critical. For example, in the same figure, the model should avoid blocking the persons by understanding the intermediate rendered image. To this end, we resort to LMMs, which possess remarkable understanding capabilities, to model the layered design process for multimodal design elements. As shown in Figure 2b, in the i-th design layer, LMMs predict the attribute set Yi for current layer’s elements Xi. The preceding layers are rendered as an image Gi−1 and reintroduced into the LMMs to be part of the contextual input, guiding the generation of Yi. Next, we will detail the representation of Xi and Yi, as well as the model architecture.

Representation of Xi and Yi. In general, we represent Xi and Yi by concatenating the element content and attributes of current layer, respectively. For background, underlay, logo/image, and embellishment layers, the inputs are all visual elements. Hence, Xi is a combination of element images (i.e., pixel values). For the text layer, we concatenate its text content together to construct Xi (see Figure 2b). In terms of Yi, we follow the trend of structured representation in existing approaches [12, 22, 27], and serialize the element attributes into JSON strings, as implemented in OpenCOLE [12]. Notably, within each layer, the elements are randomly shuffled to prevent information leakage (e.g., the top-down placement order). In the design layers that do not have any elements, we represent their inputs as null and outputs as an empty JSON string {}. Please refer to the supplementary materials for an example of the input-output representation.

Model Architecture. The model consists of three components: a vision encoder, a projector and the LMM backbone. The vision encoder is responsible for encoding the element images and intermediate designs, generating image embeddings. The projector then projects these embeddings to match the hidden state dimension required by the backbone. Finally, the backbone is used to model the joint distribution across layers, ensuring cohesion in the layered design process. To reduce computational complexity, a 2D average pooling operation is applied to the output of vision encoder to compress the image tokens effectively.

###### 3.5. Training and Inference

Training. Ground truth attributes for training samples are available, allowing us to pre-render and cache the intermediate canvas states {Gi}5i=1 based on the layer planning re-

Methods

Val Ove Ali Undl Unds (i) (ii) (iii) (iv) (v)

FlexDM [11] 5.34 5.29 5.41 5.09 4.54 0.8757 0.3242 0.0016 0.7286 0.7298 GPT-4o [1] 6.53 6.49 6.60 6.27 5.69 0.9968 0.0595 0.0001 0.3780 0.5708 LaDeCo (Ours) 8.08 7.92 8.00 7.82 6.98 0.9365 0.0865 0.0013 0.6922 0.6580

GT 8.35 8.21 8.30 8.01 7.26 0.9265 0.0768 0.0015 0.6848 0.6732

- Table 1. Quantitative comparison on the design composition task. LLaVA-OV evaluation includes the following aspects: (i) design and layout, (ii) content relevance, (iii) typography and color, (vi) graphics and images, and (v) innovation and originality. The score closest to the one calculated from real data (denoted as GT) is highlighted in bold, indicating the best performance among different methods.

sults (see Figure 4). During training, we fine-tune the model by minimizing the negative log-likelihood of Yi across all layers:

L = −

5

log P(Yi|Y<i,X≤i,G<i).

i=1

Inference. At inference time, LaDeCo iteratively generates design layers from G1 to G5, thereby achieving the goal of design composition. Compared to generating the whole attributes at once, LaDeCo adds only about a 20% increase in rendering time to obtain the intermediate designs, making it an effective and efficient method.

Notably, LaDeCo offers remarkable flexibility at inference. It can handle other design subtasks without any taskspecific training. For example, when the ground truth background layer (G1) is provided, and the model is tasked with generating G2 through G5, it can effectively achieve content-aware layout generation. Similarly, when G1 to G3 are given as ground truth, and the model is asked to generate G4, it performs typography generation.

###### 4. Experiments

###### 4.1. Setup

Datasets. We conduct experiments on the publicly available Crello-v4 [31] dataset, which includes 23,421 graphic designs sourced from VistaCreate 1. Since Crello has provided separate rendered pixel images for all elements, we can conveniently build the training inputs as described in Section 3.4. Besides, based on the pixel images as well as the layer planning results obtained in Section 3.3, we can easily render the intermediate designs for training samples through the renderer 2 developed in OpenCOLE [12]. We adopt the same data splits of Crello-v4, dividing the dataset into 19,095 training, 1,951 validation, and 2,375 test samples. To enhance training efficiency, we filter out training samples with more than 25 elements (a total of 938 examples). In addition, we gather a large-scale commercial

- 1https://create.vista.com/
- 2https://github.com/CyberAgentAILab/OpenCOLE/src/opencole/renderer

dataset similar to Crello to study the effect of dataset size on performance. We refer to it as LargeCrello. LargeCrello dataset contains a total of 109,235 samples. We also filter out its samples exceeding 25 elements and manually verify that LargeCrello has no overlap with the test set of Crello.

Implementation Details. We choose Llama-3.1-8B 3, one of the most advanced open-source large language models (LLMs), as the model backbone. The vision encoder is initialized from the CLIP ViT-L/14 model 4, and the projector is structured as a two-layer MLP using GELU [6] activation functions. To enable efficient training, we utilize the LoRA technique [9] on the backbone, jointly optimizing LoRA parameters and the projector while keeping the vision encoder parameters fixed. We conduct training on four A100-80G GPUs with a global batch size of 128, and employ AdamW [23] to optimize for about 7K iterations with a learning rate of 2e-4. As for the hyper-parameters, the rank number of LoRA is set to 32, the rank alpha is 64, and the token number of an input image is set to 5 (1 cls token, 2 × 2 compressed tokens). At inference, we set the sampling temperature to 0.7 and Top-p (nucleus sampling) to 0.95, balancing diversity and quality for generated designs. Baselines. To demonstrate the effectiveness of LaDeCo, we compare it with existing methods, i.e., FlexDM [11] and GPT-4o [1]. FlexDM originally conducts experiments on Crello-v2, which has different dataset splits of Crello-v4. We re-train it on the upgraded v4 dataset. To accommodate design composition, we mask the position and text attribute fields, and predict them simultaneously. In the GPT4o baseline, we sequentially concatenate element contents in a manner similar to our approach, and prompt GPT-4o to generate their attributes one by one. For the categorical attributes (e.g., font), we provide all options in the context.

Evaluation Metrics. We follow previous work to prepare the evaluation metrics. (1) Overall metrics. Following COLEs [12, 13], we introduce a robust proxy model for comprehensive evaluation. Specifically, we use the LLaVAOV-7B model [19] to evaluate quality across five aspects: design and layout, content relevance, typography and color,

- 3https://huggingface.co/meta-llama/Llama-3.1-8B
- 4https://huggingface.co/openai/clip-vit-large-patch14-336

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

##### FlexDMGPT-4oOursGT

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

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

Figure 3. Qualitative comparison. We also show the ground truth designs for these samples. Please zoom in for a better view.

[Figure 72]

elements along the left side of the canvas to achieve a low alignment (Ali) score, but such an arrangement would not produce meaningful and high-quality designs.

###### 4.2. Quantitative Evaluation

Table 1 shows quantitative results. LaDeCo significantly outperforms the baseline models in overall metrics (i.e., LLaVA-OV scores), indicating the superiority of LaDeCo in design composition. Particularly, LaDeCo achieves very good scores in design and layout as well as typography and color. This suggests that LaDeCo excels at layout generation and nuanced attribute prediction, both of which are critical for design creation. In terms of geometry-related metrics, LaDeCo achieves closet scores to the one calculated on real data (denoted as GT) on most metrics, showcasing its strong capabilities to model real-world data. In contrast, baseline models often struggle with some metrics. For example, FlexDM exhibits a serious overlap issue, while GPT4o has low underlay effectiveness.

- Figure 4. The rendered results of different layers from LaDeCo.

graphics and images, innovation and originality. We use the same prompts as presented in COLE [13]. (2) Geometryrelated metrics. These metrics focus purely on the geometric attributes of elements without considering their content, including element validity (Val), overlap (Ove), alignment (Ali), and underlay effectiveness (Undl, Unds) [8, 22, 27]. For each metric, the closer it is to the one calculated from real data (denoted as GT in the table), the better. Note that higher or lower values alone are not indicative of better performance. For example, a model could always put all the

###### 4.3. Qualitative Evaluation

We show rendered generated designs for each method and the corresponding ground truth designs in Figure 3. The results demonstrate that LaDeCo is proficient in composing input elements into high-quality, visually pleasing and

Settings

Val Ove Ali Undl Unds (i) (ii) (iii) (iv) (v)

Llama-3.1-8B (rank 16) 8.03 7.89 8.00 7.75 6.90 0.9347 0.0796 0.0012 0.6900 0.6564 Llama-3.1-8B (rank 64) 8.10 7.94 8.04 7.83 6.98 0.9352 0.0787 0.0013 0.7084 0.6715

llava-v1.5-7b (rank 32) 8.00 7.86 8.02 7.78 6.90 0.9403 0.0940 0.0015 0.6703 0.6208 Llama-3.1-8B-Instruct (rank 32) 8.08 7.89 8.03 7.82 6.99 0.9388 0.0804 0.0015 0.6867 0.6640

w/o LP, w/o LDC (rank 32) 7.23 7.12 7.28 6.99 6.29 0.9325 0.0954 0.0013 0.6194 0.5875 w/ LP, w/o LDC (rank 32) 7.84 7.67 7.78 7.56 6.66 0.9389 0.0843 0.0013 0.6568 0.6242

Llama-3.1-8B* (rank 32) 8.22 8.06 8.22 7.94 7.09 0.9335 0.1029 0.0005 0.7321 0.7116 Llama-3.1-8B (rank 32) 8.08 7.92 8.00 7.82 6.98 0.9365 0.0865 0.0013 0.6922 0.6580 GT 8.35 8.21 8.30 8.01 7.26 0.9265 0.0768 0.0015 0.6848 0.6732

- Table 2. Ablation studies. Our investigation covers four aspects (from top to bottom): (1) the rank number in LoRA, (2) the base model,

(3) the key techniques in LaDeCo, where LP denotes layer planning , and LDC represents layered design composition, (4) dataset size. The model with * to is trained on the combined Crello and LargeCrello datasets, while the models without * are trained on Crello only.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Figure 7. LaDeCo creates diverse designs with the same elements.

- Figure 5. LaDeCo composes the same input elements to designs with different canvas sizes.

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

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Before

New Elements

After Before

New Elements

After

- Figure 6. LaDeCo adds new elements on a existing design to achieve a more appealing design.

LaDeCo can generate meaningful layers.

Besides, LaDeCo also enables some interesting applications in graphic design. Figure 5 shows that LaDeCo can achieve design composition on the condition of different canvas sizes (called resolution adjustment). The predicted attributes will be adjusted to suit the canvas size, making the final designs appealing in various canvas sizes. Figure 6 presents that LaDeCo can add new element on a existing design to make it more pleasing (called element filling). Figure 7 shows that the given the same input elements, LaDeCo can compose them to create diverse designs, which provides multiple choices to users (called design variations).

balanced designs. On the contrast, the baselines all suffer from some serious problems, such as composition failure (FlexDM, column 2, 6), poor readability (FlexDM, column 1) and imbalance (GPT-4o, column 1, 3, 4). Additionally, LaDeCo can accurately capture relationships between elements. For instance, text elements are precisely positioned on underlay elements (Ours, column 1, 4, 5, 7), and embellishment elements contribute meaningfully to decorate the designs (Ours, column 2, 6).

We further shows the rendered results of different layers generated by LaDeCo in Figure 4. These results demonstrate that with the proposed layer planning module,

###### 4.4. Ablation Studies

Table 2 shows the results of ablation studies. (1) Rank number in LoRA: When the rank number varies from 16 to 32, and then to 64, the quantitative metrics show little variation, indicating the robustness of LaDeCo with respect to the amount of training parameters. (2) Base model: We adopt two other pre-trained LLMs, llava-v1.5-7b and Llama-3.18B-Instruct, as the backbone. Similar to previous ablation study, our LaDeCo is robust to the choice of base model. (3) Key techniques in LaDeCo: We consider two settings. First, we remove the layered design composition (LDC) and use the model to predict all element attributes sequentially

Methods Val Ove Ali Undl Unds Uti Occ Rea

PosterLLaVa [32] 0.9269 0.0685 0.0011 0.7879 0.7375 0.4199 0.1936 0.0747 PosterLlama [27] 0.8701 0.0868 0.0014 0.8483 0.7798 0.4115 0.1772 0.0694 LaDeCo (Ours) 0.9340 0.0805 0.0016 0.6851 0.6540 0.4414 0.1835 0.0768

GT 0.9265 0.0768 0.0015 0.6848 0.6732 0.4737 0.1628 0.0709

- Table 3. Quantitative results on the content-aware layout generation subtask. The score closest to the one calculated from real data (denoted as GT) is highlighted in bold, indicating the best performance among different methods.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

PosterLlama PosterLLaVA Ours GT

Figure 8. Qualitative comparison on the content-aware layout generation task. The yellow, red, green, pink boxes represent underlay, image, text, and embellishment elements, respectively.

without taking the intermediate rendered layers as input (denoted as w/ LP, w/o LDC (rank 32) in the table). This leads to drops in five overall metrics and two underlay effectiveness metrics, indicating the importance of LDC in design composition. Second, we continue to remove layer planning (LP) module and construct the input-output representation using a random ordered design elements (denoted as w/o LP, w/o LDC (rank 32) in the table). The aforementioned metrics show a significant decline, which indicates that arranging elements following the hierarchical structure does play a critical role in design composition. (4) Dataset size: When the model is trained on the combined datasets, most quantitative metrics are considerably improved and get closer to ground truth values. The results suggest that LaDeCo is a scalable algorithm with respect to dataset size.

- 4.5. Comparison with Task-specific Baselines

FlexDM OpenCOLE Ours GT

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Figure 9. Qualitative comparison on typography generation.

blocking the main content within the given canvas.

Typography generation. We consider FlexDM [11] and OpenCOLE [12] as baselines. For FlexDM, we only mask the text attribute fields, leaving the position fields accessible. For OpenCOLE, which has already been trained on Crello-v4, we leverage the released Typography LMM model 5 for comparison. Figure 9 shows the resutls. LaDeCo takes into consideration various aspects such as text layout, aesthetics, and readability during typography generation process. In contrast, the baseline models suffer from text overlap and poor readability.

###### 5. Conclusion

In this work, we introduce LaDeCo for design composition. The main idea is integrating the inherent hierarchical structure, which emerges from the layered principle during the practical design process, into LMMs. Specifically, a layer planning module categorizes input elements into different layers based on their contents, and a layered design composition process predicts the attributes that controls the composition by using the rendered previous layers as context. In the future, we plan to expand our model to accommodate more user requirements, e.g., textual descriptions. We also plan to explore the integration of our design composition model with image generation models for element content creation, aiming to achieve end-to-end design generation.

We further compare LaDeCo with task-specific baselines, which focus on handling sub-tasks of design composition.

Content-aware layout generation. We consider two state-of-the-art baselines specialized for this subtask, including PosterLLava [32] and PosterLlama [27]. We retrain them on the Crello dataset for a fair comparison. Be-

- sides geometry-related metrics, we further adopt metrics in DS-GAN [8] to assess content-related quality based on the given canvas, i.e., canvas utility (Uti), occlusion (Occ), and text readability (Rea). Table 3 shows quantitative results. Compared to specialized methods, LaDeCo achieves best performance on most metrics. Figure 8 shows qualitative results. LaDeCo excels in generating layouts that prevent

5https://huggingface.co/cyberagent/opencole-typographylmm

###### References

- [1] https://openai.com/index/hello-gpt4o/. [Accessed 04-11-2024]. 2, 3, 4, 5, 11
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2, 3
- [3] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2
- [4] Yutao Cheng, Zhao Zhang, Maoke Yang, Hui Nie, Chunyuan Li, Xinglong Wu, and Jie Shao. Graphic design with large multimodal model. arXiv preprint arXiv:2404.14368, 2024. 2
- [5] Yifan Gao, Jinpeng Lin, Min Zhou, Chuanbin Liu, Hongtao Xie, Tiezheng Ge, and Yuning Jiang. Textpainter: Multimodal text image generation with visual-harmony and text-comprehension for poster design. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7236–7246, 2023. 2, 3
- [6] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415,

2016. 5

- [7] Daichi Horita, Naoto Inoue, Kotaro Kikuchi, Kota Yamaguchi, and Kiyoharu Aizawa. Retrieval-augmented layout transformer for content-aware layout generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 67– 76, 2024. 2
- [8] Hsiao Yuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong, and Qing Zhang. Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6018–6026, 2023. 2, 6, 8
- [9] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685,

2021. 5

- [10] Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. LayoutDM: Discrete Diffusion Model for Controllable Layout Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10167–10176, 2023. 3

- [11] Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. Towards flexible multi-modal document models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14287–14296, 2023. 2, 3, 5, 8
- [12] Naoto Inoue, Kento Masui, Wataru Shimoda, and Kota Yamaguchi. Opencole: Towards reproducible automatic graphic design generation. arXiv preprint arXiv:2406.08232, 2024. 3, 4, 5, 8
- [13] Peidong Jia, Chenxuan Li, Zeyu Liu, Yichao Shen, Xingru Chen, Yuhui Yuan, Yinglin Zheng, Dong Chen, Ji Li, Xiaodong Xie, et al. Cole: A hierarchical generation framework for graphic design. arXiv preprint arXiv:2311.16974, 2023. 2, 3, 5, 6
- [14] Zhaoyun Jiang, Jiaqi Guo, Shizhao Sun, Huayu Deng, Zhongkai Wu, Vuksan Mijovic, Zijiang James Yang, Jian-Guang Lou, and Dongmei Zhang. Layoutformer++: Conditional graphic layout generation via constraint serialization and decoding space restriction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18403– 18412, 2023. 3
- [15] Kotaro Kikuchi, Naoto Inoue, Mayu Otani, Edgar Simo-Serra, and Kota Yamaguchi. Multimodal markup document models for graphic design completion. arXiv preprint arXiv:2409.19051, 2024. 3
- [16] Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. Constrained graphic layout generation via latent optimization. In Proceedings of the 29th ACM International Conference on Multimedia, pages 88–96, 2021. 3
- [17] Xiang Kong, Lu Jiang, Huiwen Chang, Han Zhang, Yuan Hao, Haifeng Gong, and Irfan Essa. Blt: Bidirectional layout transformer for controllable layout generation. In European Conference on Computer Vision, pages 474–490. Springer, 2022. 3
- [18] Hsin-Ying Lee, Lu Jiang, Irfan Essa, Phuong B Le, Haifeng Gong, Ming-Hsuan Yang, and Weilong Yang. Neural design network: Graphic layout generation with constraints. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part III 16, pages 491–506. Springer, 2020. 3
- [19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326,

2024. 5

- [20] Jianan Li, Jimei Yang, Jianming Zhang, Chang Liu, Christina Wang, and Tingfa Xu. Attribute-conditioned layout gan for automatic graphic design. IEEE

- Transactions on Visualization and Computer Graphics, 27(10):4039–4048, 2020. 3
- [21] Jiawei Lin, Jiaqi Guo, Shizhao Sun, Weijiang Xu, Ting Liu, Jian-Guang Lou, and Dongmei Zhang. A parse-then-place approach for generating graphic layouts from textual descriptions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23622–23631, 2023. 3
- [22] Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang Yang, Jian-Guang Lou, and Dongmei Zhang. Layoutprompter: Awaken the design ability of large language models. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 4, 6
- [23] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5
- [24] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424,

2023. 3

- [25] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of-thought prompting for large multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14420–14431, 2024. 3
- [26] KhayTze Peong, Seiichi Uchida, and Daichi Haraguchi. Typographic text generation with off-the-shelf diffusion model. In International Conference on Document Analysis and Recognition, pages 52–69. Springer, 2024. 2, 3
- [27] Jaejung Seol, Seojun Kim, and Jaejun Yoo. Posterllama: Bridging design ability of langauge model to contents-aware layout generation. arXiv preprint arXiv:2404.00995, 2024. 2, 4, 6, 8
- [28] Wataru Shimoda, Daichi Haraguchi, Seiichi Uchida, and Kota Yamaguchi. Towards diverse and consistent typography generation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7296–7305, 2024. 2
- [29] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805,

2023. 2

- [30] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 3
- [31] Kota Yamaguchi. Canvasvae: Learning to generate vector graphic documents. ICCV, 2021. 2, 5

- [32] Tao Yang, Yingmin Luo, Zhongang Qi, Yang Wu, Ying Shan, and Chang Wen Chen. Posterllava: Constructing a unified multi-modal layout generator with llm. arXiv preprint arXiv:2406.02884, 2024. 2, 8
- [33] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499,

2023. 3

- [34] Junyi Zhang, Jiaqi Guo, Shizhao Sun, Jian-Guang Lou, and Dongmei Zhang. Layoutdiffusion: Improving graphic layout generation by discrete diffusion probabilistic models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7226–7236, 2023. 3
- [35] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 2, 3
- [36] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chainof-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023. 3
- [37] Min Zhou, Chenchen Xu, Ye Ma, Tiezheng Ge, Yuning Jiang, and Weiwei Xu. Composition-aware graphic layout gan for visual-textual presentation designs. arXiv preprint arXiv:2205.00303, 2022. 2
- [38] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2, 3

### From Elements to Design: A Layered Approach for Automatic Graphic Design Composition

#### Supplementary Material

###### A. Implementation Details of Layer Planning

Layer planning is achieved by predicting semantic labels, including background, underlay, logo/image, text, and embellishment, for input elements. In the implementation, we can easily identify text elements since they have the text content attribute. For non-text elements, we then leverage GPT-4o [1] to categorize them into the other four labels in a zero-shot manner. We carefully design the following Element Labeling Prompt to accomplish this effectively.

###### Element Labeling Prompt

You are an excellent graphic designer. Your task is to determine the role of the given element, which is rendered as an image. There are 4 possible options: Background, Underlay, Logo/Image or Embellishment. Please refer to the detailed descriptions below to make your prediction.

Background: The foundational layer of the design, typically large in size and covering the entire canvas. It may consist of a solid color, gradient, landscape image, or similar visual foundation.

Underlay: A supportive layer placed beneath key content, often used to create contrast or highlight the main design elements, such as borders, buttons, color overlays, and so on.

Logo/Image: A core visual element that represents a brand, product, or entity. It combines both imagery and logo elements to capture attention and convey the primary message.

Embellishment: Decorative elements that enhance visual appeal without conveying core information. These elements add style to the design. Note that they are usually small in size. When you respond, please output only one word from the 4 options. Do not include any additional explanations or irrelevant information. The element is <image>. Please predict the given element role:

For training samples, we further include the complete design, canvas size, and element size in the prompt. Such information helps enhance prediction accuracy. For instance, if an element is small in size, it is likely to be an embellishment.

###### Additional Prompt for Training Samples

The overall design is <image>. The canvas width is wcanvaspx, canvas height is hcanvaspx. The element is <image>. The element width is welementpx, element height is helementpx. Please also consider the provided canvas and element width/height, as they might be helpful in making a decision. Please predict the given element role:

###### B. Input-Output Structure in Layered Design Composition

Here we present an example to demonstrate the model input and output in layered design composition. The <image> token is a placeholder which is replaced by image embeddings during data loading, as described in Section 3.4. For text elements, we directly put their content in the model input (e.g., Spring Clean, Best hacks). Since all intermediate designs can be rendered in advance, the training process is parallelized across the five design layers, with the loss function only calculated in the ASSISTANT parts. This ensures the training efficiency of LaDeCo.

###### C. More Qualitative Results

We show more qualitative results and the layered design composition process in Figure 10 and Figure 11, respectively. The results indicate that LaDeCo is proficient in creating high-quality and visually appealing designs with the novel layered design composition technique.

###### D. Interesting Applications in Graphic Design

LaDeCo enables interesting applications in graphic design, including design variation, resolution adjustment, and element filling. We show more results in Figure 12, Figure 13, and Figure 14, respectively.

###### E. More Qualitative Comparison

Figure 15, 16, 17 show more qualitative comparison of the design comparison, content-aware layout generation, and typography generation tasks, respectively. From these results, we see that LaDeCo significantly outperforms the baseline methods in design composition, and it even surpasses the specialized models in two design subtasks.

###### A Data Example

HUMAN: a poster of canvas width 1080px, canvas height 1920px. Please predict step by step according to the semantics of the elements. After each prediction, there will be an intermediate rendering result as a reference to better make the next prediction. Now predict the background elements: element 0: <image>

###### ASSISTANT:

|{<br><br>"index": 0, "left": 3, "top": -5, "width": 1101, "height": 460<br><br>}<br><br>|
|---|

HUMAN: current canvas state: <image>. Now predict the underlay elements: null ASSISTANT:

|{}<br><br>|
|---|

HUMAN: current canvas state: <image>. Now predict the logo/image elements: element 1: <image> ASSISTANT:

|{<br><br>"index": 1, "left": -78, "top": 378, "width": 1228, "height": 1842<br><br>}<br><br>|
|---|

HUMAN: current canvas state: <image>. Now predict the text elements: element 2: Spring Clean, element 3: Best hacks ASSISTANT:

|{ { "index": 2, "index": 3, "left": 98, "left": 272, "top": 375, "top": 547, "width": 874, "width": 537, "height": 125, "height": 68, "angle": 0, "angle": 0, "font": "Raleway", "font": "Raleway", "font_size": 125, "font_size": 68, "color": [29, 29, 27], "color": [0, 0, 0], "text_align": "center", "text_align": "center", "capitalize": "false", "capitalize": "false", "letter_spacing": 0.0, "letter_spacing": 0.0, "line_height": 1.0, "line_height": 1.0<br><br>} }<br><br>|
|---|

HUMAN: current canvas state: <image>. Now predict the embellishment elements: null ASSISTANT:

|{}<br><br>|
|---|

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

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

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

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

###### Figure 10. A gallery of graphic designs created by LaDeCo.

Background + Underlay + Logo/Image + Text + Embellishment

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

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

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

Figure 11. The layered design composition process in LaDeCo. Our approach generates a holistic design in the order of background, underlay, logo/image, text, and embellishment layers.

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

###### Figure 12. More results to demonstrate that LaDeCo can create diverse designs with the same input.

1920px

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

1080px

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

- Figure 13. More results to demonstrate that LaDeCo is able to generate graphic designs with different aspect ratios.

Original Design

+ New Elements

Original Design

+ New Elements

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

- Figure 14. More results to demonstrate that LaDeCo can add new elements to an existing design in a plausible way.

## FlexDM GPT-4o Ours GT

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

Figure 15. More qualitative comparison to demonstrate the superiority of LaDeCo in design composition.

PosterLlama PosterLLaVA Ours GT

PosterLlama PosterLLaVA Ours GT

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

###### FlexDM OpenCOLE Ours GT

###### FlexDM OpenCOLE Ours GT

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

Figure 17. More qualitative comparison to demonstrate the superiority of LaDeCo in typography generation.

Figure 16. More qualitative comparison to demonstrate the superiority of LaDeCo in content-aware layout generation.

