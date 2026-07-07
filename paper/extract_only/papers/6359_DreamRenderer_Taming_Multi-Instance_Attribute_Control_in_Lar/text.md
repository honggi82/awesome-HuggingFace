## DreamRenderer: Taming Multi-Instance Attribute Control in Large-Scale Text-to-Image Models

Dewei Zhou1, Mingwei Li1, Zongxin Yang2, Yi Yang1∗ RELER, CCAI, Zhejiang University 1 DBMI, HMS, Harvard University 2 {zdw1999,mingweili,yangyics}@zju.edu.cn {Zongxin Yang}@hms.harvard.edu Project Page: https://limuloo.github.io/DreamRenderer/

Depth Map + Layout (with 16 Instances) FLUX

3DIS Ours

# arXiv:2503.12885v2[cs.CV]12Apr2025

[Figure 1]

[Figure 2]

[Figure 3]

|1|2|3|4|
|---|---|---|---|
|5|6|7|8|
|9|10|11|12|
|[Figure 4]<br><br>13|14|15|16|

“A shiny {color} pool ball with number {number}”, with combinations as follows:

1) Red, number ‘1’; 2) Blue, number ‘2’; 3) Yellow, number ‘3’; 4) Green, number ‘4’; 5) Purple, number ‘5’; 6) Orange, number ‘6’; 7) Magenta, number ‘7’; 8) Black, number ‘8’;

9) Pink, number ‘9’; 10) Turquoise, number ‘10’; 11) Gold, number ‘11’; 12) Silver, number ‘12’; 13)Maroon, number ‘13’; 14) Teal, number ‘14’; 15) Lavender, number ‘15’; and 16) a pure white pool ball.

“many famous people in a tea gathering.“ 1) Caption America 2) Michelangelo-inspired marble statue 3) Black Widow 4) Spider-Man with a pair of high-tech glasses

[Figure 5]

[Figure 6]

[Figure 7]

|1|
|---|

|2<br><br>|3|
|---|
|
|---|

|[Figure 8]| | | | | |
|---|---|---|---|---|---|
| | | | | | |

[Figure 9]

[Figure 10]

“A black clock with seven birds evenly distributed around the clock’s edge.” 1) A black crow; 2) A purple bird; 3) A green bird; 4) A golden eagle; 5) A red phoenix; 6) A white dove; 7) A brown sparrow.

|5|
|---|

|4|
|---|

| |3|
|---|---|
|2| |

|5|
|---|

|4|
|---|

|6|
|---|

|6|
|---|

“A regal queen on a gothic throne, holding a sword surrounded by skulls, in a dark medieval fantasy setting with dramatic lighting.”

|1|
|---|

|7|
|---|

1) Fire sword. 2) Ice-covered frozen throne. 3) Brown & golden hair.

4) White skull with mask. 5) Sleek black high-heeled boots. 6) Golden skull.

[Figure 11]

[Figure 12]

|3<br><br>Brave Fights Three loongs.”|
|---|

“The B 1) <loong1>; 2) <loong2>; 3) <loong3>;

[Figure 13]

[Figure 14]

[Figure 15]

1

2

<loong1> <loong2> <loong3>

Figure 1. Images generated using DreamRenderer. DreamRenderer is a plug-and-play controller that grants users fine-grained control over the content of each region and instance during depth- or canny-conditioned generation without any training. By leveraging Redux [3] to translate images into text embeddings, it further allows users to seamlessly control generated content based directly on visual inputs.

1

### Abstract

Image-conditioned generation methods, such as depth- and canny-conditioned approaches, have demonstrated remarkable abilities for precise image synthesis. However, existing models still struggle to accurately control the content of multiple instances (or regions). Even state-of-the-art models like FLUX and 3DIS face challenges, such as attribute leakage between instances, which limits user control. To address these issues, we introduce DreamRenderer, a trainingfree approach built upon the FLUX model. DreamRenderer enables users to control the content of each instance via bounding boxes or masks, while ensuring overall visual harmony. We propose two key innovations: 1) Bridge Image Tokens for Hard Text Attribute Binding, which uses replicated image tokens as bridge tokens to ensure that T5 text embeddings, pre-trained solely on text data, bind the correct visual attributes for each instance during Joint Attention; 2) Hard Image Attribute Binding applied only to vital layers. Through our analysis of FLUX, we identify the critical layers responsible for instance attribute rendering and apply Hard Image Attribute Binding only in these layers, using soft binding in the others. This approach ensures precise control while preserving image quality. Evaluations on the COCO-POS and COCO-MIG benchmarks demonstrate that DreamRenderer improves the Image Success Ratio by 17.7% over FLUX and enhances the performance of layoutto-image models like GLIGEN and 3DIS by up to 26.8%. Project Page: https://limuloo.github.io/DreamRenderer/.

### 1. Introduction

Beyond conventional text-to-image models, imageconditioned generation methods [31, 56], such as depthand canny-conditioned approaches, have emerged as powerful tools for more controllable content creation, finding applications in animation, game development, visual restoration, and virtual reality [18, 22, 27, 42, 49, 54, 56].

However, existing techniques cannot accurately constrain the content generated within each specific instance (or region), and their performance deteriorates significantly as the number of controlled elements increases. As shown in Fig. 1, even state-of-the-art methods such as FLUX [3] and 3DIS [63, 64] often fail to follow user-specified inputs, leading to unsatisfactory outcomes in which attributes bleed across different instances. Such limitations ultimately hinder more refined and controllable user-directed creation.

To address these challenges and grant users greater creative control, we introduce DreamRenderer, a trainingfree approach built upon the FLUX model [3]. As illustrated in Fig. 1, DreamRenderer functions as a plug-andplay tool, allowing users to further regulate the content of each instance via bounding boxes or masks, all while pre-

serving overall visual harmony.

When implementing DreamRenderer, we encounter two main challenges. 1) Ensuring text embeddings bind the correct visual information. Current state-of-the-art textto-image models [3, 8] employ the T5 text encoder [37], pre-trained purely on textual data, to extract text embeddings lacking intrinsic visual information. These embeddings instead rely on Joint Attention [7, 16, 25, 51] with image embeddings to incorporate visual information [64]. In scenarios where multiple instances are controlled, attribute confusion can easily arise among different instances and regions. 2) Rendering each instance accurately while preserving overall visual harmony. To accurately generate controlled instances, it is necessary to constrain the image tokens’ attention masks [34, 44, 47] during Joint Attention. Yet overly restrictive constraints can undermine the resulting image quality, highlighting the need for more nuanced strategies to guide the attention masks of image tokens.

A naive solution to ensure each instance’s text embedding binds the correct visual information during Joint Attention is to constrain the tokens (both image and text) of a single instance to attend only to themselves. This strategy effectively simulates a single-instance generation process, thereby preserving the intended visual attributes for each text embedding. However, completely separating each instance’s tokens from those of others significantly degrades the image quality, as shown in Fig. 7.

To address the two challenges outlined above, DreamRenderer introduces two key innovations: 1) Bridge tokens for Hard Text Attribute Binding. Instead of directly using an instance’s original image tokens to help the text tokens bind the correct attributes, we replicate the instance’s image tokens, referred to as Bridge Image Tokens. These bridge tokens are not included in the model’s final output but instead simulate a single-instance generation process during Joint Attention, thereby guiding the text embeddings to acquire the correct visual attributes. In other words, each instance’s text embedding and Bridge Image Tokens attend only to each other, ignoring all other tokens. 2) Hard Image Binding only applied to vital layers. After ensuring each text embedding accurately captures its instance’s visual attributes through Hard Text Attribute Binding, we further ensure each instance’s image embedding incorporates the correct visual information. We adopt two types of “image attribute binding”: a hard bind, where an instance’s image tokens attend exclusively to themselves, and a soft bind, where they can attend to all tokens in the image. From experimental observation, layers near the input and output of FLUX primarily encode global information, while the intermediate layers are crucial for rendering each instance. Therefore, we apply hard binding only in these middle layers, and soft binding elsewhere, thus preserving overall visual harmony while maintaining precise control over each

[Figure 16]

(a) DreamRenderer

Image Tokens Text Tokens

Input Layout

[Figure 17]

Bridge Image Tokens FLUX

Input Tokens

###### 1

(b) Attention Masks

[Figure 18]

Hard Text Attribute Binding

Joint Attention ...

Dream Renderer

###### 2

K Q

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

...

T5Encdoer

“an ice cat and a fire dog”

| |
|---|

Joint Attention

[Figure 19]

Instance1: “an ice cat”

...

Instance2: “a fire dog”

Copy

Text Tokens

Bridge Tokens

Joint Attention

Hard Image Attribute Binding

Copy

K Q

Depth or Canny

...

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |

[Figure 20]

[Figure 21]

Dream Renderer

VAE

Drop Text Tokens & Bridge Tokens

Soft Image Attribute Binding

[Figure 22]

Q K

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |

Noise / Last Output

Image Tokens

Output Tokens

Figure 2. The overview of DreamRenderer. (§ 3.3) (a) The pipeline of DreamRenderer. (b) Attention maps in Joint Attention, which includes 1) Hard Text Attribute Binding (§ 3.5), 2) Hard Image Attribute Binding (§ 3.6), and 3) Soft Image Attribute Binding (§ 3.6). In the attention maps shown in (b), rows represent queries and columns represent keys. We use different patterns to distinguish between image tokens , text tokens , and bridge image tokens , while different colors ( for an ice cat, for a fire dog and for the global text tokens and background image tokens) represent tokens from different instances.

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

FLUX model, clarifying which layers handle global operations and which are pivotal for rendering individual instances, offering fresh insights for subsequent research.

instance’s attributes.

We evaluated DreamRenderer on two widely used benchmarks, COCO-POS [24] and COCO-MIG [61, 62], demonstrating its effectiveness in balancing image quality with fine-grained control. On COCO-POS, we extracted depth maps and canny maps from the COCO dataset as conditional inputs and leveraged their layouts to guide DreamRenderer. The results show that DreamRenderer boosts the Image Success Ratio by 17.7% over the original FLUX model, without sacrificing image quality. On COCO-MIG, we applied DreamRenderer to re-render the outputs of several state-of-the-art layout-to-image models, significantly enhancing both image quality and controllability. Specifically, DreamRenderer increases the Image Success Ratio for GLIGEN [19], InstanceDiffusion [44], MIGC [61], and 3DIS [63] by 26.8%, 19.9%, 8.3%, and 7.4%, respectively.

### 2. Related Works

Text-to-Image Generation. In recent years, text-to-image generation has progressed rapidly [1, 4, 9, 12, 20, 21, 28– 30, 32, 41, 48, 53], with generative models now producing high-quality images aligned with user descriptions. Early approaches—such as SD1.5 [39] and SDXL [35]—adopted a U-Net-based [40, 57–60] architecture together with a CLIP text encoder [36] for extracting text embeddings, injecting textual information into the network via crossattention [10, 43] to ensure correspondence between text and output content. Later, the Diffusion Transformer (DiT) [6] structure was introduced, replacing the U-Net with a Transformer, and using AdaLN to incorporate text information. To further refine text control, models like SD3 [8] and FLUX [3] switched to a T5-XXL [37] text encoder, employing Joint Attention to align image and text embeddings and thereby ensure that the final outputs adhere closely to the textual prompts.

The key contributions of this paper are as follows:

- • We propose DreamRenderer, a training-free method that enables users to control the generation content for each region and instance in depth-conditioned or cannyconditioned generation.
- • We introduce a novel Hard Text Attribute Binding technique, ensuring that text embeddings bind the correct visual attributes during the Joint Attention process.
- • For multi-instance generation, we provide the first indepth analysis of each layer’s potential function in the

Controllable Text-to-Image Generation. With further improvements in image quality, users and researchers have turned their attention to more controllable content generation. Image-conditioned generation methods, like

depth- and canny-conditioned techniques [2, 18, 23, 33, 45, 52, 56], provide an outline-based or scene-level structure for generated images. However, current methods lack the ability to precisely regulate individual regions or instances—particularly in complex scenes with multiple controlled elements—often leading to inaccurate or undesirable results. To address this limitation, we introduce DreamRenderer, a plug-and-play controller that empowers users to specify the generation content of each region or instance with heightened accuracy. In addition, DreamRenderer can be applied to re-render the outputs of layout-toimage [5, 11, 17, 19, 44, 46, 50, 55, 61, 63] models, further enhancing both content fidelity and overall image quality.

### 3. Method

#### 3.1. Preliminaries

FLUX. The FLUX [3] model is a state-of-the-art text-toimage generation model that leverages the rectified flow [8, 26] framework to iteratively generate images. In each iteration, FLUX first encodes the text prompt into a text embedding using a T5 text encoder [37], which is trained solely on textual data. This text embedding is then concatenated with a randomly sampled noise-based image embedding; if needed, a control image embedding (e.g., depth or canny) is also included. Subsequently, multiple layers of Joint Attention [7, 25, 64] iteratively refine both the image and text embeddings, ensuring they remain in alignment. Finally, FLUX decodes the refined image embedding through a Variational Autoencoder (VAE) [14], generating an output image that faithfully reflects the user’s textual description.

Joint Attention. In Joint Attention, both text and image embeddings are processed simultaneously. Each embedding is first transformed into token representations via a linear layer: the text embedding is converted into Qtext, Ktext, and Vtext, while the image embedding is mapped to Qimage, Kimage, and Vimage. These token sets are then concatenated as follows: Q = Qtext ⊕ Qimage, K = Ktext ⊕ Kimage, and V = Vtext ⊕ Vimage, where ⊕ denotes the concatenation operation. The attention mechanism is subsequently applied to the concatenated tokens.

QKT √dk

V (1)

Attention(Q,K,V ) = Softmax

where dk is the dimension of the key vectors. This Joint attention mechanism allows for bidirectional information flow between text and image embeddings, enabling the model to align visual content with textual descriptions.

#### 3.2. Problem Definition and Challenges

Problem Definition. Given a depth/canny map, a global prompt, and detailed instance (or region) descriptions, the objective is to generate an image that not only adheres to the global structural constraints of the depth/canny map and

Layer-wise Performance Analysis

- 0

- 1

- 2

- 3

Input Layers Middle Layers Output Layers

AccuracyImprovement(%)

- 0 10 20 30 40 50 Layer Index

5

4

3

2

- 1

Avg: -0.22% Avg: +0.64% Avg: -0.21%

High Performance Layers

Low Performance Layers

Figure 3. Vital Binding Layer Search (§ 3.6). We apply Hard Image Attribute Binding layer by layer and observe that applying it in the FLUX model’s input or output layers degrades performance, whereas applying it in the middle layers yields improvements.

global prompt, but also ensures that each controlled instance accurately adheres the specified descriptions.

Challenges. We build DreamRenderer upon the FLUX model (§ 3.1), and there are two main challenges in achieving our objectives: 1) Correctly binding text embedding attributes in multi-instance scenarios. Unlike previous textto-image models that use CLIP [36], FLUX employs the T5 text encoder, which is pre-trained on pure text data and does not contain visual information. This creates a challenge in ensuring that the text embedding correctly binds the visual attributes of each instance during the Joint Attention process, particularly when multiple instances are involved. 2) Enhancing the overall harmony of the generated image while maintaining correct attributes for each instance. In multi-instance scenarios, the image tokens corresponding to different instances may inadvertently leak attributes [34] during Joint Attention. Our goal is to minimize such leakage while ensuring that the final generated image retains a cohesive and harmonious composition across all instances.

#### 3.3. Overview

Fig. 2 illustrates the overview of DreamRenderer. In the Joint Attention mechanism, DreamRenderer introduces a novel Hard Text Attribute Binding (§3.5) algorithm to ensure that the text embedding for each instance correctly binds the relevant visual information. Additionally, to enhance the overall harmony of the generated image while maintaining accurate image embedding attributes for each instance, we conducted an experimental analysis (§3.6) of each layer in FLUX and decided to apply Hard Image Attribute Binding (§3.6) only in the middle layers of the FLUX model. In all other layers, Soft Image Attribute Binding (§3.6) is used.

#### 3.4. Preparation

As shown in Fig. 2 (a), DreamRenderer first embeds the input text descriptions for each instance and the global prompt separately through the T5 text encoder. These encoded embeddings are then concatenated to form the complete text embedding for the generation process. Our method requires users to provide either a depth map or a canny edge map as structural guidance, which serves as the foundation for the spatial arrangement of instances in the generated image. For instance localization, we utilize bounding boxes or masks provided by the user to identify each instance’s region within this structural guidance.

#### 3.5. Hard Text Attribute Binding

Motivation. When generating a single instance, the FLUX model generally produces images that align with the textual prompt, exhibiting minimal attribute errors. In such scenarios, image and text tokens in Joint Attention focus exclusively on the information of that single instance, thereby enabling the text embedding to bind accurate visual attributes. Building on this insight, we propose that in multi-instance scenarios, the image and text tokens of each instance should primarily attend to themselves rather than to tokens belonging to different instances, allowing the text embedding to effectively bind the correct visual information.

Naive Solution. A straightforward approach to ensure that each instance’s text embedding is bound to the correct attributes is to process each instance independently during Joint Attention. In this method, both image and text tokens for a given instance interact exclusively with themselves, remaining isolated from tokens of other instances. However, this complete isolation introduces a significant drawback: it disrupts the visual harmony of the overall image and substantially lowers the quality of the generated result (as shown in Fig. 7).

Bridge Image Token for Advanced Solution. Since strictly isolating original image tokens for each instance in joint Attention degrades the image quality, DreamRenderer proposes an advanced solution: during Joint Attention, an additional copy of each instance’s image tokens are created, referred to as Bridge Image Tokens. These Bridge Image Tokens do not contribute to the final output image but serves solely to assist each instance’s text embedding in binding the correct visual attributes during the Joint Attention. As shown in Fig. 2, the Bridge Image Tokens and text tokens for each instance align exactly as in a single-instance generation process, ensuring that the visual attributes in the final text embedding are consistent with the textual description. Formally, for the i-th instance, the Hard Text Attribute Binding attention mask Mtexti is defined as:

Mtexti [q,k] =

1, if q,k ∈ Ti ∪ Bi 0, otherwise

(2)

where q and k are the query and key indices respectively, Ti and Bi denote the token indices of the i-th instance’s text and Bridge Image Tokens respectively.

#### 3.6. Image Attribute Binding

Overview. After ensuring the accuracy of the text embedding’s attributes, the next step is to guarantee the correctness of the visual attributes in each instance’s image tokens. As shown in Fig. 2, DreamRenderer employs Hard Image Attribute Binding at the vital binding layer to ensure that each instance is rendered with the correct attributes. In the remaining layers, Soft Image Attribute Binding is used to ensure that all instances ultimately form a coherent image. In the following sections, we will detail the mechanisms of hard and soft image attribute binding and explain how we identify the vital layers for Hard Image Attribute Binding.

Hard Image Attribute Binding. As shown in Fig. 2, the Hard Image Attribute Binding imposes attention mask constraints on each instance’s image tokens during Joint Attention. First, it ensures that each instance attends only to its corresponding text token—which, following Hard Text Attribute Binding, contains the correct visual information. Second, to prevent attribute leakage across different image tokens, Hard Image Attribute Binding further restricts each instance’s image tokens to attend its own image tokens. For the i-th instance, the Hard Image Attribute Binding mask Mhardi is constructed as:

Mhardi [q,k] =

1, if q ∈ Ii,k ∈ Ti ∪ Ii 0, otherwise

(3)

where Ii represents the set of image token indices belonging to the i-th instance, and Ti represents its corresponding text token indices.

Soft Image Attribute Binding. To ensure the overall coherence of the final generated image, Soft Image Attribute Binding relaxes the constraints imposed by Hard Image Attribute Binding. Specifically, for each instance’s image tokens, Soft Image Attribute Binding allows it to attend to the image tokens of the entire image, thereby enhancing the cohesion of the entire scene. For the i-th instance, the Soft Image Attribute Binding mask Msofti is constructed as:

Msofti [q,k] =

1, if q ∈ Ii,k ∈ Iall 0, otherwise

(4)

where Iall = Ibackground∪ Ni=1 Ii represents the set of all image token indices, Ibackground is the set of all image tokens that are not associated with any instance.

Search Vital Binding Layers. As shown in Fig. 3, we applied Hard Image Attribute Binding layer by layer on the FLUX network, which consists of 57 Joint Attention layers, and compared the result obtained using Soft Image Binding

###### Depth & Layout FLUX-Depth Ours

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

|1|
|---|

2

- 3 4

|5|
|---|

1) Bread 2) Apple 3) Apple 4) Orange 5) Orange

Depth & Layout 3DIS-Depth Ours

|1|
|---|

|2|
|---|

|3|
|---|

|4|
|---|

|5|
|---|

1) Broccoli 2) Broccoli 3) Broccoli 4) Broccoli 5) Spoon Canny & Layout FLUX-Canny Ours

||1 4|
|---|
|
|---|

2

- 3 4 5

1) Chair 2) TV 3) Coach 4) Ship 5) Couch

Canny & Layout 3DIS-Canny Ours

1

3

5

4

- 1) Sandwich 2) Sandwich 3) Banana 4) Bowl 5) Apple 6) Apple

[Figure 39]

[Figure 40]

- 2

6

[Figure 41]

[Figure 42]

[Figure 43]

1 2

3

|5|
|---|

Figure 4. Qualitative Comparison on the COCO-POS benchmark (§ 4.2).

Table 1. Quantitative results on COCO-POS benchmark (§ 4.2). SR: Success Ratio of the Entire Image, ISR: Instance Success Ratio, MIoU: Mean Intersection over Union, AP: Average Precision. ∗: canny-guided. †: depth-guided.

Method SR (%)↑ ISR (%)↑ MIoU (%)↑ AP (%)↑ CLIP↑

FLUX∗ 16.38 64.95 59.71 31.88 19.03 3DIS∗ 18.07 68.78 62.97 31.36 19.46 Ours∗ 23.28 74.61 66.95 37.00 20.03

FLUX† 44.83 85.13 76.86 50.72 19.82 3DIS† 53.88 90.33 81.26 54.26 20.23 Ours† 62.50 94.51 84.36 58.95 20.74

across all layers to identify which layers are more suitable for binding specific instance attributes. The results from

- Fig. 3 indicate that applying Hard Image Binding near the output and input layers of FLUX leads to a significant decrease in performance. Conversely, we observed that implementing hard image attribute binding in the middle layers of FLUX often enhances attribute fidelity. Based on these findings, we argue that the layers at the input and output of FLUX are primarily involved in handling global image information, while the mid-layers play a critical role in rendering the attributes of instances within the image. Therefore, we perform Hard Image Binding in the mid-layers of FLUX, while employing Soft Image Binding in the remaining layers. This approach optimally balances the fidelity of instance attributes with the overall coherence of the image.

- 4. Experiments

- 4.1. Experiment Setup

[Figure 44]

sample images over 20 steps. During depth-conditioned generation, we set the Classifier-Free Guidance (CFG) [13] scale to 10.0, while for canny-conditioned generation, the CFG scale is set to 30. In our experiments, for instances whose positions are specified by bounding boxes, we further segment them using the SAM-2 [15, 38] model to obtain more precise instance masks, which is consistent with previous work [63].

Evaluation Benchmarks. We conduct experiments on two widely used benchmarks: 1) COCO-POS Benchmark, which requires generating images according to specified layouts. We extract depth maps or canny edges from COCO dataset images as conditioning signals and utilize the dataset’s inherent layouts for rendering. Models must generate results matching the instance categories at designated locations. Here we compare our approach with training-free rendering methods including Multi-Diffusion and 3DIS. 2) COCO-MIG Benchmark, which tests multiinstance generation with accurate position and attribute control. We evaluate DreamRenderer’s integration capabilities with state-of-the-art MIG models by first generating RGB images with these models, then extracting depth maps to combine with layouts for instance rendering. This evaluates DreamRenderer’s attribute control effectiveness when applied to existing MIG frameworks.

Evaluation Metrics. We used the following metrics to evaluate the model: 1) Mean Intersection over Union (MIoU): Measures the overlap ratio between the rendered instance positions and the target positions. 2) Local CLIP Score: Assesses the visual consistency of the rendered instances with their corresponding textual descriptions. 3) Average Precision (AP): Evaluates the accuracy of the rendered image’s layout. 4) Instance Success Ratio (ISR): Calculates the ratio of correctly rendered instances. 5) Image Success Ratio (ISR): Measures the ratio of images where all instances are correctly rendered.

Baselines. Beside the FLUX model, we evaluate our method against several state-of-the-art methods for multiinstance generation. Since DreamRenderer is designed as a plug-and-play solution, we conduct experiments by integrating it with existing methods: GLIGEN [19], InstanceDiffusion [44], MIGC [61], and 3DIS [64].

Implementation Details. We perform canny-conditioned and depth-conditioned generation using FLUX.1-Canny [3] and FLUX.1-Depth [3], respectively. For both cases, we

Table 2. Quantitative results on COCO-MIG benchmark (§ 4.2). Li means that the count of instances needed to generate is i.

Instance Success Ratio (%)↑ Mean Intersection over Union (%)↑ Image Success Ratio (%)↑

Method L2 L3 L4 L5 L6 AVG L2 L3 L4 L5 L6 AVG L2 L3 L4 L5 L6 AVG GLIGEN [CVPR23] 41.6 31.6 27.0 28.3 27.7 29.7 37.1 29.2 25.1 26.7 25.5 27.5 16.8 4.6 0.0 0.0 0.0 4.4

+Ours 75.5 73.6 67.9 62.6 66.2 67.8 64.3 63.5 58.1 54.2 56.6 58.2 57.4 45.1 27.0 11.4 13.0 31.2

vs. GLIGEN +33.9 +42.0 +40.9 +34.3 +38.5 +38.1 +27.2 +34.3 +33.0 +27.5 +31.1 +30.7 +40.6 +40.5 +27.0 +11.4 +13.0 +26.8 InstanceDiff [CVPR24] 58.4 51.9 55.1 49.1 47.7 51.3 53.0 47.4 49.2 43.7 42.6 46.0 37.4 15.0 11.5 5.0 2.6 14.5

+Ours 77.4 75.4 72.5 65.3 67.2 70.1 68.8 66.4 63.0 57.8 57.5 61.2 58.7 45.8 34.5 13.6 17.5 34.4

vs. InstanceDiff +19.0 +23.5 +17.4 +16.2 +19.5 +18.8 +15.8 +19.0 +13.8 +14.1 +14.9 +15.2 +21.3 +30.8 +23.0 +8.6 +14.9 +19.9 MIGC [CVPR24] 74.8 66.2 67.2 65.3 66.1 67.1 64.5 56.9 57.0 55.5 57.3 57.5 54.8 32.7 22.3 12.9 17.5 28.4

+Ours 79.7 76.3 70.6 67.7 69.4 71.4 68.8 65.0 60.3 57.6 59.3 61.0 64.5 47.7 30.4 18.6 20.1 36.7

vs. MIGC +4.9 +10.1 +3.4 +2.4 +3.3 +4.3 +4.3 +8.1 +3.3 +2.1 +2.0 +3.5 +9.7 +15.0 +8.1 +5.7 +2.6 +8.3 3DIS [ICLR25] 76.5 68.4 63.3 58.1 58.9 62.9 67.3 61.2 56.4 52.3 52.7 56.2 61.3 36.0 25.0 12.9 11.7 29.7

+Ours 79.0 77.3 71.8 68.7 69.4 71.9 69.8 68.2 63.7 61.4 61.2 63.7 63.2 52.9 29.7 18.6 18.8 37.1 vs. 3DIS +2.5 +8.9 +8.5 +10.6 +10.5 +9.0 +2.5 +7.0 +7.3 +9.1 +8.5 +7.5 +1.9 +16.9 +4.7 +5.7 +7.1 +7.4

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

GLIGEN + Ours InstanceDiffusion + Ours

| | | |
|---|---|---|

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

MIGC + Ours 3DIS + Ours

Figure 5. Qualitative comparison on the COCO-MIG benchmark (§ 4.2).

#### 4.2. Comparison with State-of-the-Art Methods

improvement becomes more pronounced as the number of instances to be controlled increases: for example, DreamRenderer’s performance gain over 3DIS is only 2.5% when controlling two instances, but rises to 10.5% when controlling six. These benefits stem from our Hard Text Attribute Binding algorithm, which ensures that each instance’s text embedding accurately binds its visual attributes during Joint Attention, even for a large number of instances.

COCO-POS Benchmarks. Tab. 1 presents quantitative results comparing our method against FLUX and 3DIS. Our approach achieves consistent superiority across all metrics in both depth-guided and canny-guided generation scenarios. In the depth-guided setting, DreamRenderer shows a substantial improvement in SR (62.50% vs. 53.88% for 3DIS), indicating more coherent scene structures. The high ISR (94.51%) and MIoU (84.36%) further corroborate its precise instance-level control. In the more demanding canny-guided scenario, DreamRenderer also exceeds the 3DIS by 5.21% in SR. Meanwhile, as shown in

Table 3. User study results with 31 participants (1-5 scale)(§ 4.2).

Method Layout Accuracy↑ Visual Quality↑

FLUX 3.51 3.26 3DIS 3.63 3.31 Ours 4.04 3.80

- Fig. 4, our method does not compromise the original FLUX model’s image generation quality—thanks to applying hard image attribute binding only within the vital layers.

User Study. Tab. 3 shows a user study with 31 participants comparing our method to FLUX [3] and 3DIS [64] on perceptual quality. Participants viewed paired outputs and rated each on (1) Layout Accuracy and (2) Image Quality, using a 5-point scale, in a blind comparison with randomized presentation. Each participant evaluated 17 pairs, with the input layout and text descriptions displayed. The results show that our proposed DreamRenderer not only enhances

COCO-MIG Benchmarks. Tab. 2 and Fig. 5 present the results of applying DreamRenderer to various stateof-the-art layout-to-image methods. As shown, DreamRenderer substantially enhances the instance-attribute control accuracy, ultimately boosting the Image Success Ratio by 26.8% over GLIGEN, 19.9% over InstanceDiffusion, 8.3% over MIGC, and 7.4% over 3DIS. Notably, this

Table 4. Ablation study on Hard Text Attribute Binding (§ 3.5). Li means that the count of instances needed to generate is i.

Instance Success Ratio (%)↑ Mean Intersection over Union (%)↑ Image Success Ratio (%)↑

Method L2 L3 L4 L5 L6 AVG L2 L3 L4 L5 L6 AVG L2 L3 L4 L5 L6 AVG Ours (Full) 79.0 77.3 71.8 68.7 69.4 71.9 69.8 68.2 63.7 61.4 61.2 63.7 63.2 52.9 29.7 18.6 18.8 37.1

Naive Solution 59.7 56.0 48.8 41.1 46.4 48.5 53.8 50.9 43.8 38.7 42.2 44.2 30.3 17.7 7.4 1.4 4.6 12.5

vs. Full -19.3 -21.3 -23.0 -27.6 -23.0 -23.4 -16.0 -17.3 -19.9 -22.7 -19.0 -19.5 -32.9 -35.2 -22.3 -17.2 -14.2 -24.6 w/o Hard Text Attribute Binding 75.5 72.3 68.1 64.6 63.2 67.2 66.9 65.3 60.6 58.1 56.1 60.0 58.1 41.8 24.3 11.4 9.7 29.5

vs. Full -3.5 -5.0 -3.7 -4.1 -6.2 -4.7 -2.9 -2.9 -3.1 -3.3 -5.1 -3.7 -5.1 -11.1 -5.4 -7.2 -9.1 -7.6

Table 5. Ablation study on Image Attribute Binding (§ 3.6). Li means that the count of instances needed to generate is i.

Instance Success Ratio (%)↑ Mean Intersection over Union (%)↑ Image Success Ratio (%)↑

Method L2 L3 L4 L5 L6 AVG L2 L3 L4 L5 L6 AVG L2 L3 L4 L5 L6 AVG w/o Hard Image Bind 71.3 69.5 56.9 51.4 47.6 56.2 63.8 61.3 52.3 47.5 43.9 51.2 53.6 37.9 16.2 4.3 3.9 23.6

Hard Image Bind Input Layer 71.0 68.4 56.6 52.9 50.3 57.1 62.4 60.5 50.4 47.2 45.4 50.9 51.0 34.0 14.9 5.7 5.2 22.5 Hard Image Bind Middle Layer (Ours) 79.0 77.3 71.8 68.7 69.4 71.9 69.8 68.2 63.7 61.4 61.2 63.7 63.2 52.9 29.7 18.6 18.8 37.1

Hard Image Bind Output Layer 64.5 63.8 57.4 49.4 58.3 57.6 56.9 57.2 51.8 45.0 52.1 51.7 43.9 28.8 14.9 2.9 7.8 20.0

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

| |
|---|

| | |
|---|---|
| | |

Depth w/o

Depth w/o

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Input Layers Output Layers Mid Layers (ours)

Input Layers Output Layers Mid Layers (ours)

##### Figure 6. Ablation study on Hard Image Attribute Binding (§ 3.6).

improving accuracy without compromising image quality. As the number of controlled instances increases, the benefit of Hard Text Attribute Binding becomes more pronounced: for example, moving from 2 to 6 instances raises the Instance Success Ratio improvement from 3.5% to 6.2%.

[Figure 69]

[Figure 70]

[Figure 71]

| | |
|---|---|
| | |

| | |
|---|---|

w/o

Naive Solution Ours

Figure 7. Ablation study on Hard Text Attribute Binding (§ 3.5). We use the same layout from Fig. 1 for testing. Due to space limitations in the main text, additional image results are provided in the Supplementary Materials.

Vital Layers for Image Attribute Binding. Tab. 5 and Fig. 6 present our ablation study on the Hard Image Attribute Binding mechanism. Applying Hard Image Attribute Binding at the FLUX input or output layers yields no clear performance gains and substantially degrades image quality, indicating that these layers are critical for the model’s global information processing. Imposing instance or region isolation at these stages severely disrupts the intermediate feature distribution, ultimately causing a sharp drop in performance. In contrast, restricting Hard Image Attribute Binding to the mid layers preserves image quality while significantly improving performance—for instance, boosting the Instance Success Ratio by 15.7%. This finding shows that FLUX’s mid layers play a pivotal role in determining each instance’s visual content, making them more suitable for binding instance’s attribute.

the FLUX model’s layout control capabilities but also generates outputs that are more visually appealing to users.

#### 4.3. Ablation Study

Bridge Image Tokens for Hard Text Attribute Binding. Tab. 4 and Fig. 7 present our ablation study on the Hard Text Attribute Binding mechanism. The Naive Solution (§ 3.5) isolates each instance during Joint Attention, disrupting the model’s inherent feature distribution and thus causing a performance drop. Introducing Bridge Image Tokens—which are not part of the final output—can effectively address this issue, enabling text tokens to bind the correct attributes and

### 5. Conclusion

We present DreamRenderer, a plug-and-play approach that enables depth- and canny-conditioned generation to control the content of specific regions and instances, without compromising the original model’s image quality. Our work makes two key contributions. First, we introduce a novel Hard Text Attribute Binding mechanism that employs Bridge Image Tokens, ensuring each instance’s text embedding binds the correct visual information during Joint Attention. Second, through an experimental analysis of FLUX’s individual layers, we apply hard image attribute binding only to the vital layers, maintaining precise instance-level control while preserving global image coherence. Extensive experiments on the COCO-POS and COCO-MIG benchmarks demonstrate DreamRenderer’s superior performance. In the depth-guided setting, our method achieves 62.50% SR, 94.51% ISR, and 84.36% MIoU, substantially outperforming existing approaches. Even under the more challenging canny-guided setting, DreamRenderer remains robust, achieving 74.61% ISR, and 66.95% MIoU. Furthermore, DreamRenderer can serve as a re-renderer, significantly improving the accuracy of layout-to-image approaches. Its training-free nature allows DreamRenderer to be easily applied to various foundation models, offering high flexibility. In the future, we will further explore DreamRenderer’s integration with additional image-conditioned generation approaches.

### References

- [1] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 3
- [2] Shariq Farooq Bhat, Niloy Mitra, and Peter Wonka. Loosecontrol: Lifting controlnet for generalized depth conditioning. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11,

2024. 4

- [3] BlackForest. Black forest labs; frontier ai lab, 2024. 1, 2, 3, 4, 6, 7
- [4] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023.

- 3

[5] Zhennan Chen, Yajie Li, Haofan Wang, Zhibo Chen, Zhengkai Jiang, Jun Li, Qian Wang, Jian Yang, and Ying Tai. Region-aware text-to-image generation via hard binding and soft refinement. arXiv preprint arXiv:2411.06558, 2024.

- 4

- [6] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. In ICLR 2023 (Eleventh International Conference on Learning Representations), 2023. 3
- [7] Yusuf Dalva, Kavana Venkatesh, and Pinar Yanardag. Fluxs-

- pace: Disentangled semantic editing in rectified flow transformers. arXiv preprint arXiv:2412.09611, 2024. 2, 4
- [8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 2, 3, 4

- [9] Aditya Ramesh et al. Hierarchical text-conditional image generation with clip latents, 2022. 3
- [10] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Reddy Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Trainingfree structured diffusion guidance for compositional text-toimage synthesis. In ICLR, 2023. 3
- [11] Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-to-image diffusion for accurate instruction following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4744–4753, 2024. 4
- [12] Daiheng Gao, Shilin Lu, Shaw Walters, Wenbo Zhou, Jiaming Chu, Jie Zhang, Bang Zhang, Mengxi Jia, Jian Zhao, Zhaoxin Fan, et al. Eraseanything: Enabling concept erasure in rectified flow transformers. arXiv preprint arXiv:2412.20413, 2024. 3
- [13] Jonathan Ho. Classifier-free diffusion guidance. ArXiv, abs/2207.12598, 2022. 6
- [14] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013. 4
- [15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 6
- [16] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [17] Yuseung Lee, Taehoon Yoon, and Minhyuk Sung. Groundit: Grounding diffusion transformers via noisy patch transplantation. Advances in Neural Information Processing Systems, 37:58610–58636, 2024. 4
- [18] Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet++: Improving conditional controls with efficient consistency feedback. arXiv preprint arXiv:2404.07987, 2024. 2, 4
- [19] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. CVPR,

- 2023. 3, 4, 6

[20] You Li, Fan Ma, and Yi Yang. Anysynth: Harnessing the power of image synthetic data generation for generalized vision-language tasks. arXiv preprint arXiv:2411.16749,

- 2024. 3

- [21] You Li, Fan Ma, and Yi Yang. Imagine and seek: Improving composed image retrieval with an imagined proxy. arXiv preprint arXiv:2411.16752, 2024. 3

- [22] Chao Liang, Fan Ma, Linchao Zhu, Yingying Deng, and Yi Yang. Caphuman: Capture your moments in parallel universes. In CVPR, 2024. 2
- [23] Han Lin, Jaemin Cho, Abhay Zala, and Mohit Bansal. Ctrladapter: An efficient and versatile framework for adapting diverse controls to any diffusion model. arXiv preprint arXiv:2404.09967, 2024. 4
- [24] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Doll´ar. Microsoft coco: Common objects in context, 2015. 3
- [25] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 2, 4
- [26] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4
- [27] Xiaoyu Liu, Yuxiang Wei, Ming Liu, Xianhui Lin, Peiran Ren, Xuansong Xie, and Wangmeng Zuo. Smartcontrol: Enhancing controlnet for handling rough visual conditions. In European Conference on Computer Vision, pages 1–17. Springer, 2024. 2
- [28] Shilin Lu, Yanzhu Liu, and Adams Wai-Kin Kong. Tf-icon: Diffusion-based training-free cross-domain image composition. In ICCV, 2023. 3
- [29] Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, and Adams Wai-Kin Kong. Mace: Mass concept erasure in diffusion models. CVPR, 2024.
- [30] Shilin Lu, Zihan Zhou, Jiayou Lu, Yuanzhi Zhu, and Adams Wai-Kin Kong. Robust watermarking using generative priors against image editing: From benchmarking to advances. arXiv preprint arXiv:2410.18775, 2024. 3
- [31] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence, pages 4296–4304, 2024. 2
- [32] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. ICML, 2022. 3
- [33] Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, MingChang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024. 4
- [34] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7932–7942, 2024. 2, 4
- [35] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3

- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 3, 4
- [37] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. 2, 3, 4
- [38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 6
- [39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 3
- [40] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. MICCAI, abs/1505.04597, 2015. 3
- [41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, Seyedeh Sara Mahdavi, Raphael Gontijo Lopes, Tim Salimans, Jonathan Ho, David Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In NIPS, 2022. 3
- [42] Gabriela Ben Melech Stan, Diana Wofk, Scottie Fox, Alex Redden, Will Saxton, Jean Yu, Estelle Aflalo, Shao-Yen Tseng, Fabio Nonato, Matthias Muller, et al. Ldm3d: Latent diffusion model for 3d. arXiv preprint arXiv:2305.10853,

2023. 2

- [43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [44] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. Instancediffusion: Instance-level control for image generation, 2024. 2, 3, 4, 6
- [45] Yilin Wang, Haiyang Xu, Xiang Zhang, Zeyuan Chen, Zhizhou Sha, Zirui Wang, and Zhuowen Tu. Omnicontrolnet: Dual-stage integration for conditional image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7436–7448, 2024. 4
- [46] Yinwei Wu, Xianpan Zhou, Bing Ma, Xuefeng Su, Kai Ma, and Xinchao Wang. Ifadapter: Instance feature control for grounded text-to-image generation. arXiv preprint arXiv:2409.08240, 2024. 4
- [47] Jiayu Xiao, Liang Li, Henglei Lv, Shuhui Wang, and Qingming Huang. R&b: Region and boundary aware zeroshot grounded text-to-image generation. arXiv preprint arXiv:2310.08872, 2023. 2
- [48] Rui Xie, Ying Tai, Kai Zhang, Zhenyu Zhang, Jun Zhou, and Jian Yang. Addsr: Accelerating diffusion-based blind superresolution with adversarial diffusion distillation. ArXiv, abs/2404.01717, 2024. 3

- [49] Yifeng Xu, Zhenliang He, Shiguang Shan, and Xilin Chen. Ctrlora: An extensible and efficient framework for controllable image generation. arXiv preprint arXiv:2410.09400,

2024. 2

- [50] Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Reco: Region-controlled text-toimage generation. In CVPR, 2023. 4
- [51] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2
- [52] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arxiv:2308.06721,

2023. 4

- [53] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation, 2022. 3
- [54] Fan Zhang, Shaodi You, Yu Li, and Ying Fu. Atlantis: Enabling underwater depth estimation with stable diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11852–11861, 2024. 2
- [55] Hui Zhang, Dexiang Hong, Tingwei Gao, Yitong Wang, Jie Shao, Xinglong Wu, Zuxuan Wu, and Yu-Gang Jiang. Creatilayout: Siamese multimodal diffusion transformer for creative layout-to-image generation. arXiv preprint arXiv:2412.03859, 2024. 4
- [56] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, pages 3836–3847, 2023. 2, 4
- [57] Chen Zhao, Weiling Cai, Chenyu Dong, and Chengwei Hu. Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration. CVPR, 2024. 3
- [58] Chen Zhao, Weiling Cai, Chengwei Hu, and Zheng Yuan. Cycle contrastive adversarial learning with structural consistency for unsupervised high-quality image deraining transformer. Neural Networks, 2024.
- [59] Chen Zhao, Chenyu Dong, and Weiling Cai. Learning a physical-aware diffusion model based on transformer for underwater image enhancement. arXiv preprint arXiv:2403.01497, 2024.
- [60] Dewei Zhou, Zongxin Yang, and Yi Yang. Pyramid diffusion models for low-light image enhancement. In IJCAI, 2023. 3
- [61] Dewei Zhou, You Li, Fan Ma, Zongxin Yang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. CVPR, 2024. 3, 4, 6
- [62] Dewei Zhou, You Li, Fan Ma, Zongxin Yang, and Yi Yang. Migc++: Advanced multi-instance generation controller for image synthesis, 2024. 3
- [63] Dewei Zhou, Ji Xie, Zongxin Yang, and Yi Yang. 3dis: Depth-driven decoupled instance synthesis for text-to-image

generation. arXiv preprint arXiv:2410.12669, 2024. 2, 3, 4, 6

[64] Dewei Zhou, Ji Xie, Zongxin Yang, and Yi Yang. 3dis-flux: simple and efficient multi-instance generation with dit rendering. arXiv preprint arXiv:2501.05131, 2025. 2, 4, 6, 7

