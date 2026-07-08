## InstructDiffusion: A Generalist Modeling Interface for Vision Tasks

Zigang Geng*, Binxin Yang*, Tiankai Hang*, Chen Li*, Shuyang Gu†, Ting Zhang, Jianmin Bao, Zheng Zhang, Han Hu, Dong Chen, Baining Guo Microsoft Research Asia

https://gengzigang.github.io/instructdiffusion.github.io/

# arXiv:2309.03895v1[cs.CV]7Sep2023

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Input Transform it to van Gogh, starry night style Paint the pixels of cheetah on the right in blue and maintain the current appearance of the other pixels

Help the elephant wear a crown

and maintain the appearance of others

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Mark the pixels of the cat in the mirror to blue and leave the rest unchanged

Withdraw the watermark applied Detection to this photograph

Use yellow to encircle the left knee of the people on the far left and draw a blue circle over the nose of the tallest people

Figure 1. We introduce InstructDiffusion, a generalist modeling interface for vision tasks. Given input image and human instruction, our unified model effectively accomplishes tasks such as image editing, segmentation, keypoint estimation, detection, and low-level vision.

### Abstract

We present InstructDiffusion, a unifying and generic framework for aligning computer vision tasks with human instructions. Unlike existing approaches that integrate prior knowledge and pre-define the output space (e.g., categories and coordinates) for each vision task, we cast diverse vision tasks into a human-intuitive image-manipulating process whose output space is a flexible and interactive pixel space. Concretely, the model is built upon the diffusion process and is trained to predict pixels according to user instructions, such as encircling the man’s left shoulder in red or applying a blue mask to the left car. InstructDiffusion could handle a variety of vision tasks, including understanding tasks (such as segmentation and keypoint detection) and generative tasks (such as editing and enhancement). It even exhibits the ability to handle unseen tasks and outperforms prior methods on novel datasets. This represents a significant step towards a generalist modeling in-

*Equal contribution. †Corresponding Author.

terface for vision tasks, advancing artificial general intelligence in the field of computer vision.

### 1. Introduction

In recent years, the field of artificial intelligence has witnessed remarkable advancements, particularly in natural language processing (NLP) [7,13,52,53]. The Generative Pre-trained Transformer (GPT) has successfully unified multiple NLP tasks by providing a single, coherent framework for diverse applications. Building on this success, our research aims to achieve a similar unification in the realm of computer vision, i.e. [10,11], developing a unifying framework capable of handling multiple vision tasks simultaneously. However, compared with NLP tasks, unifying computer vision tasks is more challenging due to the diversity of various tasks.

Diversity of Tasks and Outputs: Computer vision tasks encompass a wide range of applications, such as object recognition, segmentation, image generation, and keypoint detection, among others. Each of these tasks has a different

output format, including coordinates, binary masks, images, and categories. This diversity makes it difficult to find a uniform representation for all tasks. In contrast, NLP tasks often have text-based outputs that can be more easily represented in a standard format.

Different Methodologies and Techniques: Computer vision tasks often require distinct methodologies and techniques depending on the specific problem being addressed. For example, image generation tasks are commonly dominated by Generative Adversarial Networks (GANs) [17,28, 29] and Denoising Diffusion Models (DDPM) [19,22,65], which are rarely used for image understanding tasks such as object recognition or image classification. Additionally, the output dimensionality of generative models is relatively higher, adding to the challenge of unifying these tasks. In contrast, NLP tasks tend to rely on a more consistent set of techniques, such as Transformer-based models [71], which can be applied across various NLP applications.

Continuous Input and Output: Both the input and output of computer vision tasks are usually continuous, like coordinates or images. This continuous nature makes it challenging to develop a unified approach that can accurately handle such data. If discretizing the continuous data using techniques like Vector Quantized-Variational AutoEncoders (VQ-VAE) [58,70], there will be quantization errors, leading to inaccuracies in the results. This issue is less prominent in NLP tasks, where the input and output data can be more easily discretized [7,13,71] into text tokens.

In this paper, we take advantage of the DDPM and propose a novel approach to address these challenges by treating all computer vision tasks as image generation, specifically instructional image editing tasks. We instruct image editing tasks using a more natural and intuitive way that closely aligns with how humans process images. For instance, the instruction for a segmentation task could involve turning the pixels of an object in the image into a specific color, while the remaining pixels remain unchanged. The keypoint detection task can be described as placing an opaque colored circle at a specific position in the image. The instruction for a classification task could change the object to different colors according to its category. Compared with some methods that have attempted to formulate vision tasks as inpainting problems [5, 78], our approach ensures accurate reflection of human intentions which simplifies the process of handling multiple vision tasks. At the same time, since the input and output of DDPM are continuous [22,65], discretization is unnecessary, which solves the problem of quantization error.

We mainly focus on three types of output formats: 3channel RGB images, binary masks, and keypoints. These three outputs are sufficient to cover most vision tasks, such as semantic segmentation, referring segmentation, keypoint detection, image manipulation, and so on. Since the out-

put of the denoising diffusion model is a 3-channel image, we propose a unified representation that encodes masks and keypoints into 3-channel images to handle various image understanding tasks. Then we use a post-processing module to extract the commonly used output format for evaluation.

During the training phase, we use a diverse set of tasks to train a single model uniformly. We also collect a new dataset for image editing. The experimental results demonstrate that our approach achieves good performance in each task. Furthermore, we observed that, compared to training individual models for each task, joint training of multiple tasks can enhance the generalization ability.

Remarkably, our model also exhibits the ability of AGI to a certain extent, as it can handle tasks not seen during the training phase, such as image detection and classification. Moreover, it performs better than previous methods on datasets that were not seen during training. This study thus presents a significant step towards the development of a generalist modeling interface for vision tasks, paving the way for future research in the quest for AGI in computer vision.

### 2. Related Work

Building a general-purpose model that is capable of solving any arbitrary task has been a longstanding desire for artificial intelligence research. There exists a substantial number of related works in the literature, aiming to unify a broad spectrum of tasks. We present a brief overview of recent efforts in this direction.

Vision Language Foundation Models. The vast amount of easily accessible web-scale image-text pairs has brought about a wave of research innovations in vision language foundation models [15,36,39,45,66,75,97]. The pioneering works, CLIP [55] and ALIGN [26], are trained with contrastive loss, showing impressive generalization capabilities for downstream tasks by aligning pairs of images and texts in a cross-modal shared embedding space. Subsequent efforts extend the image-text contrastive method to a broader spectrum, such as the image-text-label space proposed in UniCL [88] and a wider range of tasks as well as modalities supported in Florence [94] and INTERN [62]. However, contrastive-based methods lack the ability to generate language, which limits their application in open-ended tasks such as captioning or visual question answering.

On the other hand, the success of large language models such as GPT series [7, 53, 56, 57], PaLM [4, 12], and LLaMA [68], has been attracting a lot of research interest [24, 38, 64, 69, 73, 74, 80] in augmenting the large language models with visual capabilities. Mostly, these models cast a wide range of open-ended vision tasks as text prediction problems, mapping visual input content to language semantics to enable general-purpose visual and language understanding. BEIT3 [77] unifies the pretraining

task in a masked data modeling manner. CoCa [92] and BLIP [34, 35] unifies contrastive learning and generative learning. Flamingo [2] accepts arbitrarily interleaved visual data and text as input and generates text in an open-ended manner by learning on a broad diversity of vision language tasks. LLaVA [43] exploits visual instruction tuning by converting image-text pairs into an instruction-following format. GLIP v2 [95] and Kosmos v2 [54] leverage grounded image-text pairs to further unlock the grounding capability of multimodal large language models. Our work differs from LLaVA [43] in that, unlike open-ended visual tasks such as visual question answering that can be naturally formulated in an instruction-following format, we attempt to formulate vision tasks, such as segmentation and keypoint detection, into an instruction-following framework. This is challenging due to the unclear instructions and lack of specific guidelines in these tasks.

Vision Generalist Models. Seeking a unified model that, once trained, can be directly used to seamlessly address a wide variety of vision tasks, has been an enduring aspiration in the computer vision community. Multi-task learning [25, 25, 33, 99] has become more and more popular. The key challenge lies in the diversity and complexity of the various structure of task outputs. Currently, there are two major interfaces for output unification: language-like generation and image-resembling generation. Most existing attempts for vision generalists take inspiration from sequence-to-sequence models in the NLP field and model a sequence of discrete tokens through next token prediction [10,20,59,74,76]. Pix2Seq v2 [11] unifies object detection, instance segmentation, keypoint detection, and image captioning by quantizing the continuous image coordinates for the first three tasks. Unified IO [46] further unifies dense structure outputs such as images, segmentation masks, and depth maps using a vector quantization variational auto-encoder (VQ-VAE) [70].

As quantization inevitably introduces information loss during discretization, another direction of unification aims to explore the image itself as a natural interface for vision generalists [5, 78]. Painter [78] formulates the dense prediction task as a masked image inpainting problem and demonstrates in-context capability in vision tasks such as depth estimation, semantic segmentation, instance segmentation, keypoint detection, and image restoration. Recently, PromptDiffusion [79] also exploits in-context visual learning with a text-guided diffusion model [60] and integrates the learning of six different tasks, i.e., image-to-depth, image-to-HED, image-to-segmentation and vice versa. Our work also examines image-resembling generation. However, in contrast to in-context learning, Unlike previous works [78, 79] that also explore natural language instructions, our method introduces a more favorable instruction alignment compared to the implicit task intention deducted

from in-context learning. Moreover, with such explicit instructions, we further unify semantic image editing tasks, which are crucial use cases in image-resembling generation.

### 3. Method

We present InstructDiffusion, a novel generalist modeling interface designed for a diverse range of vision tasks. By leveraging the Denoising Diffusion Probabilistic Model (DDPM), we treat all computer vision tasks as humanintuitive image manipulation processes with outputs in a flexible and interactive pixel space. Several existing multimodal models, such as Flamingo [2] and BLIP2 [34], inherently produce natural language as their target output, thereby restricting their capabilities to visual question answering and image captioning. In contrast, our approach posits that formulating various vision tasks, including segmentation, keypoint detection, and image synthesis as image-resembling generation processes, is more intuitive, straightforward, and readily assessable for human evaluation.

Our primary focus is on three output formats: 3-channel RGB images, binary masks, and key points. These outputs adequately encompass a wide range of vision tasks, including keypoint detection, semantic segmentation, referring segmentation, semantic image editing, and several image enhancement tasks such as deblurring, denoising, and watermark removal. We first discuss the essential instructional format design for the vision tasks currently covered in Section 3.1, followed by an in-depth explanation of the training data preparation to ensure optimal model performance in Section 3.2. Lastly, we describe a unified framework with a simple architecture in Section 3.3.

#### 3.1. Unified Instructional for Vision Tasks

The unified modeling interface for all tasks is referred to as Instructional Image Editing. By denoting the training set as {xi}, each training data xi can be represented in the form of {ci,si,ti}, where ci signifies the control instruction, while si and ti represent the source and target images, respectively. Within this context, our method aims to generate a target image ti that adheres to the given instruction ci when provided with an input source image si.

In the context of semantic image editing tasks, InstructPix2Pix [6] is a recent representative work that demonstrates a natural fit. For other vision tasks, the challenge involves creating appropriate instructions and subsequently establishing a corresponding target image. Although natural language instruction has been utilized extensively in previous approaches, such as Pix2Seq [10] and UnifiedIO [46], we contend that terms like ”semantic segmentation” or ”keypoint detection” are better perceived as indicators rather than instructions. In contrast, our approach involves providing highly detailed instructions, enabling the

model to comprehend the instructions rather than merely model a fixed bias based on the indicator.

Keypoint detection. It endeavors to precisely locate key object components within an image, such as the left eye of a face, the right shoulder of an individual, or the nose of a dog. Traditionally, heatmap regression has served as the standard learning approach, where ground truth heatmaps are generated by overlaying 2D Gaussian kernels on all keypoints. In contrast, this work introduces a more natural and easily assessable output by providing extensively detailed instructions, thereby enhancing the overall process of keypoint detection in various applications. An exemplary instruction might be, ”Please use red to encircle the left shoulder of the man.” In this instance, the output image should exhibit a red circle at the corresponding location (i.e., the left shoulder of the man in the image), while the rest of the region remains unaltered. This innovative approach facilitates a more intuitive comprehension of the keypoint detection process while simultaneously refining the model’s capacity to understand the meaning of different object components.

Segmentation. For semantic and referring segmentation, the objective is to identify the region of a particular object within the input image. An illustrative example of this instruction would be ”apply a blue semi-transparent mask to the rightmost dog while maintaining the remainder unaltered.” Consequently, the resulting image is determined and features a blue mask on the appropriate dog. We require the mask to be semi-transparent instead of opaque, thereby facilitating the human evaluation of the predicted mask’s accuracy. Moreover, our experiments indicate that the semi-transparent mask also augments the segmentation performance.

Image enhancement and image editing. Image enhancement such as deblurring, denoising, and watermark removal inherently yields output images, and the same applies to image editing. Consequently, we only need to construct instructions which shall clearly specify the operation to be performed. Detailed examples include “Make the image much sharper” for image deblurring, “Please remove the watermark on the image” for watermark removal, and “add an apple in the woman’s hand” for image editing.

To enhance the diversity of instructions, we first manually write 10 instructions for each task. Then we use GPT4 to rewrite and expand the diversity of these instructions, thereby mimicking user input to the system. Subsequently, one instruction is chosen at random during the training process. This approach, which incorporates diverse and intuitive instructions, has been observed to substantially augment the model’s multi-task fusion capabilities.

#### 3.2. Training Data Construction

As a proof-of-concept, we focus on investigating whether different tasks benefit each other under such imageresembling unification, instead of scaling data as much as possible for optimal performance at the extreme limits. We adopt widely used publicly available datasets and construct the ground truth target image according to the instruction template. For example, we use COCO-Stuff [8] for semantic segmentation and use COCO [41], MPII [3], CrowPose [37] and AIC [83] for keypoint detection. More details will be presented in Sec 4.1.

For image editing, InstructPix2Pix (IP2P) [6] pioneered the use of a synthetic training dataset by leveraging GPT3 [7] for generating instructions and Prompt2Prompt [21] for creating output images. However, the synthesized source and target images exhibit varying quality and nonnegligible artifacts, with most instructions focusing on global style modifications rather than local alterations. Furthermore, MagicBrush [96] introduced a dataset comprising over 10,000 manually annotated triples, but its size is limited when compared to other vision tasks. Consequently, in addition to existing datasets such as IP2P [6], GIER [63], GQA [90], and MagicBrush [96], we propose a novel dataset called Image Editing in the Wild (IEIW), which encompasses 159,000 image editing pairs that cover a wide range of semantic entities and diverse levels of semantic granularity. To expand the scope of image editing data, we assemble the IEIW dataset by drawing from the following three distinct resources:

Object removal. Object removal is a very common type of image editing. Inspired by Inst-Inpaint [90], we use the referring segmentation dataset PhraseCut [82] to construct the instructional object removal data. PhraseCut offers images with referring phrases for corresponding regions. We set these regions as a mask and use LAMA [67] to inpaint them, transforming them into instructional inpainting datasets. Notably, we also swap input and output images, and reverse the instructions like ”remove the blue bird on top of the tree” to ”add a blue bird on top of the tree” to further supplement data from the perspective of adding components.

Object replacement. We propose a data construction pipeline for generating training data that targets the scenario of substituting certain specific objects, which is another essential feature for image editing. To automatically generate training triplets, we rely on SA-1B [31] and OpenImages [32] datasets, which provide multiple regions in an image with semantic meaning. Specifically, we first build a gallery database consisting of diverse image patches based on those semantic-aware regions. Given a source image from OpenImages or SA-1B, we randomly select a semantic region, which is used as a query patch to retrieve its nearest neighbors from the aforementioned constructed gallery

plex image distributions. As illustrated in Figure 2, our training procedure comprises three stages: pretraining adaptation, task-specific training, and instruction tuning.

##### Step 1 Step 2 Step 3 Step 4

|A running tiger| |
|---|---|
| | |

|A running tiger with yellow and blue circles| |
|---|---|
| | |

|Use the color yellow to encircle the nose of the tiger| |
|---|---|
| | |

|Use the color yellow to encircle the nose of the tiger| |
|---|---|
| | |

Pretraining adaptation. Stable Diffusion (SD) [60] is recognized as one of the most robust open-source text-to-image models currently accessible, prompting our decision to utilize Stable Diffusion v1.5 as the foundation for our work. Initially, stable diffusion operates as a mapping mechanism that converts textual captions into natural images. However, our desired images might encompass segmentation masks or keypoint indicators, which substantially deviate from typical natural images. Consequently, our preliminary phase involves fine-tuning the stable diffusion model and adjusting the diffusion output distribution.

Text-to-image Model

Pretraining Adaptation

Task-specific Training

Human Alignment

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

- Figure 2. Training pipeline of our method. To illustrate concisely, we take keypoint detection as an example.

database. The retrieved similar patches are regarded as reference images to the source image, both of which are fed to PaintByExample [87] for generating a target image. In this way, we obtain the source image as well as the modified target image. To produce instruction, we utilize an image captioning tool, such as BLIP2 [34], to yield the source caption as well as the target caption, and then generate a possible instruction through a large language model. For example, given the captions “a running dog” and “a cute cat with black and white stripes”, a possible instruction is “please change the running dog to a cute cat with black and white stripes”. We can generate quite an amount of paired data for training using this construction pipeline.

Since we require diffusion models to be capable of generating images “with a foreground mask” or “with some special mark”, we employ existing segmentation or keypoint detection datasets to produce such data. The remaining challenge lies in the development of suitable captions that accurately depict these images while maintaining the intrinsic text-to-image generation capability. This is achieved by augmenting the original image caption with a suffix, such as ”with a few different color patches here and there” or ”surrounded with a red circle.” By fine-tuning the diffusion model with these modified image captions, we can theoretically empower the model to generate any images within the desired output domain.

Web crawl. In order to achieve greater alignment with authentic user needs and enhance the overall user experience, we gather genuine user requests along with the corresponding outcomes delivered by seasoned Photoshop professionals sourced from the website. To ensure the accuracy and relevance of the data, we search in Google by utilizing the keyword ”photoshop request”. This approach enables us to amass a substantial dataset comprising over 23,000 data triplets, which further aids in refining our understanding of user requirements and reduces the domain gap between training and inference.

Task-specific training. In the second stage, our goal is to further fine-tune the diffusion model, enhancing its comprehension of various instructions for different tasks. We follow InstructPix2Pix [6] and inject source images by concatenating them with the noise input, subsequently expanding the input channels of the first layer. We train our model using all data containing various tasks. Since the amount of data for each task is quite different, in order to maintain a balance, we manually set different sampling weights for different databases. The number of effective training samples used for different tasks is shown in Table 1. For a data triplet si,ci,ti, the diffusion process adds noise to the encoded latent z = E(ti) producing a noisy latent zt. We fine-tune the diffusion network ϵθ by minimizing the following latent diffusion objective:

In order to guarantee the quality of the training data, we further utilize image quality assessment tools to eliminate substandard data. Specifically, we apply Aesthetics Score and GIQA [18] as image quality evaluation metrics, specifically utilizing LAION-Aesthetics-Predictor [61] for Aesthetics Score and constructing a KNN-GIQA model on LAION-600M [61] images for calculating GIQA scores. We exclude two categories of data: i) target images with low-quality scores, and ii) a significant discrepancy in quality scores between the source image and its corresponding target image. Our findings indicate that this data-filtering process is of vital importance.

i,ci,ti)∼P(x),ϵ∼N(0,1),t ∥ϵ−ϵθ(zt,t,si,ti)∥22 (1)

L = E(s

Human alignment. To further improve the quality of editing, we have followed the idea of instruction tuning [81] from Large Language Models. In LLM literature, instruction tuning [81] is used to teach the model to solve a task following the instruction. However, we conduct instruction tuning differently from that in LLM. For each sample in the benchmark, we generate different editing results using 20 different sampling classifier-free guidance [23]. Then, we

#### 3.3. Unified Framework

Our framework is based on diffusion, as diffusion models have experienced significant success in modeling com-

Table 1. The number of effective training samples used for different tasks.

|Task<br><br>|Keypoint Detection Segmentation Image Enhancement Image Editing|
|---|---|
|# Effective training samples|245k 239k 46k 425k|

ask subjects to select the best 0-2 edited images to formulate the instruction-tuning dataset. The whole dataset contains 1,000 images. We use this dataset to further fine-tune our model for about 10 epochs.

### 4. Experiments 4.1. Settings

Training samples. Our model is trained on samples consisting of {instruction, source image, target image}, encompassing the aforementioned vision tasks, i.e., keypoint detection, semantic segmentation, referring segmentation, image enhancement including denoising, deblurring and watermark removal, and image editing. Specifically for keypoint detection, we adopt four classical datasets, namely COCO [41] containing 149K images with each labeled 17 keypoints, CrowdPose [37] consisting of 35K images each with 14 keypoints, MPII [3] with 22K images labeled with 16 keypoints, and AIC [83] including 378K images annotated with 14 keypoints. Throughout our training process, for each image, we employ a random selection of between 1 and 5 keypoints, and assign these keypoints with random colors. Accordingly, the instruction is produced through templates filled with the class of keypoints and the specific color, and the target image is generated by positioning small circles on the chosen keypoints, each circle taking on the color corresponding to its respective keypoint. For segmentation, we select COCO-Stuff [8] as semantic segmentation training dataset while gRefCOCO [42] and RefCOCO [93] as referring segmentation training dataset. We collect a series of prompt templates with the help of large language models to serve as text instructions. An example is “place a color mask on object.” During training, we randomly select a color for “color” and replace “object” with the corresponding category name in semantic segmentation or referring in referring segmentation. The target image is generated by placing a mask using its corresponding color with a transparency of 0.5 over the object. For image enhancement, we focus on three tasks: deblurring, denoising, and watermark removal. For these tasks, we utilize the GoPro [51] containing 2103 images and REDS [50] dataset with 24,000 images for deblurring, the SIDD [1] dataset composed of 320 images for denoising, and the CLWD [44] dataset containing 60,000 images for watermark removal. Lastly for image editing, as mentioned in Sec. 3.2, we adopt 7 editing datasets, including filtered InstructPix2Pix [6] dataset containing 561K samples, 8K samples in MagicBrush [96] training dataset, GIER [63]

Table 2. Average precision comparison on the COCO val2017, HumanArt and AP-10K datasets. We evaluate the official large models of the competitors to ensure fairness. The ground truth bounding boxes are used for all results. The best-performing generalist models are highlighted in bold.

Method COCO val HumanArt AP-10K

Specialized Models

PCT [16] 80.2 63.7 14.6 ViTPose [86] 82.0 64.1 14.7

Generalist Models

Unified-IO [46] 25.0 15.7 7.6 Painter [78] 70.2 12.4 15.3 Ours 71.2 51.4 15.9

with 5K samples, GQA [90] inpainting dataset with 131K samples, VGPhraseCut [82] composed of 85K samples, our generated dataset with 51K produced samples, and an internal dataset representing real editing scenario, which contains 23K training triplets. To ensure balanced sampling across each task, we implemented distinct sampling weights due to the considerable variance in the number of training images across different datasets. Table 1 illustrates the number of effective training samples we used in our framework. Implementation details. We utilize Stable Diffusion [60] v1.5 as initialization to leverage a text-to-image generation prior. The input image resolution is preprocessed to 256 × 256, and the learning rate is fixed to 1 × 10−4 during training. In addition, we adopt an EMA rate of 0.9999 to stabilize the training. Our model is trained using a batch size of 3072 for a total of 200 epochs, which requires approximately 4 days of computation on 48 NVIDIA V100 GPUs. Once trained, our model is readily applicable and can be directly used for different vision tasks. For each task, we provide a comprehensive comparison and an in-depth analysis of its performance in the subsequent sections. During the human alignment stage, we use an EMA rate of 0.99 to help the model quickly adapt to the instruction-tuning dataset.

#### 4.2. Keypoint Detection

We evaluate our model on both the close-set scenario using the COCO validation dataset as well as the open-set generalization capability over the unseen dataset: HumanArt dataset [27] and AP-10K animal dataset [91]. The HumanArt dataset is an artificial human dataset comprised of various forms such as cartoons, shadow plays, and murals, exhibiting a distinct data distribution compared to COCO

- (a) (b) (c) (d)

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Figure 3. The keypoint detection results generated by our model. The instructions are as follows: (a) Mark the car logo with a blue circle.

- (b) Put a blue circle on the nose of the white tiger and use the red color to draw a circle around the left shoulder of the white tiger. (c) Create a yellow circle around the right eye of the whale. (d) Use the color blue to encircle the right wrist of the person on the far left and draw a yellow circle over the left wrist of the person on the far right.

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

(a) (b) (c) (d)

- Figure 4. The segmentation results generated by our model. The instructions are as follows: (a) Mark the pixels of cat in the mirror to blue and leave the rest unchanged. (b) Fill in the pixels of neutrophil with yellow, retaining the existing colors of the remaining pixels. (c) Modify the pixels of Oriental Pearl Tower to red without affecting any other pixels. (d) Paint the pixels of shadow in blue and maintain the current appearance of the other pixels.

dataset. The AP-10K Animal dataset is a collection of annotated animal keypoints, which effectively highlights the ability of our model to handle animal keypoints despite being trained only on human keypoint datasets. To enable a more detailed and thorough evaluation, it is essential to extract accurate pose coordinate information, namely precise horizontal and vertical coordinates, rather than simply marking the location with a distinct symbol. To achieve this, we employ a lightweight U-Net structure that postprocesses the output image to generate a multi-channel heatmap. We employ the standard AP (average precision) based on the OKS as our evaluation metrics. Additionally, we utilize the ground truth bounding boxes for all results. Notably, for the AP-10K animal dataset, in order to facilitate comparison with other methods, the OKS is calculated exclusively on the keypoints that overlap with the COCO annotated joints. However, it should be noted that our model possesses the capability to detect keypoints beyond the confines of the training dataset.

The results of the keypoint detection are presented in

- Table 2. Our approach outperforms other generalist models, Unified-IO [46] and Painter [78], across all evaluated datasets. Particularly we demonstrate a significantly higher level of performance over HumanArt and AP-10K, indicating the powerful generalization ability of our framework. In comparison to methods specifically designed for keypoint detection, our unified model does not exceed their performance due to localization accuracy limitations. However, it showcases exceptional performance on the entirely un-

seen animal keypoints dataset, AP-10K. Figure 3 (a-c) display our results for car and animals keypoint detection. Our model can accurately detect the logo of the car and the keypoints of animals that have never appeared in the keypoint detection training dataset. Figure 3 (d) demonstrate our capability for referring keypoint detection, showcasing our versatile detection abilities.

#### 4.3. Segmentation

Our primary focus lies in assessing the open-vocabulary capability of our model, particularly when evaluating images that contain unseen classes not present during the training phase. Therefore, besides the COCO-stuff [8], gRefCOCO [42] and RefCOCO [93] datasets, we conduct evaluation over additional eight datasets, i.e., RefCOCO+ [93], G-Ref [47], RefClef [30] for referring segmentation, and ADE20K-150 [98], ADE20K-847 [98], Pascal Context59 [49], Pascal Context-459 [49], Pascal VOC [14] for semantic segmentation. Similar to keypoint detection, we employ a lightweight U-Net structure that post-processes the output image to extract the binary mask of each individual object. Adhering to the prevailing convention [42], we adopt cumulative IoU (cIoU) to measure the performance for referring segmentation. On the other hand, our approach involves predicting a mask for each semantic category individually. As a result, semantic segmentation can also be perceived as referring to segmentation based on semantic categories. Thus, we choose to utilize the mean of classwise cumulative intersection over union (mcIoU) to quan-

- Table 3. Quantitative results on referring segmentation in terms of cIoU. U: UMD split. G: Google split. The best-performing generalist models are highlighted in bold.

Method

gRefCOCO RefCOCO RefCOCO+ G-Ref RefClef

val val test A test B val test A test B val(U) test(U) val(G) val testA testB testC

Specialized Models

LAVT [89] 57.64 72.73 75.82 68.79 56.86 62.29 48.14 58.65 59.17 61.16 21.22 44.77 24.78 47.08 ReLA [42] 62.42 73.21 75.24 68.72 56.10 62.26 47.89 59.71 60.40 61.37 20.68 43.08 22.57 45.94

Generalist Models

Unified-IO [46] 17.31 46.42 46.06 48.05 40.50 42.17 40.15 48.74 49.13 44.30 40.13 43.33 48.07 32.47 Ours 67.36 61.74 65.20 60.17 46.57 52.32 39.04 51.17 52.02 52.18 49.58 54.73 54.82 40.34

- Table 4. Quantitative results on semantic segmentation in terms of mcIoU. The best-performing generalist models are highlighted in bold.

Method ADE-847 PC-459 ADE-150 PC-59 VOC COCO-Stuff

Specialized Models

SimSeg [40] 10.43 13.98 25.89 53.55 39.27 40.26 OvSeg [85] 13.85 22.72 36.50 60.93 38.50 48.76 SAN [84] 18.84 33.32 38.79 63.31 46.14 50.15

Generalist Models

Painter [78] 5.00 8.68 54.50 33.67 4.67 11.91 PromptDiffusion [79] 0.99 2.19 8.97 13.07 11.69 2.71 Unified-IO [46] 8.96 13.69 15.65 27.21 31.46 22.52 Ours 19.68 28.29 33.62 59.00 72.55 53.17

tify the performance of semantic segmentation.

Table 3 reports the results for referring segmentation. To the best of our knowledge, Unified-IO [11] stands as the sole generalist model with the capability to perform referring segmentation. It can be seen that our model largely outperforms Unified-IO across almost all datasets. We also present methods that are specially designed for referring segmentation. Interestingly, our approach achieves an unexpectedly significant improvement over the RefClef dataset. Table 4 presents the quantitative comparison results of semantic segmentation. Both specialized models as well as our model have undergone training exclusively on the COCO-Stuff dataset. It is evident that our model not only surpasses specialized models in the close-set scenario, specifically the COCO-Stuff dataset, but also achieves comparable performance across other datasets that represent open-set scenarios. Notably, in the case of the VOC dataset, we observe a substantial improvement. When compared to generalist models, our approach outperforms other competitors by a considerable margin, except in the case of Painter on the ADE-150K dataset. This is largely attributable to Painter being specifically trained on this dataset. Interestingly, we notice that both Painter [78] and PromptDiffusion [79] lack awareness of the colors associated with unseen categories during evaluations in open-set scenarios. This is due to their reliance on example images to instruct the model regarding the color corresponding to each seman-

tic. In contrast, our approach establishes the color corresponding to each semantic category through text instructions, resulting in significantly superior performance. Figure 4 illustrates several visual examples for referring segmentation to demonstrate our model’s capability.

#### 4.4. Image Enhancement

We evaluate the low-level vision performance of our model using the widely employed benchmarks, i.e., GoPro [51], SIDD [1] and CLWD [44] dataset respectively for deblurring, denoising, and watermark removal task. The standard PSNR metric is adopted to measure the difference between the output processed image and the ground truth image. We evaluate our model’s deblurring capability on the GoPro benchmark with 1280×720 resolution, while for SIDD and CLWD, evaluations are conducted under 256×256 resolution to align with other works. The numerical comparison is reported in Table 5. We have made the following observations. Firstly, specialized models trained for image editing tasks tend to exhibit poor generalization when applied to image enhancement tasks. Secondly, the generalist model Painter [78] performs better in the denoising task but encounters challenges when it comes to seamlessly integrating image editing tasks through in-context learning. Lastly, the performance of our model in image enhancement is constrained by the VAE model, which introduces information loss. We have conducted an experi-

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Sharpen this blurry image Purify this photo by removing noise

Improve the focus of this hazy image

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Remove watermark from this picture

Erase the watermark from this photograph

- Figure 5. InstructDiffusion is also applicable to low-level vision tasks, including image deblurring, denoising, and watermark removal.

- Table 5. Quantitative results on image editing and image enhancement. For editing tasks (Replace, remove, and add), the results are CLIPSim/AP score. For enhancement tasks, the number reflects the PSNR metric. The numbers in parentheses indicate the results obtained by reconstructing the ground truth images using VAE, representing the performance upper bound achievable with the used VAE model.

| |Editing (CLIP-Sim↑ / AP Score↑)<br><br>|LowLevel (PSNR↑)|
|---|---|---|
|Method|Replace Remove Add<br><br>|Deblur Denoise Watermark remove|

Specialized Models

NAFNet [9] - - - 33.71 40.30 WDNet [44] - - - - - 40.24 Null-text [48] 30.71/5.04 29.69/4.80 30.14/4.95 24.52 23.29 18.31 InstructPix2Pix [6] 29.61/4.99 28.82/4.69 30.11/4.94 22.71 15.14 14.96 MagicBrush [96] 30.50/4.94 29.07/4.67 30.69/4.90 22.64 16.10 15.46 EDICT [72] 29.91/4.91 29.33/4.80 30.19/4.93 24.16 24.48 19.88

Generalist Models

Painter [78] - - - - 38.66 Ours 30.19/4.90 28.88/4.65 30.39/4.87 23.58 (29.54) 34.26 (36.56) 23.60 (24.80)

ment by feeding the ground truth image to the VAE model and calculating the PSNR for the output, which serves as an upper bound for our model and is indicated in parentheses.

We also present some “in-the-wild” visual results in Figure 5 to qualitatively show our model’s real-world applicability in low-level vision tasks. We can observe that the resulting images have been effectively processed in line with the provided instruction, which includes sharpening, denoising, and watermark removal.

#### 4.5. Image Editing

To better demonstrate the editing quality of our method, we build a benchmark containing 1,000 samples. Each sample contains the source image, the caption of the source image provided by BLIP2 [34], the editing instruction, and the target caption of the edited image. We manually classify each sample into one of three distinct editing scenarios: replacement, removal, and addition. This meticulous categorization aims to provide a nuanced reflection of the

model’s editing capabilities. We adopt two commonly used metrics, CLIP similarity (CLIP-Sim) and Aesthetic Predictor’s Score (AP) [61], to evaluate the editing results. CLIPSim measures the semantic similarity between an image and a text. We utilize BLIP2 [34] to obtain the caption of the input image and invoke GPT-3.5-turbo to acquire the target caption of the edited image. The CLIP-Sim score is calculated between the edited image and the target caption. The AP score assesses the aesthetic quality of the generated images, a methodology akin to LAION-5B, which employs the CLIP+MLP Aesthetic Score Predictor. A higher quality score reflects better perceptual quality.

We report the numerical results in Table 5. It is important to emphasize that none of the existing generalist models have the capability to perform image editing. Compared with specific models, it is evident from the table that even with joint training, our model achieves superior CLIP-Sim compared to Instruct-Pix2Pix [6] and on-par results with MagicBrush [96]. When assessing the editing task, it is

Input InstructPix2Pix MagicBrush EDICT Null-text Inversion Ours

[Figure 32]

Remove all magnets and stickers

[Figure 33]

Replace the truck with a train

[Figure 34]

Make the cat look like Lil Wayne

- Figure 6. Comparison between different instruction guided image editing. From left to right: input, Prompt-to-prompt [21], MagicBrush [96], EDICT [72], Null-text Inversion [48], and our results.

crucial to take into account not only semantic similarity and aesthetic quality but also factors such as background consistency and manipulation accuracy. Quantitative results may be somewhat constrained for comparison. For instance, a model that fails to make substantial edits and produces an image that remains almost unchanged could potentially receive a higher score than models that have successfully carried out meaningful edits.

We further illustrate several visual examples in Figure 6 compared with competitive baseline methods that have been shown impressive editing quality, including InstructPix2Pix [6], MagicBrush [96], EDICT [72] and Null-text Inversion [48]. It is evident that our model effectively edits the image in line with the given instructions. For instance, by following the directives, our model can successfully eliminate magnets and stickers, convert a truck into a train, and transform a cat into a particular style. We showcase additional editing results of our model in Figure 7, further highlighting the remarkably precise editing quality achieved. Given a source image, our model is capable of successfully adding, removing, and replacing elements. The image undergoes precise editing based on the prompt while maintaining the integrity of the background and preserving intricate details.

#### 4.6. The Benefit of Highly Detailed Instruction

Our hypothesis is that the ability to generalize is the skill of learning through understanding the specific meanings of individual elements rather than memorizing entire instructions. Unlike previous unified models like Pix2seq [10] and Unified-IO [46], which simply treat natural language as task indicators, our approach employs detailed descriptions for each task as instructions. Such detailed instructions enable the model to understand comprehensively and then prioritize accurate execution instead of simple instructions that favor mimicking. To show this, we try to replace our instructions within our framework with simpler task indicators, such as ”semantic segmentation” and ”keypoint detection,” while assigning fixed colors to each keypoint or object class. As demonstrated in Table 6, the results of simple instructions are extremely poor, especially when handling new types of keypoints or novel object categories. This highlights that our detailed instructions provide enhanced flexibility and adaptability in the open domain.

#### 4.7. The Benefit of Multi-task Training

Multi-task learning has grown increasingly popular, enabling models to achieve greater generalization by concurrently addressing multiple related tasks during training.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Input Add a pair of elegant earrings for her Remove the red tie Make her hair black Add sunglasses Replace her clothes with suit

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Input Remove the cat Remove the dog Help the cat and dog wear baseball caps Van gogh, sunflowers Replace the cat with little dog

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Input Make it a cute pig Make it a cute cat Add a hat Add beard on the hedgehog Add glasses

Figure 7. Image editing results generated by our model.

- Table 6. Ablation study on the instruction. The term ”Simple Instruction” denotes coarse-grained instructions like ”semantic segmentation” or ”keypoint detection”. In contrast, our approach utilizes highly detailed and more flexible instructions. The training datasets include COCO for keypoint detection and COCO-Stuff for semantic segmentation.

|Method| |Keypoint Detection<br><br>| |Semantic Segmentation| |
|---|---|---|---|---|---|
| | |COCO|HumanArt AP-10K|COCO-Stuff<br><br>|ADE-847 PC-459 ADE-150 PC-59 VOC|
|Simple Instruction Ours| |22.7 71.2<br><br>|7.0 5.2 51.4 15.9<br><br>|41.28 53.17|1.39 3.96 13.65 18.49 20.22 19.68 28.29 33.62 59.00 72.55<br><br>|

This approach often results in improved model generalization performance compared to specialized single-task training. To further provide empirical validation for this observation, we experiment with our model when trained only on the segmentation dataset and report the performance difference in Figure 8. This illustration compares the results of our single-task model and multi-task model evaluated over four unseen test datasets. It is evident that our jointly trained model performs significantly better in open-domain testing scenarios compared to the specific models. Furthermore, we observe that this benefit also extends to image editing. In Figure 9, the visual comparison demonstrates that when integrated with other tasks, the model can more effectively discern which objects require editing, potentially benefiting from the integration of referring segmentation.

55

single-task training

49.6

multi-task training

46

EvaluationMetrics

37

33.6

29.7

28.3

28

24.4

23.1

19.7

19

15.1

10

RefClef ADE-847 PC-459 ADE-150 Validation Dataset

Figure 8. Ablation study on multi-task training. We evaluate our models on four unseen datasets, RefClef, ADE-847, PC-459, and ADE-150. It demonstrates that joint training significantly enhances the capability to handle open-set scenarios.

window

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

dolphin

flower

vase

girl

Input Multi-task Single-task

book laptop

- Figure 9. Multi-task training’s effect on editing. When given the instruction “put a hat on the leopard”, joint training with other tasks can lead to more precise editing outcomes in comparison to single-task training.

|Alignment tuning<br><br>Baseline|
|---|

0 2 4 6 8 10 12 14 Tuning Epochs

29.6

29.7

29.8

29.9

CLIP-Sim

- Figure 10. Effect of human alignment. Further fine-tuning on selected human alignment data enhances the CLIP-Sim metric, reaching its peak after approximately 10 epochs.

[Figure 58]

[Figure 59]

Figure 11. Adopt InstructDiffusion to unseen tasks, including detection, classification, and face alignment.

our approach involves directly instructing our model to encircle the specific facial region of interest, such as the nose or right ear. Remarkably, we have found that this approach performs admirably even when applied to animal faces. We argue that this underscores its versatility in adapting to new challenges beyond its initial training scope.

#### 4.8. The Benefit of Human Alignment

Our model undergoes a subsequent fine-tuning phase using a filtered dataset with human alignment. In this evaluation, we examine its effectiveness and present the finetuning progress in Figure 10, which showcases the relationship between CLIP-Sim performance and the number of epochs. Initially, the CLIP-Sim score stands at 29.6. Remarkably, we observe a noticeable enhancement in imagetext alignment, which increases from 29.6 to 29.9 over approximately 10 epochs. It is important to highlight the significance of this improvement, particularly considering that the dataset consists of only 1000 samples.

#### 4.9. Generalization Capability to Unseen Tasks

We demonstrate that our model exhibits a degree of Artificial General Intelligence (AGI) capabilities by leveraging the wealth of tasks and diverse datasets through this highly detailed instruction-following format. We validate its capacity to handle tasks that were not part of its training repertoire, including image detection, classification, and even intricate fine-grained tasks like face alignment in Figure 11. In the context of detection and classification, we employ a prompt that resembles referring segmentation, enabling us to derive the bounding box coordinates by identifying the top, bottom, left, and right boundaries of the marked region. Moreover, we can also verify the class label using a versatile prompt structure. In the realm of face alignment,

### 5. Discussion and conclusion

In conclusion, this paper presents InstructDiffusion, a novel and unifying framework for aligning computer vision tasks with human instructions. InstructDiffusion treats all computer vision tasks as image generation, with a focus on three types of output formats: 3-channel RGB images, binary masks, and keypoints. We demonstrated that our approach achieves good performance in individual tasks, and joint training of multiple tasks enhances the generalization ability. Remarkably, InstructDiffusion exhibits AGI capabilities to some extent, handling tasks not seen during training and outperforming previous methods on unseen datasets. This research marks a significant step towards a generalist modeling interface for vision tasks and sets the stage for future advancements in the pursuit of artificial general intelligence in computer vision.

In future work, we plan to focus on the following aspects to further improve the performance and capabilities of InstructDiffusion: 1) Improve the unified representation: We aim to explore alternative encoding schemes and techniques to better represent a more diverse range of outputs associated with various computer vision tasks. 2) Investigate the role of self-supervised and unsupervised learning: To enhance the generalization ability of InstructDiffusion, we will explore the use of self-supervised and unsupervised learning techniques to leverage large-scale unlabeled data for model training and adaptation.

### References

- [1] Abdelrahman Abdelhamed, Stephen Lin, and Michael S. Brown. A high-quality denoising dataset for smartphone cameras. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. 6, 8
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736,

2022. 3

- [3] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2d human pose estimation: New benchmark and state of the art analysis. In Proceedings of the IEEE Conference on computer Vision and Pattern Recognition, pages 3686–3693, 2014. 4, 6
- [4] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. 2
- [5] Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei Efros. Visual prompting via image inpainting. Advances in Neural Information Processing Systems, 35:25005–25017, 2022. 2, 3
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 3, 4, 5, 6, 9, 10
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1, 2, 4
- [8] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In Computer vision and pattern recognition (CVPR), 2018 IEEE conference on. IEEE, 2018. 4, 6, 7
- [9] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun. Simple baselines for image restoration. arXiv preprint arXiv:2204.04676, 2022. 9
- [10] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. arXiv preprint arXiv:2109.10852, 2021. 1, 3, 10
- [11] Ting Chen, Saurabh Saxena, Lala Li, Tsung-Yi Lin, David J Fleet, and Geoffrey E Hinton. A unified sequence interface for vision tasks. Advances in Neural Information Processing Systems, 35:31333–31346, 2022. 1, 3, 8
- [12] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022. 2
- [13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional

- transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 1, 2
- [14] Mark Everingham and John Winn. The pascal visual object classes challenge 2012 (voc2012) development kit. Pattern Anal. Stat. Model. Comput. Learn., Tech. Rep, 2007(1-45):5,

2012. 7

- [15] Andrea Frome, Greg S Corrado, Jon Shlens, Samy Bengio, Jeff Dean, Marc’Aurelio Ranzato, and Tomas Mikolov. Devise: A deep visual-semantic embedding model. Advances in neural information processing systems, 26, 2013. 2
- [16] Zigang Geng, Chunyu Wang, Yixuan Wei, Ze Liu, Houqiang Li, and Han Hu. Human pose as compositional tokens. In CVPR, 2023. 6
- [17] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2
- [18] Shuyang Gu, Jianmin Bao, Dong Chen, and Fang Wen. Giqa: Generated image quality assessment. In European conference on computer vision, pages 369–385. Springer, 2020. 5
- [19] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10696–10706, 2022. 2
- [20] Tanmay Gupta, Amita Kamath, Aniruddha Kembhavi, and Derek Hoiem. Towards general purpose vision systems: An end-to-end task-agnostic vision-language architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16399–16409, 2022. 3
- [21] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2022. 4, 10
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2
- [23] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [24] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045, 2023. 2
- [25] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021. 3
- [26] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR,

2021. 2

- [27] Xuan Ju, Ailing Zeng, Wang Jianan, Xu Qiang, and Zhang Lei. Human-art: A versatile human-centric dataset bridging

- natural and artificial scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 6
- [28] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2
- [29] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 2
- [30] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014. 7
- [31] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 4
- [32] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4. International Journal of Computer Vision, 128(7):1956–1981, 2020. 4
- [33] Hao Li, Jinguo Zhu, Xiaohu Jiang, Xizhou Zhu, Hongsheng Li, Chun Yuan, Xiaohua Wang, Yu Qiao, Xiaogang Wang, Wenhai Wang, et al. Uni-perceiver v2: A generalist model for large-scale vision and vision-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2691–2700, 2023. 3
- [34] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 3, 5, 9
- [35] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–

12900. PMLR, 2022. 3

- [36] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705, 2021. 2
- [37] Jiefeng Li, Can Wang, Hao Zhu, Yihuan Mao, Hao-Shu Fang, and Cewu Lu. Crowdpose: Efficient crowded scenes pose estimation and a new benchmark. In CVPR, 2019. 4, 6
- [38] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965–10975, 2022. 2
- [39] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu

- Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16, pages 121–137. Springer, 2020. 2
- [40] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7061–7070, 2023. 8
- [41] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 4, 6
- [42] Chang Liu, Henghui Ding, and Xudong Jiang. GRES: Generalized referring expression segmentation. In CVPR, 2023. 6, 7, 8
- [43] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 3

- [44] Yang Liu, Zhen Zhu, and Xiang Bai. Wdnet: Watermarkdecomposition network for visible watermark removal. In 2021 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2021. 6, 8, 9
- [45] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019. 2
- [46] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. arXiv preprint arXiv:2206.08916, 2022. 3, 6, 7, 8, 10
- [47] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016. 7
- [48] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 9, 10
- [49] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 891–898, 2014. 7
- [50] Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, and Kyoung Mu Lee. Ntire 2019 challenge on video deblurring and superresolution: Dataset and study. In CVPR Workshops, June

2019. 6

- [51] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In CVPR, July 2017. 6, 8

- [52] OpenAI. Gpt-4 technical report, 2023. 1
- [53] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022. 1, 2
- [54] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 3
- [55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 2
- [56] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 2
- [57] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 2
- [58] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019. 2
- [59] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al. A generalist agent. arXiv preprint arXiv:2205.06175, 2022. 3
- [60] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 3, 5, 6
- [61] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 5, 9
- [62] Jing Shao, Siyu Chen, Yangguang Li, Kun Wang, Zhenfei Yin, Yinan He, Jianing Teng, Qinghong Sun, Mengya Gao, Jihao Liu, et al. Intern: A new learning paradigm towards general vision. arXiv preprint arXiv:2111.08687, 2021. 2
- [63] Jing Shi, Ning Xu, Trung Bui, Franck Dernoncourt, Zheng Wen, and Chenliang Xu. A benchmark and baseline for language-driven image editing. In Proceedings of the Asian Conference on Computer Vision, 2020. 4, 6
- [64] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. Flava: A foundational language and vision alignment model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15638–15650, 2022. 2

- [65] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [66] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visuallinguistic representations. arXiv preprint arXiv:1908.08530,

2019. 2

- [67] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2149–2159, 2022. 4
- [68] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [69] Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34:200–212, 2021. 2
- [70] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2, 3
- [71] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2
- [72] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22532–22541, 2023. 9, 10
- [73] Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Luowei Zhou, Yucheng Zhao, Yujia Xie, Ce Liu, Yu-Gang Jiang, and Lu Yuan. Omnivl: One foundation model for image-language and video-language tasks. Advances in neural information processing systems, 35:5696–5710, 2022. 2
- [74] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100, 2022. 2, 3
- [75] Liwei Wang, Yin Li, and Svetlana Lazebnik. Learning deep structure-preserving image-text embeddings. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5005–5013, 2016. 2
- [76] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, pages 23318–23340. PMLR, 2022. 3
- [77] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a

- foreign language: Beit pretraining for all vision and visionlanguage tasks. arXiv preprint arXiv:2208.10442, 2022. 2
- [78] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6830–6839, 2023. 2, 3, 6, 7, 8, 9
- [79] Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, and Mingyuan Zhou. In-context learning unlocked for diffusion models. arXiv preprint arXiv:2305.01115, 2023. 3, 8
- [80] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904, 2021. 2
- [81] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2021. 5
- [82] Chenyun Wu, Zhe Lin, Scott Cohen, Trung Bui, and Subhransu Maji. Phrasecut: Language-based image segmentation in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10216–10225, 2020. 4, 6
- [83] Jiahong Wu, He Zheng, Bo Zhao, Yixin Li, Baoming Yan, Rui Liang, Wenjia Wang, Shipei Zhou, Guosen Lin, Yanwei Fu, et al. Ai challenger: a large-scale dataset for going deeper in image understanding. arXiv preprint arXiv:1711.06475, 2017. 4, 6
- [84] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for open-vocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2945– 2954, 2023. 8
- [85] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for openvocabulary semantic segmentation with pre-trained visionlanguage model. In European Conference on Computer Vision, pages 736–753. Springer, 2022. 8
- [86] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. Vitpose: Simple vision transformer baselines for human pose estimation, 2022. 6
- [87] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391,

2023. 5

- [88] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. Unified contrastive learning in image-text-label space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19163–19173, 2022. 2
- [89] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In CVPR,

2022. 8

- [90] Ahmet Burak Yildirim, Vedat Baday, Erkut Erdem, Aykut Erdem, and Aysegul Dundar. Inst-inpaint: Instructing to remove objects with diffusion models. arXiv preprint arXiv:2304.03246, 2023. 4, 6
- [91] Hang Yu, Yufei Xu, Jing Zhang, Wei Zhao, Ziyu Guan, and Dacheng Tao. Ap-10k: A benchmark for animal pose estimation in the wild. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. 6
- [92] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 3
- [93] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 6, 7
- [94] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021. 2
- [95] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Li, Xiyang Dai, Lijuan Wang, Lu Yuan, JenqNeng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and vision-language understanding. Advances in Neural Information Processing Systems, 35:36067–36080, 2022. 3
- [96] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. arXiv preprint arXiv:2306.10012,

2023. 4, 6, 9, 10

- [97] Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D Manning, and Curtis P Langlotz. Contrastive learning of medical visual representations from paired images and text. In Machine Learning for Healthcare Conference, pages 2–

25. PMLR, 2022. 2

- [98] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641,

2017. 7

- [99] Xizhou Zhu, Jinguo Zhu, Hao Li, Xiaoshi Wu, Hongsheng Li, Xiaohua Wang, and Jifeng Dai. Uni-perceiver: Pretraining unified architecture for generic perception for zeroshot and few-shot tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16804–16815, 2022. 3

