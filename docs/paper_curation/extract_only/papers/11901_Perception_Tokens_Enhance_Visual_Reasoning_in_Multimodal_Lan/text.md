# arXiv:2412.03548v2[cs.CV]8Dec2024

## Perception Tokens Enhance Visual Reasoning in Multimodal Language Models

### Mahtab Bigverdi1 Zelun Luo2 Cheng-Yu Hsieh1

Ethan Shen 1 Dongping Chen 1 Linda G. Shapiro1 Ranjay Krishna1 1University of Washington, 2Google Research {mahtab,cydhsieh,ethans03,shapiro,ranjay}@cs.washington.edu, alanzluo@gmail.com, cdp0612@uw.edu

### Abstract

Multimodal language models (MLMs) still face challenges in fundamental visual perception tasks where specialized models excel. Tasks requiring reasoning about 3D structures benefit from depth estimation, and reasoning about 2D object instances benefits from object detection. Yet, MLMs can not produce intermediate depth or boxes to reason over. Finetuning MLMs on relevant data doesn’t generalize well and outsourcing computation to specialized vision tools is too compute-intensive and memoryinefficient. To address this, we introduce Perception Tokens, intrinsic image representations designed to assist reasoning tasks where language is insufficient. Perception tokens act as auxiliary reasoning tokens, akin to chain-ofthought prompts in language models. For example, in a depth-related task, an MLM augmented with perception tokens can reason by generating a depth map as tokens, enabling it to solve the problem effectively. We propose AURORA, a training method that augments MLMs with perception tokens for improved reasoning over visual inputs. AURORA leverages a VQVAE to transform intermediate image representations, such as depth maps into a tokenized format and bounding box tokens, which is then used in a multi-task training framework. AURORA achieves notable improvements across counting benchmarks: +10.8% on BLINK, +11.3% on CVBench, and +8.3% on SEED-Bench, outperforming finetuning approaches in generalization across datasets. It also improves on relative depth: over +6% on BLINK. With perception tokens, AURORA expands the scope of MLMs beyond language-based reasoning, paving the way for more effective visual reasoning capabilities. Code will be released at the project page.

### 1. Introduction

In contrast to the growing emphasis on building multimodal language models (MLMs), computer vision was originally attempting to interpret images as projections of indescrib-

[Figure 1]

Figure 1. We introduce Perception Tokens, intermediate reasoning tokens that allow MLMs to go beyond using language in reasoning. With it, we develop AURORA, a framework that trains multimodal language models to leverage visual perception tokens, allowing them to use depth estimation and bounding box predictions while reasoning.

able 3D intrinsics, not just processing 2D arrays of language patterns [9, 36, 37]. Towards this endeavor, early vision research developed a series of intermediate image representations—enabling geometric reasoning through depth estimation [44] and instance reasoning through bounding box grounding [14]. As pointed out by recent work, we have focused less on such perceptual representations and instead tackled reasoning problems that require limited visual involvement [11, 42, 43]. This is likely because many traditional vision tasks remain ambiguous through natural language. Consider the task of identifying which of a set of N points is furthest away from the camera. While language doesn’t lend itself to reason over this problem, a depth estimation would provide the appropriate abstraction to reason over.

Numerous attempts have been made to enable MLMs to reason over intrinsic image representations. The default approach is to finetune MLMs on data tailored to the specific

[Figure 2]

- Figure 2. We demonstrate relative depth estimation and counting questions where LLaVA fails. In contrast, by learning to utilize visual perception tokens as intermediate reasoning steps, LLaVA-AURORA successfully complete these tasks requiring perceptual understanding.

perception task of interest, hoping that the model implicitly learns the required intrinsic representations [4]. Another option is to outsource the computation to external tools: the MLM can invoke a depth estimator or object detector to produce the appropriate intrinsic [17]. Unfortunately, relying on external models makes the task more computationally expensive and requires loading additional models with more memory. Similarly, vanilla fine-tuning (even with advancements like LoRA [16]) has shown marginal improvements.

Conceptually, we introduce Perception Tokens, intrinsic image representations that aid in reasoning where language is insufficient. To solve the aforementioned task, an MLM augmented with perception tokens can solve the task similar to how language models use chain-of-thought. They will produce a response like the following: “The depth map is <perception tokens>. Therefore, point D is closest to the camera.” Here, <perception tokens> is a set of tokens that implicitly estimate the depth of the image. Similarly, for a counting task, the model can first generate perception tokens that represents the location of the relevant bounding boxes of the desired object, and count the number of boxes to support its final answer.

To demonstrate the utility of perception tokens, we introduce AURORA, a training algorithm to augment MLMs with the ability to use perception tokens as intermediate reasoning steps. For certain intermediate representations (e.g. depth maps), we train a VQVAE to transform them into a set of tokens, treating the learned VQVAE codebook indices as a collection of perception tokens. while for others, such as bounding boxes, we use directly encoded structured tokens. Next, we follow a multi-task training approach [15, 18] to train MLMs to use perception tokens as chain-of-thought tokens (see Fig. 1). Additionally, we adopt a curriculum learning approach to avoid catastrophic forgetting.

We apply the AURORA training algorithm to the LLaVA [28] model, resulting in our LLaVA-AURORA variant. Our LLaVA-AURORA model significantly outperforms standard fine-tuning approaches across multiple perceptiondemanding tasks, demonstrating the generality and effec-

tiveness of our method. LLaVA-AURORA achieves stateof-the-art results on both relative depth estimation and object counting tasks. For instance, on BLINK relative depth estimation, LLaVA-AURORA delivers a performance boost of 6.4% points compared to the fine-tuning baseline. Similarly, on counting tasks, LLaVA-AURORA drives improvements of 10.8% points on BLINK, 11.3% points on CVBench, and 8.3% on SEED-Bench. Fig. 2 illustrates examples from these tasks. Perception tokens open up a whole new modality through which MLMs can begin to reason, tackling tasks beyond just language reasoning.

### 2. Related work

Multimodal language models (MLMs). MLMs aim to solve a variety of tasks (e.g., Visual Question Answering (VQA) and captioning) based on vision and language inputs. Most modern architectures accomplish this by relying on either cross-attention [1, 2] or visual instruction tuning [26, 28, 30, 53] to interleave multimodal information. Cross-attention architectures operate by independently encoding images and then cross-attending to a language model backbone. On the other hand, visual instruction tuning produces token embeddings from image representations that can be interleaved with language tokens to ground generation. While these techniques work well for high level tasks, MLMs still struggle with mid-level and low-level tasks such as counting, depth reasoning, and segmentation. Most MLMs can be classified as either end-to-end MLMs or tool-using MLMs.

End-to-end MLMs. End-to-end MLMs use a single, unified architecture [1, 8, 24, 26, 27] that can be repeatedly used for different tasks without requiring any architectural changes. While training end-to-end MLMs can be costly due to the need for large amounts of multi-task data, the use of diverse datasets allows end-to-end MLMs to generalize effectively and learn nuanced visual representations.

Tool-using MLMs. Tool-using MLMs enable LLMs to perform vision tasks by attaching specialized vision mod-

ules such as segmentation or captioning networks [32, 35, 48, 54]. Tool-using MLMs use a router to select the optimal vision module to use for a given input. The use of specialized modules allows MLMs to achieve higher performances on many tasks. However, tool-using MLMs are sensitive to errors because they link together several networks, which are each a potential point of failure.

Any-to-any networks. To address this, recent works [19, 20, 40, 47] such as Unified-IO [33, 34] have experimented with shared embedding spaces and visual decoders for vision and language tasks, training MLMs to generate segmentation masks, keypoints, and depth maps. Similarly, [49] generates special task tokens to route language model representations to diffusion heads for image, video, and audio generation. To handle complex segmentation scenarios, LISA [22] and GSVA [50] train language models to generate an additional segmentation token that can be used to produce segmentation masks, grounding the mask in language reasoning. While these approaches can generate visual outputs, they cannot reason over their own generations to solve related visual perception tasks. In contrast, augmenting MLMs with perception tokens as chain-of-thought tokens enables them to perform visual reasoning directly, providing significant gains in detail-oriented question-answering tasks such as depth estimation and counting.

### 3. Perception Tokens & Aurora

We introduce Perception Tokens and our Aurora training algorithm (see Fig. 3), which augment multimodal language models with perception tokens, enabling the MLM to leverage these tokens effectively during training and incorporate them into its chain-of-thought reasoning process for enhanced visual reasoning.

#### 3.1. Problem formulation

In autoregressive large language models, chain-of-thought (CoT) reasoning can be formulated as a multi-step inferential process in which the model iteratively generates intermediate reasoning steps to arrive at a final answer. Given a task input x, the model generates a response y conditioned on the input and a sequence of m intermediate reasoning steps {si}mi=1, where x, y, and each si are sequences of tokens from the model’s vocabulary V .

Existing multimodal language models often rely on limited vocabulary tokens derived from text or pre-trained image embeddings like CLIP, restricting their capacity to interpret other representations crucial for reasoning. Midlevel and low-level visual features such as depth maps, instance segmentation masks, human poses and bounding boxes which could substantially improve visual reasoning, are currently incompatible with the model and cannot be in-

tegrated during training or inference. Our key insight is to introduce auxiliary perception tokens for these intermediate steps with an expanded vocabulary V ′ = V ∪ Vaux, bridging this gap by allowing the model to integrate richer visual representations into its reasoning process. Conditioning the final output on these tokens enhances the model’s accuracy and interpretability across multimodal tasks.

#### 3.2. Perception token prediction and reasoning

Introducing an expanded vocabulary to enhance multimodal reasoning presents two main challenges. The first challenge is enabling the model to generate tokens from the new auxiliary vocabulary Vaux, which includes specialized tokens for low- and mid-level visual features, such as depth maps and bounding boxes. The second challenge is ensuring that the model can effectively condition on these auxiliary tokens to improve reasoning, particularly for multi-step inference tasks.

Perception token prediction. To address the first challenge, we employ a specialist-to-generalist distillation approach, using pre-trained specialist models (e.g., depth estimation or instance segmentation) to guide auxiliary token generation through cross-entropy loss. For each input x, the specialist model provides a target probability distribution qi over its tokens. Let M : Vspec → Vaux denote a one-to-one mapping from the specialist model’s vocabulary Vspec to the auxiliary token vocabulary Vaux. We define the distillation loss as:

qi log pM(i) , (1)

−

ℓdist = min

M

i

where pM(i) is the probability assigned by our model to the auxiliary token corresponding to the mapped token M(i). This consistent mapping allows the model to effectively align its predictions with the specialist model’s output distribution, enhancing the relevance and accuracy of its auxiliary token predictions.

In addition to distillation, we incorporate a reconstruction loss to enhance the token prediction and the interpretability of our model. Each auxiliary token corresponds to a specific representation, such as a depth map or a bounding box vector, and is trained to directly predict this feature. To achieve this, we introduced a lightweight decoder g that maps the tokens into the feature space, allowing for efficient and interpretable transformations. Formally, for a token t ∈ Vaux, decoder g, and its target feature f, the reconstruction loss is defined as:

ℓrec = ∥g(t) − f∥22, (2)

where g(t) is the decoded representation for token t in the feature space. This reconstruction process not only aligns each token with a meaningful feature, improving interpretability, but also reinforces the accuracy of auxiliary

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Figure 3. The overall AURORA training framework. We first learn visual perception tokens using VQVAE. We then finetune MLMs with a multi-task training approach where we distill intrinsic image representations (e.g., depth map) into MLMs by training them to decode the visual tokens as intermediate reasoning steps towards completing the tasks.

token predictions through direct feature supervision. In practice, while the reconstruction objective improves performance and interpretability, it is optional, as models can be effectively trained with the distillation objective alone to reduce computational overhead.

Reasoning with perception tokens. The second challenge is enabling the model to condition on tokens from Vaux effectively when generating each subsequent reasoning step, thereby enhancing its reasoning capabilities. To achieve this, we introduce chain-of-thought reasoning progressively, beginning with simpler, single-step reasoning tasks and advancing to more complex, multi-step inference. The model begins by learning single-step reasoning, predicting an initial reasoning step s1 based on the input x. It then progresses to multi-step reasoning, predicting sequences s1,··· ,sm and effectively utilizing auxiliary tokens to support extended chains of inference. We further reinforce this process with constrained decoding and an information bottleneck: in constrained decoding, we restrict sampling to auxiliary tokens, ensuring they serve as intermediate reasoning steps; in the information bottleneck approach, we truncate the reasoning chain before the auxiliary token when generating subsequent reasoning steps, forcing the model to rely solely on auxiliary tokens to reach the correct answer. Lastly, we provide a multi-task data synthesis approach to train the model using curriculum learning across various synthetic tasks. Further details are provided in Section 3.4.

#### 3.3. Tokenization

A unified tokenization space is crucial for multimodal models as it creates a consistent framework through which varied visual tasks can be represented, processed, and interpreted. Inspired by [38], we establish a unified tokeniza-

tion space which enables the model to enable the model to learn varied visual features within a shared representation seamlessly. In our experiments, we implement two tailored tokenization schemes for commonly used visual representations. Importantly, our framework is designed with flexibility, allowing it to generalize to a broad range of visual representations beyond those presented here.

Pixel-level representation. This tokenization scheme captures fine-grained spatial information, such as depth maps and segmentation masks, providing the model with detailed pixel-level data essential for accurate visual processing. For these types of tokens, we leverage visual tokenizers like VQVAE and VQGAN, which take in the ground truth masks or depth maps and return discrete target tokens [10, 33, 34, 46].

Structured representation. This scheme encodes structured yet abstracted visual features, such as human poses, bounding boxes, and coordinates, allowing the model to reason with higher-level spatial relationships and object hierarchies. For these tokens, we define the domain of the tokens based on specific properties; for example, the domain for coordinates can range from 0 to the maximum number of pixels in the image’s height or width [5].

#### 3.4. Curriculum learning with progressive CoTs

The objective of training the model is to develop a data and computationally efficient method for learning to predict novel, fine-grained visual tokens and using them to complete complex visual reasoning tasks. We observed that the standard approach, which relies on a fixed mixture data, encounters a trade-off between the accuracy of novel tokens predictions and the model’s reasoning capability, primarily due to catastrophic forgetting and challenges in reasoning with new tokens. Conversely, fine-tuning the model with the

###### Training

Model Direct Labeling Data

Depth Generation Data

CoT Data

BLINK [11] 2 Points

HardBLINK 3 Points

HardBLINK 4 Points

HardBLINK 5 Points

Average

LLaVA OneVision ✗ ✗ ✗ 51.6 33.1 22.6 18.5 31.4 LLaVA 1.5 13B ✗ ✗ ✗ 54.0 35.5 37.9 29 39.1 Fine-tunned LLaVA ✓ ✗ ✗ 68.5 58.9 52.4 41.1 55.2 LLaVA-AURORA (Ours) ✓ ✓ ✓ 64.5 66.9 60.5 54.8 61.6

GPT-4o ✗ ✗ ✗ 53.2 58.9 50 36.3 49.6 GPT-4 Turbo ✗ ✗ ✗ 58.1 54.8 41.9 32.2 46.7 GPT-4 Turbo + Tool ✗ ✗ ✗ 70.2 57.2 44.3 26.6 49.6

- Table 1. Performance comparison between our LLaVA-AURORA model, the fine-tunning baseline, and the original base model on the relative depth accuracy (%) task. Results demonstrate that our approach, utilizing depth tokens and intermediate reasoning steps, significantly outperforms both the baseline and the base model, particularly on more challenging configurations with 3, 4, and 5 points sampled from the image’s mid-height region.

original training mixture significantly raises computational costs and may be impractical if the original data, particularly proprietary datasets, is unavailable, making this approach less scalable for incorporating new tokens in the future.

We propose a curriculum learning-inspired training scheme that begins with atomic tasks and gradually advances to more complex ones requiring sophisticated, multi-hop reasoning. Let dt represent the difficulty of task t, with difficulties d1 < d2 < ···dT across T tasks, and let p(t,s) denote the probability of sampling data points from task t at training step s. We define p(dt,s) using a temperature-scaled Softmax formulation as follows:

exp(−dt/τ(s)) m i=1 exp(−di/τ(s))

, (3)

p(dt,s) =

where τ(s) modulates the task difficulty over time, allowing a smooth shift in the probability distribution toward harder tasks. This temperature function is defined as:

τ0 1 + λ · s/S

. (4)

τ(s) =

Here, τ0 is the initial temperature, λ is the annealing rate, and S is the total number of training steps.

Our approach for defining dt values is based on the inherent complexity of each task, which corresponds to the depth of reasoning involved. Specifically, we assign d1 to the most atomic task, involving the prediction of newly introduced tokens. At the other end of the spectrum, dm represents the final task, requiring the full chain-of-thought (CoT) reasoning steps. Between them, intermediate tasks {dt}Tt=1, serves to bridge the gap between the atomic tasks and the comprehensive chain-of-thought responses.

In this project, we introduce three types of data subsets for each downstream task, organized in increasing levels of

difficulty. The first and most atomic task involves teaching the model to generate tokens from the new auxiliary vocabulary . For instance, in depth-related tasks, we train the model to learn depth maps; in segmentation-related tasks, we teach the model to generate masks; and so on.

The other two data subsets involve Chain-of-Thought (CoT) prompts and direct labeling which help with reasoning with auxiliary tokens. In the CoT subset, we use the new intermediate visual perception tokens to answer downstream-specific questions, encouraging the model to reason step by step. In the direct labeling subset, we pose the same questions but instruct the model to provide direct answers without step-by-step reasoning.

Inspired by [15], we employ a multitasking approach for these two data subsets. For each image, we sequentially present both the CoT and direct labeling questions, allowing the model to tackle each image with both reasoning styles in sequence. We use a sequential sampler rather than a random sampler, shuffling the images beforehand. This strategy enables the model to learn from both types of reasoning tasks effectively, enhancing its ability to perform complex visual reasoning.

### 4. Experiments

We base our work on LLaVA 1.5 13B as the foundation for our model, which we refer to as LLaVA-AURORA. Our approach augments the MLM with perception tokens to enhance reasoning and improve performance across both 3D and 2D visual tasks. We evaluate our approach on relative depth estimation (3D) using pixel-level depth map tokens for fine-grained depth capture, and on object counting (2D) with mid-level bounding box tokens for precise localization. These tokens not only enhance task-specific results but also highlight our framework’s potential to generalize effectively across a broad spectrum of visual reasoning tasks.

#### 4.1. 3D reasoning task

We choose relative depth estimation as our 3D task because it enables the model to determine spatial relationships within a scene by identifying which points are closer to or farther from the camera. This foundational skill is crucial for scene understanding and applications requiring spatial awareness, such as robotics and autonomous systems. Specifically, this task involves identifying the point closest to the camera among multiple marked points in an image. To support the model’s reasoning, we use discrete depth map tokens that capture spatial depth information, enhancing the model’s understanding of proximity.

Tokenization. To capture fine-grained spatial details, we tokenize depth maps into sequences of discrete tokens. Inspired by the approach in AiT [38], we use a Vector Quantized Variational Autoencoder (VQVAE) [46] with a codebook size of 128. In this setup, each depth map is encoded as a grid of embeddings, with each embedding matched to the nearest entry in the codebook, yielding a compact depth representation. The VQVAE decoder reconstructs the depth map from this sequence of latent codes, and the entire model is optimized with a reconstruction loss to ensure precise encoding. During inference, each 320x320 depth map is compressed into a 10x10 grid of code indices, resulting in a 100token sequence where each token represents one of the 128 discrete depth tokens, labeled DEPTH 0 to DEPTH 127. To organize the sequence, we encapsulate it with special tokens DEPTH START and DEPTH END, adding a total of 130 depth-related tokens to the model’s vocabulary.

Training data. We train the VQVAE model on pseudodepth maps generated from the ADE20k dataset [56, 57] using the Depth Anything model [51, 52]. This dataset provides a diverse range of scenes, enhancing the model’s ability to generalize.

For fine-tuning, we prepare three types of data tailored for relative depth estimation (as detailed in Sec. 3.4). First, we generate depth maps for 20k ADE20k images, tokenize them with the pre-trained VQVAE, and format each sample in a Q&A structure, prompting the model with a depth estimation question and an answer sequence of depth tokens. Additionally, we construct a dataset of 500 ADE20k images for chain-of-thought (CoT) reasoning, with 2–5 markers in each image. Here, the prompt guides the model to generate the coordinates of the markers and then the depth map as intermediate steps, then identify the marker closest to the camera. This CoT training improves sequential reasoning and relative depth estimation accuracy. Finally, we use the same 500 images for direct labeling, prompting the model to directly identify and label the marker closest to the camera (More details in Supplementary).

This fine-tuning setup enables our model to effectively handle step-by-step reasoning for the relative depth estima-

tion task.

#### 4.2. 2D reasoning task

We select counting as a critical 2D visual task. For object counting, we incorporate bounding box predictions as an intermediate reasoning step to improve accuracy in answering counting queries. Given an image and a question about the number of specific objects, the model first identifies and predicts bounding boxes for each instance of the target object. These bounding box tokens serve as structured, intermediate representations, enabling the model to understand spatial arrangements and accurately count object instances.

Tokenization. To represent bounding boxes as discrete tokens, we resize all input images to a fixed resolution of 336x336 pixels. This preprocessing step allows us to add 336 unique tokens to the model’s vocabulary, each representing a specific pixel position within the resized image. These tokens, labeled PIXEL 0 to PIXEL 335, enable the model to uniquely reference each pixel location. Bounding boxes are encoded as tuples of four tokens, formatted as (PIXEL i, PIXEL j, PIXEL k, PIXEL m), where PIXEL i and PIXEL j denote the coordinates of the top-left corner and PIXEL k and PIXEL m represent the bottom-right corner (i.e., (x1,y1,x2,y2)). This discrete representation allows the model to interpret and use bounding box locations effectively, providing the spatial structure needed for accurate object counting.

Training data. To fine-tune the model for object counting, we draw on three types of data tailored to this task (as detailed in Sec. 3.4). First, we use task-specific data from the LVIS dataset [13], selecting 5k images with objects whose counts range from 0 to 15. For each selected image, we specify an object type (e.g., ”beds”) and structure the fine-tuning samples to prompt the model for bounding box predictions of the specified objects within the image.

To enhance the model’s reasoning ability, we include a small subset of 250 LVIS images for chain-of-thought (CoT) training. Here, each question prompt encourages step-by-step reasoning by instructing the model to first generate bounding boxes for the target object, followed by providing the final count. Additionally, we create a direct labeling subset using the same 250 images. In this subset, we prompt the model to directly identify and label the total count of the specified objects without the intermediate step of bounding box generation.

#### 4.3. Benchmarks

Relative depth. A recent benchmark, BLINK [11], introduces tasks designed to be intuitive for humans yet challenging for multimodal models, with relative depth estimation as one of its tasks. BLINK provides 124 images, each containing two marked points labeled as A and B, and asks

###### Training

Model Direct Labeling Data

Bounding Box Data

CoT Data CV-Bench Counting

SEED-Bench Counting

BLINK Counting

LLaVA One Vision ✗ ✗ ✗ 34.4 31.7 35.8 LLaVA 1.5 13B ✗ ✗ ✗ 40.9 52.2 35.0 Fine-tunned LlaVA ✓ ✗ ✗ 44.7 46.3 0.2 LLaVA-AURORA (Ours) ✓ ✓ ✓ 56.0 54.6 45.8 GPT-4o ✗ ✗ ✗ 70.18 64.6 47.5 GPT-4 Turbo ✗ ✗ ✗ 61.3 64.8 57.5 GPT-4 Turbo + Tool ✗ ✗ ✗ 48.6 29.9 26.7

- Table 2. Comparison of object counting accuracy (%) across three benchmarks (CV-Bench, SEED-Bench, and BLINK). Our LlaVAAURORA model, using auxiliary perception tokens to encode bounding box information for intermediate reasoning, demonstrates superior performance compared to the fine-tunning baseline models and the original base model.

which point is closer to the camera. To reduce biases that language models have toward answering multiple-choice questions [41, 55], we modify the original BLINK questions by removing the answer choices. To further evaluate the model’s reasoning and 3D understanding in relative depth, we curated a series of more challenging benchmark sets, collectively called HardBLINK. We progressively increase task difficulty by altering the prompts and image configurations as follows:

- 1. Prompt Modification: In the prompts, we exclude the number of markers and their labels, requiring the model to infer these details solely from the image.
- 2. Increased Point Complexity: We generate four variations of BLINK by adding more markers to each image, using Depth Anything to produce pseudo-depth maps for precise placement. These curated sets— HardBLINK 3points, HardBLINK 4points, and HardBLINK 5points—contain the same images as BLINK but with 3, 4, and 5 randomly placed markers, respectively, each with reasonable depth and distance differences. This setup tests the model’s depth reasoning across more complex spatial configurations.
- 3. Mitigating Height-Based Bias: To prevent the model from assuming that higher points are farther from the camera, we place markers at mid-height within the image. This approach encourages reliance on depth information rather than positional cues [6].

Counting. For counting, we evaluate the model on CVBench [42], SEED-Bench [23], and BLINK’s counting [11] subtask. To better capture the model’s capabilities, we also remove multiple-choice options, requiring it to generate an exact count.

- 4.4. Baselines

We evaluate a diverse range of models, including closedsource models like GPT-4o [39] and GPT-4 Turbo [39], as well as state-of-the-art open-source models such as LLaVA OneVision [25]. Another key baseline for our work involves

[Figure 20]

Figure 4. Depth maps generated by Aurora are imperfect but resemble the ground-truths from Depth Anything [51].

fine-tuning the base model AURORA is applied on, solely on the direct labeling portion of the training data for each task, omitting the newly introduced tokens. This approach allows us to isolate the impact of our token-based enhancements. Additionally, we evaluate the base model for AURORA, LLaVA 1.5 13B, to assess its performance without task-specific adaptations. For comparison, we use a toolaugmented baseline with an LLM.

In relative depth estimation, we employ GPT-4 Turbo, providing it with the ground truth depth maps generated by Depth Anything for each image, allowing it to use this information in its responses. Details on the exact format are provided in the supplementary materials [17]. For counting, we also use GPT-4 Turbo in a tool-augmented setup. In this process, GPT-4 Turbo first identifies the object specified in the question, then uses Grounding DINO [31] to locate the bounding boxes for each instance of the object and finally counts them.

#### 4.5. How new tokens improve 3D reasoning?

Our experiments demonstrate that incorporating new visual tokens significantly enhances the model’s 3D reasoning abilities, specifically in the relative depth estimation task. Results in Tab. 1 show that our model outperforms both the

BLINK [11] 2 points

HardBLINK 3 Points

HardBLINK 4 Points

HardBLINK 5 Points

Average

Model

VQGAN [10] (16384 codes) 82.2 66.1 53 37 59.6 Unified-IO 70.2 75.8 75.8 75.8 74.4 Unified-IO2 54 37.9 21 27.4 35.1 LLaVA-AURORA (Ours) 91.9 78.2 71.7 75.8 79.2 Our VQVAE (128 codes) 96.7 94.3 95.2 96.7 95.7

- Table 3. While not the main aim of our work, we report the depth generation performance across benchmarks with 2, 3, 4, and 5 marked points using BLINK [11]’s relative depth subtask images. We report relative depth estimation accuracy (%), calculated by programmatically extracting depth values at specific coordinates from model-generated depth maps. Our model consistently outperforms other multimodal models, including Unified-IO [33] and Unified-IO 2 [34]

primary baseline, which is fine-tuned solely on direct labeling data, and the original base model, indicating that the added tokens contribute meaningfully to task accuracy.

As task complexity increases—for instance, when additional markers are introduced in the HardBLINK benchmarks—our model’s performance advantage becomes even more pronounced. Not only does it maintain high accuracy in distinguishing depth relationships, but it also surpasses both advanced closed-source models, such as GPT-4 Turbo, and GPT-4 Turbo + Tool that use ground truth depth maps. This suggests that the new tokens enable our model to develop a more nuanced understanding of depth cues, even in cases where complex spatial reasoning is required.

Overall, these results highlight the effectiveness of our token-based approach in enhancing 3D reasoning, allowing the model to handle increasingly difficult tasks with robust performance and accuracy.

- 4.6. How new tokens improve 2D reasoning?

ically focusing on its ability to represent spatial relationships in visual scenes. For this evaluation, we use the relative depth images from the BLINK benchmark [11], ensuring consistency with our relative depth assessments. LLaVA-AURORA generates depth tokens from BLINK images, which are then reconstructed into full depth maps via the decoder of our pre-trained VQVAE model. Notably, the depth maps output by our pre-trained VQVAE provide an upper bound on the quality of the depth maps generated by LLaVA-AURORA. We use programmatic relative depth accuracy as our metric, which measures how well the reconstructed depth maps capture the relative depth of marked points. This evaluation spans several configurations in the benchmark, including images with 2, 3, 4, and 5 labeled points. By comparing the model’s predicted depth with the ground-truth marker coordinates, we calculate relative depth accuracy, allowing us to assess the precision of depth map generation in reflecting spatial depth relationships. As shown in Tab. 3, LLaVA-AURORA outperforms UnifiedIO [33, 34] in this task. Furthermore, qualitative analysis in Fig. 4 reveals that LLaVA-AURORA ’s depth maps capture spatial details effectively highlighting its capacity to interpret and represent fine-grained depth information.

In the 2D task of object counting, incorporating new visual tokens provides a significant advantage over the primary baselines. As shown in Tab. 2, our model outperforms both the baseline fine-tuned solely on direct labeling data and the original base model, as well as the state-of-the-art open-source model LLaVA-OneVision and GPT-4 Turbo + Tool, underscoring the value of these tokens in enhancing counting accuracy.

### 5. Conclusion

Our algorithm enables the lightweight and scalable integration of perception tokens, such as depth maps and bounding box coordinates, into MLMs, allowing them to perform intermediate reasoning steps akin to chain-of-thought processes. Our method achieves state-of-the-art results on challenging tasks like 2D object counting and 3D relative depth estimation. It also enhances model generalization and interpretability without relying on external tools or task-specific finetuning. The framework is inherently adaptable, incorporating new perception tokens as they emerge, making it a future-proof solution for advancing multimodal reasoning.

Although our model does not yet surpass advanced closed-source models, the results demonstrate that the new tokens yield a meaningful improvement in 2D reasoning, enabling more reliable and accurate object counting compared to standard fine-tuning approaches and open-source alternatives.

#### 4.7. Perception token decoding

Our approach enables the decoding of learned perception tokens into specialist features, such as depth maps and object bounding boxes, to assess their fidelity and utility. We, in particular, evaluate the accuracy and correctness of the depth maps generated by LLaVA-AURORA, specif-

Acknowledgements. This work is partially supported by Amazon Science. We also thank Yushi Hu and Lindsey Li for their insightful comments and suggestions.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning,

2022. 2

- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. 2
- [3] Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Guiding llms the right way: Fast, non-invasive constrained generation. arXiv preprint arXiv:2403.06988, 2024. 3
- [4] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465,

2024. 2

- [5] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. arXiv preprint arXiv:2109.10852, 2021. 4
- [6] Weifeng Chen, Zhao Fu, Dawei Yang, and Jia Deng. Singleimage depth perception in the wild. Advances in neural information processing systems, 29, 2016. 7
- [7] Aidan Cooper. A guide to structured outputs using constrained decoding, 2024. 3
- [8] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 2

- [9] PHILOSO EPHY DO CT OR OF. MACHINE PERCEPTION OF THREE-DIMENSIONAL, SO LIDS. PhD thesis, MASSACHUSETTS INSTITUTE OF TECHNOLOGY,

1961. 1

- [10] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 4, 8
- [11] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390,

2024. 1, 5, 6, 7, 8

- [12] Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. Grammar-constrained decoding for structured nlp tasks without finetuning. arXiv preprint arXiv:2305.13971,

2023. 3

- [13] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In Pro-

- ceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019. 6
- [14] Chris Harris, Mike Stephens, et al. A combined corner and edge detector. In Alvey vision conference, pages 10–5244. Citeseer, 1988. 1
- [15] Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. arXiv preprint arXiv:2305.02301,

2023. 2, 5

- [16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2, 3
- [17] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. arXiv preprint arXiv:2406.09403, 2024. 2, 7
- [18] Yushi Hu, Otilia Stretcu, Chun-Ta Lu, Krishnamurthy Viswanathan, Kenji Hata, Enming Luo, Ranjay Krishna, and Ariel Fuxman. Visual program distillation: Distilling tools and programmatic reasoning into vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9590–9601, 2024. 2
- [19] Yang Jin, Kun Xu, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Quzhe Huang, Bin Chen, Chenyi Lei, An Liu, Chengru Song, Xiaoqiang Lei, Di Zhang, Wenwu Ou, Kun Gai, and Yadong Mu. Unified language-vision pretraining in llm with dynamic discrete visual tokenization, 2024. 3
- [20] Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Generating images with multimodal language models, 2023. 3
- [21] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 1
- [22] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model, 2024. 3
- [23] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 7, 1
- [24] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimic-it: Multi-modal in-context instruction tuning, 2023. 2
- [25] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 7
- [26] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. 2
- [27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2, 1

- [28] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 2
- [29] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 3
- [30] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2024. 2
- [31] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 7
- [32] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024. 3
- [33] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022. 3, 4, 8
- [34] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action, 2023. 3, 4, 8
- [35] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models, 2023. 3
- [36] David Marr. Vision: A computational investigation into the human representation and processing of visual information. MIT press, 2010. 1
- [37] Marvin Minsky and Seymour Papert. An introduction to computational geometry. Cambridge tiass., HIT, 479(480): 104, 1969. 1
- [38] Jia Ning, Chen Li, Zheng Zhang, Chunyu Wang, Zigang Geng, Qi Dai, Kun He, and Han Hu. All in tokens: Unifying output space of visual tasks via soft token. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19900–19910, 2023. 4, 6
- [39] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 7
- [40] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models, 2024. 3
- [41] Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. arXiv preprint arXiv:2308.11483, 2023. 7
- [42] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms,

2024. 1, 7, 2

- [43] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the

- visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024. 1
- [44] Antonio Torralba and Aude Oliva. Depth estimation from image structure. IEEE Transactions on pattern analysis and machine intelligence, 24(9):1226–1238, 2002. 1
- [45] Vivien Tran-Thien. Fast, high-fidelity llm decoding with regex constraints, 2024. 3
- [46] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 4, 6
- [47] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022. 3
- [48] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models, 2023. 3
- [49] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm, 2024. 3
- [50] Zhuofan Xia, Dongchen Han, Yizeng Han, Xuran Pan, Shiji Song, and Gao Huang. Gsva: Generalized segmentation via multimodal large language models, 2024. 3
- [51] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024. 6, 7, 1
- [52] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv preprint arXiv:2406.09414, 2024. 6
- [53] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl: Modularization empowers large language models with multimodality, 2024. 2
- [54] Haoxuan You, Rui Sun, Zhecan Wang, Long Chen, Gengyu Wang, Hammad A. Ayyubi, Kai-Wei Chang, and Shih-Fu Chang. Idealgpt: Iteratively decomposing vision and language reasoning via large language models, 2023. 3
- [55] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. In The Twelfth International Conference on Learning Representations, 2023. 7
- [56] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641,

2017. 6

- [57] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127:302–321, 2019. 6

## Perception Tokens Enhance Visual Reasoning in Multimodal Language Models Supplementary Material

### 6. Ablation study

In this section, we analyze the impact of various design choices and data configurations on the performance of our proposed method. We focus on three aspects: (1) the impact of including or excluding specific steps in the chain-ofthought (CoT) reasoning process for the 3D task of relative depth estimation, (2) the use of standard text tokens versus new perception pixel tokens for the 2D task of object counting, and (3) the effect of incorporating a perception token reconstruction loss during fine-tunning.

#### 6.1. Chain-of-thought steps

For our 3D task of relative depth estimation, the chain-ofthought (CoT) questions in the fine-tunning data include two steps: (1) identifying the coordinates or locations of the points marked in the image, and (2) generating the depth map and determining which point is closer to the camera based on pixel values in the depth map. This study evaluates the impact of including or excluding these steps in the question prompts during fine-tunning.

We experiment with three variations of fine-tunning data configurations:

- 1. Direct Labeling Baseline: The model is fine-tuned solely on direct labeling data, where the question prompts directly ask which point is closer to the camera and provide the label as the answer. These prompts do not include either step (1) or step (2), see baselines section.
- 2. Step (2) Only: This model is fine-tuned with prompts that exclude step (1) (point location identification) but include step (2), asking the model to answer based on the depth map alone.
- 3. AURORA: Our proposed AURORA technique uses prompts that include both steps (1) and (2), explicitly guiding the model through point location identification before generating the depth map.

All models are evaluated on the harder BLINK datasets we introduced. As shown in Tab. 4, the results demonstrate that having both steps in the prompts provides the most significant performance improvement. This suggests that guiding the model through a multi-step reasoning process in the prompts enables it to better capture spatial relationships and achieve more accurate depth estimations.

#### 6.2. Text tokens vs. Perception tokens

In this ablation study, we evaluate the impact of using perception tokens compared to standard text tokens for the object counting subtask. Perception tokens are represented in the format PIXEL X, where X is a number between

0 and 336, indicating pixel locations for object bounding boxes. For comparison, we replace these perception tokens with regular text tokens in the fine-tunning data, such that PIXEL 100 is replaced with 100, and so on.

As shown in Tab. 5, models utilizing perception tokens achieve higher performance across all three counting benchmarks: BLINK [11], SEED-Bench [23], and CVBench [42]. This demonstrates the effectiveness of perception tokens in explicitly encoding spatial information for improved counting accuracy.

#### 6.3. Perception token reconstruction loss

The aim of this ablation study is to assess whether adding the perception token reconstruction loss, despite its increased computational cost, significantly improves model performance. Incorporating this loss requires adding the decoder for the specific task, which increases computation time and resource requirements. Not using it makes the system lighter and faster by just using the token classification loss. Therefore, we evaluate whether the performance gains justify the additional overhead.

To this end, we fine-tune two models based on LLaVA 1.5 13B [27] using a dataset of 20,000 samples only for depth map generation. Each sample includes a prompt such as ”What is the depth map for the image?” and a response containing sequences of depth tokens. Both models are finetuned for 10 epochs: one with the reconstruction loss and one without it (both with cross entropy loss).

The reconstruction loss is computed as the mean squared error (MSE) between the ground truth depth map, which is the output of the VQVAE decoder when provided with the ground truth depth tokens, and the predicted depth map, which is generated by decoding the depth tokens produced by the LLM. A soft merging technique is used in reconstruction, where a ”soft token” is created by averaging the embeddings of all potential tokens, weighted by their prediction probabilities from the LLM.

The models are evaluated on two datasets: (1) 124 images from the relative depth subtask of BLINK [11], and (2) 1000 random images from the Visual Genome dataset [21], for which depth maps were generated using Depth Anything [51]. The evaluation metric is the mean squared error (MSE) between the ground truth decoded depth maps and the depth maps reconstructed from the model’s output tokens.

As shown in Tab. 6, the results indicate that incorporating the reconstruction loss does not significantly improve model performance. Fig. 5 further illustrates qualitative results, highlighting the visual differences in the predicted

CoT steps Model Coordinates Depth HardBLINK

HardBLINK 4 Points

HardBLINK 5 Points

3 Points

Direct Labeling Baseline ✗ ✗ 58.9 52.4 41.1 Step (2) only ✗ ✓ 56.4 56.4 50 LLaVA-AURORA (Ours) ✓ ✓ 66.9 60.5 54.8

- Table 4. Performance comparison of models trained with different Chain of Thought question prompt variations for relative depth estimation on the harder BLINK datasets. Models with both steps in the prompts (AURORA) achieve the best performance.

Model Token Type CV-Bench

Counting

SEED-Bench Counting

BLINK Counting

LLaVA-AURORA Standard 52.2 50.6 38.3 LLaVA-AURORA Perception 56.0 54.6 45.8

- Table 5. Comparison of model performance using perception tokens and standard tokens for the object counting task across three benchmarks: BLINK, SEED-Bench, and CV-Bench. Perception tokens consistently improve accuracy.

depth maps with and without the reconstruction loss. While the reconstruction loss enforces consistency between the generated and ground truth depth tokens, its overall contribution is marginal in this setup. This study suggests that omitting the reconstruction loss may be a more efficient choice, especially when computational cost is a concern. Future work could explore its impact in larger datasets or more complex tasks to better understand its potential benefits.

Mean Squared Error↓ Model Recons

Loss

BLINK Visual Genome

LLaVA 1.5 ✓ 0.092 0.074 LLaVA 1.5 ✗ 0.087 0.076

- Table 6. MSE evaluation of models with and without reconstruction loss on subsets BLINK and Visual Genome datasets.
- 7. Cross-task generalization

[Figure 21]

To assess the generalizability of AURORA trained on depth generation and Chain of Thought (CoT) data for the relative depth task, we evaluate it on a different depth-related task. Specifically, we use the Depth subtask from CVBench [42], which involves identifying which of two objects, highlighted with red and blue bounding boxes, is closer to the camera. Similar to the BLINK evaluations for relative depth, we remove options from question prompts in these evaluations too.

Figure 5. Qualitative comparison of predicted depth maps with and without reconstruction loss.

tasks.

### 8. Implementation details

As shown in Tab. 7, our model outperforms both the base LLaVA 1.5 13B model and the fine-tuning baseline, demonstrating its generalization capabilities across depth-related

Computation resources. We train Aurora models on single-node machines equipped with 8 A40 GPUs. Each

Model CV-Bench Depth

LLaVA 1.5 13B 62.2 Fine-tunned LLaVA 60.0 AURORA (Ours) 64.8

Table 7. Performance comparison on the CV-Bench Depth subtask, highlighting our model’s generalization ability.

training run completes in less than 10 hours.

Model architecture and token expansion. Our approach builds on the LLaVA 1.5 13B model [29], a pre-trained multimodal language model. To support depth-related tasks, we expand the tokenizer by introducing 130 tokens for depth maps and 336 tokens for bounding box coordinates, increasing the vocabulary size beyond the original 32,000 tokens. These additions require modifications to the token embedding layer (embed tokens) and the language model head (lm head) to accommodate the new tokens.

Fine-tuning approach. We apply LoRA [16] to the language model for efficient fine-tuning. The vision backbone is kept frozen while the embed tokens and lm head layers are fully trained. This strategy enables the model to integrate depth and bounding box information without overwriting its pre-trained knowledge.

We fine-tune the model for 10 epochs, using the same LoRA parameters and learning rates as LLaVA. Finetunning follows a cross-entropy loss for next-token prediction, treating the new tokens identically to the original vocabulary.

Inference and decoding. During inference, we use a temperature of 0 for deterministic generation and employ constrained decoding techniques [3, 7, 12, 45]. For depth map generation, the model outputs exactly 100 depth tokens between DEPTH START and DEPTH END, ensuring consistent and structured results.

Curriculum learning for reasoning. To enhance the model’s reasoning capabilities, we employ progressive chain-of-thought (CoT) for curriculum learning. Finetuning starts with atomic tasks, such as depth map estimation and bounding box predictions, and gradually incorporates multi-tasking data, including CoT reasoning and direct labeling tasks. For instance, in the depth-related task, we use 20,000 samples for depth generation and 1,000 multitasking samples (comprising 500 unique images with sequential CoT and direct labeling questions).

In the first epoch, the model is fine-tuned exclusively on 20,000 depth generation samples. Starting from the second epoch, we introduce multi-tasking data by mixing 18,000 random depth generation samples with 2,000 multi-tasking samples (the 1,000 multi-tasking samples repeated twice).

This ratio is progressively adjusted in subsequent epochs, culminating in the 10th epoch, where the model is exposed to 2,000 depth generation samples and 18,000 multi-tasking samples. This staged approach ensures a smooth transition from basic tasks to complex reasoning, effectively reinforcing the model’s ability to handle multi-step reasoning challenges.

Fine-tuning data. As discussed in the Methods section, each task is supported by three sub-datasets. For the depth task, these include (1) depth generation data, (2) chain-ofthought (CoT) reasoning data, and (3) direct labeling data. Similarly, for the counting task, the sub-datasets consist of (1) bounding box prediction data, (2) CoT reasoning data, and (3) direct labeling data.

Fig. 6 and Fig. 7 present representative samples from each sub-dataset for the depth and counting tasks, respectively.

[Figure 22]

###### Figure 6. Examples of sub-datasets for the depth task: (1) depth generation, (2) Chain-of-Thought reasoning, and (3) direct labeling.

[Figure 23]

###### Figure 7. Examples of sub-datasets for the counting task: (1) bounding box prediction, (2) Chain-of-Thought reasoning, and (3) direct labeling.

