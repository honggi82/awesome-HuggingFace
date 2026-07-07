# arXiv:2411.07975v2[cs.CV]24Mar2025

## JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation

Yiyang Ma1,2 Xingchao Liu1,† Xiaokang Chen1,† Wen Liu1,† Chengyue Wu1,3 Zhiyu Wu1,2 Zizheng Pan1 Zhenda Xie1 Haowei Zhang1 Xingkai Yu1 Liang Zhao1 Yisong Wang1,4 Jiaying Liu2 Chong Ruan1,‡

1DeepSeek-AI 2Peking University 3The University of Hong Kong 4Tsinghua University †Equal contribution, ‡Corresponding author Project Page: https://github.com/deepseek-ai/Janus

### Abstract

We present JanusFlow, a powerful framework that unifies image understanding and generation in a single model. JanusFlow introduces a minimalist architecture that integrates autoregressive language models with rectified flow, a state-of-the-art method in generative modeling. Our key finding demonstrates that rectified flow can be straightforwardly trained within the large language model framework, eliminating the need for complex architectural modifications. To further improve the performance of our unified model, we adopt two key strategies: (i) decoupling the understanding and generation encoders, and (ii) aligning their representations during unified training. Extensive experiments show that JanusFlow achieves comparable or superior performance to specialized models in their respective domains, while significantly outperforming existing unified approaches across standard benchmarks. This work represents a step toward more efficient and versatile vision-language models.

#### 1. Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in learning diverse knowledge and generalizing to new scenarios [1, 7, 8, 69, 91]. Leveraging these capabilities, researchers have developed sophisticated models specialized in image comprehension [2, 15, 47, 49, 56, 58] and text-to-image generation [23, 73, 76, 79].

The field has recently shifted toward creating unified systems capable of handling both tasks simultaneously. One prominent direction involves utilizing pre-trained text-to-image models for high-quality generation while training LLMs to generate conditions for these models [19, 25–27, 87]. However, this approach introduces architectural complexity and potentially constrains the model’s capabilities through maintaining separate LLM and generative components. Alternative approaches [88, 97, 99, 100, 108] propose training a single LLM for both tasks, typically incorporating either diffusion models [32, 83] or vector-quantized autoregressive models [22, 86].

Our approach builds upon recent breakthroughs in rectified flow models [3, 23, 55, 61, 62], which provide a simple framework for generative modeling while delivering exceptional empir-

POPE

VQAv2

GQA

84.54

76.18

57.55

73.03

64.12

48.36

61.52

52.06

39.18

MME-Perception

MMBench

1257.36

68.55

1004.91

47.36

752.46

26.18

-16.82

42.27

40.0

16.33

-13.64

54.54

-10.46

66.82

50.0

22.66

-MJHQ FID

SEEDB

60.0

29.0

Emu3-Chat (8B)

InstructBLIP (7B)

LLaVA-v1.5-Phi (1.4B)

Show-o (1.3B)

GenEval MM-Vet

JanusFlow (Ours, 1.3B)

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

[Figure 13]

[Figure 14]

[Figure 15]

(a) Benchmark Performances. (b) Visual Generation Results.

- Figure 1 | Multimodal understanding and image generation with JanusFlow. JanusFlow surpasses the state-of-the-art unified multimodal models and several task-specific understanding models on visual understanding benchmarks. It is also capable of generating high-quality images. The resolution of the images is 384 × 384.

ical performance [23, 36, 45]. Building on these advances, we propose JanusFlow, a powerful unified multimodal model that seamlessly integrates rectified flow with LLM architecture. Following a minimalist design principle, our architecture requires only a lightweight encoder and decoder to adapt the LLM for rectified flow operations. To optimize JanusFlow’s performance, we implement two key strategies: First, we maintain separate vision encoders for understanding and generation tasks, preventing task interference and thus enhancing comprehension capabilities. Second, we align the intermediate representations between generation and understanding modules during training, strengthening semantic coherence in the generation process.

JanusFlow shows state-of-the-art performances in both multimodal comprehension and text-to-image generation compared to existing unified approaches, and even outperforms several specialized methods. Specifically, on text-to-image generation benchmarks, MJHQ FID-30k [48], GenEval [28] and DPG-Bench [34], JanusFlow achieves scores of 9.51, 0.63 and 80.09%, surpassing established text-to-image models including SDv1.5 [77] and SDXL [73]. In multimodal comprehension benchmarks, JanusFlow attains scores of 74.9, 70.5 and 60.3 on MMBench [63], SeedBench [46], and GQA [35], respectively, exceeding specialized models such as LLaVA-v1.5 [56] and Qwen-VL-Chat [4]. Notably, these results are achieved with a compact LLM architecture with only 1.3B parameters.

#### 2. Related Work

Visual Generation with Flow-based Generative Models. Recent years have witnessed remarkable progress in visual generation through diffusion models [32, 83], leading to impressive models like [67, 73, 76–79]. Building on these advances, flow-based generative models [3, 55, 61] emerged as a simplified alternative framework. These approaches have recently enabled advanced visual generation models [23, 36] that achieve superior empirical performance with faster sampling. Our work demonstrates that rectified flow [60–62] can be effectively integrated into LLMs, creating unified models that excel in both understanding and generation tasks.

Unified Models For Understanding and Generation. The development of multimodal large language models (MLLMs) has enabled effective integration of text and visual information. Building upon powerful LLMs [7, 91, 92], recent MLLMs [2, 15, 49, 56, 58, 64] have demonstrated exceptional multimodal understanding capabilities. Current research increasingly focuses on architectures that can simultaneously handle visual understanding and generation tasks. One approach extends MLLMs with pre-trained diffusion models [19, 25–27, 87, 101]. However, these systems essentially utilize diffusion models as external tools, where the MLLM generates conditions for image generation without possessing direct generative capabilities. This separation often results in suboptimal performance compared to standalone diffusion models [25, 87]. Another line of work [88, 97, 99, 100, 108] aim to train a single LLM for both tasks. Many of these methods employ vector-quantization [22, 86] to convert images into discrete tokens, enabling unified autoregressive processing [88, 97]. While straightforward to implement, these approaches are inherently limited by their image tokenization quality.

Our work focuses on developing unified models that combine autoregressive capabilities with flow/diffusion models, leveraging their proven effectiveness in visual generation. Compared to similar approaches [100, 107, 108], JanusFlow offers three key advantages: (i) a simple yet effective generation process using rectified flow, (ii) enhanced performance through decoupled vision encoders that resolve inter-task conflicts, and (iii) improved generation quality through representation alignment regularization, enabled by our decoupled encoder design.

- 3. JanusFlow In this section, we introduce the architecture of JanusFlow and our training strategies.

###### 3.1. Background

Multimodal LLMs. Given a dataset D containing discrete token sequences, each of which can be formulated as 𝑥 = (𝑥1, · · · , 𝑥ℓ), large language models (LLMs) are trained to model the sequence distribution in an autoregressive manner,

∑︁ℓ−1

logP𝜃

logP𝜃

𝐿𝐿𝑀 (𝑥𝑖+1|𝑥1, . . . , 𝑥𝑖), (1)

𝐿𝐿𝑀 (𝑥) =

𝑖=0

where 𝜃𝐿𝐿𝑀 denotes the parameters of the LLM and ℓ is the sequence length. After being trained on large-scale datasets, LLMs exhibit the ability to generalize across various tasks and follow diverse instructions [1, 8, 69]. To extend these models to handle visual inputs, LLMs are augmented with vision encoders [2, 56, 58]. For instance, LLaVA [58] integrates an LLM with a pre-trained CLIP [75] image encoder via a projection layer, transforming the extracted image features into a joint embedding space that the LLM can process as word embeddings. By leveraging large-scale multimodal datasets and increasingly powerful LLMs, this architecture has facilitated the development of advanced multimodal models capable of addressing a wide range of vision-language tasks [4, 47, 56, 64].

Rectified Flow. For a dataset D consisting of continuous 𝑑-dimensional data points 𝑥 = (𝑥1, · · · , 𝑥𝑑) drawn from an unknown data distribution 𝜋1, rectified flow [55, 61] models the data distribution by learning an ordinary differential equation (ODE) defined over time 𝑡 ∈ [0,1]:

d𝑧𝑡 d𝑡

= 𝑣𝜃𝑁𝑁 (𝑧𝑡,𝑡), 𝑧0 ∼ 𝜋0, (2)

(a) Understanding: Autoregression

(b) Generation: Rectified Flow

Response

Velocity

(Next Token Prediction)

(All Image Tokens)

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Gen. Decoder 𝑔𝑑𝑒𝑐

Text De-Tokenizer

𝑧𝑡+d𝑡 = 𝑧𝑡 + 𝑣𝑡d𝑡

[Figure 20]

[Figure 21]

##### Large Language Model

Overwrite 𝑧𝑡 with 𝑧𝑡+d𝑡

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Und. Encoder 𝑓𝑒𝑛𝑐 Text Tokenizer

Gen. Encoder 𝑔𝑒𝑛𝑐

Text Tokenizer

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Repeat until 𝑡 = 1

Clean Image Und. Prompt Noisy Image 𝑧𝑡

Gen. Prompt

- Figure 2 | Architecture of the proposed JanusFlow. For visual understanding, the LLM performs autoregressive next-token prediction to generate responses. For image generation, the LLM employs images with rectified flow. Starting from Gaussian noise at 𝑡 = 0, the LLM iteratively updates 𝑧𝑡 by predicting velocity vectors until reaching 𝑡 = 1. We omit the VAE encoder, the skip connection leveraged in generation and the linear layer after 𝑓𝑒𝑛𝑐 for simplicity.

where 𝜃𝑁𝑁 represents the parameters of the velocity neural network and 𝜋0 is a simple distribution, typically standard Gaussian noise N(0, 𝐼). The network is trained by minimizing the Euclidean distance between the neural velocity and the directions of linear paths connecting random points from 𝜋0 and 𝜋1,

E𝑡∼P(𝑡),𝑧0∼𝜋0,𝑥∼𝜋1 𝑣𝜃𝑁𝑁 (𝑧𝑡,𝑡) − (𝑥 − 𝑧0) 2 , where 𝑧𝑡 = 𝑡𝑥 + (1 − 𝑡)𝑧0. (3)

min

𝜃

Here, P(𝑡) is a distribution over time 𝑡 ∈ [0,1]. When the network has sufficient capacity and the objective is perfectly minimized, the optimal velocity field 𝑣𝜃∗

maps the elementary distribution 𝜋0 to the true data distribution 𝜋1. More precisely, the distribution of 𝑧1 = ∫ 1

𝑁𝑁

(𝑧𝑡,𝑡)d𝑡, with 𝑧0 ∼ 𝜋0, follows 𝜋1. Despite its conceptual simplicity, rectified flow has shown superior performance in various generative modeling tasks, including text-to-image generation [23], audio generation [40] and biological structure generation [38].

0 𝑣𝜃∗

𝑁𝑁

###### 3.2. A Unified Framework for Multimodal Understanding and Generation

JanusFlow presents a unified framework designed to address both vision understanding and image generation tasks. Next we outline how JanusFlow handles these two tasks within a single LLM architecture.

Multimodal Understanding. In multimodal understanding tasks, the LLM processes an input sequence consisting of interleaved text and image data. The text is tokenized into discrete tokens, each of which is transformed into an embedding of dimension 𝐷𝑒𝑚𝑏. For the images, an image encoder 𝑓𝑒𝑛𝑐 encodes each image 𝑥𝑖𝑚 into a feature map of shape 𝐻𝑖𝑚 × 𝑊𝑖𝑚 × 𝐷𝑒𝑛𝑐. This feature map is flattened and projected through a linear transformation layer into a sequence of embeddings with shape 𝐻𝑖𝑚𝑊𝑖𝑚 × 𝐷𝑒𝑚𝑏. 𝐻𝑖𝑚 and 𝑊𝑖𝑚 are determined by the image encoder. The text and image embeddings are concatenated to form the input sequence to the LLM, which then autoregressively predicts the next tokens based on the input sequence of embeddings. According to common practice [88, 97, 100], we add special token |BOI| before the image and |EOI| after the image to help the model locate the image embeddings in the sequence.

Image Generation. For image generation, our LLM takes a text sequence 𝑥𝑐𝑜𝑛 as condition and generates a corresponding image using rectified flow. To improve computational efficiency, generation occurs in the latent space using a pre-trained SDXL-VAE [73].

The generation process begins by sampling Gaussian noise 𝑧0 of shape 𝐻𝑙𝑎𝑡𝑒𝑛𝑡 ×𝑊𝑙𝑎𝑡𝑒𝑛𝑡 × 𝐷𝑙𝑎𝑡𝑒𝑛𝑡 in the latent space, which is then processed by a generation encoder 𝑔𝑒𝑛𝑐 into a sequence of embeddings 𝐻𝑔𝑒𝑛𝑊𝑔𝑒𝑛 × 𝐷𝑒𝑚𝑏. This sequence is concatenated with a time embedding representing the current time step 𝑡 (𝑡 = 0 at the beginning), resulting in a sequence of length 𝐻𝑔𝑒𝑛𝑊𝑔𝑒𝑛 + 1. Unlike previous approaches that employ various attention masking strategies [100, 108], we found that causal attention suffices, as our preliminary experiments showed no performance benefits from alternative masking schemes. The LLM’s output corresponding to 𝑧0 is transformed back into the latent space by a generation decoder 𝑔𝑑𝑒𝑐, producing a velocity vector of shape 𝐻𝑙𝑎𝑡𝑒𝑛𝑡 ×𝑊𝑙𝑎𝑡𝑒𝑛𝑡 × 𝐷𝑙𝑎𝑡𝑒𝑛𝑡. The state is updated by a standard Euler solver,

𝑧𝑡+d𝑡 = 𝑧𝑡 + 𝑣(𝑧𝑡,𝑡)d𝑡, (4)

where d𝑡 is a user-defined step size. We replace 𝑧0 with 𝑧d𝑡 on the input and iterate the process until we get 𝑧1, which is then decoded into the final image by the VAE decoder. To enhance generation quality, we employ classifier-free guidance (CFG) when computing the velocity:

𝑣(𝑧𝑡,𝑡) = 𝑤𝑣(𝑧𝑡,𝑡 | 𝑥𝑐𝑜𝑛) + (1 − 𝑤)𝑣(𝑧𝑡,𝑡 | ∅), (5)

where 𝑣(𝑧𝑡,𝑡 | ∅) denotes the velocity inferred without text conditioning and 𝑤 ⩾ 1 controls the magnitute of CFG. Empirically, increasing 𝑤 yields higher semantic alignment [23, 62, 73, 77]. Analogous to multimodal understanding, we prepend the special token |BOI| to indicate the start of image generation in the sequence.

Decoupling Encoders for the Two Tasks. Previous approaches that unify autoregressive generation and diffusion models within a joint LLM training framework [100, 108] employ identical encoders (𝑓𝑒𝑛𝑐 and 𝑔𝑒𝑛𝑐) for both understanding and generation tasks. For instance, Zhou et al. [108] performs both tasks in the same VAE latent space using a shared U-Net or linear encoder, while Xie et al. [100] leverages MAGVIT-v2 [102] to encode image patches into discrete tokens for both tasks.

However, recent work on unified autoregressive models has shown this shared encoder design to be suboptimal [97], particularly in models that generate images through autoregression on vector-quantized tokens. Drawing from these insights, JanusFlow adopts a decoupled encoder design. Specifically, we employ a pre-trained SigLIP-Large-Patch/16 [106] model as 𝑓𝑒𝑛𝑐 to extract semantic continuous features for multimodal understanding, while using separate ConvNeXt blocks [96] initialized from scratch as 𝑔𝑒𝑛𝑐 and 𝑔𝑑𝑒𝑐 for generation, chosen for its effectiveness. Following established practices [5, 14, 93], we incorporate a long skip connection between 𝑔𝑒𝑛𝑐 and 𝑔𝑑𝑒𝑐. Our controlled experiments in Sec. 4.5 demonstrate that this decoupled encoder design significantly improves the performance of our unified model. The complete architecture of JanusFlow is illustrated in Fig. 2.

- 3.3. Training Schemes As illustrated in Fig. 3, we train our model in three sequential stages, detailed below.

- Stage 1: Adaptation of Randomly Initialized Components. In the first stage, we focus on training only the randomly initialized components: the linear layers, generation encoder, and

Stage 1

Stage 2

Stage 3

Adaptation

Unified Pre-Training

Supervised Fine-Tuning

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Text De-Token. Gen. Dec. 𝑔𝑑𝑒𝑐

Text De-Token. Gen. Dec. 𝑔𝑑𝑒𝑐

Text De-Token. Gen. Dec. 𝑔𝑑𝑒𝑐

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

###### LLM

LLM

LLM

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

Linear Gen. Enc. 𝑔𝑒𝑛𝑐

Linear Gen. Enc. 𝑔𝑒𝑛𝑐

Linear Gen. Enc. 𝑔𝑒𝑛𝑐

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Und. Enc. 𝑓𝑒𝑛𝑐

Und. Enc. 𝑓𝑒𝑛𝑐

Und. Enc. 𝑓𝑒𝑛𝑐

VAE Enc.

VAE Enc.

VAE Enc.

- Figure 3 | Three training stages of JanusFlow. The trainable modules are marked with flame and the frozen modules are marked with snowflakes.

generation decoder. This stage serves to adapt these new modules to work effectively with the pre-trained LLM and SigLIP encoder, essentially functioning as an initialization phase for the newly introduced components.

- Stage 2: Unified Pre-Training. Following the adaptation stage, we train the entire model except for the visual encoder, consistent with previous approaches [58, 64]. The training incorporates three data types: multimodal understanding, image generation, and text-only data. We initially allocate a higher proportion of multimodal understanding data to establish the model’s understanding capabilities. Subsequently, we increase the ratio of image generation data to accommodate the convergence requirements of diffusion-based models [18, 72].
- Stage 3: Supervised Fine-Tuning (SFT). In the final stage, we fine-tune the pre-trained model using instruction tuning data, which comprises dialogues, task-specific conversations, and highquality text-conditioned image generation examples. During this stage, we also unfreeze the SigLIP encoder parameters [64, 90, 97]. This fine-tuning process enables the model to effectively respond to user instructions for both multimodal understanding and image generation tasks.

###### 3.4. Training Objective

Training JanusFlow involves two types of data, multimodal understanding data and image generation data. Both types of data contain two parts: “condition” and “response”. “Condition” refers to the prompting of the tasks (e.g., text prompts in the task of generation and images in the task of understanding) while “response” refers to the corresponding responses of the two tasks. The data can be formatted as 𝑥 = (𝑥𝑐𝑜𝑛, 𝑥𝑟𝑒𝑠), where the superscript 𝑐𝑜𝑛 denotes “condition” and 𝑟𝑒𝑠 denotes “response”. We denote the length of the whole sequence 𝑥 as ℓ, the length of 𝑥𝑐𝑜𝑛 as ℓ𝑐𝑜𝑛 and the length of 𝑥𝑟𝑒𝑠 as ℓ𝑟𝑒𝑠. We use 𝜃 to represent the collection of all the trainable parameters in JanusFlow, including the LLM, 𝑓𝑒𝑛𝑐, 𝑔𝑒𝑛𝑐, 𝑔𝑑𝑒𝑐 and the linear transformation layers.

Autoregression Objective. For mutimodal understanding tasks, 𝑥𝑟𝑒𝑠 contains only text tokens. JanusFlow is trained using the maximum likelihood principle,

###### L𝐴𝑅(𝜃) = −E𝑥∼D

𝑢𝑛𝑑

###### ∑︁ ℓ−1

logP𝜃 (𝑥𝑖+1|𝑥1, . . . , 𝑥𝑖) , (6)

𝑖=ℓ𝑐𝑜𝑛

- Table 1 | Hyper-parameters of the proposed JanusFlow. Data ratio denotes the proportion of multimodal understanding data, image generation data and text-only data. In the initial 10,000 steps of Stage 2, we apply a data ratio of 30 : 50 : 20 to boost the understanding ability.

Stage 1 Stage 2 Stage 3

Learning Rate 1.0 × 10−4 1 × 10−4 2.0 × 10−5 LR Scheduler Constant Constant Constant Weight Decay 0.0 0.0 0.0 Gradient Clip 1.0 1.0 1.0 Optimizer AdamW (𝛽1 = 0.9, 𝛽2 = 0.95) Warm-up Steps 2,000 2,000 1,000 Training Steps 10,000 390,000 26,000 Batch Size 512 512 256 Data Ratio 50 : 50 : 0 14 : 80 : 6 21 : 70 : 9

where the expectation is taken over all (𝑥𝑐𝑜𝑛, 𝑥𝑟𝑒𝑠) pairs in our multimodal understanding dataset D𝑢𝑛𝑑, computing loss only over tokens in 𝑥𝑟𝑒𝑠.

Rectified Flow Objective. For image generation tasks, 𝑥𝑐𝑜𝑛 consists of text tokens and 𝑥𝑟𝑒𝑠 is the corresponding image. JanusFlow is trained with the rectified flow objective,

𝑔𝑒𝑛,𝑡∼P(𝑡),𝑧0∼N(0,𝐼) ||𝑣𝜃(𝑧𝑡,𝑡 | 𝑥𝑐𝑜𝑛) − (𝑥𝑟𝑒𝑠 − 𝑧0)||2 , (7)

L𝑅𝐹(𝜃) = E𝑥∼D

where 𝑧𝑡 = 𝑡𝑥𝑟𝑒𝑠 + (1 − 𝑡)𝑧0. Following Stable Diffusion 3 [23], we set the time distribution P(𝑡) to the logit-normal distribution. To enable CFG inference, we randomly drop 10% of the text prompts in training.

Representation Alignment Regularization. Recent work [103] has shown that aligning intermediate representations between diffusion transformers and semantic vision encoders enhances diffusion model generalization. Our decoupled vision encoder design enables efficient implementation of this alignment as a regularization term. Specifically, for generation tasks, we align features from the understanding encoder 𝑓𝑒𝑛𝑐 with the LLM’s intermediate features,

sim stop_grad( 𝑓𝑒𝑛𝑐(𝑥𝑟𝑒𝑠)),ℎ𝜑(𝑞𝜃(𝑧𝑡)) , (8)

###### L𝑅𝐸𝑃𝐴(𝜃, 𝜑) = −E𝑥∼D

𝑔𝑒𝑛

where 𝑞𝜃(𝑧𝑡) denotes an intermediate LLM representation given input 𝑧𝑡, and ℎ𝜑 is a small trainable MLP that projects 𝑞𝜃(𝑧𝑡) to dimension 𝐷𝑒𝑛𝑐. The function sim(·, ·) computes the mean of element-wise cosine similarity between embeddings. Before computing the loss, we reshape ℎ𝜑(𝑞𝜃(𝑧𝑡)) to 𝐻𝑔𝑒𝑛 × 𝑊𝑔𝑒𝑛 × 𝐷𝑒𝑛𝑐. To simplify the implementation, we intentionally adjust the configuration of 𝑔𝑒𝑛𝑐 and 𝑔𝑑𝑒𝑐 to ensure 𝐻𝑔𝑒𝑛 = 𝐻𝑖𝑚 and 𝑊𝑔𝑒𝑛 = 𝑊𝑖𝑚. The gradient of L𝑅𝐸𝑃𝐴 is not back-propagated through the understanding encoder. This alignment loss helps the LLM’s internal feature space (given noisy input 𝑧𝑡) align with the understanding encoder’s semantic feature space, thereby improving generation quality when producing images from new random noise and text conditions during inference.

Summary. All three objectives are applied across all training stages. Multimodal understanding tasks use L𝐴𝑅, while image generation tasks employ the combined loss L𝑅𝐹 + L𝑅𝐸𝑃𝐴. Detailed experimental settings are provided in Sec. 4.1.

#### 4. Experiments

We conduct extensive experiments to evaluate the capabilities of JanusFlow in both multimodal understanding and generation tasks. First, we describe our experimental setup and implementation details. Then, we present results on standard benchmarks for multimodal understanding and image generation. Finally, we perform ablation studies to validate our key design choices.

###### 4.1. Experiment Setup and Implementation Details

Our framework builds upon an enhanced version1 of DeepSeek-LLM (1.3B) [7, 64]. The LLM consists of 24 transformer blocks and supports a sequence length of 4,096. In our model, both understanding and generation exploits images of resolution 384.

For multimodal understanding, we leverage SigLIP-Large-Patch/16 [106] as 𝑓𝑒𝑛𝑐. For image generation, we utilize the pre-trained SDXL-VAE [73] for its latent space. The generation encoder 𝑔𝑒𝑛𝑐 comprises a 2 × 2 patchify layer followed by two ConvNeXt [96] blocks and a linear layer. The generation decoder 𝑔𝑑𝑒𝑐 combines two ConvNeXt blocks, a pixel-shuffle layer to upsample the feature map, and a linear layer. Our SigLIP encoder contains ∼ 300M parameters. 𝑔𝑒𝑛𝑐 and 𝑔𝑑𝑒𝑐 are light-weight modules, containing ∼ 70M parameters in total. Table 1 details the hyperparameters for each training stage. In the alignment regularization, we use the LLM features after the 6th block as 𝑞𝜃(𝑧𝑡) and a three-layer MLP as ℎ𝜑. We employ an exponential moving average (EMA) with a ratio of 0.99 to ensure training stability.

For data preprocessing, we deal with understanding and generation data differently. For understanding tasks, we maintain all image information by resizing the long side to the target size and padding the image to squares. For generation tasks, we resize the short side to the target size and apply random square cropping to avoid padding artifacts. During training, multiple sequences are packed to form a single sequence of length 4,096 for training efficiency. Our implementation is based on the HAI-LLM platform [31] using PyTorch [74]. Training was conducted on NVIDIA A100 GPUs, with each model requiring ∼ 1,600 A100 GPU days.

###### 4.2. Training Data Settings

We follow Janus [97] to construct the training data. The data configuration for each training stage is listed below.

Data for Stage 1 and Stage 2. The first two stages of our framework uses three types of data: multimodal understanding data, image generation data and text-only data.

1. Multimodal Understanding Data. This type of data contains several sub-categories: (a) Image caption data. We incorporate caption datasets from [20, 41, 50, 51, 53, 82] and generate additional captions for images from [16, 43] using open-source multimodal understanding models. The names of the datasets are provided in the supplementary materials. The data follows template formats, e.g., “<image>Generate the caption of this picture. <caption>”. (b) Charts and tables. We directly adopt the chart and table data from the training data of DeepSeek-VL [64]. (c) Task data. ShareGPT4V [11] data is utilized to facilitate basic question-answering capabilities during pre-training,

1This version, trained on an expanded text corpus compared to the one in Janus [97], has been demonstrated to possess better performance on multiple-choice benchmarks (e.g., MMBench [63] and SEED Bench [46]). Our preliminary experiments suggest that it has minimal impact on the quality of visual generation.

- Table 2 | Performances on GenEval benchmark. “Gen.” denotes “generation” and “Unified” denotes unified understanding and generation models. Models using external pre-trained generative models are signed with †.

Type Method Params Single Obj. Two Obj. Count. Colors Pos. Color Attri. Overall↑

LlamaGen [86] 0.8B 0.71 0.34 0.21 0.58 0.07 0.04 0.32 LDM [77] 1.4B 0.92 0.29 0.23 0.70 0.02 0.05 0.37 SDv1.5 [77] 0.9B 0.97 0.38 0.35 0.76 0.04 0.06 0.43 PixArt-𝛼 [9] 0.6B 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SDv2.1 [77] 0.9B 0.98 0.51 0.44 0.85 0.07 0.17 0.50

Gen. Only

- DALL-E 2 [76] 6.5B 0.94 0.66 0.49 0.77 0.10 0.19 0.52 Emu3-Gen [95] 8B 0.98 0.71 0.34 0.81 0.17 0.21 0.54 SDXL [73] 2.6B 0.98 0.74 0.39 0.85 0.15 0.23 0.55 IF-XL [17] 4.3B 0.97 0.74 0.66 0.81 0.13 0.35 0.61
- DALL-E 3 [6] - 0.96 0.87 0.47 0.83 0.43 0.45 0.67

Chameleon [88] 34B - - - - - - 0.39 LWM [59] 7B 0.93 0.41 0.46 0.79 0.09 0.15 0.47 SEED-X† [27] 17B 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Show-o [100] 1.3B 0.95 0.52 0.49 0.82 0.11 0.28 0.53 Janus [97] 1.3B 0.97 0.68 0.30 0.84 0.46 0.42 0.61 Transfusion [108] 7.3B - - - - - - 0.63 JanusFlow (Ours) 1.3B 0.97 0.59 0.45 0.83 0.53 0.42 0.63

Unified

structured as “<image><question><answer>”. (d) Interleaved text-image data. This sub-category is sourced from [42, 84].

- 2. Image Generation Data. Our image generation dataset combines high-quality images from [16, 21, 41, 43, 68, 71, 82, 85] and 2 million in-house data. We enhance them with machinegenerated captions using multimodal understanding models. We filter the images in [16, 82] with aspect ratios and aesthetic scores, retaining approximately 20% of the original datasets. 25% of the data contains single-sentence captions. These kind of data assist the model to be able to process short prompts. All the data points are formatted as “<prompt><image>”.
- 3. Text-Only Data. We directly use the text corpus of DeepSeek-LLM [7].

Data for Stage 3. The SFT stage also uses three types of data:

- 1. Multimodal Instruction Data. We leverage the instruction tuning datasets from [29, 33, 35, 47, 65, 80].
- 2. Image Generation Data. We reformat the high-quality text-image pairs from [16, 82, 85] into an instruction format: “User:<user prompt>\n\n Assistant:<image>”.
- 3. Text-Only Data. We directly incorporate the text-only data from [47].

###### 4.3. Evaluation Settings

Image Generation. We evaluate the generated images using both visual quality and semantic accuracy metrics. For visual quality assessment, we employ the Fréchet Inception Distance [30] (FID) metric and compute FID between 30,000 generated images and their corresponding reference images from the MJHQ dataset [48]. The FID computation follows the implementation from GigaGAN [39]. To evaluate semantic accuracy, we utilize two specialized frameworks: GenEval [28] and DPG-Bench [34]. These frameworks are designed to assess whether the

- Table 3 | Performances on DPG-Bench. The methods in this table are all generation-specific models except our method.

Method Global Entity Attribute Relation Other Overall↑

SDv1.5 [77] 74.63 74.23 75.39 73.49 67.81 63.18 PixArt-𝛼 [9] 74.97 79.32 78.60 82.57 76.96 71.11 Lumina-Next [110] 82.82 88.65 86.44 80.53 81.82 74.63 SDXL [73] 83.27 82.43 80.91 86.76 80.41 74.65 Playground v2.5 [48] 83.06 82.59 81.20 84.08 83.50 75.47 Hunyuan-DiT [54] 84.59 80.59 88.01 74.36 86.41 78.87 PixArt-Σ [10] 86.89 82.89 88.94 86.59 87.68 80.54 Emu3-Gen [95] 85.21 86.68 86.84 90.22 83.15 80.60 JanusFlow (Ours) 87.03 87.31 87.39 89.79 88.10 80.09

generated images accurately contain the objects and relationships specified in the input prompts, providing a broad evaluation of the generation capabilities.

Multimodal Understanding. We evaluate JanusFlow’s multimodal understanding abilities across a diverse set of vision-language benchmarks for general understanding capabilities, including POPE [52], MME [24], MMBench [63], SEEDBench [46], VQAv2 [29], GQA [35], MMVet [104], MMMU [105], ChartQA[70] and TextVQA[81]

###### 4.4. Quantitative Results

Image Generation Performances. We report the performances on GenEval, DPG-Bench and MJHQ FID-30k. In Tab. 2, we give comparisons on GenEval including the scores of all the sub-tasks and the overall score. JanusFlow achieves an overall score of 0.63, surpassing the previous unified framework and several generation specific models including SDXL [73] and DALL-E 2 [76]. In Tab. 3, We show results on DPG-Bench and the corresponding comparisons. It is noted that all the methods in Tab. 3 are generation-specific models except our model. The results on GenEval and DPGBench demonstrate the ability of instruction following of our model. We give the comparisons on MJHQ FID-30k in Tab. 4. The images which are sampled to calculate FID are generated with a CFG factor 𝑤 = 2 and a number of sampling steps 30. We sweep the CFG factor and the sampling steps and provide the results in the appendix. Our method achieves the best performance among all the models with 1.3B LLM. The results prove that the rectified flow is able to improve the quality of generated images over autoregressive models such as Janus [97].

Table 4 | Results of MJHQ FID30k. The models which have similar scales to our model are marked with blue background. JanusFlow achieves the best FID among 1.3B models.

Method Params FID↓ LWM [59] 7B 17.77 VILA-U 256 [99] 7B 12.81 VILA-U 384 [99] 7B 7.69 Show-o [100] 1.3B 15.18 Janus [97] 1.3B 10.10 JanusFlow (Ours) 1.3B 9.51

Multimodal Understanding Performances. We show comparisons of our method and other methods including understanding-specific models and unified understanding and generation models in Tab. 5. Our model reaches the best performances among all the models with similar number of parameters and even surpasses multiple understanding-specific methods with larger scales. Our results demonstrate that our method harmonizes autoregressive LLM and rectified flow, achieving satisfying performance in both understanding and generation.

- Table 5 | Comparison with other methods on multimodal understanding benchmarks. “Und.” denotes “understanding” and “Unified” denotes unified understanding and generation models. The models employing external pre-trained generative models are marked with †. The models with LLMs which have similar number of parameters to us are marked with blue background under the line of dashes.

Type Model LLM Param POPE MME-P MMBdev SEED VQAv2test GQA MMMU MM-Vet ChartQA TextVQA

Und. Only

MobileVLM [12] 2.7B 84.9 1288.9 59.6 - - 59.0 - - - 47.5 MobileVLM-V2 [13] 2.7B 84.7 1440.5 63.2 - - 61.1 - - - 57.5 LLaVA-Phi [109] 2.7B 85.0 1335.1 59.8 - 71.4 - - 28.9 - 48.6 LLaVA [58] 7B 76.3 809.6 38.7 33.5 - - - 25.5 - LLaVA-v1.5 [56] 7B 85.9 1510.7 64.3 58.6 78.5 62.0 35.4 31.1 - 58.2 InstructBLIP [15] 7B - - 36.0 53.4 - 49.2 - 26.2 - 50.1 Qwen-VL-Chat [4] 7B - 1487.5 60.6 58.2 78.2 57.5 - - 66.3 61.5 LLaVA-NeXT [57] 7B - 1519.3 - - - - 35.1 - 54.8 Qwen2-VL [94] 7B - - - - - - 54.1 62.0 83.0 84.3 IDEFICS-9B [44] 8B - - 48.2 - 50.9 38.4 - - - 25.9 Emu3-Chat [95] 8B 85.2 - 58.5 68.2 75.1 60.3 31.6 - 68.6 64.7 InstructBLIP [15] 13B 78.9 1212.8 - - - 49.5 - 25.6 - 50.7 LLaVA-v1.5-Phi-1.5 [100] 1.3B 84.1 1128.0 - - 75.3 56.5 30.7 - - MobileVLM [12] 1.4B 84.5 1196.2 53.2 - - 56.1 - - - 41.5 MobileVLM-V2 [13] 1.4B 84.3 1302.8 57.7 - - 59.3 - - - 52.1

Unified

Gemini-Nano-1 [89] 1.8B - - - - 62.7 - 26.3 - 53.6 62.5 LWM [59] 7B 75.2 - - - 55.8 44.8 - 9.6 - VILA-U [99] 7B 85.8 1401.8 - 59.0 79.4 60.8 - 33.5 - 60.8 Chameleon [88] 7B - - - - - - 22.4 8.3 - DreamLLM† [19] 7B - - - - 72.9 - - 36.6 - 41.8 LaVIT† [37] 7B - - - - 66.0 46.8 - - - Emu† [87] 13B - - - - 52.0 - - - - NExT-GPT† [98] 13B - - - - 66.7 - - - - Show-o [100] 1.3B 73.8 948.4 - - 59.3 48.7 25.1 - - Janus [97] 1.3B 87.0 1338.0 69.4 63.7 77.3 59.1 30.5 34.3 - JanusFlow (Ours) 1.3B 88.0 1333.1 74.9 70.5 79.8 60.3 29.3 30.9 64.6 55.5

- Table 6 | Ablation studies. The weights of the modules with † are frozen during training. “Exp.” denotes “experiment”. “FID” in this table is MJHQ FID-10k with CFG factor 𝑤 = 7.5 and 30 steps. “CLIP” denotes CLIP similarity with the backbone of CLIP-ViT-Large-Patch/14. Exp. F is the final configuration for training JanusFlow.

Evaluation Benchmarks REPA Und. Modules Gen. Modules Type POPE↑ VQAv2𝑣𝑎𝑙 ↑ GQA↑ FID↓ CLIP ↑

Model Setting

Exp. ID

Train. Iter.

- A × SigLIP VAE†+ConvNeXt Unified 50,000 82.40 69.62 54.43 19.84 24.94

- B ✓ Shared VAE†+ConvNeXt Unified 50,000 78.13 53.94 44.04 18.05 26.38
- C ✓ VAE+ConvNeXt VAE†+ConvNeXt Unified 50,000 75.30 55.41 44.44 17.53 26.32

- D ✓ SigLIP - Und. Only 13,000 85.03 69.10 54.23 - -
- E ✓ - VAE†+ConvNeXt Gen. Only 37,000 - - - 16.69 26.89

- F ✓ SigLIP VAE†+ConvNeXt Unified 50,000 84.73 69.20 54.83 17.61 26.40

###### 4.5. Ablation Studies

We conduct comprehensive ablation studies to validate the effectiveness of our key design choices. For computational efficiency, all ablation experiments are performed on 256 × 256 resolution images2. All models are trained on our unified pre-training dataset for 50,000 iterations, except for the understanding-only and generation-only variants, which are trained for proportionally fewer iterations based on their respective data ratios in the pre-training phase. The quantitative results of these ablation studies are presented in Tab. 6.

Impact of Representation Alignment. The comparison between Exp. A and F demonstrates the significant benefits of incorporating representation alignment regularization [103] during training. Specifically, models trained with representation alignment show notably lower FID

2The understanding encoders in the 256 × 256-based ablation studies is also SigLIP-Large-Patch/16 which is pre-trained on 256 × 256 images.

[Figure 79]

[Figure 80]

[Figure 81]

A corgi’s head depicted as an explosion of a

Beautiful surreal symbolism the

A lone figure in dark robes ascends

nebula, with vibrant cosmic colors like deep

mesmerizing vision of a Cleopatra Queen

worn stone steps toward a glowing light

purples, blues, and pinks swirling around. The corgi’s fur blends seamlessly into the nebula, with stars and galaxies forming the texture of its fur. Bright bursts of light emanate from its eyes, and faint constellations can be seen in the background, giving the image a surreal, otherworldly feel.

of Egypt, mesmerizing brown eyes, black hair and ethereal features, radiating celestial aura, super high definition, true lifelike color, perfect exposure, razor sharp focus, golden ratio, soft reflections, bokeh effect, fine art photography, cinematic compositing, authentic, professional.

in an ancient temple entrance. Ornate arches, lush greenery, and intricate

carvings adorn the scene, evoking a mystical, high-fantasy atmosphere reminiscent of works by artists like Randy Vargas, with cinematic lighting and epic storytelling.

- Figure 4 | Image generation results of JanusFlow. Our model can generate high-quality images that are semantically consistent with text prompts.

scores on MJHQ dataset and higher CLIP scores, indicating simultaneous improvements in both image quality and semantic alignment. Importantly, our architecture differs from previous studies [66, 72] examined in [103] due to our incorporation of LLM and an additional skip connection between 𝑔𝑒𝑛𝑐 and 𝑔𝑑𝑒𝑐. The effectiveness of representation alignment in our modified architecture suggests its broad applicability and generalization capability across different network structures.

Impact of Decoupling Visual Encoders. e efficacy of using powerful pre-trained visual encoders in multimodal understanding. The comparison among Exp. B, C, and F demonstrates the advantages of using separate visual encoders for understanding and generation tasks. In Exp. B, following a design similar to Transfusion [108], we implement shared ConvNeXt blocks in the SDXL-VAE latent space for both understanding and generation encoders. Exp. C employs separate encoders with identical architectures and initialization parameters, but trained independently. The performance differences between these configurations validate the necessity of decoupled visual encoders in improving our unified model’s capabilities. Moreover, the superior results in Exp. C and F highlight the benefits of leveraging pre-trained semantic visual encoders for multimodal understanding tasks.

Fair Comparison with Understanding / Generation-Only Models. To establish meaningful benchmarks, we evaluate task-specific models trained under identical conditions - using the same pre-training dataset, infrastructure, and hyperparameters. Exp. D and E represent these specialized models, trained with data volumes matching the unified models in Tab. 6. The minimal performance gap between Exp. F and these task-specific baselines demonstrates that our unified framework successfully integrates understanding and generation capabilities without significant compromise in either task’s performance.

###### 4.6. Qualitative Results

We present qualitative evaluations of our method for both image generation and understanding tasks. Fig. 1(b) and Fig. 4 showcases the image generation capabilities of JanusFlow. These results demonstrate both the high visual quality of our generated images and our framework’s ability

[Figure 82]

[Figure 83]

[Figure 84]

User: Describe this image.

###### JaunsFlow (Ours): Kid's

Favourite Fruits is a bar graph.

The x-axis shows the fruits. The y-axis shows the number of kids who like the fruit. The orange is

User: How many penguins in the picture?

User: What are the kinds of fruits in this picture?

the most popular fruit. The

JaunsFlow (Ours): There are 11 penguins in the picture.

JaunsFlow (Ours): The fruits in the picture are banana,

banana is the least popular fruit.

strawberry, mango, persimmon, blueberry, and lime.

- Figure 5 | Visual Understanding with JanusFlow. Our model effectively handles various visual understanding tasks, such as question answering, plot interpretation and object counting.

to faithfully follow diverse instructions. For multimodal understanding, Fig. 5 presents example conversations that show our model’s understanding capabilities across various scenarios. These interactions demonstrate the model’s ability to understand and reason about visual content in natural language dialogues. Additional qualitative examples showcasing the versatility and effectiveness of JanusFlow are provided in the appendix.

#### 5. Conclusion

We present JanusFlow, a unified framework that successfully harmonizes autoregressive and rectified flow models for multimodal understanding and generation tasks. Our extensive experiments demonstrate that this unification achieves comparable performance to task-specific models. The successful integration of these fundamentally different model architectures not only addresses current challenges in multimodal learning but also opens new possibilities for future research in training unified models.

#### References

- [1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2022.

- [3] M. Albergo and E. Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In Proc. Int’l Conf. Learning Representations, 2023.

- [4] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. QwenVL: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

- [5] F. Bao, S. Nie, K. Xue, Y. Cao, C. Li, H. Su, and J. Zhu. All are worth words: A ViT backbone for diffusion models. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2023.

- [6] J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo, et al. Improving image generation with better captions. Computer Science, 2023.

- [7] X. Bi, D. Chen, G. Chen, S. Chen, D. Dai, C. Deng, H. Ding, K. Dong, Q. Du, Z. Fu, et al. DeepSeek LLM: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.

- [8] S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. Lundberg, et al. Sparks of artificial general intelligence: Early experiments with GPT-4. arXiv preprint arXiv:2303.12712, 2023.

- [9] J. Chen, J. Yu, C. Ge, L. Yao, E. Xie, Y. Wu, Z. Wang, J. Kwok, P. Luo, H. Lu, et al. PixArtalpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

- [10] J. Chen, C. Ge, E. Xie, Y. Wu, L. Yao, X. Ren, Z. Wang, P. Luo, H. Lu, and Z. Li. PixArtSigma: Weak-to-strong training of diffusion transformer for 4K text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.

- [11] L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin. ShareGPT4V: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.

- [12] X. Chu, L. Qiao, X. Lin, S. Xu, Y. Yang, Y. Hu, F. Wei, X. Zhang, B. Zhang, X. Wei, et al. MobileVLM: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023.

- [13] X. Chu, L. Qiao, X. Zhang, S. Xu, F. Wei, Y. Yang, X. Sun, Y. Hu, X. Lin, B. Zhang, et al. MobileVLM V2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024.

- [14] K. Crowson, S. A. Baumann, A. Birch, T. M. Abraham, D. Z. Kaplan, and E. Shippole. Scalable high-resolution pixel-space image synthesis with hourglass diffusion transformers. In Proc. Int’l Conf. Machine Learning, 2024.

- [15] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2023.

- [16] dclure. LAION-Aesthetics-UMAP, 2022. URL https://huggingface.co/datasets/ dclure/laion-aesthetics-12m-umap.
- [17] DeepFloyd. DeepFloyd IF, 2023. URL https://huggingface.co/DeepFloyd/IF-I

-XL-v1.0.

- [18] P. Dhariwal and A. Nichol. Diffusion models beat GANs on image synthesis. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2021.

- [19] R. Dong, C. Han, Y. Peng, Z. Qi, Z. Ge, J. Yang, L. Zhao, J. Sun, H. Zhou, H. Wei, et al. DreamLLM: Synergistic multimodal comprehension and creation. In Proc. Int’l Conf. Learning Representations, 2024.

- [20] echo840. Detailed caption, 2023. URL https://huggingface.co/datasets/echo84 0/Detailed_Caption.
- [21] B. Egan, A. Redden, XWAVE, and SilentAntagonist. DALLE-3 1 million+ high quality captions, 2024. URL https://huggingface.co/datasets/ProGamerGov/synthe tic-dataset-1m-dalle3-high-quality-captions.
- [22] P. Esser, R. Rombach, and B. Ommer. Taming transformers for high-resolution image synthesis. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2021.

- [23] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proc. Int’l Conf. Machine Learning, 2024.

- [24] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang, X. Zheng, K. Li, X. Sun, Y. Wu, and R. Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2024.

- [25] Y. Ge, Y. Ge, Z. Zeng, X. Wang, and Y. Shan. Planting a SEED of vision in large language model. arXiv preprint arXiv:2307.08041, 2023.

- [26] Y. Ge, S. Zhao, Z. Zeng, Y. Ge, C. Li, X. Wang, and Y. Shan. Making LLaMA SEE and draw with SEED tokenizer. arXiv preprint arXiv:2310.01218, 2023.

- [27] Y. Ge, S. Zhao, J. Zhu, Y. Ge, K. Yi, L. Song, C. Li, X. Ding, and Y. Shan. SEED-X: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

- [28] D. Ghosh, H. Hajishirzi, and L. Schmidt. GenEval: An object-focused framework for evaluating text-to-image alignment. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2024.

- [29] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the v in VQA matter: Elevating the role of image understanding in visual question answering. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2017.

- [30] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. Proc. Annu. Conf. Neural Inf. Process. Systems, 2017.

- [31] High-flyer. HAI-LLM: Efficient and lightweight training tool for large models, 2023. URL https://www.high-flyer.cn/en/blog/hai-llm.
- [32] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2020.

- [33] Y.-C. Hsiao, F. Zubach, G. Baechler, V. Carbune, J. Lin, M. Wang, S. Sunkara, Y. Zhu, and J. Chen. ScreenQA: Large-scale question-answer pairs over mobile app screenshots. arXiv preprint arXiv:2209.08199, 2022.

- [34] X. Hu, R. Wang, Y. Fang, B. Fu, P. Cheng, and G. Yu. ELLA: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

- [35] D. A. Hudson and C. D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2019.

- [36] Y. Jin, Z. Sun, N. Li, K. Xu, H. Jiang, N. Zhuang, Q. Huang, Y. Song, Y. Mu, and Z. Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.

- [37] Y. Jin, K. Xu, L. Chen, C. Liao, J. Tan, Q. Huang, C. Bin, C. Song, D. ZHANG, W. Ou, et al. Unified language-vision pretraining in llm with dynamic discrete visual tokenization. In Proc. Int’l Conf. Learning Representations, 2024.

- [38] B. Jing, B. Berger, and T. Jaakkola. AlphaFold meets flow matching for generating protein ensembles. In Proc. Int’l Conf. Machine Learning, 2024.

- [39] M. Kang, J.-Y. Zhu, R. Zhang, J. Park, E. Shechtman, S. Paris, and T. Park. Scaling up GANs for text-to-image synthesis. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2023.

- [40] S. Kim, K. Shih, J. F. Santos, E. Bakhturina, M. Desta, R. Valle, S. Yoon, B. Catanzaro, et al. P-Flow: a fast and data-efficient zero-shot tts through speech prompting. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2024.

- [41] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al. Segment anything. In Proc. IEEE Int. Conf. Comput. Vision, 2023.

- [42] M. Koupaee and W. Y. Wang. WikiHow: A large scale text summarization dataset. arXiv preprint arXiv:1810.09305, 2018.

- [43] A. Kuznetsova, H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov, M. Malloci, A. Kolesnikov, et al. The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale. Int’l Journal of Computer Vision, 2020.

- [44] H. Laurençon, D. van Strien, S. Bekman, L. Tronchon, L. Saulnier, T. Wang, S. Karamcheti, A. Singh, G. Pistilli, Y. Jernite, et al. Introducing IDEFICS: An open reproduction of state-of-the-art visual language model, 2023, 2023. URL https://huggingface.co/b log/idefics.

- [45] M. Le, A. Vyas, B. Shi, B. Karrer, L. Sari, R. Moritz, M. Williamson, V. Manohar, Y. Adi, J. Mahadeokar, et al. VoiceBox: Text-guided multilingual universal speech generation at scale. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2024.

- [46] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan. SEED-Bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.

- [47] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, Y. Li, Z. Liu, and C. Li. LLaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

- [48] D. Li, A. Kamko, E. Akhgari, A. Sabet, L. Xu, and S. Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.

- [49] J. Li, D. Li, S. Savarese, and S. Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Proc. Int’l Conf. Machine Learning, 2023.

- [50] L. Li, Y. Wang, R. Xu, P. Wang, X. Feng, L. Kong, and Q. Liu. Multimodal arXiv: A dataset for improving scientific comprehension of large vision-language models. In Annual Meeting of the Association for Computational Linguistics, 2024.

- [51] X. Li, F. Zhang, H. Diao, Y. Wang, X. Wang, and L.-Y. Duan. DenseFusion-1M: Merging vision experts for comprehensive multimodal perception. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2024.

- [52] Y. Li, Y. Du, K. Zhou, J. Wang, X. Zhao, and J.-R. Wen. Evaluating object hallucination in large vision-language models. In Proc. Conf. on Empirical Methods in Natural Language Process., 2023.

- [53] Z. Li, X. Yang, K. Choi, W. Zhu, R. Hsieh, H. Kim, J. H. Lim, S. Ji, B. Lee, X. Yan, et al. MMSci: A multimodal multi-discipline dataset for phd-level scientific comprehension. In AI for Accelerated Materials Design, 2024.

- [54] Z. Li, J. Zhang, Q. Lin, J. Xiong, Y. Long, X. Deng, Y. Zhang, X. Liu, M. Huang, Z. Xiao, et al. Hunyuan-DiT: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024.

- [55] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In Proc. Int’l Conf. Learning Representations, 2023.

- [56] H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2024.

- [57] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge, 2024. URL https://llava-vl.github.io/ blog/2024-01-30-llava-next/.
- [58] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2024.

- [59] H. Liu, W. Yan, M. Zaharia, and P. Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024.

- [60] Q. Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.

- [61] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In Proc. Int’l Conf. Learning Representations, 2023.

- [62] X. Liu, X. Zhang, J. Ma, J. Peng, et al. InstaFlow: One step is enough for high-quality diffusion-based text-to-image generation. In Proc. Int’l Conf. Learning Representations, 2024.

- [63] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. MMBench: Is your multi-modal model an all-around player? In Proc. European Conf. Computer Vision, 2024.

- [64] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, H. Yang, et al. DeepSeek-VL: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.

- [65] P. Lu, L. Qiu, J. Chen, T. Xia, Y. Zhao, W. Zhang, Z. Yu, X. Liang, and S.-C. Zhu. IconQA: A new benchmark for abstract diagram understanding and visual language reasoning. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2021.

- [66] N. Ma, M. Goldstein, M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, and S. Xie. SiT: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024.

- [67] Y. Ma, H. Yang, W. Wang, J. Fu, and J. Liu. Unified multi-modal latent diffusion for joint subject and text conditional image generation. arXiv preprint arXiv:2303.09319, 2023.

- [68] madebyollin. Megalith-10M, 2024. URL https://huggingface.co/datasets/made byollin/megalith-10m.
- [69] B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

- [70] A. Masry, X. L. Do, J. Q. Tan, S. Joty, and E. Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Annual Meeting of the Association for Computational Linguistics, 2022.

- [71] mehdidc. YFCC-15M, 2024. URL https://huggingface.co/datasets/mehdidc/yf cc15m.
- [72] W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proc. IEEE Int. Conf. Comput. Vision, 2023.

- [73] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In Proc. Int’l Conf. Learning Representations, 2024.

- [74] PyTorch-Contributors. PyTorch, 2024. URL https://pytorch.org.
- [75] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In Proc. Int’l Conf. Machine Learning, 2021.

- [76] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125, 2022.

- [77] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2022.

- [78] L. Ruan, Y. Ma, H. Yang, H. He, B. Liu, J. Fu, N. J. Yuan, Q. Jin, and B. Guo. MM-Diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2022.

- [79] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2022.

- [80] S. Shah, A. Mishra, N. Yadati, and P. P. Talukdar. KVQA: Knowledge-aware visual question answering. In Proc. AAAI Conf. on Artificial Intelligence, 2019.

- [81] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach. Towards VQA models that can read. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2019.

- [82] V. Singla, K. Yue, S. Paul, R. Shirkavand, M. Jayawardhana, A. Ganjdanesh, H. Huang, A. Bhatele, G. Somepalli, and T. Goldstein. From pixels to prose: A large dataset of dense image captions. arXiv preprint arXiv:2406.10328, 2024.

- [83] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Scorebased generative modeling through stochastic differential equations. In Proc. Int’l Conf. Learning Representations, 2021.

- [84] K. Srinivasan, K. Raman, J. Chen, M. Bendersky, and M. Najork. WIT: Wikipedia-based image text dataset for multimodal multilingual machine learning. In Proc. ACM SIGIR Conf. Research and Develop. in Info. Retrieval, 2021.

- [85] K. Sun, J. Pan, Y. Ge, H. Li, H. Duan, X. Wu, R. Zhang, A. Zhou, Z. Qin, Y. Wang, et al. JourneyDB: A benchmark for generative image understanding. In Proc. Annu. Conf. Neural Inf. Process. Systems, 2024.

- [86] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan. Autoregressive model beats diffusion: LLaMA for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

- [87] Q. Sun, Q. Yu, Y. Cui, F. Zhang, X. Zhang, Y. Wang, H. Gao, J. Liu, T. Huang, and X. Wang. Generative pretraining in multimodality. In Proc. Int’l Conf. Learning Representations, 2024.

- [88] C. Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

- [89] G. Team. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [90] S. Tong, E. Brown, P. Wu, S. Woo, M. Middepogu, S. C. Akula, J. Yang, S. Yang, A. Iyer, X. Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024.

- [91] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [92] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. LLaMA 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

- [93] C. N. Vasconcelos, A. Rashwan, A. Waters, T. Walker, K. Xu, J. Yan, R. Qian, Y. Li, S. LUO, Y. Onoe, et al. Greedy growing enables high-resolution pixel-based diffusion models. Transactions on Machine Learning Research, 2024.

- [94] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

- [95] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

- [96] S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie. ConvNeXt v2: Co-designing and scaling ConvNets with masked autoencoders. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2023.

- [97] C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024.

- [98] S. Wu, H. Fei, L. Qu, W. Ji, and T.-S. Chua. NExT-GPT: Any-to-any multimodal LLM. In Proc. Int’l Conf. Machine Learning, 2024.

- [99] Y. Wu, Z. Zhang, J. Chen, H. Tang, D. Li, Y. Fang, L. Zhu, E. Xie, H. Yin, L. Yi, et al. VILA-U: A unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

- [100] J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

- [101] H. Ye, D.-A. Huang, Y. Lu, Z. Yu, W. Ping, A. Tao, J. Kautz, S. Han, D. Xu, P. Molchanov, et al. X-VILA: Cross-modality alignment for large language model. arXiv preprint arXiv:2405.19335, 2024.

- [102] L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, A. Gupta, X. Gu, A. G. Hauptmann, et al. Language model beats diffusion-tokenizer is key to visual generation. In Proc. Int’l Conf. Learning Representations, 2024.

- [103] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

- [104] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang. MM-Vet: Evaluating large multimodal models for integrated capabilities. In Proc. Int’l Conf. Machine Learning, 2024.

- [105] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2024.

- [106] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In Proc. IEEE Int. Conf. Comput. Vision, 2023.

- [107] C. Zhao, Y. Song, W. Wang, H. Feng, E. Ding, Y. Sun, X. Xiao, and J. Wang. MonoFormer: One transformer for both diffusion and autoregression. arXiv preprint arXiv:2409.16280, 2024.

- [108] C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

- [109] Y. Zhu, M. Zhu, N. Liu, Z. Ou, X. Mou, and J. Tang. LLaVA-Phi: Efficient multi-modal assistant with small language model. arXiv preprint arXiv:2401.02330, 2024.

- [110] L. Zhuo, R. Du, H. Xiao, Y. Li, D. Liu, R. Huang, W. Liu, L. Zhao, F.-Y. Wang, Z. Ma, et al. Lumina-Next: Making Lumina-T2X stronger and faster with Next-DiT. arXiv preprint arXiv:2406.18583, 2024.

Appendix

- A. Performance Analysis of 256 Resolution Model

We trained our model at two resolutions: 256 × 256 and 384 × 384. The main paper presents results from the 384 × 384 model as our primary results. Here, we provide a comprehensive evaluation of the 256 × 256 model’s performance. The visual understanding performances are presented in Tab. 1. The generation capabilities are evaluated using GenEval [28], DPGBenchmark [34], and MJHQ FID-30k [48], with results shown in Tab. 2 and 3.

- Table 1 | Results on visual understanding tasks.

Model LLM Params POPE↑ MME-P↑ MMB𝑑𝑒𝑣 ↑ SEED↑ VQAv2𝑡𝑒𝑠𝑡↑ GQA↑ MM-Vet↑

JanusFlow 256 1.3B 85.3 1203.0 71.9 67.6 76.3 58.4 27.4 JanusFlow 384 1.3B 88.0 1333.1 74.9 70.5 79.8 60.3 30.9

- Table 2 | Results on GenEval [28].

Method LLM Params Single Obj. Two Obj. Count. Colors Pos. Color Attri. Overall↑

JanusFlow 256 1.3B 0.98 0.73 0.54 0.83 0.63 0.53 0.70 JanusFlow 384 1.3B 0.97 0.59 0.45 0.83 0.53 0.42 0.63

- Table 3 | Results on DPG-Bench [34] and MJHQ FID-30k [48].

Method

Global Entity AttributeDPG-BenchRelation↑ Other Overall MJHQ FID-30k↓

JanusFlow 256 91.20 88.83 88.00 87.60 89.53 81.23 12.70 JanusFlow 384 87.03 87.31 87.39 89.79 88.10 80.09 9.51

As expected, the 256×256 model shows slightly lower performance compared to the 384×384 model on visual understanding metrics due to its reduced resolution. Interestingly, however, the 256 × 256 model outperforms its higher-resolution counterpart on GenEval and DPG-Bench benchmarks specifically designed to evaluate instruction following capabilities and semantic accuracy. This superior performance on semantic tasks can be attributed to the model’s better control over lower-resolution images, where reduced visual complexity allows for more precise semantic manipulation.

- B. Details of the Datasets

The datasets used in the pre-training stage for understanding include DetailedCaption [20], SAM [41], arXivQA[50], DenseFusion-1M [51], MMSci[53], PixelProse [82], re-captioned LAIONAesthetics [16], re-captioned Open Images V4 [43], ShareGPT4V [11], WikiHow [42] and WIT [84]. The datasets used in the pre-training stage for generation include re-captioned LAION-Aesthetics [16], DALL-E 3 1M [21], SAM [41], Open Images V4 [43], Megalith-10M [68], YFCC-15M [71], PixelProse[82] and JourneyDB [85].

28.0

27.5

- 10

- 11

- 12

- 13

27.0

26.5

CLIPSimilarity

MJHQFID-30k

26.0

25.5

25.0

24.5

24.0

1 2 3 4 5 6 7 8 9 CFG Factor

(a) Results of varying CFG Factors

26.640

9.70

26.635

9.65

9.60

CLIPSimilarity

MJHQFID-30k

26.630

9.55

26.625

9.50

26.620

9.45

9.40

26.615

25 30 35 40 45 50 Number of Sampling Step

(b) Results of Varying Numbers of Sampling Steps

- Figure 1 | Results of varying CFG factors and numbers of sampling steps. In Fig. (a), the number of sampling steps is set to 30. In Fig. (b), the CFG factor is set to 2.

0 10 20 30 40 50 Training Iterations / K Iteration

20

40

60

80

100

MJHQFID-30k

18

20

22

24

26

CLIPSimilarity

w/ REPA

w/o REPA

- Figure 2 | The FID and CLIP similarity during the first 50,000 iterations.

#### C. Analysis of CFG Factor and Sampling Steps

We investigate the impact of two key generation parameters: the Classifier-Free Guidance (CFG) factor and the number of sampling steps. While our main results use 𝑤 = 2 for CFG and 30 sampling steps to calculate FID, here we present a comprehensive analysis of these hyperparameters. Fig. 1(a) shows the effect of varying CFG factors while maintaining 30 sampling steps. The results reveal an optimal CFG value for FID scores, while CLIP [75] similarity continues to improve with increasing CFG values, consistent with findings from previous work [73]. Fig. 1(b) demonstrates the impact of different sampling steps while maintaining a CFG factor of 2. The number of sampling steps shows relatively minor influence on performance. Our choice of 30 steps in the main paper represents a balance between generation quality and computational efficiency.

#### D. Details of REPA Ablation

We provide the FID and CLIP similarity of the first 50,000 training iterations of the pre-train stage in Fig. 2 with and without representation alignment regularization. The gap between the two models demonstrates the benefits of using representation alignment regularization.

#### E. Additional Qualitative Results

Additional qualitative examples for both understanding and generation tasks are presented in Fig. 3 and Fig. 4, respectively. The understanding examples demonstrate JanusFlow’s diverse capabilities, including code generation, person identification, character recognition, and visual reasoning. For image generation, our model exhibits strong performance in both visual quality and semantic alignment with input prompts.

[Figure 85]

[Figure 86]

User: Explain why this image is funny. JanusFlow: This image is likely to be a humorous or artistic rendition of the famous

painting "The Mona Lisa" with a dog

instead of a human. The dog's expression and pose are similar to the iconic pose of the original painting, which adds a playful and unexpected twist to the artwork.

User: Please write a python code to draw this plot. JanusFlow: Here is the Python code to draw this plot:

User: What are the texts in this passage? JanusFlow: To increase the accessibility of this powerful model class and at the same time reduce its significant resource consumption, a method is needed that

```python import matplotlib.pyplot as plt

reduces the computational complexity for

# Data values = [3, 7, 5, 2]

[Figure 87]

both training and sampling. Reducing the computational demands of DMs without impairing their performance is, therefore, key to enhance their accessibility.

# Create the bar chart plt.figure(figsize=(10, 6)) plt.bar(range(len(values)), values)

# Title plt.title('Simple Bar Chart')

[Figure 88]

# Labels

User: Who is this in the figure? JanusFlow: George W. Bush.

plt.xlabel('Categories') plt.ylabel('Values')

# Show the plot plt.show() ```

- Figure 3 | More multimodal understanding cases.

[Figure 89]

[Figure 90]

[Figure 91]

Massive cathedral church, battle between Heaven and hell, church on fire, 8k hyper real ultra sharp renaissance by Francisco Goya.

A handsome 24-year-old boy in the middle with sky color background wearing eye glasses, it's super detailed with anime style.

Happy dreamy owl monster sitting on a tree branch, colorful glittering particles, forest background, detailed feathers.

[Figure 92]

[Figure 93]

[Figure 94]

A vivid depiction of the Northern Lights dancing above the snow-covered mountains in Iceland, casting a

A dark, high-contrast render of a psychedelic Tree of Life glowing brilliantly, illuminating swirling dust

A man wearing Fedora hat with mafia style,

realistic photography, intricate details, magical lighting, vibrant background, complex textures, rich colors, realistic style, front-facing view.

mesmerizing glow across the sky.

particles in a mystical, cavernous setting.

[Figure 95]

[Figure 96]

[Figure 97]

The image features a mushroom growing on grassy ground amidst fallen leaves. Their caps are light brownish-white with visible gills underneath; the stems appear dark and sturdy. In the background, there's an out-of-focus scene that includes greenery and possibly some structures or trees shrouded by mist or fog,

The image captures a vast ocean view at either sunrise or sunset, with soft pink hues near the horizon blending into darker clouds above. Waves crash against rugged black rocks on the right, where water flows down onto smaller stones below. In the foreground, dry grass contrasts with the

A serene Chinese ink painting depicts a tranquil mountain village. Simple homes nestle at the foot of misty peaks, while a gentle river winds through the village. Bamboo and pine trees dot the landscape. The minimalist brushstrokes reflect a

giving it a serene yet slightly eerie atmosphere.

smooth sea surface. The scene feels tranquil

harmonious relationship between

This photograph employs shallow depth of field to emphasize the mushrooms while blurring the surroundings for artistic effect.

but also reveals the raw power of nature through the interaction between the dynamic waves and the solid land.

nature and human life, capturing the peaceful essence of the scene with elegant simplicity.

- Figure 4 | More text-to-image generation results.

