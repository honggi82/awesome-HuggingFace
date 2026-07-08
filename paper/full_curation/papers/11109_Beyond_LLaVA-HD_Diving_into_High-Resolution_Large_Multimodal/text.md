# arXiv:2406.08487v3[cs.CV]14Jun2024

[Figure 1]

## Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models

Yi-Fan Zhang1,2, Qingsong Wen3, Chaoyou Fu, Xue Wang4, Zhang Zhang1,2, Liang Wang1,2, Rong Jin5 1State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS), Institute of Automation 2School of Artificial Intelligence, University of Chinese Academy of Sciences (UCAS) 3Squirrel AI Learning; 4Alibaba Group; 5Meta AI https://github.com/yfzhang114/SliME

### Abstract

Seeing clearly with high resolution is a foundation of Large Multimodal Models (LMMs), which has been proven to be vital for visual perception and reasoning. Existing works usually employ a straightforward resolution upscaling method, where the image consists of global and local branches, with the latter being the sliced image patches but resized to the same resolution as the former. This means that higher resolution requires more local patches, resulting in exorbitant computational expenses, and meanwhile, the dominance of local image tokens may diminish the global context. In this paper, we dive into the problems and propose a new framework as well as an elaborate optimization strategy. Specifically, we extract contextual information from the global view using a mixture of adapters, based on the observation that different adapters excel at different tasks. With regard to local patches, learnable query embeddings are introduced to reduce image tokens, the most important tokens accounting for the user question will be further selected by a similarity-based selector. Our empirical results demonstrate a ‘less is more’ pattern, where utilizing fewer but more informative local image tokens leads to improved performance. Besides, a significant challenge lies in the training strategy, as simultaneous end-to-end training of the global mining block and local compression block does not yield optimal results. We thus advocate for an alternating training way, ensuring balanced learning between global and local aspects. Finally, we also introduce a challenging dataset with high requirements for image detail, enhancing the training of the local compression layer. The proposed method, termed LMM with Sophisticated Tasks, Local image compression, and Mixture of global Experts (SliME), achieves leading performance across various benchmarks with only 2 million training data.

### 1 Introduction

In the past years, we are fortunate to witness a great flourish of LMMs [2, 10, 34]. However, they still struggle with complex visual perception [58, 14] and reasoning tasks [27, 53]. Empirical studies have shown that employing higher resolutions is a good solution [3, 32, 29, 41]. Approaches like LLaVA-Next [33] segment high-resolution images into multiple patches, encoding each one independently before concatenating all local patch tokens with the original global image tokens, albeit at an escalated computational cost. The other models like Monkey [29] and LLaVA-UHD [51] also split images into patches, but subsequently compress them to avoid redundant tokens. In such cases, for high-resolution images, the local image tokens dominate the feature space. For example, in a 1024 × 1024 image divided into 9 patches, the global image token accounts for only 1/10.

In contrast, our core idea posits that global information should be prioritized, thus we aim to extract and retain as much global context as possible while enhancing it with local image details. In this study, we initially segment the images into patches according to their resolution. The image tokens are then categorized into two groups: the global view and local patches. For the former, we preserve the token count to retain all contextual information and utilize a mixture of adapters to further explore global context. As displayed in Fig. 2 (b), we employ a Multilayer Perceptron (MLP) to project image features into the feature space of the LLM, and a set of learnable queries named qformer are employed to extract crucial global information. Softly mixing outputs from the two adapters aids the LLM in comprehending the global context more effectively. Considering the local patches, they provide additional image details but are compressed using a querying transformer to mitigate computational costs. As shown in Fig. 2 (d), a text-guided router is further proposed to select the most relevant local image tokens corresponding to the input instruction or question, thereby avoiding excessive image tokens and focusing on pertinent image information.

At the same time, we find that it is challenging to train the global projection and local compression simultaneously. The simplicity of the projection layer makes it easy to train, but also causes the model to degenerate rapidly due to over-reliance on global features and neglect of local counterparts. We formalize this as a bi-linear problem and theoretically show that simultaneously updating both blocks does not converge to optimal results. Instead, we propose to alternatively train the global projection block and local compression block, by which we ensure that both global and local features are effectively learned and utilized.

The training is known to be data-driven. The current data instances primarily stem from real-world captions, general QA, and a limited number of real-world conversations sampled from robust LLMs. Most of these instances revolve around basic perception, recognition, and reasoning tasks, such as understanding relationships among objects. However, there are two notable flaws: firstly, the tasks are not challenging enough and largely lack intricate visual reasoning tasks; secondly, many questions only pertain to specific objects or actions, neglecting the need for all image details. This limitation hampers the full utilization of the capabilities offered by our high-resolution framework. To this end, this paper meticulously gathers and filters datasets to create the Science and Mathematical Reasoning dataset (SMR), which encompasses nine challenging tasks spanning natural science, mathematical problems, and scientific chart comprehension. Some of these tasks provide complete reasoning paths, compelling the model to articulate the entire reasoning process. Importantly, many images in the SMR dataset contain rich annotations. Completing such intricate reasoning tasks necessitates a thorough understanding of image details, which will greatly benefit the training of our framework.

LMM with Sophisticated Tasks, Local image augmentation, and Mixture of global Experts (SliME) can be readily instantiated with a range of LLMs. Extensive empirical studies validate the effectiveness of our proposed method. Remarkably, our approach achieves leading performance across various settings, even matching the performance of well-established models such as Gemini Pro [46] and Qwen-VL-Plus [3] in about 10 benchmarks with only 8B LLM and 2 million data. These results underscore the potential of SliME to set new benchmarks, highlighting its advanced capabilities.

### 2 Method

We delineate our method aimed at enhancing LMMs’ image understanding capabilities in this section. We utilize adaptive slicing to scale input resolution, and refine global context via a soft mixture of experts. Additionally, we compress local features using a query transformer architecture 1, select features optimally with a text-guided router, and employ an alternating training scheme to optimize the bilinear optimization problem. These strategies collectively improve both the computational efficiency and performance of LMMs. Furthermore, we introduce the SMR dataset, known for its challenging nature and high demand for understanding image details, making it an ideal choice for training high-resolution frameworks.

1In this context, the abbreviation ’qformer’ refers to query former, where we utilize learnable query embeddings as described in previous works [2, 17], rather than employing the Qformer approach [10]

#### 2.1 Refining Global Context with a Soft Mixture of Experts

Scaling Input Resolution by Adaptive Slicing. Initially, we explore various grid options for slicing images, similar to LLaVA-Next, but with finer granularity (see Fig. 2(a)). We investigate resolutions ranging from 336 × (m,n) with m = 1,n = 1 to m = 6,n = 6 to determine the most efficient option. To provide a global context, we pad and resize the image to a uniform size of 336 × 336 and concatenate it with local features. For images with shapes W and H, we iterate through all available partition strategies. For instance, when using the strategy m ∗ n, the resize scale can be calculated as s = min{m ∗ 336/W,n ∗ 336/H}. The utilized resolution after scaling will be min{W ∗ H,W ∗ s ∗ H ∗ s}, and the wasted resolution will be m ∗ 336 ∗ n ∗ 336 − min{W ∗ H,W ∗ s ∗ H ∗ s}. We select the best partition by maximizing the utilized resolution and minimizing the wasted resolution when the utilized resolution is the same.

Why not Compress Global Image Tokens for Efficiency? Our approach is inspired by empirical observations, consistent with previous findings [56]: when employing attention-based models as adapters to reduce tokens or bridge the modality gap, a more intricate hyper-parameter search may be required to achieve performance comparable to simpler MLP. As depicted in Fig. 1, replacing the MLP adapter of LLaVA-v1.5 with the query former of the same number of tokens yields significantly inferior performance on most benchmarks. A simpler projector compels the LLM to better understand visual inputs, leading to enhanced generalization [30]. Consequently, we refrain from reducing token numbers for global images and instead preserve all global information through simple projection.

Global Context Refinement by Soft Mixture of Experts. Although query former is inferior to MLP on most benchmarks, the learnable query embeddings and attention mechanism allow for a different feature selection strategy, and on some benchmarks such as ScienceQA (SQA) [38], query former achieves better performance. Building on the insights from our analysis, we propose a novel approach to refine global context features by leveraging the strengths of both MLP and query former adapters. Specifically, we employ a noisy Mixture of Experts (MOE) framework to combine the benefits of these two types of frameworks. In this framework, for feature x from the vision encoder, a learned gating network G determines the weights for two adapters: G(x)0fm(x) + G(x)1fq(x). The gating network learns to dynamically adjust the importance of each adapter based on the input feature. To prevent the gating network from predominantly activating the same adapter, we introduce learnable noise during training2. This is achieved through the following equation: G(x) = Softmax {(x · Wg)i + Normal(0,1) · Softplus((x · Wnoise)i)}2i=1 .

#### 2.2 Local Feature Mining with Compression and Selection

Local Feature Compression. In our approach to local feature compression, we implement a query former architecture leveraging Nq query embeddings, denoted as MQ ∈ RN

q×DI. Here, DI represents the dimensionality of the image features obtained from the vision encoder. Notably, we strategically set Nq to be smaller than the original token count derived from the vision encoder. This deliberate reduction serves to alleviate the computational burden while preserving essential information. Through the application of an attention mechanism, we orchestrate the interaction between these query embeddings and the local image features. Here, the learnable embeddings act as the query for attention, directing the model’s focus towards pertinent aspects of the local features. The resultant outcome is a condensed representation of local features, consisting of Nq tokens. By judiciously balancing computational efficiency with information retention, our compression strategy enhances the scalability and effectiveness of LMMs in handling diverse tasks.

###### LLaVA qformer qformer w alt Local Local w alt

Figure 1: Significance of Alternating Training. The reported values represent the performance ratio of baselines to the best one: Local with alternating training.

2As mentioned earlier, the training of MLP is easy to converge quickly, potentially causing the gated network to assign higher weight to MLP, hindering full training of query former.

Text-Guided Router. Our approach seeks to further alleviate computational burden by feature selection. We argue that not all parts of the local image are relevant to the questions posed. For instance, in Fig. 2(d), the question "What breed is the dog?" pertains only to specific local image regions, indicating that discarding irrelevant features can significantly reduce abundant image information. In this work, we explore a simple cosine-similarity routing strategy for its simplicity and effectiveness. Given the text embedding zx ∈ RL

v×D, we compute scores as S ∈ RL

x×D and the projected local image feature zv ∈ RL

v×Lx = zvzxT. Averaging text tokens and applying softmax to image tokens yields Scosine ∈ RL

v. Once scores or relevance indicators are obtained for each local feature, we employ an adaptive selection strategy. Specifically, we sort scores from highest to lowest and select features until the accumulated score surpasses a threshold γ. This hyperparameter balances the efficiency and completeness of local features. Our experiments reveal that selecting specific local features does not diminish performance. On the contrary, by disregarding irrelevant features and using fewer tokens, we achieve superior performance across most benchmarks. During training, Gaussian noise from N(0,0.1) is added to the selection score to maintain the diversity of representations.

Alternating Training Scheme. Our training methodology for the vision-language adapter and local compression layer involves a nuanced three-stage process. Initially, in Stage I (see Fig. 2(b)), the adapter undergoes training using the global image. Subsequently, in Stage II (see Fig. 2(c)), the adapter remains fixed while the local compression layer is exclusively trained using local patches. Finally, in Stage III (see Fig. 2(d)), both global and local features are simultaneously trained. While we delve into theoretical underpinnings in Section 2.3, empirical insights also bear significance. Our experimentation reveals that simultaneously training the adapter and local compression layer in a single stage yields suboptimal performance. This discrepancy arises from the model’s predominant focus on global features, as the global feature requires only projection with no information loss, making it easier to learn. Hence, we confine the use of local patches to Stage II for compression layer training. This approach ensures sequential learning, first projection, then compression of local features (Local vs Local w. alt in Fig. 1). Additionally, alternating training can bridge the performance gap between two common adapters: MLP and query former, as mentioned earlier. Specifically, when employing attention-based models as adapters, which offer more flexibility but may exhibit inferior performance compared to simple MLP adapters [56], we find that alternating training significantly enhances performance (query former vs query former w. alt3 in Fig. 1). Such a scheme may illuminate future work, facilitating the training of more complex yet flexible adapter options.

#### 2.3 Importance of Alternating Training for Optimizing Bilinear Functions

Takeaway: Alternating Training is pivotal for the success of SliME. Our demonstration in this subsection will also shed light on why it’s common practice to initially freeze one modality in multimodal learning and optimize the adapter of one modality before engaging in joint optimization across multiple modalities. All the proof can be found in appendix C.

Bilinear forms are prevalent in deep learning models, especially in multi-modal learning where the representation from two different modalities is frequently aligned through a dot product. Let the target matrix X ∈ Rd×d be expressed as X = ab⊤ + ba⊤, where a,b ∈ Rd are two normalized vectors. Our objective is to find the rank-1 matrix to approximate X, which leads to the following optimization problem:

- 1

- 2

L(u,v) =

min

u,v∈Rd

uv⊤ − X 2

In LMMs, the vision encoder and adapter can be perceived as the vision modality, while others are categorized as the text modality, and the target X can be seen as the best LMM. Within our framework, we treat the adapter and local compression layer as distinct functions, aiming to approximate the optimal modality adaptation parameter. We recognize that assuming both a and b to be merely vectors is a simplification that may not fully capture the complexity of the entire model. However, this simplification allows us to analyze the problem more effectively and derive valuable insights from it.

3Firstly, the model learns query embeddings to interact with image features, and secondly, it projects the image features to the LLM dimension. Similarly, in this alternating training approach, we first learn the projection head and then focus on learning attention mechanism parameters and query embeddings.

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

LLMs

[Figure 13]

||[Figure 14]| | |
|---|---|---|
| | | |
| | | |
<br><br>|[Figure 15]| |
|---|---|
| | |
<br><br>|[Figure 16]|
|---|
| |
<br><br>| |[Figure 17]|
|---|---|
<br><br>| | |
|---|---|
|[Figure 18]| |
| | |
<br><br>Adaptive Slicing|
|---|

[Figure 19]

[Figure 20]

[Figure 21]

Adapter Visual Encoder

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

MLP Q-former

[Figure 27]

| |[Figure 28]| |
|---|---|---|
| | | |
| | | |

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Router

The breed of the dog in the picture is what?

(a) Adaptive Slicing.

###### (b) Stage I.

[Figure 33]

[Figure 34]

LLMs

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Adapter

[Figure 39]

[Figure 40]

[Figure 41]

Local Compression Layer

[Figure 42]

[Figure 43]

[Figure 44]

Visual Encoder

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

| | |[Figure 54]|
|---|---|---|
| | | |
| | | |

The breed of the dog in the picture is what?

Adaptive Slicing

[Figure 55]

[Figure 56]

LLMs

[Figure 57]

Global & Top-k Local Features

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

Adapter

[Figure 71]

[Figure 72]

Text-Guided Router

[Figure 73]

[Figure 74]

[Figure 75]

Local Compression Layer

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Visual Encoder

The breed of the dog in the picture is what?

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

(c) Stage II.

(d) Stage III.

Figure 2: Innovative Training Approach: (b) Refining the visual adapter with mixture of experts, (c) optimizing local compression layers, and (d) instructions fine-tuning. Here, SliME efficiently processes images via slicing, projection, and selecting pertinent local features aligned with query.

It is well known that the optimal solution for u is aligned with the top eigenvector of XX⊤, i.e.

1 a⊤b a⊤b 1

XX⊤ = (ab⊤ + b⊤a)(ba⊤ + a⊤b) = A

##### A⊤

:=M

where A = (a,b) ∈ Rd×2 and M ∈ R2×2. Since X is constructed through a and b, u has to lie in the subspace spanned by a and b and thus can be written u as

α β

u = αa + βb = A

:=z Hence, the optimal solution for z = (α,β)⊤ should be aligned with the largest eigenvector of matrix M. Let u0 and v0 be the initial solution and is given in the following form

u0 = α0a + β0b, v0 = β0a + α0b

where α0,β0 ∈ R are two scales. Here, we utilize the fact that u and v have to lie in the subspace spanned by a and b. Then we state the following theorem:

- Theorem 1. Using the gradient descent method, we update the solution ut and vt as

ut+1 = ut − η utvt⊤ − X vt, vt+1 = vt − η vtu⊤t − X ut (1) Simultaneously updating u and v using Eq. (1) is less ideal for optimizing the objective function of bilinear form, as the gradient descent update does not necessarily converge to the optimal solution.

We will demonstrate that the issue with gradient descent (or more accurately, simultaneously updating u and v) can be effectively addressed by alternating optimization. Specifically, we will optimize v with fixed u, and then optimize u with fixed v. We will show that this approach converges to the optimal solution by alternating optimization.

- Theorem 2. Let u0 = α0a + β0b. We rewrite the sequential solution ut obtained by alternating

optimization as ut = αta + βtb. zt = (αt,βt)⊤ evolves over iterations by zt+1 = |u 1

t|2|vt|2M2zt

That is, alternating optimization ensures that zt ∝ M2tz0, implying that zt is guaranteed to converge to the largest eigenvector of M, thus resolving the limitation of gradient descent.

|[Figure 90]<br><br>[Figure 91]<br><br>Textbook QA #9.5K Life Science, Earth Science and Physical Science textbooks<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>Science QA #12.7K<br><br>Natural science, Language science, Social science<br><br>Question: Complete the statement. Tungsten carbide is ().<br><br>A. an elementary Substance<br>B. a compound<br><br><br>[Figure 95]<br><br>Question: What is the outer surrounding part of the Nucleus?<br><br>A. Nuclear Membrane<br>B. Golgi Body<br>C. Cell Membrane<br>D. Nucleolus<br><br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>Arxiv QA #100K<br><br>ArXiv papers spanning various scientific domains<br><br>Question: What can be inferred about thearticle's motion in figure c? In figure c, the trajectory forms a closed loop with a direction of motion indicated, suggesting circular motion. So the answer is particle exhibits circular motion with changing velocity<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>AI2 Diagrams#12K Diagram Understanding Primary Natural Science<br><br>Question: If there were a sudden decrease in the amount of large sharks, which organism would see an increase in population? Shrimp;Tuna;Squid tuna<br><br>[Figure 104]<br><br>[Figure 105]<br><br>GeoQA3& GeoQA+ #76K<br><br>Arithmetic, Trigonometric, Theorem Formula, Constant<br><br>[Figure 106]<br><br>Question:如图,四边形ABCD内接于⊙O, 如果它的一个外角∠DCE=62°, 那么∠BOD=() A. 124° B. 120° C. 62° D. 31°<br><br>四边形ABCD内接于⊙O,∴∠A=∠DCE= 62°∴∠BOD=2∠A=124°.故选A<br><br>[Figure 107]<br><br>[Figure 108]<br><br>Chart QA #28.3K<br><br>logical and arithmetic operations Understanding<br><br>Question: What is the least difference between light blue bar and the dark blue bar? 39.32<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>DVQA #200K chart understanding<br><br>Question: How many bars are there?<br><br>5<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>TabMWP #30K<br><br>mathematical reasoning on tabular data<br><br>Question: Each vase has 2 tulips. How many tulips are in 4 vases? Count by twos. Use the chart: there are 8 tulips in 4 vases.<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]|
|---|

Figure 3: The used science and mathematical reasoning tasks in this paper.

- 2.4 Expanding Dataset Scope with Challenging Reasoning Tasks

Generation of Source Data and Instruction Data. The creation of SMR involves a meticulous amalgamation of publicly available datasets, comprising Arxiv-QA [26], ScienceQA [38], MATH-Vision [50], TextBookQA [21], GeoQA3 [6], Geometry3K [37], TabMWP [39], DVQA [19], AI2D [20], and ChartVQA [40]. The variety of question types and associated images sourced from these datasets is depicted in Fig. 3, presenting a distinctive challenge to existing instruction datasets, as illustrated in Fig. 6. The disparities between SMR and conventional instruction tuning datasets manifest in two key aspects: (1) Challenging Reasoning Tasks. Many of the tasks in Physical/Social science and mathematics demand advanced reasoning abilities. Notably, datasets such as Arxiv-QA, GeoQA3, and TabMWP offer complete reasoning paths, including intermediate steps for deriving final results. In such cases, the model is tasked not only with mastering foundational knowledge but also with articulating complex reasoning processes—a notably more demanding endeavor. (2) Demand for Image Detail Understanding. All tasks necessitate a profound understanding of visual details because many images contain rich annotation information or questions requiring comprehensive visual analysis. This aspect is particularly beneficial for training our high-resolution framework. Further elucidation on datasets and specific construction methodologies can be found in appendix A. To ensure the accuracy of our data, we carefully filter it after collection. This involves identifying and fixing issues like blurry images or jumbled text, unrelated image-text pairs, and incorrect reasoning paths that can’t lead to correct answers or might lead to wrong conclusions. For the latter, we use GPT-4V to create new, accurate reasoning paths.

Statistics and Analysis. In Fig. 4, we illustrate the differences in statistics between SMR and existing instruction tuning datasets. To standardize multi-round conversations, we aggregate them into oneround and calculate the average length. We employ LLaVA to determine the maximum length such that 99% of the data falls within the interval. LLaVA [32] comprises 665K instruction tuning data instances characterized by short queries and answers. Similarly, LLaVAR [57] exhibits comparable patterns to LLaVA. Conversely, ShareGPT4V [7], a more comprehensive dataset derived from 100K high-quality captions generated by advanced GPT4-Vision models, features longer generation lengths, indicative of more detailed and complex captions. In contrast, SMR demonstrates longer query texts compared to existing training corpora, reflecting the need for detailed descriptions or background information to elucidate scientific or mathematical problems. Additionally, since some of our datasets focus solely on question-answer tasks, approximately 50% of instances feature shorter answer lengths. However, owing to the task complexity, particularly those datasets with extensive reasoning paths aiming to train models to comprehend intricate chains of reasoning, SMR exhibits a higher ratio of longer sentences compared to LLaVA and LLaVAR.

### 3 Experiment

SliME is evaluated against both open-source and closed-source models across various domains, encompassing General QA and Open-ended Generation, Math Reasoning, Science, and Hallucination benchmarks, totaling 15 benchmarks. We conduct a comprehensive analysis of the evaluation benchmarks, their associated metrics, and the training hyperparameters for both the initial vision-

SampleCountratio

SampleCountratio

Dataset

Dataset

0.4

0.75

LLaVA LLaVAR

ShareGPT4V SMR

LLaVA LLaVAR

ShareGPT4V SMR

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.50

0.2

0.25

0.0

0.00

1 4 8 11 14 17 20 24 27 30

12 36 60 84 108 132 156 180 204 228

Lengths

Lengths

(a) Average length of query text.

(b) Average length of answer text.

Figure 4: Comparison of the average lengths of query and answer texts across different datasets.

- Table 1: Comparison with SoTA methods on academic-task-oriented datasets and benchmarks for instruction-following LMMs. VQAT: TextVQA [44], MMB: MMBench [35], MMBC:

MMBench-Chinese [35]; MMMUv: validation set of MMMU [55]; LLaVAW: LLaVA-Bench (Inthe-Wild) [34]; MMEP,C: Perception and Cognition in MME [12]. The best result is marked in bold, and the second best result is underlined. † means using LoRA during the instruction tuning phase.

Method LLM VQAT GQA VQAv2 MMB MMBC MMMUv LLaVAW MMEP MMEC MM-Vet InstructBLIP [10] Vicuna-7B 50.10 49.20 - 36.00 23.70 32.90 60.90 - - 26.20 Qwen-VL [3] Qwen-7B 63.80 59.30 78.80 38.20 7.40 35.90 - - - LLaVA-1.5 [32] Vicuna-7B 58.20 62.00 78.50 64.30 58.30 - 65.40 1510 - 30.50 LLaVA-1.5 [32] Vicuna-13B 61.30 63.30 80.00 67.70 63.60 36.40 72.50 1531 295 35.40 ShareGPT4V [7] Vicuna-7B - - 80.60 68.80 62.20 - 72.60 1567 303 37.60 LLaVA-1.5 [32] Llama3-8B 58.94 61.94 79.49 72.94 67.70 38.00 70.50 1544 328 34.80

With High Resolution OtterHD-8B [23] Fuyu-8B - - - 58.30 - - - 1223 331 26.30 Monkey [29] Qwen-7B - 60.70 80.30 72.40 67.50 - - 1522 401 33.00 LLaVA-HD [32] Vicuna-13B 62.50 64.70 81.80 68.80 61.90 - 72.00 1500 - LLaVA-NeXT [33] Vicuna-13B - - - - - 35.90 72.30 1575 316 48.40

Ours

- SliME-7B Vicuna-7B 64.39 63.13 80.32 69.32 61.85 37.20 76.10 1544 383 35.40

- SliME-8B Llama3-8B 64.76 63.94 80.69 75.00 71.80 41.20 73.90 1578 337 37.40 SliME-8B† Llama3-8B 65.26 63.94 80.79 75.42 70.96 40.80 64.90 1573 346 36.80 SliME-13B Vicuna-13B 66.11 63.60 80.43 71.13 65.20 38.00 73.10 1606 293 41.20

Private models Gemini Pro [46] - 74.60 - - 73.60 74.30 47.90 - 1496 436 59.20 Qwen-VL-Plus [3] - - - - - 68.00 45.20 - 1681 502 GPT-4V [43] - 78.00 - - 77.00 74.40 58.10 - 1409 517 56.80

language alignment pretraining and the subsequent visual instruction tuning stages. Details are provided in Appendix D. The training process utilizes 8×A100 (80G) GPUs.

#### 3.1 Numerical Results

General QA and Open-ended Generation. We assess the performance of SliME across a range of benchmarks, encompassing both academic-task-oriented assessments and recent benchmarks tailored for instruction-following LLMs, totaling 9 in all. Our results demonstrate that SliME consistently achieves superior performance across all benchmarks, even when compared to LLMs of similar scale, despite utilizing significantly smaller pretraining and instruction tuning datasets than other methods [29, 3, 23]. Notably, SliME-8B even surpasses Gemini Pro on several benchmarks such as MMB and MME. Prior research has frequently indicated that the LoRA model performs comparably to full fine-tuning, a trend that holds true across many of our benchmarks. However, we observe that models trained with LoRA struggle in instruction-following tasks. This observation is bolstered by the performance gap observed in the LLaVA-bench between SliME-8B and SliME-8B†. Furthermore, in our evaluation of MathVerse, we find that while the model prompt explicitly requested concise answers, SliME-8B† consistently generates responses with intermediate reasoning, a behavior absent in SliME-8B. We delve into a detailed analysis of these inconsistencies in appendix D.2.

Scientific, Mathematical, and Hallucination. We further assess the hallucinatory property and mathematical proficiency of SliME. As demonstrated in Table. 2, SliME achieves state-of-the-art performance, comparable to Gemini Pro, across all three mathematical benchmarks. Moreover, its performance on the ScienceQA-Img split and hallucination benchmarks is particularly noteworthy, affirming the efficacy of the proposed SliME.

- Table 2: Comparison with SoTA methods on Science and Mathematical Reasoning benchmarks. SQAI: ScienceQA-IMG; MMEH: the sum of scores in existence, count, position, color splits of MME benchmark. † means using LoRA during the instruction tuning phase

Science and Mathematical Reasoning Hallucation Method LLM MathVerse MathVista MathVision ScienceQA POPE AMBER MMEH InstructBLIP [10] Vicuna-7B - 25.30 - 60.50 - 81.70 Qwen-VL [3] Qwen-7B - 33.80 10.53 67.10 - 84.90 606 LLaVA-1.5 Vicuna-7B - 26.10 10.20 66.80 85.90 74.70 LLaVA-1.5 Vicuna-13B 7.60 26.10 13.10 71.60 85.90 - 643 ShareGPT4V Vicuna-7B 13.10 25.80 12.50 68.40 - - LLaVA-1.5 LLama3-8B 13.80 28.40 14.75 77.64 85.10 85.00 634

With High Resolution

OtterHD-8B [23] Fuyu-8B - 23.40 - - 86.00 89.10 Monkey [29] Qwen-7B - 34.80 - 69.40 - 86.00 LLaVA-HD [32] Vicuna-13B - - - 71.00 86.30 - -

Ours

- SliME-7B Vicuna-7B 17.50 37.50 16.12 76.80 85.40 87.80 633

- SliME-8B LLama3-8B 22.90 43.30 16.78 84.18 86.00 88.90 671 SliME-8B† LLama3-8B 21.80 43.60 16.12 84.13 86.00 90.10 645 SliME-13B Vicuna-13B 19.00 40.80 18.09 80.17 86.30 89.40 673

Private models

Gemini Pro [46] - 22.30 45.20 17.11 - - - 560 Qwen-VL-Plus [3] - 11.80 43.30 - - - - 670 GPT-4V [43] - 38.30 49.90 22.37 - - 87.40 595

- Table 3: Ablation results on the global MOE

Table 4: Ablation results on different treatments for global and local features , and two different training strategies .

,the token number of the compression layer , the router parameter γ and the training data .

Dataset POPE GQA SQA VQAT AMBER

Baseline 83.80 61.94 74.30 58.94 87.70 Global MOE 84.10 62.58 77.00 60.57 88.70 Nq = 64 83.80 62.91 76.15 63.09 87.60 Nq = 196 84.00 62.51 75.56 62.48 87.40 Nq = 144 84.20 62.96 77.09 63.58 88.20 γ = 90% 84.20 62.74 76.05 64.05 88.10 γ = 75% 84.50 63.09 77.44 63.83 88.40 γ = 50% 84.90 63.11 77.09 62.75 88.40 With SMR 84.50 62.58 82.29 59.89 88.40 Final 84.9 63.94 84.18 64.76 88.90

Dataset POPE GQA SQA VQAT AMBER

Baseline 84.20 61.94 74.30 58.94 87.70 LLaVA-HD 85.00 62.48 72.97 61.48 88.20 Monkey 83.10 60.70 73.85 62.14 86.90 E2E 82.90 61.90 74.12 59.69 86.00

only Global 82.10 62.43 75.26 43.65 84.30 only Local 43.40 38.70 71.24 58.22 78.20

Alternating 84.20 62.96 77.44 63.58 88.40 only Global 84.10 62.78 76.01 59.42 87.30 only Local 82.10 53.21 76.23 62.45 85.60

#### 3.2 Ablation Studies and Analysis

Why Different Strategies for Global and Local Treatment are Necessary? When comparing two strategies from LLaVA-HD [32] and Monkey [29] with identical hyperparameters and slicing strategies, it becomes apparent why different treatment strategies for global and local features are essential. LLaVA-HD does not compress local features, in contrast, all image features are directly projected by an MLP, resulting in a maximal context size of 4096. That is, this approach significantly increases both training and inference times. Conversely, Monkey compresses all global and local image tokens using 144 learnable query embeddings, akin to SliME. Despite LLaVA-HD introducing more image tokens, our approach outperforms it. For instance, in the SQA dataset, only global of SliME achieves commendable performance, highlighting the importance of global context in SQA tasks. However, as image features are primarily dominated by local image details, LLaVA-HD is detrimental to the SQA dataset. Conversely, for datasets like VQAT, which demand more image details, LLaVA-HD consistently achieves performance gains. Notably, LLaVA-HD slightly outperforms SliME in the POPE benchmark. This is likely because POPE questions are simplistic and focus on single objects in the image. Therefore, even with some loss of local detail information, LLaVA-HD can answer such questions more effectively. Monkey’s approach, compressing all features, surpasses LLaVA-HD in SQA and VQAT by nearly 1 point. However, it performs inferiorly in other benchmarks, emphasizing the importance of maintaining the global context without compression. In contrast, SliME maintains all the global context and provides additional image detail with compression, yielding promising results regardless of whether the datasets prioritize global context or local details.

Impact of Alternating Training on Performance. In this part, we investigate the effect of alternating training on model performance. To assess the significance of Alternating Training, we initially compare the performance directly (lines 4 & 7 in Table. 3), revealing a substantial performance gap between them. To further explore this phenomenon, we isolate the global and local features as image tokens respectively to assess the amount of image information provided by each. Notably, for the model that is trained end-to-end, we observe that utilizing only global features yields satisfactory results, while the local features are inadequately trained, resulting in poor performance across most benchmarks. Conversely, when employing only local features for SliME, performance improves markedly. This improvement can be attributed to the model’s dedicated learning of local feature compression, resulting in well-trained local features. Despite this improvement, utilizing only local features proves insufficient across benchmarks, underscoring the crucial of global view.

Effect of Number of Reserve Tokens. Additionally, we validate our hypothesis that more image tokens do not always yield superior results. For instance, when γ is set to 75%, consistent performance gains are evident across most benchmarks. This indicates that by discarding irrelevant image tokens and padding tokens, the model can focus more on those most pertinent to the posed question.

Ablation Studies. In Table 3, we provide a comprehensive analysis of the effectiveness of each component within SliME. Through the utilization of various adapters for global features, we have witnessed a notable improvement in image understanding, resulting in enhanced performance across a range of benchmarks. This underscores the significance of leveraging complex adapters for better extraction of global information. The inclusion of a detailed local compression layer has proven particularly advantageous for tasks demanding intricate visual analysis, such as TextVQA. Increasing Nq from 64 to 144 further amplifies performance. However, as the number of local feature tokens increases, performance does not consistently rise. This is attributed to the overshadowing of global context by the abundance of local tokens, as evidenced by performance dips across most benchmarks. Notably, excessive local detail can also hinder performance gains in VQAT. Therefore, we adopt 144 as our default setting. Lastly, the proposed SMR data demonstrates a significant enhancement in the model’s reasoning ability, especially evident in mathematical and scientific datasets, while mitigating hallucination. Importantly, rigorous checks have been conducted on all training data to ensure there are no concerns regarding data leakage between training and test instances.

Qualitative Results of SliME in high-resolution image perception are illustrated in Fig. 5, and Figs. 9 to 9. These results emphasize the importance of local features to SliME, as they enrich image details and facilitate a deeper understanding of vision information. Moreover, the ultimate SliME demonstrates robust open-generation capabilities, including tasks such as code generation from flowcharts (Fig. 10), story creation based on images (Fig. 11), and providing suggestions (Fig. 12).

[Figure 119]

[Figure 120]

Query: What is written at the top of the yellow sticker on the fridge?

[Figure 121]

The yellow sticker on the fridge has the word "Warning" written at the top.

[Figure 122]

[Figure 123]

The yellow sticker on the fridge says "no smoking"

[Figure 124]

[Figure 125]

[Figure 126]

Figure 5: High-resolution image perception.

### 4 Conclusion and Discussion

In this paper, we delve into elucidating the intricacies of designing large multimodal models, with a specific focus on high-resolution image processing. Unlike previous studies that treat both the global view and sliced local image patches indiscriminately, our approach involves projecting and extracting global context using a mixture of experts, all without any feature compression. This methodology is rooted in the belief that global context encapsulates the majority of image information and holds greater significance than local patches. Local features undergo compression and selection based on their relevance to the query, thereby mitigating computation costs. Although training the framework end-to-end initially yields subpar performance, we address this by formulating the problem into a bilevel formulation and employing an alternating training strategy. This strategic maneuver circumvents optimization dilemmas inherent in end-to-end training. Our framework, dubbed SliME, demonstrates

promising performance across more than 10 benchmarks and even matches the performance of proprietary LMMs trained on significantly larger datasets, all with only 2 million training data points.

Limitation and Future Work. One main limitation lies in the 3-stage training approach. While alternating training proves superior to E2E training both theoretically and empirically, it inevitably extends the training duration. A promising avenue for improvement involves delving deeper into optimization methods tailored for such a bilinear formulation [59], potentially converting the alternating training strategy into a soft constraint within the gradient during E2E training. Another fruitful direction for future research is image token reduction. Given that existing studies consolidate all local and global features into LLMs, the computational cost for processing very-high-resolution images becomes prohibitively high. Therefore, an open question remains: can we further reduce image tokens, drawing inspiration from techniques such as token merging [4] in computer vision? By doing so, we may preserve sufficient local details without necessitating additional image tokens for LLMs.

### References

- [1] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.
- [2] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [4] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. In ICLR, 2023.
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020.
- [6] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021.
- [7] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.
- [8] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023.
- [9] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. JMLR, 2023.
- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. NeurIPS, 2024.
- [11] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

- [12] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.
- [13] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.
- [14] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. CVPR, 2024.
- [15] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, pages 6700–6709, 2019.
- [16] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. CVPR, 2019.
- [17] Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. Perceiver: General perception with iterative attention, 2021.
- [18] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [19] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In CVPR, 2018.
- [20] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV. Springer, 2016.
- [21] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In CVPR, 2017.
- [22] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. NeurIPS, 2024.
- [23] Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A high-resolution multi-modality model. arXiv preprint arXiv:2311.04219, 2023.
- [24] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023.
- [25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [26] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. arXiv preprint arXiv:2403.00231, 2024.
- [27] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024.
- [28] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.
- [29] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607, 2023.

- [30] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2023.
- [31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.
- [32] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.
- [33] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.
- [35] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024.
- [36] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.
- [37] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.
- [38] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. NeurIPS, 2022.
- [39] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint arXiv:2209.14610, 2022.
- [40] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.
- [41] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024.
- [42] Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786, 2022.
- [43] OpenAI. Gpt-4 technical report. 2023.
- [44] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019.
- [45] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model, 2023.
- [46] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [48] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [49] Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Ming Yan, Ji Zhang, and Jitao Sang. An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397, 2023.
- [50] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv preprint arXiv:2402.14804, 2024.
- [51] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: an lmm perceiving any aspect ratio and highresolution images. arXiv preprint arXiv:2403.11703, 2024.
- [52] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.
- [53] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023.
- [54] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In ICML, 2024.
- [55] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023.
- [56] Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, and Tao Kong. What matters in training a gpt4-style language model with multimodal inputs? arXiv preprint arXiv:2307.02469, 2023.
- [57] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023.
- [58] Yi-Fan Zhang, Weichen Yu, Qingsong Wen, Xue Wang, Zhang Zhang, Liang Wang, Rong Jin, and Tieniu Tan. Debiasing large visual language models. arXiv preprint arXiv:2403.05262, 2024.
- [59] Bo Zhao, Robert M. Gower, Robin Walters, and Rose Yu. Improving convergence and generalization using parameter symmetries, 2024.
- [60] Bo Zhao, Boya Wu, and Tiejun Huang. Svit: Scaling up visual instruction tuning. arXiv preprint arXiv:2307.04087, 2023.
- [61] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

|[Figure 127]<br><br>[Figure 128]<br><br>COCO Captions<br><br>Object recognition, Scene understanding<br><br>[Figure 129]<br><br>[Figure 130]<br><br>GQA<br><br>Visual reasoning and compositional question answering,<br><br>Question: What type of fruit in the image is round? Answer: Apple Question: What color is the fruit on the right side,<br><br>red or green? Answer: Green<br><br>Question: Please describe the Image.<br><br>Answer: the man at bat Readies to swing at the Pitch while the umpair Looks on.<br><br>[Figure 131]<br><br>[Figure 132]<br><br>TextVQA<br><br>Visual Reasoning<br><br>Question: which processor Brand is featured on the top left? Anwer: intel<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>OCR-VQA<br><br>Optical character recognition<br><br>Question: Who wrote this book? Answer: Jacob Grier<br><br>Question: Is this book related to cooking? Answer: Yes<br><br>[Figure 136]<br><br>[Figure 137]<br><br>Visual Genome<br><br>Recognizing and Reasoning<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>VQA v2<br><br>Commonsense knowledge<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>OK-VQA<br><br>Outside Knowledge<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>A-OK-VQA<br><br>commonsense and world knowledge<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>Flickr30k<br><br>Object recognition, Scene understanding<br><br>[Figure 161]<br><br>Question: Please describe the Image. Answer: A man stands on one foot while holding on to a waste basket.<br><br>[Figure 162]|
|---|

Figure 6: Visual perception and reasoning tasks in existing visual instruction tuning datasets.

### A Related Work

Multimodal Large Language Models have undergone significant evolution, initially rooted in BERTbased language decoders and later incorporating advancements in LLMs. Leveraging advanced LLMs such as GPTs [43, 5], PaLM [9, 1], BLOOM [42], LLaMA [47, 48], Alpaca [45], Vicuna [8], and Mistral [18], Multimodal Large Language Models (LVLMs) exhibit enhanced capabilities and performance, particularly through end-to-end training techniques. Recent model developments, including Flamingo [2], PaLI [22], PaLM-E [11], BLIP-2 [25], InstructBLIP [10], Otter [24], MiniGPT-4 [61], mPLUG-Owl [52], LLaVA [34], and QWen-VL [3], bring unique perspectives to challenges such as scaling pre-training, enhancing instruction-following capabilities, and overcoming alignment issues. Our work is built upon LLaVA [34], but enhances all training datasets, model architecture, and alignment strategies, achieving state-of-the-art performance among existing LLMs.

Advancements in Visual Instruction Tuning: The efficacy of multimodal models heavily relies on the availability of high-quality image-text data for fine-tuning, a process known as visual instruction tuning [34]. Previous studies have highlighted the limitations of constructing training sets solely based on existing Visual Question Answering (VQA) datasets [15], often resulting in degraded model performance. In an effort to address this, MiniGPT-4 [61] meticulously curated 3,500 high-quality image-text pairs through a refinement process using ChatGPT, leading to more natural and reliable responses post-fine-tuning. In a pioneering initiative, LLaVA [34] systematically constructed the LLaVA-Instruct-150K dataset for visual instruction tuning. Employing GPT-4, they generated questions and answers by providing image-level captions and object bounding boxes from the COCO dataset [31]. To delve deeper into text-rich images, LLaVAR [57] collected 422K noisy instruction-following instances using Optical Character Recognition (OCR) results and 16K high-quality instances using GPT-4. InstructBLIP [10] amalgamated 26 public datasets, including LLaVA-Instruct-150K, to construct visual instruction tuning data. Innovatively, SVIT [60] constructed a comprehensive dataset comprising 4.2 million visual instruction tuning instances, including conversation question-answer pairs, complex reasoning QA pairs, referring QA pairs, and detailed image descriptions. However, many of these public datasets predominantly focus on visual perception and image captioning. In this study, we elevate the complexity of the Visual Instruction Tuning process by incorporating nine diverse datasets, encompassing scientific questions, mathematical/chart reasoning tasks, and even full reasoning paths. This augmentation aims to enhance LMMs to achieve significantly improved reasoning capabilities.

### B Visual instruction tuning datasets

Arxiv-QA [26]4 is a dataset featuring diverse images from scientific domains. Leveraging GPT-4V, the authors generated instruction-tuning datasets for generating QA pairs based on figures extracted from scientific papers. After filtering out invalid samples, the dataset consists of 100K QA pairs. The questions in the dataset have an average word count of 16.98, while the options for each question

4https://huggingface.co/datasets/MMInstruction/ArxivQA

have an average word count of 31.86. On average, there are 4.20 options per question. For ArxivQA, we utilize rationales as answers for supervised fine-tuning.

ScienceQA [38]5 is a dataset characterized by rich domain diversity across three subjects: natural science, language science, and social science. Questions within each subject are categorized first by topic (e.g., Biology, Physics, Chemistry) and then further categorized by subtopics (e.g., Plants, Cells, Animals) and skills (e.g., Classify fruits and vegetables as plant parts, Identify countries of Africa). ScienceQA encompasses 26 topics, 127 categories, and 379 skills, providing comprehensive coverage across various domains. For ScienceQA, we utilize both the question and the instructions on how to solve the question as the prompt.

MATH-Vision [50]6 is an intricately assembled compilation of 3,040 meticulously selected mathematical problems accompanied by visual contexts sourced from authentic math competitions. Encompassing 16 distinct mathematical disciplines and graded across 5 levels of difficulty, the dataset offers a comprehensive and diverse array of challenges to assess the mathematical reasoning capabilities of Large Multimodal Models (LMMs).

AI2D [20]7 AI2 Diagrams (AI2D) is a dataset of over 5000 grade school science diagrams with over 150000 rich annotations, their ground truth syntactic parses, and more than 15000 corresponding multiple choice questions.

TextBookQA [21]8 is derived from middle school science curricula and comprises 1,076 lessons extracted from Life Science, Earth Science, and Physical Science textbooks. The dataset encompasses a total of 26,260 questions, with 12,567 of them accompanied by diagrams. We leverage the instructional content of images with the prompt that chosen from table 5 and consider questions with accompanying diagrams necessary for answering as VQA tasks.

GeoQA3 [6]9 is a dataset comprising 4,998 diverse real-world geometric problems sourced from Chinese middle school exams. Each problem is further annotated with specific programs that describe the problem-solving process. The dataset encompasses three main problem types: angle calculation, length calculation, and others, which include various types of problems such as area calculation.

The Geometry3K Dataset[37]10 comprises 3,002 SAT-style problems sourced from two high-school textbooks covering diverse graph and goal types. Additionally, each problem in Geometry3K is annotated with dense descriptions in formal language.

Tabular Math Word Problems (TabMWP)[39]11 introduces a novel dataset containing 38,431 opendomain grade-level problems requiring mathematical reasoning on both textual and tabular data. Each question in TabMWP is associated with a tabular context, presented in the form of an image, semi-structured text, and a structured table. Questions are categorized into free-text and multi-choice types, with each problem annotated with gold solutions elucidating the multi-step reasoning process.

Data Visualization Question Answering (DVQA)[19]12 is a dataset designed to assess various aspects of bar chart understanding in a question answering framework. Unlike Visual Question Answering (VQA), DVQA necessitates processing words and answers unique to a particular bar chart. DVQA facilitates the automatic querying of extensive repositories of charts within scientific documents, web pages, and business reports. We randomly selected 100K data from the reasoning splits for our experiments.

ChartVQA[40]13 serves as a benchmark for question answering about charts with visual and logical reasoning. We exclusively selected human-authored QA pairs and excluded all machine-generated question-answer pairs to ensure data quality, resulting in a total of 9.6K question-answer pairs.

- 5https://huggingface.co/datasets/derek-thomas/ScienceQA
- 6https://huggingface.co/datasets/mathvision/mathvision
- 7https://github.com/allenai/dqa-net
- 8https://allenai.org/data/tqa
- 9https://github.com/chen-judge/GeoQA
- 10https://github.com/lupantech/InterGPS
- 11https://github.com/lupantech/PromptPG
- 12https://github.com/kushalkafle/DVQA_dataset
- 13https://github.com/vis-nlp/ChartQA

Table 5: The list of instructions for detailed image description. Index Description

- 1 Describe the following image in detail
- 2 Provide a detailed description of the given image
- 3 Give an elaborate explanation of the image you see
- 4 Share a comprehensive rundown of the presented image
- 5 Offer a thorough analysis of the image
- 6 Explain the various aspects of the image before you
- 7 Clarify the contents of the displayed image with great detail
- 8 Characterize the image using a well-detailed description
- 9 Break down the elements of the image in a detailed manner
- 10 Walk through the important details of the image
- 11 Portray the image with a rich, descriptive narrative
- 12 Narrate the contents of the image with precision
- 13 Analyze the image in a comprehensive and detailed manner
- 14 Illustrate the image through a descriptive explanation
- 15 Examine the image closely and share its details
- 16 Write an exhaustive depiction of the given image

### C Proof of Theoretical Statements

- C.1 Gradient Descent for Optimizing Bilinear Problem Let u0 and v0 be the initial solution and is given in the following form

##### u0 = α0a + β0b, v0 = β0a + α0b (2)

where α0,β0 ∈ R are two scales. Here, we use the fact that u and v have to lie in the subspace spanned by a and b. Using the gradient descent method, we update the solution ut and vt as

ut+1 = ut − η utvt⊤ − X vt, vt+1 = vt − η vtu⊤t − X ut

The following theorem describes the dynamics of u and v over the iterations of gradient descent.

- Theorem 3. Define zt = (αt,βt)⊤. Under the initialization of u and v given in Eq. 2, we have, for all t ≥ 0

ut = αta + βtb, v0 = βta + αtb with |ut|2 = |vt|2, where

1 + η(1 − |ut|2) ηa⊤b ηa⊤b 1 + η(1 − |ut|2)

zt (3)

zt+1 =

:=Ft

Proof. We prove the result by induction. It is easy to see that the theorem hold for t = 0. We will then assume it is true for the case of t and show it also holds for t + 1. We first have ut+1 as

ut+1 = ut − η utvt⊤ − X v0

= ut 1 − η|vt|2 + ηXvt

= ut 1 − η|vt|2 + η ab⊤ + ba⊤ (βta + αtb)

= ut 1 − η|ut|2 + η βtb⊤aa + βtb + αta + αta⊤bb

= (1 − η|ut|2 + η)αt + ηa⊤bβt a + (1 − η|ut|2 + η)βt + a⊤bαt b

where we utilize the assumption |vt| = |ut|. Hence, by writing ut+1 = αt+1a + βt+1b, we have

1 + η(1 − |ut|2) ηa⊤b ηa⊤b 1 + η(1 − |ut|2)

αt+1 βt+1

αt βt

=

Similarly, we apply gradient descent to update vt as follows

vt+1 = vt − η vtu⊤t − X ut

= vt 1 − η|ut|2 + ηXut

= vt 1 − η|ut|2 + η ab⊤ + ba⊤ (αta + βtb)

= vt 1 − η|vt|2 + η b⊤aαta + αtb + βta + a⊤bβtb

= (1 − η|vt|2 + η)βt + ηa⊤bαt a + (1 − η|vt|2 + η)αt + a⊤bβt b Thus, it is easy to see that vt+1 = βt+1a + αt+1b.

| |
|---|

Next, we will show that the updating in Eq. 3 will not converge the largest eigenvector of M. Let λ+ > λ− denote the largest and smallest eigenvalues of M, and let w+ and w− denote the corresponding eigenvectors, respectively. We have

λ+ = 1 + a⊤b, λ− = 1 − a⊤b and

1 √2

1 √2

1 1

1 −1

w+ =

, w− =

Let τt = zt⊤w+ and νt = zt⊤w−. Using these notation, we can write |ut|2 as

|ut|2 = zt⊤Mzt = 1 + a⊤b τt2 + 1 − a⊤b νt2 (4) The following lemma describes how τt and νt evolves over iterations. Lemma 1. We have

##### τt+1 = 1 + η(1 + a⊤b − |ut|2) τt, νt+1 = 1 + η(1 − a⊤b − |ut|2) νt (5) Proof. According to Eq. 3, we can write

Ft = (1 − η|ut|2)I + ηM and therefore can decompose it into the following form

Ft = (1 − η|ut|2 + ηλ+)w+w+⊤ + (1 − η|ut|2 + ηλ−)w−w−⊤ Since zt = τ+w+ + νtw−, we have

Ftzt = (1 − η|ut|2 + ηλ+)τtw+ + (1 − η|ut|2 + ηλ−)νtw− and therefore

τt+1 = 1 + η(1 + a⊤b − |ut|2) τt, νt+1 = 1 + η(1 − a⊤b − |ut|2) νt

| |
|---|

When η is very small, we can write the updating equation in Eq. 5 as differential equations, i.e.

dνt dt

dτt dt

= 1 − a⊤b − |ut|2 νt (6) The following reveals the convergence property of Eq. 6.

= 1 + a⊤b − |ut|2 τt,

- Theorem 4. ODE in Eq. 6 have two converged solutions, with one being τt → 1,νt → 0 and the other being τt → 0,νt → 1 when t → ∞

Proof. Using the expression of |ut|2 in Eq. 4, we have

d|ut|2 dt

dτt dt

dνt dt

= 2 1 + a⊤b τt

+ 2 1 − a⊤b νt

= 2 1 + a⊤b 1 + a⊤b − |ut|2 τt2 + 2 1 − a⊤b 1 − a⊤b − |ut|2 νt2

= 2(1 − |ut|2)|ut|2 + 2a⊤b (1 + a⊤b)τt2 − (1 − a⊤b)νt2

When the ODE converges to a fix point u∗, we should have

2(1 − |u∗|2)|u∗|2 + 2a⊤b (1 + a⊤b)τ∗2 − (1 − a⊤b)ν∗2 = 0 (7) At the same time, according to Eq. 6, when ODE converges, we should have either

|u∗|2 = 1 + a⊤b, v∗ = 0 or

|u∗|2 = 1 − a⊤b, τ∗ = 0.

We complete the proof by combining these two conditions with the condition in Eq. 7.

| |
|---|

As indicated by Theorem 4, the iterative updating by gradient descent does not necessarily converge to the optimal solution, i.e., τ∗ = 1 and ν∗ = 0, rendering the simultaneous updating of u and v less ideal for optimizing the objective function of the bilinear form.

- C.2 Optimizing Bilinear Problem by Alternating Optimization We will demonstrate that the issue with gradient descent (or more precisely, simultaneously updating

- u and v) can be effectively addressed through alternating optimization. Specifically, we will optimize
- v with u fixed, and then optimize u with v fixed. We will illustrate that this approach leads to convergence towards the optimal solution. Let u0 = α0a + β0b. We denote the sequential solution ut obtained through alternating optimization as ut = αta + βtb. The following theorem describes the evolution of zt = (αt,βt)⊤ over iterations.

#### Theorem 5. We have

1 |ut|2|vt|2

M2zt

zt+1 =

Proof. Let ut = αta + βtb. By fixing ut, we have the optimal solution for vt

1 |ut|2

vt =

1 |ut|2

=

By writing vt = βt′a + αt′b, we have

Xut

a⊤bαta + αtb + βta + a⊤bβtb

αt′ βt′

1 a⊤b a⊤b 1

1 |ut|2

αt βt

=

We then fix vt and find the optimal solution for ut+1, i.e. ut+1 =

1 |vt|2

Xvt

1 |vt|2

a⊤bβt′a + βt′b + αt′a + a⊤bαt′b By writing ut+1 = αt+1a + βt+1b, we have

=

αt+1 βt+1

1 |vt|2

=

1 a⊤b a⊤b 1

αt′ βt′

1 |ut|2|vt|2

=

1 a⊤b a⊤b 1

2 αt βt

| |
|---|

According to Theorem 5, by alternating optimization, we have

zt ∝ M2tz0

implying that zt will guarantee to converge to the largest eigenvector of M, which resolves the limitation of gradient descent.

### D Experiments

#### D.1 Experimental Details

General QA and Open-ended Generation Benchmarks The MMMU dataset [55] comprises meticulously curated multimodal questions totaling 11.5K, sourced from college exams, quizzes, and textbooks. Spanning six core disciplines—Art and Design, Business, Science, Health and Medicine, Humanities and Social Science, and Tech and Engineering—it covers 30 subjects and 183 subfields. With a focus on advanced perception and reasoning embedded with domain-specific knowledge, MMMU challenges models to tackle tasks akin to those encountered by experts. It incorporates 30 highly diverse image types, including charts, diagrams, maps, tables, music sheets, and chemical structures, diverging from existing benchmarks by emphasizing sophisticated perception and reasoning. The MME dataset [12] distinguishes itself by concurrently evaluating perception and cognition capabilities, encompassing tasks such as OCR, coarse-grained object recognition (including existence, count, position, and color), and fine-grained object recognition (encompassing movie posters, celebrities, scenes, landmarks, and artworks). With a total of 14 subtasks, MME provides a comprehensive evaluation, catering to the need for a thorough assessment of MLLMs across diverse modalities and cognitive domains. Additionally, traditional benchmarks such as MMBench-EN [35], MMVet [54], GQA [16], VQA [13], and Text-VQA [44] were utilized to gauge the model’s visual understanding and reasoning abilities. Further evaluations included testing the model’s proficiency in Chinese and its understanding of Chinese culture through the MMBench-CN test [35], along with assessing conversation abilities via LLaVA-Bench [34].

Math Reasoning and Science QA Benchmarks: MathVista [36] presents a multifaceted benchmark that integrates challenges from diverse mathematical and visual tasks. Comprising 6,141 examples, it amalgamates data from 28 existing multimodal datasets alongside the introduction of three novel datasets: IQTest, FunctionQA, and PaperQA. Solving these tasks demands intricate visual comprehension and compositional reasoning, posing significant challenges even for state-of-the-art foundation models. MathVision [50] meticulously curates a collection of 3,040 high-quality mathematical problems with visual contexts drawn from real math competitions. Encompassing 16 distinct mathematical disciplines and graded across 5 difficulty levels, this dataset offers a comprehensive and diverse range of challenges for assessing the mathematical reasoning abilities of LMMs. MathVerse provides an inclusive visual math benchmark by collecting 2,612 high-quality, multi-subject math problems with accompanying diagrams sourced from publicly available materials. Each problem undergoes transformation by human annotators, resulting in six distinct versions with varying levels of information content in multimodality, culminating in 15K test samples. This approach enables MathVerse to thoroughly evaluate the ability of LMMs to comprehend visual diagrams for mathematical reasoning. ScienceQA [38] covers 26 topics, 127 categories, and 379 skills, offering comprehensive coverage across various domains.

Hallucination The existence and count subsets of the MME dataset [12] are applicable for objectlevel hallucination, while the position and color subsets are suitable for attribute-level hallucination. POPE [28] introduces a streamlined method for evaluating object hallucination in LMMs. In this assessment framework, LMMs are tasked with determining the presence of a specific object in a given image. The benchmark ensures a balanced ratio between queries probing existent and non-existent objects (i.e., 50% each). AMBER [49] is an LLM-free Multi-dimensional Benchmark for LMMs hallucination evaluation, which can be used to evaluate fir the discriminative task including existence, attribute and relation hallucination.

We provide a thorough examination of the evaluation benchmarks employed, accompanied by their respective metrics as detailed in table 6. Additionally, we outline the training hyperparameters for both the initial vision-language alignment pretraining and the subsequent visual instruction tuning stages in table 7.

#### D.2 Extended Experiments

Instruction Following Concerns Regarding Fine-tuning with LoRA. As illustrated in Fig. 7, within the MathVerse benchmark, despite our prompt explicitly requesting the model to “Please directly answer the question and provide the correct option letter”, SliME-8B† consistently generates the entire reasoning path. While this does not necessarily result in incorrect answers, it diverges from our intended approach. Intriguingly, when we modify the prompt to “Please directly answer the question

#### Table 6: Summary of the evaluation benchmarks.

Data Response formatting prompts Metric Text-VQA Answer the question using a single word or phrase. Accuracy (↑)

GQA Answer the question using a single word or phrase. Accuracy (↑) VQA-v2 Answer the question using a single word or phrase. Accuracy (↑)

MMBench, MMBench-CN Answer with the option’s letter from the given choices directly. Accuracy (↑) MMMU (Multi-choice) Answer with the option’s letter from the given choices directly. Accuracy (↑) MMMU (Short answer) Answer the question using a single word or phrase. Accuracy (↑)

LLaVA-Bench - Score Ratio compared to GPT-4 (↑)

MME Answer the question using a single word or phrase. Total Score (↑) Science-QA Answer with the option’s letter from the given choices directly. IMG-Accuracy (↑) MathVision Answer the question using a single word or phrase. Accuracy (↑) MathVista - Accuracy (↑) MathVerse - Accuracy (↑)

POPE Answer the question using a single word or phrase. F1 score (↑) AMBER Answer the question using a single word or phrase. Accuracy (↑)

Table 7: Hyperparameters of SliME. The hyper-parameters of Vicuna-13B is similar to Vicuna-7B.

LLM Vicuna-7B Llama3-8B Llama3-70B (LoRA)

Stage I & II Stage III Stage I & II Stage III Stage I & II Stage III batch size 256 128 256 128 256 128

lr 1.00E-03 2.00E-05 1.00E-03 1.00E-05 1.00E-03 1.00E-04 Deepspeed ZeRO Stage 2 3 2 3 3 3 offload

lr schedule Cosine Annealing with Linear Warmup lr warmup ratio 0.03 Weight Decay 0

Epoch 1 Optimizer AdamW Precision BF16

using a single word or phrase and provide the correct option letter”, the model responds with just "C," highlighting the sensitivity of SliME-8B†. Additionally, we observed that the performance of SliME-8B† is inferior to that of SliME-8B, as exemplified in Fig. 7(b). Thus, while LoRA fine-tuning maintains similar performance across most benchmarks, the constrained parameter updates and limited alterations to the parameters of LLMs may make it more challenging than fully fine-tuning to strictly adhere to instructions, particularly when image prompts are involved.

Text: Describe this painting in detail.

[Figure 163]

[Figure 164]

Image:

[Figure 165]

Text: Please directly answer the question and provide the correct option letter, e.g., A, B, C, D. Question: As shown in the figure, if angle A = 36.0, then angle C is equal to () Choices: A:36° B:54° C:60° D:27°

[Figure 166]

[Figure 167]

[Figure 168]

Image:

[Figure 169]

The image is a portrait of a woman painted in the Renaissance style. The woman is depicted with long hair and a red dress. The background features a landscape with mountains and a river. The painting is executed in oil on a wooden panel .

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

The image is a captivating representation of a woman's face, painted in the distinctive style of the Renaissance period. The woman's face is the central focus of the painting, her features rendered with meticulous detail. Her hair, a rich shade of red, cascades down her shoulders, adding a vibrant contrast to the otherwise muted palette of the painting. The background is a serene landscape, with a distant mountain range under a pale blue sky. The painting is executed in oil on canvas, a medium that allows for the subtle blending of colors and the creation of depth and texture. The overall composition and the artist's use of color and light are characteristic of the Renaissance genre .

解:∵AB∥CD,∴∠BEF=∠1=50°,∵EG平分 ∠BEF,∴∠BEG=\frac{1}{2}∠BEF=25°,∴∠2=∠BEG+∠G=25°+90°=115°,故选:C．.

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

C

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

(a) MathVerse.

(b) LLaVA-Bench.

#### Figure 7: Failure cases of SliME-8B†.

[Figure 195]

[Figure 196]

Query: When does it say it'll be open?

[Figure 197]

[Figure 198]

It says that the smoothie shop will be open soon.

[Figure 199]

[Figure 200]

The sign says it will be open at 11:00.

[Figure 201]

[Figure 202]

#### Figure 8: High-resolution image perception.

Query: What is the advertisement in the white board?

[Figure 203]

[Figure 204]

[Figure 205]

The advertisement in the white board is for Southern Homes.

[Figure 206]

[Figure 207]

The advertisement on the white board is for a dental office.

[Figure 208]

[Figure 209]

Query: What is the number for southern homes?

[Figure 210]

[Figure 211]

The number for Southern Homes is 648-Home.

[Figure 212]

[Figure 213]

The number for Southern Homes is 648-3958.

[Figure 214]

[Figure 215]

#### Figure 9: High-resolution image perception.

[Figure 216]

Query: Can you translate this into Python code?

[Figure 217]

[Figure 218]

Certainly! Here's the Python code that corresponds to the flowchart in the image:

[Figure 219]

This code defines a function `read_guess()` to read the user's guess, a function `generate_random_number()` to generate a random number between 1 and 10, a function `check_guess()` to check if the guess is correct or not, and a function `play_game()` to play the game. The `play_game()` function calls the other functions and prints the result of the game.

[Figure 220]

[Figure 221]

#### Figure 10: Code generation.

[Figure 222]

Query: My son drew this, can you tell a short bedtime story based on his drawing?

[Figure 223]

Once upon a time, there was a man named Jack who loved the sea. He lived in a small coastal town where he spent most of his days fishing and sailing. Jack was a skilled sailor and knew the waters like the back of his hand.

[Figure 224]

One day, Jack decided to go on a solo adventure. He set sail on his small boat, the "Sea Dragon," and headed out into the open waters. The wind was in his favor, and the "Sea Dragon" glided smoothly across the waves.

As Jack sailed further and further from the shore, he encountered a group of playful dolphins. They swam alongside the "Sea Dragon," leaping and splashing in the water. Jack laughed with joy, feeling at one with nature.

Suddenly, a fierce storm rolled in. The winds howled, and the waves crashed against the "Sea Dragon." Jack held on tight, using all his skill and experience to navigate the treacherous waters.

Just when it seemed like all hope was lost, the storm subsided, and the sun broke through the clouds. Jack breathed a sigh of relief and looked around at the calm sea. He spotted a small island in the distance and decided to explore it.

As he approached the island, Jack noticed that it was covered in lush green forests and sandy beaches. He anchored the "Sea Dragon" and went ashore, eager to explore the island.

Jack spent the day exploring the island, discovering hidden coves and secret waterfalls. He even found a hidden treasure chest filled with gold and jewels.

As the sun began to set, Jack returned to the "Sea Dragon" and set sail for home. He was tired but content, knowing that he had experienced a truly magical adventure.

[Figure 225]

From that day on, Jack sailed the seas, always seeking out new adventures and discovering hidden wonders. And he never forgot the magical island and the treasure that lay within. The end.

[Figure 226]

#### Figure 11: Story writing.

[Figure 227]

Query: These holes are appearing around the screws in my deck. Is it rotten? Should I get my boards replaced?

[Figure 228]

[Figure 229]

Yes, the image shows that the wooden boards have holes around the screws, which indicates that the wood is rotten. It is advisable to replace the rotten boards to ensure the safety and structural integrity of your deck. Rotten wood can weaken the deck's support and lead to accidents or further damage. It is essential to address this issue promptly to prevent any potential hazards.

[Figure 230]

[Figure 231]

#### Figure 12: Suggestion.

