## Mitigating Object Hallucinations via Sentence-Level Early Intervention

### Shangpin Peng1∗ Senqiao Yang2∗ Li Jiang3 Zhuotao Tian1

1Harbin Institute of Technology, Shenzhen 2The Chinese University of Hong Kong 3The Chinese University of Hong Kong, Shenzhen

### Abstract

# arXiv:2507.12455v3[cs.CV]22May2026

Human Annotators

Proprietary Models

[Figure 1]

[Figure 2]

Model output

Factual Hallucinated

Multimodal large language models (MLLMs) have revolutionized cross-modal understanding but continue to struggle with hallucinations - fabricated content contradicting visual inputs. Existing hallucination mitigation methods either incur prohibitive computational costs or introduce distribution mismatches between training data and model outputs. We identify a critical insight: hallucinations predominantly emerge at the early stages of text generation and propagate through subsequent outputs. To address this, we propose SENTINEL (Sentence-level Early iNtervention Through IN-domain prEference Learning), a framework that eliminates dependency on human annotations. Specifically, we first bootstrap high-quality in-domain preference pairs by iteratively sampling model outputs, validating object existence through cross-checking with two open-vocabulary detectors, and classifying sentences into hallucinated/non-hallucinated categories. Subsequently, we use context-coherent positive samples and hallucinated negative samples to build context-aware preference data iteratively. Finally, we train models using a context-aware preference loss (C-DPO) that emphasizes discriminative learning at the sentence level where hallucinations initially manifest. Experimental results show that SENTINEL can reduce hallucinations by over 90% compared to the original model and outperforms the previous state-of-the-art method on both hallucination benchmarks and general capabilities benchmarks, demonstrating its superiority and generalization ability. The models, datasets, and code are available at https://github.com/pspdada/SENTINEL.

Lightweight Tools

Object Detectors

[Figure 3]

[Figure 4]

###### Additional cost

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Distinguish

[Figure 10]

MLLM

(a) Ultra-large proprietary model/human annotator-dependent methods

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Rewrite

MLLM

[Figure 16]

Domain discrepancy

(b) Response rewriting method

Sample

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

###### Discriminative Effective & Efficient

Distinguish

Non-halluci context

[Figure 22]

Iterative Contextual Bootstrapping

(c) SENTINEL (Ours)

Figure 1. Comparative analysis of data construction strategies for hallucination mitigation in MLLMs. Our proposed approach demonstrates superior efficiency and effectiveness in generating high-quality, domain-specific preference learning datasets, offering a robust solution for reducing hallucination in MLLMs.

the development of general-purpose AI systems [2, 10, 34–

- 36, 44, 63, 85]. However, a critical challenge persists in multimodal settings: the phenomenon of hallucinations [4,
- 37, 52], wherein models generate factually inconsistent or fabricated information that deviates from the image content provided by users. This issue not only degrades user trust and experience but also poses substantial risks in real-world applications of MLLMs, thereby impeding the realization of trustworthy general AI systems [6, 19, 66].

### 1. Introduction

Recent advancements in multimodal large language models (MLLMs) have demonstrated significant progress in aligning visual and textual representations through crossmodal feature integration, marking a pivotal step toward

To address this challenge, recent work has explored enhanced decoding strategies [9, 20, 28] as a means to mitigate hallucinations. While these approaches show promise, they often introduce trade-offs, including increased computational overhead during inference, higher latency, and re-

* Equal contribution. Corresponding author (tianzhuotao@hit.edu.cn).

liance on specific dependencies, which may limit their scalability and practicality in resource-constrained scenarios.

On the other hand, preference alignment methods [32, 50, 58] avoid additional inference costs but face other challenges. As shown in Fig. 1a, many of them rely on large proprietary models (e.g., GPT [1]) [22, 67, 73, 78, 80, 84] or human annotators [14, 74], incurring high costs. Additionally, Fig. 1b highlights that output rewriting [67, 80, 84] can create distributional discrepancies, while Lai et al. [26] and our experiments in Tab. 2 show that out-of-domain training data harms generalization. Therefore, the high costs and the distribution disparities inherent in the curated training data may compromise hallucination mitigation efforts.

Key observations. To address hallucination with greater efficacy and efficiency, we investigate the dynamics of hallucination within the model’s output. Our analysis reveals that hallucination intensity escalates with the length of generated text, while mitigating hallucinations at specific sentences significantly reduces their prevalence in subsequent outputs, as detailed in Figs. 2a and 2b. These findings suggest that early intervention—targeting hallucinations at their initial occurrence—is crucial to preventing their propagation in later generations. This raises a key question: How can we effectively implement an early intervention strategy to address hallucinations of MLLMs as they arise? Our solution. In this work, we propose SENTINEL (Sentence-level Early iNtervention Through IN-domain prEference Learning), which provides early intervention for the initial occurrence of hallucinations during generation. Unlike existing methods, SENTINEL operates without relying on external large language models for rewriting, ensuring that the learning targets remain strictly within the domain of the model’s original outputs. This approach preserves the model’s intrinsic distribution and expression patterns while effectively curbing hallucination propagation.

Specifically, SENTINEL first employs an in-domain candidate bootstrapping strategy, which performs multiple sampling rounds on the current model, extracts objects from the outputs, and applies consistency cross-checking to classify objects as hallucinated, uncertain, or factual. This is followed by a context-aware preference data generation process, which constructs preference pairs using non-hallucinated positive samples and hallucinated negative ones, enhanced by iterative contextual bootstrapping. Finally, context-aware preference learning is performed using the modified context-aware DPO loss, maximizing the likelihood of generating context-coherent positive samples while minimizing hallucinated negative ones. By focusing on captions where hallucinations first emerge, SENTINEL effectively halts their propagation in subsequent outputs.

Experimental results across various benchmarks demonstrate that SENTINEL effectively mitigates object hallucination while preserving the generalization capabilities of

MLLMs. Specifically, on Object Halbench [53] and AMBER [61], hallucinations are reduced by about 92% and 65%, respectively, with consistent improvements on HallusionBench [13]. Furthermore, SENTINEL preserves its performance on VQAv2 [11] and TextVQA [57], and achieving decent gains on both ScienceQA [41] and MM-Vet [76].

To summarize, our contributions are as follows:

- • We demonstrate that early intervention at the first occurrence of hallucination is crucial for preventing its propagation in subsequent model outputs of MLLMs.
- • We propose SENTINEL, which effectively and efficiently mitigates hallucinations without requiring extensive external resources or manual effort.
- • The model-agnostic SENTINEL achieves state-of-the-art performance on hallucination benchmarks without compromising MLLMs’ general capabilities.

### 2. Background and Motivation

In this section, we briefly introduce the foundational concepts and methods relevant to this study in Sec. 2.1, establishing the necessary background. Following this, in Sec. 2.2, we outline our key insights and elucidate the motivations behind our proposed designs.

#### 2.1. Related Work and Preliminaries

Object Hallucination (OH) in Multimodal Large Language Models (MLLMs) is characterized by the generation of text that is semantically coherent yet inconsistent with the visual content of the provided image [4, 52]. To mitigate this issue, recent advancements have focused on innovative decoding strategies, which aim to reduce the prevalence of OH by refining the generation process of MLLMs [7, 9, 20, 28].

Concurrently, preference learning has emerged as an alternative approach for addressing OH, leveraging its capacity to align MLLMs with human expectations for truthfulness and traceability [14, 29, 33]. Notably, the Proximal Policy Optimization algorithm (PPO) [55] enhances model reliability by training an auxiliary reward model to assess response quality and then guide the model in optimizing its outputs based on the reward signals. Moreover, Direct Preference Optimization (DPO) [50] has emerged as a simpler alternative, learning directly from pre-collected feedback without requiring a reward model. The DPO loss is:

πθ(yw|x) πref(yw|x)

LDPO(θ)=− E(x,y

w,yl)∼D log σ β log

(1)

πθ(yl|x) πref(yl|x)

− β log

,

where x=[v,q].

Here, D represents the preference dataset for learning, σ denotes the sigmoid function, πθ indicates the policy

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| |Real<br><br>Hallucinated| | | | | |
| | | | | | | |

- 0.00

0.25

0.50

0.75

1.00

1.25

1.50

- 1.75

Frequency

(a)

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |Eliminate OH<br><br>| | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

1 2 3 4 5 6

Sentence Index

0

50

100

150

200

250

300

350

400

450

FactualObjectCount

Greedy Decode Factual

Eliminate OH Factual

Greedy Decode Hallucinated

Eliminate OH Hallucinated

0

20

40

60

80

100

120

HallucinatedObjectCount

(b)

- Figure 2. Object position distribution in MLLM hallucination analysis. (a) illustrates the progressive deterioration of hallucination effects in Multimodal Large Language Models (MLLMs) with increasing description length in the image captioning task, while (b) demonstrates the effectiveness of early-stage intervention in mitigating the propagation of hallucination.

model under training, πref represents the unchanged reference model, yw stands for the positive sample, and yl represents the negative sample, both based on the input x, which includes the image v and prompt q. The hyperparameter β governs the separation between the policy model and the reference model. Many recent methods [17, 67, 75, 80, 84] leverage DPO to mitigate hallucinations by curating preference data to guide the models. More related works of this study are discussed in Sec. F.

- 2.2. Motivation

0 20 40 60 80 100

Object Position (%)

This section outlines the motivations behind this work. The implementation details of related experiments are provided in Sec. A.

Hallucination grows with text length. To better understand the causes of Object Hallucination (OH), we analyze the distributions of hallucinated and factual objects in image captions generated by MLLMs. Specifically, as shown in Fig. 2a, where the horizontal axis represents the normalized position of an object in the caption (as a percentage), while the vertical axis denotes the normalized frequency (probability density), the blue curve corresponds to objects present in the image, and the orange curve represents hallucinated objects. The comparison reveals that as caption length increases, the model becomes more prone to hallucinations, with fewer factual objects described and more hallucinated ones introduced. This trend is further corroborated by sentence-level analysis in Fig. 2b. These findings lead us to hypothesize that intervening at the initial occurrence of hallucination could be critical in reducing its recurrence in subsequent model outputs.

Early intervention mitigates hallucinations. To evaluate the effectiveness of early intervention in curbing hallucination propagation, we analyze the impact of addressing hallucinations at the sentence level in image captioning tasks.

Specifically, as illustrated in Fig. 2b, eliminating hallucinated objects in the second sentence—compared to vanilla greedy decoding—significantly reduces the likelihood of hallucinated objects in subsequent sentences while increasing the probability of factual objects present in the image. Similar results are observed when addressing hallucinations in the third sentence, as shown in Sec. A.2. These findings underscore the necessity of early intervention to mitigate hallucinations effectively.

To enable early intervention, an open-vocabulary object detector [8, 38] could be employed during inference to verify the presence of the objects generated by the model within the image. While this method effectively reduces hallucinations without sacrificing caption diversity, as demonstrated in Sec. A.2, it is time-consuming; despite the object detector being efficient, the model’s sampling process incurs significant computational overhead.

Consequently, we opt for a preference learning strategy during model training, which mitigates hallucinations without compromising the original inference efficiency.

### 3. Method

#### 3.1. Overview

Existing preference learning methods may use an external model to rewrite sentences or rely on model-generated responses as training data. However, these methods may introduce discrepancies in distribution and expression patterns between the training data and the model’s original output. Hence, we propose SENTINEL, which performs sentence-level early intervention to mitigate object hallucinations through preference learning with in-domain data, without manual effort or dependence on extensive LLMs.

As shown in Fig. 3, the proposed SENTINEL method takes six essential steps. Specifically, Sec. 3.2 presents the process of generating the in-domain candidates containing the factual and hallucinated objects. Subsequently, Sec. 3.3 introduces the construction of preference data pairs derived from these in-domain candidates. These two steps can be integrated into the In-domain Preference Data Construction phase (shown in Algorithm 1). Finally, in Sec. 3.4 we elaborate on how SENTINEL leverages the curated preference data to achieve preference learning.

#### 3.2. In-domain Candidate Bootstrapping

To construct positive and negative preference data pairs without relying on external models for rewriting, we perform multiple sampling rounds on the current model and extract objects from the outputs. We then apply a consistency cross-checking method to classify the model’s output objects into three categories: hallucinated, uncertain, and factual, which are used to construct preference data in subsequent steps. This process is termed In-domain Candidate

###### (1) In-domain Candidate Sampling (2) Object Extraction

###### (3) Object Presence Validation

- Candidate 1 The table is surrounded by chairs, and a vase of flowers can be seen on the table, adding a touch of elegance to the setting.
- Candidate 2 Several pendant lights hang from the ceiling, providing soft lighting over the dining area.
- Candidate 3 The table is set with a variety of cutlery, including forks, knives, and spoons, and there are multiple vases placed on the table, adding a decorative touch.

[Figure 23]

Factual: Uncertain: Hallucinated:

table, chair, vase...

[Figure 24]

[Figure 25]

table vase chair

fork knive spoon

[Figure 26]

[Figure 27]

spoon...

[Figure 28]

[Figure 29]

[Figure 30]

knive, fork...

...

Sample

......

###### (4) Preference Data Construction

Context: The image features a spacious, open kitchen with a dining table and chairs in the

[Figure 31]

(5) Iterative Contextual Bootstrapping

[Figure 32]

Candidate 1 The table is surrounded by chairs, and a vase of

kitchen

table chairs vase flowers

0.

table chairs

flowers can be seen on the table, adding a touch of elegance to the setting.

MLLM center.

[Figure 33]

(6) Context-aware Preference Learning

[Figure 34]

Q: Describe this image in detail.

Candidate 2 Several pendant lights hang from the ceiling,

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

pendant lights

[Figure 39]

[Figure 40]

providing soft lighting over the dining area.

[Figure 41]

Non-hallu context

Image Prompt

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Candidate 3 The table is set with a variety of cutlery, including

[Figure 46]

Preference Dataset

Prefered Response

forks knives

Inferior Response

forks, knives, and spoons, and there are multiple vases placed on the table, adding a decorative touch.

[Figure 47]

[Figure 48]

Factual with rich contextual connections

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Factual

Hallucinated

Context-aware DPO (C-DPO)

MLLM

- Figure 3. The overview of SENTINEL. The proposed SENTINEL takes six essential steps: (1) Generate multiple in-domain responses conditioned on the input image, prompt, and context c. (2) Identify and extract all mentioned objects from each generated sentence. (3) Utilizing two object detectors to validate the existence of extracted objects through cross-referencing. (4) Categorize generated sentences into hallucinated and non-hallucinated groups based on detection results. (5) Extend the generation context with verified non-hallucinated sentences to guide subsequent outputs. (6) Fine-tune the model using the context-aware DPO (C-DPO) loss with the in-domain, styleconsistent, and context-varying preference data.

Booststrapping, as illustrated in Fig. 3 (1)-(3).

types: (1) hallucinated (both models confirm absence), (2) factual (both models confirm presence), and (3) uncertain (conflicting results). Sentences containing hallucinated objects are tagged as “hallucinated”, whereas those only containing factual objects are tagged as “non-hallucinated”, forming positive-negative sample pairs for preference learning. To ensure data quality and minimize detector bias, we ignore uncertain objects.

In-domain candidate sampling. In our approach, we use sampling-based decoding to obtain n candidate samples. This ensures that the positive (yw) and negative (yl) samples are drawn from the same distribution as the current model, preserving consistency in textual styles and linguistic structures. The generation halts upon sentence completion (e.g., detection of a period), at which point sentences are automatically segmented for subsequent discrimination. Object extraction. After generating candidate sentences, we extract the mentioned objects from the text for hallucination detection. To achieve this, we utilize the SceneGraphParser [31] model to transform the textual descriptions into a series of triplet-based scene graphs. By parsing these scene graphs, we identify specific noun entities from the subjects and objects, which are subsequently used as candidate objects for existence verification.

Algorithm 1 In-domain Preference Data Construction

Input: Image v, prompt q, context c (initially empty) Output: Training samples (v, q, c, y+w, yl)

- 1: while Model M does not generate </s> do
- 2: Sample n in-domain candidates si using v, q, and c
- 3: for each sample si do
- 4: Extract entities from the sample
- 5: Validate the presence of entities using object detectors
- 6: Select y+w as context-coherent non-hallucinated sample
- 7: Select yl as hallucinated sample
- 8: Construct preference samples (v, q, c, y+w, yl)
- 9: Append a non-hallucinated sample y+w to the context c

Object presence validation. Following object extraction, we apply cross-checking to validate the presence of candidate objects in the image. Specifically, we utilize two openvocabulary object detectors, GroundingDINO [38] and Yolo World [8], for cross-validation. This approach demonstrates superior performance compared to using a single detector, as shown in Fig. 8 of the ablation study.

#### 3.3. Context-aware Preference Data Generation

With sentences labeled as “hallucinated” or “nonhallucinated” from Sec. 3.2, this section introduces contextaware preference data generation. As illustrated in Fig. 3

The cross-checking results are categorized into three

The scene features a historic brick building with a clock on top, located on a busy street.

The image features a spacious, open kitchen with a dining table and chairs in the center.

[Figure 53]

[Figure 54]

Describe this image in detail.

[Figure 55]

[Figure 56]

Describe this image in detail.

[Figure 57]

[Figure 58]

[Figure 59]

Image

[Figure 60]

###### (1) In-domain Candidate Sampling

[Figure 61]

###### MLLM

###### MLLM

cars bikes

building,

In front of the building, there is a mix of cars and bikes passing by, contributing to the bustling atmosphere.

[Figure 62]

Candidate 1 The table is surrounded by chairs, and a vase of flowers can be seen on the table, adding a touch of elegance to the setting.

car

People

eople are walking along a park path nearby, with a car parked near the entrance. A bus is parked directly outside the building, blocking the road.

[Figure 63]

... Candidate 3 The table is set with a variety of cutlery, including forks, knives, and spoons, and there are multiple vases placed on the table, adding a decorative touch.

bus

[Figure 64]

[Figure 65]

Image

Factual with rich contextual connections

[Figure 66]

[Figure 67]

[Figure 68]

Preference learning data

Factual

[Figure 69]

After steps (2), (3), and (4), categorize Candidate 1 as and Candidate 3 as .

[Figure 70]

improve generalization

Hallucinated

[Figure 71]

[Figure 72]

[Figure 73]

Dataset

Dataset

###### (5) Iterative Contextual Bootstrapping

[Figure 74]

- Figure 4. Categories of in-domain candidates. The in-domain candidates fall into three types. Employing non-hallucinated,

The image features a spacious, open kitchen with a dining table and chairs in the center. The table is surrounded by chairs, and a vase of flowers can be seen on the table, adding a touch of elegance to the setting.

[Figure 75]

context-coherent descriptions (y+w) as positive samples, paired with hallucinated descriptions (yl), enhances the model’s generalization performance and robustness.

[Figure 76]

(1) In-domain Candidate Sampling

[Figure 77]

: current context : context for next iter

MLLM

[Figure 78]

. . .

(4)-(5), this process extracts contextually relevant data, ensuring the training data better represents the model’s output distribution. The specifics are elaborated below.

Figure 5. Visualization of the Iterative Contextual Bootstrapping (ICB) framework. Given an input image and corresponding question, this pipeline iteratively generates diverse contextual samples, enabling robust hallucination mitigation across varying contexts and significantly improving model generalization.

Preference data construction. The preference data is typically composed of the image, the corresponding prompt, the positive sample, the negative sample, and the context (i.e., all generated sentences excluding the current one). In the construction of sample pairs, positive samples yw are selected from the non-hallucinated sentences, while negative samples yl are derived from the hallucinated sentences.

ness across diverse contexts, we introduce the Iterative Contextual Bootstrapping (ICB) strategy, as depicted in Fig. 5.

Specifically, given the query q, the input image v, and the current context ci, we generate multiple candidate outputs by repeatedly sampling from the MLLM. These candidates are then processed through a structured pipeline consisting of (2) object extraction, (3) object presence validation, and (4) preference data construction, as illustrated in Fig. 3. This pipeline is designed to identify a nonhallucinated positive sample y+w and a hallucinated negative sample yl. By aggregating v,q,ci,y+w and yl, we construct a preference data pair (v,q,ci,y+w,yl), which is subsequently appended to the dataset for preference learning.

Subsequently, we partition the positive samples yw into two categories: (1) the context-coherent positive sample y+w, wherein some of the described objects are explicitly referenced in the context, and (2) the context-agnostic positive sample y−w, where none of the objects are mentioned in the context. In essence, the objects described in y+w exhibit a strong correlation with the context, while those in y−w display a weaker or negligible correlation. Illustrative examples are provided in Figs. 3 and 4.

Furthermore, to bootstrap the preference data with different hallucination-free contexts, we construct ci+1 = ci+y+w for the next iteration by appending the positive sample y+w to the current context ci. The updated context ci+1 is then processed through the same procedure as described above to generate a new preference data pair. This iterative approach ensures that the preference data is enriched with progressively more complex and varied contexts, enabling the model to generalize its hallucination mitigation capabilities across different scenarios. The effectiveness of this pipeline is validated and discussed in Sec. B.2.

We observe that the context-coherent sample y+w can effectively mitigate hallucinations without compromising the model’s generalization capabilities, and incorporating y−w as the positive samples results in performance reduction, as shown in Tab. 3. This observation underscores the importance of contextual signals in guiding the model’s generation process. Specifically, the richer contextual information in y+w samples appears to enhance the model’s ability to preserve contextual coherence and prioritize salient content, resulting in performance improvements [16].

Iterative Contextual Bootstrapping (ICB). The proposed SENTINEL framework is designed to enable early intervention for mitigating hallucinations in generative models. Given the context c, which represents the hallucination-free content preceding the current output, the model is trained to distinguish between a non-hallucinated positive sample y+w and a hallucinated negative sample yl. To enhance robust-

#### 3.4. Context-aware Preference Learning

The preference data generated through the processes outlined in Sec. 3.2 and Sec. 3.3 can be formally represented as (x,c,y+w,yl), where x is the input, including the image v and the prompt q, c denotes the context, y+w is the context-

Hallucination benchmarks General benchmarks Object HalBench [53] AMBER [61] HallusionBench [13] VQAv2 [11] TextVQA [57] ScienceQA [41] MM-Vet [76] Resp.↓ Ment.↓ CHAIR↓ Hal.↓ Cog.↓ Question Acc.↑ Acc.↑ Acc.↑ Image Acc.↑ Overall↑

Model Method

baseline 52.7 28.0 8.4 35.5 4.0 46.86 78.5 58.2 66.8 31.0 VCD [28] 51.3 25.9 9.1 39.8 4.2 - 77.0 56.1 68.7 29.8 OPERA [20] 45.3 22.9 6.5 28.5 3.1 - 78.2 58.2 68.2 30.3 DoLa [9] 44.0 25.1 6.2 27.7 2.9 - 76.3 56.6 67.5 30.8 EFUF [68] 39.3 22.6 5.8 28.2 3.1 47.03 78.1 57.2 66.4 31.2 HA-DPO [80] 37.0 20.9 6.7 30.9 3.3 47.74 77.6 56.7 69.7 30.6 POVID [84] 33.4 16.6 5.3 28.7 3.0 46.59 77.2 56.6 68.8 31.8 CLIP-DPO [45] - - 3.7 16.6 1.3 - - 56.4 67.6 RLAIF-V [75] 7.8 4.2 2.8 15.7 0.9 35.43 75.2 55.1 68.2 29.9 TPO [17] 5.6 3.2 3.6 20.5 1.6 40.12 75.9 55.3 67.1 25.7 Ours 4.3 2.6 2.9 14.6 1.2 47.56 78.4 58.2 69.2 32.6

LLaVA-v1.5-7B

baseline 46.0 23.0 6.9 31.9 3.3 46.43 80.0 61.2 71.6 36.0 VCD [28] 43.7 21.6 7.8 36.2 3.7 - 78.5 59.5 72.0 33.7 vanilla-DPO [67] 6.7 3.6 2.8 15.5 1.6 46.41 79.2 60.4 71.8 35.0 HSA-DPO [67] 5.3 3.2 2.1 13.4 1.2 46.14 78.3 60.0 71.3 33.7 Ours 3.3 1.9 2.7 11.7 0.9 46.77 79.9 61.0 72.8 36.2

LLaVA-v1.5-13B

Table 1. Comparison of hallucination mitigation methods in MLLMs: effectiveness and general capabilities. This evaluation highlights the best and second-best results in bold and underlined, respectively. All comparisons are performed under identical model size constraints. “Resp.” and “Ment.” denote response-level and mention-level hallucination rates, while “Hal.” and “Cog.” represent the Hallucination Score and Cognitive Score, respectively. More evaluation details are provided in Sec. D.

coherent positive sample, and yl is the negative sample.

The learning objective is to guide the model, conditioned on the input x and the context c, to maximize the likelihood of generating the contextually coherent positive sample y+w while minimizing the likelihood of producing the negative sample yl. To achieve this, we adapt the Direct Preference Optimization (DPO) loss by incorporating the context c as part of the input. We term this modified loss as contextaware DPO (C-DPO), which is formulated as follows:

πθ(y+w|x′) πref(y+w|x′)

LC-DPO(θ) = − E(x′,y+w,yl)∼D log σ β log

πθ(yl|x′) πref(yl|x′)

− β log

,

where x′ =[x,c] = [v,q,c].

(2)

In C-DPO, the context c is excluded from the loss computation, and gradients are only derived from the discrimination between y+w and yl. This design ensures that the model focuses on learning the contextual coherence of the positive sample without being directly influenced by the context during gradient updates. Further discussions and comparisons between the proposed C-DPO and the standard DPO are provided in Sec. C.3.

### 4. Experiments

In this section, we conduct comprehensive experiments to evaluate the effectiveness of our SENTINEL in reducing hallucinations while improving the general abilities of the model. We first introduce the experimental setup in Sec. 4.1, then present the main results in Sec. 4.2, and finally conduct ablation studies in Sec. 4.3 to analyze our method’s effectiveness. More results are in Secs. C and D.

#### 4.1. Experimental Setup

Training. To ensure a fair comparison, we follow the settings of prior works [17, 20, 27, 28, 35, 45, 54, 64, 67, 67, 68, 75, 77, 80, 84], using LLaVA-v1.5 as the reference model across all experiments. For data collection, we prompt the model with detailed image descriptions [74] to generate training data, with images sourced from the Visual Genome dataset [24]. Model training is conducted using C-DPO (Eq. (2)) in combination with LoRA [18], and optimized with AdamW [40]. The 7B and 13B models are trained for one epoch on 8.6K and 7.0K samples, respectively, with learning rates of 2 × 10−7 and 3 × 10−7. Additional training details are provided in Sec. C.

Evaluation benchmarks. We evaluate the hallucination extent and general capabilities of our SENTINEL method across multiple benchmarks. For hallucination evaluation, we use widely adopted benchmarks, including Object HalBench [53], AMBER [61], and HallusionBench [13]. To assess general capabilities, we employ VQAv2 [11], TextVQA [57], ScienceQA [41], and MM-Vet [76]. Further details of these benchmarks are provided in Sec. D.1.

Baselines. To show the effectiveness of our method, we compare SENTINEL with several state-of-the-art (SOTA) methods. Specifically, VCD [28], OPERA [20], and DoLa [9] focus on enhanced decoding strategies, while HADPO [80], POVID [84], CLIP-DPO [45], RLAIF-V [75], and TPO [17] leverage preference training. Additionally, Vanilla DPO applies the original DPO objective Eq. (1) using training data from HSA-DPO, while EFUF [68] is an unlearning-based approach. Details are in Sec. D.2.

#### 4.2. Main Results

Comparison with recent SOTAs. As shown in Tab. 1, we compare our method with baseline methods across several

[Figure 79]

[Figure 80]

State Attribute

[Figure 81]

LLaVA-v1.5-7B

###### 61.8

Yes, the persons in the image are likely to be living in the house, as they are walking through the hallway and interacting with each other. The presence of chairs, a dining table, and other furniture in the room suggests that it is a living space. The people in the image are dressed in period clothing, which indicates that they might be attending a special event or gathering in the house. The fact that they are walking together and engaging in conversation further supports the idea that they are residents of the house.

###### 67.6

61.3 57.7

65.6 64.0

55.5

61.2

[Figure 82]

[Figure 83]

LLaVA-v1.5-7B + SENTINEL (Ours)

Number 88.7

Existence

74.8 69.9 67.5 65.1

81.1 82.4 85.3

Do the persons live in this house? Answer it and give the rationale.

No, the persons in the image do not live in the house. They are dressed in period costumes and are walking through the hallway of a large, ornate mansion. This suggests that they are likely actors or performers participating in a historical reenactment, a theatrical performance, or a film set. The presence of the costumes and the grand setting indicate that the scene is staged and not a typical everyday living situation.

66.7

67.4

80.4

Required capabilities: Recognition, Knowledge and Language generation

67.7

80.9

81.1

Figure 7. Qualitative results of SENTINEL. Our method can effectively eliminate hallucinations in MLLMs while enhancing the model’s general capabilities.

82.1

70.6

Action Relation

Figure 6. Impact on different hallucination types. Comparison between multiple methods shows that our method reduces hallucination in all six hallucination types.

Object HalBench AMBER MM-Vet Resp. ↓ Ment. ↓ Acc ↑ F1 ↑ Overall ↑

Method

LLaVA-v1.5-7B 52.7 27.9 71.5 74.1 31.1 Ours (8.6K (y+w,yl)) 4.3 2.6 76.1 79.3 32.6 Ours (8.6K Rewrited (y+w,yl)) 4.8↑0.5 2.9↑0.3 75.0↓1.1 78.0↓1.3 31.3↓1.3

benchmarks. The results demonstrate that SENTINEL significantly reduces the models’ hallucination rate. Specifically, for the 7B model, our method achieves a 4.3 responselevel (Resp.) and a 2.6 mention-level (Ment.) hallucination rate. Compared to the previous SOTA method, TPO [17], which achieves a 5.6 response-level and 3.2 mention-level hallucination rate, our proposed SENTINEL surpasses it by further reducing hallucinations by a total of 24% on Object Halbench. Furthermore, even on the 13B model, compared to the baseline, which achieves 6.9 CHAIR, 31.9 Hallucination score (Hal), and 4.0 Cognitive score (Cog), our proposed SENTINEL significantly improves performance, achieving 2.7 CHAIR, 11.7 Hal score, and 0.9 Cog score, respectively. These results demonstrate that our method is also effective on larger models.

Table 2. Effects of rewritten samples. Rewriting the preference training samples (y+w, yl) results in performance reduction.

lucinations but also improves general capability.

Qualitative results. To further demonstrate the effectiveness of our method, we conduct case studies. As shown in Fig. 7, the baseline model misinterprets the image content, leading to an incorrect conclusion. In contrast, our model effectively understands image content and provides a more detailed and precise description. This example highlights how our approach effectively reduces hallucinations while simultaneously enhancing the model’s overall capability. We conduct more case studies in Sec. G.

#### 4.3. Ablation Studies

Comprehensive hallucination mitigation. To further evaluate the effect of our method on various hallucination types, we conducted experiments on the discriminative part of the AMBER [61] benchmark and report the F1 scores for each hallucination type. As shown in Fig. 6, LLaVA-v1.5 with SENTINEL outperforms the baseline across all six hallucination types, demonstrating the effectiveness of our method in mitigating various hallucination issues. Notably, for the Existence hallucination type, our method improves the 7B model by 6.3 and the 13B model by 7.6 compared to the baseline. Detailed results are provided in Sec. D.4.

In this section, we conduct a series of ablation experiments to further analyze the effectiveness of SENTINEL. More discussions can be found in Sec. D.5.

Effectiveness of data style consistency. To analyze the effect of preference data style, we train the model using rewritten data for comparison. Specifically, we instructed GPT-4 [1] to rewrite (y+w, yl) while ensuring coherence with the context c. As shown in Tab. 2, the rewriting results show performance degradation in reducing hallucinations and general ability. This highlights the advantage of our approach in preserving data style consistency. Furthermore, we conduct a detailed analysis in Sec. D.5, which shows that models trained on in-domain data converge to a lower preference optimization loss and achieve better differentiation between positive and negative samples, whereas training with rewritten data provides fewer improvements.

Improved general capabilities. As shown in Tab. 1, SENTINEL enhances the general capabilities of the model on multiple benchmarks. Specifically, SENTINEL demonstrates stable performance on VQAv2 and TextVQA, whereas previous methods designed for hallucination mitigation suffer from significant performance degradation. Moreover, on ScienceQA and MM-Vet, our method not only surpasses previous hallucination mitigation methods but also outperforms the vanilla LLaVA-v1.5. These results show that SENTINEL not only significantly mitigates hal-

Effectiveness of cross-checking. To validate the effectiveness of cross-checking for object presence, we conduct experiments using only the Grounding DINO or YOLO World for detection. In this setting, if the model determines that an

Object HalBench TextVQA ScienceQA MM-Vet Resp. ↓ Ment. ↓ Acc I-Acc↑ Overall ↑

Data Scale

Method

LLaVA-v1.5-7B - 52.7 27.9 58.2 66.8 31.1 y+w 100% 8.6K 4.3 2.6 58.2 69.2 32.6 y+w 50% + y−w 50% 10.0K↑1.4K 4.8↑0.5 2.9↑0.3 58.1↓0.1 69.0↓0.2 32.0↓0.6 y−w 100% 14.0K↑5.4K 4.6↑0.3 3.0↑0.4 58.1↓0.1 68.7↓0.5 31.6↓1.0

- Table 3. Comparison between context-coherent samples y+w and context-agnostic samples y−w. This table reveals that incorporating context-coherent samples y+w yields better performance.

Method

Object HalBench[53] AMBER[61] Resp. ↓ Ment. ↓ CHAIR ↓ Hal ↓ Cog ↓

LLaVA-v1.5-7B 52.7 27.9 8.4 35.5 4.0 Non-hallucinated context 4.3 2.6 2.9 14.6 1.2 Natural context 8.6 4.7 3.3 15.6 1.5 Hallucinated context 14.3 7.1 3.9 18.6 1.8

- Table 4. Comparison between different new context formation strategies during the iterative contextual bootstrapping

pipeline. Appending non-hallucinated sample y+w to the existing context ci yields superior performance compared to incorporating hallucinated samples yl or greedy decoding contexts, highlighting the effectiveness of our proposed approach.

object is absent, it is directly classified as hallucinated. As shown in Fig. 8, leveraging two object detectors for crossvalidation significantly outperforms using a single model, effectively reducing the hallucination rate.

Effect of different yw types on model performance. As shown in Tab. 3, we conduct a detailed study on the impact of different types and proportions of the positive data yw on model performance. The results show that y+w samples, which contain richer contextual information, enhance the model’s generalization ability while achieving similar hallucination reduction with less data.

Effect of non-hallucinated sentences as context c. To analyze the impact of using non-hallucinated sentences as context c, we evaluate three different settings for generating new context: selecting a hallucinated sentence, selecting a non-hallucinated sentence, or directly using a model-generated sentence from greedy decoding. As shown in Tab. 4, using a non-hallucinated sentence as context improves the model’s ability to distinguish hallucinations and significantly reduces their occurrence in the output. This further demonstrates that intervening at the first instance of hallucination is critical for minimizing its recurrence.

Effect of data scale. To analyze the impact of the training data scale on our method, we train the model using different dataset sizes (1k/2k/4k/6k/8k) and evaluate its performance on Object Halbench. As shown in Fig. 8, our method further mitigates model hallucinations as data scale up. This demonstrates the potential and scalability of SENTINEL. Furthermore, since our method does not rely on ultra-large proprietary models or human annotators for dataset construction, it can efficiently collect more training data.

Integrating with existing preference learning methods.

Object level Hal Rate

Response level Hal Rate

50

25

40

20

30

15

91.8%

90.7%

20

10

10

5

0 1 2 4 6 8 8.6

0 1 2 4 6 8 8.6

Training Data Size (k)

Training Data Size (k)

Grounding DINO Only YOLO World Only SENTINEL

Figure 8. Impact of training data quantity on hallucination rate in Object Halbench [53]. The results show that SENTINEL demonstrates better efficiency, effectiveness, and scalability, while effectively reducing hallucination rates across varying data scales.

Object HalBench [53] AMBER [61] HallusionBench [13] TextVQA [57] MM-Vet [76] Resp. ↓ Ment. ↓ Acc↑ F1↑ Question Acc↑ Acc↑ Overall ↑

Method

LLaVA-v1.5-7B 52.7 28.0 71.5 74.1 46.86 58.2 31.0

HA-DPO [80] 37.0↓29.8% 20.9↓25.4% 74.2↑2.7 78.0↑3.9 47.74↑0.88 56.7↓1.5 30.6↓0.4 HA-DPO + Ours (6K) 8.0 ↓78.4% 4.6↓78.0% 76.6↑2.4 84.2↑6.2 48.72↑0.98 57.1↑0.4 33.5↑2.9

Table 5. Effectiveness of combining the proposed SENTINEL with HA-DPO. Only a subset of our training data is needed to reduce hallucinations while enhancing generalization effectively.

To further demonstrate SENTINEL’s generalization, we explore integrating with previous hallucination mitigation approaches. As shown in Tab. 5, incorporating a subset of our data into the GPT-generated dataset collected by HADPO [80] effectively mitigates hallucinations while significantly enhancing the model’s generalization. This highlights SENTINEL’s complementarity with other preference learning methods and its potential for broader applicability.

### 5. Concluding Remarks

Summary. In this work, we address the critical challenge of hallucinations in multimodal large language models (MLLMs). While prior methods have shown promise, they often introduce significant computational overhead, rely on costly resources, or create distributional discrepancies. To tackle these issues, we propose SENTINEL, a framework that intervenes early at the onset of hallucinations by leveraging in-domain preference learning. SENTINEL employs an in-domain candidate bootstrapping strategy, context-aware preference data generation, and a context-aware DPO (C-DPO) loss to effectively curb the propagation of hallucinations while preserving the model’s intrinsic distribution. Experimental results across multiple benchmarks demonstrate the superiority of SENTINEL, establishing it as a scalable, efficient, and model-agnostic solution for enhancing the reliability of MLLMs.

Limitation. Currently, as SENTINEL lacks the capability to incorporate spatiotemporal information, it might not be able to effectively address the hallucination issues that require long-term reasoning in video MLLMs. This limitation highlights the need for further research in this area.

Acknowledgments. This work is supported by the Shenzhen Science and Technology Innovation Program (JCYJ20240813105901003, KJZD20240903102901003), Guangdong Basic and Applied Basic Research Foundation (2025A1515011546), and National Key R&D Program of China (2024YFE0215300).

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 2, 7, 6, 8, 9

- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A versatile vision-language model for understanding, localization. arXiv preprint arXiv:2308.12966,

2023. 1, 9

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025. 8, 9
- [4] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024. 1, 2, 9
- [5] Steven Bird. NLTK: The natural language toolkit. In Proceedings of the COLING/ACL 2006 interactive presentation sessions, 2006. 2
- [6] Long Chen, Oleg Sinavski, Jan H¨unermann, Alice Karnsund, Andrew James Willmott, Danny Birch, Daniel Maund, and Jamie Shotton. Driving with LLMs: Fusing objectlevel vector modality for explainable autonomous driving. In IEEE International Conference on Robotics and Automation (ICRA), 2024. 1
- [7] Zhaorun Chen, Zhuokai Zhao, Hongyin Luo, Huaxiu Yao, Bo Li, and Jiawei Zhou. HALC: Object hallucination reduction via adaptive focal-contrast decoding. arXiv preprint arXiv:2403.00425, 2024. 2, 9
- [8] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. YOLO-World: Real-time openvocabulary object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,

2024. 3, 4

- [9] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. DoLa: Decoding by contrasting layers improves factuality in large language models. arXiv preprint arXiv:2309.03883, 2023. 1, 2, 6, 7, 9
- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. InstructBLIP: Towards general-purpose visionlanguage models with instruction tuning. Advances in Neural Information Processing Systems, 2023. 1, 9
- [11] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question

- answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. 2, 6, 5, 9
- [12] Yuzhe Gu, Wenwei Zhang, Chengqi Lyu, Dahua Lin, and Kai Chen. Mask-DPO: Generalizable fine-grained factuality alignment of LLMs. arXiv preprint arXiv:2503.02846, 2025. 4
- [13] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. HallusionBench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2, 6, 8, 5, 9
- [14] Anisha Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024. 2, 7
- [15] Yudong Han, Liqiang Nie, Jianhua Yin, Jianlong Wu, and Yan Yan. Visual perturbation-aware collaborative learning for overcoming the language prior problem. arXiv preprint arXiv:2207.11850, 2022. 9
- [16] Zongbo Han, Zechen Bai, Haiyang Mei, Qianli Xu, Changqing Zhang, and Mike Zheng Shou. Skip\n: A simple method to reduce hallucination in large vision-language models. arXiv preprint arXiv:2402.01345, 2024. 5
- [17] Lehan He, Zeren Chen, Zhelun Shi, Tianyu Yu, Jing Shao, and Lu Sheng. A topic-level self-correctional approach to mitigate hallucinations in MLLMs. arXiv preprint arXiv:2411.17265, 2024. 3, 6, 7
- [18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 6
- [19] Mingzhe Hu, Shaoyan Pan, Yuheng Li, and Xiaofeng Yang. Advancing medical imaging with language models: A journey from N-grams to ChatGPT. arXiv preprint arXiv:2304.04920, 2023. 1
- [20] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. OPERA: Alleviating hallucination in multimodal large language models via over-trust penalty and retrospection-allocation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 1, 2, 6, 7, 9
- [21] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024. 7, 8
- [22] Liqiang Jing and Xinya Du. FGAIF: Aligning large visionlanguage models with fine-grained AI feedback. arXiv preprint arXiv:2404.05046, 2024. 2, 7
- [23] Liqiang Jing, Ruosen Li, Yunmo Chen, and Xinya Du. FaithScore: Fine-grained evaluations of hallucinations in large vision-language models. arXiv preprint arXiv:2311.01477,

2023. 9

- [24] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual Genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision,

2017. 6, 3

- [25] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. LISA: Reasoning segmentation via large language model. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 9
- [26] Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-DPO: Step-wise preference optimization for long-chain reasoning of LLMs. arXiv preprint arXiv:2406.18629, 2024. 2
- [27] Seongyun Lee, Sue Hyun Park, Yongrae Jo, and Minjoon Seo. Volcano: mitigating multimodal hallucination through self-feedback guided revision. arXiv preprint arXiv:2311.07362, 2023. 6
- [28] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 1, 2, 6, 7, 9
- [29] Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, and Lingpeng Kong. Silkie: Preference distillation for large visual language models. arXiv preprint arXiv:2312.10665, 2023. 2
- [30] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-Gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2023. 9

- [31] Zhuang Li, Yuyang Chai, Terry Yue Zhuo, Lizhen Qu, Gholamreza Haffari, Fei Li, Donghong Ji, and Quan Hung Tran. FACTUAL: A benchmark for faithful and consistent textual scene graph parsing. arXiv preprint arXiv:2305.17497,

2023. 4, 1, 2

- [32] Sheng-Chieh Lin, Luyu Gao, Barlas Oguz, Wenhan Xiong, Jimmy Lin, Scott Yih, and Xilun Chen. FLAME: Factualityaware alignment for large language models. Advances in Neural Information Processing Systems, 2024. 2
- [33] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023. 2
- [34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 2023. 1, 9
- [35] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 6, 1, 7
- [36] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge, 2024. 1, 8, 9

- [37] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253, 2024. 1, 9
- [38] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. In Proceedings of the European Conference on Computer Vision,

2024. 3, 4, 1

- [39] Yijun Liu, Jiequan Cui, Zhuotao Tian, Senqiao Yang, Qingdong He, Xiaoling Wang, and Jingyong Su. Typicalnessaware learning for failure detection. arXiv preprint arXiv:2411.01981, 2024. 9
- [40] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6, 3
- [41] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 2022. 2, 6, 5, 9
- [42] Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems, 2024. 9
- [43] Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, XianSheng Hua, and Ji-Rong Wen. Counterfactual VQA: A cause-effect look at language bias. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2021. 9
- [44] OpenAI. GPT-4V(ision) system card, 2023. 1, 6, 9
- [45] Yassine Ouali, Adrian Bulat, Brais Martinez, and Georgios Tzimiropoulos. CLIP-DPO: Vision-language models as a source of preference for fixing hallucinations in LVLMs. In Proceedings of the European Conference on Computer Vision, 2024. 6
- [46] Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with DPO-Positive. arXiv preprint arXiv:2402.13228, 2024. 9
- [47] Shangpin Peng, Weinong Wang, Zhuotao Tian, Senqiao Yang, Xing Wu, Haotian Xu, Chengquan Zhang, Takashi Isobe, Baotian Hu, and Min Zhang. Uni-DPO: A unified paradigm for dynamic preference optimization of LLMs. arXiv preprint arXiv:2506.10054, 2025. 9
- [48] Tianyuan Qu, Longxiang Tang, Bohao Peng, Senqiao Yang, Bei Yu, and Jiaya Jia. Does your vision-language model get lost in the long video sampling dilemma? arXiv preprint arXiv:2503.12496, 2025. 9
- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, 2021. 3, 9
- [50] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a

- reward model. Advances in Neural Information Processing Systems, 2023. 2, 4, 9
- [51] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. ZeRO: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, 2020. 3
- [52] Vipula Rawte, Amit Sheth, and Amitava Das. A survey of hallucination in large foundation models. arXiv preprint arXiv:2309.05922, 2023. 1, 2, 9
- [53] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. arXiv preprint arXiv:1809.02156, 2018. 2, 6, 8, 1, 4, 5, 7, 9
- [54] Pritam Sarkar, Sayna Ebrahimi, Ali Etemad, Ahmad Beirami, Sercan O¨ Arık, and Tomas Pfister. Data-augmented phrase-level alignment for mitigating object hallucination. arXiv preprint arXiv:2405.18654, 2024. 6, 7
- [55] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 2
- [56] Tong Shao, Zhuotao Tian, Hang Zhao, and Jingyong Su. Explore the potential of CLIP for training-free open vocabulary semantic segmentation. In Proceedings of the European Conference on Computer Vision, 2024. 9
- [57] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019. 2, 6, 8, 4, 5, 9
- [58] Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D Manning, and Chelsea Finn. Fine-tuning language models for factuality. In The Twelfth International Conference on Learning Representations, 2023. 2
- [59] Zhuotao Tian, Michelle Shu, Pengyuan Lyu, Ruiyu Li, Chao Zhou, Xiaoyong Shen, and Jiaya Jia. Learning shape-aware embedding for scene text detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4234–4243, 2019. 9
- [60] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 2
- [61] Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, et al. AMBER: An LLM-free multi-dimensional benchmark for MLLMs hallucination evaluation. arXiv preprint arXiv:2311.07397, 2023. 2, 6, 7, 8, 5
- [62] Junjie Wang, Bin Chen, Yulin Li, Bin Kang, Yichi Chen, and Zhuotao Tian. DeCLIP: Decoupled learning for openvocabulary dense perception. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 9
- [63] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 9

- [64] Kai Wu, Boyuan Jiang, Zhengkai Jiang, Qingdong He, Donghao Luo, Shengzhi Wang, Qingwen Liu, and Chengjie Wang. NoiseBoost: Alleviating hallucination with noise perturbation for multimodal large language models. arXiv preprint arXiv:2405.20081, 2024. 6
- [65] Yike Wu, Yu Zhao, Shiwan Zhao, Ying Zhang, Xiaojie Yuan, Guoqing Zhao, and Ning Jiang. Overcoming language priors in visual question answering via distinguishing superficially similar instances. In Proceedings of the 29th International Conference on Computational Linguistics, 2022. 9
- [66] Zhenyu Wu, Ziwei Wang, Xiuwei Xu, Jiwen Lu, and Haibin Yan. Embodied task planning with large language models. arXiv preprint arXiv:2307.01848, 2023. 1
- [67] Wenyi Xiao, Ziwei Huang, Leilei Gan, Wanggui He, Haoyuan Li, Zhelun Yu, Fangxun Shu, Hao Jiang, and Linchao Zhu. Detecting and mitigating hallucination in large vision language models via fine-grained AI feedback. arXiv preprint arXiv:2404.14233, 2024. 2, 3, 6, 7, 9
- [68] Shangyu Xing, Fei Zhao, Zhen Wu, Tuo An, Weihao Chen, Chunhui Li, Jianbing Zhang, and Xinyu Dai. EFUF: Efficient fine-grained unlearning framework for mitigating hallucinations in multimodal large language models. arXiv preprint arXiv:2402.09801, 2024. 6, 7, 9
- [69] Senqiao Yang, Jiaming Liu, Ray Zhang, Mingjie Pan, Zoey Guo, Xiaoqi Li, Zehui Chen, Peng Gao, Yandong Guo, and Shanghang Zhang. LiDAR-LLM: Exploring the potential of large language models for 3d LiDAR understanding. arXiv preprint arXiv:2312.14074, 2023. 9
- [70] Senqiao Yang, Tianyuan Qu, Xin Lai, Zhuotao Tian, Bohao Peng, Shu Liu, and Jiaya Jia. An improved baseline for reasoning segmentation with large language model. arXiv preprint arXiv:2312.17240, 2023.
- [71] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. VisionZip: Longer is better but not necessary in vision language models. arXiv preprint arXiv:2412.04467, 2024. 9
- [72] Senqiao Yang, Zhuotao Tian, Li Jiang, and Jiaya Jia. Unified language-driven zero-shot domain adaptation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 9
- [73] Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. Science China Information Sciences,

2024. 2, 1, 7

- [74] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. RLHF-V: Towards trustworthy MLLMs via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2, 6, 7
- [75] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, et al. RLAIF-V: Open-source AI feedback leads to super GPT-4V trustworthiness. arXiv preprint arXiv:2405.17220, 2024. 3, 6, 5, 7, 9
- [76] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang.

- MM-Vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 2, 6, 8, 4, 9
- [77] Zihao Yue, Liang Zhang, and Qin Jin. Less is More: Mitigating multimodal hallucination from an EOS decision perspective. arXiv preprint arXiv:2402.14545, 2024. 6, 1
- [78] Mengxi Zhang, Wenhao Wu, Yu Lu, Yuxin Song, Kang Rong, Huanjin Yao, Jianbo Zhao, Fanglong Liu, Haocheng Feng, Jingdong Wang, et al. Automated multi-level preference for MLLMs. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2, 7
- [79] Tiancheng Zhao, Peng Liu, and Kyusong Lee. OmDet: Large-scale vision-language multi-dataset pre-training with multimodal detection network. IET Computer Vision, 2024. 3
- [80] Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, and Conghui He. Beyond hallucinations: Enhancing LVLMs through hallucination-aware direct preference optimization. arXiv preprint arXiv:2311.16839, 2023. 2, 3, 6, 8, 1, 4, 7, 9
- [81] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations). Association for Computational Linguistics, 2024. 8
- [82] Zhisheng Zhong, Chengyao Wang, Yuqi Liu, Senqiao Yang, Longxiang Tang, Yuechen Zhang, Jingyao Li, Tianyuan Qu, Yanwei Li, Yukang Chen, et al. Lyra: An efficient and speech-centric framework for omni-cognition. arXiv preprint arXiv:2412.09501, 2024. 9
- [83] Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. Analyzing and mitigating object hallucination in large vision-language models. arXiv preprint arXiv:2310.00754,

2023. 9

- [84] Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024. 2, 3, 6, 7
- [85] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1, 9

## Mitigating Object Hallucinations via Sentence-Level Early Intervention Supplementary Material

### Overview

This material provides supplementary details to the main paper, including the following sections:

- • (A) Motivation Details

- – (A.1) Object Position Distribution
- – (A.2) Decode Based Early Intervention

- • (B) Method Details

- – (B.1) Object Extraction
- – (B.2) Iterative Contextual Booststrapping
- – (B.3) Selection of Object Detector
- – (B.4) Treatment of Uncertain Objects

- • (C) Training Details

- – (C.1) Training Dataset
- – (C.2) Training Setup
- – (C.3) Training Objective

- • (D) Evaluation Details

- – (D.1) Evaluation Benchmarks
- – (D.2) Evaluation Counterparts
- – (D.3) Evaluation Settings
- – (D.4) Evaluation Results
- – (D.5) Details of Ablation Study

- • (E) SENTINEL with Other Baselines
- • (F) Related Work
- • (G) Additional Case Studies

### A. Motivation Details

In this section, we deepen the discussion supporting the key observations from the main paper.

#### A.1. Object Position Distribution

Following the approach of Caption Hallucination Assessment with Image Relevance [53], we select 300 images from the COCO2014 dataset and use the provided captions and segmentation annotations as references to determine whether the objects described by the model exist in the images. As shown in the main paper Fig. 2, as the model generates longer outputs, the number of real objects described decreases while hallucinated objects increase, indicating that hallucinations of the model become more severe with output length. Notably, towards the end of the generation (around the last 10% tokens), both the number of hallucinated and real objects decreases. This is because, towards the end of the image description, the model tends to conclude with abstract summaries about the atmosphere or emotions rather than providing concrete object descriptions.

450

120

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |Eliminate OH<br><br>| | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

HallucinatedObjectCount

400

100

FactualObjectCount

350

300

80

250

60

200

150

40

100

20

50

0

0

1 2 3 4 5 6

Sentence Index

Greedy Decode Factual

Greedy Decode Hallucinated

Eliminate OH Factual

Eliminate OH Hallucinated

Figure 9. Effect of intermediate hallucination mitigation on subsequent generations. Showing the effectiveness of early-stage intervention in mitigating the propagation of hallucinations.

Object HalBench AMBER Resp. ↓ Ment. ↓ CHAIR ↓ Hal ↓ Cog ↓

Model Method

baseline 52.7 27.9 8.4 35.5 4.0 Woodpecker [73] 39.6 26.4 - - VCD [28] 52.7 27.3 9.1 39.8 4.2 OPERA [20] 40.0 21.9 6.5 28.5 3.1 EOS [77] 40.0 22.2 6.4 27.4 2.6 HA-DPO [80] 37.0 20.9 6.7 30.9 3.3 Decode based early intervention

LLaVA-v1.5-7B [35]

33.5 17.6 5.5 26.8 2.6

Table 6. Effectiveness of decode based early intervention.

#### A.2. Decode Based Early Intervention

As a preliminary investigation, we explore a training-free approach to mitigating object hallucinations in MLLMs. In essence, our method dynamically verifies each generated sentence against the image content and filters out any hallucinated ones before proceeding. Specifically, for the image captioning task, we sample multiple candidate sentences (n = 5) from the model’s output, stopping generation at the first period. These candidate sentences are then parsed using SceneGraphParser [31] to extract mentioned objects. We subsequently employ an open-vocabulary object detector, Grounding DINO [38], to verify the existence of these objects in the image. A sentence without hallucinated objects is selected as the current generated sentence, and then continues generating the subsequent content.

This approach effectively prevents the further propagation of hallucinations. As shown in the main paper Fig. 2b, even when applied at just a single sentence, eliminating hallucinations as early as the second sentence significantly reduces the likelihood of generating hallucinated objects in subsequent outputs. A similar effect is observed when intervention occurs only at the third sentence, as shown in Fig. 9.

Time Taken by Different Methods Time Distribution by Type

12000

10000

8000

Time(s)

6000

4000

2000

MLLM Generate (83.8%) Parse (9.4%) Detect (6.5%) Others (0.2%)

0

Greedy DoLa VCDEarlyInterventionOPERA

| |
|---|

| |
|---|

| |
|---|

- Figure 10. Time cost analysis of decode-based methods. Decode-based early intervention increases inference time, primarily due to the additional generation steps required by MLLM sampling, whereas the object detector remains highly efficient.

When this early intervention strategy is applied throughout the entire caption generation process, as shown in Tab. 6, it effectively mitigates object hallucinations when evaluated on the Object Halbench [53] benchmark. However, as illustrated in Fig. 10, it increases inference time, primarily due to the additional sampling time of the MLLM, while the object detector remains highly efficient. These findings highlight the detector’s role as both an effective and computationally efficient component, reinforcing its potential for constructing high-quality training data for hallucination mitigation.

### B. Method Details

In this section, we detail our methods for extracting concrete objects from models’ outputs in Sec. B.1, and propose iterative contextual bootstrapping (ICB) to enhance robustness in Sec. B.2. In Sec. B.3, we discuss the selection of the object detector. Finally, in Sec. B.4, we describe how we handle uncertain objects. Our approach reduces hallucinations efficiently without relying on large auxiliary models.

#### B.1. Object Extraction

In this section, we detail our approach to extracting the mentioned objects from the model’s output automatically and efficiently. Our objective is to obtain identifiable and concrete entity descriptions, following a structured pipeline.

First, we employ SceneGraphParser [31] to convert the input descriptions into a series of triplets representing relationships within the scene. Specifically, each triplet is treated as a (subject, predicate, object) tuple. For example:

“A little black cat sits on a chair next to a table.” is parsed into the following structured triplets:

(cat, is, little) (cat, is, black) (chair, next to, table) (cat, sit on, chair)

Next, we extract entities from these triplets. We apply the following rules:

- • If the predicate belongs to {“is”, “are”}, it represents an attribute relationship. In this case, we consider only the subject as a potential entity.
- • Otherwise, both the subject and object are considered potential entities.

To refine the entity extraction process, we leverage the SpaCy natural language processing library to analyze the part of speech (POS) of the extracted candidates and filter out words that are neither nouns nor proper nouns. Furthermore, we utilize NLTK’s WordNet Lemmatizer [5] in conjunction with a lexicographic filtering mechanism to exclude non-entity nouns. Specifically, we examine the lexicographer category of each word, and if it falls within the following non-concrete categories, it is removed:

noun.feeling, noun.attribute, noun.state, noun.shape, noun.time, noun.quantity, noun.cognition, noun.event, noun.communication, noun.relation, noun.act, noun.location.

Our method effectively extracts entities without the need for large auxiliary models such as GPT-4 [1] or LLaMA-270B [60]. Instead, it relies solely on lightweight NLP tools and libraries, ensuring both high extraction accuracy and maintaining an open-vocabulary nature.

#### B.2. Iterative Contextual Booststrapping

To ensure robustness across different contexts, we introduce the iterative contextual bootstrapping (ICB) strategy, as shown in the main paper Fig. 5. By leveraging contextually bootstrapped data, early intervention can be seamlessly integrated into diverse contexts, effectively mitigating hallucinations and enhancing robustness.

To further investigate the impact of iterative contextual bootstrapping (ICB), we conduct an ablation study where we exclude ICB and instead sample a non-hallucinated description y+w only at the first occurrence of hallucination, using it as the positive sample during constructing pairs, while the original hallucinated description serves as the negative sample yl. We then train the model using the same method and dataset size mentioned in the main paper. The results, as presented in Tab. 7, demonstrate that our approach, when incorporating ICB, exhibits greater robustness and effectively reduces hallucinations across different scenarios.

#### B.3. Selection of Object Detector

Detectors are more cost-effective for providing training guidance for MLLMs than human annotators. SENTINEL is not constrained to particular detectors; any model with open-world recognition ability can be employed. As shown

Object HalBench AMBER MM-Vet Resp. ↓ Ment. ↓ CHAIR ↓ Hal ↓ Cog ↓ Overall ↑

Method

LLaVA-v1.5-7B 52.7 27.9 8.4 35.5 4.0 31.1 Ours w/ ICB 4.3 2.6 2.9 14.6 1.2 32.6 Ours w/o ICB 5.3↑1.0 3.2↑0.6 3.1↑0.2 14.9↑0.3 1.4↑0.2 31.8↓0.8

- Table 7. Effect of Iterative Contextual Booststrapping. Iterative Contextual Bootstrapping (ICB) enables early intervention to be seamlessly integrated into various contexts, effectively mitigating hallucinations and ensuring robustness across different scenarios.

Method

Object HalBench Resp.↓ Ment.↓

LLaVA-v1.5-7B 52.7 28.0 OmDet [79] 19.3 9.9 Grounding DINO [38] 14.3 7.7 YOLO World [8] 12.3 6.9 Grounding DINO [38] + YOLO World [8] 6.6 3.8

- Table 8. Results with different detectors. We observe that detector OmDet [79] often produces false positives, identifying objects that do not exist in the images, which may lead to less reliable results. Generally, detectors with more human-like real-world perception abilities yield better performance.

Method

Object HalBench MM-Vet Resp. ↓ Ment. ↓ Overall ↑

LLaVA-v1.5-7B 52.7 28.0 31.0 Ignore uncertain 4.3 2.6 32.6 Uncertain as factual 10.3 6.9 31.8 Uncertain as hallucinated 8.3 5.0 32.0

- Table 9. Treatments of uncertain objects. Ignoring uncertain objects can improve the quality of training data, thereby enhancing final model performance.

in Tab. 8, more effective detectors lead to superior performance, and the cross-validation technique effectively mitigates the phenomenon of false positives.

#### B.4. Treatment of Uncertain Objects

As mentioned in the main paper, we ignore uncertain objects to maintain data quality and reduce detector bias. We also conduct ablation studies that treat uncertain objects alternately as factual or hallucinated. Tab. 9 shows that ignoring uncertain objects yields better results. We hypothesize that it is because 1) ‘uncertain’⇒‘factual’ may bring hallucinations to the context during iterative contextual bootstrapping (ICB), contradicting the early intervention strategy based on the hallucination-free contexts. 2) ‘uncertain’⇒‘hallucinated’ may introduce noisy and ambiguous negative samples for preference learning.

### C. Training Details

In this section, we provide a detailed overview of the preference training process. The dataset used for training is described in Sec. C.1, the training setup is outlined in Sec. C.2, and the training objective is analyzed in detail in Sec. C.3.

Model

###### LLaVA-v1.5-7B LLaVA-v1.5-13B

Setting

LLM Vicuna-v1.5-7B Vicuna-v1.5-13B Vision encoder CLIP ViT-L336px/14[49] Projector mlp2x gelu Learning rate 2e-6 3e-6 Batch size per GPU 16 8 Trainable parameters LoRA trains only LLM’s linear layers. LoRA rank r 128 LoRA alpha α 256 LoRA beta β 0.1 Projector lr 0 Learning rate scheduler Cosine Optimizer AdamW [40] Model max lenght 2048 Weight decay 0. Epochs 1 Global batch size 64 Memory optimization ZeRO stage 2 [51]

Table 10. Training hyperparameters used in our experiments.

#### C.1. Training Dataset

Visual Genome. Visual Genome (VG) [24] is a publicly available large-scale vision-language dataset that provides dense annotations for about 108K images, with each image containing an average of 21 objects, 18 attributes, and 18 object relationships. In addition to object annotations, VG includes 1.7 million visual question-answering pairs in a multi-choice format, covering six question types: What, Where, When, Who, Why, and How. Compared to traditional VQA datasets, VG offers a more balanced distribution of question types while also serving as one of the most comprehensive resources for bridging visual concepts with language. In our study, VG images are utilized for constructing the training dataset.

Training Data. We use approximately 4K images from VG for training dataset construction, selected based on their appropriate information density and appropriate level of object diversity. Notably, we do not utilize any labels or ground-truth annotations from VG or other datasets when constructing the preference dataset. Instead, our approach automatically and efficiently generates highly discriminative preference training data in a cost-effective manner.

#### C.2. Training Setup

We strictly follow the official setup provided by LLaVA to ensure reproducibility. The details of the training hyperparameters used in our training are presented in Tab. 10.

#### C.3. Training Objective

As shown in the main paper Eq. (2), we employ the contextaware DPO (C-DPO) objective to train the model to differentiate between hallucinated and non-hallucinated content at the first occurrence of hallucination, aiming to mitigate its propagation. In this section, we provide a detailed anal-

ysis of (1) the rationale for excluding context c from the loss computation, (2) the key differences between our proposed C-DPO and the standard DPO, and (3) a comparison between our training objective and Mask-DPO [12].

Why mask context in loss calculation? We implemented a pseudocode for calculation based on the context-aware DPO (C-DPO) formula. As shown in Algorithm 2, to compute the C-DPO loss, we need to evaluate the log probabilities (logps) of the output tokens given an input. If we do not mask out the context during loss computation, the context c remains identical in both positive and negative samples. Since the context and its preceding tokens are the same, for the policy model, the logps of the context tokens will be the same across both forward passes. This adds an identical term C to both policy chosen logps and policy rejected logps, which cancels out in the policy logratios computation at line 7, leaving the loss unaffected.

From a gradient perspective, since C is derived from the same model parameters θ based on identical preceding tokens in both forward passes, its gradient remains the same due to the autoregressive nature of the model. As a result, this gradient term cancels out as well and does not affect model training. Therefore, to reduce unnecessary computation and mitigate potential numerical errors, we exclude the context c from the loss calculation in C-DPO.

Algorithm 2 Pseudocode for C-DPO Training

Input: Training sample (v, q, c, y+w, yl) Output: C-DPO loss

- 1: import torch
- 2: import torch.nn.functional as F
- 3:
- 4: def get cdpo loss(self, (v, q, c, y+w, yl)) → torch.Tensor:

- 5: # policy model forward pass
- 6: policy chosen logps = model.dpo forward((v, q, c, y+w))

- 7: policy rejected logps = model.dpo forward((v, q, c, yl))

- 8: policy logratios = policy chosen logps - policy rejected logps

- 9:
- 10: # reference model forward pass
- 11: with torch.no grad():

- 12: ref chosen logps = ref model.dpo forward((v, q, c, y+w))

- 13: ref rejected logps = ref model.dpo forward((v, q, c, yl))

- 14: ref logratios = ref chosen logps - ref rejected logps

- 15:
- 16: # compute C-DPO loss
- 17: logits = policy logratios - ref logratios

- 18: loss = -F.logsigmoid(dpo beta * logits)

- 19: return loss.mean()

▷ model.dpo forward() returns the sum of the log probabilities of all tokens that have not been masked out.

Comparison with Standard DPO. To validate the effectiveness of our proposed context-aware DPO (C-DPO), we conducted an additional experiment using a standard DPO for training. In this setup, no context c was included, and both yw and yl are complete image captions based on the given image v and prompt q. yw consisted of sentences with minimal hallucinations (using non-hallucinated context and ensuring the current sentence itself is hallucination-

free until the end of generating), while yl contained sentences with maximal hallucinations (using hallucinated context and ensuring the current sentence itself contained hallucinations until the end of generating). Both methods were trained on the same scale of data (8.6K samples).

As shown in Tab. 11, our proposed context-aware DPO (C-DPO) more effectively guides the model in distinguishing hallucinated content from non-hallucinated content, leading to improved hallucination suppression while maintaining generalization capabilities.

To further analyze the underlying reasons, we track the training dynamics of both objectives, including policy model log probabilities (logps) and loss. As illustrated in Fig. 11, the standard DPO exhibits greater logps variations between yw and yl during training due to the substantial differences between sentence pairs. Prior studies by Rafailov et al. [50] and Zhao et al. [80] suggest that such variability can dominate gradient updates, potentially compromising training stability. This instability may hinder the model’s ability to capture long-range dependencies, leading to slower convergence and a more gradual reduction in training loss.

Object HalBench[53] TextVQA[57] MM-Vet[76] Resp. ↓ Ment. ↓ Acc ↑ Overall ↑

Method

LLaVA-v1.5-7B 52.7 27.9 58.2 31.0 C-DPO Eq. (2) (8.6K data) 4.3 2.6 58.2 32.6 Standard DPO Eq. (1) (8.6K data) 10.1↑5.8 5.5↑2.9 58.1↓0.1 31.7↓0.9

Table 11. Effectiveness of C-DPO. Compared to standard DPO, C-DPO enables the model to better learn to distinguish between correct and incorrect responses at the onset of hallucination, effectively mitigating hallucinations from the outset.

Differences between our objective and DPO with Mask. Gu et al. [12] propose a preference learning approach that selectively retains factually correct sentences from preferred samples while avoiding penalties on factual content within inferior samples, thereby mitigating ambiguity issues inherent in preference learning. While this method effectively prioritizes high-quality samples, it primarily relies on masking certain parts of the training data without fully considering their potential impact on model learning. As demonstrated in the main paper Tab. 4, within our workflow, the choice of context c—which is masked from loss calculation during training—plays a crucial role in shaping the final training outcomes. This highlights the importance of carefully considering these factors to ensure that our approach effectively guides the model toward learning accurate and reliable knowledge.

### D. Evaluation Details

In this section, we provide detailed information about the evaluation process. The evaluation benchmarks we used are described in Sec. D.1, where we showcase the strong performance of our method. In Sec. D.2, we outline the

###### C-DPO

###### DPO

55

18

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

60

20

PolicyLogProbability

22

65

24

70

26

75

28

80

30

85

32

90

34

0 20 40 60 80 100 120

0 20 40 60 80 100 120

Step

Step

chosen rejected

(a) Logps comparison

###### C-DPO

DPO

0.70

0.70

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.65

0.65

0.60

0.60

Loss

0.55

0.55

0.50

0.50

0.45

0.45

0 20 40 60 80 100 120

0 20 40 60 80 100 120

Step

Step

(b) Loss comparison

- Figure 11. Comparison between C-DPO and standard DPO during model training. The proposed C-DPO promotes more stable gradient updates, enhancing training stability.

counterparts used for comparison. In Sec. D.3, we present the detailed evaluation setup. In Sec. D.4, we provide detailed results from some of the experiments. Additionally, in Sec. D.5, we present specific details of the ablation studies.

#### D.1. Evaluation Benchmarks

We provide a detailed description of the evaluation benchmarks used in our study.

- • Object HalBench. Object HalBench [53] is a widely used benchmark for assessing common object hallucinations in detailed image descriptions. Following [75], we incorporate eight diverse prompts to enhance evaluation stability. We report two key metrics: the response-level hallucination rate (Resp.), which measures the proportion of responses containing hallucinations, and the mentionlevel hallucination rate (Ment.), which quantifies the percentage of hallucinated object mentions.
- • AMBER. AMBER [61] is a widely used metric for hallucination evaluation, assessing the frequency of hallucinatory objects in model-generated responses. In the generative component of AMBER, hallucination is quantified using the following three metrics: CHAIR score:

′

obj ∩ Aobj) len(Robj′ )

len(R

CHAIR(R) = 1 −

. (3)

where Robj′ represents the set of objects mentioned in the model’s response, and Aobj denotes the set of objects that actually exist in the image.

Hal score: Measures the proportion of responses containing hallucinations. A response is considered hallucinatory if CHAIR(R) ̸= 0. It is computed as:

1 if CHAIR(R) ̸= 0, 0 otherwise.

Hal(R) =

(4)

Cog score: This metric assesses the alignment between model-generated hallucinations and human cognitive tendencies. It measures the probability of the model generating objects from a predefined set of hallucinatory target objects Hobj, calculated as:

′

obj ∩ Hobj) len(Robj′ )

len(R

Cog(R) =

. (5)

In the discriminative component of AMBER, hallucination severity is evaluated based on six factors: object existence, attributes, relationships, state, number, and actions. We report the F1 score to assess the model’s performance across these aspects.

- • HallusionBench. HallusionBench [13] is a benchmark designed to assess multimodal large language models (MLLMs) in image-context reasoning, specifically focusing on hallucination and illusion phenomena. By incorporating a carefully curated set of challenging reasoning tasks, HallusionBench enables a systematic evaluation of both language-based hallucinations and visiondriven illusions. To quantify model performance, we report the overall accuracy across all questions, covering both straightforward and complex cases.
- • VQAv2. VQAv2 [11] is a widely used general visual question answering benchmark that enhances dataset balance by collecting complementary images for each question.
- • TextVQA. TextVQA [57] is a benchmark designed for text-rich visual question answering, requiring models to not only recognize textual content within images but also reason about the extracted information. This task evaluates a model’s ability to accurately identify text characters while effectively handling the inherent noise present in OCR-generated outputs.
- • ScienceQA. ScienceQA [41] is a multiple-choice benchmark designed to evaluate zero-shot generalization in scientific question answering. It features multimodal questions covering a diverse range of science topics, with annotated answers supported by corresponding lectures and explanations. These annotations provide general external knowledge and specific reasoning for deriving the correct answer. In our study, we conduct experiments on the image subset of ScienceQA to assess model performance in

multimodal scientific reasoning.

- • MM-Vet. MM-Vet [76] is a comprehensive benchmark designed to assess a model’s ability to engage in visual conversations across diverse tasks. It evaluates response correctness and helpfulness through GPT-4 [1] scoring. The dataset includes a wide range of image types, such as real-world scenes, artworks, statistical graphs, and memes, paired with open-ended questions that require multimodal reasoning. MM-Vet focuses on six core evaluation capabilities: recognition, knowledge, optical character recognition (OCR), spatial awareness, language generation, and math.

#### D.2. Evaluation Counterparts

We compare our SENTINEL approach with various methods designed to mitigate hallucinations in MLLMs, all of which are trained on or applied to LLaVA-v1.5 [35] to ensure fairness.

- • VCD. VCD [28] is a training-free method designed to mitigate hallucinations in vision-language models by enhancing their focus on image content. It achieves this by contrasting output distributions derived from both original and distorted visual inputs. This contrastive approach helps the model better align its responses with actual image content rather than relying on spurious correlations. The computational cost of a single inference step using VCD is approximately twice that of standard greedy decoding.
- • OPERA. OPERA [20] addresses hallucination in multimodal language models through two strategies: OverTrust Penalty and Retrospection-Allocation. The OverTrust Penalty reduces overconfidence by adjusting model logits during beam search, while RetrospectionAllocation revisits previously generated tokens to correct potential errors, improving response accuracy.
- • DoLa. DoLa [9] enhances factual accuracy by leveraging contrastive decoding across different model layers. This approach effectively reduces the generation of incorrect facts and consistently improves truthfulness in model responses.
- • EFUF. EFUF [68] mitigates hallucinations without requiring paired data by employing gradient ascent and three specialized loss functions. It applies gradient descent when encountering real objects and gradient ascent when detecting hallucinated ones, effectively refining the model’s output.
- • HA-DPO. HA-DPO [80] formulates hallucination mitigation as a preference selection task, training the model to prefer non-hallucinated responses when given two outputs for the same image. To ensure training stability, it incorporates a causal language modeling objective into the DPO loss. Additionally, both positive and negative samples are rewritten in GPT-4’s style to maintain stylistic

- consistency.
- • POVID. POVID [84] highlights the role of inferior responses in training and enhances them by modifying images and introducing extra hallucinations via GPT4V [44]. The approach then fine-tunes LLaVA-1.5-7B using a set of 17K preference data.
- • RLAIF-V. RLAIF-V [75] employs a “Feedback From Peer” strategy, where the overall response score is derived by aggregating scores from decomposed sub-responses, reducing reliance on costly, ultra-large proprietary models like GPT4. The model is trained using an iterative alignment approach, conducting DPO training over four iterations, with each iteration consisting of four epochs.
- • TPO. TPO [17] is a self-correction approach that enables the model to mitigate its hallucinations at the topic level. Using a deconfounded strategy, it replaces each topic in the response with either the best or worst alternatives generated by the model. This process creates more contrasting pairwise preference feedback, improving the quality of feedback without requiring human intervention or proprietary models.
- • HSA-DPO. HSA-DPO [67] first trains a hallucination detection model using datasets constructed by GPT-4V [1]. This model is then leveraged in a detect-then-rewrite pipeline to generate 6K preference data for training. Finally, MLLMs are aligned using the proposed hallucination severity-aware DPO method.

#### D.3. Evaluation settings

Our overall evaluation setup strictly follows the guidelines provided by LLaVA-v1.5 [35], with certain hyperparameter settings detailed in Tab. 12.

#### D.4. Evaluation Results

Detailed results of MM-Vet. We present the detailed results of the MM-Vet [76] benchmark in Tab. 13. The results indicate that, compared to existing methods, our approach achieves the most significant improvement on the 7B model, with an increase of 1.6 points. Notably, for the 13B model, while other methods exhibit varying degrees of performance degradation, our method continues to yield improvements. This demonstrates the effectiveness of our approach in enhancing both the correctness and helpfulness of model responses.

Detailed results of AMBER. We present the detailed results of the discriminative part of the AMBER [76] benchmark in Tab. 14. The results show that some of the existing methods may experience a decline in performance across certain specific hallucination categories. In contrast, our approach demonstrates improvements in every specific hallucination category, regardless of whether the 7B or 13B model is used. Notably, for the Existence hallucination type, our method improves the 7B model by 6.3 and the

###### Method Parameters Value

Amplification Factor α 1.0 Adaptive Plausibility Threshold 0.1 Diffusion Noise Step 500

VCD [28]

Repetition Penalty θ 1.2 Adaptive Plausibility Threshold β 0.1

DoLa [9]

Pre-mature Layers [0,2··· ,32]

Self-attention Weights Scale Factor θ 50 Attending Retrospection Threshold 15

OPERA [20]

Beam Size 3 Penalty Weights 1

- Table 12. Evaluation hyperparameters of decode-based methods.

Method Rec OCR Know Gen Spat Math Overall LLaVA-v1.5-7B [35] 35.9 23.3 17.1 22.0 25.9 11.5 31.0±0.2

VCD [28] 34.5 21.9 18.3 20.6 24.8 3.8 29.8↓1.2 OPERA [20] 34.9 21.6 18.7 21.1 25.7 7.7 30.3↓0.7 DoLa [9] 36.1 21.3 19.4 20.9 26.9 7.7 30.8↓0.2 EFUF [68] 36.5 21.4 17.1 19.5 27.9 7.7 31.2↑0.2 HA-DPO [80] 35.5 22.1 18.3 21.9 26.3 7.7 30.6↓0.4 POVID† [84] - - - - - - 31.8↑0.8 RLAIF-V [75] 34.4 23.4 18.7 23.7 27.7 7.3 29.9↓1.1 TPO [17] 31.8 15.4 16.7 19.6 22.1 7.7 25.7↓5.3

- Ours 37.7 23.1 22.7 25.6 26.8 7.7 32.6↑1.6 Ours + HA-DPO [80] 38.4 25.0 21.2 23.7 29.3 7.7 33.5↑2.5 LLaVA-v1.5-13B [35] 39.7 28.8 23.2 24.2 34.5 11.5 36.0 VCD [28] 38.7 24.4 22.4 26.4 30.1 7.7 33.7↓2.3 vanilla-DPO [67] 38.4 29.7 17.9 21.0 35.6 11.5 35.0↓1.0 HSA-DPO [67] 35.9 28.4 16.4 18.9 34.5 15.0 33.7↓2.3

- Ours 38.9 30.2 22.6 23.1 32.7 15.0 36.2↑0.2

- Table 13. Full evaluation results of MM-Vet benchmark. †indicates that the results are from [84]. 13B model by 7.6 compared to the baseline.

Method Existence Attribute State Number Action Relation Overall LLaVA-v1.5-7B 82.4 64.0 57.7 69.9 81.1 67.7 74.1 VCD [28] 81.1↓1.3 65.6 61.8 67.5↓2.4 80.9↓0.2 66.7↓1.0 73.9↓0.2 DoLa [9] 87.6 67.5 62.1 72.8 82.4 56.3↓11.4 77.8 EFUF [68] 85.3 61.2↓2.8 55.5↓2.2 65.1↓4.8 80.4↓0.7 67.4↓0.3 75.0 HA-DPO [80] 88.2 66.1 56.5↓1.2 78.5 82.3 68.7 78.0 Ours 88.7↑6.3 67.6↑3.6 61.3↑3.6 74.8↑4.9 82.1↑1.0 70.6↑2.9 79.3↑5.2

LLaVA-v1.5-13B 78.5 70.0 66.0 74.2 82.2 44.9 73.1 VCD [28] 78.5 71.7 69.0 73.6↓0.6 81.6↓0.6 45.6 73.8 Ours 86.1↑7.6 72.6↑2.6 66.6↑0.6 81.6↑7.4 82.6↑0.4 51.5↑6.6 78.7↑5.6

- Table 14. Full evaluation results of AMBER’s descriminative part. We report F1 scores for each category and the overall score.

#### D.5. Details of Ablation Study

In this section, we provide more specific details of the ablation studies to validate the effectiveness of our method.

Effect of style consistency. Many preference training methods adopt rewriting techniques to construct nonhallucinated training samples [17, 67, 80]. To validate the negative impact of rewritten training data on the model’s generalization performance, we follow the approach of HADPO [80] and design prompts to instruct GPT to rewrite the preference training samples. Specifically, we prompt GPT4[1] to rewrite yw and yl in a different style while ensuring coherence with the given context. The prompt template is provided in Tab. 18, and the results are presented in the main paper Tab. 2.

To evaluate how our in-domain training data affects the model’s linguistic qualities, we adopt the approach from [54] and use GPT-4o-mini [21] as a judge. Responses are rated on a scale of 0 to 10 across four aspects: grammatical correctness, fluency, detailedness, and choice of words. We assess the model’s performance on 300 image description tasks from Object HalBench [53]. The evaluation prompt is shown in Tab. 17. As demonstrated in Tab. 15, our training not only preserves the model’s linguistic capabilities but also improves the detailedness of the descriptions.

Method Grammatical Correctness Fluency Detailedness Choice Of Words LLaVA-v1.5-7B 9.92 9.28 8.21 8.94

- SENTINEL 9.97↑0.05 9.53↑0.25 8.32↑0.11 8.97↑0.03 LLaVA-v1.5-13B 9.95 9.44 8.29 8.95

- SENTINEL 9.98↑0.03 9.60↑0.16 8.40↑0.11 8.98↑0.03

- Table 15. Language quality evaluation results. Our in-domain training data preserves the model’s language quality in image detail description tasks while improving the level of detailedness.

To further investigate the impact of rewritten data on training, we analyze the log probabilities (logps) and loss trends of the policy model when trained with in-domain data versus rewritten data, as shown in Fig. 12. Our observations indicate that the rewritten data, due to its deviation from the model’s original output style and linguistic domain, significantly lowers the logps of both positive and negative samples. Additionally, the rewriting process obscures the fundamental distinction between positive and negative samples (i.e., whether hallucinations are present), thereby weakening the model’s ability to distinguish between them and diminishing the effectiveness of the training signal. As a result, models trained on in-domain data converge to a lower loss and achieve superior differentiation between positive and negative samples, whereas training with rewritten data fails to provide comparable improvements.

Effect of data scaling up. Since our proposed SENTINEL does not rely on ultra-large proprietary models [22, 67, 73, 78, 80, 84] or human annotators [14, 74] for preference learning dataset construction, it can efficiently collect more training data. As shown in Tab. 16, although RLHF-V [74] leverages high-quality human-annotated training data to achieve a lower hallucination rate with fewer training samples, their high cost limits the scalability of the training data. Our method enables cost-effective scaling up, leading to improved model performance.

Method Data Scale

Object HalBench[53] AMBER[61] Resp. ↓ Ment. ↓ Acc ↑ F1 ↑

LLaVA-v1.5-7B - 52.7 28.0 71.5 74.1 RLHF-V [74] 1.4K 12.2 7.5 72.6 75.0

SENTINAL 2.0K 39.0 20.0 72.2 74.9 SENTINAL 8.6K 4.3 2.6 76.1 79.3

- Table 16. Impact of training data quantity. The results show that SENTINEL demonstrates better efficiency and scalability.

In Domain Data

Rewrited Data

18

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

42

20

44

PolicyLogProbability

22

46

24

26

48

28

50

30

52

32

54

34

0 20 40 60 80 100 120

0 20 40 60 80 100 120

Step

Step

chosen rejected

(a) Training log probabilities (logps) comparison

In domain

Rewrited Data

0.70

0.70

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.65

0.65

0.60

0.60

Loss

0.55

0.55

0.50

0.50

0.45

0.45

0 20 40 60 80 100 120

0 20 40 60 80 100 120

Step

Step

(b) Training loss comparison

- Figure 12. Impact of rewriting on the training process. Training with rewritten data fails to achieve the same level of convergence, resulting in higher final loss and weaker differentiation between positive and negative samples, demonstrating the necessity of indomain training data.

Complement with existing preference learning methods. HA-DPO [80] employs a GPT-4 [1]-based rewriting approach to modify both positive and negative samples in the preference training data, ensuring stylistic consistency between them. However, this rewriting process introduces stylistic discrepancies between the training data and the target model’s original outputs, potentially affecting its generalization ability.

To assess the effectiveness of in-domain preference learning data, we augment the HA-DPO [80] training dataset (approximately 4.4K samples) with a subset of our constructed dataset (6K samples from the full 8.6K) and train LLaVA-v1.5-7B under the same training settings as HA-DPO. As shown in the main paper Tab. 5, integrating even a partial set of our training data significantly reduces hallucinations while enhancing the model’s overall performance. These results further demonstrate that our sentencelevel preference training approach is complementary to existing sample-level preference learning methods.

### E. SENTINEL with Other Baselines

In this section, we explore the effectiveness of our SENTINEL approach when applied to other baselines, specifically LLaVA-v1.6 [36], Qwen2-VL [3] and Qwen2.5VL [3]. The results are presented in Tab. 19. The findings

Following is a detailed image description. Your task is to assess the response on the following criteria:

- 1. Grammatical Correctness: Analyze the response for grammar, punctuation, and syntax accuracy.
- 2. Fluency: Evaluate whether the response flows smoothly, reads naturally, and maintains coherence throughout.
- 3. Detailedness: Check if the response provides sufficient and relevant detail to address the topic comprehensively, without redundancy or unnecessary information.
- 4. Choice of Words: Assess if the words used are appropriate, varied, and effectively convey the intended message. Rate each criterion on a scale from 0 to 10, where 0 indicates poor quality and 10 signifies an excellent response. Here is the image description to evaluate: {description}

Your response should be in this format: Grammatical Correctness: SCORE Fluency: SCORE Detailedness: SCORE Choice of Words: SCORE

- Table 17. Prompts for linguistic quality evaluation. Responses are rated on a scale of 0 to 10 across four aspects: grammatical correctness, fluency, detailedness, and choice of words.

Rewrite training data

Please help me rewrite the given sentences in a style different from the original. You will be provided with three parts: “context” refers to the previously generated sentences, and “option one” and “option two” represent two choices for the sentence that follows the context. Your goal is to make the new versions from the original while preserving all details and information. Avoid adding any new information or changing the original meaning. Please rewrite the two options that differ in tone, structure, word choice, and phrasing compared to the original, while ensuring coherence and natural flow with the given context.

The format of your output should be: Option one: ... Option two: ...

The sentences are: Context: {context} Option one: {y win} Option two: {y lose}

- Table 18. Prompts for rewriting. We prompt GPT-4 [1] to rewrite

y+w and yl in a style different from the original while ensuring coherence with the given context c to show the effect of rewriting on the model’s generalization performance.

indicate that our SENTINEL approach consistently reduces hallucinations across a range of model families and sizes, while preserving or even enhancing overall performance, thereby demonstrating its robustness and effectiveness.

During these experiments, to generate training data for each target model, we simply replace the sampling model within the SENTINEL framework with the corresponding model, thereby demonstrating SENTINEL’s modelagnostic design. For training, we employ the widely used LLaMA-Factory [81] framework to ensure fairness and reproducibility. Evaluation follows the same protocol described above1. All training data, configuration details, and associated resources will be released publicly.

1For efficiency, in this set of experiments we use the GPT-4o [21] model for HallusionBench [13] evaluation, which makes these results not directly comparable to those reported for the benchmark in the main paper.

Hallucination benchmarks General benchmarks Object HalBench [53] HallusionBench [13] VQAv2 [11] TextVQA [57] ScienceQA [41] MM-Vet [76] Resp.↓ Ment.↓ Question Acc.↑ Acc.↑ Acc.↑ Image Acc.↑ Overall↑

Model

LLaVA-v1.6-vicuna-7B 15.3→5.0 10.1→3.4 36.73→37.80 81.5→81.5 59.4→59.4 74.3→74.2 40.9→45.4 LLaVA-v1.6-vicuna-13B 13.7→4.0 7.7→2.6 41.10→41.36 82.2→82.2 63.6→63.5 77.7→78.0 47.8→48.5 Qwen2-VL-2B-Instruct 15.3→2.3 8.6→1.7 41.28→42.16 81.5→81.5 78.3→78.5 76.9→77.4 49.4→49.8 Qwen2-VL-7B-Instruct 14.3→4.8 8.5→4.0 51.55→53.41 83.7→83.8 82.2→82.2 85.7→86.9 62.7→62.8 Qwen2.5-VL-7B-Instruct 15.0→4.7 9.2→2.8 52.00→52.08 84.0→84.0 77.7→77.7 88.6→88.5 72.0→72.2

Table 19. Comparison of hallucination mitigation methods with other baseline models: effectiveness and general capabilities (baseline→SENTINEL). This evaluation highlights the best and second-best results in bold and underlined, respectively. All comparisons are performed under identical model size constraints. “Resp.” and “Ment.” denote response-level and mention-level hallucination rates, while “Hal.” and “Cog.” represent the Hallucination Score and Cognitive Score, respectively.

### F. Related Work

Multimodal large language models. In recent years, vision-language models (VLMs) have made remarkable progress [39, 49, 56, 59, 62, 72]. With the advancement of large language models (LLMs), multimodal large language models (MLLMs) have achieved impressive alignment between visual and textual representations through crossmodal feature integration, marking a crucial milestone toward truly general-purpose AI systems [2, 3, 10, 25, 30, 34– 36, 44, 48, 63, 69–71, 82, 85]. However, mitigating hallucination and building reliable models for real-world scenarios remain significant challenges.

Object Hallucination. Object Hallucination (OH) refers to the phenomenon where MLLMs generate text that is semantically coherent but misaligned with the given image [4, 37, 52]. Prior studies suggest that this issue may arise during generation due to an over-reliance on linguistic priors and insufficient attention to visual features [15, 43, 65]. Furthermore, research indicates that hallucination tends to intensify over time [23, 83].

Mitigate OH with improved decoding strategies. Several approaches have explored enhanced decoding strategies to mitigate object hallucination. VCD [28] enhances the model’s focus on image content during generation by applying contrastive decoding between the original image and a noise-corrupted version. DoLa [9] improves factual accuracy by leveraging contrastive decoding across layers to better surface factual knowledge and reduce incorrect outputs. OPERA [20] introduces Over-Trust Penalty and Retrospection-Allocation to address hallucination in multimodal language models. HALC [7] reduces object hallucination through an adaptive focal-contrast decoding approach, incorporating a dynamic auto-focal grounding mechanism for real-time token correction and a refined beam search strategy to effectively suppress hallucinations while maintaining text quality.

Mitigate OH by preference learning. Preference learning

is a powerful paradigm for aligning large language models with human judgments and values. Recently, Direct Preference Optimization (DPO) [50] and its variations [42, 46, 47] have made preference learning more accessible and easier to integrate. Another line of research on mitigating OH employs preference learning to tackle object hallucination by reformulating it as a preference optimization problem. These approaches construct high-quality, stylistically consistent positive-negative sample pairs to enhance model training. Rewriting is an effective method for obtaining “non-hallucinated” training data. HA-DPO [80] utilizes GPT [1] to directly detect and rewrite the model’s original output, ensuring that both positive and negative samples undergo rewriting. HSA-DPO [67] distills a smaller hallucination detection model from the proprietary model GPT and applies it to detect hallucinations and refine responses through rewriting. In contrast, RLAIF [75] does not employ rewriting; instead, it constructs datasets using the “Feedback from Peer” approach, leveraging open-source models’ outputs as feedback. This method directly utilizes the model’s full outputs as both positive and negative samples.

Eliminating OH via unlearning. EFUF [68] employs an unlearning-based approach to mitigate OH in MLLMs. Specifically, it applies gradient ascent when hallucinated objects are detected during the model’s generation, effectively suppressing hallucinations in a targeted and efficient manner. This method enables a precise removal of hallucinated content during captioning without compromising the model’s overall performance.

### G. Additional Case Studies

In this section, we provide more case studies to further illustrate the effectiveness of our proposed SENTINEL in mitigating hallucinations and enhancing its generalization capabilities. To ensure a fair comparison, results for both LLaVA-v1.5 and SENTINEL are obtained using greedy decoding, while GPT-4V refers to gpt-4-vision-preview [44].

[Figure 84]

LLaVA-v1.5-7B

[Figure 85]

The image features a large blue bus parked in a parking lot. The bus is quite long, occupying a significant portion of the scene. The bus is parked under a tree, providing shade and a pleasant atmosphere. There are several people in the scene, with one person standing close to the bus on the left side, another person standing further away on the right side, and a third person standing near the middle of the scene. The people appear to be either waiting for the bus or simply passing by.

[Figure 86]

LLaVA-v1.5-7B + SENTINEL (Ours)

The image features a large blue and white bus parked in a parking lot. The bus is prominently displayed in the scene, occupying a significant portion of the image. The bus is likely a tour bus, as it is parked in a lot and not in motion. There is a man walking towards the bus, possibly preparing to board or just passing by. The man is located on the left side of the bus, closer to the front. The scene captures the moment when the man is approaching the bus, creating a sense of anticipation for the journey ahead.

[Figure 87]

Describe this image.

[Figure 88]

GPT-4V

The image shows a full-size blue tour bus parked on the side of a street. The bus features a colorful design with white stars and portraits of a man's profile silhouetted in white against a darker blue background. These repeated profile images create a distinctive pattern across the bus's exterior. There is the text on the side of the bus, but I cannot provide details on what it says due to image policy. A pedestrian is walking in front of the bus, moving towards the left side of the frame. The setting appears to be a parking area or a bus stop, with green trees in the background that suggest an urban park or a street lined with some greenery.

[Figure 89]

LLaVA-v1.5-7B

[Figure 90]

The image features a person riding a bicycle through a lush green field. The cyclist is positioned in the middle of the scene, surrounded by the vibrant grass. The field appears to be a mix of grass and bushes, creating a picturesque and serene environment for the cyclist's journey. The person is actively pedaling the bike, enjoying the outdoor experience and the beauty of the landscape.

[Figure 91]

LLaVA-v1.5-7B + SENTINEL (Ours)

The image features a person riding a bicycle through a lush green field, surrounded by tall grass and bushes. The cyclist is wearing a helmet for safety while enjoying the outdoor activity. The field appears to be a large, open space, providing an ideal environment for cycling. The cyclist is the main focus of the scene, with the bicycle and rider occupying a significant portion of the image.

[Figure 92]

Describe this image.

[Figure 93]

GPT-4V

The image depicts an individual riding a mountain bike on a narrow trail through what appears to be a natural grassland or prairie. The biker is dressed in a casual outdoor outfit suitable for biking, which includes shorts and a short-sleeved top. The terrain is mostly flat with sparse shrubbery and a wide-open sky above. It looks like a sunny day with few clouds, indicating good weather conditions for outdoor activities. The horizon is distant, hinting at a vast open space, and the path ahead seems to provide an enjoyable ride through the natural landscape.

[Figure 94]

[Figure 95]

LLaVA-v1.5-7B

The image features a white car parked in a parking lot, occupying a single parking space. The car is positioned in the middle of the parking lot, with its front end visible. The parking lot is filled with multiple parking spaces, with the white car being the only one currently parked. The scene gives a sense of an empty parking lot with the white car being the focal point.

[Figure 96]

LLaVA-v1.5-7B + SENTINEL (Ours)

The image features a white car parked in a parking lot, occupying a single parking space. The car is positioned in the middle of the parking lot, with its front end visible. The parking lot appears to be empty, with no other cars visible in the scene. The white car stands out against the gray pavement, making it the focal point of the image.

[Figure 97]

GPT-4V

[Figure 98]

The image shows an aerial view of a single car parked in a large parking lot with many empty spaces. The lot is marked with white lines delineating individual parking spaces, and the car is parked within one such designated area. The car appears to be white or light-colored and is centered within one of the parking spaces, aligned properly within the markings. The orientation of the photograph indicates that the vehicle is parked head-in, facing the bottom of the image, with the front of the car visible and darker shaded areas indicating the windshield and front windows. The image appears to be taken from a height that shows the car in significant detail while also displaying a number of the surrounding empty spaces, emphasizing the car's isolation in a large, mostly vacant lot.

Describe this image.

###### Figure 13. Comparing general image description results between SENTINAL and its base model LLaVA-v1.5-7B. Our method

[Figure 99]

LLaVA-v1.5-7B

[Figure 100]

The image features a large, old clock tower with a clock prominently displayed on the side. The tower is made of stone and has a brick-like texture. The clock is situated at the top of the tower, and there is a bird perched on the tower as well. In the background, there are a few trees visible, adding to the overall atmosphere of the scene. The clock tower stands tall, towering over the surrounding landscape, making it a prominent landmark in the area.

[Figure 101]

LLaVA-v1.5-7B + SENTINEL (Ours)

The image features a large clock tower with a prominent clock face on its side. The clock is positioned towards the top of the tower, making it a focal point. The tower is made of stone and has a tall, imposing structure. The clock face is white, contrasting with the darker stone of the tower. The clock is surrounded by a variety of flags, adding a sense of color and vibrancy to the scene. The flags are positioned at different heights and angles, creating a dynamic and visually interesting display.

[Figure 102]

GPT-4V

[Figure 103]

Please provide a detailed description of the image, including its visual elements, such as colors, shapes, textures, objects, and people.

This is an image of a tall clock tower structure with a classic architectural design. The tower features a prominent clock face set within an ornate frame, beneath which are three arched openings resembling windows. The top of the tower has a smaller arched structure with a pointed roof that appears to be topped with a weathervane or finial. There are also some colorful flags or bunting visible in the background, indicating that there may be a festive occasion or public celebration occurring. The presence of rooftops in the foreground suggests that the tower is set against an urban backdrop. The weather seems to be overcast, with a gray sky overhead.

[Figure 104]

LLaVA-v1.5-7B

[Figure 105]

The image captures a man walking through a large, empty train station with his luggage. He is pulling a suitcase behind him, possibly arriving at or departing from his destination... In the background, there is a traffic light, indicating that the station might be located near a busy road or intersection. Additionally, there is a handbag visible in the scene, suggesting that the man might be accompanied by someone or has additional belongings with him. The overall atmosphere of the image conveys the sense of travel and movement associated with train stations.

[Figure 106]

LLaVA-v1.5-7B + SENTINEL (Ours)

The image depicts a man walking through a large, empty train station. He is wearing a black coat and carrying a suitcase, likely traveling or commuting. The man is walking down the middle of the station, making his way towards his destination. The suitcase he is carrying suggests that he might be traveling for work or personal reasons, and the emptiness of the station implies that it is either early in the morning or late at night when fewer people are present. The man's presence and the suitcase he carries create a sense of movement and purpose in the otherwise quiet and empty station.

[Figure 107]

GPT-4V

[Figure 108]

Explain the narrative or story that the image seems to convey, detailing each part that contributes to it.

The image shows an indoor setting that resembles a modern transportation hub, such as a train station or airport. A person appears to be walking from right to left, pulling a rolling suitcase behind them, indicating they might be a traveler... There is also a staircase in the background leading to an upper level, with its base obscured by a column. The time seems to be either early morning or late evening, as the location is not crowded. The glass panel on the right suggests that the area is enclosed, and reflects the environment, adding depth to the scene. The lighting is artificial and cool, contributing to the sterile atmosphere of the building.

[Figure 109]

[Figure 110]

LLaVA-v1.5-7B

The image depicts a well-maintained bathroom with a white bathtub, a sink, and a shower. The sink is positioned to the left of the bathtub, while the shower is located to the right of the bathtub. The bathroom features a large mirror above the sink, which is placed in the middle of the scene. There are several bottles placed around the bathroom, with one near the sink, another close to the bathtub, and two others located near the shower. A toothbrush can be seen on the countertop near the sink, and there is also a cup placed on the counter.

[Figure 111]

LLaVA-v1.5-7B + SENTINEL (Ours)

The image features a bathroom with a large mirror above two sinks. The sinks are positioned side by side, with one sink closer to the left side of the mirror and the other sink closer to the right side. The bathroom has a blue and white color scheme, giving it a clean and modern appearance. The mirror reflects the surrounding bathroom space, creating a sense of depth and spaciousness.

[Figure 112]

What is this photo about? Please answer in great detail.

[Figure 113]

GPT-4V

This is an image of a bathroom interior. It features a double sink vanity with a wooden countertop and a blue cabinet base with both wood and blue colored drawers and cupboard doors. Above the sinks are two faucets and a large mirror fitted within a wooden frame that extends beyond the width of the vanity, offering a spacious look. A window is placed above the mirror, showing a glimpse of vegetation outside, suggesting the bathroom might be in a rural area or has a view of some greenery. To the right, there's a shower enclosure with clear glass walls and a showerhead visible. The room is tiled with light-colored tiles on the floor and the shower wall is covered with white tiles. To the right of the shower is a small mounted towel rack.

###### Figure 14. Comparing detailed image description results between SENTINAL and its base model LLaVA-v1.5-7B. Our method

[Figure 114]

[Figure 115]

LLaVA-v1.5-7B

In the image, there are two jars with different colored balls representing particles. The first jar has green and yellow balls, while the second jar has blue and purple balls. The average kinetic energy of the particles in each sample can be calculated by dividing the total kinetic energy by the number of particles. Sample A: Green balls: 10 balls, average kinetic energy = 10*100 m/s = 1000 m/s... Both samples have the same number of particles and the same average kinetic energy. Therefore, the temperature of both samples is the same.

[Figure 116]

LLaVA-v1.5-7B + SENTINEL (Ours)

[Figure 117]

The average kinetic energy of the particles in sample A is higher than that of sample B. This is because the particles in sample A have a higher average kinetic energy due to their higher temperature. The higher temperature in sample A results in a higher average kinetic energy for the particles, making them move faster and more energetically than the particles in sample B.

Compare the average kinetic energies of the particles in each sample. The diagrams below show two pure samples of gas in identical closed, rigid containers. Each colored ball represents one gas particle. Both samples have the same number of particles. Which sample has the higher temperature? Sample A or sample B?

[Figure 118]

[Figure 119]

LLaVA-v1.5-7B

The image features a man dressed as the iconic character, the Joker, from the movie "The Dark Knight." He is holding a playing card in his hand, possibly a reference to the famous scene in the movie where the Joker deals a card to Batman. The man's makeup and costume are designed to resemble the Joker's appearance, making him look like a convincing representation of the character.

[Figure 120]

LLaVA-v1.5-7B + SENTINEL (Ours)

The movie "The Dark Knight" is a 2008 superhero film directed by Christopher Nolan. It is the second installment in the Batman film series and stars Heath Ledger as the Joker, a psychopathic criminal mastermind who wreaks havoc on Gotham City. The movie follows the story of Batman, played by Christian Bale, as he tries to stop the Joker's reign of terror and protect the city from his evil plans. The film is known for its intense action sequences, complex plot, and the iconic performance of Heath Ledger as the Joker.

[Figure 121]

Can you give a short introduction to this movie?

Required capabilities: Recognition, Knowledge and Language generation

[Figure 122]

[Figure 123]

LLaVA-v1.5-7B

The image features a man with curly hair, a mustache, and a beard, who is wearing a suit and tie. He is smiling and appears to be a well-dressed individual. The man is also wearing glasses, which adds to his distinctive appearance. It is not possible to determine the exact identity of the person from the image alone, but he could be a businessman, a public figure, or someone attending a formal event.

[Figure 124]

LLaVA-v1.5-7B + SENTINEL (Ours)

The image features a man with curly hair and a beard, who is smiling and looking at the camera. He is wearing a suit and tie, giving him a formal appearance. This man is widely recognized as Albert Einstein, a renowned physicist and the creator of the

[Figure 125]

Can you give a short introduction to this person?

theory of relativity. His contributions to science have had a significant impact on the understanding of the universe and have shaped modern physics.

Required capabilities: Recognition, Knowledge and Language generation

- Figure 15. Comparing visual question answering results between SENTINAL and LLaVA-v1.5-7B. Our method effectively mitigates hallucinations while enhancing the general performance of the base model, leading to more accurate and detailed answers.

[Figure 126]

[Figure 127]

LLaVA-v1.5-13B

The image features a red stop sign located on the side of a road, with a lush green field in the background. The stop sign is positioned near the edge of the road, possibly to ensure the safety of drivers and pedestrians. In the background, there are several cars parked or driving along the road, with one car closer to the left side of the image and two others further away on the right side. The scene also includes a few people scattered around the area, possibly enjoying the outdoors or attending to their vehicles.

[Figure 128]

LLaVA-v1.5-13B + SENTINEL (Ours)

The image features a red stop sign situated on the side of a road. The stop sign is positioned on the grass, close to the edge of the road. The road appears to be a paved road, and the stop sign is clearly visible for drivers to see and obey. The scene is set in a mountainous area, with the mountains in the background, creating a picturesque and serene atmosphere.

[Figure 129]

GPT-4V

[Figure 130]

Describe this image.

The image shows a red octagonal stop sign in the foreground, slightly tilted, with a picturesque backdrop of a mountain range. The mountain range has lush green foliage covering its base and sharp, rugged peaks that seem to rise dramatically from the green landscape. There are a few clouds scattered in the sky. Below the mountains, an expanse of green grass dotted with trees creates a serene park or rural setting. The road is visible to the side of the stop sign, suggesting that this location is accessible by vehicles. The lighting suggests it's daytime with clear weather, giving the entire scene a vibrant and natural look.

[Figure 131]

LLaVA-v1.5-13B

The image features a row of thatched huts or straw huts, which are situated in a lush green field. The huts are made of natural materials and are arranged in a line, creating a picturesque scene. The field is filled with tall grass, adding to the serene atmosphere of the location. In addition to the huts, there are several birds scattered throughout the scene, some of which are perched on the huts or flying in the sky. The presence of the birds and the natural surroundings make the scene feel like a peaceful and idyllic setting.

[Figure 132]

[Figure 133]

LLaVA-v1.5-13B + SENTINEL (Ours)

The image features a row of thatched huts or straw huts, lined up along a path. The huts are made of natural materials and have a rustic appearance. They are situated in a lush green field, surrounded by a beautiful landscape. The huts are positioned close to each other, creating a sense of unity and harmony in the scene. The overall atmosphere of the image is serene and picturesque, showcasing the beauty of nature and traditional architecture.

[Figure 134]

Describe this image.

[Figure 135]

GPT-4V

The image shows a tranquil rural scene with traditional thatched-roof buildings that appear to be made of wood or bamboo. These structures are surrounded by lush green paddy fields indicating a rice-growing area. Banana plants are visible to the left, providing a tropical feel to the environment. A burst of yellow flowers can be seen growing along one building, adding color to the scene.

...

[Figure 136]

LLaVA-v1.5-13B

[Figure 137]

The image features a beach scene with a large, colorful umbrella providing shade on the sandy shore. The umbrella is positioned near the center of the scene, covering a significant portion of the beach. There are two chairs placed under the umbrella, one on the left side and the other on the right side. The chairs are arranged in a way that allows for comfortable relaxation while enjoying the beach view. The overall atmosphere of the scene is inviting and perfect for a day at the beach.

[Figure 138]

LLaVA-v1.5-13B + SENTINEL (Ours)

The image features a large, colorful umbrella sitting on a sandy beach. The umbrella is open and positioned in the center of the scene, providing shade and a vibrant touch to the beach setting. The umbrella's design is predominantly yellow and blue, making it stand out against the sandy background. The umbrella is placed on the sand, likely for someone to enjoy the beach while staying protected from the sun.

[Figure 139]

Describe this image.

[Figure 140]

GPT-4V

The image shows a sandy beach scene with a colorful beach umbrella and a reclining beach chair partially covered with a yellow towel. The sky is partly cloudy, suggesting a fairly comfortable, mild weather condition. The ocean is visible in the background, indicating that the beach is seaside. There's the presence of some seaweed or debris on the sand near the waterline. The beach does not appear to be crowded, as there are no visible people or significant clusters of activity within the scope of the image. It conveys a sense of relaxation and quiet leisure.

###### Figure 16. Comparing general image descriptions between SENTINAL and its base model LLaVA-v1.5-13B. Our method effectively

[Figure 141]

[Figure 142]

LLaVA-v1.5-13B

The image features a laptop computer sitting on a desk, with a mug of tea placed nearby. The mug has a tea bag inside, and it appears to be filled with a beverage. The laptop screen is lit up, indicating that it is in use. In addition to the laptop and mug, there are two keyboards on the desk, one located in front of the laptop and another slightly to the side. There is also a cell phone placed on the desk, suggesting that the user might be multitasking or working with multiple devices.

[Figure 143]

LLaVA-v1.5-13B + SENTINEL (Ours)

The image features a laptop computer sitting on a bed, with a cup of coffee placed right in front of it. The cup is filled with a beverage, likely coffee, and is positioned close to the

laptop. The laptop screen is illuminated, indicating that it is turned on and in use. The scene creates a cozy atmosphere, as the person using the laptop enjoys their coffee while working or browsing the internet.

[Figure 144]

Compose a detailed account of the image, encompassing its visual characteristics, like colors, shapes, textures, objects, and any human subjects, by paying careful attention to the specifics.

[Figure 145]

GPT-4V

The image features a laptop computer on a bed. The laptop screen is facing us and displays an open application with a white background, which has information listed in tables or lists, possibly resembling a file browser or email client. The keyboard is illuminated from backlighting. To the right of the laptop is a white ceramic mug with a design on it, possibly including a map or graphic print. A square tea bag label is visible hanging over the rim, suggesting that it contains a hot beverage, likely tea.

...

[Figure 146]

[Figure 147]

LLaVA-v1.5-13B

The image features a man and a dog playing a game of frisbee in a park. The man is holding the dog in his arms, and the dog is attempting to catch a red frisbee that is flying in the air between them. In the background, there are several other people scattered throughout the scene, possibly enjoying their time at the park as well. The atmosphere seems lively and fun, with the man and his dog engaging in an entertaining activity.

[Figure 148]

LLaVA-v1.5-13B + SENTINEL (Ours)

The image features a man and a dog playing together in a grassy field. The man is kneeling down, and the dog is jumping in the air to catch a frisbee. The frisbee is positioned above the dog, as it attempts to catch it mid-air. The man appears to be enjoying the game with his dog, creating a fun and engaging scene.

[Figure 149]

Provide an intricate description of the image, capturing its visual elements, including colors, shapes, textures, objects, and any people present.

[Figure 150]

GPT-4V

The image shows a man outdoors in what appears to be a grassy area with trees suggesting a park or a well-maintained garden. He’s facing away from the camera and is in an active, crouching position typically associated with throwing or catching. He’s holding a bright orange frisbee in his right hand, preparing to throw or having just caught it. In mid-air to the right of the person, we see another orange frisbee, indicative of this activity regarding multiple disks being handled or an ongoing game.

...

[Figure 151]

LLaVA-v1.5-13B

[Figure 152]

The scene features two young women riding skateboards down a street. They are enjoying themselves as they skate together, possibly learning from each other. The street is filled with various vehicles, including a car parked on the left side of the road, a truck on the right side, and another car further back. Additionally, there are two other cars in the scene, one parked behind the truck and another one parked near the back of the street. The two skateboarders are the main focus of the image, with their skateboards visible beneath them as they glide down the street.

[Figure 153]

LLaVA-v1.5-13B + SENTINEL (Ours)

The image features two young girls riding skateboards down a street. They are both wearing jeans and appear to be enjoying their time together. The girls are positioned close to each other, with one girl slightly ahead of the other. The street is lined with parked cars on both sides, creating a sense of depth in the scene. The cars are of various sizes and are parked at different distances from the girls, with some closer to the foreground and others further in the background. The girls' skateboards are clearly visible, adding to the dynamic nature of the scene.

[Figure 154]

Provide a thorough description of the given image.

[Figure 155]

GPT-4V

This image features two individuals on skateboards on a street during what appears to be the late afternoon, judging by the sunset lighting. The person in the foreground has one foot on a skateboard and is reaching out to hold or stabilize the shoulder of the other person, who is fully standing on another skateboard and appears to be learning or being guided. The context suggests a skateboarding lesson or a mutual skating session. Behind them, there are parked cars and residential homes lining the street, indicating a suburban neighborhood setting. The plate of the car in the frame must be avoided in trajectory or it should be switched out for a version with a blurred license plate.

###### Figure 17. Comparing detailed image descriptions between SENTINAL and its base model LLaVA-v1.5-13B. Our method effectively

[Figure 156]

[Figure 157]

LLaVA-v1.5-13B

The farthest north highlighted city is Minneapolis, Minnesota.

[Figure 158]

LLaVA-v1.5-13B + SENTINEL (Ours)

[Figure 159]

Which of these highlighted cities is the farthest north?

The farthest north highlighted city is Chicago.

###### Required capabilities:

OCR and Spatial awareness

[Figure 160]

LLaVA-v1.5-13B

[Figure 161]

The trash is full, and the woman is taking it out of the trash can.

[Figure 162]

LLaVA-v1.5-13B + SENTINEL (Ours)

The trash is empty.

[Figure 163]

Is the trash empty or full?

###### Required capabilities:

Recognition

[Figure 164]

[Figure 165]

LLaVA-v1.5-13B

There are four people eating in this kitchen.

[Figure 166]

LLaVA-v1.5-13B + SENTINEL (Ours)

There are no people eating in this kitchen, as it is currently empty.

[Figure 167]

How many people are eating in this kitchen?

Question type: adversarial Question topic: kitchen

- Figure 18. Comparing visual question answering between SENTINAL and its base model LLaVA-v1.5-13B. Our method effectively mitigates hallucinations while enhancing the general performance of the base model, leading to more accurate answers.

