ImagerySearch: Adaptive Test-Time Search for Video Generation Beyond Semantic Dependency Constraints

Meiqi Wu1,3∗ Jiashu Zhu2 Xiaokun Feng1,3 Chubin Chen4 Chen Zhu5 Bingze Song2 Fangyuan Mao2 Jiahong Wu2† Xiangxiang Chu2 Kaiqi Huang1,3‡ 1UCAS 2AMAP, Alibaba Group 3CRISE 4THU 5SEU

GitHub: https://github.com/AMAP-ML/ImagerySearch/

# arXiv:2510.14847v2[cs.CV]22Oct2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

(a) Normal Distance Semantic Prompt: " The camel walks on the desert."

(Walk)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

VideoT1EvosearchWan2.1Wan2.1

Realistic Scenarios

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Short-distance Semantic

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

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

Long-distance Semantic

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Imaginative Scenarios

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

(Pack)

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

###### ImagerySearch(Ours)

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

(b) Long-distance Semantic Prompt: " The camel packs its belongings with care."

Figure 1: The motivation of ImagerySearch. The figure illustrates two semantic dependency scenarios related to camels. Left: The distance depicts the corresponding strength of prompt tokens during the denoising process. LDT-Bench consists of imaginative scenarios with long-distance semantics, whose semantic dependencies are typically weak. Right: Wan2.1 performs well on short-distance semantics but fails under long-distance. Test time scaling methods (e.g., Video T1 (Liu et al., 2025a), Evosearch (He et al., 2025a)) also struggle. However, ImagerySearch generates coherent, context-aware motions (orange box).

Abstract

Video generation models have achieved remarkable progress, particularly excelling in realistic scenarios; however, their performance degrades notably in imaginative scenarios. These prompts often involve rarely co-occurring concepts with longdistance semantic relationships, falling outside training distributions. Existing methods typically apply test-time scaling for improving video quality, but their fixed search spaces and static reward designs limit adaptability to imaginative scenarios.

∗Work done during the internship at AMAP, Alibaba Group. †Project leader. ‡Corresponding author.

To fill this gap, we propose ImagerySearch, a prompt-guided adaptive test-time search strategy that dynamically adjusts both the inference search space and reward function according to semantic relationships in the prompt. This enables more coherent and visually plausible videos in challenging imaginative settings. To evaluate progress in this direction, we introduce LDT-Bench, the first dedicated benchmark for long-distance semantic prompts, consisting of 2,839 diverse concept pairs and an automated protocol for assessing creative generation capabilities. Extensive experiments show that ImagerySearch consistently outperforms strong video generation baselines and existing test-time scaling approaches on LDT-Bench, and achieves competitive improvements on VBench, demonstrating its effectiveness across diverse prompt types. We will release LDT-Bench and code to facilitate future research on imaginative video generation.

- 1 Introduction

Imagine describing a surreal scene–“a panda playing violin on Mars during a sandstorm”–and instantly seeing it come to life as a video. Text-to-video generation promises just that: the ability to turn language into vivid, dynamic worlds. Recent video generation models have made significant progress in generating realistic scenes (Wang et al., 2023; Yang et al., 2024; Peng et al., 2025; OpenAI, 2025; Wan Team et al., 2025); however, their performance drops sharply when handling subjectively imaginative scenarios, hindering the advancement of truly creative video generation. Why is imagination so hard to generate?

This limitation arises from two primary factors. (1) The model’s semantic dependency: Generative models exhibit strong semantic dependency constraints on long-distance semantic prompts, making it difficult to generalize to imaginative scenarios beyond the training distribution (Fig. 1). (2) The scarcity of imaginative training data: Mainstream video datasets (Huang et al., 2024b; Liu et al.,

- 2024b; Sun et al., 2024; Liu et al., 2023; Liao et al., 2025; Ling et al., 2025) predominantly contain realistic scenarios, offering limited imaginative combinations characterized by long-distance semantic relationships (Fig. 3(d)). Recent test-time scaling approaches (Liu et al., 2025a; He et al., 2025a) alleviate data scarcity by sampling multiple candidates and selecting the most promising one. However, their predefined sampling spaces and static reward functions constrain adaptability to the open-ended nature of creative generation.

The Imagery Construction theory (Thomas, 1999; Pylyshyn, 2002) posits that humans create mental scenes for imaginative scenarios by iteratively refining visual imagery in response to language. Motivated by this principle, we introduce ImagerySearch, a test-time search strategy that enhances prompt-based visual generation. ImagerySearch comprises two core components: (i) Semanticdistance-aware Dynamic Search Space (SaDSS), which adaptively modulates sampling granularity according to the semantic span of the prompt; and (ii) Adaptive Imagery Reward (AIR), which incentivizes outputs that align more closely with the intended semantics.

To assess generative models in imaginative settings, we propose LDT-Bench, the first benchmark designed specifically for long-distance semantic prompts. It comprises 2,839 challenging concept pairs, constructed by maximizing semantic distance across object–action and action–action dimensions from diverse recognition datasets (e.g., ImageNet-1K (Deng et al., 2009), Kinects-600 (Carreira et al., 2018)). In addition, LDT-Bench includes an automatic evaluation protocol, ImageryQA, which quantifies creative generation with respect to element coverage, semantic alignment, and anomaly detection.

Extensive experiments reveal that general models (e.g., Wan14B (Wan Team et al., 2025), Hunyuan13B (Kong et al., 2024), CogVideoX (Yang et al., 2024)) and TTS-based models (e.g., VideoT1 (Liu et al., 2025a), EvoSearch (He et al., 2025a)) suffer from significant degradation in video quality and semantic alignment when conditioned on long-distance semantics. In contrast, our framework consistently improves generation fidelity and alignment, demonstrating superior capability in handling long-distance semantic prompts.

Our contributions can be summarized as follows:

- • We propose ImagerySearch, a dynamic test-time scaling law strategy inspired by mental imagery that adaptively adjusts the inference search space and reward according to prompt semantics.
- • We present LDT-Bench, the first benchmark specifically designed for video generation from long-distance semantic prompts. It comprises 2,839 prompts–spanning 1,938 subjects and 901 actions–and offers an automatic evaluation framework for assessing model creativity in imaginative scenarios.
- • Extensive experiments on LDT-Bench and VBench reveal that our approach consistently improves imaging quality and semantic alignment under long-distance semantic prompts.

- 2 Related Work

Text-to-Video Generation Models. With advances in generative modeling (Ho et al., 2020; Chu et al., 2024; Lei et al., 2025; Chu et al., 2025; Chen et al., 2025a) and increased training resources, large-scale T2V models (OpenAI, 2025; Kwai, 2025; Runway, 2025; Bao et al., 2024; Zheng et al.,

- 2024a; Peng et al., 2025; Genmo Team, 2024; Kong et al., 2024; Wan Team et al., 2025) have emerged, capable of generating coherent videos, understanding physics, and generalizing to complex scenarios. But they require massive data, and collecting enough long-range semantic prompts is impractical. Although fine-tuning (Fan and Lee, 2023; Lee et al., 2023; Black et al., 2023; Wallace et al., 2024; Clark et al., 2023; Domingo-Enrich et al., 2024; Mao et al., 2025) and post-training (Yuan et al., 2024a; Prabhudesai et al., 2024; Luo et al., 2023; Li et al., 2024a;b) methods mitigate data requirements to some extent, the extreme scarcity of long-distance semantic videos still hinders effective training. In contrast, the Test-Time Scaling (TTS) methods (Oshima et al., 2025; Xie et al., 2025; Yang et al.,
- 2025; Liu et al., 2025a; He et al., 2025a) used in ImagerySearch require no additional training and achieve strong performance through a highly general approach.

Test-Time Scaling in T2V Models. TTS improves performance by using rewards to select better outputs (Jaech et al., 2024; Guo et al., 2025). In T2V generation, TTS are primarily explored in two aspects: selection strategies and reward strategies. Selection strategies mainly include Best-of-N, particle sampling, and beam search. The Best-of-N (Ma et al., 2025; Liu et al., 2025a) selects the top N outputs from multiple generations. Particle sampling (Singhal et al., 2025; Li et al., 2024c;

- 2025; Singh et al., 2025; Sunwoo Kim, 2025) improves upon this by performing importance-based sampling across the denoising process. Beam search (Liu et al., 2025a; Yang et al., 2025; Xie et al., 2025; Oshima et al., 2025; Liu et al., 2025a; He et al., 2025b) keeps multiple candidates at each step, expanding the sequence set over time. Reward strategies are based on various evaluation metrics, such as VisionReward (Xu et al., 2024), ImageReward (Xu et al., 2023), Aesthetic score (Schuhmann et al., 2022), which guide the selection process by quantifying the quality of generated output. These reward functions are crucial for aligning outputs with desired visual and semantic characteristics.

Current TTS methods optimize search and reward strategies for general T2V generation to enhance overall performance. In this work, we investigate this specific challenge and explore how TTS can be leveraged to improve model performance in long-distance semantic prompts.

Evaluation of Video Generative Models. Early video-generation metrics are simplistic: some diverged from human judgment (Unterthiner et al., 2018; Radford et al., 2021b), while others reused real-video tests unsuited to synthetic clips (Soomro et al., 2012; Xu et al., 2016). Later, studies (Szeto and Corso, 2022; Liu et al., 2023; 2024b; Huang et al., 2024c; Sun et al., 2025; Zheng et al., 2025; Chen et al., 2025b) such as VBench (Huang et al., 2024b) evaluated AI-generated videos from a comprehensive, multi-dimensional perspective. Several studies (Liu et al., 2024a; Yuan et al., 2024b; 2025; Ling et al., 2025) refine evaluation along single dimensions such as frame realism or temporal coherence.

Although existing methods focus on video quality and human perception, semantic content assessment remains underexplored. Current benchmarks struggle to effectively evaluate long-distance semantic prompts, which are key to advancing video generation capabilities. To address this, LDT-Bench was introduced as the first benchmark for evaluating long-distance semantic understanding in video generation.

𝑡∈Imageryschedulers

###### Semantic-aware DynamicSearchSpace

###### AdaptiveImagery Reward

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

[Figure 89]

[Figure 90]

T2V

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

𝛼 MQ + 𝛽 TA + γ VQ + ……

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

𝛼 𝛽

𝒟

[Figure 109]

[Figure 110]

[Figure 111]

γ

[Figure 112]

……

[Figure 113]

Select the number of videos

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Prompt: "The native bear skillfully uses remote controls."

Keyword Extractor

camel

Text Encoder

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

packs

[Figure 124]

𝒟

#### ConstraintSemanticScorer ImagerySearch

belongings

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Select Parameter

Iterative

Reward Container Bad Video Clip Good Video Clip Select Space KeyWord Features Feature Cosine Similarity

[Figure 130]

- Figure 2: Overview of our ImagerySearch. The prompt is scored by the Constrained Semantic Scorer

(producing D¯sem) and simultaneously fed to the T2V backbone (Wan2.1). At every step t specified by the imagery scheduler, we sample a set of candidate clips, rank them with a reward function

conditioned on D¯sem, and retain only a D¯sem-controlled subset. The loop repeats until generation completes.

- 3 ImagerySearch

Text-to-video generation aims to synthesize coherent videos conditioned on prompts. Diffusion models inherently possess the flexibility to adjust test-time computation via the number of denoising steps. To further improve generation quality, we formulate a search problem that identifies better noise inputs for the diffusion sampling process. We organize the design space along two axes: the reward functions that evaluate video quality, and the search algorithms that explore and select optimal noise candidates.

- 3.1 Preliminaries

In standard diffusion frameworks, sampling starts from Gaussian noise xT ∼ N(0,I), and the model iteratively denoises the latent through a learned network fθ. As a widely used sampling paradigm, DDIM performs the following step-wise denoising update:

xt−1 = ζt−1(

xt − σtfθ(xt,t,c) ζt

) + σt−1fθ(xt,t,c), (1) Where ζt−1, ζt, σt−1 denote predefined schedules.

Prior test-time scaling approaches (Liu et al., 2025a; He et al., 2025a; Yang et al., 2025) operate within a fixed noise search space and use static reward functions–such as VideoScore (He et al., 2024), VideoAlign (Liu et al., 2025b), or their combinations–to rank candidates. By contrast, our framework supports flexible reward design and adaptive noise selection, substantially improving both sample efficiency and generation quality.

- 3.2 Dynamic Search Space

Inspired by imagery cognitive theory (Thomas, 1999; Pylyshyn, 2002; Feng et al., 2023)—which posits that humans expend more effort and time to construct mental imagery for semantically distant concepts—we likewise adapt the candidate-video search space to a prompt’s semantic distance: shrinking it for short-distance prompts to boost test-time efficiency, and enlarging it for long-distance prompts to explore a broader range of possibilities. Therefore, we propose a Semantic-distance-aware Dynamic Search Space (SaDSS).

- As shown in Fig. 2, this adaptive resizing is driven by a Constrained Semantic Scorer, which dynamically modulates the search space. Specifically, we define semantic distance as the average

embedding distance between key entities (objects and actions) extracted from the prompt. Given a prompt p, we extract its compositional units {pi}ni=1 and compute:

1 |E|

∥ϕ(pi) − ϕ(pj)∥2 , (2)

D¯sem(p) =

(i,j)∈E

where ϕ(·) denotes the embedding function (e.g., T5 encoder), and E is the set of key entity pairs in the prompt.

- At inference time, we adapt the sampling procedure based on D¯sem. Specifically, the search space dynamically adapts based on semantic distance. Formally, the number of candidates Nt at timestep t is dynamically adjusted as:

## Nt = Nbase · 1 + λ · D¯sem(p) , (3)

where Nbase is the base number of samples, and λ is a scaling factor that controls the sensitivity to semantic distance. In this work, we set λ = 1.

By tailoring the search scope to the inherent difficulty of the prompt, SaDSS encourages the model to explore more diverse visual hypotheses when needed, improving visual plausibility under challenging conditions, without incurring unnecessary computational costs for simple prompts.

- 3.3 Adaptive Imagery Reward

Based on our observations, adjacent denoising steps alter the latent video only marginally, so we invoke ImagerySearch at a few key noise levels S = {5, 10, 20, 45}, termed the Imagery Schedule (see Appendix A). As shown in Fig. 2, starting from a partially denoised latent xt, we produce xˆ0 by completing the denoising trajectory and compute the reward on xˆ0 to assess the influence of different denoising stages on the final video quality.

To enhance semantic alignment between generated videos and prompts with long-distance semantics, we introduce an Adaptive Imagery Reward (AIR) that modulates evaluation feedback based on the prompt’s semantic difficulty. Specifically, we incorporate the semantic distance as a soft re-weighting factor into the reward formulation. The reward RAIR(xˆ0) for each candidate video x0 is defined as:

RAIR(xˆ0) = (α · MQ + β · TA + γ · VQ + ω · Rany) · D¯sem(xˆ0), (4) where α, β, γ, and ω are scaling factors that adaptively adjust the reward based on the prompt semantic distance D¯sem. MQ, TA, and VQ are from VideoAlign (Liu et al., 2025b), and Rany denotes an extensible reward (e.g., VideoScore (He et al., 2024), VMBench (Ling et al., 2025), UnifiedReward (Wang et al., 2025), VisionReward (Xu et al., 2024)).

- 4 LDT-Bench

The rapid progress of video generation models is closely tied to the development of targeted evaluation benchmarks. Existing benchmarks primarily assess models using text prompts designed to depict realistic scenarios. However, as video generation models have achieved impressive performance in realistic scenarios, it is timely to shift the focus towards imaginative scenarios. Generally, such complex settings involve prompts in which entities–such as objects and actions–exhibit long semantic distances, meaning these entities rarely co-occur (e.g., “a panda piloting a helicopter”). These corner cases reveal the robustness limits of generative models. Nonetheless, most existing works remain limited to qualitative analysis on a few cases, and there is a lack of a unified benchmark specifically designed for this task.

To fill this gap, we propose a novel benchmark LDT-Bench, designed to systematically analyze the generalization ability of video generation models in complex scenarios induced by prompts with Long-Distance semantic Texts. In the following sections, LDT-Bench is introduced from two perspectives: the construction of the prompt suite and the design of evaluation metrics. The core components of LDT-Bench are illustrated in Fig. 3.

- 4.1 Prompt Suite

Meta-information Extraction. Considering that objects and actions are the main entities in text prompts, we construct our prompts using the following two structural types. (1) Object–Action: An

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

##### (a) Meta-Information Extraction (b) Long-Distance Prompt Generation

##### (c) Evaluation Metrics

Evaluation prompts

Generated videos

[Figure 143]

[Figure 144]

[Figure 145]

###### Object Set

###### Action Set

Prompt Generation LLM-Human Double Check

[Figure 146]

- • The camel packs its belongings with care.
- • A girl applies eye makeup in her bedroom, then ties a colorful ribbon at her crafting desk.

###### Initial Prompt

|[Figure 147]<br><br>Evaluated questions<br><br>• ElementQA<br>• AlignQA<br>• AnomalyQA<br>|
|---|

Final prompts

Kinetics-600

[Figure 148]

|[Figure 149]<br><br>[Figure 150]<br><br>MLLMs|
|---|

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Validation by Human

Validation by LLM

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

✅ ❌

Semanticcalculationdistance& filtering T5

[Figure 159]

✅

[Figure 160]

……

[Figure 161]

[Figure 162]

❌

|[Figure 163]<br><br>Answers & Metric results|
|---|

[Figure 164]

Object-Action Pairs & Action-Action Pairs

Regeneration

[Figure 165]

[Figure 166]

[Figure 167]

##### (d) (e)

|Benchmark|Prompt|Object|Action|ASD|
|---|---|---|---|---|
|VBench|800|125|132|0.33|
|EvalCrafter|700|169|237|0.4|
|T2V-CompBench|1,400|308|545|0.31|
|FETV|619|168|334|0.38|
|DEVIL|810|270|556|0.39|
|VMBench|1,050|340|1,216|0.35|
|LDT-Bench(Ours)|2,839|1,938|901|0.86|

[Figure 168]

[Figure 169]

- Figure 3: Overview of our LDT-Bench. Upper: (a) LDT-Bench is built by first extracting metainformation from existing recognition datasets; (b) GPT-4o is then used to generate candidate prompts, which are filtered jointly by DeepSeek and humans to obtain the final prompt set; (c) Additionally, we design a set of three MLLM-based QA tasks that serve as the creativity metric. Lower: (d) Compared with other benchmarks, LDT-Bench covers a much richer variety of categories; (e) its prompts also exhibit a semantic-distance distribution that is shifted toward substantially longer ranges. Note that “ASD” denotes the average semantic distance of prompts.

object combined with an uncommon or incompatible action. (2) Action–Action: Two semantically distant or even contradictory actions.

To cover a wide range of objects and actions, we build our object and action sets from representative large-scale datasets. Specifically, the object set is derived from ImageNet-1K (Deng et al., 2009) and COCO (Lin et al., 2014) (covering 1,938 objects), while the action set is collected from ActivityNet (Caba Heilbron et al., 2015), UCF101 (Soomro et al., 2012), and Kinetics-600 (Carreira et al., 2018) (covering 901 actions). These collections serve as the foundation for subsequent prompt generation.

We first encode each object and action element texti using a pretrained T5 text encoder (Raffel et al., 2020), obtaining a high-dimensional textual feature hi ∈ Rd. These embeddings are then projected into a shared 2D semantic space via Principal Component Analysis (PCA):

zi = PCA(hi) = PCA(T5(texti)), zi ∈ R2, (5)

where zi represents the semantic position of the i-th element in the 2D space. T5 can be replaced with other encoders, such as CLIP (Radford et al., 2021b); see Appendix B.1 for details.

To measure semantic divergence, we compute the Euclidean distance between each pair of elements as a criterion for selecting long-distance semantic prompts. We then construct two candidate sets: one by pairing each object with its most distant action (1,938 object–action pairs), and the other by matching each action with its most distant counterpart (901 action–action pairs). From each set, we select the 160 most distant pairs, resulting in 320 high-distance prompts that challenge the model with long-distance semantic combinations. For more analysis of the prompt suite, please refer to Appendix B.2.

Long-distance Prompt Generation. Based on the obtained text element pairs, we employ a large language model, i.e., GPT-4o (Hurst et al., 2024), to generate fluent and complete text prompts by filling in necessary sentence components. Subsequently, each prompt is double-checked by both DeepSeekR1 (Guo et al., 2025) and human annotators to ensure quality, resulting in our final prompt suite. The detailed generation process and several illustrative cases are presented in Fig. 3 (b).

- 4.2 Imagery Evaluation Metrics

To quantitatively evaluate the performance of video generation models under long-distance semantic settings, we develop targeted evaluation metrics. Inspired by recent MLLMs-based evaluation methods (Cho et al., 2023; Feng et al., 2025), we generate questions based on the text prompts. Subsequently, MLLMs with strong semantic comprehension capabilities analyze the generated videos in response

to these questions, yielding quantitative evaluation results. Specifically, our assessment framework encompasses three primary dimensions.

ElementQA. Because our prompts focus on objects and actions, ElementQA primarily consists of targeted questions revolving around these elements. For example, given the prompt “The traffic light is dancing.”, we can generate two questions: “Does the traffic light appear in the video?” and “Is the traffic light performing a dancing action?”

AlignQA. In addition to the basic semantic information covered by ElementQA, we also evaluate the generated videos in terms of visual quality and aesthetics (Murray et al., 2012). Given the challenging and inherently subjective nature of this assessment, we employ recently developed MLLMs that have been specifically optimized for alignment with human perception to perform the evaluation (Huang et al., 2024a; Wu et al., 2023).

AnomalyQA. We have observed that current video generation models frequently produce anomalous outputs. Consequently, we also leverage MLLMs to analyze the generated frames and answer targeted questions aimed at identifying these anomalies.

Implementation Details. For ElementQA, we employ Qwen2.5-VL-72B-Instruct (Bai et al., 2025) as the underlying MLLM, whereas for AlignQA we adopt Q-Align (Wu et al., 2023), a model specifically optimized for rating visual quality and aesthetics. Given the broader generalization required by AnomalyQA, we utilize the more powerful GPT-4o (OpenAI, 2024) for evaluation. We collectively refer to these three components as ImageryQA. Further implementation details are provided in Appendix B.3.

- 5 Experiments

- 5.1 Experimental Setup Datasets & Metrics. To assess the imaginative capacity of video-generation models, we evaluate

- them on both LDT-Bench and VBench (Huang et al., 2024b), using each benchmark’s full prompt suite and associated metrics.

Compared Models. We compare two categories of models: (1) General models: Hunyuan (Kong et al., 2024), Wan2.1 (Wan Team et al., 2025), Open-Sora (Zheng et al., 2024b), CogVideoX (Yang

- et al., 2024); (2) TTS methods: Video-T1 (Liu et al., 2025a) and EvoSearch (He et al., 2025a). We use Wan2.1 as the base model and generate 33-frame clips with the default settings (see Appendix C for details).

Experimental Environment. All experiments are run on a server equipped with 8 × NVIDIA H20 GPUs (96 GB each), an Intel Xeon Gold 6348 CPU (32 cores, 2.6 GHz), and 512 GB of RAM, under Ubuntu 20.04 LTS (kernel 5.15). We used Python 3.9 with PyTorch 2.5.1 (CUDA 12.4, cuDNN 9.1), torchvision 0.20.1, and Transformers 4.50.3.

- 5.2 Comparison with Other Generation Models

Performance on LDT-Bench. As shown in Tab. 1, we adopt Wan2.1 as the base model. Our method achieves a significant improvement of 8.83%, demonstrating a clear advantage. Furthermore, compared to other test-time scaling approaches, ImagerySearch also delivers consistently superior performance. These results highlight the effectiveness of our method in handling long-distance semantic prompts and its robustness in imagination-driven scenarios.

LDT-Bench (%) ↑ Model

ElementQA AlignQA AnomalyQA ImageryQA (All)

Wan2.1 (Wan Team et al., 2025) 1.66 31.62 15.00 48.28 Video-T1 (Liu et al., 2025a) 1.91 38.16 14.68 54.75 Evosearch (He et al., 2025a) 1.92 36.10 16.46 54.48 ImagerySearch (Ours) 2.01 36.82 18.28 57.11

- Table 1: Quantitative comparison on LDT-Bench. ImagerySearch achieves the best average performance.

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

CogvideoX

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Opensora

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

Opensora-plan

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Hunyuan

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Wan2.1

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

Prompt: “The native bear skillfully uses remote controls.”

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

Video-T1

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

Evosearch

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

###### Ima (Ours)

| | |
|---|---|
| | |
| | |

|gerySe|[Figure 282]<br><br>arch|
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| |[Figure 283]|
|---|---|
| | |
| | |

|[Figure 284]| |
|---|---|
| | |
| | |

| |[Figure 285]|
|---|---|
| | |
| | |

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

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

Frame15

Frame0 Frame5 Frame10 Frame25 Frame30 Frame33

Frame20

- Figure 4: Visualization of examples. Upper: Results from general models. Lower: ImagerySearch versus other test-time scaling methods. Ours produces more vivid actions under long-distance semantic prompts.

VBench (%) ↑ Model Aesthetic

Quality

Background Consistency

Dynamic Degree

Imaging Quality

Motion Smoothness

Subject Consistency

Average

Wan2.1 (Wan Team et al., 2025) 50.50 91.80 82.85 58.25 97.50 90.25 78.53 Opensora (Peng et al., 2025) 48.80 95.25 73.15 61.35 99.05 92.95 78.43 CogvideoX (Yang et al., 2024) 48.80 95.30 47.20 65.05 98.55 94.65 74.93

General

Hunyuan (Kong et al., 2024) 50.45 92.65 85.00 59.55 95.75 90.55 78.99 Video-T1 (Liu et al., 2025a) 57.20 95.65 54.05 60.25 99.30 94.80 76.88

TTS Evosearch (He et al., 2025a) 55.55 94.80 80.95 68.90 97.70 94.55 82.08 ImagerySearch (Ours) 57.70 96.00 84.05 69.20 98.00 95.90 83.48

- Table 2: Quantitative comparison of video generation models on VBench. ImagerySearch achieves the best average performance across multiple metrics, indicating better alignment and generation quality.

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

(a) (b) (c) (d) (e) (f)

- Figure 5: (a) Effect of semantic distance across different models. As semantic distance increases, our method remains the most stable. (b-e) Our AIR consistently delivers superior performance. Scaling behavior of ImagerySeach and baselines as inference-time computation increases. From left to right, the y-axes represent the score changes for MQ, TA, VQ, and Overall (VideoAlign (Liu et al., 2025b)). (f) Effect of reward weight.

Performance on VBench. For a balanced evaluation, we compare two classes of methods on VBench. The upper rows of Tab. 2 report general generators, while the lower rows list test-time scaling approaches–Video-T1 (Liu et al., 2025a), EvoSearch (He et al., 2025a), and our proposed ImagerySearch. All models are evaluated on long-distance prompts from LDT-Bench using the VBench metrics. ImagerySearch achieves the best overall score and ranks highest on the fine-grained Dynamic Degree, Subject Consistency metrics and so on, indicating its strong ability to preserve

VBench (%) Model Aesthetic

Background Consistency

Dynamic Degree

Imaging Quality

Motion Smoothness

Subject consistency

Average Baseline Wan2.1 (Wan Team et al., 2025) 50.50 91.80 82.85 58.25 97.50 90.25 78.53

Quality

w/o AIR 56.25 94.60 81.85 68.05 97.50 94.40 82.11 Modules

w/o SaDSS 55.35 95.10 77.20 68.00 97.60 94.55 81.30

SaDSS-static 0.5 57.25 96.15 70.00 70.75 97.45 95.45 81.18 weight 0.9 57.40 96.05 70.00 70.80 97.55 95.50 81.22

BON (Ma et al., 2025) 57.40 95.00 83.01 68.10 97.70 94.63 82.64 Search Particle Sampling (Ma et al., 2025) 56.51 93.52 81.72 67.04 96.18 93.38 81.39

ImagerySearch (Ours) 57.70 96.00 84.05 68.50 97.65 94.70 83.10

- Table 3: Ablation Study. “Baseline” is the plain backbone; “Modules” successively add our two novel modules; “SaDSS-static weight” denotes the performance obtained when the selection space is kept at a fixed size; “Search” swaps in alternative search strategies. The full configuration (ImagerySearch) yields the best performance.

prompt fidelity under wide semantic gaps. Fig. 4 illustrates this strength: ImagerySearch accurately reproduces both the specified subjects (e.g., bear, controls) and their associated actions (e.g., uses). Additional examples in Appendix D further demonstrate its robustness in handling complex long-distance prompts.

Robustness Analysis Across Semantic Distances. As illustrated in Fig. 5(a), our approach maintains nearly constant VBench scores as semantic distance increases, whereas competing methods exhibit pronounced fluctuations. This stability highlights the superior robustness of our model across a wide range of semantic distances. Additional error analysis is provided in the Appendix E.

- 5.3 Test-time Scaling Law Analysis

We measure the inference-time computation by the number of function evaluations (NFEs). As shown in Fig. 5(b–d), where performance is assessed with the MQ, TA, and VQ metrics from VideoAlign (Liu et al., 2025b), ImagerySearch exhibits monotonic performance improvements as inference-time computation increases. Notably, on Wan2.1 (Wan Team et al., 2025), ImagerySearch continues to gain as NFEs grow, whereas baseline methods plateau at roughly 1 × 103 NFEs (corresponding to the 30th timestep). Computation details are provided in the Appendix F. Moreover, our method shows an even more pronounced advantage in the overall VideoAlign score, as illustrated in Fig. 5(e).

- 5.4 Ablation Study

Effect of SaDSS and AIR. As shown in the first three rows of Tab. 3, adding either the SaDSS or the AIR module individually already surpasses the baseline, while combining SaDSS with AIR achieves the best performance, confirming the complementary nature of semantic guidance and adaptive selection.

Effect of Search Space Size. The SaDSS–static weight rows in Tab. 3 compare fixed and dynamic search-space configurations. With static weights of 0.5, and 0.9, performance improves gradually, reaching a VBench score of 81.22%. In contrast, the dynamic approach attains a markedly higher score of 83.48%, demonstrating its superior ability to optimize the search space and thus boost model performance.

Effect of Search Strategy. The Search rows in Tab. 3 compare different search strategies (e.g., BON, Particle Sampling (Ma et al., 2025)). The experimental results demonstrate that our search strategy delivers the best performance.

Effect of Reward Dynamic Adjustment Mechanism. Fig. 5(f) demonstrates the impact of varying reward weights on VBench scores across different models (MQ, TA, VQ). As weights change from 0.2 to 1.2, TA shows notable improvement while MQ and VQ maintain relatively stable performance. The consistent superiority of the Ours approach, represented by the dashed line, underscores the effectiveness of dynamic reward adjustment, achieving optimal performance irrespective of weight changes.

- 6 Conclusion

In this study, we propose ImagerySearch, an adaptive test-time search method that improves video-generation quality for long-distance semantic prompts drawn from imaginative scenarios. Additionally, we present LDT-Bench, the first benchmark designed to evaluate such challenging prompts. ImagerySearch attains state-of-the-art results on both VBench and LDT-Bench, with especially strong gains on LDT-Bench, demonstrating its effectiveness for text-to-video generation under long-range semantic conditions. In future, we will explore more flexible reward mechanisms to further enhance video-generation performance.

References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015.

Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics-600. arXiv preprint arXiv:1808.01340, 2018.

Chubin Chen, Jiashu Zhu, Xiaokun Feng, Nisha Huang, Meiqi Wu, Fangyuan Mao, Jiahong Wu, Xiangxiang Chu, and Xiu Li. S2-guidance: Stochastic self guidance for training-free enhancement of diffusion models. arXiv preprint arXiv:2508.12880, 2025a.

Rui Chen, Lei Sun, Jing Tang, Geng Li, and Xiangxiang Chu. Finger: Content aware fine-grained evaluation with reasoning for ai-generated videos. arXiv preprint arXiv:2504.10358, 2025b.

Jaemin Cho, Yushi Hu, Roopal Garg, Peter Anderson, Ranjay Krishna, Jason Baldridge, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation. arXiv preprint arXiv:2310.18235, 2023.

Xiangxiang Chu, Jianlin Su, Bo Zhang, and Chunhua Shen. Visionllama: A unified llama backbone for vision tasks. In European Conference on Computer Vision, pages 1–18. Springer, 2024.

Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified self-supervised pretraining for image generation and understanding. arXiv preprint arXiv:2503.06132, 2025.

Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. arXiv preprint arXiv:2409.08861, 2024.

Ying Fan and Kangwook Lee. Optimizing ddpm sampling with shortcut fine-tuning. arXiv preprint arXiv:2301.13362, 2023.

Xiaokun Feng, Shiyu Hu, Xiaotang Chen, and Kaiqi Huang. A hierarchical theme recognition model for sandplay therapy. In Chinese Conference on Pattern Recognition and Computer Vision (PRCV), pages 241–252. Springer, 2023.

Xiaokun Feng, Haiming Yu, Meiqi Wu, Shiyu Hu, Jintao Chen, Chen Zhu, Jiahong Wu, Xiangxiang Chu, and Kaiqi Huang. Narrlv: Towards a comprehensive narrative-centric evaluation for long video generation models. arXiv preprint arXiv:2507.11245, 2025.

Genmo Team. Mochi 1. https://github.com/genmoai/models, 2024. Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong

Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Haoran He, Jiajun Liang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Ling Pan. Scaling image and video generation via test-time evolutionary search. arXiv preprint arXiv:2505.17618,

- 2025a.

Haoran He, Jiajun Liang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Ling Pan. Scaling image and video generation via test-time evolutionary search. arXiv preprint arXiv:2505.17618,

- 2025b.

Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, et al. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. arXiv preprint arXiv:2406.15252, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Yipo Huang, Xiangfei Sheng, Zhichao Yang, Quan Yuan, Zhichao Duan, Pengfei Chen, Leida Li, Weisi Lin, and Guangming Shi. Aesexpert: Towards multi-modality foundation model for image aesthetics perception. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 5911–5920, 2024a.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024b.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024c.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Kwai. Kling. Accessed February 25, 2025 [Online] https://klingai.com/, 2025. URL https://klingai.com/.

Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.

Jiachen Lei, Keli Liu, Julius Berner, Haiming Yu, Hongkai Zheng, Jiahong Wu, and Xiangxiang Chu. Advancing end-to-end pixel space generative modeling via self-supervised pre-training. arXiv preprint arXiv:2510.12586, 2025.

Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, and William Yang Wang. T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback. Advances in neural information processing systems, 37:75692–75726, 2024a.

Jiachen Li, Qian Long, Jian Zheng, Xiaofeng Gao, Robinson Piramuthu, Wenhu Chen, and William Yang Wang. T2v-turbo-v2: Enhancing video generation model post-training through data, reward, and conditional guidance design. arXiv preprint arXiv:2410.05677, 2024b.

Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, et al. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. arXiv preprint arXiv:2408.08252, 2024c.

Xiner Li, Masatoshi Uehara, Xingyu Su, Gabriele Scalia, Tommaso Biancalani, Aviv Regev, Sergey Levine, and Shuiwang Ji. Dynamic search for inference-time alignment in diffusion models. arXiv preprint arXiv:2503.02039, 2025.

Mingxiang Liao, Qixiang Ye, Wangmeng Zuo, Fang Wan, Tianyu Wang, Yuzhong Zhao, Jingdong Wang, Xinyu Zhang, et al. Evaluation of text-to-video generation models: A dynamics perspective. Advances in Neural Information Processing Systems, 37:109790–109816, 2025.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

Xinran Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. Vmbench: A benchmark for perception-aligned video motion generation. arXiv preprint arXiv:2503.10076, 2025.

Fangfu Liu, Hanyang Wang, Yimo Cai, Kaiyan Zhang, Xiaohang Zhan, and Yueqi Duan. Video-t1: Test-time scaling for video generation. arXiv preprint arXiv:2503.18942, 2025a.

Jiahe Liu, Youran Qu, Qi Yan, Xiaohui Zeng, Lele Wang, and Renjie Liao. Fr\’echet video motion distance: A metric for evaluating motion consistency in videos. arXiv preprint arXiv:2407.16124, 2024a.

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025b.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22139–22149, 2024b.

Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation. Advances in Neural Information Processing Systems, 36:62352–62387, 2023.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, et al. Inference-time scaling for diffusion models beyond scaling denoising steps. arXiv preprint arXiv:2501.09732, 2025.

Fangyuan Mao, Aiming Hao, Jintao Chen, Dongxia Liu, Xiaokun Feng, Jiashu Zhu, Meiqi Wu, Chubin Chen, Jiahong Wu, and Xiangxiang Chu. Omni-effects: Unified and spatially-controllable visual effects generation. arXiv preprint arXiv:2508.07981, 2025.

Naila Murray, Luca Marchesotti, and Florent Perronnin. Ava: A large-scale database for aesthetic visual analysis. In 2012 IEEE conference on computer vision and pattern recognition, pages 2408–2415. IEEE, 2012.

OpenAI. Gpt-4o: Openai’s new flagship model. https://openai.com/index/ gpt-4o-and-gpt-4-api-updates/, 2024. Accessed: 2024-06-05.

OpenAI. Sora. Accessed February 25, 2025 [Online] https://openai.com/index/sora/,

2025. URL https://openai.com/index/sora/. Yuta Oshima, Masahiro Suzuki, Yutaka Matsuo, and Hiroki Furuta. Inference-time text-to-video alignment with diffusion latent beam search. arXiv preprint arXiv:2501.19252, 2025.

Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, and et al. Open-sora 2.0: Training a commercial-level video generation model in $200k. arXiv preprint arXiv:2503.09642, 2025.

Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024.

Zenon W Pylyshyn. Mental imagery: In search of a theory. Behavioral and brain sciences, 25(2): 157–182, 2002.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages

- 8748–8763. PmLR, 2021a.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages

- 8748–8763. PmLR, 2021b.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Runway. Runway gen3. Accessed February 25, 2025 [Online] https://app.runwayml.com/,

2025. URL https://app.runwayml.com/.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.

Anuj Singh, Sayak Mukherjee, Ahmad Beirami, and Hadi Jamali Rad. Code: Blockwise control for denoising diffusion models. ArXiv, abs/2502.00968, 2025. URL https://api. semanticscholar.org/CorpusID:276094284.

Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of diffusion models. arXiv preprint arXiv:2501.06848, 2025.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

John Stam. Stable diffusion: High-resolution image synthesis with latent diffusion models, 2023. Placeholder entry. Please update with correct details.

Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2vcompbench: A comprehensive benchmark for compositional text-to-video generation. arXiv preprint arXiv:2407.14505, 2024.

Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8406–8416, 2025.

Dongmin Park Sunwoo Kim, Minkyu Kim. Test-time alignment of diffusion models without reward over-optimization. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=vi3DjUhFVm.

Ryan Szeto and Jason J Corso. The devil is in the details: A diagnostic evaluation benchmark for video inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21054–21063, 2022.

Nigel JT Thomas. Are theories of imagery theories of imagination? an active perception approach to conscious mental content. Cognitive science, 23(2):207–245, 1999.

Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

Wan Team, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.

Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. In International Conference on Machine Learning, pages 54015–54029. PMLR, 2024a.

Meiqi Wu, Kaiqi Huang, Yuanqiang Cai, Shiyu Hu, Yuzhong Zhao, and Weiqiang Wang. Finger in camera speaks everything: Unconstrained air-writing for real-world. IEEE Transactions on Circuits and Systems for Video Technology, 34(9):8602–8613, 2024b.

Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 15903–15935, 2023.

Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016.

Haolin Yang, Feilong Tang, Ming Hu, Yulong Li, Yexin Liu, Zelin Peng, Junjun He, Zongyuan Ge, and Imran Razzak. Scalingnoise: Scaling inference-time search for generating infinite videos. arXiv preprint arXiv:2503.16400, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6463–6474, 2024a.

Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Rui-Jie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems, 37:21236–21270, 2024b.

Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation. arXiv preprint arXiv:2505.20292, 2025.

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv

- preprint arXiv:2412.20404, 2024a.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv

- preprint arXiv:2412.20404, 2024b.

- A The Selection of Imagery Schedule

As illustrated in Fig. S1, we observe that adjacent denoising steps modify the latent video only marginally; substantial deviations from earlier stages emerge only at several pivotal steps. To improve generation efficiency, we therefore trigger ImagerySearch at a limited set of noise levels, S = {5, 20, 30, 45}, which we term the Imagery Schedule. This schedule specifies the exact timesteps at which ImagerySearch is invoked.

- B More Details About Imagery Evaluation Metrics

- B.1 More Text Encoders.

In our current implementation, T5 serves three purposes: it encodes the key entities in each prompt, measures their semantic distances, and then uses those distances to adjust the search space and reward weights during generation. The same pipeline can be run with a CLIP text encoder (Radford et al., 2021a; Blattmann et al., 2023). Trained on large-scale image–text pairs, CLIP yields text embeddings whose cosine distances correlate well with visual concepts, so these distances can play exactly the same role in deciding when to expand or shrink the search space. In addition, CLIP similarities are widely used as a measure of text–image or text–video alignment, which makes them a natural choice for the alignment term in our reward function (Stam, 2023). Because CLIP, like T5, produces a fixed-length vector in a single forward pass, it can be swapped in as a drop-in replacement without changing any downstream components while fully preserving the effectiveness of our adaptive search and reward mechanisms.

- B.2 More Analysis about Prompt Suite.

- As shown in Fig. S2, we provide a multi-faceted overview of the LDT-Bench prompt suite and underscore its advantages for long-distance semantic evaluation. (a) Examining the distribution of actions, a pronounced long-tail pattern emerges: of the five super-categories, Sports & Wellness and Daily Services each supply 300 prompts, ensuring ample coverage of everyday yet highly diverse actions. (b) For objects, a treemap of 14 super-categories—scaled by instance count—reveals that Animal and Artifact jointly exceed half of all samples, while still leaving room for rarer classes; this balance of head and tail categories is largely missing in prior benchmarks. (c) The object word cloud (after stop-word filtering) highlights high-frequency nouns such as cricket, person, and remote, evidencing fine-grained lexical diversity across domains. (d) The action word cloud reveals a wide semantic span—verbs like play, join, use, and handle—that challenges models to cope with imaginative, long-distance dependencies.

Taken together, these statistics show that LDT-Bench not only covers a richer mix of objects and actions than existing datasets but also accentuates long-distance semantic relationships that current models find most difficult, making it a uniquely effective testbed for stress-testing creative video generation systems.

B.3 ImageryQA Implementation Details.

As described in Sec. 4.2 of the paper, our metric is primarily composed of three components: ElementQA, AlignQA, and AnomalyQA (Fig. S3 (a)). In this subsection, we provide further clarification using specific examples and illustrating the metric computation process.

- As shown in Fig. S3 (b), given the evaluation prompt, “A person polishes furniture attentively at home,

- then packs cleaning products for organization.”, two videos generated by different video generation models. First, ElementQA formulates questions based on the objects and actions within the prompt, i.e., “person,” “polishes furniture,” and “packs cleaning products for organization”, resulting in the questions Q1, Q2, and Q3 in Fig. S3. Next, AlignQA assesses the first frame of each video in terms of image quality and aesthetics. Finally, AnomalyQA evaluates abnormal events in both videos, as illustrated by Q5 in Fig. S3. Based on these questions, we employ different MLLMs and answer strategies. Recent studies (Feng

- et al., 2025; Liu et al., 2025b; Wu et al., 2024a; Zheng et al., 2025; Wu et al., 2024b) suggest that for

Prompt: “The bear walks on the grassland.” Adjacent denoising steps

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

timestep = 3 timestep = 4 timestep = 6

Frame 1

Key denoising steps

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

timestep = 5 timestep = 20 timestep = 30 timestep = 45

- Figure S1: Imagery schedule. The heatmaps visualize 13th-layer attention projected onto the first video frame at successive denoising steps. Adjacent steps show nearly identical focus regions, whereas only a few key steps exhibit pronounced changes. Concentrating analysis and search on these pivotal steps therefore captures the prompt-to-frame semantic correspondence more efficiently.

[Figure 316]

[Figure 317]

[Figure 318]

(a) (b) (c) (d)

[Figure 319]

- Figure S2: LDT-Bench prompt suite analysis: (a) Action super-category distribution shown as a horizontal bar chart. (b) Object super-category distribution displayed as a treemap, with area proportional to class count. (c) Word cloud highlighting the most frequent object-action prompts. (d) Word cloud highlighting the most frequent action-action prompts.

questions with inherent uncertainty, having a general-purpose MLLM (Bai et al., 2025; OpenAI, 2024) answer the same question multiple times and averaging the results yields more reliable evaluations. Therefore, for ElementQA, we prompt Qwen2.5-VL-72B-Instruct (Bai et al., 2025) to answer each question five times. For AnomalyQA, considering the higher cost of GPT-4o (OpenAI, 2024), we collect three responses per question. For Q-Align (Wu et al., 2023) in AlignQA, since it is a dedicated model trained for aesthetic quality assessment and directly outputs a quantitative score, we use a single response.

- C Experimental Setup–Model details

Parameter settings. In our implementation, the baseline model is Wan2.1-1.3B (Wan Team et al., 2025). And we set the imagery schedule to {5,20,30,45} and set the imagery size schedule to {10,5,5,5,5}. As shown in Fig. S4, V Q and MQ exhibit the same selection trend, whereas TA shows the opposite. Therefore, regarding the parameters in Equation (5), we set β = γ = 1.0, and α are dynamically adjusted.

AlignQAElementQAAnomalyQA

- Q1: "Is a person present in the scene?"
- Q2: "Is a person performing the action 'polishes furniture attentively' at home?"
- Q3: "Is a person performing the action 'packs cleaning products for organization'?"
- Q4: image quality and aesthetic score
- Q5: "This is a generated video. Please help me determine whether there are any anomalies in the video frames, such as abnormal appearance or structure of objects, abnormal deformation of objects, motion that is unreasonable or violates physical laws, disappearance or discontinuity of objects, as well as artifacts or motion ghosting."

ImageryQA

- (a)
- (b)

Prompt: “A person polishes furniture attentively at home, then packs cleaning products for organization.”

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

Wan2.1

- Q1: 1.00
- Q2: 0.00
- Q3: 0.00
- Q4: 2.18
- Q5: 0.00

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

ImagerySearch (Ours)

- Q1: 1.00
- Q2: 0.20
- Q3: 0.80
- Q4: 4.69
- Q5: 0.33

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

- Figure S3: Evaluation with ImageryQA. (a) We design a structured question set ImageryQA, consisting of ElementQA, AlignQA, and AnomalyQA.(b) Comparison between Wan2.1 and ImagerySearch on the same prompt. Wan2.1 fails to depict a person and the actions described, resulting in low aesthetic quality (Q4) and visual anomalies (Q5). In contrast, ImagerySearch successfully captures both actions–polishing furniture and packing cleaning products–scoring higher in both Q4 and Q5.

- D More Examples

Additional qualitative examples are provided in Fig. S5, Fig. S6, and Fig. S7. Specifically, Fig. S5 reports results on LDT-Bench, where the first five rows correspond to action–action prompts and the last three to object–action prompts. Fig. S6 and Fig. S7 show further action–action cases drawn from VBench. Across all examples, our method produces vivid and coherent videos, even under long-distance semantic prompts, illustrating its capacity to handle challenging imaginative scenarios.

- E Error Analysis

In the VBench (Huang et al., 2024b) error analysis (Fig. S8), ImagerySearch shows a higher mean score with a tighter interquartile range, indicating more stable performance across prompts. Evosearch (He et al., 2025a) attains a comparable median but displays greater dispersion, whereas wan2.1 (Wan Team et al., 2025) and Video-T1 (Liu et al., 2025a) exhibit lower central scores and wider quartile spans. Overall, dynamically adjusting the search space and rewarding by semantic distance helps maintain generation quality while reducing sensitivity to prompt difficulty.

Prompt: “A girl applies eye makeup in her bedroom, then Prompt: “The camel packs its belongings with care.” ties a colorful ribbon at her crafting desk.”

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

MQ=0.6

- MQ=0.6

- MQ=1.0

[Figure 359]

[Figure 360]

TA=0.6

- TA=0.6

- TA=1.0

[Figure 361]

[Figure 362]

- VQ=0.6

- VQ=1.0

VQ=0.6

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

MQ=1.0

[Figure 373]

[Figure 374]

TA=1.0

[Figure 375]

[Figure 376]

VQ=1.0

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

- Figure S4: Reward-Weight Analysis. The left of figure shows an action–action example and the right of figure is an object–action one, visualizing the videos under different weight settings. MQ and V Q follow almost identical trends, whereas TA moves in the opposite direction. Accordingly, we fix the MQ and V Q coefficients to 1 and vary the TA coefficient with the prompt, selecting videos that better fit imaginative scenarios.

A girl applies eye makeup in her bedroom, then ties a colorful ribbon at her crafting desk.

[Figure 385]

Cricket players celebrate victories on lush pitches, then use remote controllers for strategic insights.

[Figure 386]

Pole vaulters compete on athletic fields, then tie knots in practice equipment bags.

[Figure 387]

A person washes face carefully in quiet bathrooms, then ties knots in towels for storage.

[Figure 388]

Dancers embrace elegance with belly dance routines at cultural fairs, then tie knots in costume shawls.

[Figure 389]

The tiger enjoys competitive trampoline events.

[Figure 390]

The shoe shop efficiently utilizes a remote controller.

[Figure 391]

The shovel springs vigorously onto the trampoline.

[Figure 392]

### Figure S5: More examples on LDT-Bench. The images below the prompt show the result of frame sampling, where 16 frames are uniformly extracted from a 33-frame video.

CG game concept digital art, a snowboarder with a sleek black snowboard, wearing a black hoodie and cargo pants. They are standing on a snowy mountain slope, with a clear blue sky above and fluffy white clouds in the distance. The snowboarder has long wavy brown hair blowing in the wind, and their face is covered in snow. They are holding the snowboard tightly with one hand and are about to take off down the slope. The background features towering peaks and deep valleys. The overall scene is vibrant and dynamic, with natural lighting and shadows. Low-angle view, medium shot focusing on the snowboarder's action.

[Figure 393]

Photorealistic astronaut riding a horse in the vast expanse of outer space. The astronaut, wearing a sleek, silver spacesuit with reflective panels, is astride a majestic steed adorned with metallic accents. Both the astronaut and horse are illuminated by the stark glow of stars and distant planets. The horse's coat is a blend of metallic silver and subtle blue hues, reflecting the cold, distant universe around them. The astronaut leans forward slightly, gripping the reigns tightly as they navigate the void. The horse moves with grace and purpose, its hooves creating a soft, rhythmic sound against the vacuum. The background showcases a surreal, star-filled cosmos, with nebulae and cosmic dust swirling around them. The image captures the awe-inspiring moment of this unlikely duo journeying through the cosmos. High-definition photorealistic rendering. Full-body astronaut and horse, medium shot in space environment.

[Figure 394]

A person leans in for a tender kiss, their lips gently touching as they look into each other's eyes. The person has tousled brown hair, soft features, and warm, inviting skin. They are wearing a cozy sweater in a soft pastel color, paired with jeans and sneakers. The background is a softly lit bedroom, with subtle shadows creating depth. A small bedside lamp casts a warm glow, and there are hints of fluffy bedding and a vintage alarm clock on the nightstand. The couple is positioned mid-shot, facing each other with a sense of intimacy. Soft romantic music plays in the background. Medium shot, close-up of the faces.

[Figure 395]

CG game concept digital art, a tranquil tableau featuring a serene cup resting on a polished wooden surface. The cup is made of delicate porcelain, with intricate hand-painted designs in shades of pale blue and green. It sits gracefully on a small, intricately carved wooden coaster. The wooden surface is smooth and gleaming, with subtle grain patterns. Soft, warm ambient lighting casts gentle shadows, highlighting the intricate details of the cup and the coaster. The background is a minimalist, softly textured room with hints of pastel colors, creating a peaceful and serene atmosphere. Low-angle view, medium shot focusing on the detailed cup.

[Figure 396]

### Figure S6: More examples on VBench (Part I). The images below the prompt show the result of frame sampling, where 16 frames are uniformly extracted from a 33-frame video.

A graceful woman in flowing white robes, adorned with intricate silver jewelry, stands gracefully in a well-lit garden. She expertly arranges vibrant roses, lilies, and peonies with practiced precision. Her long, wavy hair flows gently behind her as she speaks softly to herself, her serene expression reflecting the beauty of nature around her. Soft sunlight casts a warm glow over the scene, highlighting the soft petals and delicate textures of the flowers. She pauses occasionally to admire her work, a content smile playing on her lips. The background features lush greenery, winding paths, and a gentle breeze rustling through the leaves. The composition includes various angles and shots, showcasing the woman from the side, full body, and a close-up of her hands deftly arranging the flowers. The scene is captured in a cinematic lighting style, emphasizing the harmony between the artist and her creations.

[Figure 397]

CG game concept digital art, a skilled blacksmith bending metal with intense focus and determination. The blacksmith wears sturdy leather gloves and a heavy apron, standing amidst a cluttered forge filled with glowing coals and molten metal. Sparks fly as he skillfully shapes a piece of iron, his muscles tensing with effort. The environment is dimly lit, with flickering torches casting shadows on the walls. The blacksmith's face is lined with concentration, his jaw set tightly. He holds the metal with both hands, applying pressure and guiding it with precise motions. The metal bends and twists under his touch, creating a rhythmic sound that echoes through the forge. The background is a chaotic yet orderly scene, with tools scattered around and tools hanging from pegs on the wall. Close-up, low-angle view.

[Figure 398]

A person gracefully ice skating on a frozen lake at night. The figure wears a sleek black ice skating outfit with reflective silver accents, allowing their every movement to catch the glow of the streetlights casting a soft, ethereal light. Their hair is pulled back into a sleek ponytail, flowing slightly with each spin. They glide effortlessly across the ice, occasionally stopping to pose with arms outstretched, creating intricate patterns in the snow. The background is a blurred view of the city skyline with twinkling lights, giving a sense of adventure and excitement. Night-time, winter setting with a touch of magic. Low-angle shot focusing on the skater's face, capturing their joy and determination.

[Figure 399]

A person is making a large snowman in a winter landscape. The person is wearing a cozy red coat and a woolen hat, with mittens on their hands. They stand in front of a soft, fluffy snowbank, surrounded by pristine white snow. The person begins by shaping a large ball of snow for the snowman's body, then carefully forming smaller balls for the snowman's arms and head. They use their scarf as a makeshift tool to smooth out the snow, adding details like buttons and a carrot nose. The background is a vast snowy field with distant pine trees and a clear blue sky. The scene captures the joy and determination of building something from the purest of materials. Winter-themed lighting with soft, diffused sunlight highlights the intricate details of the snowman. Close-up shots of the person's hands working, medium shots of the snowman taking shape, and wide shots of the entire setup in progress. Handheld camera movements capture the person's gestures and expressions, providing a sense of movement and engagement.

[Figure 400]

### Figure S7: More examples on VBench (Part II). The images below the prompt show the result of frame sampling, where 16 frames are uniformly extracted from a 33-frame video.

Error Analysis

82.1

80

74.3

VBench(%)

70

63.3 64.2

60

50

40

ImagerySearch Evosearch wan2.1 video_t1

Model

- Figure S8: Error analysis about VBench scores on long-distance semantic prompts. Each box shows the score distribution for one model (mean marked by a white diamond); individual data points are overlaid in matching colors. ImagerySearch (orange) attains the highest mean with the tightest spread, while the other methods exhibit lower central tendencies and larger variances.

