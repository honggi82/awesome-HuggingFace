## Taming Hallucinations: Boosting MLLMs’ Video Understanding via Counterfactual Video Generation

Zhe Huang1,3,*,†, Hao Wen2,3,*,†, Aiming Hao3,*, Bingze Song3, Meiqi Wu3,†, Jiahong Wu3,‡,§, Xiangxiang Chu3, Sheng Lu2,§, Haoqian Wang1,§ 1Tsinghua University 2Beihang University 3AMAP, Alibaba Group https://amap-ml.github.io/Taming-Hallucinations/

# arXiv:2512.24271v1[cs.CV]30Dec2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

|[Figure 9]<br><br>[Figure 10]|
|---|

| |
|---|

Intuitive Answer.Correct Answer. video edit part

[Figure 11]

[Figure 12]

[Figure 13]

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

[Figure 72]

Q: What is the trajectory of the harvested corn as it is being transferred from the harvester?

[Figure 73]

Q: What happens to the pile of rocks the girl is standing on during the video?

- A. The rocks are washed away by the ocean waves.
- B. The pile of rocks remains the same throughout the video.
- C. The rocks change color from grey to green.
- D. The rocks multiply and grow into a much larger pile beneath the girl.

- A. It arches upwards and floats into the sky.
- B. The video does not show the corn being transferred.
- C. It is transferred downwards into a large green trailer.
- D. It is sprayed sideways across the field.

User

<think>The video shows the girl standing on a pile of rocks near the ocean. There is …… The background of the video does not suggest any unusual effects like rocks growing into a larger pile. Given this analysis, the correct answer is that the rocks remain the same throughout the video, and the pile of rocks does not change in any of the options provided.</think> <answer>B</answer>

<think>The video clearly shows …… The corn is being expelled from the harvester's arm in a horizontal direction towards the trailer, not upwards or in any other direction. The trajectory is direct and downward into the trailer, as the corn is collected and contained in the trailer.</think> <answer>C</answer>

[Figure 74]

[Figure 75]

Qwen-VL

Qwen-VL

[Figure 76]

<think>Upon analyzing the frames, …… Its trajectory is upwards in an arc before falling downwards into the large green trailer. This matches the description given in option C …… </think> <answer>C</answer> GPT-4o

<think>In the video, the girl is standing on a pile of rocks. Throughout the video, ……, ruling out option C. The pile of rocks does ……, ruling out option D. The rocks don’t …… eliminating option A. Therefore, the most logical conclusion is that the pile of rocks remains the same throughout the video.</think> <answer>B</answer>

[Figure 77]

###### GPT-4o

<think> Initially, the video shows a green tractor transferring harvested corn into a large green trailer. As the video progresses, the corn is seen being transferred in an arc, rising high above the tractor and then curving back down towards the trailer.</think> <answer>A</answer>

[Figure 78]

[Figure 79]

<think> Initially, the girl is standing on a pile of rocks. As she moves her arms, there is a sudden transformation where the pile of rocks beneath her grows significantly in size and becomes much larger.</think> <answer>D</answer>

DNA-Train-7B (Ours)

DNA-Train-7B (Ours)

Figure 1. Comparison of MLLMs on counterfactual video QA. Each example shows key frames, the multiple-choice question, and the models’ reasoning and answers. General models (Qwen-VL, GPT-4o) rely on language priors, picking plausible but visually incorrect answer. Our DNA-Train-7B model grounds reasoning in the video and selects the counterintuitive yet correct answer.

#### Abstract

terfactual data synthesis framework that employs controllable, diffusion-based video editing to transform real-world videos into counterfactual scenarios. By embedding structured contextual information into the video editing and QA generation processes, the framework automatically produces high-quality QA pairs together with original–edited video pairs for contrastive training. Based on this, we build DualityVidQA, a large-scale video dataset designed to reduce MLLM hallucinations. In addition, to fully exploit the contrastive nature of our paired data, we propose Duality-Normalized Advantage Training (DNA-Train), a two-stage SFT-RL training regime where the RL phase applies pair-wise ℓ1 advantage normalization, thereby enabling a more stable and efficient policy optimization. Experiments on DualityVidQA-Test demonstrate that our method substantially reduces model hallucinations on coun-

Multimodal Large Language Models (MLLMs) have made remarkable progress in video understanding. However, they suffer from a critical vulnerability: an over-reliance on language priors, which can lead to “visual ungrounded hallucinations”, especially when processing counterfactual videos that defy common sense. This limitation, stemming from the intrinsic data imbalance between text and video, is challenging to address due to the substantial cost of collecting and annotating counterfactual data. To address this, we introduce DualityForge, a novel coun-

*Equal contribution. †Work done during the internship at AMAP, Alibaba Group. ‡Project lead. §Corresponding author.

terfactual videos, yielding a relative improvement of 24.0% over the Qwen2.5-VL-7B baseline. Moreover, our approach achieves significant gains across both hallucination and general-purpose benchmarks, indicating strong generalization capability. We will open-source our dataset and code.

#### 1. Introduction

Despite the remarkable advances in Multimodal Large Language Models (MLLMs) [1, 3, 53, 70, 74], studies have revealed a critical vulnerability of them: an over-reliance on language priors at the expense of genuine visual reasoning. This bias fosters “visual ungrounded hallucinations”, whereby models rely predominantly on learned commonsense priors instead of grounding their responses in the visual content [11, 34]. This issue becomes particularly severe when MLLMs process videos depicting counterfactual phenomena, as shown in Fig. 1. When confronted with contents that defy such priors—such as an object vanishing or violating physical laws–MLLMs often disregard the critical visual anomalies. As a result, they produce narratives that are linguistically plausible yet inconsistent with the actual events depicted in the video.

Most prior efforts to mitigate hallucinations in MLLMs have focused on modifying textual data [7, 37, 63], for example, altering video captions, to rebalance the distribution within the text modality. However, a primary cause of these hallucinations lies in the inherent data imbalance of MLLMs, where the scale and diversity of text far surpass those of video [47, 62]. To address this, we advocate enhancing the model’s visual perception through counterfactual data. However, this approach faces two key bottlenecks: (1) producing scalable counterfactual videos (e.g., with visual effects) is both resource-intensive and cost-intensive; and (2) generating high-quality QA pairs is hampered by a paradox: the models’ own limited comprehension precludes reliable automatic data collection and annotation, resulting in a circular dependency that obstructs scalability.

Inspired by the recent advances in AI-Generated Content (AIGC) [2, 44, 45], we introduce a novel data synthesis framework DualityForge that leverages controllable video editing [39, 41], powered by diffusion models [24, 52], to transform real-world videos into counterfactual scenarios, such as erasing an object mid-clip to simulate a sudden disappearance. This type of method enables precise control over the generated events and, critically, embeds structured context (e.g., event type, temporal location) into the editing process. This embedded context provides MLLMs with explicit cues to comprehend counterfactual phenomena, facilitating the automated, scalable creation of highquality QA pairs. Furthermore, this process naturally yields paired data (original vs. edited videos), enabling an inno-

vative contrastive QA training strategy. By requiring the model to provide different answers to identical questions for each video in a pair, we compel it to ground its reasoning in critical visual evidence instead of relying on language priors. Building upon this framework, we construct DualityVidQA, a large-scale video understanding dataset specifically designed to mitigate hallucinations in MLLMs. It comprises 104K samples for SFT and 40K for RL, totaling 144K training samples, and includes 81K unique videos with an overall duration of approximately 100 hours.

In terms of training methodology, we propose Duality-Normalized Advantage Training (DNA-Train), a two-stage regime—Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL)—to mitigate hallucinations while preserving real-world performance. In the initial SFT stage, a hybrid dataset of real and counterfactual videos is used to enable the model to detect anomalies without compromising its performance on real videos. The subsequent RL stage further strengthens this capability by leveraging the previously introduced pair-wise contrastive task. Further, to balance the learning magnitude across different samples and avoid bias towards real videos, we apply ℓ1 normalization to the advantages for each real–counterfactual pair during RL, ensuring stable and balanced gradient updates, thereby better aligning with the contrastive nature of the training set and improving hallucination mitigation.

To evaluate model hallucinations and counterfactual video understanding capabilities, we introduce DualityVidQA-Test, a challenging benchmark of 600 manually-curated paired samples, structured into four finegrained counterfactual classes. Extensive experiments show our model achieves significant performance improvements not only on hallucination (e.g., EventHallusion [67]) but also across leading general-purpose video understanding benchmarks, including TempCompass [40], MVBench [31], TOMATO [49] and TVBench [17], demonstrating its robustness and broad applicability.

We summarize our major contributions as follows:

- • We propose DualityForge, the first counterfactual data synthesis framework that leverages diffusion-based controllable video editing with embedded structured priors to generate precise counterfactual scenarios. Building upon this framework, we introduce DualityVidQA, a large-scale video understanding dataset (144K videoQA pairs) for training and evaluating hallucinations in MLLMs, featuring paired videos with contrastive QA to systematically assess and mitigate model hallucinations.
- • We introduce DNA-Train, a two-stage regime to compel the model to ground its reasoning in visual evi-

dence. In addition, it ℓ1-normalizes the advantages for each real–counterfactual video pair during RL, enabling a more stable and efficient policy optimization.

• Extensive experiments demonstrate that our approach achieves significant gains (24.0% on DualityVidQA-Test) across both hallucination and general benchmarks(e.g., TempCompass, MVBench), indicating strong generalization capability and validating the principle that generation can effectively enhance understanding.

R1 [26] addresses cold-start via a 200K multimodal CoT corpus and GRPO with strict formatting; R1-VL [68] introduces StepGRPO for step-wise rewards that better align intermediate steps with final answers; R1-ShareVL [61] expands the question space and shares reasoning signals to mitigate sparse rewards. VL-Rethinker [55] promotes slow thinking via selective replay and rethinking, and OpenVLThinker [20] interleaves SFT with RL to iteratively refine chains of thought. VLM-R1 [51] emphasizes training stability with rule-based objectives to curb reward hacking; ThinkLiteVL [57] mines hard cases through Monte Carlo Tree Search; and VisionaryR1 [58] encourages grounding with a caption–reason–answer format and LLM-based caption rewards. Despite these advances, most methods still optimize textual traces (e.g., CoT tokens) more than visual evidence, which limits robustness—especially against counterfactual or visually deceptive content. We stress that video understanding is not equivalent to textual reasoning: it requires discriminating visually plausible from counterfactual cues and aligning decisions with grounded evidence.

#### 2. Related Works

##### 2.1. Language Prior in MLLMs

MLLMs inherit strong language priors from LLMs, which can lead to outputs that sound reasonable but conflict with visual evidence. Training-free contrastive decoding reduces this effect by contrasting the original logits with an auxiliary distribution [15, 33], built via image masking, instruction perturbation, visual augmentation, or cross-modal conversion [30, 56, 66, 75]. However, this approach requires additional negative views, increases inference costs, is sensitive to hyperparameters, and does not allow updates to the base model. As a result, performance improvements on video and other temporal tasks are often unstable. Trainingbased methods construct specialized datasets [7, 22, 37], but this involves expensive prompting, filtering, annotation, and QA. In contrast, we propose an automated, scalable data synthesis framework that minimizes manual effort and applies naturally to video.

#### 3. DualityVidQA

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Pipeline 1

[Figure 85]

MLLMs

[Figure 86]

[Figure 87]

Pipeline 2

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

validates

[Figure 92]

##### 2.2. Video Understanding Datasets

[Figure 93]

MLLM

“man”

[Figure 94]

Grounding DINO + SAM

A large body of datasets support research on video understanding across tasks such as action recognition, temporal localization, retrieval, and question answering. Real-world collections include general action and activity corpora (e.g., Kinetics [29], ActivityNet [65], EPIC-KITCHENS [19]), captioning and retrieval sets (e.g., MSR-VTT [60], WebVid-10M [5], HowTo100M [42]). However, curating high-quality video-language annotations is expensive due to spatiotemporal complexity, which constrains the scale and granularity of labeled corpora. To mitigate these costs, recent studies leverage vision language models (VLM) to synthesize video language supervision at scale. LLaVA-Hound [69] and ShareGPT4Video [10] prompt GPT-4 [1] to generate instruction–response and question–answer (QA) pairs from videos, and LLaVA-Video [71] releases about 170K video–instruction examples via a scalable pipeline. These real video-based annotation pipelines show limitations in covering rare events, long-range dependencies and edited counter-commonsense scenarios, while facing category and domain imbalance issues.

“remove the man”

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Pipeline 3

MLLM “almostA personfullholdingglass ofanbeer” FLUX

[Figure 99]

[Figure 100]

validates

MLLMs

Figure 2. Overview of video editing pipelines. There are three pipelines for different types of counterfactual context: Visual Anomaly: pixel-level video editing via OpenCV Semantic Anomaly: an MLLM selects an object for editing, followed by mask generation, VACE-based editing, and majority-vote verification using multiple SOTA MLLMs. Common Sense Anomaly: an MLLM propose commonsense violations, FLUX-Kontext edits frames, edits are re-verified by multiple MLLMs, and VACE interpolates the final video.

##### 3.1. Problem Formulation

Our work is motivated by a critical vulnerability in MLLMs: an inclination to favor dominant language priors over visual evidence [25, 30]. This bias from disproportionate text pre-training over limited video fine-tuning causes visual ungrounded hallucination. To mitigate this, our goal is to craft a large-scale video QA dataset comprising videos that depict visually salient counterfactual events. Each

##### 2.3. Visual Reinforcement Learning

Recent studies extend RL from text-only LLMs to multimodal settings to strengthen VLM understanding. Vision-

QA Generation

CounterFactual (CF) Video Generation

Real Video Dataset

option

Seed QA

[Figure 101]

[Figure 102]

[Figure 103]

Counterfactual Contexts

Q: What is the person in the video doing with the flowers? A: The person is carefully arranging a bouquet of orange tulips in a clear, square glass vase.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

MLLM

|Caption|
|---|

|QA|
|---|

Mask Generation Model

Image/Video Edit Model

MLLM

MLLM

[Figure 108]

[Figure 109]

|Prompt|
|---|

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

CF Video

[Figure 114]

Correct Answer video edit part

video edit part

Visual anomalies

[Figure 115]

[Figure 116]

|[Figure 117]| | |
|---|---|---|
|[Figure 118]| |brightness change|

[Figure 119]

[Figure 120]

Q: How would you describe the visual flow of the splashing lava throughout the video?

background change

- A. The lava splashes and churns continuously and naturally without

any interruptions.

- B. The lava flow reverses and goes back into the crater.
- C. The splashing lava freezes in place for several seconds.
- D. The lava splashes and churns, but a portion of it unnaturally

[Figure 121]

Semantic anomalies

[Figure 122]

[Figure 123]

disappears and reappears towards the end.

[Figure 124]

[Figure 125]

[Figure 126]

object disappear

[Figure 127]

object appear

[Figure 128]

Commonsense anomalies

Q: How many houses are visible in the distance by the end of the video?

[Figure 129]

[Figure 130]

- A. There is a row of several identical houses.
- B. There are no houses visible.
- C. There is one house.
- D. The house disappears by the end of the video.

[Figure 131]

[Figure 132]

uncommon sense

object replacement

[Figure 133]

[Figure 134]

[Figure 135]

CF Video

[Figure 136]

Real Video

600 Pairs

20,000 Pairs

50,000

54,879

DualityVidQA - SFT

DualityVidQA - Test

DualityVidQA - RL

- Figure 3. Overview of the DualityForge framework and DualityVidQA dataset. Starting with real, web-sourced videos, the DualityForge framework first embeds the counterfactual (CF) context, including visual, semantic, and commonsense, into it with video editing pipeline. The embedded context is then provided alongside the video to an MLLM to produce detailed captions and QA pairs. The dataset comprises three splits: DualityVidQA-SFT with real and counterfactual video-QA pairs (54K + 50K) for SFT; DualityVidQA-RL with 20K shared-question contrastive video-answer pairs (one question, two real/CF videos) for RL; and DualityVidQA-Test (600 pairs), which shares the same contrastive structure as DualityVidQA-RL and covers diverse counterfactual categories.

where D is a divergence measure, ϵ and δ are small and large thresholds, respectively. The above optimization problem is presented to articulate our conceptual objective: maximizing the divergence between MLLM and human responses with visual condition, while keeping the divergence with text-only condition low. This formulation is not applied literally in our pipeline, instead, it frames the desired characteristics of effective samples.

video is paired with questions designed to explicitly probe these anomalies, thereby encouraging the model to anchor its reasoning in visual evidence rather than linguistic bias. Formally, let V be a video and C denote the context embodied in V . Our goal is to identify a counterfactual context C within a video V that induces a mismatch between answers based on common-sense language priors and those grounded in visual evidence. We construct a question-answer pair (Q,A) where the question Q specif-

However, solving this optimization problem for automatic large-scale dataset constructing is, in practice, intractable due to two primary bottlenecks:

ically probes this context C, and A = {ai}Ni=1 represents the set of possible answers. To model this discrepancy, we

distinguish between two conditional probabilities P∗ (a | ·) for any agent ∗ ∈ {human,LLM,MLLM}: P∗ (a | Q) conditioned on the question, and P∗ (a | Q,V ) conditioned on the question and video. Our objective is to find the most challenging contexts C that reveal an MLLM’s hallucinations. A data sample is considered effective if it adheres to the following criteria:

- 1. Data Scarcity. Videos featuring naturally occurring counterfactual contexts C are inherently scarce and challenging to collect at scale.
- 2. The Automation Paradox. The MLLMs’ perceptual blindness to these very phenomena prevents us from leveraging them to automate the data collection and annotation, resulting in a circular dependency that obstructs scalability.

D (PMLLM (a | Q,V ),Phuman (a | Q,V )), s.t. D (PLLM (a | Q),Phuman (a | Q)) ≤ ϵ,

max

To overcome these bottlenecks, we propose a paradigm shift that reframes the optimization from a search problem to a synthesis problem. Our approach leverages pre-

C

(1)

D (Phuman (a | Q),Phuman (a | Q,V )) ≥ δ,

defined counterfactual context C with a novel duality: first, it guides controllable diffusion-based video editing to transform a real-world video into a counterfactual video; second, it serves as a semantic blueprint to ground an MLLM’s comprehension of the anomaly, unlocking a fully automated and scalable pipeline for high-quality QA generation, yielding QA pairs that adhere to the following principles:

D (PMLLM (a | Q,V ),Phuman (a | Q,V )) ≥ δ D (PMLLM (a | Q,V,C),Phuman (a | Q,V )) ≤ ϵ.

(2)

##### 3.2. DualityForge

We categorize counterfactual context C into three hierarchical levels of increasing complexity. At the most fundamental level, visual anomalies refer to pixel-wise distortions (e.g., abnormal contrast, saturation) that degrade visual quality without changing scene semantics. Next, semantic anomalies disrupt object-level logic, introducing temporal inconsistencies such as object disappearance or substitution. Finally, commonsense anomalies, the most abstract category, encompass violations of real-world physics and plausibility, including unnatural deformations, impossible movements, or illogical agent interactions. We designed three distinct pipelines corresponding to three different categories of anomalies, as shown in Fig. 2.

Based on the pre-defined C by MLLM, we propose a novel counterfactual data synthesis framework DualityForge (as shown in Fig. 3) that transforms them into a comprehensive counterfactual dataset via a two-stage framework. The first stage involves employing the video editing pipeline in Fig. 2 to embed the context C into a real-world source video, thereby generating the counterfactual video V . The second stage uses the same context C, which acts as a semantic blueprint, enabling an MLLM to first generate an “oracle” caption and then self-produce a diverse set of grounded QA pairs (both multiple-choice and open-ended). Furthermore, we leverage the dual nature of our data (original vs. edited videos) to construct shared-question contrastive QA pairs. In this setup, the same question Q is designed to yield different correct answers when applied to the original video (Vori) versus the edited video (Vedit). This forces the VLM to ground reasoning in actual visual content and detect subtle changes, rather than relying on language prior. Formally, this is achieved when:

D (PMLLM (a | Q,Vori),PMLLM (a | Q,Vedit)) ≥ δ (3)

To ensure the quality of our dataset, we implement a rigorous, model-based quality assurance process. This process validates the success of the video editing in the first stage and verifies the correctness of the generated QA pairs in the second stage. Built upon it, a large-scale, high-quality video understanding dataset, DualityVidQA, is constructed

and partitioned into three dedicated splits: DualityVidQASFT (104K QA pairs from 25K original/edited video pairs), DualityVidQA-RL (20K shared-question contrastive video pairs; 40K QA pairs in total), totaling about 144K training QA pairs, and a human-annotated test set, DualityVidQATest (600 pairs). DualityVidQA-Test is further organized into four primary counter-commonsense scenarios derived from cluster analysis: counter physical, object/scene deformation, attribute change, and causal reversal. The dataset contains 81,274 video clips with a total duration of 100 hours. The majority of videos (80%) last between 2 and 6 seconds, and the remaining 20% exceed 6 seconds in length. Further implementation details and data statistic are available in the supplementary material.

#### 4. DNA-Train

Motivated by the dual nature of our dataset, we present DNA-Train, a two-stage regime, SFT+RL, for mitigating hallucinations without sacrificing real-world performance, which employs a novel dual advantage normalization strategy to balance gradient updates. The structure of the DNA-Train is presented in Fig. 4.

##### 4.1. Supervised Fine-Tuning

Our training begins with a supervised fine-tuning (SFT) stage on DualityVidQA-SFT. The primary objective is twofold: to instill the ability to recognize the embedded context C in edited videos (Vedit), while crucially maintaining robust performance on original, real-world videos (Vori). To prevent the model from developing a bias towards either domain, we employ a balanced sampling strategy, ensuring each training batch contains an equal number of original and counterfactual samples. The training objective follows the cross-entropy loss: LSFT = − Ni=1 log pθ(yi|xi), where (xi,yi) represents the input-output pairs in our dataset, θ denotes the model parameters, and pθ is the model’s probability distribution over tokens.

##### 4.2. Reinforcement Learning

While SFT provides a foundational understanding, it lacks an explicit mechanism to directly penalize hallucinations and reward correct visual grounding. To further sharpen the model’s reasoning, we introduce a second reinforcement learning (RL) stage. Our task has a verifiable, ground-truth outcome, as the model must identify the sole correct answer from a list of choices. This singular ground truth makes our problem a natural fit for the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm [23, 54], which uses a deterministic verifier R : (q,o)  → R to provide unbiased, ground-truth rewards. Within the RLVR framework, algorithms like GRPO [50] have shown promise but often suffer from instability and entropy collapse on complex, long-chain-of-thought tasks—a common scenario in

|Duality-Normalized Advantage<br><br>|[Figure 137]|
|---|
<br><br>|[Figure 138]|
|---|
<br><br>[Figure 139]<br><br>[Figure 140]<br><br>| |
|---|
<br><br>|[Figure 141]|
|---|
<br><br>Normal Advantage<br><br>Step0-Step7<br><br>Step0-Step7<br><br>Step24-Step30<br><br>Advantage Strategy|
|---|

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

|Real Video Forward CF Video Forward RL Backward|
|---|

The image shows a glass being filled with a liquid that appears …… There are no signs of overflow, fish, ice cubes, or lemon slices in the image. Therefore, the correct answer is: C.

[Figure 148]

Real Video

APairofRealVideo

andCFVideo

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

CF Video

[Figure 157]

…

[Figure 158]

Reward

[Figure 159]

MLLM

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

After SFT

The correct answer is B. A small, colorful goldfish is swimming in clear water inside a decorative fishbowl. In the video, as the liquid is poured into the glass, a small, colorful goldfish becomes ……

Q: What is observed inside the container being filled with liquid towards the end of the video?

###### Policy Gradient

[Figure 166]

[Figure 167]

- A. The liquid overflows and spills onto the surface.
- B. A small, colorful goldfish is swimming in clear water inside a

decorative fishbowl.

- C. The container is nearly full of amber liquid, with ripples on

the surface.

- D. The container is filled with ice cubes and a slice of lemon.

Gradient weight update

- Figure 4. Overview of DNA-Train framework. We first perform SFT on our dual dataset to initialize the model. During RL, we sample a group of responses for both real and CF videos, compute their rewards based on task correctness, and calculate the ℓ1 norm of intra-group advantages. Finally, we normalize the advantages across the dual groups to ensure balanced gradients.

video QA. Among the following improvements on GRPO [13, 64], DAPO [64] was specifically designed to overcome these limitations with enhancements for stable optimization over long trajectories. Therefore, we build the RL component of the advantage-normalization strategy upon the robust and scalable DAPO framework. Formally, for each QA pair (q,a), DAPO samples a group of outputs {oi}Gi=1 with their corresponding rewards {Ri}Gi=1, and then optimizes the policy via the following objective:

where

1, if oi is correct, 0, otherwise,

(7)

rc =

is the correctness reward and rf is the format reward.

Duality Advantages Normalization. The gradient of JDAPO(θ) can be expressed1 as:

∇θJDAPO(θ) = E(q,a)∼D, {o

i}Gi=1∼πθ(·|q)

(8)

|oi|

G

1

Aˆi,t∇θ log πθ(oi,t|q,oi,<t) .

G i=1 |oi|

JDAPO(θ) =E(q,a)∼D, {o

i}Gi=1∼πθold(·|q)

t=1

i=1

|oi|

G

As shown in Eq. (8), the DAPO gradient is modulated by the advantage Aˆi,t. We use the ℓ1 norm of advantages, S =

1

min ri,t(θ)Aˆi,t,

G i=1 |oi|

(4)

t=1

i=1

A ˆi , as a proxy for the total learning signal magnitude

i

clip ri,t(θ),1 − ϵlow,1 + ϵhigh A ˆi,t ,

from a group of responses, where Aˆi is the average of tokenlevel advantages. With binary rewards, S becomes a simple function of the average accuracy R in the group:

s.t. 0 < {oi | is equivalent(a,oi)} < G,

where

A ˆi = 2 (1 − R)R, (9)

###### S = |G|

πθ(oi,t | q, oi, < t) πθ

i∈G

ri,t(θ) =

,

(oi,t | q, oi, < t)

This formulation reveals a critical property: the learning signal peaks for tasks of intermediate difficulty (R = 0.5) and diminishes as tasks become trivial or impossible. As shown in Fig. 4, we visualized SR and SCF under real (GR) and counterfactual (GCF) data. During the initial phase of training, the inherent accuracy gap between them creates a systematic imbalance in their learning signals, potentially destabilizing the training process. To counteract this, we introduce Duality-Normalized Advantage, which normalizes the advantages from each group to guarantee equal contribution to the gradient update. It computes scaling factors α∗ = Starget/S∗ (where Starget is the mean of SR and

(5)

old

Ri − mean({Ri}Gi=1) std({Ri}Gi=1)

Aˆi,t =

.

Reward Design. Our RL stage is guided by a dualcomponent reward signal derived from the shared-question contrastive QA pairs. The first component is a correctness reward, a binary score assigned for selecting the single right answer, which forces the model to capture subtle visual information. This is supplemented by a format reward, which encourages adherence to a prescribed reasoning structure. The overall reward is formulated as:

R = rf + rc, (6)

1We assume πθold = πθ for simplicity.

Table 1. Performance comparison of different models on predefined anomaly categories (where CF indicates Counterfactual videos) from the DualityVidQA-test set. For each column, bold denotes the best score and underline denotes the second-best score.

Attribute Change Causal Reversal Counter Physical Object/Scene Deformation Overall

Model

Real CF Both Real CF Both Real CF Both Real CF Both Real CF Both Random 27.3 27.3 9.1 25.3 20.3 5.1 19.0 22.2 2.3 28.9 28.3 5.9 24.2 23.9 4.5

GPT-4o-mini [27] 84.8 51.5 36.4 89.9 58.2 50.0 91.4 53.4 48.9 95.2 62.6 59.9 91.8 57.4 51.9 GPT-4o [27] 87.9 75.8 63.6 93.7 74.7 69.6 91.0 68.3 61.1 94.7 73.8 68.4 92.7 72.1 65.8 GPT-4.1 [46] 84.8 84.8 69.7 89.2 81.6 73.4 86.4 68.8 59.7 87.2 76.5 65.2 87.3 75.5 65.6 Gemini-2.5 Flash [16] 75.8 72.7 54.5 88.6 74.1 67.7 89.1 62.0 55.2 92.0 66.8 59.4 89.1 67.3 59.8 Gemini-2.5 Pro [16] 84.8 81.8 69.7 91.8 88.0 80.4 92.8 78.3 73.3 94.1 75.9 71.1 92.5 80.3 74.3

Qwen2.5-VL-7B [3] 87.9 60.6 48.5 88.0 57.0 46.2 93.7 53.8 49.3 93.0 69.5 63.1 91.7 59.9 52.8 Qwen2.5-VL-32B [3] 87.9 54.5 45.5 94.3 68.4 63.3 95.5 43.0 39.4 96.8 59.4 56.1 95.2 55.4 51.3 Qwen2.5-VL-72B [3] 84.8 60.6 45.5 93.7 71.5 65.2 96.8 52.9 50.7 98.4 67.4 65.8 95.8 62.8 58.9 VideoChat2-HD [31] 21.2 27.3 3.0 27.2 27.2 1.3 20.8 26.7 0.0 29.9 27.8 0.5 25.4 27.2 0.7 LLaVA-Next-Video [70] 57.6 33.3 9.1 67.1 29.7 13.9 69.2 31.7 16.3 71.1 42.8 21.4 68.6 34.7 16.9 Video-LLaVA-7B [35] 54.5 39.4 15.2 56.3 42.4 17.1 71.5 33.5 16.3 58.3 51.3 20.3 62.4 41.7 17.7

DNA-Train-7B(ours) 97.0 72.7 72.7 94.3 74.1 69.0 94.6 83.3 79.2 98.4 82.9 81.3 95.8 80.1 76.8

SCF) and applies them to their respective advantages. This elegant re-weighting scheme (Aˆ′∗ = α∗Aˆ∗) guarantees a balanced learning signal across disparate data types, fostering robust and equitable optimization. Further derivation details are available in the supplementary material.

#### 5. Experiment 5.1. Experimental Setup

Benchmarks. We evaluate our model’s performance across two categories of benchmarks: those focused on hallucination detection (DualityVidVQA-Test and EventHallusion [67]) and general video understanding benchmarks, including TempCompass [40], MVBench [31], TOMATO [49], and TVBench [17]. Crucially, for DualityVidVQA-Test, we employ a stricter pairwise accuracy, where a sample is only counted if the model correctly answers for both the original and edited videos. Frame sampling adheres to each benchmark’s standard protocol: 16 frames for DualityVidQA-Test and TOMATO, 64 for TempCompass, and 8 for MVBench and TVBench. Since our constructed dataset primarily consists of short video clips, we select evaluation benchmarks in which the video durations are within 30 seconds, ensuring that the temporal scope of the benchmarks is consistent with the characteristics of our dataset. Moreover, these benchmarks collectively assess a broad spectrum of abilities, providing a comprehensive evaluation of the model’s performance across key aspects of video comprehension.

Implementation Details. We leverage LLamaFactory [73] for SFT and SWIFT [72] for RL, applying both to the powerful Qwen2.5-VL base model. In the SFT stage, all models were trained for one epoch with a learning rate of 1×10−6 and batch size of 4, using 8 H200 GPUs for 7B models and 16 for 32B/72B models. The RL stage main-

tained the same learning rate but with batch size of 64 and 16 sampled responses per prompt, running for 600, 60, and 20 steps for the 7B, 32B, and 72B models, respectively. For evaluation, we use greedy decoding (temperature=0) to ensure deterministic outputs.

##### 5.2. Experimental Results

Table 2. Performance comparison of different models on various benchmarks. For each task, bold denotes the best score and underline denotes the second-best score.

Hallucinations General Video Understanding EventHallusion DualityVidQA-Test TempCompass MVBench TOMATO TVBench

Model

GPT-4o [27] 73.3 65.8 73.8 47.8 37.7 35.8 VideoChat2-HD [31] 20.0 0.7 38.5 51.1 - 34.7 LLaVA-Next-Video [70] 12.1 16.9 44.7 42.2 20.1 38.2 Video-LLaVA-7B [35] 29.7 17.7 49.8 42.5 23.6 33.8 Qwen2.5-VL-7B [3] 33.5 52.8 71.4 62.6 26.8 51.7 DNA-Train-7B(ours) 61.3↑ 27.8 76.8↑ 24.0 73.5↑ 2.1 63.8↑ 1.2 32.6↑ 5.8 53.0↑ 1.3

Our analysis in Tab. 1 highlights a significant and consistent weakness across all evaluated MLLMs: a dramatic performance drop when moving from real to counterfactual videos. While leading closed-source models like GPT4.1 and Gemini-2.5 Pro achieve 92% accuracy on “Real” videos, their performance on “Counterfactual” (CF) content is substantially lower. This gap is most evident in the overall results, where even the top-performing model, Gemini2.5 Pro, drops from 92.5% (Real) to 80.3% (CF). This vulnerability is particularly acute in more challenging scenarios. For instance, in the “Counter Physical” category, most models struggle. However, our DNA-Train-7B demonstrates superior resilience, achieving a remarkable 79.2% in this category. As further confirmed in Tab. 2, our training methodology yields a dual benefit. First, DNA-Train-7B establishes itself as state-of-the-art in hallucination detection, achieving a top score of 76.8% on DualityVid-Test and massively outperforming other open-source models on Even-

tHallusion. Critically, this specialization does not come at the cost of general video understanding. On the contrary, DNA-Train-7B consistently improves upon its base model (Qwen2.5-VL-7B) across all general benchmarks and remains highly competitive with, or even superior to, closed-source leaders like GPT-4o on benchmarks such as MVBench and TVBench. This ability to mitigate hallucinations while preserving broad video understanding capabilities marks a significant advance.

##### 5.3. Ablation Studies

Ablations on Data Configurations. As shown in Tab. 3, our ablation study on data configuration clearly demonstrates the necessity of our paired-data approach. Training on a single data type is markedly detrimental to our core task: using real data alone causes DualityVidTest performance from the paired-data baseline of 52.8 to 29.0, while counterfactuals alone are even more damaging, with accuracy dropping to 13.1. In contrast, the paireddata setting produces a clear synergistic effect—boosting DualityVid-Test performance to 70.6 and achieving the highest average improvement (+1.8) on the general video understanding benchmark. Intriguingly, training solely on counterfactual data improves general understanding (+1.7), suggesting that such data encourages the model to acquire more robust and generalizable visual representations.

Table 3. Ablation Study on Different Dataset Configurations.

Hallucinations

General Video Understanding

Avg Impr. EventHallusion DualityVidQA-Test TempCompass MVBench TOMATO TVBench

Avg Impr.

Setting

Base 33.5 52.8 - 71.4 62.6 26.8 51.6 Real Data 29.4 29.0 ↓ 7.9 72.4 61.5 23.5 50.9 ↓ 2.1 CF Data 57.5 13.1 ↓ 18.0 70.4 63.7 32.2 52.8 ↑ 1.7 Paired Data 49.0 70.6 ↑ 16.7 73.6 64.2 30.7 51.2 ↑ 1.8

Ablations on Duality-Normalized Advantages. To isolate the effectiveness of our DNA strategy, we conducted an ablation study comparing it against strong RL baselines (GRPO, DAPO), starting from the same SFT-trained model. As shown in Tab. 4, DNA demonstrates clear superiority on the primary task of hallucination detection with an average improvement of 10.8. Furthermore, DNA also outperforms DAPO across every single general video understanding benchmark, demonstrating the effectiveness of our advantage normalization strategy.

Table 4. Ablation Study on Different RL Training Strategies.

Hallucinations

General Video Understanding

Avg Impr. EventHallusion DualityVidQA-Test TempCompass MVBench TOMATO TVBench

Avg Impr.

Method

Base 57.8 58.7 - 72.2 63.7 31.6 51.5 GRPO 60.8 74.6 ↑ 9.5 73.5 63.6 32.5 52.6 ↑ 0.8 DAPO 60.6 74.8 ↑ 9.5 73.0 63.0 32.5 52.6 ↑ 0.5 DNA 61.3 76.8 ↑ 10.8 73.5 63.8 32.6 53.0 ↑ 1.0

Ablations on Model Scales and Training Stages. Table 5 reports results on different model scales and two training stages. DNA-Train consistently improves the Qwen2.5VL model across all evaluated scales. The largest gains are in hallucination detection, with the full DNA-Train increasing the average score by 25.9 points for the smallest model

Table 5. Ablation Study on Different Model Sizes and Training Stages.

Hallucinations

General Video Understanding

Avg Impr. EventHallusion DualityVidQA-Test TempCompass MVBench TOMATO TVBench

Avg Impr.

Type Model

Base 33.5 52.8 - 71.4 62.6 26.8 51.6 -

7B

+ SFT 57.8 58.7 ↑ 15.1 72.2 63.7 31.6 51.5 ↑ 1.7 + SFT+RL 61.3 76.8 ↑ 25.9 73.5 63.8 32.6 53.0 ↑ 2.6

Base 34.0 51.2 - 75.2 61.5 31.0 51.5 -

32B

+ SFT 55.6 60.0 ↑ 15.2 74.1 61.7 33.6 54.3 ↑ 1.1 + SFT+RL 58.8 60.8 ↑ 17.2 74.2 61.9 34.6 54.7 ↑ 1.4

Base 54.6 58.9 - 77.6 64.8 36.3 55.5 -

72B

+ SFT 64.6 68.3 ↑ 9.7 78.0 65.7 35.7 56.9 ↑ 0.5 + SFT+RL 65.4 69.4 ↑ 10.7 78.3 65.9 36.5 57.3 ↑ 0.9

variant. Crucially, these gains are accompanied by consistent improvements in general video understanding across all scales. In this process, SFT provides a strong foundation, while the subsequent RL step yields the largest boosts, particularly on the challenging DualityVid-Test benchmark. The smaller performance gain observed for the 72B model is primarily attributable to its reduced RL training schedule20 optimization steps compared to 60 for the 32B and 600 for the 7B -an intentional trade-off necessitated by computational resource constraints.

Ablations on Model Type. We evaluated two opensource MLLMs: LLaVA-Next-Video[70] and Qwen2.5VL. As shown in Tab. 6, after training on DualityVidQA with our DNA-Train, both models consistently outperformed their baselines across all metrics. Specifically, on LLaVA-Next-Video, which starts from a lower baseline, the performance gain is 42.0 and 3.7 on hallucination and general benchmarks. These results indicate that our DNA-Train method not only enhances counterfactual reasoning ability significantly, especially on DualityVidQA-Test, but also improves general video understanding performance across different model architectures, demonstrating its robustness and broad applicability.

Table 6. Ablation Study on Different Model Types.

Hallucinations

General Video Understanding

Avg Impr. EventHallusion DualityVidQA-Test TempCompass MVBench TOMATO TVBench

Avg Impr.

Model Stage

Base 33.5 52.8 - 71.4 62.6 26.8 51.6 -

Qwen2.5vl 7B

+DNA-Train 61.3 76.8 ↑ 25.9 73.5 63.8 32.6 53.0 ↑ 2.6

Base 12.1 16.9 - 44.7 42.2 20.1 38.2 -

LLaVA-Next-Video

+DNA-Train 51.9 67.6 ↑ 42.0 52.9 46.8 21.4 38.7 ↑ 3.7

#### 6. Conclusion

In this work, we address the critical issue of visual hallucinations in MLLMs, which stems from an over-reliance on language priors when processing visual content. To this end, we introduce DualityForge, a novel framework that uses controllable video editing to generate a largescale (144K) contrastive dataset, DualityVidQA, comprising paired real and counterfactual videos. Building on this, we propose DNA-Train, a two-stage regime that ℓ1normalizes advantages per real-counterfactual pair during RL to ensure balanced training and compel the model to ground its reasoning in visual evidence. Extensive experiments demonstrate that our approach not only significantly reduces hallucinations but also boosts performance

on general video understanding benchmarks. By converting counterfactual, commonsense-defying videos into highquality training data, we tame hallucinations and thus boost MLLMs’ video understanding.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 2, 3

- [2] Andrea Agostinelli, Timo I Denk, Zal´an Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325, 2023. 2
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 7
- [4] Sule Bai, Mingxing Li, Yong Liu, Jing Tang, Haoji Zhang, Lei Sun, Xiangxiang Chu, and Yansong Tang. Univg-r1: Reasoning guided universal visual grounding with reinforcement learning. arXiv preprint arXiv:2505.14231, 2025.
- [5] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021. 3
- [6] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506,

2025. 1

- [7] Cong Chen, Mingyu Liu, Chenchen Jing, Yizhou Zhou, Fengyun Rao, Hao Chen, Bo Zhang, and Chunhua Shen. Perturbollava: Reducing multimodal hallucinations with perturbative visual training. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3
- [8] Chubin Chen, Jiashu Zhu, Xiaokun Feng, et al. S2-guidance: Stochastic self guidance for training-free enhancement of diffusion models. arXiv preprint arXiv:2508.12880, 2025.
- [9] Chubin Chen, Jiashu Zhu, Chenyang Zhu, Jiangshan Wang, Nisha Huang, Chengyu Fang, Jiahong Wu, Xiangxiang Chu, and Xiu Li. Storyctrl: Customized story visualization with fine-grained control, 2025.
- [10] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 3
- [11] Meiqi Chen, Yixin Cao, Yan Zhang, and Chaochao Lu. Quantifying and mitigating unimodal biases in multimodal large language models: A causal perspective. arXiv preprint arXiv:2403.18346, 2024. 2
- [12] Rui Chen, Lei Sun, Jing Tang, Geng Li, and Xiangxiang Chu. Finger: Content aware fine-grained evaluation

- with reasoning for ai-generated videos. arXiv preprint arXiv:2504.10358, 2025.
- [13] Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025. 6
- [14] Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified self-supervised pretraining for image generation and understanding. arXiv preprint arXiv:2503.06132, 2025.
- [15] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. arXiv preprint arXiv:2309.03883, 2023. 3
- [16] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 7
- [17] Daniel Cores, Michael Dorkenwald, Manuel Mucientes, Cees GM Snoek, and Yuki M Asano. Tvbench: Redesigning video-language evaluation. arXiv preprint arXiv:2410.07752, 2024. 2, 7
- [18] Corran. Pexel Videos, 2022. Accessed: 2025-09-19. 1
- [19] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Scaling egocentric vision: The epic-kitchens dataset. In Proceedings of the European conference on computer vision (ECCV), pages 720–736, 2018. 3
- [20] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. arXiv preprint arXiv:2503.17352, 2025. 3
- [21] Xiaokun Feng, Haiming Yu, Meiqi Wu, Shiyu Hu, Jintao Chen, Chen Zhu, Jiahong Wu, Xiangxiang Chu, and Kaiqi Huang. Narrlv: Towards a comprehensive narrative-centric evaluation for long video generation models. arXiv preprint arXiv:2507.11245, 2025.
- [22] Anisha Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 18135–18143, 2024. 3
- [23] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 5
- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [25] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multimodal large language models via over-trust penalty and retrospection-allocation. In Proceedings of the IEEE/CVF

- Conference on Computer Vision and Pattern Recognition, pages 13418–13427, 2024. 3
- [26] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749,

2025. 3

- [27] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 7
- [28] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 1
- [29] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950,

2017. 3

- [30] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13872–13882, 2024. 3
- [31] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195– 22206, 2024. 2, 7
- [32] Mingxing Li, Rui Wang, Lei Sun, Yancheng Bai, and Xiangxiang Chu. Next token is enough: Realistic image quality and aesthetic scoring with multimodal large language model. arXiv preprint arXiv:2503.06141, 2025.
- [33] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. arXiv preprint arXiv:2210.15097,

2022. 3

- [34] Zongxia Li, Xiyang Wu, Guangyao Shi, Yubin Qin, Hongyang Du, Tianyi Zhou, Dinesh Manocha, and Jordan Lee Boyd-Graber. Videohallu: Evaluating and mitigating multi-modal hallucinations on synthetic video understanding. arXiv preprint arXiv:2505.01481, 2025. 2
- [35] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 7
- [36] Xinran Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. Vmbench: A benchmark for perception-aligned video motion generation. arXiv preprint

- arXiv:2503.10076, 2025.

- [37] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning. In The

- Twelfth International Conference on Learning Representations, 2024. 2, 3
- [38] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024. 1
- [39] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 2
- [40] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024. 2, 7
- [41] Fangyuan Mao, Aiming Hao, Jintao Chen, Dongxia Liu, Xiaokun Feng, Jiashu Zhu, Meiqi Wu, Chubin Chen, Jiahong Wu, and Xiangxiang Chu. Omni-effects: Unified and spatially-controllable visual effects generation. arXiv preprint arXiv:2508.07981, 2025. 2
- [42] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019. 3
- [43] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. arXiv preprint arXiv:2407.02371, 2024. 1
- [44] OpenAI. DALL-E 3, 2023. Accessed: 2025-09-20. 2
- [45] OpenAI. Sora, 2024. Accessed: 2025-09-20. 2
- [46] OpenAI. Introducing GPT-4.1 in the API, 2025. Accessed: 2025-09-19. 7
- [47] Renjie Pi, Tianyang Han, Wei Xiong, Jipeng Zhang, Runtao Liu, Rui Pan, and Tong Zhang. Strengthening multimodal large language model with bootstrapped preference optimization. In European Conference on Computer Vision, pages 382–398. Springer, 2024. 2
- [48] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 1
- [49] Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models. arXiv preprint arXiv:2410.23266,

2024. 2, 7

- [50] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 5
- [51] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao,

Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint

- arXiv:2504.07615, 2025. 3

- [52] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 2
- [53] Team Gemini, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2
- [54] Team Kimi, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025. 5
- [55] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025. 3
- [56] Xintong Wang, Jingheng Pan, Liang Ding, and Chris Biemann. Mitigating hallucinations in large vision-language models with instruction contrastive decoding. arXiv preprint arXiv:2403.18715, 2024. 3
- [57] Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. arXiv preprint arXiv:2504.07934, 2025. 3
- [58] Jiaer Xia, Yuhang Zang, Peng Gao, Yixuan Li, and Kaiyang Zhou. Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning. arXiv preprint arXiv:2505.14677, 2025. 3
- [59] Feng Xiong, Hongling Xu, Yifei Wang, Runxi Cheng, Yong Wang, and Xiangxiang Chu. Hs-star: Hierarchical sampling for self-taught reasoners via difficulty estimation and budget reallocation. arXiv preprint arXiv:2505.19866, 2025.
- [60] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 3
- [61] Huanjin Yao, Qixiang Yin, Jingyi Zhang, Min Yang, Yibo Wang, Wenhao Wu, Fei Su, Li Shen, Minghui Qiu, Dacheng Tao, et al. R1-sharevl: Incentivizing reasoning capability of multimodal large language models via share-grpo. arXiv preprint arXiv:2505.16673, 2025. 3
- [62] Yongqiang Yao, Jingru Tan, Feizhao Zhang, Jiahao Hu, Yazhe Niu, Bo Li, Pengfei Liu, Ruihao Gong, Dahua Lin, Ningyi Xu, et al. Omnibal: Towards fast instruction-tuning for vision-language models via omniverse computation balance. In Forty-second International Conference on Machine Learning, 2025. 2
- [63] Qifan Yu, Juncheng Li, Longhui Wei, Liang Pang, Wentao Ye, Bosheng Qin, Siliang Tang, Qi Tian, and Yueting Zhuang. Hallucidoctor: Mitigating hallucinatory toxicity in visual instruction data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12944–12953, 2024. 2

- [64] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 6
- [65] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9127–9134, 2019. 3
- [66] Ce Zhang, Zifu Wan, Zhehan Kan, Martin Q Ma, Simon Stepputtis, Deva Ramanan, Russ Salakhutdinov, LouisPhilippe Morency, Katia Sycara, and Yaqi Xie. Selfcorrecting decoding with generative feedback for mitigating hallucinations in large vision-language models. arXiv preprint arXiv:2502.06130, 2025. 3
- [67] Jiacheng Zhang, Yang Jiao, Shaoxiang Chen, Na Zhao, Zhiyu Tan, Hao Li, and Jingjing Chen. Eventhallusion: Diagnosing event hallucinations in video llms. arXiv preprint arXiv:2409.16597, 2024. 2, 7
- [68] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025. 3
- [69] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258, 2024. 3
- [70] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. Accessed: 2025-09-20. 2, 7, 8
- [71] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 3
- [72] Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, et al. Swift: a scalable lightweight infrastructure for fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 29733–29735, 2025. 7
- [73] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. arXiv preprint arXiv:2403.13372, 2024. 7
- [74] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2
- [75] Lanyun Zhu, Deyi Ji, Tianrun Chen, Peng Xu, Jieping Ye, and Jun Liu. Ibd: Alleviating hallucinations in large visionlanguage models via image-biased decoding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1624–1633, 2025. 3

## Taming Hallucinations: Boosting MLLMs’ Video Understanding via Counterfactual Video Generation

### Supplementary Material

#### A. Datset Detail

We categorize video anomalies into three levels: Visual anomalies refer to pixel-wise distortions, including abnormal contrast, saturation, brightness, blurring, and local distortions, etc., which primarily affect visual quality without explicit semantic alteration. Semantic anomalies involve violations of scene semantics, such as object disappearance, unexpected object emergence, and object substitution, which result in temporal inconsistencies. Commonsense anomalies capture more abstract and holistic violations involving spatio-temporal or physical implausibility, such as unnatural deformations, implausible object movements, unreasonable interaction and human motion anomalies, etc.

##### A.1. DualityForge

Table A.1. Definitions of video anomaly categories.

Category Definition Visual Pixel-wise distortions that primarily affect visual quality without explicit

semantic alteration. These include abnormal contrast, saturation, brightness, blurring, and local distortions.

Semantic Violations of scene semantics, such as object disappearance, unexpected object emergence, and object substitution, resulting in temporal inconsistencies.

Commonsense Abstract and holistic violations involving spatio-temporal or physical implausibility (e.g., unnatural deformations, implausible object movements, unreasonable interactions, and human motion anomalies).

Video Source. To improve video-editing quality and dataset diversity, we adopt two widely used public datasets Pexels [18] and OpenVid [43] which are commonly employed in video-generation research. From OpenVid, we randomly sample around 3,000 videos from each of the 20 most populated categories, yielding a candidate pool of 61,591 clips. From Pexels, we additionally sample 36,333 clips, for a total of 97,924 videos.

Visual anomalies. We employ OpenCV to synthesize visual anomalies within the video data. We divide visual anomalies into entire-frame level, region level, and object level. To introduce anomalies, we randomly select a temporally consistent segment in which to insert visual perturbations. At the object level, we first extract all noun entities present in the video and randomly select one object. Then we utilize Grounding DINO[38] and SAM[48] to localize the position of the selected object, on which the visual anomaly synthesis operation is performed.

Semantic anomalies. We categorize semantic anomalies to include both the temporal instability of entities (e.g., un-

expected appearance, disappearance, or substitution) and appearance-level abnormalities (such as unreadable text or blurred faces). To enable controlled injection of anomalies into the video while keeping the other part unchanged, we utilize the advanced video editing model, VACE[28], to edit the specific area in the video.

Common sense anomalies. We categorize anomalies that contradict common sense into the following types: violations of physical laws, causal inconsistencies, material abnormalities, and abnormal human movements. To introduce the first three types of anomalies into videos, we first employ a Multimodal Large Language Model (MLLM) to analyze the visual elements within an image and generate an editing instruction targeting the anomaly. Next, we use FLUX-Kontext[6] to edit the image according to this instruction. After validating the edited image, we create a video by performing frame interpolation with VACE using the original and edited image pair.

Finally, we collect a total of 135,168 videos with anomalies, which are subsequently subjected to an additional screening process to ensure quality prior to their use in QA construction. The statistics of video types are shown in Tab. A.2. This takes around 40k GPU hours on NVIDIA H20 GPUs.

Table A.2. Video dataset type statistics

Type Count color 27,353 replacement 9,961 appearance 6,092 disappear 5,016 common sense 86,746 All 135,168

##### A.2. DUALITYVIDQA

Training Data Construction. To enhance VLM counter-commonsense reasoning while preserving general VideoQA performance, we adopt a two-stage training framework: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). For each stage, we curate a tailored dataset to support its specific training objective. We conducted two rounds of data curation to ensure optimal training quality. In our first round, we constructed initial datasets for both SFT and RL stages. We generated 200k QA pairs from 80k videos. After analyzing the training performance,

we observed that samples with zero reward were predominantly associated with failed video edits where no meaningful visual changes were created. Thus, we use the first stage trained model to filterout around 30% of the samples with zero reward and low-quality video. This insight led us to create a refined dataset through the following process:

Table A.3. Question type frequency statistics

QA Type Real Video Counterfactual Video

Multiple Choice 12,210 10,224 Open-Ended 42,669 39,776

All 54,879 50,000

(1) SFT data construction through two stages: dense captioning and question-answer (QA) generation. During dense captioning, a red box is used to indicate the anomaly region, and video editing metadata is provided to the model to generate detailed, high-coverage captions under controlled conditions. The detailed prompt is Dense Caption Prompt Template in Fig. A.1. During QA generation, we followed LLaVA-Video, categorizing questions into 16 types and using GPT-5 and Gemini 2.5 Pro to generate questions and answers based on video content and dense captions. To ensure diversity and stability, we sampled 5,000 examples from LLaVA-Video’s 170k dataset as a pool, randomly selecting three same-category examples

Table A.4. 16 Question type frequency statistics with descriptions

QA Type Real Video Counterfactual Video Description Attribute Change 1,436 8,674 Questions about changes in attributes of objects or char-

acters between scenes or frames.

Binary 2,009 1,009 Involves yes or no questions related to the video content. Camera Direction 1,601 4,887 Tests understanding of the camera’s movement or shoot-

ing direction within the video. Causal 737 216 Focuses on explaining actions/events, determining intentions of actions or causes for events. Count 363 438 Tests ability to count instances of objects, people, or ac-

tions. Description Human 15,360 4,324 Involves describing actions or attributes of people. Description Object 8,450 4,404 Assesses ability to describe attributes of objects. Description Scene 19,067 8,317 Assesses ability to describe the major scene of the video. Fine-grain Action Understanding 811 1,303 Creates questions challenging comprehension of subtle

actions. Non-Existent Actions with Existent Scene Depictions 29 113 Tests ability to identify actions that did not occur despite related scene elements being present. Object Direction 420 3,374 Tests understanding of the movement or facing direction

of objects within the video. Plot Understanding 981 151 Challenges ability to interpret the plot in the video. Spatial 2,074 8,641 Tests ability to perceive spatial relationships between ob-

served instances in a video scene. Speed 221 998 Involves estimating or comparing the speed of moving objects or actions. Temporal 768 2,789 Designed to assess reasoning about temporal relationships between actions/events. Time Order Understanding 552 362 Tests comprehension of the chronological order of events or actions in the video.

All 54,879 50,000 Aggregate counts for all question types.

at each generation step as in-context references to maintain stylistic consistency and content diversity. Finally, we curated 25K real videos and 25K edited videos, generating 100K QA pairs with an 8:2 ratio of open-ended to multiplechoice items using Real Video QA Generation Prompt Template in Fig. A.3 and Counterfactual Video QA Generation Prompt Template in Fig. A.2 respectively. Then we use GPT-4o to classify each QA into question types based on the LLaVA-Video taxonomy. The qa detail statistics are shown in Tab. A.4 and Tab. A.3. The examples of

SFT QA are shown in Fig. A.5.

- (2) RL data construction centers on creating shared-

question counterfactual QA pairs: for each real and edited video pair, we design the same question and identical answer candidates, but the correct answer differs between the two videos. This forces the VLM to ground reasoning in actual visual content and detect subtle changes, rather than relying on prior plausibility. We construct the RL dataset using Gemini2.5-Pro, which generates counterfactual QA pairs from video captions by identifying visual differences. The prompting strategy follows the RL Question Generation Prompt in Fig. A.4. In total, we curate 20K counterfactual QA pairs as the RL training dataset. The examples of RL QA are shown in Fig. A.6.

Table A.5. Counterfactual video category statistics in DualityVidQA-Test

Tag Count

causal reversal 158 counter physical 221 object/scene deformation 187 attribute change 33 All 599

- (3) Test Set. We construct a high-quality test set,

DualityVidQA-Test, to evaluate counterfactual understanding. Firstly, we sampled around 2000 pairs from our paired video pool. Then, we employ Gemini 2.5 Pro to generate candidate based on video content and dense captions. The prompt is RL Question Generation Prompt in Fig. A.4. Then we employ 3 human annotators and 3 expert reviewers to filter and refine the generated QA pairs, ensuring each question is valid, unambiguous, and answerable based on the video content.

The final test set consists of 600 real-counterfactual video pairs, each with a shared question and options but different answers. We then cluster the test set into 12 categories, then manually cluster them into 4 major categories: counter physical, object/scene deformation, causal reversal, and attribute change. The statistics of counterfactual video categories are shown in Tab. A.5. The examples of test QA are shown in Fig. A.7.

#### B. Derivation

Here we show the derivation of

A ˆi = 2 (1 − R)R. (B.10)

###### S = |G|

i∈G

We consider the case where the reward values Ri are binary, i.e.,

Ri ∈ {0,1}. (B.11)

###### Dense Caption Prompt Template

You are a professional video understanding and visual anomaly detection expert. Please generate a description for the given video. Given [num frames] uniformly sampled frames from the video (total duration approximately [video duration] seconds), the time periods are [time list], please generate a detailed description in chronological order. I will use a red box to figure out the anomaly region.

- • anomaly type: [anomaly type]

- • region type: [region type]

- • region name: [region name]

- • anomaly start time(s): [anomaly start time]

- • anomaly end time(s): [anomaly end time] Please pay special attention to the following points:

- 1. Describe in detail the important objects, actions, and relationships in each time period of the video, following chronological order, and merge the same content to ensure the dense caption is efficient and clear.
- 2. Carefully analyze and point out any visual anomalies, such as:

- • Unnatural changes in object appearance (distortion, warping, blurring, etc.)
- • Perspective distortion or geometric deformation in specific regions
- • Discontinuities or unnatural transitions in object edges
- • Abnormal changes in texture or color
- • Unnatural changes in lighting effects
- • Anomalous behavior in specific regions of the video (e.g., lens, objects)

- 3. For each detected anomaly, please specify in detail:

- • The exact time period when the anomaly appears
- • The specific region or object affected by the anomaly
- • You need to convert the anomaly parameters into natural language descriptions (do not output values like ’saturation factor is xx’, and do not output specific region coordinates).

- 4. If there are other anomalies, such as blurring, missing content, unrecognizable scenes, etc., please clearly point them out in the relevant paragraphs. But don’t need mention red box in the description.

The output format should be JSON, including the following content: {

"spatial_location": "[region_name]", "merged_timestamps": ["[0.0s - ...]", "[... - ...]"], "dense_captions": [

"[0.0s - ...]: ...", "[... - ...]: ...", "...", "[... - video_end_time]: ..."

] }

Figure A.1. Dense Caption Prompt Template

Let |G| be the size of the group, and let

R =

1 |G| i∈G

Ri (B.12)

denote the accuracy of the group (i.e., the fraction of correct responses).

###### Standard Deviation of rewards.

R · |G| · (1 − R)2 + (1 − R) · |G| · (0 − R)2 |G|

std({Ri}Gi=1) =

= R · (1 − R)

(B.13) The magnitude of the advantage is therefore:

Counterfactual Video Multiple Choice and Open-Ended Question Generation Prompt

Task Given a detailed description that summarizes the content of the generate-video, generate questionanswer pairs to build LLM training data. Reference Examples: Here is one question dimension and its explanation and example question-answer pairs for reference: Question Type: [question type]

- Example 1:

- ## caption-1: [Video description]

- ## question-1: [Question text]

- ## answer-1: [Answer text]

Example 2:

## caption-2: [Video description] ## question-2: [Question text]

- ## answer-2: [Answer text]

Example 3:

## caption-3: [Video description] ## question-3: [Question text]

- ## answer-3: [Answer text]

You need to generate similar question-answer pairs like the examples. Guidelines For Question Generation:

- • Please formulate questions using only objectively observable information, without presupposing or emphasizing any abnormal, strange, or logically impossible phenomena in the questions themselves.
- • The questions should be neutral and natural, while the answers may accurately describe the observed phenomena.
- • Each multiple-choice question should have 4 options (A, B, C, D), with only one correct answer.
- • The answer must be correct with respect to the video visual content.
- • For abnormal object/event questions, include an option stating “The video is normal” as a distractor.
- • Generate 1-4 question-answer pairs.
- • Do not mention people’s reactions to abnormal phenomena.
- • For open-ended questions, provide detailed descriptions including speed and direction of actions/camera movements.

Input: Dense Caption: [dense caption] Output Format: Your output should be formatted as a JSON file: For Multiple Choice Questions:

[{

"Question": "<question-1>", "Options": ["<option-0>", "<option-1>",

"<option-2>", "<option-3>"], "Answer": "index of correct option"

}] For Open-Ended Questions: [{

"Question": "<question-1>", "Answer": "<a detailed answer-1>"

}]

Figure A.2. Counterfactual Video QA Generation Prompt Template

Sum of ℓ1 norm. The sum of ℓ1 norm of Aˆi over the group is:

1 |G| i∈G

|Aˆi|

S =

1 − R R · (1 − R)

1 |G|

|G| · R ·

=

R R · (1 − R)

+ |G| · (1 − R) ·

=2 R · (1 − R)

(B.15)

|Aˆi| =

 

√ 1−R

, if ri = 1, √ R

R·(1−R)

, if ri = 0.



R·(1−R)

(B.14)

###### Real Video Multiple Choice and Open-Ended Question Generation Prompt

Task: Given a detailed description that summarizes the content of video, generate question-answer pairs to build LLM training data. Reference Examples: Question Type: [question type] For Multiple Choice:

## caption-1: [Video description] ## question-1: [Question text] ## options-1: [A. Option1, B. Option2, C. Option3, D. Option4] ## answer-1: [Correct answer]

For Open-Ended:

## caption-1: [Video description] ## question-1: [Question text] ## answer-1: [Detailed answer]

You need to generate similar question-answer pairs like the examples. Guidelines For Question Generation: For Multiple Choice Questions:

- • Generate appropriate multiple-choice question-answer pairs based on the description
- • Each question should have 4 options (A, B, C, D)
- • Only one option should be correct
- • Other options should be plausible distractors
- • Distractor options must be reasonable, relevant to the question, and not obviously wrong For Open-Ended Questions:
- • Generate appropriate question-answer pairs based on the description
- • Answers should be detailed and comprehensive General Guidelines:
- • Generate 1-4 question-answer pairs
- • Questions should focus on observable content in the video
- • Maintain natural and objective question formulation Output Format: For Multiple Choice Questions:

[{

"Question": "<question-1>", "Options": ["<option-0>", "<option-1>",

"<option-2>", "<option-3>"], "Answer": "index of correct option"

}] For Open-Ended Questions: [{

"Question": "<question-1>", "Answer": "<a detailed answer-1>"

}]

Figure A.3. Real Video QA Generation Prompt Template

RL Question Generation Prompt

Task: Given two captions — TRUE CAPTION (original video description) and MOCK CAPTION (edited video description after applying an edit instruction) — design a question that can be answered differently for the TRUE and MOCK videos. The goal is to produce high-quality, dimension-specific question-answer pairs for training multimodal models.

Reference Example: TRUE CAPTION: The man places a cake on the table and lights the candles. MOCK CAPTION: The man places a cake on the table without lighting any candles. Edit Instruction: Remove the candle lighting action. Question: What does the man do with the cake after placing it on the table? Answer for TRUE: He lights the candles on the cake. Answer for MOCK: He leaves the cake as it is without lighting candles. Wrong Answers: [“He cuts the cake into slices”, “He puts the cake back into the oven”] Guidelines for Question Generation: Core Requirements:

- • Base questions strictly on differences between the TRUE and MOCK videos.
- • Do not refer to or mention captions directly in the question.
- • No timestamps or meta-information in the question.
- • Use the provided edit instruction as a design hint.
- • Questions must belong to one of the predefined task dimensions.
- • If no suitable question for the chosen dimension, output an empty question string.
- • Wrong answers must be incorrect for both videos, but still plausible.
- • Generate answers for each video independently without inferring from the other. Available Dimensions: Refer to the predefined TASK EXAMPLES set for dimensions and descriptions. Output Format: The result must be valid JSON with the following structure:

{

"dimension": "<task dimension>", "question": "<generated question>", "answers_for_true_caption": ["<answer based on TRUE CAPTION>"], "answers_for_mock_caption": ["<answer based on MOCK CAPTION>"], "wrong_answers": ["<wrong answer 1>", "<wrong answer 2>", ...]

}

Figure A.4. RL Question Generation Prompt

[Figure 168]

[Figure 169]

Real Video

Q: What visual change happens to the rice paddies later in the video?

CounterFactual Video

Q: What visual change happens to the rice paddies later in the video?

[Figure 170]

- A. They are harvested and become bare brown immediately
- B. Livestock enter and graze across the fields
- C. The video is normal; nothing unusual changes
- D. Their surfaces turn smooth white and ripple like waves

[Figure 171]

[Figure 172]

- A. They are harvested and become bare brown immediately
- B. Livestock enter and graze across the ﬁelds
- C. The video is normal; nothing unusual changes
- D. Their surfaces turn smooth white and ripple like waves

[Figure 173]

[Figure 174]

D. Their surfaces turn smooth white and ripple like waves

[Figure 175]

C. The video is normal; nothing unusual changes

[Figure 176]

[Figure 177]

CounterFactual Video

[Figure 178]

Real Video

[Figure 179]

[Figure 180]

Q: At the beginning of the clip, how does the woman operate the chest press machine?

[Figure 181]

Q: What is the person using to stir the contents of the pot?

[Figure 182]

From 0 to about 1.3 seconds, she sits back against the machine’s pad, grips both handles, and pushes them straight forward in a smooth, controlled motion. The pace is steady with no jerky movements, and the press follows a typical chest-press path from near the torso outward.

The person is using a light-colored wooden spoon to stir the liquid in the pot.

[Figure 183]

[Figure 184]

[Figure 185]

Real Video

[Figure 186]

CounterFactual Video

[Figure 187]

[Figure 188]

[Figure 189]

Q: What are the two women in the background doing?

Q: Describe the motion and shape of the water feature that appears in the background.

- A. Playing with a dog
- B. Exercising
- C. Waiting for a bus
- D. Sitting on a bench

The water erupts upward at high speed in a near-vertical plume, bright white and foamy against the blue sky. The column has a tapered, rocket-like profile: wider at the base where it meets the sea and narrowing toward the top as it climbs. It shows little sideward drift, maintaining a mostly straight, skyward trajectory that dwarfs the gentle swells around it.

[Figure 190]

[Figure 191]

D. Sitting on a bench

###### Figure A.5. Examples of DualityVidQA-SFT. We show the real video and counterfactual video pair and the question and answer pair generated based on the counterfactual video.

[Figure 192]

Real Video

[Figure 193]

CounterFactual Video

Q: What change, if any, occurs to the large rocks in the stream during the video?

[Figure 194]

- A. The large rocks are washed away downstream by the current.
- B. Some of the large rocks begin to float upwards into the air.
- C. The large rocks change color from grey to green.
- D. The large rocks remain stationary in the stream bed throughout the video.

[Figure 195]

Real Video Answer

D. The large rocks remain stationary in the stream bed throughout the video.

[Figure 196]

B. Some of the large rocks begin to ﬂoat upwards into the air.

CF Video Answer

###### Figure A.6. Examples of DualityVidQA-RL. We show the real video and counterfactual video pair and the generated question and answer.

[Figure 197]

[Figure 198]

Real CounterFact

Real CounterFact

[Figure 199]

[Figure 200]

counter physical

Q: What is the boy in the plaid shirt doing in the background while the girl in the orange dress is inflating her balloon?

causal reversal Q: What happens to the lanterns in the video?

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

- A. He is talking to the girl in the white shirt
- B. He is playing with a green balloon on the ground.
- C. He is not visible at first, but then suddenly appears holding a purple balloon.
- D. He is twisting a purple balloon into a shape.

- A. Some of the lanterns float up into the sky.
- B. The lanterns are hanging from balconies and are on boats on the water.
- C. People are seen releasing the lanterns into the sky.
- D. The lanterns fall into the river.

attribute change

object/scene deformation

[Figure 206]

[Figure 207]

Real CounterFact

Real CounterFact

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Q: What is the spatial relationship and movement pattern of the celestial objects, other than the spaceship, seen in the video?

[Figure 213]

Q: What is notable about the attire of the man standing in front of the BMW logo?

- A. He is wearing a plain black suit with no logos on it.
- B. He is not wearing a suit jacket, only a dress shirt and tie.
- C. A glowing BMW logo badge appears on the chest of his suit.
- D. The BMW logo is embroidered on the cuff of his sleeve.

- A. Multiple asteroids move in a square or box-like path around a

central pink planet.

- B. A stationary nebula is visible in the background, which the

spaceship moves away from.

- C. A single planet spins rapidly while multiple moons orbit it in a

circular path.

- D. A black hole is shown pulling in surrounding stars and debris.

###### Figure A.7. Examples of DualityVidQA-Test. We show the real video and counterfactual video pair and the generated question. Answers for the counterfactual video are shown in red, and answers for the real video are shown in green.

