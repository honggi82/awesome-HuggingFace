# arXiv:2511.11113v2[cs.CV]20Apr2026

## VIDEOP2R: Video Understanding from Perception to Reasoning

Yifan Jiang1∗, Yueying Wang2, Rui Zhao2†, Toufiq Parag3∗, Zhimin Chen2 Zhenyu Liao2, Jayakrishnan Unnikrishnan2 1USC, 2Amazon, 3Keystone AI

1yifjia@isi.edu,2{yueyingw,zhaori,zhiminch,zyliao,jayunn}@amazon.com 3toufiq.parag@gmail.com

### Abstract

Reinforcement fine-tuning (RFT), a two-stage framework consisting of supervised fine-tuning (SFT) and reinforcement learning (RL) has shown promising results on improving reasoning ability of large language models (LLMs). Yet extending RFT to large video language models (LVLMs) remains challenging. We propose VIDEOP2R, a novel process-aware video RFT framework that enhances video reasoning by modeling perception and reasoning as distinct processes. In the SFT stage, we develop a three-step pipeline to generate VIDEOP2R-CoT-162K, a high-quality, processaware chain-of-thought (CoT) dataset for perception and reasoning. In the RL stage, we introduce a novel process-aware group relative policy optimization (PAGRPO) algorithm that supplies separate rewards for perception and reasoning. Extensive experiments show that VIDEOP2R achieves state-of-the-art (SotA) performance on six out of seven video reasoning and understanding benchmarks. Ablation studies further confirm the effectiveness of our process-aware modeling and PAGRPO and demonstrate that model’s perception output is information-sufficient for downstream reasoning. Our project page is available at https://videop2r. github.io/videop2r/.

### 1. Introduction

Understanding visual information in video has long been a core challenge in computer vision [15, 32, 48, 49]. Reinforcement fine-tuning (RFT) [34], a two-stage framework of supervised fine-tuning (SFT) followed by reinforcement learning (RL), has recently emerged as a powerful approach for enhancing the reasoning capabilities of large language models (LLMs) [1, 4, 21]. Within the RFT framework, the SFT stage leverages Chainof-Thought (CoT) [59] annotations to warm up mod-

∗Work done while at Amazon. †Corresponding author.

els with explicit guidance, while the RL stage further improves reasoning quality through reward-driven optimization. Motivated by the success of RFT, recent studies [7, 14, 26, 56] have adapted RFT to video reasoning by designing diverse CoT generation pipelines and developing RL variants, with particular attention to Group Relative Policy Optimization (GRPO) [17] for its simple rule-based rewards without a critic model. These adaptations have demonstrated clear advantages over conventional SFT across multiple video benchmarks [14, 52].

However, exploration of adapting RFT from text to video remains preliminary, partly due to simply treating video as another alternative modality to text [50]. This simplicity assumption ignores the decomposability of visual reasoning, which consists of two different processes [43, 66]. The first one is perception, which extracts salient information from the visual input, and reasoning, which organizes the visual evidence and draws inferences. Each process can introduce error in distinct ways (e.g., missed visual evidence, faulty inference), and can impact video understanding [2, 22]. Yet existing video RFT frameworks are process-agnostic: they collapse perception and reasoning into a single procedure and assign a single final reward to the whole trajectory, which blurs credit assignment. As Fig. 1 (top right) shows, a perception error (e.g., turning back) induces the reasoning error, but without process awareness, the entire trajectory is evaluated as incorrect. Therefore, assigning a single reward to the entire process can prevent the model from effectively correcting mistakes that emerge in different processes.

The challenge to extend RFT to explicitly account for different processes in visual reasoning is two-fold. (i) Lack of process-aware CoT data: existing CoT annotations conflate perception and reasoning rather than explicitly disentangling the two; and (ii) Coarse rewards: training typically collapses feedback into a single final reward for the entire reasoning process [14, 52, 56], hindering credit assignment across processes. To address these challenges, we propose VIDEOP2R,

|Question: Are the events occurring in this order: painting a wall while walking, and then turning back? Options: A. Yes, B. No<br><br>[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]|
|---|

Process-agnostic

###### GRPO

<think>Let me think about this for a moment. The question is asking about the sequence of events in the video. First, let's break it down. The person is indeed holding a paint roller and appears to be painting a wall. This is evident from the blue paint streaks on the wall and the person's posture, which suggests they are actively engaged in painting. Now, the question is whether this painting activity is followed by the person turning back. From what I can observe, the person continues to paint the wall while walking. There is no indication of the person stopping to turn back. … Let’s break it down one last time to ensure. The person is painting and walking, and there is no turning back. This is consistent with the video content.</think> <answer>B</answer>

Entire Trajectory Incorrect

###### VIDEOP2R

Process-aware

<observation>Based on the question, I need to notice the sequence of actions performed by the woman in the video. The woman is seen walking while holding a paint roller, and she appears to be painting a wall. After a while, she turns back and continues walking in the direction. </observation> <think>Based on the observations, the woman is indeed painting a wall while walking and then turns back. Let me think, this sequence matches the description in the question. Does that make sense? Yes, it does. The video clearly shows her painting and then turning back, so the events are in the correct order. Therefore, the correct answer should be A.</think> <answer>A</answer>

Perception Correct

Reasoning Correct

Figure 1. Comparison between GRPO-based video RFT framework (process-agnostic) and VIDEOP2R (process-aware).

a novel process-aware video RFT framework that models perception and reasoning as distinct processes to enhance video reasoning. Same as conventional RFT, our framework consists of two stages of training. In the SFT stage, we construct a three-step CoT generation pipeline that produces high-quality perception and reasoning traces. Given visual question-answer (VQA) samples, our pipeline first generates visual perception and reasoning traces in order; the perceptions are then fed to a reasoning-capable LLM [5] to verify whether they contain sufficient visual evidence to reach the correct answer. Running this pipeline on 260K VQA pairs [14] yields VIDEOP2R-CoT-162K after filtering low-quality samples. We use this dataset in SFT to warm up the model, encouraging process separation during inference and providing a strong initialization for subsequent RL. In the RL stage, we propose PA-GRPO, a process-aware variant of GRPO. Unlike GRPO, which assigns a single reward to the entire trajectory, PAGRPO supplies two separate rewards specific for perception and reasoning and assigns them to the corresponding output segments: 1) an LLM-judged [16] perception reward, which evaluates whether the model’s perception captures the necessary information from the video, and 2) a rule-based reasoning reward verifying the correctness of the final answer. As illustrated in Fig. 1, after two-stage training, VIDEOP2R enables LVLMs to conduct process-aware inference with calibrated perception and reasoning.

We conduct comprehensive experiments on seven widely used video understanding and reasoning benchmarks [15, 20, 25, 32, 40, 62, 72], comparing VIDEOP2R with representative process-agnostic video RFT baselines (e.g., Video-R1 [14] and VideoRFT [52]). Results show that VIDEOP2R achieves SotA on six out of seven benchmarks, with robust gains of 1.9%–9.1% average accuracy over base models across benchmarks. Ablation studies further validate the effectiveness of the

process-aware modeling and the PA-GRPO. In addition, we provide fine-grained analysis of VIDEOP2R’s perception and PA-GRPO’s improvements over GRPO to support future process-aware research in the video domain. Our key contributions are: (i) A novel processaware video RFT framework, VIDEOP2R, that models perception and reasoning separately to enhance video reasoning. (ii) A process-aware RL algorithm based on GRPO, PA-GRPO, that provides separate rewards for perception and reasoning, improving credit assignment in RL. (iii) An automatic three-step CoT generation pipeline that produces perception and reasoning annotations, yielding VIDEOP2R-CoT-162K for warm start in SFT. (iv) Comprehensive evaluation confirms VIDEOP2R’s SotA performance, with ablations verifying the effectiveness of process-aware modeling and PAGRPO.

### 2. Related work

Video Perception and Reasoning in LVLMs. Video reasoning poses coupled spatial–temporal challenges beyond text/image only settings [15, 32, 48], demanding coordinated perception and reasoning [43, 66]. Early approaches handle this by introducing modular perception preprocessors (e.g., frame captioners [53, 57, 68] or spatio-temporal scene graphs [27, 47, 61]) to aid downstream reasoning. However, because these preprocessors are typically frozen, they bottleneck the pipeline and cannot be improved to curb error propagation or information loss [50]. Recent methods organize perception–reasoning into pre-defined stages, either by attributes (frames/objects/actions) [13, 18, 37, 41, 55, 58] or steps (planning/grounding) [33, 45]. However, rigid designs often limit generalization across scenarios [52], and mixing perception and reasoning at each stage makes it vulnerable to perception errors [2, 22]. In contrast, VIDEOP2R separates perception and reasoning,

###### Three-Step CoT Generation Pipeline

###### Image&Video QA Dataset (~260k)

[Figure 7]

###### (1)Process-aware CoT Generation

Question: What distinctive feature is seen on the hand that places the lid on the pot? Options: A. A wristwatch, B. A gold ring, C. A tattoo, D. A silver bracelet

Perception

[Figure 8]

[Figure 9]

[Figure 10]

| |
|---|

<observation>Based on the question, I need to notice any distinctive features on the hand, such as the gesture or object being held. Observing the video, I can see that there is a gold ring on the ring ﬁnger of the hand that places the lid on the pot. There are no visible wristwatches, tattoos, or silver bracelets on the hand.</observation>

Reasoning

<think>Based on the observations, the distinctive feature on the hand that places the lid on the pot is a gold ring. Let me think, the question asks for a speciﬁc feature, and the gold ring is clearly visible. Does that make sense? Yes, it does. Let me double-check the other options; there are no wristwatches, tattoos, or silver bracelets. Therefore, the correct answer should be the one that matches the gold ring.</think> <answer>B</answer>

###### SFT Stage

Qwen2.5-VL-7B

###### VIDEOP2R-CoT-162K

[Figure 11]

Image&Video QA Process-aware CoT

[Figure 12]

<observation>… </observation> <think>…</think> <answer>…</answer>

###### (2)CoT Verification

VIDEOP2R-SFT

<observation>… </observation> <think>…</think> <answer>…</answer>

Format check Quality check

|RL Stage| | |
|---|---|---|
|[Figure 13]<br><br>Image&Video QA|PA-GRPO Process-Aware Reward| |

###### R

I

(3)Observation Sufficiency Verification

For {question}, can the {answer} be deduced from {perception}?

[Figure 14]

[Figure 15]

<observation>… </observation> <think>…</think> <answer>…</answer>

[Figure 16]

Policy Model VIDEOP2R

Based on the question … there is a gold ring on the ring ﬁnger …

Figure 2. Illustration of overall VIDEOP2R RFT framework (left) and the three-step CoT generation pipeline (right).

aligning with the two-process modeling of video reasoning [43] for better generalization, while preserving an end-to-end training pipeline for continual refinement.

Reinforcement Learning for LVLMs. Building on GRPO, recent work adapts RL/RFT to video understanding and reasoning [7, 14, 26, 36, 39, 52, 56, 71]. Time-R1 uses timestamp-aware and template rewards [56]; Video-R1 and STAR-R1 reward sensitivity to correct temporal order [14, 28]; Videochat-R1 and VersaVid-R1 adopt task-specific rewards [7, 26]; VideoRFT adds a stage-aware semantic reward [52]. However, most prior efforts model video reasoning as a single trajectory and apply a single final reward over the entire output sequence, with no process-aware distinction between perception and reasoning. This also holds for recent image-domain studies on perception–reasoning separation [29, 60, 64]. VIDEOP2R instead uses PAGRPO to assign separate perception and reasoning rewards to their token segments, providing more clear signals and more precise error attribution.

### 3. VIDEOP2R RFT Framework

In this section, we introduce the overall design of the VIDEOP2R RFT framework (Fig. 2 left), which follows the standard RFT setup [34] with a specific focus on modeling video reasoning into perception and reasoning: (1) SFT stage, we use a three-step CoT generation pipeline to construct a process-aware CoT dataset, VIDEOP2R-CoT-162K. We train the base model on this dataset to enhance its perception and reasoning ca-

pabilities while warming up the model for the RL stage. (2) RL stage, we propose a process-aware reinforcement learning scheme, PA-GRPO, which refines the model’s reasoning by providing separate rewards for perception and reasoning, enabling the model to move beyond supervised learning boundaries [38].

#### 3.1. Process-aware CoT Dataset

To address the challenge of lacking process-aware CoT dataset, we develop a strategy to curate CoT data at scale, which is then used for fine-tuning LVLM in SFT. Particularly, we first standardize a processaware CoT template that explicitly disentangles perception from reasoning in different segments as follows:

<observation>...</observation> <think>...</think><answer>...</answer>

⟨observation⟩ segment represents the perception process, where the model extracts relevant visual evidence based on the question. ⟨think⟩ and ⟨answer⟩ segment captures the reasoning process, where the model reasons (⟨think⟩) over the extracted visual evidence and states the final answer (⟨answer⟩). All generated CoTs follow this template, and the model is trained to adhere to it at inference.

##### 3.1.1. Three-Step CoT Generation Pipeline

Building upon the proposed template, we design a threestep pipeline (Fig. 2 right) to generate process-aware CoT data containing both perception and reasoning traces for a diverse set of VQA tasks. The overall workflow comprises three major steps (Details in the Supple-

mentary). 1) Process-aware CoT Generation. For each VQA sample, we use Qwen2.5-VL-72B-Instruct [4] to generate an initial CoT trace for both perception and reasoning in corresponding segments. 2) CoT Verification. To ensure consistency and correctness, we evaluate the final answer of each generated response with taskspecific metrics (e.g., exact word match and word error rate for generation tasks), discarding samples that yield low-quality answers or deviate from the expected CoT template. 3) Observation Sufficiency Verification. We further filter generated data using a cross-modal validation strategy to validate the ⟨observation⟩ segment in isolation from raw visual inputs [52]. Specifically, for each sample, we provide only the ⟨observation⟩ segment, along with the corresponding question and answer, to Claude 3.7 Sonnet [3], which assesses whether the visual evidence in ⟨observation⟩ are adequate to support the final reasoning process.

##### 3.1.2. Data Construction and Statistics

To ensure reproducibility and fair comparison, we apply this pipeline to a public and representative image- and video-based QA dataset [14], which includes multiple question types such as multiple-choice, numerical QA, Optical Character Recognition (OCR), free-form QA, and regression. Applying our pipeline on 260K VQA data [14] produces 162K high-quality process-aware CoT data with perception and reasoning traces, termed VIDEOP2R-CoT-162K. We provide detailed analyses of VIDEOP2R-CoT-162K (e.g., embedding visualization and word-frequency statistics) in the Supplementary, which show that our annotations inherently separate perception from reasoning.

#### 3.2. Process-Aware Reinforcement Learning

After the SFT stage, we further refine the model through reinforcement learning. Building upon GRPO [17], we introduce a process-aware variant, PA-GRPO (Fig. 3), which provides separate reward signals for perception and reasoning processes to encourage more structured and efficient policy optimization [30]. This section first revisits the standard GRPO framework and then presents our process-aware extension designed to align reward signals with perception–reasoning separation.

##### 3.2.1. Group Relative Policy Optimization (GRPO)

GRPO [17] extends Proximal Policy Optimization [42] by removing the dependency on a learned critic model and directly comparing responses within groups. Given a question q, the policy model πθ samples G candidate responses o = {o1,o2,...,oG} as a group, each assigned a rule-based reward ri. Rewards are then normalized within the group to yield the relative advantage:

Question: Are the events occurring in the order of Painting a wall while walking, then turning back? Options: A. Yes, B. No

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

<observation>Based on the question, I need to notice the sequence of actions performed by the woman in the video. The woman is seen walking while holding a paint roller and a paint can, and she appears to be painting a wall. After a while, she turns back and continues walking in the opposite direction. </observation> <think>Based on the observations, the woman is indeed painting a wall while walking and then turns back. Let me think, this sequence matches the description in the question. Does that make sense? Yes, it does. The video clearly shows her painting and then turning back, so the events are in the correct order. Therefore, the correct answer should be A.</think> <answer>A</answer>

Perception Reward

Accuracy Format

Question

A1,P

A1,R

r1,P

r1,R

Length

- oG,P

- o1,P

- oG,R

- o1,R

Process-Aware Advantage Computation

A2,P

A2,R

r2,P

r2,R

Policy

Model o…

o2,R

…

…

…

…

2,P

rG,R

AG,P

AG,R

rG,P

Reasoning Reward

Reference Rewards Advantages Model

Accuracy Format

Group Outputs

Length

Figure 3. The illustration of the PA-GRPO algorithm.

ri − mean({rj}Gj=1) std({rj}Gj=1)

. (1)

Ai =

where Ai represents the relative advantage of all tokens in the i-th response within the group. With the relative advantage computed, the GRPO overall optimization objective is formulated as:

G

1 G

JGRPO(θ) = Eq,{o

min(ρiAi, clip ρi, 1−ϵ, 1+ϵ Ai − β DKL πθ ∥πref . (2)

i}

i=1

θ(oi|q)

where ρi = π

πθold(oi|q) is the likelihood ratio between updated policy πθ and old policy πθ

, and πref is a fixed reference model (e.g., a frozen copy of policy model) after SFT, providing KL regularization weighted β. This formulation constrains large policy deviations while promoting high-reward samples, ensuring stable optimization during reinforcement learning.

old

3.2.2. Process-Aware Group Relative Policy Optimization (PA-GRPO)

While GRPO performs well on textual reasoning, its single scalar reward provides a limited training signal for the two-process based visual reasoning [43, 66]. To address this, we propose PA-GRPO, which introduces separate perception and reasoning rewards for tokens in each process, enabling fine-grained credit assignment during reinforcement learning, as illustrated in Fig. 3. Formally, for each question q, the sampled response from the policy model πθ is represented as o = {(o1,P,o1,R),(o2,P,o2,R),...,(oG,P,oG,R)}, where oi,P denotes perception process tokens within the

⟨observation⟩ segment, and oi,R denotes reasoning process tokens within the ⟨think⟩ and ⟨answer⟩ segments. For tokens within each process, PA-GRPO supplies separate accuracy rewards to provide reliable supervision. We further introduce length and format rewards to encourage well-structured, concise outputs, following prior RFT frameworks [14, 17, 26, 52]. We demonstrate the accuracy reward design, and our configurations for the format and length rewards below.

Perception Accuracy Reward (Racc,P). The perception accuracy reward evaluates whether the model correctly perceives visual information from video input. We adopt an LLM-as-Judge evaluation [16] in a textonly setting, a procedure shown to be reliable in various scenarios [9, 12]. Concretely, we provide only the ⟨observation⟩ segment, along with the corresponding question and answer, to Claude 3.7 Sonnet [3], which judges whether the ⟨observation⟩ segment is sufficient to supcont the correct answer. Formally,

Racc,P = 1 (judged sufficient); 0 otherwise. (3)

Reasoning Accuracy Reward (Racc,R). The reasoning accuracy reward evaluates whether the model produces accurate reasoning outcomes. We apply taskspecific evaluation metrics to accommodate different question types, including exact word match for categorical tasks, ROUGE-based similarity for open-ended generation, and error-based scores for numerical or regression problems. Formally,

Racc,R = Acct(oi,R,ytrue), (4)

where Acct(·) ∈ [0,1] denotes the task-specific accuracy metric for task type t.

Format Reward (Rform) and Length Reward (Rlen). To ensure clear reward assignment, the format and the length reward in PA-GRPO are also provided separately for each process (oi,P and oi,R). We use the format reward (Rform) to encourage adherence to the process-aware inference template. Specifically, perception process tokens (oi,P) must appear within the ⟨observation⟩, the reasoning process tokens (oi,R) must present the reasoning trace under ⟨think⟩, and the final answer must be provided within ⟨answer⟩. We verify compliance using regular expression matching and assign a binary reward ({0,1}) accordingly. The length reward (Rlen) is included to favor concise yet informative responses while avoiding overthinking. The reward is assigned only if both accuracy and format rewards are non-zero and the model’s response in each segment (oi,P and oi,R) falls within the target length ([lmin, lmax]). In line with prior RFT work [14, 52], we fix Rlen = 0.2, lmin = 128 and lmax = 320 for oi,P, lmin = 320 and lmax = 512 for oi,R.

##### 3.2.3. Process-Aware Reward Assignment

The overall reward of perception tokens (oi,P) and reasoning tokens (oi,R) are defined as

Ri,k = Racc,k + Rform,k + Rlen,k, k ∈ {P,R}. (5)

Unlike Eq. (1), which normalizes all rewards within a single group, we split rewards into separate groups (perception process vs. reasoning process) and normalize each to get process-aware advantage, since their scales and distributions are not directly comparable [67]:

Ri,k − mean({Rj,k}Gj=1) std({Rj,k}Gj=1)

, k ∈ {P,R}. (6)

Ai,k =

We assign each process-aware advantage only to its corresponding tokens (e.g., the perception advantage Ai,P is applied to oi,P). The overall optimization objective of PA-GRPO is then formulated as:

  1

G

JPA-GRPO(θ) = Eq,{o

min(ρi,kAi,k,

i}

G

i=1 k∈{P,R}

clip ρi,k, 1−ϵ, 1+ϵ Ai,k − β DKL πθ ∥πref , (7)

πθ(oi,k | q) πθ

where ρi,k =

denotes the likelihood ratio

(oi,k | q)

old

for perception (k=P) or reasoning (k=R), and Ai,k is the process-aware advantage.

### 4. Experiment Setup

Two-stage Training. We adopt Qwen2.5-VL-7BInstruct [4] as the base LVLM in our training pipeline. Following the same training setups used in prior videoRFT studies [14, 52], we perform one epoch of SFT on VIDEOP2R-CoT-162K, followed by 1K RL updates over original 260K visual QA data using PA-GRPO. The model obtained after the SFT stage is referred to as VIDEOP2R-SFT, and the final model after the RL stage is denoted as VIDEOP2R.

Benchmarks. Following prior works [15, 52], we evaluate our approach on seven benchmarks, including four video reasoning datasets (VSI-Bench [62], VideoMMMU [20], MMVU [72], and VCR-Bench [40]) and three video understanding datasets (MVBench [25], TempCompass [32], and VideoMME [15]). These benchmarks jointly cover spatial reasoning, knowledgeintensive QA, temporal logic, and general video understanding. We follow the official evaluation protocols of each benchmark.

Baselines. (1) RFT on Qwen2.5-VL-7B. We compare against recent video RFT approaches built upon Qwen2.5-VL-7B using GRPO or its variants, including Video-R1 [14], Time-R1 [56], VideoRFT [52],

- Table 1. Performance comparison on video reasoning and understanding benchmarks. Best/second-best result of each column is in bold/underline. Missing entries indicate unreported results (all numbers unit in %).

Video Reasoning Video Understanding

Model

Avg VSI. VideoMMMU MMVU VCR. MV. TempCom. VideoMME

###### Open-Source 7B Models

LLaVA-OneVision-7B 32.4 33.8 49.2 – 56.7 – 58.2 – LongVA-7B 29.2 23.9 – – – 56.9 52.6 – Video-UTR-7B – – – – 58.8 59.7 52.6 – VideoLLaMA2-7B – – 44.8 – 54.6 – 47.9 – Qwen2.5-VL-7B 30.1 48.1 60.0 44.3 59.0 72.6 56.6 52.9

###### RFT on Qwen2.5-VL-7B

Video-R1 35.8 52.3 63.8 49.0 63.9 73.2 59.3 56.8 VideoChat-R1 33.9 54.0 63.0 49.0 67.9 72.5 57.7 56.9 Time-R1 29.0 51.0 62.9 49.6 63.1 73.7 59.3 55.5 VersaVid-R1 33.7 51.9 64.3 49.8 62.9 74.0 58.8 56.5 VideoRFT 36.8 51.1 68.5 49.6 62.1 73.7 59.8 57.4

VIDEOP2R (Ours) 36.8 55.0 65.4 51.0 68.1 74.5 60.0 58.7

VideoChat-R1 [26], and VersaVid-R1 [7] (2) OpenSource Models. We further include Qwen2.5-VL-7B along with other 7B-scale models for a comprehensive evaluation: LLaVA-OneVision [24], LongVA [69], Video-UTR [63], and VideoLLaMA2 [8]. We follow the prompt templates in each baseline’s official publication. We provide all setup details in the Supplementary.

### 5. Results

Our experiments focus on addressing five research questions from Section 5.1 to 5.5 : (1) How does VIDEOP2R perform across various video understanding benchmarks? (2) What is the contribution of each component in VIDEOP2R? (3) Does the perception representations of VIDEOP2R effectively support downstream reasoning? (4) Does the process-aware reward design of PAGRPO improve RL efficiency and reliability? (5) What are the success and failure mode of VIDEOP2R?

#### 5.1. Main Results

The main evaluation results of VIDEOP2R and other baselines are shown in Tab. 1. Compared with prior video RFT approaches, VIDEOP2R achieves highly competitive performance across seven benchmarks, setting SotA results on six of them and ranking second on the remaining one. In contrast to previous RFT methods that often bring improvement on specific datasets (e.g., Video-R1 ranks second on VSI-Bench, while VideoRFT is SotA on MMVU but last on MVBench), VIDEOP2R delivers consistent gains across all benchmarks, surpassing the previous SotA by 1.3% in average accuracy. This consistency underscores the effectiveness and generalizability of modeling visual reasoning through distinct perception and reasoning processes. Compared to the base model Qwen2.5-VL, VIDEOP2R exhibits clear

and steady improvements, with accuracy gains ranging from 1.9% to 9.1% across benchmarks. More broadly, a performance gap exists between models trained with RFT and those trained only with SFT/Instruction Tuning [31] (i.e., open-source models), highlighting the superiority of RFT for expanding capability boundaries. We further analyze VIDEOP2R’s performance drop on MMVU (Sec. 5.5) and attribute it to the lack of domainspecific knowledge (e.g., chemistry) in our training data.

#### 5.2. Ablation Study

The success of VIDEOP2R underscores the importance of decomposing visual reasoning into distinct process stages. To further analyze the contribution of each process-aware component, we perform an ablation study on the two-stage training in the RFT framework, process-aware modeling, and reward design (Tab. 2). Two-stage Training: SFT-only and RL-only improves the baseline by 2.7% and 3.1% respectively. But combining both yield a more significant 5.8% improvement, suggesting that single-stage training is insufficient, whereas using both stages can further extend the model’s capability.

Process-aware Modeling: We evaluate process-aware modeling against a process-agnostic counterpart in both SFT and RL. In SFT, the process-aware variant (same as VIDEOP2R-SFT) yields an average accuracy gain of 2.1% over the process-agnostic variant, which uses reasoning-only segments. In RL, the agnostic variant follows GRPO—assigning a single reasoning reward across all output tokens, while the process-aware variant (same as VIDEOP2R) again leads on six benchmarks with 2.3% on average. Under identical visual inputs, adding perception annotations in SFT and using processaware credit assignment in RL provide clearer training

- Table 2. Ablation studies of VIDEOP2R on two-stage training, process-aware modeling and reward design (all numbers unit in %).

Ablation Factor

Video Reasoning Video Understanding

Model Variant

Avg. VSI. VideoMMMU MMVU VCR. MV. TempCom. VideoMME

Two-stage Training

- - VIDEOP2R (Ours) 36.8 55.0 65.4 51.0 68.1 74.5 60.0 58.7+5.8

- - SFT-only (VIDEOP2R-SFT) 35.2 53.7 61.6 46.9 62.3 72.4 57.2 55.6+2.7

- - RL-only 35.8 54.6 64.6 46.3 60.8 73.8 55.9 56.0+3.1 Process-aware Modeling

- - VIDEOP2R (Ours) 36.8 55.0 65.4 51.0 68.1 74.5 60.0 58.7+5.8

- - process-agnostic RL (GRPO) 37.4 53.6 62.8 48.3 63.8 73.3 55.4 56.4+3.5

- - process-aware SFT (no RL) 35.2 53.7 61.6 46.9 62.3 72.4 57.2 55.6+2.7

- - process-agnostic SFT (no RL) 34.3 48.9 61.6 47.3 59.0 69.7 54.0 53.5+0.6 Reward Design

- - VIDEOP2R (Ours) 36.8 55.0 65.4 51.0 68.1 74.5 60.0 58.7+5.8

- - without RR 36.0 51.6 60.3 46.8 62.1 72.5 57.9 55.3+2.4

- - without RP 37.4 53.6 62.8 48.3 63.8 73.3 55.4 56.4+3.5

- - without RL 40.0 52.7 63.2 48.4 65.5 73.9 60.0 57.7+4.8

- - without separation 37.1 53.2 64.9 48.8 65.0 73.2 59.7 57.4+4.5 Baseline: Qwen2.5-VL-7B 30.1 48.1 60.0 44.3 59.0 72.6 56.6 52.9

signals, improving video understanding and reasoning.

Reward design: We examine the PA-GRPO reward function by ablating each reward component and the separation reward assignment. Removing any component causes a notable drop, sometimes even below the SFT baseline (e.g., removing the perception reward yields worse performance than SFT on VideoMME). This indicates that all components are necessary and that their joint design in PA-GRPO enables fine-grained credit assignment in RL. When we remove separation and assign both perception and reasoning rewards to all output tokens, the results remain competitive but still lag behind PA-GRPO, indicating that separation reward assignment provides clearer and more effective training signals. Notably, removing Rlen improves performance on VSI-Bench. Further analysis (Supplementary) indicates that its questions often require long, fine-grained descriptions to ensure sufficient perception, where the length reward becomes counter-productive. We propose to have a dynamic length reward [51] in future work. Finally, we further ablate different judge models in the Supplementary for perception accuracy reward assignment, and find that VIDEOP2R remains robust.

#### 5.3. Effectiveness of Perception Representations

The ablation study highlights the effectiveness of process-aware modeling in VIDEOP2R. In this section, we further examine whether the perception outputs produced by VIDEOP2R can enhance the reasoning ability of generic LVLMs (Examination details in the Supplementary). Specifically, we compare Qwen2.5-VL7B’s zero-shot performance on (i) text-only questions, (ii) text plus video inputs. For each scenario, we further augment question text with a perception segment gener-

100

Qwen2.5-VL(7B)

+Per.(Qwen)

+Per.(VIDEOP2R)

| |
|---|

| |
|---|

80

AverageScore

60

54.9 56.0

55.5

52.9

48.6

40

35.1

20

0

Video + Text Text

Figure 4. Effect of perception on downstream reasoning

ated either by VIDEOP2R or by pre-trained Qwen. The results, as shown in Fig. 4, yield two key insights. First, when Qwen2.5-VL-7B is provided only with text questions augmented by VIDEOP2R’s perceptions segments, its performance (55.5%) even surpasses that under raw video input (52.9%), indicating that the textual perceptions generated by VIDEOP2R capture semantically rich perceptual information that directly supports reasoning. Second, in both text and video modality, when we augment the prompt with a perception segment, using VIDEOP2R’s perceptions consistently outperforms using Qwen’s own perceptions, indicating that VIDEOP2R’s perceptions provide more accurate and useful evidence for downstream reasoning.

#### 5.4. Advantages of PA-GRPO over GRPO

Figure 6(a-b) illustrates the training dynamics of VIDEOP2R. Both perception and reasoning accuracy re-

|Question: Assume that 2.24 liters of gas fully participates in the reaction shown in the video under the standard temperature and pressure condition, how many grams of precipitate are produced approximately? Options: A. 10.0, B. 5.0, C. 12.0, D. 15.0, E. 20.0<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|---|

|Question: Why does the little monkey want to take back the bone? Options: A. Regarded it as the owner's toy., B. This is his own food.,<br><br>C. Mistook it for a dinosaur fossil., D. This is food for his pet dog.<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]|
|---|

<observation>Based on the question, I need to notice the interaction between the little monkey and the bone. The monkey is seen holding the bone and appears to be examining it closely with a magnifying glass, suggesting that he might have mistaken it for something of interest. The bone is also shown in a context that seems to be part of a dinosaur exhibit, which might have inﬂuenced the monkey's perception.</observation> <think>Based on the observations, the monkey seems to have mistaken the bone for a dinosaur fossil. He is examining it closely with a magnifying glass, which is a common way to inspect fossils. The setting also includes a dinosaur exhibit, which further supports this idea. Let me think, if the monkey had mistaken it for a dinosaur fossil, he would likely want to take it back to the exhibit. Does that make sense? Yes, it does. Let me double-check the options, and C seems to be the most ﬁtting answer.</think> <answer>C</answer>

<observation>Based on the question, I need to notice the volume of gas in the ﬂask and the relationship between the gas volume and the precipitate produced. The video shows a person holding a ﬂask with a volume marked on it. The person is blowing air into the ﬂask through the glass tube. </observation> <think>Based on the observations, the volume of gas in the ﬂask is 2.24 liters, which is the amount that fully participates in the reaction under standard temperature and pressure conditions. I need to use the stoichiometry of the reaction. The reaction between the gas and the solution in the ﬂask produces a precipitate. The molar mass of the precipitate can be calculated using the volume of gas and the molar volume of a gas at standard temperature and pressure (44.8 liters/mol). The molar mass of the precipitate is approximately 10 grams/mol. Therefore, the grams of precipitate produced are 2.24 liters / 44.8 liters/mol * 100 grams/mol = 5 grams. … so the grams of precipitate produced are 5 grams. </think> <answer>B</answer>

Figure 5. Success (Left) and Failure (Right) case of VIDEOP2R. Correct statement and incorrect statement are colored.

(a) Perception Reward (b) Reasoning Reward

[Figure 29]

[Figure 30]

(c) Number of Equal-Reward Group (d) Think-Answer Mismatch

[Figure 31]

[Figure 32]

- Figure 6. Training Dynamics and Think-Answer Mismatch Analysis of VIDEOP2R. Details in Section 5.4.

wards exhibit an increasing trend, which, combined with the quantitative gains in Tab. 2, confirms that PA-GRPO is more effective than standard GRPO in improving the model’s perception and reasoning traces. In the following, we further analyze the underlying reasons for PAGRPO’s advantages over standard GRPO. (1) Training Efficiency. In advantage-based policy gradient methods [17, 42, 54], including GRPO, when all sampled responses (o1,o2,...,oG) for a given prompt receive nearly identical rewards, this leads to advantage collapse [70], where the advantages shrink toward zero, leaving little to no effective learning signal and causing updates to stagnate [65]. PA-GRPO mitigates this by decomposing the overall reward into perception and

reasoning components, so even when reasoning rewards saturate, perception rewards can still provide non-zero gradients. We visualize the number of samples in a batch with advantage collapse for PA-GRPO and GRPO during the RL stage in Fig. 6(c). Compared with GRPO, PA-GRPO consistently exhibits fewer advantage collapse samples, indicating better utilization of training samples and improved training efficiency.

(2) Mitigating Think–Answer Mismatch. Reasoningaugmented models often exhibit Think–Answer Mismatch [44], where generated reasoning traces diverge from the actual decision process yet still produce correct answers (e.g., “the man is on the right side, thus the answer is B. left”). In GRPO, such inconsistencies can lead to reward hacking [10, 46], as a single final reward can reinforce unfaithful reasoning traces that coincidentally yield correct outcomes. To quantify this issue, we perform an alignment check (Details in Supplementary) using Claude 3.7 Sonnet to extract answers from the ⟨think⟩ segments and compare them with the final output answer. Figure 6(d) reports mismatch rates across VIDEOP2R and two single-reward trained models: while all SFT models maintain stable reasoning consistency (≤5%), both Video-R1 and VideoRFT’s RL models show significant degradation (≥16%). In contrast, VIDEOP2R shows notably lower mismatch, indicating that PA-GRPO’s process-aware rewards, which separately encourage faithful perception traces and correct final answers, effectively mitigate Think–Answer Mismatch and promote more reliable reasoning.

#### 5.5. Qualitative Results of VIDEOP2R

We present one success and one failure case of VIDEOP2R in Fig. 5 to illustrate both its strengths and areas of improvement (More examples in the Supple-

mentary). The left example shows an Aha Moment [17], where VIDEOP2R performs process-aware inference by accurately describing visual cues, such as the monkey’s actions and its imagination, and reasoning over them to reach the correct answer. In contrast, the right example depicts a failure case: although the model correctly identifies relevant visual details about the person and her behavior, it produces an incorrect conclusion due to missing domain-specific knowledge (e.g., the molar volume of a gas should be 22.4). Overall, while VIDEOP2R exhibits strong capabilities in general video understanding and reasoning, its performance can be further improved by injecting factual and domain-specific knowledge.

### 6. Conclusion

In this work, we introduced VIDEOP2R, a processaware RFT framework that models perception and reasoning as distinct processes for video understanding. Through a three-step CoT generation pipeline, we constructed VIDEOP2R-CoT-162K, a large-scale processaware dataset enabling fine-grained supervision in the SFT stage. In the RL stage, we proposed PA-GRPO, a process-aware extension of GRPO that provides separate rewards for perception and reasoning to improve credit assignment. Experiments across seven benchmarks demonstrate SotA performance and strong generalization, while ablations verify the effectiveness of process-aware modeling and PA-GRPO.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1
- [2] Kian Ahrabian, Zhivar Sourati, Kexuan Sun, Jiarui Zhang, Yifan Jiang, Fred Morstatter, and Jay Pujara. The curious case of nonverbal abstract reasoning with multi-modal large language models. arXiv preprint arXiv:2401.12117, 2024. 1, 2
- [3] Anthropic. Claude 3, 2024. 4, 5
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 4, 5
- [5] Maciej Besta, Julia Barth, Eric Schreiber, Ales Kubicek, Afonso Catarino, Robert Gerstenberger, Piotr Nyczyk, Patrick Iff, Yueling Li, Sam Houliston, et al. Reasoning language models: A blueprint. arXiv preprint arXiv:2501.11223, 2025. 2
- [6] Mustafa Chasmai, Gauri Jagatap, Gouthaman KV, Grant Van Horn, Subhransu Maji, and Andrea Fanelli. Moment sampling in video llms for long-form video qa. arXiv preprint arXiv:2507.00033, 2025. 6
- [7] Xinlong Chen, Yuanxing Zhang, Yushuo Guan, Bohan Zeng, Yang Shi, Sihan Yang, Pengfei Wan, Qiang Liu, Liang Wang, and Tieniu Tan. Versavid-r1: A versatile video understanding and reasoning model from question answering to captioning tasks. arXiv preprint arXiv:2506.09079, 2025. 1, 3, 6
- [8] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 6
- [9] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E Gonzalez, et al. Chatbot arena: An open platform for evaluating llms by human preference. In Forty-first International Conference on Machine Learning, 2024. 5
- [10] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. 8
- [11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024. 4
- [12] Yann Dubois, Bal´azs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024. 5
- [13] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Meishan Zhang, Mong Li Lee, and Wynne Hsu. Video-

- of-thought: step-by-step video reasoning from perception to cognition. In Proceedings of the 41st International Conference on Machine Learning, pages 13109– 13125, 2024. 2
- [14] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025. 1, 2, 3, 4, 5, 8
- [15] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multimodal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 1, 2, 5, 3, 7
- [16] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024. 2, 5
- [17] Daya Guo and et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 4, 5, 8, 9
- [18] Songhao Han, Wei Huang, Hairong Shi, Le Zhuo, Xiu Su, Shifeng Zhang, Xu Zhou, Xiaojuan Qi, Yue Liao, and Si Liu. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26181–26191,

2025. 2

- [19] Wei Han, Hui Chen, Min-Yen Kan, and Soujanya Poria. Self-adaptive sampling for accurate video question answering on image text models. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2522–2534, 2024. 6
- [20] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826,

2025. 2, 5, 3, 7

- [21] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 1
- [22] Yifan Jiang, Kexuan Sun, Zhivar Sourati, Kian Ahrabian, Kaixin Ma, Filip Ilievski, Jay Pujara, et al. Marvel: Multidimensional abstraction and reasoning through visual evaluation and learning. Advances in Neural Information Processing Systems, 37:46567–46592, 2024. 1, 2
- [23] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 3
- [24] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei

- Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 6
- [25] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 2, 5, 3
- [26] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025. 1, 3, 5, 6
- [27] Yiming Li, Xiaoshan Yang, Bing-Kun Bao, and Changsheng Xu. Graph prompts: Adapting video graph for video question answering. 2025. 2
- [28] Zongzhao Li, Zongyang Ma, Mingze Li, Songyou Li, Yu Rong, Tingyang Xu, Ziqi Zhang, Deli Zhao, and Wenbing Huang. Star-r1: Spatial transformation reasoning by reinforcing multimodal llms. arXiv preprint arXiv:2505.15804, 2025. 3
- [29] Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding visionlanguage model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025. 3
- [30] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023. 4
- [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 6
- [32] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics ACL 2024, pages 8731–8772, 2024. 1, 2, 5, 3
- [33] Ye Liu, Kevin Qinghong Lin, Chang Wen Chen, and Mike Zheng Shou. Videomind: A chain-of-lora agent for long video reasoning. arXiv preprint arXiv:2503.13444,

2025. 2

- [34] Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. Reft: Reasoning with reinforced fine-tuning. arXiv preprint arXiv:2401.08967,

2024. 1, 3

- [35] Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018. 2
- [36] Desen Meng, Rui Huang, Zhilin Dai, Xinhao Li, Yifan Xu, Jun Zhang, Zhenpeng Huang, Meng Zhang, Lingshu Zhang, Yi Liu, et al. Videocap-r1: Enhancing mllms for video captioning via structured thinking. arXiv preprint arXiv:2506.01725, 2025. 3
- [37] Juhong Min, Shyamal Buch, Arsha Nagrani, Minsu Cho, and Cordelia Schmid. Morevqa: Exploring modular rea-

- soning models for video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13235–13245, 2024. 2
- [38] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022. 3
- [39] Jinyoung Park, Jeehye Na, Jinyoung Kim, and Hyunwoo J Kim. Deepvideo-r1: Video reinforcement finetuning via difficulty-aware regressive grpo. arXiv preprint arXiv:2506.07464, 2025. 3
- [40] Yukun Qi, Yiming Zhao, Yu Zeng, Xikun Bao, Wenxuan Huang, Lin Chen, Zehui Chen, Jie Zhao, Zhongang Qi, and Feng Zhao. Vcr-bench: A comprehensive evaluation framework for video chain-of-thought reasoning. arXiv preprint arXiv:2504.07956, 2025. 2, 5, 3
- [41] Haiyi Qiu, Minghe Gao, Long Qian, Kaihang Pan, Qifan Yu, Juncheng Li, Wenjie Wang, Siliang Tang, Yueting Zhuang, and Tat-Seng Chua. Step: Enhancing videollms’ compositional reasoning by spatio-temporal graphguided self-training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3284– 3294, 2025. 2
- [42] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 4, 8
- [43] Ramprasaath R Selvaraju, Purva Tendulkar, Devi Parikh, Eric Horvitz, Marco Tulio Ribeiro, Besmira Nushi, and Ece Kamar. Squinting at vqa models: Introspecting vqa models with sub-questions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10003–10011, 2020. 1, 2, 3, 4
- [44] Si Shen, Peijun Shen, Wenhua Zhao, and Danhao Zhu. Mitigating think-answer mismatch in llm reasoning through noise-aware advantage reweighting. arXiv preprint arXiv:2508.05928, 2025. 8
- [45] Yudi Shi, Shangzhe Di, Qirui Chen, and Weidi Xie. Enhancing video-llm reasoning via agent-of-thoughts distillation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8523–8533, 2025. 2
- [46] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460–9471, 2022. 8
- [47] Zihan Song, Xin Wang, Zi Qian, Hong Chen, Longtao Huang, Hui Xue, and Wenwu Zhu. Modularized selfreflected video reasoner for multimodal llm with application to video question answering. In Forty-second International Conference on Machine Learning. 2
- [48] Elizabeth S Spelke and Katherine D Kinzler. Core knowledge. Developmental science, 10(1):89–96, 2007. 1, 2
- [49] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large lan-

- guage models: A survey. IEEE Transactions on Circuits and Systems for Video Technology, 2025. 1
- [50] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024. 1, 2
- [51] Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, et al. Srpo: Enhancing multimodal llm reasoning via reflection-aware reinforcement learning. arXiv preprint arXiv:2506.01713, 2025. 7
- [52] Qi Wang, Yanrui Yu, Ye Yuan, Rui Mao, and Tianfei Zhou. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. arXiv preprint arXiv:2505.12434, 2025. 1, 2, 3, 4, 5
- [53] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision, pages 58–76. Springer,

2024. 2

- [54] Yuhui Wang, Hao He, and Xiaoyang Tan. Truly proximal policy optimization. In Uncertainty in artificial intelligence, pages 113–122. PMLR, 2020. 8
- [55] Yueqian Wang, Yuxuan Wang, Kai Chen, and Dongyan Zhao. Stair: spatial-temporal reasoning with auditable intermediate results for video question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 19215–19223, 2024. 2
- [56] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025. 1, 3, 5
- [57] Zhenhailong Wang, Manling Li, Ruochen Xu, Luowei Zhou, Jie Lei, Xudong Lin, Shuohang Wang, Ziyi Yang, Chenguang Zhu, Derek Hoiem, et al. Language models with image descriptors are strong few-shot videolanguage learners. Advances in Neural Information Processing Systems, 35:8483–8497, 2022. 2
- [58] Zikang Wang, Boyu Chen, Zhengrong Yue, Yi Wang, Yu Qiao, Limin Wang, and Yali Wang. Videochat-a1: Thinking with long videos by chain-of-shot reasoning. arXiv preprint arXiv:2506.06097, 2025. 2
- [59] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 1
- [60] Jiaer Xia, Yuhang Zang, Peng Gao, Yixuan Li, and Kaiyang Zhou. Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning. arXiv preprint arXiv:2505.14677, 2025. 3
- [61] Junbin Xiao, Pan Zhou, Tat-Seng Chua, and Shuicheng Yan. Video graph transformer for video question answering. In European Conference on Computer Vision, pages 39–58. Springer, 2022. 2

- [62] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632– 10643, 2025. 2, 5, 3, 7
- [63] En Yu, Kangheng Lin, Liang Zhao, Yana Wei, Zining Zhu, Haoran Wei, Jianjian Sun, Zheng Ge, Xiangyu Zhang, Jingyu Wang, et al. Unhackable temporal rewarding for scalable video mllms. arXiv preprint arXiv:2502.12081, 2025. 6
- [64] En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954, 2025. 3
- [65] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 8
- [66] Yuanyuan Yuan, Shuai Wang, Mingyue Jiang, and Tsong Yueh Chen. Perception matters: Detecting perception failures of vqa models using metamorphic testing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16908– 16917, 2021. 1, 2, 4
- [67] Siliang Zeng, Quan Wei, William Brown, Oana Frunza, Yuriy Nevmyvaka, and Mingyi Hong. Reinforcing multiturn reasoning in llm agents via turn-level credit assignment. arXiv preprint arXiv:2505.11821, 2025. 5
- [68] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video questionanswering. arXiv preprint arXiv:2312.17235, 2023. 2
- [69] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 6
- [70] Xingjian Zhang, Siwei Wen, Wenjun Wu, and Lei Huang. Edge-grpo: Entropy-driven grpo with guided error correction for advantage diversity. arXiv preprint arXiv:2507.21848, 2025. 8
- [71] Xingjian Zhang, Siwei Wen, Wenjun Wu, and Lei Huang. Tinyllava-video-r1: Towards smaller lmms for video reasoning. arXiv preprint arXiv:2504.09641, 2025. 3
- [72] Yilun Zhao, Haowei Zhang, Lujing Xie, Tongyan Hu, Guo Gan, Yitao Long, Zhiyuan Hu, Weiyuan Chen, Chuhan Li, Zhijian Xu, et al. Mmvu: Measuring expertlevel multi-discipline video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8475–8489, 2025. 2, 5, 3

## VIDEOP2R: Video Understanding from Perception to Reasoning Supplementary Material

We provide additional details and illustrations for our main content in the following section:

- • Process-Aware CoT Generation (Sec. 7).
- • Evaluation Setup (Sec. 8)
- • Ablation Study on Judge Model (Sec. 9)
- • Perception Effectiveness Experiment (Sec. 10)
- • RL Training Trend (Sec. 11).
- • Think-Answer Mismatch (Sec. 12).
- • More Qualitative Results (Sec. 13).
- • Impact of Model Size and Dataset Composition (Sec. 14)

### 7. Details of Process-Aware CoT Generation and Data Analysis

#### 7.1. Prompt Used

Figure 7 illustrates the prompt template for processaware CoT generation. We employ Qwen2.5-VL-72BInstruct with a temperature of 0 for the generation.

Prompt Template for Process-aware CoT Generation

{Question} You are required to answer the question using the visual content provided. Your output must include **three parts**: <observation>, <think>, and <answer>. Do

**not** stop after the first part. Follow the instructions below for each section:

- Step 1: Visual Observation

- • First, analyze what the question is asking.
- • Identify what type of visual information is necessary to answer it (e.g., objects, actions, text, spatial relations, etc.).
- • Write a **single paragraph** that begins with: **'Based on the question, I need to notice ...'**
- • Then directly continue with your visual observations in the same paragraph.
- • Ensure that the motivation ('I need to notice...') and the actual observations are logically connected.
- • Wrap this section between <observation> and </observation> tags.

<observation> Based on the question, I need to notice ... [Continue your observation here in the same paragraph.] </observation>

- Step 2: Reasoning

- • Use your observations to reason step by step toward the answer.
- • Begin with: **'Based on the observations, ...'**
- • Engage in internal dialogue (e.g., 'let me think', 'hmm', 'wait a second', etc.).
- • It's encouraged to include **self-reflection or verification**, such as: 'Does that make sense?', 'Let me doublecheck...', etc.
- • Wrap your reasoning between <think> and </think> tags.

<think> Based on the observations, ... [Insert your reasoning here.] </think>

- Step 3: Final Answer

- • Provide a concise and specific final answer.
- • Ensure it is logically consistent with both the observations and the reasoning.
- • Wrap your answer between <answer> and </answer> tags.

<answer> [Insert your final answer here.] </answer>

- Figure 7. Prompt Template for Process-aware CoT Generation. We use the same prompt for training and inference.
- Figure 8 illustrates the prompt template for observa-

tion sufficiency verification. We use Claude 3.7 to judge the sufficiency of the observation segment.

#### 7.2. Data Source and Metric for CoT Verification

Our data source [14] encompasses five distinct question types to enhance the model’s flexibility and its generalization across diverse tasks and formats: (1) Multiple

###### Prompt Template for Observation Suﬃciency Veriﬁcation

You will receive a question, its correct answer, and a set of observations extracted from {data_type}. Your task is to decide—strictly based on the observations—whether the correct answer can be deduced. Do **not** assume any information beyond what is provided.

### Question {Question}

### Correct Answer {CorrectAnswer}

### Observations {Observations}

### Instruction

- 1. Carefully read the question and the correct answer.
- 2. Briefly explain whether (and how) the observations support the correct answer.
- 3. Finally output your judgement, either <judgement>Yes</judgement> or <judgement>No</judgement>.

Figure 8. Prompt Template for Observation Sufficiency Verification. We use the same prompt for perception correctness judgment in RL stage.

Choice, (2) Numerical QA, (3) OCR, (4) Free-form QA, and (5) Regression. Each data sample includes a question, data source, correct answer, and optional choices when applicable (e.g., for multiple-choice questions). In the subsequent CoT Verification stage, task-specific accuracy metrics are adopted to assess annotation reliability, and samples below a preset threshold of 0.6 are filtered out. The task-specific metrics are listed as follows:

- • Multiple Choice: 1 if the predicted option matches the ground truth; 0 otherwise.
- • Numerical QA: 1 for exact match with the reference value.
- • OCR: reward based on Word Error Rate (WER) between prediction and reference.
- • Free-form QA: reward is the average of ROUGE-1, ROUGE-2, and ROUGE-L scores.
- • Regression: reward = 1− relative error between prediction and ground truth.

These task-specific metrics are also used for computing the reasoning accuracy reward in the RL stage.

#### 7.3. Data Statistic

Adapting our generation pipeline to the data source yields VIDEOP2R-CoT-162K, consisting of 162,062 image and video visual QA pairs with high-quality annotations on perception and reasoning. We present the data statistics in Tab. 3. The dataset covers both image and video modalities, and spans multiple question types including multiple-choice, numerical, OCR, free-form, and regression. Multiple-choice questions constitute the majority, providing stable evaluation signals, while the inclusion of numerical, OCR, and free-form questions introduces diverse reasoning skills such as count-

ing, reading, grounded description, and open-ended inference. This heterogeneous composition enables comprehensive assessment of process-aware perception and reasoning across modalities.

To analyze our constructed data, we visualize the embedding distributions (using UMAP [35]) of perception and reasoning annotations (Fig. 9). The two clusters are clearly separated, indicating that our annotated data inherently distinguishes perception and reasoning.

#### 7.4. Word Count and Word Cloud Analysis

Figure 10 presents the word length distribution and word cloud visualization of VIDEOP2R-CoT-162K. As shown on the left of the figure, perception and reasoning annotations exhibit a comparable number of words across the entire annotation set, suggesting a balanced contribution of both processes. The word clouds further highlight the intrinsic difference in focus between the two processes: perception annotations are dominated by video-centric terms such as “video”, “person”, and “observing”, reflecting their emphasis on factual and descriptive content; in contrast, reasoning annotations frequently contain introspective expressions such as “double check” and “make sense”, which indicate deeper reflective reasoning.

[Figure 33]

Figure 9. Embeddings visualization of VIDEOP2R-CoT-162K

#### 7.5. Annotation Examples

We provide annotation examples in Figs. 11 and 12 to illustrate how our annotations explicitly separate perception from reasoning. Figure 11 presents a video QA example where the perception segment successfully captures the key visual cue (the zigzag pattern), and the reasoning segment then derives the correct answer based on this evidence. Figure 12 presents an image QA example in which the perception segment accurately extracts the numerical information from the table, and the reasoning segment performs the required mathematical reasoning

[Figure 34]

- Figure 10. Word length (Left) and Word cloud (Right) Visualization for VIDEOP2R-CoT-162K.

over these numbers, followed by validation to doublecheck the final answer.

|Question: In which direction does the person pour the oil into the wok? Options: A.In a diagonal line. B. In a circular motion. C.In a zigzag pattern. D.In a straight line.<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]|
|---|

<observation>Based on the question, I need to notice the direction in which the oil is poured into the wok. Observing the video, the oil is being poured in a way that creates a zigzag pattern across the surface of the wok. This is evident from the way the oil spreads and the lines it forms as it hits the wok's surface.</observation> <think>Based on the observations, the oil is being poured in a zigzag pattern. Let me think, this pattern is distinct and not a straight line, diagonal line, or circular motion. The zigzag pattern is clearly visible as the oil spreads across the wok. Does that make sense? Yes, it does. Let me double-check the options provided, and the zigzag pattern matches option C.</think> <answer>C</answer>

- Figure 11. An Annotation Example of the Video QA Sample

|Question: What is the total number of computers in all the classrooms combined? Answer: 55<br><br>[Figure 39]|
|---|

<observation>Based on the question, I need to notice the number of computers in each classroom as listed for each teacher. The table provides the necessary data: Miss Murphy has 20 computers, Mr. Schmidt has 10, Ms. Rojas has 19, Ms. Cohen has 1, and Mr. Watson has 5. These numbers are essential for calculating the total number of computers across all classrooms.</observation> <think>Based on the observations, I need to add up the number of computers for each teacher to ﬁnd the total. Let me think: 20 (Miss Murphy) + 10 (Mr. Schmidt) + 19 (Ms. Rojas) + 1 (Ms. Cohen) + 5 (Mr. Watson) equals 55. Does that make sense? Let me double-check the addition: 20 + 10 is 30, plus 19 is 49, plus 1 is 50, and plus 5 is indeed

55. The calculation seems correct.\n</think> <answer>55</answer>

- Figure 12. Annotation Example of the Image QA Sample

Table 3. Distribution of question types across VIDEOP2R-CoT-162K.

Question Type

Sum Multiple Choice Numerical OCR Free-form Regression

Image 47,091 18,476 4,014 2,501 693 72,775 Video 86,910 1,371 – 1,006 – 89,287

Sum 134,001 19,847 4,014 3,507 693 162,062

### 8. Experiment Set up

#### 8.1. Implementation Details

The whole two-stage training is conducted on 8× NVIDIA A100 GPUs. For efficiency, we limit the video input to 16 frames at a resolution of 128 × 28 × 28 during training, where 28×28 denotes the patch size and 128 the number of patches. For the SFT stage, we use a batch size of 8 with gradient accumulation = 2. For the RL stage, we adopt a batch size of 561 (8 rollouts per sample). We use Claude 3.7 Sonnet for observation judgment to speed up the training process. We note that since video token processing dominates training time, the additional text-only judgment introduces little overhead: GRPO takes 16.5 hours for 1k steps, while VIDEOP2R takes 18 hours on the same hardware.

During inference, we increase the number of frames and resolution to 32 and 256 × 28 × 28, respectively, and apply the decoding configuration (top p = 0.001, temperature = 0.01) consistent with the Qwen2.5-VLA official demo. During both training and inference, we adopt the same prompt (Fig. 7) as in process-aware CoT generation, and use the prompt shown in Fig. 8 for perception accuracy judgment.

#### 8.2. Main Table Evaluation Setup

This section introduces the evaluation benchmarks used in Tab. 1 and the evaluation metrics. We selected seven widely used video understanding and reasoning benchmarks to provide a comprehensive analysis of VideoP2R:

- • VSI-Bench [62] is a video-based benchmark designed to evaluate models’ visual–spatial reasoning capability. It includes two types of questions: (1) numerical and (2) multiple-choice. Numerical questions are evaluated using Mean Relative Accuracy (MRA), while multiple-choice questions are evaluated using Accuracy (ACC). Following the original benchmark protocol, we report the overall performance as the average of MRA and ACC.
- • VideoMMMU [20] is a multi-modal, multi-

1To accelerate training, we integrate vLLM [23] for sampling, assigning one GPU exclusively for sample generation and the remaining seven for model updates.

disciplinary video benchmark, designed to evaluate models’ ability to acquire and apply knowledge from expert-level lecture videos. In our experiments, models are evaluated with accuracy over all questions.

- • MMVU [72] is an expert-level multi-disciplinary video understanding benchmark aimed at assessing models’ capability to perform domain-specific reasoning across diverse scientific and technical fields. In our experiments, models are evaluated with accuracy over all multiple-choice questions (1858 of 3000).
- • VCR-Bench [40] is a benchmark crafted to assess video Chain-of-Thought reasoning. VCR-Bench selected and integrated data from multiple existing video benchmarks. In our experiments, models are evaluated with accuracy over all multiple-choice questions (510 of 1034).
- • MVBench [25] is a multi-modal video understanding benchmark designed to stress test models’ temporal reasoning capabilities across diverse domains. In our evaluation, models are assessed using accuracy on multiple-choice QA derived from temporally grounded tasks.
- • TempCompass [32] is a temporal reasoning benchmark designed to dissect video LLMs’ ability to perceive dynamic changes over time. It constructs paired videos that share identical static content but differ in temporal aspects (e.g., speed, direction) to prevent shortcut solutions based on static frames. In our evaluation, models are measured by accuracy over temporal reasoning questions under the official protocols.
- • Video-MME [15] is a comprehensive multi-modal evaluation benchmark for video-centric large language models, designed to assess their analysis capabilities across diverse video types and modalities. We evaluate using the official metrics and configuration, reporting accuracy over the QA pairs without subtitles.

For all result numbers of Open-Source Models in Tab. 1, we use the reported number in the original paper. For all result numbers of RFT Models, we run the evaluation locally. We additionally include Qwen2.5-VL72B in Tab. 8 as an upper bound for our model. While VIDEOP2R still trails Qwen2.5-VL-72B on average, it

significantly boosts the base model (Qwen2.5-VL-7B) and even outperforms Qwen2.5-VL-72B on MVBench, underscoring the effectiveness of our approach.

### 9. Ablation Study on Judge Model

- Table 4 presents the results of using different judge models for perception correctness judgement. We conduct the same two-stage training process, but only change the Claude3.7 to Llama3.1 [11] families for providing perception correctness judgment.

Compared with the base model, all VIDEOP2R variants using different judge models achieve consistent improvements, confirming the effectiveness of perception reward supervision. To further assess judge reliability, we randomly annotate the perception correctness of 200 samples with human labels and evaluate each judge’s decision accuracy. We observe a clear upward trend in accuracy as the judge model becomes larger and more capable. Additionally, the fact that even Llama3.1-8B attains reasonable reliability on this relatively simple perception correctness judgement suggests that perception correctness can be robustly handled by current LLMs, and our pipeline is broadly applicable across a wide range of judge models. Moreover, the positive correlation between judge capability and the downstream performance of the trained model indicates that stronger judges provide more reliable perception feedback and lead to larger gains, with Claude3.7 achieving the highest agreement with human annotations and the best overall process-aware performance.

10. Details of the Perception Examination

- 10.1. Prompt Used and Detailed Set up

The perception examination experiment involves three types of experiments on either text or video domains. We compare the zero-shot performance of Qwen2.5-VL7B across different input settings and examine how perception segments influence its answers: (i) performance on text-only questions, (ii) performance with both text and video inputs, and (iii) performance when the textonly prompt is augmented with a perception segment generated by VIDEOP2R or Qwen2.5-VL-7B. We used the prompt “Prompt for Qwen Inference” (Fig. 13 Top) for (i) and (ii). The prompt “Prompt for Qwen Inference with Perception Segment” (Fig. 13 Bottom) is used for (iii). For (iii), We use the same prompt in Fig. 7 to get the perception segment from VIDEOP2R or Qwen2.5VL-7B first and then augment the segment within the prompt for inference.

- 10.2. Full Results

- Table 5 presents the full results of the perception examination experiment (Fig. 4).

###### Prompt for Qwen Inference

{Question} Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions It's encouraged to include self-reflection or verification in the reasoning process. Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.

###### Prompt for Qwen Inference with Perception Segment

{Question} You are provided with an observation description instead of raw visual content. Here is the observation: {observation} Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions It's encouraged to include self-reflection or verification in the reasoning process. Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.

Figure 13. Prompt Template for Perception Examination Experiment.

###### Prompt Template for Answer Extraction

You are given a multiple-choice question with options and a model’s response, where its reasoning is enclosed in <think>…</think> tags. Extract the model’s **final chosen option** from the text strictly inside <think>…</think>.

Guidelines:

- • Consider only the content inside <think>…</think>.
- • Output **only** the option letter (e.g., A, B, C, D).
- • A statement counts as a **definitive selection** if it matches patterns like: 'the answer is', 'final answer is', 'I choose', 'I pick', 'correct answer is', 'Therefore/Thus/So, the answer is', or 'option <letter>'.
- • If the model states the answer in words (option text), map it to the corresponding option letter using the provided Options.
- • **Ignore** speculative or non-final phrases such as 'would be', 'could be', 'might be', 'closest match', 'I lean toward', unless no definitive selection is present.
- • If the reasoning indicates refusal/cannot choose/none fit, output 'No'.
- • Do **not** judge correctness and do **not** add explanations—just extract.
- • Return the result in <extraction><option_letter_or_No></extraction>. Here is the question and options:

Question: {Question} Here is the model reasoning response:

Model response: {model_response}

- • Please focus more on extracting final answer in the **end** of the <think>…</ think> rather within the reasoning process.
- • If the model already provide the option letter at the end of the <think>…</think>, you can directly extract it.
- • If the model does not provide the actual option letter at the end of the <think>… </think>, you can extract the final answer and try to map it to the option letter.
- • Only if you can not find the final answer explicitly or map it to the option letter, you can output 'No' as the final answer.

Figure 14. Prompt Template for Answer Extraction.

#### 10.3. Examples of Qwen Inference Output

We present examples of Qwen’s outputs under different configurations in our perception examination experiment in Fig. 21. When given only the text question (Top Left), Qwen fails to perform meaningful reasoning due to the absence of video information and resorts to guessing from the answer choices. When conditioned on the text question plus Qwen’s own perception segment (Top Right), it mentions some relevant visual cues (e.g., the person placing books into the backpack) but omits critical details such as the exact number of books, resulting in unreliable reasoning. Even with access to both the text question and the video input (Bottom Left), Qwen still produces inaccurate perception, confusing the top pocket with the main compartment; this misperception propagates into the reasoning process and yields an incorrect answer. In contrast, the perception segment generated by VIDEOP2R is clear and sufficient (Bottom Right), explicitly capturing both the number of books and their correct placement. This improvement can be attributed to PA-GRPO, which guides the perception process to target information directly relevant to answer-

Table 4. Ablation studies of VIDEOP2R on judge models. “Judge Acc.” reports perception decision accuracy on 200 human-labeled samples. Best results within each group are in bold.

|Video Reasoning<br><br>|Video Understanding|
|---|---|
|VSI. VideoMMMU MMVU VCR.<br><br>|MV. TempCom. VideoMME|

Model Judge Acc.

Avg

- • Base Models Qwen2.5-VL(7B) – 30.1 48.1 60.0 44.3 59.0 72.6 56.6 52.9

- • Judge Model VIDEOP2R (Llama3.1-8B) 82 39.0 52.2 64.0 49.2 64.7 73.8 59.2 57.4 VIDEOP2R (Llama3.1-70B) 88 35.8 52.4 64.6 50.2 65.0 74.5 60.5 57.6 VIDEOP2R (Llama3.1-405B) 91 38.2 54.4 64.5 49.2 66.5 75.0 58.4 58.0 VIDEOP2R (Claude3.7) 95 36.8 55.0 65.4 51.0 68.1 74.5 60.0 58.7

Table 5. Comparison of Qwen2.5-VL-7B with and without perception segments augmentation (video/text modalities).

|Video Reasoning<br><br>|Video Understanding|
|---|---|
|VSI. VideoMMMU MMVU VCR.|MV. TempCom. VideoMME|

Model

Avg

- • Video + Text Qwen2.5-VL(7B) 30.1 48.1 60.0 44.3 59.0 72.6 56.6 52.9 Qwen2.5-VL(7B) + Obs. 29.7 54.1 61.5 49.4 65.4 73.4 58.8 56.0 Qwen2.5-VL(7B) + Obs.(Qwen) 30.4 53.1 61.5 48.6 61.6 72.8 56.4 54.9

- • Text Qwen2.5-VL(7B) 22.1 34.4 45.8 33.5 30.9 46.2 33.1 35.1 Qwen2.5-VL(7B) + Obs. 33.8 51.5 61.0 48.4 63.6 72.9 57.5 55.5 Qwen2.5-VL(7B) + Obs.(Qwen) 28.8 45.8 60.0 41.5 49.1 67.3 48.0 48.6 VIDEOP2R 36.8 55.0 65.4 51.0 68.1 74.5 60.0 58.7

ing the question, thereby enabling Qwen to arrive at the correct answer even without direct access to the video.

### 11. RL Training Dynamics

Perception Accuracy Reward Perception Format Reward Perception Response Length

[Figure 40]

[Figure 41]

[Figure 42]

Reasoning Accuracy Reward Reasoning Format Reward Reasoning Response Length

[Figure 43]

[Figure 44]

[Figure 45]

Figure 15. RL training Dynamics of VIDEOP2R

We provide the full RL training dynamics of VIDEOP2R in Fig. 15 to comprehensively illustrate our RL stage. Both the perception accuracy reward and the reasoning accuracy reward exhibit an overall increasing trend, indicating that the model progressively improves its ability to produce correct perception and reasoning traces. The perception format reward and reasoning for-

mat reward remain close to 1 throughout training, showing that the model consistently adheres to the processaware inference template and maintains stable format compliance. Since the length reward is conditioned on both accuracy and format rewards, we instead visualize the lengths of the perception and reasoning segments during RL. We observe an initial increase followed by a decrease in both segments, indicating that the model adaptively adjusts its outputs and eventually converges to concise yet sufficiently informative perception and reasoning traces.

### 12. Think-Answer Mismatch Analysis

#### 12.1. Pilot Experiment

We conduct a pilot experiment with Claude 3.7 to assess the reliability of answer extraction using the prompt in Fig. 14. We first sample 400 responses to multiplechoice questions and ask human annotators to extract the model’s answers from the ⟨think⟩ segments. The annotators achieve 95% agreement, with most disagreements arising from cases where the reasoning is unclear and the model appears to guess the answer. Using the same setting, Claude 3.7 reaches 96.5% accuracy, confirming the reliability of this assessment.

#### 12.2. Think-Answer Mismatch Example

Figure 16 presents an example of Think–Answer Mismatch, where the model conducts correct reasoning in ⟨think⟩ but produces an incorrect final answer in ⟨answer⟩. This mismatch highlights how relying solely on final-answer rewards can reinforce unfaithful or inconsistent behavior, underscoring the necessity of process-aware rewards in PA-GRPO.

Think-Answer Mismatch Example

Question: For the video example, if we change student 7's Test 2 score from 79 to 89, what will be the new t value. Options:

- A. -1.030
- B. 1.020
- C. -0.500
- D. -1.250
- E. -0.950
- F. -0.300
- G. 0.750
- H. 1.030
- I. -1.500
- J. -0.600

<think>To solve this problem, we need to recalculate the t-value after changing the test score for student 7 from 79 to 89. The t-value is calculated using the formula: \[ T = \frac{\bar{d}}{SE(\bar{d})} \] where:

- - \(\bar{d}\) is the mean of the differences,
- - \(SE(\bar{d})\) is the standard error of the difference. First, let's calculate the new mean of the differences and the new standard error of the difference.

- 1. Calculate the new mean of the differences:

- - Original differences: -7, 5, 12, 8, 10, -6, -11
- - New difference for student 7: -11 + 10 = -1
- - New differences: -7, 5, 12, 8, 10, -6, -1
- - Mean of new differences: \(\bar{d} = \frac{-7 + 5 + 12 + 8 + 10 + (-6) + (-1)} {7} = \frac{19}{7} \approx 2.71\)

- 2. Calculate the new standard error of the difference:

- - Original standard deviation of differences: \(s_d = 9.325\)
- - New standard deviation of differences: \(s_d' = \sqrt{\frac{\sum(d_i \bar{d})^2}{n-1}}\)
- - Sum of squared differences: \((-7)^2 + 5^2 + 12^2 + 8^2 + 10^2 + (-6)^2 + (-1)^2 = 49 + 25 + 144 + 64 + 100 + 36 + 1 = 419\)
- - New standard deviation: \(s_d' = \sqrt{\frac{419}{6}} \approx 9.05\)
- - Standard error: \(SE(\bar{d}) = \frac{s_d'}{\sqrt{n}} = \frac{9.05}{\sqrt{7}} \approx 3.44\)

- 3. Calculate the new t-value:

- New t-value: \(T = \frac{2.71}{3.44} \approx 0.79\) The closest option to 0.79 is 0.750. </think>

<answer>H</answer>

Figure 16. Example of Think-Answer Mismatch.

- 12.3. Think-Answer Mismatch Results

Table 62 reports Think–Answer Mismatch rates across benchmarks. All results are computed on the multiplechoice subsets of each benchmark.

- 13. More Qualitative Results of VIDEOP2R

#### 13.1. Success Case

We provide two additional success cases of VIDEOP2R in Fig. 19. In the left example, VIDEOP2R effectively tracks key visual information throughout the video: in the ⟨observation⟩ segment, it identifies the three positions where the yellow clothing is presented, supporting subsequent reasoning. In the right example, VIDEOP2R accurately captures relevant visual cues, including the person’s gestures and the background context, and leverages them to produce the correct final answer.

|Question: How many chair(s) are in this room? Answer: 2<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]|
|---|

|[Figure 51]|
|---|

Key-frame Missing

<observation>Based on the question, I need to notice the number of chairs present in the room. Observing the video, I see a room with a table and a chair. The chair is positioned next to the table, and there are no other chairs visible in the room. The room also includes a sofa, a coffee table, and a kitchen area, but these do not contribute to the count of chairs. The focus is on identifying the chairs, and only one chair is clearly visible in the video. The room appears to be a living space with minimal furniture, and the chair is the only piece of furniture that ﬁts the description of a chair. The video does not show any additional chairs in the background or any other area of the room.</observation> <think>Based on the observations, the room contains only one chair. Let me think, the question speciﬁcally asks for the number of chairs in the room. The video clearly shows one chair next to the table, and there are no other chairs visible in the room. Does that make sense? Let me double-check the video again to ensure there are no other chairs hidden or partially visible. After rechecking, it is conﬁrmed that there is only one chair in the room. The other furniture items like the sofa and coffee table do not count as chairs.</think> <answer>1</answer>

- Figure 17. Failure Cases of Key-frame Missing in VIDEOP2R

|Question: If I am standing by the stool and facing the sofa, is the tv to my front-left, front-right, back-left, or back-right? The directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis)? Options: A. front-right, B. back-left, C. back-right, D. front-left<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]|
|---|

<observation>Based on the question, I need to notice the spatial relationship between the stool, the sofa, and the TV. Observing the video, a table is positioned on my left, holding an oven, a blender, and a green plant. To the left of this table is a window with the curtain half open, and several plants placed along the window threshold; nothing notable is visible outside. Another table is placed close to the window, accompanied by four white chairs and four bowl mats with speckled decoration. The stool is positioned near the center of the room, and the sofa is located towards the back-right corner of the room. The TV is mounted on the wall to the left of the sofa, which means it is not directly in front of the sofa but rather to its left.</<observation> <think>Based on the observations, the TV is to the left of the sofa, which means it is in the back-left position relative to the sofa. Let me think, if I am standing by the stool and facing the sofa, the TV would be to my back-left. Does that make sense? Yes, it does. Let me double-check the options, and A. back-left seems to be the correct choice.</think> <answer>A<answer>

- Figure 18. Failure Cases of Overly detailed visual configuration in VIDEOP2R

#### 13.2. Failure Case

We identify two representative types of failure cases for VIDEOP2R, illustrated in Fig. 17 and Fig. 18. (1) Key-frame missing. During inference, we adopt uniform frame sampling, which may omit question-critical key frames containing essential visual evidence [6]. As shown in Fig. 17, the question asks for the number of chairs in the room. However, the sampled frames only include the view where one chair appears on the left side of the table, while the key frame showing another chair on the right side is missed. In this case, the question becomes unsolvable given the incomplete observations. Increasing the number of sampled frames can mitigate this issue, and an adaptive sampling strategy [19] can further reduce the risk of missing question-

2*VersaVid-R1 has too few available traces on VSI-Bench for meaningful statistics.

Table 6. Think–Answer Mismatch rates across models and benchmarks (lower is better).

|Video Reasoning<br><br>|Video Understanding|
|---|---|
|VSI. VideoMMMU MMVU VCR.|MV. TempCom. VideoMME|

Model

Avg

Qwen2.5-VL(7B) 4.9 12.2 3.5 6.3 7.8 4.6 5.1 6.3 Video-R1 (SFT) 2.0 2.7 3.2 1.0 0.6 0.5 0.8 1.6 VideoRFT (SFT) 5.4 4.2 9.1 5.0 7.3 1.6 3.6 5.2 VideoP2R (SFT) 0.9 4.2 0.9 1.0 1.4 0.9 1.2 1.5 Video-R1 25.5 19.0 18.8 26.7 21.7 14.5 24.3 21.5 VideoChat-R1 13.6 11.9 11.9 13.3 12.4 7.3 11.3 11.7 VersaVid-R1* – 21.5 11.1 18.6 16.7 9.5 15.6 15.5 VideoRFT 23.5 22.9 12.9 18.5 15.1 10.9 14.4 16.9 VIDEOP2R 6.8 9.6 7.9 7.6 7.4 6.4 7.1 7.5

critical evidence, which we leave for future work. (2) Overly detailed visual configuration. The second failure type arises when questions require tracking an excessive number of fine-grained visual details, which is particularly common in VSI-Bench [62]. Questions in VSI-Bench demand precise modeling of object layouts and relative positions across multiple regions, often exceeding the length targets ([128,320]) used during VIDEOP2R’s training. When the required descriptions fall outside this familiar length regime, the model tends to compress or drop critical perceptual details, leading to incomplete observations and subsequent reasoning errors. As shown in Fig. 18, the question targets the relative positions among the stool, sofa, and TV, while accurately specifying this configuration also requires the locations of surrounding reference objects, such as the table near the stool and the table next to the window. VIDEOP2R initially exhibits high-quality perception (e.g., correctly identifying the tables and window) but gradually introduces errors for later elements (e.g., the stool and sofa), which ultimately leads to failure in subsequent reasoning. We propose to have a more dynamic length reward system [51] in future work.

#### 13.3.FromBaseModelto VIDEOP2R:AStepwise Capability Evolution

We present a representative example in Fig. 20 to illustrate how the model’s perception and reasoning capabilities evolve from the base Qwen2.5-VL-7B to VIDEOP2R-SFT and finally to the RL-optimized VIDEOP2R. The question asks where the cat stays for the longest time. In the video the cat briefly starts on the stool and then spends the remaining time on the robot’s thigh. The base Qwen2.5-VL-7B only captures the early details where the cat is on the stool and misses the later details on the robot’s thigh, leading to an incorrect answer. VIDEOP2R-SFT attends to both the stool and the robot’s thigh but misjudges the relative durations.

In contrast, VIDEOP2R produces a comprehensive and faithful perception trace that correctly tracks the cat’s locations and time spent, even explicitly ruling out distractors such as the carpet and nest, thereby enabling reliable downstream reasoning. This example demonstrates that each stage (SFT, RL) of the VIDEOP2R framework is both effective and necessary, enabling stable, long-term improvement in process-aware perception and reasoning.

#### 13.4. Impact of Task Difficulty

We analyze VIDEOP2R’s adaptability across difficulty levels using the fine-grained categorizations of our benchmarks Sec. 8.2. VideoMMMU [20] spans three reasoning difficulties from basic knowledge identification (Identify) to interpretation (Interpret) and adaptation (Adapt), while Video-MME [15] categorizes perception difficulty by video duration (Short, Medium, Long). As shown in Tab. 7, VIDEOP2R obtains its largest gains on the more challenging splits: +3.66% on Interpret (VideoMMMU) and +3.24% on Medium videos (Video-MME), while the smallest gains occur on the easiest splits (+2.31% on Identify, +1.66% on Short). These results indicate that our method is most beneficial for challenging tasks, while providing moderate gains for simpler tasks that already fall within the model’s intrinsic capability boundary.

### 14. Impact of Model Size and Dataset Composition

We report the full results in Tab. 8, including VIDEOP2R’s adaptation to a smaller model. VIDEOP2R also scales effectively to smaller architectures: applying our pipeline to Qwen2.5-VL-3B improves average accuracy from 38.4% to 53.3%, surpassing even the Qwen2.5-VL-7B baseline (52.9%).

18:44To demonstrate that VIDEOP2R-CoT-162K is

|Question: Which caption matches the video better? Options: A. A man is dunking a basketball. B. The man is shooting a basketball.<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]|
|---|

|Question: Where is the highest position of the yellow clothes located? Options: A. On the tree, B. In the attic, C. On the ground, D. On the table in the room<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]|
|---|

<observation>Based on the question, I need to notice the action being performed by the person in the video. The video shows a basketball player in silhouette performing a dunk, as the ball is seen going through the hoop. The player's arm is extended upwards, and the ball is in motion, indicating a successful dunk. The background is dark with smoke, which adds to the dramatic effect of the dunk. </observation> <think>Based on the observations, the action in the video is a dunk, not a regular shot. The player's extended arm and the ball going through the hoop conﬁrm this. Let me think, does that make sense? Yes, it does. The question asks which caption matches the video better. Option A describes a dunk, which aligns with the visual evidence. Option B describes a shot, which does not match the action shown. Let me double-check the observations, and they still support the conclusion that the correct caption is A.</think> <answer>A</answer>

<observation>Based on the question, I need to notice the highest position of the yellow clothes in the video. Observing the video, I see a man wearing yellow clothes who is initially outside near a tree. Later, he is seen inside a room, and then in an attic. The yellow clothes are most elevated when he is in the attic.</observation> <think>Based on the observations, the yellow clothes are most elevated when the man is in the attic. Let me think, the question asks for the highest position of the yellow clothes. The man is standing on a ladder in the attic, which is higher than any other position in the video. Does that make sense? Yes, it does. Let me double-check the other options. The tree is not the highest, the ground is lower, and the table is also lower. Therefore, the highest position is indeed in the attic.</think> <answer>B</answer>

Figure 19. Success Cases of VIDEOP2R

- Table 7. Performance across difficulty levels on Video-MME (perception, by video duration) and VideoMMMU (reasoning). Best result of each column is in bold (all numbers in %).

Model

Video-MME (Perception) VideoMMMU (Reasoning) Short Medium Long Identify Interpret Adapt

Qwen2.5-VL-7B 68.49 55.92 51.06 68.67 48.67 42.00 VIDEOP2R 70.80 59.16 53.65 70.33 52.33 44.33

∆ +2.31 +3.24 +2.59 +1.66 +3.66 +2.33

- Table 8. Performance comparison on video reasoning and understanding benchmarks. Best/second-best result of each column is in bold/underline. Missing entries indicate unreported results (all numbers unit in %).

Video Reasoning Video Understanding

Model

Avg

VSI. VideoMMMU MMVU VCR. MV. TempCom. VideoMME Qwen2.5-VL-72B 37.2 67.0 73.4 54.9 61.7 74.9 64.5 61.9 Open-Source 7B Models LLaVA-OneVision-7B 32.4 33.8 49.2 – 56.7 – 58.2 – LongVA-7B 29.2 23.9 – – – 56.9 52.6 – Video-UTR-7B – – – – 58.8 59.7 52.6 – VideoLLaMA2-7B – – 44.8 – 54.6 – 47.9 – Qwen2.5-VL-3B 26.9 34.3 51.5 35.1 47.0 23.5 50.7 38.4 Qwen2.5-VL-7B 30.1 48.1 60.0 44.3 59.0 72.6 56.6 52.9 RFT on Qwen2.5-VL-7B

Video-R1 35.8 52.3 63.8 49.0 63.9 73.2 59.3 56.8 VideoChat-R1 33.9 54.0 63.0 49.0 67.9 72.5 57.7 56.9 Time-R1 29.0 51.0 62.9 49.6 63.1 73.7 59.3 55.5 VersaVid-R1 33.7 51.9 64.3 49.8 62.9 74.0 58.8 56.5 VideoRFT 36.8 51.1 68.5 49.6 62.1 73.7 59.8 57.4

VIDEOP2R (3B) (Ours) 38.7 45.3 58.7 45.3 63.2 66.4 55.9 53.3 VIDEOP2R (Ours) 36.8 55.0 65.4 51.0 68.1 74.5 60.0 58.7

beneficial and generalizes to other RFT pipelines, we compare it against Video-R1-CoT-165K, the CoT dataset used for SFT in Video-R1 [14]. Specifically, in the SFT stage, we train Qwen2.5-VL-7B on each dataset separately, and in the RL stage, we apply Video-R1’s temporal GRPO (T-GRPO) on top of both SFT checkpoints under identical settings. As shown in Tab. 9, training on VIDEOP2R-CoT-162K yields 55.6% aver-

age accuracy in the SFT stage, outperforming Video-R1CoT-165K (53.9%) by 1.7%. After applying T-GRPO, our dataset further improves to 57.0%, still surpassing the Video-R1 counterpart (56.8%). These results suggest that VIDEOP2R-CoT-162K not only provides a stronger foundation in the SFT stage, but also generalizes well to other RFT pipelines.

|Question: Where is the place where the cat in the video stays the longest? Options: A. On the carpet, B. In the cat's nest, C. On the stool, D. On the robot's thigh<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]|
|---|

On the stool On the robot's thigh

VIDEOP2R

###### Qwen2.5-VL-7B

<think> Let me think about this. The cat is shown on the stool in the beginning, then it jumps down and interacts with the robot. After that, the cat is not shown again.

<observation>Based on the question, I need to notice the location where the cat stays the longest in the video. Observing the video, the cat is seen initially on a stool, then it moves to the robot's thigh, and ﬁnally, it stays on the robot's thigh for a signiﬁcant amount of time. The cat does not spend much time on the carpet, in the cat's nest, or on the stool after the initial moment. The robot's thigh becomes a consistent spot for the cat throughout the video. This observation is crucial for answering the question accurately. </observation> <think>Based on the observations, the cat stays on the robot's thigh for the longest period. Let me think, the cat initially jumps onto the stool but then moves to the robot's thigh, where it remains for a considerable time. The other options do not match the cat's behavior as closely. Does that make sense? Yes, it does. The cat's prolonged stay on the robot's thigh is evident in the video. Let me double-check the other options; none of them seem to ﬁt as well as the robot's thigh. Therefore, the correct answer should be the option that matches the cat's longest stay.</think> <answer>D</answer>

The stool is the only place the cat is seen for a signiﬁcant amount of time. The cat is on the stool for the initial scene and then jumps down, so it's the only place it's seen for a longer duration.</think>

<answer>C</answer>

###### VIDEOP2R (SFT)

<observation>Based on the question, I need to notice the location where the cat stays the longest in the video. Observing the video, the cat is seen initially on a stool, then it jumps down and walks around the room. It brieﬂy interacts with the robot and the orange robot, but most of its time is spent on the stool. The cat does not stay on the carpet, in the cat's nest, or on the robot's thigh for an extended period.</observation> <think>Based on the observations, the cat spends the most time on the stool. Let me think, the cat is seen sitting on the stool at the beginning and then walks around the room, but it returns to the stool multiple times. Does that make sense? Yes, it does. The cat does not stay on the carpet, in the cat's nest, or on the robot's thigh for an extended period. Let me double-check... The cat's movements and interactions conﬁrm that the stool is the longest stay location.</think> <answer>C</answer>

- Figure 20. From base Qwen2.5-VL-7B to VIDEOP2R-SFT and VIDEOP2R: a representative example illustrating the stepwise improvement in model’s perception and reasoning.

- Table 9. Comparison between VIDEOP2R-CoT-162K and Video-R1-CoT-165K (all numbers in %). Best result of each column is in bold.

Video Reasoning Video Understanding

Method

Avg VSI. VideoMMMU MMVU VCR. MV. TempCom. VideoMME

###### SFT Stage

SFT (Video-R1-CoT-165K) 33.3 49.4 63.5 45.5 60.5 69.9 55.4 53.9 SFT (VIDEOP2R-CoT-162K) 35.2 53.7 61.6 46.9 62.3 72.4 57.2 55.6

###### RL Stage

SFT+T-GRPO (Video-R1-CoT-165K) 35.8 52.3 63.8 49.0 63.9 73.2 59.3 56.8 SFT+T-GRPO (VIDEOP2R-CoT-162K) 39.9 50.0 62.8 48.9 64.5 73.5 58.2 57.0

|Question: In the video, after the character zips up the top of the backpack for the ﬁrst time, how many books are left in the main compartment of the backpack? Options: A. 0, B. 2, C. 4, D. 6<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]|
|---|

Two books were placed in the main compartment of the back pack.

The top of the backpack was zipped.

###### Qwen with the text question only

###### Qwen with the text question plus the perception segments (Qwen)

<think>The question asks how many books are left in the main compartment after this action. However, the video itself is not provided, so I can't directly observe the movement. If we assume

<observation>Based on the question, I need to notice the number of books left in the main compartment of the backpack after the character zips it up for the ﬁrst time. The video shows a person placing books into a backpack and then zipping it up. The books are visible on the table before being placed in the backpack, and the person's hands are shown closing the backpack's top zipper.</observation>

that the character is packing books into the backpack, and the top compartment is being closed, it's likely that the books are being placed in the main compartment.

<think>Let me think about this. The question asks speciﬁcally about the number of books left in the main compartment of the backpack after the ﬁrst zip. The observation mentions that the person places books into the backpack and then zips it up. However, it doesn't specify how many books are placed in the backpack or how many are left after the ﬁrst zip. Without more speciﬁc information about the number of books initially placed or the number left after the ﬁrst zip, it's impossible to determine the exact number of books left in the main compartment.

Given the options:

- A. 0 - This seems unlikely because the character would need to have some books to pack.
- B. 2 - This is a small number, but it's possible.
- C. 4 - This is a reasonable number, considering a typical backpack can hold several books.
- D. 6 - This is a larger number, but it's still possible depending on the size of the backpack.

Since the question doesn't provide enough details to make a deﬁnitive choice, and the options given are speciﬁc numbers, it seems the question might be based on a misunderstanding or lack of information.</think>

Without more speciﬁc information about the number of books or the size of the backpack, the most reasonable answer based on typical backpack usage would be a moderate number. Therefore, the most likely answer is C.</think>

No answer

<answer>C</answer>

###### Qwen with the text question plus the perception segments (VIDEOP2R)

Qwen with the text question plus the video input

<observation>Based on the question, I need to notice the number of books left in the main compartment of the backpack after the character zips up the top for the ﬁrst time. In the video, the character is seen placing books into the backpack and then zipping it up. After the ﬁrst zip, there are two books visible in the main compartment of the backpack. </observation>

<think>Let me think through this step-by-step. Initially, there are two books on the table: one red and one with a yellow cover. The person picks up the red book and places it into the backpack. Then, they pick up the yellow book and place it into the backpack as well. After these actions, the person zips up the top of the backpack, which presumably closes the main compartment. At this point, both books should be inside the backpack. Now, let's consider the question: "After the character zips up the top of the backpack for the ﬁrst time, how many books are left in the main compartment of the backpack?" Since both books were placed inside the backpack and the top was zipped up, there should be no books left outside the backpack. Therefore, the number of books left in the main compartment of the backpack is zero. </think> <answer>A</answer>

<think> The question asks for the number of books left in the main compartment of the backpack after the character zips up the top for the ﬁrst time. The observation clearly states that after the ﬁrst zip, there are two books visible in the main compartment. This directly corresponds to option B. 2.

Let me think through this step-by-step:

- 1. The character places books into the backpack.
- 2. The character zips up the top of the backpack for the ﬁrst time.
- 3. After the ﬁrst zip, the observation mentions that there are two books visible in the main compartment.
- 4. The options provided are A. 0, B. 2, C. 4, and D. 6.
- 5. The observation aligns perfectly with option B. 2.

Therefore, the number of books left in the main compartment of the backpack after the ﬁrst zip is 2.</think>

<answer>B</answer>

- Figure 21. Examples of Perception Examination: Top Left: Qwen with the text question only; Top Right: Qwen with the text question plus the perception segments from Qwen; Bottom Left: Qwen with the text question plus the video input; Bottom Right: Qwen with the text question plus the perception segments from VIDEOP2R. Green text denotes correct visual information or reasoning traces, while red text denotes incorrect or insufficient visual information or reasoning traces.

