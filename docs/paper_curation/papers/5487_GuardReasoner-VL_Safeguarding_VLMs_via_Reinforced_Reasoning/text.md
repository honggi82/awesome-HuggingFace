## GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning

# arXiv:2505.11049v1[cs.AI]16May2025

Yue Liu1, Shengfang Zhai1, Mingzhe Du2,1 Yulin Chen1, Tri Cao1, Hongcheng Gao1, Cheng Wang1 Xinfeng Li2, Kun Wang2, Junfeng Fang1, Jiaheng Zhang1, Bryan Hooi1

- 1National University of Singapore
- 2Nanyang Technological University yliu@u.nus.edu

### Abstract

To enhance the safety of VLMs, this paper introduces a novel reasoning-based VLM guard model dubbed GuardReasoner-VL. The core idea is to incentivize the guard model to deliberatively reason before making moderation decisions via online RL. First, we construct GuardReasoner-VLTrain, a reasoning corpus with 123K samples and 631K reasoning steps, spanning text, image, and text-image inputs. Then, based on it, we cold-start our model’s reasoning ability via SFT. In addition, we further enhance reasoning regarding moderation through online RL. Concretely, to enhance diversity and difficulty of samples, we conduct rejection sampling followed by data augmentation via the proposed safety-aware data concatenation. Besides, we use a dynamic clipping parameter to encourage exploration in early stages and exploitation in later stages. To balance performance and token efficiency, we design a length-aware safety reward that integrates accuracy, format, and token cost. Extensive experiments demonstrate the superiority of our model. Remarkably, it surpasses the runner-up by 19.27% F1 score on average, as shown in Figure 1. We release data, code, and models (3B/7B) of GuardReasoner-VL1.

Warning: This Paper Contains Potentially Harmful Content.

VLM Guard Models

VLM Guard Models

[Figure 1]

[Figure 2]

Avg. F1 Score (%)

Avg. F1 Score (%)

LLM Guard Models

LLM Guard Models

Azure Content Safety API

Azure Content Safety API

Qwen2.5-

Qwen2.5-

Aegis Guard Permissive 7B

LLaMA Guard 3 Vision 11B

Guard ReasonerVL 7B (ours)

Aegis Guard Permissive 7B

LLaMA Guard 3 Vision 11B

Guard ReasonerVL 7B (ours)

Guard Reasoner

Guard Reasoner

Wild Guard 7B

OpenAI Moderation API

Wild Guard 7B

OpenAI Moderation API

LLaMA Guard 3 8B

Shield Gemma 9B

LLaMA Guard 3 8B

Shield Gemma 9B

VLInstruct 7B

VLInstruct 7B

8B

8B

(a) Prompt Harmfulness Detection Task.

(b) Response Harmfulness Detection Task.

#### Figure 1: Mean Performance of GuardReasoner-VL on Multi-modal Guardrail Benchmarks.

1https://github.com/yueliu1999/GuardReasoner-VL/

Preprint. Under review.

### 1 Introduction

Built upon large language models (LLMs), vision-language models (VLMs) achieve remarkable success in a wide range of real-world applications such as computer use [68], deep research [71], embodied AI [14], etc. However, when deploying VLMs in safety-critical domains such as education [12], finance [80], or government, they remain vulnerable to manipulations and attacks [50, 21, 51, 41]. To alleviate this problem, safety alignment methods [49, 95] are proposed by training VLMs to align with human values and expectations. While effective, it imposes the alignment tax [27, 42], compromising the fundamental capabilities of models, such as creativity, helpfulness, and reasoning.

To mitigate this drawback, VLM guard models [16, 11, 30] are developed to safeguard VLMs without direct modifications to the victim VLMs. For example, VLMGuard [16] detects malicious text-image prompts using unlabeled data. In addition, LLaMA Guard 3-Vision [11] moderates both text-image prompts and text responses by SFT. Then, Beaver-Guard-V [30] is developed via RL with a welltrained reward model. The existing VLM guard models are trained to output only classification results. Although effective, they lack interpretability, as the models do not justify their decisions. Besides, the harmful categories are fixed, restricting the generalization to new categories.

Therefore, this paper aims to build a reasoning-based VLM guard model. It has three challenges as follows. 1) Limited Data. The available training data is limited in terms of the number of samples, input modalities, and reasoning processes. 2) Offline Training. Current guard models are typically restricted to offline training, which hampers their performance. 3) Token Efficiency. The reasoning process increases token costs, reducing inference efficiency.

To this end, we propose a novel reasoning-based VLM guard model termed GuardReasoner-VL by incentivizing it to reason-then-moderate via online RL. 1) First, to solve data limitations, we create GuardReasoner-VLTrain, a reasoning corpus with 123K samples and 631K reasoning steps. Unlike the existing data, we collect a mixture of text, image, and text-image samples (see Figure 3) to match the diverse input modalities of VLMs, and generate reasoning processes by prompting GPT-4o. Based on GuardReasoner-VLTrain, we cold-start our model via SFT. 2) Then, we conduct online RL to incentivize our model. To increase the diversity and difficulty of the data, we perform data augmentation via our proposed safety-aware data concatenation. The main principle is to guide the model to detect harmful content hidden among predominantly harmless content. We concatenate the inputs of different samples and assign new safety labels based on whether any of the original samples are labeled as harmful. Besides, we use a dynamic clipping parameter to encourage the model to explore in the early stage and exploit in the later stage. 3) To balance the model performance and token efficiency, we design a length-aware safety reward, integrating accuracy, format, and reasoning tokens. We develop two model versions: GuardReasoner-VL, a more powerful version, and GuardReasoner-VL-Eco, a more token-economical version. The contributions are listed as follows.

- • We develop GuardReasoner-VL, a novel VLM guard model that first reasons and then moderates.
- • We curate a reasoning corpus for VLM guard termed GuardReasoner-VLTrain, containing 123K samples with 631K reasoning steps, covering text, image, and text-image paired samples.
- • We incentivize the reasoning ability of our model through online RL, incorporating the proposed safety-aware data concatenation, dynamic clipping parameter, and length-aware safety reward.
- • Extensive experiments and analyses verify the superiority of our proposed GuardReasoner-VL.

### 2 GuardReasoner-VL

This section outlines the methodology of the proposed GuardReasoner-VL. First, we define the moderation task of VLM guard models. Then, we present the data curation for our training data. In addition, we introduce the training pipeline of our proposed reasoning-based VLM guard model. The overview training pipeline is shown in Figure 2. The basic notations are summarized in Table 4.

Moderation Task. Given a victim VLM F, a user inputs a prompt X and receives a response S = F(X), where X can be represented by one of the following modalities: a text T , an image I, or an text-image pair {T ,I}. The VLM guard model G moderates the input and output of the victim VLM F by detecting whether they are harmful, formulated as follows.

Yˆ = (Yˆprom,Yˆres) = G(X,S), (1)

[Figure 3]

- Figure 2: Overview Training Pipeline of GuardReasoner-VL.It mainly contains three processes, including data curation, model cold-start, and online RL. Concretely, we first build a reasoning corpus, which contains 123K samples with 631K reasoning steps, spanning text, image, and textimage modalities. We cold-start the model via reasoning SFT. Then, we perform data augmentation to improve the difficulty and diversity of the data via safety-aware data concatenation. In addition, we conduct online RL with a dynamic clipping parameter and the designed length-aware safety reward.

where Yˆprom ∈ {harmful,unharmful} is the predicted label for the prompt harmfulness detection task, and Yˆres ∈ {harmful,unharmful} is the predicted label for the response harmfulness detection task. The performance of G is evaluated using the F1 score between the predicted label Yˆ and the ground-truth Y = {Yprom,Yres}. The harmful/unharmful samples are treated as positives/negatives.

However, existing VLM guard models [16, 11, 30] merely provide classification results, limiting performance, explainability, and generalization. Thus, we aim to develop a reasoning-based VLM guard model Greasoner to first deliberatively reason and then make moderation decisions as follows.

{Yˆ,R} = Greasoner(X,S), (2) where R are reasoning processes, improving performance, explainability, and generalization.

- 2.1 Data Curation

First, to match the diverse input modalities of VLMs, we collect a mixture of text, image, and text-image samples. The distribution and cases are demonstrated in Figure 3.

[Figure 4]

- Figure 3: Input Modalities and Distribution of Our Training Data GuardReasoner-VLTrain. It contains 123K samples with 631K reasoning steps, spanning 3 input modalities, including text, image, and text-image. The prompt and response can be classified as harmful or unharmful.

Text. Following GuardReasoner [48], we collect and combine the data of WildGuardTrain [25], AegisTrain [19], BeaverTailsTrain [29], and ToxicChatTrain [43]. To balance the ratios of different input modalities, we use 50% of the mixed text data.

Image. We collect and combine the data of UnsafeBench [61], BadNews [88], HatefulMemes [34], HatefulPoliticalMemes (HatefulPMemes) [60], and HOD [24]. For HatefulMemes and HatefulP-

Memes, we utilize the processed data from VLGuard [95]. For HOD, we use 60% of the original dataset to balance the harmful and unharmful categories of the images. For this constructed image data, we use 80% for training and 20% for testing. The test set is named as HarmImageTest.

Text-Image. We utilize the SPA-VL-Train dataset [91] as the text-image paired training data. To balance the ratios of different input modalities, we use 50% of the SPA-VL-Train dataset.

Then, to train the reasoning-based VLM guard models, we generate the reasoning processes via prompting GPT-4o [48], as shown in Figure 10. As a result, we obtain a reasoning corpus termed GuardReasoner-VLTrain, consisting of 123K samples and 631K reasoning steps. The detailed statistics is listed in Table 6. In Figure 7, we show the distribution of data sources, the distribution of harmful categories, and representative cases of each modality in GuardReasoner-VLTrain.

#### 2.2 Model Cold-Start

Based on the curated reasoning dataset GuardReasoner-VLTrain, denoted as D, we cold-start the base model via Reasoning Supervised Fine-Tuning (R-SFT). Specifically, given the guardrail instruction Q, the user prompt X, and the victim model’s response S, we train the base model Mbase to generate both the reasoning process R and the moderation result Y. The objective is formulated as follows.

LR-SFT = −E(X,S,R,Y)∼D log Pθ(R,Y | Q,X,S), (3)

where θ denotes the model parameters. The input X can be a text, an image, or a text-image pair. The instruction, input, and output are showcased in Figure 11. Through R-SFT, we endow the model with basic reasoning ability for moderation, resulting in a reasoning model MR-SFT.

#### 2.3 Online Reinforcement Learning

Then, we perform online RL on MR-SFT to further enhance the reasoning ability regarding moderation. It contains three parts, including data augmentation, training process, and reward design.

#### 2.3.1 Data Augmentation

We generate harder and more diverse samples to better facilitate the generalization of online RL. First, we perform rejection sampling on MR-SFT over the reasoning corpus D. We run the entire dataset four times with high randomness and select the samples for which all predictions are incorrect.

[Figure 5]

- Figure 4: Safety-Aware Data Concatenation for Data Augmentation. Given two samples with labels {X1,Y1} and {X2,Y2}, we generate a new sample Xnew by concatenating text and merge image. We assign the new label Ynew as harmful if any of the original labels Y1,Y2 is harmful. It enables the guard model to identity harmful content hidden among predominantly harmless content.

Then, to further improve the diversity and the difficulty of the data, we conduct data augmentation via safety-aware data concatenation, as shown in Figure 4. Our core idea is to enable the guard model to identify harmful content hidden among predominantly harmless content. Take the prompt harmfulness

detection task as an example, given two text-image paired inputs X1 = {T1,I1},X2 = {T2,I2} and their labels Y1,Y2, the augmented sample is constructed as follows.

Tnew = text_concat(T1,T2), Inew = image_merge(I1,I2), Xnew = {Tnew,Inew}, (4) Ynew =

unharmful if Y1 = Y2 = unharmful harmful otherwise

, (5)

where text_concat denotes concatenating two textual inputs into a single context. image_merge denotes combining two image inputs through image-level transformations. For the new label Ynew of the augmented sample Xnew, we assign it as harmful if any of the original samples is harmful. In this manner, it can enhance the guard model’s ability to detect harmfulness in more complex and challenging cases. Through rejection sampling and safety-aware data augmentation, we generate a hard-sample reasoning corpus DRL for online RL.

#### 2.3.2 Training Process

Based on DRL, we train MR-SFT via online RL. We implement it by using group relative policy optimization (GRPO) [65]. Unlike standard GRPO, we omit the KL divergence loss to reduce constraints on the model’s behavior. In addition, we propose to encourage exploration in the early training stages and exploitation in the later training stages. The objective is formulated as follows.

G

1 G

(min(Ki,clip(Ki,1 − B,1 + B)) · Ai), (6)

LRL = −E(X,S,R,Y)∼D

RL,{Ri,Yˆi}Gi=1∼Pθold

i=1

s

Pθ(Ri,Yˆi|Q,X,S) Pθ

ri − mean({r1,r2,...,rG}) std({r1,r2,...,rG})

stotal − i stotal · ϵ,

Ki =

, Ai =

, Bs =

(Ri,Yˆi|Q,X,S)

old

i=1

(7)

where Ki is the policy ratio, Ai denotes the estimated advantage, {r1,r2,...,rG} is a group of rewards. We introduce a dynamic clipping parameter Bs in Formula (7), where s is the current training step, and stotal is the total number of training steps. In the early stage, the clipping threshold is set to a large value, allowing the model to explore more freely. In the later stages, it is gradually reduced to encourage more stable and fine-grained updates.

#### 2.3.3 Reward Design

We design a safety reward to guide our guard model to finish two guardrail tasks, i.e., prompt harmfulness detection and response harmfulness detection. First, the model should output in a correct format to ensure the predicted results are extracted correctly. Then, based on the correct format, we calculate the correctness between the predicted results and the ground truth of these two tasks, and combine them linearly. This safety reward is formulated as follows.

rsafety = Iformat × (rprompt × 0.5 + rresponse × 0.5), (8) rprompt =

1 if Yˆprom = Yprom 0 otherwise

1 if Yˆres = Yres 0 otherwise

, (9)

, rresponse =

where Iformat indicates whether the output format satisfies the required structure, i.e., Iformat = 1 if the model places the reasoning process R between the “<think>” and “</think>” tags, and the predicted

label between the “<result>” and “</result>” tags; otherwise, Iformat = 0.

Based on rsafety, to balance the performance and token efficiency, we incorporate the length of the reasoning process into the reward. The basic idea is that when the model fails to complete these guardrail tasks correctly, it is encouraged to improve its accuracy by scaling up the reasoning length, while remaining within a constraint. This length-aware safety reward is formulated as follows.

r = −1 + rsafety min(lnorm,β)2

, (10)

where lnorm ∈ [0,1] is the normalized length of the reasoning R, and β is a cut-off hyper-parameter to alleviate over-thinking. Note that the numerator rsafety is constrained to be non-positive, i.e., rsafety ∈ [−1,0]. Thus, when the model fails to complete all tasks correctly, i.e., rsafety ∈ [−1,0), it is encouraged to improve its accuracy by increasing the reasoning length, subject to the constraint β.

Through online RL with these designs, we obtain a reasoning-based VLM guard model Greasoner.

- Table 1: F1 score (%) of 21 Models on 8 Benchmarks of Prompt Harmfulness Detection. The bold and underlined values denote the best and the runner-up. “-” denotes that the result is unavailable.

OpenAI Moderation

Average (All) LLM Guard Models

Aegis SafetyTest

Simple SafetyTests

WildGuard Test

Average (Text)

HarmImage Test

SPA-VLEval

Method ToxicChat HarmBench

LLaMA Guard 7B 61.60 67.20 75.80 74.10 93.00 56.00 64.89 00.00 00.00 33.43

- LLaMA Guard 2 8B 47.10 94.00 76.10 71.80 95.80 70.90 63.62 00.00 00.00 32.77
- LLaMA Guard 3 8B 53.12 98.94 79.69 71.39 99.50 76.18 68.47 00.00 00.00 35.27

Aegis Guard Defensive 7B 70.00 77.70 67.50 84.80 100.00 78.50 72.99 00.00 00.00 37.60 Aegis Guard Permissive 7B 73.00 70.50 74.70 82.90 99.00 71.50 73.83 00.00 00.00 38.03

Aegis Guard 2.0 8B - - 81.00 - - 81.60 - 00.00 00.00 -

ShieldGemma 2B 06.91 11.81 13.89 07.47 05.83 09.36 09.38 00.00 00.00 04.83 ShieldGemma 9B 67.92 67.96 78.58 77.63 91.89 57.74 68.77 00.00 00.00 35.42

WildGuard 7B 70.80 98.90 72.10 89.40 99.50 88.90 77.99 00.00 00.00 40.17 GuardReasoner 1B 72.09 94.92 69.02 89.34 98.99 87.13 77.18 00.00 00.00 39.76 GuardReasoner 3B 78.38 88.58 71.88 91.19 100.00 88.97 80.80 00.00 00.00 41.62 GuardReasoner 8B 79.43 93.30 71.24 90.27 100.00 88.59 81.09 00.00 00.00 41.77

VLM Guard Models OpenAI Moderation API 25.40 09.60 79.00 31.90 63.00 12.10 35.28 44.39 63.00 44.20

Azure Content Safety API 57.61 37.41 74.27 46.75 74.21 32.54 54.30 26.42 43.64 44.95

LLaMA Guard 3 Vision 11B 58.19 96.09 67.64 70.62 97.96 75.19 67.24 00.48 54.86 48.03 Qwen2.5-VL-Instruct 3B 34.61 90.11 52.03 82.15 100.00 64.05 51.47 48.66 62.81 53.53 Qwen2.5-VL-Instruct 7B 40.99 91.61 57.21 81.58 100.00 74.77 58.04 43.88 66.02 56.53

GuardReasoner-VL-Eco 3B 73.47 88.58 70.87 89.04 99.50 89.16 78.43 66.79 85.82 77.39 GuardReasoner-VL 3B 74.45 89.10 70.83 88.79 99.50 88.92 78.77 70.93 86.47 78.73 GuardReasoner-VL-Eco 7B 76.26 98.73 70.82 90.34 99.50 88.54 79.82 64.84 85.26 77.49 GuardReasoner-VL 7B 76.51 98.30 70.98 90.13 98.99 88.35 79.88 70.84 85.60 79.07

### 3 Experiments

Environment. All experimental results are obtained on two servers with 8 NVIDIA H100 (80 GB) GPUs, and one server with 4 NVIDIA H200 (141GB) GPUs. For SFT, we use the LLaMA Factory [92] training platform. For online RL, we use the EasyR1 [93] training platform.

Benchmark. We evaluate our method on 14 benchmarks across two guardrail tasks, including prompt harmfulness detection and response harmfulness detection. For prompt harmfulness detection, we use 8 benchmarks, covering text-only inputs (ToxicChat [43], OpenAIModeration [53], AegisSafetyTest [19], SimpleSafetyTests [75], HarmBench [54], WildGuardTest [25]), image-only inputs (HarmImageTest), and text-image paired inputs (SPA-VL-Eval [91]). For response harmfulness detection, we use 6 benchmarks, including HarmBench [54], SafeRLHF [13], BeaverTails [29], XSTestResponse [64], WildGuardTest [25], and SPA-VL-Eval [91]. The statistical information of these benchmarks is listed in Table 5. We use F1 score (harmful category as positive samples) for evaluation. Due to the varying sample sizes across benchmarks (0.1K to 3K), we use a sample-weighted average of F1 scores across benchmarks to evaluate the performance. “Average (Text)” is the average performance on text guardrail benchmarks. “Average (All)” is the average performance on all guardrail benchmarks, including text, image, and text-image guardrail benchmarks. We do not evaluate response harmfulness in the image modality, as VLM responses are absent in the collected image benchmark.

Baseline. Since the used benchmarks contain text, image, and text-image inputs, we compare our model with both LLM guard models (LLaMA Guard 7B [28], LLaMA Guard 2 8B [17], LLaMA Guard 3 8B, Aegis Guard Defensive 7B, Aegis Guard Permissive 7B [19], Aegis Guard 2.0 8B [20], ShieldGemma 2B, ShieldGemma 9B [89], HarmBench LLaMA 13B, HarmBench Mistral 7B [54], MD-Judge 7B [39], BeaverDam 7B [29], WildGuard 7B [25]) and VLM guard models (LLaMA Guard 3-Vision [11], OpenAI Moderation API [53], Azure Content Safety API [4]). For Azure Content Safety API, we use text moderation for the text inputs, image moderation for image inputs, and multimodal moderation for text-image inputs. We did not compare with [30], as their models were not fully released at the time of our work.

#### 3.1 Performance

The performance is shown in Table 1 (prompt harmfulness detection) and Table 2 (response harmfulness detection). In Figure 1 (“Average (All)” metric) and Figure 8 (“Average (Text)” metric), we show the average performance of these two tasks. From the results, we draw 4 findings. 1) LLM guard models, limited to text inputs, underperform on image and text-image modalities, yield-

- Table 2: F1 score (%) of 25 Models on 6 Benchmarks of Response Harmfulness Detection. The bold and underlined values denote the best and the runner-up. “-” denotes the result is unavailable.

Average (All) LLM Guard Models

WildGuard Test

Average (Text)

SPA-VL -Eval

Method HarmBench SafeRLHF BeaverTails XSTestReponse

LLaMA Guard 7B 52.00 48.40 67.10 82.00 50.50 58.27 00.00 41.07

- LLaMA Guard 2 8B 77.80 51.60 71.80 90.80 66.50 66.99 00.00 47.22
- LLaMA Guard 3 8B 85.07 44.36 67.84 87.67 70.80 64.97 00.00 45.79

Aegis Guard Defensive 7B 62.20 59.30 74.70 52.80 49.10 62.79 00.00 44.25 Aegis Guard Permissive 7B 60.80 55.90 73.80 60.40 56.40 63.55 00.00 44.79

Aegis Guard 2.0 8B - - - 86.20 77.50 - 00.00 -

ShieldGemma 2B 35.36 16.92 30.97 65.55 20.13 27.24 00.00 19.20 ShieldGemma 9B 56.44 47.07 63.61 73.86 47.00 55.67 00.00 39.24

HarmBench LLaMA 13B 84.30 60.00 77.10 64.50 45.70 65.49 00.00 46.16 HarmBench Mistral 7B 87.00 52.40 75.20 72.00 60.10 66.70 00.00 47.01

MD-Judge 7B 81.60 64.70 86.70 90.40 76.80 78.67 00.00 55.45 BeaverDam 7B 58.40 72.10 89.90 83.60 63.40 76.60 00.00 53.99 WildGuard 7B 86.30 64.20 84.40 94.70 75.40 77.95 00.00 54.94

GuardReasoner 1B 84.75 68.39 85.84 90.12 74.81 79.06 00.00 55.72 GuardReasoner 3B 85.66 69.02 86.72 91.36 79.70 80.80 00.00 56.95 GuardReasoner 8B 85.47 70.04 87.60 94.34 78.20 81.22 00.00 57.24

VLM Guard Models OpenAI Moderation API 20.60 10.10 15.70 46.60 16.90 16.68 47.21 25.69

Azure Content Safety API 44.16 36.56 51.52 57.80 38.12 44.47 39.35 42.96

LLaMA Guard 3 Vision 11B 80.95 41.72 64.98 81.08 56.51 59.28 41.43 54.01 Qwen2.5-VL-Instruct 3B 62.14 64.71 73.30 31.40 29.79 58.05 52.84 56.51 Qwen2.5-VL-Instruct 7B 65.21 59.73 77.29 47.06 42.21 62.25 60.00 61.58

GuardReasoner-VL-Eco 3B 84.72 66.96 85.39 93.59 77.39 79.31 72.01 77.14

GuardReasoner-VL 3B 85.76 66.37 85.16 93.08 76.07 78.83 71.19 76.56 GuardReasoner-VL-Eco 7B 86.22 66.15 85.51 93.33 78.60 79.51 70.81 76.94

GuardReasoner-VL 7B 87.22 66.37 84.76 92.72 79.04 79.42 73.22 77.58

Online RL

Online RL

[Figure 6]

[Figure 7]

F1 Score (%)

F1 Score (%)

78.73%

79.07%

Reasoning SFT

78.03%

78.47%

77.81%

Reasoning SFT

77.29% 77.11%

77.09%

77.11%

76.85%

75.88%

74.67%

74.92%

74.57%

73.49%

73.25%

61.61%

65.65%

R-SFT (Text)

R-SFT (Image)

R-SFT (T-I)

w/o Dyn.

w/o Len. Reward

w/o

R-SFT (Text)

R-SFT (Image)

R-SFT (T-I)

w/o Dyn.

w/o Len. Reward

w/o

SFT R-SFT Ours

SFT R-SFT Ours

Aug.

Aug.

- Figure 5: Ablation Studies of 3B (left) and 7B Models (right) on Prompt Harmfulness Detection. Y-axis denotes F1 score (%), and X-axis denotes model variants in reasoning SFT and online RL.

ing unpromising average performance. 2) Existing VLM guard models, typically trained as pure classifiers on text-image pairs, struggle with image-only moderation. 3) Our models achieve the best performance by learning to reason for moderation across modalities. 4) Our models achieve comparable performance on text guardrail benchmarks with the state-of-the-art LLM guard models.

#### 3.2 Ablation Study

This section verifies the effectiveness of modules in GuardReasoner-VL. As shown in Figure 5, we conduct ablation studies on 3B and 7B models over the prompt harmfulness detection task. They are grouped into two stages, including the reasoning SFT stage and the online RL stage.

- Table 3: Performance and Token Costs of GuardReasoner-VL and GuardReasoner-VL-Eco. The F1 score is averaged over the prompt harmfulness detection and response harmfulness detection.

3B 7B F1 Score (%) Output Tokens F1 Score (%) Output Tokens GuardReasoner-VL 77.65 213.32 78.33 208.33

Model

GuardReasoner-VL-Eco 77.27 187.30 77.22 180.08 Relative Change 0.48%↓ 12.20%↓ 1.42%↓ 13.56%↓

[Figure 8]

[Figure 9]

#### Figure 6: Response Length and Reward During Training of Our Models.

First, at the reasoning SFT stage, “SFT” denotes conducting supervised fine-tuning on the collected multimodal data (text, images, text-image pairs) without reasoning processes. “R-SFT (Text)” denotes conducting SFT on the collected text data with reasoning processes. “R-SFT (Image)” denotes conducting SFT on the collected image data with reasoning processes. “R-SFT (T-I)” denotes conducting SFT on the collected text-image data with reasoning processes. “R-SFT” denotes conducting SFT on our GuardReasoner-VLTrain data. We have the conclusions as follows. 1) The reasoning processes help the model achieve better performance, e.g., “R-SFT” outperforms “SFT”. 2) Each modality of the reasoning data contributes to the performance improvement. However, SFT on images alone degrades the textual capability of the model, leading to unpromising performance.

Second, at the online RL stage, “Ours” denotes our GuardReasoner-VL model. “w/o Aug.” denotes our model without safety-aware data augmentation. “w/o Dyn.” denotes our model without the dynamic clipping strategy. “w/o Len. Reward” denotes our model without the length term in the reward. We find that 1) Each design contributes to the performance improvement. 2) GuardReasonerVL achieves the best performance, showing the effectiveness of the combination of these designs. Similar conclusions hold for the response harmfulness detection task, as shown in Figure 9.

#### 3.3 Token Efficiency

Although our reasoning-based VLM guard models achieve promising performance, their multi-step reasoning process incurs higher token consumption, which increases moderation latency. To mitigate

this issue, we set a constraint parameter β = 16 in Formula (10), developing a more token-efficient variant, termed GuardReasoner-VL-Eco. As shown in Table 3, this variant achieves comparable

performance (1%∼2% F1 score drops) while reducing around 10% token usage.

#### 3.4 Analyses

Training Process. We analyze the training process of our models. As shown in Figure 6, we visualize the training curves of GuardReasoner-VL 7B and GuardReasoner-VL-Eco 7B. We observe that GuardReasoner-VL 7B tends to increase its response length to gain higher rewards. In contrast, GuardReasoner-VL-Eco 7B initially increases the length slightly but soon stabilizes, still achieving competitive rewards under the imposed constraint.

Case Study. To further verify the effectiveness of our proposed GuardReasoner-VL, we conduct case studies on our GuardReasoner-VL 7B and “Qwen2.5-VL-Instruct 7B + SFT”. “Qwen2.5-VL-Instruct 7B + SFT” denotes conducting SFT on the collected multimodal data (text, images, text-image pairs) without reasoning processes for the Qwen2.5-VL-Instruct 7B model. The cases are demonstrated in Figure 12 (text input data), Figure 13 (image input data), and Figure 14 (text-image input data). From

these cases, we observe that GuardReasoner-VL can accurately identify harmful content in both user requests and AI responses. Also, it can effectively infer the underlying reasons for its predictions.

### 4 Related Work

- 4.1 Vision-Language Models

Motivated by the great success of the large language models (LLMs) [1, 72], Vision-language models (VLMs) are developed to extend the strong ability of LLMs to process both visual and textual information. The pioneer models like Flamingo [2], CLIP [62], and the BLIP series [36, 37] aim to align the visual encoders and LLMs in the latent space. Then, LLaVA is [44] proposed to construct the visual instruction data and conduct visual instruction tuning. This visual instruction tuning pipeline has become mainstream, and researchers [7, 45] pay attention to the construction of visual instruction data. Besides, any-resolution methods [9, 46] enable VLMs to handle images with any resolutions and ratios, improving the adaptability of VLMs in real-world applications. More recently, state-of-the-art open-sourced VLMs such as the LLaVA series [46, 35], InternVL series [9, 8, 10], and QwenVL [5, 77, 84] series have significantly advanced the capabilities of vision-language understanding.

- 4.2 Safety of VLMs

Despite their impressive performance, current VLMs remain susceptible to manipulations and attacks [50, 21, 51, 41], posing substantial risks in safety-critical applications such as autonomous driving [52], robotic manipulation [31], and education [12]. To alleviate this problem, the 3H principle [3] (Helpful, Honest, and Harmless) provides a foundational guideline for constraining model behaviors. Safety alignment techniques are proposed to better align VLMs with human values and expectations [86]. For example, [49] implements the safety alignment of VLMs by training the additional safety modules. In addition, ADPO [81], Safe RLHF-V [30], and [40] enhance the safety alignment of VLMs via DPO [63], RLHF [58], and GRPO [65], respectively. Besides, open-sourced datasets [91, 30, 22] contributed to high-quality alignment data and benchmarks. Differently, [78, 18, 15, 47] propose to conduct safety alignment at inference time.

Although effective, safety alignment on the VLM itself compromises its capabilities in other dimensions, e.g., creativity, reasoning, and helpfulness. As an alternative, safeguarding methods [79, 67, 90, 55, 47] are proposed to perform content moderation, aiming to ensure the safety of VLMs without directly degrading VLMs’ core abilities. Among these, one promising approach is to train a separate VLM-based guard model to moderate the inputs and outputs of the victim VLM. For example, based on LLaVA-OneVision [35] and the collected multimodal safety dataset, LLaVAGuard [26] is built to conduct large-scale dataset annotation and moderate the text-image models. However, it is merely designed to moderate the images rather than the text-image pairs. In addition, VLMGuard [16] is proposed to conduct malicious text-image prompt detection by leveraging the unlabeled user prompts. Moreover, LLaMA Guard 3-Vision [11] is developed to moderate both the text-image input and text output of VLMs via SFT. To improve the generalization ability, [30] presents Beaver-Guard-V by training a reward model and then applying reinforcement learning. Recently, GuardReasoner [48] has been proposed to enhance the performance, explainability, and generalization of the LLM guard model by guiding it to learn to reason. Motivated by its success, this paper develops a reasoning-based VLM guard model named GuardReasoner-VL.

### 5 Conclusion

This paper presents GuardReasoner-VL, a novel reasoning-based VLM guard model that moderates harmful multimodal inputs by first performing deliberative reasoning. To enable this, we construct a large-scale reasoning dataset, GuardReasoner-VLTrain, spanning diverse input modalities and complex safety cases. We further enhance the guard model via online reinforcement learning, leveraging a set of tailored techniques including safety-aware data concatenation, dynamic clipping, and a length-aware safety reward to balance safety performance and token efficiency. Extensive experiments demonstrate that GuardReasoner-VL significantly outperforms existing VLM guard models across multiple benchmarks. We hope our work offers a new direction for building interpretable, generalizable VLM guard models, and we release all data, code, and models to support future research. In the future, it is worthy building reasoning-based guard models for agentic systems.

### A Appendix

- A.1 Impact Statement

We introduce a guard model designed to enhance the safety of VLMs. By implementing this guard model, we aim to mitigate the potential risks and harmful impacts that VLMs may pose to society. The key aim of this paper is to demonstrate that the performance, explainability, and generalizability of the guard model can be improved by learning to reason. Inspired by this work, companies can build their own guard models for commercial use.

- A.2 Notations We list the basic notations of this paper in Table 4.

Table 4: Basic Notations of This Paper.

Notations Meanings Notations Meanings F Victim VLM D Reasoning Corpus for R-SFT X User Input Xnew Augmented Use Input T Text Input DRL Reasoning Corpus for RL I Image Input Mbase Base Model

{T ,I} Text-image Paired Input MR-SFT Trained Model via R-SFT S Response of Victim VLM Greasoner Reasoning-based VLM Guard Model G VLM Guard Model LR-SFT Objective of R-SFT

- Q Instruction for Guardrail Task Bs Dynamic Clipping Parameter

- R Reasoning Process r Overall Reward Yˆ Predicted Label lnorm Normalized Length of Reasoning Y Ground Truth LRL Objective of RL

- A.3 Datasets

We list the statistical information of the used benchmarks in Table 5. We list statistics of GuardReasoner-VLTrain in Table 6.

- A.4 Additional Experiments

We show the average performance of our model on text guardrail benchmarks in Figure 8. We list the additional experiments regarding ablation studies in Figure 9.

- A.5 Implementation

- A.5.1 Baseline

We use the original codes of the baselines to replicate their results. We introduce the baselines and provide the implementation details as follows, including 16 LLM guard models and 5 guard models.

#### LLM Guard Models

- • LLaMA Guard 7B. LLaMA Guard 7B [28] is Meta’s AI content guard model. It is instruct-tuned from the base model LLaMA 2 7B [74]. The training data is private and contains 13K samples.
- • LLaMA Guard 2 8B. LLaMA Guard 2 8B is the second version of the LLaMA Guard series. It is based on LLaMA 3 8B [17]. They flip labels to conduct data augmentation on the training data.
- • LLaMA Guard 3 8B. LLaMA Guard 3 8B is the third version of LLaMA Guard series. The base model is LLaMA 3.1 8B [17]. It supports 8 languages and has a context window of 128K tokens.

[Figure 10]

[Figure 11]

(a) Text. (b) Image.

[Figure 12]

(c) Text-Image.

Figure 7: Data Sources and Cases of Different Modalities in GuardReasoner-VLTrain Dataset. The Y-axis denotes the number of samples. The X-axis denotes the different data sources.

[Figure 13]

Avg. F1 Score (%)

[Figure 14]

Avg. F1 Score (%)

LLM Guard Models VLM Guard Models

LLM Guard Models VLM Guard Models

Azure Content Safety API

Azure Content Safety API

Qwen2.5-

Aegis Guard Permissive 7B

LLaMA Guard 3 Vision 11B

Guard ReasonerVL 7B (ours)

Qwen2.5-

Aegis Guard Permissive 7B

LLaMA Guard 3 Vision 11B

Guard ReasonerVL 7B (ours)

Guard Reasoner

Guard Reasoner

Wild Guard 7B

OpenAI Moderation API

LLaMA Guard 3 8B

Shield Gemma 9B

Wild Guard 7B

OpenAI Moderation API

LLaMA Guard 3 8B

Shield Gemma 9B

VLInstruct 7B

VLInstruct 7B

8B

8B

(a) Prompt Harmfulness Detection.

(b) Response Harmfulness Detection.

#### Figure 8: Mean Performance of GuardReasoner-VL on Text Guardrail Benchmarks.

- • Aegis Guard Defensive/Permissive 7B. They are developed by NVIDIA. It is based on LLaMA Guard 7B and uses LoRA to train the model. The defensive version classifies samples that need caution as harmful, and the permissive version classifies them as benign.
- • Aegis Guard 2.0 8B. It is the second version of the Aegis Guard series. The base model is LLaMA 3.1-instruct 8B. Ghosh et al. [20] propose a new safety corpus with 12 top-level hazard categories.
- • ShieldGemma 2B/9B. ShieldGemma 2B/9B is Google’s AI content moderation model. It is based on Gemma 2 2B/9B [70] and targets on four harm categories: sexually explicit, dangerous content, hate, and harassment.

Online RL

Online RL

[Figure 15]

F1 Score (%)

[Figure 16]

F1 Score (%)

77.58%

Reasoning SFT 76.56%

76.21%

76.05%

75.99%

Reasoning SFT

75.89%

75.78%

76.52% 76.59%

76.37%

75.52%

76.27%

76.02%

74.83%

75.06%

72.41%

01.46%

00.06%

R-SFT (Text)

R-SFT (Image)

R-SFT (T-I)

w/o Dyn.

w/o Len. Reward

w/o

R-SFT (Text)

R-SFT (Image)

R-SFT (T-I)

w/o Dyn.

w/o Len. Reward

w/o

SFT R-SFT Ours

SFT R-SFT Ours

Aug.

Aug.

Figure 9: Ablation Studies of 3B (left) and 7B Models (right) on Response Harmfulness Detection. X-axis denotes model variants in reasoning SFT and online RL.

Table 5: Statistics of 14 Benchmarks on 2 Guardrail Tasks.

Guardrail Task Benchmark # Sample Input Modality

ToxicChat 2,853 Text OpenAIModeration 1,680 Text

AegisSafetyTest 359 Text SimpleSafetyTests 100 Text HarmBenchPrompt 239 Text

Prompt Harmfulness Detection

WildGuardTest 1,756 Text HarmImageTest 3,295 Image

SPA-VL-Eval 3,282 Text-Image

HarmBenchResponse 602 Text SafeRLHF 2,000 Text BeaverTails 3,021 Text

Response Harmfulness Detection

XSTestReponseHarmful 446 Text WildGuardTest 1,768 Text

SPA-VL-Eval 3,282 Text-Image

#### Table 6: Statistics of our Reasoning Corpus GuardReasoner-VLTrain.

###### Modality # Sample # Step Mean Step Mean Len. per Step

Text 63,799 353,440 5.54 163.25 Image 13,267 57,322 4.32 154.03

Text-Image 46,030 221,033 4.80 160.79 Overall 123,096 631,795 5.13 159.36

- • HarmBench LLaMA 13B. HarmBench LLaMA 13B is based on LLaMA 2 13B [74]. The training data comes from GPT-4. It is used to evaluate jailbreak attacks in HarmBench [54].
- • HarmBench Mistral 7B. HarmBench Mistral 7B is based on Mistral 7B [32]. The training data is constructed by prompting GPT-4. It is used to evaluate jailbreak attacks in HarmBench [54].
- • MD-Judge 7B. MD-Judge 7B [39] is based on Mistral 7B [32]. The training data is private.

- • BeaverDam 7B. BeaverDam 7B [29] is based on LLaMA 7B [73] and is instruction-tuned on BeaverTails training dataset [29].
- • WildGuard 7B. WildGuard 7B is based on Mistral 7B [32]. It unifies the tasks of prompt/response harmfulness detection and refusal detection. They release the training data, WildGuardTrain.
- • GuardReasoner 1B. WildGuard 1B is based on LLaMA 3.2 1B [17]. It is a reasoning-based LLM guard model. They release the reasoning corpus GuardReasonerTrain.
- • GuardReasoner 3B. WildGuard 3B is based on LLaMA 3.2 3B [17]. It is a reasoning-based LLM guard model. They release the reasoning corpus GuardReasonerTrain.
- • GuardReasoner 8B. WildGuard 8B is based on LLaMA 3.1 8B. It is a reasoning-based LLM guard model. They release the reasoning corpus GuardReasonerTrain.

#### VLM Guard Models.

- • OpenAI Moderation API. It [53] is a tool that automatically detects and filters harmful or inappropriate user-generated content using AI, helping developers maintain safe environments.
- • Azure Content Safety API. The cloud-based Azure AI Content Safety API [4] provides developers with access to advanced algorithms for processing images and text and flagging content that is potentially offensive, risky, or otherwise undesirable.
- • LLaMA Guard 3 Vision 11B. LLaMA Guard 3 Vision [11] is a LLaMA-3.2-11B pretrained model [17], fine-tuned for content safety classification. It can be used to safeguard content for both LLM inputs and LLM responses.
- • Qwen2.5-VL-Instruct 3B/7B. Qwen2.5-VL-Instruct 3B/7B are fine-tuned for instructionfollowing, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. We prompt them to finish VLM guardrail tasks.

- A.5.2 GuardReasoner-VL We provide the implementation details of our proposed GuardReasoner-VL as follows.

- (I) In the R-SFT stage, we adopt 2 base VLM models with different scales, including Qwen2.5-VLInstruct 3B and Qwen2.5-VL-Instruct 7B. We use our synthesized GuardReasoner-VLTrain as the training data of R-SFT. It contains 123K samples with 631K reasoning steps. The chat template is set to qwen2_vl. The cutoff length is set to 2048 tokens. The initial learning rate is set to 5e-05, and we use the cosine learning rate scheduler. We use the BFloat16 training, and we adopt the full-parameter fine-tuning. We adopt AdamW optimizer. The number of epochs is set to 3. The total batch size is set to 192 = 8(accumulate step) × 6(batch size) × 4(device). The DeepSpeed stage is set to 3.
- (II) In the online RL stage, we first perform rejection sampling. We generate 4 candidate responses using temperature = 1.0 and top_p = 0.95, and retain only those hard samples where all responses are incorrect. Then, we perform data augmentation on these hard samples by randomly selecting pairs of the samples and conducting safety-aware data concatenation. We set the augmented samples to comprise 20% of the training data for online RL. We obtain training data for online RL, consisting of 12K samples. During training, the number of rollouts is set to 16 and temperature = 1.2. The batch size of rollouts is set to 512. The batch size for the actor model is 256. The initial learning rate for the actor model is set to 1e-6, and the weight decay is set to 1e-2. The clipping ratio ϵ is set to 0.2. The length constrain β is set to 1 for GuardReasoner-VL, and 16 for GuardReasoner-VL-Eco.

#### A.6 Additional Related Work

Reasoning Ability of VLMs. Recent advances in vision-language reasoning have enabled VLMs to tackle increasingly complex multimodal tasks, including math [76], code [38], and agent systems [82]. Early efforts focused on eliciting reasoning capabilities through improved visual encoding strategies [33], task-specific modules [23], in-context learning [94], and prompt tuning [87]. More recently, inspired by models such as OpenAI o1/o3 [56, 57] and DeepSeek R1 [69], researchers have shifted toward actively incentivizing VLMs to learn the reasoning processes [83, 85, 6, 59, 66].

[Figure 17]

Figure 10: Prompt for Reasoning Data Synthesis.

- A.7 Case Studies

The cases are demonstrated in Figure 12 (text input data), Figure 13 (image input data), and Figure 14 (text-image input data).

- A.8 Limitations

Although the proposed GuardReasoner-VL achieves promising performance, the token efficiency is still limited. In the future, we could solve this problem via techniques like model merge, agentic router, pruning, etc. Besides, for the attacks in the wild, e.g., indirect attacks in the environment for a computer-use agent, our models may achieve unpromising performance. In the future, it is worthy developing the reasoning-based guard models for the computer-use agents or multi-agent systems.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35: 23716–23736, 2022.

- [3] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.

- [4] Microsoft Azure. Azure ai content safety. https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety/, 2024.

[Figure 18]

#### Figure 11: Instruction, Input, and Output for Reasoning SFT.

- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

- [6] Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/Deep-Agent/ R1-V, 2025. Accessed: 2025-02-02.
- [7] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024.

- [8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

- [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024.

- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.

- [11] Jianfeng Chi, Ujjwal Karn, Hongyuan Zhan, Eric Smith, Javier Rando, Yiming Zhang, Kate Plawiak, Zacharie Delpierre Coudert, Kartikeya Upasani, and Mahesh Pasupuleti. Llama guard 3 vision: Safeguarding human-ai image understanding conversations. arXiv preprint arXiv:2411.10414, 2024.

- [12] Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo Yan, Jinheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang, Philip S Yu, et al. Llm agents for education: Advances and applications. arXiv preprint arXiv:2503.11733, 2025.

[Figure 19]

- Figure 12: Case Study on Text Input Data. This case is sampled from WildGuardTest [25].

- [13] Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe rlhf: Safe reinforcement learning from human feedback. arXiv preprint arXiv:2310.12773, 2023.

[Figure 20]

- Figure 13: Case Study on Image Input Data. This case is sampled from HatefulMemes [34].

- [14] Google Deepmind. Gemini robotics brings ai into the physical world. https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/, 2025.

- [15] Yi Ding, Bolian Li, and Ruqi Zhang. Eta: Evaluating then aligning safety of vision language models at inference time. arXiv preprint arXiv:2410.06625, 2024.

- [16] Xuefeng Du, Reshmi Ghosh, Robert Sim, Ahmed Salem, Vitor Carvalho, Emily Lawton, Yixuan Li, and Jack W Stokes. Vlmguard: Defending vlms against malicious prompts via unlabeled data. arXiv preprint arXiv:2410.00296, 2024.

- [17] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [18] Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, and Amrit Singh Bedi. Immune: Improving safety against jailbreaks in multi-modal llms via inference-time alignment. arXiv preprint arXiv:2411.18688, 2024.

[Figure 21]

- Figure 14: Case Study on Text-Image Input Data. This case is sampled from SPA-VL-Eval [91].

- [19] Shaona Ghosh, Prasoon Varshney, Erick Galinkin, and Christopher Parisien. Aegis: Online adaptive ai content safety moderation with ensemble of llm experts. arXiv preprint arXiv:2404.05993, 2024.

- [20] Shaona Ghosh, Prasoon Varshney, Makesh Narsimhan Sreedhar, Aishwarya Padmakumar, Traian Rebedea, Jibin Rajan Varghese, and Christopher Parisien. Aegis2. 0: A diverse ai safety dataset and risks taxonomy for alignment of llm guardrails. In Neurips Safe Generative AI Workshop 2024, 2024.

- [21] Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. Figstep: Jailbreaking large vision-language models via typographic visual prompts. arXiv preprint arXiv:2311.05608, 2023.

- [22] Tianle Gu, Zeyang Zhou, Kexin Huang, Liang Dandan, Yixu Wang, Haiquan Zhao, Yuanqi Yao, Yujiu Yang, Yan Teng, Yu Qiao, et al. Mllmguard: A multi-dimensional safety evaluation suite for multimodal large language models. Advances in Neural Information Processing Systems, 37:7256–7295, 2024.

- [23] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14953–14962, 2023.

- [24] Eungyeom Ha, Heemook Kim, Sung Chul Hong, and Dongbin Na. Hod: A benchmark dataset for harmful object detection. arXiv preprint arXiv:2310.05192, 2023.

- [25] Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. arXiv preprint arXiv:2406.18495, 2024.

- [26] Lukas Helff, Felix Friedrich, Manuel Brack, Patrick Schramowski, and Kristian Kersting. Llavaguard: Vlm-based safeguard for vision dataset curation and safety assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8322–8326, 2024.

- [27] Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, Yichang Xu, and Ling Liu. Safety tax: Safety alignment makes your large reasoning models less reasonable. arXiv preprint arXiv:2503.00555, 2025.

- [28] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674, 2023.

- [29] Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. Beavertails: Towards improved safety alignment of llm via a human-preference dataset. Advances in Neural Information Processing Systems, 36, 2024.

- [30] Jiaming Ji, Xinyu Chen, Rui Pan, Han Zhu, Conghui Zhang, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, et al. Safe rlhf-v: Safe reinforcement learning from human feedback in multimodal large language models. arXiv preprint arXiv:2503.17682, 2025.

- [31] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. arXiv preprint arXiv:2502.21257, 2025.

- [32] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

- [33] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13700–13710, 2024.

- [34] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. Advances in neural information processing systems, 33:2611–2624, 2020.

- [35] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

- [36] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022.

- [37] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

- [38] Kaixin Li, Yuchen Tian, Qisheng Hu, Ziyang Luo, and Jing Ma. Mmcode: Evaluating multimodal code large language models with visually rich programming problems, 2024.
- [39] Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, and Jing Shao. Salad-bench: A hierarchical and comprehensive safety benchmark for large language models. arXiv preprint arXiv:2402.05044, 2024.

- [40] Xuying Li, Zhuo Li, Yuji Kosuga, and Victor Bian. Optimizing safe and aligned language generation: A multi-objective grpo approach. arXiv preprint arXiv:2503.21819, 2025.

- [41] Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. Images are achilles’ heel of alignment: Exploiting visual vulnerabilities for jailbreaking multimodal large language models. In European Conference on Computer Vision, pages 174–189. Springer, 2024.

- [42] Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, Rui Pan, Haoxiang Wang, Wenbin Hu, Hanning Zhang, et al. Mitigating the alignment tax of rlhf. arXiv preprint arXiv:2309.06256, 2023.

- [43] Zi Lin, Zihan Wang, Yongqi Tong, Yangkun Wang, Yuxin Guo, Yujia Wang, and Jingbo Shang. Toxicchat: Unveiling hidden challenges of toxicity detection in real-world user-ai conversation. arXiv preprint arXiv:2310.17389, 2023.

- [44] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

- [45] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.

- [46] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. URL https: //llava-vl.github.io/blog/2024-01-30-llava-next/.
- [47] Qin Liu, Fei Wang, Chaowei Xiao, and Muhao Chen. Vlm-guard: Safeguarding vision-language models via fulfilling safety alignment gap. arXiv preprint arXiv:2502.10486, 2025.

- [48] Yue Liu, Hongcheng Gao, Shengfang Zhai, Jun Xia, Tianyi Wu, Zhiwei Xue, Yulin Chen, Kenji Kawaguchi, Jiaheng Zhang, and Bryan Hooi. Guardreasoner: Towards reasoning-based llm safeguards. arXiv preprint arXiv:2501.18492, 2025.

- [49] Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, and Bo Zheng. Safety alignment for vision language models. arXiv preprint arXiv:2405.13581, 2024.

- [50] Weimin Lyu, Lu Pang, Tengfei Ma, Haibin Ling, and Chao Chen. Trojvlm: Backdoor attack against vision language models. In European Conference on Computer Vision, pages 467–483. Springer, 2024.

- [51] Weimin Lyu, Jiachen Yao, Saumya Gupta, Lu Pang, Tao Sun, Lingjie Yi, Lijie Hu, Haibin Ling, and Chao Chen. Backdooring vision-language models with out-of-distribution data. arXiv preprint arXiv:2410.01264, 2024.

- [52] Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, and Chaowei Xiao. Dolphins: Multimodal language model for driving. In European Conference on Computer Vision, pages 403–420. Springer, 2024.

- [53] Todor Markov, Chong Zhang, Sandhini Agarwal, Florentine Eloundou Nekoul, Theodore Lee, Steven Adler, Angela Jiang, and Lilian Weng. A holistic approach to undesired content detection in the real world. In Proceedings of the AAAI Conference on Artificial Intelligence, 2023.

- [54] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249, 2024.

- [55] Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, and Srijan Kumar. Uniguard: Towards universal safety guardrails for jailbreak attacks on multimodal large language models. arXiv preprint arXiv:2411.01703, 2024.

- [56] OpenAI. Learning to reason with llms. https://openai.com/index/learning-to-reason-with-llms/, 2024.

- [57] OpenAI. Openai o3-mini. https://openai.com/index/openai-o3-mini/, 2024.

- [58] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- [59] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.

- [60] Shraman Pramanick, Shivam Sharma, Dimitar Dimitrov, Md Shad Akhtar, Preslav Nakov, and Tanmoy Chakraborty. Momenta: A multimodal framework for detecting harmful memes and their targets. arXiv preprint arXiv:2109.05184, 2021.

- [61] Yiting Qu, Xinyue Shen, Yixin Wu, Michael Backes, Savvas Zannettou, and Yang Zhang. Unsafebench: Benchmarking image safety classifiers on real-world and ai-generated images. arXiv preprint arXiv:2405.03486, 2024.

- [62] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

- [63] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

- [64] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. arXiv preprint arXiv:2308.01263, 2023.

- [65] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [66] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

- [67] Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, and Chaowei Xiao. Safeguarding vision-language models against patched visual prompt injectors. arXiv preprint arXiv:2405.10529, 2024.

- [68] Claude Team. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku. https://www.anthropic.com/news/3-5-models-and-computer-use, 2024.

- [69] Deepseek Team. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [70] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

- [71] OpenAI Team. Introducing deep research. https://openai.com/index/introducing-deep-research/, 2025.

- [72] OpenAI Team. Introducing chatgpt. https://openai.com/index/chatgpt/, 2025.

- [73] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [74] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

- [75] Bertie Vidgen, Nino Scherrer, Hannah Rose Kirk, Rebecca Qian, Anand Kannappan, Scott A Hale, and Paul Röttger. Simplesafetytests: a test suite for identifying critical safety risks in large language models. arXiv preprint arXiv:2311.08370, 2023.

- [76] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=QWTCcxMpPA.

- [77] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

- [78] Pengyu Wang, Dong Zhang, Linyang Li, Chenkun Tan, Xinghao Wang, Ke Ren, Botian Jiang, and Xipeng Qiu. Inferaligner: Inference-time alignment for harmlessness through cross-model guidance. arXiv preprint arXiv:2401.11206, 2024.

- [79] Yu Wang, Xiaogeng Liu, Yu Li, Muhao Chen, and Chaowei Xiao. Adashield: Safeguarding multimodal large language models from structure-based attack via adaptive shield prompting. In European Conference on Computer Vision, pages 77–94. Springer, 2024.

- [80] Ziao Wang, Yuhang Li, Junda Wu, Jaehyeon Soon, and Xiaofeng Zhang. Finvis-gpt: A multimodal large language model for financial chart analysis. arXiv preprint arXiv:2308.01430, 2023.

- [81] Fenghua Weng, Jian Lou, Jun Feng, Minlie Huang, and Wenjie Wang. Adversary-aware dpo: Enhancing safety alignment in vision language models via adversarial training. arXiv preprint arXiv:2502.11455, 2025.

- [82] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024.
- [83] Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step, 2024. URL https://arxiv.org/abs/2411.10440.
- [84] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

- [85] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.

- [86] Mang Ye, Xuankun Rong, Wenke Huang, Bo Du, Nenghai Yu, and Dacheng Tao. A survey of safety on large vision-language models: Attacks, defenses and evaluations. arXiv preprint arXiv:2502.14881, 2025.

- [87] J Diego Zamfirescu-Pereira, Richmond Y Wong, Bjoern Hartmann, and Qian Yang. Why johnny can’t prompt: how non-ai experts try (and fail) to design llm prompts. In Proceedings of the 2023 CHI conference on human factors in computing systems, pages 1–21, 2023.

- [88] Eric Zeng, Tadayoshi Kohno, and Franziska Roesner. Bad news: Clickbait and deceptive ads on news and misinformation websites. In Workshop on Technology and Consumer Protection, pages 1–11, 2020.

- [89] Wenjun Zeng, Yuchi Liu, Ryan Mullins, Ludovic Peran, Joe Fernandez, Hamza Harkous, Karthik Narasimhan, Drew Proud, Piyush Kumar, Bhaktipriya Radharapu, et al. Shieldgemma: Generative ai content moderation based on gemma. arXiv preprint arXiv:2407.21772, 2024.

- [90] Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Ming Hu, Jie Zhang, Yang Liu, Shiqing Ma, and Chao Shen. Jailguard: A universal detection framework for llm prompt-based attacks. arXiv preprint arXiv:2312.10766, 2023.

- [91] Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin, Senjie Jin, Yu Qiao, Xuanjing Huang, et al. Spa-vl: A comprehensive safety preference alignment dataset for vision language model. arXiv preprint arXiv:2406.12030, 2024.

- [92] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

- [93] Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework. https://github. com/hiyouga/EasyR1, 2025.
- [94] Yucheng Zhou, Xiang Li, Qianning Wang, and Jianbing Shen. Visual in-context learning for large vision-language models. arXiv preprint arXiv:2402.11574, 2024.

- [95] Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. Safety fine-tuning at (almost) no cost: A baseline for vision large language models. arXiv preprint arXiv:2402.02207, 2024.

