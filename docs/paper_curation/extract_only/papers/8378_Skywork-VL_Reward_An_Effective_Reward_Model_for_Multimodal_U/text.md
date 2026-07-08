arXiv:2505.07263v2[cs.CV]9Jun2025

# Skywork-VL Reward: An Effective Reward Model for Multimodal Understanding and Reasoning

Xiaokun Wang∗, Peiyu Wang∗, Jiangbo Pei∗, Wei Shen∗, Yi Peng, Yunzhuo Hao, Weijie Qiu, Ai Jian, Tianyidan Xie, Xuchen Song†, Yang Liu, Yahui Zhou

Skywork AI, Kunlun Inc. xuchen.song@kunlun-inc.com

#### Abstract

We propose Skywork-VL Reward, a multimodal reward model that provides reward signals for both multimodal understanding and reasoning tasks. Our technical approach comprises two key components: First, we construct a large-scale multimodal preference dataset that covers a wide range of tasks and scenarios, with responses collected from both standard vision-language models (VLMs) and advanced VLM reasoners. Second, we design a reward model architecture based on Qwen2.5-VL-7B-Instruct, integrating a reward head and applying multi-stage fine-tuning using pairwise ranking loss on pairwise preference data. Experimental evaluations show that Skywork-VL Reward achieves state-of-the-art results on multimodal VL-RewardBench and exhibits competitive performance on the text-only RewardBench benchmark. Furthermore, preference data constructed based on our Skywork-VL Reward proves highly effective for training Mixed Preference Optimization (MPO), leading to significant improvements in multimodal reasoning capabilities. Our results underscore Skywork-VL Reward as a significant advancement toward general-purpose, reliable reward models for multimodal alignment. Our model has been publicly released to promote transparency and reproducibility‡.

#### 1 Introduction

Large language models (LLMs) and vision-language models (VLMs) have recently achieved remarkable progress [1–7], demonstrating impressive capabilities across a wide range of tasks. Despite these advances, aligning their behavior with human preferences remains a significant challenge [8, 9, 6]. Reward models (RMs) have become indispensable in tackling this issue, serving as key components in both the training and inference stages of LLMs and VLMs [10–12].

While reward models for text-only LLMs have been extensively studied, the development of multimodal RMs remains in its early stages, with two major limitations: Existing multimodal RMs lack generalizability across diverse tasks and struggle to effectively evaluate advanced VLM reasoners with complex inference. Hence, there is a pressing need for multimodal RMs capable of assessing outputs from both standard VLMs and advanced VLM-based reasoners across diverse domains and tasks.

In this paper, we introduce Skywork-VL Reward, a multimodal RM designed to serve as a comprehensive and robust evaluator for VLM outputs. Our approach addresses previous limitations in domain

∗Equal contribution †Corresponding author ‡https://huggingface.co/Skywork/Skywork-VL-Reward-7B

Tech report. Preprint.

coverage and reasoning capacity by incorporating two critical improvements: (i) creating a carefully curated multimodal preference dataset derived from various sources, and (ii) developing a strong base model and training paradigm to enable effective vision-language understanding and reasoning. Specifically, we compile high-quality preference pairs from both publicly available datasets and internal annotations, spanning tasks from basic image descriptions to intricate reasoning scenarios. The collected preference pair includes the image (when applicable), textual prompt, and candidate responses sourced from standard VLMs [13, 14] and advanced VLM reasoners [6]. Building on this dataset, we construct Skywork-VL Reward based on Qwen2.5-VL-7B-Instruct, with an integrated reward head designed to output scalar scores aligned with human preferences. The model is trained using a two-stage training paradigm that combines both pure-text and multimodal data, which enhances its generalization and performance across a wide range of multimodal scenarios. Experimental evaluations confirm that Skywork-VL Reward achieves state-of-the-art results on VL-RewardBench [15] while maintaining competitive performance in text-only scenarios. Furthermore, preference data constructed based on our Skywork-VL Reward proves highly effective when used for training with Mixed Preference Optimization (MPO) [16], leading to significant improvements in multimodal reasoning capabilities.

Our contributions are summarized as follows. First, we introduce Skywork-VL Reward, a multimodal reward model capable of evaluating outputs from both standard VLMs and advanced VLM reasoners across diverse domains and tasks. Second, our model achieves state-of-the-art results on VLRewardBench, while maintaining competitive performance in text-only scenarios. Third, preference data generated using Skywork-VL Reward proves highly effective in MPO training, demonstrating the practical value of our model.

#### 2 Related Work

Reward Models for Text-only Large Language Models. Reward modeling has become a cornerstone of aligning LLM behavior with human preferences [17–22]. Generally, RMs are trained on comparisons of outputs (chosen vs. rejected responses) to predict which output is better, often using data collected from human raters or AI assistants. Existing RMs can be categorized along two axes: (1) the model form [23] (discriminative RMs vs. generative RMs vs. implicit RMs) and (2) the feedback target (outcome-based vs. process-based supervision). Discriminative RMs [24, 9, 25] treat preference prediction as a binary (or scalar) regression problem: given an input and a candidate response, the model directly outputs a score (or probability of being the preferred response). By contrast, generative RMs use a language-model head to generate an evaluation or verdict based on a specific prompt [26, 23, 2], rather than directly outputting a numeric score. A third category, implicit RMs [27], effectively reparameterize preference learning within the model itself via Direct Preference Optimization (DPO) [28], enables construct preference pairs without requiring an explicit RM. The second axis pertains to the nature of the feedback signal. Outcome-based RMs [29] generate a single scalar reward for the entire response, reflecting its overall quality or correctness. Process-based RMs [30] assess intermediate steps within a response, producing a sequence of rewards that reflect the quality of reasoning throughout the generation.

Reward Models for Vision-Language Models. Motivated by the observed benefits of preferencebased alignment for VLMs [16], the extension of reward modeling to the multimodal domain is an active research area. Most studies focused on generative RMs [31–33] based on open-source VLMs and explored data augmentation for response generation [34, 35]. For discriminative RMs, LLaVA-RLHF [36] pioneered the application of Reinforcement Learning from Human Feedback (RLHF) [37] to VLMs by leveraging human feedback to train a multimodal RM. Building on this, they further proposed Fact-RLHF, which enhances reward signals by incorporating additional information (e.g., image captions). To address the challenge of data scarcity, recent efforts have focused on the synthesis of large-scale, multi-domain preference datasets using advanced models as proxies for human evaluation. For instance, IXC-2.5-Reward [12] leverages GPT-4o and verifier functions [23] to automatically generate preference labels, leading to significant performance gains.

Most of the current approaches belong to ORM, providing reward signal for the final outputs of VLMs. More recently, Wang et al. introduced VisualPRM [38], a model trained on multimodal process supervision data that provides fine-grained multimodal process reward signals and serves as an effective critic model for test-time scaling of MLLMs, thereby enhancing the reasoning capabilities of existing VLMs.

[Figure 1]

Figure 1: Distribution of Training Data from Open-Source Sources.

#### 3 Method

Our objective is to develop a multimodal reward model named Skywork-VL Reward, which takes as input an optional image, a textual prompt, and a candidate response generated by either a multimodal understanding model or a reasoning model. The model outputs a scalar reward score that reflects the quality or degree to which the response aligns with human preferences. We achieve this by fine-tuning a pretrained VLM on a curated set of preference comparison data. In this section, we describe our dataset construction pipeline, the model architecture and modifications for reward modeling, the loss function used for pairwise preference training, and the overall training strategy.

###### 3.1 Dataset Construction

Open Source Data. We construct a comprehensive training dataset for Skywork-VL Reward by integrating multiple open-source preference datasets and additional in-house annotations. The dataset primarily includes three sources: (1) LLaVA-Critic-113k [33], (2) Skywork-Reward-Preference-80Kv0.2 [10], and (3) RLAIF-V-Dataset [32].

- • LLaVA-Critic-113k is an open-source dataset consisting of 113k multimodal instructionresponse examples. Each example contains an image, a user query, and one or more model responses with associated quality judgments. This dataset uniquely provides both pointwise scores and pairwise rankings generated by GPT-4o, covering tasks from straightforward image descriptions to complex reasoning challenges. Each pair is often accompanied by explanatory annotations, enriching our understanding of judgment criteria.
- • Skywork-Reward-Preference-80K-v0.2 is a high-quality dataset comprising 80k pairs of human-preferred textual responses, covering diverse domains such as general Q&A and creative writing. By eliminating noisy and inconsistent judgments through careful filtering, this dataset significantly enhances the text comprehension and alignment capabilities of Skywork-VL Reward, enabling it to effectively handle purely textual inputs.
- • RLAIF-V-Dataset is a large-scale multimodal feedback dataset containing 83,132 preference pairs. The instructions in this dataset are sourced from diverse datasets. Incorporating this

Table 1: Distribution of In-house Training Data.

Field Mathematics Physics Biology Chemistry Others Percentage (%) 35.4 24.6 14.7 20.2 5.1

Table 2: Percentage of Generation Approaches of In-house Training Data.

Approach Direct generation Two-step generation Percentage (%) 47.4 52.6

dataset greatly enhances the general multimodal understanding abilities of Skywork-VL Reward, enabling robust performance across varied tasks and contexts.

In-house Reasoning Data. We augment existing datasets with a proprietary in-house dataset consisting of approximately 50,000 preference comparisons focused on complex reasoning tasks. The tasks primarily involved carefully curated multimodal problems spanning mathematics, physics, biology, and chemistry (Table 1). These comparisons were collected through human annotation, where annotators assessed the correctness and reasoning quality of various VLM-generated reasoning-style responses.

###### 3.2 Data Curation Procedure

Our data curation involved three stages. Open-source datasets were utilized in the initial two stages. For the final stage, we employed a reasoning dataset derived from internal human annotation resources.

- Stage 1: The first data curation stage involved deduplication and filtering of the aggregated dataset. Specifically, our data curation involved the following:

- • Deduplication: Removal of identical pairs across different sources.
- • Similarity Filtering: Elimination of highly similar samples based on semantic similarity.
- • Judgment Filtering: Discarding pairs with ambiguous or low-confidence preference judgments (compared to GPT-4o), where "equal quality" was considered ambiguous.

This process yielded approximately 200,000 distinct, high-confidence preference pairs. This refined dataset was subsequently used to train a surrogate RM, which then scored the entire dataset.

- Stage 2: We further refined the collected dataset based on scores predicted by the surrogate RM in the second stage, with the following principles:

- • Low-Scoring Chosen Responses: If the chosen response in a preference pair received a low score from the surrogate RM, we regenerated that response using GPT-4o, effectively replacing the original.
- • Minimal Score Difference: For preference pairs where the score difference between the chosen and rejected responses was minimal, we also regenerated the chosen response with GPT-4o to enhance clarity.

Following this refinement process, a total of 150,000 data samples were retained. The distribution of this resulting training data is presented in Figure 1.

- Stage 3: In this stage, reasoning-style responses were generated via two distinct methods, with the percentage breakdown presented in Table 2:

- • Direct generation: Using our vision-language reasoning model Skywork R1V to directly produce reasoning-style responses conditioned on the original multimodal input.

- • Two-step generation: First employ a standard VLM model (we use the InternVL series [39]) to generate a textual description of the image, which serves as a substitute for the original visual input. Then, this textual description is fed into llm-reasoner (we use the Deepseek R1 series [40]) to generate the final reasoning-style response.

After the above procedure, our final training dataset consists of approximately 190k comparison samples (each a prompt with two responses and a preference), of which about 70% include images. We believe this dataset is one of the largest and most diverse multimodal preference collections to date, and it is a key enabler of Skywork-VL Reward’s strong generalization.

###### 3.3 Model Architecture

Our RM is built upon the Qwen2.5-VL-7B-Instruct [13]. Qwen2.5-VL-7B-Instruct is a 7-billionparameter VLM recently released by Alibaba Cloud, which extends the Qwen-7B LLM with visual understanding capabilities. The architecture consists of three main components: a Vision Transformer (ViT) as the visual encoder, a vision-language adapter/projector, and the Qwen language model decoder. The visual encoder processes an input image into a sequence of patch features. These raw image features are then passed through a small adapter module that projects them into a compact set of visual tokens that the language model can attend to. The language model then takes these visual tokens, along with text tokens from the prompt, and generates a response.

For Skywork-VL Reward, we modify this architecture to output a scalar reward instead of a text sequence. We remove Qwen’s causal LM head used for token prediction and replace it with a reward head that produces a single score. Concretely, we attach a fully-connected layer on top of the final hidden state to predict the reward.The reward head processes the final hidden state following the answer’s last token, producing a raw score rθ. This score is used during training to calculate a preference loss across various answers. At inference, the Skywork-VL Reward model evaluates a given prompt and response by outputting a quality score.

###### 3.4 Reward Model Loss Function

We train Skywork-VL Reward using a standard pairwise preference loss [41] commonly used in the reward modeling stage of RLHF. For a given comparison example, we have a prompt (with optional image) x, a preferred response y+ (the one chosen by the annotator or judge), and a dispreferred response y− (the one that was rejected). The model generates a scalar reward score for each response:

s+ = rθ(x,y+) for the preferred answer and s− = rθ(x,y−) for the dispreferred one, where rθ represents the reward model’s output. Our loss function aims to maximize the difference between s+ and s−, which can be formulated as:

LRM(θ) = −log σ rθ(x,y+) − rθ(x,y−) , (1)

where σ(z) = 1+1e−z is the sigmoid function. This formulation focuses solely on learning the relative ranking of responses, encouraging the model to assign higher scores to preferred answers without explicitly calibrating the absolute reward scores. Consequently, examples with equal or near-equal preferences are prone to introducing ambiguity and were therefore excluded from our training data.

#### 4 Experiment

###### 4.1 Training Details

Training Parameter. To efficiently fine-tune the model as a reward scorer, we adopt a partial parameter freezing strategy [42]. In particular, we freeze the entire visual encoder of Qwen2.5VL-7B-Instruct to preserve its visual abilities pretrained on massive image-text data. The trainable weights in Skywork-VL Reward are limited to the projector, language backbone, and the reward head.

Two-Stage Fine-Tuning Procedure. We formulate preference learning as a supervised learning task over the constructed preference dataset. The fine-tuning follows a two-stage training strategy. In the first stage, the model is trained exclusively on multimodal preference data, allowing it to develop strong vision-language alignment capabilities. In the second stage, we additionally incorporate

Table 3: Evaluation Results on VL-RewardBench.

Models Model Size General Hallucination Reasoning Overall Accuracy Macro Average Proprietary Models

Claude-3.5-Sonnet(2024-06-22) - 43.4 55.0 62.3 55.3 53.6 Gemini-1.5-Flash (2024-09-24) - 47.8 59.6 58.4 57.6 55.3 GPT-4o(2024-08-06) - 49.1 67.6 70.5 65.8 62.4

- Gemini-1.5-Pro(2024-09-24) - 50.8 72.5 64.2 67.2 62.5
- Gemini-2.0-flash-exp(2024-12) - 50.8 72.6 70.1 68.8 64.5 Open-Source Models

Qwen2-VL-7B-Instruct 7B 31.6 19.1 51.1 28.3 33.9 MAmmoTH-VL-8B 8B 36.0 40.0 52.0 42.2 42.7 Qwen2.5-VL-7B-Instruct 7B 43.4 42.0 63.0 48.0 49.5 InternVL3-8B 8B 60.6 44.0 62.3 57.0 55.6 IXC-2.5-Reward-7B 7B 80.3 65.3 60.4 66.3 68.6 Qwen2-VL-72B-Instruct 72B 38.1 32.8 58.0 39.5 43.0 Molmo-72B-0924 72B 33.9 42.3 54.9 44.1 43.7 QVQ-72B-Preview 72B 41.8 46.2 51.2 46.4 46.4 Qwen2.5-VL-72B-Instruct 72B 47.8 46.8 63.5 51.6 52.7 InternVL3-78B 78B 67.8 52.5 64.5 63.3 61.6

Skywork-VL Reward (Ours) 7B 66.0 80.0 61.0 73.1 69.0

pure-text preference data to further improve the model’s generalization and reasoning abilities in text-only scenarios.

We use AdamW [43] with a moderate learning rate for optimization in the first stage (10−5) and a lower learning rate in the second stage (10−6 ). And the model is fine-tuned for 2 epochs per stage, which we find sufficient for convergence.

###### 4.2 Evaluation Benchmarks

We evaluate Skywork-VL Reward on two benchmarks: VL-RewardBench [15] and RewardBench [23]. VL-RewardBench is designed to assess vision-language reward modeling. It contains 1,250 carefully curated examples spanning general multimodal queries, visual hallucination detection, and complex reasoning tasks involving images. RewardBench is a pure-text benchmark targeting reward functions for language models. This dataset includes prompt-chosen-rejected triplets covering a diverse range of topics within general chat, safety, and reasoning. We report the performance on both benchmarks, specifically focusing on their key evaluation dimensions and the aggregated overall accuracy.

###### 4.3 Baselines

For VL-RewardBench, we compare Skywork-VL Reward against a broad range of RMs, including both cutting-edge proprietary models and leading open-source alternatives. The proprietary multimodal RMs (closed-source) in our evaluation include GPT-4o [1], Claude 3.5 [44] with Vision, and Google Gemini 1.5 [2]. These models represent top-performing industrial models and serve as upper-bound references for RM performance. The involved prominent open-source models include Qwen2-VL-7B-Instruct [45], MAmmoTH-VL-8B [46], Qwen2.5-VL-7B-Instruct [13], InternVL3-8B [47], Qwen2-VL-72B-Instruct [45], IXC-2.5-Reward-7B [12], Molmo-72B [48], QVQ-72B-Preview [49], Qwen2.5-VL-72B-Instruct [13], and InternVL3-78B [47].

For RewardBench, we evaluate several advanced language-only RMs, including InternLM2-7BReward [24], Skywork-Reward-Llama3.1-8B [10], Skywork-Reward-Llama3.1-8B-v0.2 [10], and QRM-Llama3.1-8B-v2 [50]. We also evaluate multimodal RMs that are comparable in size to our own models including Qwen2-VL-7B-Instruct, InternVL3-8B, IXC-2.5-Reward-7B, and Qwen2.5VL-7B-Instruct.

In our experiments, Qwen2.5-VL-7B-Instruct, InternVL3-8B, Qwen2.5-VL-72B-Instruct, IXC-2.5Reward-7B, and InternVL3-78B were reproduced by ourselves, whereas the results for the remaining models were obtained from official reports.

Table 4: Evaluation Results on RewardBench.

Models Chat Chat Hard Safety Reasoning Avg Score Language-Only Reward Models

InternLM2-7B-Reward 99.2 69.5 87.2 94.5 87.6 Skywork-Reward-Llama3.1-8B 95.8 87.3 90.8 96.2 92.5 Skywork-Reward-Llama-3.1-8B-v0.2 94.7 88.4 92.7 96.7 93.1 QRM-Llama3.1-8B-v2 96.4 86.8 92.6 96.8 93.1

MultiModal Reward Models

Qwen2-VL-7B-Instruct 65.1 50.9 55.8 68.3 60.0 InternVL3-8B 97.2 50.4 83.6 83.9 78.8 Qwen2.5-VL-7B-Instruct 94.3 63.8 84.1 86.2 82.1 IXC-2.5-Reward-7B 90.8 83.8 87.8 90.0 88.1

Skywork-VL Reward (Ours) 90.0 87.5 91.1 91.8 90.1

###### 4.4 VL-RewardBench Evaluation

- Table 3 presents comparisons with both proprietary and open-source models on VL-RewardBench.

In the general category, skywork-VL Reward achieves a score of 66.0%, significantly outperforming even the strongest proprietary model, Gemini-2.0-flash-exp (50.8%). However, a gap remains compared to IXC-2.5-Reward-7B (80.3%). In the hallucination category, our model achieves the best score (80.0%), surpassing both proprietary models (e.g., Gemini-2.0-flash-exp at 72.6%) and the top-performing open-source model, IXC-2.5-Reward-7B (65.3%). This result highlights our model’s strong capability in mitigating factual inconsistencies. Our model also demonstrates robust performance in the reasoning category. It achieves a reasoning score of 61.0%, which is comparable to that of the much larger InternVL3-78B (64.5%), despite having 10× fewer parameters.

Our model achieves an overall accuracy of 73.1% and a Macro Average of 69.0%, demonstrating superior performance across diverse task types and surpassing the best proprietary model, Gemini2.0-flash-exp (68.8% overall accuracy and 64.5% Macro Average), and the second-best open-source model, IXC-2.5-Reward-7B (66.3% overall accuracy and 68.6% Macro Average). These results demonstrate the effectiveness of our method in providing reliable reward signals for multimodal tasks.

4.5 RewardBench Evaluation

- Table 4 reports the results on RewardBench, a language-focused reward benchmark.

Our model achieves an average score of 90.1% on RewardBench, achieve advanced performance among all open-source multimodal RMs of comparable scale and outperforming the second-best model, IXC-2.5-Reward-7B, by 2.0%. It also shows competitive performance against advanced language-specific RMs, such as QRM-Llama3.1-8B-v2 (93.1%).

Compared to other multimodal RMs of similar size, our model leads in Chat Hard (87.5%), Safety (91.1%), and Reasoning (91.8%), outperforming the second-best models by 3.7%, 3.3%, and 1.8%, respectively. These results highlight the robustness and well-rounded performance of our model across challenging and safety tasks. Furthermore, the results validate that our model is not only effective in handling multimodal data but also exhibits strong capabilities on pure-text inputs.

###### 4.6 Case Study

We present two illustrative examples that highlight the efficacy of our Skywork-VL Reward across distinct reasoning scenarios. For each example, we supply a multimodal prompt together with a good and a bad answer. For each given answer, Skywork-VL Reward produces a scalar reward value upon querying.

The first example (Figure 2) is a geometry problem that asks for the area of a circular sector. While both candidate answers arrive at the same numerical conclusion, the good answer showcases accurate reasoning in its derivation, unlike the bad answer’s rambling, self-corrective approach. This indicates that Skywork-VL Reward strongly favors the concise solution, demonstrating its sensitivity to the quality of reasoning rather than just the final correctness.

## Source: OneVision

[Figure 2]

|Question:<br><br>What is the area of the sector given the following choices: (A)𝜋/6 (B) 𝜋/4 (C)3𝜋/5 (D) 9𝜋/10<br><br>Hint:<br><br>Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.|
|---|

Good Answer :

Okay, let's see. The problem is asking for the area of a sector with a central angle of 10 degrees and a radius of 3 inches. Hmm, I remember that the formula for the area of a sector is something like(𝜃/360)multiplied by𝜋r squared. Let me make sure I recall that correctly. First, the central angle𝜃 here is 10 degrees. The radius r is given as 3 inches. So the area should be (10/360) times𝜋 times (3)^2. Let me write that out step by step to avoid mistakes. Calculating the fraction of the circle first: 10 divided by 360. That simplifies to 1/36, right? Because 10 dividedby10 is 1, and 360 divided by 10 is 36. So 10/360 = 1/36. Wait, is that correct? Let me check again. 10 divided by 360: yes, dividing numerator and denominator by 10 gives 1/36.Okay, that part seems right. Now, the area of the entire circle would be𝜋r², which is𝜋∗3 . 3 squared is 9, so that's 9𝜋. So multiplying the fraction by the area of the whole circle: (1/36) * 9𝜋. Let me compute that. 9 divided by 36 is 0.25, so that's 0.25𝜋? Wait, 9 divided by 36 is actually 1/4. Because 9 divided by 9 is 1, and 36 divided by 9 is 4. So yeah, 9/36 reduces to 1/4. Therefore, the area would be (1/4)𝜋. Hmm, but looking at the answer choices, option B is 𝜋/4.That seems to match. But let me double-check everything. Maybe I made a mistake in the formula. The formula for the area of a sector is indeed ∗𝜋𝑟 . So𝜃is 10 degrees, radius 3. So substituting in, it's ∗𝜋∗3 . Let's compute this step by step again. First, 10 divided by 360 is 1/36. Then 3 squared is 9. Multiply those together: (1/36)*9 = 9/36 = 1/4. So yes, 1/4 times𝜋 is𝜋/4. So that's answer choice B. Wait a second, but let me make sure that the angle is indeed in degrees and that I didn‘t mix up radians. The problem states the angle is 10 degrees, so that’s correct. If it were in radians, the formula would be different, but since it‘s specified in degrees, the formula I used is appropriate. Alternatively, maybe I should think in terms of proportions. A full circle is 360 degrees, so a 10-degree sector is 10/360 of the circle. The area of the circle is𝜋𝑟 = 𝜋 ∗ 9, so multiplying gives the sector area. Yep, that still gives 𝜋/4. Looking at the answer choices again:(A)𝜋/6(B)𝜋/4 (C) 3𝜋/5(D) 9𝜋/10. So𝜋/4 is option B. That seems right. Wait, but hold on, let me check if I messed up the multiplication somewhere. Let me recompute: 10 divided by 360 is 1/36. Then 1/36 multiplied by 9𝜋:9fivided by 36 is indeed 0.25, which is 1/4. So 1/4π. So yes, that's correct. Alternatively, maybe the question is a trick question? Let me check the figure again. The problem mentions a figure with a central angle of 10 degrees and radius 3 inches. The sector is the shaded part, which is 10 degrees. So there's no trick here. Hmm, I think that's solid. So the answer should be B,𝜋/4. </think> The area of the sector is calculated using the formula: \[ \text{Area} = \frac{\theta}{360^\circ} \times \pi r^2 \] Substituting the given values (\(\theta = 10^\circ\), \(r = 3\) in): \[ \text{Area} = \frac{10}{360} \times \pi \times 3^2 = \frac{1}{36} \times 9\pi = \frac{\pi}{4} \] Thus, the correct answer is **B**.

**Answer:** B ---**Final Answer** \boxed{B}

[Figure 3]

Reward Score: 5.86

Bad Answer:

Okay, let‘s see. Okay, let’s tackle this problem step by step. The question is asking for the area of a sector in a circle, and the options are given. First, I need to recall the formula for the area of a sector. I remember that the area of a sector is a fraction of the area of the entire circle. The formula is something like(𝜃/360)multiplied by𝜋r², where𝜃 is the central angle in degrees and r is the radius of the circle. Let me confirm that. So, the problem mentions a sector with a central angle of 10 degrees and a radius of 3 inches.Wait, the image shows a central angle of 10 degrees and the radius is labeled as 3 inches. That's the key information here. The fraction of the circle that the sector represents is𝜃 divided by 360 degrees. So that would be 10/360. Simplifying that fraction: both numerator and denominator are divisible by 10, so that becomes 1/36.Wait, 10 divided by 360 is indeed 1/36. Hmm, but let me check the arithmetic again. 10 divided by 360 is the same as 1/36. Yes, that's correct. Therefore, the area of the sector should be (1/36) multiplied by the area of the whole circle. The area of the whole circle is 9𝜋, so multiplying those together: (1/36)*9𝜋. Let me compute that. 9 divided by 36 is 0.25, so 0.25𝜋？ Wait, but 9 divided by 36 is actually 1/4.Wait, hold on. Let me recalculate:Wait, 9 divided by 36 is indeed 1/4? Wait, no. 9 divided by 36 is 0.25, which is 1/4.Wait, but 10 degrees over 360 is 1/36, so 1/36 times 9𝜋. Let me do the math again step by step. First, the area of the sector formula is ∗𝜋𝑟 . Plugging in the numbers:𝜃=10, so 10/360 = 1/36,r = 3 inches, so r² = 9. Therefore, the area is ∗ 𝜋 ∗ 9 = 𝜋. Wait, so that would be 𝜋/4？ But hold on,𝜋/4 is one of the options, option B. However, I need to check if I made a mistake here because sometimes there might be a different approach.Wait, 10 degrees is the central angle. The formula is ∗𝜋𝑟 . So 10 divided by 360 is indeed 1/36. Multiply that by𝜋∗3 which is 9𝜋. So ∗ 𝜋 ∗ 9 = 𝜋. So𝜋/4. That's correct. Butwait, let me check if the radius is indeed 3 inches. The problem states "3 in." which is the radius? The image might have the radius labeled as 3 inches. So yes, radius is 3. Therefore, the calculation seems right. Therefore, the area is𝜋/4, which is option B.But wait, let me check the options again to confirm. The options are (A)𝜋/6(B)𝜋/4 (C) 3𝜋/5(D) 9𝜋/10. Yes, B is𝜋/4. So that's the answer.Wait a second, but let me think again. Let me recast the problem in another way. Suppose the sector area formula is (1/2)*r²*𝜃, but in radians.Wait, that's another formula. So maybe if the angle was in radians, but here it's given in degrees, so that formula would require converting𝜃 to radians first.Wait, let me clarify the formulas. The area of a sector can be calculated in two ways depending on whether the central angle is in degrees or radians.Therefore, the answer should be𝜋/4, which is option B. Hmm, but let me check once more. Wait, let me compute the numerical value to see if it matches any of the options. Let's see: Option A:𝜋/6 ≈ 0.5236 Option B:𝜋/4 ≈ 0.7854Option C:3𝜋/5 ≈ 1.885Option D:9𝜋/10 ≈ 2.827. The area of the sector we calculated is𝜋/4 ≈ 0.7854, which matches option B. So that's correct. Thus, the area of the sector is \(\boxed{B}\).

[Figure 4]

Reward Score: -15.5 Explanation: The word ‘wait’ appears multiple times and is redundant.

Figure 2: Evaluating Skywork R1V on Mathematical Problems.

### Source: ChartQA

[Figure 5]

|Question:<br><br>Which country has longest bar?<br><br>Hint:<br><br>Please answer the question and provide the final answer at the end.|
|---|

Good Answer :

Okay, let me look at the data provided. The question is asking which country has the longest bar in the bar chart. The bars represent the projected share of the population in extreme poverty for 2023. First, I need to check the values for each country. The bars are color-coded, so each country's value is along a horizontal bar.The values are as follows:

-Nigeria: 43.54%-Extreme fragility: 31.44%-Africa: 29.05%-Fragile: 18.46%-World: 6.35%-India: 0.76% The length of each bar corresponds to these percentages. The longer the bar, the higher the percentage. Comparing all the values, Nigeria has the highest percentage at 43.54%, which would make its bar the longest. The next is Extreme fragility at 31.44%, then Africa at 29.05%, and so on. So, the country with the longest bar is Nigeria because it has the highest value. The answer should be straightforward based on the given data. The country with the longest bar is Nigeria, as it has the highest projected share of the population in extreme poverty at 43.54% in 2023. Answer: Nigeria.

[Figure 6]

Reward Score: 7.53

Bad Answer:

Okay, let's see. The user is asking which country has the longest bar in the given bar chart. The chart shows the projected share of the population in extreme poverty for 2023 across different countries and regions. The data is presented as percentages for each category. First, I need to look at the values for each country. The bars are horizontal, so the length of each bar corresponds to the percentage value. The countries listed are Nigeria, Extreme Fragility, Africa, Fragile, World, and India. The values are as follows:-Nigeria: 43.54%-Extreme Fragility: 31.44%-Africa: 29.05%-Fragile: 18.46%-World: 6.35%-India: 0.76% The longest bar would be the one with the highest percentage value. Comparing all the values, Nigeria has the highest at 43.54%,so that's the longest bar. The next one is Extreme Fragility at 31.44%, then Africa at 29.05%, and so on. The user might be confused if they think "longest" refers to physical length, but in a bar chart, the length is proportional to the value.So Nigeria's bar is the longest here. The country with the longest bar is Nigeria, with a projected share of 43.54% of its population in extreme poverty in 2023. This is the highest value among all the listed categories, making its bar the longest in the chart. Key Values for Clarity:-Nigeria: 43.54% -Extreme Fragility: 31.44% -Africa: 29.05% -Fragile: 18.46% -World: 6.35% -India: 0.76% The length of the bars directly corresponds to these percentage values,so Nigeria's bar is the longest.

[Figure 7]

Reward Score: -10.36 Explanation: Repeating the answer multiple times makes it verbose.

Figure 3: Evaluating Skywork R1V on Chart Problems.

- Table 5: Performance Evaluation on the MathVista Benchmark Using MPO with Different Reward Models.

Model Base Qwen2.5-VL-7B-Instruct InternVL3-8B Ours Performance (%) 69.2 71.2 71.8 73.5

The second example (Figure 3), involving identifying the country with the longest bar in an extreme poverty rate chart, further illustrates this. The good answer concisely states the label and cites relevant percentages, while the bad answer redundantly lists the same numbers. Again, Skywork-VL Reward strongly favors the compact explanation, demonstrating robustness across a different visual domain. These two cases, spanning distinct domains, highlight Skywork-VL Reward’s consistent ability to differentiate well-structured reasoning from verbose or confused discourse. The significant reward gap observed in both settings suggests the model captures a valuable alignment signal for downstream reinforcement learning.

###### 4.7 Skywork-VL Reward for Mixed Preference Optimization

We examine the effect of Skywork-VL Reward as a reward signal for MPO [16], a recent strategy to further improve model alignment. MPO refers to optimizing the behavior of a model using a mixture of preference signals rather than a single RM. This approach was proposed to stabilize and enhance training, especially for complex reasoning tasks that benefit from diverse feedback.

Moreover, preference data generated using our Skywork-VL reward demonstrates high effectiveness in training MPO, resulting in substantial gains in multimodal reasoning abilities. We leverage Skywork-VL Reward to generate preference data, which is subsequently used to fine-tune a VLM reasoner (the base model of Skywork R1V2 [7]) via MPO. Performance is evaluated on MathVista [51], a challenging benchmark for mathematical reasoning over visual content. As shown in Table 5, incorporating Skywork-VL Reward as an additional reward yields a notable improvement: the model’s MathVista score increases from 69.2% to 73.5%. This improvement demonstrates the potential of Skywork-VL Reward as a critical component in training VLMs capable of long-CoT reasoning.

#### 5 Conclusion

In this work, we introduce Skywork-VL Reward, a multimodal reward model for VLMs, aimed at addressing the critical need for reliable and general-purpose evaluators in multimodal understanding and reasoning tasks. Through the construction of a large-scale, meticulously curated preference dataset encompassing various tasks and scenarios, coupled with a two-stage training paradigm, our model is able to effectively assess responses generated by both standard VLMs and VLM reasoners. Empirical results demonstrate the state-of-the-art performance of Skywork-VL Reward on the VL-RewardBench benchmark and its competitive capabilities on the text-only RewardBench. Furthermore, integrating Skywork-VL Reward to provide supervised signals for MPO significantly enhances the multimodal reasoning abilities of VLMs, highlighting its practical value.

#### References

- [1] OpenAI. Gpt-4 technical report, 2023. 1, 6
- [2] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 6
- [3] OpenAI. Gpt-4o system card, 2024.
- [4] Gemini Team. Introducing gemini 2.0: our new ai model for the agentic era. https://blog.google/technology/google-deepmind/ google-gemini-ai-update-december-2024/#ceo-message, 2024.

- [5] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [6] Yi Peng, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, Li Ge, et al. Skywork r1v: Pioneering multimodal reasoning with chain-ofthought. arXiv preprint arXiv:2504.05599, 2025. 1, 2
- [7] Chris, Yichen Wei, Yi Peng, Xiaokun Wang, Weijie Qiu, Wei Shen, Tianyidan Xie, Jiangbo Pei, Jianhao Zhang, Yunzhuo Hao, Xuchen Song, Yang Liu, and Yahui Zhou. Skywork r1v2: Multimodal hybrid reinforcement learning for reasoning, 2025. 1, 10
- [8] Zhilin Wang, Yi Dong, Olivier Delalleau, Jiaqi Zeng, Gerald Shen, Daniel Egert, Jimmy J Zhang, Makesh Narsimhan Sreedhar, and Oleksii Kuchaiev. Helpsteer2: Open-source dataset for training top-performing reward models. arXiv preprint arXiv:2406.08673, 2024. 1
- [9] Rui Yang, Ruomeng Ding, Yong Lin, Huan Zhang, and Tong Zhang. Regularizing hidden states enables learning generalizable reward model for llms. arXiv preprint arXiv:2406.10216, 2024.

- 1, 2

[10] Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms, 2024.

- 1, 3, 6

- [11] Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark, 2025.
- [12] Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Ziyu Liu, Shengyuan Ding, Shenxi Wu, Yubo Ma, Haodong Duan, Wenwei Zhang, et al. Internlm-xcomposer2. 5-reward: A simple yet effective multi-modal reward model. arXiv preprint arXiv:2501.12368, 2025. 1, 2, 6
- [13] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923,

2025. 2, 5, 6

- [14] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 2
- [15] Lei Li, Yuancheng Wei, Zhihui Xie, Xuqing Yang, Yifan Song, Peiyi Wang, Chenxin An, Tianyu Liu, Sujian Li, Bill Yuchen Lin, Lingpeng Kong, and Qi Liu. Vlrewardbench: A challenging benchmark for vision-language generative reward models, 2024. 2, 6
- [16] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization, 2025. 2, 10
- [17] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 2
- [18] Gemma Team. Gemma. 2024.
- [19] Donald Joseph Hejna III and Dorsa Sadigh. Few-shot preference learning for human-in-the-loop rl. In Conference on Robot Learning, pages 2014–2025. PMLR, 2023.
- [20] Ali Narin. Evolutionary reward design and optimization with multimodal large language models. In Proceedings of the 3rd Workshop on Advances in Language and Vision Research (ALVR), pages 202–208, 2024.
- [21] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.

- [22] Jialun Zhong, Wei Shen, Yanzeng Li, Songyang Gao, Hua Lu, Yicheng Chen, Yang Zhang, Wei Zhou, Jinjie Gu, and Lei Zou. A comprehensive survey of reward models: Taxonomy, applications, challenges, and future. arXiv preprint arXiv:2504.12328, 2025. 2
- [23] Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, et al. Rewardbench: Evaluating reward models for language modeling. arXiv preprint arXiv:2403.13787, 2024. 2, 6
- [24] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, et al. Internlm2 technical report, 2024. 2, 6
- [25] Haoxiang Wang, Yong Lin, Wei Xiong, Rui Yang, Shizhe Diao, Shuang Qiu, Han Zhao, and Tong Zhang. Arithmetic control of llms for diverse user preferences: Directional preference alignment with multi-objective rewards. In ACL, 2024. 2
- [26] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023. 2
- [27] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training, 2025. 2
- [28] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024. 2
- [29] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. 2
- [30] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023. 2
- [31] Yassine Ouali, Adrian Bulat, Brais Martinez, and Georgios Tzimiropoulos. Clip-dpo: Visionlanguage models as a source of preference for fixing hallucinations in lvlms, 2024. 2
- [32] Tianyu Yu, Haoye Zhang, Qiming Li, Qixin Xu, Yuan Yao, Da Chen, Xiaoman Lu, Ganqu Cui, Yunkai Dang, Taiwen He, Xiaocheng Feng, Jun Song, Bo Zheng, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. Rlaif-v: Open-source ai feedback leads to super gpt-4v trustworthiness,

2024. 3

- [33] Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llava-critic: Learning to evaluate multimodal models, 2025. 2, 3
- [34] Shijian Deng, Wentian Zhao, Yu-Jhe Li, Kun Wan, Daniel Miranda, Ajinkya Kale, and Yapeng Tian. Efficient self-improvement in multimodal large language models: A model-level judgefree approach, 2024. 2
- [35] Yihe Deng, Pan Lu, Fan Yin, Ziniu Hu, Sheng Shen, Quanquan Gu, James Zou, Kai-Wei Chang, and Wei Wang. Enhancing large vision language models with self-training on image comprehension, 2024. 2
- [36] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer, and Trevor Darrell. Aligning large multimodal models with factually augmented rlhf, 2023. 2
- [37] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. 2

- [38] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, et al. Visualprm: An effective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291, 2025. 2
- [39] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271,

2024. 5

- [40] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. 5
- [41] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. 5
- [42] Rui Yang, Ruomeng Ding, Yong Lin, Huan Zhang, and Tong Zhang. Regularizing hidden states enables learning generalizable reward model for llms, 2024. 5
- [43] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. 6
- [44] Anthropic. Claude-3.5, 2024. 6
- [45] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191,

2024. 6

- [46] Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale, 2024. 6
- [47] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. 6
- [48] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models,

2024. 6

- [49] Qwen Team. Qvq: To see the world with wisdom, December 2024. 6
- [50] Nicolai Dorka. Quantile regression for distributional reward models in rlhf. arXiv preprint arXiv:2409.10164, 2024. 6
- [51] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 10

