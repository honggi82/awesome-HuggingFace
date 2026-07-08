## PaCo-RL: Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling

[Figure 1]

Bowen Ping1*, Chengyou Jia1*, Minnan Luo1†, Changliang Xia1, Xin Shen1, Zhuohang Dang1, Hangwei Qian2† 1School of Computer Science and Technology, MOEKLINNS Lab, Xi’an Jiaotong University 2CFAR and IHPC, A*STAR jayceping6@gmail.com, cp3jia@stu.xjtu.edu.cn

# arXiv:2512.04784v2[cs.CV]15Mar2026

Figure 1. Two Representative Tasks in Consistent Image Generation. In the image editing task, the model needs to modify specific attributes while preserving the overall appearance. In the text-to-image set generation task, the goal is to generate multiple coherent images that remain consistent in identity, style, and context under a unified description.

#### Abstract

by task-aware instructions and CoT reasons. The second component, PaCo-GRPO, leverages a novel resolutiondecoupled optimization strategy to substantially reduce RL cost, alongside a log-tamed multi-reward aggregation mechanism that ensures balanced and stable reward optimization. Extensive experiments across the two representative subtasks show that PaCo-Reward significantly improves alignment with human perceptions of visual consistency, and PaCo-GRPO achieves state-of-the-art consistency performance with improved training efficiency and stability. Together, these results highlight the promise of PaCo-RL as a practical and scalable solution for consistent image generation. Project page.

Consistent image generation requires faithfully preserving identities, styles, and logical coherence across multiple images, which is essential for applications such as storytelling and character design. Supervised training approaches struggle with this task due to the lack of largescale datasets capturing visual consistency and the complexity of modeling human perceptual preferences. In this paper, we argue that reinforcement learning (RL) offers a promising alternative by enabling models to learn complex and subjective visual criteria in a data-free manner. To achieve this, we introduce PaCo-RL, a comprehensive framework that combines a specialized consistency reward model with an efficient RL algorithm. The first component, PaCo-Reward, is a pairwise consistency evaluator trained on a large-scale dataset constructed via automated sub-figure pairing. It evaluates consistency through a generative, autoregressive scoring mechanism enhanced

#### 1. Introduction

Image generation has made remarkable progress in producing high-quality and diverse images from textual prompts [3, 12, 23, 25, 44, 56, 69, 70, 78, 85]. However, achieving consistent image generation remains both challenging and crucial for various applications such as story-

*Equal Contribution. †Corresponding author.

telling, advertising, and character creation [5, 22, 39, 42, 65, 97]. In this work, we study consistent image generation through two representative tasks: Image Editing [40] and Text-to-ImageSet [24], where the objective is to either align generated images with given references (Fig. 1, left) or produce a coherent set of images from a single prompt (Fig. 1, right). This is a challenging problem for supervised training approaches, due to the lack of large-scale datasets for visual consistency and the complexity of modeling human perceptions of consistency [7, 26, 36, 83, 93].

Recently, reinforcement learning (RL) has demonstrated strong potential in advancing image generation models [37, 88, 95], offering a promising pathway to address the above challenges. Instead of relying on explicit supervision from carefully curated datasets, RL algorithms optimize generative models using feedback from reward models (RMs), which are often trained to capture human preferences across diverse perceptual dimensions. This enables models to learn complex and subjective visual criteria [83, 87] in a data-free way. Despite its promise, applying RL to consistent image generation remains an unsolved challenge, primarily due to the absence of both (1) Suitable Reward Model and (2) Efficient RL Algorithm. On the reward side, existing reward models predominantly evaluate aesthetics and prompt alignment [41, 76, 80, 81], but the lack of explicit consistency evaluation makes them insufficient for consistent generation. On the optimization side, consistent generation is significantly more computationally demanding than singleimage synthesis and requires carefully balancing consistency preservation and prompt fidelity. However, current RL algorithms struggle to achieve this trade-off efficiently.

In this paper, we present a comprehensive methodology, Pairwise Consistency Reinforcement Learning (PaCoRL), to overcome the above challenges. It focuses on building a specialized, state-of-the-art consistency reward model and an efficient online RL algorithm for consistent image generation. In summary, our contributions are:

(1) PaCo-Reward. To build an effective and scalable reward model to assess visual consistency, we first design an automated data synthesis pipeline based on subfigure pairing, which supports the construction of a largescale, human-annotated consistency ranking dataset covering diverse image pairs with real-world consistency patterns. Leveraging this dataset, we introduce PaCo-Reward, which reformulates reward modeling as a generative task for pairwise comparisons. Rather than relying on an extra regression head, it aligns rewards with the next-token prediction process of the underlying vision-language model (VLM), mapping consistency scores to the probability of generating the “yes” token. To enhance generalization and interpretability, PaCo-Reward incorporates task-aware instructions and CoT-style reasons, providing a more robust and perceptually aligned consistency evaluator.

(2) PaCo-GRPO. Recent RL methods for image generation [31, 37, 88] achieve strong results but remain computationally heavy, particularly for consistency tasks requiring high resolutions or multiple images per prompt. To improve efficiency and stability, we introduce PaCo-GRPO, which integrates two core strategies: (1) a resolution-decoupled training scheme that trains on low-resolution images, while preserving full-resolution generation at inference, effectively reducing computational cost without compromising final performance; and (2) a log-tamed multi-reward aggregation mechanism that balances rewards to prevent domination and ensure stable training. Together, these techniques significantly boost training efficiency and stability without compromising performance, and can be combined with prior optimizations [31, 37] to make consistent image generation both effective and computationally practical.

We conduct evaluations across multiple benchmarks for RMs, where PaCo-Reward consistently outperforms existing reward models by a margin of 8.2%-15.0% in correlation with human preferences, demonstrating its superior alignment with human perceptions of visual consistency. Building on this strong reward foundation, we integrate PaCo-Reward into our PaCo-GRPO, which achieves stateof-the-art performance in both Text-to-ImageSet and Image Editing tasks, yielding substantial improvements of 10.3%11.7% in consistency metrics while preserving prompt fidelity, together with nearly doubled training efficiency and greater stability by keeping the reward ratio under 1.8 to avoid domination. These results underscore the broad potential of RL for advancing consistent image generation, paving the way for future research in this important area.

#### 2. Related Work

##### 2.1. Consistent Image Generation

Consistent Image Generation aims to produce images coherent with given contexts, constraints, or attributes, encompassing tasks such as Text-to-ImageSet [20, 24, 54, 62, 65] and Image Editing [40, 73, 81, 81] While differing in formulation, both share the goal of generating images faithful to shared identities, styles, or visual conditions. Textto-ImageSet methods rely on grouped or pairwise data with strong internal coherence [21, 22, 33, 39, 42, 97], whereas image editing approaches [2, 7, 71, 72, 78] learn controlled modifications from paired original-edited images, preserving fidelity while adjusting specific attributes.

Despite these differences, both task families face similar issues: dependence on curated paired or grouped data, narrow task-specific formulations, and limited generalization under complex or multi-condition constraints. In this work, we unify these under a single Consistent Image Generation framework, enabling PaCo-RL to enhance consistency across diverse generative scenarios.

##### 2.2. Reinforcement Learning for Image Generation

###### 2.2.1. Reward Models for Image Generation

Early works built reward models using existing image quality metrics or fine-tuned CLIP-based models on human preference datasets [7, 26, 36, 82, 84, 93]. Recent studies adopt multimodal large language models (MLLMs) as backbones for reward modeling [40, 41, 59, 76, 80, 81], demonstrating strong ability to understand complex human preferences and much greater flexibility across diverse evaluation tasks. However, these works most remain focused on general image-text alignment or aesthetic evaluation, largely ignoring the aspects of visual consistency.

In contrast, consistency-related preferences require pairwise image comparisons rather than simple image-text alignment or single-image quality assessment. Although models such as CLIP [52] and DreamSim [15] can assess image similarity, they are not designed to capture the multifaceted nature of human perception of consistency. To bridge this gap, we propose a pairwise reward modeling framework that effectively captures human preferences for consistency across multiple aspects.

###### 2.2.2. RL Algorithms for Image Generation

RL-style algorithms such as Proximal Policy Optimization (PPO) [4, 14, 17, 43, 58, 94], Direct Preference Optimization (DPO) [16, 34, 35, 38, 53, 67, 89, 91, 92], and Group Relative Policy Optimization (GRPO) [19, 31, 32, 37, 60, 68, 88] have emerged as powerful paradigms to enhance generative models [13, 48, 61, 63, 75, 90, 95], and have demonstrated significant success in aligning text-to-image models with human preferences [8, 11, 30, 49, 50, 87].

Recent methods like Flow-GRPO [37] and DanceGRPO [88] introduce stochasticity to flow-matching models by converting ordinary differential equations (ODEs) into stochastic differential equations (SDEs), improving sampling diversity. However, RL-based image generation remains computationally expensive, with sampling being a key bottleneck. Techniques such as MixGRPO [31, 37] reduce training costs by mixing SDE and ODE sampling, but sampling overhead persists, particularly for consistent image generation tasks that handle multiple images.

To mitigate this, we propose PaCo-GRPO, a resolutiondecoupled training strategy that samples low-resolution images during RL, significantly reducing computational demands. This approach complements existing methods like MixGRPO [31] and FlowGRPO-Fast [37], offering additional efficiency gains when combined.

#### 3. PaCo-Reward

To enable effective RL for consistent image generation, we first build PaCo-Dataset, a large-scale human-annotated dataset covering diverse consistency patterns, designed for

training and evaluating consistency reward models. Based on this dataset, we introduce PaCo-Reward, a novel framework for pairwise consistency reward modeling. It reformulates consistency evaluation as a generative task, accurately capturing human preferences for visual consistency while ensuring computational efficiency and interpretability.

##### 3.1. Paco-Dataset: Pairwise Consistency Dataset

Overview. To address the shortage of high-quality datasets for consistency reward modeling, we introduce PacoDataset, which spans 6 major categories and 32 subcategories of consistent image generation scenarios, including character generation, design style generation, and process generation, covering diverse types of visual consistency such as identity, style, and logic. The full categorization is provided in Appendix 11. Each entry consists of one reference image and four comparison images, accompanied by human-annotated rankings reflecting visual consistency. This dataset enables more effective training of reward models aligned with nuanced human preferences and includes a benchmark, ConsistencyRank, for standardized evaluation.

Data Synthesis. Consistent generation involves multiple images sharing common elements while varying in other aspects, making data collection particularly challenging. To efficiently construct a large-scale dataset with diverse consistency patterns, we first generate 2,000 text prompts for Text-to-ImageSet synthesis using Deepseek-V3.1 [10], then select 708 diverse prompts through graph-based diversification on text embeddings [52]. Inspired by [22], we use FLUX.1-dev [28] to generate m×n image grids with strong internal consistency. To enhance diversity and reduce cost, we generate four grids with different seeds per prompt and apply a sub-figure combinatorial pairing strategy, where we divide each grid into m×n subfigures and exhaustively pair them across grids from the same prompt. We empirically choose a 2 × 2 grid setting as it offers the best trade-off between quality and efficiency. This process yields 33,984 unique ranking instances from 708 prompts and 2,832 images, substantially expanding dataset scale and diversity at minimal cost. Each instance comprises one reference image and four comparison candidates to be ranked by consistency, as illustrated in Fig. 2 (top-left).

Data Annotation. To obtain reliable human preference data, we recruited six trained annotators, each labeling about 5,664 instances. Annotators were given basic guidelines on the task but made their rankings based on personal judgment of consistency with the reference image. A random subset of 3,136 instances is reserved as the ConsistencyRank benchmark for evaluating reward models

Directly training on ranking data is computationally demanding and often requires non-trivial architectural changes [81]. To retain both simplicity and generality, we convert rankings into pairwise comparisons (see Fig. 2, top-

[Figure 2]

Figure 2. Overview of the proposed PaCo-Reward framework.

score between two images, thus facilitating efficient pairwise consistency evaluation for RL training. Moreover, human preference rankings can be inferred from the consistency scores of candidate images relative to the reference image, as shown in Fig. 2 (bottom-left).

right), preserving rich supervision while providing clear positive and negative pairs for consistency evaluation. To further improve diversity and balance, we incorporate 5,695 manually verified consistent pairs from ShareGPT4o-Image [6]. In total, the dataset contains 54,624 annotated image pairs (27,599 consistent and 27,025 inconsistent), each labeled through human evaluation and augmented with CoT reasoning annotations generated by GPT-5 [45]. These reasoning annotations enhance dataset interpretability and help mitigate overfitting during Paco-Reward training.

Training Objective. Training solely on binary responses can lead to overfitting and limited generalization, while directly optimizing the full CoT sequence may weaken the main supervision signal. To balance these factors, we design a weighted likelihood objective. As shown in Fig. 2 (bottom-right), given two images IA and IB under a shared prompt P (collectively denoted as input I = (IA,IB,P)), the model generates a binary answer (“Yes” or “No”) followed by a CoT-style reasoning sequence of n − 1 token. The objective maximizes the weighted likelihood of the ground-truth response:

##### 3.2. PaCo-Reward: Pairwise Reward Modeling

Motivation. Consistency evaluation inherently requires comparing multiple images to assess their alignment on shared attributes. Recent works [40, 81] demonstrate that VLM-based reward models are effective for such multiimage, text-conditioned tasks. However, existing approaches have key limitations. Some introduce extra regression heads to output scalar rewards, which mismatch with the next-token prediction nature of VLMs. Others rely on long CoT reasoning to infer reward scores, causing high computational overhead during RL training.

n−1

(1 − α) n − 1

LPaCo = − α log p(y0 | I) +

log p(yi | I) ,

i=1

where y0 denotes the first token (“Yes” or “No”), yi is the i-th token of the reasoning sequence, and α ∈ [0,1] controls the balance between the decision token and reasoning supervision. When α = n1, the formulation reduces to standard maximum likelihood estimation. Through a hyperparameter search, we find that setting α = 0.1 achieves the best generalization performance in our setting.

Reward Modeling. Inspired by prior works [24, 88], we propose PaCo-Reward, a pairwise reward modeling framework that simplifies consistency learning. Rather than modeling full rankings, it learns relative preferences from pairwise comparisons to provide rich and efficient supervision. To align with the autoregressive nature of VLMs, PaCoReward further reformulates consistency evaluation as a generative task, predicting the likelihood of “Yes” or “No” to indicate the consistency between two images. This formulation naturally fits the autoregressive next-token prediction paradigm of VLMs and can be further enhanced with CoT reasoning to improve interpretability and robustness, as illustrated in Fig. 2 (top-right). During inference, the predicted probability of “Yes” serves as the consistency

#### 4. Paco-GRPO

##### 4.1. Preliminaries

Recently, FlowGRPO [37] and DanceGRPO [88] have successfully adapted Group Relative Policy Optimization (GRPO) [60] to image generation models. These methods introduce stochasticity into flow-matching models by converting deterministic ODEs into SDEs. The corresponding

Resolution-decoupled Sampling Reward Functions

Multi-rewards Aggregation

Multi-rewards

Generate a set of images depicting the same fox in different scenarios...

[Figure 3]

[Figure 4]

Naive Weighted Aggregation

1, 21, . . . , 1

[Figure 5]

1

1

[Figure 6]

- 1
- 2, 22, . . . , 2

2

Flow-Matching Generative Model

[Figure 7]

Log-tamed Aggregation

[Figure 8]

...

...

...

[Figure 9]

1 , 2 , . . . ,

Advantage Computation

Wow! Low-resolution images can drive effective RL!

Policy Optimization

= − ( |  ) 1, 2, . . . ,

Loss Computation

Figure 3. Overview of our proposed PaCo-GRPO framework on Text-to-ImageSet generation task.

RL training: (i) a resolution-decoupled training strategy that improves both sampling and training efficiency, and (ii) a log-tamed multi-reward aggregation scheme that adaptively mitigates reward domination and stabilizes optimization.

SDE update rule is:

√

σt2 2t

(xt + (1 − t)vθ) ∆t + σt

xt+∆t = xt + vθ +

∆tϵ,

Resolution-decoupled Training. In Text-to-ImageSet generation, models often produce high-resolution outputs containing multiple sub-figures, each matching the standard resolution of text-to-image tasks. This requirement dramatically increases computational cost, scaling quadratically with image resolution in Transformer-based architectures [66]. Such scaling makes RL particularly expensive.

where x is the image latent, t denotes the time step, vθ is the learned velocity field for xt at t, ϵ ∼ N(0,I) is standard Gaussian noise, and σt is the noise scale. This stochastic sampling promotes diversity and enhances exploration during RL training. The GRPO objective is then maximized:

Jθ = Jclip − βDKL(πθ||πref), Jclip = meani,t min(rti(θ)Aˆit,clip rti(θ),ε A ˆit) .

FlowGRPO [37] has observed that even low-quality images generated with fewer denoising steps can still provide effective reward signals for RL optimization. Building on this insight, we introduce a resolution-decoupled training strategy to improve efficiency. While the model generates high-resolution (h × w) images during inference and evaluation, it produces lower-resolution images (e.g., h2 × w2 ) during training for reward computation and optimization. This approach substantially reduces the overhead of both sampling and training, thereby accelerating RL optimization (see Fig. 4). It also integrates seamlessly with existing strategies [19, 31, 37] for further efficiency gains.

(1)

where Aˆit is the estimated advantage, rti(θ) is the ratio between the current policy πθ and the previous policy πθ

,

old

and the KL-loss DKL ensures that the updated policy πθ remains close to the reference policy πref.

Consistent image generation tasks often involve multiple images and require the simultaneous optimization of several reward signals, such as visual consistency and prompt alignment. This presents significant challenges in two aspects: (i) RL efficiency: optimizing large or multiple images substantially increases computational cost. Prior work [19, 31, 37] demonstrates that training can be reduced to 1-2 steps without compromising performance, yet sampling efficiency, which remains the primary computational bottleneck, is largely unaddressed. (ii) RL stability: multireward optimization is prone to reward domination [18, 55], where one signal can dominate others, resulting in suboptimal outcomes or destabilized training.

Log-tamed Multi-reward Aggregation During RL training [53, 57, 58, 60], multiple reward signals are often combined to guide models toward desired behaviors. For consistent image generation, both visual consistency and prompt alignment are essential for ensuring high-quality outputs. Given N input conditions {ci}Ni=1, the model generates a group of G samples {xji}Gj=1 for each condition ci. Each reward model Rk outputs a scalar score Rk(xji,ci), and the aggregated reward and advantage are computed as

This motivates the following question: Can we improve both sampling and training efficiency simultaneously while ensuring stable multi-reward optimization?

K

rˆij − meanj(ˆrij) stdj(ˆrij)

wkRk(xji,ci), Aˆji =

rˆij =

, (2)

- 4.2. PaCo-GRPO Strategies

k=1

To address these challenges, we propose PaCo-GRPO, which introduces two key strategies for efficient and stable

where wk is the weight for the k-th reward. However, this naive aggregation often suffers from reward domina-

tion [18, 55], where one reward dominates optimization, leading to suboptimal or even degenerate results.

This issue can be mitigated by manual weight tuning, but the process is labor-intensive and lacks generality. To address this, we propose a log-tamed aggregation strategy, which adaptively compresses rewards with large fluctuations, thus preventing them from overwhelming the overall optimization. Specifically, we first compute the coefficient of variation [77] for the k-th reward, denoted as hk:

stdi,j Rk(xji,ci) meani,j Rk(xji,ci)

hk =

. (3)

A high hk indicates that the k-th reward exhibits large fluctuations, leading to large absolute values in the aggregated reward and potentially dominating optimization. To temper such effects, we apply a logarithmic transformation:

Rk(xji,ci) =

log(1 + Rk(xji,ci)), if hk > δ, Rk(xji,ci), otherwise,

(4)

where δ is a threshold hyperparameter, which can be dynamically set as the mean of {hk}Kk=1 or as a fixed value (e.g., 0.2) based on prior knowledge. This transformation compresses large reward values while preserving the relative order of samples, thus mitigating reward domination without distorting the underlying preferences.

#### 5. Experiments

Our experiments are designed to address the following research questions (RQs):

- RQ1: Can PaCo-Reward better capture human preferences for visual consistency than existing methods?
- RQ2: Does integrating PaCo-Reward into RL training improve the performance of consistent image generation?
- RQ3: Do the proposed PaCo-GRPO strategies improve the efficiency and stability of RL training? 5.1. Experimental Setup

Our experiments consist of two stages: reward modeling evaluation and RL evaluation.

###### 5.1.1. Reward Modeling Evaluation Setup.

Models. We trained two variants of PaCo-Reward based on Qwen2.5-VL-7B-Instruct [1]: (1) PaCo-Reward-7B-Fast, trained with only binary labels for fast convergence; (2) PaCo-Reward-7B, trained on the full PaCo-Dataset with reasoning-augmented labels for improved performance.

Benchmarks. We evaluated PaCo-Reward on two benchmarks: (1) ConsistencyRank, which contains ∼3k humanannotated instances, each containing one reference image and four candidate images ranked by visual consistency. (2)

EditReward-Bench [81], comprising 3k pairs of original and edited images with human preference labels over Prompt Following (PF) and Consistency (C).

Baselines. We compared PaCo-Reward against state-ofthe-art reward models, including CLIP-I [52], DreamSim [15], InternVL3.5-8B [74], and Qwen2.5-VL-7B [1] on ConsistencyRank; and GPT-4.1 [47], GPT-5 [45], Gemini2.5-Pro [9], the Qwen2.5-VL series [1], and the EditScore series [40] on EditReward-Bench.

Metrics. We evaluate using Accuracy, Kendall’s rank correlation coefficient (τ), Spearman’s rank correlation coefficient (ρ), and Top-1-Bottom-1 (T1-B1) Accuracy on ConsistencyRank, and PF, C, and Overall (geometric mean of PF and C) on EditReward-Bench, following [40].

- 5.1.2. RL Evaluation Setup.

Models. For models, we used FLUX.1-dev [28] for Text-toImageSet generation, and FLUX.1-Kontext-dev [29] as well as Qwen-Image-Edit [78] for Image Editing tasks. PaCoGRPO was employed to train these models using PaCoReward-7B as the reward model. The prompt templates for each task are provided in Appendix 8.

Benchmarks and Metrics. For benchmarks, we conducted evaluations on T2IS-Bench [24] for Text-to-ImageSet and GEdit-Bench [78] for Image Editing. Evaluation metrics followed the corresponding benchmarks: comprehensive metrics from T2IS-Bench for quality and consistency, and Semantic Consistency (SC), Prompt Quality (PQ), Overall scores from GEdit-Bench for editing performance.

Table 1. Benchmark results on EditReward-Bench.

Method

Accuracy Prompt Following Consistency Overall

GPT-4.1 0.673 0.602 0.705 GPT-5 0.777 0.669 0.755 Gemini2.5-Pro 0.703 0.560 0.722 Qwen2.5-VL-7B 0.458 0.325 0.432 Qwen2.5-VL-32B 0.498 0.376 0.563 Qwen2.5-VL-72B 0.540 0.435 0.621 EditScore-7B 0.592 0.591 0.659 EditScore-32B 0.638 0.556 0.680 EditScore-72B 0.635 0.586 0.703 PaCo-Reward-7B-Fast 0.748 0.697 0.728 PaCo-Reward-7B 0.777 0.709 0.751

Table 2. Benchmark results on ConsistencyRank.

Method Accuracy ↑ τ ↑ ρ ↑ T1-B1 ↑

Clip-I 0.394 0.178 0.206 0.475 DreamSim 0.403 0.184 0.214 0.493 InternVL3.5-8B 0.359 0.149 0.176 0.420 Qwen2.5-VL-7B 0.344 0.118 0.138 0.401 PaCo-Reward-7B-Fast 0.441 0.240 0.278 0.544 PaCo-Reward-7B 0.449 0.250 0.288 0.557

- 5.2. Experimental Results

RA1: As shown in Tab. 1 and Tab. 2, PaCo-Reward effectively captures human preferences for visual con-

- Table 3. Comparisons with various Text-to-ImageSet generation methods on T2IS-Bench. Scores for Visual Consistency are evaluated by two independent evaluators, Qwen2.5-VL-7B and Gemma-3-4B (values before/after the slash), to ensure cross-model reliability.

Method Aesthetics

Prompt Alignment Visual Consistency

Avg. Entity Attribute Relation Identity Style Logic

Close Source GPT-4o 0.445 0.663 0.683 0.693 0.400 / 0.792 0.463 / 0.795 0.383 / 0.840 0.501 / 0.697 Gemini 2.0 Flash 0.430 0.738 0.747 0.743 0.428 / 0.750 0.383 / 0.739 0.392 / 0.793 0.509 / 0.673 AutoT2IS + GPT-4o 0.567 0.754 0.761 0.763 0.441 / 0.806 0.520 / 0.825 0.416 / 0.860 0.571 / 0.754 Seedream 4.0 0.551 0.813 0.800 0.805 0.486 / 0.809 0.479 / 0.763 0.458 / 0.862 0.589 / 0.758

Open Source Gemini & SD3 0.500 0.796 0.794 0.789 0.287 / 0.640 0.244 / 0.621 0.320 / 0.755 0.480 / 0.648 Gemini & Pixart 0.447 0.743 0.765 0.747 0.206 / 0.615 0.279 / 0.638 0.268 / 0.737 0.440 / 0.617 Gemini & Hunyuan 0.410 0.758 0.774 0.765 0.197 / 0.605 0.276 / 0.622 0.271 / 0.741 0.436 / 0.615 Gemini & FLUX.1-dev 0.533 0.791 0.790 0.786 0.249 / 0.636 0.302 / 0.619 0.328 / 0.754 0.490 / 0.659 AutoT2IS 0.520 0.729 0.756 0.743 0.359 / 0.723 0.414 / 0.737 0.356 / 0.794 0.515 / 0.686 FLUX.1-dev + PaCo-Reward-7B 0.555 0.721 0.732 0.731 0.508 / 0.837 0.549 / 0.867 0.422 / 0.859 0.576 / 0.757

- Table 4. Benchmark results on GEdit-Bench. EN-I and EN denote English instructions, while CN-I and CN denote Chinese instructions.

EN-I EN CN-I CN SC PQ Overall SC PQ Overall SC PQ Overall SC PQ Overall

Method

###### Close Source

SeedEdit3.0 [73] 7.396 7.899 7.137 7.222 7.885 6.983 7.370 7.870 7.105 7.195 7.851 6.942 GPT-Image-1 [46] 7.867 8.097 7.590 7.743 8.133 7.494 7.840 8.075 7.560 7.708 8.095 7.451

Open Source OmniGen [86] 6.037 5.856 5.154 5.879 5.871 5.005 6.015 5.830 5.122 5.850 5.845 4.976 OmniGen2 [79] - - - 6.72 7.20 6.28 - - - - - OmniGen2 + EditScore [40] - - - 7.20 7.46 6.68 - - - - - Step1X-Edit [81] 7.289 6.962 6.618 7.131 6.998 6.444 7.464 7.076 6.779 7.647 7.398 6.983 Step1X-Edit + EditReward [81] 7.895 6.946 7.131 7.854 6.931 7.086 7.757 7.024 7.074 7.658 6.995 7.001 FLUX.1-Kontext [29] 6.842 6.888 6.143 6.599 6.873 5.956 - - - - - FLUX.1-Kontext + PaCo-Reward-7B 7.279 7.036 6.636 7.033 7.088 6.469 - - - - - Qwen-Image-Edit [78] 7.746 7.805 7.307 7.642 7.807 7.223 7.727 7.977 7.344 7.633 7.938 7.264 Qwen-Image-Edit + PaCo-Reward-7B 7.799 8.053 7.451 7.693 7.987 7.325 7.866 8.125 7.520 7.678 8.093 7.366

3.0

512x512 384x384 256x256

Without Log-tame

0.400

2.8

With Log-tame Manual Tuning

0.38

Small Variance Large Variance

EvaluationReward

2.5

0.395

TrainingReward

RewardRatio

2.2

0.390

0.36

Reward Domination Mitigation

2.0

0.385

1024x1024 (12.0h)

0.34

1.8

512x512 (6.0h) 384x384 (5.7h) 256x256 (5.5h) Small Variance Large Variance

0.380

| |
|---|

| |
|---|

| |
|---|

| |
|---|

1.5

| |
|---|

0.375

| |
|---|

| |
|---|

0.32

1.2

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

Training Epoch

Training Epoch

Training Epoch

Figure 4. Training processes under different image resolutions.

Figure 5. Evaluation processes under different training image resolutions.

Figure 6. Ablation of log-tamed aggregation on the reward ratio.

sistency. On ConsistencyRank, advanced MLLMs such as InternVL3.5-8B and Qwen2.5-VL-7B perform worse than traditional similarity-based methods like CLIP-I and DreamSim, suggesting that current MLLMs remain misaligned with human perception of visual consistency and underscoring the need for specialized reward modeling. After fine-tuning on our PaCo-Dataset, both variants of PaCoReward consistently surpass all baselines. Notably, PaCoReward-7B achieves a 10.5% gain in Accuracy and a 0.150 increase in Spearman’s ρ over the original Qwen2.5-VL7B, highlighting the effectiveness of our pairwise reward

modeling paradigm in aligning with human preferences. Benefiting from the diverse and high-quality supervision provided by PaCo-Dataset, PaCo-Reward-7B attains highly competitive results on EditReward-Bench, outperforming all open-source baselines and approaching the performance of proprietary models such as GPT-5. These results demonstrate that PaCo-Reward generalizes effectively in modeling human preferences for diverse visual consistency.

RA2: PaCo-Reward improves consistency in RL-based image generation. We integrate PaCo-Reward-7B into PaCo-GRPO for both Text-to-ImageSet generation and Im-

Create four café menu displays with chalkboard font, featuring the following headings: “Fresh Brew”,“Daily Specials”, “Homemade” and “Sweet Treats”.

Generate four images depicting a progressive pencil drawing sequence of a young woman’s portrait.

Generate four images depicting the same dentist in scrubs in various medical scenarios.

[Figure 10]

[Figure 11]

[Figure 12]

Step0Step40Step80

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Figure 7. Progressive improvement of images generated with a fixed seed during PaCo-RL training for Text-to-ImageSet generation.

age Editing tasks, as shown in Tabs. 3 and 4. To ensure cross-model evaluation of visual consistency on T2ISBench, we additionally employ Gemma-3-4B-IT [64], since PaCo-Reward-7B is fine-tuned from Qwen2.5-VL-7B.

On T2IS-Bench, FLUX.1-dev enhanced with PaCoReward-7B substantially outperforms the strongest opensource baseline [24] across all visual consistency metrics, achieving average absolute gains of 0.117 under the Qwen2.5-VL-7B evaluator and 0.103 under the Gemma-34B-IT evaluator. Moreover, it reaches performance comparable to closed-source models such as GPT-4, narrowing the gap between open-source and closed-source solutions.

On GEdit-Bench, PaCo-Reward-7B consistently improves SC and PQ across all base models. Unlike EditReward [81], which degrades perceptual quality for Step1XEdit, our method achieves a balanced improvement, enhancing consistency while preserving image quality. Even for strong baselines such as Qwen-Image-Edit, PaCoReward-7B further boosts both consistency and perceptual quality. These gains stem from the strong reward signal provided by PaCo-Reward-7B, which guides RL optimization toward consistent yet visually coherent outputs.

RA3: PaCo-GRPO improves RL training efficiency and stability. To evaluate PaCo-GRPO’s effectiveness, we conduct ablation studies on its two key components using the challenging Text-to-ImageSet task.

Resolution-decoupled training: Fig. 4 illustrates training under different image resolutions. Each training image uses a 2 × 2 grid layout and is divided into four sub-images for reward computation, effectively halving the resolution in each dimension. As shown in Fig. 4, low-resolution training (512 × 512) starts with lower rewards but reaches the performance of 1024×1024 training after about 50 epochs, demonstrating that lower-quality images still provide reliable reward feedback [37]. Low-resolution training also shows higher reward variance (indicated by circles in Figs. 4 and 5), which promotes exploration and produces more diverse samples that enhance RL optimization. However, as shown in Fig. 5, 256 × 256 training fails due to insufficient visual detail for reliable reward evaluation. These results

indicate that moderate resolution reduction can greatly accelerate training without degrading final performance.

Log-tamed multi-reward aggregation: To assess the effect of log-tamed aggregation in mitigating reward domination, we compare it with standard weighted-sum aggregation (see Fig. 6). Two reward components are used: Consistency (PaCo-Reward-7B) and Prompt Alignment (CLIPT [52]). The plot shows the ratio of Consistency to Prompt Alignment rewards during training. Under standard aggregation without careful weight tuning, this ratio quickly exceeds 2.5 after 50 epochs, indicating that the Consistency reward dominates optimization. In contrast, log-tamed aggregation keeps the ratio below 1.8 throughout training, preventing any single reward from dominating and producing balanced improvements across all objectives.

##### 5.3. Case Studies

Fig. 7 visualizes progressive improvements achieved during PaCo-RL training with a fixed random seed. In the identity case, the dentist’s facial and hairstyle attributes gradually converge toward a consistent appearance across different medical scenarios. In the style case, the chalkboard font across café menus becomes increasingly uniform, with higher prompt fidelity and improved overall aesthetics. In the logic case, the model learns a correct sketch-to-drawing progression, where each subsequent panel extends and refines the previous one rather than altering it. Together, these cases demonstrate PaCo-RL’s capability to jointly enhance diverse dimensions of visual consistency, validating effectiveness of both reward model and RL framework.

#### 6. Conclusion

Our work introduces PaCo-RL, a novel reinforcement learning framework designed for consistent image generation. It consists of PaCo-Reward and PaCo-GRPO, which together enable more human-aligned and computationally efficient training for consistent image generation tasks.

#### Acknowledgment

This work is supported by the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM101), the National Natural Science Foundation of China (No. 62272374, No. 62192781), the Natural Science Foundation of Shaanxi Province (No.2024JC-JCQN-62), the State Key Laboratory of Communication Content Cognition under Grant No. A202502, the Key Research and Development Project in Shaanxi Province (No. 2023GXLH-024) and by A*STAR Career Development Fund <Project No. C243512010>.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, et al. Qwen2.5-vl technical report, 2025. 6
- [2] Stephen Batifol, Andreas Blattmann, Frederic Boesel, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv– 2506, 2025. 2, 1
- [3] James Betker, Gabriel Goh, Li Jing, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 1
- [4] Kevin Black, Michael Janner, Yilun Du, et al. Training diffusion models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024. 3
- [5] Dongping Chen, Ruoxi Chen, Shu Pu, et al. Interleaved scene graphs for interleaved text-and-image generation assessment, 2025. 2
- [6] Junying Chen, Zhenyang Cai, Pengcheng Chen, et al. Sharegpt-4o-image: Aligning multimodal models with gpt4o-level image generation, 2025. 4
- [7] Sherry X Chen, Yi Wei, Luowei Zhou, et al. Adiee: Automatic dataset creation and scorer for instruction-guided image editing evaluation. arXiv preprint arXiv:2507.07317,

2025. 2, 3

- [8] Kevin Clark, Paul Vicol, Kevin Swersky, et al. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023. 3
- [9] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. 6
- [10] DeepSeek-AI, Aixin Liu, Bei Feng, et al. Deepseek-v3 technical report, 2025. 3
- [11] Hanze Dong, Wei Xiong, Deepanshu Goyal, et al. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023. 3
- [12] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12873–12883, 2021. 1
- [13] Jiajun Fan, Shuaike Shen, Chaoran Cheng, et al. Online reward-weighted fine-tuning of flow matching with wasser-

- stein regularization. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [14] Ying Fan, Olivia Watkins, Yuqing Du, et al. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 3
- [15] Stephanie Fu, Netanel Yakir Tamir, Shobhita Sundaram, et al. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 3, 6
- [16] Hiroki Furuta, Heiga Zen, Dale Schuurmans, et al. Improving dynamic object interactions in text-to-video generation with ai feedback. arXiv preprint arXiv:2412.02617, 2024. 3
- [17] Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin, et al. A simple and effective reinforcement learning method for text-to-image diffusion fine-tuning. arXiv preprint arXiv:2503.00897, 2025. 3
- [18] Conor F Hayes, Roxana R˘adulescu, Eugenio Bargiacchi, et al. A practical guide to multi-objective reinforcement learning and planning. arXiv preprint arXiv:2103.09568,

2021. 5, 6

- [19] Xiaoxuan He, Siming Fu, Yuke Zhao, et al. Tempflow-grpo: When timing matters for grpo in flow models, 2025. 3, 5
- [20] Amir Hertz, Andrey Voynov, Shlomi Fruchter, et al. Style aligned image generation via shared attention, 2024. 2
- [21] Jiehui Huang, Xiao Dong, Wenhui Song, et al. Consistentid: Portrait generation with multimodal fine-grained identity preserving, 2024. 2
- [22] Lianghua Huang, Wei Wang, Zhi-Fan Wu, et al. In-context lora for diffusion transformers, 2024. 2, 3
- [23] Chengyou Jia, Minnan Luo, Zhuohang Dang, et al. Ssmg: Spatial-semantic map guided diffusion model for free-form layout-to-image generation, 2024. 1
- [24] Chengyou Jia, Xin Shen, Zhuohang Dang, et al. Why settle for one? text-to-imageset generation and evaluation, 2025. 2, 4, 6, 8
- [25] Chengyou Jia, Changliang Xia, Zhuohang Dang, et al. Chatgen: Automatic text-to-image generation from freestyle chatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13284–13293, 2025. 1
- [26] Yuval Kirstain, Adam Polyak, Uriel Singer, et al. Pick-apic: An open dataset of user preferences for text-to-image generation, 2023. 2, 3
- [27] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 1
- [28] Black Forest Labs. Announcing black forest labs - flux.1 [dev] suite released. https://bfl.ai/blog/24-0801-bfl, 2024. Accessed: 2025-10-29. 3, 6, 1
- [29] Black Forest Labs, Stephen Batifol, Andreas Blattmann, et al. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. 6, 7
- [30] Kimin Lee, Hao Liu, Moonkyung Ryu, et al. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. 3

- [31] Junzhe Li, Yutao Cui, Tao Huang, et al. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde, 2025. 2, 3, 5, 1
- [32] Yuming Li, Yikai Wang, Yuying Zhu, et al. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models, 2025. 3
- [33] Zhen Li, Mingdeng Cao, Xintao Wang, et al. Photomaker: Customizing realistic human photos via stacked id embedding. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [34] Zhanhao Liang, Yuhui Yuan, Shuyang Gu, et al. Step-aware preference optimization: Aligning preference with denoising performance at each step. arXiv preprint arXiv:2406.04314,

2024. 3

- [35] Zhanhao Liang, Yuhui Yuan, Shuyang Gu, et al. Aesthetic post-training diffusion models from generic preferences with step-by-step preference optimization. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13199–13208, 2025. 3
- [36] Zhiqiu Lin, Deepak Pathak, Baiqi Li, et al. Evaluating textto-visual generation with image-to-text generation. arXiv preprint arXiv:2404.01291, 2024. 2, 3
- [37] Jie Liu, Gongye Liu, Jiajun Liang, et al. Flow-grpo: Training flow matching models via online rl, 2025. 2, 3, 4, 5, 8, 1
- [38] Runtao Liu, Haoyu Wu, Zheng Ziqiang, et al. Videodpo: Omni-preference alignment for video diffusion generation. arXiv preprint arXiv:2412.14167, 2024. 3
- [39] Tao Liu, Kai Wang, Senmao Li, et al. One-prompt-one-story: Free-lunch consistent text-to-image generation using a single prompt. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [40] Xin Luo, Jiahao Wang, Chenyuan Wu, et al. Editscore: Unlocking online rl for image editing via high-fidelity reward modeling, 2025. 2, 3, 4, 6, 7, 1
- [41] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, et al. Hpsv3: Towards wide-spectrum human preference score. arXiv preprint arXiv:2508.03789, 2025. 2, 3
- [42] Jiawei Mao, Xiaoke Huang, Yunfei Xie, et al. Story-adapter: A training-free iterative framework for long story visualization, 2024. 2
- [43] Zichen Miao, Jiang Wang, Ze Wang, et al. Training diffusion models towards diverse image generation with reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10844–10853, 2024. 3
- [44] OpenAI. Introducing 4o image generation. https:// openai.com/index/introducing-4o-imagegeneration/, 2025. Accessed: 2025-03-25. 1
- [45] OpenAI. Introducing gpt-5. https://openai.com/ index/introducing-gpt-5/, 2025. Accessed: 202510-29. 4, 6
- [46] OpenAI. Gpt-image-1. https://platform.openai. com/docs/guides/image-generation?imagegeneration-model=gpt-image-1, 2025. OpenAI’s image generation model. Accessed September 2025. 7
- [47] OpenAI, Josh Achiam, Steven Adler, et al. Gpt-4 technical report, 2024. 6

- [48] Xue Bin Peng, Aviral Kumar, Grace Zhang, et al. Advantageweighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019. 3
- [49] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, et al. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023. 3
- [50] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, et al. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024. 3
- [51] Qwen, :, An Yang, et al. Qwen2.5 technical report, 2025. 1
- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, et al. Learning transferable visual models from natural language supervision, 2021. 3, 6, 8, 1
- [53] Rafael Rafailov, Archit Sharma, Eric Mitchell, et al. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024. 3, 5
- [54] Tanzila Rahman, Hsin-Ying Lee, Jian Ren, et al. Makea-story: Visual memory conditioned consistent story generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2493–2502,

2023. 2

- [55] Diederik M Roijers, Peter Vamplew, Shimon Whiteson, et al. A survey of multi-objective sequential decisionmaking. Journal of Artificial Intelligence Research, 48:67– 113, 2013. 5, 6
- [56] Chitwan Saharia, William Chan, Saurabh Saxena, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 1
- [57] John Schulman, Sergey Levine, Philipp Moritz, et al. Trust region policy optimization, 2017. 5
- [58] John Schulman, Filip Wolski, Prafulla Dhariwal, et al. Proximal policy optimization algorithms, 2017. 3, 5
- [59] Team Seedream, :, Yunpeng Chen, et al. Seedream 4.0: Toward next-generation multimodal image generation, 2025. 3
- [60] Zhihong Shao, Peiyi Wang, Qihao Zhu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. 3, 4, 5
- [61] Jiaming Song, Qinsheng Zhang, Hongxu Yin, et al. Lossguided diffusion models for plug-and-play controllable generation. In International Conference on Machine Learning, pages 32483–32498. PMLR, 2023. 3
- [62] Yiren Song, Cheng Liu, and Mike Zheng Shou. Makeanything: Harnessing diffusion transformers for multi-domain procedural sequence generation, 2025. 2
- [63] Zhiwei Tang, Jiangweizhi Peng, Jiasheng Tang, et al. Tuning-free alignment of diffusion models with direct noise optimization. arXiv preprint arXiv:2405.18881, 2024. 3
- [64] Gemma Team, Aishwarya Kamath, Johan Ferret, et al. Gemma 3 technical report, 2025. 8
- [65] Yoad Tewel, Omri Kaduri, Rinon Gal, et al. Training-free consistent text-to-image generation, 2024. 2
- [66] Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. Attention is all you need. Advances in neural information processing systems, 30, 2017. 5

- [67] Bram Wallace, Meihua Dang, Rafael Rafailov, et al. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024. 3
- [68] Feng Wang and Zihao Yu. Coefficients-preserving sampling for reinforcement learning with flow matching, 2025. 3
- [69] Jiyuan Wang, Chunyu Lin, Lang Nie, et al. Digging into contrastive learning for robust depth estimation with diffusion models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 4129–4137. ACM, 2024. 1
- [70] Jiyuan Wang, Chunyu Lin, Cheng Guan, et al. Jasmine: Harnessing diffusion prior for self-supervised depth estimation. arXiv preprint arXiv:2503.15905, 2025. 1
- [71] JiYuan Wang, Chunyu Lin, Lei Sun, et al. From editor to dense geometry estimator. arXiv preprint arXiv:2509.04338,

2025. 2

- [72] Jiyuan Wang, Chunyu Lin, Lei Sun, et al. Geometry-guided reinforcement learning for multi-view consistent 3d scene editing, 2026. 2
- [73] Peng Wang, Yichun Shi, Xiaochen Lian, et al. Seededit 3.0: Fast and high-quality generative image editing, 2025. 2, 7
- [74] Weiyun Wang, Zhangwei Gao, Lixin Gu, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025. 6
- [75] Yibin Wang, Zhimin Li, Yuhang Zang, et al. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning, 2025. 3
- [76] Yibin Wang, Yuhang Zang, Hao Li, et al. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025. 2, 3
- [77] Wikipedia contributors. Coefficient of variation Wikipedia, the free encyclopedia, 2025. Accessed: 202511-11. 6
- [78] Chenfei Wu, Jiahao Li, Jingren Zhou, et al. Qwen-image technical report, 2025. 1, 2, 6, 7
- [79] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, et al. Omnigen2: Exploration to advanced multimodal generation, 2025. 7
- [80] Jie Wu, Yu Gao, Zilyu Ye, et al. Rewarddance: Reward scaling in visual generation, 2025. 2, 3
- [81] Keming Wu, Sicong Jiang, Max Ku, et al. Editreward: A human-aligned reward model for instruction-guided image editing, 2025. 2, 3, 4, 6, 7, 8
- [82] Xiaoshi Wu, Yiming Hao, Keqiang Sun, et al. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023. 3
- [83] Xiaoshi Wu, Yiming Hao, Keqiang Sun, et al. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis, 2023. 2
- [84] Xiaoshi Wu, Keqiang Sun, Feng Zhu, et al. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105, 2023. 3
- [85] Changliang Xia, Chengyou Jia, Zhuohang Dang, et al. From ideal to real: Unified and data-efficient dense prediction for real-world scenarios, 2025. 1

- [86] Shitao Xiao, Yueze Wang, Junjie Zhou, et al. Omnigen: Unified image generation, 2024. 7
- [87] Jiazheng Xu, Xiao Liu, Yuchen Wu, et al. Imagereward: Learning and evaluating human preferences for textto-image generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2, 3
- [88] Zeyue Xue, Jie Wu, Yu Gao, et al. Dancegrpo: Unleashing grpo on visual generation, 2025. 2, 3, 4
- [89] Kai Yang, Jian Tao, Jiafei Lyu, et al. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8941–8951, 2024. 3
- [90] Po-Hung Yeh, Kuang-Huei Lee, and Jun-Cheng Chen. Training-free diffusion model alignment with sampling demons. arXiv preprint arXiv:2410.05760, 2024. 3
- [91] Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, et al. Self-play fine-tuning of diffusion models for text-to-image generation. arXiv preprint arXiv:2402.10210, 2024. 3
- [92] Jiacheng Zhang, Jie Wu, Weifeng Chen, et al. Onlinevpo: Align video diffusion model with online video-centric preference optimization. arXiv preprint arXiv:2412.15159,

2024. 3

- [93] Sixian Zhang, Bohan Wang, Junqiang Wu, et al. Learning multi-dimensional human preference for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8018– 8027, 2024. 2, 3
- [94] Hanyang Zhao, Haoxian Chen, Ji Zhang, et al. Score as action: Fine-tuning diffusion generative models by continuous-time reinforcement learning. arXiv preprint arXiv:2502.01819, 2025. 3
- [95] Kaiwen Zheng, Huayu Chen, Haotian Ye, et al. Diffusionnft: Online diffusion reinforcement with forward process, 2025. 2, 3
- [96] Yaowei Zheng, Richong Zhang, Junhao Zhang, et al. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. 1
- [97] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, et al. Storydiffusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems, 37:110315–110340, 2024. 2

Ref. Image

[Figure 19]

Ref. Image

[Figure 20]

Ref. Image

[Figure 21]

Ref. Image

[Figure 22]

“Make him look stronger”

[Figure 23]

“Make the action of the woman to laughing”

[Figure 24]

“Make him gain 20 pounds”

[Figure 25]

“Add abs to the original photo”

[Figure 26]

Figure 8. Training progression visualization for Image Editing with different prompts.

Trainingprogression

Prompt: “Depict the chronological decomposition of a single leaf on a forest floor. All images maintain a realistic style with consistent lighting and environmental elements, focusing on the gradual transformation of the leaf while adhering to natural decay processes. The forest floor setting includes subtle elements like soil texture, scattered debris, and occasional fungi or insects.”

[Figure 27]

Trainingprogression

Prompt: “Health beverage labels featuring honey drip font with viscous liquid texture and hexagonal comb patterns. All labels utilize the honey drip font style, integrating hexagonal comb motifs and natural/organic themes. Consistency in color palette (golden, amber, earthy tones) and texture emphasis ensures visual harmony across the set.”

[Figure 28]

Trainingprogression

Prompt: “Step-by-step progression of creating a cheerful chef emoji. All images use a minimalist, cartoonish style with a clean white background. Bright and cohesive color schemes unify the stages, maintaining continuity in character proportions and playful energy.”

[Figure 29]

Prompt: “Please generate four different perspective images of a 3D animated parrot with a vibrant and colorful plumage. The parrot exhibits a stunning array of colors, including shades of red, green, blue, and yellow, with detailed feather textures that reflect light and give a sense of depth...”

#### OursAutoT2ISSeedream

[Figure 30]

[Figure 31]

[Figure 32]

Figure 12. Comparison of different methods for multi-image generation.

Prompt: “Design product mockups featuring a retro, pixel art logo that reimagines our brand in an 8-bit style paired with a futuristic digital font. Apply the logo on 4 products: a portable gaming console, a vintage-style gaming t-shirt, a pixel art coffee mug, and a limited edition poster, using a monochromatic color scheme.”

#### AutoT2ISSeedreamOurs

[Figure 33]

[Figure 34]

[Figure 35]

Figure 13. Comparison of different methods for multi-image generation.

Prompt: “This artwork represents the gradual creation of a traditional Chinese ink painting featuring pumpkins and vines. The progression follows these steps:...”

#### SeedreamOursAutoT2IS

[Figure 36]

[Figure 37]

[Figure 38]

Figure 14. Comparison of different methods for Text-to-ImageSet generation.

## PaCo-RL: Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling

### Supplementary Material

#### 7. Implementation Details

Infrastructure. All experiments are conducted on a server equipped with eight NVIDIA H100 GPUs, each with 80 GB of memory.

PaCo-Reward Implementation Details. PaCo-Reward is fine-tuned using the LlamaFactory framework [96] with a customized weighted cross-entropy loss defined in Sec. 3.2. Both PaCo-Reward-7B-Fast and PaCo-Reward-7B are finetuned from Qwen2.5-VL-7B-Instruct [51], employing a LoRA rank of 32, α = 64, a learning rate of 2 × 10−4, and a batch size of 8. Training on the PaCo-Dataset for two epochs requires approximately 18 GPU hours. For PaCo-Reward-7B, the first-token weight α in Sec. 3.2 is set to 0.1 to emphasize the importance of the initial token. Since PaCo-Reward-7B-Fast is trained solely on binary labels (“Yes.”/“No.”), it does not apply token weighting.

PaCo-RL Implementation Details. During PaCo-RL training, one GPU is dedicated to running a vLLM [27] server for reward computation, while the remaining seven GPUs are used for reinforcement-learning fine-tuning. Unless otherwise specified, all experiments adopt a learning rate of 3 × 10−4, a batch size of 1, a group size of G = 16, and 42 unique samples per epoch. The clipping parameter ε in Eq. (1) is set to 1×10−4. Following [37], we monitor KL loss throughout training but assign it zero weight (β = 0) in Eq. (1) to achieve better performance.

Text-to-ImageSet. For the Text-to-ImageSet generation task, we fine-tune FLUX.1-dev [28] using LoRA with rank 64 and α = 128, and apply a classifier-free guidance (CFG) scale of 3.5 during both training and inference. The noise

scale σt in Sec. 4.1 is defined as σt = a 1−t t with a = 0.7, following FlowGRPO [37]. We use 10 denoising steps and perform SDE sampling only at timestep t = 1 (from 0,1,...,9), leveraging MixGRPO [31] and FlowGRPOFast [37] strategies for efficient training. In our PaCo-RL setup, the training resolution is 512 × 512 while the evaluation resolution is 1024×1024. We use δ = 0.2 in Eq. (4) to aggregate the Consistency Score (from PaCo-Reward-7B) and the Text-Image Alignment Score (from CLIP-T [52]).

Image Editing. For Image Editing, we fine-tune both FLUX.1-Kontext-dev [2] and Qwen-Image-Edit [78] using LoRA with rank 64 and α = 128. The CFG scales are set to 2.5 and 4.0 for FLUX.1-Kontext-dev and Qwen-ImageEdit, respectively. We apply the MixGRPO [31] strategy for efficient training, setting the noise scale a = 0.9 at timestep 1 for FLUX.1-Kontext-dev [2], and a = 1.0 for timesteps

1-4 for Qwen-Image-Edit [78]. The training and evaluation resolutions are 384 × 384 and 1024 × 1024, respectively. As the Image Editing task relies on a single reward signal, multi-reward aggregation and the log-tame strategy in Eq. (4) are not employed. The prompt template for Image Editing reward computation is provided in Sec. 8.

#### 8. Prompt Templates

We provide the prompt templates used for reward computation in both tasks below.

For the Text-to-ImageSet generation task, we design two versions of the prompt template: (1) one incorporating detailed consistency criteria derived from the original dataset for enhanced evaluation reliability, and (2) another containing only the input prompt information, which is used for generalization evaluation.

For the Image Editing task, we adopt a modified version of the prompt template from EditScore [40]. This template consists of two components: Semantic Consistency (SC) and Prompt Following (PF).

Prompt Template for Text-to-ImageSet (v1)

Do images meet the following criteria? {consistency_criteria} Please answer “Yes” or “No” first, then provide detailed reasons.

###### Prompt Template for Text-to-ImageSet (v2)

Given two subfigures generated based on the theme: {main_prompt}

do the two images maintain consistency in terms of style, logic and identity? Answer “Yes” or “No” first, and then provide detailed reasons.

###### Prompt Template for Image Editing (SC)

Compare the edited image (second) with the original image (first). Instruction: {prompt}. Except for the parts that are intentionally changed according to the instruction, does the edited image remain consistent with the original in style, logic, and identity? Answer ’Yes’ or ’No’ first, then provide detailed reasons.

###### Prompt Template for Image Editing (PF)

Compare the edited image (second) with the original image (first). Instruction: {prompt}. Does the edited image accurately follow this instruction? Answer ’Yes’ or ’No’ first, then provide detailed reasons.

#### 9. Resolution-Decoupled Training Analysis

To validate the reliability of resolution-decoupled training, we conduct experiments generating image sets at three resolutions: 256 × 256 (0.25x), 512 × 512 (0.5x), and 1024×1024 (1x) using FLUX.1-dev, and analyze the Pearson correlation of evaluation metrics across resolutions.

- 0

- 1

- 2

- 3

0.5xResolution

1

Pickscore ( =0.848)

2

Clip ( =0.753)

Consistency ( =0.725)

3

y=x

3 2 1 0 1 2 3 1x Resolution

- 0

- 1

- 2

- 3

0.25xResolution

1

Pickscore ( =0.771)

2

Clip ( =0.765)

Consistency ( =0.669)

3

y=x

3 2 1 0 1 2 3 1x Resolution

(a) 1x vs. 0.5x

(b) 1x vs. 0.25x

Figure 15. Pearson correlation of evaluation metrics across different training-to-inference resolution ratios. Strong correlations at 0.5x confirm that reward signals remain reliable under moderate resolution reduction.

As shown in Fig. 15, all metrics exhibit strong positive correlations (0.725–0.848) when comparing 1x vs. 0.5x resolutions, indicating that the relative quality ordering of image sets is largely preserved and reward signals remain reliable under moderate resolution reduction. However, correlations weaken substantially when comparing 1x vs. 0.25x, especially for Aesthetics and Consistency, suggesting that extremely low resolutions lose detailed visual information crucial for these fine-grained aspects.

We further validated these findings on additional reward models (Text-Rendering, GenEval) and generators (QwenImage, FLUX2), confirming that 0.5x reduction consistently preserves reward reliability while 0.25x fails, particularly for fine-grained metrics. Regarding higher-resolution inference, no systematic artifacts were observed when generating at full resolution after 0.5x training, as the reward improvements transfer effectively across resolutions. This supports the use of resolution-decoupled training as a practical strategy for reducing computational cost without sacrificing final generation quality.

#### 10. Ablation on PaCo-GRPO Components

We quantify the individual contributions of two key components in PaCo-GRPO: resolution-decoupled training (ResDec.) and log-tamed reward aggregation (Log-Agg.). Results are reported in Tab. 5.

Resolution-Decoupled Training. Removing resolutiondecoupled training results in suboptimal performance even

Table 5. Ablation study on PaCo-GRPO components. Removing resolution-decoupled training leads to suboptimal performance even with doubled training time, while removing log-tamed aggregation causes reward collapse.

Method Aes.↑ P.F.↑ V.C.↑ Time↓ Full PaCo-GRPO 0.555 0.728 0.493 6.0h W/O Res-Dec. 0.542 0.698 0.452 12.0h W/O Log-Agg. 0.471 0.616 0.557 6.0h

with doubled training time (12h vs. 6h), as the model fails to converge fully at higher resolution. This demonstrates that training at a reduced resolution (0.5x) not only halves the computational cost but also leads to better optimization dynamics by enabling more effective exploration within the same time budget.

Log-Tamed Reward Aggregation. Without log-tamed aggregation, the visual consistency reward dominates the training signal, causing the model to collapse into generating near-identical, low-quality image sets. While the Visual Consistency (V.C.) score increases to 0.557, both Aesthetics and Prompt Following degrade significantly, confirming that the log-tamed strategy in Eq. (4) is essential for balancing multiple reward objectives and preventing reward hacking.

#### 11. PaCo-Dataset Details

To ensure the quality and consistency of PaCo-Dataset annotations, all annotators followed detailed guidelines accompanied by illustrative examples to establish a shared understanding of the evaluation criteria across different consistency dimensions (i.e., style, logic, and identity). Each image pair was independently evaluated by multiple annotators, and the final annotations were obtained via majority voting. Ties were discarded to ensure clear binary preferences in the dataset.

The main categories, subcategories, and their corresponding consistency dimensions are summarized in Tab. 6.

Table 6. Main categories, subcategories, and their corresponding consistency dimensions.

###### Main Category Subcategory Consistency Dimensions

Home Decoration Style IP Product Style, Identity Font Design Style Poster Design Style, Logic Creative Style Style

Design Style Generation

Children Book Logic, Identity, Style Hist. Narrative Logic, Identity Movie Shot Logic, Identity, Style Comic Story Logic, Identity, Style News Illustration Logic, Style

Story Generation

Evolution Illustration Logic Draw progression Logic, Style Growth progression Logic Arch. Building Logic Cooking progression Logic Physical Law Logic

progression Generation

Historical Panel Logic, Style Activity Arrange Logic Evolution Illustration Logic Education Illustration Logic, Style Travel Guide Logic, Style, Identity Product Instruction Logic, Style

Instruction Generation

Multi-view Identity, Style Multi-pose Identity Portrait Design Identity, Style Multi-Expression Identity Multi-Scenario Identity, Logic

Character Generation

Inpainting and replacement Identity Element manipulation Identity, Style Background modification Identity, Style, Logic Attribute and effect manipulation Style Image editing and manipulation Identity, Style, Logic

Editing

