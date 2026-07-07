# arXiv:2509.08826v1[cs.CV]10Sep2025

[Figure 1]

## RewardDance: Reward Scaling in Visual Generation

### Jie Wu∗,† Yu Gao∗ Zilyu Ye Ming Li Liang Li Hanzhong Guo Jie Liu Zeyue Xue Xiaoxia Hou Wei Liu Yan Zeng Weilin Huang‡

ByteDance Seed ∗Equal contribution, ‡Corresponding authors, †Project lead

### Abstract

Reward Models (RMs) are critical for improving generation models via Reinforcement Learning (RL), yet the RM scaling paradigm in visual generation remains largely unexplored. It primarily due to fundamental limitations in existing approaches: CLIP-based RMs suffer from architectural and input modality constraints, while prevalent Bradley-Terry losses are fundamentally misaligned with the next-token prediction mechanism of Vision-Language Models (VLMs), hindering effective scaling. More critically, the RLHF optimization process is plagued by Reward Hacking issue, where models exploit flaws in the reward signal without improving true quality. To address these challenges, we introduce RewardDance, a scalable reward modeling framework that overcomes these barriers through a novel generative reward paradigm. By reformulating the reward score as the model’s probability of predicting a "yes" token, indicating that the generated image outperforms a reference image according to specific criteria – RewardDance intrinsically aligns reward objectives with VLM architectures. This alignment unlocks scaling across two dimensions: (1) Model Scaling: Systematic scaling of RMs up to 26 billion parameters; (2) Context Scaling: Integration of task-specific instructions, reference examples, and chain-of-thought (CoT) reasoning. Extensive experiments demonstrate that RewardDance significantly surpasses state-of-the-art methods in text-to-image, text-to-video, and image-to-video generation. Crucially, we resolve the persistent challenge of "reward hacking": Our large-scale RMs exhibit and maintain high reward variance during RL fine-tuning, proving their resistance to hacking and ability to produce diverse, high-quality outputs. It greatly relieves the mode collapse problem that plagues smaller models.

###### Seedream-3.0 (T2I)

###### Seedance-1.0 (T2V)

86

Small reward Std. +49.0%

Small reward Std. Large reward Std.

50

Large reward Std.

84

AlignmentGSBImprovement(%)

84.8

45

82

+45.0%

AlignmentScore

40

81.6

+41.0%

80

79.5

35

78

+32.0%

76

30

75.3

74.9

+28.0%

74

1B 2B 4B 8B 26B

1B 2B 4B 8B 26B

Reward Model Size (Billions of Parameters)

Reward Model Size (Billions of Parameters)

Figure 1 RewardDance consistently boosts T2I and T2V generation (validated by alignment/GSB). The reward variance during the later stages of RL training, represented by bubble size, serves as an indicator of policy "hacking." Low variance implies mode collapse, where the model tends to generate uniform, high-reward outputs. High variance signifies the policy maintains output diversity across various prompts, indicating that it has not collapsed.

### 1 Introduction

Diffusion models have achieved immense progress in visual generation. State-of-the-art models such as FLUX [26] and Seedream [14, 16] for image generation, alongside Wan2.1 [56] and Seedance [15] for video generation, have unlocked vast creative potential. These capabilities are further enhanced by paradigms like Reinforcement Learning (RL) [11, 13, 18, 20, 28, 29, 34, 55, 67] and Test-time Scaling [39], where the Reward Model (RM) plays a pivotal role. While a robust and accurate RM can significantly improve generation quality, the community has lacked clear guidance on designing superior RMs. In this paper, we introduce RewardDance, a new framework built on the principle that scalability is the key to creating better visual RMs.

|Methods|Task<br><br>|Aligning Stage<br><br>|Base Model<br><br>|Modeling Paradigm<br><br>|Model Scaling|Context Scaling<br><br>Task Instruction Reference Examples CoT Data|
|---|---|---|---|---|---|---|
|ImageReward [64] PickScore [23] HPSv2 [63] VisionReward [65] VideoAlign [35] HPSv3 [40] WorldPM [57] DeepSeek-GRM [36] Pairwise RM [66] UnifiedReward [59] RewardDance<br><br>|Visual Visual Visual Visual Visual Visual Understanding Understanding Understanding Multimodal Visual|RL RL RL RL RL RL&Infer RL Infer RL RL&Infer<br><br>|CLIP CLIP CLIP VLM VLM VLM VLM VLM VLM VLM VLM<br><br>|Regressive Regressive Regressive Regressive Regressive Regressive Regressive Generative Generative Generative Generative|× × × × × ✓ ✓ × × × ✓<br><br>|× × × × × × × × × ✓ × × ✓ × × ✓ × × ✓ × × ✓ ✓ × ✓ × × ✓ ✓ ✓ ✓ ✓ ✓<br><br>|

Table 1 A comprehensive comparison of visual and multimodal reward models. Our RewardDance is the first framework for visual generation to successfully integrate a generative paradigm with comprehensive scalability across both reward model size and reward context dimensions (task instructions, reference examples, and CoT data).

Our work is motivated by a comprehensive analysis of existing reward modeling methods, as summarized in

- Table 1. Early CLIP-based reward models [63, 64] were limited by CLIP’s architecture, which was difficult to scale [31] and generalized poorly to diverse tasks [30]. Later VLM-based models explored new paradigms and scaling strategies, but progress has been fragmented: some achieve large-scale models yet remain limited to regressive paradigms (e.g., HPSv3 [40] , WorldPM [57]), while others adopt stronger generative paradigms without effective scaling (e.g., UnifiedReward [59]). As shown in the Figure 2 , the regression-based reward model is very susceptible to reward hacking.

###### Regressive RM (2B)

###### Generative RM (2B)

###### Generative RM (26B)

- 0.8

- 1.0

1.0

1.0

|=6.<br><br>|1e-3| | |
|---|---|---|---|
| | | | |
|Raw Re Smoot Small Large|ward Sco hed Rewa Reward V Reward V|re rd Score<br><br>ariance ariance| |
| | | | |
| | | | |

=4.2e-3

###### =2.1e-2

=7.2e-3

0.8

0.8

=5.4e-2

=5.8e-2

AlignmentScore

AlignmentScore

AlignmentScore

- 0.6

0.6

0.6

- 0.4

0.4

0.4

0.2

0.2

0.2

Raw Reward Score

Raw Reward Score

Smoothed Reward Score

Smoothed Reward Score

Small Reward Variance Large Reward Variance

Small Reward Variance Large Reward Variance

0.0

0.0

0.0

0 1000 2000 3000 4000 5000 6000 7000 8000 RL Iteration

0 1000 2000 3000 4000 5000 6000 7000 8000 RL Iteration

0 1000 2000 3000 4000 5000 6000 7000 8000 RL Iteration

- Figure 2 Comparison of training dynamics for Regressive vs. Generative reward models during diffusion RL fine-tuning. At the same 2B model scale (Left vs. Middle panel), the generative reward model exhibits significantly superior training dynamics compared to the regression-based one: it facilitates higher exploration magnitude, manifested as greater reward variance, and a more favorable reward growth trend. This higher diversity in reward signals indicates that the generative RM exhibits stronger robustness against reward hacking. Under a regression-based RM, the diffusion model risks learning to exploit reward loopholes to achieve high scores without making substantive progress. This inherent robustness is key to the generative RM’s successful scaling to 26B parameters (Right panel).

We introduce RewardDance to address above challenges. It represents the first framework designed to achieve this unification, leveraging a generative paradigm that enables effective scaling across both model size and

context—including task-specific instructions, reference examples, and Chain-of-Thought (CoT) reasoning—to unlock the full potential of VLMs for advanced visual reward modeling.

Our RewardDance framework resolves this fundamental paradigm mismatch by rethinking reward modeling as a generative task. Instead of appending a regression head, we convert the reward score into the VLM’s predicted probability of a "yes" token. This approach natively aligns the reward objective with the VLM’s autoregressive, next-token prediction mechanism, thereby enabling effective reward scaling along two primary dimensions:

- • Model Scaling: We break from the conventional practice of using smaller, fixed-size models by systematically scaling our RM with VLMs of increasing capacity (1B to 26B parameters). This directly links model parameter count to both reward modeling performance and final generation quality.
- • Context Scaling: In contrast to traditional methods that rely on simple image-text pairs, RewardDance enriches input context by incorporating task-aware instructions, reference examples, and CoT reasoning data, enabling more robust and accurate reward judgments.

To validate RewardDance, we conducted extensive experiments on both open-source and proprietary models across a range of tasks, including text-to-image, text-to-video, and image-to-video generation. We observed that as the parameter of the Reward Model (RM) increases, the diffusion model exhibits stronger exploratory capacity at higher RLHF iteration steps and is less susceptible to reward hacking. Moreover, incorporating additional context information significantly enhances model performance and robustness. Crucially, our findings reveal a pronounced scaling effect: under both RL and test-time scaling paradigms, visual generation quality consistently and stably improves in direct correspondence with reward model enhancement.

Our contributions are summarized as:

- • Scalability as the Principle for RMs: We establish scalability as the fundamental design principle for visual RMs, addressing a critical yet underexplored dimension in existing work and providing new insights for the field. More importantly, our experiments demonstrate that scaling up the Reward Model (RM) enables less reward hacking (Fig. 2) and further achieves significant improvements in generation quality (Fig. 1).
- • Generative Reward Modeling: We propose a novel generative reward modeling paradigm that resolves the fundamental mismatch in prior work by regarding reward prediction as a next-token prediction task. This naturally aligns with VLMs’ autogressive mechanism, unlocking the potential for effective reward scaling.
- • Comprehensive Reward Scaling: We propose and validate a holistic methodology for scaling RMs across two dimensions: reward model size (from 1B to 26B parameters) and reward context (including task-aware instructions, reference examples, and CoT reasoning). We are the first to systematically demonstrate that enhancing these dimensions yields stable and consistent scaling effects, where improved reward models directly contribute to higher-quality visual generation.

### 2 Related Work

##### 2.1 Diffusion Models

Generalized diffusion models encompass both traditional diffusion processes [21, 52, 53] and flow-based models [33]. These methods learn to map a simple prior distribution, typically a Gaussian, to a target data distribution by reversing a learned noise-injection process. Owing to their effectiveness in modeling complex, high-dimensional data, diffusion models have demonstrated remarkable performance across a wide range of tasks, including image [12, 14, 16, 26, 32, 44, 49, 71] and video [1, 4, 6, 8, 15, 19, 25, 38, 51, 56, 69, 70] generation.

##### 2.2 Reward Models

Reward Models (RMs) play a pivotal role in aligning the outputs of visual generation models with human preferences [14–16, 48, 71]. Early approaches, such as ImageReward [64], PickScore [23], and HPSv2 [63], fine-tuned CLIP models to produce scores that reflect human preferences. Subsequent works [35, 65] adopted

Vision-Language Models (VLMs) as backbones, typically appending a regression head to output reward signals. These methods, predominantly trained with the Bradley–Terry loss [5, 43, 72], are categorized as regression-based RMs. More recently, advancements in LLMs and multimodal learning have inspired new paradigms. For instance, WorldPM [57] investigated the scaling potential of RMs through enhanced data and model architectures. Concurrently, a new class of generative reward models was proposed [36, 59, 66], aiming to better leverage the native capabilities of pre-trained models. Building upon these developments, our work introduces the generative RM paradigm to the visual domain and, for the first time, systematically explores its scaling properties with respect to both reward model size and reward context.

##### 2.3 Reinforcement Learning from Human Feedback

Reinforcement Learning from Human Feedback (RLHF) has been increasingly adopted in diffusion models to align generated outputs with human preferences. The typical pipeline involves training a reward model on human preference data and leveraging it to guide the generative model. For instance, DDPO [3] adapts Proximal Policy Optimization (PPO) to diffusion frameworks by computing image log-likelihoods. In contrast, ReFL [64] circumvents likelihood computation challenges by directly optimizing diffusion outputs through gradients from a frozen reward model. Beyond diffusion models, recent works [34, 35, 67, 68] have extended RLHF techniques to flow-based generative models, demonstrating the broader applicability of preference-based optimization in generative modeling. Complementing training-time alignment methods, test-time optimization techniques have emerged to enhance diffusion model performance during inference without requiring model retraining. These approaches [35, 39, 42] typically involve reward-guided adjustments to noise prediction outputs or dynamic modifications to sampling schedules, thereby improving generation quality in a plug-andplay manner. In this work, we integrate our RewardDance framework into both reinforcement learning and inference-time optimization stages, demonstrating its effectiveness in aligning generated content with human preferences across diverse visual generation tasks.

### 3 Method

This section introduces RewardDance, our framework for scalable visual reward modeling. We first detail its core generative paradigm, then describe our methodologies for scaling it along two primary dimensions: context and model size. Finally, we outline the training pipeline in aligning diffusion models.

##### 3.1 Preliminary

As illustrated in Figure 3 (top panel), conventional reward modeling approaches follow a pointwise regressive paradigm. The term "pointwise" indicates that these RMs evaluate individual images independently, while "regressive" refers to the scalar reward prediction mechanism. This paradigm encompasses both CLIP-based models, which compute rewards through cosine similarity between CLIP image and text embeddings, and VLM-based approaches, which typically employ additional regression heads to map VLM hidden states to scalar reward values. These models are commonly optimized using preference pairs with the Bradley-Terry (BT) loss:

LBT = −E(y,xw,xl)∼D log σ r(xw,y) − r(xl,y) , (1)

where y is the input prompt, xw and xl are the chosen (better) and rejected (worse) images, r(·,·) is the reward model, and σ(·) is the sigmoid function. These approaches ultimately provide reward signals to optimize diffusion models through reward-based feedback learning.

However, this regressive paradigm suffers from fundamental limitations. CLIP-based approaches are constrained by their dual-encoder architecture and single-modality design, which impose inherent scalability bottlenecks. For VLM-based models, the regression head introduces a critical paradigm mismatch with the model’s native next-token prediction ability. Collectively, these limitations prevent full utilization of pre-trained knowledge and fundamentally restrict effective scaling across both reward context and model size dimensions.

Image, Text, Instruction

Image

Image, Text, Instruction, Ref Image, COT

Text

Vision-Language Model

Vision-Language Model

Text Encoder

Image Encoder

Hidden State

Image Emb Text Emb

Hidden State

Regression Head

LLM Head

Cosine Similarity

Cosine Similarity Scalar Value Probability Value

Reward Scores

Reward Scores

Reward Scores

CLIP-based RM Architecture VLM-based RM Architecture

###### RewardDance Architecture

[Figure 2]

🔥

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

Please evaluate two images generated from the prompt: “two monkeys in the cage". Is image 2 better than image 1?

Diffusion Model

Please answer "yes" or "no", and given the detailed reason.

Image1 Image2

Ref Image Generated Image

… … …

… … …

Specific evaluation criteria

Specific evaluation criteria

Image1 Tokens Image2 Tokens Prompt & CoT Task Instruction

Image1 Tokens Image2 Tokens Prompt & CoT Task Instruction

[Figure 7]

🔥 🧊

[Figure 8]

RewardDance (1B ~ 26B)

RewardDance (1B ~ 26B)

Maximize Reward: P (yes | ... )

…

Task-aware CoT Instruction:

yes/no Reasons

RewardDance Inference: Reward Feedback Learning

RewardDance Training

- Figure 3 Overview of RewardDance framework compared with existing reward model Architecture. (Top): Previous works use CLIP-based or VLM-based reward models to provide scalar reward scores for diffusion model training. (Bottom): Our RewardDance approach (from 1B to 26B parameters) uses task-aware CoT instructions for reward modeling with reasoning. Red flames indicate trainable components; blue ice cubes indicate frozen parameters.

##### 3.2 Reward Model Learning

To address the limitations of conventional regressive reward modeling, we introduce RewardDance, a novel framework designed for scalable visual reward modeling. Our approach centers on a generative paradigm that regards reward modeling as a token generation task, naturally aligning with the native capabilities of modern VLMs. This core innovation enables systematic exploration of scalability along two primary dimensions: Model Scaling and Context Scaling. As shown in Figure 3 (bottom panel), our RewardDance framework supports models ranging from 1B to 26B parameters. The framework operates through two main stages: (1) RewardDance Training: where we train the reward model on task-aware Chain-of-Thought (CoT) instruction data, and (2) RewardDance Inference: where the trained reward model provides feedback signals to optimize diffusion models through reward feedback learning.

###### 3.2.1 Context Scaling

We scale the context of our reward model beyond simple image-prompt pairs by incorporating three key elements: task-aware instructions, reference images, and Chain-of-Thought (CoT) reasoning.

First, we regard the reward scoring problem as a comparative judgment task as shown in Figure 3 (bottom left). The model’s input consists of image1 tokens x1, image2 tokens x2, prompt y, and CoT task instruction i. The task instruction guides the model to evaluate whether one image is superior to the other based on a set of predefined criteria. The model is then trained to predict the next token with "yes" or "no". In practice, the reward score is the predicted probability of the "yes" token:

rθ(x1,x2,y,i) = Pθ("yes" | x1,x2,y,i), (2) where Pθ is the probability distribution from our VLM-based reward model, RewardDance.

[Figure 9]

Figure 4 Examples of task-aware instruction and CoT response used for training our RewardDance model.

To further scale the context and enhance the RM’s reasoning capabilities, we train it to generate a rationale for its decision in a Chain-of-Thought (CoT) manner. As illustrated in the Figure 3, the model outputs both "yes/no" and "Reasons" tokens. We construct two data formats for this: one where the model outputs the "yes/no" decision before the reasoning, and another where it generates the reasoning first and concludes with the decision. This detailed reasoning data, distilled from a powerful teacher model (SEED-VL 1.5), not only improves performance but also makes the reward judgments interpretable. For efficiency during the feedback alignment stage (bottom right in the Figure 3), we employ the first format and derive the reward signal by maximizing the probability of the "yes" token: P(yes|...) to optimize the diffusion model.

- Figure 4 provides a concrete illustration of two COT examples in RM training process. It consist of a prompt , two images for comparison (Image 1 and Image 2) , a task-aware instruction, a "yes/no" judgment and a Chain-of-Thought (CoT) response to elaborate on the detailed reasoning. This structure, which combines instructions, comparative judgment, and detailed reasoning, greatly enriches the reward model’s training context, allowing it to move beyond simple text-image matching to achieve more precise and interpretable evaluations.

###### 3.2.2 Model Scaling

To investigate the effects of model scale, we systematically scale our reward model using variants of the InternVL [10] architecture, ranging from 1B to 26B parameters. Our experiments reveal a strong positive correlation between the reward model’s parameter count and the final quality of generated outputs, demonstrating that larger reward models yield superior reward evaluation performance and enhanced generation quality.

###### 3.2.3 Reward Model Training Variants

Pointwise Generative Variant: While our primary reference (pairwise) generative reward paradigm requires reference images for reward prediction, we also design a pointwise generative variant. This variant receives only prompts and generated images, incorporating additional instructions typically used to evaluate whether generated results meet high-quality standards. The reward value is measured by the probability of the "yes" token. During training, we employ Bradley-Terry loss for preference pairs. To facilitate better model convergence, we add a weighted cross-entropy (CE) loss with a small coefficient, assigning "yes" labels to preferred samples and "no" labels to rejected samples.

##### 3.3 Reward Feedback Aligning

###### 3.3.1 Reward Finetuning

For Reinforcement Learning (RL)-based fine-tuning, we adopt the ReFL algorithm [64]. Our generative reward model, RewardDance, provides crucial preference signals. Since our model operates comparatively, we utilize a Best-of-N (BoN) sampling strategy to establish high-quality references for each prompt. Specifically, we first generate N candidate images using the SFT model. RewardDance then performs pairwise comparisons among candidates to identify the top-ranked image, which serves as a reference in subsequent fine-tuning steps. We explore different strategies for selecting reference images based on win counts, with detailed experiments presented in Section 4.4.

###### 3.3.2 Inference-Time Scaling

To further enhance inference-time performance, we implement a Search over Paths approach [36]. This strategy works by pruning a search space of multiple generation paths to select the most promising trajectories. The process begins by initializing N distinct sampling trajectories from different noise vectors. During ODE sampling, these trajectories are iteratively extended using a re-noising and re-sampling mechanism. Specifically, a verifier (our RewardDance) prunes the search space at each step by selecting the most promising trajectories. For this verification role, we deploy a computationally efficient point-wisee generative variant of our model. This lightweight variant is optimized for speed by omitting reference images and evaluating individual candidate images based solely on task instructions.

### 4 Experiments

In this section, we present a comprehensive evaluation of RewardDance on both text-to-image, text-to-video and image-to-video tasks. We analyze the impact of reward model scaling, inference-time scaling and various architectural choices through detailed experimental results and ablation studies.

##### 4.1 Evaluation Baselines, Benchmarks and Metrics

###### 4.1.1 Baselines

Image Baselines. Our primary baselines include representative state-of-the-art (SOTA) models from academia (e.g., LlamaGen [54], SD-v1.5 [49], PixArt-α [7], SD-v2.1 [49], DALL-E 2 [45], Emu3-Gen [58], SDXL [44], DALL-E 3 [2], SD-v3 [12] and FLUX.1-dev [27]) and industry (e.g., Luma [37], Ideogram [22], Midjourney V6.1 [41], DALLE-3 [2], FLUX1.1 [27], Hunyuan [25] , Imagen 3, Recraft [47]s, Seedream 2.0 [16] and Qwen-Image [61]).

Video Baselines. We compare our model with SOTA industry products, e.g., Kling 2.1 Master [24], Veo-3.0 [17], Kling 2.0 Master [24], Veo-2.0 [17], Wan 2.1 [56], Sora [6], RunwayGen 4 [50].

###### 4.1.2 Evaluation Benchmarks

Image Evaluation Benchmarks. We adopt Bench-240, the evaluation prompts utilized in Seedream 2.0 and Seedream 3.0. This comprehensive set of 240 prompts is constructed by integrating representative prompts from publicly available benchmarks with manually curated examples. The prompt distribution is rigorously calibrated based on extensive user preference surveys. The maximum score for Bench-240 is 100 points.

Video Evaluation Benchmarks. For a comprehensive evaluation of video generation models across diverse scenarios, we utilize SeedVideoBench-1.0, introduced in Seedance 1.0. It comprises 300 prompts applicable to both text-to-video and image-to-video tasks, covering a wide spectrum of application domains. The taxonomy for the image-to-video task aligns with that of text-to-video, augmented with detailed annotations for the initial frame.

###### 4.1.3 Evaluation Metrics

Image Metrics. For all text-to-image experiments, we adopt the image-text alignment score from [14]. This metric involves human evaluators to verify whether the generated images correctly depict the key elements

described in the prompts.

Video Metrics. For text-to-video and image-to-video evaluation, we use the human preference-based GoodSame-Bad (GSB) score. Here, G, S, and B represent the counts of "Good," "Same," and "Bad" judgments, respectively. The final GSB score is defined as:

G − B G + S + B

GSB =

(3)

To enable performance comparison with industry products, we developed the Video-Text Alignment Score. The scoring rubric is defined as follows: 0 indicates a complete mismatch between the video content and the text description; 1 indicates a partial match; and 2 indicates a perfect match.

##### 4.2 Comparison with Reward Model Scaling

Image Generation. As demonstrated in Table 2, deploying and scaling RewardDance delivers substantial performance improvements across both RL fine-tuning and test-time scaling paradigms. For RL fine-tuning, scaling the RM from 1B to 26B parameters significantly enhances FLUX.1-dev performance from 67.0 to 73.6. This scaling effect becomes even more pronounced with Seedream-3.0, where scores dramatically increase from a baseline of 74.1 to an impressive 84.8 using the 26B RM. A consistent scaling trend emerges for test-time scaling with Seedream-3.0, with performance steadily climbing from 74.1 to 80.5. This robust correlation across diverse base models and optimization paradigms confirms that larger VLM-based reward models more effectively capture human preferences, thereby guiding diffusion models toward superior output quality.

Stage Metric Base Model No RM 1B RM 2B RM 4B RM 8B RM 26B RM

RM Training RM ID Accuracy (%) RewardDance – 64.70 69.36 65.37 74.92 78.44 RM Training RM OOD Accuracy (%) RewardDance – 69.10 69.59 71.92 71.94 80.90

Alignment Score FLUX.1-dev 67.0 70.7+3.7 72.4+5.4 72.2+5.2 73.0+6.0 73.6+6.6

RL Fine-tuning

Alignment Score Seedream-3.0-SFT 74.1 74.9+0.8 75.3+1.2 79.5+5.4 81.6+7.5 84.8+10.7 Test-Time Scaling Alignment Score Seedream-3.0-SFT 74.1 75.1+1.0 76.3+2.2 78.4+4.3 79.3+5.2 80.5+6.4

- Table 2 The impact of Reward Model (RM) scaling on both RM accuracy and final text-to-image alignment score. As the RM size and its corresponding accuracy increase, the performance of the diffusion models under both RL and Test-time Scaling consistently improves.

Furthermore, we constructed two evaluation datasets: In-Domain (ID) preference dataset contains 2,500 sample pairs. This dataset was constructed by partitioning a subset of data points from the Stage 2 training set of RewardDance that were held out and unseen during training. Out-Of-Domain (OOD) preference dataset contains over 4,000 sample pairs. This dataset was curated by sampling from public benchmark datasets, including ImageReward [64] and HPS [62], to assess the model’s generalization capability on out-of-distribution samples.

As shown in Table 2, we observe no strict positive correlation between the Reward Model (RM) accuracy on the ID dataset and its parameter scale, particularly evident in models with 1B, 2B, and 4B parameters. This finding aligns with existing studies [9, 46, 60], which suggest that higher RM accuracy does not guarantee corresponding improvements in Reinforcement Learning (RL) performance. However, the RM’s accuracy on the OOD dataset – indicative of its generalization capability – can be viewed as a more significant predictor of the final outcome. An RM that demonstrates accurate judgment on unseen distributions and strong generalization capability possesses substantially higher guiding value. Consequently, for evaluating RM efficacy, OOD accuracy (generalization capability) emerges as a more critical core metric than ID accuracy. Building upon this insight, developing evaluation benchmarks that better assess RM generalization constitutes an important direction for future research. This involves enhancing the consistency between the RM evaluation set and the RL test set, as well as integrating sample pairs of varying difficulty levels to enable a more comprehensive assessment of reward model capabilities.

Video Generation. We demonstrate that the benefits of reward scaling extend seamlessly to video generation tasks, as presented in Table 3. We employ GSB metric for this evaluation. For Seedance-1.0 in the Text-to-

Video (T2V) RL setting, RewardDance exhibits a compelling scaling trend: performance improves by +28% with a 1B RM compared to Seedance-1.0-SFT, and progressively increases to an impressive +49% improvement when utilizing the 26B RM. This scaling phenomenon is equally pronounced in the Image-to-Video (I2V) paradigm, where performance gains systematically increase from +29% with the 1B RM to a substantial +47% with the 26B RM. These consistent improvements across both video generation modalities demonstrate the broad applicability and robustness of our RewardDance framework.

Task Base Model No RM 1B RM 2B RM 4B RM 8B RM 26B RM

T2V RL Seedance-1.0-SFT - +28% +32%+4% +41%+13% +45%+17% +49%+21% I2V RL Seedance-1.0-SFT - +29% +34%+5% +37%+8% +41%+12% +47%+18%

- Table 3 Video text alignment results (using GSB as the metric), showing that increasing the reward model size yields significant and consistent performance gains for both T2V and I2V RL over the SFT baseline.

##### 4.3 Comparison with State-of-the-art Generation Models

To rigorously assess the competitiveness of our framework, we benchmark our RewardDance-optimized models against leading academic and industrial systems on three benchmarks: GenEval, Bench-240, and SeedVideoBench-1.0. As shown in Tables 4, 5, and 6, models enhanced by RewardDance consistently achieve state-of-the-art performance.

GenEval. As shown in Table 4, models enhanced with RewardDance demonstrate significant superiority across multiple dimensions of text-to-image generation. Seedream-3.0 w RewardDance achieves the top overall score of 0.79, while FLUX.1-dev w RewardDance follows closely at 0.75, outperforming strong baselines like SD3 (0.74). RewardDance confers overall performance gains of +0.10 and +0.09 on Seedream-3.0 w/o RewardDance and FLUX.1-dev, respectively. These advancements validate the efficacy of RewardDance in substantially enhancing semantic understanding and generative precision.

Method Single Obj. Two Obj. Counting Colors Position Color Attri. Overall↑

LlamaGen [54] 0.71 0.34 0.21 0.58 0.07 0.04 0.32 SD-v1.5 [49] 0.97 0.38 0.35 0.76 0.04 0.06 0.43 PixArt-α [7] 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SD-v2.1 [49] 0.98 0.51 0.44 0.85 0.07 0.17 0.50

- DALL-E 2 [45] 0.94 0.66 0.49 0.77 0.10 0.19 0.52 Emu3-Gen [58] 0.98 0.71 0.34 0.81 0.17 0.21 0.54 SDXL [44] 0.98 0.74 0.39 0.85 0.15 0.23 0.55
- DALL-E 3 [2] 0.96 0.87 0.47 0.83 0.43 0.45 0.67 SD-v3 [12] 0.99 0.94 0.72 0.89 0.33 0.60 0.74

FLUX.1-dev [27] 0.98 0.81 0.74 0.79 0.22 0.45 0.66

FLUX.1-dev w RewardDance 0.98+0.00 0.92+0.11 0.83+0.09 0.84+0.05 0.30+0.08 0.64+0.19 0.75+0.09

Seedream-3.0 [14] SFT (w/o RewardDance) 0.98 0.86 0.67 0.86 0.43 0.31 0.69

Seedream-3.0 w RewardDance 1.00+0.02 0.96+0.10 0.88+0.21 0.94+0.08 0.52+0.09 0.44+0.13 0.79+0.10

Table 4 Evaluation of text-to-image generation ability on GenEval benchmark.

Bench-240. On the Bench-240 benchmark (Table 5), Seedream-3.0 w RewardDance achieves the highest Overall Score of 0.848 when compared against a suite of powerful, closed-source models. It surpasses top commercial offerings such as Imagen 3 (0.79), Luma (0.77), and Midjourney V6.1 (0.63). A detailed breakdown reveals that our model demonstrates exceptional performance in complex categories like Action (0.87) and is highly competitive in Attribute understanding (0.89), validating the effectiveness of our training and alignment methodology.

SeedVideoBench-1.0. We also benchmark our method with Video-Text Alignment Score on video generation tasks (Table 6), evaluating our RewardDance-aligned Seedance 1.0 against leading proprietary models. The final score is the average rating calculated over all samples. In the Text-to-Video (T2V) evaluation, Seedance

- 1.0 achieves the highest average score of 1.66, outperforming strong competitors like Veo-3.0 (1.63) and Kling
- 2.1 (1.57). For the Image-to-Video (I2V) task, our model is also state-of-the-art, achieving a top-tier score of

Model Overall Score Action Counting Relation Attribute

Hunyuan [25] 0.59 0.59 0.60 0.48 0.65 Midjourney V6.1 [41] 0.63 0.62 0.51 0.52 0.68 Recraft [47] 0.64 0.58 0.50 0.51 0.68 DALLE-3 [2] 0.67 0.70 0.57 0.53 0.71 FLUX1.1 [27] 0.67 0.68 0.56 0.62 0.74 Seedream 2.0 [16] 0.71 0.67 0.67 0.66 0.77 Qwen-Image [61] 0.77 0.82 0.73 0.76 0.88 Luma [37] 0.77 0.71 0.68 0.81 0.82 Ideogram [22] 0.77 0.78 0.70 0.69 0.80 Imagen 3 0.79 0.77 0.66 0.73 0.90

Seedream3.0 [14] w RewardDance 0.848 0.87 0.78 0.81 0.89

Table 5 Comparison of human evaluation of different generation models on text-to-image generation task.

- 1.65 and tying with the best-performing model, Kling 2.1. These findings underscore our framework’s ability to produce highly competitive, state-of-the-art video generations.

Model T2V Avg. Score I2V Avg. Score

RunwayGen 4 [50] – 1.37 Sora [6] 1.37 –

- Veo-2.0 [17] 1.47 1.19 Wan 2.1 [56] 1.49 1.36 Kling 2.1 Master [24] 1.57 1.65 Kling 2.0 Master [24] 1.58 1.57
- Veo-3.0 [17] 1.63 1.59 Seedance 1.0 [15] w RewardDance 1.66 1.65

- Table 6 Comparison of human evaluation of different models on text-to-video and image-to-video generation tasks.

##### 4.4 Ablation Study

Reward Dynamics Analysis. Figure 5 and Figure 6 (first five line graphs) illustrates the reward curves during ReFL training for reward models of varying scales across image and video tasks. In these visualizations, solid lines represent smoothed reward trajectories, and the surrounding light-colored bands capture the actual fluctuation ranges of raw reward values. The bubbles quantify reward variance magnitude within a sliding window of 1,000 RL iteration, simultaneously reflecting the exploration intensity of the current policy.

Key findings from these analyses include: i) Sustained Exploration in Large-Scale Reward Models: The 26B RM exhibits substantially greater variance throughout the training. Critically, even at advanced training iterations, these error bars remain considerably wide, indicating that the model maintains significant exploration capabilities well into the later training phases. ii) Accelerated Convergence in Smaller Models: In contrast, error bars for 2B/1B RMs converge to narrow ranges much earlier in training (virtually disappearing in later stages), signifying that their policies achieve stable convergence states sooner and cease effective exploration. iii) Positive Scaling-Exploration Correlation: Consistently, larger model scales correspond to wider error bars across all experimental conditions. This strongly suggests that large-scale reward models demonstrate greater resistance to reward hacking and maintain superior sustained exploration capacity.

- Figure 5 and Figure 6 (final bubble chart) further visualizes the relationship between final performance metrics and exploration intensity (measured by reward variance in late training stages) across RMs of varying scales. This analysis corroborates our observations: as RM scale increases, both model performance and exploration capability (variance) improve monotonically. These concurrent improvements across both dimensions highlight

###### Reward Model: 1B

###### Reward Model: 2B

###### Reward Model: 4B

| | | | | |
|---|---|---|---|---|
|=3<br><br>|.2e-2<br><br>=7.|0e-3| |=5.5e-3<br><br>|
|0 1000 2000<br><br>|3000 4000 5000 6000 7000 8000<br><br>RL Iteration<br><br>Raw Reward Score<br><br>Smoothed Reward Score<br><br>Small Reward Variance Large Reward Variance<br><br>| | | |

| | |
|---|---|
|=2.5e-2| |
| | |
|Score| |
| | |
|8000| |

1.0

1.0

1.0

0.9

0.9

0.9

=4.7e-3 =2.6e-3

=1.2e-2

0.8

0.8

0.8

=3.6e-2

RewardScore

RewardScore

RewardScore

=5.3e-2

0.7

0.7

0.7

0.6

0.6

0.6

0.5

0.5

0.5

0.4

0.4

0.4

Raw Reward Score

Raw Reward Score Smoothed Reward Small Reward Variance Large Reward Variance

Smoothed Reward Score

0.3

0.3

0.3

Small Reward Variance Large Reward Variance

0.2

0.2

0.2

0 1000 2000 3000 4000 5000 6000 7000 8000 RL Iteration

0 1000 2000 3000 4000 5000 6000 7000

RL Iteration

###### Reward Model: 8B

###### Reward Model: 26B

###### Reward Statistics with Different RMs

| | | |
|---|---|---|
| |=4.1e-2 =4.3e-2| |
|=5.4e-2<br><br>|Raw Reward Score Smoothed Reward Score Small Reward Variance| |
|0 1000 2000 3000 40<br><br>RL Ite|00 5000 6000 7000 8000<br><br>ration<br><br>Large Reward Variance| |

| | | |
|---|---|---|
| |=5.3e-2<br><br>=5.1e-2| |
|=6.7e-2|Raw Reward Score<br><br>Smoothed Reward Score<br><br>| |
| |Small Reward Variance Large Reward Variance| |
|0 1000 2000 3000 40<br><br>RL It|00 5000 6000 7000 8000 eration| |

86

1.0

1.0

0.9

0.9

84

0.8

0.8

=5.3e-2

AlignmentScore

82

RewardScore

RewardScore

0.7

0.7

80

=4.1e-2

0.6

0.6

78

0.5

0.5

=3.6e-2

76

0.4

0.4

0.3

0.3

74

=7.0e-3

=4.7e-3

0.2

0.2

1B 2B 4B 8B 26B Reward Model Size

- Figure 5 This figure shows the reward curves for Seedream during the RL stage, experiments with a separate reward model of varying sizes (1B to 26B). While reward scores consistently improve with more RL iterations across all models, a key trade-off emerges with RM size: larger RMs tend to exhibit a higher standard deviation, suggesting stronger robustness and less susceptibility to reward hacking.

0 1000 2000 3000 4000 5000 6000 RL Iteration

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0

RewardScore

Reward Model: 1B

Raw Reward Score

Smoothed Reward Score

Small Reward Variance Large Reward Variance

=3.3e-2

=1.7e-2

0 1000 2000 3000 4000 5000 6000 RL Iteration

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0

RewardScore

Reward Model: 2B

Raw Reward Score

Smoothed Reward Score

Small Reward Variance Large Reward Variance

=3.5e-2

=2.7e-2

0 1000 2000 3000 4000 5000 6000 RL Iteration

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0

RewardScore

Reward Model: 4B

Raw Reward Score

Smoothed Reward Score

Small Reward Variance Large Reward Variance

=4.1e-2

=3.4e-2

0 1000 2000 3000 4000 5000 6000 RL Iteration

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0

RewardScore

Reward Model: 8B

Raw Reward Score

Smoothed Reward Score

Small Reward Variance Large Reward Variance

=5.7e-2

=5.1e-2

0 1000 2000 3000 4000 5000 6000 RL Iteration

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0 RewardScore

Reward Model: 26B

Raw Reward Score

Smoothed Reward Score

Small Reward Variance Large Reward Variance

=1.0e-1

=7.2e-2

1B 2B 4B 8B 26B Reward Model Size

25

30

35

40

45

50

GSBAlignmentImprovement(%)

Reward Statistics with Different RMs

=1.7e-2

=2.7e-2

=3.4e-2

=5.1e-2

=7.2e-2

- Figure 6 This figure shows the GSB alignment scores for Seedance during the RL stage, experiments with a separate reward model of varying sizes (1B to 26B). Note that the apparent sparsity of fluctuations in this plot (compared to the image task results in Figure 5) stems from the lower sampling frequency of data points recorded for the video task. While reward scores consistently improve with more RL iterations, which phenomenon is consistent with Seedream.

Reference Examples

Image-Text Alignment

Base Model Paradigm

Pointwise Regressive × 70.8 Pointwise Generative × 71.6

FLUX.1-dev [27]

Pairwise Generative ✓ 73.0

Pointwise Regressive × 80.7 Pointwise Generative × 81.0

Seedream-3.0 SFT

Pairwise Generative ✓ 81.6

- Table 7 Both generative reward modeling paradigm and reward context scaling consistently improve performance.

Base Model Type

Image-Text Alignment

Seedream-3.0 SFT

Best-of-16 Top-2 83.6

Best-of-16 Bottom-2 80.6 Best-of-2 Top-2 82.7 Best-of-6 Top-2 83.1

Table 8 Ablation of the BON Reference.

Base Model Type Image-Text Alignment

Seedream-3.0 SFT

Baseline 81.6 +CoT Finetuing 83.6

Table 9 Ablation of COT Finetuning.

2B 8B 26B

Reward Model Size

65

70

75

80

85

AlignmentScore

75.3

63.4

81.6

65.5

84.8

67.3

+6.3

+9.5

+2.1

+3.9

Reward Scaling for Different Diffusion Model Sizes

Seedream

Seedream-Lite

Figure 7 The performance of Seedream and SeedreamLite models with different RM model sizes. While both models benefit from larger RMs, the performance improvement is substantially more pronounced for the larger model. This suggests that larger generative models require commensurately larger reward models to realize their full potential.

the advantages of large-scale RMs in preserving exploration vitality while achieving superior final performance. DIT Parameter Analysis. Figure 7 presents performance curves for Seedream-3.0 Lite (approximately one-fifth the parameters of Seedream-3.0) and Seedream-3.0 under reward models of varying scales. Key observations include: i) Scaling Laws Apply to Smaller Architectures: Compact DiT models (Seedream-3.0 Lite) also demonstrate scaling phenomena, where performance improves with increasing RM size, consistent with their larger counterparts. ii) Enhanced Scaling Benefits for Larger Models: However, the performance gain rate for Seedream-3.0 Lite does not exceed that of the full-scale model. Specifically, when scaling the RM from

- 8B to 26B parameters: Seedream-3.0 achieves a performance improvement of +3.92% (81.6% → 84.8%), while Seedream-3.0 Lite exhibits a more modest gain of +2.75% (65.5% → 67.3%). This comparative analysis demonstrates that larger DiT architectures derive greater benefits from reward scaling, yielding more substantial performance improvements.

Generative Reward Paradigm. We first evaluate the impact of paradigm shift from regression-based to generation-based reward modeling via our 8B RewardDance. As demonstrated in Table 7, transitioning from regressive to generative paradigms consistently improves performance across both FLUX.1-dev (70.8 → 71.6) and Seedream (80.7 → 81.0). Subsequently, incorporating reference images yields further improvements: FLUX.1-dev advances from 71.6 to 73.0, while Seedream progresses from 81.0 to 81.6. These results validate our hypothesis that generative approaches, which align more naturally with VLM architectures, provide a more effective foundation for reward modeling and scaling.

BON Reference Examples. Building upon the generative paradigm, we assess the effectiveness of incorporating Best-of-N (BoN) reference examples for enabling comparative evaluations. Due to computational constraints in image generation, we limit N to a maximum of 16. Table 8 reveals that higher-quality reference images correspond to incremental performance enhancements in Seedream-3.0 (Best-of-16 Top-2 > Best-of-6 Top-2 > Best-of-2 Top-2 > Best-of-16 Bottom-2), where ’Best-of-16 Top-2’ refers to selecting the two best-quality samples from 16 inferred samples using the method described in 3.3.1. This indicates that the quality of the reference images is of critical importance in comparative reward modeling.

Reasoning. We isolate the benefits of training with rich, explanatory reasoning data. As shown in Table 9, incorporating CoT data provides substantial performance improvements, elevating the Image-Text Alignment

#### Baseline 1B RM 4B RM 8B RM 26B RM

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

一只豹在雾中捕猎 鹿，以动态姿势描 绘，单色调。

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

1个玻璃杯，

两瓶红酒， 三听啤酒。

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

A cloud in the shape of a teacup.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

画一个球被放置在 盒子里，盒子又被 放在椅子下面。

Figure 8 Text-to-Image generation comparison across reward models of increasing size (Baseline, 1B, 4B, 8B, 26B). Larger reward models demonstrate progressively better prompt adherence, visual quality, and semantic understanding.

score from 81.6 to 83.6 for Seedream-3.0. This demonstrates that leveraging explicit reasoning pathways enhances the reward model’s capability to generate more accurate and human-aligned judgments.

##### 4.5 Visualizations

- Figure 8 and Figure 9 visualize the effects of Reward Scaling on the image and video tasks, respectively. It can be observed from the figures that as the RM model size increases, the performance on both image and video tasks shows an improving trend. Taking the image task as an example, smaller-scale models struggle to accurately generate descriptions involving multi-instance quantity relationships, whereas the large 26B model is capable of generating them completely and correctly.

### 5 Discussion and Further work

Although RewardDance has achieved significant progress on image and video generation tasks, substantial room for exploration remains:

Parameter Scaling. The current maximum model size is 26B parameters. Empirical observations suggest that further scaling to larger sizes (e.g., 70B/100B) is likely to yield greater performance gains, which we will investigate in future work.

Capability Dimension Scaling. This work focuses mainly on foundational vision language capabilities, such as alignment. Future research will explore other critical dimensions of visual generation, such as motion modeling and aesthetic generation.

Task Scope Scaling. Unified models for joint understanding and generation are a current research hotspot.

2B RM 8B RM

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

|[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]|[Figure 53]<br><br>[Figure 54]|[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]|[Figure 60]<br><br>[Figure 61]|
|---|---|---|---|

|[Figure 62]<br><br>[Figure 63]|[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]|[Figure 69]<br><br>[Figure 70]|[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]|
|---|---|---|---|

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

The female statue holds her⼥性雕像chin with⼿托住下巴her hand, the，镜头向右缓缓移动，随后前景处出现⼀道⽩⾊虚化墙慢慢挡住画⾯。camera slowly moves to the right. Then a white blur appears in the foreground and slowly blocks the picture.

[Figure 78]

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

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

|[Figure 102]|[Figure 103]|[Figure 104]|[Figure 105]|
|---|---|---|---|

2BRM2BRM2BRM8BRM8BRM8BRM

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

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

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

|[Figure 154]|[Figure 155]|[Figure 156]|[Figure 157]|
|---|---|---|---|

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

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

The rider weaves his motorcycle through narrow alleys. The camera switches to a shaky handheld shot, showcasing the rider's subjective perspective.

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

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

|[Figure 206]|[Figure 207]|[Figure 208]|[Figure 209]|
|---|---|---|---|

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

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

|[Figure 258]|[Figure 259]|[Figure 260]|[Figure 261]|
|---|---|---|---|

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

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

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Close-up, in a theater dressing room, a clown faces a mirror with tears in his eyes. He then wipes away his tears and suddenly bursts into laughter.

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

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

|[Figure 310]|[Figure 311]|[Figure 312]|[Figure 313]|
|---|---|---|---|

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

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

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

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

[Figure 359]

[Figure 360]

[Figure 361]

|[Figure 362]|[Figure 363]|[Figure 364]|[Figure 365]|
|---|---|---|---|

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

A close-up shot, the silvery moonlight pours down on the archery field. A Japanese archer slowly draws the bow, and the sharp arrow flies out and hits the bull's eye.

###### Figure 9 Video generation quality comparison between 2B and 8B reward models. Top two examples: Image-to-Video generation; Bottom two examples: Text-to-Video generation. The 8B reward model shows improved visual quality and temporal consistency compared to the 2B reward model.

An important direction is to explore whether Reward Scaling techniques can be effectively applied to such models to synergistically enhance performance across tasks like visual understanding, generation, and editing.

Multimodal Scaling. Future many-to-vision tasks (e.g., audio/video-to-video generation) are expected to become mainstream. Consequently, advancing Cross-modal Reward Signal Scaling is a crucial research trend.

Context Scaling. Integrating richer reference information, more complex instructions, reflection capabilities, and in-context learning mechanisms holds promise for further boosting model performance.

### 6 Conclusion

In this work, we address a critical gap in visual Reward Models (RMs) for diffusion-based generation, where existing methods are constrained by architectural limitations or paradigm mismatches that preclude effective reward scaling. We introduce RewardDance, a scalable RM framework built on a novel generative paradigm. This approach reframes reward prediction as a token generation task by converting reward scores into a Vision-Language Model’s (VLM’s) predicted probability for a "yes" token, which natively aligns with the VLM’s autoregressive mechanism. This core innovation unlocks scaling along two key dimensions: model size (from 1B to 26B parameters) and context richness (task-aware instructions, reference examples, and Chain-ofThought reasoning). Our comprehensive experiments across text-to-image, text-to-video, and image-to-video tasks demonstrate that scaling along these dimensions consistently improves the quality of the reward signal, which in turn drives stable and significant gains in final generation quality under both RL fine-tuning and Test-time Scaling. Ultimately, our work establishes scalability as a foundational principle for visual RMs, paving the way for more powerful and robust reward modeling in the future of visual generation.

### References

- [1] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.

- [2] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

- [3] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

- [5] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

- [6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.

- [7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

- [8] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. CoRR, 2023.

- [9] Yanjun Chen, Dawei Zhu, Yirong Sun, Xinghao Chen, Wei Zhang, and Xiaoyu Shen. The accuracy paradox in rlhf: When better reward models don’t yield better language models. arXiv preprint arXiv:2410.06554, 2024.

- [10] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

- [11] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023.

- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [13] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS) 2023. Neural Information Processing Systems Foundation, 2023.

- [14] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

- [15] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [16] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025.

- [17] Google. Veo. https://deepmind.google/models/veo/, 2025.
- [18] Xin Gu, Ming Li, Libo Zhang, Fan Chen, Longyin Wen, Tiejian Luo, and Sijie Zhu. Multi-reward as condition for instruction-based image editing. arXiv preprint arXiv:2411.04713, 2024.

- [19] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

- [20] Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin, Sreya Dutta Roy, Harrie Oosterhuis, Maarten de Rijke, and Satya Narayan Shukla. A simple and effective reinforcement learning method for text-to-image diffusion fine-tuning. arXiv preprint arXiv:2503.00897, 2025.

- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [22] Ideogram. Ideogram. https://about.ideogram.ai/1.0., 2024.
- [23] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

- [24] klingai. klingai. https://app.klingai.com/cn/, 2025.
- [25] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [26] Black Forest Labs. Flux: Official inference repository for flux.1 models, 2024. URL https://github.com/ black-forest-labs/flux. Accessed: 2024-11-12.
- [27] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [28] Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet++: Improving conditional controls with efficient consistency feedback: Project page: liming-ai. github. io/controlnet_plus_plus. In European Conference on Computer Vision, pages 129–147. Springer, 2024.

- [29] Ming Li, Xin Gu, Fan Chen, Xiaoying Xing, Longyin Wen, Chen Chen, and Sijie Zhu. Superedit: Rectifying and facilitating supervision for instruction-based image editing. arXiv preprint arXiv:2505.02370, 2025.

- [30] Siting Li, Pang Wei Koh, and Simon Shaolei Du. Exploring how generative mllms perceive more than clip with the same vision encoder. arXiv preprint arXiv:2411.05195, 2024.

- [31] Xianhang Li, Zeyu Wang, and Cihang Xie. An inverse scaling law for clip training. Advances in Neural Information Processing Systems, 36:49068–49087, 2023.

- [32] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

- [33] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

- [34] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

- [35] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.

- [36] Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025.

- [37] lumalabs. lumalabs. https://lumalabs.ai/, 2024.
- [38] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.

- [39] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, et al. Inference-time scaling for diffusion models beyond scaling denoising steps. arXiv preprint arXiv:2501.09732, 2025.

- [40] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. arXiv preprint arXiv:2508.03789, 2025.

- [41] midjourney. midjourney. https://www.midjourney.com/home, 2024.
- [42] Yuta Oshima, Masahiro Suzuki, Yutaka Matsuo, and Hiroki Furuta. Inference-time text-to-video alignment with diffusion latent beam search. arXiv preprint arXiv:2501.19252, 2025.

- [43] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- [44] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

- [45] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

- [46] Noam Razin, Zixuan Wang, Hubert Strauss, Stanley Wei, Jason D Lee, and Sanjeev Arora. What makes a reward model a good teacher? an optimization perspective. arXiv preprint arXiv:2503.15477, 2025.

- [47] recraft. recraft. https://www.recraft.ai/, 2024.
- [48] Yuxi Ren, Jie Wu, Yanzuo Lu, Huafeng Kuang, Xin Xia, Xionghui Wang, Qianqian Wang, Yixing Zhu, Pan Xie, Shiyin Wang, et al. Byteedit: Boost, comply and accelerate generative image editing. In European Conference on Computer Vision, pages 184–200. Springer, 2024.

- [49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [50] Runway. Runway. https://runwayml.com/research/introducing-runway-gen-4, 2025.
- [51] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685, 2025.

- [52] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

- [53] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

- [54] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

- [55] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

- [56] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [57] Binghai Wang, Runji Lin, Keming Lu, Le Yu, Zhenru Zhang, Fei Huang, Chujie Zheng, Kai Dang, Yang Fan, Xingzhang Ren, et al. Worldpm: Scaling human preference modeling. arXiv preprint arXiv:2505.10527, 2025.

- [58] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

- [59] Yibin Wang, Zhimin Li, Yuhang Zang, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Unified multimodal chain-of-thought reward model through reinforcement fine-tuning. arXiv preprint arXiv:2505.03318, 2025.

- [60] Xueru Wen, Jie Lou, Yaojie Lu, Hongyu Lin, Xing Yu, Xinyu Lu, Ben He, Xianpei Han, Debing Zhang, and Le Sun. Rethinking reward model evaluation: Are we barking up the wrong tree? arXiv preprint arXiv:2410.05584, 2024.

- [61] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

- [62] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Better aligning text-to-image models with human preference. arXiv preprint arXiv:2303.14420, 1(3), 2023.

- [63] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105, 2023.

- [64] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

- [65] Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

- [66] Wenyuan Xu, Xiaochen Zuo, Chao Xin, Yu Yue, Lin Yan, and Yonghui Wu. A unified pairwise framework for rlhf: Bridging generative reward modeling and policy optimization. arXiv preprint arXiv:2504.04950, 2025.

- [67] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

- [68] Zilyu Ye, Zhiyang Chen, Tiancheng Li, Zemin Huang, Weijian Luo, and Guo-Jun Qi. Schedule on the fly: Diffusion time prediction for faster and better image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23412–23422, 2025.

- [69] Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make pixels dance: High-dynamic video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8850–8860, 2024.

- [70] Jiacheng Zhang, Jie Wu, Weifeng Chen, Yatai Ji, Xuefeng Xiao, Weilin Huang, and Kai Han. Onlinevpo: Align video diffusion model with online video-centric preference optimization. arXiv preprint arXiv:2412.15159, 2024.

- [71] Jiacheng Zhang, Jie Wu, Yuxi Ren, Xin Xia, Huafeng Kuang, Pan Xie, Jiashi Li, Xuefeng Xiao, Weilin Huang, Shilei Wen, et al. Unifl: Improve latent diffusion model via unified feedback learning. Advances in Neural Information Processing Systems, 37:67355–67382, 2024.

- [72] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

