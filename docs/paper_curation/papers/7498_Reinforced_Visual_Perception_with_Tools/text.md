# Reinforced Visual Perception with Tools

Zetong Zhou1, Dongping Chen‡1,3, Zixian Ma2, Zhihan Hu1, Mingyang Fu1, Sinan Wang1, Yao Wan†1, Zhou Zhao4, and Ranjay Krishna†2

1ONE Lab, HUST, 2University of Washington, 3University of Maryland, 4Zhejiang University

Visual reasoning, a cornerstone of human intelligence, encompasses complex perceptual and logical processes essential for solving diverse visual problems. While advances in computer vision have produced powerful models for various perceptual tasks, leveraging these for general visual reasoning remains challenging. Prior work demonstrates that augmenting LLMs with vision models via supervised finetuning improves performance, but faces key limitations such as expensive data generation, reliance on careful data filtering, and poor generalization. To address these issues, we propose ReVPT to enhance multi-modal LLMs’ abilities to reason about and use visual tools through reinforcement learning. We introduce a novel RL algorithm based on GRPO, designed to train models to reason with a suite of four visual tools. Through extensive experiments, we show that our method achieves state-of-the-art performance on several perception-heavy benchmarks, including SAT, CV-Bench, BLINK and MMStar, outperforming the supervised and text-based RL finetuning baselines. Notably, Our ReVPT-3B and ReVPT-7B outperform the instruct models by 9.03% and 9.44% on CV-Bench. Finally, we bring to the community new insights on RL-based visual tool-usage through extensive ablations. Our code is available at https://github.com/ls-kelvin/REVPT.

## arXiv:2509.01656v1[cs.CV]1Sep2025

### 1. Introduction

Visual reasoning is a core component of human intelligence. It enables us to solve a wide range of problems, from daily activities such as navigation to challenging geometry problems. Unlike verbal reasoning, visual reasoning is a more complex multifaceted process that requires not just straightforward logical reasoning but also sound visual perception, which further relies on atomic abilities such as object recognition, edge

81.2

84.2

83.3

65.8 74.8

69.4 80.2

80.4

72.9

68.4

64.3

68.3 74.4

60.1

69.0

75.3

72.3

60.6 63.6

76.2

68.3

60.7

59.0

73.6

CV-Bench BLINK

CV-Bench BLINK

57.9

66.0

AVG

AVG

61.0

[Figure 1]

[Figure 2]

53.5

50.0

51.6

62.0

41.7

BLINK-HARD

BLINK-HARD

MathVista

MathVista

56.7

55.0

59.7

63.2

61.2

60.5

68.1

62.4

MMBench

MMBench

MMVP

MMVP

3B

###### 7B

49.7

62.7

MMStar

74.0

MMMU

83.2

MMStar

MMMU

87.2

81.0

85.8

69.0

75.3

70.0

86.8

83.7

72.0

81.0

85.1

68.7

43.0

46.6

45.4

48.4

61.1

60.1

52.0

51.2

53.9

52.9

51.9

49.0

50.9

52.2

62.3

61.8

REVPT REVPT-SFT Qwen2.5VL+RL Qwen2.5VL

REVPT REVPT-SFT Qwen2.5VL+RL Qwen2.5VL

Figure 1: Our ReVPT-3B and 7B models outperform their instruct and text-only GRPO counterparts on perceptioncentric tasks while maintaining strong general capabilities across multiple benchmarks.

Dongping Chen is the project leader. Corresponding author(s): Yao Wan: wanyao@hust.edu.cn, Ranjay Krishna: ranjay@cs.washington.edu, Dongping Chen: dongping@umd.edu.

|<think>The object detection tool has identified one bottle near the window, which appears to be a glass vase. …</think><br><br><answer>… The answer is 1</answer><br><br>Q: How many bottles are in the image?<br><br>A: 1<br><br>GPT-4.1: 2<br><br>Counting<br><br>[Figure 3]<br><br>[Figure 4]<br><br><think>Let's analyze the … I don't immediately see any bottles. I will use object detection to check for any hidden bottles.</think><br><br><tool_call> object_detection<br><br>objects: [ bottles]<br><br></tool_call> <result> <image>Detected 1 object(s) in image 0:…<br><br></result><br><br>[Figure 5]<br><br>[Figure 6]|
|---|

|<think> In the depth map, warmer colors indicate objects that are closer to the camera…</think> <answer>Based on the depth map, the lamp is closer to the camera …</answer><br><br>Q: Which object is closer to the camera, the lamp or the refrigerator? A: lamp<br><br>GPT-4.1: refrigerator<br><br>Depth<br><br><think>To determine which object is closer to the<br><br>camera, I need to analyze the spatial relationship …Using<br><br>a depth estimation tool will help me accurately assess their relative distances from the camera.<think><br><br><tool_call> depth_estimation<br><br></tool_call><br><br><result> <image>Here is the colored depth map for image 0.<br><br></result><br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]|
|---|

|Q: Considering the<br><br>relative positions of the chair (annotated by the red box) and the tennis racket in the image provided,<br><br>where is the located<br><br>with respect to the tennis racket? Select from the following choices. (A) above(B) below<br><br>A:(B)below<br><br>GPT-4.1: The chair (annotated by the red box) is positioned above the tennis racket in the image. The answer is (A) above.<br><br><think>The tennis racket is being held by the player, while the red box marks a chair in the background. To determine the spatial relationship, I will analyze their vertical positions in the image. The chair appears lower and farther away, suggesting it is below the tennis racket in terms of position.</think> <answer>The chair is located below the tennis racket, as indicated by its position in the image. The answer is B.<br><br></answer><br><br>Relationship<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]|
|---|

- Figure 2: ReVPT-3B and 7B are able to solve problems through thinking and selectively employing visual tools to reach strong visual perception. We illustrate some step-by-step visual reasoning breakdowns for challenging examples that model leveraging visual tools to solve complex perception tasks and outperform commercial models GPT-4.1.

detection, depth estimation, etc. Due to the complexity of visual perception, the computer vision community has developed specialized models for different perceptual tasks, such as RecognizeAnything for object recognition, DepthAnything for depth estimation, and SegmentAnything for segmentation (Yang et al., 2024, Liu et al., 2023d, Zhang et al., 2023).

To leverage these advances in perception for enhanced visual reasoning and problem-solving, prior work attempts to augment (multimodal) language models with vision specialists. For example, VisProg first showcases that augmenting large language models (LLMs) with various vision models boosts models’ performance on diverse visual tasks (Gupta and Kembhavi, 2022). Similarly, VisualSketchPad finds that prompting GPT-4o to use sketching tools and depth models improves its performance on challenging perception and visual math benchmarks by large margins (Hu et al., 2024a). More recently, a few works demonstrate that open-source multimodal language models also benefit from using visual tools and reasoning about their outputs after supervised finetuning on tool-use data (Qi et al., 2024, Liu et al., 2023c).

Nonetheless, there are a few limitations to learning to reason with visual tools with supervised finetuning (SFT): first, it relies on expensive data curation. Prior work heavily relies on expensive commercial models like GPT-4 to generate high-quality tool-use and reasoning traces (Qi et al., 2024, Ma et al., 2024, Shao et al., 2024a). Second, it often requires careful data filtering. Previous efforts reveal that aggressive filtering based on data format, answer correctness and other heuristics is crucial to performance gains (Hu et al., 2024b, Ma et al., 2024). Most importantly, supervised finetuning results in limited generalization: it relies on offline trajectories that pre-define particular tools to use; there is no incentive for the model to explore alternative tools or adapt to new environment outputs, thus limiting model’s generalization to unseen tools or problems (Guo et al., 2025, Chen et al., 2025a).

To mitigate these limitations, we borrow inspiration from recent work on enhancing LLM reasoning and generalization with reinforcement learning (RL) (Zhang et al., 2025a, Guo et al., 2025, Huang et al., 2025,

➢Visual Tool Controller

#### Executor

|Here are some useful tools<br><br>for you to solve visual question answering better.<br><br>1. Depth. <Json><br>2. Edge Detection. <Json><br>3. Object Detection. <Json><br>4. Zoom In <Json><br>5. …<br><br><br>Prompt|
|---|

||Processed<br><br>Image<br><br>[Figure 14]<br><br>… …<br><br>[Figure 15]<br><br>[Figure 16]<br><br>Local Server<br><br>[Figure 17]<br><br>Python API<br><br>[Figure 18]<br><br>Depth<br><br>Estimation Zoom In<br><br>[Figure 19]<br><br>[Figure 20]<br><br>Object<br><br>Detection<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|---|
|
|---|

[Figure 25]

Parsing

[Figure 26]

[Figure 27]

[Figure 28]

Rule-Based

[Figure 29]

###### T1 A1 O1

R

- Reward 1

…

…

- Reward 2

- Adv 1
- Adv 2

… …

Reward

[Figure 30]

[Figure 31]

- Output 1

T1 A1 O1 R

- Output 2

[Figure 32]

[Figure 33]

Image

[Figure 34]

…

Policy Model

…

…

Reference

[Figure 35]

[Figure 36]

[Figure 37]

…

Question

…

Model

Direct Answer

T1 R Output G

Adv G

Reward G

[Figure 38]

DKL

Lpg

Update

➢Visual Tools Training Process Based on GRPO Reinforce Learning.

- Figure 3: An overall pipeline of our ReVPT. (Top): Model-generated tool requests are managed by a local environmentbased Tool Controller, which independently deploys visual tool services (e.g., Depth, Object Detection). These tools’ outputs are then fed back to the LVLM for iterative reasoning. (Bottom): When processing a visual reasoning problem, ReVPT employs K-turn rollouts where the model interacts with the tool environment to learn an adaptive policy, culminating in the final model.

Zhan et al., 2025, Shen et al., 2025) and propose Reinforced Visual Perception with Tools (ReVPT). ReVPT enhances multimodal language model’s visual perception and general question-answering by training them to reason with visual tools via RL. ReVPT consists of cold-start and GRPO stages to enable effective RL on multimodal language models with 4 visual tools. Unlike SFT, where all the tool-use reasoning traces need to be generated and filtered in advance, we only need to select the appropriate visual questions to perform RL on. Moreover, instead of supervising the model with one correct tool-use trajectory for each question, RL allows the model to explore and learn from different possible solutions and incentivizes the model to output the best one.

Our experiments with models of different sizes—3B and 7B—demonstrate that ReVPT achieves state-ofthe-art performance (Figure 1), consistently outperforming SFT baselines across five perception-intensive benchmarks, including MMVP, MMStar, CV-Bench, BLINK, and BLINK-Hard. Notably, our approach surpasses the original instruct-tuned models by significant margins, achieving over 5% on average across all benchmarks, with a significant 9% improvement on CV-Bench. Furthermore, our models outperform commercial models GPT-4.1 and Gemini-2.0-Flash on the challenging BLINK-Depth and Relation subsets. Lastly, we contribute a valuable resource to the community by fully open-sourcing ReVPT along with its code and datasets.

### 2. ReVPT: Thinking with Images and Tools

We introduce ReVPT, a novel two-stage framework designed to train multi-modal language models for better visual perception. It is built upon the GRPO algorithm (Shao et al., 2024b), aiming to enhance the model’s ability to tackle complex visual problems by integrating visual processing tools as reasoning steps. Given a multi-modal query, ReVPT agent first generates a reasoning process about analyzing the query and deciding whether to call tools or answer directly. Then, by iteratively analyzing the results from the tools, the model generates a final response to the query. The overall architecture are illustrated in Figure 3.

###### 2.1. Preliminary: GRPO

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that evaluates policy performance by directly comparing a group of candidate responses.

The process begins with a prompt q, for which the policy πθ samples a set of N responses, {o1, o2, . . . , oN}. Each response is evaluated by a reward function, yielding rewards ri = R(q, oi). GRPO then calculates a group-relative advantage Ai for each response by normalizing its reward against the statistics of the entire group:

ri −stdmean{r1,{rr21, . . . ,,r2, . . . ,rN}rN}

. (1)

Ai =

The policy is then updated to responses with higher relative advantages by maximizing a clipped surrogate objective function. This objective is designed to ensure stable training by preventing excessively large policy updates. The full objective is:

G

JGRPO(θ) = Eq∼P(Q),{o

min(siAi, clip(si,1 − ε, 1 + ε)Ai) − β DKL(πθ ∥ πref)]. (2)

i}iG=1∼πθold(O∣q)[

1 G

∑

i=1

Here, si = ππθ(oi∣q)

θold(oi∣q) is the importance sampling ratio that measures the change between the new policy πθ and the old policy πθold used to generate the samples. The Kullback-Leibler (KL) divergence penalty, DKL(πθ∣∣πref) regularizes the policy update by penalizing large deviations from a reference policy πref.

###### 2.2. Visual Tools

The efficacy of leveraging visual tools for visual reasoning has been well-established. We choose to focus on four specific visual tools as they are the most suitable for perception-oriented tasks. While we experiment with a larger suite of tools, we find that the model is reluctant to use them. Our framework integrates these four high-performing visual tools, enabling their execution and subsequent result incorporation during the MLLM inference process to construct a comprehensive rollout. While Table 1 offers a detailed summary of each tool’s parameters and specifications, their core functionalities and representative use cases are elaborated upon below.

▷ Object Detection (Fu et al., 2025) is capable of open-vocabulary detection. It takes an input image Iin and a textual query qtext to locate instances of described objects, submitting their boundaries ℬout for objects localization.

▷ Zoom In facilitates focused analysis by taking an input image ℐin and a specified region of interest ℛ

###### Tool Description Arguments Tool Output

Object Detection Open-vocabulary object detection Image + Text Prompt Annotated Image + Boxes Zoom In Crop a region and augment it Image + Coordinates + Factor Cropped Image Edge Detection Edge detection using scharr algorithm Image Edge Map Depth Estimation Return depth map in colorful style Image Depth Map

- Table 1: Visual tools integrated within ReVPT. We illustrate their arguments, result formats, and core funtions description.

to output a magnified view ℐfocus of that region. It helps model to concentrate on pertinent areas, thereby mitigating the influence of irrelevant information and amplifying core features.

▷ Edge Detection is designed to identify and delineate the perimeters of objects and significant textural variations, this tool transforms an input image ℐin into a feature map ℰout that emphasizes these structural edges. This grants models a sharper perception of object silhouettes and surface patterns.

▷ Depth Estimation (Yang et al., 2024) computes spatial depth from a monocular visual input. Given an image ℐin , it generates a corresponding depth map 𝒟out which encodes the estimated distance of scene elements from the observer. We use depth_anything_v2 to serve as the tool, equipping models with a more nuanced understanding of three-dimensional arrangements and the relative positioning of objects.

[Figure 39]

- Figure 4: Reinforced visual tool-usage training requires high-quality and verified data. We transform two datasets into multiple-choice question, then filter out too easy or hard questions for Qwen2.5-VL-7B by multiple inference.

###### 2.3. Cold Start

Initially, our objective is to train a multimodal agent using the R1-Zero method. However, during the training process, we observe a progressive decline in the agent’s propensity to utilize tools. This phenomenon likely stemmed from the fact that solving visual tasks did not inherently require tool usage, and reasoning based on processed images represented a distribution shift from the model’s initial training data. Consequently, we opt to incorporate cold-start data during the initial training phases as shown in Figure 4. This strategy aims to facilitate the model’s initial acquisition of tool utilization skills for solving visual problems.

Currently, we employ GPT-4.1 as the agent to synthesize high-quality data for tool-augmented reasoning in

visual problem solving. We require the GPT model to follow a predefined format when answering questions. Specifically, it should first engage in a reasoning process, considering the potential utility of external tools, and then employ a multi-turn generation strategy to address the problem. We explicitly encourage the model in our prompt to generate more tool-assisted rollouts. Subsequently, we filter out rollout trajectories generated by the GPT model that contain errors.

After synthesizing tool-integrated reasoning data, we perform supervised fine-tuning on it. Denote the query

- as Qi, the rollout trajectory τi as a sequence of actions ait and observations oti. We optimize the cross-entropy loss:

N

ℒSFT(θ) = −

log Pθ(τi ∣ Qi) (6)

1 N

∑

i=1

ni

N

log Pθ(ait,∣ Qi, a<i t, o<i t). (7)

1 N

∑

∑

= −

t=1

i=1

By minimizing ℒSFT, the model acquires a robust Cold-Start policy for sequential vision-tool invocation, providing a solid foundation for the subsequent reinforcement learning phase.

###### 2.4. Reward Design

Vision tasks often possess ground truth data, facilitating the use of verified rewards to evaluate model’s responses. This approach circumvents the introduction of neural network-based reward models, thereby preventing reward hacking.

- • Correctness. We restructure the dataset into a verifiable format. The model is required to put the final direct answer or option in the box, enabling reliable rule-based verification of correctness. If the answer match the ground truth, it gets through checking.
- • Format. In each turn, the response should enclose its thinking process in <think></think> and enclose its tool call in <tool_call></tool_call> or answer in <answer></answer>. If the response matches the pattern, it gets through format checking. Given the query q and the rollout o, the reward is defined as:

⎧ ⎨ ⎪⎩

1, I f FormatCorrect(o)

Reward(q, o) =

∧ AnswerCorrect(o) −1, otherwise

We adopt this naive binary reward design because previous research has demonstrated the effectiveness of using simple, rule-based reinforcement learning to produce strong reasoning behaviors (Zhou et al., 2025).

### 3. Experiment

###### 3.1. Experiment Setups

Models. We conduct most of our experiments on Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct (Wang et al., 2024a) for their strong visual perception capability and how different “perception baseline” contribute to final capability.

CV-Bench

Model

BLINK(sub) MMVP MMSTAR BLINK-HARD MMMU MMBench MathVista Count Relation Depth Distance Avg.

Avg.

Gemini-2.0-Flash 76.09 71.95 86.92 87.50 82.17 81.50 76.37 79.34 69.40 68.28 71.7 89.05 73.1 GPT-4.1 76.06 67.77 92.00 94.50 89.50 84.76 68.80 88.00 69.80 66.13 74.00 86.60 70.40 TACO (Qwen2-VL-7B) 58.55 63.32 81.08 59.83 57.26 66.00 65.50 67.00 49.53 33.13 44.0 81.80 41.9

Qwen2.5-VL-3B-Instruct 58.98 68.65 74.92 76.00 71.67 72.55 64.34 62.67 53.40 41.67 49.00 80.97 61.20 Qwen2.5-VL-3B-SAT-SFT 60.99 60.66 90.15 87.33 84.50 79.42 63.79 58.33 50.84 56.71 42.11 80.09 56.66 Qwen2.5-VL-3B-GRPO 60.60 69.16 81.23 80.17 61.50 72.90 60.66 69.00 46.60 51.61 45.44 83.69 54.95 ReVPT-3B (Ours) 65.67 70.43 92.62 87.50 78.33 81.20 72.35 68.70 53.87 60.48 49.00 83.21 57.90 ∆ +6.69 +1.78 +17.70 +11.50 +6.66 +8.65 +8.06 +6.03 -0.47 +18.81 0.00 +2.24 -4.20

Qwen2.5-VL-7B-Instruct 68.23 64.97 88.46 72.00 74.00 74.41 80.41 74.00 61.80 50.00 52.00 85.08 68.10 Qwen2.5-VL-7B-SAT-SFT 62.70 60.28 84.46 85.33 82.67 77.03 67.91 62.67 49.80 57.00 45.22 82.49 59.50 Qwen2.5-VL-7B-GRPO 69.38 72.59 95.08 83.17 84.83 83.32 75.87 75.33 62.27 61.00 51.88 86.83 63.20 ReVPT-7B (Ours) 69.42 74.11 92.31 88.67 82.67 84.23 73.64 72.00 61.07 62.37 50.89 87.20 66.00 ∆ +5.73 +9.14 +3.85 +16.67 +8.67 +9.82 -6.77 -2.00 -0.73 +12.37 -1.11 +2.12 -2.10

- Table 2: Comparison between different models and method on vision-centric benchmarks. Our ReVPT outperform direct SFT and text-space GRPO in perception-specific tasks and preserve general capability.

Dataset Construction. A high quality cold-start dataset and a difficult dataset for RL training is very important for model to learn test-time scaling itself (Yu et al., 2025). Therefore, we filter the SAT dataset (Ray et al., 2024) and Taco (Ma et al., 2024) training set with Qwen2.5-VL-7B-Instruct and retain the samples it answers incorrectly. SAT is a synthetic VQA dataset designed to improve a model’s visual perception, which has been shown to be effective in prior work (Zhou et al., 2025). TACO dataset was also utilized in previous research (Ma et al., 2024) for multi-modal chain-of-thought SFT. We selected it based on the assessment that its VQA content is better suited for tool-assisted responses. Then, we split it into cold-start data synthetic and training data. To construct high-quality cold-start dataset, we leverage GPT-4.1 to synthesize tool trajectory and retain the subset that it answers correctly. Finally, we obtain a 1.5k cold-start dataset with well-curated reasoning chain and tool trajectory and 20k question for RL process.

Baselines. We compare ReVPT against the following models and approaches: (1) Commercial Models: We select GPT-4.1 (OpenAI, 2025a) and Gemini-2.0-Flash (Google, 2024) as strong multimodal baselines. Both are evaluated in a zero-shot setting without external tools as strong generalist benchmarks. (2) Taco: Learns to invoke 15 external tools (e.g., OCR, depth, etc.) by generating Chain-of-Thought-and-Action (CoTA) sequences via supervised learning on synthetic data. TACO typically executes tools within a single process, contrasting with our RL-based approach and distributed architecture (Ma et al., 2024). (3) Qwen-Instruct: We implement two instruct version models without any tool usage, employing basic prompt formats for a robust evaluation. (4) Qwen-SAT-SFT: Two models after supervised finetuning with the SAT training set (13k samples) as a strong baseline. We also include an enhanced SFT baseline with rephrased answers generated by the model itself as reported by previous research (Wang et al., 2024b). (5) Qwen-SFT (cold start): The model state after cold-start training. (6) Text-based RL: An RL agent trained similarly to ReVPT but without any visual feedback from tools. This follows the native GRPO training in VisualThinker-R1-Zero (Zhou et al., 2025).

Evaluation. We select 8 multi-modal benchmarks covering diverse capabilities with a focus on visual perception and reasoning tasks. This includes perception-centric benchmarks CV-Bench (Tong et al., 2024a), Blink (Fu et al., 2024), MMVP (Tong et al., 2024b), Blink-Hard (Bigverdi et al., 2024). We also include MMStar (Chen et al., 2024), MMMU-val (Yue et al., 2024) and MMBench (Liu et al., 2024b) for general capabilties assessment. Reasoning benchmark MathVista (Lu et al., 2024) is also included. We choose subset

Data Construction

CV-Bench BLINK

BLINK-HARD

Hint Avg.

MMVP MMSTAR

Tool-usage General Count Relation Depth Distance Avg. Count Depth Relation 3 4 5

- ✔ ✔ 78.28 66.62 87.69 82.50 76.83 77.75 65.00 73.39 83.92 68.67 60.40 65.32 54.84 51.61

- ✔ ✔ 77.26 65.74 88.62 82.33 72.83 76.76 60.00 79.03 81.12 69.33 58.33 61.29 54.03 59.68

✔ ✔ 78.97 68.15 87.23 80.67 80.33 78.46 65.83 72.58 81.82 69.00 58.33 63.71 50.81 41.94

- ✔ ✔ ✔ 80.86 66.24 91.54 87.17 79.17 80.17 65.83 79.84 81.82 72.67 61.80 68.55 55.65 60.48

- Table 3: Ablation study on ReVPT-7B SFT data construction demonstrates that synthesized tool-usage data and general data from TACO collaboratively contribute to an effective cold-start process, preserving the model’s general capabilities while enabling visual tool usage. Specifically, training with synthesized data where the model provides 1-2 sentence reasoning about task suitability for visual tools proves beneficial for this dual objective.

with only one image as input. All results are reported under average of three runs.

Experiment Details. We conduct model training using configurations with 8 × NVIDIA A800 GPUs. We leverage LLaMA-Factory (Zheng et al., 2024) as SFT platform and Verl as visual tool-base RL platform (Sheng

- et al., 2024). The training process involved two phases: (1) Cold-start Period: Models are trained for 2 epochs with a learning rate of 1e-5 and a global batch size of 64. (2) RL Period: For this phase, models are trained for 200 steps to reach reward converge. We provide training loss and reward curve in Appendix.

###### 3.2. Experiment Results

ReVPT outperform supervised finetuning and text-based RL on visual perception tasks. As shown in Table 2, our method significantly outperforms the original model on both CV-Bench and BlinkHard benchmarks, while also demonstrating improvements over SFT cold start models. The performance gains are particularly pronounced in depth and distance tasks, with improvements of 9.5% and 13.23% respectively compared to the base model, indicating that reinforcement learning effectively teaches the model to utilize tools and interpret tool-generated information. The consistent improvements across ReVPT-3B (+5.81%) and ReVPT-7B (+5.73%), with striking gains on perception-heavy benchmarks like CV-Bench (+8.65% and +9.82%), demonstrate that reinforcement learning unlocks latent capabilities inaccessible through conventional training. This ceiling elevation stems from RL’s unique ability to enable exploratory tool selection and adaptive reasoning during inference, allowing models to dynamically adjust problem-solving strategies rather than being constrained to pre-defined patterns from static supervised trajectories. The fact that ReVPT substantially outperforms not only base instruct models but also specialized baselines suggests that reward-driven optimization creates a synergistic learning dynamic that transcends single training methodologies, indicating the true potential of visual perception capabilities may be significantly higher than previously demonstrated.

Performance tradeoffs in perception-specialized training. Our training on the curated perception subset from SAT and TACO resulted in a slight reduction of the model’s general performance capabilities. Direct supervised fine-tuning on the SAT dataset (SAT-SFT) revealed substantial degradation across several capabilities required for BLINK and MMStar evaluations. These findings underscore the critical relationship between cold-start sample distribution and the resulting model capabilities. Our key insight is that developing effective visual tool-using agents requires not only sophisticated interaction environments but also carefully balanced, comprehensive, and high-quality data construction that preserves general capabilities while enhancing targeted skills.

###### Visual Tool Usage Proportion Across Models and Benchmarks

1.0

| |0.62<br><br>0.49<br><br>0.71<br><br>0.18<br><br>0.54<br><br>0.45<br><br>0.55<br><br>0.17| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

ToolUsageProportion

0.6

0.4

0.2

0.0

CV-Bench BLINK MMVP MMStar

Object Detection Zoom In Depth Estimation Edge Detection

- Figure 5: ReVPT-3B effectively learns tool utilization after cold-start process. ReVPT-3B shows bias toward object detection and depth estimation over zoom and edge detection tools due to cold-start data construction. Future work will address data balance and generalized perception objectives.

Figure 6: ReVPT-3B learns to improve accuracy by changing its tool usage after reinforcement learning. This strategic shift results in a higher proportion of correct answers, demonstrating that the RL phase optimizes model’s tool selection.

Visual Tools

CV-Bench BLINK

BLINK-HARD

RL

MMVP MMSTAR

Object Depth Count Relation Depth Distance Avg. Count Depth Relation 3 4 5

- ✔ ✔ 70.05 91.08 85.17 77.67 80.40 60.00 70.16 76.22 63.67 54.53 66.13 57.26 58.06

✔ ✔ 68.53 90.46 86.33 71.33 78.62 55.83 69.35 84.62 63.33 53.93 62.10 58.87 58.87

- ✔ ✔ 66.62 87.54 86.50 69.67 76.99 59.17 73.39 74.83 51.00 53.47 61.29 55.65 54.03

- ✔ ✔ ✔ 70.43 92.62 87.50 78.33 81.58 60.83 75.81 81.12 70.33 53.87 66.13 57.26 60.48

- Table 4: Ablation study on visual tools and RL phase showing that each component within ReVPT-3B collaboratively contribute to our outstanding results on vision perception tasks.

###### 3.3. In-Depth Analysis

Ablation study on how to introduce tool-use capability to current models. To address the above mentioned challenge, we conducted an ablation study on cold-start process (shown in Table 3) to determine how to introduce tool-use agent capabilities to the current model without compromising its general performance. Our empirical experiments demonstrate that synthesizing hints about whether and why to use tools can drastically preserve the model’s general capabilities. Moreover, including general data from TACO also contributes to maintaining general model capability, while incorporating tool-use data enhances the model’s tool-using performance.

Ablation study on visual tools. We conduct ablation studies on the two most frequently used visual tools—object detection and depth estimation—by removing these tools from both data and instructions to validate their importance as core components of ReVPT. The experiments follow the same cold-start and RL training procedures, with results shown in Table 4. When removing object detection, performance declines significantly across most metrics: BLINK relation dropped by 5 points (8.2% decrease) from 60.83 to 55.83, and MMVP scores fall by 12.33 points (9.6% decrease) from 70.33 to 63.33. This confirms that visual tool usage is fundamental to ReVPT’s effectiveness.

|The object detection model mistakenly identify a pillow as a cushion, and ReVPT fails to recognize this<br><br>error, leading to an<br><br>incorrect answer.<br><br>Q: How many cushions are in the image? A: 2<br><br>REVPT: 5<br><br>Incorrect Tool Result<br><br>[Figure 40]<br><br>[Figure 41]|The depth estimation<br><br>model returns the correct result, but ReVPT analysis is incorrect, leading to an incorrect answer.<br><br>Q: Which point is<br><br>closer to thecamera?<br><br>A: B REVPT: A<br><br>Incorrect Tool Analysis<br><br>[Figure 42]<br><br>[Figure 43]|
|---|---|
|The zoomed-in region<br><br>is completely useless for answering, and instead interfered with normal perception, leading to an incorrect<br><br>response.<br><br>Q: How many clocks are in the image? A: 2 REVPT: 1<br><br>Inappropriate Tool Usage<br><br>[Figure 44]<br><br>[Figure 45]|ReVPT chooses to use edge detection, but this tool is completely unhelpful for answering the question.<br><br>Q: Is the knife laid flat or standing up? A: flat REVPT: standing up<br><br>Incorrect Tool Selection<br><br>[Figure 46]<br><br>[Figure 47]|

Figure 7: Case studies illustrating various failure modes of ReVPT when using visual tools.Examples include incorrect tool output (top left) , misinterpretation of a correct tool output (top right) , inappropriate tool usage that interferes with perception (bottom left) , and the selection of an unhelpful tool for the given task (bottom right).

Tool usage analysis. We analyze the frequency of tool usage by cold-start and RL phase models across different benchmarks, as shown in Figure 5. On perception-intensive benchmarks like CV-Bench and MMVP, models exhibit high tool usage frequency, while on the more general MMSTAR benchmark, tool usage frequency decreases substantially. Furthermore, we observe that object detection and edge detection usage decline significantly after RL training, resulting in an overall reduction in tool usage frequency. To figure out how vision tools contribute to model’s improvement, we examine the changing portion (Figure 6) between ReVPT-3B and base model. Across 4 benchmarks, we find that ReVPT-3B achieves higher accuracy when using tools, indicating that visual tool-usage provide more information and select more appropriate tools for visual perception tasks.

Limitations of visual tools. During our experimental process, we observe that these visual tools often hinder rather than enhance model performance, sometimes degrading or interfering with the model’s reasoning capabilities. In some cases, the model needed to correct erroneous tool outputs to arrive at accurate answers. For instance, in Figure 7 (top left), the object detection model misclassify a mattress as a pillow, resulting in an incorrect response despite the model’s inherent capability to solve the problem correctly. Case study in

- Figure 7 (top right) shows that although the depth estimation tool returned a color-coded depth map, the model do not acknowledge this information and use it during reasoning, ultimately producing an incorrect answer. We attribute this phenomenon to the model’s original perception capabilities constraining its ability to effectively and accurately utilize external tools, limiting tool-use efficacy.

### 4. Discussion and Future Works

Do we really need visual tool specialist? The utility of visual tool specialists follows a non-monotonic relationship with model capability. For resource-constrained models like our 3B variants, visual tools provide

a compelling shortcut to enhanced perception, with ReVPT achieving over 9% improvements on perceptionheavy benchmarks. This aligns with findings from TACO (Ma et al., 2024), where chain-of-thought-and-action approaches enabled performance gains of 30-50% compared to few-shot prompting. However, as models scale to moderate sizes, the marginal benefit diminishes—larger vision encoders reduce dependency on external tools, and our analysis reveals cases where tool outputs contradict correct model intuitions, leading to degraded performance. Paradoxically, the most advanced models like GPT-o3 (OpenAI, 2025b) may experience a tool renaissance through dynamic code generation, representing a shift from using pre-defined tools to creating sophisticated reasoning chains. This suggests tool integration is worthwhile across the capability spectrum, with beneficial tools evolving from compensating for perceptual deficits in small models to enabling sophisticated multi-step reasoning in large ones.

Which visual tools should we select for tool-use training? The selection of visual tools fundamentally shapes model capabilities and constrains downstream performance. Our domain-specific approach prioritized tools aligned with perceptual challenges: object detection, depth estimation, edge detection, and zoom-in functionality. While this reflects broader patterns where tool selection is guided by target applications, it reveals fundamental limitations in current approaches that constrain generalizability. The core challenge lies in the cold-start nature of tool learning, which necessitates committing to a specific tool repertoire during initial training (Zhang et al., 2025b, Wang et al., 2025c). Our experiments initially included broader tools (e.g., auxiliary line drawing, highlighting, rotation) but found extremely low utilization rates, underscoring that smaller models struggle to learn multiple tools simultaneously due to limited world knowledge. The constraint is data-driven—achieving proficiency across diverse tool sets requires comprehensive training datasets. Our reliance on SAT and TACO datasets lacks the breadth for more specialized tools, suggesting that advancing visual tool learning requires systematic development of richer, more diverse training ecosystems.

The Bitter Lesson of Tool Use: Human Intuitions vs. Computational Discovery. A fundamental tension exists in tool-using AI systems: while any approach producing correct solutions should be considered valid regardless of human expectations, current training paradigms systematically inject human biases about tool selection. This occurs inevitably through cold-start phases where synthetic data generation relies on humandesigned demonstrations, creating imitation learning dynamics that persist even after reinforcement learning begins. The history of AI suggests that methods leveraging greater computation for autonomous discovery will ultimately outperform those constrained by human intuitions about appropriate tool usage. However, practical limitations force this bias injection—open-source agents employing cold-start supervised fine-tuning require initial human demonstrations because smaller models lack the capability to explore effectively from scratch. Approaches like process reward models (Lightman et al., 2023, Wang et al., 2025b) likely amplify rather than resolve these human-centric biases. This creates a computational scaling challenge: as systems grow more capable, they should increasingly discover tool-use strategies aligned with their computational strengths rather than human cognitive patterns, yet current training paradigms embed human preferences that may limit this autonomous discovery.

### 5. Related Work

Multimodal language models. Recently, there have been many advances on open-source multimodal models Awadalla et al. (2023), Chen et al. (2023), Liu et al. (2023b,a, 2024a), Dai et al. (2024), Li et al. (2022, 2023), Deitke et al. (2024). These efforts include training multimodal models to take in multiple images, engage in multi-turn conversations, and even understand videos Liu et al. (2024a), Jiang et al. (2024),

Li et al. (2024). For example, LLaVA-Next achieves strong multi-image understanding through large-scale interleaved visual instruction tuning with M4-Instruct Liu et al. (2024a). Similarly, Mantis introduces a new large-scale multi-image instruction tuning dataset Mantis-Instruct for multi-image training Jiang et al. (2024). These efforts pave the foundation for our work on learning vision-language models with image-text interleaved reasoning traces. over perceptual outputs from vision specialists.

Multimodal tool-use. Recently, there has been increasing interest in enhancing multimodal language models with improved tool-use capabilities Liu et al. (2023c), Qi et al. (2024), Shao et al. (2024a). LLaVa-Plus demonstrated the feasibility of training such models to utilize visual tools Liu et al. (2023c). Visual Program Distillation transfers tool-use and reasoning skills into a multimodal model using chain-of-thought (CoT) data derived from programs Hu et al. (2024b). Similarly, Visual CoT introduces a synthetic CoT dataset aimed

- at boosting the reasoning abilities of multimodal models Shao et al. (2024a). More recently, LLaVa-CoT incorporates both perception and reasoning capabilities inspired by GPT-4o Xu et al. (2025). CogCoM identifies six key manipulation strategies and trains multimodal models with synthetic chain-of-manipulation (CoM) data Qi et al. (2024), while TACO contributes 273K multimodal reasoning traces derived from the outputs of 15 visual tools Ma et al. (2024). Recent works also extent multimodal tool-use with RL, enhancing general reasoning capability (Wang et al., 2025c, Zhang et al., 2025b, Zhao et al., 2025) and visual search capabiltiy (Zheng et al., 2025, Chen et al., 2025b, Wang et al., 2025a).

Multimodal language model reasoning with RL. DeepSeek-R1 (Guo et al., 2025) has demonstrated that simple rule-based reinforcement learning can effectively induce strong reasoning behaviors. This R1-style reasoning paradigm has shown success in multimodal language models (Shen et al., 2025, Li et al., 2025, Chen

- et al., 2025c). VLM-R1 (Shen et al., 2025) applies reward-driven fine-tuning to improve visual reasoning and generalization beyond supervised baselines. Vision-R1 (Zhan et al., 2025) introduces human-free alignment through vision-guided rewards, while another Vision-R1 (Huang et al., 2025) leverages CoT-style tasks and reward shaping to boost multi-step reasoning. UI-R1 (Lu et al., 2025) enhances action prediction in GUI agents via GRPO-based RL, achieving strong performance with compact models. In the video domain, Video-R1 (Feng et al., 2025) and VideoChat-R1 (Li et al., 2025) improves temporal reasoning through T-GRPO and mixed-modality rewards across diverse video types.

### 6. Conclusion

In this work, we explore tool-usage reinforcement learning to enable models to utilize external vision tools for test-time scaling. We propose ReVPT, a training suite designed to integrate tool-usage with RL, enabling models to optimize tool selection and interleaved text-image-tool reasoning through direct interaction and reward feedback. Our experiments across multiple benchmarks demonstrate that ReVPT successfully raises performance beyond base model, achieving results significantly higher than those obtained through supervised fine-tuning and text-only RL. We hope that ReVPT, along with our fully open-sourced code, dataset, and platform, will serve as a valuable resource for the community.

### Acknowledgment

We would like to thank Jieyu Zhang, Weikai Huang, and Xirui Li for their insightful feedback and support.

### References

Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models, 2023.

Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G Shapiro, and Ranjay Krishna. Perception tokens enhance visual reasoning in multimodal language models. arXiv preprint arXiv:2412.03548, 2024.

Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, and Georgia Gkioxari. Omni3D: A large benchmark and model for 3D object detection in the wild. In CVPR, Vancouver, Canada, June

2023. IEEE.

Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468, 2025a.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.

Yang Chen, Yufan Shen, Wenxuan Huang, Shen Zhou, Qunshu Lin, Xinyu Cai, Zhi Yu, Botian Shi, and Yu Qiao. Learning only with images: Visual reinforcement learning with reasoning, rendering, and visual feedback. arXiv preprint arXiv:2507.20766, 2025b.

Zhangquan Chen, Xufang Luo, and Dongsheng Li. Visrl: Intention-driven visual perception via reinforced reasoning. arXiv preprint arXiv:2503.07523, 2025c.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 36, 2024.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.

Shenghao Fu, Qize Yang, Qijie Mo, Junkai Yan, Xihan Wei, Jingke Meng, Xiaohua Xie, and Wei-Shi Zheng. Llmdet: Learning strong open-vocabulary object detectors under the supervision of large language models. arXiv preprint arXiv:2501.18954, 2025.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024.

Google. Gemini 2.5 flash. https://deepmind.google/technologies/gemini/flash/, 2024. Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma,

Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. ArXiv, abs/2211.11559, 2022.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models, 2024a. URL https://arxiv.org/abs/2406.09403.

Yushi Hu, Otilia Stretcu, Chun-Ta Lu, Krishnamurthy Viswanathan, Kenji Hata, Enming Luo, Ranjay Krishna, and Ariel Fuxman. Visual program distillation: Distilling tools and programmatic reasoning into visionlanguage models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9590–9601, 2024b.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Dongfu Jiang, Xuan He, Huaye Zeng, Con Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483, 2024.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.

Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. CoRR, abs/1405.0312, 2014. URL http://arxiv.org/abs/1405.0312.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning,

2023a. Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023b. Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next:

Improved reasoning, ocr, and world knowledge, January 2024a. URL https://llava-vl.github. io/blog/2024-01-30-llava-next/.

Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang, Jianfeng Gao, and Chunyuan Li. Llava-plus: Learning to use tools for creating multimodal agents, 2023c. URL https://arxiv.org/abs/2311.05437.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023d.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024b.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. Uir1: Enhancing action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025.

Zixian Ma, Jianguo Zhang, Zhiwei Liu, Jieyu Zhang, Juntao Tan, Manli Shu, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Caiming Xiong, et al. Taco: Learning multi-modal action models with synthetic chains-of-thought-and-action. arXiv preprint arXiv:2412.05479, 2024.

OpenAI. Gpt-4.1. https://openai.com/index/gpt-4-1/, 2025a. Large language model. Accessed:

2025-05-11. OpenAI. Openai o3-mini, 2025b. URL https://openai.com/index/openai-o3-mini/. Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao

Dong, and Jie Tang. Cogcom: Train large vision-language models diving into details through chain of manipulations, 2024. URL https://arxiv.org/abs/2402.04236.

Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, Kuo-Hao Zeng, et al. Sat: Spatial aptitude training for multimodal language models. arXiv preprint arXiv:2412.07755, 3, 2024.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Unleashing chain-of-thought reasoning in multi-modal language models, 2024a.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024b.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024a.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms, 2024b.

Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, et al. Vgr: Visual grounded reasoning. arXiv preprint arXiv:2506.11991, 2025a.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, et al. Visualprm: An effective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291, 2025b.

Ye Wang, Qianglong Chen, Zejun Li, Siyuan Wang, Shijie Guo, Zhirui Zhang, and Zhongyu Wei. Simple o3: Towards interleaved vision-language reasoning. arXiv preprint arXiv:2508.12109, 2025c.

Yifei Wang, Yuyang Wu, Zeming Wei, Stefanie Jegelka, and Yisen Wang. A theoretical understanding of self-correction through in-context alignment. arXiv preprint arXiv:2405.18634, 2024b.

Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step, 2025. URL https://arxiv.org/abs/2411.10440.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng,

Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.

Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning. arXiv preprint arXiv:2503.18013, 2025.

Shaokun Zhang, Yi Dong, Jieyu Zhang, Jan Kautz, Bryan Catanzaro, Andrew Tao, Qingyun Wu, Zhiding Yu, and Guilin Liu. Nemotron-research-tool-n1: Tool-using language models with reinforced reasoning. arXiv preprint arXiv:2505.00024, 2025a.

Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025b.

Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514, 2023.

Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, and Chen Wei. Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998, 2025.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-zero’s "aha moment" in visual reasoning on a 2b non-sft model, 2025. URL https://arxiv.org/abs/2503. 05132.

### A. Dataset Construction Details

We leverage the SAT dataset and the CoTA dataset from TACO for our training. The SAT (Ray et al., 2024) dataset is a synthetic VQA dataset designed to enhance the spatial reasoning capabilities of MLLMs. The CoTA dataset from TACO (Ma et al., 2024) is a synthetic dataset comprising Chain-of-Thought-Action data generated by GPT-4o. However, we leverage it solely as a VQA filter for potential tool utilization.

The CoTA data comprises trajectories of thought, action, and observation. We derive QA pairs from this data using Qwen3-30B-A3B. For open-ended questions within this set, we employ Qwen2.5-VL-32B to synthesize multiple-choice options, thereby transforming the entirety of the CoTA data into a MCQA format. Subsequently, we task Qwen2.5-VL-7B with answering all the questions and retain only those that are answered incorrectly. These erroneous examples are more likely to exhibit both correct and incorrect responses during the sampling of rollouts in GRPO training. The resulting dataset is then randomly split into a cold-start dataset and a RL dataset. For the cold-start dataset, we utilize GPT-4.1 as the agent to synthesize tool-integrated reasoning rollouts, filtering out any rollouts that lead to incorrect answers.

The SAT data is inherently in a MCQA format. We randomly shuffled the answer options and subsequently filtered out data instances for which Qwen2.5-VL-7B provided an incorrect response.

### B. Experiment Setup Details

###### B.1. Benchmark and Dataset Details

In this paper, we evaluate five multi-modal benchmarks covering diverse visual reasoning capabilities: CV-Bench (Tong et al., 2024a), Blink (Fu et al., 2024), Blink-Hard (Bigverdi et al., 2024), MMVP (Tong et al., 2024b), and a 1,000 randomly selected subset from SAT (Ray et al., 2024). Our dataset construction incorporates single-image samples from both SAT (Ray et al., 2024) and Taco (Ma et al., 2024). The following sections provide detailed descriptions of these benchmarks and datasets:

- • CV-Bench (Tong et al., 2024a): A vision-centric benchmark containing 2,638 manually-inspected examples for evaluating MLLMs. It repurposes standard vision benchmarks (ADE20K (Zhou et al., 2017), COCO (Lin et al., 2014), Omni3D (Brazil et al., 2023)) to assess fundamental 2D and 3D understanding capabilities including spatial relationships, object counting, depth ordering, and relative distance estimation. Questions are programmatically constructed and manually verified for clarity and accuracy.
- • Blink (Fu et al., 2024): A benchmark with 3,807 multiple-choice questions reformatting 14 classic computer vision tasks to test fundamental visual perception abilities. Despite humans achieving 95.70% accuracy, even top models like GPT-4V and Gemini achieve only 51.26% and 45.72% accuracy respectively. Blink highlights significant gaps between human-level visual perception and current MLLM capabilities. Specifically, we experiment with relative depth, count and spatial reasoning tasks in Blink.
- • Blink-Hard (Bigverdi et al., 2024): A challenging benchmark focused on visual perception tasks requiring deeper 3D understanding and spatial reasoning. It evaluates whether MLMs can produce and reason with intermediate representations like depth maps and bounding boxes. The benchmark serves as a testbed for approaches like Perception Tokens that assist models in tackling complex visual reasoning problems.
- • MMVP (Tong et al., 2024b): Contains 300 questions based on 150 pairs of “CLIP-blind” images that appear

- similar to CLIP models despite clear visual differences. The benchmark categorizes questions across nine visual patterns and reveals systematic shortcomings in MLLMs’ visual perception abilities, with even top models showing performance gaps of over 50% compared to humans.
- • MMMU (Yue et al., 2024): A massive, multi-discipline benchmark requiring expert-level knowledge to evaluate advanced MLLMs. It consists of 11.5K questions spanning six core disciplines from college-level exams and textbooks. MMMU tests perception and reasoning abilities on complex visual materials like diagrams, charts, and chemical formulas.
- • MMBench (Liu et al., 2024b): A comprehensive benchmark designed to evaluate MLLMs on a wide range of capabilities, from basic perception to advanced reasoning. It features a carefully curated set of multiple-choice questions covering 20 skill areas. MMBench employs a circular evaluation strategy on a held-out test set to provide a robust assessment of model performance while preventing overfitting.
- • MathVista (Lu et al., 2024): A benchmark focused on visual mathematical reasoning, designed to test MLLMs on a diverse set of 6,141 problems. Sourced from 28 existing math datasets and 3 newly created ones, its questions cover a wide range of tasks, including geometric problems, statistical chart interpretation, and scientific puzzles. MathVista requires models to integrate visual perception, mathematical logic, and multi-step reasoning.
- • SAT (Ray et al., 2024): Contains 218K question-answer pairs covering 22K synthetic scenes testing both static and dynamic spatial reasoning. Unlike benchmarks focused on static relationships, SAT (Ray et al.,

2024) incorporates perspective-taking and egocentric action recognition. Generated using a photo-realistic physics engine, it can be scaled and extended to include new scenarios.

- • Taco (Ma et al., 2024): A framework and dataset with over 1 million synthetic chains-of-thought-andaction traces (filtered to 293K high-quality examples). TACO (Ma et al., 2024) enables models to perform step-by-step reasoning while invoking external tools (OCR, depth estimation, calculators) to solve complex visual problems, showing performance improvements of up to 15% on challenging multimodal tasks.

###### B.2. Training Details

We fine-tune Qwen2.5-VL-3B and Qwen2.5-VL-7B on eight A800 GPUs. The detailed training parameters for cold-start and reinforcement learning are listed in Table 5 and Table 6. The reward curve of our method REVPT compared to text RL training is shown in Figure 8.

Table 5: Training hyperparameters

Name Value

bf16 True tf32 True per_device_train_batch_size 4 gradient_accumulation_steps 2 lr 1e-5 weight_decay 0 warmup_ratio 0.1 lr_scheduler_type cosine max_seq_len 16384

###### Table 6: Training hyperparameters

Name Value

bf16 True tf32 True per_device_train_batch_size 8 ppo_mini_batch_size 128 num_generation 8 kl_loss_coef 1e-3 lr 2e-6 weight_decay 0.01 warmup_ratio 0.03 lr_scheduler_type cosine max_seq_len 16384

###### Training Reward Curves Comparison

1.0

0.8

NormalizedReward(0-1)

0.6

0.4

0.2

ReVPT

Text-only GRPO

0.0

0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190

Training Step

- Figure 8: Our reward gradually upgrade and reach converge.

The hyperparameters used for the agent are detailed in Table 7. During training with the veRL framework, we observed a degradation in the response quality when batch-inferencing with Qwen2.5-VL by vllm as the batch size increased. Consequently, we imposed a limit on the batch size used for each inference step.

Table 7: Agent config

Name Value max_turns 5 max_tokens_per_turn 1024 inference_batch_size 8

### C. Case Study

We show some case studies that our model successfully using tools in Figure 9 ,10 ,11 and 12. We also include some failure cases that visual tools return wrong output and disturb model’s reasoning chain in 13, 14 and 15.

Question: Considering the relative positions of the bottle (annotated by the red box) and the fork in the image provided, where is the bottle (annotated by the red box) located with respect to the fork? (A) left (B) right

Answer: (B)

[Figure 48]

[Figure 49]

Assistant:

<think> bottle (red box) is located on the table near the center of the image. The fork is also visible, likely placed on the table as well. To determine their relative positions, I should analyze the locations of both objects. The bottle appears to be more towards the middle-right side of the table, while the fork is closer to the center or slightly left. To confirm, I will use the edge detection tool to better distinguish the boundaries and positions of these objects.</think>

<tool_call>{"name": "edge_detection", "arguments": {"image_id": 0}}</tool_call>’ User: <image> <result>The edge map for image 0.</result> Assistant:

<think>From the edge detection result, it is clear that the bottle (in the red box) is positioned more towards the center-right area of the table. The fork is visible beside the bottle, further to the right. This confirms that the bottle is to the right of the fork.</think>

<answer>Based on the image and the edge detection result, the bottle (red box) is located to the right of the fork. Therefore, the answer is \boxedB.</answer>

- Figure 9: Case study for the tool type: Edge Detection.

Question: A bounding box is an annotated rectangle surrounding an object. The edges of bounding boxes should touch the outermost pixels of the object that is being labeled. Given the two bounding boxes on the image, labeled by A and B, which bounding box more accurately localizes and encloses the knife? Select from the following options. (A) Box A (B) Box B Answer: (B)

[Figure 50]

[Figure 51]

###### Assistant:

<think>Letś carefully examine the positions of the two bounding boxes, A and B. Box A is around what appears to be a book or a stack of papers, while Box B is around a silver object that resembles a knife. The knife is clearly visible and positioned in Box B, whereas Box A does not contain any knife-like object. To confirm, Iĺl double-check the boundaries of both boxes and their contents.</think>

<tool_call>{"name": "zoom_in", "arguments": {"image_id": 0, "bbox": [200, 490, 480, 720], "factor": 1.5}}</tool_call> User: <image> <result>Zoomed image 0 on [200, 490, 480, 720] with 1.5x magnification.</result>

###### Assistant:

<think>The zoomed-in view shows the region where both bounding boxes A and B overlap. By closely inspecting this area, I can see that Box B encompasses the handle and blade of a knife, which is not present in Box A. This confirms my initial assessment. Therefore, Box B is the one that correctly encloses the knife.</think>

<answer>The bounding box labeled B more accurately localizes and encloses the knife, as it includes the knife’s handle and blade. Therefore, the correct answer is \boxed{B}.</answer>

- Figure 10: Case study for the tool type: Zoom In.

Question:How many ties are in the image? Select from the following choices. (A) 6 (B) 5 (C) 3 (D) 2 (E) 4 (F) 0 Answer: (E)

[Figure 52]

[Figure 53]

Assistant: <think>To accurately determine the number of ties in the image, it’s best to use the object detection tool targeting "tie" to ensure all visible ties are counted, even those partially obscured or at different angles.</think>

<tool_call>{"name": "object_detection", "arguments": {"image_id": 0, "objects": ["tie"]}}</tool_call> User: <image>

<result>Image 1:Detected 4 object(s) in image 0: 1. tie(0.76): [87, 144, 108, 223] 2. tie(0.46): [370, 117, 387, 153] 3. tie(0.72): [247, 115, 262, 153] 4. tie(0.66): [505, 116, 517, 138]</result>

Assistant: <think>The object detection tool has identified four distinct "ties" in the image, each marked with a bounding box around the neck area. The bounding boxes correspond to the visible neckwear worn by each individual in the image. Since the tool’s output matches the visible evidence and there are no other neck accessories present, I can confidently answer the question based on this information.<//think>

<answer>Based on the object detection results and visual confirmation, there are 4 ties in the image. The tool detected one tie for the man on the left (wearing a suit), and three ties for the three men in military uniforms. \boxed{B}.</answer>

- Figure 11: Case study for the tool type: Object Detection.

Question:Which object is closer to the camera taking this photo, the table (highlighted by a red box) or the television (highlighted by a blue box)? Select from the following choices. (A) table (B) television Answer: (E)

[Figure 54]

[Figure 55]

Assistant: <think>To determine which object is closer to the camera.the table (red box) or the television (blue box).it’s best to use depth estimation. This will allow me to visually compare the relative distances of these two objects from the camera.<//think>

<tool_call>{"name": "depth_estimation", "arguments": {"image_id": 0}}</tool_call> User: <image> <result>Image 1:The colored depth map for image 0.</result>

###### Assistant:

<think>In the depth map, warmer colors indicate areas closer to the camera, while cooler colors represent regions farther away. The area corresponding to the table (red box in the original image) appears much warmer compared to the region where the television (blue box) is located. This suggests that the table is closer to the camera than the television. No further tools are needed as the depth map provides clear evidence.<//think>

<answer>By analyzing the depth map, it is evident that the table (red box) is closer to the camera than the television (blue box). The table’s region shows significantly warmer colors, indicating proximity, while the television’s area is cooler, signifying it is farther away. //boxed{A}. <//answer>

- Figure 12: Case study for the tool type: Depth Estimation.

Question:How many pictures are in the image?? Select from the following choices. (A) 0 (B) 5 (C) 2 (D) 1 (E) 3 (F) 4 Answer: (E)

[Figure 56]

[Figure 57]

Assistant: <think> To accurately determine the number of pictures in the image, I should use the object detection tool to identify all the pictures present. This will help ensure I don’t miss any that may not be immediately obvious.<//think>

<tool_call>{"name": "object_detection", "arguments": {"image_id": 0, "objects": ["picture"]}} </tool_call> User: <image> <result>Image 1:Detected 3 object(s) in image 0: 1. picture(0.76): [483, 144, 570, 279] 2. picture(0.72): [115, 158, 188, 247] 3. picture(0.42): [747, 194, 781, 280] </result>

###### Assistant:

<think>The object detection tool has identified three pictures in the image, each marked with a bounding box and label. The bounding boxes correspond to the framed objects on the wall: one on the left above the bookshelf, one in the middle above the cabinet, and one on the right wall. Since the detection confidence is high and matches the visible frames, there is no need for further tool usage. I am ready to answer the question based on this information.</think>

<answer>Based on the object detection results, there are three pictures in the image. The bounding boxes and labels confirm this count. \boxed{C}. </answer>

- Figure 13: Case study for the error:Model’s Misunderstanding of Tool Results (Object Detection).

Question: Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera? Select from the following choices. (A) A is closer (B) B is closer Answer: (B)

[Figure 58]

[Figure 59]

###### Assistant:

<think>Looking at the image, both points A and B are marked near the face of the doll. To determine which one is closer to the camera, I need to analyze their positions relative to the viewerś perspective. Point A appears lower on the dollś face compared to point B, which is higher up on the forehead. However, this could be due to the camera angle or perspective. To be more certain, I should use depth estimation to compare the actual spatial distances of points A and B from the camera.</think>

<tool_call>{"name": "depth_estimation", "arguments": {"image_id": 0}}</tool_call> User: <image> <result>Image 1:The colored depth map for image 0.</result>

###### Assistant:

<think>In the depth map, warmer colors indicate areas that are closer to the camera, while cooler colors represent regions farther away. Observing the map, the region around point A (near the lower part of the face) appears much warmer compared to the area around point B (higher on the forehead). This confirms that point A is indeed closer to the camera than point B.</think>

<answer>Based on the depth map, point A is closer to the camera than point B. The warm color at point A supports this conclusion. \boxed{A}</answer>

- Figure 14: Case study for the error: Model’s Misunderstanding of Tool Results (Depth Estimation).

Question: How many pillows are in the image? Select from the following choices. (A) 2 (B) 0 (C) 1 (D) 3 Answer: (B)

[Figure 60]

###### Assistant:

<think>To accurately determine the number of pillows in the image, it’s best to use object detection targeting pillowör¨ similar objects. This will help ensure I don’t miss any pillows that may not be immediately obvious.</think>

<tool_call>{"name": "object_detection", "arguments": {"image_id": 0, "objects": ["pillow"]}}</tool_call> User: <result>No objects matching ’pillow.’ detected in image 0.</result> Assistant:

<think>The object detection tool did not find any pillows, but it is not always reliable. Visually inspecting the image, there appears to be one pillow visible on the bed in the background. The rest of the items look like bags, clothes, or bedding. Since the detection tool missed it, and only one pillow is clearly visible, it seems reasonable to conclude that there is only one pillow in the image.</think>

<answer>After careful inspection and considering the output of the object detection tool, there is only one visible pillow in the image.\boxed{C}</answer>

- Figure 15: Case study for the error: Flawed Tool Results (Object Detection).

