[Figure 1]

## EditThinker: Unlocking Iterative Reasoning for Any Image Editor

Hongyu Li1,2 Manyuan Zhang2† Dian Zheng2,3 Ziyu Guo2,4 Yimeng Jia2 Kaituo Feng2,3 Hao Yu5 Yexin Liu2 Yan Feng2 Peng Pei2

Xunliang Cai2 Linjiang Huang1 Hongsheng Li3 Si Liu1‡ 1Beihang University 2Meituan 3CUHK MMLab 4CUHK IMIXR 5Tsinghua University Porject Page: https://appletea233.github.io/think-while-edit

|[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>Animate the cat in the image<br><br>Specify cartoon style, preserve pose and add motion cues, keep scene consistent<br><br>[Figure 8]<br><br>✍<br><br>Pose and background differ from original，fur pattern only loosely preserved<br><br>[Figure 9]<br><br>No visible motion or cartoon 🧐 style; cat remains static and unanimated.<br><br>[Figure 10]<br><br>🧐 Fur markings and texture not<br><br>faithfully preserved， motion effects minimal.<br><br>[Figure 11]<br><br>🧐 Pose, fur pattern, and bedding<br><br>detail still deviate from the original.<br><br>[Figure 12]<br><br>🧐 Animates the cat with clear motion while preserving its identity and surroundings.<br><br>[Figure 13]<br><br>🥳<br><br>Enforce original lying pose and curled paws，need bedding background consistency.<br><br>[Figure 14]<br><br>✍<br><br>Strengthen demands on fur, facial markings, dark bedding, and playful motion effects.<br><br>[Figure 15]<br><br>✍<br><br>Emphasize strict pose preservation, precise blackand-white pattern.<br><br>[Figure 16]<br><br>✍<br><br>Source Image<br><br>Adjust the background to a city<br><br>Maintain the original text, clarity, and sharpness of the street signs<br><br>[Figure 17]<br><br>✍<br><br>The street signs experienced distortion, losing their original sharpness and clarity<br><br>[Figure 18]<br><br>The text on the signs is 🧐 distorted and partially unreadable ('Mt tn Park Rd')<br><br>[Figure 19]<br><br>🧐 'Mt Lookout Rd' is missing<br><br>entirely, and 'North Park Rd' text has significant warping<br><br>[Figure 20]<br><br>🧐 While the background was<br><br>replaced, the street signs exhibit significant issues.<br><br>[Figure 21]<br><br>🧐 Appears authentic and remain<br><br>well-integrated and preserved details.<br><br>[Figure 22]<br><br>🥳<br><br>Maintain clarity of the street signs exactly as they appear in the original image<br><br>[Figure 23]<br><br>✍<br><br>avoiding any warping and distortion during the background replacement process<br><br>[Figure 24]<br><br>✍<br><br>The clarity of the street signs ('Mt Lookout Rd' and 'North Park Rd') must remain<br><br>[Figure 25]<br><br>✍<br><br>Source Image<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>Adjust the image style to a bubble-like aesthetic<br><br>Does not make the overall scene or subject appear 'bubble-like'<br><br>[Figure 33]<br><br>🧐<br><br>Adding semi-transparent, shiny bubble elements of various sizes throughout the scene<br><br>[Figure 34]<br><br>✍<br><br>The overall scene still retains the dry leaves background<br><br>[Figure 35]<br><br>🧐<br><br>Change the background to an airy, light, and colorful ambience<br><br>[Figure 36]<br><br>✍<br><br>The background has been transformed into a colorful ambience that fits bubbles<br><br>[Figure 37]<br><br>🥳<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>Change the color of umbrellas to brown<br><br>The brown overlays appear flat and artificial<br><br>[Figure 45]<br><br>🧐<br><br>The brown overlays appear flat and artificial<br><br>[Figure 46]<br><br>✍<br><br>Fails to preserve the original patterns, textures<br><br>[Figure 47]<br><br>🧐<br><br>Ensuring that all original paper-like textures are clearly preserved<br><br>[Figure 48]<br><br>✍<br><br>Appears authentic and remain well-integrated and preserved details<br><br>[Figure 49]<br><br>🥳<br><br>(a) Source Image Source Image<br><br>Origin Prompt Reﬁned Prompt (simplify) Reasoning Process (simplify)<br><br>Turn 1 Turn 2 Turn 3 Turn 1 Turn 2 Turn 3<br><br>Turn 1 Turn 2 Turn 3 Turn 4 Turn 5<br><br>Turn 1 Turn 2 Turn 3 Turn 4 Turn 5<br><br>[Figure 50]<br><br>✍ 🧐 🥳<br><br>[Figure 51]<br><br>[Figure 52]|
|---|

# arXiv:2512.05965v1[cs.CV]5Dec2025

|[Figure 53]<br><br>[Figure 54]<br><br>(b)|
|---|

Figure 1. Overview of EditThinker. Subfigure (a) illustrates our multi-turn Think-while-Edit pipeline that iteratively Critiques, Refines, and Repeats the editing instruction, while subfigure (b) reports results on four image editing benchmarks, showing large gains for three existing editing methods and we use the dev version of FLUX.1 Kontext (denoted as FLUX.1 Kontext in the figure).

1

#### Abstract

Instruction-based image editing has emerged as a prominent research area, which, benefiting from image generation foundation models, have achieved high aesthetic quality, making instruction-following capability the primary challenge. Existing approaches improve instruction adherence via supervised or reinforcement learning, yet single-turn success rates remain limited due to inherent stochasticity and a lack of deliberation. In this work, we propose a deliberative editing framework to “think” while they edit, which simulates the human cognitive loop by iteratively executing a Think-while-Edit cycle: Critiquing results and Refining instructions , followed by Repeating the generation until satisfactory. Specifically, we train a single MLLM, EditThinker, to act as the reasoning engine of this framework, which jointly produce the critique score, reasoning process, and refined instructions. We employ reinforcement learning to align the EditThinker’s thinking with its editing, thereby generating more targeted instruction improvements. Extensive experiments on four benchmarks demonstrate that our approach significantly improves the instruction-following capability of any image editing model by a large margin. We will release our data construction framework, datasets, and models to benefit the community.

#### 1. Introduction

Instruction-based image editing aims to edit a user-given image following the given instructions, which has a wide range of applications in content creation and world simulation. Current state-of-the-art editing methods [2, 28, 30] are typically built by fine-tuning strong image generation foundation models, contributing to excellent aesthetic quality of edited images. Consequently, the primary challenge has instead shifted toward achieving precise instruction-following capability. Instruction-based image editing is emerging as a core capability for interactive visual systems, enabling practical applications such as digital content creation, virtual avatar design, and controllable world simulation. Compared to text-to-image generation, this task is inherently more challenging as it requires the model to simultaneously preserve identity, perform localized semantic modifications, and respect long-range visual consistency, all under freeform natural language instructions.

Recently, inspired by the remarkable success of reinforcement learning (RL) in eliciting reasoning capabilities [6, 7, 14, 32, 43], the RL paradigm has also been extended to image editing [15, 20]. However, as shown in Figure 1, even after RL, the instruction-following performance in single-

†Project Leader. ‡Corresponding Author.

turn (i.e., Turn1 in the image) remains limited. In practice, a single-turn editing model is tasked with jointly performing instruction understanding, visual planning, and content generation within a single step. Due to this coupled and one-pass nature, the model is deprived of the opportunity to self-correct intermediate errors, leading to issues such as missing attributes. In essence: Current models mainly act as reactive executor, rather than a reflective thinker.

In this work, we explore a novel perspective: enabling the editing system to “think” while it edits. Instead of improving the editor model itself, we equip it with a Thinker —implemented as a Multimodal Large Language Model (MLLM)— that executes a Critique-Refine-Repeat loop. Specifically, the Thinker evaluates the editing result (Critique), refines the instruction based on identified deficiencies (Refine), and resubmits it to the editor for regeneration (Repeat). This pipeline can effectively address instructionfollowing limitations across different models. To validate this concept, we employed GPT-4.1 [8] as an expert Thinker to conduct multi-round instruction iterations on several state-of-the-art editing models (Qwen-ImageEdit [28], Flux-Kontext [2], Omnigen2 [30]). Remarkably, without fine-tuning the editing models, we achieved significant performance improvements across all models.

Furthermore, we propose our EditThinker, a MLLM with reasoning capability that implements this Think-whileEdit paradigm for any image editor. To achieve this, our framework incorporates two key contributions. First, we train a single MLLM as the EditThinker to jointly output the critique score, the refined instruction, and its underlying reasoning process. After supervised fine-tuning (SFT) to adapt to the output format, we employ reinforcement learning (RL) to bridge the Think-while-Edit gap, aligning the EditThinker’s planning with the practical capabilities and failure modes of the image edit models. Second, we construct THINKEDIT-140k via a comprehensive multi-round instruction refinement framework. This automated pipeline generates tuples of high-fidelity source images, diverse editing requests, and detailed reasoning traces.

Extensive experiments on four widely used benchmarks demonstrate the effectiveness of our EditThinker across diverse editing scenarios and edit models, yielding consistent performance improvements in all evaluated settings. We further conduct comprehensive ablation studies to analyze the impact of key components, including the thinking paradigm, the number of reasoning turns, the training strategy, and the choice of expert thinker.

In summary, our main contributions are as follows:

- 1. We identify the limitation within single-turn instructionfollowing and propose a novel Think-while-Edit paradigm, reframing the editing task as an iterative reasoning process.
- 2. We propose EditThinker, a reasoning-driven MLLM

- trained with SFT and RL to iteratively critique, refine, and re-plan editing instructions.
- 3. We introduce THINKEDIT-140k, a large-scale multiround dataset with unified supervision signals for instruction refinement and reasoning-based training.
- 4. Extensive experiments on four widely used benchmarks demonstrate the effectiveness of our method across diverse editing scenarios and edit models.

#### 2. Related Work

###### 2.1. Image Editing

The emergence of diffusion models marked a paradigm shift in Text-to-Image (T2I) synthesis [16, 19, 25, 26, 37]. Image editing, however, imposes stricter constraints to balance attribute modification with background preservation. Early solutions, ranging from inversion-based techniques [10, 21, 22] to explicit spatial controls [39, 41], improved precision but often suffered from computational overhead or limited semantic flexibility. While initial instruction-tuning attempts [3] introduced natural language control, they faced generalization bottlenecks. Recently, the field has advanced towards robust instruction-tuned models [2, 17, 45] and general-purpose Multimodal LLMs or Unifed Model [15, 24, 29, 30], evolving alongside foundational architectures like flow matching [12]. Although the foundational capabilities of editing models continue to improve, their instruction-following ability remains limited due to the inherent stochasticity and lack of deliberation in single-turn editing. In this work, we pioneer a multi-round instruction iterative refinement paradigm that achieves performance improvements across any editing model, demonstrating the importance of the multi-round editing paradigm.

###### 2.2. Reward Models for Image Editing

Feedback and Reward Modeling in Image Editing. The correlation between Multimodal Large Language Models (MLLMs) and human perception [4, 42] has established the ”MLLM-as-a-Judge” paradigm, facilitating their use as reward models (RMs) for generative tasks [23, 36]. However, translating holistic evaluations into effective training signals for image editing is non-trivial. Early attempts using discrete scores [9] or dense logit-based values [31] often failed to capture the fine-grained nuances required for precise visual modifications. To address these limitations, recent research has pivoted towards domain-specialized reward modeling. Notably, EditReward [33] constructed a largescale human preference dataset to train a reward model capable of rigorous data filtering and alignment. Building on this, EditScore [20] developed a series of specialized reward models that surpass general-purpose VLM judges, successfully unlocking effective online reinforcement learning (RL) for editing policies.

Despite these advancements in RL application, a fundamental ”feedback lag” remains. Existing specialized RMs primarily provide outcome-oriented feedback—they evaluate the edited image after generation. This post-hoc signal acts as an external judge rather than an internal guide. In complex editing scenarios requiring multi-step reasoning, such scalar rewards fail to correct the intermediate logic of the generation process [44]. Consequently, an emerging paradigm seeks to utilize MLLMs not merely as judges, but as internal planners [19, 38]. In this work, we shift from maximizing a static post-hoc reward to harnessing the MLLM’s structured reasoning process to actively guide the editing model during execution.

#### 3. Think-while-Edit

To address the inherent limitations of current editing models in single-turn instruction following, we propose Thinkwhile-Edit framework, mimicking the human cognitive process of “critique, reflect, and edit” during creation.

###### 3.1. Overall Framework

Previous methods mainly operates in a single turn: given a source image Isrc and the origin instruction Ts, the editing model directly produces the final edited image. This process lacks the ability to iteratively refine the output or recover from a failed edit.

To address this limitation, we introduce a MLLM-based Thinker that transforms single-pass editing into an iterative, multi-turn process. Our framework explicitly decouples the editing workflow into two distinct roles: a Thinker for judging and reasoning, an Editor for execution, where the Thinker is trained via SFT and RL and the Editor is any existing image editing models (e.g., Qwen-Image-Edit, Flux-Kontext). Specifically, at each iteration t, the Thinker evaluates the previous output Ieditt−1 and generates the instruction following score St, refined instruction Tt and the reasoning process Rt at the same time as:

(St,Rt,Tt) = Thinker(Isrc,Ieditt−1,Tt−1.Ts). (1) Then the Editor executes the new instruction Tt on the

source image Isrc, generating the updated result Ieditt as:

Ieditt = Editor(Isrc,Tt). (2)

This iterative process, termed the Critique-RefineRepeat cycle, continues until the editing goal is achieved.

###### 3.2. Design of the EditThinker

We formulate EditThinker as a dual-role model that simultaneously evaluates and plans. Unlike decoupled approaches that use separate models for evaluation (a MLLMbased scorer) and planning (a LLM-based rewriter), EditThinker performs both tasks in a single forward pass.

Origin Prompt Ts Source Image 𝑰𝒔𝒓𝒄

Refined PromptT1 Source Image 𝑰𝒔𝒓𝒄

Refined PromptTt Source Image 𝑰𝒔𝒓𝒄

[Figure 55]

[Figure 56]

[Figure 57]

×t

Do not add a lighthouse to the foreground or center.

“Replace the sailboat on the right with a lighthouse.”

Make the size appropriately for the distance

###### …

###### Edit Model

###### Edit Model

###### Edit Model

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

Ts 𝑰𝒔𝒓𝒄

Ts 𝑰𝒔𝒓𝒄

Ts 𝑰𝒔𝒓𝒄

##### T1

##### Tt

Edited Image 𝑰𝒆𝒅𝒊𝒕𝒕 𝑆 < 𝑆

Edited Image 𝑰𝒆𝒅𝒊𝒕

Edited Image 𝑰𝒆𝒅𝒊𝒕 𝑆 < 𝑆

[Figure 69]

[Figure 70]

###### EditThinker

[Figure 71]

###### EditThinker

EditThinker

𝑆 ≥ 𝑆

Semantic

0 10

Semantic

Semantic

0 10

0 10

Successfully removes the sailboat on the right and placed the lighthouse right with correct size.

Successfully removes the sailboat on the right, but fails to add a lighthouse in its place.

Successfully removes the sailboat on the right, but the size of it still too big.

0 10 0 10

Quality Overall

0 10 0 10

0 10 0 10

Quality Overall

Quality Overall

Finish

Reasoning R1 Edit Score S1

Reasoning R2 Edit Score S2

Reasoning Rt Edit Score St

- Figure 2. The Pipeline of Think-while-Edit. EditThinker is a multi-round instruction iterative refinement framework. In the first round,

the original image Isrc and instruction Ts are fed into an editor to produce an initial edited image Ieditt . This edited image, along with the original image and instruction, is then fed into EditThinker, which generates the edit score St, refined prompt Tt, and corresponding reasoning process Rt. If the score falls below a threshold, the framework proceeds to the next iteration with the refined prompt until a satisfactory result is achieved.

Our key insight is that effective planning requires deep evaluation: the model must first critique the previous output (generating score St and reasoning Rt) before producing a refined instruction Tt. By generating Rt before Tt, EditThinker creates an explicit chain of thought that grounds instruction refinement in the visual critique of Isrc and Ieditt−1.

lowed by reinforcement learning (RL) to optimize instruction refinement based on actual editing feedback. The data construction process is detailed in Section 4.

###### 3.3.1. Supervised Fine-Tuning (Cold Start)

Using the expert (GPT-4.1) demonstration dataset (detailed in Sec. 4), the base MLLM learns to adopt our structured I/O format (e.g., <think>, <score>, <answer>), mimic the expert’s reasoning style, and understand the principles of critiquing and refining instructions.

To implement this dual-role design, we define a structured input-output format that explicitly encodes the evaluation-then-planning process.

Input Tuple. EditThinker receives a multimodal tuple (Isrc, Ieditt−1, Ts, Tt−1) at each iteration t, providing complete context of the editing state: Isrc and Ts represents the original reference, Ieditt−1 is the current result to be critiqued, and Tt−1 is the previous instruction that produced it.

###### 3.3.2. Reinforcement Learning Tuning (RLT)

The SFT model learns how the expert would ideally reason, but this reasoning is not grounded in the practical limitations of real editors. The model has never observed actual editing failures or learned which types of instructions are prone to misinterpretation by specific editors. Consequently, an instruction Tt that appears optimal to the SFT model may still fail when executed by actual editors like Qwen-Image-Edit. This creates a gap between ideal reasoning and practical execution.

Structured Output Format. The output is a structured text string that serializes EditThinker’s reasoning process:

|<think> Reasoning process... </think> <score> [Ssem, Squal] </score> <answer> Refined prompt Tt </answer>|
|---|

Here, Squal is the perceptual quality of Ieditt−1, and Ssem is the semantic alignment with the original instruction Ts relative to Isrc. Both scores range from 0 to 10.

To bridge this gap, we introduce an RL stage that optimizes EditThinker based on actual editing feedback. We employ standard GRPO (Group Relative Policy Optimization) with a carefully designed reward function. As defined in Sec. 3.2, EditThinker performs as a dual roles agent (i.e., Critic and refiner), we design a multi-component reward that provides learning signals for both aspects as follows:

###### 3.3. Training of EditThinker

Training EditThinker to perform this dual-role task requires a specialized dataset and a multi-stage training strategy. We adopt a two-stage approach: first, supervised fine-tuning (SFT) to learn the output format and basic reasoning, fol-

Critic Reward. This component trains the EditThinker to be a more accurate critic. The model outputs predicted

###### Edit Data Pooling

Trajectory Pooling

Selected Trajectories

Selected Steps

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

T1 3 Tn 4 T1 5

[Figure 86]

[Figure 87]

Trajectory Filter

Trajectory Generation

Step-Wise Filter

[Figure 88]

[Figure 89]

[Figure 90]

… …

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

[Figure 102]

- 3

- 4

T1

T1

T1

T1 5

T1

[Figure 103]

… … …

…

[Figure 104]

[Figure 105]

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

…

[Figure 117]

Tn

Tn

Tn

Tn

Tn

2

###### Think While Edit

###### Task Distribution Score Distribution

[Figure 118]

[Figure 119]

Ts: turn to red

Ts: remove face

[Figure 120]

35

[Figure 121]

𝑰𝒔𝒓𝒄 Ts

Edit Thinker Expert

[Figure 122]

[Figure 123]

[Figure 124]

30

- T1 R1 2

[Figure 125]

[Figure 126]

[Figure 127]

- T2 R2 3

[Figure 128]

###### 8

- T1 R1

[Figure 129]

[Figure 130]

- T2 R2

- S1
- S2

- S1
- S2

25

20

[Figure 131]

[Figure 132]

[Figure 133]

EditScorer

[Figure 134]

[Figure 135]

[Figure 136]

- 2

- 3

###### Tt <stop?>

Rt

15

…

…

…

…

10

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

5

Tt Rt 9

Tt Rt

St

St

0

subject-driven style

…

…

…

…

0 2 4 6 8 10

background reasoning

<stopx> <stop√>

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

𝑰𝒆𝒅𝒊𝒕𝒕

[Figure 150]

Sn 1

Tn Rn 4

Tn Rn

others

[Figure 151]

Sn

Tt

[Figure 152]

[Figure 153]

Trajectory 1 √ Trajectory 2 x

[Figure 154]

[Figure 155]

Image Prompt S Score

𝑰𝒔𝒓𝒄

Image Editor

𝑀𝑎𝑥(𝑆 ) ≥ 𝑆 𝑀𝑎𝑥(𝑆 ) < 𝑆

- Figure 3. Data construction pipeline of our THINKEDIT. We construct our dataset through four sequential steps: (1) Trajectory Generation: We use several image edit models and expert evaluator GPT-4.1 to iteratively edit image, evaluate it and generates refined instructions until issuing a ⟨stop⟩ token. (2) Trajectory Filter: An edit scorer assigns scores St to each step, retaining only trajectories where max(St>1) ≥ S1 and truncating them at the highest-scoring step k. (3) Step-wise Filter: We unroll trajectories into individual

training samples pairing inputs (Isrc, Ieditt−1, Ts, Tt−1) with outputs (Rt, Tt), then balance the dataset across task types and score distributions. (4) Data Partition: The filtered data is split for SFT and RL training.

where Rformat is the basic reasoning format reward, and α + β + γ = 1 .

scores St (including Ssem and Squal) that should align with the actual quality of the edited result. We employ GPT-4.1

as the critic expert (E) to evaluate the resulting image Ieditt . The critic reward, Rcritic, penalizes the prediction error as:

#### 4. THINKEDIT Dataset

To train the EditThinker, we require a high-quality dataset that captures the multi-turn Think while Edit cycle. As shown in Figure 3, we designed an automated data construction pipeline to simulate this process, consisting of four sequential steps: Trajectory Generation, Trajectory Filter, Step-wise Filter, and Data Partition. This pipeline allowed us to construct our THINKEDIT-140k dataset. We detail each step below.

Rcritic = −|St − E(Isrc,Ieditt ,Ts)|. (3)

This reward encourages EditThinker to calibrate its selfassessment: overestimating quality (predicting 9 when the actual score is 5) or underestimating both incur penalties. Through this feedback, the model learns to align its internal critique with the actual editing outcomes.

Edit Reward. This is the primary reward that trains the EditThinker to be a better refiner. It incentivizes the model to generate an instruction Tt that leads to a measurable improvement in image quality and instruction following . We use a differential reward, comparing the “before” state (Ieditt−1) and the “after” state (Ieditt ) using the same expert E:

###### 4.1. Trajectory Generation

The first stage focuses on simulating the multi-turn “Think while Edit” cycle. The pipeline begins with an Edit Data Pool containing diverse (Isrc, Ts) pairs. At each step t, the edit thinker expert (GPT-4.1) evaluates the current state (based on Isrc, Ts, and Ieditt−1) and generates a new instruction (Tt), reasoning process (Rt) and ⟨stop⟩ token.

Redit = E(Isrc,Ieditt ,Ts) − E(Isrc,Ieditt−1,Ts). (4)

This reward is positive only if the generated instruction Tt successfully prompted the Editor to produce a better image than the previous step. This directly grounds the planning ability of EditThinker in the practical execution results.

Notably, the expert does not output a score (St). Instead, it directly determines when to halt the process by issuing a ⟨stop⟩ token. This design choice stems from our finding that a single expert struggles to maintain high performance in both task refinement and output scoring simultaneously.

The final reward Rtotal is as follows:

If a ⟨stop⟩ token is not issued, the image editor uses the

Roverall = αRformat + βRcritic + γRedit, (5)

new Tt to produce Ieditt . This loop continues until the expert triggers the ⟨stop⟩ condition (or a max-iteration limit N is

hit), thus completing a full trajectory.

###### 4.2. Trajectory Filter

Since the edit thinker expert only generates refined instructions and a ⟨stop⟩ token without quality scores, we employ

an additional edit scorer to evaluate each step Iedit(t) and assign a score St. After scoring all steps (S1,...,Sn), we apply a two-stage filtering process:

Filter Failed Trajectories. We retain only trajectories where at least one subsequent step (t > 1) achieves a score higher than or equal to the initial step (i.e., max(St>1) ≥ S1). Trajectories failing this condition are discarded.

Truncate Kept Trajectories. For retained trajectories, we identify the step k with the highest score (Sk = max(St≥1)) and truncate the trajectory to include only steps from 1 to k. All subsequent steps (t > k) are discarded.

###### 4.3. Step-wise Filter

Finally, we process the curated trajectories from the Trajectory Filter to create the final training data through two steps: Sample Extraction. First, we unroll the truncated trajectories. Each individual step t within a trajectory is converted into a distinct training sample. This sample pairs an input tuple (Isrc, Ieditt−1, Ts, Tt−1) with its corresponding ground-truth expert output (Rt,Tt). The score St for that step, while retaining the score St as metadata for subsequent filtering.

Distribution Balancing. We apply a final filtering step to balance the dataset along two dimensions:

- • Task Distribution: We balance samples across different task types (e.g., object removal, color modification, adding items) to ensure uniform coverage.
- • Score Distribution: We normalize samples across score levels to ensure balanced representation of editing quality.

###### 4.4. SFT and RL Data Split

After the Trajectory Filter, we obtained a large pool of curated, high-quality trajectories. From this collection, we create two distinct datasets for our Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) phases. The split is based on the principle that SFT requires stable, highquality examples, while RL benefits most from dynamic examples of improvement.

RL Dataset. We first identify trajectories that are most valuable for reinforcement learning. The key criterion is high intra-trajectory score variance (i.e., ”high-fluctuation” scores, Var(St) > θ). These trajectories represent challenging cases where the model initially struggled but then managed to improve, providing a rich reward signal for learning.

We filtered for a pool of 10k such high-variance trajectories, while also ensuring this set was balanced across differ-

ent task types and score distributions. When unrolled, these trajectories yielded 27k step-wise samples, which constitute our RL dataset.

SFT Dataset. The SFT dataset is intended to teach the model the correct, stable refinement behavior. We therefore selected samples characterized by low score variance or consistent high quality. These ”low-fluctuation” steps typically represent more straightforward, correct, and reliable refinement examples. This process resulted in a separate dataset of 140k step-wise samples for SFT.

#### 5. Experiments

- 5.1. Experimental Setup

Implementation Detail. EditThinker is built upon the Qwen3-VL-8B-Instruct [1]. We perform SFT on our newly constructed THINKEDIT-SFT-140k dataset for one epoch. Key hyperparameters for training include a learning rate of 2 × 10−5, a batch size of 32. And we preform RL on THINKEDIT-RL-10k dataset for one epoch. Key hyperparameters for training include a learning rate of 2 × 10−6, a global batch size of 128, and a rollout number(N) of 8 for generation, a KL divergence penalty with a coefficient of 1 × 10−3. MAX PIXELS is set to 1024 × 1024.The entire training process is conducted on 8 H800 GPUs and takes approximately 48 hours. For inference, we employ our “think while edit” paradigm with OmniGen2[30], Flux Kontext [dev][2] and Qwen-Image-Edit[28].

Benchmarks and Baselines. To comprehensively validate the effectiveness of our ”think while edit” paradigm, we conduct a composite evaluation on four distinct benchmarks: ImgEdit-Bench [40], GEdit-Bench [18] , RISEBench [47], and KRIS-Bench [34]. This suite of benchmarks was chosen for a multi-faceted assessment, with RISEBench and KRIS-Bench specifically focusing on evaluating the reasoning capabilities of the edit models.

- 5.2. Main Results

We evaluate our EditThinker framework across a comprehensive suite of four benchmarks to assess its performance on both general and reasoning-based editing tasks. For general image editing, we use ImgEdit-Bench and GEditBench-EN (results in Table 1). For complex reasoningbased editing, we utilize RISE-Bench and Kris-Bench (results in Table 2).

Performance on General Editing. As shown in Table 1, our Think-while-Edit framework consistently and significantly enhances the performance of all base models. On ImgEdit-Bench, EditThinker boosts the Overall score of FLUX.1-Kontext [Dev] from 3.44 to 3.98, OmniGen2 from

- 3.4 to 3.5, and Qwen-Image-Edit from 4.36 to 4.37. This achieves highly competitive performance, surpassing several state-of-the-art models. This strong performance gen-

- Table 1. Comparison of fine-tuning results of different models on our dataset on ImgEdit-Bench. ‡ indicates results from our own tests without fine-tuning. Note that the performance of +EditThinker-Expert-GPT4.1 represents the oracle upper bound.

Model

ImgEdit-Bench GEdit-Bench-EN Add Adjust Extract Replace Remove Background Style Hybrid Action Overall G SC G PQ G O

Open-source Models IP2P [3] 2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88 3.58 5.49 3.68 AnyEdit [11] 3.18 2.95 1.88 2.47 2.23 2.23 2.85 1.56 2.65 2.45 3.18 5.82 3.21 UltraEdit [46] 3.44 2.81 2.13 2.96 1.45 2.86 3.76 1.91 2.98 2.70 - - OmniGen [35] 3.47 3.04 1.71 2.94 2.43 3.21 4.19 2.24 3.38 2.96 5.96 5.89 5.06 Step1X-Edit [17] 3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06 7.66 7.35 6.97 ICEdit [45] 3.58 3.39 1.73 3.15 2.93 3.08 3.84 2.04 3.68 3.05 - - BAGEL [5] 3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20 7.36 6.83 6.52 OmniGen2 [30] 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44 7.16 6.77 6.41 Ovis-U1 [27] 4.13 3.62 2.98 4.45 4.06 4.22 4.69 3.45 4.61 4.00 - - 6.42 FluxKontext dev [13] 3.76 3.45 2.15 3.98 2.94 3.78 4.38 2.96 4.26 3.52 6.52 7.38 6.00 UniWorld-V2 [15] 4.29 4.44 4.32 4.69 4.72 4.41 4.91 3.83 4.83 4.49 8.39 8.02 7.83

Proprietary Models GPT-4o 4.61 4.33 2.9 4.35 3.66 4.57 4.93 3.96 4.89 4.20 - - 7.49

Think-while-Edit OmniGen2‡ 3.91 3.23 2.03 2.84 3.11 3.94 4.59 2.76 4.69 3.41 6.47 7.04 6.03 + EditThinker-8B 3.68 2.9 3.14 2.83 3.16 3.88 4.62 2.35 4.48 3.52 6.59 7.16 6.28 + EditThinker-Expert-GPT4.1 4.21 3.28 3.04 3.80 3.39 4.16 4.61 2.97 3.39 3.81 7.34 7.24 6.78

Flux-Kontext-dev‡ 3.83 3.55 2.18 3.91 2.74 3.79 4.42 2.82 4.18 3.44 6.62 7.61 6.18 + EditThinker-8B 3.82 3.80 3.52 4.09 3.88 4.09 4.52 3.21 4.44 3.98 7.59 7.63 7.02 + EditThinker-Expert-GPT4.1 4.08 4.01 3.45 4.44 3.75 4.19 4.59 3.73 4.57 4.13 7.83 7.66 7.19

Qwen-Image-Edit‡ 4.59 4.32 3.79 4.57 3.86 4.54 4.83 3.85 4.7 4.36 8.01 7.87 7.49 + EditThinker-8B 4.23 4.43 4.24 4.20 4.21 4.44 4.76 3.91 4.68 4.40 8.30 7.86 7.73 + EditThinker-Expert-GPT4.1 4.47 4.27 4.18 4.58 4.59 4.55 4.81 3.72 4.77 4.49 8.57 7.86 7.90

- Table 2. Comparison of model performance on RISE-Bench. ‡ indicates results from our own tests with official model checkpoint.

which tests complex spatial, causal, and temporal reasoning, our EditThinker framework provides a stable performance lift for all models. FLUX.1-Kontext [Dev] improves from 5.8 to 14.4, OmniGen2 from 3.1 to 3.4, and QwenImage-Edit from 8.9 to 17.8.

Model Temporal Causal Spatial Logical Overall Proprietary Models

Seedream-4.0 12.9 12.2 11.0 7.1 10.8 GPT-Image-1 34.1 32.2 37.0 10.6 28.9 Gemini-2.5-Flash-Image 25.9 47.8 37.0 18.8 32.8

Effect of the Expert Model’s Capability. We also observe that the performance of our framework scales with the capability of the EditThinker (Expert Model) itself. The tables show results for the same base model (e.g., FLUX.1-Kontext [Dev]) paired with different experts, such as EditThinker-8B and the stronger EditThinker (GPT-4.1). On ImgEdit-Bench, EditThinker-8B improves the FLUX score to 3.98, while the stronger EditThinker (GPT-4.1) boosts it even further to 4.13. This pattern holds across other models and benchmarks, demonstrating that using a more capable expert model as the ”thinker” directly translates to a greater performance enhancement in the final editing results.

Open-source Models

Step1X-Edit 0.0 2.2 2.0 3.5 1.9 Ovis-U1 1.2 3.3 4.0 2.4 2.8 FLUX.1-Kontext-Dev 2.3 5.5 13.0 1.2 5.8 BAGEL 2.4 5.6 14.0 1.2 6.1 Qwen-Image-Edit 4.7 10.0 17.0 2.4 8.9 BAGEL (w/ CoT) 5.9 17.8 21.0 1.2 11.9

Think-while-Edit

OmniGen2‡ 2.4 1.1 7.0 1.2 3.1 + EditThinker-8B 4.7 7.8 5.0 3.5 3.4 + EditThinker-Expert-GPT4.1 17.6 8.9 8.0 2.4 9.2

FLUX.1-Kontext-Dev‡ 2.3 5.5 13.0 1.2 5.8

+ EditThinker-8B 11.8 17.8 20.0 7.1 14.4 + EditThinker-Expert-GPT4.1 16.5 24.4 33.0 5.9 20.6

Qwen-Image-Edit‡ 4.7 10.0 17.0 2.4 8.9

+ EditThinker-8B 10.8 23.3 27.0 8.2 17.8 + EditThinker-Expert-GPT4.1 25.9 32.2 40.0 9.4 27.5

eralizes to the GEdit-Bench-EN dataset, where our method again provides stable gains, improving FLUX.1-Kontext [Dev] from 6.18 to 7.05, OmniGen2 from 6.19 to 6.28, and Qwen-Image-Edit from 7.49 to 7.73.

Performance on Reasoning Editing. Crucially, our method’s advantages are not limited to general edits; it provides equally consistent improvements on tasks requiring deep reasoning, as detailed in Table 2. On the RISE-Bench,

###### 5.3. Ablation Study

We conduct a series of ablation studies to validate the effectiveness of the key components within our EditThinker framework. We use the FLUX.1-Kontext [Dev] model as our baseline and evaluate on GEdit-Bench-EN and ImgEditBench, unless specified otherwise.

Think Pattern Analysis We categorize model editing thinking paradigms into two main approaches: Think before Edit and Think while Edit. Think before Edit rewrites an op-

Table 3. Ablation on GEdit-Bench-EN with Thinking Paradigm

GEdit-Bench-EN ↑ G SC G PQ G O FLUX.1-Kontext [Dev] 6.62 7.61 6.18 + Think before Edit 7.34 7.64 6.82 + Think while Edit 7.83 7.66 7.19 + Think before and while Edit 7.75 7.60 7.06

Model

Table 4. Ablation of turn number on GEdit-Bench-EN

GEdit-Bench-EN ↑ G SC G PQ G O FLUX.1-Kontext [Dev] 6.62 7.61 6.18 Trun 2 7.57 7.55 6.95 Trun 4 7.79 7.60 7.13 Trun 6 7.85 7.59 7.16 Trun 8 8.00 7.61 7.30

Model

timized prompt using only the source image, while Think while Edit denotes our proposed iterative reasoning-andediting framework. As shown in Table 3 of the main paper, Think before Edit provides a noticeable improvement but is consistently outperformed by Think while Edit. Furthermore, initializing Think while Edit with a Think before Edit step leads to a performance drop from 7.19 to 7.06. We hypothesize that the initial Think before Edit introduces a bias in the first-round reasoning, which results in incomplete information transfer and negatively impacts downstream performance.

Effectiveness of Thinking Rounds We first analyze the impact of the iterative refinement loop’s depth. As detailed in Table 4, the baseline model (equivalent to a single pass, or ”Trun 1”) achieves a G O score of 6.18. Introducing our Think While Edit framework with a maximum of two turns (Trun 2) immediately provides a substantial performance boost to 6.95 G O. We observe a clear and consistent performance scaling as we increase the maximum number of allowed turns. The G O score climbs to 7.13 at 4 turns, 7.16 at 6 turns, and reaches a peak of 7.30 at 8 turns. This strong positive correlation demonstrates that our framework effectively utilizes deeper, multi-step reasoning, allowing the model to iteratively correct errors and progressively enhance the editing outcome.

Analysis on Training Stage We then ablate the contributions of our EditThinker-8B model’s two-stage training process. Table 5 presents this breakdown. The SFT stage alone (+ EditThinker-8B-SFT) is responsible for a significant performance gain, lifting the G O score from 6.18 to 6.93 and

- Table 5. Ablation on GEdit-Bench-EN and ImgEdit-Bench with Training Stage with Think While Edit

Model

GEdit-Bench-EN ↑ ImgEdit-Bench ↑ G SC G PQ G O Overall ↑

FLUX.1-Kontext [Dev] 6.62 7.61 6.18 3.44 + Qwen-VL3-8B 6.70 7.60 6.23 3.42 + EditThinker-8B-SFT 7.55 7.54 6.93 3.57 + EditThinker-8B-RL 7.59 7.63 7.02 3.95

- Table 6. Ablation on GEdit-Bench-EN with Expert Model in Think-While-Edit pipeline.

GEdit-Bench-EN ↑ G SC G PQ G O FLUX.1-Kontext [Dev] 6.62 7.61 6.18 + GPT 4.1 7.83 7.66 7.19 + Gemini 2.5 Pro 7.76 7.65 7.11 + Doubao 1.5 7.36 7.59 6.80 + GPT-4o 7.65 7.67 7.05

Model

the ImgEdit-Bench Overall score from 3.44 to 3.57. Subsequently, applying the Reinforcement Learning (RL) stage (+ EditThinker-8B-RL) provides an additional and crucial optimization. While it offers a modest gain on GEdit-Bench (7.02 G O), its impact is most pronounced on the ImgEditBench benchmark, where it elevates the Overall score from 3.57 (SFT) to 3.95 (RL). This demonstrates that SFT is vital for imparting the foundational refinement capabilities, while RL is highly effective in optimizing the expert’s judgment and fine-tuning its decision-making policy.

Ablation of Different EditThinker Expert Finally, we investigate the scalability of our framework by ”plugging in” different expert models, replacing our trained EditThinker8B. The results in Table 6 are striking. The baseline FLUX model scores 6.00 G O in this setup. When we simply substitute the expert with a powerful, off-the-shelf proprietary model like GPT 4.1, the G O score leaps to 7.19. This result confirms two key insights: 1) Our Think While Edit framework is a general and highly scalable paradigm, not limited to our specific trained expert. 2) The framework’s performance is directly and positively correlated with the underlying reasoning and critical capabilities of the expert model employed.

#### 6. Conclusion

We propose a deliberative editing framework EditThinker that enables image editing models to “think while they edit”, addressing the limited instruction-following capability caused by inherent stochasticity and lack of deliberation in existing single-turn approaches. Our framework simu-

lates the human cognitive process by iteratively executing a Think-while-Edit cycle: Critiquing results, Refining instructions, and Repeating generation until satisfactory outcomes are achieved. Specifically, EditThinker is a single MLLM trained to jointly produce critique scores, reasoning processes, and refined instructions. We employ reinforcement learning to align EditThinker’s reasoning with actual editing outcomes, enabling more targeted instruction improvements. Extensive experiments on four benchmarks demonstrate that our approach significantly enhances the instruction-following capability of any image editing model by a large margin. We release our data construction framework, datasets, and models to benefit the research community.

#### References

- [1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 6
- [2] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, 2025. 2, 3, 6
- [3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 3, 7
- [4] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Fortyfirst International Conference on Machine Learning, 2024. 3
- [5] Anne de Jong, Sacha AFT van Hijum, Jetta JE Bijlsma, Jan Kok, and Oscar P Kuipers. Bagel: a web-based bacteriocin genome mining tool. Nucleic acids research, 2006. 7
- [6] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776,

2025. 2

- [7] Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, et al. Onethinker: All-inone reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025. 2
- [8] Luciano Floridi and Massimo Chiriatti. Gpt-3: Its nature, scope, limits, and consequences. Minds and machines, 2020. 2
- [9] Yuan Gong, Xionghui Wang, Jie Wu, Shiyin Wang, Yitong Wang, and Xinglong Wu. Onereward: Unified mask-guided image generation via multi-task human preference learning. arXiv preprint arXiv:2508.21066, 2025. 3
- [10] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt im-

- age editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3
- [11] Houcheng Jiang, Junfeng Fang, Ningyu Zhang, Guojun Ma, Mingyang Wan, Xiang Wang, Xiangnan He, and Tat-seng Chua. Anyedit: Edit any knowledge encoded in language models. arXiv preprint arXiv:2502.05628, 2025. 7
- [12] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

2025. 3

- [13] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 7

- [14] Hongyu Li, Songhao Han, Yue Liao, Junfeng Luo, Jialin Gao, Shuicheng Yan, and Si Liu. Reinforcement learning tuning for videollms: Reward design and data efficiency. arXiv preprint arXiv:2506.01908, 2025. 2
- [15] Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, and Li Yuan. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888, 2025. 2, 3, 7
- [16] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3
- [17] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 3, 7
- [18] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 6
- [19] Zheyuan Liu, Munan Ning, Qihui Zhang, Shuo Yang, Zhongrui Wang, Yiwei Yang, Xianzhe Xu, Yibing Song, Weihua Chen, Fan Wang, et al. Cot-lized diffusion: Let’s reinforce t2i generation step-by-step. arXiv preprint arXiv:2507.04451, 2025. 3
- [20] Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Liu, et al. Editscore: Unlocking online rl for image editing via high-fidelity reward modeling. arXiv preprint arXiv:2509.23909, 2025. 2, 3
- [21] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 3
- [22] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of

- the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023. 3
- [23] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025. 3
- [24] OpenAI. Image generation API. https://openai. com/index/image-generation-api/, 2025. 3
- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3
- [26] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3
- [27] Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, et al. Ovis-u1 technical report. arXiv preprint arXiv:2506.23044, 2025. 7
- [28] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint

- arXiv:2508.02324, 2025. 2, 6

[29] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint

- arXiv:2508.02324, 2025. 3

- [30] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 2, 3, 6, 7
- [31] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, et al. Q-bench: A benchmark for general-purpose foundation models on low-level vision. In ICLR, 2024. 3
- [32] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965,

2025. 2

- [33] Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, and Wenhu Chen. Editreward: A human-aligned reward model for instruction-guided image editing. arXiv preprint arXiv:2509.26346, 2025. 3
- [34] Yongliang Wu, Zonghui Li, Xinting Hu, Xinyu Ye, Xianfang Zeng, Gang Yu, Wenbo Zhu, Bernt Schiele, MingHsuan Yang, and Xu Yang. Kris-bench: Benchmarking next-level intelligent image editing models. arXiv preprint arXiv:2505.16707, 2025. 6
- [35] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In CVPR, 2025. 7
- [36] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Wang, Weiyun Ye, Shihao Geng, Yiren Zhao, Jiaming Li, Cunjian Li, Hang

- Sun, et al. Imagereward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, 2023. 3
- [37] Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782, 2025. 3
- [38] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning, 2024. 3
- [39] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [40] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 6
- [41] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 3
- [42] Qihui Zhang, Munan Ning, Zheyuan Liu, Yue Huang, Shuo Yang, Yanbo Wang, Jiayi Ye, Xiao Chen, Yibing Song, and Li Yuan. Upme: An unsupervised peer review framework for multimodal large language model evaluation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9165–9174, 2025. 3
- [43] Xiaoying Zhang, Hao Sun, Yipeng Zhang, Kaituo Feng, Chaochao Lu, Chao Yang, and Helen Meng. Critique-grpo: Advancing llm reasoning with natural language and numerical feedback. arXiv preprint arXiv:2506.03106, 2025. 2
- [44] Yinan Zhang, Eric Tzeng, Yilun Du, and Dmitry Kislyuk. Large-scale reinforcement learning for diffusion models. In European Conference on Computer Vision, pages 1–17. Springer, 2024. 3
- [45] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with incontext generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690, 2025. 3, 7
- [46] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024. 7
- [47] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025. 6

[Figure 156]

## EditThinker: Unlocking Iterative Reasoning for Any Image Editor

### Supplementary Material

#### A. Kris-Bench Result

To further evaluate reasoning-centric editing capability, we additionally report results on Kris-Bench. As shown in Table 7, our method demonstrates strong performance on reasoning-driven edits. We observe consistent performance gains. The Overall Score for FLUX.1 Kontext [Dev] is lifted from 61.81 to 69.53, OmniGen2 from 50.52 to 53.09, and Qwen from 64.43 to 71.91. This further demonstrates the performance improvements achieved by our method on the Reasoning Editing task.

#### B. More Ablation Analysis

Multi-round Reasoning for EditThinker. The main paper reports GPT-4.1’s multi-round reasoning performance as an approximate theoretical upper bound for the Think while Edit paradigm. Here, we further evaluate the multiround behavior of EditThinker-8B, as presented in Table 8. We observed a continuous performance improvement from the baseline to Turn 8, rising from 6.18 to 7.03. The largest performance boost was observed at Turn 2, where the score jumped from 6.18 to 6.90. This is often because the initial prompt performs the worst, so the first refinement brings the most direct improvement. In contrast, the stages after Turn 2 typically involve further reflection on the previously rewritten prompts.

#### C. Additional Implementation Details C.1. Details of EditThinker Expert

To supervise EditThinker with high-quality reasoning traces and refined editing instructions, we employ GPT-4 as the expert model. At the t-th editing iteration, we provide the tuple (Isrc, Ieditt−1, Ts, Tt−1) as input to the expert. The model then generates a reasoning trace Rt, a refined editing instruction Tt, and a <stop> flag indicating whether the current edit successfully satisfies the user’s intent. The maximum number of iterations for this think-while-edit process is set to N = 5.

The expert prompt is meticulously designed to explicitly encourage a multi-step Critique–Revise cycle. It requires the model to: (1) evaluate whether the edited image fulfills the original instruction Ts, (2) identify failure causes through detailed reasoning, and (3) synthesize an improved instruction that corrects these errors without introducing new inconsistencies. The full prompt template used for the expert is provided in Figure 6.

###### C.2. Details of EditThinker

EditThinker adopts a unified prompt format for both training and inference. This design ensures that the behavior learned during supervision aligns seamlessly with the capabilities required at inference time, enabling the model to (1) evaluate the current result, (2) reason about potential issues, and (3) refine the instruction for the next round.

At each iteration t, EditThinker receives a multimodal tuple (Isrc,Ieditt−1,Ts,Tt−1) that provides the complete context of the editing state. Here, Isrc and Ts represent the original source image and user instruction; Ieditt−1 denotes the intermediate result from the previous turn; and Tt−1 is the specific instruction that produced it. The maximum number of iterations for EditThinker is set to N = 5.

Based on this input, EditThinker outputs three components: a scalar instruction-following score St, a naturallanguage reasoning trace Rt, and a refined editing instruction Tt. Diverging from prior systems that rely on a binary <stop> flag, we implement a continuous scoring scheme comprising a Semantic Score and a Quality Score. This offers two key advantages: (1) it provides a smoother, more informative supervision signal for learning nuanced failure patterns; and (2) it enables precise control over inference quality, allowing users to trigger refinement only when the predicted score falls below a specific threshold. The full prompt template used for the expert is provided in Figure 7.

#### D. Details of ThinkEdit-140K Dataset

We obtain Ts and Isrc from three data sources: OpenGPT4o-Image, ShareGPT-4o-Image, and Pico-Banana-400K. From these sources, we sample 40K, 40K, and 60K editing instances respectively, ensuring that the editing categories are as evenly distributed as possible, resulting in a total of 140K raw samples. We divide these samples into three splits and use GPT-4.1 as the EditThinker-Expert, while selecting OmniGen2, FLUX.1 Kontext [Dev], and QwenImage-Edit as the editors. After trajectory filtering, we retain 70K valid trajectories. Among them, 10K trajectories are selected for RL, while the remaining 60K undergo stepwise refinement and filtering, ultimately producing 140K high-quality samples for SFT. Additionally, a subset of 27K filtered trajectories is used for RL training.

#### E. Visualization

We provide visualizations of the outputs generated by our framework across different settings. As shown in Figure 4,

- Table 7. Comparison of model performance on Kris-Bench. ‡ indicates results from our own tests with official model checkpoint. 0.0∗ indicates that the model was not evaluated on multi-image editing. Since our method currently does not support multi-image inputs, we excluded the Temporal subset of Factual Knowledge to ensure a fair comparison.

Model

Factual Knowledge Conceptual Knowledge Procedural Knowledge

Overall Score Attribute Spatial Temporal Average Social Sci. Natural Sci. Average Logical Instruction Average

Proprietary Models

Doubao 70.92 59.17 40.58 63.30 65.50 61.19 62.23 47.75 60.58 54.17 60.70 Step 3o vision 69.67 61.08 63.25 66.70 66.88 60.88 62.32 49.06 54.92 51.99 61.43 Gemini 2.0 66.33 63.33 63.92 65.26 68.19 56.94 59.65 54.13 71.67 62.90 62.41 GPT-4o 83.17 79.08 68.25 79.80 85.50 80.06 81.37 71.56 85.08 78.32 80.09

Open-source Models

InstructPix2Pix 30.33 21.33 0.00∗ 23.33 22.56 26.56 25.59 19.81 14.75 17.28 22.82 OmniGen 37.92 28.25 21.83 33.11 30.63 27.19 28.02 11.94 35.83 23.89 28.85 MagicBrush 53.92 39.58 0.00∗ 41.84 42.94 38.06 39.24 30.00∗ 23.08 26.54 37.15 AnyEdit 47.67 45.17 0.00∗ 39.26 38.56 42.94 41.88 36.56 26.92 31.74 38.55 Emu2 51.50 48.83 22.17 45.40 34.69 38.44 37.54 24.81 45.00 34.91 39.70 Step1X-Edit 55.50 51.75 0.00∗ 45.52 44.69 49.06 48.01 40.88 22.75 31.82 43.29 HiDream-E1 52.75 49.42 0.00∗ 43.31 52.56 49.25 50.05 45.19 30.08 37.64 44.72 ByteMorph 61.17 62.00 0.00∗ 51.27 45.50 47.38 46.92 32.00 31.33 31.67 44.85 FLUX.1 Kontext [Dev] 64.83 60.92 0.00∗ 53.28 48.94 50.81 50.36 46.06 39.00 42.53 49.54 OmniGen2 59.92 52.25 54.75 57.36 47.56 43.12 44.20 32.50 63.08 47.79 49.71 UniWorld-V1 58.17 54.50 63.00 47.71 47.50 43.94 44.80 42.00 53.83 47.92 50.27 Step1X-Edit v1.1 64.17 61.75 0.00∗ 53.05 52.06 55.06 54.34 52.56 36.75 44.66 51.59 BAGEL 64.27 62.42 42.45 60.26 55.40 56.01 55.86 52.54 50.56 51.69 56.21 BAGEL-Think 67.42 68.33 58.67 66.18 63.55 61.40 61.92 48.12 50.22 49.02 60.18 Uni-CoT 72.76 72.87 67.10 71.85 70.81 66.00 67.16 53.43 73.93 63.68 68.00

Think-while-Edit

OmniGen2‡ 60.21 54.67 0.00∗ 58.73 53.60 46.76 48.42 37.67 56.67 45.84 50.52 + EditThinker-8B 62.18 53.92 0.00∗ 59.98 61.50 49.90 52.71 37.04 58.78 46.43 53.09 + EditThinker-Expert-GPT4.1 65.55 56.83 0.00∗ 63.22 63.35 55.26 57.22 44.38 60.44 51.26 57.34

FLUX.1 Kontext [Dev]‡ 71.12 67.25 0.00∗ 70.09 56.60 58.91 58.35 56.75 63.72 59.75 61.81 + EditThinker-8B 77.82 65.50 0.00∗ 73.73 73.44 69.04 70.09 62.29 65.33 63.60 69.53 + EditThinker-Expert-GPT4.1 81.03 74.67 0.00∗ 79.33 77.60 74.98 75.62 71.38 65.50 68.86 74.93

Qwen-Image-Edit ‡ 72.73 73.33 0.00∗ 72.89 63.50 60.40 61.15 57.47 67.97 61.70 64.43 + EditThinker-8B 78.48 73.83 0.00∗ 77.24 76.20 70.69 72.02 65.23 66.89 65.94 71.91 + EditThinker-Expert-GPT4.1 83.70 76.08 0.00∗ 81.67 81.99 80.53 80.91 71.94 76.07 73.40 79.34

- Table 8. Ablation of turn number on GEdit-Bench-EN for EditThinker-8B

GEdit-Bench-EN ↑ G SC G PQ G O FLUX.1 Kontext [Dev] 6.62 7.61 6.18 Turn 2 7.44 7.60 6.90 Turn 4 7.51 7.60 6.94 Turn 6 7.57 7.62 7.01 Turn 8 7.61 7.58 7.03

Model

EditThinker produces high-quality edits when combined with various editors, including FLUX.1 Kontext [Dev], OmniGen2, and Qwen-Image-Edit. In addition, Figure 5 visualizes EditThinker’s iterative reasoning dynamics with FLUX.1 Kontext [Dev], highlighting how the Thinker critiques intermediate results and progressively refines the instruction over multiple rounds.

Source Turn 1 Turn 3

Turn 2

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Image

- (a)

[Figure 161]

- (b)

Turn 4 Turn 5

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Image

Prompt Animate the cat in the image.

[Figure 166]

Transform the cat in the image into an animated, cartoon-style character while maintaining its pose and fur pattern. Add subtle motion effects, such as motion lines or a playful background, to convey a sense of animation, but keep the rest of the scene consistent with the original.

Transform the cat in the image into an animated, cartoon-style character while strictly maintaining its original lying-on-back pose and curled paw position. Preserve the cat's unique fur pattern and ensure the background remains consistent with the original bedding, adding subtle motion effects (such as motion lines or playful animated elements) to convey a sense of animation.

Transform the cat in the image into an animated, cartoon-style character, strictly maintaining its original lying-on-back pose and curled paw position. Accurately replicate the cat's unique fur pattern and facial markings from the original image. Ensure the background remains consistent with the original dark, textured bedding, not a simplified or color-changed version. Add playful animated motion effects, such as dynamic motion lines or cartoon sparkles, to convey a lively sense of animation.

Transform the cat in the image into an animated, cartoon-style character while strictly preserving its original lying-on-back pose and curled paw position. Accurately replicate the cat's unique blackand-white fur pattern and facial markings as seen in the original image. Ensure the background remains consistent with the original dark, textured bedding, maintaining its folds and shading rather than simplifying or replacing it. Add playful animated motion effects, such as dynamic motion lines or cartoon sparkles, to convey a lively sense of animation, but do not alter the cat's pose or background details.

[Figure 167]

- (c)

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Image

Draw what it will look like if it Prompt were crumpled into a ball.

Transform the original sheet of paper into a realistic crumpled ball, preserving its white color and paper texture. Show the crumpled ball in the same context as the original image, such as clipped to the wall or placed on the same background, and avoid stylized or abstract representationsmake it look like an actual piece of paper that has been crumpled into a ball.

Transform the original sheet of paper into a realistic, naturally crumpled ball, preserving its white color and authentic paper texture. Ensure the crumpled ball displays irregular, organic folds and creases typical of a hand-crumpled piece of paper, and avoid any geometric, faceted, or stylized abstraction. Show the crumpled ball in the same context as the original image, such as clipped to the wall or placed on the same background, maintaining the natural look and integration with the scene.

- (d)

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Image

Transform it into an oil painting Prompt style.

[Figure 176]

Transform the image into an oil painting style, strictly preserving the original winter atmosphere, including the snow-covered trees, muted color palette, and overcast sky. Maintain all key details and composition from the original photo, only applying painterly brushstroke texture and oil painting effects.

Transform the image into an oil painting style, strictly maintaining the original winter atmosphere: keep the sky overcast and pale, preserve the muted and cool color palette, and ensure the snow-covered trees and architectural details remain unchanged. Apply painterly brushstroke texture and oil painting effects only to the surface, without altering the lighting or color mood of the scene.

Transform the image into an oil painting style by applying painterly brushstroke textures and oil painting effects, but strictly preserve the original winter atmosphere: keep the sky overcast and pale with no added warm tones, maintain the muted and cool color palette throughout the scene, and ensure the snow-covered trees and architectural details remain unchanged in color and form. Do not introduce any warm or saturated colors; focus on replicating the original lighting and mood in the oil painting effect.

Transform the image into an oil painting style by applying painterly brushstroke textures and oil painting effects, but strictly preserve the original winter atmosphere: keep the sky overcast and pale with cool, muted tones only, avoid introducing any warm or yellow hues, and maintain the snow-covered trees and architectural details exactly as in the original image in terms of color and form. Replicate the original lighting and mood, ensuring the scene retains its cold, wintry appearance.

[Figure 177]

- (e)

[Figure 178]

[Figure 179]

[Figure 180]

Image

Prompt Make the cat in the image run.

Transform the cat in the image so it is clearly running across the couch, with its body stretched out, legs extended in a running motion, and a sense of dynamic movement. Add subtle motion blur to the cat and background to emphasize speed, while maintaining the cat's original appearance and the integrity of the scene.

- (f)

Replace the entire baby with a realistically sized puppy sitting in the same position on the chair, ensuring the puppy is naturally integrated into the scene. The puppy should be holding the piece of bread in its paw or mouth, and all background and chair details should be preserved. Remove all traces of the baby, including hands, arms, and clothing, and ensure the puppy's fur and posture look natural for the setting.

Replace the baby sitting in the chair with a realistically sized puppy in the same position, ensuring the puppy is naturally integrated into the scene, holding the piece of bread in its paw or mouth, and maintaining the original background and chair details.

Prompt Replace the baby with a puppy.

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

Image

Transform the zebra into a realistic concrete sculpture, ensuring its entire body displays authentic concrete texture, surface roughness, and appropriate lighting. Retain the bold black-and-white striped pattern as coloration or surface treatment on the concrete, making the stripes appear painted or etched onto the concrete surface. Preserve the zebra’s natural pose and position directly on the road, without adding any pedestal or base. The zebra should look as if it is sculpted from concrete and standing naturally on the road.

Change the zebra’s material to concrete, ensuring it retains its bold black-and-white striped pattern as distinct coloration or surface treatment on the concrete. Preserve the zebra’s natural pose and context on the road, without adding a pedestal or base. The zebra should appear as if sculpted from concrete, with realistic concrete texture and lighting, but still clearly display its iconic stripes.

Transform the zebra into a realistic concrete sculpture, ensuring its entire body displays authentic concrete texture, surface roughness, and appropriate lighting. Retain the bold black-and-white striped pattern as coloration or surface treatment on the concrete, making the stripes appear painted or etched onto the concrete surface. Preserve the zebra’s natural pose and position directly on the road, with its hooves touching the road surface—do not add any pedestal, base, or platform beneath the zebra.

Change the zebra’s material to concrete, but preserve its distinctive black-and-white striped pattern and overall zebra shape. The zebra should appear as if sculpted from concrete, with visible concrete texture and realistic lighting, while maintaining all original zebra features.

Change the zebra’s material to Prompt concrete.

- Figure 4. Qualitative visualizations of EditThinker paired with different editors. Subfigures (a) and (b) show results with FLUX.1 Kontext [Dev], (c) and (d) use OmniGen2, and (e) and (f) use Qwen-Image-Edit.

###### Source Turn 1 Turn 3 Turn 4 Turn 5

Turn 2

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

###### Image

Show this bread as it would appear after being left out for a long time: incorporate visual signs of aging such as spots of green and blue mold, dried and curled edges, a harder and slightly shrunken texture, and possible discoloration. Preserve the bread's recognizable shape and position from the original image, but make it clear that it is no longer fresh.

Transform the bread to show clear signs of prolonged aging: add multiple patches of green and blue mold throughout several slices, make the edges visibly dried, curled, and slightly hardened, subtly shrink the loaf to indicate moisture loss, and add areas of yellowish and grayish discoloration to the surface. Make sure the bread retains its recognizable shape and arrangement from the original image, but ensure it clearly looks stale, old, and unappetizing.

Transform the bread to convincingly depict prolonged aging: add irregular, textured patches of green and blue mold with realistic gradients on both surface and interior, introduce subtle edges that are visibly dried, curled, and slightly hardened, shrink the loaf slightly to indicate moisture loss, and apply scattered areas of yellowish and grayish discoloration for varied staleness. Ensure the mold looks naturally integrated (not flat or cartoonish), and maintain the bread's overall recognizable shape and arrangement from the original image.

Transform the bread to convincingly depict prolonged aging: add irregular, realistically textured patches of green, blue, yellowish, and grayish mold to both the crust and exposed interior, using gradients and soft transitions for natural blending. Vary the density and color of mold patches for authentic diversity. Emphasize dried, curled, and slightly hardened edges on both whole and sliced pieces, and shrink the loaf perceptibly to indicate moisture loss. Ensure the mold penetrates unevenly into the exposed areas, not just the loaf's exterior. Preserve the recognizable shape and arrangement of the original bread, but make the overall appearance clearly aged, stale, and spoiled.

Prompt long time

This bread after being left for a

The edited image does not visibly reflect the changes one would expect for 'this bread after being left for a long time.' There are no signs of aging such as mold, dryness, staleness, discoloration, or hardening. The bread in the edited image looks nearly identical to the original, retaining a fresh and soft appearance. The rewritten prompt failed to specify or encourage visual cues that indicate old or spoiled bread, resulting in little to no actual transformation.

The edited image does introduce a few spots of mold on the bread, partially addressing the visual signs of aging. However, other critical features requested in the rewritten prompt are missing or insufficient: the bread's edges do not appear dried, curled, or shrunken, and the overall texture still looks soft and fresh rather than hard. Discoloration is limited to small areas; most of the bread maintains its original color, failing to convey a convincing sense of staleness or prolonged exposure. The bread's shape and position are correctly preserved, but the overall effect does not clearly communicate that the bread has been left out for a long time.

Reasoning

The edited image applies green and blue patches that resemble mold, which partially fulfills the intent of showing prolonged aging. However, several key quality aspects are missing or incorrect:

The edited image adds a widespread green mold-like coloration to the crust and edges of the bread, visually suggesting aging and spoilage. However, several key aspects of the rewritten prompt were not adequately fulfilled: (1) The mold effect is limited mostly to uniform green coloring and does not convincingly include irregular textured patches, gradients, or the requested blue, yellowish, and grayish hues—the appearance is flat and lacks natural diversity. (2) The mold is applied in a way that appears digitally overlaid rather than truly integrated into the bread's surface and interior as requested. (3) There is insufficient evidence of dried, curled, or hardened edges, nor is the loaf noticeably shrunken to reflect moisture loss; the interior remains pristine. (4) The overall bread shape is preserved, but staleness and aging features (discoloration, textural changes, edge curling) are underdeveloped. The edit falls short on detail, realism, and color variation, making the bread appear more like dyed bread than authentically aged and moldy.

- (1) The mold coloration appears as large, unrealistically uniform patches, lacking the texture and irregularity typical of real mold.
- (2) There is no visible drying, curling, or hardening of the bread edges—the slices retain their fresh, soft appearance and straight edges. (3) No signs of shrinkage or moisture loss are present; the loaf maintains its original fullness. (4) There is an absence of yellowish and grayish discoloration, which was explicitly requested to show varied stages of staleness. (5) The overall effect looks artificial and lacks subtlety, making the bread appear unnaturally colored rather than convincingly old and unappetizing.

###### Source Turn 1 Turn 3 Turn 4 Turn 5

Turn 2

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

###### Image

Replace the zebra with a realistic giraffe, ensuring the giraffe has accurate proportions (long neck and legs) and a classic giraffe coat pattern with large, irregular brown patches separated by light tan lines. Remove all zebra stripes and features, and integrate the giraffe naturally into the scene with matching lighting and shadows.

Replace all zebras in the scene with realistic giraffes, ensuring each giraffe has accurate proportions including a long neck, long legs, and a classic giraffe coat pattern with large, irregular brown patches separated by light tan lines. Remove all zebra stripes and features, and integrate the giraffes naturally into the environment with matching lighting and shadows.

Replace all zebras in the scene with realistic giraffes, ensuring each giraffe has accurate proportions including a long neck, long legs, and a classic giraffe coat pattern: large, irregular brown patches separated by light tan lines. Remove all zebra stripes and features completely, and integrate the giraffes naturally into the environment with matching lighting and shadows.

Replace the zebra with a realistic giraffe, ensuring the giraffe has accurate proportions, long legs, and a spotted coat pattern typical of giraffes. Remove all zebra features, and integrate the giraffe naturally into the scene with appropriate lighting and shadows.

Prompt

Replace the zebra with a giraffe.

The edited image does not successfully fulfill the original instruction to 'replace the zebra with a giraffe.' While one zebra has been removed and the remaining animal's neck has been elongated, the body, legs, and coloration still clearly resemble a zebra rather than a giraffe. The animal retains the zebra's stripes and proportions, rather than the giraffe's distinctive spotted pattern, long legs, and overall body shape. The edit is essentially a zebra with a giraffelike neck, not a true replacement with a giraffe. Therefore, the core intent of the instruction is not met.

The edited image replaced the zebra with an animal that has the body shape and proportions of a giraffe (long neck and legs), but the coat pattern is incorrect. The animal retains zebra-like stripes rather than the characteristic spotted or patchwork pattern of a giraffe. The prompt specified a 'spotted coat pattern typical of giraffes,' which was not achieved. The integration and lighting are natural, but the core intent—to fully replace the zebra with a realistic giraffe—was not met due to the incorrect coat pattern.

The edited image does not fully satisfy the original instruction. While the zebra's stripes have been recolored to brown and tan, the animal's body shape, proportions, and facial features remain distinctly zebra-like. The neck and legs are not elongated as expected for a giraffe, and the coat pattern is still striped rather than the classic giraffe pattern of large, irregular patches. Additionally, only one zebra was replaced, leaving the second zebra untouched. The rewritten prompt was clear, but the execution failed to deliver a realistic giraffe replacement.

The edited image replaced the zebras with animals that have giraffe-like body shapes (long necks and legs), but the coat pattern is incorrect. The new animals retain zebra-like stripes rather than the classic giraffe pattern of large, irregular brown patches separated by light tan lines. This fails to fully meet the original instruction and the rewritten prompt, which specifically requested removal of zebra stripes and accurate giraffe coat patterns. The integration and proportions are good, but the most visually distinctive feature of a giraffe—the coat pattern—is missing.

Reasoning

###### Figure 5. Visualization of EditThinker’s reasoning traces and intermediate editing results when paired with FLUX.1 Kontext [Dev]. The figure illustrates how the Thinker evaluates the current output, identifies issues, and iteratively refines the instruction over multiple rounds.

You are an expert image editing evaluator and prompt engineer. Your task is to:

- 1. Evaluate whether an edited image successfully fulfills the original user instruction.
- 2. If not satisfied, generate an improved rewritten prompt that addresses the shortcomings.

###### Input Information. You will receive:

- • Original Image: the input image before editing.
- • Original User Instruction: the user’s initial editing request.
- • Rewritten Prompt: the refined instruction that was used for editing.
- • Edited Image: the resulting image after applying the rewritten prompt.

###### Evaluation Criteria.

- A. Intent Alignment

- • Does the edited image achieve the core goal of the user instruction?
- • Are all requested changes present and correctly implemented?

- B. Quality Assessment

- • Subject/Object Changes: correctness of additions/removals/replacements.
- • Appearance Modifications: accuracy of color/style/material edits.
- • Scene Changes: correctness of background/environment edits.
- • Detail Preservation: important details remain intact.
- • Visual Coherence: edited image looks natural and well-integrated.

- C. Common Failure Patterns

- • Missing requested elements.
- • Incorrect positioning or scale.
- • Wrong colors or materials.
- • Unnatural blending or artifacts.
- • Lost subject details.
- • Style inconsistency.
- • Text errors (if applicable).
- • Over-editing or under-editing.

Evaluation Decision. SATISFIED: the edited image fulfills the original instruction with acceptable quality. Minor imperfections are acceptable if the core intent is achieved. NOT SATISFIED: the edit fails in key aspects. Major elements missing, incorrect, or severe quality issues → refinement required. Prompt Refinement Strategy (If Not Satisfied).

- 1. Identify what went wrong.

- • Compare original instruction → rewritten prompt → edited result.
- • Identify mismatches between intent and execution.
- • Determine whether the issue is clarity, specificity, or contradiction.

- 2. Refinement approaches.

- • If vague: add specific descriptors, spatial relations, or context.
- • If contradictory: resolve conflicts and simplify.
- • If important details were lost: explicitly require preservation.
- • If scale/position wrong: add precise location and size cues.
- • If style incorrect: specify textures, lighting, materials.
- • If over/under-edited: specify degree of modification.

- 3. Leverage all information.

- • Reference visible content in the original and edited images.
- • Retain what worked; correct what failed.

###### Output Format.

{ ’’is satisfied’’: true/false, ’’reason’’: ’’Detailed explanation of evaluation. If satisfied, explain why it meets requirements. If not satisfied, describe specific shortcomings.’’, ’’new rewritten prompt’’: ’’Only include if is satisfied is false. If satisfied, set to null.’’ }

Examples.

- Example 1: Satisfied

{ ’’is satisfied’’: true, ’’reason’’: ’’The edited image successfully adds a cat...’’, ’’new rewritten prompt’’: null }

- Example 2: Not Satisfied — Lack of Specificity

{ ’’is satisfied’’: false, ’’reason’’: ’’The rewritten prompt was too vague...’’, ’’new rewritten prompt’’: ’’Change the car color to blue...’’ }

Input Data:

- • Original User Instruction: Ts
- • Rewritten Prompt Used: Tt−1
- • Images Order: [Original Image, Edited Image]

Images: Isrc, Ieditt−1

- Figure 6. EditThinker Expert Prompt. The full expert instruction used for EditThinker Expert. At each iteration, the Expert observes (Isrc, Ieditt−1, Ts, Tt−1) and produces stop flag, reasoning, and a refined instruction.

Images: Isrc, Ieditt−1 Edit Evaluation and Prompt Refinement System. You are an expert image editing evaluator and prompt engineer. Your task is to:

- 1. Score the edited image from two perspectives and output the result in JSON format.
- 2. If you think the edited image is not good enough, generate an optimized rewritten prompt that addresses the original shortcomings; if you think it is good enough, output the original rewritten prompt.

Input Information. You are shown two images in sequence:

- • Original Image: the input image before editing.
- • Edited Image: the latest edited image generated using the previous instruction.

The textual instructions involved in this process are:

- • Original User Instruction : “Ts”
- • Previous Rewritten Instruction : “Tt−1” Evaluation Criteria (Score 0–10).

- A. Semantic Score (Instruction Following and Preservation).

- • Evaluates how accurately the edit was performed. The edit fails if it either (A) fails to follow the text instruction, or (B) over-edits the image by changing content that was not supposed to be changed.
- • (10) Follows the instruction perfectly and preserves all unchanged content.
- • (0) Fails to follow the instruction or needlessly changes the original scene.

- B. Quality Score (Naturalness & Artifacts).

- • Evaluates the technical quality of the newly edited image.
- • (10) Looks natural, has no artifacts, and integrates seamlessly.
- • (0) Looks unnatural (wrong shadow, lighting, or sense of distance) or contains severe artifacts (distortion, blurred faces, unusual body parts, disharmony).

Prompt Refinement Strategy (if Not Good Enough). When generating a new rewritten prompt, follow these steps:

- 1. What went wrong?

- • Compare original instruction → rewritten prompt → edited result.
- • Identify gaps between intent and execution.
- • Determine if the issue is clarity, specificity, or contradiction.

- 2. Refinement approaches.

- • If the rewritten prompt was too vague: add more specific descriptors (exact colors, positions, sizes), include spatial relationships and context, and specify interaction with existing elements.
- • If the rewritten prompt was contradictory: resolve conflicts between requirements, prioritize core intent over secondary details, and simplify complex multi-part instructions.
- • If important details were lost: explicitly state preservation requirements, add “maintain [aspect]” or “preserve [feature]” clauses, and reference specific elements from the original image.
- • If positioning/scale was wrong: use more precise spatial descriptors, add relative size/scale indicators, and specify foreground/midground/background placement.
- • If style/appearance was incorrect: use more specific visual vocabulary, add reference to the original image’s style elements, and include material/texture/lighting specifications.
- • If the edit was over/under-processed: add modifiers such as “subtle”, “gentle”, “dramatic”, or “significant”, specify the degree of change more clearly, and balance enhancement with naturalness.

- 3. Leverage all information.

- • Reference what is visible in the original image.
- • Learn from what the previous rewritten prompt missed.
- • Use the edited image as feedback on what went wrong.
- • Maintain what worked, and only modify what needs to be improved.

Output Format. The output consists of three parts:

- 1. A statement: analysis process and reasoning.
- 2. A JSON object: scores in different dimensions.
- 3. A prompt: either the optimized rewritten prompt or the original rewritten prompt. An example output is shown below:

<think> Detailed explanation of evaluation and new rewritten prompt. If the edited image is good enough, explain why it meets the requirements. If it is not good enough, explain the specific shortcomings. </think> <score> { ’’semantic’’: [0--10], ’’quality’’: [0--10] } </score> <answer> ’’Improved rewritten prompt that addresses the identified issues and enhances clarity, specificity, and preservation requirements’’ (if NOT GOOD ENOUGH) ’’Original rewritten prompt’’ (if GOOD ENOUGH) </answer>

- Figure 7. EditThinker prompt. The unified prompt template used for EditThinker’s edit evaluation and instruction refinement. At each iteration, the Thinker observes (Isrc, Ieditt−1, Ts, Tt−1) and produces semantic and quality scores, reasoning, and a refined instruction.

