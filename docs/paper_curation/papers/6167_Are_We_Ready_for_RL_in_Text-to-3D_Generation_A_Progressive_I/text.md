# arXiv:2512.10949v1[cs.CV]11Dec2025

## Are We Ready for RL in Text-to-3D Generation? A Progressive Investigation

Yiwen Tang1,4*, Zoey Guo3*, Kaixin Zhu2*, Ray Zhang3*, Qizhi Chen4, Dongzhi Jiang3, Junli Liu4, Bohan Zeng2, Haoming Song4, Delin Qu4, Tianyi Bai5, Dan Xu5, Wentao Zhang2, Bin Zhao1,4 1Northwestern Polytechnical University 2Peking University 3The Chinese University of Hong Kong 4Shanghai AI Lab 5The Hong Kong University of Science and Technology

##### Abstract

Reinforcement learning (RL), earlier proven to be effective in large language and multi-modal models, has been successfully extended to enhance 2D image generation recently. However, applying RL to 3D generation remains largely unexplored due to the higher spatial complexity of 3D objects, which require globally consistent geometry and fine-grained local textures. This makes 3D generation significantly sensitive to reward designs and RL algorithms. To address these challenges, we conduct the first systematic study of RL for text-to-3D autoregressive generation across several dimensions. (1) Reward designs: We evaluate reward dimensions and model choices, showing that alignment with human preference is crucial, and that general multi-modal models provide robust signal for 3D attributes. (2) RL algorithms: We study GRPO variants, highlighting the effectiveness of token-level optimization, and further investigate the scaling of training data and iterations. (3) Text-to-3D Benchmarks: Since existing benchmarks fail to measure implicit reasoning abilities in 3D generation models, we introduce MME-3DR. (4) Advanced RL paradigms: Motivated by the natural hierarchy of 3D generation, we propose Hi-GRPO, which optimizes the global-to-local hierarchical 3D generation through dedicated reward ensembles. Based on these insights, we develop AR3D-R1, the first RL-enhanced text-to-3D model, expert from coarse shape to texture refinement. We hope this study provides insights into RL-driven reasoning for 3D generation. Code is released at https://github.com/Ivan-Tang-3D/3DGenR1.

##### 1. Introduction

Large Language Models (LLMs) and Large Multi-modal Models (LMMs) have achieved strong results in tasks like text generation, image grounding [15], and video under-

standing [16], yet still struggle with complex reasoning tasks such as mathematical problem solving [41] and code generation [22]. Recently, driven by Chain-of-Thought (CoT) reasoning capabilities that emerge through reinforcement learning (RL), advanced models such as OpenAI o3 [18] and DeepSeek-R1 [9] achieve significant gains on these challenging tasks. As shown in Figure 1, RL training has expanded beyond understanding tasks to multimodal generation, particularly in autoregressive text-toimage models. The prior work Image Generation with CoT [10] demonstrated the effectiveness of Direct Preference Optimization (DPO) [20] in improving intermediate generation processes. More recently, several studies [13, 28] have explored the application of Group Relative Policy Optimization (GRPO) [23] to 2D generation. However, 3D autoregressive generation models [11, 25] have primarily focused on pre-training and fine-tuning approaches.

This raises the question: Can RL training be applied to text-to-3D generation, strengthening the step-by-step process of 3D autoregressive models? While RL has shown promise in text-to-image generation, these strategies cannot be directly applied to text-to-3D generation. 3D assets involve coupled geometric and textural properties operating in higher spatial dimensionality, making RL training more sensitive to reward designs and algorithmic choices. Moreover, 3D generation requires coherent joint optimization across multiple object components

Therefore, we systematically investigate the potential of the RL training for 3D autoregressive generation. Building upon the GRPO algorithm and the 3D discrete model ShapeLLM-Omni [39], and inspired by recent advances in 2D generation [13], we introduce a reasoning-guided framework where the model first generates textual reasoning that subsequently guides token-level 3D generation. We evaluate our approach on Toys4K [26]. As shown in the right part of Figure 1, our analysis focuses on following perspectives:

• Impact of Different Reward Models. In 3D gener-

### RL in Text Generation

[Figure 1]

Investigation Result

Question

Julie read 12 ×

There are 120 − 36 = 84 pages left to be read.

2 = 24 pages today.

RL Loss

Julie is reading a 120-page book. She read 12 pages yesterday and twice that today. If she plans to read half of what’s left tomorrow, how many pages will she read?

Answer Step IV

[Figure 2]

Step I Step II Step III

CLIP Score

[Figure 3]

Reasoning Process

42

Since she wants to read half of the remaining pages tomorrow, then she should

So she was able to read a total of 12 +

22.7

24 = 36 pages since yesterday.

24.0 25.2

read 842 = 42 pages.

26.5

RL in Image Generation

Textual Reasoning

29.3

[Figure 4]

[Figure 5]

[Figure 6]

Yellow cat in front, brown dog behind, close together outdoors in natural light...

Image Prompt

Reward Design

[Figure 7]

Kernel Distance

RL Loss

[Figure 8]

Step I Step II Step III

A yellow cat and a HPS

ImageReward ORM

Reasoning Process

[Figure 9]

0.249 0.241

brown dog.

[Figure 10]

[Figure 11]

[Figure 12]

RL Algorithm

[Figure 13]

0.228 0.210

GRPO DAPO GSPO

0.156

RL in 3D Generation

Textual Reasoning

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Small red toy train with cowcatcher, smokestack, and yellow accents...

Baseline

3D Prompt

Textual Reasoning

Reward Design

RL Loss

[Figure 19]

Step I Step II Step III

Toy train with red boiler and blue domes

Reward Design

HPS Qwen UnifiedReward

[Figure 20]

Reasoning Process

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

RL Algorithm RL Paradigm

RL Algorithm

[Figure 26]

GRPO DAPO GSPO

Figure 1. The Illustration of RL in Text, Image and 3D Generation Tasks. Left: In text generation, RL induces textual reasoning, whereas in 2D and 3D autoregressive generation, RL primarily improves token-level generation. Although recent 2D studies have explored reasoning-guided generation, diverse reward models and RL algorithms, such approaches remain examined in 3D generation. Right: We present the effects of different strategies on RL performance, using ShapeLLM-Omni as the baseline model. KD is reported ×100.

ation, reward models serve diverse roles, including modeling human preference distributions, measuring prompt alignment, and enforcing multi-view consistency. These rewards vary from task-specific trained models to promptdriven general LMMs. Understanding how different reward sources shape policy behavior is critical for RL training.

Observations. 1) Functions of Reward Models. Aligning with human preferences is crucial for 3D autoregressive generation. Additionally, enforcing consistency with text prompt and incorporating 3D aesthetic priors further enhance generation quality. 2) Forms of Reward Models. Compared to specialized reward models, exclusive reliance on general LMMs for task-specific evaluation introduces systematic bias. However, LMMs surprisingly demonstrate strong robustness for 3D-relevant attributes.

• Impact of Different RL Algorithms. Recent works have introduced GRPO variants [37, 40] to improve LLM reasoning. For instance, DAPO [40] enforces consistent token-level averaging in loss computation and promotes sequence diversity, whereas GSPO [37] clips sequence-level likelihood differences between new and old policies to align with sequence-level rewards. We therefore ask whether these improvements also benefit 3D generation.

Observations. 1) The RL loss of 3D autoregressive models benefits more from token-level averaging, as it better captures global structural differences during generation. In

contrast, sequence-level operations provide limited gains.

- 2) Simple techniques in DAPO, such as dynamic sampling, are sufficient to stabilize training for text-to-3D generation.
- 3) Data scaling effectively improves performance, whereas iteration scaling demands careful calibration.

• Exploration of Text-to-3D Benchmarks. Current text-to-3D benchmarks fail to evaluate models under reasoning-heavy conditions. While models perform well on simple prompts, we observe consistent failures across five categories: (1) Spatial & structural geometry, (2) Mechanical affordances, (3) Biological & organic shapes, (4) Worldknowledge rare objects, and (5) Stylized representation. As a result, existing benchmarks overestimate generation models and ignore their intrinsic reasoning abilities. To bridge this gap, we propose MME-3DR, the first benchmark designed for these reasoning-intensive 3D cases. It contains 249 annotated 3D objects spanning the five challenging categories. Experiments on several text-to-3D models validate the effectiveness of MME-3DR.

Observations. 1)Recent text-to-3D models demonstrate reasonable performance on biological objects and those with well-defined mechanical structures, yet they remain fragile across other categories. 2) After RL training, the model achieves substantial improvements across all five categories compared to the base model. 3) MME-3DR serves dual purposes: measuring generation quality and evaluat-

ing implicit reasoning capabilities.

• Exploration of RL Paradigms. During training, we observe that the model first constructs the global geometry and then progressively refines local textures in later stages, resembling human 3D perception process. This indicates that RL paradigms leveraging textual reasoning for direct 3D guidance can be further enhanced. To jointly optimize the hierarchical 3D generation within a single iteration, we introduce Hi-GRPO. With unified generation and understanding abilities of ShapeLLM-Omni, our method first prompts the model to plan the global structure and produce high-level semantic reasoning for token-level generation, yielding a coarse 3D shape that captures geometry but lacks texture details. In the second step, we feed both the firststep CoT and the original prompt into the model to obtain low-level visual reasoning and generate a texture-refined 3D object. In each iteration, we sequentially generate multiple coarse shapes and corresponding refined models for each prompt, and introduce two specialized ensembles of expert reward models to compute group-relative rewards for both steps. Building on these strategies, we develop AR3D-R1, the first RL—enhanced 3D autoregressive model.

Observations. 1) AR3D-R1 demonstrates a coarseto-fine progression during inference, evolving from rough shapes to detailed textures. This behavior aligns with our training procedure and validates the effectiveness of HiGRPO. 2) AR3D-R1 exhibits strong reasoning capability on MME-3DR and outperforms Trellis on benchmarks.

In summary, our core contributions are as follows:

- • We are the first to systematically introduce reinforcement learning into text-to-3D autoregressive generation, conducting an in-depth analysis from multiple perspectives.
- • By examining reward model design and RL algorithm selection, we show that both must be tailored to 3D domain knowledge and that appropriate reward models significantly enhance overall performance.
- • From the perspective of text-to-3D benchmark, we observe that existing benchmarks focus on object diversity while neglecting evaluation of model capability, and therefore introduce MME-3DR to assess the intrinsic reasoning abilities of 3D generation models.
- • From the RL paradigm perspective, we reveal an inherent hierarchy in 3D generation—from coherent geometry to texture-refined object, and propose Hi-GRPO, an advanced RL paradigm that jointly optimizes both steps in a single iteration. Building on these, we develop AR3D-R1, which outperforms current text-to-3D models.

##### 2. Related Work

###### 2.1. RL for LLM

Advanced LLMs such as OpenAI o3 [18] and DeepSeekR1 [9] have demonstrated strong reasoning abilities by

combining Chain-of-Thought (CoT) reasoning with reinforcement learning (RL). DeepSeek-R1 introduces rulebased rewards and GRPO [23], enabling models to conduct extensive internal reasoning before producing an answer, with rewards guiding correctness and format compliance. This paradigm has also been extended to multimodal LLMs [8, 12, 24, 45], where RL is adapted for visual understanding by jointly processing images and text for step-bystep reasoning. These RL-driven approaches have proven effective across mathematical problem-solving [23, 41] and code generation [22], establishing RL as a key technique for eliciting advanced capabilities in large-scale models.

###### 2.2. RL for 2D Generation

RL has also been effectively applied to text-to-image generation. Image-Generation-CoT [10] first frames progressive image token generation as a reasoning process and applies DPO [20] accordingly. T2I-R1 [13] extends this idea by distinguishing two levels of CoT—semantic-level planning and token-level patch generation—and introduces BiCoT-GRPO to jointly optimize both using an ensemble of vision experts as reward models. Recent work [28] comparing DPO and GRPO shows that GRPO offers better text–image alignment and aesthetic quality through group-relative policy updates. Together, these studies highlight that well-designed sequence-level rewards and multidimensional evaluation are essential for producing semantically consistent and visually appealing images in autoregressive models. For diffusion models, Dance-GRPO [36] introduces a stepwise, motion-aware reward that aligns policy updates with temporal dynamics, enabling more coherent and physically plausible generation. Flow-GRPO [17] extends GRPO to flow-matching models by coupling policy optimization with flow objectives, yielding smoother training and improved stability. These methods show that RL can effectively enhance controllability and consistency in diffusion and flow-based generative models.

###### 2.3. Text-to-3D Generation

Text-to-3D generation has progressed from two-stage pipelines [35, 38] to native diffusion models [5, 34] and, more recently, autoregressive approaches [2, 25, 39, 42]. Two-stage methods, like Dream3D [35], first generate a high-quality 3D shape prior from text using a text-toimage diffusion model and then refine it as a neural radiance field, but this pipeline suffers from error accumulation between stages and limited 3D consistency inherited from the 2D diffusion backbone. Native diffusion models, like Trellis [34], leverage structured 3D latent representations to directly generate high-fidelity 3D content, but their strong performance comes at the cost of significant computational demands. Autoregressive models alleviate these limitations by discretizing 3D content into token sequences.

Textual Prompt

HPS Unified HPS & LMM HPS & Unified HPS & Unified & LMM

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

"Acoustic guitar with dark fingerboard, six strings, and a circular soundhole."

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

"Stylized dolphin with smooth blue body, dorsal and pectoral fins, bifurcated tail, and subtle mouth line."

Figure 2. Visualization Results of the Impact of Different Reward Models.

Table 1. Quantitative comparisons using Toys4k for Different Reward Models. HPS refers to HPS v2.1, which outputs human preference rewards. Unified denotes UnifiedReward-2.0Qwen7B, which jointly evaluates aesthetic quality and prompt alignment. LMMalign employs Qwen2.5-VL-7B to replace UnifiedReward functionality. LMM3D utilizes Qwen2.5-VL to assess 3D consistency. (KD is reported ×100)

MeshGPT [25] uses decoder-only transformer to model triangle meshes as sequences, and MeshAnything [2] demonstrates scalable artist-grade mesh generation using autoregressive transformers. DeepMesh [42] provides an early attempt to incorporate DPO [20] into autoregressive 3D creation, and LLaMA-Mesh [30] represents 3D OBJ files as text to unify language and 3D representations. ShapeLLMOmni [39] proposes a unified multimodal LLM for 3D generation and understanding by discretizing 3D shapes into tokens with a 3D VQVAE. The model follows a text→voxel pipeline, where the LLM predicts discrete 3D latent tokens that are decoded by the VQVAE into voxel grids, which are then further converted into meshes using Rectified Flow model [34] for rendering. This design enables a single LLM to support text-to-3D generation, 3D understanding, and editing within one coherent framework. Despite these developments, RL training for 3D autoregressive models remains largely unexplored. In contrast to 2D generation, where RL has shown clear benefits, 3D generation introduces additional challenges—greater spatial complexity, stricter global geometry constraints, and fine-grained local details—making it more sensitive to reward design and optimization choices. These factors highlight the need for systematic RL strategies tailored to text-to-3D generation.

Reward Model Metrics HPS Unified LMMalign LMM3D CLIP Score↑ KDincep↓

- - - - - 22.7 0.249

✓ - - - 24.0 0.241

- - ✓ - - 23.5 0.246
- - - - ✓ 23.3 0.245

✓ ✓ - - 24.6 0.235 ✓ - ✓ - 24.2 0.238 ✓ ✓ - ✓ 25.2 0.228

can be found in the supplementary material.

RL Algorithm. GRPO [23] is an on-policy reinforcement learning algorithm that enhances PPO [21] by removing the value function and using group-wise reward comparisons. For a given prompt, G responses {oi}Gi=1 are sampled from the old policy πθ

. Each response receives a reward Ri from the reward model, and the advantage is computed by normalizing rewards within the group:

old

##### 3. Preliminary

Ri − mean({Ri}Gi=1) std({Ri}Gi=1)

Ai =

.

3D Autoregressive Generation. Compared with two-stage pipelines [35] and native 3D diffusion models [32, 38, 43], 3D autoregressive generation models [2, 25, 31, 42] directly discretize 3D objects into token sequences. This can be achieved by compressing and quantizing 3D shapes with VQVAE [3, 39], or by applying mesh tokenization techniques [4, 42] to discretize vertices and faces. Furthermore, LLaMA-Mesh [30] treats 3D OBJ files as plain text, utilizing natural language as the interface for mesh generation and understanding. Building further, ShapeLLM-Omni [39] integrates Qwen2.5-VL with a 3D VQVAE module to unify 3D generation and understanding, enabling autoregressive prediction over discrete 3D tokens and text. More details

GRPO uses PPO-like clipping and introduces a KL penalty term between the policy πθ and the reference πθ

.

ref

In text reasoning tasks, rewards are depend on the correctness of the final answer and the output format.

Experimental Settings. We adopt ShapeLLM-Omni as our baseline, a recent state-of-the-art 3D autoregressive model. We curate 8,400 short captions from 3D object datasets as training prompts [6, 7, 14]. To systematically evaluate different strategies, we randomly select 800 samples from Toys4K as our test set. Given ShapeLLM-Omni understanding-and-generation capability and the recent ad-

- Table 2. Quantitative comparisons using Toys4k for Different RL algorithms. In DAPO, Clip, Sampling, Token Avg., and KL Remov. correspond to Decoupled Clip, Dynamic Sampling, Token-level Loss Aggregation, and KL Penalty Removal, respectively. For GSPO, Seq. Opt. indicates that both importance sampling and clipping are performed at the sequence level. (KD is reported ×100)

DAPO GSPO Metrics Clip Sampling Token Avg. KL Remov. Seq. Opt. CLIP Score↑ KDincep↓

- - - - - - 25.2 0.228
- - ✓ - - - 25.8 0.219
- - ✓ - - ✓ 25.5 0.223
- - ✓ ✓ - - 26.3 0.214
- - ✓ ✓ ✓ - 25.9 0.213

✓ ✓ ✓ - - 26.5 0.210

1X 1.5X 3X

25.5

26.0

26.5

27.0

27.5

2X

Data Scaling

Iteration Scaling

Figure 3. Effects of Scaling Strategies. We examine the effects of data scaling and training iteration scaling strategies on clip score.

- Table 3. Effectiveness of textual reasoning. We employ HPS V2.1 as the reward and adopt GRPO.

Textual Reasoning CLIP Score

Base Model 22.7 W/O 23.4 W/ 24.0

vances in 2D generation [13], we do not directly generate 3D objects.

Instead, each training iteration begins by prompting the model to imagine the object and produce G textual descriptions, followed by generating one 3D object conditioned on each description, where G = 8. Table 3 shows that textual reasoning prior to 3D token-level generation yields greater RL potential than direct generation. In the following sections, we explore the applicability of GRPO to text-to-3D generation from four perspectives: Reward models (Sec. 4), RL algorithm choice (Sec. 5), Text-to-3D benchmarks (Sec. 6) and the advanced RL paradigm design (Sec. 7).

- 4. Impact of Different Reward Models

integrity; (2) Multi-view assessment: Evaluation must ensure cross-view consistency, verifying that shapes and textures across viewpoints form structurally valid objects. We therefore investigate reward model paradigms across different evaluation dimensions and their combination strategies.

- • Human Preference. Human preference models, such as the HPS series [33], are vision–language models trained on large-scale human-annotated image-ranking datasets. The model takes a prompt and multiple rendered views as input, assigns a score to each view, and uses the highest score to represent the overall 3D object visual quality.
- • Prompt Alignment & Aesthetic Quality. Specialized reward models, such as UnifiedReward [29], evaluate each rendered view of a 3D object with three scores: (1) Prompt—image alignment, (2) Image logical coherence, and (3) Image style appealing. These scores are summed, and the maximum score across views is taken as the final reward. The latter two terms capture aesthetic quality. In contrast, general LMMs, like Qwen2.5-VL, jointly process the prompt and all views to generate a single reasoning-based score, used for either alignment or aesthetics.
- • 3D Consistency. There is no reward model trained specifically for 3D consistency. However, we observe that advanced LMMs, such as Qwen2.5-VL [1], exhibit strong 3D understanding and can assess cross-view spatial consistency. The model evaluates consistency across three dimensions: (1) shape outline across views, (2) appearance, and (3) object parts. Each dimension is rated from 0 to 1, and their sum is used as the overall 3D consistency score.

Reinforcement learning has been proven effective for large language models and 2D generation, using reward models to capture human preferences and improve aesthetic quality. However, RL for 3D autoregressive generation remains underexplored. We therefore first study how reward model capability affects RL performance using GRPO with group size G = 8 and one policy update per iteration.

Experimental Analysis and Insights. Detailed results are reported in Table 1, and qualitative visualizations are shown in Figure 2. We standardize the reward evaluation by sampling six rendered views for each 3D-generated object. Based on these results, we draw two key insights:

Reward Model Design. Unlike code or math tasks, where deterministic verification functions provide direct rewards, multi-modal generation requires learned reward models. 3D generation faces unique challenges compared to 2D: (1) Complex design: 3D objects lack canonical viewpoints, requiring multi-dimensional reward systems that jointly evaluate realism, semantic alignment, and structural

• Human Preference reward serves as the core signal for RL in 3D autoregressive generation. Other reward dimensions offer limited standalone benefit but consistently improve performance when added on top of the preference re-

[Figure 49]

[Figure 50]

ShapeLLM-Omni

###### Benchmark Composition

[Figure 51]

Performance Comparison

Spatial & Structural Geometry

[Figure 52]

Mechanical Trellis Affordance

Stylized Representations

MME-3DR RL: Reward+Algorithm

Toys4K

s

26.3

23.

9

26.8 26.5

[Figure 53]

40

[Figure 54]

64

25.7%

20.5

16.1%

Biological &

Spatial & Structural Geometry

Organic Shapes 25.5

24.4

23.024.3

[Figure 55]

World-Knowledge Rare Objects

TOTAL 249

20.

3

19.1

22.7

15.4%

[Figure 56]

[Figure 57]

Mechanical Affordances

38

21.5%

[Figure 58]

19.2

19.

8

21.3%

Biological & Organic Shapes

22.7

23.2

[Figure 59]

54 53

[Figure 60]

24.

8

ShapeLLM-Omni RL: Reward+Algorithm

Trellis

26.0

Stylized

World-knowledge Rare Objects Representation

Figure 4. MME-3DR Benchmark Analysis. Left: MME-3DR contains 249 complex 3D objects across five categories. Right: We compare the performance of ShapeLLM-Omni, the SOTA model Trellis, and our RL-enhanced model on MME-3DR and Toys4K. Using CLIP Score as the metric, the results highlight the importance of implicit reasoning ability.

ward. As shown in Table 1, HPS V2.1 delivers the strongest gains among single-reward settings, and combining it with UnifiedReward or Qwen2.5-VL yields up to an additional

- 0.6 performance improvement.

• For a given reward dimension, specialized reward models show greater robustness than LMMs. However, for multi-view objectives such as 3D consistency, LMMs exhibit superior generalization. As shown in Table 1, combining HPS V2.1 with UnifiedReward outperforms pairing it with Qwen2.5-VL by 0.4. In contrast, when Qwen2.5-VL is used to assess 3D consistency, it delivers a 0.6 improvement in CLIP score. As shown in Figure 2, the 3D consistency reward effectively enhances coherence in color, texture, and geometry for both the guitar and the dolphin.

##### 5. Impact of Different RL Algorithms

GRPO [23], a widely used on-policy RL algorithm, has been applied to LLMs, LMMs, and 2D generation to enhance reasoning capabilities and generalization. Recentl GRPO variants, such as DAPO [40] and GSPO [37], have emerged, demonstrating superior efficiency and effectiveness on mathematical and coding tasks. However, their application to generation tasks remains underexplored. We therefore train and evaluate GRPO, DAPO, and GSPO on 3D autoregressive generation to systematically assess their respective advantages.

DAPO. DAPO [40] mitigates entropy collapse and training instability in vanilla GRPO by introducing several techniques, some of which are promising for 3D generation: (1) decoupled clipping bounds to enhance exploration and avoid oversmoothing of 3D object, (2) dynamic sample filtering to focus on medium-complexity 3D cases, (3) tokenlevel loss aggregation to reduce bias toward trivial shapes, and (4) removal of KL regularization for more flexible policy updates. We adopt the same settings as GRPO, with a group size of 8 and one iteration per update.

GSPO. To mitigate expert-activation fluctuations from token-level optimization in GRPO [44], GSPO [37] shifts optimization to the sequence level. It performs importance

sampling and clipping based on sequence likelihood under the current and reference policy models. This ensures that each 3D object is optimized as a coherent whole, preventing local token-level conflicts that could lead to inconsistent geometry. We adopt the same training setting as GRPO.

Experimental Analysis and Insights. As shown in Table 2, we systematically analyze RL algorithms—GRPO, DAPO, and GSPO—for 3D autoregressive generation, examining their strengths, limitations, and effective combinations. We further validate scaling strategies across training data and training iterations based on the optimal RL configuration in Figure 3. The key findings are as follows:

- • Compared to sequence-level operations, RL for 3D autoregressive generation favors token-level strategies. As shown in Table 2, under identical reward model settings, token-level averaging yields much larger gains than the sequence-level importance sampling and clipping.
- • Simple techniques can already stabilize training, particularly Dynamic Sampling, as long as policy updates remain properly constrained. As shown in Table 2, Dynamic Sampling improves vanilla GRPO by 0.6. However, completely removing the KL penalty leads to a 0.4 drop, while a more controlled method such as Decoupled Clip still yields gains by encouraging low-probability token exploration.
- • Scaling training data effectively mitigates preference bias and improves performance, while moderate iteration increases optimize results—though excessive training risks generalization degradation. As shown in Figure 3, expanding the dataset by 1.5×, 2×, and 3× yields gains of 0.4, 0.2, and 0.4, respectively. Doubling training iterations improves performance by 0.9, yet tripling them causes performance decline. This indicates significant generalization deterioration, likely attributable to overfitting on preference features.

##### 6. Exploration of Text-to-3D Benchmarks

Current text-to-3D models generalize well on simple prompts, yet remain vulnerable to certain categories—a phenomenon observed in both autoregressive and diffusion models. As shown in Figure 4, these challenging cases

[Figure 61]

[Figure 62]

[Figure 63]

Textual Prompt

[Figure 64]

[Figure 65]

Step 200 Step 400 Step 600

[Figure 66]

"Cyan pickup truck with beige seats, hexagonal wheels, square headlights, bar grill, rectangular bumpers, and red rear lights."

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

"Low-poly deer, elongated body, slender legs, narrow head, branching antlers, grazing pose."

Figure 5. Visualization results across different training stages

mainly fall into five types:

- 1. Spatial & Structural Geometry: objects with complex spatial layouts and component arrangements;
- 2. Mechanical Affordances: objects involving physical functionality or interactive mechanical components;
- 3. Biological & Organic Shapes: organisms (e.g., animals, plants) with dynamic organic characteristics;
- 4. World-Knowledge Rare Objects: low-frequency concepts requiring broader real-world knowledge;
- 5. Stylized Representation: non-photorealistic forms, , including cartoon, abstract, or stylistic interpretations.

These types correspond to five core implicit reasoning abilities essential for 3D generation: spatial, physical, dynamic, knowledge-based, and abstract reasoning. The absence of these abilities reveals that current models depend heavily on memorization rather than genuine 3D understanding. To address this gap, we propose MME-3DR, the first benchmark designed to evaluate the implicit reasoning capabilities of text-to-3D generation models.

MME-3DR. We curate 249 complex 3D objects across five categories, carefully selected from Toys4K [26], which are used neither in previous text-to-3D model training nor in current RL training. As shown in Figure 4, the benchmark comprises 16.1% objects with complex spatial structures, 21.5% with explicit mechanical and interactive components, 21.3% non-rigid dynamic objects such as animals and plants, 15.4% rare conceptual objects (e.g., fine-grained flower species), and 25.7% stylized or non-realistic objects originating from artistic or toy-like designs. Compared with our previous setup—randomly sampling 800 objects from the 105-category Toys4K dataset—MME-3DR intentionally balances samples across reasoning types critical for 3D generation, while maintaining broad object diversity.

Experimental Analysis and Insights. We evaluate ShapeLLM-Omni [39], Trellis [34], and our RL-enhanced model on the proposed MME-3DR benchmark and the randomly sampled subset of Toys4K [26]. As shown in the right panel of Figure 4, our key findings are as follows:

• Recent text-to-3D models perform reasonably on me-

chanical structures and non-rigid biological objects but struggle on the other three categories. RL training achieves substantial improvements across all five types. Figure 4 (radar chart) shows ShapeLLM-Omni and Trellis leading by over 1 point on mechanical affordances and biological & organic shapes, likely due to higher training data prevalence. RL training improves ShapeLLM-Omni by 5-6 points overall, with particularly notable gains in stylized representation, driven by enhanced implicit reasoning capabilities.

• MME-3DR evaluates implicit reasoning while simultaneously assessing general generation capability. The bar chart in Figure 4 reveals that Trellis significantly outperformes ShapeLLM-Omni on the randomly sampled Toys4K test set. This performance gap persists in MME-3DR, validating the effectiveness of its diverse object coverage.

##### 7. Exploration of RL Paradigms

In the text-to-3D task, we construct an RL paradigm for reasoning-guided 3D generation by leveraging the capabilities of ShapeLLM-Omni in text generation and token-wise 3D generation. Given a 3D textual prompt, the model first performs semantic reasoning to clarify user intent and resolve ambiguities. It then jointly models global structure and local texture details conditioned on both the prompt and the inferred reasoning, ultimately generating 3D tokens that are decoded into the final mesh.

While effective, the RL paradigm leaves substantial room for improvement. We observe that in early training stage, the model focuses on global geometry, producing coarse shapes with limited texture fidelity. As training progresses, reward signals drive refinement of materials and fine-grained textures, leading to clear gains in aesthetic quality and alignment with human preference. As shown in Figure 5, we evaluate checkpoints at steps 200, 400, and 600 on identical prompts. Early-stage outputs resemble only a rough cyan pickup-truck-like shape; later, features such as beige seats, square headlights, rectangular bumpers, and red taillights gradually emerge. A similar trend is observed for

Low-Level Visual Instruction

###### Step 1

###### Step 2

High-level Semantic Instruction

3D Prompt

“Stylized flower with gradient pink petals, yelow .”

“Produce a concise high-level semantic plan for.Focus on global shape primitives.”

“Given the text prompt and a high-level plan, write ONLY a detailed.Expand parts into precise geometric details.”

[Figure 73]

3D AutoRegressive Model

High-level Semantic Reasoning

Low-Level Visual Reasoning

Group Size G

Group Size G

Decode

Decode

Group Size G

Group Size G

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

"Stylized flower features a gradient of pink.with a bright yelow stamen.green stem.The spatial proportion."

"To create the stylized flower with gradient pink petals.irst identify the placement and size of.edge and gradualy transitioning.downwards from the center."

…

[Figure 78]

[Figure 79]

…

Coarse Shape Refined Object

Figure 6. Framework of Hi-GRPO. In Step 1, we instruct the model to generate high-level semantic reasoning based on the 3D prompt, and use it together with the prompt to produce a coarse 3D shape. In Step 2, conditioned on the 3D prompt and the high-level semantic CoT, the model generates low-level visual reasoning focused on local appearance details, which is used to produce the refined 3D object.

the deer example: the antlers are largely absent early on, and subsequently evolve into well-defined, branched structures. This coarse-to-fine progression intuitively aligns with human 3D perception, where global geometry is recognized first, followed by fine-grained visual cues.

Prompt Alignment

Shape Semantic Consistency

Human Preference

- Step 1
- Step 2

Prompt Alignment & Aesthetic Quality

Human Preference

Part Semantic-Visual Consistency

Appearance Consistency

This raises a question: Can hierarchical 3D generation process be integrated into RL paradigm to better align with the intrinsic nature of text-to-3D generation? To this end, we introduce Hi-GRPO, which disentangles RL training into hierarchical coarse-to-fine steps. In each iteration, the model first predicts global structure and then refines local textures and details, producing high-fidelity 3D assets.

[Figure 80]

Rendered Imag

e

HPS

Human aesthetic preferences?

0.16

3D Mesh

Prompt

Reward

[Figure 81]

[Figure 82]

Rendered Image

2.99 3.69 3.04 0.0

Unified Reward

Prompt Alignment? Logical Coherence?

Reward

Prompt Style Appealing?

[Figure 83]

Rendered Images

Hi-GRPO. As shown in Figure 6, our RL paradigm decomposes each training iteration into two steps, progressing from coarse geometry to fine-grained appearance. In the first step, the model first performs semantic planning at the global geometry level, guided by the 3D prompt and high-level instruction. This semantic reasoning serves three purposes: (1) understanding object subcategories to better capture generation intent, (2) establishing spatial layouts of key components to prevent geometric deviations, and (3) elaborating ambiguous terms to improve generation quality. This process produces |si| semantic tokens {si,1,si,2,...,si,|s

Qwen 2.5-VL

Shape Semantic Consistency?

Prompt

Reward

Appearance Consistency?

Prompt

1.94

“Wooden black chair with four legs and a rectangular backrest.”

[Figure 84]

3D Point

1.0

Clou Shape LLM

d Object Part Existence?

Prompt

Reward Object Part Completeness?

0.85

Figure 7. Illustration of the Reward Ensemble Design. We design reward ensembles for steps in Hi-GRPO: step 1 focuses on global alignment, while step 2 emphasizes local refinement.

i|}, where i ∈ {0,...,G − 1}. In Figure 6, the semantic reasoning determines the spatial layout and proportions of components (petals, stamen, stem), ensuring “the proportion of the flower is balanced”, and specifies color distribution as “a gradient of pink from the center to the outer”. The 3D prompt, semantic reasoning, and mesh start token <mesh start> are then fed into the model to generate 3D tokens grid by grid, where each grid depends on previous ones. This yields M 3D tokens {ti,1,ti,2,...,ti,M}, where M denotes the compressed grid count. The coarse 3D shape is obtained via decoding and re-

construction. As shown in Figure 6, this produces geometrically consistent flower shape with accurate colors. In the second step, conditioned on the prompt, semantic reasoning, and a low-level visual instruction, the model generates visual reasoning that focuses on refining local appearance through: (1) detailed component textures and interactions, and (2) local attributes such as element counts and symmetry. This produces |vi| visual tokens {vi,1,vi,2,...,vi,|v

i|}. As shown in Figure 6, this step clarifies the petal textures, stamen-petal spatial relations, and leaf counts. Finally, The model then generates the refined 3D object tokens

Textual Prompt

AR3D-R1 ShapeLLM-Omni Trellis LGM

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

"Standing dog with white and brown smooth fur, muscular build, raised head. "

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

"Three-step folding ladder, metallic steps, black A-frame, orange handrail, stabilizing bars."

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

"Abstract octopus with a bulbous head and cylindrical eyes."

###### Figure 8. Qualitative Comparison of Text-to-3D Models.

Table 4. Quantitative Comparison on Text-to-3D Generation Benchmarks. (KD is reported ×100. †: evaluated using shaded images of PBR meshes.)

MME-3DR Toys4K CLIP↑ KDincep ↓ KDdinov2 ↓ FDincep ↓ CLIP↑ KDincep ↓ KDdinov2 ↓ FDincep ↓

Method

LGM [27] 16.3 1.507 49.10 47.8 20.6 1.192 36.79 41.0 3DTopia-XL [5] 15.9† 1.635† 78.41† 61.9† 18.8† 1.439† 56.23† 54.3† SAR3D [3] 16.7 1.374 16.89 38.6 20.0 0.650 15.84 29.5 Trellis [34] 23.4 0.302 4.27 27.5 26.8 0.175 2.67 23.1 ShapeLLM-Omni [39] 19.8 0.451 6.73 34.1 22.7 0.249 3.27 27.7 AR3D-R1 28.5 0.194 2.74 25.9 29.3 0.156 1.85 20.4

Textual Prompt

hacking. As shown in Figure 7, our ensemble includes the following expert models, with some shared between steps:

STEP I STEP II

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

"Bright yellow dump bed, red cabin, large grey wheels, simplified dump truck."

- • Human Preference Model. We adopt HPS V2.1 [33] mentioned in Sec. 4 and apply it to both steps, computing the maximum prompt-view similarity across all viewpoints for the coarse shape and refined object. This yields human preference scores R1HPM and R2HPM as global reward signals.
- • Unified Reward Model. Given that HPS relies on similarity computation at 224×224 resolution, we introduce UnifiedReward [29] to evaluate prompt relevance and aesthetic quality. In step 1, UnifiedReward Think-qwen7B [29] scores prompt-coarse shape alignment (1-5) across views, taking the maximum as R1unified for geometric supervision. In step 2, we further evaluate the appearance quality. UnifiedReward-2.0-qwen-7b scores textured objects across views on logical coherence, style appeal, and prompt alignment (1-5 each). The maximum sum yields R2unified.

[Figure 113]

[Figure 114]

[Figure 115]

"Corinthian helmet featuring T-shaped face opening, nose guard, gold-trimmed crest, flared rim, horsehair crest."

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

"All-in-one desktop computer with a sleek white stand and curved back."

Figure 9. Results of Different Steps during Inference.

{oi,1,oi,2,...,oi,M}. We adopt the loss formulation from Sec. 5 with two modifications: (1) the reward from step 2 is backpropagated to the step 1 via Rhigh = Rhigh + λ · Rlow, allowing final quality to supervise global planning through configurable weight λ; (2) each step independently computes advantages and policy losses from its own rewards, yielding total loss L = Lhigh + Llow.

###### • 2D Large Multi-modal Model. Since existing spe-

cialized reward models inadequately handle 3D consistency verification, we adopt Qwen2.5-VL [1] for its strong multiview 3D understanding. Following the coarse-to-fine progression, in step 1, Qwen2.5-VL-7B verifies whether the generated shape matches the object category, assigning a binary score R1consist (0/1) for geometric constraint. In step 2, Qwen2.5-VL-7B assesses appearance consistency across views: color smoothness, material realism and coherence, and texture rationality, each scored 0-1 with sum R2consist.

Reward Ensemble Design. Text-to-3D quality assessment requires multi-view evaluation across multiple dimensions: aesthetic quality, prompt alignment, component completeness, and appearance consistency. To this end, we introduce tailored reward ensembles for Hi-GRPO. Multiple reward functions across steps effectively prevent reward

###### • 3D Large Multi-modal Model. However, 2D LMMs

struggle to accurately detect 3D components from multi-

view observations and may misidentify parts. To address this, in step 2 we sample the refined mesh into a 3D point cloud and employ ShapeLLM [19] to directly detect the existence (binary 0/1) and completeness (scored 0-1) of key components mentioned in the prompt, summing to R2part.

Experimental Analysis and Insights. We develop AR3D-R1 via Hi-GRPO training, the first RL-enhanced 3D autoregressive generation model, and evaluate it against existing text-to-3D methods on MME-3DR and sampled Toys4K test set. Table 4 shows quantitative results, and Figure 8 and 9 presents qualitative comparisons. For reward evaluation, we sample six views per object. Key findings:

• Hi-GRPO effectively enables hierarchical reasoning from global to local in 3D autoregressive generation. AR3D-R1 exhibits a coarse-to-fine progression, evolving from rough shapes to refined 3D objects. As illustrated in Figure 9, the model first generates basic shapes of the truck, helmet, and computer, then refines local details including colors, textures, and part structures to match the prompt.

- • AR3D-R1 achieves superior performance on both

MME-3DR and sampled Toys4K test set. As shown in Table 4, AR3D-R1 attains optimal results across benchmarks and demonstrates improved cross-dataset robustness compared to other text-to-3D models. As illustrated in Figure 8, AR3D-R1 produces high-quality meshes across diverse categories, including animals and mechanical objects.

8. Conclusion

This paper presents the first systematic study of reinforcement learning for text-to-3D autoregressive generation. We identify key factors in reward design, RL algorithms, text-to-3D benchmarks and RL paradigms. Our proposed Hi-GRPO leverages the hierarchical nature of 3D generation through dedicated reward ensembles, optimizing global-to-local generation from coarse shapes to fine textures. Building on insights, we develop AR3D-R1, the first RL-enhanced text-to-3D model, which demonstrates superior performance on our proposed MME-3DR and existing benchmarks, achieving significant improvements in geometry consistency and texture quality. Our work provides valuable insights for research in RL-driven 3D generation.

Contributions of Co-first Authors

- • Yiwen Tang: Conducted experiments, primary manuscript writing, and idea discussion.
- • Zoey Guo: Contributed to experiments and manuscript refinement.
- • Kaixin Zhu: Contributed to experiments and manuscript refinement.
- • Ray Zhang: Project leader, responsible for idea proposal, experimental design, and overall manuscript planning.

##### Overview

- • Sec. A: Experimental settings.
- • Sec. B: Details of Hi-GRPO.
- • Sec. C: Ablation study.
- • Sec. D: Additional visualizations.

##### A. Experimental Settings

We employ ShapeLLM-Omni [39] as the base model with a learning rate of 1 × 10−6, β of 0.01, and group size of 8. Training is conducted on 8 GPUs with a batch size of

- 1 per device and gradient accumulation over 2 steps. The model is trained for 1,200 steps. The configurable weight λ for supervising global planning with final quality is set to 1.0. Our reward models are deployed via the vLLM API framework. We select training prompt from ObjaverseXL [7], HSSD [14], and ABO [6], and evaluate our method on Toys4K [26].

- • Objaverse-XL: Objaverse-XL is one of the largest 3D object datasets currently available, comprising over 10 million 3D objects sourced from diverse platforms including GitHub, Thingiverse, Sketchfab and Polycam. The dataset undergoes rigorous deduplication and rendering validation, covering a range of categories and fine-grained attributes.
- • HSSD: HSSD contains 211 high-quality indoor synthetic 3D scenes with approximately 18,656 real-world object models, emphasizing indoor layouts, semantic structures, and object relationships.
- • ABO: ABO focuses on real-world household objects and provide approximately 147,000 product listings, nearly 400,000 catalog images, and about 8,000 3D models with rich material, geometric, and attribute annotations.
- • Toys4K: Toys4K includes approximately 4,000 3D object instances spanning around 105 categories, featuring diverse categories and significant shape variations.

##### B. Details of Hi-GRPO

###### B.1. Two-Step Generation Process

Step 1: The model generates semantic reasoning tokens si = {si,1,...,si,|s

i|} for global geometric planning, followed by 3d tokens ti = {ti,1,...,ti,M}, where M is the number of compressed grids. The coarse triangular mesh

M(1)i is decoded through the VQVAE decoder.

Step 2: Conditioned on semantic reasoning, the model generates visual reasoning tokens vi = {vi,1,...,vi,|v

i|} focused on local details, followed by 3d tokens oi = {oi,1,...,oi,M}, which are decoded into mesh M(2)i .

###### B.2. Hierarchical Reward Ensemble Design B.2.1. Human Preference Model

We adopt HPS V2.1 [33] in both steps. For generated 3D objects rendered from 6 uniformly distributed viewpoints

{v1,...,v6}, we compute text-image similarity at each viewpoint and take the maximum:

RiHPM,k = max

HPS(x,Render(M(ik),vj)) (1) This reward evaluates human preference with range [0, 1].

j=1,...,6

###### B.2.2. Unified Reward Model

Step 1: UnifiedReward Think-qwen-7B [29] evaluates geometric alignment between prompts and coarse shapes. Each of the 6 viewpoints is scored (1-5), and the maximum is:

Riunified,1 = max

fUR-Think(x,Render(M(1)i ,vj)) (2) This reward evaluates prompt alignment with range [1, 5].

j=1,...,6

Step 2: UnifiedReward-2.0-qwen-7b [29] performs three-dimensional evaluation of textured objects: (1) prompt alignment (1-5), (2) logical coherence (1-5), (3) style appeal (1-5). The maximum sum across 6 viewpoints:

Riunified,2 = max

fUR(ℓ)(x,Render(M(2)i ,vj)) (3)

j=1,...,6

ℓ∈Aapp

Aapp = {logic, style, align}. This reward evaluates 3 dimensions with range [3, 15].

###### B.2.3. 2D Large Multi-modal Model

Step 1: Qwen2.5-VL-7B [1] verifies whether the generated shape matches the object category in the prompt based on joint observation of 6 viewpoints:

Riconsist,1 = fQwencategory(x,{Render(M(1)i ,vj)}6j=1) (4) This reward evaluates category matching with range {0,1}.

Step 2: Qwen2.5-VL-7B [1] evaluates three dimensions of cross-view appearance consistency: (1) color smoothness (0-1), (2) material realism and coherence (0-1), (3) texture rationality (0-1):

Riconsist,2 =

fQwen(ℓ) x,{Render(M(2)i ,vj)}6j=1 . (5)

ℓ∈Aapp

Aapp = {color, material, texture}. This reward evaluates 3 dimensions with range [0, 3].

###### B.2.4. 3D Large Multi-modal Model

2D LMMs struggle to accurately detect 3D components from multi-view observations. To obtain accurate component completeness assessment, we employ direct evaluation based on 3D point clouds in step 2.

###### 1) Mesh to Dense Point Cloud Sampling: The refined

triangular mesh M(2)i = (V(2),F(2),T ) is converted to dense point cloud Pi. The sampling process:

1. Area-Weighted Sampling: For each triangle face f ∈ F(2), allocate sample points nf = ⌈ρ·Af⌉ based on area Af, where ρ is the sampling density parameter

###### Table 5. Quantitative comparisons using Toys4k for Reward Analysis.

Step 1 Reward Step 2 Reward Metrics

R1HPM R1unified R1consist R2HPM R2unified R2consist R2part CLIP Score↑ KDincep↓

- - - - ✓ ✓ - - 25.0 0.235
- - - - ✓ ✓ ✓ - 25.7 0.223

✓ ✓ - ✓ ✓ ✓ - 27.8 0.194 ✓ ✓ ✓ ✓ ✓ ✓ - 28.3 0.182 ✓ ✓ - ✓ ✓ ✓ ✓ 28.6 0.178 ✓ ✓ ✓ ✓ ✓ ✓ ✓ 29.3 0.156

###### Table 6. Quantitative comparisons using Toys4k for Different RL Paradigms.

Training Strategy Metrics GRPO Textual Reasoning Step1 Reward Step2 Reward Hi-GRPO CLIP Score↑ KDincep↓

- - - - - - 22.7 0.249

✓ - - - - 24.3 0.237 ✓ ✓ - - - 25.2 0.228 ✓ ✓ ✓ - - 24.8 0.235 ✓ ✓ - ✓ - 26.0 0.214

- - - - - ✓ 28.7 0.182

###### 2. Barycentric Uniform Sampling: Within face f =

(v1,v2,v3), generate random barycentric coordinates (α,β,γ) satisfying α + β + γ = 1 and α,β,γ ≥ 0. Sample point coordinates: p = αv1 + βv2 + γv3

###### 3. Texture Color Sampling: Interpolate UV coordinates using barycentric coordinates uv = αuv1 + βuv2 + γuv3, and sample RGB color from texture map T

The result is point cloud Pi = {(pk,ck)}Nk=1p , where pk ∈ R3 is position, ck ∈ R3 is RGB color.

2) Per-Component Evaluation: Parse prompt x to extract component list C = {c1,...,cN

c} and expected quantities {n1,...,nN

c}. ShapeLLM [19] processes point cloud Pi and component queries. For each component cp:

- • Existence: ep ∈ {0,1} determines the existence.
- • Completeness: qp ∈ [0,1] evaluates geometric completeness, shape correctness, and quantity matching

Average scores across Nc components:

1 Nc

Ripart,2 =

Nc

(ep + qp) (6)

p=1

This reward evaluates 2 dimensions per component (existence + completeness), with averaged range [0, 2].

###### B.2.5. Dimension-Normalized Reward Ensemble

Each reward is normalized by its number of evaluation dimensions to ensure fair contribution:

###### Step 1 Total Reward:

Rihigh = RiHPM,1 + Riunified,1 + Riconsist,1 (7)

###### Step 2 Total Reward:

Ripart,2 2

Riconsist,2 3

Riunified,2 3

Rilow = RiHPM,2 +

(8)

+

+

This normalization strategy ensures: (1) each reward’s contribution is proportional to its number of evaluation dimensions; (2) multi-dimensional evaluations do not dominate through simple summation; (3) the system remains stable when adding or removing rewards. Rewards from step 2 are backpropagated to step 1 through weight λ:

R˜ihigh = Rihigh + λ · Rilow (9)

When λ = 1.0, the high-level step is directly supervised by final output quality. For each step, advantages are normalized within prompt groups to eliminate reward scale differences across prompts:

R˜ihigh − µ(1)g σg(1) + ϵ

Rilow − µ(2)g σg(2) + ϵ

A(1)i =

, A(2)i =

(10)

where µ(gk) = G1 Gj=1 Rj(k), σg(k) =

G j=1(Rj(k) − µ(gk))2, ϵ = 10−4.

1 G

###### B.3. Loss Computation

For each step, we compute token-level log probabilities by concatenating the log probabilities of reasoning tokens and mesh tokens. In step 1, the complete sequence log probability concatenates semantic reasoning and

coarse mesh generation: log πθ(yi(1)) = concat(log πθ(si |

Textual Prompt +Reasoning-GRPO +Step1 Reward +Step2 Reward +Hi-GRPO

+GRPO

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

"Knife with unique grooved black handle and guard. "

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

"Brown high-top sneaker with black stripes, leather, padded collar, stitched cap, metal eyelets, grey sole."

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

Figure 10. Visualization Results of Small Objects for Different RL Paradigms.

x),log πθ(ti | x,si)). In step 2, it concatenates visual reasoning and refined mesh generation: log πθ(yi(2)) = concat({log πθ(vi,t | x,si,vi,<t)}|tv=1i|,{log πθ(oi,t | x,vi,oi,<t)}Mt=1). Reference policy log probabilities log πref(yi(k)) are computed similarly.

For stage k ∈ {1,2}, the complete loss function is:

L(k) = −Eq∼D,{y(k)

i }Gi=1∼πθold

Ti(k)

G

1

G i=1 Ti(k)

t=1

i=1

m(i,tk) min ri,t(k)(θ)A(ik),clip(ri,t(k)(θ),

1 − εlow,1 + εhigh)A(ik) − β · KL(i,tk)

(11) We highlight and describe key components as follows:

###### • Policy Ratio:

πθ(yi,t(k)|yi,<t(k) ) πθ

ri,t(k)(θ) =

(12)

(yi,t(k)|yi,<t(k) )

old

###### • Decoupled Clipping: Asymmetric clipping thresholds

εlow and εhigh. The higher threshold allows low-probability tokens greater probability increase space, promoting exploration and preventing entropy collapse.

- • Token-Level Averaging: The loss is normalized by the token count Gi=1 Ti(k), where Ti(k) = Tt=1max m(i,tk) is the number of valid tokens and m(i,tk) is the completion mask.
- • KL Regularization: Token-level KL divergence

πref(yi,t(k)|yi,<t(k) ) πθ(yi,t(k)|yi,<t(k) )

πref(yi,t(k)|yi,<t(k) ) πθ(yi,t(k)|yi,<t(k) )

KL(i,tk) =

− 1 (13)

− log

with penalty coefficient β = 0.01 prevents the policy from deviating too far from the reference. The two steps compute losses independently. The total optimization objective:

Ltotal = L(1) + L(2) (14)

##### C. Ablation Study

###### C.1. Reward Analysis.

This section investigates reward function selection and combination for AR3D-R1. Table 5 presents our findings. We first examine whether Step-2 rewards can simultaneously optimize both generation steps. Results show that rewards from refined objects struggle to control both coarse geometry and fine texture effectively. Even with combined rewards (R2HPM+R2unified+R2consist), improvements remain marginal. However, introducing step-specific rewards, adding R1HPM+R1unified for Step 1, yields substantial gains,

Textual Prompt +GRPO +Reasoning-GRPO +Step1 Reward +Step2 Reward +Hi-GRPO

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

"An hourglassshaped violin with reddish-brown polished wood, two F-holes, a bridge, scroll, and four silver strings."

[Figure 156]

[Figure 157]

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

"Silver-framed motorcycle with blue body panels, hexagonal rims, black low-profile tires, geometric handlebars, yellow headlight, and black seat. "

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

Figure 11. Visualization Results of Large Objects for Different RL Paradigms.

improving CLIP scores by 2.1 point. Notably, componentlevel rewards prove critical for ensuring correct part positioning, quantity, and structural plausibility.

###### C.2. Effectiveness of Hi-GRPO.

To validate Hi-GRPO, we conduct ablation studies using the baseline GRPO algorithm. Table 6 presents quantitative results, while Figures 10 and 11 provide extensive visualizations. We first compare direct 3D token optimization against textual reasoning-guided GRPO, both evaluated with HPSV2.1+UnifiedReward+Qwen2.5-VL reward system. Quantitatively, textual reasoning yields a 0.9-point CLIP score improvement, while qualitatively it enables effective global planning for both large and small objects. We then examine Hi-GRPO’s hierarchical reward ensemble by separately applying Step-1 and Step-2 reward systems. Since Step-1 rewards focus on high-level geometric structure, Table 6 shows performance degradation, with visualizations revealing noticeably reduced texture fidelity. Ultimately, the hierarchical RL paradigm of Hi-GRPO, combining global-to-local generation with step-specific reward ensembles, achieves substantial improvements across geometry, fine-grained textures, and prompt alignment.

##### D. Additional Visualizations

Figures 12, 13, 14, 15, and 16 visualize the generation results of our proposed AR3D-R1, ShapeLLM-Omni, and

Trellis across the five categories in our MME-3DR benchmark. Figures 17, 18, and 19 visualize AR3D-R1’s hierarchical generation process across different object categories.

"Metallic helmet with side ear discs and textured details."

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

"Dark brown wooden rocking chair with curved armrests and spindle backrest. "

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

###### Figure 12. Visualization Results of Spatial & Structural Geometry in MME-3DR.

Textual Prompt

AR3D-R1 Trellis ShapeLLM-Omni

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

"A deep red rose, with smooth curled petals on a slender green stem. "

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

"White 3D model of standing elephant with large ears and tusks. "

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

###### Figure 13. Visualization Results of Biological & Organic Shapes in MME-3DR.

"Stylized, lowpoly teal frog with triangular mouth, no eyes, and white oval markings. "

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

"White oval abstract chicken with pink eyes, light brown beak, wavy pink comb, brown tail and beige feet. "

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

###### Figure 14. Visualization Results of Stylized Representations in MME-3DR.

Textual Prompt

AR3D-R1 Trellis ShapeLLM-Omni

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

"Electric guitar with green asymmetrical body, white pickguard, yellow neck. "

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

"Office chair with gray padding, curved backrest, black framed, five legs, adjustable height. "

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

###### Figure 15. Visualization Results of Mechanical Affordances in MME-3DR.

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

“Parrot fish 3D object with elongated shape, multiple fins, and color transition from yellowishorange to white. "

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

"Low-poly, white chess pawn with spherical top, disc middle, and wide base. "

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Figure 16. Visualization Results of World-Knowledge Rare Objects in MME-3DR.

#### AR3D- R1 STEP II

Reasoning Process

[Figure 301]

[Figure 302]

"The global shape of the dump truck is rectangular with a large gray rectangular base and a bright yellow frame. The cabin is red and sits atop the gray base, and there are large wheels in proportion to the size of the vehicle. The dump bed is wide and deep, designed to hold a significant amount of material. "

Textual Prompt

#### AR3D- R1

Reasoning Process

"Bright yellow dump bed, red STEP I cabin, large grey wheels, simplified dump truck. "

"The gray color of the base creates a durable and sturdy platform for the truck to operate on. The bright yellow frame and red cabin stand out against the gray backdrop, making it easy to spot in traffic. The large wheels, proportional to the size of the vehicle, offer stability and support to the heavy weight of the truck. The dump bed is wide and deep, providing ample space to hold a substantial amount of material. The dump truck's design is simple and streamlined, with clean lines and geometric shapes that give it a professional and modern look. The overall structure of the truck is symmetrical, ensuring balance and harmony in its appearance. Consistency in color and style creates a cohesive and polished look for this industrial vehicle. "

[Figure 303]

[Figure 304]

Figure 17. Visualization of the Two-Step Reasoning Generation Process in Cabin.

#### AR3D- R1 STEP II

Reasoning Process

[Figure 305]

[Figure 306]

"The all-in-one desktop computer has a rectangular shape with a sleek white stand and curved back, designed to fit comfortably on any desk. The main components are arranged in a compact and efficient manner, with the monitor positioned above the keyboard and mouse. "

Textual Prompt

#### AR3D- R1

Reasoning Process

"All-in-one desktop computer STEP I with a sleek white stand and curved back."

[Figure 307]

[Figure 308]

"The all-in-one desktop computer features a rectangular shape that is designed to fit comfortably on any desk. The sleek white stand and curved back add a touch of elegance to the computer's overall appearance. The main components are arranged in a compact and efficient manner, with the monitor positioned above the keyboard and mouse. The keyboard and mouse are designed to be ergonomic, making typing and navigation a breeze. The overall design emphasizes simplicity and ease of use, ensuring that even novice users can navigate the system with ease. With its sleek and compact design, the all-inone desktop computer is an ideal choice for both home and office use. "

Figure 18. Visualization of the Two-Step Reasoning Generation Process in Desktop Computer.

#### AR3D- R1 STEP II

Reasoning Process

"The stylized flower features a gradient of pink from the center to the outer petals, with a bright yellow stamen at its core. The green stem is vivid and contrasts with the petals' softness. The arrangement of the petals is symmetric, with three large central petals surrounded by smaller ones. The spatial proportion of the flower is balanced, with the stem erect and the petals spreading outward in all directions. "

[Figure 309]

[Figure 310]

Textual Prompt

#### AR3D- R1

Reasoning Process

"Stylizedpink petals,floweryellowwithstamen,gradient STEP I bright green stem, and vivid leaves. "

"To create the stylized flower with gradient pink petals, yellow stamen, bright green stem, and vivid leaves, first identify the placement and size of the center flower part. Then, begin drawing the petals in a circular pattern, with the gradient pink starting from the outer edge and gradually transitioning to the center. Ensure that each petal is symmetrical and evenly spaced. Next, draw the vivid green stem extending downwards from the center, with proportional length and width to support the flower. Finally, add several leaves, varying in size and shape, surrounding the flower to provide context and balance. The result will be a visually pleasing and harmonious design. "

[Figure 311]

[Figure 312]

Figure 19. Visualization of the Two-Step Reasoning Generation Process in Stylized Flower.

##### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5, 9, 11
- [2] Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, et al. Meshanything: Artist-created mesh generation with autoregressive transformers. arXiv preprint arXiv:2406.10163,

2024. 3, 4

- [3] Yongwei Chen, Yushi Lan, Shangchen Zhou, Tengfei Wang, and Xingang Pan. Sar3d: Autoregressive 3d object generation and understanding via multi-scale 3d vqvae. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28371–28382, 2025. 4, 9
- [4] Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, and Guosheng Lin. Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13922–13931, 2025. 4
- [5] Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling highquality 3d asset generation via primitive diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26576–26586, 2025. 3, 9
- [6] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21126– 21136, 2022. 4, 11
- [7] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36:35799–35813, 2023. 4, 11
- [8] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776,

2025. 3

- [9] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 3
- [10] Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Rui Huang, Haoquan Zhang, Manyuan Zhang, Jiaming Liu, Shanghang Zhang, Peng Gao, et al. Can we generate images with cot? let’s verify and reinforce image generation step by step. arXiv preprint arXiv:2501.13926, 2025. 1, 3
- [11] Zekun Hao, David W Romero, Tsung-Yi Lin, and Ming-Yu Liu. Meshtron: High-fidelity, artist-like 3d mesh generation at scale. arXiv preprint arXiv:2412.09548, 2024. 1

- [12] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749,

2025. 3

- [13] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025. 1, 3, 5
- [14] Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X Chang, and Manolis Savva. Habitat synthetic scenes dataset (hssd-200): An analysis of 3d scene scale and realism tradeoffs for objectgoal navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16384–16393,

2024. 4, 11

- [15] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1
- [16] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5971–5984, 2024. 1
- [17] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 3
- [18] OpenAI: Introducing OpenAI o3 and o4 mini. 2025. (2025), https://openai.com/o3/. 1, 3
- [19] Zekun Qi, Runpei Dong, Shaochen Zhang, Haoran Geng, Chunrui Han, Zheng Ge, Li Yi, and Kaisheng Ma. Shapellm: Universal 3d object understanding for embodied interaction. In European Conference on Computer Vision, pages 214–

238. Springer, 2024. 10, 12

- [20] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023. 1, 3, 4
- [21] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 4
- [22] ByteDance Seed, Yuyu Zhang, Jing Su, Yifan Sun, Chenguang Xi, Xia Xiao, Shen Zheng, Anxiang Zhang, Kaibo Liu, Daoguang Zan, et al. Seed-coder: Let the code model curate data for itself. arXiv preprint arXiv:2506.03524,

2025. 1, 3

- [23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 1, 3, 4, 6
- [24] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao,

- Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 3
- [25] Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. Meshgpt: Generating triangle meshes with decoder-only transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19615–19625, 2024. 1, 3, 4
- [26] Stefan Stojanov, Anh Thai, and James M Rehg. Using shape to categorize: Low-shot learning with an explicit shape bias. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1798–1808, 2021. 1, 7, 11
- [27] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pages 1–18. Springer, 2024. 9
- [28] Chengzhuo Tong, Ziyu Guo, Renrui Zhang, Wenyu Shan, Xinyu Wei, Zhenghao Xing, Hongsheng Li, and Pheng-Ann Heng. Delving into rl for image generation with cot: A study on dpo vs. grpo. arXiv preprint arXiv:2505.17017, 2025. 1, 3
- [29] Yibin Wang, Zhimin Li, Yuhang Zang, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Unified multimodal chain-of-thought reward model through reinforcement finetuning. arXiv preprint arXiv:2505.03318, 2025. 5, 9, 11
- [30] Zhengyi Wang, Jonathan Lorraine, Yikai Wang, Hang Su, Jun Zhu, Sanja Fidler, and Xiaohui Zeng. Llama-mesh: Unifying 3d mesh generation with language models. arXiv preprint arXiv:2411.09595, 2024. 4
- [31] Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. Crm: Single image to 3d textured mesh with convolutional reconstruction model. In European conference on computer vision, pages 57–74. Springer, 2024. 4
- [32] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, et al. Direct3d-s2: Gigascale 3d generation made easy with spatial sparse attention. arXiv preprint arXiv:2505.17412, 2025. 4
- [33] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 5, 9, 11

- [34] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21469–21480, 2025. 3, 4, 7, 9
- [35] Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, and Shenghua Gao. Dream3d: Zero-shot text-to-3d synthesis using 3d shape prior and text-to-image diffusion models. In Proceedings of the IEEE/CVF Con-

- ference on Computer Vision and Pattern Recognition, pages 20908–20918, 2023. 3, 4
- [36] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025. 3
- [37] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2, 6
- [38] Xianghui Yang, Huiwen Shi, Bowen Zhang, Fan Yang, Jiacheng Wang, Hongxu Zhao, Xinhai Liu, Xinzhou Wang, Qingxiang Lin, Jiaao Yu, et al. Hunyuan3d 1.0: A unified framework for text-to-3d and image-to-3d generation. arXiv preprint arXiv:2411.02293, 2024. 3, 4
- [39] Junliang Ye, Zhengyi Wang, Ruowen Zhao, Shenghao Xie, and Jun Zhu. Shapellm-omni: A native multimodal llm for 3d generation and understanding. arXiv preprint arXiv:2506.01853, 2025. 1, 3, 4, 7, 9, 11
- [40] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 2, 6
- [41] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024. 1, 3
- [42] Ruowen Zhao, Junliang Ye, Zhengyi Wang, Guangce Liu, Yiwen Chen, Yikai Wang, and Jun Zhu. Deepmesh: Autoregressive artist-mesh creation with reinforcement learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10612–10623, 2025. 3, 4
- [43] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 4
- [44] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. 6
- [45] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 3

