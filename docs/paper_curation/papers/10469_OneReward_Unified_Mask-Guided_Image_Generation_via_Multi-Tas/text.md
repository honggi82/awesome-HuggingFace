# arXiv:2508.21066v1[cs.CV]28Aug2025

ONEREWARD: UNIFIED MASK-GUIDED IMAGE GENERATION VIA MULTI-TASK HUMAN PREFERENCE LEARNING

Yuan Gong∗ Xionghui Wang∗† Jie Wu Shiyin Wang Yitong Wang Xinglong Wu ByteDance Inc.

ABSTRACT

In this paper, we introduce OneReward, a unified reinforcement learning framework that enhances the model’s generative capabilities across multiple tasks under different evaluation criteria using only One Reward model. By employing a single vision-language model (VLM) as the generative reward model, which can distinguish the winner and loser for a given task and a given evaluation criterion, it can be effectively applied to multi-task generation models, particularly in contexts with varied data and diverse task objectives. We utilize OneReward for mask-guided image generation, which can be further divided into several subtasks such as image fill, image extend, object removal, and text rendering, involving a binary mask as the edit area. Although these domain-specific tasks share same conditioning paradigm, they differ significantly in underlying data distributions and evaluation metrics. Existing methods often rely on task-specific supervised fine-tuning (SFT), which limits generalization and training efficiency. Building on OneReward, we develop Seedream 3.0 Fill, a mask-guided generation model trained via multi-task reinforcement learning directly on a pre-trained base model, eliminating the need for task-specific SFT. Experimental results demonstrate that our unified edit model consistently outperforms both commercial and open-source competitors, such as Ideogram, Adobe Photoshop, and FLUX Fill [Pro], across multiple evaluation dimensions. Code and model are available at: https://one-reward.github.io

1 INTRODUCTION

Recent advancements in diffusion model(Rombach et al. (2022); Podell et al. (2023); Labs (2024)) have enabled a diverse range of challenging tasks, such as inpainting, outpainting, object removal, and text rendering. Although these tasks share a common mask-guided input format, they exhibit significant divergence in conditional distributions and evaluation metrics, presenting considerable challenge to the development of a unified, versatile model. Inpainting, also known as image fill, involves modifying or adding specific objects within a localized masked area, emphasizing accurate prompt alignment, aesthetic coherence, and structural integrity. Outpainting, or image-extend, requires generating extensive content around an existing image, expanding beyond its original borders, with a strong emphasis on visual aesthetics, seamless integration, and structural consistency. Object removal entails filling a masked region based on surrounding context, requires avoiding the generation of extra object, and ensuring texture consistency with the original image. Text rendering specifically targets accurate rendering of textual elements, emphasizing precision in generating and aligning fonts according to given instructions. Current state-of-the-art generative models typically excel within specific editing tasks but struggle to maintain consistently high performance across multiple tasks simultaneously. Existing methods or community models often rely on taskspecific supervised fine-tuning (SFT), or LoRA(Hu et al. (2022)) with limited data base on SD1.5Inpaint(Rombach et al. (2022)) and FLUX Fill(Labs (2024)), which restricts their generalization

*Equal contribution. †Project leader.

Overall Usability Rate (%)

TextRendering(%)

TextAlignment

Texture Consistency

Aesthetics

StyleConsistency

Structure

Seedream 3.0 Fill Higgsfield Flux Fill [pro] Ideogram Adobe Photoshop

(a) Image Fill

Overall Usability Rate (%)

TextAlignment

Aesthetics

TextureConsistency

Structure

Style Consistency

Seedream 3.0 Fill Ideogram Flux Fill [pro] Adobe Photoshop

(b) Image Extend with Prompt

Overall Usability Rate (%)

Texture Consistency

Aesthetics

StyleConsistency

Structure

Seedream 3.0 Fill Ideogram Flux Fill [pro] Midjourney Adobe Photoshop

(c) Image Extend without Prompt

Overall Usability Rate (%)

RemovalQuality(%)

TextureConsistency

Structure

Seedream 3.0 Fill Adobe Photoshop Ideogram Flux Fill [pro]

(d) Object Removal

- Figure 1: Overall evaluation across four image editing tasks, and text rendering is included in image fill. For each sub-task, we selected only state-of-the-art models or closed-source APIs as competitors and conducted detailed evaluations in multiple dimensions. Note that different tasks have different evaluation criteria.

to diverse editing scenarios. This reveals the difficulty of designing a unified framework capable of supporting multiple image editing tasks while avoiding the inefficiencies of task-specific fine-tuning.

Reinforcement learning from human feedback (RLHF) methods for diffusion and flow matching model, such as Direct Preference Optimization (DPO)(Rafailov et al. (2023); Wallace et al. (2024); Xu et al. (2024); Liu et al. (2025b)), reward-base method (Xu et al. (2023); Zhang et al. (2024); Li et al. (2024); Gao et al. (2025b)) and RL-based method (Black et al. (2023); Liu et al. (2025a); Xue et al. (2025)) have shown strong promise in aligning generative outputs with human preferences across text-to-image and text-to-video domains. However, DPO faces fundamental limitations in handling diverse tasks and evaluation dimensions concurrently, as it inherently assumes a well-defined preference order that may not hold across heterogeneous tasks and criteria.For instance, DPO cannot unambiguously determine the winner and loser when an image is better in aesthetics but worse in structure than its counterpart. Reward Feedback Learning (ReFL), while

significantly boosting model performance in specific dimensions, typically requires training separate reward models for each evaluation criterion when using traditional multimodal architectures such as BLIP(Li et al. (2022)) and CLIP(Radford et al. (2021)), increasing training and tuning complexity. Furthermore, ReFL encounters reward conflicts in multi-task scenarios, where high quality object generation may receive completely opposite evaluations in the task of image-fill and objectremoval. FlowGRPO(Liu et al. (2025a)) and DanceGRPO(Xue et al. (2025)) introduce GRPO(Shao

- et al. (2024)), which is powerful in Large Language Model(LLM), into flow matching models, by converting deterministic Ordinary Differential Equation(ODE) sampleing into a Stochastic Differential Equation (SDE) framework. While GRPO-based methods significantly enhance performance on vision generation tasks, they rely on policy-based estimation by introducing a group-relative formulation to estimate the advantage, without explicitly maximizing reward signals during optimization. This often results in slower convergence compared to reward-driven approaches.

To overcome these limitations, we introduce OneReward, a unified reinforcement learning framework for mulit-task image generation using only one VLM as the reward model. By incorporating task category and evaluation metric information (e.g. aesthetics, structure, consistency) directly into its queries, the VLM can effectively distinguish between tasks and evaluation criteria, enabling it to make pairwise judgments and determine which output is better under certain setting. Base on OneReward, we adopt Seedream 3.0(Gao et al. (2025a)) as the pre-trained base model and develop Seedream 3.0 Fill, a state-of-the-art (SOTA) mask-guided image generation model that consistently delivers superior performance across a diverse set of tasks, including image fill, image extension, object removal, and text rendering. Seedream 3.0 Fill is directly optimized via reinforcement learning from a pre-trained model, without any SFT. During training, we treat the initial pre-trained model as the reference model and the training one as the policy model, optimizing the latter to generate results that surpass the reference model in each task-specific evaluation metric. The reward signal is derived from the probability of the token “Yes” generated by the VLM, which is then used for gradient backward. To the best of our knowledge, this is the first work to employ reinforcement learning as a direct optimization paradigm in the context of multi-task image editing. The main contributions of our work are threefold:

- 1. We propose OneReward, a novel reward model framework for the visual domain by employing VLM as the generative reward model to enhance multi-task reinforcement learning, significantly improving the policy model’s generation ability across diverse scenarios.
- 2. Building on OneReward, we develop Seedream 3.0 Fill, a unified SOTA image editing model capable of effectively handling diverse tasks including image fill, image extend, object removal, and text rendering. It surpasses several leading commercial and opensource models, including Ideogram, Adobe Photoshop, and FLUX Fill [Pro].
- 3. By applying our multi-task reinforcement learning approach on FLUX Fill [dev], we introduce and open-source FLUX Fill [dev][OneReward], a generalized image editing model that outperforms the original model on both inpainting and outpainting tasks, serving as a powerful new baseline for future research in unified mask-guided image generation.

2 RELATED WORK

Mask-guided image generation: Image inpainting and outpainting focus on generating coherent and seamless content for missing or external regions of an image. With the advent of deep learning, methods based on Generative Adversarial Networks (GAN)(Goodfellow et al. (2020)) became dominant. Notably, Large Mask Inpainting(LaMa)(Suvorov et al. (2022)) introduced Fast Fourier Convolutions, significantly improving the ability to handle large, complex masks while preserving global structural consistency, a common failure point for earlier CNN-based methods. More recently, diffusion models(Ho et al. (2020); Song et al. (2020a); Rombach et al. (2022); Song et al. (2020b)) have become the state-of-the-art due to their superior generative quality. RePaint (Lugmayr et al. (2022)) was an early method that applied a pre-trained unconditional diffusion model to inpainting by repeatedly sampling the unknown region and blending it with the known context, although its iterative nature can be computationally intensive. Subsequent models, such as the native inpainting variant of Stable Diffusion(Rombach et al. (2022); Podell et al. (2023)), adopted a more efficient approach by concatenating the latent representations of the mask and the original image as input to its origin text-to-image model. This paradigm established a strong foundation

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

red wedding dress Canton Tower Temple of Heaven

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

a cat Kimono

a big sunflower

girl on a bicycle

Image Fill

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Two burgundy Chinese style lanterns colorful hot air balloon cable car

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Image Extend

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Object Removal

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

“李老太自动洗碗器” “随便” “澳门”

“运”

Text Rendering

- Figure 2: Visual showcase of Seedream 3.0 Fill results across four scenario: image fill, image extend, object removal and text rendering. Each column presents a representative example with corresponding prompts and outputs, demonstrating the model’s unified capability across diverse generation objectives.

for high-fidelity, text-guided editing. Follow-up works, such as MagicBrush(Zhang et al. (2023)) and Inst-Inpaint (Yildirim et al. (2023)), introduced more refined instruction-based datasets to improve the accuracy of image editing. ByteEdit(Ren et al. (2024)) explored the use of feedback learning to boost performance in these tasks but separate SFT and RL procedures are applied in different sub-task. Recently, FLUX Fill(Labs (2024)) has emerged as a powerful open source baseline demonstrating strong performance in both inpainting and outpainting. However, these models are often specialized or lack robust generalization across multiple, distinct editing modalities. Our unified edit model builds directly upon these foundations, but addresses their limitations by leveraging a novel multi-task RLHF framework, unify inpaint, outpainting, object removal, and text rendering within a single, proficient model.

RLHF for diffusion model: Aligning generative models with human preferences has emerged as a rapidly advancing research area, aiming to enhance the aesthetic quality, instruction alignment, and overall user expectation of generated visual content. The success of RLHF critically depends on the quality of the reward model. ReFL (Xu et al. (2023)) makes an important step toward generalpurpose reward modeling by training on a large-scale dataset of expert comparisons. It further proposes an algorithm to directly fine-tune diffusion models by treating reward scores as human preference losses and backpropagating them to randomly selected later steps in the denoising process. Subsequent research, such as VisionReward (Xu et al. (2024)), has explored more fine-grained, multi-dimensional reward modeling by decomposing human preferences into interpretable axes such as fidelity, composition, safety and text alignment. However, its reliance on logistic regression to weight these dimensions introduces additional complexity, limiting its applicability in fully end-toend training pipelines and reducing generalizability to broader scenarios. Adapting RLHF algorithms from the large language model (LLM) domain to diffusion models presents a set of unique challenges. Direct Preference Optimization (DPO) (Rafailov et al. (2023)) was proposed as a simpler and stable alternative to the full RL pipeline. Instead of relying on explicit reward model, DPO optimizes the policy model by directly maximizing the difference in log-probability ratios between preferred and dispreferred responses. This method was effectively extended to the visual domain with Diffusion-DPO (Wallace et al. (2024)), which reformulates the objective in terms of diffusion model likelihood, enabling direct and stable preference alignment. Denoising Diffusion Policy Optimization (DDPO) (Su et al. (2024)) was a pioneering work that successfully applied policy gradient methods to diffusion models by casting the denoising process as a multi-step decision-making problem. Further algorithmic advances include Group Relative Policy Optimization (GRPO) (Shao et al. (2024)), which has demonstrated strong performance in aligning both diffusion and flow matching models, as evidenced by its applications in FlowGRPO (Liu et al. (2025a)) and DanceGRPO (Xue

- et al. (2025)).

OneReward synthesizes recent advances in alignment strategies into a unified framework. Our work pushes this frontier further by leveraging only one VLM as the generative reward model to produce task-aware feedback for our multi-task reinforcement learning. It addresses a key limitation of traditional algorithms such as DPO, which struggle to distinguish the winner from the loser when preferences vary across different evaluation dimensions. With OneReward, we develop a SOTA image editing model that jointly learns multiple sub-tasks within a unified reinforcement learning framework.

- 3 PRELIMINARIES

- 3.1 FLOW MATCHING

Flow Matching(Lipman et al. (2022)) represents a new class of generative models that offers a more efficient and stable training paradigm compared to traditional diffusion models. Instead of learning the score function of a data distribution, flow matching models learn a velocity vector field that transports a simple prior distribution (e.g., normal distribution) to a complex data distribution through a continuous normalizing flow (CNF). A CNF is defined by an ordinary differential equation (ODE) that describes the trajectory of a sample x over a continuous time variable t ∈ [0,1]: dxt

dt = vt(xt). Here, x0 is sampled from the prior distribution p0, and x1 follows the target data distribution p1. The function vt(xt) is the time-dependent vector field. Flow matching aims to train a neural network vθ(x,t,c) (where c represents conditioning information such as text or binary mask) to

approximate a target vector field ut(x|c). The conditional flow matching loss is formulated as a simple regression objective:

t(x|c),c ∥vθ(x,t,c) − ut(x|c)∥2

LFM(θ) = Et,p

Rectified Flow(Liu et al. (2022)) is a powerful special case of flow matching that linearizes the transport paths to maximize sampling efficiency. It simplifies the objective by defining the target vector field as the constant direction between a data sample x1 and its corresponding noise sample x0, such that ut(x|c) = x1 − x0. This formulation avoids the complex score-matching objective of diffusion models, often leading to faster convergence and more efficient generation. Our model, is trained upon this efficient trainging methodology.

- 3.2 REINFORCEMENT LEARNING FROM HUMAN FEEDBACK

RLHF is a powerful technique for aligning generative models with complex, hard-to-specify human preferences. Human preference data is collected, typically in the form of comparisons. For a given input c, annotators are shown two outputs, winner xw and loser xl, and asked to choose which one they prefer. This pairwise preference dataset is used to train a reward model rϕ(c,x) that predicts a scalar score reflecting human preference. The reward model is often trained using a binary cross-entropy loss based on the Bradley-Terry(Bradley & Terry (1952)) model, which states that the probability of preferring xw over xl is:

### P(xw ≻ xl|c) = σ(rϕ(xw,c) − rϕ(xl,c)) where σ is the sigmoid function.

ReFL(Xu et al. (2023)) proposes a direct gradient-based fine-tuning method using scalar rewards from a pre-trained reward model. Unlike language models, diffusion models lack a tractable likelihood for complete samples, making traditional RL-based optimization less straightforward to apply. ReFL circumvents this by leveraging the insight that partially denoised samples at latter timesteps already exhibit distinguishable reward scores. During training, ReFL randomly selects a late denoising step t, predicts the corresponding image x′0, and computes a scalar reward r(c,x′0) from the reward model. The gradient is then backpropagated through the denoising step using a truncated reward:

Lreward = ReLU(r(c,xˆ0)) By applying feedback directly on generation trajectories, ReFL enables preference-driven finetuning of diffusion models in a scalable and model-agnostic manner.

- 4 METHODOLOGY

- 4.1 OVERVIEW

As show in Fig. 3, we introduce a unified framework for multi-task learning, leveraging reward model r trained with human preferences data to fine-tunes the policy model πθ across a range of downstream tasks. We assume that the complete dataset D = {Dk}Kk=1 can be partitioned into K subsets. For each task, Dk = {(xi,ci,sk,Ek)}Ni=1 share a common format for the image xi ∈ RH×W×3 and corresponding input condition ci. The subset Dk is characterized by a unique identifier sk and a set of evaluation metrics Ek. Note that Dk is associated with a distinct set of evaluation criteria, i.e., Ek may vary significantly across tasks. Given current task ids sk and a certain evaluation metric e ∈ Ek, the reward model r will be used to calculate the probability how the evaluation image xθ is better than reference image xref, which are generated by policy model πθ and reference model πref, respectively. In the process of our RL pipeline, The objective of our training scheme is to increase the probability of generated reward-aligned tokens, as determined by the reward model r. OneReward enables the model to efficiently learn a versatile, multi-task generation policy that satisfies multidimensional human preferences within a single, unified process.

This section is structured as follows: we first detail our data construction pipeline in Section 4.2. Next, in Section 4.3, we describe the training procedure for our reward model, termed OneReward. Finally, we present the multi-task, multi-criteria reinforcement learning strategy in Section 4.4.

[Figure 43]

- Figure 3: Overall pipeline of our unified RL procedure. We first random sample image and conditions from different task with a certain probability. Start with same condition and different init noise, the reference image is fully denoised using the reference model, denoted as πref(·). While the evaluation image is partially denoised with randomly selected step and directly predict x′0 based on the policy model, denoted as πθ(·). The reward model guides learning by encouraging the policy model to achieve superior performance to the reference model across all evaluation dimensions and tasks.

- 4.2 HUMAN PREFERENCE DATA COLLECTION

To support the development and evaluation of our unified image editing framework, we constructed a large-scale, high-quality human preference dataset for multi-task image generation. This dataset spans four major editing tasks: image fill, image extend, object removal, and text rendering, each presenting unique challenges and requiring distinct evaluation perspectives.

In image fill and image extend, the model is required to synthesize plausible content in user-specified regions, guided by natural language prompts. These prompts typically describe the content to be generated, along with contextual details to ensure stylistic and semantic consistency with the surrounding image. In contrast, the object removal task centers on the elimination of specified elements from the input image. Its objective is not to insert new content, but rather to achieve visually seamless completion of the masked area. Since removal does not involve user-defined content, there is no need to provide a unique prompt as condition for each sample. To ensure input format consistency across all tasks, we apply a fixed, generic prompt (e.g., “remove the specified object”) to all object removal instances. Each sample is structured as a triplet (Isrc,M,P), where Isrc ∈ RH×W×3 is the source image, M ∈ RH×W is the binary mask indicating the region to be edited, and P is the corresponding text prompt. This unified input representation enables consistent processing across different tasks within our framework. In the data collection pipeline, we employ a pre-trained diffusion model as the base generator and produce a set of candidate images for each sample by varying key inference parameters, thereby promoting diversity in output quality. This parameter randomization introduces controlled variability across the candidates, enabling the learning of nuanced human preferences. Specifically, we randomly sample the following parameter in the diffusion inference pipeline:

- • Inference Steps: Uniformly sampled from different denoise steps, controlling the quality of generation.
- • Negative prompt: Including optional descriptors like “nsfw”, “blurry” or “low quality”, which guide the generator away from undesired content.
- • Classifier-Free Guidance Scale: Sampled from a range of values, controlling the trade-off between prompt adherence and creativity.

[Figure 44]

- Figure 4: Illustration of the pairwise annotation process. Given multiple candidate outputs for the same prompt and binary mask, annotators identify the best and worst samples under each evaluation dimension to form a winner/loser pair. If the differences between candidates are negligible, the dimension is discarded (denoted by ∅), ensuring that only informative comparisons are retained. To clarify, this showcase uses an all-one mask, meaning the entire image region is generated.

• Initial Noise Strength: Sampled from a predefined range to simulate varying levels of denoising starting point.

To meet the needs of multi-tasking and multi-dimensional evaluation, we design a task-specific annotation protocol:

- • Structure: Measures whether the generated content maintains spatial and geometric coherence with the original image layout, such as object contours, perspective, and scene topology.
- • Consistency: Evaluate how smoothly the edited region integrates with its surroundings in terms of color, texture, and illumination, ensuring visual seamlessness.
- • Text Alignment: Assesses how accurately the generated image content aligns with the semantics of the input prompt, especially in terms of object identity, attributes, and positioning.
- • Aesthetic: Reflects the overall visual appeal of the result, including composition, realism, and fidelity to high-quality image standards.
- • Removal Quality: Judging the effectiveness of object removal, focusing on whether the target is completely eliminated and whether the area is cleanly filled without visible artifacts or unintended content.

For image fill and image extend, each set of candidate images is evaluated along the first four dimensions. And for object removal, where no user text is present, we reduce the evaluation to the final single criterion. As shown in Fig.4, annotators are presented with the input triplet (Isrc,M,P), and N generated candidate images. For each evaluation dimension, they are instructed to perform a Bestof-N and Worst-of-N selection, identifying the most and least preferred outputs. This process yields a winner/loser pair for each dimension of one sub-task, enabling fine-grained and dimension-specific preference supervision. Since judgments are made independently across multiple dimensions, the same candidate may be selected as a winner in one dimension and a loser in another. For instance, a particular output may exhibit stronger semantic alignment with the prompt than another candidate, while simultaneously being less aesthetically appealing. The design retains disagreements across evaluation dimensions, instead of forcing them into a fixed but potentially misleading label.

By collecting preference pairs at the metric level, rather than through holistic or averaged scoring, our annotation strategy overcomes a major limitation of prior work by enabling the disentanglement of conflicting judgments across distinct evaluation dimensions. This comparative and multidimensional evaluation scheme promote high-quality and robust human perference data, which serves as a strong foundation for training reward models in multi-objective image generation tasks.

[Figure 45]

- Figure 5: The detail of our one reward model. We utilize VLM to judge whether the first image is better than the second one. In the process of reward feedback learning, the probability of y+ token is treated as the reward to the diffusion models. We simplely add the edit task and the evaluation dimensions to the user query, achieving the goal of training for different task and dimensions. The content of angle brackets is optional, only add when the evaluation dimension is Text Alignment.

- 4.3 MULTI-DIMENSIONAL PAIRWISE REWARD MODEL TRAINING

Designing a multi-task learning framework for image editing presents significant challenges, particularly in constructing a reward mechanism that is both scalable across heterogeneous tasks and robust against reward hacking. A naive solution, training separate reward models for each evaluation dimension within each task, is computationally expensive and difficult to tune, requiring considerable resources for both training and optimization. Conventional scalar-based reward models are particularly inadequate for mask-guided generation tasks, as their assessments are easily dominated by unchanged background regions. As a result, they fail to capture the true quality of the edited content within the masked area, producing reward signals that are misaligned with the actual effectiveness of the intended edits.

To address these limitations, we propose OneReward, a unified reward model designed to assess outputs across multiple image editing tasks and evaluation dimensions with only one reward model. OneReward is built upon the perceptual capabilities of a pre-trained VLM, which serves as the backbone for all-dimension reward prediction. Rather than introducing task-specific output heads, we guide the model through a textual evaluation query q , which encodes both the task identifier sk and the evaluation dimension e ∈ Ek. This formulation enables dynamic conditioning of the VLM on the specific aspect of quality to be evaluated. Formally, the evaluation query is constructed as:

### q = Φ(sk,e,P) (1)

where Φ is a instruction template as shown in Fig. 5, P is the original input prompt, used only when evaluating prompt–image alignment. For all other dimensions, such as aesthetics or structural, the query remains prompt-free, ensuring that judgments reflect intrinsic visual quality without semantic bias.

To further enhance robustness and address the shortcomings of absolute scoring, OneReward employs a comparative evaluation scheme. It receives a pair of generated images and is asked to identify the preferred one, based on relative quality. As illustrated in Fig. 5, the reward model takes a pair of images (xw,xl) along with an evaluation query q as input, and outputs a binary classification that aligns with the human preference for that specific query. It is defined as:

### y = r(xw,xl,q) (2)

Our training methodology derives directly from the Best-of-N and Worst-of-N annotation protocol in Sec. 4.2. Specifically, for each set of four candidate images, annotators independently select the best image xw and the worst one xl for each evaluation dimension, forming pairs of human preference for training. This process allows each preference pair to be labeled with one or more evaluation dimensions, resulting in a multi-label learning scenario. For instance, in Fig. 4, the fourth image excels in text alignment, but underperforms in consistency. As a result, it is labeled as the winner for text alignment and the loser for consistency. For each training sample, we construct

##### Accuracy(%) Text Alignment Consistency Structure Aesthetics Removal Quality

Image Fill 83.77 74.23 74.63 72.10 – Image Extend 80.36 72.50 74.29 71.10 – Object Removal – – – – 84.93

Table 1: Accuracy of the reward model across multiple editing tasks and evaluation dimensions. Each entry denotes the model’s accuracy in distinguishing winners from losers on the test set for a given dimension (e.g., Text Alignment, Consistency, Structure, Aesthetics, Removal Quality). Higher values indicate more reliable preference discrimination by the reward model.

the corresponding evaluation queries q for all the annotated dimensions and compute the standard cross-entropy loss. The loss function is defined as follows:

- 1

- 2

L(ϕ) = −

E(xw,xl,q)∈D log Pϕ y+ | xw, xl, q + log Pϕ y− | xl, xw, q (3)

where y+ denotes affirmative token (e.g. “Yes”), y− denotes negative token (e.g. “No”), and Pϕ represents the probability assigned to the corresponding token by the reward model. xw and xl refer to the winner and loser pair under a specific evaluation criterion. Note that this formulation is noncommutative, as the query is designed to expect the response y+ when the image in the first position is of higher quality. Otherwise, the expected response is y−. This training strategy enables the efficient utilization of our annotations, empowering the reward model with the capacity to perform nuanced, multidimensional evaluations across a range of tasks.

- Table 1 reports the accuracy of our reward model in different sub-task and evaluation dimensions. The results show that text alignment achieves the highest accuracy, exceeding 80% in both image fill and image extend. Other dimensions such as consistency, structure, and aesthetics yield moderate but stable accuracies, mostly in the low–mid 70% range. For the object removal task, the model reaches 84.93% in removal quality, indicating that it is particularly effective at capturing human preferences in this setting. Since the underlying VLM is pre-trained with multimodal supervision, it is naturally more sensitive to text–image correspondence, making text alignment easier to discriminate, whereas dimensions related to intrinsic visual quality remain more challenging. Overall, these results suggest that the model generalizes well across tasks and metrics.

- 4.4 MULTI-TASK REWARD FEEDBACK LEARNING

Based on OneReward, we propose a novel framework that systematically aligns a pre-trained diffusion model with complex, multi-dimensional human preferences across a suite of tasks. As shown in Fig. 3, the whole train pipeline is composed of three main components: a frozen reference model πref, a trainable policy model πθ, and the trained OneReward model r. In each iteration, we first sample data from the sub-dataset Dk within the full dataset {Dk}Kk=1, according to a predefined sampling probability pk. To promote learning in more difficult tasks, we assign higher sampling probabilities to the task that are estimated to be more difficult based on prior knowledge. The reference model πref is initialized by the parameters of the pre-trained diffusion model. It generates a baseline image xref via a full denoising trajectory from gaussian noise xT to the final output x0, representing the model’s initial generative capacity. Inspired by ReFL (Xu et al. (2023)), the evaluate image xθ start from the same condition c but with a different noise vector, performing partial denoising up to a randomly selected intermediate timestep t before directly predicting the final latent x′0. The vae decoder, denote as vae decode(·), is used to reconstructs pixel-space images from these latents and gradients are backpropagated only through this final single-step prediction, making the training process efficient. Both the prediction of the evaluate xθ and the reference image xref, along with an evaluation query q, are passed into the reward model. The reward model evaluates whether xθ is preferred over xref under the given task and evaluation dimension, and returns a binary output token y+ and y−:

Image Fill: Consistency

Image Fill: Structure

Image Fill: Text Alignment

0.9

0.9

0.9

0.8

0.8

0.7

0.8

0.7

0.6

RewardValue

RewardValue

RewardValue

0.6

0.7

0.5

0.5

0.6

0.4

0.4

0.3

0.3

0.5

0.2

0.2

0.4

0.1

0 1000 2000 3000 4000 5000 6000 7000 8000 Iteration

0 1000 2000 3000 4000 5000 6000 7000 8000 Iteration

0 1000 2000 3000 4000 5000 6000 7000 8000 Iteration

Image Fill: Aesthetic

Image Extend: Consistency

Image Extend: Structure

0.90

0.9

0.9

0.85

0.8

0.8

0.80

0.7

0.7

RewardValue

RewardValue

RewardValue

0.75

0.6

0.6

0.70

0.5

0.5

0.4

0.65

0.4

0.3

0.60

0.2

0.3

0.55

0 1000 2000 3000 4000 5000 6000 7000 8000 Iteration

0 1000 2000 3000 4000 5000 6000 7000 8000 Iteration

0 1000 2000 3000 4000 5000 6000 7000 8000 Iteration

Image Extend: Text Alignment

Image Extend: Aesthetic

Object Removal: Removal Quality

0.9

0.9

0.8

0.8

0.8

0.7

0.7

0.7

RewardValue

RewardValue

RewardValue

0.6

0.6

0.6

0.5

0.5

0.5

0.4

0.4

0.4

0.3

0.3

0.3

0 1000 2000 3000 4000 5000 6000 7000 8000 Iteration

0 1000 2000 3000 4000 5000 6000 7000 8000 Iteration

0 1000 2000 3000 4000 5000 6000 7000 8000 Iteration

- Figure 6: We visualize the reward curves of Consistency, Structure, Text Alignment, Aesthetics for image fill(blue) and image extend(green), Removal Quality for object removal(orange).

### xθ = vae decode(x′0)

xref = vae decode(x0)

### y = r(xθ,xref,q)

(4)

As xθ and xref are generated from the same condition c and different model parameters, we simplify the rollout procedure as πθ(·) and πref(·), respectively. In the process of RL, The probability to response y+ is treated as the reward signal. And the objective of the policy model is to maximize this expected reward, the loss function is defined as follows.

### J(θ) = max 0,λ − Pϕ y+ | πθ(c), πref(c), q (5)

where y+ denotes the affirmative token defined in Sec. 4.3 and Pϕ is the probability predicted by the reward model parameterized by ϕ, λ is the predefined reward upper bound to avoid hacking, the query q contains the information of task ids and its task-specific evaluation dimensions, as shown in Fig. 5. Our training objective for the policy model involves simultaneously optimizing for all relevant evaluation metrics of a given task. At each step, dimension-wise rewards are computed and averaged to guide optimization, effectively casting the training process as a form of multi-objective reinforcement learning. This encourages the model to achieve balanced improvements across diverse evaluation criteria, rather than overfitting to any single aspect of generation.A detailed description of the multi-task reinforcement learning process can be found in Alg. 1.

Fig. 6 illustrates the reward curves across different evaluation dimensions during training, covering three tasks: image fill (blue), image extend (green), and object removal (orange). Each subplot shows the evolution of reward values over training iterations for specific task–dimension pairs, including Consistency, Structure, Text Alignment, Aesthetics, and Removal Quality. Compared with

- Algorithm 1 Multi-Task Reinforcement Learning from Human Feedback

Dataset: Multi-Task image-condition datasets {Dk}Kk=1, with data sample probability distribution P = {p1,p2,...,pK}, task ids S = {s1,s2,...,sK} and each evaluation dimension {Ek}Kk=1. Input: Reference diffusion model πref, policy model πθ with parameters θ, unified reward model

r with parameters ϕ, hyperparameters [t1,t2] for the generation of evaluate image.

- 1: Init reference model πref ← πθ
- 2: Init ema model πema ← πθ
- 3: for iteration = 1, ..., N do
- 4: Sample condition c from the k-th dataset Dkwith probability pk
- 5: Sample init noise ϵ1,ϵ2 from normal distribute N(0,1)
- 6: Random sample denoise timesteps t from [t1,t2]
- 7: Generate the reference image xref with full denoise procedure πref(ϵ1,c)
- 8: Generate the evaluate image xθ with random denoise steps πθ(ϵ2,c,t)
- 9: for e ∈ Ek do
- 10: Generate query q with task id sk and current evalution dimension e as shown in Fig.5
- 11: Compute RL loss Je(xθ,xref,q) in Equation5 with reward model r
- 12: end for
- 13: Updata policy model via gradient ascent:πθ ← πθ + |E1

k|∇πθ

e∈Ek

Je

- 14: EMA update πema ← τπema + (1 − τ)πθ
- 15: end for Output: πθ,πema

single-task settings, multi-task training introduces moderate fluctuations in the reward trajectories. Nevertheless, all curves display a clear upward trend, indicating consistent performance gains across dimensions. These results suggest that while multi-task optimization inherently introduces greater variability due to shared capacity and competing objectives, the model still converges reliably with steady improvements. The effectiveness of our reward-driven training scheme is further validated by the high final reward values achieved across all dimensions.

- 5 EXPERIMENTS

- 5.1 IMPLEMENTATION DETAILS

Dataset. Our dataset for multi-task image editing was constructed through a multi-stage pipeline. We began by extracting image embeddings using the CLIP model(Radford et al. (2021)), followed by K-means clustering to obtain a diverse and representative subset for reinforcement learning. Based on the selected subset, we further generated and annotated a large number of human preference pairs, which serve as training data for both the reward model and the image generation model.

Experimental Settings. For our reward model, we fine-tuned the open-source Qwen2.5-VL-7BInstruct( Bai et al. (2025)) on our preference dataset with batch size 16, learning rate 1e-6. For Seedream 3.0 Fill, We adopt Seeddream 3.0( Gao et al. (2025a)) as the text-to-image base model, data were sampled with probabilities of 50% for image fill, 25% for image extend, and 25% for object removal. Training was performed with batch size 8, learning rate 1e-5.

- 5.2 COMPARISONS WITH STATE OF THE ARTS

- 5.2.1 EVALUATION SETTINGS

To comprehensively evaluate the effectiveness of OneReward, we conducted extensive comparisons with SOTA models and APIs, including Ideogram, Higgsfield, Adobe Photoshop, Midjourney, and Flux Fill [Pro]. For evaluation, we constructed a diverse benchmark consisting of 130 images for image fill, 100 for object removal, and 200 for image extend (split evenly between text-guided and text-free). It covers a wide range of scenes, such as portraits, landscapes, pets, and typography, as well as artistic styles such as photorealistic, anime, watercolor, and AI-generated.

###### Task Model Usability Rate(%) Text Alignment Texture Consistency Style Consistency Structure Aesthetics Text Rendering(%) Removal Quality(%)

Adobe Photoshop 45.22 4.09 4.05 3.83 2.89 2.16 26.69 – Ideogram 50.65 3.78 4.25 4.13 3.80 2.65 24.81 – Flux Fill [pro] 50.97 4.12 4.16 3.72 3.46 2.47 29.32 – Higgsfield 52.11 4.43 4.29 3.63 3.54 2.40 45.49 – SeedDream 3.0 Fill 69.04 4.57 4.33 3.76 4.02 2.91 70.68 –

Image Fill

Adobe Photoshop 44.07 4.21 4.04 4.08 2.79 2.27 – – Flux Fill [pro] 44.26 4.23 3.77 3.79 3.57 2.36 – – Ideogram 63.13 4.44 3.63 4.02 3.80 2.61 – – SeedDream 3.0 Fill 64.72 4.26 4.19 3.60 4.05 2.89 – –

Image Extend w Prompt

Adobe Photoshop 61.10 – 3.98 4.49 2.93 2.55 – – Midjourney 69.59 – 4.24 4.33 3.47 2.95 – – Flux Fill [pro] 70.47 – 4.10 4.37 3.68 2.84 – – Ideogram 73.71 – 3.99 4.40 3.86 2.92 – – SeedDream 3.0 Fill 87.54 – 4.36 4.02 4.19 3.29 – –

Image Extend w/o Prompt

Flux Fill [pro] 15.92 – 3.97 – 3.39 – – 18.71 Ideogram 70.14 – 4.04 – 4.07 – – 72.00 Adobe Photoshop 73.98 – 4.03 – 3.68 – – 83.16 SeedDream 3.0 Fill 82.22 – 3.87 – 4.32 – – 86.33

Object Removal

- Table 2: Quantitative comparison of our model against SOTA competitors across four editing tasks. Metrics with percentages (e.g., Usability Rate, Text Rendering, Removal Quality) are reported as success rates, while other dimensions (e.g., Text Alignment, Texture Consistency, Style Consistency, Structure, Aesthetics) are Mean Opinion Scores (MOS) rated on a 1–5 scale. Higher scores indicate better performance in all dimensions. Our model demonstrates consistently superior performance across most dimensions compared with existing SOTA models, especially in overall usability rate.

| |32.3%<br><br>68.0%<br><br>54.2%<br><br>66.1%<br><br>Seedream 3.0 Fill [OneReward]<br><br>| | |48.3%<br><br>23.6%<br><br>38.4%<br><br>27.9%<br><br>19.3%<br><br>8.4%<br><br>7.4%<br><br>6.1%<br><br>Seedream 3.0 Fill [Base]<br><br>Good<br><br>Same<br><br>Bad| | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Image Fill

Image Extend w Prompt

Image Extend w/o Prompt

Object Removal

0 20 40 60 80 100 Percentage

- Figure 7: Comparison of performance between Seedream 3.0 [OneReward] and Seedream 3.0 [Base] using Good–Same–Bad (GSB) evaluation. Each group corresponds to a specific image editing task. Bars represent the relative proportions of outputs judged as Good (orange), Same (green), or Bad (blue) across different model pairs. This visualization highlights the distribution of relative preferences, showing where OneReward-enhanced models outperform the base model.

- 5.2.2 HUMAN EVALUATION

To comprehensively evaluate the performance of our model, we conducted a user study involving 40 participants. Each participant rated the generated images across multiple dimensions: overall quality, text alignment, texture consistency, style consistency, structural plausibility, aesthetics, text rendering, and removal quality. Among these, overall usability, text rendering, and removal quality were treated as binary judgments, where each image was assessed as either acceptable or not. The reported values for these metrics therefore represent success rates expressed as percentages. In contrast, the remaining dimensions were rated on a 1–5 Likert scale, and the scores were averaged to produce Mean Opinion Scores (MOS), where higher values indicate better quality.

Comparative results for the four image editing tasks are presented in Fig. 1 and Tab. 2. SeedDream 3.0 Fill demonstrates the strongest overall performance across all tasks. For image fill, our model achieves a usability rate of 69.04%, outperforming the second-best competitor (52.11%) by 16.93 percentage points. It also obtains the highest scores in most dimensions, including text alignment, texture consistency, structure, aesthetics, and text rendering, with the only exception being style consistency, where Ideogram shows a slight advantage. On the text-guided image extend task, SeedDream 3.0 Fill performs comparably to Ideogram while clearly surpassing Flux Fill [pro] and Adobe

Image Fill Image Extend Text Rendering Object Removal

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Source

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Seedream 3.0 Fill

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Ideogram

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Flux Fill [pro]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

PhotoShop

A black iphone Goodfish Word "Black Forest Labs", next line word "The diary"

tiger Butterfly clothes with wings

Change the text to "Intense Summer"

- Figure 8: Visual comparison of editing results for Seedream 3.0 Fill and its competitors across different tasks. Rows correspond to different methods, columns show task-specific prompts and outputs. The source images are shown in the first row. The blank row at the bottom indicates that the case is prompt-free.

Photoshop in usability. And in the text-free setting, SeedDream 3.0 Fill shows pronounced superiority, achieving the highest usability rate (87.54%) and leading across all reported dimensions. For object removal, our model again delivers the best results, with a usability rate of 82.22% and a removal quality score of 86.33%, significantly outperforming other SOTA competitors. The high removal quality indicates that our model produces the fewest unwanted objects in this task, behavior that typically conflicts with goals in other generation tasks such as image fill or extend. This demonstrates the effectiveness of our RL strategy in multi-task human preference learning.

To further assess the impact of OneReward, we conduct a Good–Same–Bad (GSB) evaluation comparing SeedDream 3.0 Fill models trained with and without reward guidance. As shown in Fig. 7, each bar represents the distribution of human preferences across different tasks. Compared to the base model, the OneReward variant receives a higher proportion of “Good” ratings in all tasks. The GSB results demonstrate that our unified reward model generally shifts model outputs toward preferred generations.

Notably, all of the above results were achieved using a unified generation model, trained with only one reward model shared across different task and evaluation dimension, without any task-specific SFT.

- 5.3 DYNAMIC REINFORCEMENT LEARNING

As shown in Alg. 1, our training pipeline typically maintain three models in parallel: a policy model, a reference model, and an EMA variant. This design leads to high memory consumption and increases the engineering complexity of model synchronization. On the other hand, if the reference images are of insufficient quality, the policy model may be trained on overly easy preference pairs, potentially leading to reward hacking and hindering effective learning.

To address these limitations, we propose a dynamic reinforcement learning strategy (Alg. 2), in which the EMA model is directly reused as the reference model. As training progresses, the EMA model gradually improves in generative quality, thereby providing an increasingly strong baseline

Image Fill Image Extend Text Rendering Object Removal

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Source

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Flux Fill [dev]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Flux Fill [pro]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Flux Fill [dev] [OneReward]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Flux Fill [dev] [OneReward Dynamic]

A pile of deep red roses, with many rose petals scattered in the background

a teal colored Ice and snow a sign that says "different" a sign that says "Coffee" couch

Blond caucasian boy wearing hat

- Figure 9: Visual comparison of editing results for Flux Fill and our RL model across different tasks. Rows correspond to different methods, columns show task-specific prompts and outputs. The source images are shown in the first row. The last two rows stands for our RL-enhanced model, trained via Alg. 1 and Alg. 2.

for policy comparison. This design not only reduces memory overhead by eliminating the need for a separate reference model, but also yields more stable and adaptive reward signals throughout training. A conceptual visualization of the reward computation process is provided in Figure 10, highlighting the difference between the baseline setup and this dynamic design. In the dynamic variant, the reference model is continuously enhanced during training, ensuring that the policy is always compared against a strong and progressively improving baseline. Beyond its simplicity and efficiency, the dynamic framework demonstrates highly competitive performance, as shown in the last row of Fig. 9.

- 5.4 QUALITATIVE RESULTS

Fig. 8 presents visual comparisons in different situations. We compare Seedream 3.0 Fill with Ideogram, Flux Fill [pro], and Adobe Photoshop using the same input conditions. Based on Flux Fill [dev], we applied both Alg. 1 and Alg. 2 for RL training. Fig. 9 shows a comparison of the resulting models with Flux Fill [dev] and Flux Fill [pro]. It shows that our RL-enhanced model, trained via OneReward, demonstrates superior visual quality compared to the base model and even the closedsource API. We will release the both trained models as part of our open-source contribution.

- 6 CONCLUSIONS

We present OneReward, a unified reward model designed for multi-task, multi-dimensional reinforcement learning of diffusion and flow mathching model. It enables fine-grained supervision across diverse tasks by leveraging VLM as the reward model. Built on OneReward, our develop Seedream 3.0 Fill, which achieves SOTA performance on image fill, image extend, object removal, and text rendering, outperforming both commercial APIs and open-source models on most dimensions. While style consistency remains a relative weakness, we leave it for next optimization through detailed data annotation and post-training refinement. To further support the community, we opensource FLUX Fill [dev][OneReward], an enhanced variant of flux model trained with our RL framework, offering stronger generalization and broader task applicability.

REFERENCES

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025a.

Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025b.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-

training for unified vision-language understanding and generation. In International conference on machine learning, pp. 12888–12900. PMLR, 2022.

Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet++: Improving conditional controls with efficient consistency feedback: Project page: liming-ai. github. io/controlnet plus plus. In European Conference on Computer Vision, pp. 129–147. Springer, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025a.

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025b.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11461–11471, 2022.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Yuxi Ren, Jie Wu, Yanzuo Lu, Huafeng Kuang, Xin Xia, Xionghui Wang, Qianqian Wang, Yixing Zhu, Pan Xie, Shiyin Wang, et al. Byteedit: Boost, comply and accelerate generative image editing. In European Conference on Computer Vision, pp. 184–200. Springer, 2024.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020a.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.

Hongzu Su, Lichao Meng, Lei Zhu, Ke Lu, and Jingjing Li. Ddpo: Direct dual propensity optimization for post-click conversion rate estimation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1179–1188, 2024.

Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2149–2159, 2022.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Ahmet Burak Yildirim, Vedat Baday, Erkut Erdem, Aykut Erdem, and Aysegul Dundar. Inst-inpaint: Instructing to remove objects with diffusion models. arXiv preprint arXiv:2304.03246, 2023.

Jiacheng Zhang, Jie Wu, Weifeng Chen, Yatai Ji, Xuefeng Xiao, Weilin Huang, and Kai Han. Onlinevpo: Align video diffusion model with online video-centric preference optimization. arXiv preprint arXiv:2412.15159, 2024.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.

## A APPENDIX

###### Reward Computation: Default vs Dynamic

Policy (Default)

14

Reference (Default:Fixed)

Policy (Dynamic)

Reference (Dynamic:EMA)

12

10

ModelCapability

8

6

Dynamic reward

4

Default reward

2

0

0 20 40 60 80 100 Training Steps

- Figure 10: Schematic illustration of reward computation in the baseline and our dynamic framework. In the baseline 1, rewards are measured as the vertical gap between the policy (blue) and a fixed reference (red). In the dynamic method 2, rewards are computed against an EMA-updated reference (orange) that evolves smoothly with the policy (green), forming a dynamic baseline. This figure is for conceptual understanding only and does not reflect actual parameter values or training dynamics.

- Algorithm 2 Dynamic Multi-Task Reinforcement Learning from Human Feedback

Dataset: Multi-Task image-condition datasets {Dk}Kk=1, with data sample probability distribution P = {p1,p2,...,pK}, task ids S = {s1,s2,...,sK} and each evaluation dimension {Ek}Kk=1. Input: Reference diffusion model πref, policy model πθ with parameters θ, unified reward model

r with parameters ϕ, hyperparameters [t1,t2] for the generation of evaluate image.

- 1: Init reference model πref ← πθ
- 2: for iteration = 1, ..., N do
- 3: Sample condition c from the k-th dataset Dkwith probability pk
- 4: Sample init noise ϵ1,ϵ2 from normal distribute N(0,1)
- 5: Random sample denoise timesteps t from [t1,t2]
- 6: Generate the reference image xref with full denoise procedure πref(ϵ1,c)
- 7: Generate the evaluate image xθ with random denoise steps πθ(ϵ2,c,t)
- 8: for e ∈ Ek do
- 9: Generate query q with task id sk and current evalution dimension e as shown in Fig. 5
- 10: Compute RL loss Je(xθ,xref,q) in Equation 5 with reward model r
- 11: end for
- 12: Updata policy model via gradient ascent:πθ ← πθ + |E1

k|∇πθ

e∈Ek

Je

- 13: ▷ EMA update πref ← τπref + (1 − τ)πθ
- 14: end for Output: πθ,πref

