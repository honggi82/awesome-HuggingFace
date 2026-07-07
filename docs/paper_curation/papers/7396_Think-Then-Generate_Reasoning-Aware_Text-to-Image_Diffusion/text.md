# arXiv:2601.10332v1[cs.CV]15Jan2026

## Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders

Siqi Kou1∗, Jiachun Jin1∗, Zetong Zhou1∗, Ye Ma2, Yugang Wang1, Quan Chen2, Peng Jiang2, Xiao Yang3, Jun Zhu3, Kai Yu1, Zhijie Deng1† 1Shanghai Jiao Tong University 2Kuaishou Technology 3Tsinghua University https://github.com/zhijie-group/think-then-generate

#### Abstract

Recent progress in text-to-image (T2I) diffusion models (DMs) has enabled high-quality visual synthesis from diverse textual prompts. Yet, most existing T2I DMs, even those equipped with large language model (LLM) based text encoders, remain text–pixel mappers—they employ LLMs merely as text encoders, without leveraging their inherent reasoning capabilities to infer what should be visually depicted given the textual prompt. To move beyond such literal generation, we propose the think-then-generate (T2G) paradigm, where the LLM-based text encoder is encouraged to reason about and rewrite raw user prompts; the states of the rewritten prompts then serve as diffusion conditioning. To achieve this, we first activate the think-thenrewrite pattern of the LLM encoder with a lightweight supervised fine-tuning process. Subsequently, the LLM encoder and diffusion backbone are co-optimized to ensure faithful reasoning about the context and accurate rendering of the semantics via Dual-GRPO. In particular, the text encoder is reinforced using image-grounded rewards to infer and recall world knowledge, while the diffusion backbone is pushed to produce semantically consistent and visually coherent images. Experiments show substantial improvements in factual consistency, semantic alignment, and visual realism across reasoning-based image generation and editing benchmarks, achieving 0.79 on WISE score, nearly on par with GPT-4o. Our results constitute a promising step toward next-generation unified models with reasoning, expression, and demonstration capacities.

#### 1. Introduction

Text-to-image (T2I) diffusion models (DMs) have demonstrated remarkable capabilities in generating high-fidelity

∗Equal contribution. Work done during Jiachun and Siqi’s internships at Kuaishou Technology.

†Corresponding author.

[Figure 1]

Figure 1. An overview of our think-then-generate method. Beyond using LLM as a frozen text encoder, we train it to think and refine the raw user prompts guided by the reward of output images.

and diverse images given textual prompts, as exemplified by Stable Diffusion [11, 27] and the FLUX series [2, 19, 20]. To capture richer semantic representations of textual prompts, researchers have progressively adopted stronger text encoders—from early CLIP models [27] to large language models (LLMs) [7, 8] and their variants with visual inputs, i.e., vision language models (VLMs). Representative examples include the open-source Qwen-Image [34], which employs Qwen2.5-VL [1] as the encoder, and advanced commercial systems such as Seedream 4.0 [29], GPT-4o [17], and Gemini-2.5-Flash-Image [14]. The LLM encoder offers two key advantages: (1) native understanding of mixed text–image prompts, and (2) extensive world knowledge that enhances instruction following. These capabilities may alleviate the burden of iteratively crafting and refining prompts, as in prior work [23, 32, 35], potentially enabling more natural interactions for visual generation.

However, in practice, existing models often fail to fully exploit the potential of LLMs for reasoning. Typically, the models are trained on large-scale descriptive image–caption pairs with the LLM encoder frozen [2, 4, 34] (i.e., solely as a feature extractor). Consequently, they can handle only literal and descriptive instructions (e.g., specific object colors or textures) but struggle with conceptual instructions (e.g., illustrating how a machine operates), as shown in Figure 2. I.e., they function as simple text–pixel mappers. Ideally, the LLM encoder should transcend prompt encoding to reason over raw instructions, leverage inherent world knowledge

Qwen-Image Ours FLUX.1 dev GPT 4o Nano Banana Bagel

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

A math teacher stands at the blackboard explaining how to solve the equation 2x − 4 = 10, with all solution steps shown on the board.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

A multi-panel illustration showing the story of marking the boat to find a sword, with clear steps from dropping the sword to carving a mark on the boat.

Figure 2. Comparison of T2I models on conceptual visual generation. Our Qwen-Image under think-then-generate pipeline produces semantically aligned and visually coherent results, correctly interpreting user intent given prompt “Holiday celebrating the birth of Jesus Christ.” (e.g., generating a warm Christmas celebration rather than literally depicting Jesus), whereas vanilla Qwen-Image behaves like a simple text–pixel mapper and often fails to capture conceptual meanings.

for conceptual tasks, and generate semantically enriched prompts for diffusion generation. However, such a functionality cannot be trivially incentivized by current post-training frameworks for DMs [21, 22, 37, 39].

Bagel [10] and Janus-Pro-7B [9]. Notably, this performance is on par with the commercial GPT-4o [17]. Moreover, our method achieves a score of 92.2 on T2I-ReasonBench [31], surpassing the strong closed-source model Gemini-2.0 [13]. For image editing, we apply our method to Qwen-ImageEdit [34], and qualitative results demonstrate its ability to produce more faithful, fine-grained, and instruction-aligned modifications. Moreover, in more challenging and realistic scenarios—such as generating schematic illustrations of human activities (e.g., teaching math)—our trained model demonstrates superior knowledge grounding, visual plausibility, and aesthetic quality, highlighting its potential as next-generation unified models with reasoning, expression, and demonstration capacities in real-world applications.

This paper implements the think-then-generate (T2G) paradigm for reasoning-aware text-to-image diffusion, where the LLM encoder is able to reason over and rewrite raw user prompts, with the embeddings of rewritten prompts fed to the diffusion transformer (DiT) [11, 25] as generation conditioning, as shown in Figure 1. To realize it, we first construct a supervised fine-tuning dataset that enriches raw user prompts with chain-of-thought (CoT) reasoning as well as rewritten prompts. We finetune the LLM on it to acquire the think-then-rewrite pattern. We then co-optimize the LLM encoder and the DiT decoder via a Dual-GRPO strategy, where the rewritten prompts act as the bridge between text reasoning and image synthesis and image-based rewards are maximized in an end-to-end manner. Considering the distinct roles of the encoder and the DiT for image generation, we tailored the reward objective to each component, where the former is optimized for semantic alignment and conceptual understanding and the latter focuses on visual realism and aesthetic quality. This optimization not only activates the world knowledge embedded in LLMs but also adapts DiT to the evolved representation space of the text encoder.

#### 2. Preliminary: Group Relative Policy Optimization

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm popularized in optimizing large generative models [15, 30]. It can be viewed as a variant of policy gradient methods such as PPO [28], with introduced group-wise relative normalization of rewards to better handle high-variance signals during training.

Formally, given a group of rollout trajectories {og}Gg=1 sampled from the current policy πθ, unlike actor–critic methods, GRPO groups samples with similar input prompts or conditions, and then directly calculates relative advantages from normalized group rewards rather than learning

Empirically, on the T2I task, we apply our method to the state-of-the-art open-source model Qwen-Image [34], and observe a score of 0.79 on the WISE benchmark [24], which surpasses the pre-trained Qwen-Image by 30%, and substantially outperforms other open-source models such as

- them through a value model:

Rg − mean {Rg}Gg=1 std {Rg}Gg=1

Aˆg =

, (1)

where Rg is the scalar reward for the trajectory og. This design eliminates the need for a value model, making GRPO more efficient and easier to scale for large models such as LLMs [16] and diffusion/flow matching models [22, 36]. The typical GRPO objective is:

g}Gg=1∼πθold (2)

JGRPO(θ) = E{o

G

1 G

min rg(θ)Aˆg,clip(rg(θ),1 − ϵ,1 + ϵ)Aˆg

g=1

− βDKL πθ ∥ πθ

,

ref

where rg(θ) = ππθ(og)

is the reference policy, and β is the KL divergence regularization parameter.

θold(og), πθ

ref

GRPO for LLMs. For LLMs, the policy πθ(ot | o<t,q) is a language model that generates text tokens ot conditioned on the previous tokens o<t and the user prompt q, this holds a tractable likelihood and is easy to compute. In typical GRPO applications, training relies on outcome-based rewards, where a reward model evaluates only the final generated text. All intermediate tokens in the rollout trajectory are assigned the same reward, which implicitly assumes that each token contributes equally to the final outcome.

Flow-GRPO for Flow Matching Models. Although GRPO has been successfully applied to LLMs, applying it to flow matching models is nontrivial since their ODEbased sampling dynamics do not provide stochasticity to generate diverse trajectories for advantage estimation and policy exploration. Flow-GRPO [22] addresses this limitation by transforming the deterministic Flow-ODE into an equivalent stochastic differential equation (SDE) and discretizing it via the Euler-Maruyama scheme. This yields the following transition kernel:

πθ(xt−1 | xt) = N(xt−1;µθ(xt),gt2∆tI), (3)

here gt = a 1−t t controls the level of stochasticity, ∆t is the discretization step size and µθ(xt) equals to:

xt + vθ(xt,t) +

gt2 2t

(xt + (1 − t)vθ(xt,t)) ∆t. (4)

Crucially, the transition kernels are reduced to tractable Gaussian distributions, which enables the direct application of the GRPO policy update to flow matching models.

#### 3. Method

In this section, we first describe our data curation pipeline for constructing the supervised fine-tuning dataset used to

Figure 3. t-SNE visualization of the embedding distributions before and after SFT. The distributions remain highly consistent, indicating that our SFT procedure preserves the embedding space structure and thus maintains compatibility with the DiT, enabling it to render stable and coherent visual outputs.

train the LLM to replicate the T2G pattern. We then introduce Dual-GRPO, a reinforcement learning strategy that jointly optimizes both the LLM and the DiT using imagebased rewards.

##### 3.1. Reasoning-aware Behavior Activation

Given a raw user prompt set Q = {q}, we aim to enable the model to follow a T2G paradigm. Specifically, for each raw prompt, the model first performs CoT reasoning using its world knowledge to explicitly outline the content to be depicted, followed by summarizing this CoT into a descriptive refined prompt. During image generation, the embedding of the refined prompt serves as the conditioning signal for the DiT. To learn this pattern, we process Q to construct a supervised fine-tuning dataset using Gemini-2.5 [14]. We prompt it to perform CoT reasoning, descriptively inferring what should be depicted and generating a refined prompt. This curated dataset follows the format: raw user prompt → [long CoT] → refined prompt. We then fine-tune the text encoder on this curated dataset.

Qwen2.5-VL serves as both a rewriter and an encoder in our T2G paradigm. While SFT activates its rewriting potential, it remains unclear whether SFT degrades its encoding function: does SFT alter the embedding space in a way that harms the DiT’s ability to generate coherent images? To examine this, we visualize Qwen2.5-VL [1] text embeddings before and after SFT using t-SNE [5]. It reveals distributional shifts by showing whether embeddings overlap or separate in a low-dimensional space. Interestingly, Figure 3 shows that the embeddings totally overlap, indicating that SFT preserves the distribution of the embedding space

and thereby maintains DiT’s capacity to produce stable and reasonable visual outputs.

##### 3.2. Dual-GRPO for LLM-DiT Composite Models

Given q, which may be vague or implicitly specified, conventional DMs employ an LLM text encoder pϕ to encode q into a latent representation pϕ(q), which serves as the conditioning input for the DiT pλ to generate the corresponding image. After our supervised fine-tuning, the LLM encoder first performs CoT reasoning to generate text tokens z and

- then encodes them into a reasoning-aware representation zˆ that serves as conditions for the DiT decoding.

To be specific, the DMs can be viewed as a composite policy model parameterized by θ = {ϕ,λ}. Formally, it can be defined as

pϕ(zt | z<t,q) t ≤ ℓ, pλ(xt | xt−1,zˆ) t > ℓ.

πθ(ot | st) =

And the rollout trajectory is represented as

(5)

o = {z1,··· ,zℓ,xℓ+1,··· ,xℓ+m}.

In the above, zi denotes the i-th generated text token by LLM in z, with reasoning budget ℓ. And xi refers to the i-th prediction during the reverse sampling process x with total iterative steps m.

To apply the policy gradient algorithm to the composite model πθ, we can optimize the following objective:

1,···,xℓ+m}∼πθ (6) 1 ℓ + m

Eq∼p(Q),{z

max

θ={ϕ,λ}

ℓ+m

ℓ

R2(xt,xt−1,zˆ) ,

R1(zt,z<t,q) +

t=1

t=ℓ+1

where p(Q) is the distribution of user prompts, R1 and R2 respectively denote the reward functions for the LLM and DiT components, details about the reward functions are discussed in Section 3.3.

To take advantage of the group relative formulation for the estimation of advantages, we first sample a collection of outputs from the previous policy πθ

. Specifically, as illustrated in Figure 4, given a user prompt q, we sample J reasoning sequences {zj}Jj=1 from pϕ

old

, and for each zˆj, we sample K images {xj,k}Kk=1 from pλ

old

. This hierarchical sampling strategy enables us to compute group-relative advantages for both components within one rollout. By incorporating the standard PPO clipping mechanism and KL divergence regularization, we arrive at the Dual-GRPO objective:

old

1,···,xℓ+m}Ji=1×K∼πθold (7) 1 l + m

Eq∼p(Q),{z

max

θ={ϕ,λ}

l

l+m

Lt(ϕ) +

Lt(λ) ,

t=1

t=l+1

where Lt(ϕ) follows the standard GRPO formulation for language models with group size J:

J

1 J

min(rj,t(ϕ),clip(rj,t(ϕ),1 − ϵ,1 + ϵ))Aˆj,t (8)

j=1

− βDKL pϕ(zj,t) ∥ pϕ

(zj,t) ,

ref

where rj,t(ϕ) = ppϕ(zj,t|zj,<t,q)

ϕold(zj,t|zj,<t,q), the group-relative advantage Aˆj,t for the LLM pϕ is calculated by aggregating the rewards from all K images generated from the same reasoning output zˆj.

Similarly, denote rj,k,t(λ) = ppλ(xj,k,t|xj,k,t−1,zˆj)

λold(xj,k,t|xj,k,t−1,zˆj), the loss term Lt(λ) for the DiT pλ employs a formulation of Flow-GRPO with an extra batch dimension with size J and group size K:

J

K

1 J

1 K

min (rj,k,t(λ), clip(rj,k,t(λ), 1 − ϵ, 1 + ϵ)) Aˆj,k,t

j=1

k=1

(9)

− βDKL pλ(xj,k,t | xj,k,t−1, zˆj) ∥ pλref (xj,k,t | xj,k,t−1, zˆj) .

##### 3.3. The Reward Function and Scheduler

When GRPO is applied to language models or flow matching models, it is common to use an outcome-based reward that depends only on the final prediction, and all intermediate states within the rollout trajectory share the same reward. Such a design implicitly assumes that each step contributes equally to the final outcome, overlooking the fact that different stages of the trajectory influence distinct aspects of the final result.

Taking into account that the composite model is a twostage model, we can design stage-specific reward functions that align with the distinct characteristics of each stage.

During the LLM reasoning stage, most of the semantic content of the generated image is determined within the reasoning output zˆ. And the reward for each reasoning step is the averaged semantic consistency score Rsem over all K generated images sampled from pλ

(x | zˆ):

old

K

1 K

Rsem(xj,k,q). (10)

R1(zj,t,zj,<t,q) = β1(τ)

k=1

In the above, β1(τ) is the scheduler for reward weighting, which is a function of the current training step τ. With this scheduler, we can assign different emphases to the rewards of the LLM and the diffusion model at different stages of training. The corresponding advantage is calculated as

R1(zj,t,zj,<t,q) − mean {R1(zj,t,zj,<t,q)}Jj=1 std {R1(zj,t,zj,<t,q)}Jj=1

Aˆj,t =

.

(11)

(a)

(b)

[Figure 14]

[Figure 15]

###### =  

Alignment Aesthetic

[Figure 16]

[Figure 17]

###### = ퟐ

[Figure 18]

Alignment

- 11

rewards

- 11

- 12

- 13

advantages for DiT

groupcomputation

- 12

<think> This implies a need for specific knowledge about Einstein’ s ...... <\think>

 

Aesthetic

- 11

- 12

- 13

| | |
|---|---|
| | |
| | |
| | |

  

[Figure 19]

Revised Prompt: A classic wooden violin, rich brown ……

 

 ퟐ

iteration = 0

[Figure 20]

[Figure 21]

[Figure 22]

- 1 13

- 2

- 1

- 2

###### groupcomputation

Generate Einstein’s favorite instrument

  

advantages for LLM

rewards

=  

advantages for DiT

rewards

[Figure 23]

- 21

- 22

- 23

- 21

- 22

- 23

<think> While Einstein was known to play the violin, simply stating …… <\think>

groupcomputation

hanging melon Ferris wheel

ퟐ

- 21

- 22

- 23

ퟐ 

[Figure 24]

[Figure 25]

[Figure 26]

Revised Prompt: A beautifully crafted, classic wooden ……

ퟐퟐ

ퟐ

[Figure 27]

ퟐ 

iteration = 240

- Figure 4. Dual-GRPO training trajectories. (a) Tree-structured rollout for a given user prompt q: the LLM encoder samples J reasoning traces, each rewritten prompt conditions the DiT to generate K images. Image-grounded rewards are aggregated to compute group-relative advantages for updating both the LLM and the DiT. (b) Evolution of alignment and style scores during training, demonstrating how DiT training improves both semantic alignment and visual quality over time.

During the diffusion sampling stage, the model needs to render the reasoning output into an aesthetically pleasing and physically consistent image. For a reverse time trajectory xj,k conditioned on a previous reasoning output zˆj, the reward for each step is defined as a weighted sum of the aesthetic score Raes, the physical consistency score Rcon, and the semantic consistency score Rsem of the final generated image [31]:

from MM-DiT [11]. Our post-training pipeline is separated into two stages. For the supervised fine-tuning stage, each model is trained with a learning rate of 5e-6 and a batch size of 32. In the subsequent Dual-GRPO stage, we apply a simple reward scheduler with constant and balanced weights: β1(τ) = β2(τ) = 0.5, where both the LLM and the DiT are updated jointly at every iteration. In Appendix 1, we explore different designs of the scheduler. During the LLM training, we employ a learning rate of 2e-6 with a batch size of 256, generating 5 responses per input. For the DiT training, we adhere to the default configuration of FlowGRPOfast[22], utilizing a learning rate of 3e-4, a batch size of 32, and a clipping range of 1e-4. The generation process involves sampling 16 images for each prompt over 10 inference steps, with an SDE window of 2.

R2(xj,k,t,xj,k,t−1,zˆj) = (12) β2(τ)(ω1Raes(xj,k) + ω2Rcon(xj,k) + ω3Rsem(xj,k)),

ω1,ω2,ω3 are the weights for different components of the reward. Similarly, β2(τ) is the scheduler for the diffusion sampling stage.

#### 4. Experiments

In this section, we evaluate the effectiveness of our T2G paradigm across T2I and image editing tasks. Both quantitative and qualitative results validate the advantages of our approach.

##### 4.1. Implementation Details

We post-train Qwen-Image [34] and Qwen-Image-edit [34] for the T2I and image-editing tasks, respectively. Consequently, the LLM encoder is initialized from Qwen2.5VL [1], and the corresponding DiT backbone is initialized

##### 4.2. Text-to-Image Generation

###### 4.2.1. Supervised Fine-tuning

To activate the T2G pattern of Qwen2.5-VL, we first carefully curate a raw user prompt dataset comprising 7,000 samples. A core criterion for curation is that each prompt requires integration of world knowledge and reasoning to generate a semantically coherent corresponding image. For example, prompts like “Generate an image of traditional food associated with the Dragon Boat Festival”. Then we instruct Gemini-2.5 [14] to perform CoT reasoning using its world knowledge to explicitly outline what content should

- Table 1. Comparison of image generation models on WISE. Numbers in bold indicate the highest score among open-source models. Underlined numbers denote the highest score.

Model Type Cultural Time Space Biology Physics Chemistry Overall ↑

FLUX.1-dev diffusion 0.48 0.58 0.62 0.42 0.51 0.35 0.50 SD-3.5-medium diffusion 0.43 0.50 0.52 0.41 0.53 0.33 0.45 SD-3.5-large diffusion 0.44 0.50 0.58 0.44 0.52 0.31 0.46 Bagel w/CoT unified 0.76 0.69 0.75 0.65 0.75 0.58 0.70 Janus-Pro-7B unified 0.30 0.37 0.49 0.36 0.42 0.26 0.35 HunyuanImage-3.0 unified 0.58 0.57 0.72 0.56 0.68 0.35 0.58 T2I-R1 unified 0.56 0.55 0.63 0.54 0.55 0.30 0.54 Uni-CoT unified 0.76 0.70 0.76 0.73 0.81 0.73 0.75 GPT 4o proprietary 0.81 0.71 0.89 0.83 0.79 0.74 0.80 Qwen-Image diffusion 0.62 0.63 0.78 0.55 0.67 0.35 0.61 Ours (w/o SFT+GRPO) diffusion 0.68 0.58 0.77 0.62 0.76 0.41 0.65 Ours (w/o GRPO) diffusion 0.76 0.66 0.79 0.74 0.84 0.65 0.74 Ours diffusion 0.80 0.74 0.83 0.81 0.85 0.66 0.79

be depicted and summarize it into a descriptive refined prompt. We then fine-tune Qwen2.5-VL on this dataset to replicate the T2G pattern when given new prompts.

###### 4.2.2. Quantitative Results

Benchmarks. We evaluate performance across various T2I benchmarks requiring reasoning capacity. WISE [24] evaluates models with 1000 meticulously crafted prompts across 25 sub-domains in cultural common sense, spatio-temporal understanding, and natural science. T2I-ReasonBench [31] comprises 800 prompts organized into four dimensions: idiom interpretation, textual image design, entity-reasoning, and scientific-reasoning. It introduces a two-stage evaluation framework to quantify performance: an LLM first generates prompt-specific question-criterion pairs evaluating if the image includes the essential elements resulting from correct reasoning, and a multimodal LLM then scores the generated image against these criteria.

Baselines. We compare with 10 state-of-the-art T2I models, including 4 diffusion-based T2I models, 4 unified multimodal models, and 2 proprietary models. The diffusion-based T2I models are FLUX.1-dev [19], StableDiffusion-3-Medium [11], Stable-Diffusion-3.5-Large [11], and Qwen-Image [34]. The unified multimodal models are Bagel [10], Uni-CoT [26], Emu3 [33], Janus-Pro-7B [9], HunyuanImage-3.0 [6], and T2I-R1 [18]. The proprietary models include Gemini-2.0 [13] and GPT-4o [17].

Results analysis. From Table 1 and 2, we find that DMs exhibit weak performance on both the WISE and T2I-ReasonBench benchmarks. This stems from its fundamental misalignment between their architectural design (i.e., treat LLMs only as a text encoder) and the demands of knowledge-intensive, reasoning-driven T2I generation. These models excel at pixel-level detail synthesis, but their

reliance on shallow text-image alignment restricts them to capturing surface-level semantic correlations—insufficient for tasks requiring deep knowledge integration. For instance, DMs such as vanilla Qwen-Image only achieves 0.35 in WISE chemistry domains, failing to translate abstract concepts (e.g., reaction stoichiometry) into logically consistent visuals, underscoring the importance of our work.

Moreover, we evaluate the zero-shot performance of Qwen-Image by introducing a CoT step: for raw user prompts, we first prompt the model to generate reasoning before outputting a refined prompt. As shown in the results, this modification yields marginal improvement: the overall WISE score increases from 0.61 to 0.65. However, it still lags significantly behind optimized variants, as it lacks task-specific supervised signals to align its reasoning process with the need to infer visually actionable attributes suitable for DiT decoding.

To address this misalignment between reasoning and visual generation requirements, we further conduct SFT training. However, the CoT generation process remains unaware of the image generation module DiT under the SFT pipeline. This decoupling can lead to the LLM generating some strange tokens that the DiT cannot render into a reasonable image. This can be alleviated by further dualGRPO training, where the image-grounded rewards are used for LLM optimization and DiT is adapted for better visual rendering. As seen, our models after SFT and GRPO achieves breakthrough performance on both benchmarks (0.79 on WISE, 68.3 accuracy on T2I-ReasonBench) by addressing the long-standing challenge of reasoninggeneration co-optimization. It achieves a leading performance across open-source T2I models and outperforms the leading proprietary model Gemini-2.0, approaching the

Reference Image Ours Qwen-Image Emu2 Bagel Nano Banana

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Draw what it will look like after being exposed to the sun for one hour.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Draw what it will look like after being hit by a baseball.

- Figure 5. Comparison of T2I models on conceptual image editing. Vanilla Qwen-Image fails to interpret instructions (e.g., showing an ice cream under sunlight instead of melting), behaving as a text–pixel mapper. Our model correctly infers intended semantics, producing coherent, aesthetically pleasing edits with high consistency to the original image, outperforming unified models like Emu2 and Bagel.

- Table 2. Comparison of generation models on T2I-ReasonBench. Numbers in bold are the highest score among open-source models. Underlined numbers denote the highest score.

Model Type Idiom Textual Entity Scientific Overall ↑ Acc. Qual. Acc. Qual. Acc. Qual. Acc. Qual. Acc. Qual.

FLUX.1-dev diffusion 39.1 83.4 56.9 76.5 45.1 90.6 46.7 80.9 47.0 82.8 SD-3.5-medium diffusion 34.4 80.6 58.0 70.1 44.8 92.1 49.9 83.0 46.8 81.4 SD-3.5-large diffusion 35.6 85.3 62.2 75.4 46.6 92.6 52.9 84.5 49.3 84.4 Bagel w/CoT unified 44.6 84.3 44.0 73.7 52.4 91.6 57.7 88.3 49.7 84.5 Janus-Pro-7B unified 25.5 78.0 37.2 70.9 38.5 87.6 44.9 77.8 36.5 78.6 HunyuanImage-3.0 unified 25.4 80.2 54.2 80.9 52.3 92.2 56.8 84.4 47.2 84.4 UniCoT unified 49.0 84.2 58.1 92.3 73.5 92.9 51.9 71.7 58.1 85.3 Gemini-2.0 proprietary 52.4 87.8 73.0 83.3 67.0 94.3 66.7 89.3 64.8 88.7 GPT-4o proprietary 75.7 94.5 86.9 97.6 77.5 96.6 74.7 94.3 78.7 95.8 Qwen-Image diffusion 48.1 84.3 66.5 85.8 57.1 84.7 59.5 85.3 57.8 87.5 Ours (w/o SFT+GRPO) diffusion 51.7 86.3 71.7 83.8 57.3 92.8 62.8 87.6 60.9 86.6 Ours (w/o GRPO) diffusion 59.5 90.4 71.5 87.1 61.29 93.9 72.2 91.8 66.1 90.8 Ours diffusion 58.5 90.6 75.2 89.5 68.8 95.2 72.9 93.5 68.3 92.2

SOTA of closed systems GPT-4o, proving that with our T2G paradigm, open models can match or even surpass closed alternatives in knowledge-intensive visual generation tasks.

###### 4.2.3. Qualitative Results

We analyze the CoT trajectories generated before and after GRPO and observe errors in the roll-out samples (e.g., labeling dumplings as a traditional food for the Dragon Boat Festival in China). These incorrect samples are penalized during GRPO, leading to improved accuracy. Moreover, Figure 2 illustrates images generated by Qwen-Image under the T2G paradigm, demonstrating its superior ability

to infer and understand raw user prompts, while producing text-consistent, visually coherent, and aesthetically pleasing images.

##### 4.3. Image Editing

###### 4.3.1. Supervised Fine-tuning

For raw user instructions for image editing, we select the UniREdit-Data-100K [26], a large-scale synthetic dataset specifically curated for image editing with high-quality CoT reasoning annotations. Following the same training data format as T2I generation tasks, we augment this dataset by

- Table 3. Comparisons of image editing results on UniREditBench and RISEBench. Numbers in bold indicate the highest score among open-source models. Underlined numbers denote the highest score. Detailed scores are shown in Appendix 2.

Model Type UniREdit ↑ RISE ↑

Gemini-2.5-Flash-Image proprietary 68.3 32.8 GPT-Image-1 proprietary 73.4 28.9 Seedream-4.0 proprietary 55.8 10.8 Bagel w/ CoT unified 51.0 9.2 UniWorld-V2 unified 54.9 FLUX.1-Kontext diffusion 45.8 5.8 Qwen-Image-Edit diffusion 56.5 8.9 Ours (w/o GRPO) diffusion 61.1 20.2 Ours diffusion 68.7 23.9

instructing Gemini-2.5 [14] to conclude a refined prompt for each raw instruction and corresponding CoT. This augmented dataset is used to fine-tune Qwen-Image-Edit. For Dual-GRPO training, we randomly select 5k raw user prompts from this dataset.

- 4.3.2. Quantitative Results

Benchmarks. We evaluate performance across various image editing benchmarks requiring reasoning capacity. UniREditBench [26] evaluates models with 2,700 meticulously curated samples, covering both real- and game-world scenarios. RISEBench [38] comprises 327 prompts designed to evaluate models on reasoning-aware tasks across temporal, causal, spatial, and logical dimensions. It proposes a robust evaluation framework that assesses instruction reasoning, appearance consistency, and visual plausibility using the LMM-as-a-judge approach.

Results analysis. As shown in Table 3, after our posttraining, QwenImage-Edit achieves strong performance on image editing tasks, even surpassing Gemini-2.5-FlashImage on the UniREdit benchmark. Moreover, it exhibits a substantial improvement over the SFT-only baseline, demonstrating the effectiveness of our Dual-GRPO in enhancing CoT reasoning through image-based rewards.

###### 4.3.3. Qualitative Results

As shown in Table 5, vanilla Qwen-Image struggles to interpret conceptual editing instructions. For the prompt “draw what the ice cream looks like after being exposed

- to the sun,” it merely renders the reference image under sunlight, resulting in visually incoherent output and revealing its behavior as a simple text–pixel mapper. In contrast, our model trained under the T2G paradigm correctly infers the intended semantics—e.g., depicting the ice cream melting—and produces coherent edits. Moreover, the qualitative results show that our model generates more aesthetically pleasing images than unified models such as Emu2 and Bagel, while achieving the highest consistency between

Table 4. Ablation studies on the importance of SFT. w/o SFT denotes the vanilla Qwen-Image with thinking system prompt.

Model WISE↑ T2I-Reason↑ Acc. Qual.

w/o SFT 0.65 60.9 86.6 w/o SFT + GRPO 0.70 66.9 91.0 w/ SFT 0.74 66.1 90.8 w/ SFT + GRPO 0.79 68.3 92.2

the edited image and the original one. These findings highlight its potential as a next-generation unified model for real-world applications.

##### 4.4. Ablation Studies

To evaluate the effectiveness of our SFT, we conduct an ablation study, noting that the vanilla Qwen-VL already possesses a certain degree of reasoning capability. Specifically, we compare models trained with and without the SFT stage that teaches the T2G pattern. As shown in Table 4, the model with SFT significantly outperforms the variant without it. This improvement stems from SFT explicitly aligning the reasoning process (CoT → refined prompt) with the generation pipeline, enabling the model to produce semantically grounded refined prompts rather than relying on shallow pattern matching or implicit reasoning. Consequently, the DiT receives more coherent conditioning signals, leading to consistently higher-quality image generation.

#### 5. Related Work 5.1. RL for Diffusion Models

Reinforcement learning has emerged as a powerful paradigm for aligning generative models with human preferences, yet its application to diffusion models faces unique challenges due to deterministic sampling and intractable likelihood estimation. Early approaches like DDPO [3] discretized the reverse process to enable policy gradient updates but suffered from solver restrictions and instability when scaling to complex prompts. Flow-GRPO [22] pioneered online RL integration for flow matching by converting ODEs to equivalent SDEs for exploration and introducing denoising reduction to improve sampling efficiency. While effective for compositional generation and text rendering, it optimizes only the diffusion backbone while freezing the text encoder—a limitation restricting its capacity for knowledge-intensive tasks. Similarly, DanceGRPO [36] enhanced stability through group-wise relative policy optimization across multiple generative paradigms but maintained the same encoder-freezing paradigm. DiffusionNFT [39] circumvented likelihood estimation by optimizing directly on the forward process via contrastive

preference learning, achieving remarkable efficiency gains. However, all these methods treat the text encoder as a static feature extractor, overlooking its potential for semantic reasoning when endowed with world knowledge (e.g., in LLMbased encoders). Recent work like Diffusion-SDPO [12] addressed optimization pathologies in preference learning but still operated solely on the decoder. Our Dual-GRPO framework fundamentally diverges by jointly optimizing both the LLM encoder and diffusion decoder through a composite policy design.

##### 5.2. Multimodal Models for Image Generation

The shift toward unified multimodal architectures has accelerated with the emergence of models possessing strong image generation capabilities, such as HunyuanImage [6] and BAGEL [10], which integrate vision-language understanding and generation within a single autoregressive framework. HunyuanImage leverages MoE architectures (over 80B parameters) to unify multimodal processing, while BAGEL demonstrates emergent reasoning capabilities via training on interleaved text-image-video data. This deep integration of textual and visual modeling holds promise for improved image generation conditioned on conceptual and complex prompts. Despite these advances, such models remain biased toward literal generation, largely due to their reliance on descriptive caption datasets during training. To mitigate this, several post-training frameworks incorporate multimodal CoT reasoning. For example, T2I-R1 [18] leverages reinforcement learning with bi-level CoT reasoning to jointly enhance high-level prompt planning and lowlevel pixel generation. In parallel, Uni-CoT proposes a unified two-level reasoning paradigm, consisting of a macroLevel CoT for high-level task planning and a micro-Level CoT for fine-grained subtask execution, all within a single multimodal model. Instead of focusing on unified architectures, our framework emphasizes incorporating text-based CoT reasoning within the LLM encoder to enhance diffusion models.

#### 6. Conclusion

This work addresses a key limitation of T2I diffusion models: their literal text-to-pixel mapping fails to exploit LLMs’ reasoning and world knowledge. We propose the thinkthen-generate paradigm, turning LLMs from passive encoders into active reasoning agents. Our framework has two components: a lightweight supervised fine-tuning to activate LLMs reasoning with raw prompts; and Dual-GRPO, which jointly optimizes LLMs for image-grounded semantic consistency and DiT for visual realism. Experiments show that our approach outperforms open-source baselines, surpasses Gemini-2.0, and excels in conceptual image editing, enabling knowledge-intensive visual generation.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 3, 5
- [2] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506,

2025. 1

- [3] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 8
- [4] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Shijie Huang, Zhaohui Hou, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025. 1
- [5] T Tony Cai and Rong Ma. Theoretical foundations of t-sne for visualizing high-dimensional clustered data. Journal of Machine Learning Research, 23(301):1–54, 2022. 3
- [6] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025. 6, 9
- [7] Junsong Chen, YU Jincheng, GE Chongjian, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations. 1
- [8] Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart-delta: Fast and controllable image generation with latent consistency models. arXiv preprint arXiv:2401.05252, 2024. 1
- [9] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 2, 6

- [10] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2, 6, 9
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, pages 12606–

12633. PMLR, 2024. 1, 2, 5, 6

- [12] Minghao Fu, Guo-Hua Wang, Tianyu Cui, Qing-Guo Chen, Zhao Xu, Weihua Luo, and Kaifu Zhang. Diffusion-sdpo: Safeguarded direct preference optimization for diffusion models. arXiv preprint arXiv:2511.03317, 2025. 9
- [13] Google. Introducing gemini 2.0. https : / / blog.google/technology/google-deepmind/

google- gemini- ai- update- december- 2024,

2024. Accessed: 2025-09-24. 2, 6

- [14] Google. Introducing gemini 2.5 flash image. https : / / developers . googleblog . com / en / introducing-gemini-2-5-flash-image/, 2025. Accessed: 2025-09-24. 1, 3, 5, 8
- [15] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633– 638, 2025. 2
- [16] Alex Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, Sainbayar Sukhbaatar, and Roberta Raileanu. Teaching large language models to reason with reinforcement learning. arXiv preprint arXiv:2403.04642,

2024. 3

- [17] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1, 2, 6
- [18] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025. 6, 9
- [19] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2023. 1, 6
- [20] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 1

- [21] Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, and Shanghang Zhang. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025. 2
- [22] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 2, 3, 5, 8
- [23] Wenyi Mo, Tianyu Zhang, Yalong Bai, Bing Su, Ji-Rong Wen, and Qing Yang. Dynamic prompt optimizing for textto-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26627–26636, 2024. 1
- [24] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025. 2, 6
- [25] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2

- [26] Luozheng Qin, Jia Gong, Yuqing Sun, Tianjiao Li, Mengping Yang, Xiaomeng Yang, Chao Qu, Zhiyu Tan, and Hao

- Li. Uni-cot: Towards unified chain-of-thought reasoning across text and vision. arXiv preprint arXiv:2508.05606, 2025. 6, 7, 8
- [27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1
- [28] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 2
- [29] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward nextgeneration multimodal image generation. arXiv preprint arXiv:2509.20427, 2025. 1
- [30] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2
- [31] Kaiyue Sun, Rongyao Fang, Chengqi Duan, Xian Liu, and Xihui Liu. T2i-reasonbench: Benchmarking reasoning-informed text-to-image generation. arXiv preprint arXiv:2508.17472, 2025. 2, 5, 6
- [32] Omost Team. Omost github page, 2024. 1
- [33] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 6
- [34] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 1, 2, 5, 6
- [35] Dawei Xiang, Wenyan Xu, Kexin Chu, Tianqi Ding, Zixu Shen, Yiming Zeng, Jianchang Su, and Wei Zhang. Promptsculptor: Multi-agent based text-to-image prompt optimization. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 774–786, 2025. 1
- [36] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025. 3, 8
- [37] Zhiyuan Yan, Kaiqing Lin, Zongjian Li, Junyan Ye, Hui Han, Zhendong Wang, Hao Liu, Bin Lin, Hao Li, Xue Xu, et al. Unified multimodal model as auto-encoder. arXiv preprint arXiv:2509.09666, 2025. 2
- [38] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025. 8, 1
- [39] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025. 2, 8

## Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders

### Supplementary Material

(a)

(b)

[Figure 40]

###### 𝑲 = 𝟑

Table 5. Comparison of different reward schedulers on T2IReasonBench.

[Figure 41]

###### 𝑱 = 𝟐

[Figure 42]

- 𝑅11

rewards

- 𝐴መ11

- 𝐴መ12

- 𝐴መ13

advantages

groupcomputation

- 𝑅12

- 𝑅13

𝒛𝟏

Balanced Scheduler Staged Scheduler Category Acc. Qual. Acc. Qual.

<think>

- 𝑅11

- 𝑅12

- 𝑅13

𝒙𝟏𝟏

This implies a need for

[Figure 43]

specific knowledge about

Einstein’ s ...... <\think>

Idiom 62.1 90.6 58.5 90.6 Textual 77.5 90.1 75.2 89.5 Entity 67.1 96.0 68.8 95.2 Scientific 72.4 93.6 72.9 93.5

𝒛ො𝟏

Revised Prompt:

𝒙𝟏𝟐

iteration = 0

A classic wooden violin,

[Figure 44]

rich brown ……

[Figure 45]

[Figure 46]

- 𝐴መ1

- 𝐴መ2

- 𝑅1

- 𝑅2

###### groupcomputation

Generate Einstein’s

Overall ↑ 69.8 92.6 68.3 92.2

𝒙𝟏𝟑

favorite

rewards

advantages

###### 𝑲 = 𝟑 rewards

instrument

advantages

#### 1. Ablation on Different Reward Schedulers

[Figure 47]

- 𝐴መ21

- 𝐴መ22

- 𝐴መ23

- 𝑅21

- 𝑅22

- 𝑅23

𝒒

In Section 3.3, we introduce the reward-weighting scheduler used during training. Specifically, we adopt a balanced scheduler with constant weights, setting β1(τ) = β2(τ) = 0.5, such that both the LLM and the DiT are updated jointly at every iteration. In this section, we an alternative staged piecewise scheduler, where we set β1(τ) = 1,β2(τ) = 0 at early training steps and switch to β1(τ) = 0,β2(τ) = 1 in later steps (staged scheduler in Table 5. This design effectively updates only the LLM parameters in the early phase and only the DiT parameters in the later phase.

𝒛𝟐

groupcomputation

<think>

lavender in winter smoke billowing …

- 𝑅21

- 𝑅22

- 𝑅23

𝒙𝟐𝟏

While Einstein was known

[Figure 48]

[Figure 49]

[Figure 50]

to play the violin, simply stating ……

<\think>

𝒛ො𝟐

Revised Prompt: A beautifully crafted, classic wooden ……

𝒙𝟐𝟐

[Figure 51]

𝒙𝟐𝟑

iteration = 240

We compare the effectiveness of different schedulers in Table 5 on the T2I-ReasonBench benchmark. The results show that the balanced scheduler consistently outperforms the staged scheduler. We hypothesize that this is because joint optimization enables tighter coordination between prompt refinement and visual rendering. In Figure 6, we visualize the evolution of the rewards during training, as well as samples before and after the Dual-GRPO under this new staged scheduler.

- Figure 6. Evolution of alignment and style scores during training with the new scheduler, where β1(τ) = β2(τ) = 0.5, and samples demonstrating how DiT training improves both semantic alignment and visual quality over time.

GPT-4o Ours Qwen-Image Bagel-think

1.6

1.8

2.0

2.2

2.4

2.6

2.8

3.0

Score

3.04

2.98

2.10

1.90

- Figure 7. Comparison of scores on user study. Our model performs closely to GPT-4o and clearly surpasses Qwen-Image, highlighting its strong ability in challenging real-world T2I scenarios.

#### 2. Detailed Results on Image Editing Task

In this section, we provide detailed scores on the image editing task. We compare our model with several state-of-theart image editing models over the RISE [38] benchmark. The results are shown in 6. It can be seen that training under our think-then-generate paradigm significantly outperforms the vanilla Qwen-Image-Edit and even surpasses strong proprietary models like Seedream-4.0. These results demonstrate that incorporating reasoning into the generation process substantially improves both accuracy and visual quality of edited images.

[Figure 52]

Figure 8. User interface for our human evaluation study. Images 1–4 correspond to Bagel-think, our model, vanilla Qwen-Image, and GPT-4o, respectively. Notably, our model provides the most comprehensive deduction for this math problem, uniquely arriving at the correct final answer, thereby demonstrating its superior reasoning ability and practical effectiveness in real-world tasks.

Table 6. Detailed scores on RISEBench. Numbers in bold indicate the highest score among open-source models.

Model Type Temporal Causal Spatial Logical Overall ↑

Gemini-2.5-Flash-Image proprietary 25.9 47.8 37 18.8 32.8 GPT-Image-1 proprietary 34.1 32.2 37 10.6 28.9 Gemini-2.0-Flash-exp proprietary 8.2 15.5 23 4.7 13.3 Seedream-4.0 proprietary 12.9 12.2 11 7.1 10.8 BAGEL unified 2.4 5.6 14 1.2 6.1 BAGEL (w/ CoT) unified 5.9 17.8 21 1.2 11.9 FLUX.1-Kontext diffusion 2.3 5.5 13.0 1.2 5.8 Qwen-Image-Edit diffusion 4.7 10.0 17.0 2.4 8.9 Ours (w/o GRPO) diffusion 22.3 27.8 21.0 9.4 20.2 Ours diffusion 20.0 28.9 15.3 30.0 23.9

#### 3. User Study

In this section, we conduct a user study on T2I generation in challenging real-world scenarios. We select 46 prompts and compare our model against strong baselines, including vanilla Qwen-Image [34], Bagel-think [10], and GPT-

- 4o [17]. Users are instructed to rank the model outputs according to semantic alignment and conceptual understanding, visual realism and coherence, and aesthetic quality. Our preference-collection interface is shown in Figure 8.

For evaluation, we aggregate prompt-level rankings. For each given prompt i, the model receives a reward si =

- 5 − ri, where ri ∈ {1,2,3,4} denotes its rank among the four models. The average score is computed by adding the rewards and averaging all prompts (Figure 7). Results show that our model performs closely to GPT-4o and clearly sur-

passes Qwen-Image, highlighting its strong ability in challenging real-world T2I scenarios. Notably, as shown in Figure 8, our model produces the most comprehensive reasoning for math-related problems and uniquely arrives at the correct final answer, highlighting its superior deductive capabilities and practical effectiveness in real-world tasks.

#### 4. More Demos

We provide more demos for the T2I task in Figure 10, and additional examples for image editing in Figure 9. These results further demonstrate that our model delivers stronger conceptual instruction understanding and alignment, more robust appearance consistency across generated content, and improved overall visual plausibility and coherence compared to existing baselines.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

prompt: Draw what it will look like after many people have walked on it.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

prompt: Draw what it will look like one hour later.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

prompt: Draw what it will look like after baking for 45 minutes.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

prompt: Draw what it will look like after a year placed in a high-school classroom.

Figure 9. More demos for image editing task.

Qwen-Image Ours FLUX.1 dev Nano Banana Bagel

GPT 4o

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

prompt: The ship sinking after hitting an iceberg in 1912

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

prompt: Halloween kids' favorite treat

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

prompt: An autumnal harvest celebration in North America, where a large fowl takes center stage on the dining table

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

prompt: A sport that ignites national fervor in Argentina, often associated with legendary players and historic rivalries

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

prompt: The craft that embodies Swiss precision and artistry

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

prompt: A small, oval fruit with a fuzzy skin and a sweet, tart taste, often associated with the start of summer and the colour pink

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

prompt: A majestic striped animal, symbolizing the wildlife of Bangladesh

Figure 10. More demos for the T2I task.

