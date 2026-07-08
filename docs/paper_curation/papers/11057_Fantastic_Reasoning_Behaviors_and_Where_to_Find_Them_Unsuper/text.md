## arXiv:2512.23988v1[cs.CL]30Dec2025

[Figure 1]

2026-01-01

# Fantastic Reasoning Behaviors and Where to Find Them: Unsupervised Discovery of the Reasoning Process

###### Zhenyu Zhang*,2, Shujian Zhang1, John Lambert1, Wenxuan Zhou1, Zhangyang Wang2, Mingqing Chen1, Andrew Hard1, Rajiv Mathews1 and Lun Wang1

1Google DeepMind, 2The University of Texas at Austin, *Work done as a student researcher at Google DeepMind

Despite the growing reasoning capabilities of recent large language models (LLMs), their internal mechanisms during the reasoning process remain underexplored. Prior approaches often rely on humandefined concepts (e.g., overthinking, reflection) at the word level to analyze reasoning in a supervised manner. However, such methods are limited, as it is infeasible to capture the full spectrum of potential reasoning behaviors, many of which are difficult to define in token space. In this work, we propose an unsupervised framework (namely, RISE: Reasoning behavior Interpretability via Sparse auto-Encoder) for discovering reasoning vectors, which we define as directions in the activation space that encode distinct reasoning behaviors. By segmenting chain-of-thought traces into sentence-level ‘steps’ and training sparse auto-encoders (SAEs) on step-level activations, we uncover disentangled features corresponding to interpretable behaviors such as reflection and backtracking. Visualization and clustering analyses show that these behaviors occupy separable regions in the decoder column space. Moreover, targeted interventions on SAE-derived vectors can controllably amplify or suppress specific reasoning behaviors, altering inference trajectories without retraining. Beyond behavior-specific disentanglement, SAEs capture structural properties such as response length, revealing clusters of long versus short reasoning traces. More interestingly, SAEs enable the discovery of novel behaviors beyond human supervision. We demonstrate the ability to control response confidence by identifying confidence-related vectors in the SAE decoder space. These findings underscore the potential of unsupervised latent discovery for both interpreting and controllably steering reasoning in LLMs.

Keywords: Reasoning, Sparse Auto-Encoding, Mechanistic Interpretability

##### 1. Introduction

Recent advancements in reasoning have significantly expanded the capabilities of large language models (LLMs), moving them far beyond basic language understanding to encompass more complex reasoning tasks. These include competition-level mathematical problem solving (Ahn et al., 2024; AlphaProof and AlphaGeometry, 2024), project-level coding (Jiang et al., 2024), and planning (Huang et al., 2024; Valmeekam et al., 2023). Evidence from model responses suggests that such substantial performance improvements primarily stem from their ability to perform extended chain-of-thought (CoT) reasoning (Wei et al., 2022). Nevertheless, how such lengthy reasoning trajectories contribute to performance remains underexplored.

Recent studies have investigated the reasoning process, such as analyzing entropy mechanisms during inference (Fu et al., 2025; Zhang et al., 2025) or at the post-training stage (Cui et al., 2025; Wang et al., 2025b; Zhao et al., 2025), which can be further leveraged to achieve more accurate and concise reasoning (Yang et al., 2025). Other works suggest that current reasoning processes are often verbose, with certain parts being ineffective (Fu et al., 2024; Huang et al., 2025; Sheng et al., 2025).

© 2026 Google. All rights reserved

In addition, another line of studies argues that reasoning ability is tied to specific behaviors (Chen et al., 2025; Gandhi et al., 2025; Venhoff et al., 2025; Ward et al., 2025), such as reflection (i.e., the model revisits and verifies its previous reasoning steps) and backtracking (i.e., the model abandons the current reasoning path and pursues an alternative solution), which appear to be closely associated with the achieved performance gains.

From the perspective of mechanistic interpretability, many studies rely on activation engineering methods, particularly the Difference-of-Means (DiffMean) approach (Marks and Tegmark, 2023). This method begins by constructing a contrastive dataset with human supervision, such as categorizing samples into safe vs. harmful or positive vs. negative groups. For each category, it then extracts the latent representations of individual samples (i.e., ℎ+

𝑗 ) and computes a steering vector by averaging the representations within each category and taking their difference: 𝑣 = 𝑁1

𝑖 and ℎ−

𝑁+ 𝑖=1 ℎ+

𝑖 − 𝑁−

+

- 1

𝑗 . The resulting steering vector can then be applied to shift the response style, for example, from sad to happy or vice versa.

𝑁− 𝑗=1 ℎ−

However, such activation-based approaches are ill-suited for understanding reasoning, as they rely on supervised, predefined concepts. This is feasible in tasks like sentiment classification (Pang et al., 2002), where concepts such as “happy” and “sad” are clearly separable. In contrast, reasoning behaviors are fluid, overlapping, and difficult to annotate at scale, which has led prior studies to focus narrowly on a small set of predefined behaviors such as reflection or backtracking (Wang et al., 2025d). This restricts both the scope of analysis and the robustness of the resulting steering vectors.

|Sparse Autoencoder — Decoder Space<br><br>[Figure 2]<br><br>|
|---|

Vanilla Reasoning

c

|MLP<br><br>Attention<br><br>LLM| | |
|---|---|---|
| | | |
| | | |

|MLP<br><br>Attention<br><br>LLM| | |
|---|---|---|
| | | |
| | | |

SAE Encoder

So, we can solve it by [step1] [step2] [step3]....

c

c

c

Reasoning with Reflection

Let me double check, ..., Maybe I'm wrong, ..., but, ...

c

Reasoning with Confidence

I'm sure, we first [step1], ... , yes, then [step2]...

- Figure 1 | Illustration of our RISE framework for unsupervised reasoning behavior discovery. The pipeline consists of two stages: (i) training a Sparse Autoencoder (SAE) on unlabeled representations of reasoning steps (Left), and (ii) evaluating causal effects on the original reasoning process (Right). Notably, the intervention process on the right is applied directly, without any additional training.

In this work, we address these challenges by introducing an unsupervised framework (i.e., RISE: Reasoning behavior Interpretability via Sparse auto-Encoder) for discovering reasoning behaviors. Building on the linear representation hypothesis (Park et al., 2023), we define fine-grained reasoning behaviors as linear directions in the activation space, which we refer to as Reasoning Vectors. These vectors can be applied to modify the reasoning mechanism in specific ways and thereby influence response quality.

Unlike prior approaches, our framework does not rely on human-supervised labels; instead, it directly identifies reasoning vectors from step-level activations in chain-of-thought sequences. To achieve this, we employ sparse auto-encoders (SAEs), which learn a dictionary of sparse latent features that reconstruct hidden states while promoting disentanglement. We show that individual decoder columns correspond to interpretable reasoning behaviors (Figure 1). Visualizations further reveal that these vectors cluster into semantically coherent regions of activation space. Most importantly, we demonstrate controllability: injecting SAE-derived vectors during inference can suppress or amplify behaviors such as reflection, directly altering the reasoning trajectory without additional training.

Beyond semantic behaviors, we also observe evidence of structural organization: decoder columns consistently form clusters aligned with response length, with separability peaking in mid-to-late layers. Furthermore, we show that SAEs can reveal new behaviors that are difficult to define through wordlevel human supervision. As a case study, we examine the semantic concept of confidence and find that confidence-related directions are highly concentrated, with interventions on the corresponding vectors exerting clear causal effects on reasoning styles. Our contributions are threefold:

- • We propose an unsupervised framework, i.e., RISE, that captures the structure of reasoning behaviors within the latent space. Unlike prior methods that rely on human-defined concepts, RISE end-to-end models diverse reasoning behaviors directly within the decoder column space.
- • With RISE, we show that SAE-derived vectors align with human-interpretable behaviors and can be used to selectively modulate reasoning at inference time without additional training. In particular, we can directly control reflection and backtracking during reasoning by reducing or enhancing the components of hidden representations along the SAE column directions on the fly.
- • We further demonstrate the ability to discover novel reasoning behaviors that are difficult to define with word-level human supervision. As an example, we consider confidence, a behavior that is challenging to specify at the word level. We find that confidence-related reasoning vectors form coherent clusters and causally shift the model’s response style toward more confident answers.

##### 2. Related Works

Reasoning Models. Previous works have demonstrated that enabling models to generate longer outputs can significantly enhance their reasoning ability. This progress starts from the famous Chain-ofThought (CoT) prompting (Wei et al., 2022), to test-time scaling (Snell et al., 2024), and more recent reinforcement learning–optimized reasoning models, such as OpenAI’s o-series (Jaech et al., 2024), Anthropic’s Claude-3.7-Sonnet-Thinking (Anthropic, 2025), and Google’s Gemini-2.5-Flash (Google, 2025), as well as notable open-source counterparts (Abdin et al., 2025; Guo et al., 2025; Team, 2024, 2025; Yang et al., 2024). While longer responses generally improve performance, the mechanisms remain unclear. Recent studies link these gains to several metacognitive behaviors (Chen et al., 2025; Gandhi et al., 2025; Venhoff et al., 2025), yet they rely on human-defined behaviors. In this work, we seek to uncover the geometry of reasoning behaviors in an unsupervised manner, thereby avoiding dependence on human-labeled definitions.

Activation Steering. Activation steering modifies model outputs by editing internal activations, with notable approaches including representation engineering (Zou et al.), activation patching (Meng et al., 2022), and DiffMean (Marks and Tegmark, 2023). These methods have been effective in domains such as improving LLM truthfulness (Wang et al., 2025c), enhancing privacy (Goel et al., 2025), and controlling sentiment (Han et al., 2023). However, they typically rely on contrastive pairs (e.g., happy vs. sad) to define steering directions, which is suitable when clear oppositional concepts exist. In reasoning, such contrasts are more difficult to define: prior work has targeted specific behaviors like reflection (Chen et al., 2025; Venhoff et al., 2025) or coarse distinctions between short and long responses (Eisenstadt et al., 2025; Huang et al., 2025; Sheng et al., 2025), leaving the broader reasoning behavior space underexplored. In this work, we present a pivot study to investigate this space in an unsupervised manner. A recent effort also explored reasoning models in an unsupervised way by using a Sparse Auto-Encoder (SAE) (Wang et al., 2025a), but primarily treated SAE as a bridge to transfer reasoning ability from a reasoning model to a base model via supervised fine-tuning, offering limited insight into the internal mechanisms of the reasoning process. Our work instead aims to directly analyze these mechanisms.

##### 3. Preliminary

- 3.1. Sparse Auto-Encoder

Building on the Linear Representation Hypothesis (Olah and Jermyn, 2024; Park et al., 2023), each atomic reasoning behavior can be mapped to a specific direction in the activation space, which we define as a reasoning vector. To automatically identify such behaviors, we employ sparse auto-encoders (SAEs), which provide an effective and principled approach. An SAE consists of an encoder and a decoder that aim to reconstruct representations in the activation space:

ℎˆ = 𝑊decoder⊤ 𝜎(𝑧) + 𝑏decoder; 𝑧 = 𝜎 𝑊encoder⊤ ℎ + 𝑏encoder (1)

Here, ℎ ∈ ℝ𝑑 denotes the original representation, 𝜎 is a non-linear activation function, and ℎˆ is the reconstruction. Each row of 𝑊decoder ∈ ℝ𝐷×𝑑 corresponds to an atomic vector (i.e., a reasoning vector), where 𝐷 is the hidden dimension of the SAE and 𝑑 is the dimension of the original input. The latent feature 𝑧 represents the corresponding code. For a standard SAE (Cunningham et al., 2023), we use ReLU as the non-linear activation function 𝜎. The training objective is formulated as

L = ∥ℎˆ − ℎ∥22 + 𝜆∥𝑧∥0, (2)

which encourages the SAE to accurately reconstruct the original activations while enforcing sparsity in the latent codes. This ensures that only a small number of atomic concepts are used, thereby reducing entanglement across vectors and enhancing interpretability.

- 3.2. Thought Representation Construction

Intuitively, reasoning behaviors cannot be trivially explained by token-level attributes, since the same token can play diverse roles across different contexts. This observation, further supported by recent studies (Venhoff et al., 2025; Ward et al., 2025), motivates our choice of analyzing the reasoning process at the sentence level, where the internal structure of the whole responses can be more meaningfully captured.

The construction of thought representations for SAE training involves the following steps: (i) Collecting model responses: we feed each question from the selected training set into the target model to generate responses. (ii) Splitting responses into sentence-level steps: we divide each response into reasoning steps using the delimiter symbol <\n\n>, producing 𝑘 steps per response, where 𝑘 varies across samples. (iii) Embedding each step: we re-run inference by feeding both the question and the corresponding response into the model, and extract the hidden representations of the token <\n\n>. Each representation is then regarded as the activation of its corresponding reasoning step (Chen et al., 2025). The representations are then being concatenated, denote as {ℎ𝑙𝑖}, where 𝑖 ∈ {1, . . . , 𝑁} indexes the samples and 𝑙 denotes the layer index. We specifically use the residual stream representations after each transformer layer, and train the SAE on a single chosen layer.

##### 4. Unsupervised Reasoning Vector Discovery

###### 4.1. SAE Discovers Reasoning Behaviors in the Decoder Column Space

To begin, we first analyze the ability of the SAE to capture the ground-truth representation structure. For each activation ℎ obtained from the original model, the SAE decoder reconstructs it as: ℎˆ = 𝑊⊤

decoder 𝜎(𝑧) + 𝑏decoder. This reconstruction can be interpreted as a linear combination of selected rows of 𝑊decoder. Under the linear representation hypothesis, we posit that the true activation admits

a sparse representation over a ground-truth dictionary 𝑊 = [𝑤1, . . . , 𝑤𝑚] ∈ ℝ𝑑×𝑚. As established in Theorem 1, the SAE decoder matrix can recover this dictionary in the decoder space up to additional permutation and scaling. More details can be found in Appendices B and C.

Theorem 1. Suppose hidden representations at delimiter tokens follow the generative model

ℎ = 𝑊𝑎 + 𝜀, (3)

where ℎ ∈ ℝ𝑑, 𝑊 = [𝑤1, . . . , 𝑤𝑚] ∈ ℝ𝑑×𝑚 is a dictionary of latent behavior directions, 𝑎 is a 𝑘-sparse code, and 𝜀 is bounded noise. Assume: (i) Incoherence: max𝑖≠𝑗 ∥|⟨𝑤𝑤𝑖,𝑤𝑗⟩|

𝑖∥∥𝑤𝑗∥ ≤ 𝜇 < 1. (ii) Sparsity: 𝑘 < 𝑐/𝜇 for a universal constant 𝑐. (iii) Activation: SAE uses ReLU nonlinearity. (iv) Separation: Nonzero coefficients satisfy |𝑎𝑖| ≥ 𝛼 > 0.

Then, as the number of samples 𝑁 → ∞, any local optimum of the SAE training objective min

𝔼 ∥ℎ − ℎˆ∥22 + 𝜆∥𝑧∥0 (4)

𝑊enc,𝑊dec

recovers a decoder matrix whose columns align with the true dictionary up to permutation matrix Π and scaling diagonal matrix 𝐷:

∃Π, 𝐷 ≻ 0 s.t. 𝑊dec ≈ 𝑊Π𝐷. (5)

###### 4.2. Setup Details

We use DeepSeek-R1-Distill-Qwen-1.5B (R1-1.5B) (Guo et al., 2025), along with five hundred randomly sampled training examples from the MATH dataset (Lightman et al., 2023). We follow Section 3.2 to obtain the activations for SAE training. Since our goal is to model various reasoning behaviors, whose complexity is much lower than modeling raw language structure, we adopt a relatively small hidden dimension of 𝐷 = 2048. The SAE is trained with a batch size of 1024 and a learning rate of 1 × 10−4, with a warm-up over the initial 10% of training. We use the Adam optimizer (Kingma, 2014) with cosine annealing learning rate decay. To encourage sparsity, we apply a sparsity strength of 𝜆 = 2 × 10−3. Additionally, we extend our analysis to Qwen3-8B and report the results in Figure 5.

###### 4.3. Visualizing the Geometry of the SAE Decoder

We then conduct a human-supervised analysis of the SAE decoder geometry to assess whether the structures learned by the SAE align with human supervision. Specifically, we classify each

Geometry of SAE

w. Reflection Highlighted

w. Backtracking Highlighted

w. Others Highlighted

- 4

- 5

- 6

- 7

- 8

- 4

- 5

- 6

- 7

- 8

- 4

- 5

- 6

- 7

- 8

- 4

- 5

- 6

- 7

- 8

UMAP-2

UMAP-2

UMAP-2

UMAP-2

4 6 8 10 UMAP-1

4 6 8 10 UMAP-1

4 6 8 10 UMAP-1

4 6 8 10 UMAP-1

- Figure 2 | Visualization of SAE decoder columns projected onto a 2-D plane with UMAP. From left to right, we show the raw SAE decoder rows and the corresponding results with human-defined behaviors highlighted. Results are obtained from the final layer of R1-1.5B.

representation in {ℎ𝑙𝑖} into one of three categories: reflection, backtracking, or other. The main motivation for emphasizing reflection and backtracking is that, compared to prior chain-of-thought

prompting, current reasoning models introduce stages of refining answers through these processes, which are the most prominent features we aim to explore. The classification is performed using an LLM-as-a-judge approach, where for each reasoning step associated with ℎ𝑙𝑖, we prompt the LLM (i.e., GPT-5) with precise definitions of the behaviors: reflection (re-examining earlier steps), backtracking (switching to a new approach), and other. For each labeled step, the corresponding representation is then mapped back to the SAE latent feature space. More details on the annotation process and the consistency across different annotation methods are provided in Appendix D. We record the activation patterns and report the top-active channels, highlighting their associated decoder columns. Here, the activity of a channel is measured by the largest magnitude of its latent feature. Additionally, we apply Uniform Manifold Approximation and Projection (UMAP) (McInnes et al., 2018) to embed the decoder columns into a two-dimensional space.

Results. Figure 2 illustrates that the SAE decoder columns encode semantically meaningful behaviors in the latent space. The leftmost subfigure shows the UMAP projection of all decoder vectors without annotations. When overlaid with humanlabeled behavioral concepts, clear semantic structures emerge. We adopt UMAP for visualization because it leverages cosine similarity as the internal metric, emphasizing directional rather than Euclidean differences. This choice is particularly appropriate since activation vectors are primarily meaningful in terms of their direction rather than their magnitude. Furthermore, for behaviors such as reflection and backtracking, which are semantically clearer from a human perspective, the corresponding decoder columns cluster more tightly in localized regions. In contrast, for the other category, which potentially encompasses a mixture of behaviors, the top-active vectors are less centered and more dispersed across the entire SAE space.

1.0

NormalizedSilhouetteScore

0.8

0.6

0.4

Overall

Reflection vs Backtracking

0.2

Others vs Reflection

Others vs Backtracking

0.0

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 Layer

Figure 3 | Normalized Silhouette scores across different layers of R1-1.5B.

Layer-wise Properties of SAE Geometry. Then, we quantify the above analysis by Silhouette scores (Rousseeuw, 1987; Shahapure and Nicholas, 2020). This metric measures both the cohesion of samples within a cluster and their separation from other clusters. We normalize the raw scores across layers for better visualization. The results are shown in Figure 3. Two key observations can be made. First, later layers generally achieve higher Silhouette scores than earlier ones, indicating that behavioral concepts become more separable as representations deepen. Interestingly, a slight decline occurs near the final layers (except for the very last one), suggesting that mid-to-late representations encode the most distinct behavioral structures. This observation aligns with the oversmoothing phenomenon in current LLMs (Wang et al., 2023), where token representations become excessively similar. Second, the comparison between reflection and backtracking reveals only modest differences, whereas the separation of “other” behaviors from either reflection or backtracking is substantially stronger. This suggests that reflection and backtracking occupy more overlapping representational subspaces, while both are more clearly distinguished from the residual “other” category.

###### 4.4. Causal Examination of SAE Decoder Columns

###### 4.4.1. Incorporating SAE Columns during LLM Inference

Incorporating SAE Columns during LLM Inference. We next perform an intervention study to examine the causal effect of SAE decoder columns, focusing on those from the final layer. We first filter out decoder columns that exhibit strong activations across multiple behaviors (e.g., reflection and backtracking). From the remaining reflection-specific columns, we compute their average to obtain a single reflection vector. During inference, this vector is injected into the hidden representation of the last token at each reasoning step, as shown in Figure 4. Note that all decoder columns are independently normalized to unit length ||𝑤𝑖||2 = 1. The intervention is then performed as:

ℎ′ = ℎ − 𝑤𝑖(𝑤𝑖⊤ℎ), (6)

where 𝑤𝑖 denotes decoder column 𝑖. In this way, we project out the component of ℎ along the reflection direction to examine how the reflection behavior changes in the model’s responses.

Intervention Results. As shown in Figure 6, when DeepSeek-R1-1.5B is prompted with a detailed math question, its reasoning style can be effectively shaped while preserving correctness. Across all conditions, the model consistently produces the same final answer (3, 𝜋/2), yet the reasoning trajectory shifts in a predictable manner: negative intervention suppresses meta-cognitive phrases (i.e., reflection) and shortens the whole response process, positive intervention amplifies self-checking behaviors while lengthening it, and vanilla inference lies between. While a simple example may appear trivial, we further validate the causal effect of SAE columns across diverse tasks, behaviors, and models. As shown in Figure 5, the results consistently demonstrate the causal influence of SAE columns. Specifically, applying negative interventions leads to a consistent decline in the corresponding behavior, whereas positive interventions result in clear enhancement. Moreover, when we manually control the intervention strength with a

|| |
|---|
|
|---|

Decoder

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

MLP

Decoder

Encoder

Attention

Figure 4 | Illustration of our inference process that utilizes SAE decoder columns. For a given reasoning behavior, we compute the corresponding centroid in the SAE decoder column space and directly apply it during inference of the original model for examining how the response changes.

AIME 2025, R1-1.5B Negative

AMC23, R1-1.5B

AIME 2024, Qwen3-8B Negative

MATH500, Qwen3-8B

150

100

70

Negative Vanilla Positive

Negative Vanilla

60

125

Vanilla

Vanilla

| |
|---|

| |
|---|

150

80

Positive

Positive

Positive

| |
|---|

50

100

60

40

#Steps

#Steps

#Steps

#Steps

100

75

30

40

50

20

50

20

25

10

0

0

0

0

Reflection Backtracking

Reflection Backtracking

Reflection Backtracking

Reflection Backtracking

- Figure 5 | Statistics of reasoning behavior shifts induced by SAE column interventions are reported across different models and tasks, where the SAE columns are consistent across tasks.

- Table 1 | Generalization of SAE columns under domain shift. The SAE columns are learned from the MATH500 dataset and applied to other domains.

GPQA-Diamond KnowLogic Reflection Backtracking Reflection Backtracking

Steering Direction

Vanilla 53.23 11.83 35.56 5.42 Positive 62.77 20.31 51.00 9.38 Negative 45.42 6.47 25.99 2.33

scalar value as: ℎ′ = ℎ − 𝛼 · 𝑤𝑖(𝑤⊤

𝑖 ℎ), where 𝛼 ∈ {−1.5, −1, 0, 1, 1.5}, the corresponding number of reflection steps of DeepSeek-R1-1.5B on the AIME25 tasks changes to {58.6, 73.6, 90.5, 131.0, 166.9}, consistently evolving in accordance with the intervention strength. These results further support that SAEs learn meaningful reasoning behaviors such as reflection in an unsupervised manner.

Generalization Across Data Domains.The SAE is trained on the MATH500 dataset, after which we evaluate how the learned vectors generalize to other domains. To ensure a larger domain shift, we consider commonsense and logical reasoning tasks, specifically GPQA-Diamond (Rein et al., 2024) and KnowLogic (Zhan et al., 2025). Notably, we apply the same SAE columns during inference on these new domains with the R1-1.5B model. The results, reported in Table 1, show that the reflection and backtracking vectors consistently modulate reasoning behaviors across domains.

SAE Reveals the Underlying Geometry of Response Length. Inspired by recent findings (Huang et al., 2025; Sun et al., 2025) that response length can be distinguished in the activation space, we are further motivated to explore the geometry of response length in the SAE decoder space. Using the same procedure, the results are presented in Figures 11 and 12. We observe that early layers exhibit weak separation with respect to response length, whereas mid-to-late layers show stronger alignment with this structural property. More details can be found at Appendix E.

##### 5. Discovering New Behaviors with SAE

In the previous section, we demonstrated that an unsupervisedly trained SAE can capture semantically meaningful structures, as evidenced by visualizing its weight geometry and its alignment with humandefined reasoning behaviors. In the following, we present an initial attempt showing that SAEs can also be used to discover new behaviors.

From the decoder columns of the SAE, the key challenge becomes selecting a specific subspace that can be leveraged for a particular goal. As a starting point, we consider identifying a goal that meaningfully impacts the reasoning process. Recent studies on the entropy mechanisms of LLM reasoning (Cui et al., 2025; Fu et al., 2025; Wang et al., 2025b; Zhang et al., 2025; Zhao et al., 2025) highlight that entropy, or equivalently model confidence, plays a critical role during the reinforcement stage. For instance, incentivizing an LLM to maximize its confidence in the final answer has been shown to positively influence reasoning ability in certain scenarios. Moreover, confidence can also serve as a criterion for filtering out ineffective responses (Fu et al., 2025; Yang et al., 2025; Zhang et al., 2025), thereby improving inference efficiency. Motivated by these insights, we adopt entropy

- as our proxy objective and seek to identify decoder columns that are closely associated with entropy.

Setup. Given a trained SAE with decoder 𝑊decoder ∈ ℝ𝐷×𝑑, where 𝑑 denotes the dimension of the original model and 𝐷 is the number of learned vectors, we divide the original model into two parts. The splitting point is chosen at the layer 𝑙 from which the SAE is trained. Accordingly, we denote the

###### Question: Convert the point in rectangular coordinates to polar coordinates. Enter your answer in the form where and < 2π

Okay, [...] [non-reflection] [...] is 90 degrees. Wait, let me think again. [...] π/2 radians. So, θ is π/2. But I should make sure that [...] system. So, θ is π/2. Wait, but sometimes, [...] x is 0 and y is positive, θ is π/2. Alternatively [...] [non-reflection] [...]are (3, π/2). Wait, let me double-check. [...] the original point (0, 3). Therefore, I [...] [non-reflection] [...] are (3, π/2). But just to make sure[...] So, that seems right. Alternatively, if [...] [non-reflection] [...], it's π/2. Wait, another thought: [...] so it's straightforward. I think [...] [non-reflection] [...] for (0, 3). Final Answer [...]

### Positive Intervention

### 6 Steps for Reflection

Okay, [...] [non-reflection] [...] 90 degrees. Wait, let me think again.[...] π/2 radians. So, θ is π/2. But I should make sure that [...] system. So, θ is π/2. Let me double-check the [...] positive y-axis, θ is π/2. So, [...] [non-reflection] [...]are (3, π/2). Wait, just to make sure, [...] up. Yeah, that makes sense. I don't [...] [non-reflection] [...](3, π/2). Final Answer [...]

Vanilla Inference

### 4 Steps for Reflection

Okay, [...] [non-reflection] [...] degrees. Let me verify that. If I [...] That makes sense. I think [...] [non-reflection] [...] (3, π/2). Wait, just to make sure [...] should get (0, 3). Using the [...] [non-reflection] [...] Final Answer [...]

Negative Intervention

2 Steps for Reflection

- Figure 6 | Responses from the R1-1.5B model when intervened with SAE vectors corresponding to reflection behaviors. Reasoning steps associated with reflection are highlighted in red. Zoom in for better visualization.

remaining part of the model as ℎ = 𝑓1→𝑙(𝑥); 𝑦 = 𝑓𝑙→𝐿(ℎ), where 𝐿 is the total number of layers, 𝑥 is the input token and ℎ is the activation at layer 𝑙.

Identifying Target SAE Decoder Columns. We assign a score vector 𝑆 ∈ ℝ𝐷 to the 𝐷 decoder columns and optimize 𝑆 by minimizing the final entropy. Specifically, given the activation set {ℎ𝑙𝑖} extracted from the MATH500 training set at layer 𝑙, we solve the following objective:

∑︁|𝑉|

argmin

𝑝𝑘 log 𝑝𝑘 , 𝑝𝑘 = softmax 𝑓𝑙→𝐿(ℎ + 𝑆𝑊decoder) 𝑘, (7)

𝔼 −

𝑆

𝑘=1

where 𝑝𝑘 denotes the predicted probability of token 𝑘 and |𝑉| is the vocabulary size. We solve this optimization using the Adam optimizer with a learning rate of 0.01 for 1000 iterations, together with a cosine annealing learning rate decay schedule. The batch size is 256.

Vanilla Inference Confident Inference

|[Figure 3]|
|---|

|[Figure 4]|
|---|

(a) (b) (c)

- Figure 7 | (a) Visualization of SAE decoder columns, with top-scoring columns related to entropy highlighted, showing clear clustering. (b) Most frequent tokens during vanilla inference. (c) Most frequent tokens under intervention with the entropy-related vector. Comparing (b) and (c), tokens associated with reflection or backtracking become less frequent (e.g., Wait), while more tokens emerge that correspond to confident mathematical calculation (e.g., numbers)

###### 5.1. Emergence of Confident Reasoning Vectors

We conduct experiments on the last layer of DeepSeek-R1-1.5B. As shown in Figure 7, we select the columns with the highest scores in 𝑆, which implies that these columns play a critical role in achieving low entropy. We then inject with the corresponding vector, referred to as the confidence reasoning vector, during inference of the original model. The intervention process follows the same procedure described in the previous section 4.4.

Confidence Vectors Concentrated in the SAE Space. First, we observe that the confidence-related columns are predominantly located in the bottom-right region of the visualization, suggesting that this area is particularly important for reasoning confidence. Moreover, these columns partially overlap with the reflection and backtracking regions identified in Figure 2, indicating that reflection and backtracking behaviors may contribute to higher entropy. This observation is consistent with recent findings (Wang et al., 2025b). Further results are provided by Figures 7(b–c), where we visualize the most frequent tokens at the beginning of each reasoning step. Without intervention, frequent tokens include reflection-related cues such as “Wait” and backtracking indicators such as “Alternatively.” After intervention with the confidence vector, these are replaced by tokens more closely related to detailed mathematical calculation steps.

Causal Effect of Confidence Vectors. In addition, we measure the frequency of reflection and backtracking steps before and after intervention. On the AIME25 tasks, reflection steps are reduced from 90.53 to 33.77, and backtracking steps decrease from 35.50 to 5.93. While we observe a slight performance drop from 23.33% to 20.00%, this corresponds to only a single question out of 30 and is therefore not statistically significant. More importantly, the intervention substantially alters the reasoning style, with large reductions in reflection and backtracking. These results demonstrate that SAEs can be used not only to interpret but also to actively modify the reasoning style of LLMs. Furthermore, as shown in Table 2, we observe that the confidence vector also generalizes across domains, exhibiting a clear ability to simultaneously modulate both reflection and backtracking behaviors. This further demonstrates the semantic similarity between the confidence direction and the combined effects of reflection and backtracking.

Reasoning Enhancement via Confidence Vectors. We further explore potential applications of RISE for enhancing the reasoning process. Inspired by recent work (Hu et al., 2025) that learns a bias term

- at test time during the prefilling stage, we select the top-3 confidence vectors 𝑐1, 𝑐2, 𝑐3 and learn their combination coefficients at test time to control the reasoning process. As illustrated in Figure 4, this

- Table 2 | Generalization of confidence vectors across data domains under different steering directions.

GPQA KnowLogic Reflection Backtracking Reflection Backtracking

Steering Direction

Vanilla 53.23 11.83 35.56 5.42 Positive 61.37 16.18 48.16 5.98 Negative 37.25 6.91 20.04 3.51

yields a sample-dependent steering vector 𝑖∈1,2,3 𝛼𝑖𝑐𝑖. We compare this approach against SEAL (Chen et al., 2025) and TIP (Wang et al., 2025d). As shown in Figure 8, our careful steering strategy improves reasoning accuracy by up to 4.66 points while achieving a 13.69% reduction in token usage. These results serve as a pivotal case study demonstrating the potential of RISE to enable on-the-fly control of the reasoning process through interpretability.

MATH 500, Accuracy

MATH 500, Token Cost

AIME 2025, Accuracy

AIME 2025, Token Cost

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

80

15000

20

4000

60

Accuracy(%)

Accuracy(%)

15

3000

10000

Tokens

Tokens

40

10

2000

5000

20

5

1000

0

0

0

0

Vanilla TIP SEAL Confidence

Vanilla TIP SEAL Confidence

Vanilla TIP SEAL Confidence

Vanilla TIP SEAL Confidence

- Figure 8 | Results on reasoning accuracy and token cost under different steering methods on R1-1.5B.

##### 6. Conclusions

We presented an unsupervised framework for uncovering and manipulating reasoning behaviors in LLMs. By training SAEs on chain-of-thought activations, we discovered disentangled latent vectors that correspond to semantic reasoning behaviors, validated through clustering analyses and intervention studies. These results extend the linear representation hypothesis to reasoning, showing that abstract cognitive patterns can be linearly encoded and selectively controlled. In addition, we identify a robust length-aware organization in the latent space, where response length emerges as a structural axis influencing reasoning trajectories. Our findings suggest new opportunities for both interpretability and control. First, unsupervised discovery offers a scalable path toward mapping the cognitive landscape of reasoning models without relying on hand-crafted labels. Second, the controllability of SAE-derived vectors enables fine-grained test-time adaptation, allowing practitioners to modulate reasoning behaviors dynamically. Finally, the emergence of length as an organizing principle highlights an implicit inductive bias in reasoning LLMs, motivating future work on aligning structural and semantic features for more reliable reasoning.

##### References

M. Abdin, S. Agarwal, A. Awadallah, V. Balachandran, H. Behl, L. Chen, G. de Rosa, S. Gunasekar, M. Javaheripi, N. Joshi, P. Kauffmann, Y. Lara, C. C. T. Mendes, A. Mitra, B. Nushi, D. Papailiopoulos,

- O. Saarikivi, S. Shah, V. Shrivastava, V. Vineet, Y. Wu, S. Yousefi, and G. Zheng. Phi-4-reasoning technical report, 2025. URL https://arxiv.org/abs/2504.21318.

- J. Ahn, R. Verma, R. Lou, D. Liu, R. Zhang, and W. Yin. Large language models for mathematical reasoning: Progresses and challenges. arXiv preprint arXiv:2402.00157, 2024.

T. AlphaProof and T. AlphaGeometry. Ai achieves silver-medal standard solving international 178 mathematical olympiad problems. DeepMind blog, 179:45, 2024.

Anthropic. Claude 3.7 sonnet and claude code, February 2025. URL https://www.anthropic. com/news/claude-3-7-sonnet. Accessed: 2025-03-17.

- R. Chen, Z. Zhang, J. Hong, S. Kundu, and Z. Wang. Seal: Steerable reasoning calibration of large language models for free. arXiv preprint arXiv:2504.07986, 2025.

- G. Cui, Y. Zhang, J. Chen, L. Yuan, Z. Wang, Y. Zuo, H. Li, Y. Fan, H. Chen, W. Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.
- H. Cunningham, A. Ewart, L. Riggs, R. Huben, and L. Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600, 2023.

- R. Eisenstadt, I. Zimerman, and L. Wolf. Overclocking llm reasoning: Monitoring and controlling thinking path lengths in llms. arXiv preprint arXiv:2506.07240, 2025.

Y. Fu, J. Chen, S. Zhu, Z. Fu, Z. Dai, A. Qiao, and H. Zhang. Efficiently serving llm reasoning programs with certaindex. arXiv preprint arXiv:2412.20993, 2024.

Y. Fu, X. Wang, Y. Tian, and J. Zhao. Deep think with confidence. arXiv preprint arXiv:2508.15260, 2025.

- K. Gandhi, A. Chakravarthy, A. Singh, N. Lile, and N. D. Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

A. Goel, Y. Hu, I. Gurevych, and A. Sanyal. Differentially private steering for large language model

alignment, 2025. URL https://arxiv.org/abs/2501.18532. Google. Gemini flash thinking, February 2025. URL https://deepmind.google/technologies/ gemini/flash/. Accessed: 2025-05-11.

D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

C. Han, J. Xu, M. Li, Y. Fung, C. Sun, N. Jiang, T. Abdelzaher, and H. Ji. Word embeddings are steers for language models. arXiv preprint arXiv:2305.12798, 2023.

Y. Hu, X. Zhang, X. Fang, Z. Chen, X. Wang, H. Zhang, and G. Qi. Slot: Sample-specific language model optimization at test-time. arXiv preprint arXiv:2505.12392, 2025.

- X. Huang, W. Liu, X. Chen, X. Wang, H. Wang, D. Lian, Y. Wang, R. Tang, and E. Chen. Understanding the planning of llm agents: A survey. arXiv preprint arXiv:2402.02716, 2024.
- Y. Huang, H. Chen, S. Ruan, Y. Zhang, X. Wei, and Y. Dong. Mitigating overthinking in large reasoning models via manifold steering. arXiv preprint arXiv:2505.22411, 2025.

A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel,

- A. Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

- J. Jiang, F. Wang, J. Shen, S. Kim, and S. Kim. A survey on large language models for code generation. arXiv preprint arXiv:2406.00515, 2024.

D. P. Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever,

and K. Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023. S. Marks and M. Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. arXiv preprint arXiv:2310.06824, 2023. L. McInnes, J. Healy, and J. Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.

- K. Meng, D. Bau, A. Andonian, and Y. Belinkov. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022.

C. Olah and A. Jermyn. What is a linear representation? what is a multidimensional feature. Transformer Circuits Thread, 2024.

- B. Pang, L. Lee, and S. Vaithyanathan. Thumbs up? sentiment classification using machine learning techniques. In Proceedings of the ACL-02 Conference on Empirical Methods in Natural Language Processing - Volume 10, EMNLP ’02, page 79–86, USA, 2002. Association for Computational Linguistics.

K. Park, Y. J. Choe, and V. Veitch. The linear representation hypothesis and the geometry of large language models. arXiv preprint arXiv:2311.03658, 2023.

D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

P. J. Rousseeuw. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20:53–65, 1987.

- K. R. Shahapure and C. Nicholas. Cluster quality analysis using silhouette score. In 2020 IEEE 7th international conference on data science and advanced analytics (DSAA), pages 747–748. IEEE, 2020.
- L. Sheng, A. Zhang, Z. Wu, W. Zhao, C. Shen, Y. Zhang, X. Wang, and T.-S. Chua. On reasoning strength planning in large reasoning models. arXiv preprint arXiv:2506.08390, 2025.

- C. Snell, J. Lee, K. Xu, and A. Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

C.-E. Sun, G. Yan, and T.-W. Weng. Thinkedit: Interpretable weight editing to mitigate overly short thinking in reasoning models. arXiv preprint arXiv:2503.22048, 2025. doi: 10.48550/arXiv.2503. 22048.

- Q. Team. Qwq: Reflect deeply on the boundaries of the unknown, November 2024. URL https: //qwenlm.github.io/blog/qwq-32b-preview/.

Q. Team. Qwen3, April 2025. URL https://qwenlm.github.io/blog/qwen3/. K. Valmeekam, M. Marquez, S. Sreedharan, and S. Kambhampati. On the planning abilities of large

language models-a critical investigation. Advances in Neural Information Processing Systems, 36: 75993–76005, 2023.

C. Venhoff, I. Arcuschin, P. Torr, A. Conmy, and N. Nanda. Understanding reasoning in thinking language models via steering vectors. arXiv preprint arXiv:2506.18167, 2025.

- S. Wang, J. Asilis, Ö. F. Akgül, E. B. Bilgin, O. Liu, D. Fu, and W. Neiswanger. Resa: Transparent reasoning models via saes. arXiv preprint arXiv:2506.09967, 2025a.

- S. Wang, L. Yu, C. Gao, C. Zheng, S. Liu, R. Lu, K. Dang, X. Chen, J. Yang, Z. Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025b.
- T. Wang, X. Jiao, Y. Zhu, Z. Chen, Y. He, X. Chu, J. Gao, Y. Wang, and L. Ma. Adaptive activation steering: A tuning-free llm truthfulness improvement method for diverse hallucinations categories. In Proceedings of the ACM on Web Conference 2025, pages 2562–2578, 2025c.

Y. Wang, H. Chen, Y. Tang, T. Guo, K. Han, Y. Nie, X. Wang, H. Hu, Z. Bai, Y. Wang, et al. Pangu-pi: Enhancing language model architectures via nonlinearity compensation. arXiv preprint arXiv:2312.17276, 2023.

- Y. Wang, Q. Liu, J. Xu, T. Liang, X. Chen, Z. He, L. Song, D. Yu, J. Li, Z. Zhang, et al. Thoughts are all over the place: On the underthinking of o1-like llms. arXiv preprint arXiv:2501.18585, 2025d.

J. Ward, C. Lin, C. Venhoff, and N. Nanda. Reasoning-finetuning repurposes latent representations in base models. arXiv preprint arXiv:2507.12638, 2025.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.

A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

C. Yang, Q. Si, Y. Duan, Z. Zhu, C. Zhu, Z. Lin, L. Cao, and W. Wang. Dynamic early exit in reasoning models. arXiv preprint arXiv:2504.15895, 2025.

- W. Zhan, Y. Wang, N. Hu, L. Xiao, J. Ma, Y. Qin, Z. Li, Y. Yang, S. Deng, J. Ding, et al. Knowlogic: A benchmark for commonsense reasoning via knowledge-driven data synthesis. arXiv e-prints, pages arXiv–2503, 2025.

Z. Zhang, X. He, W. Yan, A. Shen, C. Zhao, S. Wang, Y. Shen, and X. E. Wang. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space. arXiv preprint arXiv:2505.15778, 2025.

- X. Zhao, Z. Kang, A. Feng, S. Levine, and D. Song. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.

A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A.-K. Dombrowski, et al. Representation engineering: A top-down approach to ai transparency, 2023. URL https://arxiv. org/abs/2310.01405.

##### A. Clarification of LLM Usage

In this work, we use LLMs to polish the writing, correct grammatical errors, and improve fluency. We also employ LLMs to generate the subfigures of the SAE decoder space in Figure 1. All LLM-generated content has been proofread by the authors before being included in this draft.

##### B. Theoretical Justification Proof. 1) Support identifiability by thresholding. For 𝑗 ∈ 𝑆,

###### |⟨ℎ, 𝑤𝑗⟩| = ∑︁ 𝑖∈𝑆

𝑎𝑖⟨𝑤𝑖, 𝑤𝑗⟩ + ⟨𝜀, 𝑤𝑗⟩

≥ |𝑎𝑗| − ∑︁

|𝑎𝑖| |⟨𝑤𝑖, 𝑤𝑗⟩| − ∥𝜀∥2 ≥ 𝛼 − (𝑘 − 1)𝜇 𝛼 − 𝜎.

𝑖∈𝑆\{ 𝑗}

|⟨ℎ, 𝑤𝑗⟩| ≤ ∑︁ 𝑖∈𝑆

For 𝑗 ∉ 𝑆,

|𝑎𝑖| |⟨𝑤𝑖, 𝑤𝑗⟩| + ∥𝜀∥2 ≤ 𝑘𝜇 𝛼 + 𝜎.

Hence any 𝜏 in the stated interval yields exact recovery 𝑆★(ℎ) = 𝑆. The separation margin is 𝛾 := 𝛼 1 − (2𝑘 − 1)𝜇 − 2𝜎 > 0.

- 2) Stable coefficient recovery on the true support. Condition on 𝑆: ℎ = 𝑊𝑆𝑎𝑆 + 𝜀. Then

𝑎ˆ𝑆 − 𝑎𝑆 = (𝑊𝑆⊤𝑊𝑆)−1𝑊𝑆⊤𝜀 = 𝑊𝑆+ 𝜀, so

∥𝑎ˆ𝑆 − 𝑎𝑆∥2 ≤ ∥𝑊𝑆+∥op∥𝜀∥2 = ∥𝜀∥2

𝜎min(𝑊𝑆)

.

By Gershgorin, 𝜆min(𝑊⊤

𝑆 𝑊𝑆) ≥ 1 − (𝑘 − 1)𝜇, hence 𝜎min(𝑊𝑆) ≥ √︁1 − (𝑘 − 1)𝜇 and

∥𝑎ˆ𝑆 − 𝑎𝑆∥2 ≤

𝜎 √︁1 − (𝑘 − 1)𝜇

.

- 3) Realizability of threshold+LS by a one-hidden-layer ReLU encoder. The map ℎ ↦→ 𝑆★(ℎ) is determined by finitely many half-space tests {±⟨ℎ, 𝑤𝑗⟩ > 𝜏}. On each polyhedral region with fixed 𝑆, the map ℎ ↦→ 𝑎ˆ𝑆 is affine. Thus 𝑇(ℎ) is piecewise affine. Because Step 1 provides a margin 𝛾 > 0, 𝑇 can be uniformly approximated on the 𝛾/2-interiors of these regions by a one-hidden-layer ReLU network, with the excluded boundary strip having vanishing probability as 𝜎 → 0. Hence there exist encoder parameters whose output 𝑧 coincides with 𝑇(ℎ) up to arbitrarily small error while preserving the sparsity pattern 𝑆★(ℎ).
- 4) Population objective separates true and false decoders. For any 𝑊′, define 𝜂𝑖 := dist 𝑤𝑖, span(𝑊′) .

(a) Span mismatch. If some 𝜂𝑖 > 0, then on E𝑖(𝛼, 𝛽) we can write ℎ = 𝑎𝑖𝑤𝑖 +𝑟+𝜀 with ∥𝑟∥2 ≤ 𝛽, and by Step 1 the index 𝑖 is selected. Any reconstruction using 𝑊′ must approximate 𝑎𝑖𝑤𝑖 from span(𝑊′), incurring error at least (𝛼 − 𝛽)𝜂𝑖; hence for some absolute 𝑐 > 0,

∥ℎ − 𝜙𝑊′(ℎ)∥22 ≥ (𝛼 − 𝛽)2𝜂2𝑖 − 𝑐(𝛽2 + 𝜎2).

Taking expectations and summing over 𝑖 yields a strictly positive gap 𝐽(𝑊′) ≥ minΠ,𝐷≻0 𝐽(𝑊Π𝐷) + 𝛿span > 0 for small enough 𝛽, 𝜎.

(b) Same-span misalignment. Suppose span(𝑊′) = span(𝑊) but 𝑊′ ≠ 𝑊Π𝐷. Then some 𝑤𝑖 is not colinear with any column of 𝑊′. On E𝑖(𝛼, 𝛽), reconstructing 𝑎𝑖𝑤𝑖 with 𝑊′ requires combining multiple misaligned columns, causing an oblique-projection loss bounded below by

∥𝑎𝑖𝑤𝑖 − 𝑊′𝑇(ℎ)∥22 ≥ 𝑐𝑖(𝛼 − 𝛽)2 − 𝑐(𝛽2 + 𝜎2), where 𝑐𝑖 > 0 depends only on the minimal principal angle between 𝑤𝑖 and the columns of 𝑊′. Averaging gives 𝐽(𝑊′) ≥ minΠ,𝐷≻0 𝐽(𝑊Π𝐷) + 𝛿align > 0 for small enough 𝛽, 𝜎.

Combining (a) and (b), for any 𝑊′ ∉ {𝑊Π𝐷} there exists 𝛿 > 0 such that 𝐽(𝑊′) ≥ min

𝐽(𝑊Π𝐷) + 𝛿. Note that the ℓ0 term does not diminish this gap; for 𝜆 > 0 it can only increase it because 𝑇(ℎ) is fixed.

Π,𝐷≻0

- 5) From population to empirical risk. Restrict 𝑊′ to a compact set W (e.g., each column norm in

[𝑐, 𝐶]). The map 𝑇 is piecewise-Lipschitz with finitely many linear regions and ∥𝑇(ℎ)∥0 ≤ 𝑘; hence the class

F := ℎ ↦→ ∥ℎ − 𝑊′𝑇(ℎ)∥22 + 𝜆∥𝑇(ℎ)∥0 : 𝑊′ ∈ W has finite pseudo-dimension and satisfies a uniform law of large numbers:

𝐽𝑁(𝑊′) − 𝐽(𝑊′) −−−−→ a.s.

sup

0.

𝑁→∞

𝑊′∈W

Therefore empirical local minimizers converge to the population minimizers, which by Step 4 equal {𝑊Π𝐷}.

##### C. Empirical Validation of Sparsity and Incoherence Assumptions

Furthermore, we empirically validate the key assumptions underlying the above theoretical analysis: (i) Incoherence, defined as max𝑖≠𝑗 |⟨|𝑤𝑤𝑖,𝑤𝑗⟩|

𝑖||𝑤𝑗| ≤ 𝜇 < 1, and (ii) Sparsity, characterized by 𝑘 < 𝑐/𝜇 for a universal constant 𝑐.

k-Sparsity Across Layers

Max Cosine Similarity Across Layers

Reconstruction Error Across Layers

0.08

0.30

| | | | |
|---|---|---|---|
| | | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

300

0.25

MaxCosineSimilarity

ReconstructionError

0.06

0.20

k-Sparsity

200

0.04

0.15

0.10

100

0.02

0.05

0

0.00

0.00

0 5 10 15 20 25 Layer

0 5 10 15 20 25 Layer

0 5 10 15 20 25 Layer

- Figure 9 | Results on Sparsity and Incoherence Assumptions.

Since the ground-truth dictionary 𝑊 in the theorem 1 is not directly observable, we use the reconstructed activations from the sparse autoencoder for evaluation. Given the consistently low reconstruction error in Figure 9, this serves as a reliable proxy. We observe that the maximum cosine similarity among recovered dictionary vectors is clearly bounded by a valid value , and the sparsity level k (defined as the number of active dictionary vectors used to reconstruct each activation) is also well within the theoretical bound. These results provide empirical support for the assumptions underlying our theoretical analysis.

##### D. Annotation Details for Reasoning Behaviors

We analyze the consistency of LLM annotation results in Section 4.3, using the following prompt to query GPT-5, GPT-4o, and Claude Sonnet 4.5 for classification.

You are a helpful expert that is good at classifying reasoning steps. You will be given a single reasoning step from a math/logic solution. Your task is to classify the reasoning step according to the provided

taxonomy and decision rules. The available labels are:

- (1) reflection: step checking its previous reasoning process and stating its own uncertainty.

- (2) backtracking: steps that explicitly retract/pivot , proposing an alternative strategy to replace the current one.

- (3) others: steps that do not fall into the above two categories.

You must select the class labels based on the above criteria and assign a single class for each step.

Your output should be a strict label from the above three options: "

reflection", "backtracking", or "others". If you cannot determine the label , please assign "others". Now , please classify the following reasoning step delimited by triple

backticks , according to the taxonomy and decision rules provided in the system prompt.

Reasoning Step: ‘‘‘{text}’’’

Table 3 | Keywords set for annotation of reasoning steps.

Wait, verify, make sure, hold on, think again, ’s correct, ’s incorrect

Reflection

Let me check, seems right Backtracking

Alternatively, think differenly, another way, another approach, another method, another solution, another strategy, another technique

Annotation Consistency: To analyze the robustness of the annotation method, we compare annotation results produced by different LLM judges and a keyword-matching approach, using the classification criteria defined in Table 3. The results, shown in Figure 10, report the agreement ratio (defined as the proportion of steps receiving the same annotation relative to all steps) exceeding 85% for each pair of methods. Overall, the consistency is high across most comparisons, with GPT-5 and GPT-4o exhibiting particularly strong agreement at approximately 94%. These results validate the reliability and consistency of our annotation methodology.

##### E. SAE Reveals the Underlying Geometry of Response Length.

We manually split the step-level activations {ℎ𝑙𝑖} (where 𝑙 denotes the layer index) into two categories: short responses, with sequence length less than one thousand tokens, and long responses, with sequence

length exceeding eight thousand tokens. Following the same procedure, we forward the activations from each category into the SAE and highlight the most active columns.

Structures aligned with response length emerge in the SAE column space. Figure 11 presents UMAP visualizations of SAE decoder columns across layers, revealing two distinct clusters that align with response length. Columns associated with long responses form a more diverse and dispersed

#### Annotation Overlap Between Different Judges

1.0

|[Figure 5]| |
|---|---|
| | |
| | |
| | |
| | |

Claude Sonnet 4.5

1 0.86 0.88 0.85

0.8

Keywords

0.86 1 0.86 0.85

0.6

0.4

- GPT-4o

- GPT-5

0.88 0.86 1 0.94

0.2

0.85 0.85 0.94 1

0.0

Claude Sonnet 4.5 Keywords GPT-4o GPT-5

- Figure 10 | Results of annotation consistency across different methods. The numbers represent the agreement ratio for each pair of annotation methods.

cluster, while those linked to short responses appear compact. This separation is weak in early layers but becomes increasingly clear and stable in mid-to-late layers, peaking just before the output stage. These results suggest that SAE representations progressively encode structural signals about response length alongside behavior-specific features. Also, the normalized silhouette scores across layers, based on response length clustering, are reported in Figure 12. We observe that the early layers exhibit weak separation with respect to response length, while the mid-to-late layers demonstrate stronger alignment with this structural property.

Layer 1

Layer 2

Layer 3

Layer 4

Layer 5

Layer 6

Layer 7

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

UMAP-2

Layer 8

Layer 9

Layer 10

Layer 11

Layer 12

Layer 13

Layer 14

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

UMAP-2

Layer 15

Layer 16

Layer 17

Layer 18

Layer 19

Layer 20

Layer 21

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

UMAP-2

Layer 22

Layer 23

Layer 24

Layer 25

Layer 26

Layer 27

Layer 28

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

UMAP-2

UMAP-1

UMAP-1

UMAP-1

UMAP-1

UMAP-1

UMAP-1

UMAP-1

- Figure 11 | Visualization of the SAE decoder columns. From left to right, we show the raw SAE decoder columns and the corresponding results with human-defined behaviors highlighted (green/yellow dots represents the columns related with short and long responses, respectively). Results are obtained from DeepSeek-R1-1.5B using MATH-500 training samples.

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 Layer

0.0

0.2

0.4

0.6

0.8

1.0

NormalizedSilhouetteScore

Length-based Silhouette Score

- Figure 12 | Normalized Silhouette scores across different layers of R1-1.5B.

